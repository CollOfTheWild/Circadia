#!/usr/bin/env python3
"""Nightly QLoRA fine-tune over accumulated chat logs using mlx-lm.

Optionally augments the dataset with semisynthetic pairs prior to training,
and routes factual/snippet-like conversations into a RAG store.
"""

import argparse
import json
import re
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# Optional RAG imports (graceful fallback if not installed)
try:
    from rag import ConversationMemory

    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_MEMORY = Path("data/memories.jsonl")
DEFAULT_ADAPTER_DIR = Path("adapters/nightly_update")


def load_memories(memory_path: Path) -> List[Dict]:
    if not memory_path.exists():
        raise FileNotFoundError(f"No memory file found at {memory_path}")

    records: List[Dict] = []
    with memory_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in obj and isinstance(obj["messages"], list):
                records.append({"messages": obj["messages"]})

    if not records:
        raise ValueError("Memory file is empty or malformed; nothing to train on.")
    return records


def classify_for_rag(messages: List[Dict[str, str]]) -> bool:
    """Heuristic: decide if a conversation should also go to RAG."""
    text = " ".join(m.get("content", "") for m in messages).lower()

    has_code = "```" in text or re.search(r"(def |class |;\\s*$)", text)
    has_url = "http://" in text or "https://" in text
    is_long = len(text) > 280
    looks_json = "{" in text and "}" in text
    lots_of_numbers = len(re.findall(r"\\d", text)) > 20

    return any([has_code, has_url, is_long, looks_json, lots_of_numbers])


def split_for_rag(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split records into (rag_records, lora_records)."""
    rag_records: List[Dict] = []
    lora_records: List[Dict] = []
    for rec in records:
        msgs = rec.get("messages", [])
        if classify_for_rag(msgs):
            rag_records.append(rec)
        else:
            lora_records.append(rec)
    # Ensure LoRA still sees all data (keep original behavior)
    if not lora_records:
        lora_records = records
    return rag_records, lora_records


def write_split_dataset(records: List[Dict], dataset_dir: Path) -> None:
    """Write train/valid/test JSONL; reuse train if fewer than 2 samples."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"
    test_path = dataset_dir / "test.jsonl"

    def dump(path: Path, data: List[Dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dump(train_path, records)

    # Minimal validation/test sets required by mlx_lm.lora; reuse/train slice.
    val_sample = records[:1] if records else []
    dump(valid_path, val_sample)
    dump(test_path, val_sample)


def launch_finetune(
    model_id: str,
    dataset_dir: Path,
    adapter_dir: Path,
    batch_size: int,
    iters: int,
    learning_rate: float,
    max_seq_len: int,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--train",
        "--fine-tune-type",
        "lora",
        "--model",
        model_id,
        "--data",
        str(dataset_dir),
        "--adapter-path",
        str(adapter_dir),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_len),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _adapter_if_present(adapter_dir: Path) -> Optional[Path]:
    """Return adapter_dir only if BOTH config and weights exist (handles crash recovery)."""
    cfg = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapters.safetensors"
    if cfg.exists() and weights.exists():
        return adapter_dir
    return None


def _format_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def augment_records(
    records: List[Dict],
    model_id: str,
    adapter_dir: Path,
    per_record: int,
    max_tokens: int,
    temp: float,
    top_p: float,
) -> Tuple[List[Dict], int]:
    """Use the model to generate semisynthetic Q/A pairs from existing chats."""
    adapter_path = _adapter_if_present(adapter_dir)
    model, tokenizer = load(
        model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    sampler = make_sampler(temp=temp, top_p=top_p)

    augmented: List[Dict] = []
    for rec in records:
        convo_text = _format_messages(tokenizer, rec.get("messages", []))
        prompt = (
            "You are a data augmentation helper. Given the conversation below, "
            f"produce {per_record} short, on-topic user/assistant pairs that stay faithful "
            "to the tone and facts. Output JSON as a list of objects, each with "
            "\"messages\": [{\"role\": \"user\", \"content\": ...}, {\"role\": \"assistant\", \"content\": ...}]. "
            "Return JSON only."
            "\n\nCONVERSATION:\n"
            f"{convo_text}\n\nJSON:\n"
        )

        generated = ""
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            generated += text

        try:
            parsed = json.loads(generated)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            for item in parsed:
                if (
                    isinstance(item, dict)
                    and "messages" in item
                    and isinstance(item["messages"], list)
                    and len(item["messages"]) == 2
                ):
                    augmented.append({"messages": item["messages"]})

    return records + augmented, len(augmented)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly QLoRA fine-tuning job.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature for augmentation.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for augmentation.")
    parser.add_argument("--augment", action="store_true", help="Generate semisynthetic pairs before training.")
    parser.add_argument("--augment-per-record", type=int, default=1)
    parser.add_argument("--augment-max-tokens", type=int, default=256)
    parser.add_argument("--augment-temp", type=float, default=0.8)
    parser.add_argument("--augment-top-p", type=float, default=0.9)
    parser.add_argument("--reset-adapter", action="store_true", help="Wipe adapter dir before training.")
    # RAG options
    parser.add_argument("--rag", action="store_true", help="Ingest factual/snippet conversations into RAG.")
    parser.add_argument("--rag-dir", type=Path, default=Path("data/rag_memory"))
    parser.add_argument("--rag-clear", action="store_true", help="Clear RAG store before ingesting.")
    args = parser.parse_args()

    records = load_memories(args.memory_file)
    if args.augment:
        records, added = augment_records(
            records=records,
            model_id=args.model_id,
            adapter_dir=args.adapter_dir,
            per_record=args.augment_per_record,
            max_tokens=args.augment_max_tokens,
            temp=args.augment_temp,
            top_p=args.augment_top_p,
        )
        print(f"Augmented dataset with {added} semisynthetic records.")

    # Split into RAG vs LoRA slices
    if args.rag:
        rag_records, lora_records = split_for_rag(records)
        print(f"RAG routing: {len(rag_records)} to RAG, {len(lora_records)} to LoRA")

        if not RAG_AVAILABLE:
            print("⚠️  RAG requested but rag.py/chromadb unavailable. Skipping RAG ingestion.")
        else:
            rag_memory = ConversationMemory(persist_dir=args.rag_dir)
            if args.rag_clear:
                rag_memory.clear()
            for rec in rag_records:
                rag_memory.add_conversation(rec.get("messages", []))
            print(f"RAG ingest complete. Total conversations in store: {rag_memory.count()}")
    else:
        lora_records = records

    if args.reset_adapter and args.adapter_dir.exists():
        shutil.rmtree(args.adapter_dir)
        args.adapter_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.adapter_dir / "dset"
    write_split_dataset(lora_records, dataset_dir)
    launch_finetune(
        model_id=args.model_id,
        dataset_dir=dataset_dir,
        adapter_dir=args.adapter_dir,
        batch_size=args.batch_size,
        iters=args.iters,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()

