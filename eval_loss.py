#!/usr/bin/env python3
"""Compute average token-level loss on logged conversations using mlx-lm."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
from mlx_lm import load


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_MEMORY = Path("data/memories.jsonl")
DEFAULT_ADAPTER_DIR = Path("adapters/nightly_update")


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


def tokens_from_messages(tokenizer, messages: List[Dict[str, str]], max_seq_len: int) -> Optional[List[int]]:
    prompt = _format_messages(tokenizer, messages)
    tokens = tokenizer.encode(prompt)
    if len(tokens) < 2:
        return None
    return tokens[:max_seq_len]


def loss_for_tokens(model, tokens: List[int]) -> float:
    # Teacher-forced loss: predict token[t+1] from token[t]
    input_ids = mx.array([tokens[:-1]], dtype=mx.int32)
    targets = mx.array([tokens[1:]], dtype=mx.int32)
    logits = model(input_ids)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return float(mx.mean(nll).item())


def evaluate_records(model, tokenizer, records: List[Dict], max_seq_len: int, max_samples: Optional[int]) -> float:
    losses: List[float] = []
    for rec in records:
        if max_samples is not None and len(losses) >= max_samples:
            break
        tokens = tokens_from_messages(tokenizer, rec.get("messages", []), max_seq_len)
        if tokens is None:
            continue
        losses.append(loss_for_tokens(model, tokens))
    if not losses:
        raise ValueError("No valid samples to evaluate.")
    return sum(losses) / len(losses)


def load_records(memory_path: Path) -> List[Dict]:
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
            if "messages" in obj:
                records.append({"messages": obj["messages"]})
    if not records:
        raise ValueError("No records found to evaluate.")
    return records


def _write_csv(csv_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "loss", "samples", "model", "adapter", "tag"]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(
            ",".join(
                [
                    row.get("timestamp", ""),
                    f"{row.get('loss', '')}",
                    f"{row.get('samples', '')}",
                    row.get("model", ""),
                    row.get("adapter", ""),
                    row.get("tag", ""),
                ]
            )
            + "\n"
        )


def _write_jsonl(jsonl_path: Path, row: Dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate average loss on chat logs.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--max-samples", type=int, default=10, help="Cap evaluated samples to control runtime.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--out-csv", type=Path, help="Append metrics to a CSV file.")
    parser.add_argument("--out-jsonl", type=Path, help="Append metrics to a JSONL file.")
    parser.add_argument("--tag", type=str, default="", help="Optional label for this eval run (e.g., day1-pre).")
    args = parser.parse_args()

    adapter_path = _adapter_if_present(args.adapter_dir)
    model, tokenizer = load(
        args.model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    records = load_records(args.memory_file)
    avg_loss = evaluate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
    )
    samples_used = min(len(records), args.max_samples or len(records))
    print(f"Average loss over {samples_used} samples: {avg_loss:.4f}")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "loss": avg_loss,
        "samples": samples_used,
        "model": args.model_id,
        "adapter": str(adapter_path) if adapter_path else "",
        "tag": args.tag,
    }
    if args.out_csv:
        _write_csv(args.out_csv, row)
        print(f"Appended to CSV: {args.out_csv}")
    if args.out_jsonl:
        _write_jsonl(args.out_jsonl, row)
        print(f"Appended to JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()

