#!/usr/bin/env python3
"""
Terminal chat interface using mlx-lm and Llama 3.2 3B Instruct (4-bit).

Circadia Architecture:
- LoRA adapter: Learns your style/preferences (compressed into weights)
- RAG memory: Retrieves specific facts/conversations (stored verbatim)

Together they mimic biological memory:
- LoRA = procedural memory (how to respond)
- RAG = episodic memory (what happened)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# RAG imports (optional, graceful fallback if not available)
try:
    from rag import ConversationMemory, format_context_prompt
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_MEMORY = Path("data/memories.jsonl")
DEFAULT_ADAPTER_DIR = Path("adapters/nightly_update")
DEFAULT_RAG_DIR = Path("data/rag_memory")


def format_prompt(
    tokenizer,
    history: List[Dict[str, str]],
    user_message: str,
) -> str:
    """Convert chat history + new user message into a model-ready prompt."""
    messages = history + [{"role": "user", "content": user_message}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback prompt if the tokenizer lacks a chat template.
    prompt_lines = []
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        prompt_lines.append(f"{speaker}: {msg['content']}")
    prompt_lines.append("Assistant:")
    return "\n".join(prompt_lines)


def append_memory(memory_path: Path, user: str, assistant: str) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    with memory_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def detect_adapter(adapter_dir: Path) -> Optional[Path]:
    """Return adapter_dir only if BOTH config and weights exist (handles crash recovery)."""
    cfg = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapters.safetensors"
    if cfg.exists() and weights.exists():
        return adapter_dir
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with an MLX model.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    # RAG options
    parser.add_argument("--rag", action="store_true", help="Enable RAG retrieval")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-results", type=int, default=3, help="Number of RAG results to retrieve")
    parser.add_argument("--rag-verbose", action="store_true", help="Show retrieved context")
    args = parser.parse_args()

    adapter_path = detect_adapter(args.adapter_dir)
    model, tokenizer = load(
        args.model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    # Initialize RAG memory if enabled
    rag_memory = None
    if args.rag:
        if not RAG_AVAILABLE:
            print("⚠️  RAG requested but rag.py not available. Running without RAG.")
        else:
            rag_memory = ConversationMemory(persist_dir=args.rag_dir)
            print(f"📚 RAG memory loaded: {rag_memory.count()} conversations")

    print(f"Loaded model: {args.model_id}")
    if adapter_path:
        print(f"🧠 Using LoRA adapter: {adapter_path}")
    else:
        print("Using base model (no adapter).")
    if rag_memory:
        print("🔍 RAG retrieval: enabled")
    print("Type 'exit' or Ctrl-D to quit.\n")

    history: List[Dict[str, str]] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue

        # RAG: Retrieve relevant context
        enhanced_input = user_input
        if rag_memory:
            retrieved = rag_memory.search(user_input, n_results=args.rag_results)
            if retrieved and args.rag_verbose:
                print(f"  📚 Retrieved {len(retrieved)} relevant conversations")
                for r in retrieved:
                    print(f"     - {r['id']} (relevance: {r['relevance']})")
            if retrieved:
                enhanced_input = format_context_prompt(user_input, retrieved)

        prompt = format_prompt(tokenizer, history, enhanced_input)
        print("Assistant: ", end="", flush=True)

        reply = ""
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            print(text, end="", flush=True)
            reply += text
        print()

        # Update session history (use original input, not enhanced)
        history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": reply},
            ]
        )
        
        # Save to memories (for LoRA training)
        append_memory(args.memory_file, user_input, reply)
        
        # Save to RAG memory (for retrieval)
        if rag_memory:
            rag_memory.add_conversation([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": reply},
            ])


if __name__ == "__main__":
    main()

