#!/usr/bin/env python3
"""Master orchestrator: pre-eval -> train (sleep) -> post-eval."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def ensure_memories(memory_path: Path) -> None:
    if not memory_path.exists() or memory_path.stat().st_size == 0:
        print(
            f"Memory file missing or empty at {memory_path}. "
            "Please run chat.py to log some conversations first."
        )
        sys.exit(1)


def run(name: str, cmd) -> str:
    """Run command and return stdout."""
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout


def parse_loss_from_output(output: str) -> float | None:
    """Extract loss value from eval_loss.py output."""
    match = re.search(r"Average loss over \d+ samples: ([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None


def parse_harness_from_output(output: str) -> dict | None:
    """Extract harness results from eval_harness.py output."""
    match = re.search(r"Total: (\d+), Passed: (\d+), Pass rate: ([\d.]+)", output)
    if match:
        return {
            "total": int(match.group(1)),
            "passed": int(match.group(2)),
            "pass_rate": float(match.group(3)),
        }
    return None


def write_experiment_log(log_path: Path, record: dict) -> None:
    """Append experiment record to JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n📊 Experiment logged to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-eval, sleep, post-eval cycle.")
    parser.add_argument("--tag", required=True, help="Base tag for the run, e.g., day3.")
    parser.add_argument("--model-id", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--adapter-dir", type=Path, default=Path("adapters/nightly_update"))
    parser.add_argument("--memory-file", type=Path, default=Path("data/memories.jsonl"))
    parser.add_argument("--prompts", type=Path, default=Path("eval/prompts.jsonl"))
    parser.add_argument("--loss-max-samples", type=int, default=10)
    parser.add_argument("--eval-max-prompts", type=int, help="Limit prompts for quick runs.")
    parser.add_argument("--loss-csv", type=Path, default=Path("logs/loss.csv"))
    # sleep params
    parser.add_argument("--sleep-iters", type=int, default=100)
    parser.add_argument("--sleep-batch-size", type=int, default=1)
    parser.add_argument("--sleep-lora-r", type=int, default=8)
    parser.add_argument("--sleep-lora-alpha", type=int, default=16)
    parser.add_argument("--sleep-learning-rate", type=float, default=2e-4)
    parser.add_argument("--sleep-max-seq-len", type=int, default=1024)
    parser.add_argument("--sleep-augment", action="store_true", help="Enable semisynthetic augmentation.")
    parser.add_argument("--sleep-augment-per-record", type=int, default=1)
    parser.add_argument("--sleep-augment-max-tokens", type=int, default=256)
    parser.add_argument("--sleep-augment-temp", type=float, default=0.8)
    parser.add_argument("--sleep-augment-top-p", type=float, default=0.9)
    parser.add_argument("--sleep-temp", type=float, default=0.7)
    parser.add_argument("--sleep-top-p", type=float, default=0.9)
    parser.add_argument("--sleep-reset-adapter", action="store_true", help="Wipe adapter dir before training.")
    args = parser.parse_args()

    pre_tag = f"{args.tag}-pre"
    post_tag = f"{args.tag}-post"

    args.loss_csv.parent.mkdir(parents=True, exist_ok=True)
    ensure_memories(args.memory_file)

    # Reset adapter BEFORE pre-eval so we compare against base model
    if args.sleep_reset_adapter and args.adapter_dir.exists():
        print(f"Resetting adapter at {args.adapter_dir} (--sleep-reset-adapter)")
        shutil.rmtree(args.adapter_dir)

    # If no adapter yet, ensure eval steps don't try to load one
    if not (args.adapter_dir / "adapter_config.json").exists():
        print(f"No adapter found at {args.adapter_dir}; running evals against base model.")

    # Pre: loss
    pre_loss_out = run(
        "eval_loss (pre)",
        [
            sys.executable,
            str(ROOT / "eval_loss.py"),
            "--model-id",
            args.model_id,
            "--adapter-dir",
            str(args.adapter_dir),
            "--memory-file",
            str(args.memory_file),
            "--max-samples",
            str(args.loss_max_samples),
            "--tag",
            pre_tag,
            "--out-csv",
            str(args.loss_csv),
        ],
    )
    pre_loss = parse_loss_from_output(pre_loss_out)

    # Pre: capability harness
    harness_cmd = [
        sys.executable,
        str(ROOT / "eval_harness.py"),
        "--model-id",
        args.model_id,
        "--adapter-dir",
        str(args.adapter_dir),
        "--prompts",
        str(args.prompts),
        "--tag",
        pre_tag,
    ]
    if args.eval_max_prompts:
        harness_cmd += ["--max-prompts", str(args.eval_max_prompts)]
    pre_harness_out = run("eval_harness (pre)", harness_cmd)
    pre_harness = parse_harness_from_output(pre_harness_out)

    # Train (sleep)
    sleep_cmd = [
        sys.executable,
        str(ROOT / "sleep.py"),
        "--model-id",
        args.model_id,
        "--memory-file",
        str(args.memory_file),
        "--adapter-dir",
        str(args.adapter_dir),
        "--batch-size",
        str(args.sleep_batch_size),
        "--iters",
        str(args.sleep_iters),
        "--lora-r",
        str(args.sleep_lora_r),
        "--lora-alpha",
        str(args.sleep_lora_alpha),
        "--learning-rate",
        str(args.sleep_learning_rate),
        "--max-seq-len",
        str(args.sleep_max_seq_len),
        "--temp",
        str(args.sleep_temp),
        "--top-p",
        str(args.sleep_top_p),
    ]
    if args.sleep_augment:
        sleep_cmd += [
            "--augment",
            "--augment-per-record",
            str(args.sleep_augment_per_record),
            "--augment-max-tokens",
            str(args.sleep_augment_max_tokens),
            "--augment-temp",
            str(args.sleep_augment_temp),
            "--augment-top-p",
            str(args.sleep_augment_top_p),
        ]
    if args.sleep_reset_adapter:
        sleep_cmd.append("--reset-adapter")
    run("sleep (train)", sleep_cmd)

    # Post: loss
    post_loss_out = run(
        "eval_loss (post)",
        [
            sys.executable,
            str(ROOT / "eval_loss.py"),
            "--model-id",
            args.model_id,
            "--adapter-dir",
            str(args.adapter_dir),
            "--memory-file",
            str(args.memory_file),
            "--max-samples",
            str(args.loss_max_samples),
            "--tag",
            post_tag,
            "--out-csv",
            str(args.loss_csv),
        ],
    )
    post_loss = parse_loss_from_output(post_loss_out)

    # Post: capability harness
    harness_cmd_post = [
        sys.executable,
        str(ROOT / "eval_harness.py"),
        "--model-id",
        args.model_id,
        "--adapter-dir",
        str(args.adapter_dir),
        "--prompts",
        str(args.prompts),
        "--tag",
        post_tag,
    ]
    if args.eval_max_prompts:
        harness_cmd_post += ["--max-prompts", str(args.eval_max_prompts)]
    post_harness_out = run("eval_harness (post)", harness_cmd_post)
    post_harness = parse_harness_from_output(post_harness_out)

    # Write comprehensive experiment log
    experiment_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "model": args.model_id,
        "memory_file": str(args.memory_file),
        "hyperparameters": {
            "learning_rate": args.sleep_learning_rate,
            "iters": args.sleep_iters,
            "batch_size": args.sleep_batch_size,
            "lora_r": args.sleep_lora_r,
            "lora_alpha": args.sleep_lora_alpha,
            "max_seq_len": args.sleep_max_seq_len,
            "reset_adapter": args.sleep_reset_adapter,
        },
        "pre": {
            "loss": pre_loss,
            "harness": pre_harness,
        },
        "post": {
            "loss": post_loss,
            "harness": post_harness,
        },
        "delta": {
            "loss": round(post_loss - pre_loss, 4) if pre_loss and post_loss else None,
            "harness_passed": (post_harness["passed"] - pre_harness["passed"]) if pre_harness and post_harness else None,
        },
    }
    write_experiment_log(Path("logs/experiments.jsonl"), experiment_record)

    # Print summary
    print("\n" + "=" * 60)
    print(f"🧪 EXPERIMENT SUMMARY: {args.tag}")
    print("=" * 60)
    print(f"Hyperparameters: lr={args.sleep_learning_rate}, iters={args.sleep_iters}, r={args.sleep_lora_r}")
    print(f"Pre:  loss={pre_loss:.4f}, harness={pre_harness['passed']}/{pre_harness['total']} ({pre_harness['pass_rate']:.0%})" if pre_loss and pre_harness else "Pre: N/A")
    print(f"Post: loss={post_loss:.4f}, harness={post_harness['passed']}/{post_harness['total']} ({post_harness['pass_rate']:.0%})" if post_loss and post_harness else "Post: N/A")
    if pre_loss and post_loss:
        loss_delta = post_loss - pre_loss
        print(f"Δ Loss: {loss_delta:+.4f} ({'✅ improved' if loss_delta < 0 else '⚠️ worse'})")
    if pre_harness and post_harness:
        harness_delta = post_harness["passed"] - pre_harness["passed"]
        print(f"Δ Harness: {harness_delta:+d} ({'✅ improved' if harness_delta > 0 else '⚠️ worse' if harness_delta < 0 else '— unchanged'})")
    print("=" * 60)


if __name__ == "__main__":
    main()

