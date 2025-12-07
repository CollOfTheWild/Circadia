#!/usr/bin/env python3
"""Simple evaluation harness to score prompts against the current model/adapter."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_PROMPTS = Path("eval/prompts.jsonl")
DEFAULT_ADAPTER_DIR = Path("adapters/nightly_update")


def _adapter_if_present(adapter_dir: Path) -> Optional[Path]:
    """Return adapter_dir only if BOTH config and weights exist (handles crash recovery)."""
    cfg = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapters.safetensors"
    if cfg.exists() and weights.exists():
        return adapter_dir
    return None


def load_prompts(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(prompts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt" in obj and "id" in obj:
                prompts.append(obj)
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def generate_response(model, tokenizer, prompt: str, max_tokens: int, temp: float, top_p: float) -> str:
    sampler = make_sampler(temp=temp, top_p=top_p)
    response = ""
    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        response += text
    return response.strip()


def _extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def score_response(spec: Dict[str, Any], response: str) -> Tuple[bool, str]:
    expected = spec.get("expected", {})
    etype = expected.get("type")

    if etype == "contains_any":
        keywords = expected.get("keywords", [])
        resp_low = response.lower()
        if any(k.lower() in resp_low for k in keywords):
            return True, "matched keyword"
        return False, "missing keywords"

    if etype == "regex":
        pattern = expected.get("pattern", "")
        if re.search(pattern, response, flags=re.MULTILINE | re.DOTALL):
            return True, "regex matched"
        return False, "regex failed"

    if etype == "numeric":
        target = expected.get("value")
        tol = expected.get("tolerance", 0.0)
        num = _extract_first_number(response)
        if num is None:
            return False, "no number found"
        if abs(num - target) <= tol:
            return True, f"numeric within tol ({num} vs {target})"
        return False, f"numeric off ({num} vs {target})"

    if etype == "json_keys":
        keys = expected.get("keys", [])
        try:
            obj = json.loads(response)
        except json.JSONDecodeError:
            return False, "json parse failed"
        if all(k in obj for k in keys):
            return True, "json keys present"
        return False, "missing json keys"

    if etype == "format_bullets":
        min_count = expected.get("min_count", 1)
        bullets = [line for line in response.splitlines() if line.strip().startswith("-")]
        if len(bullets) >= min_count:
            return True, "enough bullets"
        return False, "too few bullets"

    return False, "unsupported expected type"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompts against the model/adapter.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-prompts", type=int, help="Limit number of prompts for quick runs.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out-jsonl", type=Path, default=Path("logs/eval_results.jsonl"))
    parser.add_argument("--out-summary", type=Path, default=Path("logs/eval_summary.json"))
    parser.add_argument("--tag", type=str, default="", help="Label for this run, e.g., day3-pre.")
    args = parser.parse_args()

    adapter_path = _adapter_if_present(args.adapter_dir)
    model, tokenizer = load(
        args.model_id,
        adapter_path=str(adapter_path) if adapter_path else None,
    )

    prompts = load_prompts(args.prompts, args.max_prompts)
    results: List[Dict[str, Any]] = []

    for spec in prompts:
        resp = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=spec["prompt"],
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
        )
        passed, reason = score_response(spec, resp)
        results.append(
            {
                "id": spec.get("id"),
                "category": spec.get("category", "uncategorized"),
                "prompt": spec.get("prompt"),
                "response": resp,
                "passed": passed,
                "reason": reason,
                "expected": spec.get("expected", {}),
                "tag": args.tag,
                "model": args.model_id,
                "adapter": str(adapter_path) if adapter_path else "",
            }
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summaries
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"total": 0, "passed": 0})
        by_cat[cat]["total"] += 1
        by_cat[cat]["passed"] += 1 if r["passed"] else 0

    summary = {
        "tag": args.tag,
        "model": args.model_id,
        "adapter": str(adapter_path) if adapter_path else "",
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_category": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": v["passed"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in by_cat.items()
        },
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run tag: {args.tag}")
    print(f"Total: {total}, Passed: {passed}, Pass rate: {summary['pass_rate']:.2f}")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.2f})")
    print(f"Wrote details to {args.out_jsonl}")
    print(f"Wrote summary to {args.out_summary}")


if __name__ == "__main__":
    main()

