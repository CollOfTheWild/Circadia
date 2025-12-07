# Project State Log

This document tracks the current understanding of the project and logs every change made by the assistant.

## Overview
- **Goal:** Build a “Circadian Rhythm” workflow where daytime chats with an MLX-served LLM are logged and then distilled into the model’s weights via nightly QLoRA fine-tuning on Apple Silicon (Mac Mini M2, 16 GB).
- **Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (quantized to fit memory).
- **Key scripts:**
  - `chat.py`: Terminal chat using `mlx_lm`, streams responses, appends each user/assistant turn to `data/memories.jsonl`. If `adapters/nightly_update` exists, loads it so the chat uses the latest adapter.
  - `sleep.py`: Nightly fine-tune using `mlx_lm.finetune` with conservative QLoRA settings, reading `data/memories.jsonl` and saving adapters to `adapters/nightly_update`.
- **Dependencies:** Minimal (`mlx`, `mlx-lm`, `huggingface_hub`) in `requirements.txt`.
- **Data layout:** Chat logs stored as JSONL with `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}` entries. Adapters saved under `adapters/nightly_update`.

## Change Log
- **2025-12-06:** Initial assistant pass: added minimal requirements, implemented `chat.py` logging + adapter loading, implemented `sleep.py` QLoRA finetune runner, created data/adapter directories.
- **2025-12-06:** Added `STATE.md` for project tracking. Extended `sleep.py` with optional semisynthetic augmentation (via model-generated pairs) ahead of QLoRA training and supporting CLI flags.
- **2025-12-06:** Added `eval_loss.py` to compute average token loss over logged chats using the current model/adapter.
- **2025-12-06:** Added `pytest` to requirements to support the automated test suite.
- **2025-12-06:** Added pytest suite covering chat logging, dataset handling, finetune CLI construction, augmentation parsing, and loss evaluation helpers.
- **2025-12-06:** Updated `README.md` with setup, chat/sleep usage, loss evaluation, testing, and multi-day guidance.
- **2025-12-06:** Extended `eval_loss.py` with CSV/JSONL logging and tagging for plotting loss curves; noted usage in `README.md`.
- **2025-12-06:** Added `eval_harness.py` for capability scoring, `eval/prompts.jsonl` starter set, and README instructions for running pre/post evaluations.
- **2025-12-06:** Added `run_cycle.py` to orchestrate pre-loss eval, pre-capability eval, training, and post evaluations with tunable flags; documented usage in README.
- **2025-12-06:** Hardened `run_cycle.py` to require a non-empty `data/memories.jsonl` and updated README to remind running chat first.
- **2025-12-06:** Fixed stream_generate calls to use `temperature` (mlx-lm API change) in `chat.py`, `sleep.py`, and `eval_harness.py`.
- **2025-12-06:** Updated sampling to use `TopPSampler` (mlx-lm API) in chat, sleep augmentation, and eval harness.
- **2025-12-06:** Adjusted sampling to use `make_sampler` (compat with installed mlx-lm) replacing `TopPSampler`.
- **2025-12-06:** Handled mlx_lm GenerationResponse objects by extracting `.text` in chat, sleep augmentation, and eval harness streaming loops.
- **2025-12-06:** Added `--temp`/`--top-p` CLI options back to `sleep.py` (used for augmentation sampler) to align with `run_cycle.py` invocation.
- **2025-12-06:** Swapped training backend to `python -m mlx_lm.lora --train`; write datasets as JSONL under `adapters/nightly_update/dset/train.jsonl` to match mlx-lm 0.28.4 tooling.
- **2025-12-06:** Guarded adapter loading to require `adapter_config.json` (avoids missing-adapter errors) across chat, sleep, eval_loss, eval_harness, and run_cycle messages about base-model evals.
- **2025-12-06:** Removed unsupported `--lora-r/--lora-alpha` when calling `mlx_lm.lora` to match the installed CLI.
- **2025-12-06:** Ensure mlx_lm.lora has required splits by writing train/valid/test JSONL (valid/test reuse first sample if limited data).
- **2025-12-06:** Added synthetic starter dataset at `data/synthetic.jsonl` for bootstrapping fine-tuning before real chats accumulate.
- **2025-12-06:** Added `build_synthetic.py` generator, and `--reset-adapter` flag in sleep/run_cycle to optionally wipe adapters before training.
- **2025-12-06:** Fixed reset flow: when `--reset-adapter` is used, wipe adapter dir before writing train/valid/test splits to avoid empty dataset errors.
- **2025-12-06:** Added `generate_gemini_dataset.py` (OpenAI-compatible Gemini client) and `openai` dependency to build larger synthetic train/eval JSONL via GEMINI_API_KEY.
- **2025-12-07:** Updated `generate_gemini_dataset.py` to use native Gemini API (`google-genai`) and `gemini-2.5-flash-preview-05-20` model; replaced `openai` with `google-genai` in requirements.txt; added `--verbose` flag for progress output.

## In-Flight Planning
- **Router upgrade (LoRA vs RAG):** Design a lightweight classifier to replace heuristics in `classify_for_rag`:
  - Features: encoder embedding of concatenated messages (or pooled per-turn), content length, code/url/json flags, digit ratio.
  - Model: tiny logistic regression or shallow MLP trained on a labeled seed (bootstrap with heuristics, then human corrections).
  - Behavior: abstain threshold; log route + score per record; default to dual-path (LoRA+RAG) when uncertain.
  - Instrumentation: write route decisions to `logs/experiments.jsonl` with counts/ratios for monitoring.
- **RAG hygiene:** Plan to improve ingestion quality and retrieval robustness:
  - Chunking: split long turns into ~512–1k char chunks with 10–20% overlap before embedding.
  - Dedup/compaction: hash chunks, skip near-duplicates, allow periodic rebuild (`rag_clear`) with compaction.
  - Metadata: store turn-level IDs, timestamps, tags, and source adapter tag for audit.
  - Eval: add a retrieval benchmark (seed facts with queries), log recall@k to `logs/eval_summary.json`.
  - Maintenance: background job to prune stale/low-hit items and re-embed after model changes.
- **Safety & privacy filters:** Protect logs before storage/training.
  - PII pass: regex + simple NER to detect emails, phones, addresses; redact or mask by default.
  - Secrets/code guard: optional blocklist for keys/tokens; allow user tag `#no-train` or `#no-rag` to skip routing.
  - Toxicity/off-policy: lightweight classifier to drop/flag unsafe content before RAG/LoRA.
  - Audit logging: record redaction actions and filtered counts to `logs/experiments.jsonl`.
- **Eval automation:** Harden evaluation and regression catching.
  - Harness expansion: add tasks for code correctness, long-context recall, safety refusal; keep tags for per-domain scores.
  - Regression gates: in `run_cycle.py`, fail/abort if loss rises vs previous tag or harness drops by >1 task.
  - Trend tracking: emit rolling CSV/JSONL with per-task scores and plot hooks; surface delta in console summary.
  - Chat eval: add small golden chat transcripts; measure style match via BLEU/BERTScore or simple perplexity.
- **Adapter management:** Version, roll back, and audit adapters.
  - Naming: store adapters under `adapters/{date-tag}`; keep symlink `adapters/nightly_update` pointing to active.
  - Metadata: write `manifest.json` with hyperparams, dataset stats, and eval deltas beside adapters.
  - Retention: keep last N adapters (e.g., 5) and delete older unless pinned.
  - Rollback: CLI flag `--adapter-tag` for chat/sleep/eval to pick a specific adapter; add `--rollback` helper to repoint symlink.
- **Ops polish:** Make runs observable and resilient.
  - Timing/VRAM: wrap train/eval with timers; log peak memory (mlx/cupti analog) and tokens/sec.
  - Resume: detect partial `adapters/{tag}`; allow resume from last checkpoint or auto-clean on failure.
  - Dry-run: mode that routes, augments, and writes datasets without training; prints planned actions.
  - Alerts: simple notification on failure/threshold breach (e.g., macOS notification or stdout summary).
- **Docs updates:** Clarify flow and choices.
  - Diagram: quick data-flow from chat log → router → RAG/LoRA → eval → adapter selection.
  - Decision table: when to toggle `--rag`, `--rag-clear`, `--reset-adapter`, `--augment`, LR/iters presets.
  - Highlight new safeguards (routing classifier, redaction, regression gates) and adapter versioning.

