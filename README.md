## Circadian Rhythm (MLX)
Daytime chats are distilled nightly on Apple Silicon (MLX) via QLoRA, now paired with RAG to split “how to respond” (LoRA) from “what was said” (retrieval).

### Setup
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- Layout: `data/` for logs/RAG, `adapters/nightly_update/` for LoRA weights.

### Daytime Chat
- `python chat.py` (loads adapter if present).
- Logs every turn to `data/memories.jsonl`.
- RAG on/off: `python chat.py --rag --rag-verbose` (retrieves past chats from `data/rag_memory`).

### Nightly “Sleep” (LoRA + RAG routing)
- Base: `python sleep.py --iters 100 --batch-size 1`
- Optional: `--augment ...` for semisynthetic expansion.
- RAG routing: `--rag --rag-dir data/rag_memory --rag-clear`
  - Heuristics send code/URLs/long or JSON-ish chats to RAG.
  - Everything still feeds LoRA; RAG holds verbatim facts, LoRA learns style/preferences.

### Loss & Capability Eval
- Loss: `python eval_loss.py --max-samples 10 --out-csv logs/loss.csv --tag dayX-pre|post`
- Harness: `python eval_harness.py --tag dayX-pre|post` (writes to `logs/eval_results.jsonl` + `logs/eval_summary.json`), categories: math, reasoning, factual, format, summarization, style, context.

### One-Command Cycle
- `python run_cycle.py --tag dayX --sleep-iters 100 --sleep-reset-adapter`
- Runs pre-loss, pre-harness, sleep, post-loss, post-harness. Adapter reset happens before pre-eval for a fair baseline. Losses go to `logs/loss.csv`; summaries to `logs/eval_summary.json`.

### Experiment Logging
- Every run of `run_cycle.py` logs to `logs/experiments.jsonl`:
  - Hyperparams, pre/post loss, harness scores, deltas.

### Datasets (optional)
- Quick downloads in ShareGPT format:
  - `python download_dataset.py --dataset dolly` → `data/dolly_15k.jsonl` (15K)
  - `python download_dataset.py --dataset alpaca` → `data/alpaca_cleaned.jsonl` (52K)
  - `--dataset all` to grab everything; defaults use your `data/memories.jsonl`.

### Testing
- `pytest`

### Files
- `chat.py` — chat + optional RAG retrieval.
- `sleep.py` — QLoRA, optional augmentation, and RAG ingestion/splitting.
- `run_cycle.py` — full pre/train/post pipeline with experiment logging.
- `rag.py` — ChromaDB-backed retrieval for chat history.
- `download_dataset.py` — pull standard instruction-tuning datasets.
- `STATE.md` — project log.
