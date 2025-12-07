## Circadia: Teach Your Model While It Sleeps (MLX, Apple Silicon)

### The vibe
Circadia is a tiny nightly ritual for your local Llama (3.2B, 4-bit MLX). By day you chat; by night it “sleeps”: replaying the day, distilling style into LoRA weights, and shelving facts into RAG. Like human sleep:
- **Procedural** (how you speak) gets stronger → LoRA.
- **Episodic** (what was said) gets replayed → RAG.
- You wake up with yesterday’s feel, without forgetting how to math.

### Why it’s novel
- Not just “add RAG” or “run LoRA”: Circadia **routes** your own chat turns automatically—code/URLs/long snippets into RAG (verbatim), tone/preferences into LoRA (compressed).
- No doc drop required: only input is your chats (paste code, logs, search results there). Sleep decides what to retrieve vs what to fine-tune.
- Designed for Apple Silicon (MLX) with small, fast nightly jobs.

### Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Daytime: chat (optionally with RAG)
```bash
# Plain chat (loads adapter if present)
python chat.py

# Chat with retrieval of past chats
python chat.py --rag --rag-verbose
```
Logs live in `data/memories.jsonl`. RAG index persists in `data/rag_memory`.

### Nighttime: sleep (LoRA + RAG routing)
```bash
# Gentle nightly update (recommended sweet spot)
python sleep.py \
  --memory-file data/memories.jsonl \
  --adapter-dir adapters/nightly_update \
  --iters 50 --learning-rate 2e-5 \
  --rag --rag-dir data/rag_memory --rag-clear \
  --reset-adapter
```
Heuristics: code/URLs/JSON-ish/long/numeric-heavy → RAG. Everything still trains LoRA, so behavior stays stable while facts stay retrievable.

### One-command daily cycle
```bash
python run_cycle.py --tag day1 \
  --sleep-iters 50 --sleep-learning-rate 2e-5 \
  --sleep-reset-adapter --sleep-augment
```
Runs pre-eval, train, post-eval; logs to `logs/experiments.jsonl`, losses to `logs/loss.csv`, harness summaries to `logs/eval_summary.json`.

### Evaluate
```bash
# Loss on your chat logs
python eval_loss.py --max-samples 10 --out-csv logs/loss.csv --tag day1-pre

# Capability harness (math, reasoning, factual, format, summarization, style, context)
python eval_harness.py --tag day1-pre
```
Rerun with `...-post` to see deltas.

### Optional datasets (for play)
```bash
python download_dataset.py --dataset dolly   # 15K, general
python download_dataset.py --dataset alpaca  # 52K, classic
```
Defaults still use your `data/memories.jsonl`.

### What’s inside
- `chat.py` — chat with optional RAG retrieval.
- `sleep.py` — QLoRA + RAG ingestion/splitting, optional augmentation.
- `run_cycle.py` — pre/train/post driver with experiment logging.
- `rag.py` — ChromaDB-backed retrieval for past chats.
- `download_dataset.py` — grab Dolly/Alpaca in ShareGPT format.
- `logs/experiments.jsonl` — hyperparams + pre/post metrics per run.

### Notes from the lab
- Gentle LR/iters matter: 2e-5, 50 iters dropped loss ~66% with only ~1 harness task regression.
- Small, focused synthetic chats beat larger Dolly for preserving general skills (less interference).
- RAG + LoRA separation keeps style adaptive and facts verbatim, mimicking the “sleep” split between procedural and episodic memory.
