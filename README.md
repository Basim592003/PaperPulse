# PaperPulse

PaperPulse is a multi-agent pipeline that turns a research question into a synthesized digest of relevant arXiv papers. A set of agents run sequentially under an orchestrator that loops until an evaluator agent passes the output (or hits `MAX_ITERATIONS`). All LLM calls go to [Groq](https://groq.com/); papers, embeddings, extractions, and run history persist in [Supabase](https://supabase.com/) (pgvector); runs are tracked in MLflow via [DagsHub](https://dagshub.com/). A FastAPI service exposes the pipeline over HTTP with async job execution.

## How it works

The orchestrator holds a single shared `state` dict and threads it through the agents in order. If the evaluator scores the digest below `EVAL_THRESHOLD`, it loops — re-running only the relevant downstream stages based on the evaluator's `suggested_action` (`fetch_more_papers`, `re_extract`, or default `re_synthesize`).

```
query
  │
  ▼
┌──────────┐   ┌───────────┐   ┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│  Search  │──▶│ Critique  │──▶│ Extraction │──▶│ Contradiction │──▶│ Synthesis  │──▶│ Evaluator  │
└──────────┘   └───────────┘   └────────────┘   └───────────────┘   └────────────┘   └─────┬──────┘
 arXiv search   score & keep    PDF → struct.    find conflicting    structured        pass? ──▶ digest
 + embed + DB   top 5 papers    extraction       claims             digest            fail? ──▶ loop back
```

| Agent | Input | Action | Output |
|-------|-------|--------|--------|
| **Search** | `query` | Expand query → arXiv search → dedupe → save papers + embeddings to DB | candidate papers |
| **Critique** | `query`, `papers` | Score each paper via Groq, drop low scores, keep top 5 | shortlist |
| **Extraction** | `paper` (per shortlisted) | Download PDF → parse text → Groq structured extraction → save to DB | extraction dict |
| **Contradiction** | `extractions` | Send key claims to Groq, find contradicting pairs | contradictions |
| **Synthesis** | `extractions`, `contradictions` | Groq produces a structured digest | digest |
| **Evaluator** | `digest`, `extractions`, `query` | Score digest; if `overall < 0.75`, return feedback to loop | eval scores + `suggested_action` |

See `Notes.txt` for the full shared-state contract and each agent's exact I/O.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
# Required
GROQ_API_KEY=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...

# Optional — embeddings via Hugging Face API
HF_API_KEY=...

# Optional — MLflow tracking (silently no-ops if unset)
DAGSHUB_TOKEN=...
DAGSHUB_USER=...
DAGSHUB_REPO=...
MLFLOW_EXPERIMENT=paperpulse
```

The Supabase database needs a `papers` table (with an `embedding` column), an `extractions` table, a `runs` table for job history, and the pgvector `match_papers` RPC used for semantic search.

## Running

```bash
# Run the full pipeline end-to-end (optional query arg)
python agents/orchestrator.py "attention in transformer models"

# Start the FastAPI server
uvicorn api.main:app --reload

# Run batch evaluation across the benchmark queries
python eval/run_batch.py

# Scratch script for exercising individual tools/agents
python tester.py
```

## API

Run with `uvicorn api.main:app --reload` from the repo root.

| Method & path | Description |
|---------------|-------------|
| `GET /health` | Liveness check |
| `POST /runs` | Submit a query; returns `job_id` immediately, pipeline runs in the background |
| `GET /runs` | List run history |
| `GET /runs/{job_id}` | Poll job status (`running` / `done` / `failed`); returns trimmed result when done |
| `GET /papers/search?q=...&k=10` | Semantic search over cached papers via pgvector |
| `GET /papers/{paper_id}` | Fetch one paper row |
| `GET /papers/{paper_id}/extraction` | Fetch the stored extraction for a paper |

Jobs are persisted in the Supabase `runs` table, so they survive restarts and form a queryable history.

## Configuration

All tunables live in `config.py` (loaded from `.env`):

| Setting | Default | Meaning |
|---------|---------|---------|
| `DEV_MODEL` | `llama-3.1-8b-instant` | Faster/cheaper Groq model |
| `PROD_MODEL` | `llama-3.3-70b-versatile` | Higher-quality Groq model |
| `CRITIQUE_THRESHOLD` | `5` | Minimum critique score to keep a paper |
| `EVAL_THRESHOLD` | `0.75` | Evaluator score required to pass |
| `TOP_K_PAPERS` | `5` | Papers kept after critique |
| `MAX_ARXIV_RESULTS` | `20` | arXiv results fetched per query |
| `MAX_WORKERS` | `5` | Concurrency for parallel extraction |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model (384-dim) |
