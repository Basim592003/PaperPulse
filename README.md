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

## Human-in-the-loop (shortlist approval)

Over the API, the pipeline runs in **two phases** with a human approval gate at the highest-leverage point — the critique shortlist. Everything downstream (extraction, contradictions, synthesis) inherits whatever the critique agent shortlisted, so a single wrong inclusion silently skews the whole digest. Pausing here lets a human drop off-topic papers in a few seconds before any expensive work runs.

```
POST /runs ─▶ Search ─▶ Critique ──▶ [ pause: awaiting_shortlist_approval ]
                                              │  human reviews 5 papers,
                                              │  approves a subset
                                              ▼
              POST /runs/{id}/approve ─▶ Extraction ─▶ Contradiction ─▶ Synthesis ─▶ Evaluator ─▶ done
```

1. **`POST /runs`** runs phase 1 (search + critique) in the background, then parks the run at status `awaiting_shortlist_approval` instead of continuing.
2. **`GET /runs/{job_id}`** returns the shortlist for review — each paper's `id`, `title`, `abstract`, and critique `scores`.
3. **`POST /runs/{job_id}/approve`** with `{"approved_ids": [...]}` filters the shortlist to the chosen subset and launches phase 2 (extraction → … → evaluator). The run flips back to `running`, then `done`.

Notes:
- You can only **keep or drop** from the shortlisted papers — `approved_ids` must be a non-empty subset of the ids returned in step 2 (an empty list or an unknown id returns `400`).
- The gate is **only the first shortlist**. If the evaluator loops back with `fetch_more_papers`, the re-derived shortlist is used autonomously (no second pause).
- Approving twice, or approving a run that isn't awaiting, returns `409` (a compare-and-swap guards the resume transition).
- The **CLI path is unaffected**: `python agents/orchestrator.py "…"` (and the eval batch) run straight through with no human gate.

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

The Supabase database needs a `papers` table (with an `embedding` column), an `extractions` table, a `runs` table for job history, and the pgvector `match_papers` RPC used for semantic search. The `runs` table also needs a `partial_state JSONB` column, which holds the phase-1 checkpoint while a run awaits shortlist approval:

```sql
ALTER TABLE runs ADD COLUMN IF NOT EXISTS partial_state JSONB;
```

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
| `POST /runs` | Submit a query; returns `job_id` immediately, runs search + critique in the background, then pauses for approval |
| `GET /runs` | List run history |
| `GET /runs/{job_id}` | Poll job status (`running` / `awaiting_shortlist_approval` / `done` / `failed`); returns the shortlist while awaiting, and the trimmed result when done |
| `POST /runs/{job_id}/approve` | Approve a subset of the shortlist (`{"approved_ids": [...]}`) to resume the run through extraction → synthesis → evaluation |
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
