import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import run_phase1, run_phase2
from tools.embedding_tool import get_embedding
from tools.db_tool import (
    semantic_search, get_paper, get_extraction,
    create_run, set_run_result, set_run_failed, get_run, list_runs,
    set_run_awaiting, set_run_resuming,
)

app = FastAPI(title="PaperPulse API")

# Allow browser frontends to call this API. "*" permits any origin, which is
# fine for local development. Before deploying publicly, replace allow_origins
# with the specific frontend URL(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Job storage.
#
# Jobs are persisted in the Supabase `runs` table so they survive restarts and
# can be listed as history. Everything goes through these small helpers, which
# delegate to tools/db_tool.py.
# ---------------------------------------------------------------------------
def _create_job(query: str) -> str:
    job_id = uuid.uuid4().hex
    create_run(job_id, query)
    return job_id


def _set_job_result(job_id: str, result: dict) -> None:
    set_run_result(job_id, result)


def _set_job_failed(job_id: str, error: str) -> None:
    set_run_failed(job_id, error)


def _get_job(job_id: str) -> dict | None:
    return get_run(job_id)


def _trim_state(state: dict) -> dict:
    """Keep just the useful, presentable parts of the orchestrator state."""
    return {
        "query": state["query"],
        "iterations": state["iterations"],
        "digest": state["digest"],
        "eval_score": state["eval_score"],
        "shortlist": [
            {"id": p["id"], "title": p["title"]} for p in state.get("critiqued", [])
        ],
    }


def _strip_heavy(paper: dict) -> dict:
    """Drop full_text + embedding from a paper dict; both re-fetch from the DB
    on demand, so the checkpoint blob stays small (a few KB, not megabytes)."""
    return {k: v for k, v in paper.items() if k not in ("full_text", "embedding")}


def _trim_partial_state(state: dict) -> dict:
    """The phase-1 checkpoint persisted between critique and approval.

    Keeps everything phase 2 needs to resume: query, the accumulated `papers`
    (so the eval loop's fetch_more_papers branch can dedup against them), and
    the full `critiqued` survivor list (with scores, for the human to review).
    Heavy fields are stripped — extraction re-fetches full_text from the DB.
    """
    return {
        "query": state["query"],
        "papers": [_strip_heavy(p) for p in state.get("papers", [])],
        "critiqued": [_strip_heavy(p) for p in state.get("critiqued", [])],
        "iterations": 0,
        "history": [],
    }


def _shortlist_preview(critiqued: list) -> list:
    """What the client sees while a run awaits approval: enough to decide."""
    return [
        {
            "id": p["id"],
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "scores": p.get("scores", {}),
        }
        for p in critiqued
    ]


def _run_pipeline(job_id: str, query: str) -> None:
    """Background worker for phase 1: search + critique, then pause for the
    human. The run sits in awaiting_shortlist_approval until POST .../approve."""
    try:
        state = run_phase1(query)
        set_run_awaiting(job_id, _trim_partial_state(state))
    except Exception as e:
        _set_job_failed(job_id, str(e))


def _resume_pipeline(job_id: str, partial_state: dict) -> None:
    """Background worker for phase 2: extract -> ... -> evaluate on the
    human-approved shortlist, then finalize the run."""
    try:
        state = run_phase2(partial_state)
        _set_job_result(job_id, _trim_state(state))
    except Exception as e:
        _set_job_failed(job_id, str(e))


class RunRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs")
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    job_id = _create_job(req.query)
    background_tasks.add_task(_run_pipeline, job_id, req.query)
    return {"job_id": job_id, "status": "running"}


@app.get("/runs")
def list_runs_endpoint(limit: int = 20):
    return {"runs": list_runs(limit)}


@app.get("/runs/{job_id}")
def get_run_endpoint(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")

    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done" and job.get("result"):
        response.update(job["result"])
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    return response


@app.get("/papers/search")
def search_papers(q: str, k: int = 10):
    embedding = get_embedding(q)
    results = semantic_search(embedding, match_count=k)
    return {"query": q, "results": results}


@app.get("/papers/{paper_id}")
def read_paper(paper_id: str):
    paper = get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@app.get("/papers/{paper_id}/extraction")
def read_extraction(paper_id: str):
    extraction = get_extraction(paper_id)
    if extraction is None:
        raise HTTPException(status_code=404, detail="extraction not found")
    return extraction


