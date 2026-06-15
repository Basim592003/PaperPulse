import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from agents.orchestrator import orchestrator_run
from tools.embedding_tool import get_embedding
from tools.db_tool import semantic_search, get_paper, get_extraction

app = FastAPI(title="PaperPulse API")


# ---------------------------------------------------------------------------
# Job storage.
#
# For now jobs live in this in-memory dict. Everything goes through the small
# helper functions below so that swapping this for a Supabase `runs` table
# later only touches these helpers, not the endpoints.
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}


def _create_job(query: str) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "query": query, "result": None, "error": None}
    return job_id


def _set_job_result(job_id: str, result: dict) -> None:
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["result"] = result


def _set_job_failed(job_id: str, error: str) -> None:
    JOBS[job_id]["status"] = "failed"
    JOBS[job_id]["error"] = error


def _get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)


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


def _run_pipeline(job_id: str, query: str) -> None:
    """Background worker: runs the (slow, blocking) orchestrator."""
    try:
        state = orchestrator_run(query)
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


@app.get("/runs/{job_id}")
def get_run(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")

    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        response.update(job["result"])
    elif job["status"] == "failed":
        response["error"] = job["error"]
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


