import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EVAL_THRESHOLD
from agents.search_agent import search_agent_run
from agents.critique_agent import critique_agent_run
from agents.extraction_agent import extraction_agent_run
from agents.contradiction_agent import contradiction_agent_run
from agents.synthesis_agent import synthesis_agent_run
from agents.evaluator_agent import evaluator_agent_run
from tracking.mlflow_logger import track_run

MAX_ITERATIONS = 2


def _init_state(query: str) -> dict:
    return {
        "query": query,
        "papers": [],
        "critiqued": [],
        "extractions": {},
        "contradictions": [],
        "digest": None,
        "eval_score": None,
        "iterations": 0,
        "history": [],
    }


def _do_search(state: dict) -> None:
    print(f"\n[orchestrator] -> search")
    result = search_agent_run(state["query"])
    existing = {p["id"]: p for p in state["papers"]}
    for p in result["papers"]:
        existing[p["id"]] = p
    state["papers"] = list(existing.values())


def _do_critique(state: dict) -> None:
    print(f"\n[orchestrator] -> critique")
    result = critique_agent_run(state["query"], state["papers"])
    state["critiqued"] = result["shortlist"]


def _do_extract(state: dict, force_refresh: bool = False, feedback: str = "") -> None:
    print(f"\n[orchestrator] -> extract")
    result = extraction_agent_run(state["critiqued"], force_refresh=force_refresh, feedback=feedback)
    state["extractions"] = result["extractions"]


def _do_contradictions(state: dict) -> None:
    print(f"\n[orchestrator] -> contradictions")
    result = contradiction_agent_run(state["extractions"])
    state["contradictions"] = result["contradictions"]


def _do_synthesize(state: dict) -> None:
    print(f"\n[orchestrator] -> synthesize")
    result = synthesis_agent_run(
        state["query"],
        state["critiqued"],
        state["extractions"],
        state["contradictions"],
    )
    state["digest"] = result["digest"]


def _do_evaluate(state: dict) -> dict:
    print(f"\n[orchestrator] -> evaluate")
    result = evaluator_agent_run(
        state["query"],
        state["digest"],
        state["critiqued"],
        state["extractions"],
    )
    state["eval_score"] = result["evaluation"]
    return result["evaluation"]


def orchestrator_run(query: str) -> dict:
    print(f"\n=== orchestrator.run query={query!r} threshold={EVAL_THRESHOLD} ===")
    state = _init_state(query)

    with track_run(query) as logger:
        _do_search(state)
        _do_critique(state)
        _do_extract(state)
        _do_contradictions(state)
        _do_synthesize(state)
        evaluation = _do_evaluate(state)

        while not evaluation["passed"] and state["iterations"] < MAX_ITERATIONS:
            state["iterations"] += 1
            action = evaluation.get("suggested_action", "re_synthesize")
            state["history"].append({
                "iteration": state["iterations"],
                "overall": evaluation["overall"],
                "weakest": evaluation.get("weakest_dimension"),
                "action": action,
            })
            print(f"\n[orchestrator] iteration {state['iterations']}/{MAX_ITERATIONS} — "
                  f"overall={evaluation['overall']} action={action}")

            if action == "fetch_more_papers":
                _do_search(state)
                _do_critique(state)
                _do_extract(state)
                _do_contradictions(state)
                _do_synthesize(state)
            elif action == "re_extract":
                ca_feedback = (evaluation.get("feedback") or {}).get("citation_accuracy", "")
                _do_extract(state, force_refresh=True, feedback=ca_feedback)
                _do_contradictions(state)
                _do_synthesize(state)
            else:
                _do_synthesize(state)

            evaluation = _do_evaluate(state)

        logger.log_state(state)

    print(f"\n=== orchestrator.done passed={evaluation['passed']} "
          f"overall={evaluation['overall']} iterations={state['iterations']} ===")
    return state


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    q = sys.argv[1] if len(sys.argv) > 1 else "semantic segmentation scene understanding"
    final = orchestrator_run(q)
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("\n=== FINAL DIGEST ===")
    print(json.dumps(final["digest"], indent=2))
    print("\n=== EVAL ===")
    print(json.dumps(final["eval_score"], indent=2))
    print(f"\n=== TIME ===")
    print(f"Total: {minutes}m {seconds}s")