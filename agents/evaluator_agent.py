import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, PROD_MODEL, EVAL_THRESHOLD

groq_client = Groq(api_key=GROQ_API_KEY)

EVAL_PROMPT = """You are a research digest evaluator. Score the given digest on three dimensions, each from 0.0 to 1.0.

Original query: {query}

Digest to evaluate:
{digest_block}

Source extractions (ground truth — what claims actually came from each paper):
{extractions_block}

Score these three dimensions:
1. citation_accuracy — Do the digest's summary and key_findings trace back to claims/results actually present in the source extractions? Penalize fabricated facts, invented numbers, or claims attributed to papers that don't support them.
2. coverage — Does the digest actually answer the original query? Penalize off-topic content, missing major themes, or digests that ignore parts of the query.
3. coherence — Is the digest internally consistent? Penalize self-contradictions, repeated points, vague or contradictory recommendations.

Return ONLY a JSON object with this exact shape, NOTHING else:
{{
  "citation_accuracy": <float 0.0-1.0>,
  "coverage": <float 0.0-1.0>,
  "coherence": <float 0.0-1.0>,
  "feedback": {{
    "citation_accuracy": "1-2 sentences on what hurt this score (or 'ok' if >=0.85)",
    "coverage": "1-2 sentences on what hurt this score (or 'ok' if >=0.85)",
    "coherence": "1-2 sentences on what hurt this score (or 'ok' if >=0.85)"
  }},
  "weakest_dimension": "citation_accuracy | coverage | coherence",
  "suggested_action": "fetch_more_papers | re_extract | re_synthesize | none"
}}

Rules:
- Be strict but fair. A perfect score requires the digest to be fully grounded, on-topic, and consistent.
- weakest_dimension is the dimension with the lowest score (used by the orchestrator to decide what to retry).
- suggested_action: 'fetch_more_papers' if coverage is the problem, 're_extract' if citations are wrong, 're_synthesize' if coherence is the problem, 'none' if overall is acceptable."""


def _format_extractions_block(papers: list[dict], extractions: dict) -> str:
    paper_meta = {p["id"]: p for p in papers}
    lines = []
    for pid, ext in extractions.items():
        title = paper_meta.get(pid, {}).get("title", "")
        lines.append(f"Paper {pid} — {title}")
        for c in ext.get("key_claims") or []:
            lines.append(f"  - {c}")
        if ext.get("results"):
            lines.append(f"  results: {ext['results']}")
        lines.append("")
    return "\n".join(lines).strip()


def _empty_eval(reason: str) -> dict:
    return {
        "citation_accuracy": 0.0,
        "coverage": 0.0,
        "coherence": 0.0,
        "overall": 0.0,
        "passed": False,
        "feedback": {"citation_accuracy": reason, "coverage": reason, "coherence": reason},
        "weakest_dimension": "coverage",
        "suggested_action": "fetch_more_papers",
        "error": reason,
    }


def evaluator_agent_run(query: str, digest: dict, papers: list[dict], extractions: dict) -> dict:
    print(f"\n[evaluator.run] scoring digest (threshold={EVAL_THRESHOLD})")

    if not digest or digest.get("error"):
        print("[evaluator.run] empty/errored digest, returning failing score")
        return {"evaluation": _empty_eval("empty digest")}

    prompt = EVAL_PROMPT.format(
        query=query,
        digest_block=json.dumps(digest, indent=2),
        extractions_block=_format_extractions_block(papers, extractions),
    )
    response = groq_client.chat.completions.create(
        model=PROD_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"[evaluator.run] JSON parse failed: {text[:200]!r}")
        return {"evaluation": _empty_eval("parse error")}

    ca = float(data.get("citation_accuracy", 0.0) or 0.0)
    cov = float(data.get("coverage", 0.0) or 0.0)
    coh = float(data.get("coherence", 0.0) or 0.0)
    overall = round((ca + cov + coh) / 3.0, 3)
    passed = overall >= EVAL_THRESHOLD

    evaluation = {
        "citation_accuracy": ca,
        "coverage": cov,
        "coherence": coh,
        "overall": overall,
        "passed": passed,
        "feedback": data.get("feedback", {}),
        "weakest_dimension": data.get("weakest_dimension", ""),
        "suggested_action": data.get("suggested_action", "none"),
    }

    print(f"[evaluator.run] ca={ca} cov={cov} coh={coh} overall={overall} "
          f"{'PASS' if passed else 'FAIL'} — action={evaluation['suggested_action']}")
    return {"evaluation": evaluation}


if __name__ == "__main__":
    from agents.search_agent import search_agent_run
    from agents.critique_agent import critique_agent_run
    from agents.extraction_agent import extraction_agent_run
    from agents.contradiction_agent import contradiction_agent_run
    from agents.synthesis_agent import synthesis_agent_run

    q = "hentavirus"
    search_result = search_agent_run(q)
    critique_result = critique_agent_run(q, search_result["papers"])
    extraction_result = extraction_agent_run(critique_result["shortlist"])
    contradiction_result = contradiction_agent_run(extraction_result["extractions"])
    synthesis_result = synthesis_agent_run(
        q,
        critique_result["shortlist"],
        extraction_result["extractions"],
        contradiction_result["contradictions"],
    )
    eval_result = evaluator_agent_run(
        q,
        synthesis_result["digest"],
        critique_result["shortlist"],
        extraction_result["extractions"],
    )
    print("\n=== EVALUATION ===")
    print(json.dumps(eval_result["evaluation"], indent=2))
