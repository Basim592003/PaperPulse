import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, DEV_MODEL, CRITIQUE_THRESHOLD, TOP_K_PAPERS

groq_client = Groq(api_key=GROQ_API_KEY)

CRITIQUE_PROMPT = """You are a research critique agent. Score the paper below against the user's query on three axes:

- relevance: how directly the paper addresses the user's query (0-10)
- quality: methodological-quality signals from the abstract — clear method, evaluation, baselines (0-10)
- recency: how current the work is given its publication date (0-10, where 1 = very recent, 0 = >10 years old)

Return ONLY a JSON object, NOTHING else:
{{"relevance": 0.0, "quality": 0.0, "recency": 0.0, "reason": "one short sentence"}}

User query: {query}

Paper title: {title}
Published: {published}
Abstract: {abstract}"""


def _score_paper(user_query: str, paper: dict) -> dict:
    prompt = CRITIQUE_PROMPT.format(
        query=user_query,
        title=paper.get("title", ""),
        published=paper.get("published_date") or paper.get("published", ""),
        abstract=paper.get("abstract", "")[:2000],
    )
    response = groq_client.chat.completions.create(
        model=DEV_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        print(f"[_score_paper] JSON parse failed for {paper.get('id')}: {text!r}")
        return {"relevance": 0.0, "quality": 0.0, "recency": 0.0, "overall": 0.0, "reason": "parse error"}

    r = float(scores.get("relevance", 0))
    q = float(scores.get("quality", 0))
    rec = float(scores.get("recency", 0))
    overall = round((r + q + rec) / 3, 3)
    return {
        "relevance": r,
        "quality": q,
        "recency": rec,
        "overall": overall,
        "reason": scores.get("reason", ""),
    }


def critique_agent_run(user_query: str, papers: list[dict]) -> dict:
    print(f"\n[critique.run] scoring {len(papers)} papers against: {user_query!r}")
    scored = []
    for p in papers:
        s = _score_paper(user_query, p)
        print(f"[critique.run]   {p.get('id')}: overall={s['overall']} "
              f"(r={s['relevance']}, q={s['quality']}, rec={s['recency']}) — {s['reason']}")
        scored.append({**p, "scores": s})

    survivors = [p for p in scored if p["scores"]["overall"] >= CRITIQUE_THRESHOLD]
    print(f"[critique.run] {len(survivors)}/{len(scored)} survived threshold {CRITIQUE_THRESHOLD}")

    survivors.sort(key=lambda p: p["scores"]["overall"], reverse=True)
    shortlist = survivors[:TOP_K_PAPERS]
    print(f"[critique.run] shortlisted top {len(shortlist)}: {[p['id'] for p in shortlist]}")

    return {
        "shortlist": shortlist,
        "shortlist_ids": [p["id"] for p in shortlist],
        "all_scored": scored,
    }


if __name__ == "__main__":
    from agents.search_agent import search_agent_run 
    search_result = search_agent_run("denoising diffusion image inpainting")
    critique_result = critique_agent_run("denoising diffusion image inpainting", search_result["papers"])
    print(f"\nFinal shortlist IDs: {critique_result['shortlist_ids']}")
    