import sys
import os
import re
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, DEV_MODEL
from tools.db_tool import save_extraction, get_extraction, get_paper_full_text

groq_client = Groq(api_key=GROQ_API_KEY)

EXTRACTION_PROMPT = """You are a research extraction agent. Read the paper text below and extract a structured summary.

Return ONLY a JSON object with this exact shape, NOTHING else:
{{
  "methodology": "1-3 sentence description of the approach/method",
  "key_claims": ["claim 1", "claim 2", "..."],
  "results": "1-3 sentence summary of quantitative/qualitative results",
  "limitations": ["limitation 1", "limitation 2", "..."]
}}

Rules:
- key_claims and limitations are JSON arrays of short strings (one statement each).
- methodology and results are plain strings.
- Be faithful to the paper — DO NOT invent numbers or claims.

Paper title: {title}
Abstract: {abstract}

Paper text (may be truncated):
{full_text}"""

# Groq context budget for the paper body. Abstract + prompt overhead is small.
_MAX_TEXT_CHARS = 12000
# How much of the budget to reserve for the tail (conclusion/results at the end).
_TAIL_CHARS = 3000


def _prepare_text(full_text: str) -> str:
    """Strip front matter + references and keep the most informative content.

    Naive [:N] truncation wastes budget on the title page (authors, emails,
    affiliations) and can run out before reaching Results/Conclusion. We drop
    the boilerplate head, cut the References tail, then keep the body head plus
    the conclusion tail so the ending isn't lost.
    """
    text = full_text

    # Drop everything before the Abstract/Introduction (title page boilerplate).
    head_match = re.search(r"\b(abstract|introduction)\b", text, re.IGNORECASE)
    if head_match and head_match.start() < 3000:
        text = text[head_match.start():]

    # Cut the References/Bibliography tail (last occurrence).
    ref_matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n",
                                   text, re.IGNORECASE))
    if ref_matches:
        text = text[:ref_matches[-1].start()]

    text = text.strip()

    if len(text) <= _MAX_TEXT_CHARS:
        return text

    # Keep the head plus the conclusion tail so we don't lose the ending.
    head_budget = _MAX_TEXT_CHARS - _TAIL_CHARS
    return text[:head_budget] + "\n...\n" + text[-_TAIL_CHARS:]


def _empty_extraction(reason: str) -> dict:
    return {
        "methodology": "",
        "key_claims": [],
        "results": "",
        "limitations": [],
        "error": reason,
    }


def _extract_paper(paper: dict) -> dict:
    full_text = paper.get("full_text") or ""
    if not full_text:
        full_text = get_paper_full_text(paper["id"])
    full_text = _prepare_text(full_text)
    if not full_text:
        print(f"[_extract_paper] {paper.get('id')}: no full_text, skipping")
        return _empty_extraction("no full_text")

    prompt = EXTRACTION_PROMPT.format(
        title=paper.get("title", ""),
        abstract=(paper.get("abstract") or "")[:2000],
        full_text=full_text,
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
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"[_extract_paper] JSON parse failed for {paper.get('id')}: {text[:200]!r}")
        return _empty_extraction("parse error")

    return {
        "methodology": str(data.get("methodology", "") or ""),
        "key_claims": list(data.get("key_claims", []) or []),
        "results": str(data.get("results", "") or ""),
        "limitations": list(data.get("limitations", []) or []),
    }


def extraction_agent_run(papers: list[dict]) -> dict:
    print(f"\n[extraction.run] extracting from {len(papers)} papers")
    extractions = {}
    for p in papers:
        pid = p["id"]
        cached = get_extraction(pid)
        if cached:
            print(f"[extraction.run]   {pid}: using cached extraction")
            extractions[pid] = cached
            continue

        print(f"[extraction.run]   {pid}: calling Groq")
        ext = _extract_paper(p)
        if "error" not in ext:
            save_extraction(pid, ext)
            print(f"[extraction.run]   {pid}: saved "
                  f"({len(ext['key_claims'])} claims, {len(ext['limitations'])} limitations)")
        else:
            print(f"[extraction.run]   {pid}: failed ({ext['error']})")
        extractions[pid] = ext

    return {"extractions": extractions}


if __name__ == "__main__":
    from agents.search_agent import search_agent_run
    from agents.critique_agent import critique_agent_run

    q = "denoising diffusion image inpainting"
    search_result = search_agent_run(q)
    critique_result = critique_agent_run(q, search_result["papers"])
    extraction_result = extraction_agent_run(critique_result["shortlist"])
    print(f"\nExtracted {len(extraction_result['extractions'])} papers")
    for pid, ext in extraction_result["extractions"].items():
        print(f"  - {pid}: {len(ext.get('key_claims', []))} claims")
