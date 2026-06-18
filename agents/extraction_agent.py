import sys
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, DEV_MODEL
from tools.db_tool import save_extraction, get_extraction, get_paper_full_text

groq_client = Groq(api_key=GROQ_API_KEY)

EXTRACTION_WORKERS = 2

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
{feedback_block}
Paper text (may be truncated):
{full_text}"""

_MAX_TEXT_CHARS = 12000
_TAIL_CHARS = 4000


def _prepare_text(full_text: str) -> str:

    text = full_text

    head_match = re.search(r"\b(abstract|introduction)\b", text, re.IGNORECASE)
    if head_match and head_match.start() < 3000:
        text = text[head_match.start():]

    ref_matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n",
                                   text, re.IGNORECASE))
    if ref_matches:
        text = text[:ref_matches[-1].start()]

    text = text.strip()

    if len(text) <= _MAX_TEXT_CHARS:
        return text

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


def _extract_paper(paper: dict, feedback: str = "") -> dict:
    full_text = paper.get("full_text") or ""
    if not full_text:
        full_text = get_paper_full_text(paper["id"])
    full_text = _prepare_text(full_text)
    if not full_text:
        print(f"[_extract_paper] {paper.get('id')}: no full_text, skipping")
        return _empty_extraction("no full_text")

    feedback_block = ""
    if feedback:
        feedback_block = (
            "\nA previous extraction led to a digest with citation-accuracy problems. "
            "Be especially careful to ground every claim in the text below and avoid "
            f"the following issue:\n{feedback}\n"
        )

    prompt = EXTRACTION_PROMPT.format(
        title=paper.get("title", ""),
        abstract=(paper.get("abstract") or "")[:2000],
        feedback_block=feedback_block,
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


def extraction_agent_run(papers: list[dict], force_refresh: bool = False, feedback: str = "") -> dict:
    print(f"\n[extraction.run] extracting from {len(papers)} papers"
          f"{' (force_refresh)' if force_refresh else ''}")
    extractions = {}
    to_extract = []
    for p in papers:
        pid = p["id"]
        if not force_refresh:
            cached = get_extraction(pid)
            if cached:
                print(f"[extraction.run]   {pid}: using cached extraction")
                extractions[pid] = cached
                continue
        to_extract.append(p)

    results = {}
    with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as executor:
        futures = {executor.submit(_extract_paper, p, feedback): p for p in to_extract}
        for future in as_completed(futures):
            pid = futures[future]["id"]
            print(f"[extraction.run]   {pid}: called Groq")
            try:
                results[pid] = future.result()
            except Exception as e:
                results[pid] = _empty_extraction(f"groq call failed: {e}")

    for pid, ext in results.items():
        if "error" not in ext:
            save_extraction(pid, ext, overwrite=force_refresh)
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
