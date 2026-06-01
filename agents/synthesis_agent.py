import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, PROD_MODEL

groq_client = Groq(api_key=GROQ_API_KEY)

SYNTHESIS_PROMPT = """You are a research synthesis agent. Given structured extractions from multiple papers and a list of detected contradictions, produce a final digest for the user that answers their original query.

Original query: {query}

Return ONLY a JSON object with this exact shape, NOTHING else:
{{
  "summary": "3-6 sentence overview of the state of the field relative to the query",
  "key_findings": ["finding 1", "finding 2", "..."],
  "contradictions": [
    {{
      "papers": ["<paper_id>", "<paper_id>"],
      "description": "1-2 sentences describing the conflict"
    }}
  ],
  "recommended_papers": [
    {{"paper_id": "<paper_id>", "title": "<title>", "why": "1 sentence on why this paper matters for the query"}}
  ],
  "what_to_read_next": "1-3 sentences suggesting open questions, follow-up topics, or papers to look for next"
}}

Rules:
- Ground every finding in the provided extractions. Do not invent results or citations.
- key_findings should be concise, standalone statements (one insight each).
- contradictions: faithfully restate the provided contradictions; do not fabricate new ones. If none were provided, return [].
- recommended_papers should cover the papers most relevant to the query, ordered by importance.
- Keep the tone neutral and informative — this is a research briefing, not marketing copy.

Papers and extractions:
{extractions_block}

Detected contradictions:
{contradictions_block}"""


def _format_extractions_block(papers: list[dict], extractions: dict) -> str:
    paper_meta = {p["id"]: p for p in papers}
    lines = []
    for pid, ext in extractions.items():
        meta = paper_meta.get(pid, {})
        title = meta.get("title", "")
        lines.append(f"Paper {pid} — {title}")
        if ext.get("methodology"):
            lines.append(f"  methodology: {ext['methodology']}")
        if ext.get("key_claims"):
            lines.append("  key_claims:")
            for c in ext["key_claims"]:
                lines.append(f"    - {c}")
        if ext.get("results"):
            lines.append(f"  results: {ext['results']}")
        if ext.get("limitations"):
            lines.append("  limitations:")
            for l in ext["limitations"]:
                lines.append(f"    - {l}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_contradictions_block(contradictions: list[dict]) -> str:
    if not contradictions:
        return "(none detected)"
    lines = []
    for c in contradictions:
        lines.append(
            f"- {c.get('paper_a')} vs {c.get('paper_b')} [{c.get('category', 'n/a')}]: "
            f"{c.get('reason', '')}"
        )
        if c.get("claim_a"):
            lines.append(f"    A: {c['claim_a']}")
        if c.get("claim_b"):
            lines.append(f"    B: {c['claim_b']}")
    return "\n".join(lines)


def synthesis_agent_run(query: str, papers: list[dict], extractions: dict, contradictions: list[dict]) -> dict:
    print(f"\n[synthesis.run] synthesizing {len(extractions)} extractions, "
          f"{len(contradictions)} contradictions")

    extractions_block = _format_extractions_block(papers, extractions)
    contradictions_block = _format_contradictions_block(contradictions)

    if not extractions_block:
        print("[synthesis.run] no extractions to synthesize")
        return {"digest": {
            "summary": "",
            "key_findings": [],
            "contradictions": [],
            "recommended_papers": [],
            "what_to_read_next": "",
            "error": "no extractions",
        }}

    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        extractions_block=extractions_block,
        contradictions_block=contradictions_block,
    )
    response = groq_client.chat.completions.create(
        model=PROD_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        digest = json.loads(text)
    except json.JSONDecodeError:
        print(f"[synthesis.run] JSON parse failed: {text[:200]!r}")
        return {"digest": {
            "summary": "",
            "key_findings": [],
            "contradictions": [],
            "recommended_papers": [],
            "what_to_read_next": "",
            "error": "parse error",
        }}

    print(f"[synthesis.run] digest produced "
          f"({len(digest.get('key_findings', []))} findings, "
          f"{len(digest.get('recommended_papers', []))} recommendations)")
    return {"digest": digest}


if __name__ == "__main__":
    from agents.search_agent import search_agent_run
    from agents.critique_agent import critique_agent_run
    from agents.extraction_agent import extraction_agent_run
    from agents.contradiction_agent import contradiction_agent_run

    q = "denoising diffusion image inpainting"
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
    print("\n=== DIGEST ===")
    print(json.dumps(synthesis_result["digest"], indent=2))