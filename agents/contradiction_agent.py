import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, DEV_MODEL

groq_client = Groq(api_key=GROQ_API_KEY)

CONTRADICTION_PROMPT = """You are a research contradiction detector. Below are key claims extracted from multiple papers. Identify pairs of claims that DIRECTLY contradict each other.

For each contradiction, explain WHY they conflict — different benchmarks, datasets, model scale, time period, evaluation protocol, or genuinely opposing findings.

Return ONLY a JSON object with this exact shape, NOTHING else:
{{
  "contradictions": [
    {{
      "paper_a": "<paper_id>",
      "claim_a": "<exact claim text from paper A>",
      "paper_b": "<paper_id>",
      "claim_b": "<exact claim text from paper B>",
      "reason": "1-2 sentences explaining why these conflict",
      "category": "<one of: benchmark, dataset, scale, time_period, methodology, finding, evaluation_metric, training_data, architecture>"      
    }}
  ]
}}

Rules:
- Only include genuine direct contradictions. If none exist, return {{"contradictions": []}}.
- paper_a and paper_b MUST be DIFFERENT paper_ids. Never compare a paper against itself.
- If only one paper is provided, return {{"contradictions": []}}.
- Do not invent claims — quote them as given.
- Skip claims that merely differ in scope or topic without conflicting.
- Exhaustively check ALL claim pairs across papers. The same paper pair can appear multiple times with different conflicting claims — emit one object per conflicting claim pair, do not collapse to one entry per paper pair.
- A single claim may contradict multiple claims in another paper; emit each pairing as its own object.

Example shape (illustrative only, not real claims):
{{
  "contradictions": [
    {{"paper_a": "X", "claim_a": "claim 1", "paper_b": "Y", "claim_b": "claim 1", "reason": "...", "category": "benchmark"}},
    {{"paper_a": "X", "claim_a": "claim 2", "paper_b": "Y", "claim_b": "claim 3", "reason": "...", "category": "scale"}}
  ]
}}

Claims by paper:
{claims_block}"""


def _format_claims_block(extractions: dict) -> str:
    lines = []
    for pid, ext in extractions.items():
        claims = ext.get("key_claims") or []
        if not claims:
            continue
        lines.append(f"Paper {pid}:")
        for c in claims:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines).strip()


def contradiction_agent_run(extractions: dict) -> dict:
    print(f"\n[contradiction.run] checking {len(extractions)} extractions for contradictions")
    if len(extractions) < 2:
        print("[contradiction.run] need at least 2 papers to compare, skipping")
        return {"contradictions": []}
    claims_block = _format_claims_block(extractions)
    if not claims_block:
        print("[contradiction.run] no claims to compare")
        return {"contradictions": []}

    prompt = CONTRADICTION_PROMPT.format(claims_block=claims_block)
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
        print(f"[contradiction.run] JSON parse failed: {text[:200]!r}")
        return {"contradictions": []}

    contradictions = list(data.get("contradictions", []) or [])
    print(f"[contradiction.run] found {len(contradictions)} contradiction(s)")
    return {"contradictions": contradictions}


if __name__ == "__main__":
    from agents.search_agent import search_agent_run
    from agents.critique_agent import critique_agent_run
    from agents.extraction_agent import extraction_agent_run

    q = "Hentavirus"
    search_result = search_agent_run(q)
    critique_result = critique_agent_run(q, search_result["papers"])
    extraction_result = extraction_agent_run(critique_result["shortlist"])
    contradiction_result = contradiction_agent_run(extraction_result["extractions"])
    for c in contradiction_result["contradictions"]:
        print(f"- [{c.get('category')}] {c.get('paper_a')} vs {c.get('paper_b')}: {c.get('reason')}")