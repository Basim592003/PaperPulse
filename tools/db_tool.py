import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_ANON_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

_BAD_CHARS = re.compile(r"[\x00\ud800-\udfff]")

def _clean(v):
    if isinstance(v, str):
        return _BAD_CHARS.sub("", v)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v

def paper_exists(paper_id: str) -> bool:
    response = supabase.table("papers").select("id").eq("id", paper_id).execute()
    return len(response.data) > 0

def save_paper(paper: dict) -> None:
    if not paper_exists(paper["id"]):
        supabase.table("papers").insert({
            "id": paper["id"],
            "title": _clean(paper["title"]),
            "abstract": _clean(paper["abstract"]),
            "authors": _clean(paper["authors"]),
            "full_text": _clean(paper.get("full_text", "")),
            "published_date": paper["published"],
        }).execute()

def save_embedding(paper_id: str, embedding: list[float]) -> None:
    supabase.table("papers").update({
        "embedding": embedding
    }).eq("id", paper_id).execute()

def semantic_search(query_embedding: list[float], match_threshold: float = 0.5, match_count: int = 10) -> list[dict]:
    response = supabase.rpc("match_papers", {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": match_count
    }).execute()
    return response.data

def get_paper_full_text(paper_id: str) -> str:
    response = supabase.table("papers").select("full_text").eq("id", paper_id).execute()
    if response.data:
        return response.data[0].get("full_text") or ""
    return ""

def extraction_exists(paper_id: str) -> bool:
    response = supabase.table("extractions").select("id").eq("paper_id", paper_id).execute()
    return len(response.data) > 0

def save_extraction(paper_id: str, extraction: dict) -> None:
    if not extraction_exists(paper_id):
        supabase.table("extractions").insert({
            "paper_id": paper_id,
            "methodology": _clean(extraction.get("methodology", "")),
            "key_claims": _clean(extraction.get("key_claims", [])),
            "results": _clean(extraction.get("results", "")),
            "limitations": _clean(extraction.get("limitations", [])),
        }).execute()

def get_extraction(paper_id: str) -> dict | None:
    response = supabase.table("extractions").select("*").eq("paper_id", paper_id).execute()
    if response.data:
        return response.data[0]
    return None