import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_ARXIV_RESULTS
from groq import Groq
from config import GROQ_API_KEY, DEV_MODEL, TOP_K_PAPERS
from tools.arxiv_tool import search_papers
from tools.embedding_tool import get_embedding
from tools.db_tool import (
    semantic_search,
    paper_exists,
    save_paper,
    save_embedding,
)
from tools.pdf_tool import fetch_paper_text


groq_client = Groq(api_key=GROQ_API_KEY)

PDF_FETCH_WORKERS = 8

EXPAND_PROMPT = """You are a research assistant. Given a user's research question, generate 3 focused arXiv search queries that together cover the topic from different angles.

IMPORTANT: Preserve the most specific and unusual terms from the user's query in at least one of the 3 queries. Do not drop domain-specific keywords.

Use plain keywords only — no AND/OR operators, no quotes, no boolean syntax.
Keep each query concise, 4-8 words max, using technical terminology.

Return ONLY a JSON array of 3 strings, nothing else. Example:
["denoising diffusion probabilistic models image synthesis", "score based generative models sampling", "DDPM latent diffusion image generation"]

User question: {query}"""


def expand_query(user_query: str) -> list[str]:
    print(f"[expand_query] user_query={user_query!r}")
    prompt = EXPAND_PROMPT.format(
        query=user_query
    )
    
    response = groq_client.chat.completions.create(
        model=DEV_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    print(f"[expand_query] raw groq output: {text}")
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        queries = json.loads(text)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            print(f"[expand_query] parsed {len(queries)} queries: {queries}")
            return queries + [user_query]  
    except json.JSONDecodeError:
        print("[expand_query] JSON parse failed — falling back to user_query")
    return [user_query]


def search_supabase(queries: list[str]) -> list[dict]:
    print(f"[search_supabase] querying Supabase for {len(queries)} expanded queries")
    found = {}
    for q in queries:
        emb = get_embedding(q)
        hits = semantic_search(emb)
        print(f"[search_supabase]   {q!r} -> {len(hits)} hits")
        for paper in hits:
            found[paper["id"]] = paper
    print(f"[search_supabase] {len(found)} unique papers after dedup")
    return list(found.values())


def ingest_from_arxiv(queries: list[str]) -> list[dict]:
    print(f"[ingest_from_arxiv] hitting arXiv with {len(queries)} queries")
    new_papers = []
    fetched = search_papers(queries)
    print(f"[ingest_from_arxiv] arXiv returned {len(fetched)} papers")

    to_ingest = []
    for paper in fetched:
        if paper_exists(paper["id"]):
            print(f"[ingest_from_arxiv]   skip (already in db): {paper['id']}")
            continue
        to_ingest.append(paper)

    # PDF fetch is pure I/O — download/parse all candidates in parallel.
    def _fetch(paper):
        print(f"[ingest_from_arxiv]   fetching pdf: {paper['id']} — {paper['title'][:60]}")
        paper["full_text"] = fetch_paper_text(paper["pdf_url"])
        return paper

    fetched_papers = []
    with ThreadPoolExecutor(max_workers=PDF_FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch, p): p for p in to_ingest}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                fetched_papers.append(future.result())
            except Exception as e:
                print(f"[ingest_from_arxiv]   skip (pdf fetch failed: {e}): {paper['pdf_url']}")

    # DB writes stay sequential.
    for paper in fetched_papers:
        save_paper(paper)
        save_embedding(paper["id"], get_embedding(paper["abstract"]))
        new_papers.append(paper)
    print(f"[ingest_from_arxiv] ingested {len(new_papers)} new papers")
    return new_papers


def search_agent_run(user_query: str) -> dict:
    print(f"\n[run] === starting search for: {user_query!r} ===")
    queries = expand_query(user_query)
    db_hits = search_supabase(queries)

    if len(db_hits) >= max(TOP_K_PAPERS * 2, 10):
        print(f"[run] db has {len(db_hits)} hits (>= {MAX_ARXIV_RESULTS}); skipping arXiv")
        return {"queries": queries, "papers": db_hits[:MAX_ARXIV_RESULTS], "ingested": []}

    print(f"[run] db has only {len(db_hits)} hits; fetching more from arXiv")
    ingested = ingest_from_arxiv(queries)
    db_hits = search_supabase(queries)
    print(f"[run] final result: {len(db_hits[:MAX_ARXIV_RESULTS])} papers, {len(ingested)} newly ingested")
    return {
        "queries": queries,
        "papers": db_hits[:MAX_ARXIV_RESULTS],
        "ingested": ingested,
    }

if __name__ == "__main__":
    result = search_agent_run("what is life")
    print(f"Queries: {result['queries']}")
    print(f"Papers found: {len(result['papers'])}")
    print(f"Papers ingested: {len(result['ingested'])}")
    for p in result['papers']:
        print(f"  - {p['title']} ({p['published']})")