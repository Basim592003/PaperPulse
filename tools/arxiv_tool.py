import time

import arxiv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_ARXIV_RESULTS


client = arxiv.Client(page_size=10, delay_seconds=10, num_retries=5)

def search_papers(queries: list[str]) -> list[dict]:
    papers = {}
    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max(5, MAX_ARXIV_RESULTS // len(queries)),
            sort_by=arxiv.SortCriterion.Relevance
        )
        try:
            for paper in client.results(search):
                paper_id = paper.entry_id.split("/")[-1]
                if paper_id not in papers:
                    papers[paper_id] = {
                        "id": paper_id,
                        "title": paper.title,
                        "abstract": paper.summary,
                        "authors": [a.name for a in paper.authors],
                        "published": str(paper.published.date()),
                        "pdf_url": paper.pdf_url
                    }
        except arxiv.HTTPError as e:
            print(f"[arxiv] HTTP {e.status} on query '{query}' — skipping")
            time.sleep(15)
            continue
        time.sleep(10)
    return list(papers.values())