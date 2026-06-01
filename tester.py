from tools.arxiv_tool import search_papers
from tools.embedding_tool import get_embedding
from tools.db_tool import save_paper, save_embedding, paper_exists
from tools.pdf_tool import fetch_paper_text

papers = search_papers(["denoising diffusion probabilistic models"])

for paper in papers:
    if not paper_exists(paper["id"]):
        full_text = fetch_paper_text(paper["pdf_url"])
        paper["full_text"] = full_text
        save_paper(paper)
        embedding = get_embedding(paper["abstract"])
        save_embedding(paper["id"], embedding)
        print(f"Saved: {paper['title']}")
    else:
        print(f"Already exists: {paper['title']}")