from tools.arxiv_tool import search_papers
from tools.embedding_tool import get_embedding
from tools.db_tool import save_paper, save_embedding, paper_exists
from tools.pdf_tool import fetch_paper_text
from agents.search_agent import expand_query

query = 'agentic image inpainting'

expanded_queries = expand_query(query)
print(f"Expanded queries: {expanded_queries}")