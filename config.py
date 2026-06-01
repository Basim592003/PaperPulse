import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

DEV_MODEL = "llama-3.1-8b-instant"
PROD_MODEL = "llama-3.3-70b-versatile"

CRITIQUE_THRESHOLD = 5
EVAL_THRESHOLD = 0.75
MAX_WORKERS = 5
TOP_K_PAPERS = 5
MAX_ARXIV_RESULTS = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384