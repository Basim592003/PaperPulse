from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()