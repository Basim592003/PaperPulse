import requests
from config import HF_API_KEY, EMBEDDING_MODEL

_MODEL_ID = EMBEDDING_MODEL if "/" in EMBEDDING_MODEL else f"sentence-transformers/{EMBEDDING_MODEL}"
_API_URL = f"https://router.huggingface.co/hf-inference/models/{_MODEL_ID}/pipeline/feature-extraction"
_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        _API_URL,
        headers=_HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
