import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI

app = FastAPI(title="PaperPulse API")


@app.get("/health")
def health():
    return {"status": "ok"}
