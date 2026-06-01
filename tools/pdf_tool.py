import urllib.request
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader

def fetch_paper_text(pdf_url: str) -> str:
    req = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "Mozilla/5.0 (paperpulse research bot)"},
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        with urllib.request.urlopen(req) as resp:
            f.write(resp.read())
        tmp_path = f.name
    try:
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    finally:
        os.unlink(tmp_path)