"""
PDF + Text loader for the Baahubali knowledge base.
Supports both .txt (existing) and .pdf (via PyMuPDF) inputs.
Outputs list of {text, metadata} dicts.
"""
import os
from pathlib import Path
from typing import List, Dict


def load_text_file(file_path: str) -> List[Dict]:
    """
    Load a plain-text knowledge base file.
    Returns a list with a single document dict.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{"text": text, "metadata": {"source": path.name, "page": 1}}]


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a PDF file page by page using PyMuPDF (fitz).
    Returns list of {text, metadata} dicts — one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:  # skip blank pages
            pages.append({
                "text": text,
                "metadata": {"source": path.name, "page": page_num},
            })
    doc.close()

    print(f"[pdf_loader] Loaded {len(pages)} pages from {path.name}")
    return pages


def load_knowledge_base(file_path: str) -> List[Dict]:
    """
    Auto-detect format (.txt or .pdf) and load accordingly.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    else:
        return load_text_file(file_path)


if __name__ == "__main__":
    from backend.config import KNOWLEDGE_BASE_PATH
    docs = load_knowledge_base(str(KNOWLEDGE_BASE_PATH))
    print(f"Loaded {len(docs)} document(s)")
    print(f"Sample (first 300 chars): {docs[0]['text'][:300]}")
