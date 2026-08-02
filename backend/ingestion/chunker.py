"""
Text chunker for the Baahubali RAG pipeline.
Uses LangChain RecursiveCharacterTextSplitter.
Produces Setting A (500/100) and Setting B (300/50) chunk sets.
Saves both as JSONL to data/.
"""
import json
import uuid
from pathlib import Path
from typing import List, Dict

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def _make_chunks(
    documents: List[Dict],
    chunk_size: int,
    chunk_overlap: int,
    setting_label: str,
) -> List[Dict]:
    """
    Split document texts into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of {text, metadata} dicts from pdf_loader
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between consecutive chunks
        setting_label: 'A' or 'B' — used to prefix chunk IDs

    Returns:
        List of {id, text, char_count, metadata} dicts
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        raw_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 20:  # skip trivial fragments
                continue
            chunk_id = f"{setting_label}_{i:04d}_{uuid.uuid4().hex[:6]}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "metadata": doc.get("metadata", {}),
            })

    return chunks


def chunk_for_setting_a(documents: List[Dict]) -> List[Dict]:
    """Setting A: 500 char chunks, 100 overlap (dense retrieval)."""
    return _make_chunks(documents, chunk_size=500, chunk_overlap=100, setting_label="A")


def chunk_for_setting_b(documents: List[Dict]) -> List[Dict]:
    """Setting B: 300 char chunks, 50 overlap (hybrid BM25+dense retrieval)."""
    return _make_chunks(documents, chunk_size=300, chunk_overlap=50, setting_label="B")


def save_chunks_jsonl(chunks: List[Dict], output_path: str) -> None:
    """Save chunks to a JSONL file (one JSON object per line)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[chunker] Saved {len(chunks)} chunks -> {output_path}")


def load_chunks_jsonl(jsonl_path: str) -> List[Dict]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from backend.config import KNOWLEDGE_BASE_PATH, CHUNKS_A_PATH, CHUNKS_B_PATH
    from backend.ingestion.pdf_loader import load_knowledge_base

    print("Loading knowledge base...")
    docs = load_knowledge_base(str(KNOWLEDGE_BASE_PATH))

    print("Chunking for Setting A (500/100)...")
    chunks_a = chunk_for_setting_a(docs)
    save_chunks_jsonl(chunks_a, str(CHUNKS_A_PATH))

    print("Chunking for Setting B (300/50)...")
    chunks_b = chunk_for_setting_b(docs)
    save_chunks_jsonl(chunks_b, str(CHUNKS_B_PATH))

    avg_a = sum(c["char_count"] for c in chunks_a) / len(chunks_a) if chunks_a else 0
    avg_b = sum(c["char_count"] for c in chunks_b) / len(chunks_b) if chunks_b else 0

    print(f"\nSetting A: {len(chunks_a)} chunks, avg {avg_a:.0f} chars")
    print(f"Setting B: {len(chunks_b)} chunks, avg {avg_b:.0f} chars")
