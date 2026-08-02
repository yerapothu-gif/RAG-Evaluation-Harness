"""
Document loader and text splitter for the Baahubali knowledge base.
Supports .txt and .pdf files with configurable chunk sizes.
"""
import os
from typing import List, Dict


def load_document(file_path: str) -> str:
    """Load text content from a .txt or .pdf file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("pypdf is required for PDF files. Install with: pip install pypdf")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks with metadata.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    # Split into sentences first
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Split on sentence boundaries
        import re
        sent_splits = re.split(r'(?<=[.!?])\s+', paragraph)
        for s in sent_splits:
            s = s.strip()
            if s:
                sentences.append(s)

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "char_count": len(chunk_text),
            })
            chunk_id += 1

            # Calculate overlap: keep sentences from the end that fit within overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "char_count": len(chunk_text),
        })

    return chunks


def load_and_split(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, str]]:
    """Convenience function: load a document and split it into chunks."""
    text = load_document(file_path)
    return split_text(text, chunk_size, chunk_overlap)
