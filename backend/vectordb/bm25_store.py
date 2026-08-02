"""
BM25 keyword index for Setting B hybrid retrieval.
Builds BM25Okapi index, caches to disk as pickle.
"""
import os
import pickle
import re
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import BM25_INDEX_PATH

_bm25_index: Optional[BM25Okapi] = None
_bm25_chunks: Optional[List[Dict]] = None


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: List[Dict], force_rebuild: bool = False) -> None:
    """
    Build BM25Okapi index from text chunks and cache to disk.

    Args:
        chunks: List of {id, text, ...} dicts
        force_rebuild: If True, ignore existing cache
    """
    global _bm25_index, _bm25_chunks
    cache_path = str(BM25_INDEX_PATH)

    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                _bm25_index = cached["index"]
                _bm25_chunks = cached["chunks"]
                print(f"[bm25_store] Loaded BM25 index from cache ({len(_bm25_chunks)} docs).")
                return
        except Exception as e:
            print(f"[bm25_store] Cache load failed ({e}), rebuilding...")

    print(f"[bm25_store] Building BM25 index from {len(chunks)} chunks...")
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"index": _bm25_index, "chunks": chunks}, f)
    print(f"[bm25_store] BM25 index saved -> {cache_path}")


def _load_cache() -> bool:
    """Try loading BM25 index from disk cache. Returns True on success."""
    global _bm25_index, _bm25_chunks
    cache_path = str(BM25_INDEX_PATH)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                _bm25_index = cached["index"]
                _bm25_chunks = cached["chunks"]
                return True
        except Exception:
            pass
    return False


def query_bm25(query: str, top_k: int = 10) -> List[Dict]:
    """
    Query the BM25 index and return top-k results.

    Returns:
        List of {id, text, bm25_score, bm25_rank} dicts
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is None or _bm25_chunks is None:
        if not _load_cache():
            print("[bm25_store] WARNING: BM25 index not loaded. Call build_bm25_index first.")
            return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append({
                "id": _bm25_chunks[idx]["id"],
                "text": _bm25_chunks[idx]["text"],
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank + 1,
            })

    return results


def is_bm25_loaded() -> bool:
    """Check if BM25 index is currently loaded."""
    return _bm25_index is not None and _bm25_chunks is not None


def get_bm25_chunks() -> Optional[List[Dict]]:
    """Return the chunks used to build the BM25 index."""
    return _bm25_chunks
