"""
Hybrid retriever combining BM25 (keyword) + Dense Vector (semantic) search
using Reciprocal Rank Fusion (RRF) for Setting B.
"""
import pickle
import os
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
from src.config import SETTING_B, RRF_K, DATA_DIR
from src.vector_store import query_collection


_bm25_index: Optional[BM25Okapi] = None
_bm25_chunks: Optional[List[Dict]] = None
BM25_CACHE_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: List[Dict], force_rebuild: bool = False):
    """Build a BM25 index from text chunks."""
    global _bm25_index, _bm25_chunks

    if not force_rebuild and os.path.exists(BM25_CACHE_PATH):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
                _bm25_index = cached["index"]
                _bm25_chunks = cached["chunks"]
                return
        except Exception:
            pass

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks

    # Cache to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"index": _bm25_index, "chunks": chunks}, f)


def query_bm25(query: str, top_k: int = 10) -> List[Dict]:
    """Query the BM25 index and return ranked results."""
    global _bm25_index, _bm25_chunks

    if _bm25_index is None or _bm25_chunks is None:
        # Try loading from cache
        if os.path.exists(BM25_CACHE_PATH):
            with open(BM25_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
                _bm25_index = cached["index"]
                _bm25_chunks = cached["chunks"]
        else:
            return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices
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


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    k: int = RRF_K,
    top_k: int = 5,
) -> List[Dict]:
    """
    Merge dense vector and BM25 results using Reciprocal Rank Fusion.
    RRF_score(doc) = Σ 1/(k + rank_i(doc))
    """
    doc_scores = {}
    doc_texts = {}

    # Score from dense results
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        doc_texts[doc_id] = {
            "text": result["text"],
            "dense_rank": rank + 1,
            "dense_similarity": result.get("similarity", 0),
        }

    # Score from BM25 results
    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        if doc_id in doc_texts:
            doc_texts[doc_id]["bm25_rank"] = rank + 1
            doc_texts[doc_id]["bm25_score"] = result.get("bm25_score", 0)
        else:
            doc_texts[doc_id] = {
                "text": result["text"],
                "bm25_rank": rank + 1,
                "bm25_score": result.get("bm25_score", 0),
            }

    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_id, rrf_score in sorted_docs[:top_k]:
        entry = {
            "id": doc_id,
            "text": doc_texts[doc_id]["text"],
            "rrf_score": round(rrf_score, 6),
        }
        entry.update(doc_texts[doc_id])
        fused_results.append(entry)

    return fused_results


def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform hybrid search: BM25 + Dense Vector with RRF fusion.
    This is the main retrieval function for Setting B.
    """
    # Dense vector search
    dense_results = query_collection(
        SETTING_B["collection_name"],
        query,
        top_k=top_k * 2,  # Fetch more for better fusion
    )

    # BM25 keyword search
    bm25_results = query_bm25(query, top_k=top_k * 2)

    # Fuse with RRF
    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)

    return fused
