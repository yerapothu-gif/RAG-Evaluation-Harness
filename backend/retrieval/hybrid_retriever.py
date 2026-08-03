"""
Setting B: Hybrid retrieval using BM25 + Dense vectors with RRF fusion.
BM25 top-10 + Qdrant/Chroma dense top-10 → Reciprocal Rank Fusion → top-3.
RRF formula: score(d) = Σ 1/(K + rank(d))  where K=60
"""
import time
from typing import List, Dict, Tuple

from backend.config import SETTING_B, RRF_K
from backend.vectordb.chroma_store import query_collection
from backend.vectordb.bm25_store import query_bm25


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    k: int = RRF_K,
    top_k: int = 5,
) -> List[Dict]:
    """
    Merge dense and BM25 results via Reciprocal Rank Fusion.

    RRF_score(doc) = Σ 1/(K + rank_i(doc))

    Args:
        dense_results: Results from ChromaDB (with 'id', 'text', 'similarity')
        bm25_results: Results from BM25 (with 'id', 'text', 'bm25_score')
        k: RRF constant (default 60)
        top_k: Number of final results to return

    Returns:
        List of fused result dicts sorted by RRF score descending
    """
    doc_scores: Dict[str, float] = {}
    doc_meta: Dict[str, Dict] = {}

    # Score from dense results
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
        doc_meta[doc_id] = {
            "text": result["text"],
            "dense_rank": rank + 1,
            "dense_similarity": result.get("similarity", 0.0),
            "metadata": result.get("metadata", {}),
        }

    # Score from BM25 results
    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
        if doc_id in doc_meta:
            doc_meta[doc_id]["bm25_rank"] = rank + 1
            doc_meta[doc_id]["bm25_score"] = result.get("bm25_score", 0.0)
        else:
            doc_meta[doc_id] = {
                "text": result["text"],
                "bm25_rank": rank + 1,
                "bm25_score": result.get("bm25_score", 0.0),
            }

    # Sort by RRF score and take top_k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for doc_id, rrf_score in sorted_docs[:top_k]:
        entry = {
            "id": doc_id,
            "text": doc_meta[doc_id]["text"],
            "rrf_score": round(rrf_score, 6),
            **{k: v for k, v in doc_meta[doc_id].items() if k != "text"},
        }
        fused.append(entry)

    return fused


def hybrid_search(query: str, top_k: int = None) -> Tuple[List[Dict], float]:
    """
    Perform hybrid BM25 + dense retrieval with RRF fusion.

    Args:
        query: The user's question
        top_k: Final number of results (defaults to SETTING_B["top_k"])

    Returns:
        (fused_results, latency_ms)
    """
    k = top_k or SETTING_B["top_k"]
    fetch_k = k * 3  # Fetch more from each source for better fusion coverage

    t0 = time.perf_counter()

    # Dense search on Setting B collection
    dense_results = query_collection(
        SETTING_B["collection_name"],
        query,
        top_k=fetch_k,
    )

    # BM25 keyword search
    bm25_results = query_bm25(query, top_k=fetch_k)

    # RRF fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K, top_k=k)

    latency_ms = (time.perf_counter() - t0) * 1000
    return fused, round(latency_ms, 2)


def retrieve_setting_b(query: str, top_k: int = None) -> Dict:
    """
    Main retrieval function for Setting B.

    Returns:
        {chunks: [...], latency_ms: float, setting: "B"}
    """
    results, latency_ms = hybrid_search(query, top_k)
    return {
        "chunks": results,
        "latency_ms": latency_ms,
        "setting": "B",
        "retrieval_type": "hybrid_rrf",
        "collection": SETTING_B["collection_name"],
    }


if __name__ == "__main__":
    query = "Who is Kattappa?"
    result = retrieve_setting_b(query)
    print(f"Query: {query}")
    print(f"Latency: {result['latency_ms']}ms")
    for i, c in enumerate(result["chunks"], 1):
        print(f"\n[{i}] RRF={c['rrf_score']:.6f}")
        print(c["text"][:200])
