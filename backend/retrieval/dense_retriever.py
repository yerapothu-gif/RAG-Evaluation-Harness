"""
Setting A: Dense vector retrieval using ChromaDB cosine similarity.
Embeds query → searches baahubali_setting_a collection → returns top-k.
"""
import time
from typing import List, Dict, Tuple

from backend.config import SETTING_A
from backend.vectordb.chroma_store import query_collection


def dense_search(query: str, top_k: int = None) -> Tuple[List[Dict], float]:
    """
    Perform dense vector retrieval for Setting A.

    Args:
        query: The user's question
        top_k: Number of results (defaults to SETTING_A["top_k"])

    Returns:
        (results, latency_ms)
        results: List of {id, text, similarity, metadata} dicts
    """
    k = top_k or SETTING_A["top_k"]
    collection = SETTING_A["collection_name"]

    t0 = time.perf_counter()
    results = query_collection(collection, query, top_k=k)
    latency_ms = (time.perf_counter() - t0) * 1000

    return results, round(latency_ms, 2)


def retrieve_setting_a(query: str, top_k: int = None) -> Dict:
    """
    Main retrieval function for Setting A.

    Returns:
        {chunks: [...], latency_ms: float, setting: "A"}
    """
    results, latency_ms = dense_search(query, top_k)
    return {
        "chunks": results,
        "latency_ms": latency_ms,
        "setting": "A",
        "retrieval_type": "dense",
        "collection": SETTING_A["collection_name"],
    }


if __name__ == "__main__":
    query = "Who is Kattappa?"
    result = retrieve_setting_a(query)
    print(f"Query: {query}")
    print(f"Latency: {result['latency_ms']}ms")
    for i, c in enumerate(result["chunks"], 1):
        print(f"\n[{i}] Similarity={c['similarity']:.4f}")
        print(c["text"][:200])
