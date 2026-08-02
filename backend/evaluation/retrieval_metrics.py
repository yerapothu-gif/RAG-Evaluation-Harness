"""
Custom retrieval quality metrics: Hit@K and MRR.
These are computed using chunk IDs vs ground-truth chunk IDs from the test set.
"""
from typing import List, Optional


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 3) -> float:
    """
    Hit@K: 1.0 if at least one relevant chunk appears in top-K results, else 0.0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs
        relevant_ids: Ground-truth relevant chunk IDs
        k: Cutoff rank

    Returns:
        1.0 (hit) or 0.0 (miss)
    """
    if not relevant_ids:
        return 1.0  # No ground truth — assume hit

    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return 1.0 if top_k_ids & relevant_set else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first relevant retrieved chunk.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs
        relevant_ids: Ground-truth relevant chunk IDs

    Returns:
        Reciprocal rank (0.0 if no relevant chunk found)
    """
    if not relevant_ids:
        return 1.0  # No ground truth — assume perfect

    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 3) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(hits / k, 4) if k > 0 else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 3) -> float:
    """
    Recall@K: fraction of relevant docs that appear in top-K results.
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return round(hits / len(relevant_set), 4)


def compute_retrieval_metrics(
    retrieved_chunks: List[dict],
    relevant_ids: Optional[List[str]] = None,
    k: int = 3,
) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_chunks: List of retrieved chunk dicts (must have 'id' key)
        relevant_ids: Ground-truth chunk IDs (if available)
        k: Rank cutoff

    Returns:
        {hit_at_k: float, mrr: float, precision_at_k: float, recall_at_k: float}
    """
    retrieved_ids = [c["id"] for c in retrieved_chunks]
    gt_ids = relevant_ids or []

    return {
        "hit_at_k": hit_at_k(retrieved_ids, gt_ids, k),
        "mrr": mrr(retrieved_ids, gt_ids),
        "precision_at_k": precision_at_k(retrieved_ids, gt_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, gt_ids, k),
    }
