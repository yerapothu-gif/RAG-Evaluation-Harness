"""
Search analytics tracking module.
Tracks queries, response times, scores, and categories for the dashboard.
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from src.config import ANALYTICS_LOG_PATH


def _load_log() -> List[Dict]:
    """Load analytics log from disk."""
    if os.path.exists(ANALYTICS_LOG_PATH):
        try:
            with open(ANALYTICS_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_log(entries: List[Dict]):
    """Save analytics log to disk."""
    os.makedirs(os.path.dirname(ANALYTICS_LOG_PATH), exist_ok=True)
    with open(ANALYTICS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def log_query(
    query: str,
    category: str,
    setting: str,
    response_time: float,
    faithfulness_score: Optional[float] = None,
    retrieval_score: Optional[float] = None,
    is_in_scope: bool = True,
    answer_preview: str = "",
):
    """Log a query and its metrics."""
    entries = _load_log()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "category": category,
        "setting": setting,
        "response_time_ms": round(response_time * 1000, 1),
        "faithfulness_score": faithfulness_score,
        "retrieval_score": retrieval_score,
        "is_in_scope": is_in_scope,
        "answer_preview": answer_preview[:200] if answer_preview else "",
    }

    entries.append(entry)

    # Keep last 500 entries
    if len(entries) > 500:
        entries = entries[-500:]

    _save_log(entries)


def get_analytics() -> Dict:
    """Get aggregated analytics data for the dashboard."""
    entries = _load_log()

    if not entries:
        return {
            "total_queries": 0,
            "avg_response_time_ms": 0,
            "avg_faithfulness": 0,
            "avg_retrieval_score": 0,
            "category_distribution": {},
            "top_queries": [],
            "response_times": [],
            "faithfulness_over_time": [],
            "retrieval_over_time": [],
            "in_scope_rate": 0,
            "setting_distribution": {},
        }

    # Basic counts
    total = len(entries)
    in_scope = sum(1 for e in entries if e.get("is_in_scope", True))

    # Average response time
    response_times = [e["response_time_ms"] for e in entries]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Average faithfulness
    faith_scores = [e["faithfulness_score"] for e in entries if e.get("faithfulness_score") is not None]
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0

    # Average retrieval score
    ret_scores = [e["retrieval_score"] for e in entries if e.get("retrieval_score") is not None]
    avg_retrieval = sum(ret_scores) / len(ret_scores) if ret_scores else 0

    # Category distribution
    categories = {}
    for e in entries:
        cat = e.get("category", "General")
        categories[cat] = categories.get(cat, 0) + 1

    # Setting distribution
    settings = {}
    for e in entries:
        s = e.get("setting", "A")
        settings[s] = settings.get(s, 0) + 1

    # Most asked queries (group similar)
    query_counts = {}
    for e in entries:
        q = e["query"].strip().lower()
        query_counts[q] = query_counts.get(q, 0) + 1
    top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_queries": total,
        "avg_response_time_ms": round(avg_response_time, 1),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_retrieval_score": round(avg_retrieval, 3),
        "category_distribution": categories,
        "top_queries": [{"query": q, "count": c} for q, c in top_queries],
        "response_times": response_times[-50:],
        "faithfulness_over_time": faith_scores[-50:],
        "retrieval_over_time": ret_scores[-50:],
        "in_scope_rate": round(in_scope / total, 3) if total > 0 else 0,
        "setting_distribution": settings,
        "entries": entries[-50:],  # Last 50 for detailed view
    }


def clear_analytics():
    """Clear all analytics data."""
    _save_log([])
