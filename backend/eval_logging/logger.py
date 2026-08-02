"""
JSONL logger — writes one line per eval run to logs/eval_runs.jsonl.
"""
import json
import time
from pathlib import Path
from typing import Dict

from backend.config import EVAL_RUNS_JSONL


def log_eval_result(record: Dict) -> None:
    """
    Append a single evaluation result as a JSONL line.
    Adds timestamp if not present.

    Args:
        record: Any dict (typically the full eval result dict)
    """
    if "timestamp" not in record:
        record = {**record, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    log_path = Path(EVAL_RUNS_JSONL)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_query(
    query: str,
    answer: str,
    setting: str,
    latency_retrieval_ms: float,
    latency_generation_ms: float,
    query_type: str = "unknown",
    experiment_id: str = "",
) -> None:
    """
    Quick helper to log a single query (no eval metrics).
    """
    log_eval_result({
        "type": "query",
        "experiment_id": experiment_id,
        "query": query,
        "answer_preview": answer[:200],
        "setting": setting,
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_generation_ms": latency_generation_ms,
        "query_type": query_type,
    })


def read_log(last_n: int = 50) -> list:
    """
    Read the last N records from the JSONL log file.
    """
    log_path = Path(EVAL_RUNS_JSONL)
    if not log_path.exists():
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    records = []
    for line in lines[-last_n:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records
