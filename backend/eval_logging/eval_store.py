"""
SQLite-backed eval result store.
Schema matches the final_plan.md specification exactly.
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import SQLITE_DB_PATH

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id         TEXT,
    query_id              TEXT,
    question              TEXT,
    answer                TEXT,
    setting               TEXT,
    embedding_model       TEXT,
    chunk_size            INTEGER,
    top_k                 INTEGER,
    faithfulness          REAL,
    answer_relevancy      REAL,
    context_recall        REAL,
    context_precision     REAL,
    hit_at_k              REAL,
    mrr                   REAL,
    latency_retrieval_ms  REAL,
    latency_generation_ms REAL,
    llm_judge_score       REAL,
    query_type            TEXT,
    eval_status           TEXT,
    timestamp             TEXT
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiment_id ON eval_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_setting ON eval_results(setting);
"""


@contextmanager
def _get_conn():
    """Context manager for SQLite connection."""
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the SQLite database (creates table if not exists)."""
    with _get_conn() as conn:
        conn.executescript(_CREATE_TABLE_SQL + _CREATE_INDEX_SQL)
    print(f"[eval_store] Database ready at {SQLITE_DB_PATH}")


def insert_eval_result(
    experiment_id: str,
    question: str,
    answer: str,
    eval_result: Dict,
    setting_config: Dict,
    query_id: Optional[str] = None,
) -> int:
    """
    Insert a single evaluation result row.

    Args:
        experiment_id: UUID for the current experiment batch
        question: The question asked
        answer: The generated answer
        eval_result: Dict from ragas_eval.run_full_evaluation()
        setting_config: SETTING_A or SETTING_B dict
        query_id: Optional ID for the question (from test set)

    Returns:
        The new row's id
    """
    init_db()

    row = {
        "experiment_id": experiment_id,
        "query_id": query_id or str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "setting": eval_result.get("setting", "A"),
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": setting_config.get("chunk_size", 0),
        "top_k": setting_config.get("top_k", 0),
        "faithfulness": eval_result.get("faithfulness", 0.0),
        "answer_relevancy": eval_result.get("answer_relevancy", 0.0),
        "context_recall": eval_result.get("context_recall", 0.0),
        "context_precision": eval_result.get("context_precision", 0.0),
        "hit_at_k": eval_result.get("hit_at_k", 0.0),
        "mrr": eval_result.get("mrr", 0.0),
        "latency_retrieval_ms": eval_result.get("latency_retrieval_ms", 0.0),
        "latency_generation_ms": eval_result.get("latency_generation_ms", 0.0),
        "llm_judge_score": eval_result.get("llm_judge_score", 0.0),
        "query_type": eval_result.get("query_type", "factual"),
        "eval_status": eval_result.get("eval_status", "success"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    sql = """
    INSERT INTO eval_results
        (experiment_id, query_id, question, answer, setting, embedding_model,
         chunk_size, top_k, faithfulness, answer_relevancy, context_recall,
         context_precision, hit_at_k, mrr, latency_retrieval_ms,
         latency_generation_ms, llm_judge_score, query_type, eval_status, timestamp)
    VALUES
        (:experiment_id, :query_id, :question, :answer, :setting, :embedding_model,
         :chunk_size, :top_k, :faithfulness, :answer_relevancy, :context_recall,
         :context_precision, :hit_at_k, :mrr, :latency_retrieval_ms,
         :latency_generation_ms, :llm_judge_score, :query_type, :eval_status, :timestamp)
    """
    with _get_conn() as conn:
        cursor = conn.execute(sql, row)
        return cursor.lastrowid


def get_experiment_results(experiment_id: str) -> List[Dict]:
    """Get all rows for a given experiment_id."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_results WHERE experiment_id = ? ORDER BY id",
            (experiment_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_comparison_summary() -> Dict:
    """
    Aggregate average scores for Setting A vs B across all experiments.
    Returns dict with 'A' and 'B' keys, each containing metric averages.
    """
    init_db()
    metrics = [
        "faithfulness", "answer_relevancy", "context_recall",
        "context_precision", "hit_at_k", "mrr",
        "latency_retrieval_ms", "latency_generation_ms", "llm_judge_score",
    ]
    avg_cols = ", ".join(f"AVG({m}) as avg_{m}" for m in metrics)
    sql = f"SELECT setting, COUNT(*) as n, {avg_cols} FROM eval_results GROUP BY setting"

    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    result = {}
    for row in rows:
        d = dict(row)
        setting = d.pop("setting")
        result[setting] = d
    return result


def get_all_results(limit: int = 200) -> List[Dict]:
    """Get recent eval results for CSV export."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_results ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
