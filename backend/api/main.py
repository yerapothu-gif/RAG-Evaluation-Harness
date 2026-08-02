"""
FastAPI backend for the Baahubali RAG Evaluation Harness.

Endpoints:
  POST /query                   — Run query through one setting, return answer
  POST /run-eval                — Run full harness on test set, return experiment ID
  GET  /results/{experiment_id} — All scores for one experiment run
  GET  /compare                 — Aggregated Setting A vs B metric table
  GET  /export                  — Download CSV of all results

Run with:
  uvicorn backend.api.main:app --reload --port 8000
"""
import csv
import io
import time
import uuid
from typing import Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import SETTING_A, SETTING_B
from backend.guardrails.input_validator import check_query
from backend.guardrails.query_classifier import classify_query
from backend.retrieval.dense_retriever import retrieve_setting_a
from backend.retrieval.hybrid_retriever import retrieve_setting_b
from backend.generation.llm import generate_answer
from backend.evaluation.ragas_eval import run_full_evaluation, load_test_set
from backend.eval_logging.logger import log_eval_result
from backend.eval_logging.eval_store import (
    init_db,
    insert_eval_result,
    get_experiment_results,
    get_comparison_summary,
    get_all_results,
)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Baahubali RAG Evaluation Harness",
    description="FastAPI backend for RAG pipeline with dual retrieval settings and evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
async def startup_event():
    init_db()


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="The question to ask")
    setting: Literal["A", "B"] = Field("A", description="Retrieval setting: A (dense) or B (hybrid)")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Override top-K (uses setting default if None)")


class QueryResponse(BaseModel):
    question: str
    answer: str
    setting: str
    chunks: List[Dict]
    latency_retrieval_ms: float
    latency_generation_ms: float
    query_type: str
    in_scope: bool
    message: str


class EvalRequest(BaseModel):
    setting: Optional[Literal["A", "B", "both"]] = Field(
        "both", description="Which setting(s) to evaluate"
    )
    max_questions: Optional[int] = Field(
        None, ge=1, le=25, description="Limit questions (default: all)"
    )


class EvalResponse(BaseModel):
    experiment_id: str
    setting: str
    total_questions: int
    completed: int
    failed: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    avg_hit_at_k: float
    avg_mrr: float
    avg_llm_judge_score: float
    avg_latency_retrieval_ms: float
    avg_latency_generation_ms: float


# ── Helper: Run RAG Pipeline ──────────────────────────────────────────────────

def _run_rag(question: str, setting: str, top_k: Optional[int] = None) -> Dict:
    """Run the full RAG pipeline for a given question and setting."""
    if setting == "A":
        retrieval = retrieve_setting_a(question, top_k)
    else:
        retrieval = retrieve_setting_b(question, top_k)

    chunks = retrieval["chunks"]
    latency_retrieval_ms = retrieval["latency_ms"]

    answer, latency_generation_ms = generate_answer(question, chunks)

    return {
        "answer": answer,
        "chunks": chunks,
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_generation_ms": latency_generation_ms,
    }


# ── Endpoint 1: POST /query ───────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(req: QueryRequest):
    """
    Run a single query through the RAG pipeline.
    Returns the answer + retrieved chunks + latency.
    """
    # Guardrail check
    in_scope, message, confidence = check_query(req.question)
    query_type, _ = classify_query(req.question)

    if not in_scope:
        return QueryResponse(
            question=req.question,
            answer=message,
            setting=req.setting,
            chunks=[],
            latency_retrieval_ms=0.0,
            latency_generation_ms=0.0,
            query_type=query_type,
            in_scope=False,
            message=message,
        )

    result = _run_rag(req.question, req.setting, req.top_k)

    # Log the query
    log_eval_result({
        "type": "query",
        "question": req.question,
        "setting": req.setting,
        "answer_preview": result["answer"][:200],
        "query_type": query_type,
        "latency_retrieval_ms": result["latency_retrieval_ms"],
        "latency_generation_ms": result["latency_generation_ms"],
    })

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        setting=req.setting,
        chunks=result["chunks"],
        latency_retrieval_ms=result["latency_retrieval_ms"],
        latency_generation_ms=result["latency_generation_ms"],
        query_type=query_type,
        in_scope=True,
        message=message,
    )


# ── Endpoint 2: POST /run-eval ────────────────────────────────────────────────

def _run_eval_for_setting(
    setting: str,
    test_set: List[Dict],
    experiment_id: str,
    max_questions: Optional[int],
) -> EvalResponse:
    """Run the full eval harness for one setting."""
    setting_config = SETTING_A if setting == "A" else SETTING_B
    questions = test_set[:max_questions] if max_questions else test_set

    results = []
    failed = 0

    for item in questions:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        query_type = item.get("type", "factual")

        try:
            rag = _run_rag(question, setting, top_k=setting_config["top_k"])
            eval_result = run_full_evaluation(
                question=question,
                answer=rag["answer"],
                context_chunks=rag["chunks"],
                ground_truth=ground_truth,
                latency_retrieval_ms=rag["latency_retrieval_ms"],
                latency_generation_ms=rag["latency_generation_ms"],
                setting=setting,
                query_type=query_type,
            )

            # Persist to SQLite
            insert_eval_result(
                experiment_id=experiment_id,
                question=question,
                answer=rag["answer"],
                eval_result=eval_result,
                setting_config=setting_config,
                query_id=str(item.get("id", "")),
            )

            # Log to JSONL
            log_eval_result({
                "type": "eval",
                "experiment_id": experiment_id,
                "question": question,
                "setting": setting,
                **{k: v for k, v in eval_result.items() if k != "details"},
            })

            results.append(eval_result)

        except Exception as e:
            failed += 1
            log_eval_result({
                "type": "eval_error",
                "experiment_id": experiment_id,
                "question": question,
                "setting": setting,
                "error": str(e),
                "eval_status": "failed",
            })

    def _avg(key: str) -> float:
        vals = [r.get(key, 0.0) for r in results if r.get("eval_status") == "success"]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return EvalResponse(
        experiment_id=experiment_id,
        setting=setting,
        total_questions=len(questions),
        completed=len(results),
        failed=failed,
        avg_faithfulness=_avg("faithfulness"),
        avg_answer_relevancy=_avg("answer_relevancy"),
        avg_context_recall=_avg("context_recall"),
        avg_context_precision=_avg("context_precision"),
        avg_hit_at_k=_avg("hit_at_k"),
        avg_mrr=_avg("mrr"),
        avg_llm_judge_score=_avg("llm_judge_score"),
        avg_latency_retrieval_ms=_avg("latency_retrieval_ms"),
        avg_latency_generation_ms=_avg("latency_generation_ms"),
    )


@app.post("/run-eval", tags=["Evaluation"])
async def run_eval_endpoint(req: EvalRequest):
    """
    Run the full evaluation harness on the test set.
    Returns experiment_id and aggregated metric scores.
    """
    test_set = load_test_set()
    experiment_id = str(uuid.uuid4())

    if req.setting == "both":
        result_a = _run_eval_for_setting("A", test_set, experiment_id + "_A", req.max_questions)
        result_b = _run_eval_for_setting("B", test_set, experiment_id + "_B", req.max_questions)
        return {
            "experiment_id": experiment_id,
            "setting_a": result_a.dict(),
            "setting_b": result_b.dict(),
        }
    else:
        result = _run_eval_for_setting(req.setting, test_set, experiment_id, req.max_questions)
        return result.dict()


# ── Endpoint 3: GET /results/{experiment_id} ─────────────────────────────────

@app.get("/results/{experiment_id}", tags=["Evaluation"])
async def get_results(experiment_id: str):
    """
    Get all evaluation scores for a given experiment_id.
    """
    rows = get_experiment_results(experiment_id)
    if not rows:
        # Try with _A and _B suffixes
        rows_a = get_experiment_results(experiment_id + "_A")
        rows_b = get_experiment_results(experiment_id + "_B")
        rows = rows_a + rows_b

    if not rows:
        raise HTTPException(status_code=404, detail=f"No results found for experiment '{experiment_id}'")

    return {"experiment_id": experiment_id, "count": len(rows), "results": rows}


# ── Endpoint 4: GET /compare ──────────────────────────────────────────────────

@app.get("/compare", tags=["Evaluation"])
async def compare_settings():
    """
    Aggregated Setting A vs B metric comparison across all experiments.
    """
    summary = get_comparison_summary()
    if not summary:
        return {
            "message": "No evaluation results found. Run /run-eval first.",
            "A": {},
            "B": {},
        }
    return {
        "comparison": summary,
        "metrics": [
            "faithfulness", "answer_relevancy", "context_recall",
            "context_precision", "hit_at_k", "mrr",
            "latency_retrieval_ms", "latency_generation_ms", "llm_judge_score",
        ],
    }


# ── Endpoint 5: GET /export ───────────────────────────────────────────────────

@app.get("/export", tags=["Evaluation"])
async def export_csv(limit: int = Query(default=500, ge=1, le=2000)):
    """
    Download all evaluation results as a CSV file.
    """
    rows = get_all_results(limit=limit)
    if not rows:
        raise HTTPException(status_code=404, detail="No results to export.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eval_results.csv"},
    )


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Check API health and vector store status."""
    from backend.vectordb.chroma_store import collection_count
    return {
        "status": "ok",
        "setting_a_docs": collection_count(SETTING_A["collection_name"]),
        "setting_b_docs": collection_count(SETTING_B["collection_name"]),
        "version": "1.0.0",
    }
