"""
Full evaluation harness — runs all 7 metrics for a single QA pair.
Metrics: faithfulness, answer_relevance, context_recall, context_precision,
         hit@k, MRR, latency + LLM judge (1-5 score).

Ported and extended from src/evaluator.py — adds llm_judge and retrieval metrics.
"""
import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from backend.generation.llm import call_llm_raw
from backend.evaluation.llm_judge import llm_judge_faithfulness
from backend.evaluation.retrieval_metrics import compute_retrieval_metrics
from backend.config import TEST_SET_PATH


# ── JSON Parsing Helper ──────────────────────────────────────────────────────

def _parse_json_response(response: str) -> Dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if block:
        response = block.group(1)
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"error": "Failed to parse JSON", "raw": response}


# ── Heuristic Fallbacks ──────────────────────────────────────────────────────

def _keywords(text: str) -> set:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
        "of", "in", "on", "at", "to", "for", "with", "by", "this", "that",
        "it", "its", "not", "no", "so", "if", "be", "been", "do", "did",
    }
    return {w for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
            if len(w) > 2 and w not in stopwords}


def _heuristic_faithfulness(answer: str, context_chunks: List[Dict]) -> Dict:
    context_text = " ".join(c.get("text", "") for c in context_chunks)
    a_tokens = _keywords(answer)
    discourse_words = {
        "according", "context", "document", "documents", "archive", "archives",
        "stated", "states", "mentioned", "provided", "information", "available",
        "based", "royal", "archivist", "mahishmati", "saga", "story"
    }
    content_tokens = {w for w in a_tokens if w not in discourse_words} or a_tokens
    if not content_tokens:
        return {"score": 1.0, "reasoning": "Answer has minimal text.", "claims": []}
    c_tokens = _keywords(context_text)
    matched = content_tokens & c_tokens
    raw_ratio = len(matched) / len(content_tokens)
    if raw_ratio >= 0.50:
        score = round(min(1.0, 0.70 + (raw_ratio - 0.50) * 0.60), 4)
    else:
        score = round(max(0.0, raw_ratio * 1.40), 4)
    return {"score": score, "reasoning": f"Heuristic: {len(matched)}/{len(content_tokens)} core fact terms in context.", "claims": []}


def _heuristic_answer_relevance(question: str, answer: str) -> Dict:
    q_tokens = _keywords(question)
    a_tokens = _keywords(answer)
    if not q_tokens or not a_tokens:
        return {"score": 0.85, "reasoning": "Baseline relevance."}
    overlap = q_tokens & a_tokens
    score = max(0.15, min(1.0, len(overlap) / len(q_tokens)))
    return {"score": round(score, 4), "reasoning": f"Heuristic: {len(overlap)}/{len(q_tokens)} question terms in answer."}


def _heuristic_context_recall(ground_truth: str, context_chunks: List[Dict]) -> Dict:
    if not ground_truth or ground_truth == "OUT_OF_SCOPE":
        return {"score": 1.0, "reasoning": "No ground truth."}
    ctx_text = " ".join(c.get("text", "") for c in context_chunks)
    gt_tokens = _keywords(ground_truth)
    c_tokens = _keywords(ctx_text)
    if not gt_tokens:
        return {"score": 1.0, "reasoning": "Ground truth is empty."}
    matched = gt_tokens & c_tokens
    score = round(len(matched) / len(gt_tokens), 4)
    return {"score": min(1.0, score), "reasoning": f"Heuristic: {len(matched)}/{len(gt_tokens)} GT terms in context."}


def _heuristic_context_precision(question: str, context_chunks: List[Dict]) -> Dict:
    if not context_chunks:
        return {"score": 0.0, "reasoning": "No chunks.", "chunks": []}
    q_tokens = _keywords(question)
    chunk_scores = []
    for c in context_chunks:
        c_tokens = _keywords(c.get("text", ""))
        overlap = len(q_tokens & c_tokens) / len(q_tokens) if q_tokens and c_tokens else 0.5
        chunk_scores.append(1.0 if overlap >= 0.2 else 0.3)
    score = round(sum(chunk_scores) / len(chunk_scores), 4)
    return {"score": min(1.0, score), "reasoning": f"Heuristic precision: {score:.2f}"}


# ── LLM Evaluation Functions ─────────────────────────────────────────────────

def evaluate_faithfulness(question: str, answer: str, context_chunks: List[Dict]) -> Dict:
    """Evaluate faithfulness (0-1): is the answer grounded in context?"""
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""You are evaluating if an answer is faithful to context.

CONTEXT:
{context_text}

QUESTION: {question}
ANSWER: {answer}

Extract factual claims from the ANSWER and check if each is SUPPORTED by CONTEXT.
Return ONLY valid JSON:
{{
    "claims": [{{"claim": "text", "supported": true/false, "evidence": "quote or 'not found'"}}],
    "supported_count": <int>,
    "total_count": <int>,
    "score": <float 0.0-1.0>,
    "reasoning": "brief explanation"
}}"""
    response = call_llm_raw(prompt)
    result = _parse_json_response(response)
    if "error" in result or "score" not in result:
        return _heuristic_faithfulness(answer, context_chunks)
    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
        "claims": result.get("claims", []),
        "supported_count": result.get("supported_count", 0),
        "total_count": result.get("total_count", 0),
    }


def evaluate_answer_relevance(question: str, answer: str) -> Dict:
    """Evaluate answer relevance (0-1): does the answer address the question?"""
    prompt = f"""Evaluate how well this answer addresses the question.

QUESTION: {question}
ANSWER: {answer}

Return ONLY valid JSON:
{{"score": <float 0.0-1.0>, "reasoning": "brief explanation"}}"""
    response = call_llm_raw(prompt)
    result = _parse_json_response(response)
    if "error" in result or "score" not in result:
        return _heuristic_answer_relevance(question, answer)
    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
    }


def evaluate_context_recall(question: str, ground_truth: str, context_chunks: List[Dict]) -> Dict:
    """Evaluate context recall (0-1): does retrieved context contain all GT facts?"""
    if not ground_truth or ground_truth.startswith("OUT_OF_SCOPE"):
        return {"score": 1.0, "reasoning": "Out-of-scope or no ground truth."}
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""Evaluate context recall — does the retrieved context contain all facts from the ground truth?

QUESTION: {question}
GROUND TRUTH: {ground_truth}
CONTEXT: {context_text}

Return ONLY valid JSON:
{{
    "facts": [{{"fact": "text", "found_in_context": true/false}}],
    "found_count": <int>,
    "total_facts": <int>,
    "score": <float 0.0-1.0>,
    "reasoning": "brief explanation"
}}"""
    response = call_llm_raw(prompt)
    result = _parse_json_response(response)
    if "error" in result or "score" not in result:
        return _heuristic_context_recall(ground_truth, context_chunks)
    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
        "facts": result.get("facts", []),
    }


def evaluate_context_precision(question: str, context_chunks: List[Dict]) -> Dict:
    """Evaluate context precision (0-1): are retrieved chunks relevant?"""
    context_parts = [f"[Chunk {i+1}]: {c['text'][:300]}" for i, c in enumerate(context_chunks)]
    prompt = f"""Evaluate context precision — are these retrieved chunks relevant to the question?

QUESTION: {question}
CHUNKS:
{chr(10).join(context_parts)}

Return ONLY valid JSON:
{{
    "chunks": [{{"chunk_id": 1, "relevant": true/false, "reason": "brief"}}],
    "relevant_count": <int>,
    "total_chunks": <int>,
    "score": <float 0.0-1.0>,
    "reasoning": "brief explanation"
}}"""
    response = call_llm_raw(prompt)
    result = _parse_json_response(response)
    if "error" in result or "score" not in result:
        return _heuristic_context_precision(question, context_chunks)
    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
        "chunks": result.get("chunks", []),
    }


# ── Full Evaluation Runner ───────────────────────────────────────────────────

def run_full_evaluation(
    question: str,
    answer: str,
    context_chunks: List[Dict],
    ground_truth: Optional[str] = None,
    relevant_chunk_ids: Optional[List[str]] = None,
    latency_retrieval_ms: float = 0.0,
    latency_generation_ms: float = 0.0,
    setting: str = "A",
    query_type: str = "factual",
) -> Dict:
    """
    Run all 7 metrics for a single Q-A pair.

    Returns:
        Full evaluation dict with all scores and metadata.
    """
    eval_status = "success"
    try:
        faithfulness = evaluate_faithfulness(question, answer, context_chunks)
        answer_relevance = evaluate_answer_relevance(question, answer)
        context_precision = evaluate_context_precision(question, context_chunks)
        context_recall = evaluate_context_recall(question, ground_truth or "", context_chunks)
        judge = llm_judge_faithfulness(question, answer, context_chunks)
        retrieval_metrics = compute_retrieval_metrics(context_chunks, relevant_chunk_ids or [])
    except Exception as e:
        eval_status = f"failed: {str(e)}"
        # Return minimal result
        faithfulness = {"score": 0.0, "reasoning": eval_status}
        answer_relevance = {"score": 0.0}
        context_precision = {"score": 0.0}
        context_recall = {"score": 0.0}
        judge = {"judge_score": 1, "judge_score_normalized": 0.0, "method": "error"}
        retrieval_metrics = {"hit_at_k": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}

    scores = [faithfulness["score"], answer_relevance["score"],
              context_recall["score"], context_precision["score"]]
    avg_score = round(sum(scores) / len(scores), 4)

    return {
        "faithfulness": faithfulness["score"],
        "answer_relevancy": answer_relevance["score"],
        "context_recall": context_recall["score"],
        "context_precision": context_precision["score"],
        "hit_at_k": retrieval_metrics["hit_at_k"],
        "mrr": retrieval_metrics["mrr"],
        "llm_judge_score": judge["judge_score"],
        "llm_judge_normalized": judge["judge_score_normalized"],
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_generation_ms": latency_generation_ms,
        "avg_score": avg_score,
        "setting": setting,
        "query_type": query_type,
        "eval_status": eval_status,
        "details": {
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "judge": judge,
            "retrieval": retrieval_metrics,
        },
    }


def load_test_set() -> List[Dict]:
    """Load the 20 QA pairs from backend/evaluation/test_set.json."""
    with open(str(TEST_SET_PATH), "r", encoding="utf-8") as f:
        return json.load(f)
