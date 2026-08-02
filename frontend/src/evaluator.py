"""
LLM-as-a-Judge evaluation harness.
Measures: Faithfulness, Answer Relevance, Context Recall, Context Precision.
"""
import json
import re
from typing import Dict, List, Optional
from src.llm_engine import evaluate_with_llm


def _parse_json_response(response: str) -> Dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to extract JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Clean up common issues
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find any JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "Failed to parse JSON", "raw": response}


def _extract_keywords(text: str) -> set:
    """Extract meaningful lower-case word tokens, excluding common stopwords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "and", "or", "but", "if",
        "of", "in", "on", "at", "to", "for", "with", "by", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now", "it", "its", "this", "that", "these", "those"
    }
    words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    return {w for w in words if len(w) > 2 and w not in stopwords}


def _heuristic_faithfulness(answer: str, context_chunks: List[Dict]) -> Dict:
    context_text = " ".join([c.get("text", "") for c in context_chunks])
    answer_tokens = _extract_keywords(answer)
    if not answer_tokens:
        return {"score": 1.0, "reasoning": "Groundedness check: Answer contains minimal text.", "claims": []}
    
    context_tokens = _extract_keywords(context_text)
    matched = answer_tokens.intersection(context_tokens)
    score = len(matched) / len(answer_tokens)
    score = round(min(1.0, max(0.0, score)), 4)
    
    claims = [
        {"claim": f"Grounded term coverage ({len(matched)}/{len(answer_tokens)} terms in context)",
         "supported": score >= 0.5,
         "evidence": f"Overlap: {', '.join(list(matched)[:5])}"}
    ]
    return {
        "score": score,
        "reasoning": f"Algorithmic Groundedness: {len(matched)} of {len(answer_tokens)} answer terms found in context ({score*100:.1f}% grounded).",
        "claims": claims,
        "supported_count": len(matched),
        "total_count": len(answer_tokens),
    }


def _heuristic_answer_relevance(question: str, answer: str) -> Dict:
    q_tokens = _extract_keywords(question)
    a_tokens = _extract_keywords(answer)
    if not q_tokens or not a_tokens:
        return {"score": 0.85, "reasoning": "Baseline relevance check."}
    
    overlap = q_tokens.intersection(a_tokens)
    recall = len(overlap) / len(q_tokens)
    jaccard = len(overlap) / len(q_tokens.union(a_tokens))
    
    score = 0.65 * recall + 0.35 * min(1.0, jaccard * 3.0)
    if len(answer) > 40 and recall >= 0.4:
        score = max(score, 0.82)
    score = round(min(1.0, max(0.15, score)), 4)
    
    return {
        "score": score,
        "reasoning": f"Algorithmic Relevance: Answer addresses {len(overlap)} of {len(q_tokens)} question key terms ({score*100:.1f}% relevant).",
    }


def _heuristic_context_recall(question: str, ground_truth: str, context_chunks: List[Dict]) -> Dict:
    if not ground_truth or ground_truth == "OUT_OF_SCOPE":
        return {"score": 1.0, "reasoning": "Out-of-scope or missing ground truth."}
        
    context_text = " ".join([c.get("text", "") for c in context_chunks])
    gt_tokens = _extract_keywords(ground_truth)
    if not gt_tokens:
        return {"score": 1.0, "reasoning": "Ground truth contains minimal text."}
        
    context_tokens = _extract_keywords(context_text)
    matched = gt_tokens.intersection(context_tokens)
    score = len(matched) / len(gt_tokens)
    score = round(min(1.0, max(0.0, score)), 4)
    
    facts = [
        {"fact": f"Ground truth fact overlap ({len(matched)}/{len(gt_tokens)} key terms)", "found_in_context": score >= 0.5}
    ]
    return {
        "score": score,
        "reasoning": f"Algorithmic Recall: Context retrieved {len(matched)} of {len(gt_tokens)} ground truth facts ({score*100:.1f}% recall).",
        "facts": facts,
    }


def _heuristic_context_precision(question: str, context_chunks: List[Dict]) -> Dict:
    if not context_chunks:
        return {"score": 0.0, "reasoning": "No context chunks retrieved.", "chunks": []}
        
    q_tokens = _extract_keywords(question)
    chunk_scores = []
    chunk_details = []
    
    for i, c in enumerate(context_chunks):
        sim = c.get("similarity", 0) or c.get("rrf_score", 0)
        c_tokens = _extract_keywords(c.get("text", ""))
        
        if q_tokens and c_tokens:
            overlap = len(q_tokens.intersection(c_tokens))
            term_score = overlap / len(q_tokens)
        else:
            term_score = 0.5
            
        is_relevant = (sim > 0.25) or (term_score >= 0.2)
        chunk_scores.append(1.0 if is_relevant else (0.6 if sim > 0 else 0.3))
        chunk_details.append({
            "chunk_id": i + 1,
            "relevant": is_relevant,
            "reason": f"Similarity: {sim:.3f}, Term overlap: {term_score:.2f}"
        })
        
    score = sum(chunk_scores) / len(chunk_scores)
    score = round(min(1.0, max(0.0, score)), 4)
    
    return {
        "score": score,
        "reasoning": f"Algorithmic Precision: Precision computed across {len(context_chunks)} retrieved chunks ({score*100:.1f}% signal).",
        "chunks": chunk_details,
    }


def evaluate_faithfulness(
    question: str,
    answer: str,
    context_chunks: List[Dict],
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate whether the answer is grounded in the retrieved context.
    Score: 0.0 (hallucinated) to 1.0 (fully grounded).
    """
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are evaluating whether an answer is faithful to the provided context.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER: {answer}

TASK:
1. Extract all factual claims from the ANSWER (list them).
2. For each claim, determine if it is SUPPORTED by the CONTEXT (yes/no).
3. Calculate faithfulness score = (number of supported claims) / (total claims).

Return ONLY valid JSON in this exact format:
{{
    "claims": [
        {{"claim": "claim text", "supported": true/false, "evidence": "brief quote from context or 'not found'"}}
    ],
    "supported_count": <number>,
    "total_count": <number>,
    "score": <float between 0.0 and 1.0>,
    "reasoning": "brief explanation"
}}"""

    response = evaluate_with_llm(prompt, custom_api_key=custom_api_key)
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


def evaluate_answer_relevance(
    question: str,
    answer: str,
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate whether the answer actually addresses the question.
    Score: 0.0 (irrelevant) to 1.0 (perfectly relevant).
    """
    prompt = f"""You are evaluating whether an answer is relevant to the question asked.

QUESTION: {question}

ANSWER: {answer}

TASK:
Evaluate how well the answer addresses the specific question. Consider:
- Does it directly answer what was asked?
- Is it on-topic?
- Does it provide the information the question seeks?
- Is it complete (doesn't leave out critical parts of the answer)?

Return ONLY valid JSON:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "brief explanation of why this score was given"
}}"""

    response = evaluate_with_llm(prompt, custom_api_key=custom_api_key)
    result = _parse_json_response(response)

    if "error" in result or "score" not in result:
        return _heuristic_answer_relevance(question, answer)

    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
    }


def evaluate_context_recall(
    question: str,
    ground_truth: str,
    context_chunks: List[Dict],
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate whether the retrieved context contains all the facts
    needed to answer the question (compared to ground truth).
    Score: 0.0 (missed everything) to 1.0 (all facts retrieved).
    """
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""You are evaluating context recall — whether the retrieved documents contain all necessary information.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context_text}

TASK:
1. Extract key facts from the GROUND TRUTH ANSWER.
2. Check which of these facts are present in the RETRIEVED CONTEXT.
3. Calculate recall = (facts found in context) / (total facts in ground truth).

Return ONLY valid JSON:
{{
    "facts": [
        {{"fact": "fact text", "found_in_context": true/false}}
    ],
    "found_count": <number>,
    "total_facts": <number>,
    "score": <float between 0.0 and 1.0>,
    "reasoning": "brief explanation"
}}"""

    response = evaluate_with_llm(prompt, custom_api_key=custom_api_key)
    result = _parse_json_response(response)

    if "error" in result or "score" not in result:
        return _heuristic_context_recall(question, ground_truth, context_chunks)

    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
        "facts": result.get("facts", []),
    }


def evaluate_context_precision(
    question: str,
    context_chunks: List[Dict],
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Evaluate whether the retrieved context is signal (relevant) vs noise (irrelevant).
    Score: 0.0 (all noise) to 1.0 (all signal).
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(f"[Chunk {i+1}]: {chunk['text'][:300]}...")

    context_str = "\n\n".join(context_parts)

    prompt = f"""You are evaluating context precision — whether the retrieved document chunks are relevant to the question.

QUESTION: {question}

RETRIEVED CHUNKS:
{context_str}

TASK:
For each chunk, determine if it is RELEVANT to answering the question (contains useful information) or NOISE (irrelevant filler).
Calculate precision = (relevant chunks) / (total chunks).

Return ONLY valid JSON:
{{
    "chunks": [
        {{"chunk_id": 1, "relevant": true/false, "reason": "brief explanation"}}
    ],
    "relevant_count": <number>,
    "total_chunks": <number>,
    "score": <float between 0.0 and 1.0>,
    "reasoning": "brief explanation"
}}"""

    response = evaluate_with_llm(prompt, custom_api_key=custom_api_key)
    result = _parse_json_response(response)

    if "error" in result or "score" not in result:
        return _heuristic_context_precision(question, context_chunks)

    return {
        "score": min(1.0, max(0.0, float(result.get("score", 0.5)))),
        "reasoning": result.get("reasoning", ""),
        "chunks": result.get("chunks", []),
    }


def run_full_evaluation(
    question: str,
    answer: str,
    context_chunks: List[Dict],
    ground_truth: Optional[str] = None,
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Run all 4 evaluation metrics on a single Q-A pair.

    Returns:
        {
            "faithfulness": {"score": float, ...},
            "answer_relevance": {"score": float, ...},
            "context_recall": {"score": float, ...},
            "context_precision": {"score": float, ...},
            "avg_score": float,
        }
    """
    faithfulness = evaluate_faithfulness(question, answer, context_chunks, custom_api_key=custom_api_key)
    answer_relevance = evaluate_answer_relevance(question, answer, custom_api_key=custom_api_key)
    context_precision = evaluate_context_precision(question, context_chunks, custom_api_key=custom_api_key)

    if ground_truth and ground_truth != "OUT_OF_SCOPE":
        context_recall = evaluate_context_recall(question, ground_truth, context_chunks, custom_api_key=custom_api_key)
    else:
        context_recall = {"score": 1.0, "reasoning": "No ground truth available or out-of-scope"}

    scores = [
        faithfulness["score"],
        answer_relevance["score"],
        context_recall["score"],
        context_precision["score"],
    ]
    avg_score = sum(scores) / len(scores)

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "avg_score": round(avg_score, 4),
    }

