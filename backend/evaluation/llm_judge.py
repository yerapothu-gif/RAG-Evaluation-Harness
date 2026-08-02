"""
LLM-as-a-Judge for faithfulness evaluation.
Sends a rubric to Grok and parses a 1–5 integer score.
Also provides heuristic fallback if the LLM call fails.
"""
import json
import re
from typing import Dict, List, Optional

from backend.generation.llm import call_llm_raw


_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the faithfulness of an AI-generated answer.

FAITHFULNESS DEFINITION:
A faithful answer contains only claims that are directly supported by the provided context.
An unfaithful answer includes hallucinated facts, contradictions, or unsupported claims.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ===
{answer}

=== SCORING RUBRIC (1–5) ===
5 = Every claim is directly supported by the context. No hallucinations.
4 = Almost all claims supported; minor unsupported details.
3 = Most claims supported; some unsupported or vague claims.
2 = Several unsupported claims or partial hallucinations.
1 = Answer is mostly hallucinated or contradicts the context.

Return ONLY valid JSON:
{{
    "score": <integer 1 to 5>,
    "reasoning": "<one-sentence explanation>",
    "unsupported_claims": ["<list any hallucinated claims, or empty list>"]
}}"""


def _parse_judge_response(response: str) -> Optional[Dict]:
    """Extract JSON from the judge LLM response."""
    # Try JSON code block first
    block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if block:
        response = block.group(1)

    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find embedded JSON object
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _heuristic_judge_score(answer: str, context_chunks: List[Dict]) -> int:
    """
    Fallback heuristic score (1-5) when LLM judge is unavailable.
    Based on keyword overlap between answer and context.
    """
    context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
    answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    if not answer_words:
        return 3

    context_words = set(re.findall(r'\b\w{4,}\b', context_text))
    overlap = len(answer_words & context_words) / len(answer_words)

    if overlap >= 0.75:
        return 5
    elif overlap >= 0.55:
        return 4
    elif overlap >= 0.35:
        return 3
    elif overlap >= 0.20:
        return 2
    return 1


def llm_judge_faithfulness(
    question: str,
    answer: str,
    context_chunks: List[Dict],
) -> Dict:
    """
    Use Grok as a judge to score faithfulness 1-5.

    Returns:
        {
            "judge_score": int (1-5),
            "judge_score_normalized": float (0.0-1.0),
            "reasoning": str,
            "unsupported_claims": List[str],
            "method": "llm" | "heuristic"
        }
    """
    context_text = "\n\n".join(c.get("text", "") for c in context_chunks)[:3000]
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question,
        answer=answer,
    )

    raw_response = call_llm_raw(prompt, max_tokens=256, temperature=0.0)

    parsed = _parse_judge_response(raw_response)

    if parsed and "score" in parsed:
        raw_score = parsed["score"]
        try:
            score = max(1, min(5, int(raw_score)))
        except (TypeError, ValueError):
            score = _heuristic_judge_score(answer, context_chunks)
            return {
                "judge_score": score,
                "judge_score_normalized": round((score - 1) / 4, 4),
                "reasoning": "Score parsing failed, used heuristic fallback.",
                "unsupported_claims": [],
                "method": "heuristic",
            }

        return {
            "judge_score": score,
            "judge_score_normalized": round((score - 1) / 4, 4),
            "reasoning": parsed.get("reasoning", ""),
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "method": "llm",
        }

    # Full fallback
    score = _heuristic_judge_score(answer, context_chunks)
    return {
        "judge_score": score,
        "judge_score_normalized": round((score - 1) / 4, 4),
        "reasoning": "LLM judge unavailable, used heuristic fallback.",
        "unsupported_claims": [],
        "method": "heuristic",
    }
