"""
Backend LLM caller — thin wrapper over the shared src.llm_pool provider pool.
Pool order: Gemini → Groq → Grok (xAI) → Offline Direct Synthesis.
"""
import time
from typing import List, Dict, Tuple

# Re-use the single source-of-truth pool from src/
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.llm_pool import call_llm_raw, get_active_providers  # noqa: F401 (re-exported)

from backend.generation.prompt_builder import build_chat_messages


def generate_answer(
    query: str,
    context_chunks: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Tuple[str, float]:
    """
    Generate an answer using the shared provider pool (Gemini → Groq → Grok).
    Returns (answer_text, latency_ms).
    """
    from src.llm_pool import _pool_call, _fallback_synthesis

    messages = build_chat_messages(query, context_chunks)
    t0 = time.perf_counter()

    try:
        text, provider = _pool_call(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        text = _fallback_synthesis(query, context_chunks, str(e))

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return text, latency_ms
