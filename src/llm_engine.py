"""
LLM engine — thin facade over the unified provider pool (Gemini → Groq → Grok).
All generation and evaluation calls route through src.llm_pool.
This file exists for backward compatibility with existing imports in app.py and evaluator.py.
"""
from src.llm_pool import (   # noqa: F401  (re-exported)
    generate_answer,
    evaluate_with_llm,
    call_llm_raw,
    get_active_providers,
    SYSTEM_PROMPT,
)
