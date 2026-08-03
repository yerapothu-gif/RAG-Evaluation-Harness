"""
Multi-provider LLM pool: Gemini → Groq → Grok (xAI)
All LLM calls in the application route through this single module.
Providers are tried in order; the first to succeed returns.
If all fail, falls back to offline direct context synthesis.
"""
import json
import os
import time
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# ── Provider credentials ─────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "llama-3.3-70b-versatile"

XAI_API_KEY   = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL  = "https://api.x.ai/v1"
XAI_MODEL     = "grok-3-mini-fast"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("gemini_api_key", "")
GEMINI_MODEL   = "gemini-2.0-flash"


# ── Provider pool builder ─────────────────────────────────────────────────────

def _build_pool(custom_api_key: Optional[str] = None) -> List[Tuple[str, object, str, str]]:
    """
    Build an ordered list of (name, client_or_sdk, model, provider_type) tuples.
    provider_type is 'openai' for Groq/Grok, 'gemini' for Gemini.
    Pool order: Custom Key → Groq → Gemini → Grok (xAI)
    """
    from openai import OpenAI

    pool = []

    # Custom key routing by prefix
    if custom_api_key and custom_api_key.strip():
        key = custom_api_key.strip()
        if key.startswith("gsk_"):
            pool.append(("Groq (Custom)", OpenAI(api_key=key, base_url=GROQ_BASE_URL), GROQ_MODEL, "openai"))
        elif key.startswith("xai-"):
            pool.append(("Grok (Custom)", OpenAI(api_key=key, base_url=XAI_BASE_URL), XAI_MODEL, "openai"))
        else:
            # Ambiguous — try all providers
            pool.append(("Groq (Custom)", OpenAI(api_key=key, base_url=GROQ_BASE_URL), GROQ_MODEL, "openai"))

    # Groq
    if GROQ_API_KEY and not GROQ_API_KEY.startswith("your_"):
        pool.append(("Groq", OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL), GROQ_MODEL, "openai"))

    # Gemini (via google-genai SDK)
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"):
        pool.append(("Gemini", GEMINI_API_KEY, GEMINI_MODEL, "gemini"))

    # Grok / xAI
    if XAI_API_KEY and not XAI_API_KEY.startswith("your_"):
        pool.append(("Grok (xAI)", OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL), XAI_MODEL, "openai"))

    return pool


def _call_openai_provider(client, model: str, messages: List[Dict],
                           max_tokens: int, temperature: float) -> str:
    """Call an OpenAI-compatible provider (Groq or Grok)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _call_gemini_provider(api_key: str, model: str, messages: List[Dict],
                           max_tokens: int, temperature: float) -> str:
    """Call Gemini via the google-genai SDK."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    # Flatten messages to Gemini content format
    combined_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            combined_text += f"[System Instructions]\n{content}\n\n"
        elif role == "user":
            combined_text += f"{content}"
        elif role == "assistant":
            combined_text += f"\n[Previous Response]\n{content}\n\n"

    response = gemini_model.generate_content(combined_text)
    return response.text.strip()


def _pool_call(
    messages: List[Dict],
    max_tokens: int = 1024,
    temperature: float = 0.3,
    custom_api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Call the LLM provider pool in order. Returns (text, provider_name).
    Raises RuntimeError if all providers fail.
    """
    pool = _build_pool(custom_api_key)

    if not pool:
        raise RuntimeError("No LLM providers configured. Set GROQ_API_KEY, GEMINI_API_KEY, or XAI_API_KEY in .env")

    last_error = ""
    for name, client_or_key, model, ptype in pool:
        try:
            if ptype == "openai":
                text = _call_openai_provider(client_or_key, model, messages, max_tokens, temperature)
            else:
                text = _call_gemini_provider(client_or_key, model, messages, max_tokens, temperature)
            return text, name
        except Exception as e:
            last_error = f"{name}: {e}"
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


# ── Fallback synthesis ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Royal Archivist of Mahishmati. Your task is to answer questions STRICTLY and ONLY using the facts present in the provided context documents.

STRICT GROUNDING DIRECTIVES:
1. Answer ONLY using facts, names, dates, and details explicitly stated in the provided context.
2. Do NOT add outside knowledge, assumptions, or unmentioned speculations.
3. If the context does not contain enough information, state clearly: "Based on the provided archives, information about this is not available."
4. Be direct, precise, and 100% faithful to the text. Every statement in your answer must be directly verifiable in the context documents.
5. Do NOT include conversational fluff or meta-disclaimers."""


def _fallback_synthesis(query: str, context_chunks: List[Dict], error_msg: str) -> str:
    """Direct retrieval synthesis when all LLM providers fail."""
    if not context_chunks:
        return "⚠️ *No context retrieved from Mahishmati Archives and all LLM providers are unavailable.*"

    excerpts = []
    for i, c in enumerate(context_chunks[:3]):
        text = c.get("text", "").strip()
        if text:
            excerpts.append(f"**Excerpt {i+1}:** {text}")

    excerpts_str = "\n\n".join(excerpts)
    notice = (
        "⚠️ **LLM Notice:** All providers (Groq, Gemini, Grok) are unavailable or quota-limited. "
        "Showing direct retrieval synthesis.\n\n---\n\n"
    )
    return (
        f"{notice}"
        f"### 📜 Mahishmati Archives — Direct Context Synthesis\n\n"
        f"Based on top retrieved passages for *\"{query}\"*:\n\n"
        f"{excerpts_str}"
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: List[Dict],
    temperature: float = 0.0,
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Generate a strictly grounded answer using the provider pool.
    Returns {"answer": str, "model": str, "tokens_used": int}.
    """
    context_parts = [f"[Document {i+1}]\n{c['text']}" for i, c in enumerate(context_chunks)]
    context_str = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"CONTEXT DOCUMENTS:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"Provide a strictly grounded, accurate answer based ONLY on the context documents above."
        )},
    ]

    try:
        answer, provider = _pool_call(messages, max_tokens=1024,
                                      temperature=temperature, custom_api_key=custom_api_key)
        return {"answer": answer, "model": provider, "tokens_used": 0}
    except Exception as e:
        fallback = _fallback_synthesis(query, context_chunks, str(e))
        return {"answer": fallback, "model": "Direct-Synthesis (Fallback)", "tokens_used": 0}


def evaluate_with_llm(
    prompt: str,
    temperature: float = 0.1,
    custom_api_key: Optional[str] = None,
) -> str:
    """
    Generic LLM call for evaluation / judging tasks across the provider pool.
    Returns raw text, or a JSON error string if all providers fail.
    """
    messages = [
        {"role": "system", "content": "You are a precise evaluation judge. Follow instructions exactly and return valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        text, _ = _pool_call(messages, max_tokens=1024,
                             temperature=temperature, custom_api_key=custom_api_key)
        return text
    except Exception as e:
        return json.dumps({"error": str(e)})


def call_llm_raw(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    custom_api_key: Optional[str] = None,
) -> str:
    """
    Raw single-turn LLM call across the provider pool.
    Used by backend eval harness (llm_judge, ragas_eval).
    Returns raw text or JSON error string if all fail.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text, _ = _pool_call(messages, max_tokens=max_tokens, temperature=temperature, custom_api_key=custom_api_key)
        return text
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_active_providers() -> List[str]:
    """Return list of currently configured provider names (for status display)."""
    pool = _build_pool()
    return [name for name, _, _, _ in pool]
