"""
LLM engine wrapper for Grok (xAI) via OpenAI-compatible API.
Handles answer generation with strict grounding constraints.
"""
import json
from typing import List, Dict, Optional
from openai import OpenAI
from src.config import XAI_API_KEY, XAI_BASE_URL, XAI_MODEL


def _get_client(custom_api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client pointing to xAI's API."""
    key = custom_api_key or XAI_API_KEY
    if not key or key == "your_xai_api_key_here":
        raise ValueError(
            "XAI_API_KEY is not set. Please add your Grok API key to the .env file or sidebar.\n"
            "Get one from: https://console.x.ai/"
        )
    return OpenAI(api_key=key, base_url=XAI_BASE_URL)


SYSTEM_PROMPT = """You are the Royal Archivist of Mahishmati, the keeper of all knowledge about the Baahubali saga. You answer questions ONLY based on the provided context documents.

STRICT RULES:
1. ONLY use information from the provided context to answer questions.
2. If the context does not contain enough information to fully answer the question, say: "Based on the available archives, I can share that..." and provide what you can, noting what information is missing.
3. If the question is completely unrelated to the Baahubali universe (e.g., sports, politics, technology, other movies), respond EXACTLY with: "I cannot answer this question based on the provided document. As the Royal Archivist of Mahishmati, my knowledge is limited to the Baahubali saga."
4. Do NOT make up facts, dates, or events not present in the context.
5. Do NOT hallucinate character names, relationships, or plot points.
6. Keep answers comprehensive but concise (2-4 paragraphs).
7. Reference specific events, characters, or quotes from the context when possible.
8. Write in an engaging, narrative style befitting the epic nature of the Baahubali saga."""


def _generate_fallback_synthesis(query: str, context_chunks: List[Dict], error_msg: str) -> str:
    """Generate a clean grounded synthesis directly from retrieved chunks when LLM API call fails (e.g. 403 credits error)."""
    if not context_chunks:
        return "⚠️ *No relevant context chunks retrieved from Mahishmati Archives.*"

    excerpts = []
    for i, c in enumerate(context_chunks[:3]):
        text = c.get("text", "").strip()
        if text:
            excerpts.append(f"**Excerpt {i+1}:** {text}")

    excerpts_str = "\n\n".join(excerpts)
    
    notice = ""
    if "403" in error_msg or "permission-denied" in error_msg or "credits" in error_msg:
        notice = "⚠️ **xAI API Notice:** The Grok API key has no active credits (*403 Permission Denied*). You can update `XAI_API_KEY` in `.env` or in the sidebar.\n\n---\n\n"
    elif error_msg:
        notice = f"⚠️ **LLM API Notice:** {error_msg}\n\n---\n\n"

    return (
        f"{notice}"
        f"### 📜 Mahishmati Archives Direct Retrieval Synthesis\n\n"
        f"Based on the top retrieved archives for *\"{query}\"*:\n\n"
        f"{excerpts_str}"
    )


def generate_answer(
    query: str,
    context_chunks: List[Dict],
    temperature: float = 0.3,
    custom_api_key: Optional[str] = None,
) -> Dict:
    """
    Generate an answer using the LLM with retrieved context.

    Returns:
        {
            "answer": str,
            "model": str,
            "tokens_used": int,
        }
    """
    try:
        client = _get_client(custom_api_key)
    except Exception as err:
        fallback = _generate_fallback_synthesis(query, context_chunks, str(err))
        return {
            "answer": fallback,
            "model": "RAG-Direct-Synthesis (Fallback)",
            "tokens_used": 0,
        }

    # Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(f"[Document {i+1}]\n{chunk['text']}")
    context_str = "\n\n".join(context_parts)

    user_message = f"""CONTEXT DOCUMENTS:
{context_str}

QUESTION: {query}

Provide a thorough answer based ONLY on the context documents above."""

    try:
        response = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return {
            "answer": answer,
            "model": XAI_MODEL,
            "tokens_used": tokens,
        }

    except Exception as e:
        err_str = str(e)
        fallback = _generate_fallback_synthesis(query, context_chunks, err_str)
        return {
            "answer": fallback,
            "model": f"{XAI_MODEL} (Fallback)",
            "tokens_used": 0,
        }


def evaluate_with_llm(prompt: str, temperature: float = 0.1, custom_api_key: Optional[str] = None) -> str:
    """
    Generic LLM call for evaluation tasks (scoring, claim extraction, etc.).
    Uses lower temperature for more deterministic scoring.
    """
    try:
        client = _get_client(custom_api_key)
        response = client.chat.completions.create(
            model=XAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise evaluation judge. Follow instructions exactly and return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        return json.dumps({"error": str(e)})

