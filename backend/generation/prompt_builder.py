"""
Prompt builder for the Baahubali RAG pipeline.
Assembles strict grounding system instructions + retrieved context chunks + user query.
"""
from typing import List, Dict


SYSTEM_PROMPT = """You are an expert on the Baahubali film series.
Your role is to answer questions STRICTLY and ONLY based on the context documents provided.

STRICT GROUNDING DIRECTIVES:
1. Answer ONLY using information explicitly present in the provided context.
2. If the context doesn't contain enough information, state so clearly.
3. Do NOT hallucinate facts, names, dates, or events not in the text.
4. Be direct, concise, and factually precise.
5. For adversarial or off-topic questions, decline to answer based on the knowledge base.
"""


def build_prompt(
    query: str,
    context_chunks: List[Dict],
    max_context_chars: int = 4000,
) -> str:
    """
    Build the full RAG prompt from query + retrieved chunks.
    """
    context_parts = []
    total_chars = 0
    for i, chunk in enumerate(context_chunks, 1):
        chunk_text = chunk.get("text", "").strip()
        if total_chars + len(chunk_text) > max_context_chars:
            break
        context_parts.append(f"[Context {i}]\n{chunk_text}")
        total_chars += len(chunk_text)

    context_block = "\n\n".join(context_parts) if context_parts else "No context retrieved."

    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT DOCUMENTS ===
{context_block}

=== QUESTION ===
{query}

=== YOUR ANSWER (Strictly Grounded) ===
"""
    return prompt


def build_chat_messages(
    query: str,
    context_chunks: List[Dict],
) -> List[Dict]:
    """
    Build OpenAI-style chat messages list for LLM APIs.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "").strip()
        context_parts.append(f"[Context {i}]\n{text}")

    context_block = "\n\n".join(context_parts) if context_parts else "No context retrieved."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"=== CONTEXT DOCUMENTS ===\n{context_block}\n\n"
                f"=== QUESTION ===\n{query}\n\n"
                f"Provide a strictly grounded answer based ONLY on the context above."
            ),
        },
    ]
