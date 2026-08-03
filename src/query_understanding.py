import json
from typing import Optional
from src.llm_pool import call_llm_raw

EXPANSION_PROMPT = """You are a search query optimization engine for a Baahubali RAG (Retrieval-Augmented Generation) system.
Your task is to take the user's short or ambiguous question and rewrite it into a highly descriptive, comprehensive search query.

Guidelines:
1. Resolve pronouns (e.g., "he", "she", "it") to the specific characters if contextually obvious.
2. Add relevant synonyms and domain-specific keywords (e.g., if they ask about a fight, add "battle", "war", "combat").
3. Make the query highly optimized for both semantic vector search and BM25 keyword search.
4. Output ONLY the optimized query string. Do not include quotes, introductions, or conversational filler.

User Question: {query}
Optimized Search Query:"""

def expand_query(query: str, custom_api_key: Optional[str] = None) -> str:
    """
    Expands and rewrites the user's query into an optimized search string using the LLM.
    Falls back to the original query if the LLM call fails.
    """
    prompt = EXPANSION_PROMPT.format(query=query)
    
    # We call the raw LLM. Temperature is slightly > 0 to allow some creative expansion, but keep it low.
    response = call_llm_raw(prompt, max_tokens=150, temperature=0.2, custom_api_key=custom_api_key)
    
    # Check if the response was a JSON error from our fallback mechanism
    if response and response.startswith("{") and "error" in response.lower():
        return query
        
    expanded = response.strip().strip('"').strip("'")
    
    # Failsafe
    if not expanded or len(expanded) < 3:
        return query
        
    return expanded


SPELL_CORRECTION_PROMPT = """Fix any spelling or grammatical errors in the following user query. Do not answer the question or expand on it, just return the corrected text. If it is completely incomprehensible, return it as-is.

Query: {query}
Corrected:"""

def correct_spelling(query: str, custom_api_key: Optional[str] = None) -> str:
    """
    Pre-retrieval normalization step to fix spelling and grammar.
    Uses the LLM pool with deterministic settings (temperature=0.0).
    """
    prompt = SPELL_CORRECTION_PROMPT.format(query=query)
    
    response = call_llm_raw(prompt, max_tokens=100, temperature=0.0, custom_api_key=custom_api_key)
    
    if response and response.startswith("{") and "error" in response.lower():
        return query
        
    corrected = response.strip().strip('"').strip("'")
    
    if not corrected or len(corrected) < 2:
        return query
        
    return corrected
