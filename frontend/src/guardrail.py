"""
Out-of-scope guardrail for the Baahubali RAG system.
Detects and rejects queries unrelated to the Baahubali universe.
"""
from typing import Tuple
from src.config import BAAHUBALI_KEYWORDS


# Topics that are explicitly out-of-scope
OUT_OF_SCOPE_INDICATORS = [
    "cricket", "football", "soccer", "basketball", "tennis",
    "election", "president", "prime minister", "politics",
    "stock market", "bitcoin", "cryptocurrency",
    "covid", "pandemic", "vaccine",
    "recipe", "cooking", "restaurant",
    "weather", "forecast", "temperature",
    "capital of", "population of", "currency of",
    "programming", "python code", "javascript",
    "iphone", "android", "google", "apple", "microsoft",
    "social media", "instagram", "tiktok", "twitter",
    "world cup", "olympics", "champions league",
    "elon musk", "jeff bezos", "mark zuckerberg",
    "chatgpt", "artificial intelligence", "machine learning",
    "homework", "math problem", "calculate",
]


def is_in_scope(query: str) -> Tuple[bool, str, float]:
    """
    Determine whether a query is related to the Baahubali universe.

    Returns:
        (is_in_scope, message, confidence)
    """
    query_lower = query.lower().strip()

    # Empty query check
    if len(query_lower) < 3:
        return False, "Please enter a valid question about the Baahubali universe.", 1.0

    # Check for explicit out-of-scope indicators
    for indicator in OUT_OF_SCOPE_INDICATORS:
        if indicator in query_lower:
            return (
                False,
                f"⚔️ I cannot answer this question based on the provided document. "
                f"This query appears to be about '{indicator}', which is outside "
                f"the Baahubali knowledge base. Please ask about the Baahubali movies, "
                f"characters, kingdoms, or battles.",
                0.95,
            )

    # Check for Baahubali-related keywords
    keyword_matches = sum(1 for kw in BAAHUBALI_KEYWORDS if kw in query_lower)

    if keyword_matches >= 1:
        return True, "Query is related to the Baahubali universe.", 0.9 + min(keyword_matches * 0.02, 0.1)

    # Heuristic: generic movie/story questions might be in-scope
    generic_story_words = [
        "who", "why", "how", "what", "when", "where",
        "character", "story", "plot", "movie", "film",
        "hero", "villain", "war", "love", "betray",
        "son", "father", "mother", "brother", "wife",
        "death", "kill", "fight", "rescue", "save",
    ]
    story_matches = sum(1 for w in generic_story_words if w in query_lower)

    if story_matches >= 2:
        # Ambiguous — let it through with lower confidence
        return True, "Query might be related to Baahubali (generic story terms detected).", 0.6

    # Default: likely out-of-scope
    return (
        False,
        "⚔️ I cannot answer this question based on the provided document. "
        "This query does not appear to be related to the Baahubali movies. "
        "Please ask about characters, kingdoms, battles, or events from the Baahubali saga.",
        0.7,
    )
