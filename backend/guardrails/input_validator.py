"""
Input validator for the Baahubali RAG API.
Guards against empty queries, excessively long inputs, and obviously
out-of-scope questions before they hit the retrieval pipeline.
"""
from typing import Tuple

from backend.config import OUT_OF_SCOPE_INDICATORS, BAAHUBALI_KEYWORDS


MAX_QUERY_LENGTH = 500  # characters
MIN_QUERY_LENGTH = 3    # characters


def validate_input(query: str) -> Tuple[bool, str]:
    """
    Validate the raw query string.

    Returns:
        (is_valid, error_message)
        is_valid = True means the query passed all checks.
        If False, error_message explains why.
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."

    query_stripped = query.strip()

    if len(query_stripped) < MIN_QUERY_LENGTH:
        return False, f"Query too short (min {MIN_QUERY_LENGTH} characters)."

    if len(query_stripped) > MAX_QUERY_LENGTH:
        return False, (
            f"Query too long ({len(query_stripped)} chars). "
            f"Please keep it under {MAX_QUERY_LENGTH} characters."
        )

    return True, ""


def is_in_scope(query: str) -> Tuple[bool, str, float]:
    """
    Determine whether a query is related to the Baahubali universe.

    Returns:
        (is_in_scope, message, confidence)
    """
    query_lower = query.lower().strip()

    # Explicit out-of-scope indicators
    for indicator in OUT_OF_SCOPE_INDICATORS:
        if indicator in query_lower:
            return (
                False,
                (
                    f"⚔️ This question appears to be about '{indicator}', "
                    "which is outside the Baahubali knowledge base. "
                    "Please ask about Baahubali characters, kingdoms, battles, or events."
                ),
                0.95,
            )

    # Strong in-scope: direct keyword match
    keyword_matches = sum(1 for kw in BAAHUBALI_KEYWORDS if kw in query_lower)
    if keyword_matches >= 1:
        confidence = min(1.0, 0.9 + keyword_matches * 0.02)
        return True, "Query is related to the Baahubali universe.", confidence

    # Weak in-scope: generic story/film question words
    story_words = [
        "who", "why", "how", "what", "when", "where",
        "character", "story", "plot", "movie", "film",
        "hero", "villain", "war", "love", "betray",
        "son", "father", "mother", "brother", "wife",
        "death", "kill", "fight", "rescue", "save", "throne",
    ]
    story_matches = sum(1 for w in story_words if w in query_lower)
    if story_matches >= 2:
        return True, "Query might be related to Baahubali (generic story terms).", 0.6

    return (
        False,
        (
            "⚔️ This query does not appear to be related to the Baahubali movies. "
            "Please ask about characters, kingdoms, battles, or events from the Baahubali saga."
        ),
        0.7,
    )


def check_query(query: str) -> Tuple[bool, str, float]:
    """
    Combined validation + scope check.

    Returns:
        (allowed, message, confidence)
        allowed = True means the query should proceed to retrieval.
    """
    is_valid, err = validate_input(query)
    if not is_valid:
        return False, err, 1.0

    in_scope, msg, conf = is_in_scope(query)
    return in_scope, msg, conf
