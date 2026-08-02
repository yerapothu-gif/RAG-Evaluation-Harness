"""
Query classifier for categorizing Baahubali-related questions.
Categories: Character, Kingdom, Battle, Timeline, General
"""
from typing import Tuple


# Category keyword mappings
CATEGORY_KEYWORDS = {
    "Character": [
        "amarendra", "mahendra", "shivudu", "sivagami", "devasena",
        "bhallaladeva", "bhallala", "kattappa", "katappa", "bijjaladeva",
        "avantika", "kumara varma", "kalakeya", "prabhas", "rana",
        "who is", "character", "personality", "played by", "actor",
        "mother", "father", "son", "wife", "brother", "rival",
        "hero", "villain", "protagonist", "antagonist",
    ],
    "Kingdom": [
        "mahishmati", "kingdom", "palace", "throne", "kuntala",
        "architecture", "fortification", "walls", "gates",
        "political", "monarchy", "dynasty", "wealth", "trade",
        "geography", "valley", "mountain", "defense",
        "rule", "govern", "capital", "territory",
    ],
    "Battle": [
        "battle", "war", "invasion", "siege", "fight", "combat",
        "kalakeya invasion", "kuntala battle", "final battle",
        "army", "soldier", "weapon", "sword", "arrow", "shield",
        "catapult", "palm tree", "bull", "taming",
        "military", "strategy", "tactics", "cavalry", "elephant",
        "attack", "defend", "victory", "defeat",
    ],
    "Timeline": [
        "when", "timeline", "sequence", "order", "chronolog",
        "before", "after", "first", "then", "finally",
        "generation", "year", "age", "born", "died",
        "history", "event", "happened", "period",
        "25 years", "childhood", "grew up",
    ],
    "General": [
        "theme", "motif", "symbol", "significance", "meaning",
        "movie", "film", "director", "producer", "budget",
        "box office", "gross", "record", "music", "vfx",
        "rajamouli", "arka media", "language", "telugu", "tamil",
        "dialogue", "quote", "famous", "waterfall",
    ],
}


def classify_query(query: str) -> Tuple[str, float]:
    """
    Classify a query into a category.

    Returns:
        (category, confidence)
    """
    query_lower = query.lower()

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                # Longer keywords get more weight
                weight = len(keyword.split())
                score += weight
        scores[category] = score

    if not scores or max(scores.values()) == 0:
        return "General", 0.5

    best_category = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_category] / total_score if total_score > 0 else 0.5

    return best_category, round(confidence, 3)


def get_category_icon(category: str) -> str:
    """Return an icon for each category."""
    icons = {
        "Character": "👤",
        "Kingdom": "🏰",
        "Battle": "⚔️",
        "Timeline": "📅",
        "General": "📜",
    }
    return icons.get(category, "❓")
