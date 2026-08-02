"""
Query type classifier for the Baahubali RAG pipeline.
Classifies queries as: factual / character / plot / adversarial.
Uses keyword heuristics (no API call needed).
"""
import re
from typing import Tuple


_PATTERNS = {
    "adversarial": [
        r"cricket", r"football", r"politics", r"weather", r"cook",
        r"recipe", r"bitcoin", r"stock", r"covid", r"capital of",
        r"programming", r"chatgpt", r"election", r"president",
        r"population", r"calculate", r"math",
    ],
    "character": [
        r"\bwho\b", r"character", r"kattappa", r"katappa", r"sivagami",
        r"bhallaladeva", r"bhallala", r"devasena", r"amarendra",
        r"mahendra", r"avantika", r"bijjaladeva", r"shivudu",
        r"rajmata", r"relationship", r"brother", r"father", r"mother",
        r"wife", r"son", r"daughter", r"slave", r"queen", r"king",
        r"prince", r"princess",
    ],
    "plot": [
        r"why", r"how did", r"what happened", r"explain", r"sequence",
        r"story", r"reason", r"unfold", r"chain", r"manipulation",
        r"kill", r"die", r"death", r"battle", r"war", r"fight",
        r"plan", r"strategy", r"scheme", r"downfall", r"result",
    ],
    "factual": [
        r"what is", r"where", r"when", r"how many", r"which",
        r"describe", r"list", r"name", r"title", r"directed",
        r"produced", r"budget", r"box office", r"gross",
        r"mahishmati", r"kingdom", r"army", r"waterfall",
    ],
}


def classify_query(query: str) -> Tuple[str, float]:
    """
    Classify a query into one of four categories.

    Returns:
        (category, confidence)
        category: "factual" | "character" | "plot" | "adversarial"
        confidence: float 0.0-1.0
    """
    query_lower = query.lower().strip()
    scores: dict[str, int] = {cat: 0 for cat in _PATTERNS}

    for category, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                scores[category] += 1

    # "adversarial" wins if any indicator matches
    if scores["adversarial"] > 0:
        return "adversarial", 0.95

    best_cat = max(
        [cat for cat in scores if cat != "adversarial"],
        key=lambda c: scores[c],
        default="factual",
    )
    best_score = scores[best_cat]

    if best_score >= 3:
        confidence = 0.9
    elif best_score >= 2:
        confidence = 0.75
    elif best_score >= 1:
        confidence = 0.6
    else:
        best_cat = "factual"  # default
        confidence = 0.5

    return best_cat, confidence


if __name__ == "__main__":
    test_queries = [
        "Why did Kattappa kill Baahubali?",
        "Who is Sivagami?",
        "Describe Mahishmati kingdom.",
        "What happened in the final battle?",
        "Who won the cricket World Cup?",
    ]
    for q in test_queries:
        cat, conf = classify_query(q)
        print(f"[{cat:12s} {conf:.2f}] {q}")
