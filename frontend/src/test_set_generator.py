"""
Auto-generate test set using LLM-as-a-Teacher.
Parses the knowledge base and generates diverse Q-C-A triplets.
"""
import json
import os
from typing import List, Dict
from src.config import TEST_SET_PATH, KNOWLEDGE_BASE_PATH
from src.document_loader import load_document
from src.llm_engine import evaluate_with_llm


def generate_test_set(num_questions: int = 20) -> List[Dict]:
    """
    Use the LLM to auto-generate diverse Q-A triplets from the knowledge base.

    Categories:
    - Factual (10): Direct fact-based questions
    - Complex Reasoning (5): Multi-hop or inferential questions
    - Trick/Adversarial (3): Questions that test edge cases
    - Out-of-Scope (2): Questions unrelated to Baahubali
    """
    knowledge_text = load_document(KNOWLEDGE_BASE_PATH)

    # Truncate if too long for context window
    if len(knowledge_text) > 12000:
        knowledge_text = knowledge_text[:12000]

    prompt = f"""You are a test set generator for a RAG (Retrieval Augmented Generation) system about the Baahubali movies.

KNOWLEDGE BASE:
{knowledge_text}

TASK: Generate exactly {num_questions} diverse question-answer pairs as a test set.

Distribution:
- 10 FACTUAL questions (direct answers from the text)
- 5 COMPLEX REASONING questions (require connecting multiple facts)
- 3 TRICK questions (adversarial — questions that seem related but require careful reading)
- 2 OUT-OF-SCOPE questions (completely unrelated to Baahubali — sports, politics, etc.)

For out-of-scope questions, set ground_truth to "OUT_OF_SCOPE - This question is not related to the Baahubali movies."

Return ONLY a valid JSON array with this exact structure:
[
    {{
        "id": 1,
        "question": "the question",
        "ground_truth": "the expected answer",
        "category": "Character|Kingdom|Battle|Timeline|General|out_of_scope",
        "type": "factual|complex_reasoning|adversarial"
    }}
]

Make questions specific and varied. Cover different characters, events, and topics.
Include proper nouns like Bhallaladeva, Kattappa, Mahishmati, Sivagami, Devasena."""

    response = evaluate_with_llm(prompt, temperature=0.7)

    # Parse the response
    try:
        # Try to extract JSON array
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            test_set = json.loads(json_match.group())
        else:
            test_set = json.loads(response)

        return test_set

    except (json.JSONDecodeError, Exception) as e:
        print(f"Failed to generate test set: {e}")
        print("Falling back to pre-generated test set...")
        return load_test_set()


def load_test_set() -> List[Dict]:
    """Load the pre-generated test set from JSON file."""
    if os.path.exists(TEST_SET_PATH):
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_test_set(test_set: List[Dict]):
    """Save a test set to the JSON file."""
    os.makedirs(os.path.dirname(TEST_SET_PATH), exist_ok=True)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
