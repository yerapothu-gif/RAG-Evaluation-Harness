"""
Auto-generate test set using LLM-as-a-Teacher.
Parses the knowledge base and generates diverse Q-C-A triplets.
"""
import json
import os
import re
from typing import List, Dict
from src.config import TEST_SET_PATH, KNOWLEDGE_BASE_PATH
from src.document_loader import load_document
from src.llm_engine import evaluate_with_llm


def generate_test_set(num_questions: int = 20) -> List[Dict]:
    """
    Use the LLM to auto-generate diverse Q-A triplets from the knowledge base.
    """
    knowledge_text = load_document(KNOWLEDGE_BASE_PATH)

    if len(knowledge_text) > 12000:
        knowledge_text = knowledge_text[:12000]

    prompt = f"""You are a test set generator for a RAG system about the Baahubali movies.

KNOWLEDGE BASE:
{knowledge_text}

TASK: Generate exactly {num_questions} diverse question-answer pairs as a test set.

Distribution:
- 10 FACTUAL questions (direct answers from the text)
- 5 COMPLEX REASONING questions (require connecting multiple facts)
- 3 TRICK questions (adversarial)
- 2 OUT-OF-SCOPE questions (completely unrelated to Baahubali)

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
]"""

    response = evaluate_with_llm(prompt, temperature=0.2)

    try:
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            raw_data = json.loads(json_match.group())
        else:
            raw_data = json.loads(response)

        if isinstance(raw_data, list):
            valid_items = []
            for idx, item in enumerate(raw_data, start=1):
                if isinstance(item, dict) and "question" in item:
                    valid_items.append({
                        "id": item.get("id", idx),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "category": item.get("category", "General"),
                        "type": item.get("type", "factual"),
                    })
            if valid_items:
                return valid_items

    except Exception as e:
        print(f"Failed to parse LLM test set response: {e}")

    print("Falling back to pre-generated test set...")
    return load_test_set()


def load_test_set() -> List[Dict]:
    """Load the pre-generated test set from JSON file."""
    if os.path.exists(TEST_SET_PATH):
        try:
            with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict) and "question" in item]
        except Exception:
            pass
    return []


def save_test_set(test_set: List[Dict]):
    """Save a test set to the JSON file only if valid."""
    if not isinstance(test_set, list) or not test_set:
        return
    valid_list = [item for item in test_set if isinstance(item, dict) and "question" in item]
    if not valid_list:
        return

    os.makedirs(os.path.dirname(TEST_SET_PATH), exist_ok=True)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_list, f, indent=2, ensure_ascii=False)
