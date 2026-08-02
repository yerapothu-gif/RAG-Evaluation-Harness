import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.vector_store import build_index, query_collection
from src.hybrid_retriever import hybrid_search
from src.evaluator import run_full_evaluation

print("Testing Evaluation Harness...")
chunks_a, chunks_b = build_index()

test_questions = [
    {
        "q": "Why did Kattappa kill Baahubali?",
        "gt": "Kattappa killed Baahubali on the secret orders of Rajmata Sivagami, who was deceived by Bhallaladeva and Bijjaladeva."
    },
    {
        "q": "Who is the king of Mahishmati?",
        "gt": "Bhallaladeva became the king of Mahishmati, but later Mahendra Baahubali reclaimed the throne."
    }
]

for item in test_questions:
    q = item["q"]
    gt = item["gt"]
    chunks = query_collection("baahubali_dense_1000", q, 5)
    ans_text = f"Based on the archives, {gt}"
    
    scores = run_full_evaluation(q, ans_text, chunks, gt)
    print(f"\nQuestion: {q}")
    print(f"Faithfulness Score: {scores['faithfulness']['score']} ({scores['faithfulness']['reasoning']})")
    print(f"Relevance Score:    {scores['answer_relevance']['score']} ({scores['answer_relevance']['reasoning']})")
    print(f"Recall Score:       {scores['context_recall']['score']} ({scores['context_recall']['reasoning']})")
    print(f"Precision Score:    {scores['context_precision']['score']} ({scores['context_precision']['reasoning']})")
    print(f"Average Score:      {scores['avg_score']}")
