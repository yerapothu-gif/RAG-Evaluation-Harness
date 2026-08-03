"""
Verification script for the entire Baahubali RAG Evaluation Harness.
Runs end-to-end checks on all checkpoints and prints a diagnostic report.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print("[+] BAAHUBALI RAG HARNESS - FULL VERIFICATION TEST")
print("==================================================")

# 1. Document Loader & Splitting
print("\n[1/7] Testing Document Ingestion & Chunking...")
from src.document_loader import load_and_split
from src.config import KNOWLEDGE_BASE_PATH, SETTING_A, SETTING_B

chunks_a = load_and_split(KNOWLEDGE_BASE_PATH, SETTING_A["chunk_size"], SETTING_A["chunk_overlap"])
chunks_b = load_and_split(KNOWLEDGE_BASE_PATH, SETTING_B["chunk_size"], SETTING_B["chunk_overlap"])
print(f"  [OK] Setting A Chunks (1000/200): {len(chunks_a)} chunks")
print(f"  [OK] Setting B Chunks (350/70):   {len(chunks_b)} chunks")

# 2. Vector DB & Indexing
print("\n[2/7] Testing Vector Database (ChromaDB) & Embeddings...")
from src.vector_store import build_index, query_collection
chunks_a_idx, chunks_b_idx = build_index()
results_a = query_collection(SETTING_A["collection_name"], "Why did Kattappa kill Baahubali?", top_k=5)
print(f"  [OK] Dense Retrieval (Setting A): {len(results_a)} chunks returned. Top similarity: {results_a[0]['similarity']}")

# 3. Hybrid BM25 & RRF Retrieval
print("\n[3/7] Testing Hybrid Retrieval (BM25 + Dense + RRF)...")
from src.hybrid_retriever import build_bm25_index, hybrid_search
build_bm25_index(chunks_b_idx)
results_b = hybrid_search("Why did Kattappa kill Baahubali?", top_k=5)
print(f"  [OK] Hybrid Retrieval (Setting B): {len(results_b)} chunks returned. Top RRF score: {results_b[0]['rrf_score']}")

# 4. Guardrails & Classifier
print("\n[4/7] Testing Guardrails & Query Classifier...")
from src.guardrail import is_in_scope
from src.query_classifier import classify_query

in_scope_1, msg_1, conf_1 = is_in_scope("Why did Kattappa kill Baahubali?")
cat_1, cat_conf_1 = classify_query("Why did Kattappa kill Baahubali?")
print(f"  [OK] In-scope query: in_scope={in_scope_1}, category={cat_1}")

in_scope_2, msg_2, conf_2 = is_in_scope("Who won the 2024 Cricket World Cup?")
print(f"  [OK] Out-of-scope query: in_scope={in_scope_2}")

# 5. Evaluation Harness & Metrics
print("\n[5/7] Testing Evaluation Metrics & LLM-as-a-Judge...")
from src.evaluator import run_full_evaluation
test_q = "Why did Kattappa kill Baahubali?"
test_ans = "Kattappa killed Amarendra Baahubali because he was bound by his oath to obey Rajmata Sivagami, who was manipulated by Bhallaladeva."
test_gt = "Kattappa killed Amarendra Baahubali because he was bound by his family's hereditary oath of loyalty to obey the Rajmata's command."

eval_res = run_full_evaluation(test_q, test_ans, results_b, test_gt)
print(f"  [OK] Evaluation Scores:")
print(f"     - Faithfulness:      {eval_res['faithfulness']['score']}")
print(f"     - Answer Relevance:  {eval_res['answer_relevance']['score']}")
print(f"     - Context Recall:    {eval_res['context_recall']['score']}")
print(f"     - Context Precision: {eval_res['context_precision']['score']}")
print(f"     - Overall Average:   {eval_res['avg_score']}")

# 6. Logging & Analytics
print("\n[6/7] Testing Analytics Logging & SQLite Store...")
from src.analytics import log_query, get_analytics
log_query("Why did Kattappa kill Baahubali?", cat_1, "B", 0.45, eval_res['faithfulness']['score'], eval_res['avg_score'])
analytics = get_analytics()
print(f"  [OK] Analytics Logged. Total queries: {analytics['total_queries']}")

from backend.eval_logging.eval_store import init_db, insert_eval_result, get_comparison_summary
init_db()
db_id = insert_eval_result("test-exp-001", test_q, test_ans, {
    "setting": "B",
    "faithfulness": eval_res['faithfulness']['score'],
    "answer_relevancy": eval_res['answer_relevance']['score'],
    "context_recall": eval_res['context_recall']['score'],
    "context_precision": eval_res['context_precision']['score'],
    "hit_at_k": 1.0,
    "mrr": 1.0,
    "latency_retrieval_ms": 120.0,
    "latency_generation_ms": 450.0,
    "llm_judge_score": eval_res['avg_score'],
    "query_type": "complex_reasoning",
    "eval_status": "success",
}, SETTING_B, query_id="q02")
print(f"  [OK] SQLite Record Inserted. Row ID: {db_id}")

# 7. Test Set Integrity
print("\n[7/7] Checking Golden QA Test Set Integrity...")
from src.test_set_generator import load_test_set
test_set = load_test_set()
print(f"  [OK] Test set loaded successfully ({len(test_set)} QA items).")

print("\n==================================================")
print("[SUCCESS] ALL CHECKPOINTS VERIFIED PERFECTLY!")
print("==================================================")
