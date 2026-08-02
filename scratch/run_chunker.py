"""Run chunker to produce both JSONL files."""
import sys
sys.path.insert(0, '.')

from backend.config import KNOWLEDGE_BASE_PATH, CHUNKS_A_PATH, CHUNKS_B_PATH
from backend.ingestion.pdf_loader import load_knowledge_base
from backend.ingestion.chunker import chunk_for_setting_a, chunk_for_setting_b, save_chunks_jsonl

print("Loading knowledge base...")
docs = load_knowledge_base(str(KNOWLEDGE_BASE_PATH))
total_chars = sum(len(d["text"]) for d in docs)
print(f"Loaded {len(docs)} doc(s), total chars: {total_chars}")

print("\nChunking Setting A (chunk_size=500, overlap=100)...")
chunks_a = chunk_for_setting_a(docs)
save_chunks_jsonl(chunks_a, str(CHUNKS_A_PATH))
avg_a = sum(c["char_count"] for c in chunks_a) / len(chunks_a) if chunks_a else 0

print("\nChunking Setting B (chunk_size=300, overlap=50)...")
chunks_b = chunk_for_setting_b(docs)
save_chunks_jsonl(chunks_b, str(CHUNKS_B_PATH))
avg_b = sum(c["char_count"] for c in chunks_b) / len(chunks_b) if chunks_b else 0

print(f"\n✅ Setting A: {len(chunks_a)} chunks, avg {avg_a:.0f} chars/chunk")
print(f"✅ Setting B: {len(chunks_b)} chunks, avg {avg_b:.0f} chars/chunk")
print("\nCHUNKING COMPLETE")
