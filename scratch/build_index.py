"""Build ChromaDB collections and BM25 index for both settings."""
import sys
sys.path.insert(0, '.')

from backend.config import CHUNKS_A_PATH, CHUNKS_B_PATH, SETTING_A, SETTING_B
from backend.ingestion.chunker import load_chunks_jsonl
from backend.vectordb.chroma_store import build_collection, collection_count
from backend.vectordb.bm25_store import build_bm25_index

print("Loading chunks from JSONL files...")
chunks_a = load_chunks_jsonl(str(CHUNKS_A_PATH))
chunks_b = load_chunks_jsonl(str(CHUNKS_B_PATH))
print(f"  Setting A: {len(chunks_a)} chunks loaded")
print(f"  Setting B: {len(chunks_b)} chunks loaded")

print("\nBuilding ChromaDB Setting A collection (dense)...")
build_collection(SETTING_A["collection_name"], chunks_a, force_rebuild=False)
count_a = collection_count(SETTING_A["collection_name"])
print(f"  Collection '{SETTING_A['collection_name']}': {count_a} docs")

print("\nBuilding ChromaDB Setting B collection (hybrid)...")
build_collection(SETTING_B["collection_name"], chunks_b, force_rebuild=False)
count_b = collection_count(SETTING_B["collection_name"])
print(f"  Collection '{SETTING_B['collection_name']}': {count_b} docs")

print("\nBuilding BM25 index from Setting B chunks...")
build_bm25_index(chunks_b, force_rebuild=True)

print("\nVERIFICATION - quick query test...")
from backend.vectordb.chroma_store import query_collection
from backend.vectordb.bm25_store import query_bm25

results_a = query_collection(SETTING_A["collection_name"], "Who is Kattappa?", top_k=3)
results_b = query_bm25("Who is Kattappa?", top_k=3)
print(f"  Dense (A) top result similarity: {results_a[0]['similarity'] if results_a else 'N/A'}")
print(f"  BM25  (B) top result score:      {results_b[0]['bm25_score'] if results_b else 'N/A'}")
print("\nINDEX BUILD COMPLETE")
