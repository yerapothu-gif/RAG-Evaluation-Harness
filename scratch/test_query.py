import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

print("1. Testing config...")
from src.config import SETTING_A, SETTING_B, XAI_API_KEY
print(f"XAI_API_KEY loaded: {bool(XAI_API_KEY and XAI_API_KEY != 'your_xai_api_key_here')}")

print("\n2. Testing vector store index build / query...")
try:
    from src.vector_store import build_index, query_collection
    chunks_a, chunks_b = build_index(force_rebuild=False)
    print("Index built successfully!")
    res = query_collection(SETTING_A["collection_name"], "Who is Amarendra Baahubali?", 3)
    print(f"Retrieved {len(res)} chunks for Setting A.")
except Exception as e:
    print(f"Vector Store Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing LLM generation...")
try:
    from src.llm_engine import generate_answer
    ans = generate_answer("Who is Amarendra Baahubali?", res if 'res' in locals() else [])
    print(f"LLM Answer: {ans}")
except Exception as e:
    print(f"LLM Error: {e}")
    import traceback
    traceback.print_exc()
