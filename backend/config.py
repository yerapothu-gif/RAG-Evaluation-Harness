"""
Central configuration for the Baahubali RAG Evaluation Harness.
All tunable parameters, model names, API endpoints, and paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level above backend/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── LLM Configuration (Gemini -> Groq -> Grok Pool) ──────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_MODEL: str = "llama-3.3-70b-versatile"

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("gemini_api_key", "")
GEMINI_MODEL: str = "gemini-2.0-flash"

XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL: str = "https://api.x.ai/v1"
XAI_MODEL: str = "grok-3-mini-fast"

# ── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# ── Retrieval Settings ──────────────────────────────────────────────────────
SETTING_A = {
    "name": "Setting A — Dense Vector",
    "short": "A",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "collection_name": "baahubali_setting_a",
    "top_k": 5,
    "description": "Dense vector cosine similarity (chunk=500, top_k=5)",
}

SETTING_B = {
    "name": "Setting B — Hybrid BM25 + Dense (RRF)",
    "short": "B",
    "chunk_size": 300,
    "chunk_overlap": 50,
    "collection_name": "baahubali_setting_b",
    "top_k": 5,
    "description": "Hybrid BM25 + dense RRF fusion (chunk=300, top_k=5)",
}

# ── RRF Parameters ──────────────────────────────────────────────────────────
RRF_K: int = 60  # Reciprocal Rank Fusion constant

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = _ROOT
DATA_DIR = _ROOT / "data"
LOGS_DIR = _ROOT / "logs"
EVAL_DIR = _ROOT / "evaluation"
BACKEND_DIR = _ROOT / "backend"

KNOWLEDGE_BASE_PATH = DATA_DIR / "baahubali_knowledge_base.pdf"
TEST_SET_PATH = BACKEND_DIR / "evaluation" / "test_set.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
CHROMA_DB_DIR = str(DATA_DIR / "chroma_db")

CHUNKS_A_PATH = DATA_DIR / "chunks_setting_a.jsonl"
CHUNKS_B_PATH = DATA_DIR / "chunks_setting_b.jsonl"

EVAL_RUNS_JSONL = LOGS_DIR / "eval_runs.jsonl"
SQLITE_DB_PATH = EVAL_DIR / "results.db"

# ── Query Categories ─────────────────────────────────────────────────────────
QUERY_TYPES = ["factual", "character", "plot", "adversarial"]

# ── Guardrail Keywords ───────────────────────────────────────────────────────
BAAHUBALI_KEYWORDS = [
    "baahubali", "bahubali", "mahishmati", "amarendra", "mahendra",
    "sivagami", "devasena", "bhallaladeva", "bhallala", "kattappa",
    "katappa", "bijjaladeva", "avantika", "kumara varma", "kalakeya",
    "shivudu", "rajmata", "kuntala", "prabhas", "rajamouli",
    "rana daggubati", "anushka", "ramya krishnan", "sathyaraj",
    "arka media", "waterfall", "throne", "kingdom", "warrior",
    "battle", "sword", "arrow", "army", "siege", "palace",
    "prince", "princess", "king", "queen", "crown", "dynasty",
    "vikramadeva",
]

OUT_OF_SCOPE_INDICATORS = [
    "cricket", "football", "soccer", "basketball", "tennis",
    "election", "president", "prime minister", "politics",
    "stock market", "bitcoin", "cryptocurrency",
    "covid", "pandemic", "vaccine",
    "recipe", "cooking", "restaurant",
    "weather", "forecast", "temperature",
    "capital of", "population of", "currency of",
    "programming", "python code", "javascript",
    "iphone", "android", "google", "apple", "microsoft",
    "social media", "instagram", "tiktok", "twitter",
    "world cup", "olympics", "champions league",
    "elon musk", "jeff bezos", "mark zuckerberg",
    "chatgpt", "artificial intelligence", "machine learning",
    "homework", "math problem", "calculate",
]

# Ensure key directories exist at import time
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
