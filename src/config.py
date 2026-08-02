"""
Central configuration for the Baahubali RAG Evaluation Harness.
All tunable parameters, model names, API endpoints, and theme colors.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Providers (Multi-Provider Pool: Gemini -> Groq -> Grok) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("gemini_api_key", "")
GEMINI_MODEL = "gemini-2.0-flash"

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_MODEL = "grok-3-mini-fast"

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --- Retrieval Settings ---
SETTING_A = {
    "name": "Setting A — Dense Vector",
    "short": "A",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "collection_name": "baahubali_dense_1000",
    "top_k": 5,
    "description": "Broad dense context (chunk=1000)",
}

SETTING_B = {
    "name": "Setting B — Hybrid BM25 + Dense (RRF)",
    "short": "B",
    "chunk_size": 350,
    "chunk_overlap": 70,
    "collection_name": "baahubali_hybrid_350",
    "top_k": 5,
    "description": "Granular hybrid retrieval (chunk=350, BM25+Dense RRF)",
}

# --- RRF Parameters ---
RRF_K = 60  # Reciprocal Rank Fusion constant

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "baahubali_knowledge_base.txt")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.json")
ANALYTICS_LOG_PATH = os.path.join(DATA_DIR, "analytics_log.json")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# --- Theme Colors ---
ROYAL_GOLD = "#D4AF37"
DEEP_MAROON = "#22100E"
DARK_BROWN = "#120807"
WARM_CREAM = "#E8DCC8"
MUTED_GOLD = "#B8942E"
EMBER_ORANGE = "#C4722A"
BLOOD_RED = "#8B1A1A"
FOREST_GREEN = "#2D5A3D"

# --- Query Categories ---
QUERY_CATEGORIES = ["Character", "Kingdom", "Battle", "Timeline", "General"]

# --- Guardrail Keywords ---
BAAHUBALI_KEYWORDS = [
    "baahubali", "bahubali", "mahishmati", "amarendra", "mahendra",
    "sivagami", "devasena", "bhallaladeva", "bhallala", "kattappa",
    "katappa", "bijjaladeva", "avantika", "kumara varma", "kalakeya",
    "shivudu", "rajmata", "kuntala", "prabhas", "rajamouli",
    "rana daggubati", "anushka", "ramya krishnan", "sathyaraj",
    "arka media", "waterfall", "throne", "kingdom", "warrior",
    "battle", "sword", "arrow", "army", "siege", "palace",
    "prince", "princess", "king", "queen", "crown", "dynasty",
]
