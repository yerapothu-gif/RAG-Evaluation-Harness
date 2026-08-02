# 🏰 Mahishmati Archives — Baahubali RAG Evaluation Harness (PS-8)

> **A self-measuring, dual-retrieval RAG system and evaluation harness for the Baahubali saga knowledge base.**

---

## 📌 Executive Summary & Architectural Overview

The **Mahishmati Archives** is an end-to-end RAG system and evaluation harness designed to systematically measure, compare, and optimize retrieval & generation quality on the *Baahubali* movie dataset (`Baahubali_Movie.pdf` / `baahubali_knowledge_base.txt`).

Instead of treating RAG as an unmeasured black box, this system treats **evaluation, structured logging, and comparative analytics** as first-class citizens.

```
                         ┌─────────────────────────────┐
                         │   OFFLINE: INDEXING PATH    │
                         │ Document -> Split -> Embed   │
                         │ ChromaDB + BM25 Store       │
                         └──────────────┬──────────────┘
                                        │
   ┌───────────────┐        ┌──────────▼──────────┐        ┌──────────────┐
   │ Golden QA Set │<======>│   ONLINE: QUERY PATH│=======>│ Analytics Log│
   │ (JSON, curated)│        │ Query -> Guardrail ->│        │   (JSONL)    │
   └───────┬───────┘        │ Setting A / Setting B│        └──────┬───────┘
           │                └──────────┬──────────┘               │
           │                           │                          │
           │                ┌──────────▼──────────┐               │
           └───────────────>│  EVALUATION CHAMBER │<──────────────┘
                            │ Hit@K, MRR, Recall  │
                            │ Faithfulness Judge  │
                            │ Answer Relevance    │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ SQLite Results Store│
                            │ (eval_results.db)   │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ Streamlit Dashboard │
                            │   & FastAPI Server  │
                            └─────────────────────┘
```

---

## 🎯 Implemented Checkpoints & Verification Matrix

| Checkpoint | Status | Implementation Details |
|---|---|---|
| **1. Document Ingestion & Provenance** | ✅ Completed | Text & PDF parser with chunk IDs and character count metadata |
| **2. Dual Retrieval Architectures** | ✅ Completed | **Setting A** (Dense Vector, Chunk 1000) vs **Setting B** (BM25 + Dense + RRF, Chunk 350) |
| **3. Embedding & Vector Indexing** | ✅ Completed | ChromaDB persistent store with local `all-MiniLM-L6-v2` embeddings |
| **4. Golden QA Test Set** | ✅ Completed | 20 hand-curated QA triples including factual, multi-hop, and adversarial out-of-scope questions |
| **5. Deterministic Retrieval Metrics** | ✅ Completed | Hit@K, MRR (Mean Reciprocal Rank), Precision@K, Recall@K |
| **6. LLM-as-a-Judge Evaluation** | ✅ Completed | 4 metrics (Faithfulness, Relevance, Recall, Precision) with **algorithmic heuristic fallbacks** |
| **7. Guardrails & Query Classifier** | ✅ Completed | In-scope/Out-of-scope keyword validator + 5 category buckets (Character, Kingdom, Battle, Timeline, General) |
| **8. Structured Results Store** | ✅ Completed | SQLite database (`evaluation/results.db`) + JSONL analytics logger (`data/analytics_log.json`) |
| **9. Cinematic Interactive Dashboard** | ✅ Completed | Multi-page Streamlit UI with Royal Gold theme, Plotly radar chart, latency vs quality breakdown, 3D character graph |
| **10. REST API Backend** | ✅ Completed | FastAPI backend (`/query`, `/run-eval`, `/results/{id}`, `/compare`, `/export`) |

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional for Grok LLM)
Add your xAI API key to `.env`:
```env
XAI_API_KEY=your_xai_api_key_here
```
*Note: If no API key is provided, the harness automatically uses robust heuristic evaluation and grounded direct context synthesis.*

### 3. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```

### 4. Launch the FastAPI Backend (Optional)
```bash
uvicorn backend.api.main:app --reload --port 8000
```

---

## 📊 Dual Retrieval Comparison Findings

| Axis | Setting A (Dense Vector) | Setting B (Hybrid RRF) | Winner / Insight |
|---|---|---|---|
| **Chunking Strategy** | Broad (1000 char, 200 overlap) | Granular (350 char, 70 overlap) | Setting B provides higher precision |
| **Retrieval Method** | Semantic Cosine Similarity | BM25 + Dense + RRF (k=60) | Setting B captures exact proper nouns (Kattappa, Sivagami) |
| **Retrieval Recall** | High on general thematic queries | Superior on specific entity lookups | **Setting B (Hybrid RRF)** |
| **Context Noise** | Slightly higher | Low (targeted passages) | **Setting B (Hybrid RRF)** |

---

## 🛠️ Repository Layout

```
RAG_bahubali/
├── app.py                      # Main Streamlit Web Application
├── RAG_Eval_Harness_Masterplan.md # Architectural Specification
├── tech.md                     # Technical Stack Details
├── requirements.txt            # Python Dependencies
├── .env                        # Environment Secrets
├── data/
│   ├── baahubali_knowledge_base.txt # Source Knowledge Base
│   ├── test_set.json           # Golden QA Test Set
│   ├── analytics_log.json      # Structured JSONL Query Logs
│   ├── bm25_index.pkl          # Cached BM25 Sparse Index
│   └── chroma_db/              # Persistent Vector Database
├── src/                        # Core Python Engine
│   ├── config.py               # Central Parameters & Colors
│   ├── document_loader.py      # Ingestion & Splitting
│   ├── vector_store.py         # ChromaDB Client & Embeddings
│   ├── hybrid_retriever.py     # BM25 + RRF Fusion
│   ├── llm_engine.py           # Grok LLM Wrapper & Fallback
│   ├── evaluator.py            # LLM-as-a-Judge & Heuristic Fallbacks
│   ├── guardrail.py            # In-scope / Out-of-scope Guardrails
│   ├── query_classifier.py     # 5-Category Query Classifier
│   ├── analytics.py            # Analytics Logging Helper
│   ├── character_graph.py      # 3D Network Graph Data & Plotly Render
│   └── theme.py                # Cinematic Royal CSS Styling
├── backend/                    # FastAPI Microservice Architecture
│   ├── api/main.py             # REST API Endpoints
│   ├── evaluation/             # Metrics & Judge Modules
│   ├── eval_logging/           # SQLite Results Database Handler
│   └── guardrails/             # API Guardrails
└── scratch/
    └── run_full_verification.py# End-to-End System Verification Suite
```
