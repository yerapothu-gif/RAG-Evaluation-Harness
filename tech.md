# Technical Architecture and Stack for RAG Evaluation Harness

This document outlines the complete technology stack, libraries, configurations, and mechanisms required to build the Baahubali RAG Evaluation Harness (PS-8) from scratch.

## 1. Core Environment and Setup
- **Language**: Python 3.9+
- **Environment Management**: `virtualenv` or `conda` (stored in `.venv`)
- **Secret Management**: `.env` file loaded via `python-dotenv` for securely storing API keys (e.g., `XAI_API_KEY`).
- **Dependencies List**: Managed via `requirements.txt`.

## 2. Large Language Models (LLM) & API
- **Primary LLM**: `grok-3-mini-fast` (provided by x.ai).
- **SDK**: `openai>=1.30.0` (The OpenAI Python client is used as a drop-in replacement by pointing the `base_url` to `https://api.x.ai/v1`).
- **Usage**: Used for generating answers to user queries and powering the "LLM-as-a-Judge" evaluation harness.

## 3. Embedding and Vector Database
- **Embedding Model**: `all-MiniLM-L6-v2` via `sentence-transformers>=3.0.0`. This allows for local embedding generation without incurring API costs.
- **Vector Database**: `chromadb>=0.5.0` (Persistent local client).
  - Used for storing document chunks and their dense vector embeddings.
  - Utilizes cosine similarity for retrieval.

## 4. Retrieval Mechanisms & Pipeline
The project features a dual-retrieval comparison dashboard to evaluate different settings side-by-side:

### Setting A (Dense Vector Retrieval)
- **Mechanism**: Pure semantic search using ChromaDB.
- **Chunking Strategy**: Broad context window (Chunk Size: 1000, Overlap: 200).
- **Metric**: Cosine similarity.
- **Top K**: 5

### Setting B (Hybrid Retrieval)
- **Mechanism**: Combines dense vector search with sparse keyword search.
- **Sparse Retriever**: `rank-bm25>=0.2.2` (BM25 algorithm for keyword matching).
- **Chunking Strategy**: Granular context window (Chunk Size: 350, Overlap: 70).
- **Fusion Mechanism**: Reciprocal Rank Fusion (RRF) with a constant `K=60` to normalize and combine scores from both dense and sparse retrievers.
- **Top K**: 5

## 5. Evaluation Harness (LLM-as-a-Judge)
An evaluation system is built to measure the pipeline against a test set (`test_set.json`). It evaluates four key metrics:
1. **Faithfulness**: Measures if the answer is grounded in the retrieved context. (Extracts factual claims and checks them against the context).
2. **Answer Relevance**: Measures how well the answer directly addresses the question.
3. **Context Recall**: Compares retrieved context against the ground truth to see if all necessary facts were retrieved.
4. **Context Precision**: Evaluates the signal-to-noise ratio of the retrieved chunks.

**Fallback Mechanism**: If the LLM fails to return properly formatted JSON for the evaluation, the system falls back to **Heuristic/Algorithmic evaluation** using Jaccard similarity, token intersection, and keyword overlap.

## 6. Frontend and User Interface
- **Web Framework**: `streamlit>=1.35.0` for building the interactive dashboard.
- **Visualizations & Charts**: `plotly>=5.22.0` (for radar charts, bar charts, and comparing Setting A vs Setting B side-by-side).
- **Data Handling**: `pandas>=2.2.0` and `numpy>=1.26.0` for formatting metrics and analytics data.
- **Theme/Styling**: Custom CSS injected into Streamlit to apply a cinematic theme (Royal Gold, Deep Maroon, etc.).

## 7. Additional Python Libraries
- **PDF Processing**: `pypdf>=4.0.0` for extracting text from the source dataset (`Baahubali_Movie.pdf`).
- **Graphing/Knowledge Graph**: `networkx>=3.3` for visualizing character relationships and network graphs.

## 8. Guardrails and Observability
- **Query Guardrails**: A predefined list of domain-specific keywords (`BAAHUBALI_KEYWORDS`) to detect if a user query is "In-Scope" or "Out-of-Scope".
- **Query Classification**: Categorizes user questions into buckets like Character, Kingdom, Battle, Timeline.
- **Observability (Analytics)**: Logs all queries, retrieval times, chunks retrieved, and evaluation scores locally to `analytics_log.json`.

## Summary Checklist to Start Building:
1. Initialize `.venv` and install `requirements.txt`.
2. Setup `.env` with `XAI_API_KEY`.
3. Write the document loader (using `pypdf`) and text chunking logic.
4. Setup ChromaDB client and ingest chunks with Sentence Transformers.
5. Implement BM25 for the hybrid retrieval path.
6. Write the generation script using the OpenAI client hooked to x.ai.
7. Build the Evaluation module (prompts for the 4 metrics + heuristic fallbacks).
8. Connect everything via a Streamlit dashboard with Plotly visualisations.
