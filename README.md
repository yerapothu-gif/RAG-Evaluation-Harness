# 🏰 Baahubali RAG Evaluation Harness

Welcome to the **Mahishmati Archives**, a cinematic, royal-themed RAG (Retrieval-Augmented Generation) pipeline and evaluation harness built around the lore of the Baahubali universe. 

This project demonstrates a comprehensive RAG application with a dual-retrieval comparison, LLM-as-a-Judge evaluation metrics, interactive analytics dashboards, and a 3D character relationship map.

## ✨ Features

- **⚔️ Royal Query Engine**: Ask questions about the Baahubali saga and receive answers generated from the knowledge base. Features out-of-scope query rejection (Guardrails) and query classification.
- **⚖️ Dual-Retrieval Comparison**: Compare two retrieval strategies side-by-side:
  - *Setting A*: Dense Vector Search (ChromaDB cosine similarity, chunk size 1000).
  - *Setting B*: Hybrid Search (BM25 + Dense + Reciprocal Rank Fusion, chunk size 350).
- **📊 Evaluation Harness**: Use LLM-as-a-Judge to evaluate retrieval quality across four key metrics:
  - Faithfulness
  - Answer Relevance
  - Context Recall
  - Context Precision
  - *New*: Re-generate evaluation test sets dynamically using an LLM-as-a-Teacher approach.
- **📈 Search Analytics**: Track system performance, user query distribution, average response times, and retrieval metrics over time in beautiful visual dashboards.
- **🕸️ Character Map**: Explore a 3D interactive knowledge graph of the Mahishmati dynasty characters and their relationships.
- **🎨 Cinematic UI**: A fully custom, dark-themed UI with royal gold and deep maroon accents, designed with Streamlit.

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Plotly (for interactive charts and 3D graphs)
- **RAG Pipeline**: ChromaDB (Vector Store), Sentence Transformers (`all-MiniLM-L6-v2`)
- **Hybrid Search**: Rank-BM25
- **LLM Engine**: xAI API (Grok models like `grok-3-mini-fast`)
- **Evaluation**: Custom LLM-as-a-Judge implementation
- **Data Processing**: PyPDF, NetworkX, Pandas, Numpy

## 📂 Project Structure

```text
RAG_bahubali/
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── app.py                  # Main Streamlit application
├── data/                   # Data directory
│   ├── baahubali_knowledge_base.txt
│   ├── test_set.json
│   └── ...
└── src/                    # Core modules
    ├── analytics.py        # Query logging and analytics logic
    ├── character_graph.py  # 3D character graph generation
    ├── config.py           # Configuration parameters and theme settings
    ├── document_loader.py  # Document chunking and parsing
    ├── evaluator.py        # LLM-as-a-Judge evaluation logic
    ├── guardrail.py        # Out-of-scope detection
    ├── hybrid_retriever.py # BM25 + Dense retrieval with RRF
    ├── llm_engine.py       # xAI API integration for generation
    ├── query_classifier.py # Query categorization
    ├── test_set_generator.py# Automated test set generation via LLM
    ├── theme.py            # Streamlit custom CSS and UI components
    └── vector_store.py     # ChromaDB vector index management
```

## 🚀 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/RAG_bahubali.git
   cd RAG_bahubali
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Update the `.env` file in the root directory with your xAI API key:
   ```env
   XAI_API_KEY=your_xai_api_key_here
   ```

## 🎮 Usage

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

The application will launch in your default web browser. On the first run, the system will automatically build the vector indexes (ChromaDB) and BM25 indexes based on the provided knowledge base.

Navigate through the sidebar to explore the Query Engine, Evaluation Harness, Search Analytics, and Character Map.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
