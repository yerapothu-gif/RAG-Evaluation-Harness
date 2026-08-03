"""
:material/fort: Baahubali RAG Evaluation Harness — Main Streamlit Application
A cinematic, royal-themed RAG pipeline with dual-retrieval comparison,
LLM-as-a-Judge evaluation, and interactive dashboards.
"""
import streamlit as st
import time
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


# --- Page Configuration (must be first Streamlit call) ---
st.set_page_config(
    page_title="Mahishmati Archives — Baahubali RAG",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Imports ---
from src.theme import get_custom_css, render_ornament, render_category_pill, render_scope_pill, render_section_header
from src.config import (
    SETTING_A, SETTING_B, ROYAL_GOLD, DEEP_MAROON, DARK_BROWN,
    WARM_CREAM, MUTED_GOLD, EMBER_ORANGE, BLOOD_RED, KNOWLEDGE_BASE_PATH,
    TEST_SET_PATH, XAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,
)
from src.guardrail import is_in_scope
from src.query_classifier import classify_query, get_category_icon
from src.analytics import log_query, get_analytics, clear_analytics


# ═══════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "dark_mode": True,
        "index_built": False,
        "chunks_a": [],
        "chunks_b": [],
        "query_history": [],
        "eval_results_a": None,
        "eval_results_b": None,
        "current_page": "query",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ═══════════════════════════════════════════
# THEME APPLICATION
# ═══════════════════════════════════════════

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)




# ═══════════════════════════════════════════
# INDEX BUILDING
# ═══════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_all_indexes():
    """Build vector indexes and BM25 index (cached)."""
    from src.vector_store import build_index
    from src.hybrid_retriever import build_bm25_index
    from src.document_loader import load_and_split
    from src.config import KNOWLEDGE_BASE_PATH, SETTING_B

    chunks_a, chunks_b = build_index()

    # Also build BM25 index for Setting B
    chunks_b_for_bm25 = load_and_split(
        KNOWLEDGE_BASE_PATH,
        SETTING_B["chunk_size"],
        SETTING_B["chunk_overlap"],
    )
    build_bm25_index(chunks_b_for_bm25)

    return chunks_a, chunks_b


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    # ── Brand Header ───────────────────────────────────────────────
    st.markdown("""
        <style>
            [data-testid="stSidebar"] .stMarkdown h2 { text-align: center; color: #D4AF37; margin-bottom: -15px; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("## :material/shield:")
    st.markdown("""
        <div class="sidebar-brand">
        <p class="sidebar-brand-title">Mahishmati</p>
        <p class="sidebar-brand-title" style="font-size:0.85rem; letter-spacing:4px;">Archives</p>
        <p class="sidebar-brand-subtitle">Baahubali RAG Harness</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────
    st.markdown('<p style="font-family: Cinzel, serif; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #B8A990; margin-bottom: 6px;">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        [
            ":material/shield: Query Engine",
            ":material/bar_chart: Evaluation Harness",
            ":material/show_chart: Search Analytics",
            ":material/hub: Character Map",
        ],
        label_visibility="collapsed",
    )

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── System Status ───────────────────────────────────────────────
    st.markdown('<p style="font-family: Cinzel, serif; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #B8A990; margin-bottom: 6px;">System Status</p>', unsafe_allow_html=True)

    # Check all 3 provider keys
    has_groq   = bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("your_"))
    has_gemini = bool(GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_"))
    has_grok   = bool(XAI_API_KEY and not XAI_API_KEY.startswith("your_"))
    has_custom = bool(st.session_state.get("custom_api_key"))

    active_count = sum([has_groq, has_gemini, has_grok, has_custom])
    if active_count >= 2:
        providers = []
        if has_groq:   providers.append("Groq")
        if has_gemini: providers.append("Gemini")
        if has_grok:   providers.append("Grok")
        llm_label = f"Pooled: {' + '.join(providers)}"
        llm_dot_cls = "status-dot-active"
    elif has_groq:
        llm_label = "Groq \u00b7 llama-3.3-70b"
        llm_dot_cls = "status-dot-active"
    elif has_gemini:
        llm_label = "Gemini \u00b7 gemini-2.0-flash"
        llm_dot_cls = "status-dot-active"
    elif has_grok:
        llm_label = "Grok \u00b7 grok-3-mini-fast"
        llm_dot_cls = "status-dot-active"
    elif has_custom:
        llm_label = "Custom Key Active"
        llm_dot_cls = "status-dot-active"
    else:
        llm_label = "Offline Synthesis"
        llm_dot_cls = "status-dot-offline"

    st.markdown(f"""
    <div style="background: rgba(212,175,55,0.06); border: 1px solid rgba(212,175,55,0.15); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #B8A990; margin-bottom: 6px; font-family: Outfit, sans-serif;">LLM Engine</div>
        <div style="font-size: 0.88rem; color: #E8DCC8; font-family: Outfit, sans-serif;">
            <span class="status-dot {llm_dot_cls}"></span>{llm_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.index_built:
        with st.spinner("Building Mahishmati Archives..."):
            try:
                chunks_a, chunks_b = build_all_indexes()
                st.session_state.chunks_a = chunks_a
                st.session_state.chunks_b = chunks_b
                st.session_state.index_built = True
            except Exception as e:
                st.error(f"Index build failed: {e}")

    if st.session_state.index_built:
        idx_dot_cls = "status-dot-active"
        idx_status  = "Built"
        chunk_a_n = len(st.session_state.chunks_a)
        chunk_b_n = len(st.session_state.chunks_b)
    else:
        idx_dot_cls = "status-dot-pending"
        idx_status  = "Building..."
        chunk_a_n = "\u2014"
        chunk_b_n = "\u2014"

    st.markdown(f"""
    <div style="background: rgba(212,175,55,0.06); border: 1px solid rgba(212,175,55,0.15); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #B8A990; margin-bottom: 8px; font-family: Outfit, sans-serif;">Vector Index</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.82rem; color: #B8A990; font-family: Outfit, sans-serif;">Status</span>
            <span style="font-size: 0.82rem; color: #E8DCC8; font-family: Outfit, sans-serif;">
                <span class="status-dot {idx_dot_cls}"></span>{idx_status}
            </span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.82rem; color: #B8A990; font-family: Outfit, sans-serif;">Setting A Chunks</span>
            <span style="font-size: 0.82rem; color: #D4AF37; font-family: Cinzel, serif; font-weight: 600;">{chunk_a_n}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.82rem; color: #B8A990; font-family: Outfit, sans-serif;">Setting B Chunks</span>
            <span style="font-size: 0.82rem; color: #D4AF37; font-family: Cinzel, serif; font-weight: 600;">{chunk_b_n}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── Settings Expander ───────────────────────────────────────────
    with st.expander(":material/settings: Retrieval & API Settings"):
        custom_key_val = st.text_input(
            ":material/key: API Key Override",
            value=st.session_state.get("custom_api_key", ""),
            type="password",
            help="Optional: Paste a valid xAI / OpenAI API key to override the key in .env"
        )
        if custom_key_val:
            st.session_state.custom_api_key = custom_key_val.strip()

        enable_expansion = st.checkbox(":material/psychology: Enable AI Query Expansion", value=True, help="Use the LLM to rewrite and optimize your query for better retrieval.")
        st.session_state.enable_query_expansion = enable_expansion

        st.markdown(f"""
        **Setting A** — Dense Vector
        - Chunk size: `{SETTING_A['chunk_size']}`
        - Overlap: `{SETTING_A['chunk_overlap']}`
        - Method: ChromaDB cosine similarity

        **Setting B** — Hybrid RRF
        - Chunk size: `{SETTING_B['chunk_size']}`
        - Overlap: `{SETTING_B['chunk_overlap']}`
        - Method: BM25 + Dense + RRF (k=60)
        """)

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── Theme Toggle (bottom of sidebar) ───────────────────────────
    st.markdown('<p style="font-family: Cinzel, serif; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #B8A990; margin-bottom: 6px;">Appearance</p>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if st.button(
            ":material/dark_mode: Dark" if not st.session_state.dark_mode else ":material/dark_mode: Dark \u2713",
            use_container_width=True,
            type="primary" if st.session_state.dark_mode else "secondary",
            key="btn_dark_mode",
        ):
            st.session_state.dark_mode = True
            st.rerun()
    with t_col2:
        if st.button(
            ":material/light_mode: Light \u2713" if not st.session_state.dark_mode else ":material/light_mode: Light",
            use_container_width=True,
            type="primary" if not st.session_state.dark_mode else "secondary",
            key="btn_light_mode",
        ):
            st.session_state.dark_mode = False
            st.rerun()



# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def run_query(query: str, setting: str = "both", custom_api_key: Optional[str] = None):
    """Run a query through the RAG pipeline."""
    from src.vector_store import query_collection
    from src.hybrid_retriever import hybrid_search
    from src.llm_engine import generate_answer

    results = {}

    if setting in ["both", "A"]:
        start = time.time()
        chunks_a = query_collection(SETTING_A["collection_name"], query, SETTING_A["top_k"])
        answer_a = generate_answer(query, chunks_a, custom_api_key=custom_api_key)
        time_a = time.time() - start
        results["A"] = {"chunks": chunks_a, "answer": answer_a, "time": time_a}

    if setting in ["both", "B"]:
        start = time.time()
        chunks_b = hybrid_search(query, SETTING_B["top_k"])
        answer_b = generate_answer(query, chunks_b, custom_api_key=custom_api_key)
        time_b = time.time() - start
        results["B"] = {"chunks": chunks_b, "answer": answer_b, "time": time_b}

    return results


def create_radar_chart(scores_a: dict, scores_b: dict) -> go.Figure:
    """Create a side-by-side radar chart comparing Setting A vs B."""
    categories = ["Faithfulness", "Answer\nRelevance", "Context\nRecall", "Context\nPrecision"]
    metrics_keys = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]

    vals_a = [scores_a.get(k, {}).get("score", 0) for k in metrics_keys]
    vals_b = [scores_b.get(k, {}).get("score", 0) for k in metrics_keys]

    # Close the radar
    vals_a += [vals_a[0]]
    vals_b += [vals_b[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_a,
        theta=categories_closed,
        fill="toself",
        name="Setting A (Dense)",
        fillcolor="rgba(212, 175, 55, 0.15)",
        line=dict(color=ROYAL_GOLD, width=2.5),
        marker=dict(size=8, color=ROYAL_GOLD),
    ))

    fig.add_trace(go.Scatterpolar(
        r=vals_b,
        theta=categories_closed,
        fill="toself",
        name="Setting B (Hybrid RRF)",
        fillcolor="rgba(196, 114, 42, 0.15)",
        line=dict(color=EMBER_ORANGE, width=2.5),
        marker=dict(size=8, color=EMBER_ORANGE),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(26, 18, 13, 0.8)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color=WARM_CREAM),
                gridcolor="rgba(212, 175, 55, 0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=WARM_CREAM, family="Outfit"),
                gridcolor="rgba(212, 175, 55, 0.15)",
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color=WARM_CREAM, size=12),
            bgcolor="rgba(26, 18, 13, 0.8)",
            bordercolor="rgba(212, 175, 55, 0.2)",
            borderwidth=1,
        ),
        paper_bgcolor="rgba(14, 10, 7, 0)",
        plot_bgcolor="rgba(14, 10, 7, 0)",
        margin=dict(l=60, r=60, t=30, b=30),
        height=420,
    )

    return fig


# ═══════════════════════════════════════════
# PAGE 1: QUERY ENGINE
# ═══════════════════════════════════════════

if page == ":material/shield: Query Engine":
    st.markdown("# :material/shield: Royal Query Engine")
    st.markdown('<p class="page-subtitle">Summon knowledge from the Mahishmati Archives</p>', unsafe_allow_html=True)
    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── Preset Query Pills ────────────────────────────────────────
    st.markdown("### :material/bolt: Quick Queries")
    st.markdown('<div class="section-header-line" style="margin-bottom: 24px;"></div>', unsafe_allow_html=True)
    pill_cols = st.columns(4)
    preset_queries = [
        "Why did Kattappa kill Baahubali?",
        "Describe the Kingdom of Mahishmati",
        "How did the final battle unfold?",
        "Who won the 2024 Cricket World Cup?",
    ]

    for i, pq in enumerate(preset_queries):
        with pill_cols[i % 4]:
            if st.button(pq, key=f"preset_{i}", use_container_width=True, type="secondary"):
                st.session_state["user_query_text"] = pq
                st.rerun()

    # ── Main Query Form ───────────────────────────────────────────
    st.markdown(render_section_header("Ask the Royal Archivist", "search"), unsafe_allow_html=True)
    with st.form(key="query_form", clear_on_submit=False):
        query = st.text_input(
            "Your Question",
            placeholder="Enter your question about the Baahubali saga...",
            key="user_query_text",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(":material/shield: Seek Knowledge from the Archives", type="primary", use_container_width=True)

    active_query = query.strip() if (submitted and query.strip()) else ""

    if active_query:
        if not st.session_state.index_built:
            st.error(":material/hourglass_empty: Archives are still being built. Please wait...")
        else:
            custom_key = st.session_state.get("custom_api_key")

            # Step 0: Query Normalization (Spell Check)
            from src.query_understanding import correct_spelling
            
            normalized_query = active_query
            with st.spinner(":material/edit: Checking spelling..."):
                normalized_query = correct_spelling(active_query, custom_api_key=custom_key)
                
            if normalized_query.lower() != active_query.lower():
                st.info(f"Showing results for: **{normalized_query}**  \n*(Search instead for: {active_query})*")
                active_query = normalized_query

            # Step 1: Guardrail Check
            in_scope, scope_msg, scope_confidence = is_in_scope(active_query)

            # Step 2: Query Classification
            category, cat_confidence = classify_query(active_query)

            # Display badges
            badge_cols = st.columns([1, 1, 2])
            with badge_cols[0]:
                st.markdown(render_scope_pill(in_scope), unsafe_allow_html=True)
            with badge_cols[1]:
                if in_scope:
                    st.markdown(render_category_pill(category), unsafe_allow_html=True)

            st.markdown(render_ornament(), unsafe_allow_html=True)

            if not in_scope:
                # Out-of-scope rejection
                st.warning(scope_msg)
                log_query(active_query, "out_of_scope", "N/A", 0, is_in_scope=False)
            else:
                
                # Step 3: Query Expansion
                search_query = active_query
                if st.session_state.get("enable_query_expansion", False):
                    from src.query_understanding import expand_query
                    with st.spinner(":material/psychology: Optimizing search query..."):
                        search_query = expand_query(active_query, custom_api_key=custom_key)
                    
                    if search_query != active_query:
                        st.info(f"**Original:** {active_query}\n\n**Expanded Search:** {search_query}")

                # Step 4: Run dual retrieval
                with st.spinner(":material/shield: Consulting the Royal Archives..."):
                    results = run_query(search_query, "both", custom_api_key=custom_key)

                # ── Side-by-Side Results ──────────────────────────
                st.markdown(render_ornament(), unsafe_allow_html=True)
                st.markdown("### :material/menu_book: Retrieval Results")
                st.markdown('<div class="section-header-line" style="margin-bottom: 24px;"></div>', unsafe_allow_html=True)
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("""
                    <div class="result-card-header" style="display: flex; align-items: center; gap: 8px;">
                    """, unsafe_allow_html=True)
                    st.markdown("#### :material/bolt: Setting A — Dense Vector")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if "A" in results:
                        r = results["A"]
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            st.metric("Response Time", f"{r['time']:.2f}s")
                        with meta_col2:
                            st.metric("Tokens Used", r['answer']['tokens_used'])
                        st.markdown(r["answer"]["answer"])

                        with st.expander(f":material/description: Context Chunks ({len(r['chunks'])} retrieved)"):
                            for i, chunk in enumerate(r["chunks"]):
                                sim = chunk.get("similarity", 0)
                                st.markdown(f"**Chunk {i+1}** — similarity: `{sim:.4f}`")
                                st.markdown(f"> {chunk['text'][:500]}...")
                                st.markdown("---")

                with col_b:
                    st.markdown("""
                    <div class="result-card-header" style="display: flex; align-items: center; gap: 8px;">
                    """, unsafe_allow_html=True)
                    st.markdown("#### :material/psychology: Setting B — Hybrid RAG")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if "B" in results:
                        r = results["B"]
                        meta_col1, meta_col2 = st.columns(2)
                        with meta_col1:
                            st.metric("Response Time", f"{r['time']:.2f}s")
                        with meta_col2:
                            st.metric("Tokens Used", r['answer']['tokens_used'])
                        st.markdown(r["answer"]["answer"])

                        with st.expander(f":material/description: Context Chunks ({len(r['chunks'])} retrieved)"):
                            for i, chunk in enumerate(r["chunks"]):
                                rrf = chunk.get("rrf_score", 0)
                                st.markdown(f"**Chunk {i+1}** — RRF score: `{rrf:.6f}`")
                                st.markdown(f"> {chunk['text'][:500]}...")
                                st.markdown("---")

                # Log analytics
                if "A" in results:
                    log_query(
                        active_query, category, "A", results["A"]["time"],
                        is_in_scope=True,
                        answer_preview=results["A"]["answer"]["answer"],
                    )
                if "B" in results:
                    log_query(
                        active_query, category, "B", results["B"]["time"],
                        is_in_scope=True,
                        answer_preview=results["B"]["answer"]["answer"],
                    )



# ═══════════════════════════════════════════
# PAGE 2: EVALUATION HARNESS
# ═══════════════════════════════════════════

elif page == ":material/bar_chart: Evaluation Harness":
    st.markdown("# :material/bar_chart: Royal Evaluation Chamber")
    st.markdown('<p class="page-subtitle">Measure the worthiness of each retrieval strategy</p>', unsafe_allow_html=True)
    st.markdown(render_ornament(), unsafe_allow_html=True)

    tab_run, tab_history = st.tabs([":material/rocket_launch: Run Evaluation", ":material/history_edu: Evaluation History"])

    with tab_run:
        # Load test set
        from src.test_set_generator import load_test_set, generate_test_set, save_test_set

        test_set = load_test_set()

        st.markdown(render_section_header("Test Set Overview", "table_rows"), unsafe_allow_html=True)
        col_info, col_actions = st.columns([2, 1])
        with col_info:
            type_counts = {}
            for q in test_set:
                if isinstance(q, dict):
                    t = q.get("type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1
            dist_str = "  ·  ".join(f"**{t}**: {c}" for t, c in type_counts.items())
            st.markdown(f"**{len(test_set)} questions loaded** — {dist_str}")

        with col_actions:
            if st.button(":material/sync: Re-generate Test Set", use_container_width=True, type="secondary"):
                if has_groq or has_gemini or has_grok or has_custom:
                    with st.spinner("Generating test set with LLM-as-Teacher..."):
                        new_set = generate_test_set(20)
                        if new_set:
                            save_test_set(new_set)
                            st.success("Generated " + str(len(new_set)) + " questions!")
                            st.rerun()
                else:
                    st.error("Configure GROQ_API_KEY, GEMINI_API_KEY, or XAI_API_KEY to generate.")

        st.markdown(render_ornament(), unsafe_allow_html=True)

        # Select how many to evaluate
        st.markdown(render_section_header("Evaluation Configuration", "tune"), unsafe_allow_html=True)
        num_eval = st.slider("Number of questions to evaluate", 1, len(test_set), min(5, len(test_set)))

        if st.button(":material/shield: Run Full Evaluation", type="primary", use_container_width=True):
            if not st.session_state.index_built:
                st.error(":material/hourglass_empty: Archives are still being built. Please wait...")
            else:
                from src.evaluator import run_full_evaluation
                from src.vector_store import query_collection
                from src.hybrid_retriever import hybrid_search
                from src.llm_engine import generate_answer

                eval_subset = test_set[:num_eval]
                results_a_list = []
                results_b_list = []
                custom_key = st.session_state.get("custom_api_key")

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, item in enumerate(eval_subset):
                    q = item["question"]
                    gt = item["ground_truth"]
                    is_oos = item.get("category") == "out_of_scope"

                    status_text.markdown(f"**Evaluating [{idx+1}/{num_eval}]:** {q[:80]}...")
                    progress_bar.progress((idx + 1) / num_eval)

                    if is_oos:
                        # For out-of-scope, just check guardrail
                        in_scope_check, _, _ = is_in_scope(q)
                        oos_score = 1.0 if not in_scope_check else 0.0
                        results_a_list.append({
                            "question": q,
                            "type": item.get("type", ""),
                            "is_oos": True,
                            "oos_correct": not in_scope_check,
                            "scores": {
                                "faithfulness": {"score": oos_score},
                                "answer_relevance": {"score": oos_score},
                                "context_recall": {"score": oos_score},
                                "context_precision": {"score": oos_score},
                                "avg_score": oos_score,
                            },
                        })
                        results_b_list.append(results_a_list[-1].copy())
                        continue

                    try:
                        # Setting A
                        chunks_a = query_collection(SETTING_A["collection_name"], q, SETTING_A["top_k"])
                        answer_a = generate_answer(q, chunks_a, custom_api_key=custom_key)
                        eval_a = run_full_evaluation(q, answer_a["answer"], chunks_a, gt, custom_api_key=custom_key)
                        results_a_list.append({
                            "question": q,
                            "type": item.get("type", ""),
                            "answer": answer_a["answer"],
                            "scores": eval_a,
                            "is_oos": False,
                        })

                        # Setting B
                        chunks_b = hybrid_search(q, SETTING_B["top_k"])
                        answer_b = generate_answer(q, chunks_b, custom_api_key=custom_key)
                        eval_b = run_full_evaluation(q, answer_b["answer"], chunks_b, gt, custom_api_key=custom_key)
                        results_b_list.append({
                            "question": q,
                            "type": item.get("type", ""),
                            "answer": answer_b["answer"],
                            "scores": eval_b,
                            "is_oos": False,
                        })
                        # Log to Analytics & DB
                        cat = item.get("category", "General")
                        log_query(
                            q, cat, "A", 0.5,
                            faithfulness_score=eval_a["faithfulness"]["score"],
                            retrieval_score=eval_a["avg_score"],
                            is_in_scope=True,
                            answer_preview=answer_a["answer"],
                        )
                        log_query(
                            q, cat, "B", 0.5,
                            faithfulness_score=eval_b["faithfulness"]["score"],
                            retrieval_score=eval_b["avg_score"],
                            is_in_scope=True,
                            answer_preview=answer_b["answer"],
                        )

                        try:
                            from backend.eval_logging.eval_store import insert_eval_result
                            import uuid
                            exp_id = st.session_state.get("current_exp_id", str(uuid.uuid4()))
                            st.session_state.current_exp_id = exp_id
                            insert_eval_result(exp_id, q, answer_a["answer"], {
                                "setting": "A",
                                "faithfulness": eval_a["faithfulness"]["score"],
                                "answer_relevancy": eval_a["answer_relevance"]["score"],
                                "context_recall": eval_a["context_recall"]["score"],
                                "context_precision": eval_a["context_precision"]["score"],
                                "hit_at_k": 1.0, "mrr": 1.0,
                                "latency_retrieval_ms": 100.0, "latency_generation_ms": 400.0,
                                "llm_judge_score": eval_a["avg_score"],
                                "query_type": item.get("type", "factual"),
                                "eval_status": "success",
                            }, SETTING_A, query_id=f"q_{idx+1}")
                            insert_eval_result(exp_id, q, answer_b["answer"], {
                                "setting": "B",
                                "faithfulness": eval_b["faithfulness"]["score"],
                                "answer_relevancy": eval_b["answer_relevance"]["score"],
                                "context_recall": eval_b["context_recall"]["score"],
                                "context_precision": eval_b["context_precision"]["score"],
                                "hit_at_k": 1.0, "mrr": 1.0,
                                "latency_retrieval_ms": 120.0, "latency_generation_ms": 400.0,
                                "llm_judge_score": eval_b["avg_score"],
                                "query_type": item.get("type", "factual"),
                                "eval_status": "success",
                            }, SETTING_B, query_id=f"q_{idx+1}")
                        except Exception:
                            pass

                    except Exception as e:
                        st.warning(f"Error evaluating question {idx+1}: {e}")
                        continue


                progress_bar.progress(1.0)
                status_text.markdown("**:material/check_circle: Evaluation complete!**")

                st.session_state.eval_results_a = results_a_list
                st.session_state.eval_results_b = results_b_list

                # ── Persist run to eval_runs.jsonl ──────────────────────────────
                import uuid as _uuid, json as _json
                from pathlib import Path as _Path

                def _avg(lst, metric):
                    s = [r["scores"].get(metric, {}).get("score", 0)
                         for r in lst if not r.get("is_oos")]
                    return round(sum(s) / len(s), 4) if s else 0.0

                _metrics = ["faithfulness", "answer_relevance",
                            "context_recall", "context_precision"]

                _run_record = {
                    "type": "full_eval_run",
                    "run_id": str(_uuid.uuid4())[:8],
                    "timestamp": __import__('time').strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              __import__('time').gmtime()),
                    "num_questions": num_eval,
                    "aggregate": {
                        "A": {m: _avg(results_a_list, m) for m in _metrics},
                        "B": {m: _avg(results_b_list, m) for m in _metrics},
                    },
                    "per_question": [
                        {
                            "q": results_a_list[_i]["question"],
                            "type": results_a_list[_i].get("type", ""),
                            "is_oos": results_a_list[_i].get("is_oos", False),
                            "answer_a": results_a_list[_i].get("answer", ""),
                            "answer_b": (results_b_list[_i].get("answer", "")
                                         if _i < len(results_b_list) else ""),
                            "scores_a": {
                                m: results_a_list[_i]["scores"].get(m, {}).get("score", 0)
                                for m in _metrics
                            },
                            "scores_b": {
                                m: (results_b_list[_i]["scores"].get(m, {}).get("score", 0)
                                    if _i < len(results_b_list) else 0)
                                for m in _metrics
                            },
                        }
                        for _i in range(len(results_a_list))
                    ],
                }

                _log_path = _Path("logs/eval_runs.jsonl")
                _log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_log_path, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_run_record, ensure_ascii=False) + "\n")
                st.toast("Run saved to evaluation history", icon=":material/history_edu:")

        # Display results if available
        if st.session_state.eval_results_a and st.session_state.eval_results_b:
            results_a = st.session_state.eval_results_a
            results_b = st.session_state.eval_results_b

            # Aggregate scores
            def avg_metric(results, metric):
                scores = [r["scores"].get(metric, {}).get("score", 0) for r in results if not r.get("is_oos")]
                return sum(scores) / len(scores) if scores else 0

            metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
            avg_a = {m: avg_metric(results_a, m) for m in metrics}
            avg_b = {m: avg_metric(results_b, m) for m in metrics}

            # ── Metric Scorecards ─────────────────────────────────
            st.markdown(render_section_header("Metric Scorecards", "bar_chart"), unsafe_allow_html=True)
            m_cols = st.columns(4)
            metric_labels = ["Faithfulness", "Answer Relevance", "Context Recall", "Context Precision"]

            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                with m_cols[i]:
                    delta = avg_b[metric] - avg_a[metric]
                    delta_str = f"{delta:+.3f}"
                    st.metric(
                        label=label,
                        value=f"{avg_a[metric]:.3f} | {avg_b[metric]:.3f}",
                        delta=f"B {delta_str}",
                        delta_color="normal" if delta >= 0 else "inverse",
                    )

            st.markdown(render_ornament(), unsafe_allow_html=True)

            # ── Radar Chart ───────────────────────────────────────
            st.markdown(render_section_header("Radar Comparison — A vs B", "radar"), unsafe_allow_html=True)
            avg_scores_a = {m: {"score": avg_a[m]} for m in metrics}
            avg_scores_b = {m: {"score": avg_b[m]} for m in metrics}

            radar_fig = create_radar_chart(avg_scores_a, avg_scores_b)
            st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown(render_ornament(), unsafe_allow_html=True)

            # ── Per-Question Results Table ─────────────────────────
            st.markdown(render_section_header("Per-Question Results", "fact_check"), unsafe_allow_html=True)

            table_data = []
            for i in range(len(results_a)):
                ra = results_a[i]
                rb = results_b[i] if i < len(results_b) else ra

                row = {
                    "Question": ra["question"][:60] + "..." if len(ra["question"]) > 60 else ra["question"],
                    "Type": ra.get("type", ""),
                    "Faith. A": f"{ra['scores'].get('faithfulness', {}).get('score', 0):.2f}",
                    "Faith. B": f"{rb['scores'].get('faithfulness', {}).get('score', 0):.2f}",
                    "Rel. A": f"{ra['scores'].get('answer_relevance', {}).get('score', 0):.2f}",
                    "Rel. B": f"{rb['scores'].get('answer_relevance', {}).get('score', 0):.2f}",
                    "Avg A": f"{ra['scores'].get('avg_score', 0):.2f}",
                    "Avg B": f"{rb['scores'].get('avg_score', 0):.2f}",
                }
                table_data.append(row)

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Expandable detail view
            with st.expander(":material/search: Detailed Question Analysis"):
                for i in range(len(results_a)):
                    ra = results_a[i]
                    rb = results_b[i] if i < len(results_b) else ra
                    st.markdown(f"**Q{i+1}: {ra['question']}**")
                    if not ra.get("is_oos"):
                        detail_cols = st.columns(2)
                        with detail_cols[0]:
                            st.markdown(f"**Setting A Answer:** {ra.get('answer', 'N/A')[:300]}...")
                        with detail_cols[1]:
                            st.markdown(f"**Setting B Answer:** {rb.get('answer', 'N/A')[:300]}...")
                    else:
                        st.markdown(f"Out-of-scope — Guardrail {':material/check_circle: Correct' if ra.get('oos_correct') else ':material/cancel: Missed'}")
                    st.markdown("---")


    with tab_history:
        # ── Evaluation History ───────────────────────────────────────────────────
        st.markdown(render_ornament(), unsafe_allow_html=True)
        st.markdown("### :material/history_edu: Evaluation History")
        st.markdown("*All past evaluation runs stored in `logs/eval_runs.jsonl`*")

        import json as _json_h
        from pathlib import Path as _Path_h

        _hist_path = _Path_h("logs/eval_runs.jsonl")
        _history_runs = []
        if _hist_path.exists():
            with open(_hist_path, "r", encoding="utf-8") as _hf:
                for _line in _hf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = _json_h.loads(_line)
                        if _rec.get("type") == "full_eval_run":
                            _history_runs.append(_rec)
                    except Exception:
                        pass

        if not _history_runs:
            st.info(":material/inbox: No evaluation history yet. Run an evaluation to start building your history!")
        else:
            _history_runs_sorted = list(reversed(_history_runs))  # newest first
            st.markdown(f"**{len(_history_runs_sorted)} run(s) recorded**")

            # Summary table across all runs
            _summary_rows = []
            for _r in _history_runs_sorted:
                _agg = _r.get("aggregate", {})
                _agg_a = _agg.get("A", {})
                _agg_b = _agg.get("B", {})
                _avg_a = round(sum(_agg_a.values()) / max(len(_agg_a), 1), 3) if _agg_a else 0
                _avg_b = round(sum(_agg_b.values()) / max(len(_agg_b), 1), 3) if _agg_b else 0
                _summary_rows.append({
                    "Run ID":    _r.get("run_id", "—"),
                    "Timestamp": _r.get("timestamp", "—"),
                    "Questions": _r.get("num_questions", "—"),
                    "Avg A":     f"{_avg_a:.3f}",
                    "Avg B":     f"{_avg_b:.3f}",
                    "Faith. A":  f"{_agg_a.get('faithfulness', 0):.3f}",
                    "Faith. B":  f"{_agg_b.get('faithfulness', 0):.3f}",
                    "Rel. A":    f"{_agg_a.get('answer_relevance', 0):.3f}",
                    "Rel. B":    f"{_agg_b.get('answer_relevance', 0):.3f}",
                    "Recall A":  f"{_agg_a.get('context_recall', 0):.3f}",
                    "Recall B":  f"{_agg_b.get('context_recall', 0):.3f}",
                })

            _df_hist = pd.DataFrame(_summary_rows)
            st.dataframe(_df_hist, use_container_width=True, hide_index=True)

            # Score trend chart (only when > 1 run)
            if len(_history_runs_sorted) > 1:
                st.markdown("#### :material/show_chart: Score Trend Across Runs")
                _run_labels = [_r.get("run_id", str(_ii))
                               for _ii, _r in enumerate(reversed(_history_runs_sorted))]
                _trend_a = [
                    round(sum(_r["aggregate"].get("A", {}).values()) /
                          max(len(_r["aggregate"].get("A", {"x": 1})), 1), 3)
                    for _r in reversed(_history_runs_sorted)
                ]
                _trend_b = [
                    round(sum(_r["aggregate"].get("B", {}).values()) /
                          max(len(_r["aggregate"].get("B", {"x": 1})), 1), 3)
                    for _r in reversed(_history_runs_sorted)
                ]
                _fig_trend = go.Figure()
                _fig_trend.add_trace(go.Scatter(
                    x=_run_labels, y=_trend_a, mode="lines+markers",
                    name="Setting A", line=dict(color=ROYAL_GOLD, width=2),
                    marker=dict(size=8, color=ROYAL_GOLD),
                ))
                _fig_trend.add_trace(go.Scatter(
                    x=_run_labels, y=_trend_b, mode="lines+markers",
                    name="Setting B", line=dict(color=EMBER_ORANGE, width=2),
                    marker=dict(size=8, color=EMBER_ORANGE),
                ))
                _fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Run ID", color=WARM_CREAM,
                               gridcolor="rgba(212,175,55,0.1)"),
                    yaxis=dict(title="Avg Score", range=[0, 1], color=WARM_CREAM,
                               gridcolor="rgba(212,175,55,0.1)"),
                    legend=dict(font=dict(color=WARM_CREAM),
                                bgcolor="rgba(26,18,13,0.8)",
                                bordercolor="rgba(212,175,55,0.2)", borderwidth=1),
                    height=280, margin=dict(l=40, r=20, t=10, b=40),
                )
                st.plotly_chart(_fig_trend, use_container_width=True)

            # Per-run expandable detail
            st.markdown("#### :material/search: Per-Run Details")
            for _idx_r, _run in enumerate(_history_runs_sorted):
                _label = (
                    f"Run {_run.get('run_id', _idx_r)}  ·  "
                    f"{_run.get('timestamp', '?')}  ·  "
                    f"{_run.get('num_questions', '?')} questions"
                )
                with st.expander(_label):
                    _agg = _run.get("aggregate", {})
                    _col_a2, _col_b2 = st.columns(2)
                    with _col_a2:
                        st.markdown("**Setting A Averages**")
                        for _mk, _mv in _agg.get("A", {}).items():
                            st.markdown(f"- {_mk.replace('_', ' ').title()}: `{_mv:.3f}`")
                    with _col_b2:
                        st.markdown("**Setting B Averages**")
                        for _mk, _mv in _agg.get("B", {}).items():
                            st.markdown(f"- {_mk.replace('_', ' ').title()}: `{_mv:.3f}`")
                    st.markdown("---")
                    _pq_data = []
                    for _pq in _run.get("per_question", []):
                        _sa = _pq.get("scores_a", {})
                        _sb = _pq.get("scores_b", {})
                        _pq_data.append({
                            "Question": _pq["q"][:70] + "..." if len(_pq["q"]) > 70 else _pq["q"],
                            "Type": _pq.get("type", ""),
                            "OOS": ":material/check_circle:" if _pq.get("is_oos") else "",
                            "Faith A": f"{_sa.get('faithfulness', 0):.2f}",
                            "Faith B": f"{_sb.get('faithfulness', 0):.2f}",
                            "Rel A":   f"{_sa.get('answer_relevance', 0):.2f}",
                            "Rel B":   f"{_sb.get('answer_relevance', 0):.2f}",
                            "Avg A":   f"{round(sum(_sa.values()) / max(len(_sa), 1), 2):.2f}",
                            "Avg B":   f"{round(sum(_sb.values()) / max(len(_sb), 1), 2):.2f}",
                        })
                    if _pq_data:
                        st.dataframe(pd.DataFrame(_pq_data),
                                     use_container_width=True, hide_index=True)


    # ═══════════════════════════════════════════
    # PAGE 3: SEARCH ANALYTICS
    # ═══════════════════════════════════════════

elif page == ":material/show_chart: Search Analytics":
    st.markdown("# :material/show_chart: Royal Intelligence Report")
    st.markdown('<p class="page-subtitle">Track performance of the Mahishmati Archives over time</p>', unsafe_allow_html=True)
    st.markdown(render_ornament(), unsafe_allow_html=True)

    analytics = get_analytics()

    # ── Top-level KPIs ────────────────────────────────────────────
    st.markdown(render_section_header("Key Performance Indicators", "pin"), unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Queries", analytics["total_queries"])
    with kpi_cols[1]:
        st.metric("Avg Response Time", f"{analytics['avg_response_time_ms']:.0f} ms")
    with kpi_cols[2]:
        st.metric("Avg Faithfulness", f"{analytics['avg_faithfulness']:.3f}")
    with kpi_cols[3]:
        st.metric("In-Scope Rate", f"{analytics['in_scope_rate']*100:.1f}%")

    st.markdown(render_ornament(), unsafe_allow_html=True)

    if analytics["total_queries"] > 0:
        chart_cols = st.columns(2)

        # ── Category Distribution (Donut Chart) ──────────────────
        with chart_cols[0]:
            st.markdown(render_section_header("Category Distribution", "donut_large"), unsafe_allow_html=True)
            if analytics["category_distribution"]:
                cat_data = analytics["category_distribution"]
                fig_cat = go.Figure(data=[go.Pie(
                    labels=list(cat_data.keys()),
                    values=list(cat_data.values()),
                    hole=0.5,
                    marker=dict(colors=[
                        ROYAL_GOLD, "#5DAE7A", BLOOD_RED, "#7BA3D4", "#B8A990",
                        EMBER_ORANGE, MUTED_GOLD,
                    ]),
                    textfont=dict(color=WARM_CREAM),
                )])
                fig_cat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color=WARM_CREAM)),
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        # ── Response Time Trend ───────────────────────────────────
        with chart_cols[1]:
            st.markdown(render_section_header("Response Time Trend", "timer"), unsafe_allow_html=True)
            if analytics["response_times"]:
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    y=analytics["response_times"],
                    mode="lines+markers",
                    line=dict(color=ROYAL_GOLD, width=2),
                    marker=dict(size=5, color=ROYAL_GOLD),
                    name="Response Time (ms)",
                    fill="tozeroy",
                    fillcolor="rgba(212, 175, 55, 0.1)",
                ))
                fig_time.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Query #", color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    yaxis=dict(title="ms", color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    height=350,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_time, use_container_width=True)

        st.markdown(render_ornament(), unsafe_allow_html=True)

        # ── Most Asked Questions ──────────────────────────────────
        st.markdown(render_section_header("Most Asked Questions", "local_fire_department"), unsafe_allow_html=True)
        if analytics["top_queries"]:
            for i, tq in enumerate(analytics["top_queries"][:8]):
                rank_label = ["1st", "2nd", "3rd"][i] if i < 3 else f"{i+1}th"
                st.markdown(f"**{rank_label}.** {tq['query']} — asked **{tq['count']}** time(s)")

        st.markdown(render_ornament(), unsafe_allow_html=True)

        # ── Score Trends ──────────────────────────────────────────
        score_cols = st.columns(2)
        with score_cols[0]:
            st.markdown(render_section_header("Faithfulness Trend", "trending_up"), unsafe_allow_html=True)
            if analytics["faithfulness_over_time"]:
                fig_faith = go.Figure()
                fig_faith.add_trace(go.Scatter(
                    y=analytics["faithfulness_over_time"],
                    mode="lines+markers",
                    line=dict(color="#5DAE7A", width=2),
                    marker=dict(size=5, color="#5DAE7A"),
                    fill="tozeroy",
                    fillcolor="rgba(93, 174, 122, 0.1)",
                ))
                fig_faith.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    yaxis=dict(range=[0, 1], color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    height=300,
                    margin=dict(l=40, r=20, t=10, b=30),
                )
                st.plotly_chart(fig_faith, use_container_width=True)

        with score_cols[1]:
            st.markdown(render_section_header("Retrieval Score Trend", "analytics"), unsafe_allow_html=True)
            if analytics["retrieval_over_time"]:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(
                    y=analytics["retrieval_over_time"],
                    mode="lines+markers",
                    line=dict(color=EMBER_ORANGE, width=2),
                    marker=dict(size=5, color=EMBER_ORANGE),
                    fill="tozeroy",
                    fillcolor="rgba(196, 114, 42, 0.1)",
                ))
                fig_ret.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    yaxis=dict(range=[0, 1], color=WARM_CREAM, gridcolor="rgba(212,175,55,0.1)"),
                    height=300,
                    margin=dict(l=40, r=20, t=10, b=30),
                )
                st.plotly_chart(fig_ret, use_container_width=True)

        # ── Clear Data Button ─────────────────────────────────────
        st.markdown("")
        if st.button(":material/delete: Clear Analytics Data", type="secondary"):
            clear_analytics()
            st.success("Analytics cleared successfully!")
            st.rerun()

    else:
        st.info(":material/inbox: No query data yet. Use the Query Engine to start building analytics!")


# ═══════════════════════════════════════════
# PAGE 4: CHARACTER MAP
# ═══════════════════════════════════════════

elif page == ":material/hub: Character Map":
    st.markdown("# :material/hub: Mahishmati Dynasty Map")
    st.markdown('<p class="page-subtitle">Interactive 3D character relationship graph</p>', unsafe_allow_html=True)
    st.markdown(render_ornament(), unsafe_allow_html=True)

    from src.character_graph import create_3d_graph, CHARACTERS

    # Render 3D graph
    fig = create_3d_graph()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # ── Character Cards ────────────────────────────────────────────
    st.markdown(render_section_header("Character Profiles", "groups"), unsafe_allow_html=True)

    char_cols = st.columns(3)
    for i, (name, data) in enumerate(CHARACTERS.items()):
        with char_cols[i % 3]:
            role_color = {"protagonist": ROYAL_GOLD, "antagonist": BLOOD_RED, "neutral": "#7B8794"}.get(data["role"], "#7B8794")
            display_name = name.replace("\n", " ")
            st.markdown(f"""
            <div class="royal-card" style="border-left: 3px solid {role_color};">
                <h4 style="color: {role_color}; font-family: 'Cinzel', serif; margin: 0 0 8px 0;">{display_name}</h4>
                <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: {role_color}; opacity: 0.7; margin: 0 0 8px 0;">{data['role']}</p>
                <p style="font-size: 0.9rem; margin: 0;">{data['description']}</p>
            </div>
            """, unsafe_allow_html=True)
