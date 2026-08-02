"""
🏰 Baahubali RAG Evaluation Harness — Main Streamlit Application
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
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Imports ---
from src.theme import get_custom_css, render_ornament, render_category_pill, render_scope_pill
from src.config import (
    SETTING_A, SETTING_B, ROYAL_GOLD, DEEP_MAROON, DARK_BROWN,
    WARM_CREAM, MUTED_GOLD, EMBER_ORANGE, BLOOD_RED, KNOWLEDGE_BASE_PATH,
    TEST_SET_PATH, XAI_API_KEY,
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
    st.markdown("# ⚔️ Mahishmati\nArchives")
    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Dark/Light Mode Toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Dark", use_container_width=True,
                      type="primary" if st.session_state.dark_mode else "secondary"):
            st.session_state.dark_mode = True
            st.rerun()
    with col2:
        if st.button("☀️ Light", use_container_width=True,
                      type="primary" if not st.session_state.dark_mode else "secondary"):
            st.session_state.dark_mode = False
            st.rerun()

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Navigation
    st.markdown("### 📜 Navigation")
    page = st.radio(
        "Select Page",
        ["⚔️ Query Engine", "📊 Evaluation Harness", "📈 Search Analytics", "🕸️ Character Map"],
        label_visibility="collapsed",
    )

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # System Status
    st.markdown("### 🏰 System Status")

    # Build indexes on first load
    api_key_set = XAI_API_KEY and XAI_API_KEY != "your_xai_api_key_here"
    st.markdown(f"**LLM Engine:** {'✅ Grok Connected' if api_key_set else '⚠️ Set XAI_API_KEY in .env'}")

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
        st.markdown(f"**Vector Index:** ✅ Built")
        st.markdown(f"**Setting A Chunks:** {len(st.session_state.chunks_a)}")
        st.markdown(f"**Setting B Chunks:** {len(st.session_state.chunks_b)}")
    else:
        st.markdown("**Vector Index:** ⏳ Building...")

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Settings Info
    with st.expander("⚙️ Retrieval & API Settings"):
        custom_key_val = st.text_input(
            "🔑 API Key Override",
            value=st.session_state.get("custom_api_key", ""),
            type="password",
            help="Optional: Paste a valid xAI / OpenAI API key to override the key in .env"
        )
        if custom_key_val:
            st.session_state.custom_api_key = custom_key_val.strip()

        st.markdown(f"""
        **Setting A** — Dense Vector
        - Chunk size: {SETTING_A['chunk_size']}
        - Overlap: {SETTING_A['chunk_overlap']}
        - Method: ChromaDB cosine similarity

        **Setting B** — Hybrid RRF
        - Chunk size: {SETTING_B['chunk_size']}
        - Overlap: {SETTING_B['chunk_overlap']}
        - Method: BM25 + Dense + RRF (k=60)
        """)



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

if page == "⚔️ Query Engine":
    st.markdown("# ⚔️ Royal Query Engine")
    st.markdown("*Summon knowledge from the Mahishmati Archives*")
    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Preset query pills
    st.markdown("##### 🎯 Quick Queries")
    pill_cols = st.columns(4)
    preset_queries = [
        "Why did Kattappa kill Baahubali?",
        "Describe the Kingdom of Mahishmati",
        "How did the final battle unfold?",
        "Who won the 2024 Cricket World Cup?",
    ]

    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    for i, pq in enumerate(preset_queries):
        with pill_cols[i % 4]:
            if st.button(pq, key=f"preset_{i}", use_container_width=True):
                st.session_state.query_input = pq

    # Main query form (supports Enter key & Button click)
    with st.form(key="query_form", clear_on_submit=False):
        query = st.text_input(
            "🔍 Ask the Royal Archivist",
            value=st.session_state.query_input,
            placeholder="Enter your question about the Baahubali saga... (Press Enter or click below)",
            key="user_query_text",
        )
        submitted = st.form_submit_button("⚔️ Seek Knowledge", type="primary", use_container_width=True)

    active_query = query.strip() if (submitted and query.strip()) else ""

    if active_query:
        if not st.session_state.index_built:
            st.error("⏳ Archives are still being built. Please wait...")
        else:
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
                # Step 3: Run dual retrieval
                custom_key = st.session_state.get("custom_api_key")
                with st.spinner("⚔️ Consulting the Royal Archives..."):
                    results = run_query(active_query, "both", custom_api_key=custom_key)

                # Display side-by-side results
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### 🟡 Setting A — Dense Vector")
                    if "A" in results:
                        r = results["A"]
                        st.markdown(f"⏱️ Response time: **{r['time']:.2f}s**")
                        st.markdown(f"📊 Tokens used: **{r['answer']['tokens_used']}**")
                        st.markdown("---")
                        st.markdown(r["answer"]["answer"])

                        with st.expander(f"📄 Context Chunks ({len(r['chunks'])} retrieved)"):
                            for i, chunk in enumerate(r["chunks"]):
                                sim = chunk.get("similarity", 0)
                                st.markdown(f"**Chunk {i+1}** (similarity: {sim:.4f})")
                                st.markdown(f"> {chunk['text'][:500]}...")
                                st.markdown("---")

                with col_b:
                    st.markdown("### 🟠 Setting B — Hybrid RRF")
                    if "B" in results:
                        r = results["B"]
                        st.markdown(f"⏱️ Response time: **{r['time']:.2f}s**")
                        st.markdown(f"📊 Tokens used: **{r['answer']['tokens_used']}**")
                        st.markdown("---")
                        st.markdown(r["answer"]["answer"])

                        with st.expander(f"📄 Context Chunks ({len(r['chunks'])} retrieved)"):
                            for i, chunk in enumerate(r["chunks"]):
                                rrf = chunk.get("rrf_score", 0)
                                st.markdown(f"**Chunk {i+1}** (RRF score: {rrf:.6f})")
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

elif page == "📊 Evaluation Harness":
    st.markdown("# 📊 Royal Evaluation Chamber")
    st.markdown("*Measure the worthiness of each retrieval strategy*")
    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Load test set
    from src.test_set_generator import load_test_set, generate_test_set, save_test_set

    test_set = load_test_set()

    col_info, col_actions = st.columns([2, 1])
    with col_info:
        st.markdown(f"**Test Set:** {len(test_set)} questions loaded")
        type_counts = {}
        for q in test_set:
            t = q.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        st.markdown(f"Distribution: {', '.join(f'{t}: {c}' for t, c in type_counts.items())}")

    with col_actions:
        if st.button("🔄 Re-generate Test Set (LLM)", use_container_width=True):
            if XAI_API_KEY and XAI_API_KEY != "your_xai_api_key_here":
                with st.spinner("Generating test set with LLM-as-Teacher..."):
                    new_set = generate_test_set(20)
                    if new_set:
                        save_test_set(new_set)
                        st.success(f"Generated {len(new_set)} questions!")
                        st.rerun()
            else:
                st.error("Set XAI_API_KEY to use LLM generation.")

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Select how many to evaluate
    num_eval = st.slider("Questions to evaluate", 1, len(test_set), min(5, len(test_set)))

    if st.button("⚔️ Run Full Evaluation", type="primary", use_container_width=True):
        if not st.session_state.index_built:
            st.error("⏳ Archives are still being built. Please wait...")
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
                except Exception as e:
                    st.warning(f"Error evaluating question {idx+1}: {e}")
                    continue


            progress_bar.progress(1.0)
            status_text.markdown("**✅ Evaluation complete!**")

            st.session_state.eval_results_a = results_a_list
            st.session_state.eval_results_b = results_b_list

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

        # Metric Scorecards
        st.markdown("### 📊 Metric Scorecards")
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

        # Radar Chart
        st.markdown("### 🎯 Radar Comparison")
        avg_scores_a = {m: {"score": avg_a[m]} for m in metrics}
        avg_scores_b = {m: {"score": avg_b[m]} for m in metrics}

        radar_fig = create_radar_chart(avg_scores_a, avg_scores_b)
        st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown(render_ornament(), unsafe_allow_html=True)

        # Per-question Results Table
        st.markdown("### 📋 Per-Question Results")

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
        with st.expander("🔍 Detailed Question Analysis"):
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
                    st.markdown(f"Out-of-scope — Guardrail {'✅ Correct' if ra.get('oos_correct') else '❌ Missed'}")
                st.markdown("---")


# ═══════════════════════════════════════════
# PAGE 3: SEARCH ANALYTICS
# ═══════════════════════════════════════════

elif page == "📈 Search Analytics":
    st.markdown("# 📈 Royal Intelligence Report")
    st.markdown("*Track the performance of the Mahishmati Archives*")
    st.markdown(render_ornament(), unsafe_allow_html=True)

    analytics = get_analytics()

    # Top-level KPIs
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Queries", analytics["total_queries"])
    with kpi_cols[1]:
        st.metric("Avg Response Time", f"{analytics['avg_response_time_ms']:.0f}ms")
    with kpi_cols[2]:
        st.metric("Avg Faithfulness", f"{analytics['avg_faithfulness']:.3f}")
    with kpi_cols[3]:
        st.metric("In-Scope Rate", f"{analytics['in_scope_rate']*100:.1f}%")

    st.markdown(render_ornament(), unsafe_allow_html=True)

    if analytics["total_queries"] > 0:
        chart_cols = st.columns(2)

        # Category Distribution (Donut Chart)
        with chart_cols[0]:
            st.markdown("### 📊 Query Category Distribution")
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

        # Response Time Over Queries (Line Chart)
        with chart_cols[1]:
            st.markdown("### ⏱️ Response Time Trend")
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

        # Most Asked Questions
        st.markdown("### 🔥 Most Asked Questions")
        if analytics["top_queries"]:
            for i, tq in enumerate(analytics["top_queries"][:8]):
                st.markdown(f"**{i+1}.** {tq['query']} — asked **{tq['count']}** time(s)")

        st.markdown(render_ornament(), unsafe_allow_html=True)

        # Faithfulness & Retrieval Over Time
        score_cols = st.columns(2)
        with score_cols[0]:
            st.markdown("### 📏 Faithfulness Trend")
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
            st.markdown("### 📏 Retrieval Score Trend")
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

        # Clear analytics button
        if st.button("🗑️ Clear Analytics Data"):
            clear_analytics()
            st.success("Analytics cleared!")
            st.rerun()

    else:
        st.info("📭 No query data yet. Use the Query Engine to start building analytics!")


# ═══════════════════════════════════════════
# PAGE 4: CHARACTER MAP
# ═══════════════════════════════════════════

elif page == "🕸️ Character Map":
    st.markdown("# 🕸️ Mahishmati Dynasty Map")
    st.markdown("*Interactive 3D character relationship graph*")
    st.markdown(render_ornament(), unsafe_allow_html=True)

    from src.character_graph import create_3d_graph, CHARACTERS

    # Render 3D graph
    fig = create_3d_graph()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(render_ornament(), unsafe_allow_html=True)

    # Character Cards
    st.markdown("### 👥 Character Profiles")

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
