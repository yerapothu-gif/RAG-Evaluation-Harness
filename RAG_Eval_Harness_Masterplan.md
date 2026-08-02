# PS-8 — RAG Evaluation Harness: Master Plan
**Dataset:** Baahubali_Movie.pdf (single-document knowledge base)
**Base scaffold:** csi-vnrvjiet/build-with-rag-workshop (11-phase pipeline)
**Deliverable focus:** Not "a RAG bot" — a RAG *system that measures itself*. The eval harness, logging, and comparison dashboard ARE the project; the QA bot is just the thing being measured.

---

## 1. How This Will Be Graded (read this first, design backward from it)

The brief has 4 explicit requirements. Everything below is organized to hit these directly:

| Requirement | What "good" looks like |
|---|---|
| Build a RAG pipeline | Clean, swappable stages (ingest → chunk → embed → store → retrieve → generate) — not a monolith |
| Small QA test set | 15–25 hand-verified (question, ground-truth answer, ground-truth source chunk) triples, not auto-generated garbage |
| Faithfulness metric | LLM-as-judge with a strict rubric + a citation-based grounding check, not just "vibes" |
| Retrieval quality metric | Hit-rate / Recall@k / MRR against ground-truth chunks — deterministic, not judged |
| Eval harness | A script that runs N questions through the pipeline and spits out a scored table + logs, repeatable across configs |
| Bonus: dashboard | Two retrieval configs run side-by-side with the same test set, metrics compared visually |

The single biggest differentiator between a "workshop clone" submission and a strong one is **treating eval as a first-class pipeline stage with its own logs and reproducibility**, not a notebook you ran once.

---

## 2. High-Level Architecture

```
                         ┌─────────────────────────────┐
                         │   OFFLINE: INDEXING PATH     │
                         │  PDF → Parse → Chunk →       │
                         │  Embed → Vector Store        │
                         └──────────────┬───────────────┘
                                        │
   ┌───────────────┐        ┌──────────▼──────────┐        ┌──────────────┐
   │ Golden QA Set  │◄──────►│   ONLINE: QUERY PATH  │───────►│  Structured   │
   │ (JSON, curated)│        │  Query → Retrieve →  │        │  Logs (JSONL) │
   └───────┬────────┘        │  Rerank(opt) →       │        └──────┬───────┘
           │                 │  Prompt → Generate    │               │
           │                 └──────────┬───────────┘               │
           │                            │                            │
           │                 ┌──────────▼───────────┐                │
           └────────────────►│    EVAL HARNESS       │◄───────────────┘
                              │ Retrieval metrics     │
                              │ (Recall@k/MRR/nDCG)  │
                              │ Faithfulness (judge)  │
                              │ Answer relevancy      │
                              │ Latency/cost per stage│
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Results Store (SQLite│
                              │  or Parquet) + Runs   │
                              │  table (config_id)    │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Dashboard (Streamlit)│
                              │  Config A vs Config B │
                              │  side-by-side          │
                              └───────────────────────┘
```

Key architectural decision: **the eval harness is config-driven, not hardcoded.** Every run (retrieval settings, chunk size, top-k, reranker on/off, LLM model) is tagged with a `run_id` and `config` blob and written to the results store. This is what makes the bonus dashboard possible without extra plumbing later — you don't bolt it on, you get it for free.

---

## 3. Component-by-Component: Options Analysis

For each stage I'll give you the realistic options, the trade-off, and what I'd pick for a single-PDF, evaluation-focused assignment (not a 10,000-doc production system — don't over-engineer this part).

### 3.1 Document Ingestion / Parsing

| Option | Pros | Cons |
|---|---|---|
| `PyPDF2` / `pypdf` | Simple, zero deps, fast | Loses layout, bad on tables |
| `pdfplumber` | Good text + table extraction, per-page control | Slower |
| `unstructured` | Handles messy layout, auto-classifies elements | Heavy dependency, overkill for one clean PDF |
| `PyMuPDF (fitz)` | Fast, accurate text + coordinates, can extract images | Slightly more code |

**Pick:** `pdfplumber` for extraction with page-number metadata preserved. Baahubali_Movie.pdf is almost certainly prose/Wikipedia-style text (plot, cast, production, reception) — no complex tables — so you don't need `unstructured`'s overhead. **Critically: keep page number and a stable `chunk_id` on every chunk.** Your eval harness needs to know *which chunk* an answer should have come from, and you can't do that if you don't track provenance from ingestion onward.

### 3.2 Chunking Strategy

| Option | Pros | Cons | When |
|---|---|---|---|
| Fixed-size recursive split (workshop default) | Simple, predictable | Ignores semantic boundaries, can split mid-fact | Baseline config |
| Semantic chunking (embedding-based breakpoints) | Chunks align with topic shifts | Slower, needs embedding pass first | Comparison config |
| Sentence-window / small-to-big (small chunk for retrieval, larger window for generation) | Best of both — precise retrieval, rich context for answer | More moving parts | If you want a 3rd config |
| Section/heading-aware (split on document structure) | Good for the Wikipedia-style structure of a movie article (Plot / Cast / Production / Reception) | Needs the doc to have detectable headings | Strong fit for THIS dataset |

**Pick for Config A (baseline):** Recursive character splitting, chunk_size=500, overlap=50 (workshop default-ish).
**Pick for Config B (comparison):** Section-aware chunking using the PDF's headings (Plot, Cast, Soundtrack, Reception, Box Office, etc.) with chunk_size≈800, overlap=100.

This pairing is deliberate — it gives you a *meaningful* side-by-side story for the bonus dashboard: "does respecting document structure improve retrieval quality over naive fixed-size chunking?" That's a much better demo than "top_k=3 vs top_k=5," which is a shallower comparison.

### 3.3 Embedding Model

| Option | Dim | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` (workshop default) | 384 | Fast, decent quality, free, CPU-friendly |
| `bge-small-en-v1.5` / `bge-base-en-v1.5` | 384/768 | Noticeably better retrieval accuracy on benchmarks (MTEB), still free/local |
| `text-embedding-3-small` (OpenAI) | 1536 | Best quality, costs money, needs API key, adds network latency |
| `e5-small-v2` | 384 | Good middle ground, requires query/passage prefix convention |

**Pick:** `bge-small-en-v1.5` (local, free, better than MiniLM on retrieval benchmarks) as the single embedding model for both configs — keep embeddings constant so your chunking-strategy comparison isn't confounded by also changing the embedder. Vary **one variable at a time** between Config A and B; that's what makes the eval defensible.

### 3.4 Vector Database

| Option | Pros | Cons |
|---|---|---|
| Qdrant (in-memory, workshop default) | Zero setup, fast for small corpora | Not persistent across restarts unless you run the server |
| Qdrant (Docker, persistent) | Same API, durable, still simple | Needs Docker |
| ChromaDB | Even simpler local persistence, good DX | Slightly less mature filtering |
| FAISS | Fastest raw ANN search, battle-tested | No metadata filtering/payload store out of the box, more manual plumbing |
| pgvector | If you want everything in one relational DB (great for storing eval results too) | More setup |

**Pick:** Qdrant in local persistent mode (Docker or on-disk) — stick with the workshop's choice since it already gives you payload filtering and cosine search, and persistence means you don't re-embed every run of your eval harness (saves real time when iterating).

### 3.5 Retrieval Strategy

This is the stage most worth comparing for your bonus dashboard, alongside chunking.

| Strategy | What it is | Trade-off |
|---|---|---|
| Dense-only (vector similarity) | Workshop default | Misses exact keyword/name matches (e.g., "SS Rajamouli", "Kalakeya") sometimes |
| Hybrid (dense + BM25 keyword, fused) | Combines semantic + lexical | Better recall on proper nouns/entities — very relevant for a movie-facts dataset full of names | Slightly more infra (need a keyword index too, e.g. `rank_bm25`) |
| Dense + reranker (cross-encoder, e.g. `bge-reranker-base` or `ms-marco-MiniLM`) | Retrieve top-20 cheaply, rerank to top-5 precisely | Extra latency (one more model call) but meaningfully better precision |
| Dense + metadata filter (workshop's query-understanding phase) | Good when doc has strong categories | Less useful for a single narrative PDF like a movie article (no departments to filter by) |

**Pick for Config A:** Dense-only retrieval, top_k=5.
**Pick for Config B:** Hybrid (dense + BM25) with a cross-encoder reranker down to top_k=5.

This is the strongest possible bonus-dashboard story: **"naive dense retrieval vs. hybrid+rerank — does it improve Recall@5 and downstream faithfulness, and at what latency cost?"** That's a genuinely useful, real-world RAG question, and it maps directly onto your two required metrics (retrieval quality + faithfulness) plus a natural third axis (latency).

Drop the workshop's LLM-based query-understanding/filter-extraction phase for this project — it's designed for multi-department enterprise docs, not a single narrative PDF, and it adds latency + an extra LLM call without helping your metrics here.

### 3.6 Prompt Construction / Grounding

Keep the workshop's approach: numbered context blocks with source/page tags, an explicit instruction to answer *only* from context and say "not found in the document" otherwise, and ask the model to cite which context number(s) it used. That citation is what makes your faithfulness judge's job much more reliable (see §4.2) — you can check the citation against the actual chunk instead of only free-text judging.

### 3.7 Generation LLM

| Option | Notes |
|---|---|
| Groq `llama-3.3-70b-versatile` (workshop default) | Free tier, extremely low latency (Groq's LPU inference) — great for keeping your harness fast across 20+ questions × 2 configs |
| OpenAI `gpt-4o-mini` | Strong quality, cheap, but adds network latency + cost per eval run |
| Local (Ollama + Llama3/Qwen) | Zero cost, fully offline | Slower, weaker instruction-following on smaller models |

**Pick:** Keep Groq for the generator (fast, free, good enough for a factual single-document QA task) — but use a **separate, more careful model as the judge** (§4.2). Don't let the same model grade its own homework if you can avoid it.

### 3.8 Evaluation Harness — the core deliverable (full design in §4)

### 3.9 Logging / Observability (full design in §5)

### 3.10 Dashboard (full design in §6)

---

## 4. Evaluation Harness Design

### 4.1 Building the Golden QA Set (15–25 pairs)

Don't auto-generate these with an LLM and call it done — hand-verify every one, or an LLM-generated set will silently bake in the LLM's own biases (RAGAS's synthetic generation and LLM-as-judge would then share blind spots). Process:

1. Read the PDF yourself, section by section (Plot, Cast, Production, Music, Release, Reception, Box Office, etc.)
2. For each section, write 2–4 questions covering a mix of difficulty:
   - **Simple factual** ("Who directed Baahubali?") — single-chunk answer
   - **Detail/numeric** ("What was the film's reported budget?") — tests precision, numbers are easy to hallucinate
   - **Multi-hop / synthesis** ("How did the film's box office performance compare to its budget?") — needs 2+ chunks, stress-tests retrieval breadth
   - **Adversarial/unanswerable** ("What did the director say about the sequel's release date?" — if not in the doc) — tests whether the system correctly refuses instead of hallucinating. **Include at least 3–4 of these.** Faithfulness evaluation without unanswerable questions is testing the easy case only.
3. For each question, record:
   ```
   {
     "id": "q07",
     "question": "...",
     "ground_truth_answer": "...",
     "ground_truth_chunk_ids": ["chunk_014", "chunk_015"],   // which chunks SHOULD be retrieved
     "category": "multi-hop" | "factual" | "numeric" | "unanswerable",
     "difficulty": "easy" | "medium" | "hard"
   }
   ```
   The `ground_truth_chunk_ids` field is what makes retrieval metrics possible without another LLM judge — it's deterministic ground truth you write once by hand.

Store this as `eval/golden_qa.json` — version it, don't regenerate it per run.

### 4.2 Retrieval Quality Metrics (deterministic — no LLM needed)

For each question, run retrieval, get the top-k chunk_ids, compare against `ground_truth_chunk_ids`:

| Metric | Formula intuition | Why it matters |
|---|---|---|
| **Hit Rate @k** | Was *any* correct chunk in top-k? | Binary sanity check |
| **Recall @k** | What fraction of ground-truth chunks were retrieved? | Catches multi-hop failures |
| **Precision @k** | What fraction of retrieved chunks were actually relevant? | Catches noisy retrieval that would confuse the generator |
| **MRR (Mean Reciprocal Rank)** | How high up was the first correct chunk? | Rewards ranking quality, not just presence |
| **nDCG@k** | Rank-weighted relevance | Most rigorous, use if you have time |

These are cheap, fast, and 100% reproducible — run them first, before touching an LLM judge, since they're the objective backbone of your "retrieval quality" requirement.

### 4.3 Faithfulness (Groundedness) — LLM-as-Judge

Two complementary approaches, use both if time permits (they catch different failure modes):

**A. Claim-decomposition judge (most rigorous, RAGAS-style)**
1. Ask a judge LLM to break the generated answer into atomic factual claims.
2. For each claim, ask the judge: "Is this claim directly supported by the provided context? yes/no/partial."
3. Faithfulness score = (# supported claims) / (# total claims).

**B. Direct grounding judge (simpler, faster, good enough as a baseline)**
Single prompt: *"Given this CONTEXT and this ANSWER, rate on a 1–5 scale how well the answer is grounded in the context alone (5 = fully grounded, no unsupported claims; 1 = mostly fabricated). Also flag ANY specific claim not found in context."* Log both the score and the flagged claims.

**Judge model choice:** use a *different* model from the generator (e.g., generator = Groq Llama-3.3-70b, judge = Gemini 1.5 Flash or GPT-4o-mini) — this avoids self-preference bias where a model rates its own phrasing more favorably. Log every judge call (prompt + raw response) so you can manually audit disagreements.

**Sanity-check the judge itself:** hand-score 5 answers yourself first, compare to the judge's scores, and only trust the judge harness once they roughly agree. This is a step almost everyone skips and it's exactly the kind of rigor that separates a strong submission.

### 4.4 Answer Relevancy (secondary, cheap to add)

Simple LLM-judge or even embedding-similarity check: does the answer actually address what was asked (independent of whether it's grounded)? A model can be perfectly faithful to context while dodging the actual question — this metric catches that.

### 4.5 Latency & Cost Instrumentation

Every run logs, per question:
- `retrieval_latency_ms`, `rerank_latency_ms` (if used), `generation_latency_ms`, `judge_latency_ms`, `total_latency_ms`
- `prompt_tokens`, `completion_tokens`, `estimated_cost_usd` (0 for Groq free tier / local, but log token counts regardless — it's a proxy for prompt bloat)

Aggregate at the end of a harness run: p50/p95 latency, total run time, cost.

### 4.6 The Harness Itself (workflow, not code)

```
run_eval(config: RunConfig) -> RunResult:
    1. Load golden_qa.json
    2. For each question:
        a. retrieve(question, config)              → chunks + retrieval latency
        b. compute retrieval metrics vs ground truth → hit@k, recall@k, MRR
        c. generate(question, chunks, config)        → answer + generation latency
        d. judge_faithfulness(answer, chunks)         → score + flagged claims
        e. judge_relevancy(answer, question)          → score
        f. log full record (JSONL) — see §5
    3. Aggregate all per-question metrics → RunResult summary
    4. Write RunResult to results store, tagged with config_id + timestamp
    return RunResult
```

Run this once per configuration (Config A, Config B, ...). Each call is fully reproducible given the same config — that reproducibility is what the dashboard depends on.

---

## 5. Logging Strategy

Two tiers — don't conflate them:

**Tier 1 — Structured per-request JSONL log** (`logs/run_<config_id>_<timestamp>.jsonl`), one line per question:
```json
{"run_id": "...", "config_id": "hybrid_rerank_v1", "question_id": "q07",
 "retrieved_chunk_ids": [...], "retrieval_scores": [...], "retrieval_latency_ms": 42,
 "ground_truth_chunk_ids": [...], "hit_at_5": true, "recall_at_5": 0.5, "mrr": 1.0,
 "generated_answer": "...", "generation_latency_ms": 810,
 "faithfulness_score": 0.9, "flagged_claims": [], "relevancy_score": 5,
 "prompt_tokens": 1240, "completion_tokens": 181, "timestamp": "..."}
```

**Tier 2 — Trace-level observability** (optional but strong signal to the evaluator that you understand production RAG): Langfuse (workshop already wires this in) or Arize Phoenix / OpenTelemetry, capturing the full span tree (retrieval span → rerank span → generation span → judge span) per request, viewable in a trace UI. This is what "observability" means beyond flat logs — nested spans, not just log lines.

**Results store:** SQLite (simplest, queryable with SQL for the dashboard, single file, zero server) with two tables: `runs` (config_id, config_json, timestamp, aggregate metrics) and `run_details` (per-question rows, foreign key to run_id). This is what your Streamlit dashboard queries — don't have the dashboard re-run the pipeline live, have it read from this store.

---

## 6. Bonus Dashboard: Config A vs Config B

Streamlit page reading from the SQLite results store, showing:

1. **Headline metric cards** side by side: avg Faithfulness, avg Recall@5, avg MRR, p95 latency, for Config A vs Config B.
2. **Per-category breakdown** (bar chart): faithfulness/recall split by question category (factual/multi-hop/unanswerable) — this is where the "why" shows up, e.g. hybrid+rerank likely wins big on multi-hop and unanswerable-refusal, not on simple factual lookups.
3. **Per-question drill-down table**: question | Config A answer | Config B answer | Config A faithfulness | Config B faithfulness | winner — lets a reviewer spot-check specific disagreements.
4. **Latency vs quality scatter**: makes the "is the improvement worth the latency cost" trade-off visually obvious — the single most compelling chart you can put in front of an evaluator, since it shows you're thinking about engineering trade-offs, not just chasing a metric.
5. **Failure gallery**: filter to show only cases where faithfulness < 3 or hit@5 = false, for both configs — demonstrates you actually looked at the failures, not just the aggregate score.

---

## 7. Final Recommended Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Matches workshop scaffold |
| PDF parsing | `pdfplumber` | Clean text + page metadata, right-sized for one document |
| Chunking | `langchain` `RecursiveCharacterTextSplitter` (Config A) + custom heading-aware splitter (Config B) | Gives a genuine, non-trivial comparison axis |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Free, local, better MTEB retrieval scores than MiniLM |
| Vector DB | Qdrant (local persistent) | Payload filtering, cosine search, reused from workshop |
| Keyword index (Config B only) | `rank_bm25` | Lightweight hybrid retrieval, no extra server |
| Reranker (Config B only) | `BAAI/bge-reranker-base` (cross-encoder, local) | Free, meaningfully improves precision@k |
| Generator LLM | Groq `llama-3.3-70b-versatile` | Fast + free, keeps harness runs quick across 20+ Qs × 2 configs |
| Judge LLM | Gemini 1.5 Flash *or* GPT-4o-mini (different from generator) | Avoids self-preference bias, cheap |
| Eval framework | Custom harness (deterministic retrieval metrics) + RAGAS (faithfulness/relevancy as a cross-check) | Deterministic metrics you fully control + an established library to validate against |
| Logging | Python `logging` → JSONL files | Simple, greppable, git-diffable |
| Trace observability | Langfuse (cloud free tier) | Nested spans, already wired into workshop scaffold |
| Results store | SQLite | Queryable, single file, perfect for a small dashboard |
| Dashboard/UI | Streamlit | Matches scaffold, fastest to build a comparison view |
| Config management | Pydantic Settings (workshop default) + one YAML/JSON per named config (`config_a.yaml`, `config_b.yaml`) | Makes "run eval for config X" a one-line CLI call |

---

## 8. Suggested Enhanced Project Structure

Builds on the workshop skeleton, adding what the assignment actually grades on:

```
build-with-rag-workshop/
├── ingestion/            # (existing) + pdfplumber loader w/ page tracking
├── chunking/             # (existing) + heading_aware_splitter.py  [NEW]
├── embeddings/           # swap to bge-small-en-v1.5
├── vectordb/             # (existing) Qdrant
├── retrieval/
│   ├── dense_retriever.py        # Config A
│   ├── hybrid_retriever.py       # Config B: BM25 + dense fusion  [NEW]
│   └── reranker.py               # cross-encoder rerank            [NEW]
├── generation/           # (existing) Groq client
├── eval/                                                            [NEW — the core deliverable]
│   ├── golden_qa.json             # hand-curated 15-25 QA pairs
│   ├── retrieval_metrics.py       # hit@k, recall@k, MRR, nDCG
│   ├── faithfulness_judge.py      # claim-decomposition + direct grounding judge
│   ├── relevancy_judge.py
│   ├── harness.py                 # orchestrates a full eval run for a given config
│   └── configs/
│       ├── config_a_dense.yaml
│       └── config_b_hybrid_rerank.yaml
├── rag_logging/          # (existing) + JSONL per-run logger
├── results/
│   └── eval_results.db            # SQLite: runs + run_details tables
├── observability/        # (existing) Langfuse spans
├── ui/
│   ├── streamlit_app.py           # (existing) chat UI
│   └── eval_dashboard.py          # [NEW] Config A vs B comparison dashboard
└── data/
    └── Baahubali_Movie.pdf
```

---

## 9. Build Order (roadmap, not code)

1. Ingest + chunk (both strategies) → confirm chunk counts/quality by eyeballing 5–10 chunks
2. Embed + index both chunk sets into two Qdrant collections (`baahubali_dense_v1`, `baahubali_hybrid_v1`)
3. Write the golden QA set by hand (this takes longer than people expect — budget real time here, it's the foundation everything else is scored against)
4. Build retrieval metrics module, sanity-check on 3 questions manually before trusting it
5. Wire up generation + logging
6. Build the faithfulness/relevancy judge, hand-validate against your own scoring on 5 answers
7. Build the harness orchestrator, run Config A end-to-end, inspect results
8. Run Config B, compare
9. Build the SQLite results store + dashboard
10. Write up a short README section: what you compared, what won, why, and what the latency/quality trade-off looks like — this narrative is what turns "I ran two configs" into "I understand RAG evaluation."

---

## 10. Common Pitfalls to Avoid

- **Letting the generator judge its own answers** — always use a different model as judge.
- **Auto-generating the golden QA set entirely with an LLM** — you'll bake in blind spots; hand-review every pair.
- **No unanswerable questions in the test set** — faithfulness testing is meaningless if the system is never tempted to hallucinate.
- **Comparing configs that differ in more than one variable** — if Config B changes chunking *and* retrieval *and* the embedding model, you can't attribute the metric delta to anything specific.
- **Dashboard re-running the pipeline live** — read from a persisted results store; keep the dashboard and the harness decoupled.
- **Skipping latency/cost logging** — "accuracy went up" without "and latency went up 4x" is an incomplete engineering story, and evaluators notice that omission.
