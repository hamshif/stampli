# Stampli Agentic Analyst Architecture (v4)

## 1. Executive Summary
The Stampli Agentic Analyst is a production-grade RAG system designed to extract insights from 42,000+ Disney Park reviews. 

**Core Philosophy**: A "Logic-First, LLM-Second" hybrid approach. We use high-recall Keyword Heuristics to backfill structured data where pure LLM extraction initially struggled, while leveraging LLMs for nuanced semantic retrieval and synthesis.

**Current State**: The system utilizes a modular 3-stage Ingestion Pipeline and a context-aware RAG server, synchronized via a centralized versioning system.

---

## 2. The Ingestion & Enrichment Pipeline
The system moves away from monolithic scripts toward a modular Directed Acyclic Graph (DAG) controlled by a master orchestrator.

### A. Pipeline Orchestration
- **Controller**: [disney_ingestion_pipeline.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/enrichment/disney_ingestion_pipeline.py)
- **Versioning**: Centralized in [paths.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/paths.py). Updating `CURRENT_VERSION` automatically synchronizes all pipeline nodes and the Chat Server.

### B. Three-Stage Modular DAG
1. **Stage 1: Ingestion** ([ingest_raw.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/enrichment/ingest_raw.py))
   - Converts raw `DisneylandReviews.csv` to a canonical Parquet base.
   - Standardizes IDs (`review_uid`) and handles encoding (`ISO-8859-1`).

2. **Stage 2: LLM Enrichment** ([enrich_data.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/enrichment/enrich_data.py))
   - Uses `gpt-4o-mini` to extract structured sentiment, topics, and specific domain fields (Crowds, Staff, Price).
   - **Feature**: Robust Resume/Checkpointing. It identifies "gaps" in the data and only processes rows missing enrichment.

3. **Stage 3: Keyword Backfill** ([enrich_by_keywords.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/enrichment/enrich_by_keywords.py))
   - Acts as a "Safety Net". Uses high-precision Regex patterns to fill missing labels.
   - **Why**: Initial LLM runs had a high false-negative rate for obvious crowd signals. This stage recovered 11,000+ labels, improving metadata-filter recall by 10x.

---

## 3. RAG Execution Engine (`stampli_chat.py`)

### A. Router (Linguistic Intent)
- **Intent Extraction**: Converts natural language into a structured `FilterSchema`.
- **Context Management**: Prevents "Sticky Context". It resets filters during topic shifts (e.g., switching parks) but maintains history for follow-ups ("Why?").
- **Entity Normalization**: Maps loose user entities ("Anaheim", "Eurodisney") to canonical database values.

### B. Retrieval Engine (Hybrid Search)
1. **Metadata Filtering**: Pandas-based boolean masking for high-signal fields (Branch, Date, Crowd Level).
2. **Semantic Re-ranking**: Uses `OpenAIEmbeddings` (text-embedding-3-small) with Cosine Similarity.
- **Why**: Pure metadata filtering is brittle; semantic search ensures that a query about "busy queues" finds reviews even if the specific `Packed` checkbox was missed.

### C. Synthesizer (Evidence-Based Synthesis)
- **Citation-Strict**: Generates answers based *only* on the retrieved Top-K reviews.
- **Formatting**: Returns structured distribution stats and individual bulleted summaries for transparency.

---

## 4. Quality Assurance & Exploration
- **Automated Verification**: [test_assignment_qa.py](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/tests/test_assignment_qa.py) runs 9 complex scenarios including guardrail checks and context-recall tests.
- **Local Exploration**: [local_disney_exploration.ipynb](file:///home/gideon/dev/biocontext/biocontext-services/src/biocontext/apps/stampli/local_disney_exploration.ipynb) provides a lightweight, Spark-free environment for inspecting enrichment distribution.

---

## 5. Deprecated Architectures (Lessons Learned)

| Approach | Status | Reason for Deprecation |
| :--- | :--- | :--- |
| **Pure LLM Enrichment (v1)** | **Burned** | High False Negative Rate (~96%). Obvious signals were ignored by the model. |
| **Spark Ingestion** | **Frozen** | Overkill for 42k rows. Introduced unnecessary dependency/setup complexity. |
| **Naive Retrieval** | **Replaced** | `df.head()` is statistical noise. Semantic re-ranking was required for relevance. |

---

## 6. Mitigation Roadmap
1. **LLM Validation (v5)**: Use the LLM as a *validator* (True/False) on keyword-identified rows to eliminate Regex false positives.
2. **Fine-Tuning**: Train a small BERT-style classifier on the "High Confidence" dataset (Consensus between Keywords and LLM).
3. **Sarcasm Detection**: Enhance Stage 3 regex to handle negative lookahead (e.g., "NOT busy").
