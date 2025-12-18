Below is the **final, implementation-ready specification**.
It is written as a **neutral system spec** suitable for Codex CLI + implementation work.
No narrative, no audience addressing.

---

# Disney Reviews – Agentic Analyst Application Spec

---

## 1. System Objective

Provide a natural-language query system over Disney park reviews that produces:

* Statistically correct answers for aggregation and trends
* Qualitative explanations grounded in review text
* Actionable customer-experience recommendations

All numeric outputs must be derived from structured data.
All explanatory evidence must be derived from review text.

---

## 2. Data Model

### 2.1 Raw Inputs

* `review_id: string`
* `review_text: string`
* `review_date: date`
* `rating: float`
* `park_name: string`
* `park_country: string`
* `reviewer_country: string`

---

### 2.2 Derived Fields (Offline)

* `month: int`
* `year: int`
* `season: enum {winter, spring, summer, fall}`
* `sentiment_score: float` (per-review)
* `topic_tags: list[string]` (optional)

---

## 3. Storage Layer

### 3.1 Structured Store

* Pandas / DuckDB / SQLite
* Contains all raw + derived fields
* Indexed on:

  * `park_name`
  * `month`
  * `season`
  * `reviewer_country`

### 3.2 Semantic Store

* Vector database (FAISS / Chroma)
* Embeddings: `text-embedding-3-small`
* Stored with metadata filters:

  * `park_name`
  * `month`
  * `season`
  * `reviewer_country`
  * `sentiment_score`

---

## 4. Runtime Components

---

## 4.1 Orchestrator (LLM Router)

**Model:** `gpt-4o-mini`

### Input

* User natural-language question

### Output (Strict JSON Schema)

```json
{
  "quant_tasks": [
    {
      "filters": {
        "park_name": "string | null",
        "season": "string | null",
        "month": "int | null",
        "reviewer_country": "string | null"
      },
      "metrics": ["review_count", "avg_rating", "avg_sentiment"]
    }
  ],
  "qual_tasks": [
    {
      "filters": {
        "park_name": "string | null",
        "season": "string | null",
        "month": "int | null",
        "reviewer_country": "string | null"
      },
      "semantic_intent": "string"
    }
  ]
}
```

### Responsibilities

* Intent classification
* Filter extraction
* Tool routing
* Execution planning

---

## 4.2 Quant Engine (Structured Analytics)

### Input

* `quant_tasks` from orchestrator

### Execution

* Deterministic Pandas / SQL operations:

  * `COUNT(*)`
  * `AVG(rating)`
  * `AVG(sentiment_score)`
  * Period-over-period comparisons

### Output

```json
{
  "metrics": {
    "review_count": 1245,
    "avg_rating": 4.3,
    "avg_sentiment": 0.61
  },
  "baseline": {
    "review_count": 890,
    "avg_rating": 4.0
  }
}
```

### Constraints

* No LLM usage
* Fully reproducible
* Source-of-truth for all numbers

---

## 4.3 Qual Engine (Semantic Retrieval)

### Input

* `qual_tasks` from orchestrator

### Execution

* Metadata-filtered vector search
* Top-K retrieval (configurable)
* Optional topic constraint

### Output

```json
{
  "snippets": [
    {
      "review_id": "r123",
      "text": "Lines were extremely long due to summer holidays."
    }
  ]
}
```

---

## 4.4 Synthesizer (Answer Generator)

**Model:** `gpt-4o-mini`

### Input

* Quant Engine output
* Qual Engine snippets

### Responsibilities

* Combine statistics with qualitative evidence
* Generate:

  * Answer
  * Supporting facts
  * Actionable recommendation

### Output

```json
{
  "answer": "June is significantly more crowded than average.",
  "evidence": {
    "stats": "Review volume is 40% higher than baseline.",
    "quotes": ["Lines were extremely long due to summer holidays."]
  },
  "recommendation": "Increase staffing and promote early-day reservations."
}
```

### Constraints

* No new statistics may be invented
* All numeric references must trace to Quant Engine output

---

## 5. Application Flow

1. User submits question
2. Orchestrator generates execution plan
3. Quant and Qual engines execute in parallel
4. Synthesizer produces final response
5. Response returned via API / UI

---

## 6. Evaluation Strategy

### 6.1 Router Accuracy

* Golden query set
* Filter extraction correctness

### 6.2 Aggregation Accuracy

* Deterministic metric verification against raw data

### 6.3 Insight Quality

* LLM-as-Judge scoring on:

  * Helpfulness
  * Actionability
  * Evidence grounding

---

## 7. Non-Goals

* No fine-tuning
* No text-only aggregation
* No hallucinated statistics
* No opaque metric generation

---

## 8. Minimal Tech Stack

* Python
* Pandas / DuckDB
* OpenAI API (`gpt-4o-mini`, embeddings)
* FAISS / Chroma
* FastAPI or Streamlit

---

## 9. Design Invariant

**Metrics come from structured data.
Explanations come from text.
The LLM decides when to use each.**
