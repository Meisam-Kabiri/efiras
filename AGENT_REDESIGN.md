# EFIRAS Agent Redesign

Replacing the single-shot `embed → hybrid_search → generate` RAG path with a
staged, LLM-routed pipeline. This doc is built up stage by stage; each stage is
confirmed before the next is designed.

## Why (failure modes of the current raw RAG)

- No query understanding — raw question embedded as-is, one retrieval pass for every question type.
- Cross-encoder reranker exists (`hybrid_search_with_cross_encoder`) but is never called; active path is plain `hybrid_search`. Too slow for CPU-only Cloud Run anyway.
- Document names are a soft ×1.5 boost (`calculate_doc_boost`), not a filter → cross-document contamination.
- One-shot retrieval can't handle comparative / multi-hop / survey questions.
- No grounding check → hallucinated citations.
- No "I don't know" path; fixed, tiny top_k (~10 chunks ≈ 4k tokens, leaving ~98% of a long-context budget unused).

## Target architecture (high level)

```
1. ROUTE   — cheap LLM: expand query + decide scope (strict/narrowed/broad)
             + resolve documents + decompose into sub-queries if needed
2. RETRIEVE— hybrid search WITH hard doc filter, generous top-k (~40-50),
             NO cross-encoder reranker
3. ANSWER  — strong long-context model: grounded, cited answer; explicit
             "not found in these documents" when retrieval is empty
(4. VERIFY — optional, for broad queries, to catch unsupported citations)
```

Key levers chosen over the reranker: **hard document scoping** (precision) +
**generous context to a strong model** (recall). This drops the slow CPU
reranker entirely.

---

## STAGE 1 — Routing layer + removing soft doc-weighting ✅ CONFIRMED

### 1a. New first LLM layer (cheap model, e.g. gpt-4o-mini / Haiku)

Takes the raw user question, returns a small structured decision **before** any
retrieval:

```json
{
  "kind": "question | greeting | gibberish | out_of_scope",
  "scope": "strict | narrowed | broad",
  "documents": ["GDPR"],            // resolved names; [] when broad
  "expanded_query": "GDPR data subject access request response deadline ...",
  "sub_queries": ["..."]            // only for comparative / multi-hop
}
```

Scope semantics:
- **strict**  — question explicitly bound to one regulation ("based on GDPR…") → search ONLY that document.
- **narrowed** — clearly a small group ("AML obligations" → 4AMLD + 5AMLD + FATF) → search only those.
- **broad**   — genuinely vague → search everything.

`kind != question` (greeting / gibberish / out_of_scope) short-circuits: respond
directly, skip retrieval entirely.

### 1b. Changes to `search_service.py`

1. **Delete** `calculate_doc_boost()` and its use inside `rrf_combine()`.
   RRF returns to plain reciprocal-rank fusion (no document boosting).
2. **Fix** the RRF constant: `k=0` → `k=60` (standard; previously masked by the boost).
3. **Add** an optional `doc_filter` (list of filenames) through the search path
   (`search_documents` / `hybrid_search` / `bm25_search` / `vector_search`):
   - BM25: add `WHERE filename IN (...)` to the FTS query.
   - FAISS: **DECIDED — `IDSelectorBatch` + `nprobe=nlist`.** Index is
     `IndexIVFFlat` (1024-dim, 57,456 vecs, nlist=100, nprobe=20, L2 over
     normalized vectors ≡ cosine). FAISS position `i` == `chunks.id` `i`, so the
     `filename → [ids]` map is a one-line SQL query, precomputed at load.
     **Verified gotcha:** a selector with the default nprobe=20 returns ~0 for a
     single doc (its vectors are scattered across all 100 clusters, most skipped).
     Fix: when a doc filter is active, search with
     `faiss.SearchParametersIVF(sel=IDSelectorBatch(ids), nprobe=idx.nlist)` →
     scans all clusters, exact recall (tested: SFDR 60/60 in-filter). At 57k vecs
     a full scan is ~milliseconds, so no latency cost. Unfiltered (broad) keeps nprobe=20.
4. **Repurpose** the `extract_doc_keywords()` dictionary as the
   **name → filename resolver**: router outputs `"GDPR"`, resolver maps it to
   `General_Data_Protection_Regulation_(GDPR).pdf`. Same data, new job — do NOT delete it.

### Open item carried to later stages
- FAISS filtering approach (IDSelector vs over-fetch) — to confirm.

---

## STAGE 2 — Retrieval depth + reflective loop ✅ CONFIRMED

### 2a. Chunk count, scaled by scope

No cross-encoder reranker (too slow on CPU-only Cloud Run). Instead: scope hard,
then feed a generous top-k to a strong long-context model.

| scope    | docs searched | chunks returned | rationale |
|----------|---------------|-----------------|-----------|
| strict   | 1             | 50–60           | one regulation, low noise — be generous |
| narrowed | 2–4           | ~50             | balanced across the named docs |
| broad    | 23            | ~40             | more distractor risk as the pool widens |

50 chunks ≈ 20k tokens — trivial for a 200K/1M-context model. Tunable constants.

### 2b. Up-front decomposition (layer 1) + reflective loop (layer 2)

Two complementary mechanisms:

1. **Up-front splitting** — when layer 1 already sees a multi-part question
   (e.g. "compare Basel II vs III capital requirements"), it emits multiple
   `sub_queries`, each doc-scoped, and they are searched **in parallel**. Covers
   comparative + predictable multi-hop cheaply, before any loop.

2. **Reflective loop** — for the cases you can't predict up front (true
   multi-hop: "find Article X, then find the penalty for X"):

```
LLM layer 1: expand query + scope + doc filter (+ split if multi-part)
      ↓
SEARCH (hybrid bm25 + vector, doc-filtered) — all sub-queries in parallel
      ↓
LLM layer 2: "are these chunks enough to answer?"
   ├─ enough → generate final answer → DONE
   └─ not    → keep the good chunks, generate a NEW search query → loop to SEARCH
```

**Loop cap: max 2–3 iterations** (never open-ended). Strict single-doc lookups
skip the loop.

**Open sub-decision (loop style):**
- **(A)** layer 2 judges only whether the *chunks* are sufficient; the answer is
  generated **once** at the end → cheaper, lower latency. *(recommended default)*
- **(B)** generate a draft answer each round, critique the *answer*, search again
  if weak → better at catching subtle gaps, but ~2–3× the cost on hard questions.

### 2c. Streaming UX

The SSE channel already supports typed events. During the loop, emit
`{"type": "status", "content": "searching GDPR…"}` so the user sees progress
instead of a frozen screen while layers 1–2 run before the answer streams.

---

## STAGE 3 — Models per layer 🔲 UNDER DISCUSSION

Current code is OpenAI-only (`rag_generator.py` → `openai` client, gpt-4o-mini).
Switching the answer layer to Claude means adding the Anthropic SDK alongside it.

Reference (Claude, current pricing per 1M tokens):

| Model            | Context | Input | Output |
|------------------|---------|-------|--------|
| Claude Haiku 4.5 | 200K    | $1    | $5     |
| Claude Sonnet 4.6| 1M      | $3    | $15    |
| Claude Opus 4.8  | 1M      | $5    | $25    |

(gpt-4o-mini, the current model, is ~$0.15 / $0.60 — far cheaper, weaker.)

Proposed split (to confirm):

| Layer | Job | Proposed model | Why |
|-------|-----|----------------|-----|
| 1 — router / expand / scope | cheap structured-JSON classification | gpt-4o-mini **or** Haiku 4.5 | cheap, fast; quality bar is low |
| 2 — "enough?" sufficiency check | cheap judgment | gpt-4o-mini **or** Haiku 4.5 | runs each loop iteration |
| 3 — answer generation | grounded, cited, long-context reasoning | **Claude Sonnet 4.6** (or Opus 4.8 for max quality) | citation discipline + long-context fidelity is where quality is won |

If Claude is used for the answer, **prompt-cache** the big system prompt + scoped
context to cut cost/latency on the streamed generation.

---

## NOT YET DECIDED / before this is production-ready

- Loop style A vs B (Stage 2b).
- Provider + tier for each layer (Stage 3) + adding the Anthropic SDK.
- Per-layer prompt engineering (router schema prompt, sufficiency-check prompt, answer prompt).
- Evaluation: a small old-vs-new question set (lookup / comparative / multi-hop / out-of-scope) — otherwise "better" stays a vibe.
- Latency budget: each LLM stage adds time-to-first-token; measure on real queries.
