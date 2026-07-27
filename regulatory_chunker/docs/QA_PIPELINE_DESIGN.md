# EFIRAS Q&A Pipeline Design

Design for the query→answer stage: a client asks a regulatory question, the platform returns an accurate, citation-backed answer.

**Foundation (already built):** 23 documents, 13,500 chunks, paragraph-level granularity. Every chunk carries a canonical legal citation string (`GDPR Art. 1(1)`), a hierarchical path (chapter → article → paragraph), and a stable chunk_id (`GDPR/art_30/para_1`). This is exactly what a high-accuracy citation pipeline needs.

## The pipeline

```
Question
   │
   ├─► 0. Citation router (regex, free) ──► direct chunk lookup ─┐
   │                                                             │
   ├─► 1. Query analysis (Haiku, ~$0.001) ──► doc filter,        │
   │        sub-queries for multi-part questions                 │
   │                                                             │
   ├─► 2. Hybrid retrieval: BM25 + embeddings, RRF-fused → top 50│
   ├─► 3. Rerank → top ~12                                       │
   ├─► 4. Expand to parent article + follow cross-references ────┤
   │                                                             ▼
   └─► 5. Claude Opus 4.8 + Citations API ──► answer with grounded citations
```

### 0. Citation-aware router (free accuracy — build first)

A large share of compliance questions name their target: "What does Article 30 GDPR require?", "Explain DORA Art. 5(2)". Parse doc aliases + article/paragraph references with regex, look up the chunk_id directly (IDs are already `GDPR/art_30/para_1`), and feed the whole article to the answer step. No retrieval error is possible on this class. When a citation is parsed, retrieval can still run in parallel for supporting context.

### 1. Query analysis

One cheap Haiku call returning structured output:
- which document(s) the question concerns (or "all")
- a cleaned rewrite of the query
- for multi-part questions ("compare record-keeping duties under GDPR and DORA"): 2–4 sub-queries, each retrieved independently

This is where hard questions get tamed.

### 2. Hybrid retrieval

Legal text is full of exact terms of art where lexical match beats semantic ("politically exposed person", "ancillary services undertaking"), and full of paraphrases where embeddings beat lexical. Run both, fuse with reciprocal rank fusion (RRF):

- **BM25**: SQLite FTS5 — zero infrastructure at this scale.
- **Dense**: embed each chunk as `citation + article heading + chapter heading + text`. The `path` field gives this contextualization for free — a bare paragraph like "1. This Regulation lays down rules…" embeds terribly without it. Embedding model: Voyage `voyage-law-2` (legal-domain-tuned); embedding all 13,500 chunks costs well under $1. Store vectors in LanceDB or a NumPy file — 13.5k × 1024 floats fits in memory; no vector database service needed.

### 3. Rerank

Cross-encoder rerank of the top ~50 fused hits down to ~12 (Voyage `rerank-2.5` or Cohere Rerank; fractions of a cent per query). The single highest accuracy-per-dollar step in RAG — this is what saves subtly-worded questions.

### 4. Small-to-big expansion + cross-references

Retrieve at paragraph level (precise), answer at article level (complete): group hits by article and pass the full article, keeping per-paragraph citations.

Then one hop of cross-reference following — regulations constantly say "the processing referred to in Article 6(1)". Extract those references at index time (`graph_structure.py` is the natural home) and pull referenced articles into context. This fixes the classic legal-RAG failure where the retrieved paragraph is right but incomplete without its referenced definition.

### 5. Answer with the Citations API

Pass each article as a `document` content block with `citations: {enabled: true}` and its `title` set to the citation string. The API returns the answer as text blocks with attached `cited_text` + `document_index` — citations are structurally grounded by the API rather than generated as free text, so the model *cannot* cite a passage that wasn't in context. Map `document_index` back to citation strings for display.

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=4096,
    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": [
        *[{
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": art["text"]},
            "title": art["citation"],          # e.g. "GDPR Art. 30"
            "citations": {"enabled": True},
        } for art in retrieved_articles],
        {"type": "text", "text": question},
    ]}],
)
```

System prompt requirements:
- answer only from the provided provisions; quote the operative language
- state explicitly when the provided context doesn't cover the question — never fill gaps from general knowledge (the killer failure mode in compliance)
- flag Level 1 vs Level 2 instruments (AIFMD/MiFID L2 docs exist in the corpus) — name which instrument answers what

## Two data fixes before building

1. **Oversized chunks.** Outliers up to 74KB (`5amld`), 44KB (`crr`), 22KB (`cssf_18_698`) — almost certainly annexes. They dominate context and retrieve poorly. Either split them at internal structure for indexing (keeping the parent citation), or embed them in overlapping windows that all map back to the one chunk.

2. **Amendment metadata.** 4AMLD/5AMLD (and AMLR superseding both from 2027) is a correctness trap: a client asking "what are the CDD thresholds?" must get the in-force consolidated position, not the 2015 text. Add doc-level metadata (in-force status, amended-by, applies-from) that query analysis uses to pick the right instrument and the answer step uses to caveat. `amendment_checker.py` is the natural home.

## Tier 2: agentic loop for genuinely hard questions

For questions the single-pass pipeline can't nail — cross-document comparisons, "which obligations apply to a Luxembourg AIFM doing X", multi-hop chains — run an agentic loop instead: give Claude three tools via the SDK's tool runner and let it search iteratively, follow references itself, then produce the cited answer.

Tools:
- `search_chunks(query, doc_filter)`
- `get_article(doc, article)`
- `get_toc(doc)`

Route to this tier when query analysis flags the question as multi-document/multi-hop, or as a retry when the tier-1 answer reports insufficient context. Costs 3–8× a single pass — which is why it's a tier, not the default.

## Cost picture (per question, tier 1)

| Step | Cost |
|---|---|
| Haiku query analysis | ≈ $0.001 |
| Query embedding + rerank | ≈ $0.002 |
| Opus 4.8 answer (~10 articles in context, system prompt cached) | ≈ $0.05–0.15 |

If per-query cost matters at volume, the answer model is the one meaningful lever — Sonnet 5 ($3/$15 vs $5/$25 per MTok) is credible for regulatory Q&A — but for a product whose value proposition is accuracy, launch on Opus 4.8 and only consider stepping down with eval data in hand.

## Evaluation (the part most people skip)

Before tuning anything, write ~75 realistic client questions with expected citation(s). Include hard ones: paraphrased terms, cross-references, multi-document, and questions whose true answer is "the regulation doesn't address this".

Measure two numbers:
- **Retrieval recall@12** — is the right article in what was passed to the model?
- **Citation accuracy** — did the answer cite the right provisions?

Every design choice (k values, reranker on/off, expansion policy) gets judged against this set, not vibes. In practice ~90% of wrong answers in legal RAG are retrieval failures, so this shows exactly where to invest.

## Build order

1. Citation router + BM25 + embeddings + RRF (a day's work at this scale, all local except one embedding batch)
2. Answer step with Citations API + cached system prompt
3. Eval set → measure → add reranker and article expansion → measure again
4. Cross-reference hop, amendment metadata, then the agentic tier 2
