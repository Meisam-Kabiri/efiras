"""Agent pipeline: route → retrieve → (sufficiency loop) → answer."""

from typing import Any, Dict, List, Optional, Tuple

from core.rag import references as _references
from core.rag import router as _router

# Top-k by scope — tunable constants.
_TOP_K = {"strict": 55, "narrowed": 50, "broad": 40}
# Minimum results per document when several documents are searched, so a
# small document (Basel III: ~100 chunks) is not crowded out by a large one
# (CRR: ~2,900 chunks).
_MIN_PER_DOC = 10
# Minimum results per sub-query when a question is split into parts.
_MIN_PER_SUB_QUERY = 20

# Short-circuit replies for non-question kinds (avoid an extra LLM call).
_NON_QUESTION_REPLIES: Dict[str, str] = {
    "greeting":    (
        "Hello! I am EFIRAS, a financial regulatory assistant. "
        "Ask me about GDPR, SFDR, Basel III, or any of the 23 "
        "EU/US regulations in my knowledge base."
    ),
    "gibberish":   "Please ask a complete regulatory question.",
    "out_of_scope": (
        "That question is outside my scope. "
        "I specialise in EU and US financial regulations."
    ),
}


def run(
    question: str,
    embedding_service: Any,
    search_service: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    """
    Run the routing + retrieval pipeline for one user question.

    Returns:
        (route_result, chunks, short_circuit_reply)
        - route_result: dict from router.route()
        - chunks: retrieved chunk dicts (empty list when short-circuited)
        - short_circuit_reply: non-None when the pipeline should skip the
          answer LLM and return this string directly (greetings, gibberish, etc.)
    """
    route_result = _router.route(question)
    kind = route_result.get("kind", "question")

    if kind != "question":
        reply = _NON_QUESTION_REPLIES.get(kind, "I cannot help with that.")
        return route_result, [], reply

    scope = route_result.get("scope", "broad")
    doc_keys = route_result.get("documents", [])
    expanded_query = route_result.get("expanded_query") or question
    sub_queries = route_result.get("sub_queries", [])

    doc_filter: Optional[List[str]] = (
        _router.resolve_doc_ids(doc_keys) if doc_keys else None
    ) or None
    # Recorded so the answer step can tell the model which documents the
    # question is about (and say so when they don't contain the answer).
    route_result["doc_ids"] = doc_filter or []
    route_result["doc_names"] = _router.doc_names(doc_filter or [])
    top_k = _TOP_K.get(scope, 40)

    if sub_queries:
        # Search each sub-query separately, each in its own documents when the
        # router tagged it, and merge. A sub-query without its own documents
        # uses the question's documents.
        per_query = max(_MIN_PER_SUB_QUERY, top_k // len(sub_queries))
        chunks = []
        for sq in sub_queries:
            sq_docs = _router.resolve_doc_ids(sq["documents"]) or doc_filter
            sq_embedding = embedding_service.embed_text(sq["query"])
            chunks.extend(
                _retrieve(search_service, sq["query"], sq_embedding, sq_docs, per_query)
            )
        chunks = _dedupe(chunks)
    else:
        # Use expanded query for retrieval; fall back to original.
        query_embedding = embedding_service.embed_text(expanded_query)
        chunks = _retrieve(search_service, expanded_query, query_embedding, doc_filter, top_k)

    # Add the articles that the retrieved text (or the question) refers to.
    chunks = chunks + _references.follow_references(
        chunks, search_service, question=question, doc_filter=doc_filter
    )

    return route_result, chunks, None


def _retrieve(
    search_service: Any,
    query: str,
    query_embedding: Any,
    doc_ids: Optional[List[str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Search, giving every requested document its own share of the results.

    With one document (or none) this is a single search. With several, each
    document is searched separately and the results are interleaved by rank,
    so each document is represented in the context.
    """
    if not doc_ids or len(doc_ids) == 1:
        return search_service.search_documents(
            query, query_embedding, top_k=top_k, doc_filter=doc_ids or None
        )

    per_doc = max(_MIN_PER_DOC, top_k // len(doc_ids))
    per_doc_results = [
        search_service.search_documents(query, query_embedding, top_k=per_doc, doc_filter=[d])
        for d in doc_ids
    ]
    interleaved: List[Dict[str, Any]] = []
    for rank in range(max((len(r) for r in per_doc_results), default=0)):
        for results in per_doc_results:
            if rank < len(results):
                interleaved.append(results[rank])
    return _dedupe(interleaved)


def _dedupe(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop repeats: same chunk id, or the same expanded article reached twice."""
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk.get("is_expanded"):
            key = ("article", chunk.get("doc_id"), chunk.get("headers") or chunk.get("header_identifier"))
        else:
            key = ("chunk", chunk.get("id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique
