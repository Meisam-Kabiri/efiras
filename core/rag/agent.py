"""Agent pipeline: route → retrieve → (sufficiency loop) → answer."""

from typing import Any, Dict, List, Optional, Tuple

from core.rag import router as _router

# Top-k by scope — tunable constants.
_TOP_K = {"strict": 55, "narrowed": 50, "broad": 40}

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
        _router.resolve_filenames(doc_keys) if doc_keys else None
    )
    top_k = _TOP_K.get(scope, 40)

    # Use expanded query for retrieval; fall back to original.
    query_embedding = embedding_service.embed_text(expanded_query)

    if sub_queries:
        # Search each sub-query separately and merge (dedup by chunk id).
        seen: Dict[int, Dict[str, Any]] = {}
        for sq in sub_queries:
            sq_embedding = embedding_service.embed_text(sq)
            for chunk in search_service.search_documents(
                sq, sq_embedding, top_k=top_k, doc_filter=doc_filter
            ):
                seen.setdefault(chunk["id"], chunk)
        chunks = list(seen.values())
    else:
        chunks = search_service.search_documents(
            expanded_query, query_embedding, top_k=top_k, doc_filter=doc_filter
        )

    return route_result, chunks, None
