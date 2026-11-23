"""LLM Layer 1 — query router.

Classifies the user question, resolves document scope, expands the query,
and splits multi-part questions into sub-queries — all before any retrieval.
"""

from typing import Any, Dict, List, Optional

from core.rag.llm import call_llm

ROUTER_MODEL = "gpt-4o-mini"

# All 23 documents in the index. Keys match the router's output; values are
# (doc_id exactly as stored in regulatory_chunks.db, human-readable name).
# The doc_id must match the database exactly, or the document filter in
# search_service silently matches nothing.
DOC_KEYS: Dict[str, tuple] = {
    "aifmd":            ("AIFMD", "Alternative Investment Fund Managers Directive (AIFMD)"),
    "aifmd_level_2":    ("AIFMD_L2", "AIFMD Level 2 Delegated Regulation"),
    "amlr":             ("AMLR", "Anti-Money Laundering Regulation (AMLR)"),
    "4amld":            ("4AMLD", "Fourth Anti-Money Laundering Directive (4AMLD)"),
    "5amld":            ("5AMLD", "Fifth Anti-Money Laundering Directive (5AMLD)"),
    "basel_iii":        ("BASEL3", "Basel III: International regulatory framework for banks (Basel Committee)"),
    "crd_v":            ("CRD5", "Capital Requirements Directive (CRD V)"),
    "crr":              ("CRR", "Capital Requirements Regulation (CRR), the EU law implementing Basel III"),
    "cssf_18_698":      ("CSSF_18_698", "Luxembourg CSSF Circular 18/698 (investment fund managers)"),
    "dora":             ("DORA", "Digital Operational Resilience Act (DORA)"),
    "emir":             ("EMIR", "European Market Infrastructure Regulation (EMIR)"),
    "eu_taxonomy":      ("EUTAXONOMY", "EU Taxonomy Regulation"),
    "fatf":             ("FATF2012", "FATF Recommendations 2012"),
    "gdpr":             ("GDPR", "General Data Protection Regulation (GDPR)"),
    "mar":              ("MAR", "Market Abuse Regulation (MAR)"),
    "mifid_ii":         ("MIFID2", "Markets in Financial Instruments Directive II (MiFID II)"),
    "mifid_ii_level_2": ("MIFID2_L2", "MiFID II Level 2 Delegated Regulation"),
    "mifir":            ("MIFIR", "Markets in Financial Instruments Regulation (MiFIR)"),
    "psd2":             ("PSD2", "Payment Services Directive 2 (PSD2)"),
    "sfdr":             ("SFDR", "Sustainable Finance Disclosure Regulation (SFDR)"),
    "sftr":             ("SFTR", "Securities Financing Transactions Regulation (SFTR)"),
    "solvency_ii":      ("SOLVENCY2", "Solvency II Directive"),
    "ucits":            ("UCITS", "UCITS Directive"),
}

_DOC_LIST_TEXT = "\n".join(
    f"  - {key}: {name}" for key, (_doc_id, name) in DOC_KEYS.items()
)

_SYSTEM_PROMPT = f"""You are a routing assistant for a financial-regulation question-answering system.

Available documents (use the exact key from this list):
{_DOC_LIST_TEXT}

Your job: analyse the user's question and return a JSON object with these fields:

{{
  "kind": "question" | "greeting" | "gibberish" | "out_of_scope",
  "scope": "strict" | "narrowed" | "broad",
  "documents": ["<doc_key>", ...],
  "expanded_query": "<rewritten query with synonyms / full terms>",
  "sub_queries": ["<sub_query_1>", ...]
}}

Rules:
- kind:
    "question"     — a genuine regulatory / compliance / legal question.
    "greeting"     — hello, thanks, etc.
    "gibberish"    — random characters, no meaning.
    "out_of_scope" — a real question but not about financial regulation.
- scope (only matters when kind == "question"):
    "strict"   — question explicitly names ONE regulation (e.g. "under GDPR…"). Put that doc in documents[].
    "narrowed" — question clearly covers a small group of regulations (2–4 docs). List them.
    "broad"    — genuinely vague or multi-regulation. Set documents to [].
- expanded_query: rewrite the question using formal legal CONCEPT SYNONYMS and KEY CONCEPTS only.
  CRITICAL RULE: DO NOT guess, invent, or output specific Article or Section numbers (e.g., do NOT output 'Article 22' or 'Article 13'). 
  Allow the database vector and BM25 search engines to discover the exact Articles from the indexed documents. Omit if kind != "question".
- sub_queries: split ONLY if the question is clearly multi-part (e.g. "compare X and Y", "what are the
  obligations under X AND the penalties under Y?"). Each sub_query is a complete standalone question.
  Empty list otherwise.

Respond with valid JSON only. No markdown, no explanation."""


def route(question: str) -> Dict[str, Any]:
    """Classify and expand a user question. Returns the routing dict."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = call_llm(
        model=ROUTER_MODEL,
        messages=messages,
        schema={},  # forces JSON output
        temperature=0.0,
        max_tokens=500,
    )
    # Normalise: ensure required keys exist with safe defaults
    result.setdefault("kind", "question")
    result.setdefault("scope", "broad")
    result.setdefault("documents", [])
    result.setdefault("expanded_query", question)
    result.setdefault("sub_queries", [])
    return result


def resolve_doc_ids(doc_keys: List[str]) -> List[str]:
    """Convert router doc keys to database doc_ids. Unrecognised keys are dropped."""
    return [DOC_KEYS[k][0] for k in doc_keys if k in DOC_KEYS]


def doc_names(doc_ids: List[str]) -> List[str]:
    """Human-readable names for database doc_ids (falls back to the id)."""
    by_id = {doc_id: name for doc_id, name in DOC_KEYS.values()}
    return [by_id.get(d, d) for d in doc_ids]
