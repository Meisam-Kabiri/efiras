"""LLM Layer 1 — query router.

Classifies the user question, resolves document scope, expands the query,
and splits multi-part questions into sub-queries — all before any retrieval.
"""

from typing import Any, Dict, List, Optional

from core.rag.llm import call_llm

ROUTER_MODEL = "gpt-4o-mini"

# All 23 documents in the corpus. Keys match the router's output.
DOC_KEY_TO_FILENAME: Dict[str, str] = {
    "aifmd":            "Alternative_Investment_Fund_Managers_Directive_(AIFMD).pdf",
    "aifmd_level_2":    "Alternative_Investment_Fund_Managers_Directive_Level_2_(AIFMD_Level_2).pdf",
    "basel_ii":         "Basel_II_Framework_(Basel_II_2006).pdf",
    "basel_iii":        "Basel_III_Framework_(Basel_III).pdf",
    "crd_v":            "Capital_Requirements_Directive_V_(CRD_V).pdf",
    "crr":              "Capital_Requirements_Regulation_(CRR).pdf",
    "dodd_frank":       "Dodd_Frank_Wall_Street_Reform_and_Consumer_Protection_Act_(Dodd_Frank).pdf",
    "dodd_frank_2":     "Dodd_Frank_Wall_Street_Reform_and_Consumer_Protection_Act_2_(Dodd_Frank_Act).pdf",
    "emir":             "European_Market_Infrastructure_Regulation_(EMIR).pdf",
    "eu_taxonomy":      "European_Union_Taxonomy_Regulation_(EU_Taxonomy).pdf",
    "5amld":            "Fifth_Anti_Money_Laundering_Directive_(5AMLD).pdf",
    "fatf":             "Financial_Action_Task_Force_Recommendations_2012_(FATF).pdf",
    "4amld":            "Fourth_Anti_Money_Laundering_Directive_(4AMLD).pdf",
    "gdpr":             "General_Data_Protection_Regulation_(GDPR).pdf",
    "cssf_18_698":      "Luxembourg_CSSF_18_698(CSSF_18_698).pdf",
    "mifid_ii":         "Markets_in_Financial_Instruments_Directive_II_(MiFID_II).pdf",
    "mifir":            "Markets_in_Financial_Instruments_Regulation_(MiFIR).pdf",
    "psd2":             "Payment_Services_Directive_2_(PSD2).pdf",
    "sftr":             "Securities_Financing_Transactions_Regulation_(SFTR).pdf",
    "solvency_ii":      "Solvency_II_Directive_(Solvency_II).pdf",
    "solvency_ii_level_2": "Solvency_II_Directive_Level_2_(Solvency_II_Level_2).pdf",
    "sfdr":             "Sustainable_Finance_Disclosure_Regulation_(SFDR).pdf",
    "ucits":            "Undertakings_for_Collective_Investment_in_Transferable_Securities_(UCITS).pdf",
}

_DOC_LIST_TEXT = "\n".join(
    f"  - {key}: {fname}" for key, fname in DOC_KEY_TO_FILENAME.items()
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
- expanded_query: rewrite the question with full regulatory terms, article synonyms, and key concepts.
  Keep the same meaning but add useful vocabulary for retrieval. Omit if kind != "question".
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


def resolve_filenames(doc_keys: List[str]) -> List[str]:
    """Convert router doc keys to actual filenames. Unrecognised keys are dropped."""
    return [DOC_KEY_TO_FILENAME[k] for k in doc_keys if k in DOC_KEY_TO_FILENAME]
