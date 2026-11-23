"""One-hop cross-reference following.

After retrieval, find references such as "Article 429", "Articles 92 and 94"
or "Article 92(3)" in the retrieved chunks (and in the question itself) and
add the referenced articles of the same document to the context, so the
answer can explain what a referenced provision says.

Only references inside the same document are followed. References to other
acts ("Article 4 of Directive 2013/36/EU") are skipped: we cannot tell
reliably which document in the index, if any, they point to.

Articles are looked up by header_identifier, which the EUR-Lex chunker
writes as "<DOC> Art. <n>" or "<DOC> Art. <n>(<paragraph>)...".
FATF recommendations are looked up as "... Recommendation <n>".
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# "Article 429", "Article 92(3)", "Articles 92 and 94", "Articles 26 to 28", "Art. 5"
_ARTICLE_RE = re.compile(
    r"\b(?:Articles?|Art\.)\s+"
    r"(\d+[a-z]?)(?:\(\d+[a-z]?\))*"
    r"((?:\s*(?:,|and|or|to)\s*\d+[a-z]?(?:\(\d+[a-z]?\))*)*)",
)
_FOLLOWING_NUMBER_RE = re.compile(r"(?:,|and|or|to)\s*(\d+[a-z]?)")
_RANGE_RE = re.compile(r"(\d+)\s*to\s*(\d+)")
# Text right after a reference that means it points to another legal act.
_EXTERNAL_RE = re.compile(
    r"^[\s\d()a-z,]*\s*of\s+(?:Regulation|Directive|Decision|Delegated|Implementing|Council|Commission|the\s+(?:Treaty|TFEU|Charter))",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(r"\bRecommendations?\s+(\d{1,2})\b")

MAX_REFERENCED_ARTICLES = 6   # extra articles added per question
MAX_TOKENS_PER_ARTICLE = 1000  # ~4,000 characters per referenced article (keeps the prompt small)
MAX_RANGE = 10                 # "Articles 26 to 28" expands, "Articles 1 to 400" does not


def _article_numbers(text: str) -> List[str]:
    """Article numbers referenced in text, in order, skipping references to other acts."""
    numbers: List[str] = []
    for match in _ARTICLE_RE.finditer(text):
        tail = text[match.end(): match.end() + 80]
        if _EXTERNAL_RE.match(tail):
            continue
        group = [match.group(1)] + _FOLLOWING_NUMBER_RE.findall(match.group(2) or "")
        span = match.group(0)
        for start, end in _RANGE_RE.findall(span):
            start_i, end_i = int(start), int(end)
            if 0 < end_i - start_i <= MAX_RANGE:
                group.extend(str(n) for n in range(start_i + 1, end_i))
        for number in group:
            if number not in numbers:
                numbers.append(number)
    return numbers


def _recommendation_numbers(text: str) -> List[str]:
    numbers: List[str] = []
    for number in _RECOMMENDATION_RE.findall(text):
        if number not in numbers:
            numbers.append(number)
    return numbers


def _own_article(chunk: Dict[str, Any]) -> Optional[str]:
    """Article number of the chunk itself, so self-references are ignored."""
    match = re.search(r" Art\. (\d+[a-z]?)", chunk.get("header_identifier") or "")
    return match.group(1) if match else None


def _fetch_article(search_service: Any, doc_id: str, number: str) -> List[Dict[str, Any]]:
    """All chunks of one article (all paragraphs), in document order."""
    ident = f"{doc_id} Art. {number}"
    rows = search_service.conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? AND (header_identifier = ? OR header_identifier LIKE ?) ORDER BY id",
        (doc_id, ident, ident + "(%"),
    ).fetchall()
    return [search_service._row_to_chunk(row) for row in rows]


def _fetch_recommendation(search_service: Any, doc_id: str, number: str) -> List[Dict[str, Any]]:
    rows = search_service.conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? AND header_identifier LIKE ? ORDER BY id",
        (doc_id, f"% Recommendation {number}"),
    ).fetchall()
    return [search_service._row_to_chunk(row) for row in rows]


def _merge(parts: List[Dict[str, Any]], label: str, referenced_from: str) -> Optional[Dict[str, Any]]:
    """Merge an article's chunks into one context entry, capped in size."""
    if not parts:
        return None
    texts, tokens = [], 0.0
    for part in parts:
        content = part.get("content") or ""
        cost = len(content.split()) * 1.3
        if tokens + cost > MAX_TOKENS_PER_ARTICLE:
            if not texts:
                # first paragraph alone is too long: keep its beginning
                texts.append(" ".join(content.split()[: int(MAX_TOKENS_PER_ARTICLE / 1.3)]) + " [...]")
            else:
                texts.append("[... further paragraphs of this article omitted]")
            break
        texts.append(content)
        tokens += cost
    merged = dict(parts[0])
    merged["content"] = "\n\n".join(texts)
    merged["header_identifier"] = label
    merged["is_reference"] = True
    merged["referenced_from"] = referenced_from
    return merged


def follow_references(
    chunks: List[Dict[str, Any]],
    search_service: Any,
    question: str = "",
    doc_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return extra context entries for articles referenced by the retrieved chunks
    (or by the question, when the question is about a single document).

    The caller appends them after the retrieved chunks.
    """
    if not chunks or getattr(search_service, "conn", None) is None:
        return []

    already: Set[Tuple[str, str]] = set()
    for chunk in chunks:
        own = _own_article(chunk)
        if own:
            already.add((chunk.get("doc_id"), own))

    # (doc_id, kind, number, referenced_from), in priority order: question first, then chunks in rank order
    wanted: List[Tuple[str, str, str, str]] = []

    def want(doc_id: str, kind: str, number: str, source: str) -> None:
        key = (doc_id, number)
        if key in already or any((w[0], w[2]) == key for w in wanted):
            return
        wanted.append((doc_id, kind, number, source))

    if doc_filter and len(doc_filter) == 1:
        doc_id = doc_filter[0]
        for number in _article_numbers(question):
            want(doc_id, "article", number, "the question")
        if doc_id == "FATF2012":
            for number in _recommendation_numbers(question):
                want(doc_id, "recommendation", number, "the question")

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue
        source = chunk.get("header_identifier") or doc_id
        own = _own_article(chunk)
        content = chunk.get("content") or ""
        for number in _article_numbers(content):
            if number != own:
                want(doc_id, "article", number, source)
        if doc_id == "FATF2012":
            for number in _recommendation_numbers(content):
                want(doc_id, "recommendation", number, source)

    extra: List[Dict[str, Any]] = []
    for doc_id, kind, number, source in wanted:
        if len(extra) >= MAX_REFERENCED_ARTICLES:
            break
        if kind == "article":
            parts = _fetch_article(search_service, doc_id, number)
            label = f"{doc_id} Art. {number}"
        else:
            parts = _fetch_recommendation(search_service, doc_id, number)
            label = f"{doc_id} Recommendation {number}"
        # Skip if any of these chunks was already retrieved
        retrieved_ids = {c.get("id") for c in chunks}
        if any(p.get("id") in retrieved_ids for p in parts):
            continue
        merged = _merge(parts, label, source)
        if merged:
            extra.append(merged)
    return extra
