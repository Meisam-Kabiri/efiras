#!/usr/bin/env python3
"""
bucket_b_chunker.py

Chunker for the Bucket B documents (Basel III, FATF 2012, CSSF 18/698), whose
HTML in html_cache/ was generated from PDFs by scripts/run_bucket_b_pipeline.py.

Why a separate chunker: TreeChunker only recognises EUR-Lex id prefixes
(cpt_, sct_, art_, ...). The Bucket B HTML also uses prt_, tit_, sbs_, par_
and pre_, so TreeChunker merged whole parts into one giant chunk (Basel III)
or dropped most of the text (FATF, CSSF). This walker treats every <div id="xxx_...">
as a structural node, whatever its prefix, so no text is lost.

Every node's own text (its paragraphs and tables, not its sub-sections) becomes
a chunk carrying the full heading path. Long texts are split at paragraph
boundaries so no chunk exceeds MAX_CHARS, which keeps every chunk well under
the embedder's 8,000-character cap.

No LLM calls: the HTML already holds the structure.

Usage (from backend/regulatory_chunker):
    python bucket_b_chunker.py            # writes chunks_output/<doc>_chunks.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

HERE = Path(__file__).parent
HTML_DIR = HERE / "html_cache"
OUT_DIR = HERE / "chunks_output"

# doc_id as stored in the index -> HTML file in html_cache/
DOCS = {
    "BASEL3": "basel_iii.html",
    "FATF2012": "fatf_2012.html",
    "CSSF_18_698": "cssf_18_698.html",
}

MAX_CHARS = 4000  # split longer texts at paragraph boundaries
MIN_CHARS = 40    # drop fragments shorter than this (stray numbers, page marks)

# id prefix -> (node type, word used in citations)
TYPES = {
    "pre": ("preamble", "Preamble"),
    "prt": ("part", "Part"),
    "tit": ("title", "Title"),
    "cpt": ("chapter", "Chapter"),
    "sct": ("section", "Section"),
    "sbs": ("subsection", "Subsection"),
    "art": ("article", "Article"),
    "par": ("paragraph", "Paragraph"),
    "anx": ("annex", "Annex"),
    "app": ("appendix", "Appendix"),
}
# FATF numbers its recommendations as art_ in the HTML; cite them by their real name.
# (FATF also numbers some recommendations as sbs_9-, sbs_10-, ...)
DOC_TYPE_WORDS = {"FATF2012": {"art": "Recommendation", "sbs": "Recommendation"}}

ID_RE = re.compile(r"^([a-z]+)_(.*)$")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def readable_label(prefix: str, raw: str, doc_id: str) -> str:
    """'A-' -> 'Section A', 'Part-1' -> 'Part 1', 'Sub-chapter-2-1' -> 'Sub-chapter 2.1'."""
    label = raw.strip("-").replace("-", " ").strip()
    word = TYPES.get(prefix, (prefix, prefix.title()))[1]
    override = DOC_TYPE_WORDS.get(doc_id, {}).get(prefix)
    if override and re.fullmatch(r"\d+[A-Z]?", label):
        word = override
    label = re.sub(r"(\d) (\d)", r"\1.\2", label)  # 2 1 -> 2.1
    if not label:
        return word
    first = label.split(" ")[0].lower()
    if first.startswith(word.lower()[:4]) or first in ("sub", "sub chapter", "chapter", "part", "annex"):
        return label
    if prefix == "pre":
        return word
    return f"{word} {label}"


def is_structural(el: Any) -> bool:
    return isinstance(el, Tag) and el.name == "div" and bool(ID_RE.match(el.get("id") or ""))


def own_blocks(div: Tag) -> List[str]:
    """Text blocks that belong to this node itself: paragraphs, tables, lists and
    unlabelled wrapper divs, but not headings and not structural sub-nodes."""
    blocks = []
    for child in div.children:
        if isinstance(child, NavigableString):
            text = clean(str(child))
            if text:
                blocks.append(text)
            continue
        if not isinstance(child, Tag) or is_structural(child):
            continue
        if child.name == "p" and "heading" in (child.get("class") or []):
            continue
        if child.name == "div":
            # unlabelled wrapper: keep its own content, recurse for nested wrappers
            if child.find(is_structural):
                blocks.extend(own_blocks(child))
                continue
        if child.name == "table":
            rows = []
            for tr in child.find_all("tr"):
                cells = [clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            text = "\n".join(rows)
        else:
            text = clean(child.get_text(" "))
        if text:
            blocks.append(text)
    return blocks


def heading_of(div: Tag) -> Optional[str]:
    for child in div.children:
        if isinstance(child, Tag) and child.name == "p" and "heading" in (child.get("class") or []):
            text = clean(child.get_text(" "))
            if text:
                return text
    return None


def split_blocks(blocks: List[str]) -> List[str]:
    """Group blocks into parts of at most MAX_CHARS, splitting oversized blocks at sentences."""
    pieces = []
    for block in blocks:
        if len(block) <= MAX_CHARS:
            pieces.append(block)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.;:])\s+", block):
            if current and len(current) + len(sentence) + 1 > MAX_CHARS:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    parts, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > MAX_CHARS:
            parts.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


def chunk_document(doc_id: str, html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    chunks: List[Dict[str, Any]] = []

    def walk(div: Tag, path: List[Dict[str, Any]], id_path: List[str]) -> None:
        prefix, raw = ID_RE.match(div["id"]).groups()
        node_type = TYPES.get(prefix, (prefix, prefix))[0]
        node = {
            "type": node_type,
            "label": readable_label(prefix, raw, doc_id),
            "heading": heading_of(div),
        }
        new_path = path + [node]
        new_ids = id_path + [div["id"]]

        text_parts = split_blocks(own_blocks(div))
        text_parts = [t for t in text_parts if len(t) >= MIN_CHARS]
        # Short, readable citation: the last three levels of the path.
        labels = [n["label"] for n in new_path if n["type"] != "preamble"][-3:]
        citation = f"{doc_id} " + ", ".join(labels) if labels else f"{doc_id} Preamble"
        base_id = f"{doc_id}/" + "/".join(new_ids)
        for i, text in enumerate(text_parts, 1):
            chunks.append({
                "chunk_id": base_id if len(text_parts) == 1 else f"{base_id}#part{i}",
                "doc_id": doc_id,
                "path": new_path,
                "citation": citation,
                "text": text,
            })

        for child in div.find_all(is_structural):
            # only direct structural descendants (skip ones nested in another structural div)
            parent = child.find_parent(is_structural)
            if parent is div:
                walk(child, new_path, new_ids)

    for top in body.find_all(is_structural):
        if top.find_parent(is_structural) is None:
            walk(top, [], [])

    # Footnotes sit at body level, outside every section: keep them as their own chunks.
    footnote_parts = [t for t in split_blocks(own_blocks(body)) if len(t) >= MIN_CHARS]
    for i, text in enumerate(footnote_parts, 1):
        chunks.append({
            "chunk_id": f"{doc_id}/footnotes#part{i}",
            "doc_id": doc_id,
            "path": [{"type": "footnotes", "label": "Footnotes", "heading": "Footnotes"}],
            "citation": f"{doc_id} Footnotes",
            "text": text,
        })

    # chunk_ids must be unique
    seen: Dict[str, int] = {}
    for c in chunks:
        n = seen.get(c["chunk_id"], 0)
        seen[c["chunk_id"]] = n + 1
        if n:
            c["chunk_id"] = f"{c['chunk_id']}~{n}"
    return chunks


def main(doc_ids: List[str]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for doc_id in doc_ids:
        html_file = HTML_DIR / DOCS[doc_id]
        html = html_file.read_text(encoding="utf-8")
        chunks = chunk_document(doc_id, html)

        source_chars = len(clean(BeautifulSoup(html, "html.parser").get_text(" ")))
        chunk_chars = sum(len(c["text"]) for c in chunks)
        out = OUT_DIR / f"{doc_id.lower()}_chunks.json"
        out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{doc_id}] {len(chunks)} chunks, max {max(len(c['text']) for c in chunks)} chars, "
            f"text kept {chunk_chars:,}/{source_chars:,} chars -> {out.name}"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or list(DOCS))
