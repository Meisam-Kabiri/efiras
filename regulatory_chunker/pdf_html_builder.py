"""
pdf_html_builder.py

Translates pdf_to_html.py's flat event list (open/close/para/continue_para/
table/footnote) into a single HTML string that TreeChunker can walk exactly
like real EUR-Lex HTML -- same id="prefix_label" convention on structural
divs, span.no-parag markers for numbered paragraphs, direct-child <table>
elements, and top-level <p class="footnote"> siblings outside the whole
structural tree, cross-referenced via id="fn_N" / id="src.fn_N" (the same
mechanism TreeChunker._citing_context already reads for real footnotes).
No changes to TreeChunker itself are needed -- this only has to speak the
grammar it already understands.

This module defines the translation logic only -- see
scripts/build_html_from_events.py for the runner.
"""

import html as html_lib
import re

try:
    from .fetch_chunk import EurLexChunker
except ImportError:
    from fetch_chunk import EurLexChunker

# canonical type -> id prefix, reused from EurLexChunker's own level table so
# a canonical type discovered by the LLM (e.g. "sub-section") gets exactly
# the prefix EurLexChunker/TreeChunker already recognize ("sbs").
TYPE_TO_PREFIX = {v: k for k, v in EurLexChunker.DEFAULT_LEVELS.items()}
TYPE_TO_PREFIX["article"] = "art"

_RESERVED_PREFIXES = set(TYPE_TO_PREFIX.values())


def _slug(label):
    # "_" is the prefix/label separator TreeChunker splits on, "." is its
    # ancestor-path separator -- neither may appear inside a label without
    # being misread as structure, so both get folded to "-" here. This only
    # affects the id attribute; citation text is built by TreeChunker from
    # this same slugged label, so a label like "2.1" will show as "2-1" in
    # citations -- a cosmetic difference only, no content is lost.
    return re.sub(r"[_.\s]+", "-", label.strip()) or "x"


class _PrefixAllocator:
    """Hands out a short, stable, collision-free id prefix per canonical
    type -- known types reuse EurLexChunker's own table (e.g. "chapter" ->
    "cpt"); a genuinely new type gets a mechanical prefix derived from its
    own name, so the LLM never has to invent a valid one itself."""

    def __init__(self):
        self.assigned = dict(TYPE_TO_PREFIX)
        self.used = set(_RESERVED_PREFIXES)

    def prefix_for(self, type_):
        if type_ in self.assigned:
            return self.assigned[type_]
        base = re.sub(r"[^a-z]", "", type_.lower())[:3] or "typ"
        candidate = base
        n = 1
        while candidate in self.used:
            n += 1
            candidate = f"{base}{n}"
        self.assigned[type_] = candidate
        self.used.add(candidate)
        return candidate


def merge_continued_paragraphs(events):
    """Folds each continue_para's text onto the immediately preceding
    para/continue_para event. continue_para only ever appears as the first
    event of a new batch, completing a sentence the page/batch boundary cut
    off mid-way -- it is never a real standalone paragraph, so by the time
    HTML is built there should be no separate continue_para events left."""
    merged = []
    for e in events:
        if e.get("op") == "continue_para" and merged and merged[-1].get("op") in ("para", "continue_para"):
            prev = dict(merged[-1])
            prev["text"] = (prev.get("text", "") + " " + e.get("text", "")).strip()
            merged[-1] = prev
        else:
            merged.append(dict(e))
    return merged


def wrap_preamble(events):
    """If any body content (para/table/footnote) appears before the first
    `open` event, wraps that stretch in a synthetic "preamble" div. Real
    documents routinely have front matter before any real Part/Chapter
    starts -- e.g. a CSSF circular's cover letter, which is where "IFM" (a
    term used throughout the rest of the document) actually gets defined.
    Without this, that content would end up as bare <p> tags directly in
    <body>, outside every structural div -- and TreeChunker only ever walks
    structural divs, so it would be silently invisible to it. Wrapping it
    keeps TreeChunker's contract simple (every real chunk of content lives
    inside *some* div) instead of adding a special case for content that
    lives outside all of them."""
    first_open_idx = next((i for i, e in enumerate(events) if e.get("op") == "open"), None)
    if first_open_idx is None or first_open_idx == 0:
        return events  # nothing before the first open, or no open at all
    if not any(e.get("op") in ("para", "table", "footnote") for e in events[:first_open_idx]):
        return events  # nothing but other opens/closes preceded it -- shouldn't happen, but no-op either way
    return (
        [{"op": "open", "type": "preamble", "label": "Preamble", "heading": None}]
        + events[:first_open_idx]
        + [{"op": "close", "type": "preamble"}]
        + events[first_open_idx:]
    )


def events_to_html(events, doc_id):
    """Builds one HTML document from a flat event list. Returns the HTML
    string; does not touch the filesystem."""
    events = merge_continued_paragraphs(events)
    events = wrap_preamble(events)
    allocator = _PrefixAllocator()

    body_parts = []
    footnote_parts = []
    stack = []
    last_para_ref = None  # index into body_parts of the last emitted <p>, for footnote src anchors
    footnote_counter = 0

    for e in events:
        op = e.get("op")

        if op == "open":
            prefix = allocator.prefix_for(e.get("type") or "section")
            label = e.get("label") or str(len(stack) + 1)
            div_id = f"{prefix}_{_slug(label)}"
            stack.append(div_id)
            body_parts.append(f'<div id="{html_lib.escape(div_id, quote=True)}">')
            heading = e.get("heading")
            if heading:
                body_parts.append(f'<p class="heading">{html_lib.escape(heading)}</p>')

        elif op == "close":
            if stack:
                stack.pop()
                body_parts.append("</div>")

        elif op == "para":
            label = e.get("label")
            text = html_lib.escape(e.get("text") or "")
            if label:
                body_parts.append(
                    f'<p><span class="no-parag">{html_lib.escape(label)}</span> {text}</p>'
                )
            else:
                # no natural number in the source -- left unmarked, so
                # TreeChunker's _split_paragraphs treats it as this div's own
                # lead-in text (captured alongside any later numbered
                # paragraphs) rather than a numbered paragraph.
                body_parts.append(f"<p>{text}</p>")
            last_para_ref = len(body_parts) - 1

        elif op == "table":
            rows = e.get("rows") or []
            row_html = "".join(
                "<tr>" + "".join(f"<td>{html_lib.escape(str(cell))}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            # must be a *direct* child of the currently open div -- TreeChunker
            # only scans recursive=False for tables.
            body_parts.append(f"<table>{row_html}</table>")

        elif op == "footnote":
            footnote_counter += 1
            fn_id = f"fn_{footnote_counter}"
            label = e.get("label") or str(footnote_counter)
            text = html_lib.escape(e.get("text") or "")
            footnote_parts.append(
                f'<p class="footnote"><a id="{fn_id}">{html_lib.escape(label)}</a> {text}</p>'
            )
            if last_para_ref is not None:
                src_anchor = f'<a id="src.{fn_id}"></a>'
                body_parts[last_para_ref] = body_parts[last_para_ref][:-4] + src_anchor + "</p>"

    while stack:
        stack.pop()
        body_parts.append("</div>")

    return (
        "<html><body>\n"
        + "\n".join(body_parts)
        + "\n"
        + "\n".join(footnote_parts)
        + "\n</body></html>"
    )
