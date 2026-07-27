"""
pdf_to_html.py

Sequential, batch-by-batch PDF structure-extraction pipeline.

Walks a PDF page-by-page (via PyMuPDF), groups pages into batches by a
character budget, and calls an LLM once per batch to classify the batch's
raw text into a flat list of structural events (open/close/para/
continue_para/table/footnote). Unlike feeding the model the whole document
or its growing output, only a small, bounded state is carried forward
between batches:

  - the current open hierarchy stack (part/chapter/article/... currently
    "open", tracked deterministically by this module from prior events --
    never re-derived by the model)
  - a short literal tail of the previous batch's raw text (so the model can
    tell whether this batch continues a sentence cut off mid-page)
  - the native-term -> canonical-type mapping discovered so far (grows as
    genuinely new structural types are encountered; carried forward so the
    same native concept doesn't get relabeled differently in a later batch)

There is deliberately no separate upfront "discovery" pass -- the type
vocabulary is built incrementally by processing the document in order, since
a document may not have a usable table of contents and random/non-
contiguous sampling breaks the context needed to tell how a new type nests.

This module produces the event stream plus a per-batch text-fidelity
reconciliation report (word-diff against the raw PyMuPDF text) -- the only
stage of the Bucket B pipeline that costs money (one LLM call per batch).
Translating the event stream into HTML is pdf_html_builder.py's job.

This module defines the extraction logic only -- see
scripts/run_bucket_b_pipeline.py for the runner that loops over every
Bucket B document.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # "import fitz" collides with an unrelated PyPI package of that name
from anthropic import Anthropic
from dotenv import load_dotenv

try:
    from .fetch_chunk import EurLexChunker
except ImportError:
    from fetch_chunk import EurLexChunker

load_dotenv()

TARGET_CHARS = 12_000
MIN_PAGES = 4
MAX_PAGES = 8
TAIL_CHARS = 300

# The canonical type vocabulary the downstream chunker (TreeChunker) already
# understands, shown to the model on every batch call -- without this, batch
# 1 has an empty KNOWN TYPES dict (nothing discovered yet) and no reason to
# prefer an existing name, so a native term like "Sub-chapter" gets mapped to
# a brand-new invented type ("sub-chapter") instead of the equivalent
# existing "sub-section". Found via a real 3-batch CSSF test run. This list
# is a *preference*, not a hard cap -- a genuinely new kind of structural
# unit (nothing here fits) should still get its own new type.
CANONICAL_TYPES = sorted(set(EurLexChunker.DEFAULT_LEVELS.values()) | {
    "article", "paragraph", "annex", "table", "footnote", "definition",
})

SYSTEM_PROMPT = """\
You are converting a batch of raw text (extracted from a regulatory PDF via \
PyMuPDF) into a flat JSON event list, continuing a document you're reading \
sequentially -- you are NOT seeing this document from the start, you're \
mid-walk through it. Be extremely literal and faithful: do not summarize, \
paraphrase, reword, or drop any sentence. Every word of real body text in \
this batch must appear verbatim (whitespace/line-break normalization is \
fine) inside some event's "text" field. Ignore repeated page header/footer \
boilerplate (document title, "Page N/M", running section titles repeated on \
every page) -- that's not body content.

You will be told:
  - CANONICAL TYPES already understood downstream (e.g. "chapter", \
"sub-section", "article") -- always prefer mapping a native term onto one of \
these if it fits, even the first time you see that native term (e.g. a \
document's own "Sub-chapter" almost certainly IS a "sub-section" -- don't \
invent a new type name just because the native word looks different). Only \
introduce a new type name for a genuinely new *kind* of structural unit that \
none of these fit.
  - the STACK of structural levels currently open (e.g. you are currently \
inside Chapter 2 > Sub-chapter 2.1) -- continue inside it, only emit "open" \
for something new, "close" to pop back out
  - KNOWN TYPES discovered so far in this document (native term -> \
canonical type, e.g. "Sub-chapter" -> "sub-section") -- reuse these exact \
type names for the same kind of native structure; only introduce a new type \
name if you encounter a genuinely new kind of structural unit not covered \
by an existing one or by CANONICAL TYPES
  - TAIL TEXT: the last ~300 characters of raw text from the previous batch \
-- if this batch's text starts by completing that cut-off sentence, emit a \
continue_para for it before anything else

Emit a JSON array of events, each one of:
  {"op":"open","type":"<canonical type>","label":"<native label verbatim>","heading":"<text or null>"}
  {"op":"close","type":"<type>"}
  {"op":"para","label":"<native label verbatim>","text":"<verbatim text>"}
  {"op":"continue_para","text":"<verbatim text>"}
  {"op":"table","rows":[["cell","cell"], ...]}
  {"op":"footnote","label":"<label>","text":"<verbatim text>"}

Call the emit_batch_result tool exactly once with the full event list for \
this batch, plus any newly-introduced types.
"""

EMIT_TOOL = {
    "name": "emit_batch_result",
    "description": "Emit the structural events for this batch of PDF text, plus any newly introduced canonical types.",
    "input_schema": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["open", "close", "para", "continue_para", "table", "footnote"]},
                        "type": {"type": "string"},
                        "label": {"type": "string"},
                        "heading": {"type": ["string", "null"]},
                        "text": {"type": "string"},
                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    },
                    "required": ["op"],
                },
            },
            "new_types": {
                "type": "object",
                "description": "native term -> {canonical_type, id_label_pattern, citation_word}, only for types not already in KNOWN TYPES",
            },
        },
        "required": ["events"],
    },
}


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [doc[i].get_text() for i in range(doc.page_count)]


def make_batches(pages, start_page=0, target_chars=TARGET_CHARS, min_pages=MIN_PAGES, max_pages=MAX_PAGES):
    """Groups (1-indexed) pages into batches by a character budget, not a
    fixed page count -- naturally yields fewer pages/batch on dense pages,
    more on sparse ones, while keeping each LLM call's input roughly
    bounded regardless of document length."""
    batches = []
    i = start_page
    n = len(pages)
    while i < n:
        batch = [(i + 1, pages[i])]
        char_count = len(pages[i])
        i += 1
        while i < n and len(batch) < max_pages and (char_count < target_chars or len(batch) < min_pages):
            batch.append((i + 1, pages[i]))
            char_count += len(pages[i])
            i += 1
        batches.append(batch)
    return batches


def _norm_words(text):
    return Counter(re.findall(r"[\w'‘’]+", text.lower()))


def reconcile(raw_batch_text, events):
    """Deterministic word-diff between the batch's raw PyMuPDF text and the
    events' text/heading/label fields -- the safety net that doesn't trust
    the model's own claim of fidelity."""
    event_text_parts = []
    for e in events:
        event_text_parts.append(e.get("text") or "")
        event_text_parts.append(e.get("heading") or "")
        event_text_parts.append(e.get("label") or "")
        for row in e.get("rows") or []:
            event_text_parts.extend(row)
    event_words = _norm_words(" ".join(event_text_parts))

    # strip the obvious page-footer boilerplate before comparing
    clean_lines = [
        l for l in raw_batch_text.splitlines()
        if not re.match(r"^\s*Page\s+\d+/\d+\s*$", l)
    ]
    raw_words = _norm_words(" ".join(clean_lines))

    missing = raw_words - event_words
    extra = event_words - raw_words
    total = sum(raw_words.values()) or 1
    return {
        "missing_ratio": sum(missing.values()) / total,
        "extra_ratio": sum(extra.values()) / total,
        "missing_top": missing.most_common(10),
        "extra_top": extra.most_common(10),
    }


def call_batch(client, model, batch, stack, known_types, tail_text):
    batch_text = "\n".join(f"--- page {p} ---\n{t}" for p, t in batch)

    user_content = (
        f"CANONICAL TYPES (prefer these): {json.dumps(CANONICAL_TYPES)}\n"
        f"STACK (currently open): {json.dumps(stack)}\n"
        f"KNOWN TYPES so far: {json.dumps(known_types)}\n"
        f"TAIL TEXT from previous batch: {tail_text!r}\n\n"
        f"Batch text:\n\n{batch_text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        tools=[EMIT_TOOL],
        tool_choice={"type": "tool", "name": "emit_batch_result"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = tool_use.input
    return result.get("events", []), result.get("new_types", {}), response.usage, batch_text


def apply_events_to_stack(stack, events):
    stack = list(stack)
    for e in events:
        if e["op"] == "open":
            stack.append({"type": e.get("type"), "label": e.get("label"), "heading": e.get("heading")})
        elif e["op"] == "close":
            if stack:
                stack.pop()
    return stack


def run(pdf_path, doc_id, model="claude-sonnet-5", max_batches=None):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (check .env)")
    client = Anthropic()

    pages = extract_pages(pdf_path)
    batches = make_batches(pages)
    if max_batches:
        batches = batches[:max_batches]

    stack = []
    known_types = {}
    tail_text = ""
    all_events = []

    for i, batch in enumerate(batches, 1):
        page_range = f"{batch[0][0]}-{batch[-1][0]}"
        print(f"\n=== batch {i}/{len(batches)} (pages {page_range}) ===")

        events, new_types, usage, batch_text = call_batch(client, model, batch, stack, known_types, tail_text)
        print(f"  {usage.input_tokens} in / {usage.output_tokens} out tokens, {len(events)} events")

        if new_types:
            known_types.update(new_types)
            print(f"  NEW TYPES introduced: {list(new_types.keys())}")

        report = reconcile(batch_text, events)
        print(f"  reconciliation: missing {report['missing_ratio']:.2%}, extra {report['extra_ratio']:.2%}")
        if report["missing_ratio"] > 0.01 or report["extra_ratio"] > 0.01:
            print(f"    missing_top={report['missing_top']}")
            print(f"    extra_top={report['extra_top']}")

        stack = apply_events_to_stack(stack, events)
        print(f"  stack now: {[s['type'] + ':' + str(s['label']) for s in stack]}")

        tail_text = batch[-1][1][-TAIL_CHARS:]
        all_events.extend(events)

    return all_events, stack, known_types

