"""
chunks_to_markdown.py

Renders a list of chunks (as produced by EurLexChunker, or loaded from a
*_chunks.json file) into a single markdown file for fast manual review:
each chunk becomes a bold path line (e.g.
`**partIII/titleI/chapter1/section1/article93/paragraph6**`), a blank
line, its full text, then a horizontal rule separating it from the next
chunk.

Usage:  python chunks_to_markdown.py <chunks.json> <out.md>
"""

import json
import sys
from pathlib import Path


def _path_label(chunk):
    """Builds a compact path label from a chunk's `path` list of
    {"type", "label", "heading"} levels, e.g. 'partIII/titleI/article93/paragraph6'.
    Most levels have a short numeric/letter label ("93", "6"), so
    type+label reads naturally. The preamble level is the exception --
    its label is the word "Preamble" itself, which would otherwise render
    as "preamblePreamble" -- so skip the label when it's redundant with
    the type (case-insensitively)."""
    parts = []
    for level in chunk.get("path", []):
        t, l = level["type"], level["label"]
        parts.append(t if l.lower() == t.lower() else f"{t}{l}")
    return "/".join(parts)


def chunks_to_markdown(chunks, out_file):
    """
    Writes `chunks` to `out_file` as markdown: each chunk's path label
    (bold) on its own line, a blank line, then its text on its own line,
    then a `---` horizontal rule separating it from the next chunk.
    Streams straight to disk (one write per chunk) rather than building
    the whole file in memory first, so it stays cheap even for documents
    with thousands of chunks.
    """
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            text = chunk.get("text")
            if not text:
                continue
            f.write(f"**{_path_label(chunk)}**\n\n")
            f.write(text)
            f.write("\n\n---\n\n")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <chunks.json> <out.md>", file=sys.stderr)
        sys.exit(1)

    chunks_json, out_md = sys.argv[1], sys.argv[2]
    chunks = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
    out_path = chunks_to_markdown(chunks, out_md)
    print(f"wrote {len(chunks)} chunks -> {out_path}")
