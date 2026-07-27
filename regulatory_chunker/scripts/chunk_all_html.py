"""
chunk_all_html.py

Chunks every HTML document currently available, in one pass:

  - Bucket A (buckets_config.BUCKET_A): real EUR-Lex HTML, downloaded/
    cached and chunked via EurLexChunker. Entries whose format isn't
    "html" (e.g. SOLVENCY2_L2, which EUR-Lex only serves as a JS shell on
    the HTML endpoint -- see buckets_config.py's note) are skipped, not
    guessed at.
  - Bucket B documents already converted to HTML by
    scripts/run_bucket_b_pipeline.py (reads its DOCS list directly, so
    doc_ids/paths always match -- no filename-guessing): chunked via
    TreeChunker, with levels discovered from the actual id prefixes
    present in that document's HTML.

Every document's chunks + review markdown are written to chunks_output/
and skipped if already there, so this is safe to rerun any time new
documents get added to either bucket -- it only does work for whatever's
actually new. No LLM calls, no cost -- HTML downloads/parsing only.

Usage:  python scripts/chunk_all_html.py   (run from the repo root)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buckets_config import BUCKET_A
from fetch_chunk import EurLexChunker
from tree_chunker import TreeChunker
from chunks_to_markdown import chunks_to_markdown
from run_bucket_b_pipeline import DOCS as BUCKET_B_DOCS

CHUNKS_DIR = Path("chunks_output")
HTML_DIR = Path("html_cache")


def chunk_bucket_a():
    for doc_id, name, url, fmt in BUCKET_A:
        if fmt != "html":
            print(f"skip {doc_id} ({name}): format={fmt}, not plain EUR-Lex HTML")
            continue

        out_file = CHUNKS_DIR / f"{doc_id.lower()}_chunks.json"
        review_file = CHUNKS_DIR / f"{doc_id.lower()}_review.md"
        if out_file.exists():
            print(f"skip {doc_id}: {out_file} already exists")
            continue

        html_file = HTML_DIR / f"{doc_id.lower()}.html"
        print(f"\nchunking {doc_id} ({name})")
        chunker = EurLexChunker(doc_id=doc_id, source=url, cache_file=str(html_file), verbose=False)
        chunks = chunker.run(out_file=str(out_file))
        chunks_to_markdown(chunks, str(review_file))
        print(f"  {len(chunks)} chunks -> {out_file}")


def chunk_bucket_b():
    for _pdf_path, doc_id, html_out in BUCKET_B_DOCS:
        html_file = Path(html_out)
        if not html_file.exists():
            print(f"skip {doc_id}: {html_file} doesn't exist yet (run scripts/run_bucket_b_pipeline.py first)")
            continue

        out_file = CHUNKS_DIR / f"{doc_id.lower()}_chunks.json"
        review_file = CHUNKS_DIR / f"{doc_id.lower()}_review.md"
        if out_file.exists():
            print(f"skip {doc_id}: {out_file} already exists")
            continue

        html = html_file.read_text(encoding="utf-8")
        # levels discovered from this document's own id prefixes, same
        # approach used to validate the CSSF run earlier -- Bucket B HTML
        # is LLM-generated per document, so its type vocabulary isn't
        # known ahead of time the way EurLexChunker's DEFAULT_LEVELS is
        prefixes = set(re.findall(r'id="([a-z0-9]+)_', html))
        levels = {p: p for p in prefixes}

        print(f"\nchunking {doc_id} (Bucket B: {html_file})")
        chunker = TreeChunker(doc_id=doc_id, levels=levels)
        chunks = chunker.chunk(html)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        chunks_to_markdown(chunks, str(review_file))
        print(f"  {len(chunks)} chunks -> {out_file}")


if __name__ == "__main__":
    chunk_bucket_a()
    chunk_bucket_b()
