"""
gdpr_chunk.py

Downloads the GDPR (General Data Protection Regulation) from EUR-Lex,
caches the HTML locally (skips the download if the cache already exists),
chunks it via EurLexChunker into gdpr_chunks.json, and renders a
gdpr_review.md alongside it so the chunking can be sanity-checked by eye.

Usage:  python scripts/gdpr_chunk.py   (run from the repo root)
Needs:  pip install requests beautifulsoup4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_chunk import EurLexChunker
from chunks_to_markdown import chunks_to_markdown

DOC_ID = "GDPR"
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
HTML_FILE = "html_cache/gdpr.html"
OUT_FILE = "chunks_output/gdpr_chunks.json"
REVIEW_FILE = "chunks_output/gdpr_review.md"

if __name__ == "__main__":
    chunker = EurLexChunker(doc_id=DOC_ID, source=URL, cache_file=HTML_FILE)
    chunks = chunker.run(out_file=OUT_FILE)

    review_path = chunks_to_markdown(chunks, REVIEW_FILE)
    print(f"wrote review markdown -> {review_path}")
