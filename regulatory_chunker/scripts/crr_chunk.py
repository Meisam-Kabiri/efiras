"""
crr_chunk.py

Downloads the CRR (Capital Requirements Regulation) from EUR-Lex, caches
the HTML locally, chunks it via EurLexChunker into crr_chunks.json.

Usage:  python scripts/crr_chunk.py   (run from the repo root)
Needs:  pip install requests beautifulsoup4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_chunk import EurLexChunker

DOC_ID = "CRR"
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02013R0575-20260101"
HTML_FILE = "html_cache/crr.html"
OUT_FILE = "chunks_output/crr_chunks.json"

if __name__ == "__main__":
    chunker = EurLexChunker(doc_id=DOC_ID, source=URL, cache_file=HTML_FILE)
    chunker.run(out_file=OUT_FILE)
