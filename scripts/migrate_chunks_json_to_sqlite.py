"""
One-off migration: convert the existing data/indexes/chunks_metadata.json
(chunk content + metadata, in FAISS/BM25 index order) into data/indexes/chunks.db
(SQLite + FTS5 BM25 index), so chunk content no longer has to be loaded into
RAM at startup and rank-bm25/bm25_tokenized.pkl are no longer needed.

faiss.index is untouched - chunk order is unchanged, so it stays aligned.
"""

import json
from pathlib import Path

from core.rag.search_service import SearchService

CHUNKS_JSON_PATH = Path("data/indexes/chunks_metadata.json")
DB_PATH = Path("data/indexes/chunks.db")


def main():
    print(f"Loading {CHUNKS_JSON_PATH} ...")
    with open(CHUNKS_JSON_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    search = SearchService(index_dir=str(DB_PATH.parent))
    search.chunks = chunks
    search._write_chunks_db(str(DB_PATH))

    print(f"✅ Wrote {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
