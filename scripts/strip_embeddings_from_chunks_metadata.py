"""
One-off cleanup: chunks_metadata.json was built with a duplicate "embedding"
field per chunk (the vector already lives in faiss.index). This strips that
field in place without touching faiss.index or bm25_tokenized.pkl, since
those were never affected and the chunk order is unchanged.
"""

import json
import os
from pathlib import Path

CHUNKS_PATH = Path("data/indexes/chunks_metadata.json")


def main():
    print(f"Loading {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    had_embedding = sum(1 for c in chunks if "embedding" in c)
    print(f"{had_embedding} chunks contain an 'embedding' field to strip")

    for c in chunks:
        c.pop("embedding", None)

    tmp_path = CHUNKS_PATH.with_suffix(".json.tmp")
    print(f"Writing cleaned file to {tmp_path} ...")
    with open(tmp_path, "w") as f:
        json.dump(chunks, f)

    os.replace(tmp_path, CHUNKS_PATH)
    print(f"Replaced {CHUNKS_PATH} ({CHUNKS_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
