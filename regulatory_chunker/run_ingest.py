#!/usr/bin/env python3
"""
run_ingest.py
Batch runner script to process all regulatory documents in BUCKET_A and BUCKET_B
and output chunks to chunks_output/.
"""

import json
from pathlib import Path

try:
    from .api import RegulatoryChunker
    from .buckets_config import BUCKET_A, BUCKET_B
except ImportError:
    from api import RegulatoryChunker
    from buckets_config import BUCKET_A, BUCKET_B

OUT_DIR = Path(__file__).parent / "chunks_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    chunker = RegulatoryChunker()

    print("=== Processing Bucket A (HTML Regulatory Frameworks) ===")
    for doc_id, name, url, fmt in BUCKET_A:
        out_file = OUT_DIR / f"{doc_id.lower()}_chunks.json"
        print(f"[{doc_id}] Processing {name}...")
        try:
            chunks = chunker.chunk(doc_id=doc_id, source=url, fmt=fmt)
            out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{doc_id}] Saved {len(chunks)} chunks -> {out_file}\n")
        except Exception as e:
            print(f"[{doc_id}] Error: {e}\n")

    print("=== Processing Bucket B (PDF Regulatory Frameworks) ===")
    for doc_id, name, url, fmt in BUCKET_B:
        if doc_id == "DODD_FRANK_P2":
            continue
        out_file = OUT_DIR / f"{doc_id.lower()}_chunks.json"
        pdf_cache = Path(__file__).parent / "flat_cache" / f"{doc_id.lower()}.pdf"
        if not pdf_cache.exists():
            print(f"[{doc_id}] PDF file {pdf_cache} not found, skipping.")
            continue
        print(f"[{doc_id}] Processing PDF {name}...")
        try:
            chunks = chunker.chunk(doc_id=doc_id, source=str(pdf_cache), fmt="pdf")
            out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{doc_id}] Saved {len(chunks)} chunks -> {out_file}\n")
        except Exception as e:
            print(f"[{doc_id}] Error: {e}\n")

    print("Done! All regulatory framework chunks are ready in chunks_output/.")


if __name__ == "__main__":
    main()
