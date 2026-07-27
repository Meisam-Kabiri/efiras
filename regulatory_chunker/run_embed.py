#!/usr/bin/env python3
"""
run_embed.py
Batch script to generate OpenAI API embeddings (text-embedding-3-small)
for all pre-built regulatory document chunks.
"""

import sys
from pathlib import Path

try:
    from .embedder import RegulatoryEmbedder
except ImportError:
    from embedder import RegulatoryEmbedder


def main():
    print("=== Generating OpenAI API Embeddings for Regulatory Frameworks ===")
    embedder = RegulatoryEmbedder(model="text-embedding-3-small")
    
    summary = embedder.embed_all()
    print("\n=== OpenAI Embedding Batch Generation Complete ===")
    for doc_id, count in summary.items():
        print(f"[{doc_id}]: {count} chunks embedded")

    print(f"\nEmbeddings saved to: {embedder.output_dir}")


if __name__ == "__main__":
    main()
