"""
build_graph_structure.py

Walks every chunks_output/*_chunks.json file, builds one unified structure
across the whole corpus via GraphStructureBuilder, and writes both a
machine-readable JSON structure (for RAG/agentic use) and a readable
indented outline (for sanity-checking by eye).

Usage:  python scripts/build_graph_structure.py   (run from the repo root)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_structure import GraphStructureBuilder
from tree_to_markdown import corpus_to_markdown

if __name__ == "__main__":
    builder = GraphStructureBuilder()
    corpus = builder.build_corpus("chunks_output")

    out_path = Path("chunks_output/corpus_structure.json")
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    review_path = corpus_to_markdown(corpus, "chunks_output/corpus_structure_review.md")

    for doc_id, tree in corpus.items():
        print(f"{doc_id}: {len(tree['children'])} top-level nodes")
    print(f"\nwrote unified corpus structure ({len(corpus)} documents) -> {out_path}")
    print(f"wrote review outline -> {review_path}")
