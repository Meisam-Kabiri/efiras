"""
graph_structure.py

Builds a navigable graph/structure (a rich table of contents, not just a
flat chunk list) from a document's chunks -- every chunk from
EurLexChunker/TreeChunker already carries its own `path` (the ordered list
of {type, label, heading} ancestors), but that path is repeated on every
single chunk rather than existing once. This class collapses shared path
prefixes into one shared node per structural level (e.g. one "Chapter IV >
Section 1" node, not one per paragraph underneath it), so the result is an
actual walkable structure: list a node's children, jump to any subtree,
find a node's own chunks.

Useful for RAG/agentic use beyond flat vector search -- e.g. pulling a
retrieved chunk's ancestor headings for context, listing sibling chunks
under the same article, or letting an LLM navigate a document's structure
(or the whole corpus's) without everything in context at once.

This module defines the class only -- see scripts/build_graph_structure.py
for the runner.
"""

import json
from pathlib import Path


class GraphStructureBuilder:
    """
    `build(chunks)` turns one document's flat chunk list into a structure.
    `build_from_file(chunk_file)` does the same, reading the chunks from a
    JSON file on disk (as produced by EurLexChunker/TreeChunker's `run`/
    `chunk` + a JSON dump, e.g. chunks_output/gdpr_chunks.json).
    `build_corpus(chunks_dir)` walks an entire folder of chunk files and
    returns one unified structure -- {doc_id: structure, ...} -- across
    every document found.

    Each node: {"type", "label", "heading", "children": [...],
    "chunks": [...]}. `children` are this node's immediate structural
    sub-levels (also nodes); `chunks` are the actual chunks whose path
    resolves to exactly this node (a short preview only, not full text --
    look up `chunk_id` in the source file for the complete chunk).
    """

    @staticmethod
    def _make_node(level):
        return {
            "type": level["type"],
            "label": level["label"],
            "heading": level.get("heading"),
            "children": [],
            "chunks": [],
        }

    def build(self, chunks):
        root = {"type": None, "label": None, "heading": None, "children": [], "chunks": []}
        # keyed by (id(parent_node), type, label) -- id(parent_node) keeps
        # e.g. two different "Chapter 1" nodes under two different Parts
        # from colliding into one, since they're genuinely different nodes
        lookup = {}

        for chunk in chunks:
            node = root
            for level in chunk["path"]:
                key = (id(node), level["type"], level["label"])
                child = lookup.get(key)
                if child is None:
                    child = self._make_node(level)
                    node["children"].append(child)
                    lookup[key] = child
                elif level.get("heading") and not child["heading"]:
                    # same node reached via a different chunk whose path
                    # entry happened to carry the heading this time
                    child["heading"] = level["heading"]
                node = child
            node["chunks"].append({
                "chunk_id": chunk["chunk_id"],
                "citation": chunk.get("citation"),
                "text_preview": (chunk.get("text") or "")[:200],
            })

        return root

    def build_from_file(self, chunk_file):
        chunks = json.loads(Path(chunk_file).read_text(encoding="utf-8"))
        return self.build(chunks)

    def build_corpus(self, chunks_dir, pattern="*_chunks.json"):
        corpus = {}
        for path in sorted(Path(chunks_dir).glob(pattern)):
            chunks = json.loads(path.read_text(encoding="utf-8"))
            if not chunks:
                continue
            doc_id = chunks[0]["doc_id"]
            corpus[doc_id] = self.build(chunks)
        return corpus
