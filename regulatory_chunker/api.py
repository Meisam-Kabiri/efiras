"""
api.py
Unified Facade API & Repository Loader for Regulatory Chunker.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from .fetch_chunk import EurLexChunker
    from .tree_chunker import TreeChunker
    from .pdf_to_html import run as pdf_to_events
    from .pdf_html_builder import events_to_html
    from .graph_structure import GraphStructureBuilder
    from .buckets_config import BUCKET_A, BUCKET_B
except ImportError:
    from fetch_chunk import EurLexChunker
    from tree_chunker import TreeChunker
    from pdf_to_html import run as pdf_to_events
    from pdf_html_builder import events_to_html
    from graph_structure import GraphStructureBuilder
    from buckets_config import BUCKET_A, BUCKET_B


class RegulatoryChunker:
    """
    Unified Facade interface for chunking regulatory framework documents
    (EUR-Lex HTML and converted PDFs).
    """

    def __init__(self, anthropic_model: str = "claude-sonnet-5"):
        self.anthropic_model = anthropic_model
        self.graph_builder = GraphStructureBuilder()

    def chunk_html(self, doc_id: str, source: str, cache_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunks an HTML regulatory document (direct EUR-Lex HTML or local HTML file).
        """
        fetcher = EurLexChunker(
            doc_id=doc_id,
            source=source,
            cache_file=Path(cache_file) if cache_file else None,
            verbose=False,
        )
        fetcher._load_html()
        tree_parser = TreeChunker(doc_id=doc_id)
        return tree_parser.chunk(str(fetcher.soup))

    def chunk_pdf(self, doc_id: str, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Chunks a PDF regulatory document by extracting structural events via LLM,
        building structured HTML, and parsing with TreeChunker.
        """
        events, _, _ = pdf_to_events(
            pdf_path=pdf_path,
            doc_id=doc_id,
            model=self.anthropic_model,
        )
        html_content = events_to_html(events, doc_id=doc_id)
        tree_parser = TreeChunker(doc_id=doc_id)
        return tree_parser.chunk(html_content)

    def chunk(self, doc_id: str, source: str, fmt: str = "html") -> List[Dict[str, Any]]:
        """
        Unified chunking method.
        :param doc_id: Document identifier (e.g., 'GDPR', 'DORA', 'CSSF_18_698')
        :param source: URL or local file path
        :param fmt: 'html' or 'pdf'
        """
        if fmt.lower() == "html":
            return self.chunk_html(doc_id=doc_id, source=source)
        elif fmt.lower() == "pdf":
            return self.chunk_pdf(doc_id=doc_id, pdf_path=source)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'html' or 'pdf'.")

    def build_toc(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a hierarchical Table of Contents graph structure from flat chunks.
        """
        return self.graph_builder.build(chunks)


class RegulatoryRepository:
    """
    Exposes pre-built regulatory framework document chunks, status listings,
    and graph structures.
    """

    def __init__(self, chunks_dir: Optional[str] = None):
        if chunks_dir:
            self.chunks_dir = Path(chunks_dir)
        else:
            primary = Path(__file__).parent.parent / "data" / "regulatory_pipeline" / "chunks"
            fallback = Path(__file__).parent / "chunks_output"
            if primary.exists() and any(primary.glob("*_chunks.json")):
                self.chunks_dir = primary
            else:
                self.chunks_dir = fallback

    def list_documents(self) -> List[str]:
        """
        Returns a sorted list of all available regulatory document IDs on disk.
        """
        if not self.chunks_dir.exists():
            return []
        return sorted([f.stem.replace("_chunks", "").upper() for f in self.chunks_dir.glob("*_chunks.json")])

    def list_configured_documents(self, bucket: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Returns document IDs registered in buckets_config.py, separated by Bucket A (HTML) and Bucket B (PDF).
        """
        bucket_a_ids = sorted([doc[0] for doc in BUCKET_A])
        bucket_b_ids = sorted([doc[0] for doc in BUCKET_B if doc[0] != "DODD_FRANK_P2"])

        if bucket:
            b_upper = bucket.upper()
            if b_upper == "A":
                return {"A": bucket_a_ids}
            elif b_upper == "B":
                return {"B": bucket_b_ids}
            else:
                raise ValueError(f"Invalid bucket '{bucket}'. Use 'A' or 'B'.")
        return {"A": bucket_a_ids, "B": bucket_b_ids}

    def list_chunked_documents(self, bucket: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Returns document IDs that have been successfully chunked, separated by Bucket A (HTML) and Bucket B (PDF).
        """
        all_chunked = set(self.list_documents())
        configured = self.list_configured_documents(bucket=bucket)

        result = {}
        for b_name, b_ids in configured.items():
            result[b_name] = [doc_id for doc_id in b_ids if doc_id in all_chunked]
        return result

    def list_unchunked_documents(self, bucket: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Returns document IDs registered in buckets_config.py that have NOT yet been chunked into chunks_output/.
        """
        all_chunked = set(self.list_documents())
        configured = self.list_configured_documents(bucket=bucket)

        result = {}
        for b_name, b_ids in configured.items():
            result[b_name] = [doc_id for doc_id in b_ids if doc_id not in all_chunked]
        return result

    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Loads pre-built JSON chunks for a specific document ID (e.g. 'GDPR', 'CRR').
        """
        file_path = self.chunks_dir / f"{doc_id.lower()}_chunks.json"
        if not file_path.exists():
            raise FileNotFoundError(f"No pre-built chunks found for document '{doc_id}' at {file_path}")
        return json.loads(file_path.read_text(encoding="utf-8"))

    def get_all_chunks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads pre-built chunks for ALL available regulatory framework documents.
        """
        return {doc_id: self.get_chunks(doc_id) for doc_id in self.list_documents()}

    def get_toc(self, doc_id: str) -> Dict[str, Any]:
        """
        Builds and returns the hierarchical Table of Contents graph for a document ID.
        """
        chunks = self.get_chunks(doc_id)
        builder = GraphStructureBuilder()
        return builder.build(chunks)


if __name__ == "__main__":
    repo = RegulatoryRepository()
    print("Chunks Directory:", repo.chunks_dir)
    print("Chunked Documents:", repo.list_chunked_documents())
    print("Unchunked Documents:", repo.list_unchunked_documents())