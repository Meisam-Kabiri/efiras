"""
Regulatory Chunker Package
Modular parser, repository loader, and OpenAI embedder for EU and global regulatory frameworks.
"""

try:
    from .api import RegulatoryChunker, RegulatoryRepository
    from .embedder import RegulatoryEmbedder
    from .fetch_chunk import EurLexChunker
    from .tree_chunker import TreeChunker
    from .graph_structure import GraphStructureBuilder
    from .buckets_config import BUCKET_A, BUCKET_B, BUCKETS
except ImportError:
    from api import RegulatoryChunker, RegulatoryRepository
    from embedder import RegulatoryEmbedder
    from fetch_chunk import EurLexChunker
    from tree_chunker import TreeChunker
    from graph_structure import GraphStructureBuilder
    from buckets_config import BUCKET_A, BUCKET_B, BUCKETS

__all__ = [
    "RegulatoryChunker",
    "RegulatoryRepository",
    "RegulatoryEmbedder",
    "EurLexChunker",
    "TreeChunker",
    "GraphStructureBuilder",
    "BUCKET_A",
    "BUCKET_B",
    "BUCKETS",
]
