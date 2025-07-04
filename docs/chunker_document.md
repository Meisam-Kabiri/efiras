# RegulatoryChunkingSystem

`RegulatoryChunkingSystem` is a Python class for segmenting legal and regulatory documents into structured, meaningful chunks using both rule-based and semantic methods. It's designed to support tasks like Retrieval-Augmented Generation (RAG), legal compliance checks, and knowledge extraction.

## Features

* **Hierarchical Chunking**: Detects structural elements (e.g., *Article 1*, *Section 2.3*) using regular expressions and splits content accordingly.
* **Semantic Grouping**: Uses [Sentence Transformers](https://www.sbert.net/) and KMeans clustering to merge smaller semantically similar chunks.
* **Overlap Handling**: Adds configurable overlap between consecutive chunks to ensure contextual continuity.
* **Metadata Extraction**: Identifies legal references (e.g., "12 U.S.C. § 1841") and cross-references (e.g., "see Section 3.1").
* **Chunk Classification**: Tags chunks as `definition`, `requirement`, `procedure`, `penalty`, or `general` based on keyword patterns.
* **Validation & Cleanup**: Filters out undersized chunks, checks citation integrity, and prepares content for indexing.

## Output Format

Each chunk is returned as a dictionary:

```python
{
  "chunk_id": "chunk_12_24",
  "content": "...",
  "section_hierarchy": ["Article 1", "Section 1.2"],
  "chunk_type": "requirement",
  "legal_citations": ["12 U.S.C. § 1841"],
  "cross_references": ["see Section 3.1"],
  "start_position": 12,
  "end_position": 24,
  "overlap_with": ["chunk_10_12"]
}
```

## Usage Example

```python
from regulatory_chunker import RegulatoryChunkingSystem
chunker = RegulatoryChunkingSystem()
chunks = chunker.process_document(document_text)
```

## Dependencies

* `sentence-transformers`
* `scikit-learn`
* `numpy`

Install with:

```bash
pip install sentence-transformers scikit-learn numpy
```

## Applications

* Preprocessing for vector databases in RAG systems
* Automated compliance monitoring and reporting
* Extraction and analysis of legal duties, penalties, and definitions

## Future Improvements

* Cross-reference resolution across documents
* Multilingual document support
* Integration with legal ontologies or knowledge graphs
