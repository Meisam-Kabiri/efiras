# Regulatory Chunking System - Technical Documentation

## System Overview

The Regulatory Chunking System is a specialized document chunking framework designed for regulatory and financial documents. It processes structured document blocks (from the document processing pipeline) and creates optimally-sized chunks that preserve document metadata, hierarchical structure, and regulatory context for downstream RAG operations.

## Key Features

### Regulatory-Optimized Design
- **Metadata Preservation**: Maintains all original block metadata (page numbers, headers, TOC information)
- **Size-Based Chunking**: Intelligent splitting based on configurable size constraints
- **Overlap Strategy**: Configurable overlap between chunks to prevent context loss
- **Semantic Awareness**: Uses sentence transformers for semantic understanding (prepared for future enhancements)

### Integration with Document Pipeline
- **Input**: Processed document blocks from block_processor
- **Output**: Optimally-sized chunks ready for embedding and RAG
- **Automatic Saving**: Saves chunked results to structured JSON files
- **Filename Preservation**: Maintains original document naming in output files

## Core Configuration

### Initialization Parameters

```python
RegulatoryChunkingSystem(
    min_chunk_size=100,      # Minimum characters per chunk
    max_chunk_size=500,      # Maximum characters per chunk  
    overlap_percentage=0.15, # 15% overlap between chunks
    semantic_model="all-MiniLM-L6-v2"  # Sentence transformer model
)
```

### Size Strategy
- **Minimum Size**: 100 characters (filters out very small blocks)
- **Maximum Size**: 500 characters (ensures manageable chunk sizes)
- **Overlap**: 15% overlap between consecutive chunks (75 characters for 500-char chunks)

## Core Methods & Functionality

### Document Chunking (`chunk_blocks`)
**Purpose**: Process document blocks into optimally-sized chunks while preserving all metadata

**Input Format**:
```python
pdf_content = {
    'filename_without_ext': 'document_name',
    'blocks': [
        {
            'text': 'Document text content...',
            'page': 1,
            'enriched_headers': 'Section > Subsection',
            'headers': ['Section', 'Subsection'],
            'block_type': 'paragraph',
            # ... other metadata
        }
    ]
}
```

**Processing Logic**:
1. **Size Check**: Examine each block's text length
2. **Direct Pass-through**: Blocks ≤ max_chunk_size remain unchanged
3. **Size Filtering**: Skip blocks < min_chunk_size
4. **Large Block Splitting**: Split oversized blocks using overlap strategy
5. **Metadata Preservation**: Copy ALL original metadata to each chunk
6. **Chunk Identification**: Add `chunk_id` to split blocks (0, 1, 2, ...)

**Output**: List of chunks with preserved metadata plus chunk identifiers

### Large Text Splitting (`_split_large_text`)
**Purpose**: Split oversized text blocks into overlapping chunks

**Algorithm**:
```python
# Sliding window with overlap
start = 0
while start < len(text):
    end = min(start + max_chunk_size, len(text))
    chunk = text[start:end]
    
    # Next chunk starts with overlap
    start = end - overlap_size
```

**Overlap Calculation**:
- **Overlap Size**: `max_chunk_size × overlap_percentage`
- **Example**: 500 chars × 15% = 75 character overlap
- **Benefit**: Prevents context loss at chunk boundaries

### Automatic File Management
**Purpose**: Save chunked results with consistent naming and structure

**Output Path Pattern**: `data_processed/{original_filename}_chunked_blocks.json`

**File Structure**:
```json
[
    {
        "text": "Chunk text content...",
        "page": 1,
        "enriched_headers": "Section > Subsection",
        "headers": ["Section", "Subsection"],
        "block_type": "paragraph",
        "chunk_id": 0
    }
]
```

## Integration with Document Processing Pipeline

### Input Source
```python
# From document processing pipeline
manager = DocumentProcessorManager()
raw_result = manager.process_document(pdf_path)
processed_data = block_processor.process_and_chunk_blocks(raw_result)

# Input to chunking system
chunker = RegulatoryChunkingSystem(max_chunk_size=1500)
chunks = chunker.chunk_blocks(processed_data)
```

### Output to RAG System
```python
# Output to RAG system
rag = UnifiedRAGSystem()
rag.add_documents(chunks, cache_path="data_processed", cache_file_name="embeddings")
```

## Configuration Examples

### Development Setup (Small Chunks)
```python
chunker = RegulatoryChunkingSystem(
    min_chunk_size=50,
    max_chunk_size=300,
    overlap_percentage=0.10
)
```

### Production Setup (Balanced)
```python
chunker = RegulatoryChunkingSystem(
    min_chunk_size=100,
    max_chunk_size=1500,
    overlap_percentage=0.15
)
```

### Large Document Setup
```python
chunker = RegulatoryChunkingSystem(
    min_chunk_size=200,
    max_chunk_size=2000,
    overlap_percentage=0.20
)
```

## Metadata Preservation Strategy

### Original Block Metadata (Preserved)
- **`text`**: Updated with chunk text
- **`page`**: Source page number
- **`enriched_headers`**: Hierarchical TOC path
- **`headers`**: List of section headers
- **`block_type`**: Type classification (paragraph, list, etc.)
- **All other fields**: Preserved exactly as received

### Added Metadata
- **`chunk_id`**: Sequential identifier for split blocks (0, 1, 2, ...)

### Benefits for RAG System
1. **Source Attribution**: Page numbers enable precise citation
2. **Context Preservation**: Headers maintain document structure
3. **Hierarchical Navigation**: TOC paths enable section-aware retrieval
4. **Split Tracking**: Chunk IDs help reconstruct original blocks if needed

## Performance Characteristics

### Processing Speed
- **Small Documents** (<100 blocks): Instant processing
- **Medium Documents** (100-1000 blocks): <1 second
- **Large Documents** (1000+ blocks): 1-5 seconds

### Memory Usage
- **Minimal Overhead**: In-place processing with metadata copying
- **Semantic Model**: ~100MB for sentence transformer (loaded once)
- **Output Size**: Roughly 1.2-1.5x input size due to overlap

### Chunking Efficiency
- **No Loss**: Every character from qualifying blocks is preserved
- **Optimal Sizing**: Consistent chunk sizes improve embedding quality
- **Context Continuity**: Overlap prevents boundary-related context loss

## Error Handling & Edge Cases

### Empty or Invalid Blocks
- **Empty Text**: Skipped automatically
- **Missing Text Key**: Handled gracefully with empty string default
- **Invalid Metadata**: Preserved as-is, no validation errors

### Size Edge Cases
- **Text < min_chunk_size**: Filtered out completely
- **Text = max_chunk_size**: Passed through unchanged
- **Text slightly > max_chunk_size**: Split with minimal overhead

### File System Operations
- **Directory Creation**: Automatically creates `data_processed/` if missing
- **File Overwrite**: Overwrites existing chunk files with same name
- **Encoding**: UTF-8 encoding for international character support

## Future Enhancement Opportunities

### Semantic Chunking (Prepared)
- **Sentence Transformer**: Already loaded for future semantic chunking
- **Clustering Support**: KMeans clustering imported for semantic grouping
- **Similarity Analysis**: Cosine similarity tools available

### Advanced Regulatory Features
- **Section-Aware Splitting**: Split at natural section boundaries
- **Citation Preservation**: Ensure regulatory references stay within chunks
- **Table Handling**: Special processing for regulatory tables

### Performance Optimizations
- **Lazy Loading**: Load semantic model only when needed
- **Batch Processing**: Process multiple documents in single operation
- **Memory Streaming**: Handle very large documents without memory limits

## Usage Examples

### Basic Processing
```python
from src.document_chunker.block_chunker import RegulatoryChunkingSystem

# Initialize chunker
chunker = RegulatoryChunkingSystem(max_chunk_size=1500)

# Process document blocks
chunks = chunker.chunk_blocks(processed_document_data)

print(f"Created {len(chunks)} chunks from document")
```

### Custom Configuration
```python
# High-precision chunking for complex documents
chunker = RegulatoryChunkingSystem(
    min_chunk_size=200,
    max_chunk_size=1000,
    overlap_percentage=0.25,  # 25% overlap for complex content
    semantic_model="all-mpnet-base-v2"  # Higher quality model
)

chunks = chunker.chunk_blocks(document_data)
```

### Pipeline Integration
```python
# Full document processing pipeline
from src.document_processing.manager import DocumentProcessorManager
from src.document_processing.block_processor import block_processor
from src.document_chunker.block_chunker import RegulatoryChunkingSystem

# Process document
manager = DocumentProcessorManager()
raw_result = manager.process_document("document.pdf")
processed_data = block_processor().process_and_chunk_blocks(raw_result)

# Chunk for RAG
chunker = RegulatoryChunkingSystem(max_chunk_size=1500)
chunks = chunker.chunk_blocks(processed_data)

# Results automatically saved to data_processed/
```

This regulatory chunking system provides the essential bridge between structured document processing and effective RAG operations, ensuring that regulatory content is optimally prepared for semantic search and question-answering while preserving all critical metadata and context.