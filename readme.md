# Enhanced Financial Intelligence and Regulatory Analysis System

## Overview

This system processes PDF documents by extracting text, cleaning it, and making it searchable through a question-answering interface. It consists of several components that work together to transform raw PDFs into structured, queryable data.

## What It Does

### Document Processing
- **Extracts text** from PDF files using multiple processing engines
- **Cleans and structures** the extracted text by removing headers/footers and organizing content
- **Splits large documents** into manageable chunks while preserving context
- **Creates searchable databases** from processed documents

### Question Answering
- **Answers questions** about your documents using AI
- **Finds relevant information** by searching through document content
- **Provides source citations** showing where answers came from
- **Works with multiple documents** simultaneously

## System Components

### 1. Multi-Engine Document Processor
Processes PDF documents using different engines:
- **PyMuPDF**: Fast processing for text-based PDFs
- **Azure Document Intelligence**: Advanced OCR and table extraction
- **PDFMiner**: Complex layout analysis
- **Unstructured**: Multi-format document support

The system automatically selects the best engine for each document and falls back to alternatives if needed.

### 2. Block Processor
Cleans and structures extracted text:
- **Extracts Table of Contents (TOC)** from documents when available
- **Assigns hierarchical TOC titles** to each text block/chunk based on document structure
- Removes headers and footers based on positioning
- Identifies section titles and hierarchies using regex patterns
- Merges related text blocks (e.g., blocks ending with colons)
- **Enriches blocks with TOC-derived semantics** for better context
- Repairs paragraphs split across page breaks
- Cleans up formatting issues and normalizes text

### 3. Block Chunker
Splits large documents into smaller pieces:
- Tries to split by paragraphs first
- Falls back to sentences if needed
- Uses word-based splitting as last resort
- Preserves document structure and metadata

### 4. RAG (Retrieval-Augmented Generation) System
Provides question-answering capabilities:
- Creates embeddings from document chunks using either:
  - **Online embeddings**: OpenAI's text-embedding-3-large model
  - **Offline embeddings**: Local sentence-transformer models (e.g., all-mpnet-base-v2)
- Searches for relevant content using similarity matching
- Generates answers using OpenAI's GPT models
- Returns answers with source information

## Input and Output

### Input
- PDF documents (any size)
- Text-based or scanned PDFs
- Documents with tables and complex layouts

### Output
- Structured JSON files with processed text blocks including:
  - Cleaned text content
  - Hierarchical section headers and TOC titles
  - Page numbers and positioning metadata
  - Document structure information
- Searchable embeddings for fast retrieval
- Natural language answers to questions about documents
- Source citations with page numbers and confidence scores

## Usage Example

```python
from src.document_readers.base import DocumentProcessor, ProcessorConfig
from src.document_processing.block_processor import block_processor
from src.document_chunker.block_chunker import RegulatoryChunkingSystem
from src.rag.rag_simple import RAGSystem
from src.document_processing.manager import DocumentProcessorManager

# 1. Configure and process PDF document
config = ProcessorConfig(chunk_size=2000, extract_tables=True, ocr_fallback=True)
manager = DocumentProcessorManager(config)
raw_result = manager.process_document("document.pdf", preferred_processor="PYMUPDF")

# 2. Clean and structure the text
processor = block_processor()
processed_data = processor.process_and_chunk_blocks(raw_result)

# 3. Create manageable chunks
chunker = RegulatoryChunkingSystem(max_chunk_size=1500)
chunked_blocks = chunker.chunk_blocks(processed_data)

# 4. Build searchable knowledge base
rag = RAGSystem(use_local_embeddings=True)
rag.add_documents(chunked_blocks, cache_path="data", cache_file_name="embeddings")

# 5. Ask questions
result = rag.answer_with_sources("What are the main requirements?", top_k=3)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## Real-World Example

**Document:** CSSF 18/698 - Luxembourg Investment Fund Manager Regulations (96 pages)

**Query:** "What monitoring elements must IFM implement for central administration delegation?"

**System Response:**
```
The IFM (Investment Fund Manager) must implement its own control and monitoring 
system when delegating the central administration function. This system should 
cover at least the following elements:

1. Monitoring the time of delivery of the net asset value.
2. Monitoring the net asset value calculation errors.
3. Monitoring the non-compliance with the investment policy and restrictions.
4. Monitoring the transactions which were not accounted for within the usual time limits.
5. Controlling the fees and commissions to be borne by UCIs (Undertakings for Collective Investment).
6. Monitoring the reconciliation of the number of units in circulation.

These requirements are outlined in section 518 of the provided context.
```

**Key Features Demonstrated:**
- ✅ **Precise extraction** of specific regulatory requirements from complex financial documents
- ✅ **Complete enumeration** of all required monitoring elements (6/6 found)
- ✅ **Professional formatting** with numbered lists suitable for compliance documentation
- ✅ **Source attribution** referencing the exact regulatory section (518)
- ✅ **Contextual understanding** of financial terminology (IFM, UCI, net asset value)

This example shows the system successfully processing a 96-page Luxembourg regulatory document and providing comprehensive, accurate answers for compliance and regulatory questions.

## Requirements

### For Document Processing:
- Python packages: PyMuPDF, pdfminer, unstructured
- Optional: Azure Document Intelligence credentials for advanced processing

### For Question Answering:
- OpenAI API key (set as GPT_API_KEY in .env file) for answer generation and online embeddings
- sentence-transformers (for offline/local embeddings option)

## File Structure

The system processes documents through this workflow:
1. **Raw PDF** → Multi-engine processor → **Extracted text with metadata**
2. **Extracted text** → Block processor → **Cleaned and structured blocks with TOC hierarchy**
3. **Structured blocks** → Block chunker → **Manageable chunks preserving TOC context**
4. **Chunks** → RAG system → **Searchable knowledge base**

## Output Format

Processed documents are saved as JSON files containing:
- Document metadata (filename, page count, etc.)
- **Extracted Table of Contents structure** when available
- Cleaned text blocks with page numbers
- **Hierarchical section headers and TOC-derived titles** for each block
- Section headers and structure information
- Embeddings for similarity search (cached separately)

## Future Work

### Enhanced Document Processing

- **Table Extraction**: Improve extraction and preservation of financial tables and structured data
- **Multi-Document Analysis**: Enable comparative analysis across multiple documents simultaneously
- **Document Summarization**: Automated generation of executive summaries for lengthy financial documents

### Financial-Specific Features

- **Financial Entity Recognition**: Specialized identification of financial terms, regulations, and compliance references
- **Regulatory Citation Linking**: Automatic cross-referencing between related regulatory documents
- **API Integration**: RESTful API for integration with existing financial systems and workflows

These immediate enhancements would strengthen the core document processing capabilities while adding financial industry-specific value.