# Enhanced Financial Intelligence and Regulatory Analysis System

## Overview

This system processes PDF documents by extracting text, cleaning it, and making it searchable through a question-answering interface. It supports Azure OpenAI services and Azure AI Search for enterprise-grade scalability and performance. It consists of several components that work together to transform raw PDFs into structured, queryable data.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd efiras
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

3. **Environment configuration:**
   Create a `.env` file in the root directory with your API credentials:
   
   **For OpenAI:**
   ```bash
   GPT_API_KEY=your_openai_api_key_here
   ```
   
   **For Azure OpenAI:**
   ```bash
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   ```
   
   **For Azure AI Search (optional):**
   ```bash
   AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
   AZURE_SEARCH_API_KEY=your_azure_search_api_key_here
   ```

4. **Run the example:**
   ```bash
   python example.py
   ```

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
Provides question-answering capabilities with support for both OpenAI and Azure OpenAI:
- **Unified RAG System**: Single interface supporting both OpenAI and Azure OpenAI
- Creates embeddings from document chunks using either:
  - **Online embeddings**: OpenAI's text-embedding-3-large or Azure OpenAI embedding models
  - **Offline embeddings**: Local sentence-transformer models (e.g., all-mpnet-base-v2)
- Searches for relevant content using hybrid similarity matching with regulatory boosting
- Keyword term matching adds boost per matching term
- Regulatory number boosting for numbers like "517", "698"
- Regulatory reference boosting for citations like "Article 5", "Section 3.2"
- Generates answers using OpenAI's GPT models or Azure OpenAI deployments with regulatory-specific prompting
- Returns answers with source information including page numbers and hierarchical context

## Input and Output

### Input
- PDF documents
- Text-based or scanned PDFs

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
from src.rag.unified_rag import UnifiedRAGSystem
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

# 4. Build searchable knowledge base with various configurations
# Option A: Use OpenAI with local embeddings
rag = UnifiedRAGSystem(use_local_embeddings=True, use_azure=False)

# Option B: Use Azure OpenAI with local embeddings
# rag = UnifiedRAGSystem(use_local_embeddings=True, use_azure=True, model="gpt-35-turbo")

# Option C: Use Azure OpenAI with Azure AI Search backend
# rag = UnifiedRAGSystem(use_local_embeddings=True, use_azure=True, model="gpt-35-turbo", use_azure_search=True)

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

Source: Page 73, Chapter 6: Specific organisational arrangements, Sub-chapter 6.4:  Organisation of the function of UCI administration, Section 6, Sub-section 6.4.3.2
```

**Key Features Demonstrated:**
- ✅ **Precise extraction** of specific regulatory requirements from complex financial documents
- ✅ **Complete enumeration** of all required monitoring elements (6/6 found)
- ✅ **Professional formatting** with numbered lists suitable for compliance documentation
- ✅ **Accurate source attribution** with full regulatory hierarchy (Sub-section 6.4.3.2, Page 73)
- ✅ **Contextual understanding** of financial terminology (IFM, UCI, net asset value)

This example shows the system successfully processing a 96-page Luxembourg regulatory document and providing comprehensive, accurate answers for compliance and regulatory questions.

---

## Advanced Example: Basel III Regulatory Analysis

**Document:** Basel III - Capital Requirements Regulation (337 pages)

**Query:** "What are the requirements for using the Internal Assessment Approach for ABCP programmes?"

**System Response:**
```
The requirements for using the Internal Assessment Approach (IAA) for Asset-Backed Commercial Paper (ABCP) programmes are outlined in Article 259 of Title II, Chapter 5 of the Capital Requirements for Credit Risk. The following conditions must be met:

1. **Rated Positions**: The commercial paper issued from the ABCP programme must be rated positions (Article 259, paragraph 3(a)).

2. **Permission from Competent Authorities**: Institutions must obtain permission from competent authorities to use the IAA (Article 259, paragraph 3).

3. **Alternative to Unrated Positions**: As an alternative to point (b), for unrated positions in ABCP programmes, the institution may use the IAA if the competent authorities have permitted it to do so (Article 259, paragraph (c)).

4. **Regular Reviews**: Regular reviews of the internal assessment process and the quality of the internal assessments of the credit quality of the institution's exposures to an ABCP programme must be performed. These reviews can be conducted by internal or external auditors, an External Credit Assessment Institution (ECAI), or the institution's internal credit review or risk management function (Article 259, paragraph (g)).

5. **Independence of Review Functions**: If the institution's internal audit, credit review, or risk management functions perform the review, these functions must be independent of the ABCP programme business line, as well as the customer relationship (Article 259, paragraph (g)).

6. **Hierarchy of Methods**: For a rated position or a position in respect of which an inferred rating may be used, the Ratings Based Method set out in Article 261 must be used to calculate the risk-weighted exposure amount (Article 259, paragraph 1(a)).

Source: Basel III, Article 259, Page 161
```

**Advanced Features Demonstrated:**
- ✅ **Complex financial terminology** handling (ABCP, IAA, ECAI, risk-weighted exposure)
- ✅ **Multi-layered regulatory synthesis** from Basel III's intricate article structure
- ✅ **Professional regulatory formatting** with proper article citations and paragraph references
- ✅ **Comprehensive requirement analysis** covering operational, compliance, and methodological aspects
- ✅ **Perfect source attribution** with exact page and article references for regulatory compliance

This demonstrates the system's capability to handle sophisticated banking regulations and provide actionable guidance for financial institutions implementing Basel III requirements.

## Requirements

### For Document Processing:
- Python packages: PyMuPDF, pdfminer, unstructured
- Optional: Azure Document Intelligence credentials for advanced processing

### For Question Answering:
- **OpenAI**: API key (set as GPT_API_KEY in .env file) for answer generation and online embeddings
- **Azure OpenAI**: Endpoint and API key (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY) for enterprise deployment
- **Azure AI Search**: Endpoint and API key (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY) for scalable vector storage
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
