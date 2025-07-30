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

## Available Regulatory Documents

The system comes pre-loaded with **24 major financial and regulatory documents** that you can immediately query:

### European Union and Luxembourg Regulations (23 documents)
- **Luxembourg CSSF 18/698** - Investment Fund Manager regulations
- **Alternative Investment Fund Managers Directive (AIFMD)** + Level 2 Regulations
- **Basel II Framework** (2006) and **Basel III Framework** - International banking regulations
- **Capital Requirements Directive V (CRD V)** and **Capital Requirements Regulation (CRR)**
- **Dodd-Frank Wall Street Reform** (2 versions) - US financial reform legislation
- **European Market Infrastructure Regulation (EMIR)**
- **European Union Taxonomy Regulation** - Sustainable finance classification
- **Anti-Money Laundering Directives**: 4th (4AMLD) and 5th (5AMLD) versions
- **Financial Action Task Force (FATF) Recommendations 2012**
- **General Data Protection Regulation (GDPR)** - EU data protection law
- **Markets in Financial Instruments**: Directive II (MiFID II) and Regulation (MiFIR)
- **Payment Services Directive 2 (PSD2)** - European payment services regulation
- **Securities Financing Transactions Regulation (SFTR)**
- **Solvency II Directive** + Level 2 Regulations - Insurance sector prudential regulation
- **Sustainable Finance Disclosure Regulation (SFDR)** - ESG disclosure requirements
- **Undertakings for Collective Investment in Transferable Securities (UCITS)**


### Ready-to-Query Knowledge Base
All documents are **fully processed and embedded**, meaning you can immediately ask questions like:
- *"What are the reporting requirements under MiFID II?"*
- *"How does Basel III define Tier 1 capital?"*
- *"What are the data protection obligations under GDPR for financial institutions?"*
- *"What monitoring elements must IFM implement for central administration delegation according to CSSF 18/698?"*

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

### 4. EmbeddingService - Dual-Mode Embedding Generation
Advanced embedding service supporting both local and online models:

**Local Embeddings (Default):**
- **BAAI/bge-large-en-v1.5**: High-quality multilingual embeddings optimized for retrieval
- **Sentence Transformers**: Local processing with GPU acceleration (`device='cuda'`)
- **Cached Processing**: Automatic caching of embeddings to avoid recomputation
- **Offline Operation**: No API calls required, complete privacy

**Online Embeddings (Optional):**
- **OpenAI**: `text-embedding-3-large` - Latest OpenAI embedding model
- **Azure OpenAI**: Enterprise-grade embedding deployments
- **API Integration**: Seamless switching between OpenAI and Azure OpenAI
- **Environment Configuration**: Automatic credential loading from `.env`

**Key Features:**
- **Header Enrichment**: Automatically enriches text with document hierarchy and filename
- **Model Flexibility**: Easy switching between local and online models
- **Batch Processing**: Efficient processing of multiple documents
- **Memory Management**: Automatic GPU memory cleanup after processing

### 5. SearchService - Advanced Hybrid Search System  
Sophisticated multi-modal search combining semantic and keyword approaches:

**Hybrid Search Architecture:**
- **BM25 Keyword Search**: Traditional term-frequency based search using `rank_bm25`
- **Semantic Vector Search**: Dense retrieval using FAISS indexes with cosine similarity
- **Reciprocal Rank Fusion (RRF)**: Combines results from both methods for superior relevance

**Advanced Indexing:**
- **FAISS Vector Index**: Optimized for different dataset sizes:
  - Small datasets (<1K): Flat index for exact search
  - Medium datasets (1K-10K): HNSW for balanced speed/accuracy
  - Large datasets (>10K): IVF with quantization for scalability
- **Whoosh Text Index**: Full-text search with advanced tokenization
- **Intelligent Tokenization**: Preserves regulatory terms (e.g., "non-compliance")

**Cross-Encoder Reranking:**
- **BAAI/bge-reranker-large**: State-of-the-art reranking model
- **Query-Document Scoring**: Precise relevance assessment between query and chunks
- **Final Reordering**: Re-ranks hybrid search results for optimal precision

**Search Workflow:**
```
Query → [BM25 Search] + [Semantic Search] → RRF Fusion → Cross-Encoder Reranking → Final Results
```

**Performance Features:**
- **Async Support**: Non-blocking search operations
- **Result Caching**: Speeds up repeated queries
- **Index Persistence**: Save/load indexes for faster startup
- **Regulatory Boosting**: Special handling for regulatory citations and numbers

### 6. RAG Generator - Intelligent Answer Generation
Combines search results with large language models for comprehensive answers:
- **Context Assembly**: Intelligently combines multiple search results
- **Source Attribution**: Provides exact page numbers and regulatory hierarchy
- **Regulatory Formatting**: Professional formatting suitable for compliance documentation
- **Multi-Model Support**: Works with OpenAI, Azure OpenAI, and local models

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
import json
from pathlib import Path
from rag.search_service import SearchService
from rag.embedding_service import EmbeddingService
from rag.rag_generator import RAGGenerator
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem

# Example 1: Generate embeddings for all PDFs in a directory
def create_embeddings_for_directory(pdf_directory_path):
    path = Path(pdf_directory_path)
    pdf_files = path.glob("*.pdf")
    embedding_service = EmbeddingService(device='cuda')  # Use GPU if available
    
    for pdf_file in pdf_files:
        # Check if embeddings already exist
        embedding_dir = Path("data_processed")
        if list(embedding_dir.glob(f"{pdf_file.stem}*embd*.json")):
            print(f"Embeddings exist for {pdf_file.name}")
            continue
            
        print(f"Processing {pdf_file.name}...")
        
        # 1. Extract text from PDF
        reader = PyMuPDFProcessor()
        raw_blocks = reader.extract_blocks(pdf_file)
        
        # 2. Clean and structure the text
        processor = block_processor(raw_blocks)
        processed_blocks = processor.process_blocks()
        
        # 3. Create manageable chunks
        chunker = RegulatoryChunkingSystem(processed_blocks)
        chunks_doc = chunker.chunk_blocks()
        
        # 4. Generate embeddings
        embeddings_data = embedding_service.embed_all_chunks(chunks_doc)

# Example 2: Complete RAG system with all embeddings
def setup_rag_system():
    # Load all existing embeddings from processed directory
    embedding_dir = Path("data_processed")
    embedding_files = list(embedding_dir.glob("*embds_local_BAAI_bge-large-en-v1_5*.json"))
    
    document_list = []
    for file in embedding_files:
        with open(file, 'r') as f:
            document_list.append(json.load(f))
    
    print(f"Loaded {len(document_list)} documents")
    total_chunks = sum(len(doc["embeddings"]) for doc in document_list)
    print(f"Total chunks: {total_chunks}")
    
    # Initialize RAG system components
    embedding_service = EmbeddingService()
    search_service = SearchService(index_dir="indexes")
    rag_generator = RAGGenerator()
    
    # Setup search service with indexes
    if not search_service.load_indexes():
        print("Building new search indexes...")
        search_service.build_indexes(document_list)
        search_service.save_indexes()
    else:
        print("Search indexes loaded successfully!")
        search_service.set_chunks(document_list)
    
    # Process a query
    query = "What are the main requirements for risk management?"
    print(f"\nQuery: {query}")
    
    # Get query embedding and search
    query_embedding = embedding_service.embed_text(query)
    relevant_chunks = search_service.search_documents(query, query_embedding, top_k=12)
    
    print(f"Found {len(relevant_chunks)} relevant chunks")
    
    # Generate answer using RAG
    answer = rag_generator.answer_query(query, relevant_chunks)
    print(f"Answer: {answer}")
    
    return search_service, rag_generator

# Usage
create_embeddings_for_directory("data/regulatory_documents/eu")
search_service, rag_system = setup_rag_system()
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

## Embedding Training Pipeline

The EFIRAS system includes a comprehensive training pipeline for fine-tuning embedding models specifically for regulatory and financial documents. This specialized training improves retrieval accuracy and domain-specific understanding.

### Training Overview

The training pipeline consists of two main components:
1. **Dataset Creation**: Generates training pairs from processed regulatory documents
2. **Model Fine-tuning**: Trains sentence transformers with contrastive learning for improved regulatory text understanding

### Quick Start Training

```bash
# Interactive training pipeline (recommended)
python train_embeddings.py

# Choose from menu:
# 1. Create MNRL dataset with hierarchy preservation
# 2. Fine-tune embedding model with batch-aware training  
# 3. Full pipeline (dataset creation + fine-tuning)
```

### Training Pipeline Components

#### 1. Dataset Preparation
Creates training datasets using the **MNRL (Multiple Negatives Ranking Loss)** approach:

```bash
# Create dataset only
python -m src.training.embedding_dataset_preparation
```

**Features:**
- **Hierarchical Awareness**: Preserves document structure and TOC relationships
- **Fixed-size Batches**: Generates consistent batch sizes (default: 8) for stable training
- **Diverse Sampling**: Creates similar/dissimilar pairs across different document sections
- **Regulatory Focus**: Optimized for financial and regulatory document characteristics

**Output:** Training datasets saved to `training_data/mnrl_fixed_batch_dataset_*.json`
## Embedding Training Pipeline

The EFIRAS system includes a comprehensive training pipeline for fine-tuning embedding models specifically for regulatory and financial documents. This specialized training improves retrieval accuracy and domain-specific understanding.

### Training Overview

The training pipeline uses **contrastive learning** with explicit positive and negative pairs:
1. **Dataset Creation**: Generates positive/negative training pairs from processed regulatory documents
2. **Model Fine-tuning**: Trains sentence transformers with contrastive loss for improved regulatory text understanding

### Quick Start Training

```bash
# Run the complete contrastive training pipeline
python train_embeddings.py
```

### Training Pipeline Architecture

#### 1. Contrastive Pair Creation
Creates high-quality positive and negative pairs using semantic similarity:

```bash
# Create contrastive pairs automatically
python -m src.training.embedding_dataset_preparation
```

**Positive Pairs:** Sentences from the same chunk (naturally related content)  
**Negative Pairs:** Sentences from randomly shuffled distant chunks (unrelated content)

**Output:** Balanced contrastive pairs with explicit labels (1.0 for positive, 0.0 for negative)

#### 2. Contrastive Model Fine-tuning
Fine-tunes sentence transformer models using contrastive loss:

```bash
# Fine-tune with contrastive learning
python -m src.training.embedding_fine_tuning
```

**Training Configuration:**
- **Base Model**: `sentence-transformers/all-mpnet-base-v2` (proven for regulatory text)
- **Loss Function**: Contrastive Loss (much more stable than MNRL)
- **Epochs**: 3 (optimal for regulatory domain)
- **Batch Size**: 16 (efficient GPU utilization)
- **Learning Rate**: 2e-5 with warmup
- **Data Split**: 80/20 train/validation

**Output:** Fine-tuned models saved to `models/efiras_contrastive_embeddings/`

### Why Contrastive Learning Over MNRL

**MNRL Problems (Why We Switched):**
- Unstable training with regulatory documents
- Required complex batch structure preservation
- Poor convergence on domain-specific text
- Inconsistent similarity learning

**Contrastive Learning Benefits:**
- **Explicit Supervision**: Clear positive/negative labels improve learning
- **Stable Training**: Contrastive loss is more robust for regulatory text
- **Better Semantic Understanding**: Learns to distinguish relevant vs. irrelevant content
- **Proven Performance**: Superior results on financial document retrieval

### Training Data Flow

```
Regulatory Documents → Chunk Extraction → Positive/Negative Pairing → Contrastive Training → Specialized Embeddings
     (processed)          (semantic)         (explicit labels)         (stable loss)        (domain-tuned)
```

### Integration with RAG System

Fine-tuned contrastive models integrate seamlessly:

```python
# Use contrastive fine-tuned model
rag = UnifiedRAGSystem(
    use_local_embeddings=True,
    local_model_path="models/efiras_contrastive_embeddings"
)
```

### Multi-Document Training

The system automatically combines training data from multiple regulatory sources:

```python
# Example from train_embeddings.py
chunks_lux = load_chunks("Lux_cssf18_698eng_chunked_blocks.json")
chunks_basel = load_chunks("Basel_III_chunked_blocks.json")

# Create pairs from both sources
positive_pairs_lux, negative_pairs_lux = builder.create_contrastive_pairs(chunks_lux)
positive_pairs_basel, negative_pairs_basel = builder.create_contrastive_pairs(chunks_basel)

# Combine for comprehensive training
all_pairs = positive_pairs_lux + negative_pairs_lux + positive_pairs_basel + negative_pairs_basel
```

### Performance Improvements

**Contrastive Training Results:**
- **Domain Specialization**: Better understanding of regulatory terminology
- **Improved Retrieval**: Higher precision on financial document queries
- **Stable Training**: Consistent convergence across different document types
- **Balanced Learning**: Equal positive/negative examples prevent bias

**Evaluation Metrics:**
- Training/validation similarity scores saved to model directory
- Embedding quality assessment on regulatory query benchmarks
- Comparative analysis vs. base model performance

### Configuration

**Training Parameters:** 3 epochs, 2e-5 learning rate, batch size 16  
**Models Supported:** `all-mpnet-base-v2`, `e5-base-v2`  
**Output:** `models/efiras_contrastive_embeddings/`
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
