# Unified RAG System - Technical Documentation

## System Overview

The Unified RAG System is a comprehensive, multi-backend retrieval-augmented generation framework that provides flexible document-based question-answering capabilities. It supports multiple embedding providers, language models, and vector storage backends with seamless switching between different configurations for both development and production environments.

## Key Features

### Multi-Backend Architecture
- **Embedding Options**: Local (sentence-transformers), OpenAI API, Azure OpenAI
- **Search Backends**: In-memory vector database with cosine similarity, Azure AI Search with hybrid search
- **Language Models**: OpenAI GPT models, Azure OpenAI deployments
- **Storage**: Local JSON caching, Azure AI Search indexing

### Enterprise Integration
- Environment variable configuration
- Managed identity authentication
- Comprehensive error handling and logging

## System Architecture

### Configuration Matrix

The system uses independent configuration for embedding generation and search backends:

### Embedding Providers (3 options)

| Provider | Cache File | Requirements | Use Case |
|----------|------------|--------------|----------|
| Local | `_local.json` | sentence-transformers | Development, offline, no API costs |
| OpenAI API | `_openai_online.json` | GPT_API_KEY | Cloud development, high quality |
| Azure OpenAI | `_azure_online.json` | Azure credentials | Enterprise, compliance, data residency |

### Search Backends (2 options)

| Backend | Storage | Scalability | Features |
|---------|---------|-------------|----------|
| In-memory | Local vector_db list | <50K documents | Fast, simple, no setup required |
| Azure AI Search | Cloud index | Millions of documents | Hybrid search, filters, distributed |

**Total combinations**: 3 embedding providers × 2 search backends = **6 possible configurations**

## Core Methods & Functionality

### Initialization (`__init__`)
**Purpose**: Configure the unified system with multiple backend options

**Key Parameters**:
- `use_local_embeddings`: Toggle between local vs cloud embeddings
- `use_azure`: Switch between OpenAI and Azure OpenAI
- `use_azure_search`: Choose between in-memory and Azure AI Search
- `model`: Language model name or Azure deployment name
- `online_embedding_model`: Cloud embedding model specification

**Behavior**:
1. Loads environment variables from `.env` file
2. Initializes appropriate client (OpenAI vs AzureOpenAI)
3. Sets up search backend (in-memory vs Azure Search)
4. Configures local embedding model if needed

### Text Enrichment (`enrich_text_with_headers`)
**Purpose**: Combine document block text with hierarchical headers for better embeddings

**Input**: Document block with `text` and optional `enriched_headers`
**Output**: Combined text string optimized for embedding

**Logic**:
```
If enriched_headers exist:
    return f"{enriched_headers}\n\n{block_text}"
Else:
    return block_text
```

**Rationale**: Hierarchical headers provide crucial context that improves semantic search accuracy for regulatory documents.

### Embedding Generation (`embed_text`)
**Purpose**: Generate vector embeddings using the configured provider

**Three Pathways**:

1. **Local Embeddings** (`use_local_embeddings=True`):
   - Uses sentence-transformers library
   - Runs entirely offline
   - Fast for small to medium datasets
   - No API costs

2. **OpenAI Embeddings** (`use_local_embeddings=False`, `use_azure=False`):
   - Direct OpenAI API calls
   - Uses `GPT_API_KEY` environment variable
   - High-quality embeddings
   - Pay-per-use pricing

3. **Azure OpenAI Embeddings** (`use_local_embeddings=False`, `use_azure=True`):
   - Azure OpenAI service
   - Requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
   - Enterprise compliance and data residency
   - Consistent with Azure infrastructure

**Error Handling**: Returns empty list on failure with provider-specific error messages.

### Batch Embedding (`embed_blocks`)
**Purpose**: Process multiple document blocks with intelligent caching

**Caching Strategy**:
1. **Cache Hit**: Load existing embeddings from JSON file
2. **Cache Miss**: Generate embeddings and save to cache
3. **Cache Naming**: Provider-specific suffixes for proper isolation

**Performance Optimizations**:
- Progress tracking for large document sets
- Atomic cache saves (all-or-nothing)
- Embedding validation before storage

### Document Indexing (`add_documents`)
**Purpose**: Add processed documents to the search system

**Two Storage Pathways**:

1. **In-Memory Storage**:
   - Stores embeddings in `self.vector_db` list
   - Fast access for moderate document volumes
   - Cache files: `embeddings_local.json`, `embeddings_openai_online.json`, `embeddings_azure_online.json`

2. **Azure AI Search Storage**:
   - Uploads to Azure AI Search service
   - Scalable for large document collections
   - Supports hybrid search (vector + text)
   - Same cache files (embedding generation is independent of storage)

**Key Design Decision**: Embedding generation is completely separate from storage backend - cache files are named by embedding provider only, not storage method.

### Vector Search (`search`)
**Purpose**: Find relevant documents using semantic similarity

**Two Search Strategies**:

1. **In-Memory Search**:
   - Cosine similarity between query and document embeddings
   - Enhanced with regulatory-specific boosting
   - Supports exact number/term matching
   - Custom relevance scoring

2. **Azure AI Search**:
   - Hybrid search combining vector and text matching
   - Built-in relevance ranking
   - Scalable to millions of documents
   - Advanced filtering capabilities

**Regulatory Enhancement Features**:
- **Number Boosting**: Strong boost for regulatory numbers (e.g., "517")
- **Term Matching**: Enhanced scoring for regulatory terms
- **Reference Patterns**: Special handling for "Article X", "Section Y" patterns

### Context Enhancement (`_enhance_context_with_related_blocks`)
**Purpose**: Expand retrieved context with related sections

**Algorithm**:
1. Extract section headers from retrieved chunks
2. Find additional blocks with matching headers
3. Add one related block per unique section
4. Avoid duplicating already-retrieved content

**Benefit**: Provides comprehensive coverage of regulatory sections while maintaining context coherence.

### Regulatory Intelligence (`_extract_specific_regulations`)
**Purpose**: Identify and extract regulatory citations from text

**Supported Pattern Types**:
- Articles: `Article 5`, `Article 12(3)`
- Sections: `Section 2.1.3`
- CSSF Regulations: `CSSF Regulation 18-698`
- EU Regulations: `Delegated Regulation (EU) 2017/565`
- Circulars: `Circular CSSF 12/546`
- Laws: `2013 Law`

**Output**: Deduplicated list of regulatory references found in context.

### Answer Generation (`answer_query`)
**Purpose**: Generate comprehensive answers using retrieved context

**System Prompt Adaptation**:
- **Compliance-specific prompts** for appointment/officer questions
- **General regulatory prompts** for other queries
- **Structured output requirements** with bullet points and citations

**LLM Configuration**:
- Low temperature (0.1) for consistent regulatory answers
- Increased token limit (1200) for comprehensive responses
- Compatible with both OpenAI and Azure OpenAI

### Comprehensive Response (`answer_with_sources`)
**Purpose**: Provide answers with full traceability and confidence metrics

**Output Structure**:
```json
{
    "answer": "Generated response text",
    "sources": [
        {
            "id": "chunk_id",
            "preview": "First 200 characters...",
            "headers": "Hierarchical section headers",
            "page": "Source page number"
        }
    ],
    "regulatory_references": ["Article 5", "Section 2.1"],
    "confidence": 0.8
}
```

**Confidence Calculation**: `relevant_chunks_found / top_k_requested`

## Configuration Examples

### Development Setup (Local)
```python
rag = UnifiedRAGSystem(
    use_local_embeddings=True,
    use_azure=False,
    use_azure_search=False
)
```

### Cloud Development (OpenAI)
```python
rag = UnifiedRAGSystem(
    use_local_embeddings=False,
    use_azure=False,
    use_azure_search=False,
    online_embedding_model="text-embedding-3-large"
)
```

### Enterprise Production (Full Azure)
```python
rag = UnifiedRAGSystem(
    use_local_embeddings=False,
    use_azure=True,
    use_azure_search=True,
    model="gpt-35-turbo",
    online_embedding_model="text-embedding-ada-002"
)
```

### Hybrid Setup (Local Embeddings + Azure Search)
```python
rag = UnifiedRAGSystem(
    use_local_embeddings=True,
    use_azure=True,
    use_azure_search=True,
    model="gpt-35-turbo"
)
```

## Environment Variables

```bash
# OpenAI Configuration
GPT_API_KEY=your_openai_api_key

# Azure OpenAI Configuration  
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_api_key
```

## Performance Characteristics

### Embedding Generation Speed
- **Local**: ~100-500 docs/minute (CPU dependent)
- **OpenAI API**: ~60 docs/minute (rate limited)
- **Azure OpenAI**: ~60 docs/minute (rate limited)

### Search Performance
- **In-Memory**: Sub-second for <10K documents
- **Azure AI Search**: Sub-second for millions of documents

### Memory Usage
- **In-Memory**: ~1MB per 1000 documents (embedding storage)
- **Azure Search**: Minimal local memory usage

## Error Handling & Resilience

### Embedding Failures
- Returns empty embedding list
- Continues processing remaining documents
- Provider-specific error messages for debugging

### Search Failures
- Graceful degradation to empty results
- Detailed error logging with provider context
- No system crashes on individual failures

### API Rate Limiting
- Built-in retry logic for transient failures
- Exponential backoff for rate limit errors
- Clear error messages for permanent failures

## Scalability Considerations

### Document Volume Limits
- **In-Memory**: Recommended <50K documents
- **Azure Search**: Scales to millions of documents
- **Embedding Cache**: No practical limits (JSON storage)

### Concurrent Usage
- Thread-safe for read operations
- Embedding generation can be parallelized
- Azure Search handles concurrent queries natively

### Cost Optimization
- Aggressive caching reduces API calls
- Provider flexibility enables cost/performance tuning
- Local embeddings eliminate embedding costs for development

## Integration Points

### Document Processing Pipeline
```python
# Typical integration flow
manager = DocumentProcessorManager()
raw_result = manager.process_document(pdf_path)
processed_data = block_processor.process_and_chunk_blocks(raw_result)
chunked_blocks = chunker.chunk_blocks(processed_data)

# Add to RAG system
rag.add_documents(chunked_blocks, cache_path, cache_file_name)
```

### Query Processing
```python
# Simple query
answer = rag.answer_query("What are the compliance requirements?")

# Full response with sources
result = rag.answer_with_sources("What are the compliance requirements?", top_k=5)
```

### Monitoring & Debugging
```python
# System statistics
stats = rag.stats()

# Configuration information
config = rag.get_config_info()
```

This unified architecture provides maximum flexibility while maintaining simplicity and performance for regulatory document processing workflows.