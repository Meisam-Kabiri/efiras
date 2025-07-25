# Azure AI Search Backend Documentation

## Overview

The Azure AI Search Backend provides enterprise-grade vector search capabilities for the EFIRAS RAG system. It enables scalable document storage and retrieval using Microsoft's Azure AI Search service, supporting millions of documents with sub-second search performance.

## Architecture

```
Document Chunks → Azure AI Search Index → Vector + Hybrid Search → Ranked Results
```

### Key Components

- **SearchClient**: Handles document indexing and search operations
- **SearchIndexClient**: Manages index creation and configuration
- **Vector Search**: HNSW algorithm for fast similarity search
- **Hybrid Search**: Combines vector similarity with traditional text search
- **Semantic Search**: Advanced AI-powered search capabilities (premium tiers)

## Class: AzureSearchBackend

### Purpose
Provides Azure AI Search integration for the UnifiedRAGSystem, enabling scalable vector storage and retrieval as an alternative to in-memory vector databases.

### Location
`src/rag/azure_search_backend.py`

### Dependencies
```python
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
```

### Authentication Methods

#### 1. API Key Authentication
```python
backend = AzureSearchBackend(
    endpoint="https://your-service.search.windows.net",
    index_name="documents",
    api_key="your_api_key_here"
)
```

#### 2. Managed Identity Authentication
```python
backend = AzureSearchBackend(
    endpoint="https://your-service.search.windows.net",
    index_name="documents",
    use_managed_identity=True
)
```

## Index Schema

### Field Structure
```json
{
  "id": "unique_document_id",
  "content": "The actual text content of the chunk",
  "headers": "TITLE III > Article 409 > Disclosure to investors",
  "page": 239,
  "block_id": 3230,
  "embedding": [0.1, 0.8, -0.3, 0.5, ...]
}
```

### Field Types
- **id**: String (Primary Key) - Unique identifier for each document
- **content**: SearchableField - Full-text searchable document content
- **headers**: SearchableField - Document structure/hierarchy information
- **page**: Integer - Source document page number
- **block_id**: Integer - Original chunk ID from processing pipeline
- **embedding**: Vector Field - High-dimensional embedding for similarity search

## Vector Search Configuration

### HNSW Algorithm Settings
```python
{
    "m": 4,                    # Number of bi-directional links for each node
    "efConstruction": 400,     # Size of dynamic candidate list during index construction
    "efSearch": 500,          # Size of dynamic candidate list during search
    "metric": "cosine"        # Distance metric for similarity calculation
}
```

### Performance Characteristics
- **Index Build Time**: O(n log n) where n = number of documents
- **Search Time**: O(log n) for approximate nearest neighbors
- **Memory Usage**: ~4-8 bytes per dimension per document
- **Accuracy**: 95%+ recall with proper parameter tuning

## Methods

### \_\_init\_\_(endpoint, index_name, api_key, use_managed_identity)

**Purpose**: Initialize Azure Search backend with authentication

**Parameters**:
- `endpoint` (str): Azure Search service URL
- `index_name` (str): Name of the search index (default: "documents")
- `api_key` (Optional[str]): API key for authentication
- `use_managed_identity` (bool): Use Azure managed identity instead of API key

**Example**:
```python
# Production setup with API key
backend = AzureSearchBackend(
    endpoint="https://myservice.search.windows.net",
    index_name="regulatory-docs",
    api_key=os.getenv("AZURE_SEARCH_API_KEY")
)

# Enterprise setup with managed identity
backend = AzureSearchBackend(
    endpoint="https://myservice.search.windows.net",
    index_name="regulatory-docs",
    use_managed_identity=True
)
```

### ensure_index_exists(embedding_dimension=1536)

**Purpose**: Create search index with proper schema if it doesn't exist

**Parameters**:
- `embedding_dimension` (int): Dimension of embedding vectors (default: 1536 for OpenAI)

**Features**:
- Automatic schema detection and creation
- Vector search configuration with HNSW algorithm
- Semantic search setup (premium tiers only)
- Idempotent operation (safe to call multiple times)

**Index Configuration**:
```python
# Vector Search Profile
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(...)],
    profiles=[VectorSearchProfile(...)]
)

# Semantic Search Configuration  
semantic_search = SemanticSearch(
    configurations=[SemanticConfiguration(...)]
)
```

### add_documents(documents)

**Purpose**: Index document chunks with embeddings for search

**Parameters**:
- `documents` (List[Dict]): List of document chunks with embeddings

**Input Format**:
```python
documents = [
    {
        'id': 3230,
        'content': "Institutions acting as an originator...",
        'embedding': [0.1, 0.8, -0.3, ...],  # 1536-dimensional vector
        'block': {
            'enriched_headers': 'TITLE III > Article 409 > Disclosure to investors',
            'page': 239
        }
    }
]
```

**Features**:
- Automatic batch processing (100 documents per batch)
- UUID generation for unique document IDs
- Embedding dimension auto-detection
- Progress tracking and error handling

**Performance**:
- Batch Size: 100 documents (optimal for Azure limits)
- Throughput: ~1000 documents/minute
- Memory Usage: Minimal (streaming processing)

### search(query_embedding, query_text, top_k, filters)

**Purpose**: Perform vector similarity search with optional hybrid search

**Parameters**:
- `query_embedding` (List[float]): Query vector for similarity search
- `query_text` (str): Optional text for hybrid search
- `top_k` (int): Number of results to return
- `filters` (Optional[str]): OData filter expression

**Search Types**:

#### 1. Pure Vector Search
```python
results = backend.search(
    query_embedding=[0.1, 0.2, -0.3, ...],
    top_k=5
)
```

#### 2. Hybrid Search (Vector + Text)
```python
results = backend.search(
    query_embedding=[0.1, 0.2, -0.3, ...],
    query_text="Article 409 disclosure requirements",
    top_k=5
)
```

#### 3. Filtered Search
```python
results = backend.search(
    query_embedding=[0.1, 0.2, -0.3, ...],
    filters="page ge 200 and page le 300",  # Pages 200-300 only
    top_k=5
)
```

**Return Format**:
```python
[
    {
        'id': 3230,
        'content': "Institutions acting as an originator...",
        'embedding': [],  # Excluded for performance
        'block': {
            'enriched_headers': 'TITLE III > Article 409 > Disclosure to investors',
            'page': 239
        }
    }
]
```

### get_stats()

**Purpose**: Retrieve index statistics and health information

**Returns**:
```python
{
    "total_documents": 4025,
    "index_name": "regulatory-docs",
    "endpoint": "https://myservice.search.windows.net"
}
```

**Use Cases**:
- Monitoring index size
- Performance benchmarking
- Health checks in production

### delete_index()

**Purpose**: Delete the entire search index (destructive operation)

**Warning**: ⚠️ This permanently deletes all indexed data. Use with extreme caution!

**Use Cases**:
- Development environment cleanup
- Index schema migration
- Emergency data removal

## Integration with UnifiedRAGSystem

### Configuration
```python
# Enable Azure AI Search backend
rag = UnifiedRAGSystem(
    use_local_embeddings=True,
    use_azure=True,
    model="gpt-35-turbo",
    use_azure_search=True,  # This enables Azure Search backend
    azure_search_endpoint="https://myservice.search.windows.net",
    azure_search_key="your_api_key"
)
```

### Environment Variables
```bash
# Required for Azure Search integration
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_API_KEY=your_api_key_here

# Optional: Use specific index name
AZURE_SEARCH_INDEX=regulatory-documents
```

### Automatic Backend Selection
```python
# UnifiedRAGSystem automatically chooses backend
if self.use_azure_search:
    # Use Azure Search for large-scale production
    self.search_backend = AzureSearchBackend(...)
else:
    # Use in-memory storage for development
    self.vector_db = embeddings
```

## Performance Considerations

### Scalability Limits
```
Basic Tier:     15M documents, 2GB storage
Standard S1:    15M documents, 25GB storage
Standard S2:    60M documents, 100GB storage
Standard S3:    60M documents, 200GB storage
Standard S3HD:  1B documents, 2TB storage
```

### Query Performance
- **Latency**: 10-100ms typical response time
- **Throughput**: 1000+ queries/second on Standard tiers
- **Concurrent Users**: 100+ simultaneous queries

### Cost Optimization
- Use **Basic tier** for development/testing
- Use **Standard S1** for small-medium production loads
- Scale up only when needed (costs increase significantly)

### Best Practices

#### 1. Index Design
- Keep embedding dimensions consistent (1536 for OpenAI)
- Use meaningful field names for filtering
- Minimize number of searchable fields for performance

#### 2. Batch Operations
- Upload documents in batches of 100-1000
- Use async operations for large datasets
- Monitor throttling and retry with exponential backoff

#### 3. Query Optimization
- Use filters to reduce search space
- Combine vector + text search for better relevance
- Cache frequent queries when possible

#### 4. Monitoring
- Track query latency and throughput
- Monitor index storage usage
- Set up alerts for service limits

## Error Handling

### Common Issues

#### 1. Authentication Errors
```python
# Error: Invalid API key
ValueError("Either api_key must be provided or use_managed_identity must be True")

# Solution: Check environment variables
api_key = os.getenv("AZURE_SEARCH_API_KEY")
if not api_key:
    print("AZURE_SEARCH_API_KEY environment variable not set")
```

#### 2. Index Creation Failures
```python
# Error: Service tier doesn't support semantic search
# Solution: Disable semantic search for Basic/Free tiers
query_type="simple"  # Instead of "semantic"
```

#### 3. Document Upload Errors
```python
# Error: Request entity too large
# Solution: Reduce batch size
batch_size = 50  # Instead of 100
```

#### 4. Search Timeout
```python
# Error: Search request timeout
# Solution: Reduce top_k or add filters
results = backend.search(
    query_embedding=embedding,
    top_k=10,  # Instead of 100
    filters="page le 500"  # Limit search scope
)
```

## Comparison: Local vs Azure Search

### Local In-Memory Storage
**Pros**:
- ✅ Zero cost
- ✅ Fast development iteration
- ✅ No external dependencies
- ✅ Complete data control

**Cons**:
- ❌ Limited to 50K documents
- ❌ Single-server limitation
- ❌ No persistence across restarts
- ❌ Memory constraints

### Azure AI Search
**Pros**:
- ✅ Millions of documents
- ✅ Sub-second search performance
- ✅ Hybrid search capabilities
- ✅ Enterprise features (security, monitoring)
- ✅ Automatic scaling

**Cons**:
- ❌ Monthly costs ($125-$3000+)
- ❌ Network latency
- ❌ Azure dependency
- ❌ More complex setup

## Migration Path

### Development to Production
```python
# Step 1: Develop with local storage
rag_dev = UnifiedRAGSystem(use_local_embeddings=True, use_azure_search=False)

# Step 2: Test with Azure Search
rag_test = UnifiedRAGSystem(
    use_local_embeddings=True,
    use_azure_search=True,
    azure_search_endpoint=test_endpoint
)

# Step 3: Deploy to production
rag_prod = UnifiedRAGSystem(
    use_local_embeddings=False,  # Use Azure OpenAI embeddings
    use_azure=True,
    use_azure_search=True,
    azure_search_endpoint=prod_endpoint
)
```

### Data Migration
```python
# Export from local storage
local_embeddings = rag_local.vector_db

# Import to Azure Search
azure_backend = AzureSearchBackend(endpoint, index_name, api_key)
azure_backend.add_documents(local_embeddings)
```

## Security Considerations

### Authentication
- **Development**: Use API keys stored in environment variables
- **Production**: Use Azure Managed Identity for keyless authentication
- **CI/CD**: Use Service Principal authentication

### Network Security
- Enable firewall rules to restrict access
- Use Private Endpoints for enterprise deployments
- Configure VNet integration for hybrid scenarios

### Data Privacy
- All data encrypted at rest and in transit
- Regional data residency options available
- GDPR compliance for EU deployments

## Troubleshooting

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed Azure SDK logs
backend = AzureSearchBackend(endpoint, index_name, api_key)
```

### Health Check
```python
def health_check():
    try:
        stats = backend.get_stats()
        print(f"Index health: {stats}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```

### Performance Testing
```python
import time

def benchmark_search():
    start_time = time.time()
    results = backend.search(query_embedding, top_k=10)
    duration = time.time() - start_time
    print(f"Search completed in {duration:.2f}s, found {len(results)} results")
```

## Related Documentation

- [UnifiedRAGSystem Documentation](unified_rag_system.md)
- [Azure AI Search Official Docs](https://docs.microsoft.com/en-us/azure/search/)
- [Vector Search in Azure AI Search](https://docs.microsoft.com/en-us/azure/search/vector-search-overview)