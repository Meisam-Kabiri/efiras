import os
import json
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import uuid


class AzureSearchBackend:
    """Azure AI Search backend for vector and hybrid search"""
    
    def __init__(self, 
                 endpoint: str,
                 index_name: str = "documents",
                 api_key: Optional[str] = None,
                 use_managed_identity: bool = False):
        """Initialize Azure Search backend
        
        Args:
            endpoint: Azure Search service endpoint
            index_name: Name of the search index
            api_key: API key for authentication (if not using managed identity)
            use_managed_identity: Use DefaultAzureCredential instead of API key
        """
        self.endpoint = endpoint
        self.index_name = index_name
        
        # Choose authentication method
        if api_key and not use_managed_identity:
            # Use API key when available and not explicitly using managed identity
            credential = AzureKeyCredential(api_key)
        elif use_managed_identity:
            # Use Azure Identity (managed identity, service principal, etc.)
            credential = DefaultAzureCredential()
        else:
            raise ValueError("Either api_key must be provided or use_managed_identity must be True")
        
        # Initialize clients
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )
        
        self.index_client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential
        )
        
        self._index_exists = None
    
    def ensure_index_exists(self, embedding_dimension: int = 1536):
        """Create search index if it doesn't exist"""
        if self._index_exists:
            return
            
        try:
            # Check if index exists
            self.index_client.get_index(self.index_name)
            self._index_exists = True
            return
        except:
            pass
        
        # Create index schema
        from azure.search.documents.indexes.models import (
            SearchIndex, SimpleField, SearchFieldDataType, SearchableField,
            VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
            SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
            SemanticField, VectorSearchAlgorithmKind, SearchField
        )
        
        # Define fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="headers", type=SearchFieldDataType.String),
            SimpleField(name="page", type=SearchFieldDataType.Int32),
            SimpleField(name="block_id", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=embedding_dimension,
                vector_search_profile_name="default-vector-profile"
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="default-hnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw"
                )
            ]
        )
        
        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="default-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="headers"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        self.index_client.create_index(index)
        self._index_exists = True
        print(f"Created Azure Search index: {self.index_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents with embeddings to the search index"""
        if not documents:
            return
        
        # Ensure index exists with correct embedding dimension
        first_embedding = documents[0].get('embedding', [])
        embedding_dim = len(first_embedding) if first_embedding else 1536
        self.ensure_index_exists(embedding_dim)
        
        # Prepare documents for indexing
        search_documents = []
        for doc in documents:
            search_doc = {
                "id": str(uuid.uuid4()),
                "content": doc.get('content', ''),
                "headers": doc.get('block', {}).get('enriched_headers', ''),
                "page": doc.get('block', {}).get('page', 0),
                "block_id": doc.get('id', 0),
                "embedding": doc.get('embedding', [])
            }
            search_documents.append(search_doc)
        
        # Upload documents in batches
        batch_size = 100
        for i in range(0, len(search_documents), batch_size):
            batch = search_documents[i:i + batch_size]
            self.search_client.upload_documents(batch)
        
        print(f"Added {len(search_documents)} documents to Azure Search index")
    
    def search(self, query_embedding: List[float], query_text: str = "", top_k: int = 5, 
               filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using vector similarity with optional hybrid search"""
        if not query_embedding:
            return []
        
        # Create vector query
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )
        
        # Perform search
        # NOTE: Semantic search disabled due to free tier limitations - not available on this service tier
        search_results = self.search_client.search(
            search_text=query_text if query_text else None,  # Hybrid search if text provided
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
            query_type="simple"  # Free tier doesn't support semantic search
        )
        
        # Convert results to our standard format
        results = []
        for result in search_results:
            doc = {
                'id': result.get('block_id', 0),
                'content': result.get('content', ''),
                'embedding': [],  # Don't return large embeddings
                'block': {
                    'enriched_headers': result.get('headers', ''),
                    'page': result.get('page', 0)
                }
            }
            results.append(doc)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search index statistics"""
        try:
            # Get document count
            result = self.search_client.search(search_text="*", include_total_count=True, top=0)
            return {
                "total_documents": result.get_count(),
                "index_name": self.index_name,
                "endpoint": self.endpoint
            }
        except Exception as e:
            return {
                "error": str(e),
                "index_name": self.index_name,
                "endpoint": self.endpoint
            }
    
    def delete_index(self):
        """Delete the search index (use with caution!)"""
        try:
            self.index_client.delete_index(self.index_name)
            self._index_exists = False
            print(f"Deleted Azure Search index: {self.index_name}")
        except Exception as e:
            print(f"Error deleting index: {e}")