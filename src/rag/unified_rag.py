import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
import numpy as np

try:
    from .azure_search_backend import AzureSearchBackend
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False


class RAGSystem:
    """Unified RAG System focusing on retrieval and generation only"""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 use_azure: bool = False,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 api_version: str = "2024-02-01",
                 # Azure Search parameters
                 use_azure_search: bool = False,
                 azure_search_endpoint: Optional[str] = None,
                 azure_search_key: Optional[str] = None,
                 azure_search_index: str = "documents",
                 use_managed_identity: bool = False):
        
        """Initialize unified RAG system
        
        Args:
            model: LLM model deployment name
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            use_azure_search: Whether to use Azure AI Search instead of in-memory vector DB
            azure_search_endpoint: Azure Search service endpoint
            azure_search_key: Azure Search API key
            azure_search_index: Name of the search index
            use_managed_identity: Use Azure managed identity for search authentication
        """
        
        load_dotenv()
        
        self.use_azure = use_azure
        self.model = model
        self.use_azure_search = use_azure_search
        
        # Initialize Azure Search backend if requested
        if use_azure_search:
            if not AZURE_SEARCH_AVAILABLE:
                raise ImportError("Azure Search dependencies not available. Install: pip install azure-search-documents azure-identity")
            
            search_endpoint = azure_search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
            search_key = azure_search_key or os.getenv("AZURE_SEARCH_API_KEY") or os.getenv("AZURE_SEARCH_KEY")
            
            if not search_endpoint:
                raise ValueError("Azure Search endpoint is required. Set AZURE_SEARCH_ENDPOINT environment variable or pass azure_search_endpoint parameter.")
            
            self.search_backend = AzureSearchBackend(
                endpoint=search_endpoint,
                index_name=azure_search_index,
                api_key=search_key,
                use_managed_identity=use_managed_identity
            )
        else:
            self.search_backend = None
        
        # Initialize LLM client
        if use_azure:
            self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
            
            if not self.azure_endpoint or not self.azure_api_key:
                raise ValueError("Azure OpenAI endpoint and API key are required. "
                               "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables "
                               "or pass them as parameters.")
            
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=api_version
            )
        else:
            api_key = os.getenv("GPT_API_KEY")
            if not api_key:
                raise ValueError("GPT_API_KEY environment variable not set")
            
            self.client = OpenAI(api_key=api_key)
        
        print(f"   LLM Provider: {'Azure OpenAI' if use_azure else 'OpenAI'}")
        print(f"   Search Backend: {'Azure AI Search' if use_azure_search else 'In-Memory Vector DB'}")
    

    def search(self, vector_db: List[Dict[str, Any]],
               query: str, 
               query_embedding: List[float], 
               top_k: int = 5, 
               use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """Search for similar documents using pre-computed query embedding"""
        
        if not query_embedding:
            raise ValueError("Query embedding is required for search")
        
        if self.use_azure_search:
          # Lazy setup: upload to Azure Search if not done yet
          if vector_db and not hasattr(self, '_azure_setup_done'):
              print(f"Setting up Azure Search with {len(vector_db)} documents...")
              self.search_backend.add_documents(vector_db)
              self._azure_setup_done = True
              print("✅ Azure Search setup complete")
          
          # Use Azure Search backend
          return self.search_backend.search(
              query_embedding=query_embedding,
              query_text=query,
              top_k=top_k
          )
        
        
        # Use in-memory vector database
        if not vector_db:
            return []
        
        
        if use_hybrid:
            # Use hybrid search from rag.utils
            try:
                from rag.utils import hybrid_search_rrf
                contents, top_indices = hybrid_search_rrf(
                    query=query,
                    query_embedding=query_embedding,
                    vector_db=vector_db,
                    top_k=top_k
                )
                return [vector_db[i] for i in top_indices]
            except ImportError:
                print("⚠️ Hybrid search not available, falling back to vector search")
                use_hybrid = False
        
        if not use_hybrid:
            # Fallback: Pure vector search with regulatory boosting
            similarities = []
            for doc in vector_db:
                similarity = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
                
                # Regulatory boosting
                content_lower = doc['content'].lower()
                headers_lower = doc.get('block', {}).get('header', '').lower()
                search_text = content_lower + ' ' + headers_lower
                
                # Boost for regulatory numbers and terms
                regulatory_numbers = re.findall(r'\b\d+\b', query)
                regulatory_terms = re.findall(r'\b(?:article|section|sub-section|point|paragraph)\s+\d+\b', query.lower())
                
                for num in regulatory_numbers:
                    if num in search_text:
                        similarity += 0.5
                
                for term in regulatory_terms:
                    if term in headers_lower:
                        similarity += 1.0  # Super boost for header matches
                    elif term in search_text:
                        similarity += 0.7
                
                similarities.append({'document': doc, 'similarity': similarity})
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return [item['document'] for item in similarities[:top_k]]
    
    def _extract_specific_regulations(self, context: str, query: str) -> List[str]:
        """Extract specific regulatory references from context"""
        regulations = []
        
        # Patterns for different regulation types
        patterns = [
            r'Article\s+\d+(?:\([^)]+\))?(?:\s+of\s+[^.]+)?',
            r'Section\s+\d+(?:\.\d+)*(?:\.\d+)*',
            r'Sub-section\s+\d+(?:\.\d+)*(?:\.\d+)*',
            r'CSSF\s+Regulation\s+\d+-\d+',
            r'Delegated\s+Regulation\s+\([^)]+\)\s+\d+/\d+',
            r'Circular\s+CSSF\s+\d+/\d+',
            r'Point\s+\d+',
            r'\d{4}\s+Law'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            regulations.extend(matches)
        
        return list(set(regulations))  # Remove duplicates
    

    def answer_query(self, vector_db, query: str, query_embedding: List[float], top_k: int = 5, 
               max_context: int = 15000, use_hybrid: bool = True ) -> str:
        """Answer query using RAG with enhanced context"""
        
        # Search for relevant chunks
        relevant_chunks = self.search(vector_db, query, query_embedding, top_k, use_hybrid)
        
        if not relevant_chunks:
            return "No relevant information found."
        
        system_prompt = """Answer questions based on the provided context. Be specific and cite relevant sections. If information is insufficient, state this clearly. 

        Always include a "Sources" section at the end with the page numbers and headers of the sources you used, formatted nicely."""

        # Build context from relevant chunks with page numbers and headers
        # Build context from relevant chunks with page numbers and headers
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            # Use header_identifier instead of full headers
            header = chunk['block'].get('header_identifier', 'N/A')
            page = chunk['block'].get('page', 'N/A')
            content = chunk['content']
            context_parts.append(f"[Source {i} - Page {page} - {header}]\n{content}")
        
        context = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"""Context:\n{context}

      Question: {query}

      Please provide a comprehensive answer and include a Sources section at the end."""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            provider = "Azure OpenAI" if self.use_azure else "OpenAI"
            return f"Error generating response with {provider}: {e}"
  

    
    def update_vector_db(self, new_vector_db: List[Dict[str, Any]]):
        """Update the vector database with new embeddings"""
        self.vector_db = new_vector_db
        
        if self.use_azure_search:
            # Update Azure Search index
            self.search_backend.add_documents(new_vector_db)
            print(f"Updated Azure Search with {len(new_vector_db)} documents")
        else:
            print(f"Updated in-memory vector DB with {len(new_vector_db)} documents")
    
    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.use_azure_search:
            return self.search_backend.get_stats()
        else:
            return {"total_documents": len(self.vector_db)}
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration info for debugging"""
        config = {
            "llm_provider": "Azure OpenAI" if self.use_azure else "OpenAI",
            "search_backend": "Azure AI Search" if self.use_azure_search else "In-Memory Vector DB",
            "model": self.model,
            "total_documents": len(self.vector_db),
            "supports_faiss": True,
            "supports_hybrid": True
        }
        
        if self.use_azure:
            config["azure_endpoint"] = getattr(self, 'azure_endpoint', 'N/A')
        
        if self.use_azure_search:
            config["search_endpoint"] = getattr(self.search_backend, 'endpoint', 'N/A')
            config["search_index"] = getattr(self.search_backend, 'index_name', 'N/A')
        
        return config