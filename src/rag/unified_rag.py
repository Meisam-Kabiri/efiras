"""
Unified RAG (Retrieval-Augmented Generation) System

This module provides a flexible, unified interface for document-based question-answering using various
embedding models, language models, and vector storage backends. It supports both local and cloud-based
processing with seamless switching between different configurations.

Key Features:
============
1. Multi-Backend Support:
   - Local embeddings (sentence-transformers) for offline processing
   - OpenAI embeddings for high-quality cloud-based embeddings
   - Azure OpenAI integration for enterprise environments
   - Azure AI Search for scalable vector storage

2. Flexible Model Configuration:
   - Supports both OpenAI and Azure OpenAI language models
   - Configurable embedding models (local or cloud)
   - Automatic fallback mechanisms for robust operation

3. Advanced Search Capabilities:
   - Vector similarity search using cosine similarity
   - Hybrid search with Azure AI Search (vector + text)
   - Configurable retrieval parameters (top_k, filters)
   - Context-aware document ranking

4. Enterprise Features:
   - Environment variable configuration
   - Embedding caching for performance optimization
   - Batch processing for large document sets
   - Comprehensive error handling and logging

5. Document Processing Integration:
   - Seamless integration with document chunking systems
   - Metadata preservation and enrichment
   - Support for hierarchical document structures
   - TOC-aware chunking and retrieval

Architecture:
============
The UnifiedRAGSystem class serves as the main interface, coordinating between:
- Embedding generation (local or cloud-based)
- Vector storage (in-memory or Azure AI Search)
- Language model inference (OpenAI or Azure OpenAI)
- Document preprocessing and chunking

Configuration Options:
=====================
Environment Variables:
- GPT_API_KEY: OpenAI API key
- AZURE_OPENAI_API_KEY: Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
- AZURE_SEARCH_API_KEY: Azure AI Search API key
- AZURE_SEARCH_ENDPOINT: Azure AI Search endpoint

Embedding Models:
- Local: sentence-transformers models (e.g., 'all-MiniLM-L6-v2')
- OpenAI: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- Azure: Azure OpenAI embedding deployments

Language Models:
- OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo
- Azure: Azure OpenAI deployment names

Usage Examples:
==============
# Basic local setup
rag = UnifiedRAGSystem(use_local_embeddings=True)

# Azure OpenAI with local embeddings
rag = UnifiedRAGSystem(
    use_local_embeddings=True,
    use_azure=True,
    model="gpt-35-turbo"
)

# Full Azure integration with AI Search
rag = UnifiedRAGSystem(
    use_local_embeddings=False,
    use_azure=True,
    model="gpt-35-turbo",
    online_embedding_model="text-embedding-ada-002",
    use_azure_search=True
)

Performance Considerations:
==========================
- Local embeddings: Faster for small datasets, no API costs
- Cloud embeddings: Higher quality, better for large-scale applications
- Azure AI Search: Scalable for enterprise workloads, supports hybrid search
- Embedding caching: Significantly improves performance for repeated operations

"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

try:
    from .azure_search_backend import AzureSearchBackend
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False


class UnifiedRAGSystem:
    """Unified RAG System supporting both OpenAI and Azure OpenAI"""
    
    def __init__(self, 
                 model: str = "gpt-4", 
                 online_embedding_model: str = "text-embedding-3-large", 
                 use_local_embeddings: bool = True, 
                 local_embedding_model: str = "all-mpnet-base-v2",
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
            model: Model deployment name (OpenAI model or Azure deployment)
            online_embedding_model: Embedding model name (OpenAI or Azure deployment)
            use_local_embeddings: Whether to use local embeddings
            local_embedding_model: Local embedding model name
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            use_azure_search: Whether to use Azure AI Search instead of in-memory vector DB
            azure_search_endpoint: Azure Search service endpoint
            azure_search_key: Azure Search API key (optional if using managed identity)
            azure_search_index: Name of the search index
            use_managed_identity: Use Azure managed identity for search authentication
        """
        
        load_dotenv()
        
        self.use_azure = use_azure
        self.model = model
        self.online_embedding_model = online_embedding_model
        self.use_azure_search = use_azure_search
        self.vector_db = []
        
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
        
        # Initialize client based on type
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
        
        self.use_local_embeddings = use_local_embeddings
        # Initialize local embedding model if needed
        if use_local_embeddings:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(local_embedding_model)
    
    def create_embedding_text(self, block: Dict[str, Any]) -> str:
        """Create embedding text from enriched block"""
        enriched = block.get('enriched_headers')
        if enriched:
            return f"{enriched}\n\n{block['text']}"
        else:
            return block['text']

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text using Azure OpenAI, OpenAI, or local model"""
        if self.use_local_embeddings:
            try:
                return self.local_model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")
                return []
        else:
            """Generate embeddings using OpenAI or Azure OpenAI API"""
            try:
                response = self.client.embeddings.create(
                    model=self.online_embedding_model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                provider = "Azure OpenAI" if self.use_azure else "OpenAI"
                print(f"{provider} embedding error: {e}")
                return []
    
    def embed_blocks(self, blocks: List[Dict[str, Any]], cache_path: str) -> List[Dict[str, Any]]:
        """Embed blocks with caching"""
        # Try loading from cache
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                print(f"Loaded {len(cached)} embeddings from cache")
                return cached
        
        # Generate embeddings
        embeddings = []
        for i, block in enumerate(blocks):
            print(f"Embedding {i+1}/{len(blocks)}")
            
            content = self.create_embedding_text(block)
            embedding = self.embed_text(content)
            
            if embedding:
                embeddings.append({
                    'id': i,
                    'content': block['text'],
                    'embedding': embedding,
                    'block': block
                })
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(embeddings, f)
        print(f"Saved {len(embeddings)} embeddings to cache")
        
        return embeddings
    
    def add_documents(self, blocks: List[Dict[str, Any]], cache_path: str = "data_processed", cache_file_name: str = "embeddings"):
        """Add documents to vector database"""
        if self.use_azure_search:
            # Use Azure Search backend
            embeddings = self.embed_blocks(blocks, f"{cache_path}/{cache_file_name}_azure_search.json")
            self.search_backend.add_documents(embeddings)
            return
        
        # Use in-memory vector database
        if self.use_local_embeddings:
            provider_suffix = "local"
        elif self.use_azure:
            provider_suffix = "azure_online"
        else:
            provider_suffix = "openai_online"
        
        cache_file_path = f"{cache_path}/{cache_file_name}_{provider_suffix}.json"

        self.vector_db = self.embed_blocks(blocks, cache_file_path)
        provider = "Azure OpenAI" if self.use_azure else "OpenAI"
        print(f"Added {len(self.vector_db)} documents to {provider} database")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        
        if self.use_azure_search:
            # Use Azure Search backend with hybrid search
            return self.search_backend.search(
                query_embedding=query_embedding,
                query_text=query,  # Enable hybrid search
                top_k=top_k
            )
        
        # Use in-memory vector database
        if not self.vector_db:
            return []
        
        # Extract key terms (numbers, important words)
        key_terms = []
        
        # Find numbers (like "517")
        regulatory_numbers = re.findall(r'\b\d+\b', query)
        regulatory_terms = re.findall(r'\b(?:article|section|sub-section|point|paragraph)\s+\d+\b', query.lower())
        key_terms.extend(regulatory_numbers)
        
        # Find important words (skip common words)
        words = query.lower().split()
        important_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'this', 'that']]
        key_terms.extend(important_words)

        similarities = []
        for doc in self.vector_db:
            similarity = cosine_similarity([query_embedding], [doc['embedding']])[0][0]

            # Boost for key term matches
            content_lower = doc['content'].lower()
            term_matches = sum(1 for term in key_terms if term in content_lower)
            
            if term_matches > 0:
                similarity += 0.1 * term_matches

            # STRONG boost for regulatory numbers
            for num in regulatory_numbers:
                if num in content_lower:
                    similarity += 0.5  # Very strong boost
            
            # EXTRA boost for full regulatory references
            for term in regulatory_terms:
                if term in content_lower:
                    similarity += 0.7  # Even stronger boost

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
    
    def _enhance_context_with_related_blocks(self, relevant_chunks: List[Dict[str, Any]], query: str) -> str:
        """Enhance context by finding related blocks from same sections"""
        enhanced_context = []
        seen_sections = set()
        
        for chunk in relevant_chunks:
            content = chunk['content']
            enhanced_context.append(content)
            
            # Extract section information
            headers = chunk['block'].get('enriched_headers', '')
            if headers and headers not in seen_sections:
                seen_sections.add(headers)
                
                # Look for related blocks in the same section
                for doc in self.vector_db:
                    if (doc['block'].get('enriched_headers', '') == headers and 
                        doc['content'] not in [c['content'] for c in relevant_chunks]):
                        enhanced_context.append(doc['content'])
                        break  # Add only one related block per section
        
        return "\n\n".join(enhanced_context)
    
    def answer_query(self, query: str, top_k: int = 5, max_context: int = 15000) -> str:
        """Answer query using RAG with enhanced context and regulation extraction"""
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return "No relevant information found."
        
        # Build enhanced context
        enhanced_context = self._enhance_context_with_related_blocks(relevant_chunks, query)
        
        # Extract regulatory references
        regulations = self._extract_specific_regulations(enhanced_context, query)
        
        # Enhanced system prompt based on query type
        if any(keyword in query.lower() for keyword in ['compliance officer', 'appointment', 'requirements']):
            system_prompt = """You are a regulatory compliance expert. When answering questions about compliance officers or appointments:

1. Always include ALL key requirements (full-time vs part-time, approval processes)
2. List ALL required documents completely
3. Mention any exceptions or special procedures (e.g., changes, part-time appointments)
4. Reference specific regulatory sections and articles
5. Be comprehensive while remaining clear and organized

Use bullet points for document lists and organize information logically."""
        else:
            system_prompt = """Answer questions based on the provided context. Be specific and cite relevant sections. If information is insufficient, state this clearly. 

For regulatory questions:
- Include all relevant requirements and exceptions
- Reference specific articles, sections, and regulations
- Organize complex information clearly
- Be concise but thorough
"""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"""Context:\n{enhanced_context}

Regulatory References Found: {', '.join(regulations) if regulations else 'None'}

Question: {query}

Please provide a comprehensive answer that includes all relevant requirements, exceptions, and procedures."""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent regulatory answers
                max_tokens=1200   # Increased for more comprehensive answers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            provider = "Azure OpenAI" if self.use_azure else "OpenAI"
            return f"Error generating response with {provider}: {e}"
    
    def answer_with_sources(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer query and return sources with enhanced regulation tracking"""
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "regulatory_references": [],
                "confidence": 0.0
            }
        
        answer = self.answer_query(query, top_k)
        
        # Extract regulatory references from all relevant chunks
        all_context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        regulations = self._extract_specific_regulations(all_context, query)
        
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                "id": chunk['id'],
                "preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                "headers": chunk['block'].get('enriched_headers', 'N/A'),
                "page": chunk['block'].get('page', 'N/A')
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "regulatory_references": regulations,
            "confidence": len(relevant_chunks) / top_k
        }
    
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
            "embedding_model": self.online_embedding_model,
            "using_local_embeddings": self.use_local_embeddings
        }
        
        if self.use_azure:
            config["azure_endpoint"] = getattr(self, 'azure_endpoint', 'N/A')
        
        if self.use_azure_search:
            config["search_endpoint"] = getattr(self.search_backend, 'endpoint', 'N/A')
            config["search_index"] = getattr(self.search_backend, 'index_name', 'N/A')
        
        return config