import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re


class AzureOpenAIRAGSystem:
    """RAG System using Azure OpenAI Service for enterprise-grade capabilities"""
    
    def __init__(self, 
                 model: str = "gpt-35-turbo", 
                 online_embedding_model: str = "text-embedding-3-large", 
                 use_local_embeddings: bool = True, 
                 local_embedding_model: str = "all-mpnet-base-v2",
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 api_version: str = "2024-02-01"):
        """Initialize Azure OpenAI RAG system
        
        Args:
            model: Azure OpenAI deployment name for chat model
            online_embedding_model: Azure OpenAI deployment name for embeddings
            use_local_embeddings: Whether to use local embeddings instead of Azure
            local_embedding_model: Local embedding model name
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
        """
        
        load_dotenv()
        
        # Get Azure credentials from environment or parameters
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        
        if not self.azure_endpoint or not self.azure_api_key:
            raise ValueError("Azure OpenAI endpoint and API key are required. "
                           "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables "
                           "or pass them as parameters.")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=api_version
        )
        
        self.model = model
        self.online_embedding_model = online_embedding_model
        self.vector_db = []
        
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
        """Generate embeddings for text using Azure OpenAI or local model"""
        if self.use_local_embeddings:
            try:
                return self.local_model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")
                return []
        else:
            """Generate embeddings using Azure OpenAI API"""
            try:
                response = self.client.embeddings.create(
                    model=self.online_embedding_model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Azure OpenAI embedding error: {e}")
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
        if self.use_local_embeddings:
            cache_file_path = cache_path + "/" + cache_file_name + "_azure_local.json"
        else:
            cache_file_path = cache_path + "/" + cache_file_name + "_azure_online.json"

        self.vector_db = self.embed_blocks(blocks, cache_file_path)
        print(f"Added {len(self.vector_db)} documents to Azure OpenAI database")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vector_db:
            return []
        
        query_embedding = self.embed_text(query)
        if not query_embedding:
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
        """Answer query using RAG with Azure OpenAI"""
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
            return f"Error generating response with Azure OpenAI: {e}"
    
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
    
    def stats(self) -> Dict[str, int]:
        """Get database statistics"""
        return {"total_documents": len(self.vector_db)}
    
    def get_azure_info(self) -> Dict[str, str]:
        """Get Azure OpenAI configuration info for debugging"""
        return {
            "endpoint": self.azure_endpoint,
            "model": self.model,
            "embedding_model": self.online_embedding_model,
            "using_local_embeddings": self.use_local_embeddings
        }