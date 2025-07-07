from openai import OpenAI
import numpy as np
from typing import List, Dict, Any, Optional
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

class RAGSystem:
    def __init__(self, openai_api_key: str, model: str = "gpt-4", embedding_model: str = "text-embedding-3-large"):
        """
        Initialize RAG system with OpenAI API
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model for text generation (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
            embedding_model: OpenAI model for embeddings (text-embedding-3-large, text-embedding-3-small)
        """
        from dotenv import load_dotenv
        load_dotenv()
        openai_api_key = os.getenv("GPT_API_KEY")
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.vector_db = []  # Simple in-memory vector store
        self.blocks = []
        self.toc = []
    
    def set_toc(self, toc: List[Dict[str, Any]]):
        """Set the table of contents for document enrichment"""
        self.toc = toc
    
    def find_toc_match(self, header: str, toc: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find matching TOC entry for a given header
        
        Args:
            header: Header string to match (e.g., "Chapter 6", "Sub-section 6.3.2.1")
            toc: Table of contents list
            
        Returns:
            Matching TOC entry or None
        """
        header_clean = header.strip().rstrip(".").strip().lower()
        
        for entry in toc:
            # Match against the 'header' field in TOC
            toc_header = entry.get('header', '').strip().rstrip(".").strip().lower()
            
            # Exact match
            if header_clean == toc_header:
                return entry
            
        return None
    
    def enrich_blocks_with_titles(self, blocks: List[Dict[str, Any]], toc: List[Dict[str, Any]]) ->  List[Dict[str, Any]]:
        """
        Build hierarchical context considering TOC levels
        
        Args:
            enriched_headers: List of enriched header strings
            toc: Table of contents for level information
            
        Returns:
            Hierarchical context string
        """
        # Sort headers by their level in TOC if available
        enriched_blocks = blocks.copy()
        for block in blocks:
            for header in block['headers'].split(','):
                if not header:
                    continue
                
                # Find matching TOC entry
                toc_entry = self.find_toc_match(header, toc)
                if toc_entry:
                    enriched_blocks['enriched_headers'] = f"{block[header]}: {toc_entry.get('title', '')}"
        
        return enriched_blocks


           
        
        # Create hierarchical context
        return " > ".join([header for header, _ in header_with_levels])
    
    def enrich_block_with_titles(self, block: Dict[str, Any], toc: List[Dict[str, Any]]) -> str:
        """
        Enrich a block with titles from TOC
        
        Args:
            block: Block dictionary with 'headers' and 'text' keys
            toc: Table of contents
            
        Returns:
            Enriched block content with hierarchical context
        """
        headers = block['headers']  # "Chapter 6, Sub-chapter 6.3, Section 6, Sub-section 6.3.2.1"
        
        # Split into individual headers
        header_parts = [h.strip() for h in headers.split(',')]
        
        # Find title for each header level
        enriched_headers = []
        for header in header_parts:
            # Find matching TOC entry
            toc_entry = self.find_toc_match(header, toc)
            if toc_entry and toc_entry.get('title'):
                enriched_headers.append(f"{header}: {toc_entry['title']}")
            else:
                enriched_headers.append(header)
        
        # Create hierarchical context using TOC levels
        full_context = self.build_hierarchical_context(enriched_headers, toc)
        
        return f"{full_context}\n\n{block['text']}"
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def embed_blocks_with_titles(
        self,
        blocks: List[Dict[str, Any]],
        toc: List[Dict[str, Any]],
        cache_path: str = "cached_embeddings.json"
    ) -> List[Dict[str, Any]]:
        """
        Embed enriched blocks and cache the result
        """
        # ✅ First try to load from cache
        cached = self.load_embeddings_from_json(cache_path)
        if cached:
            print(f"[Cache] Loaded {len(cached)} embeddings from {cache_path}")
            return cached

        # ❌ If not cached, proceed to embed
        embeddings = []
        for i, block in enumerate(blocks):
            print(f"[Embedding] Block {i + 1}/{len(blocks)}")
            enriched_content = self.enrich_block_with_titles(block, toc)
            embedding = self.embed_text(enriched_content)

            if embedding:
                embeddings.append({
                    'id': i,
                    'content': enriched_content,
                    'embedding': embedding,
                    'original_block': block
                })

        # ✅ Save to cache
        self.save_embeddings_to_json(cache_path, embeddings)
        print(f"[Saved] Embeddings saved to {cache_path}")

        return embeddings



    
    def add_documents(self, blocks: List[Dict[str, Any]], toc: List[Dict[str, Any]] = None):
        """
        Add documents to the vector database
        
        Args:
            blocks: List of document blocks
            toc: Table of contents (optional)
        """
        if toc:
            self.set_toc(toc)
        
        self.blocks = blocks
        embedded_blocks = self.embed_blocks_with_titles(blocks, self.toc)
        self.vector_db.extend(embedded_blocks)
        
        print(f"Added {len(embedded_blocks)} blocks to vector database")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of most similar documents
        """
        if not self.vector_db:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for doc in self.vector_db:
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [query_embedding], 
                [doc['embedding']]
            )[0][0]
            
            similarities.append({
                'document': doc,
                'similarity': similarity
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return [item['document'] for item in similarities[:top_k]]
    


    @staticmethod
    def save_embeddings_to_json(path: str, data: List[Dict[str, Any]]):
            with open(path, 'w') as f:
                json.dump(data, f)
    @staticmethod
    def load_embeddings_from_json(path: str) -> Optional[List[Dict[str, Any]]]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return None
    

    def answer_query(self, query: str, top_k: int = 5, max_context_length: int = 6000, temperature: float = 0.3) -> str:
        """
        Answer a query using RAG approach with Chat Completions API
        
        Args:
            query: User question
            top_k: Number of relevant chunks to retrieve
            max_context_length: Maximum context length for the prompt
            temperature: Temperature for response generation
            
        Returns:
            Generated answer
        """
        # Search for relevant chunks
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            content = chunk['content']
            # Simple length check to avoid exceeding context limits
            if current_length + len(content) < max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # Create messages for Chat Completions API
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context. 
                You should:
                1. Answer based primarily on the provided context
                2. Be specific and cite relevant sections when possible
                3. If the context doesn't contain enough information, clearly state this
                4. Provide clear, well-structured answers
                5. Maintain professional tone appropriate for regulatory/legal documents"""
            },
            {
                "role": "user",
                "content": f"""Context: {context}

Question: {query}

Please answer the question based on the provided context."""
            }
        ]
        
        try:
            # Generate response using Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def answer_query_with_sources(self, query: str, top_k: int = 5, max_context_length: int = 6000) -> Dict[str, Any]:
        """
        Answer a query and return both answer and source information
        
        Args:
            query: User question
            top_k: Number of relevant chunks to retrieve
            max_context_length: Maximum context length for the prompt
            
        Returns:
            Dictionary with answer and source information
        """
        # Search for relevant chunks
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Get the answer
        answer = self.answer_query(query, top_k, max_context_length)
        
        # Prepare source information
        sources = []
        for i, chunk in enumerate(relevant_chunks):
            sources.append({
                "chunk_id": chunk['id'],
                "content_preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                "original_headers": chunk['original_block'].get('headers', 'N/A')
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": len(relevant_chunks) / top_k  # Simple confidence metric
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        return {
            'total_documents': len(self.vector_db),
            'total_blocks': len(self.blocks),
            'toc_entries': len(self.toc)
        }

# Example usage:
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(openai_api_key="your-openai-api-key-here")
    
    # Example TOC with actual structure
    toc = [
        {
            "page": 8,
            "level": 1,
            "header": "Part I.",
            "title": "Definitions and abbreviations"
        },
        {
            "page": 11,
            "level": 1,
            "header": "Part II.",
            "title": "Conditions for obtaining and maintaining the authorisation of an authorised investment fund manager (IFM) who engages solely in the activity of management of UCIs as laid down in Article 101(2) of the 2010 Law and Article 5(2) of the 2013 Law"
        },
        {
            "page": 11,
            "level": 2,
            "header": "Chapter 1.",
            "title": "Basic principles"
        },
        {
            "page": 15,
            "level": 2,
            "header": "Chapter 6.",
            "title": "Bodies of IFM"
        },
        {
            "page": 16,
            "level": 3,
            "header": "Sub-chapter 6.3",
            "title": "Senior Management"
        },
        {
            "page": 17,
            "level": 4,
            "header": "Section 6",
            "title": "Required Number"
        },
        {
            "page": 18,
            "level": 5,
            "header": "Sub-section 6.3.2.1",
            "title": "Contractual Relationship"
        }
    ]
    
    # Example blocks that would match the TOC structure
    blocks = [
        {
            "headers": "Part II., Chapter 6., Sub-chapter 6.3, Sub-section 6.3.2.1",
            "text": "The senior management must maintain a contractual relationship with the institution. This ensures proper governance and accountability structures are in place."
        },
        {
            "headers": "Part II., Chapter 6., Sub-chapter 6.3, Section 6",
            "text": "The minimum number of senior management members required is three, with at least one having expertise in financial management."
        },
        {
            "headers": "Part I.",
            "text": "Investment Fund Manager (IFM): An entity that manages investment funds and is subject to regulatory oversight."
        }
    ]
    
    # Add documents to the system
    # load the json File
    
    # from chunking.block_chunker import RegulatoryChunkingSystem
    # chunker = RegulatoryChunkingSystem()
    # chunks = chunker.block_chunker(blocks) #the chunks are saved under "content" key
    
    with open('src/cleaning/output_processed_text_pymupdf.json', 'r') as f:
        data = json.load(f)


    # print(data)
    # blocks = data["blocks"]
    # toc = data["table_of_contents"]
    # print(toc)
    print(f"Loaded {len(data)} blocks from JSON file.")


    rag.add_documents(blocks, toc)
    
    # # Example queries with the new Chat API
    # print("Database stats:", rag.get_database_stats())
    
    # # Answer questions using the powerful Chat Completions API
    # answer1 = rag.answer_query("What is required for senior management contractual relationship?")
    # print(f"Answer 1: {answer1}")
    
    # answer2 = rag.answer_query("How many senior management members are required?")
    # print(f"Answer 2: {answer2}")
    
    # # Get answer with source information
    # detailed_answer = rag.answer_query_with_sources("What are the definitions mentioned in Part I?")
    # print(f"Detailed Answer: {detailed_answer['answer']}")
    # print(f"Sources: {detailed_answer['sources']}")
    # print(f"Confidence: {detailed_answer['confidence']}")
    
    # # Example of using different models
    # # rag_gpt4 = RAGSystem(openai_api_key="your-key", model="gpt-4-turbo")
    # # rag_gpt35 = RAGSystem(openai_api_key="your-key", model="gpt-3.5-turbo")