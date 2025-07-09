import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


# # FASTEST - 5x faster than mpnet, still good quality
# local_embedding_model = "all-MiniLM-L6-v2"  # 384-dim, ~90MB

# # GOOD BALANCE - 2-3x faster than mpnet, better quality than L6
# local_embedding_model = "all-MiniLM-L12-v2"  # 384-dim, ~130MB

# # STILL GOOD - Similar quality to mpnet, bit faster
# local_embedding_model = "all-distilroberta-v1"  # 768-dim, ~290MB

class RAGSystem:
    def __init__(self, model: str = "gpt-4", online_embedding_model: str = "text-embedding-3-large", use_local_embeddings: bool = False, local_embedding_model: str = "all-mpnet-base-v2"):
        """Initialize RAG system with OpenAI API"""
        


        load_dotenv()
        api_key = os.getenv("GPT_API_KEY")
        if not api_key:
            raise ValueError("GPT_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
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
        """Generate embeddings for text"""
        if self.use_local_embeddings:
            try:
                return self.local_model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")
            return []
        else:
            """Generate embeddings using OpenAI API"""
            try:
                response = self.client.embeddings.create(
                    model=self.online_embedding_model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Embedding error: {e}")
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
                    'content': content,
                    'embedding': embedding,
                    'block': block
                })
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(embeddings, f)
        print(f"Saved {len(embeddings)} embeddings to cache")
        
        return embeddings
    
    def add_documents(self, blocks: List[Dict[str, Any]], cache_path: str = "data_processed", cache_file_name: str =  "embeddings"):
        """Add documents to vector database"""
        if self.use_local_embeddings:
            cache_file_path = cache_path + "/" + cache_file_name + "_local.json"
        else:
            # cache_file_path = Path(cache_path)
            # cache_file_path.mkdir(parents=True, exist_ok=True)
            cache_file_path = cache_path + "/" + (cache_file_name + "_online.json")

        self.vector_db = self.embed_blocks(blocks, cache_file_path)
        print(f"Added {len(self.vector_db)} documents to database")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.vector_db:
            return []
        
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        
        similarities = []
        for doc in self.vector_db:
            similarity = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
            similarities.append({'document': doc, 'similarity': similarity})
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['document'] for item in similarities[:top_k]]
    
    def answer_query(self, query: str, top_k: int = 5, max_context: int = 6000) -> str:
        """Answer query using RAG"""
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return "No relevant information found."
        
        # Build context within limits
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            content = chunk['content']
            if current_length + len(content) < max_context:
                context_parts.append(content)
                current_length += len(content)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        messages = [
            {
                "role": "system",
                "content": "Answer questions based on the provided context. Be specific and cite relevant sections. If information is insufficient, state this clearly."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def answer_with_sources(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer query and return sources"""
        relevant_chunks = self.search(query, top_k)
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "confidence": 0.0
            }
        
        answer = self.answer_query(query, top_k)
        
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
            "confidence": len(relevant_chunks) / top_k
        }
    
    def stats(self) -> Dict[str, int]:
        """Get database statistics"""
        return {"total_documents": len(self.vector_db)}









# Example usage:
# if __name__ == "__main__":
    
    # file = "data_processed/Lux_cssf18_698eng_processed_blocks.json"
    # file_path = Path(file)
    # with open(file_path, 'r') as f:
    #     data = json.load(f)

    # blocks = data.get("blocks", [])

    # if data["document_info"]["filename_without_ext"]:
    #     filename = data["document_info"]["filename_without_ext"]

    # print(f"Loaded {len(data)} blocks from JSON file.")

    # rag = RAGSystem(use_local_embeddings = True)
    # rag.add_documents(blocks, cache_path="data_processed", cache_file_name=filename + "_embeddings")
    
    # # # Example queries with the new Chat API
    # # print("Database stats:", rag.get_database_stats())
    
    # # # Answer questions using the powerful Chat Completions API
    # answer1 = rag.answer_query("Can the same conducting officer in an Investment Fund Manager (IFM) be responsible for both the risk management function and the investment management function?")
    # print(f"Answer 1: {answer1}")
    
    # # answer2 = rag.answer_query("How many senior management members are required?")
    # # print(f"Answer 2: {answer2}")
    
    # # # Get answer with source information
    # # detailed_answer = rag.answer_query_with_sources("What are the definitions mentioned in Part I?")
    # # print(f"Detailed Answer: {detailed_answer['answer']}")
    # # print(f"Sources: {detailed_answer['sources']}")
    # # print(f"Confidence: {detailed_answer['confidence']}")
    
    # # # Example of using different models
    # # # rag_gpt4 = RAGSystem(openai_api_key="your-key", model="gpt-4-turbo")
    # # # rag_gpt35 = RAGSystem(openai_api_key="your-key", model="gpt-3.5-turbo")