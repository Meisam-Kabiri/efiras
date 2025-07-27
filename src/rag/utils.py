# utils.py
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re


import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


from langdetect import detect
# import torch
# torch.cuda.empty_cache()  # Clear GPU memory

import asyncio
import numpy as np
import faiss
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
import os
import json


class HybridSearch:
    def __init__(self, index_dir="indexes"):
        """Initialize HybridSearch with optional index directory"""
        self.index_dir = index_dir
        self.faiss_index = None
        self.whoosh_index = None
        self.chunks = []
        
    def build_indexes(self, chunk_data):
        """Build FAISS and Whoosh indexes"""
        os.makedirs(self.index_dir + "/whoosh", exist_ok=True)
        self.chunks = chunk_data
        
        # Build FAISS
        embeddings = np.array([c["embedding"] for c in self.chunks]).astype('float32')
        faiss.normalize_L2(embeddings)
        
        if len(self.chunks) > 10000:
            # Large dataset - use IVF
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)
            self.faiss_index.train(embeddings)
            self.faiss_index.nprobe = 10
        elif len(self.chunks) > 1000:
            # Medium dataset - use HNSW
            self.faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        else:
            # Small dataset - use flat
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        
        self.faiss_index.add(embeddings)
        
        # Build Whoosh
        schema = Schema(id=ID(stored=True), content=TEXT())
        self.whoosh_index = create_in(self.index_dir + "/whoosh", schema)
        
        writer = self.whoosh_index.writer()
        for i, chunk in enumerate(self.chunks):
            writer.add_document(id=str(i), content=chunk["content"])
        writer.commit()
        
        print(f"✅ Indexes built for {len(self.chunks)} chunks")

    async def vector_search(self, query_embedding, top_k=100):
        """Fast vector search using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_indexes() first.")
            
        query_emb = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return scores[0], indices[0]

    async def bm25_search(self, query, top_k=100):
        """Fast BM25 search using Whoosh"""
        if self.whoosh_index is None:
            raise ValueError("Whoosh index not built. Call build_indexes() first.")
            
        with self.whoosh_index.searcher(weighting=scoring.BM25F()) as searcher:
            parser = QueryParser("content", self.whoosh_index.schema)
            try:
                parsed_query = parser.parse(query)
            except:
                # Fallback for malformed queries
                from whoosh.query import Every
                parsed_query = Every()
                
            results = searcher.search(parsed_query, limit=top_k)
            return [(int(r['id']), r.score) for r in results]

    def rrf_combine(self, vector_results, bm25_results, k=60):
        """Reciprocal Rank Fusion combination"""
        scores, indices = vector_results
        combined = {}
        
        # Vector rankings
        for rank, idx in enumerate(indices):
            idx = int(idx)
            if idx != -1:
                combined[idx] = combined.get(idx, 0) + 1/(k + rank + 1)
        
        # BM25 rankings  
        for rank, (idx, score) in enumerate(bm25_results):
            combined[idx] = combined.get(idx, 0) + 1/(k + rank + 1)
        
        return combined

    async def hybrid_search(self, query, query_embedding, top_k=5):
        """Hybrid search combining vector and BM25 search"""
        # Run both searches concurrently
        vector_task = self.vector_search(query_embedding, 100)
        bm25_task = self.bm25_search(query, 100)
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # Combine using RRF
        combined = self.rrf_combine(vector_results, bm25_results)
        top_indices = sorted(combined.keys(), key=combined.get, reverse=True)[:top_k]
        
        # Return content and indices
        top_content = [self.chunks[i]["content"] for i in top_indices if i < len(self.chunks)]
        
        return top_content, top_indices

    def save_indexes(self, faiss_path=None, whoosh_path=None):
        """Save indexes to disk for persistence"""
        if faiss_path is None:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
        if whoosh_path is None:
            whoosh_path = self.index_dir + "/whoosh"
            
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"✅ FAISS index saved to {faiss_path}")
            
        if self.whoosh_index is not None:
            print(f"✅ Whoosh index already saved to {whoosh_path}")

    def load_indexes(self, faiss_path=None, whoosh_path=None, chunks_data=None):
        """Load indexes from disk"""
        if faiss_path is None:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
        if whoosh_path is None:
            whoosh_path = self.index_dir + "/whoosh"
            
        try:
            # Load FAISS index
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                print(f"✅ FAISS index loaded from {faiss_path}")
            
            # Load Whoosh index
            if os.path.exists(whoosh_path):
                self.whoosh_index = open_dir(whoosh_path)
                print(f"✅ Whoosh index loaded from {whoosh_path}")
            
            # Load chunks data if provided
            if chunks_data is not None:
                self.chunks = chunks_data
                print(f"✅ Loaded {len(self.chunks)} chunks")
                
            return True
        except Exception as e:
            print(f"❌ Error loading indexes: {e}")
            return False

    def get_stats(self):
        """Get index statistics"""
        stats = {
            "num_chunks": len(self.chunks),
            "faiss_index_type": type(self.faiss_index).__name__ if self.faiss_index else None,
            "whoosh_index_exists": self.whoosh_index is not None,
            "index_directory": self.index_dir
        }
        return stats


# # Usage example
# async def main():
#     # Initialize search system
#     search = HybridSearch(index_dir="indexes")
    
#     # Load embeddings
#     path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
#     with open(path, 'r') as f:
#         load_embeddings = json.load(f)
    
#     # Build indexes
#     search.build_indexes(load_embeddings['embeddings'])
    
#     # Optional: Save indexes for persistence
#     search.save_indexes()
    
#     # Query
#     query = "What monitoring elements must IFM implement for central administration delegation?"
    
#     # Get query embedding
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu')
#     query_embed = model.encode(query)
    
#     # Search
#     results, indices = await search.hybrid_search(query, query_embed, top_k=5)
    
#     print("Search Results:")
#     print(f"Indices: {indices}")
#     for i, result in enumerate(results):
#         print(f"{i+1}. {result[:100]}...")
    
#     # Print stats
#     print("\nIndex Stats:", search.get_stats())


if __name__ == "__main__":
    # asyncio.run(main())
    search = HybridSearch(index_dir="indexes")
    
    # Load embeddings
    path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
    with open(path, 'r') as f:
        load_embeddings = json.load(f)
    
    # Build indexes
    search.build_indexes(load_embeddings['embeddings'])
    
    # Optional: Save indexes for persistence
    search.save_indexes()
    
    # Query
    query = "What monitoring elements must IFM implement for central administration delegation?"
    
    # Get query embedding
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu')
    query_embed = model.encode(query)
    
    # Search (sync version since we can't await in main)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results, indices = loop.run_until_complete(search.hybrid_search(query, query_embed, top_k=5))
    loop.close()
    
    print("Search Results:")
    print(f"Indices: {indices}")
    for i, result in enumerate(results):
        print(f"{i+1}. {result[:100]}...")
    
    # Print stats
    print("\nIndex Stats:", search.get_stats())


    

























load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("GPT_API_KEY")

def call_gpt(prompt: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content



def smart_tokenize(text):
    try:
        lang = detect(text)
        if lang == 'en':
            stop_words = set(stopwords.words('english'))
        elif lang == 'fr':
            stop_words = set(stopwords.words('french'))
        else:
            stop_words = set()  # No stopwords for unknown languages
    except:
        stop_words = set()  # Fallback to no stopwords
    
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def minimal_tokenize(text):
    # Just remove punctuation, keep most words
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and len(token) > 1]

def better_tokenize(text):
    import re
    # Keep important terms together
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    # Don't split "non-compliance" - keep as one term
    tokens = text.lower().split()
    return [token.strip('.,') for token in tokens if len(token) > 2]

def hybrid_search_rrf(query, query_embedding, vector_db,  top_k=5, k=60):
   # 1. Vector search using existing embeddings
   chunk_embeddings = np.array([chunk["embedding"] for chunk in vector_db])
   vector_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
   
   # 2. BM25 search using BM25 library
   texts = [chunk["content"] for chunk in vector_db]
   tokenized_texts = [better_tokenize(text) for text in texts]
   bm25 = BM25Okapi(tokenized_texts)
   bm25_scores = bm25.get_scores(better_tokenize(query))
   
   # 3. RRF combination (no normalization needed)
   def rrf_combine(vector_scores, bm25_scores, k=60):
       vector_ranked = np.argsort(vector_scores)[::-1] 
       bm25_ranked = np.argsort(bm25_scores)[::-1]
       
       combined_scores = np.zeros(len(vector_scores))
       
       for rank, idx in enumerate(vector_ranked):
           combined_scores[idx] += (1 / (k + rank + 1))
       
       for rank, idx in enumerate(bm25_ranked):
           combined_scores[idx] += (1 / (k + rank + 1))
       
       return combined_scores
   
   # 4. Combine using RRF
   combined_scores = rrf_combine(vector_scores, bm25_scores, k)
   
   # 5. Get top results
   top_indices = np.argsort(combined_scores)[::-1][:top_k]
   
   return [vector_db[i]["content"] for i in top_indices], top_indices


def hybrid_search_simple_comb(query, query_embedding, vector_db, weights = [0.7, 0.3],  top_k=5):
    # 1. Vector search using existing embeddings
    chunk_embeddings = np.array([chunk["embedding"] for chunk in vector_db])
    vector_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # 2. BM25 search using proper BM25 library
    texts = [chunk["content"] for chunk in vector_db]
    # tokenized_texts = [text.split() for text in texts]
    tokenized_texts = [minimal_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(query.split())
    
    # 3. Normalize BM25 scores to 0-1 range (to match cosine similarity)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # 4. Combine scores
    combined_scores = weights[0] * vector_scores + weights[1] * bm25_scores
    
    # 5. Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    return [vector_db[i]["content"] for i in top_indices], top_indices





























# # Initialize model once
# model = SentenceTransformer('all-mpnet-base-v2')

# def sklearn_similarity(chunks: List[str], query:str, k=5, path: str = ''):
#     """Find top k similar chunks using scikit-learn cosine similarity"""

#     chunk_embeddings = np.array([item['embedding'] for item in chunks], dtype=np.float32)
#     query_embedding = model.encode([query])
    
#     # Calculate similarities
#     similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
#     # Get top k indices
#     top_indices = np.argsort(similarities)[::-1][:k]
    
#     # Return results
#     results = []
#     for idx in top_indices:
#         results.append({
#             'chunk': chunks[idx],
#             'content': [item['content'] for item in chunks if item['id'] == idx],
#             'similarity': similarities[idx],
#             'index': idx
#         })
    
#     return results



def faiss_similarity(chunks:List[dict], query_embedding, query:str, k=5):
    """Find top k similar chunks using FAISS cosine similarity"""

    # Extract only embeddings
    chunk_embeddings = np.array([item['embedding'] for item in chunks], dtype=np.float32)
    
    # Convert to float32 and normalize
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(query_embedding)
    
    # Create index and search
    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    similarities, indices = index.search(query_embedding, k)
    
    # Return results
    results = []
    for score, idx in zip(similarities[0], indices[0]):
        results.append({
            'chunk': chunks[idx],
            'content': [item['content'] for item in chunks if item['id'] == idx],
            'similarity': float(score),
            'index': int(idx)
        })
    
    return results







# Example usage
# if __name__ == "__main__":
        # Get embeddings
    # path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'

    # with open(path, 'r') as f:
    #     load_embeddings = json.load(f)
    #     # chunk_embeddings = np.array([item['embedding'] for item in load_embeddings], dtype=np.float32)
    #     # chunks_content = [item['content'] for item in load_embeddings]
    
    # query = "What monitoring elements must IFM implement for central administration delegation?"
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    # query_embed = model.encode(query)

    # build_indexes(load_embeddings['embeddings'])
    # a, b = asyncio.run(hybrid_search(query, query_embed, 5))
    # print(b)
    # print(a)
    # print("Scikit-learn results:")
    # sklearn_results = sklearn_similarity(load_embeddings, query, k=1)
    # for r in sklearn_results:
    #     print(f"  {r['similarity']:.3f} - {r['content']}")
    #     print(".."*50)
    
    # print("\nFAISS results:")
    # faiss_results = faiss_similarity(load_embeddings, query, k=1)
    # for r in faiss_results:
    #     print(f"  {r['similarity']:.3f} - {r['content']}")