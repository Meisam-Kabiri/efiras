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
from elasticsearch import Elasticsearch


load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("GPT_API_KEY")




def call_gpt(prompt: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize model once
model = SentenceTransformer('all-mpnet-base-v2')

def sklearn_similarity(chunks: List[str], query:str, k=5, path: str = ''):
    """Find top k similar chunks using scikit-learn cosine similarity"""

    chunk_embeddings = np.array([item['embedding'] for item in chunks], dtype=np.float32)
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Return results
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'content': [item['content'] for item in chunks if item['id'] == idx],
            'similarity': similarities[idx],
            'index': idx
        })
    
    return results

def faiss_similarity(chunks:List[dict], query:str, k=5):
    """Find top k similar chunks using FAISS cosine similarity"""

    # Extract only embeddings
    chunk_embeddings = np.array([item['embedding'] for item in chunks], dtype=np.float32)
    query_embedding = model.encode([query])
    
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




class BM25:
    """Simple BM25 implementation"""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # Tokenize and calculate document frequencies
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        
        # Tokenize all documents
        tokenized_corpus = []
        for doc in corpus:
            tokens = self.tokenize(doc)
            tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))
            self.doc_freqs.append(Counter(tokens))
        
        self.tokenized_corpus = tokenized_corpus
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        
        # Calculate IDF for each term
        df = Counter()
        for doc_freq in self.doc_freqs:
            for term in doc_freq.keys():
                df[term] += 1
        
        for term, freq in df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
    
    def tokenize(self, text):
        """Simple tokenization"""
        # Convert to lowercase, remove special chars, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def score(self, query, doc_idx):
        """Calculate BM25 score for a document"""
        query_tokens = self.tokenize(query)
        doc_freqs = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        for term in query_tokens:
            if term in doc_freqs:
                tf = doc_freqs[term]
                idf = self.idf.get(term, 0)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        
        return score
    

# Example usage
if __name__ == "__main__":
        # Get embeddings
    path = 'data_processed/Lux_cssf18_698eng_embeddings_local.json'

    with open(path, 'r') as f:
        load_embeddings = json.load(f)
        # chunk_embeddings = np.array([item['embedding'] for item in load_embeddings], dtype=np.float32)
        # chunks_content = [item['content'] for item in load_embeddings]
    
    query = "What monitoring elements must IFM implement for central administration delegation?"
    
    
    print("Scikit-learn results:")
    sklearn_results = sklearn_similarity(load_embeddings, query, k=1)
    for r in sklearn_results:
        print(f"  {r['similarity']:.3f} - {r['content']}")
        print(".."*50)
    
    print("\nFAISS results:")
    faiss_results = faiss_similarity(load_embeddings, query, k=1)
    for r in faiss_results:
        print(f"  {r['similarity']:.3f} - {r['content']}")