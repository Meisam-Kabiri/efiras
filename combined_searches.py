# from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
# import json

# from elasticsearch import Elasticsearch

# # Simple connection (most common)
# es = Elasticsearch("http://localhost:9200")

# # Or if you need more specific config
# # es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])

# path = 'data_processed/Lux_cssf18_698eng_embeddings_local.json'

# with open(path, 'r') as f:
#     chunks = json.load(f)

# # Index your chunks (do this once)
# for i, chunk in enumerate(chunks):
#     es.index(
#         index="chunks", 
#         id=i,
#         body={
#             "content": chunk["content"],
#             "embeddings": chunk["embedding"]
#         }
#     )

# def search(query, model):
#     query_vector = model.encode(query).tolist()
    
#     result = es.search(
#         index="chunks",
#         body={
#             "query": {
#                 "bool": {
#                     "should": [
#                         {"match": {"content": query}},  # BM25
#                         {"knn": {"field": "embeddings", "query_vector": query_vector, "k": 10}}  # Vector
#                     ]
#                 }
#             }
#         }
#     )
    
#     return [hit["_source"]["content"] for hit in result["hits"]["hits"]]

# # Use it
# model = SentenceTransformer('all-mpnet-base-v2')
# results = search("What is Python?", model)


import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect

from langdetect import detect

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


def hybrid_search_rrf(query, chunks, model, top_k=5, k=60):
   # 1. Vector search using existing embeddings
   query_embedding = model.encode(query)
   chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
   vector_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
   
   # 2. BM25 search using proper BM25 library
   texts = [chunk["content"] for chunk in chunks]
   tokenized_texts = [minimal_tokenize(text) for text in texts]
   bm25 = BM25Okapi(tokenized_texts)
   bm25_scores = bm25.get_scores(minimal_tokenize(query))
   
   # 3. RRF combination (no normalization needed)
   def rrf_combine(vector_scores, bm25_scores, k=60):
       vector_ranked = np.argsort(vector_scores)[::-1]
       bm25_ranked = np.argsort(bm25_scores)[::-1]
       
       combined_scores = np.zeros(len(vector_scores))
       
       for rank, idx in enumerate(vector_ranked):
           combined_scores[idx] += 1 / (k + rank + 1)
       
       for rank, idx in enumerate(bm25_ranked):
           combined_scores[idx] += 1 / (k + rank + 1)
       
       return combined_scores
   
   # 4. Combine using RRF
   combined_scores = rrf_combine(vector_scores, bm25_scores, k)
   
   # 5. Get top results
   top_indices = np.argsort(combined_scores)[::-1][:top_k]
   
   return [chunks[i]["content"] for i in top_indices]


def hybrid_search_simple_comb(query, chunks, model, top_k=5):
    # 1. Vector search using existing embeddings
    query_embedding = model.encode(query)
    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
    vector_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # 2. BM25 search using proper BM25 library
    texts = [chunk["content"] for chunk in chunks]
    # tokenized_texts = [text.split() for text in texts]
    tokenized_texts = [minimal_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(query.split())
    
    # 3. Normalize BM25 scores to 0-1 range (to match cosine similarity)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # 4. Combine scores
    combined_scores = 0.7 * vector_scores + 0.3 * bm25_scores
    
    # 5. Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    return [chunks[i]["content"] for i in top_indices]

import json 
path = 'data_processed/Lux_cssf18_698eng_embeddings_local.json'
with open(path, 'r') as f:
    chunks = json.load(f)

q = "What monitoring elements must IFM implement for central administration delegation?"
model = SentenceTransformer('all-mpnet-base-v2')
chunks = hybrid_search_rrf(q, chunks, model )
print(chunks[0])