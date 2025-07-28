# from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import logging
old_level = logging.getLogger("sentence_transformers").level
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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
# import torch
# torch.cuda.empty_cache()  # Clear GPU memory

import asyncio
import numpy as np
import faiss
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
import os

# Global indexes - build once, use many times
faiss_index = None
whoosh_index = None
chunks = []



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

def hybrid_search_rrf(query, query_embedding, chunks,  top_k=5, k=60):
   # 1. Vector search using existing embeddings
   chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
   vector_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
   
   # 2. BM25 search using BM25 library
   texts = [chunk["content"] for chunk in chunks]
   tokenized_texts = [better_tokenize(text) for text in texts]
   bm25 = BM25Okapi(tokenized_texts)
   bm25_scores = bm25.get_scores(better_tokenize(query))
   
   # 3. RRF combination (no normalization needed)
   def rrf_combine(vector_scores, bm25_scores, k=60):
       vector_ranked = np.argsort(vector_scores)[::-1] 
       bm25_ranked = np.argsort(bm25_scores)[::-1]
       
      #  print(f"vector_ranked:{vector_ranked[1:20]}")
      #  print(f"bm25_ranked:{bm25_ranked[1:20]}")
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
   
   return [chunks[i] for i in top_indices]


def hybrid_search_simple_comb(query, query_embedding, chunks, weights = [0.7, 0.3],  top_k=5):
    # 1. Vector search using existing embeddings
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
    combined_scores = weights[0] * vector_scores + weights[1] * bm25_scores
    
    # 5. Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    return [chunks[i]["content"] for i in top_indices], top_indices


def load_questions_from_txt(filename="generated_questions.txt"):
    questions = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_chunk = {}
    for line in lines:
        line = line.strip()
        
        if line.startswith("Chunk ID:"):
            chunk_id = line.replace("Chunk ID:", "").strip()
            current_chunk = {"id": int(chunk_id)}  # ← Convert to int
            
        elif line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
            current_chunk["question"] = question
            questions.append(current_chunk)
            current_chunk = {}
    
    return questions








###########################################################################
import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from rag.search_service import SearchService
from rag.embedding_service import EmbeddingService
from rag.rag_generator import RAGGenerator
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem



# import json 
# path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
# with open(path, 'r') as f:
#     chunks_local = json.load(f)

# path = 'data_processed/Lux_cssf18_698eng_embeddings_openai_online.json'
# with open(path, 'r') as f:
#     chunks_openai = json.load(f)

# q = "What monitoring elements must IFM implement for central administration delegation?"
q = "What must a person demonstrate to the CSSF if they hold more than one mandate as a conducting officer?"
q = "What are some examples of compliance issues that an IFM must address according to the provided text?"  #chunk 119
q =  "compliance issues violations infringements non-compliance policy restrictions transactions reporting fraud"
q = "Examples include AML/CFT breaches, inadequate risk management, poor internal controls, conflicts of interest, and failure to meet CSSF reporting or governance requirements.."



# File paths
input_pdf = "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
embeddings_path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
output_dir = Path("data_processed")
output_dir.mkdir(exist_ok=True)

# Initialize services
embedding_service = EmbeddingService()
search_service = SearchService(index_dir="indexes")
rag_system = RAGGenerator()

# Step 1: Try to load existing embeddings
if os.path.exists(embeddings_path):
  print(f"Loading embeddings from {embeddings_path}")
  with open(embeddings_path, 'r') as f:
      embeddings_data = json.load(f)
else:
  print("Embeddings not found, generating new ones...")
  # Process PDF and generate embeddings
  reader = PyMuPDFProcessor()
  raw_blocks = reader.extract_blocks(input_pdf)
  
  processor = block_processor(raw_blocks)
  processed_blocks = processor.process_blocks()
  
  chunker = RegulatoryChunkingSystem(processed_blocks)
  chunks_doc = chunker.chunk_blocks()
  
  embeddings_data = embedding_service.embed_all_chunks(chunks_doc)

# Step 2: Setup search service
if not search_service.load_indexes():
  print("Building new search indexes...")
  search_service.build_indexes(embeddings_data)  # Build indexes from chunks
  search_service.save_indexes()  # Save for next time
else:
  print("Search indexes loaded successfully!")
  search_service.set_chunks([embeddings_data])  # Set chunks first

# What ongoing responsibility do members of the management body/governing body of the IFM have regarding compliance? chunk_id: 89
# What responsibilities does the executive committee have regarding the compliance of the investment policy with the prospectus? chunk_id: 106
# To whom does the sub-chapter apply within the context of the IFM? chunk_id: 119
# What information must an IFM communicate to the CSSF when delegating risk management activities? chunk_id: 238
# What is the process for an IFM to delegate the compliance function to a third party? chunk_id: 262
# Under what circumstances can the CSSF allow an IFM to delegate the internal audit function to an external expert? chunk_id: 293
# To which entities does this sub-chapter apply? chunk_id: 319
# What specific elements must be covered in the annual report regarding the identification and assessment of ML/TF risks? chunk_id: 337
# What is required of the IFM concerning the due diligence and ongoing monitoring of delegates? chunk_id: 378
# What measures must the IFM take to ensure compliance with CSSF Regulation 10-4 when delegating functions? chunk_id: 480
# What additional regulations must AIFMs comply with according to the text? chunk_id: 501
q = "What ongoing monitoring responsibility does the IFM have regarding delegated portfolio management?q = " #chunk_id: 519

# q =  "What ongoing monitoring operations must an IFM perform to ensure compliance with legal provisions for managed UCIs?" #chunk_id: 521



q_embed = embedding_service.embed_text(q)

# Step 4: Search for relevant chunks
import asyncio
relevant_chunks = asyncio.run(search_service.hybrid_search(q, q_embed, top_k=20))
for chunk in relevant_chunks:
    print(chunk["chunk_id"], ",", end="", flush=True)
print('\n=========================================================')
relevant_chunks = asyncio.run(search_service.hybrid_search_with_cross_encoder(q, q_embed, top_k=8))
for chunk in relevant_chunks:
    print(chunk["chunk_id"], ",", end="", flush=True)
print('\n=========================================================')

relevant_chunks =hybrid_search_rrf(q, q_embed, embeddings_data["embeddings"],  top_k=20, k=60)
for chunk in relevant_chunks:
    print(chunk["chunk_id"], ",", end="", flush=True)
print()

# ###########################################################################
# # # # Load the questions
# questions_list = load_questions_from_txt("all_questions.txt")
# found = 0
# not_found = 0
# i = 0
# for item in questions_list:
#     q = item["question"]
#     id = item["id"]
#     i+=1
    
#     q_embed = embedding_service.embed_text(q)
#     relevant_chunks = asyncio.run(search_service.hybrid_search(q, q_embed, top_k=8))
#     # relevant_chunks = asyncio.run(search_service.hybrid_search_with_cross_encoder(q, q_embed, top_k=8))
#     # relevant_chunks =hybrid_search_rrf(q, q_embed, embeddings_data["embeddings"],  top_k=5, k=60)
#     inds = [chunk['chunk_id'] for chunk in relevant_chunks]

#     if item['id'] in inds:
#         found+=1
#     else:
#         not_found+=1
#         print(f"not found query: {not_found}/{i}\n")
#         print(q, "chunk_id:", id)
# print(f"found are: {found}\n")
# print(f"Not found are: {not_found}\n")
    


