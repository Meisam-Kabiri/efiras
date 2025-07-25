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
# import torch
# torch.cuda.empty_cache()  # Clear GPU memory

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
  #  query_embedding = model.encode(query)
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
   
   return [chunks[i]["content"] for i in top_indices], top_indices


def hybrid_search_simple_comb(query, query_embedding, chunks, weights = [0.7, 0.3],  top_k=5):
    # 1. Vector search using existing embeddings
    # query_embedding = model.encode(query)
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






import json 
path = 'data_processed/Lux_cssf18_698eng_embeddings_local.json'
with open(path, 'r') as f:
    chunks_mpnet = json.load(f)

path = 'data_processed/Lux_cssf18_698eng_embeddings_openai_online.json'
with open(path, 'r') as f:
    chunks_openai = json.load(f)

# q = "What monitoring elements must IFM implement for central administration delegation?"
q = "What must a person demonstrate to the CSSF if they hold more than one mandate as a conducting officer?"
q = "What are some examples of compliance issues that an IFM must address according to the provided text?"  #chunk 119
q =  "compliance issues violations infringements non-compliance policy restrictions transactions reporting fraud"
q = "Examples include AML/CFT breaches, inadequate risk management, poor internal controls, conflicts of interest, and failure to meet CSSF reporting or governance requirements.."
model = SentenceTransformer('all-mpnet-base-v2', local_files_only=True)
q_embed = model.encode(q)
# chunks_res, inds = hybrid_search_rrf(q, q_embed, chunks, model, top_k=50, k=10)
chunks_res, inds = hybrid_search_simple_comb(q, q_embed, chunks_mpnet )
# print(chunks_res[0:5])
print(inds)




###########################################################################
from src.rag.unified_rag import UnifiedRAGSystem
rag_openai = UnifiedRAGSystem(use_local_embeddings = False) # for gpt embedings
rag_base_local = UnifiedRAGSystem(local_embedding_model = "all-mpnet-base-v2", cached_local_model = True) # for all-mpnet-base-v2 embedings
fine_tuned_path = "models/efiras_contrastive_embeddings"
rag_tuned = UnifiedRAGSystem(local_embedding_model = fine_tuned_path, cached_local_model = True) # for all-mpnet-base-v2 embedings

with open("data_processed/Lux_cssf18_698eng_chunked_blocks.json", 'r') as f:
    chunks_to_be_embeded = json.load(f)

chunks_tuned = rag_tuned.embed_blocks(chunks_to_be_embeded, cache_path= "data_processed/Lux_cssf18_698eng_embeddings_local_fine_tuned.json")
q = "What are some examples of compliance issues that an IFM must address according to the provided text?"  #chunk 119
q = "What rights do the persons conducting the business of the IFM have regarding the delegation of functions?" #chunk 201
# q = "What specific documents must be included in the notification regarding the branch manager(s)?" #chunk 264
# q = "What specific documents must be included with the notification to the CSSF? " #chunk 52
q_embed = rag_openai.embed_text(q)
# chunks_res, inds = hybrid_search_simple_comb(q, q_embed, chunks_openai, weights = [0.0, 1.0])
chunks_res, inds = hybrid_search_rrf(q, q_embed, chunks_openai, 5, k=10)
# print(chunks_res[0:5])
print(f"inds found using openai embedding model: {inds} \n")

q_embed = rag_base_local.embed_text(q)
chunks_res, inds = hybrid_search_simple_comb(q, q_embed, chunks_mpnet, weights = [1, 0])
# print(chunks_res[0:5])
print(f"inds found using local base mpnet: {inds} \n")

###########################################################################
# # Load the questions
questions_list = load_questions_from_txt("all_questions.txt")




# Print to verify
found = 0
not_found = 0
for item in questions_list:
    # print(f"ID: {item['id']}, Q: {item['question']}")

    q = item["question"]
    
    
    
    # model = SentenceTransformer('all-mpnet-base-v2',  device='cpu', local_files_only=True)


    q_embed = rag_openai.embed_text(q)
    chunks_res, inds = hybrid_search_rrf(q, q_embed, chunks_openai, 5, 10)

    # q_embed = rag_tuned.embed_text(q)
    # chunks_res, inds = hybrid_search_rrf(q, q_embed, chunks_tuned, 5, 10)


    # q_embed = rag_base_local.embed_text(q)
    # chunks_res, inds = hybrid_search_rrf(q, q_embed, chunks_mpnet, 5, 10)
    
    # chunks_res, inds = hybrid_search_simple_comb(q, q_embed, chunks_openai, weights = [0.5, 0.5])
    # chunks_res, inds = hybrid_search_simple_comb(q, chunks, model )
    # print(chunks_res[0:5])
    # print(inds)
    if item['id'] in inds:
        found+=1
    else:
        not_found+=1
        print(f"not found query:\n")
        print(q)
print(f"found are: {found}\n")
print(f"Not found are: {not_found}\n")
    



###################################################################################

# Now you have a list of dicts like:
# [
#   {"id": "290", "question": "What changes are subject to notification..."},
#   {"id": "291", "question": "What is the latest deadline..."},
#   ...
# ]



# What must a person demonstrate to the CSSF if they hold more than one mandate as a conducting officer?
# not found query:

# To whom are the persons responsible for the internal control functions accountable?
# not found query:

# What are some examples of compliance issues that an IFM must address according to the provided text?
# not found query:

# What qualifications and conditions must AML/CFT Compliance Officers meet according to the text?
# not found query:

# What is the purpose of the initial due diligence and ongoing monitoring by the IFM concerning delegates?
# found are: 253

# Not found are: 5