# utils.py
import json
import os
import re
from typing import List

# from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
from rank_bm25 import BM25Okapi
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
from sklearn.metrics.pairwise import cosine_similarity

# import torch
# torch.cuda.empty_cache()  # Clear GPU memory

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
        if lang == "en":
            stop_words = set(stopwords.words("english"))
        elif lang == "fr":
            stop_words = set(stopwords.words("french"))
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
    text = re.sub(r"\n+", " ", text)  # Remove newlines
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single
    # Don't split "non-compliance" - keep as one term
    tokens = text.lower().split()
    return [token.strip(".,") for token in tokens if len(token) > 2]


def hybrid_search_rrf(query, query_embedding, vector_db, top_k=5, k=60):
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
            combined_scores[idx] += 1 / (k + rank + 1)

        for rank, idx in enumerate(bm25_ranked):
            combined_scores[idx] += 1 / (k + rank + 1)

        return combined_scores

    # 4. Combine using RRF
    combined_scores = rrf_combine(vector_scores, bm25_scores, k)

    # 5. Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [vector_db[i]["content"] for i in top_indices], top_indices


def hybrid_search_simple_comb(
    query, query_embedding, vector_db, weights=[0.7, 0.3], top_k=5
):
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


def faiss_similarity(chunks: List[dict], query_embedding, query: str, k=5):
    """Find top k similar chunks using FAISS cosine similarity"""

    # Extract only embeddings
    chunk_embeddings = np.array(
        [item["embedding"] for item in chunks], dtype=np.float32
    )

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
        results.append(
            {
                "chunk": chunks[idx],
                "content": [item["content"] for item in chunks if item["id"] == idx],
                "similarity": float(score),
                "index": int(idx),
            }
        )

    return results


# Example usage
# if __name__ == "__main__":
# Get embeddings
# path = 'data/data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'

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
