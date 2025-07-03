from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from src.rag.rag_base import BaseRAG
from src.rag.utils import call_gpt

class LocalRAG(BaseRAG):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.documents = []  # List[dict], each with keys like "text", "source", etc.

    def embed_documents(self, chunks):
        texts = [chunk["content"] for chunk in chunks]  # Extract text for embedding
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.documents = chunks  # Save full dicts for metadata access

    def retrieve(self, query):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k=3)
        return [self.documents[i] for i in I[0]]  # Return full dicts

    def generate_answer(self, query):
        retrieved_chunks = self.retrieve(query)

        # Build context string with metadata if needed
        context = "\n---\n".join([chunk["content"] for chunk in retrieved_chunks])

        prompt = f"Answer the following question strictly (avoid assumptions and base answers strictly on retrieved information) based on the retrieved content:\n{context}\n\nQuestion: {query}"
        return call_gpt(prompt)