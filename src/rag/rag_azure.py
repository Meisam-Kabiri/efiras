from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from src.rag.rag_base import BaseRAG
from src.rag.utils import call_gpt

class AzureRAG(BaseRAG):
    def __init__(self, endpoint: str, index_name: str, api_key: str):
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )

    def embed_documents(self, texts):
        pass  # Assume indexing is pre-done via Azure pipeline

    def retrieve(self, query):
        results = self.search_client.search(query, top=3)
        return [doc['content'] for doc in results]

    def generate_answer(self, query):
        context = "\n".join(self.retrieve(query))
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}"
        return call_gpt(prompt)
