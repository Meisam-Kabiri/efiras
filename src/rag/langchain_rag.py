from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# OR from langchain.embeddings import HuggingFaceEmbeddings (if you had an older version)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema import Document
from src.rag.rag_base import BaseRAG
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("GPT_API_KEY")

class LangChainRAG(BaseRAG):
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Initialize LLM
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        
        # Initialize vector store (will be set after embedding documents)
        self.vector_store = None
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""Answer the following question strictly based on the retrieved content. 
            Avoid assumptions and base answers strictly on the provided information.
            
            Context: {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # Initialize retrieval chain (will be set after embedding documents)
        self.qa_chain = None

    def embed_documents(self, chunks: List[Dict[str, Any]]):
        """Convert chunks to LangChain Documents and create vector store"""
        documents = []
        for chunk in chunks:
            # Convert to LangChain Document format
            doc = Document(
                page_content=chunk["content"],
                metadata={k: v for k, v in chunk.items() if k != "content"}
            )
            documents.append(doc)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        # Retrieve documents
        docs = self.vector_store.similarity_search(query, k=3)
        
        # Convert back to your format
        retrieved_chunks = []
        for doc in docs:
            chunk = {"content": doc.page_content}
            chunk.update(doc.metadata)
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks

    def generate_answer(self, query: str) -> str:
        """Generate answer using the retrieval chain"""
        if self.qa_chain is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        return self.qa_chain.run(query)

    def generate_answer_with_sources(self, query: str) -> Dict[str, Any]:
        """Generate answer with source information"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        # Get retrieved documents
        retrieved_docs = self.retrieve(query)
        
        # Generate answer
        answer = self.generate_answer(query)
        
        return {
            "answer": answer,
            "sources": retrieved_docs,
            "query": query
        }