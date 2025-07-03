from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict
from src.rag.rag_base import BaseRAG
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("GPT_API_KEY")
print(openai_api_key)

class RAGState(TypedDict):
    """State for the RAG workflow"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str
    confidence_score: float

class LangGraphRAG(BaseRAG):
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Initialize LLM
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        
        # Initialize vector store
        self.vector_store = None
        
        # Custom prompt templates
        self.answer_prompt = PromptTemplate(
            template="""Answer the following question strictly based on the retrieved content. 
            Avoid assumptions and base answers strictly on the provided information.
            
            Context: {context}
            
            Question: {query}
            
            Answer:""",
            input_variables=["context", "query"]
        )
        
        self.confidence_prompt = PromptTemplate(
            template="""Rate the confidence of this answer on a scale of 0-1, where:
            - 1.0 means the answer is fully supported by the context
            - 0.5 means the answer is partially supported
            - 0.0 means the answer is not supported by the context
            
            Context: {context}
            Question: {query}
            Answer: {answer}
            
            Confidence Score (0-1):""",
            input_variables=["context", "query", "answer"]
        )
        
        # Build the workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the RAG workflow using LangGraph"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate_context", self._generate_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence_node)
        
        # Define the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate_context")
        workflow.add_edge("generate_context", "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_confidence")
        workflow.add_edge("evaluate_confidence", END)
        
        return workflow.compile()

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        # Retrieve documents
        docs = self.vector_store.similarity_search(state["query"], k=3)
        
        # Convert to your format
        retrieved_chunks = []
        for doc in docs:
            chunk = {"content": doc.page_content}
            chunk.update(doc.metadata)
            retrieved_chunks.append(chunk)
        
        state["retrieved_docs"] = retrieved_chunks
        return state

    def _generate_context_node(self, state: RAGState) -> RAGState:
        """Generate context string from retrieved documents"""
        context = "\n---\n".join([chunk["content"] for chunk in state["retrieved_docs"]])
        state["context"] = context
        return state

    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """Generate answer using LLM"""
        prompt = self.answer_prompt.format(
            context=state["context"],
            query=state["query"]
        )
        answer = self.llm(prompt)
        state["answer"] = answer.strip()
        return state

    def _evaluate_confidence_node(self, state: RAGState) -> RAGState:
        """Evaluate confidence in the answer"""
        prompt = self.confidence_prompt.format(
            context=state["context"],
            query=state["query"],
            answer=state["answer"]
        )
        confidence_str = self.llm(prompt).strip()
        
        try:
            confidence = float(confidence_str)
            state["confidence_score"] = max(0.0, min(1.0, confidence))
        except ValueError:
            state["confidence_score"] = 0.5  # Default confidence
        
        return state

    def embed_documents(self, chunks: List[Dict[str, Any]]):
        """Convert chunks to LangChain Documents and create vector store"""
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={k: v for k, v in chunk.items() if k != "content"}
            )
            documents.append(doc)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        initial_state = RAGState(
            query=query,
            retrieved_docs=[],
            context="",
            answer="",
            confidence_score=0.0
        )
        
        result = self.workflow.invoke(initial_state)
        return result["retrieved_docs"]

    def generate_answer(self, query: str) -> str:
        """Generate answer using the workflow"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        initial_state = RAGState(
            query=query,
            retrieved_docs=[],
            context="",
            answer="",
            confidence_score=0.0
        )
        
        result = self.workflow.invoke(initial_state)
        return result["answer"]

    def generate_detailed_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer with full workflow results"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        initial_state = RAGState(
            query=query,
            retrieved_docs=[],
            context="",
            answer="",
            confidence_score=0.0
        )
        
        result = self.workflow.invoke(initial_state)
        return {
            "query": result["query"],
            "answer": result["answer"],
            "confidence_score": result["confidence_score"],
            "sources": result["retrieved_docs"],
            "context": result["context"]
        }