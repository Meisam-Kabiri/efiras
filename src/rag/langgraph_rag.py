from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict
from src.rag.rag_base import BaseRAG
from dotenv import load_dotenv
import os
import re
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
openai_api_key = os.getenv("GPT_API_KEY")

class RAGState(TypedDict):
    """Enhanced state for the RAG workflow"""
    query: str
    expanded_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    filtered_docs: List[Dict[str, Any]]
    context: str
    answer: str
    confidence_score: float
    source_citations: List[str]

class LangGraphRAG(BaseRAG):
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2')

        from langchain_openai import ChatOpenAI

        # Initialize LLM for chat models
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",  # Use 'model' instead of 'model_name'
            openai_api_key=openai_api_key,
            max_tokens=4000
        )
        
        # Initialize vector store
        self.vector_store = None
        
        # Enhanced prompt templates
        self.query_expansion_prompt = PromptTemplate(
            template="""Given this regulatory compliance question, generate 3 alternative phrasings that might help find relevant regulatory provisions. Focus on:
            1. Different regulatory terms (e.g., "delegation" vs "outsourcing")
            2. Specific entity types (e.g., "UCITS" vs "AIF" vs "UCI")
            3. Jurisdictional variations (e.g., "Luxembourg" vs "home Member State")
            
            Original question: {query}
            
            Alternative phrasings:
            1.""",
            input_variables=["query"]
        )
        
        self.regulatory_answer_prompt = PromptTemplate(
            template="""You are a regulatory compliance expert analyzing CSSF regulations. Answer the question based STRICTLY on the provided regulatory context.

CRITICAL INSTRUCTIONS:
1. PRIORITIZE SPECIFIC RULES over general rules when they conflict
2. IDENTIFY if different rules apply to different jurisdictions (Luxembourg vs non-Luxembourg)
3. CITE specific section numbers when making claims
4. If rules prohibit something, state it clearly as "NOT permitted"
5. If information is missing, state "The provided regulations do not address this specific scenario"
6. DO NOT make assumptions or infer rules not explicitly stated

REGULATORY CONTEXT:
{context}

QUESTION: {query}

STRUCTURED ANALYSIS:
1. Applicable regulatory framework:
2. Specific rules that apply:
3. Any exceptions or special conditions:
4. Answer:""",
            input_variables=["context", "query"]
        )
        
        self.source_verification_prompt = PromptTemplate(
            template="""Verify if this answer is fully supported by the provided regulatory context. 
            
            Context: {context}
            Answer: {answer}
            
            For each claim in the answer, identify:
            1. Is it directly supported by the context? (YES/NO)
            2. Which specific section supports it?
            3. Any unsupported claims?
            
            Verification:""",
            input_variables=["context", "answer"]
        )
        
        # Build the enhanced workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the enhanced RAG workflow"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("expand_query", self._expand_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("filter_and_rank", self._filter_and_rank_node)
        workflow.add_node("generate_context", self._generate_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("verify_sources", self._verify_sources_node)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence_node)
        
        # Define the flow
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "filter_and_rank")
        workflow.add_edge("filter_and_rank", "generate_context")
        workflow.add_edge("generate_context", "generate_answer")
        workflow.add_edge("generate_answer", "verify_sources")
        workflow.add_edge("verify_sources", "evaluate_confidence")
        workflow.add_edge("evaluate_confidence", END)
        
        return workflow.compile()
    
    def _expand_query_node(self, state: RAGState) -> RAGState:
        """Expand query to capture regulatory variations"""
        # Generate alternative phrasings
        # prompt = self.query_expansion_prompt.format(query=state["query"])
        
        # # Fix: Extract content from AIMessage
        # response = self.llm.invoke(prompt)
        
        # # Handle both string and AIMessage responses
        # if hasattr(response, 'content'):
        #     expanded_text = response.content  # For ChatOpenAI (AIMessage)
        # else:
        #     expanded_text = str(response)  # For OpenAI (string)
        
        # # Extract numbered alternatives
        # expanded_queries = [state["query"]]  # Include original
        # lines = expanded_text.split('\n')
        # for line in lines:
        #     if re.match(r'^\d+\.', line.strip()):
        #         expanded_queries.append(line.strip()[2:].strip())
    
        # state["expanded_queries"] = expanded_queries

        state["expanded_queries"] = state["query"]
        return state

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Enhanced retrieval with multiple queries"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        all_docs = []
        seen_content = set()
        
        # Search with each expanded query
        for query in state["expanded_queries"]:
            docs = self.vector_store.similarity_search_with_score(query, k=5)
            
            for doc, score in docs:
                # Avoid duplicates
                if doc.page_content not in seen_content:
                    chunk = {
                        "content": doc.page_content,
                        "similarity_score": float(score),
                        "query_matched": query
                    }
                    chunk.update(doc.metadata)
                    all_docs.append(chunk)
                    seen_content.add(doc.page_content)
        
        # Sort by similarity score
        all_docs.sort(key=lambda x: x["similarity_score"])
        
        state["retrieved_docs"] = all_docs[:8]  # Take top 8
        return state

    def _filter_and_rank_node(self, state: RAGState) -> RAGState:
        """Filter and rank documents based on regulatory relevance"""
        filtered_docs = []
        
        for doc in state["retrieved_docs"]:
            # Keep documents with good similarity scores
            if doc["similarity_score"] < 0.7:  # Adjust threshold as needed
                # Check for regulatory keywords
                content = doc["content"].lower()
                regulatory_keywords = [
                    "delegation", "aifm", "ucits", "ici", "cssf", "section", 
                    "authorized", "permitted", "prohibited", "must", "shall"
                ]
                
                if any(keyword in content for keyword in regulatory_keywords):
                    filtered_docs.append(doc)
            else:
                filtered_docs.append(doc)
        
        # Prioritize specific sections over general ones
        def get_section_priority(doc):
            content = doc["content"]
            if "specific" in content.lower() or "exception" in content.lower():
                return 1
            elif "sub-section" in content.lower():
                return 2
            elif "section" in content.lower():
                return 3
            else:
                return 4
        
        filtered_docs.sort(key=lambda x: (get_section_priority(x), x["similarity_score"]))
        
        state["filtered_docs"] = filtered_docs
        return state

    def _generate_context_node(self, state: RAGState) -> RAGState:
        """Generate structured context with section identification"""
        context_parts = []
        
        for i, doc in enumerate(state["filtered_docs"]):
            section_info = ""
            if "section" in doc.get("metadata", {}):
                section_info = f"[Section {doc['metadata']['section']}] "
            
            context_parts.append(f"--- Document {i+1} ---\n{section_info}{doc['content']}")
        
        state["context"] = "\n\n".join(context_parts)
        return state

    # def _generate_answer_node(self, state: RAGState) -> RAGState:
    #     """Generate regulatory-focused answer"""
    #     prompt = self.regulatory_answer_prompt.format(
    #         context=state["context"],
    #         query=state["query"]
    #     )
    #     answer = self.llm.invoke(prompt)
    #     state["answer"] = answer.strip()
    #     return state


    def _generate_answer_node(self, state: RAGState) -> RAGState:
        """Generate regulatory-focused answer"""

        # It's generally good practice to define a system message
        # to guide the model's behavior and tone.
        system_message_content = "You are a helpful and knowledgeable assistant specializing in regulatory affairs. Provide concise and accurate answers based on the given context."

        # Format the human message content with the context and query.
        human_message_content = self.regulatory_answer_prompt.format(
            context=state["context"],
            query=state["query"]
        )

        # Create the list of messages for the chat model.
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=human_message_content)
        ]

        # Invoke the chat model. The ChatOpenAI.invoke() method returns a BaseMessage object
        # (specifically an AIMessage in this case), not a raw string.
        ai_message = self.llm.invoke(messages)

        # Access the actual text content of the AI's response using .content
        # and then apply .strip() to it.
        state["answer"] = ai_message.content.strip()
        return state


    def _verify_sources_node(self, state: RAGState) -> RAGState:
        """Verify answer against sources"""
        prompt = self.source_verification_prompt.format(
            context=state["context"],
            answer=state["answer"]
        )
        verification = self.llm.invoke(prompt)
        
        # Extract citations (simplified)
        citations = []
        for doc in state["filtered_docs"]:
            if any(keyword in state["answer"].lower() for keyword in doc["content"].lower().split()[:10]):
                citations.append(doc.get("section", "Unknown section"))
        
        state["source_citations"] = citations
        return state

    def _evaluate_confidence_node(self, state: RAGState) -> RAGState:
        """Enhanced confidence evaluation"""
        # Base confidence on source quality and verification
        base_confidence = 0.7
        
        # Reduce confidence if few sources
        if len(state["filtered_docs"]) < 2:
            base_confidence -= 0.2
        
        # Increase confidence if specific sections found
        if any("section" in doc.get("metadata", {}) for doc in state["filtered_docs"]):
            base_confidence += 0.1
        
        # Reduce confidence if answer contains uncertainty phrases
        uncertainty_phrases = ["may", "might", "appears", "seems", "unclear"]
        if any(phrase in state["answer"].lower() for phrase in uncertainty_phrases):
            base_confidence -= 0.1
        
        state["confidence_score"] = max(0.0, min(1.0, base_confidence))
        return state

    def embed_documents(self, chunks: List[Dict[str, Any]]):
        """Enhanced document embedding with metadata preservation"""
        documents = []
        for chunk in chunks:
            # Extract section information if available
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            
            # Try to extract section number from content
            content = chunk["content"]
            section_match = re.search(r'Section (\d+(?:\.\d+)*)', content)
            if section_match:
                metadata["section"] = section_match.group(1)
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate detailed answer with enhanced information"""
        if self.vector_store is None:
            raise ValueError("No documents embedded yet. Call embed_documents first.")
        
        initial_state = RAGState(
            query=query,
            expanded_queries=[],
            retrieved_docs=[],
            filtered_docs=[],
            context="",
            answer="",
            confidence_score=0.0,
            source_citations=[]
        )
        
        result = self.workflow.invoke(initial_state)
        return {
            "query": result["query"],
            "answer": result["answer"],
            "confidence_score": result["confidence_score"],
            "sources": result["filtered_docs"],
            "context": result["context"],
            "citations": result["source_citations"],
            "expanded_queries": result["expanded_queries"]
        }