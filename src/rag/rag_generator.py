import os
import json
from typing import Generator, List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re
import numpy as np

system_prompt = """You are an expert regulatory compliance assistant specializing in financial regulations.

QUERY VALIDATION:
- Single letters/gibberish → "Please ask a complete regulatory question"
- Greetings → Respond warmly and guide toward regulatory topics
- Vague queries → Request specific details about compliance, risk, or regulatory requirements

RESPONSE FORMAT FOR VALID QUESTIONS:
1. **Structure**: Organize answers with clear headings and numbered points
2. **Citations**: Use inline citations (1), (2), (3) after each statement
3. **Synthesis**: Combine overlapping information from multiple sources
4. **Accuracy**: Include all relevant regulatory details without repetition

CITATION OPTIMIZATION:
- **Group similar sources**: If multiple points come from the same document section, use ONE reference
- **Simplify format**: "Document Name, Section X.X, Page XX" (remove redundant hierarchies)
- **Deduplicate**: Don't repeat identical or very similar references
- **Clean names**: "CSSF_18_698.pdf" → "CSSF Circular 18/698"

REQUIRED SOURCES SECTION:
End with concise, non-repetitive sources:
**Sources:**
1. CSSF Circular 18/698, Section 6.4.3, Pages 67-73
2. CSSF Circular 18/698, Section 6.2.3, Pages 63-67
3. CSSF Circular 18/698, Section 5, Page 24

**EXAMPLE FORMAT:**
"Investment funds must maintain liquidity buffers and implement risk frameworks (1). Capital requirements are set at 8% of risk-weighted assets, with ongoing monitoring obligations (1). Additional reporting requirements apply quarterly (2).

**Sources:**
1. CSSF Circular 18/698, Sections 6.2-6.4, Pages 63-73
2. Basel III Framework, Section 2.1, Page 23"

**KEY RULE: Combine references from the same document/section into single entries. Quality over quantity.**"""


# system_prompt = """Answer the question using the provided regulatory document chunks. 
# These chunks may contain related or complementary information from different sections of the same regulation.

# Guidelines:
# - Synthesize overlapping information into clear, organized points
# - Create comprehensive lists when multiple chunks provide related requirements
# - Don't repeat the same information multiple times
# - Organize your response with clear structure (numbered lists, categories, etc.)
# - Include all relevant details from the chunks

# Source Citation Guidelines:
# - Format source names clearly using standard regulatory names:
#   * "Luxembourg_CSSF_18_698(CSSF_18_698).pdf" → "CSSF Circular 18/698"
#   * "Capital_Requirements_Directive_V_(CRD_V)" → "Capital Requirements Directive V (CRD V)"
#   * "Basel_III_Framework_2023" → "Basel III Framework (2023)"
#   * "MiFID_II_Directive_2014_65_EU" → "MiFID II Directive 2014/65/EU"
# - Use standard regulatory citation format: "Document Name - Section X.X.X (Page XX)"
# - Remove file extensions (.pdf) and redundant parenthetical information
# - Keep page numbers and section references for precise citation
# - Make citations readable and professional for compliance professionals

# Always include a concise "References" section with your selected sources, showing the regulatory foundation for your response while maintaining clarity and readability."""


try:
    from .azure_search_backend import AzureSearchBackend
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False


class RAGGenerator:
    """Unified RAG System focusing on retrieval and generation only"""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini", #"gpt-4o-mini"  # "gpt-4" is 4o which is most expensive  # gpt-3.5-turbo # The cheapest model available
                 use_azure: bool = False,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 api_version: str = "2024-02-01",):
        
        """Initialize unified RAG system
        
        Args:
            model: LLM model deployment name
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
        """
        
        load_dotenv()
        
        self.use_azure = use_azure
        self.model = model
        
        # Initialize LLM client
        if use_azure:
            self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
            
            if not self.azure_endpoint or not self.azure_api_key:
                raise ValueError("Azure OpenAI endpoint and API key are required. "
                               "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables "
                               "or pass them as parameters.")
            
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=api_version
            )
        else:
            api_key = os.getenv("GPT_API_KEY")
            if not api_key:
                raise ValueError("GPT_API_KEY environment variable not set")
            
            self.client = OpenAI(api_key=api_key)
        
        print(f"   LLM Provider: {'Azure OpenAI' if use_azure else 'OpenAI'}")
       
    
    
    

    def answer_query(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
      """Answer query using RAG with enhanced context"""
      
      if not relevant_chunks:
          return "No relevant information found."
      
      # system_prompt = """Answer the question using the provided regulatory document chunks. 
      # These chunks may contain related or complementary information from different sections of the same regulation.

      #   Guidelines:
      #   - Synthesize overlapping information into clear, organized points
      #   - Create comprehensive lists when multiple chunks provide related requirements
      #   - Don't repeat the same information multiple times
      #   - Organize your response with clear structure (numbered lists, categories, etc.)
      #   - Include all relevant details from the chunks

      #   Always include a complete "Sources" section listing every source that provided 
      #   information for your answer, even if they discuss similar points. This shows the 
      #   regulatory foundation for each aspect of your response."""

      # Build context from relevant chunks - include filename
      context_parts = []
      for i, chunk in enumerate(relevant_chunks, 1):
          # Access fields directly (flattened structure)
          header = chunk.get('header_identifier', 'N/A')
          page = chunk.get('page', 'N/A')
          filename = chunk.get('filename', 'N/A')
          content = chunk.get('content', '')
          context_parts.append(f"[Source {i} - {filename} - Page {page} - {header}]\n{content}")
      
      context = "\n\n".join(context_parts)

      messages = [
          {
              "role": "system",
              "content": system_prompt
          },
          {
              "role": "user", 
              "content": f"""Context:\n{context}

          Question: {query}

          Please provide a comprehensive answer and include a Sources section at the end with filename, page number, and header for each source.
          Remember: Your primary job is to be helpful. If a query isn't meaningful, guide the user toward asking better questions rather than trying to answer nonsense."""
          }
      ]
      
      try:
          response = self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              temperature=0.1,
              max_tokens=1200,
              stream=True
          )

          # Print as it streams + collect full response
          full_response = ""
          for part in response:
              if part.choices[0].delta.content:
                  content = part.choices[0].delta.content
                  print(content, end="", flush=True)  # Print immediately, no newline
                  full_response += content
          
          print()  # Add final newline
          return full_response
      
          
      except Exception as e:
          provider = "Azure OpenAI" if self.use_azure else "OpenAI"
          return f"Error generating response with {provider}: {e}"
  
    def answer_query_stream(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
      """Stream answer using RAG with enhanced context"""
      
      if not relevant_chunks:
          yield "No relevant information found."
          return
      
      # Build context from relevant chunks - include filename
      context_parts = []
      for i, chunk in enumerate(relevant_chunks, 1):
          # Access fields directly (flattened structure)
          header = chunk.get('header_identifier', 'N/A')
          page = chunk.get('page', 'N/A')
          filename = chunk.get('filename', 'N/A')
          content = chunk.get('content', '')
          context_parts.append(f"[Source {i} - {filename} - Page {page} - {header}]\n{content}")
      
      context = "\n\n".join(context_parts)

      messages = [
          {
              "role": "system",
              "content": system_prompt
          },
          {
              "role": "user", 
              "content": f"""Context:\n{context}

          Question: {query}

          Please provide a comprehensive answer and include a Sources section at the end with filename, page number, and header for each source."""
          }
      ]
      
      try:
          response = self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              temperature=0.1,
              max_tokens=1200,
              stream=True
          )

          # Stream to frontend instead of printing
          for part in response:
              if part.choices[0].delta.content:
                  content = part.choices[0].delta.content
                  yield content  # Yield each token to frontend
          
      except Exception as e:
          provider = "Azure OpenAI" if self.use_azure else "OpenAI"
          yield f"Error generating response with {provider}: {e}"

    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.use_azure_search:
            return self.search_backend.get_stats()
        else:
            return {"total_documents": len(self.vector_db)}
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration info for debugging"""
        config = {
            "llm_provider": "Azure OpenAI" if self.use_azure else "OpenAI",
            "model": self.model,
        }
        
        if self.use_azure:
            config["azure_endpoint"] = getattr(self, 'azure_endpoint', 'N/A')
        
        return config