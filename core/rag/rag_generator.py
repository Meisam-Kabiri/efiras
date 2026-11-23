import os
from typing import Any, Dict, Generator, List, Optional, Union

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from core.rag.router import doc_names

SYSTEM_PROMPT = """
You are an expert regulatory compliance assistant specializing in financial regulations.

FORMATTING RULES:
- Write in plain text only - no markdown formatting
- Use line breaks and indentation for structure
- Start new topics on new lines with clear topic sentences
- Use numbered lists only when absolutely necessary
- Never use ** for bold or ## for headers
- Let content flow naturally in paragraphs

RELEVANCE FILTERING RULE:
- You will receive multiple document chunks, but NOT ALL are relevant to the question
- ONLY use chunks that directly answer or support the specific question asked
- Do NOT cite chunks just because they exist - they must be relevant to YOUR answer
- Ignore chunks that are off-topic or don't contribute to answering the question
- Better to cite fewer relevant sources than many irrelevant ones

GROUNDING RULES (MOST IMPORTANT):
- Use ONLY the text of the provided source chunks. Never add facts, numbers, thresholds or
  requirements from your own knowledge, even if you believe them to be correct.
- Every factual claim must be supported by the specific chunk you cite for it. Never attach a
  citation to a claim that the cited chunk does not state.
- Always name the document a requirement actually comes from. For example, the CRR is the EU law
  implementing Basel III: if the answer comes from a CRR chunk, say "Under the CRR..." or
  "The CRR, which implements Basel III in the EU, requires...", never "Under Basel III...".
- If the question asks about a specific document and the chunks from that document do not contain
  the answer, say so plainly in the first sentence (for example: "The Basel III text in the
  knowledge base does not cover this."). Only then, and only if chunks from other documents do
  answer it, add what those documents say, clearly labelled with their names.
- If the chunks only mention the topic in passing or in a different context (for example, internal
  ratings mentioned inside counterparty credit risk rules), do not present that as the document's
  general requirements on the topic. Say the document does not set out those requirements, then
  describe only what it does say and in which context.
- If no chunk answers the question, say that the knowledge base does not contain the answer and
  do not guess.

CROSS-REFERENCES AND COMPARISONS:
- Some sources are marked "Referenced by: X". They were added because source X (or the question)
  refers to them. Use them to explain what a referenced article or recommendation says when that
  helps answer the question, e.g. "Article X requires ... (1). The Article Y it refers to
  defines ... (2)". Ignore them if they are not relevant.
- When comparing documents, attribute each point to its own document and cite it, state where
  they agree and where they differ, and say if one of them does not address a point.

QUERY VALIDATION:
- Single letters or gibberish → "Please ask a complete regulatory question"
- Greetings → Respond warmly and guide toward regulatory topics  
- Vague queries → Request specific details about compliance, risk, or regulatory requirements

RESPONSE APPROACH:
Write responses as flowing, professional text that reads like an expert consultant's analysis. 
Start with the key insight, then provide supporting details in logical order. 
Use natural paragraph breaks and transitions.

CRITICAL CITATION RULES:
- You will receive multiple source chunks numbered [Source 1], [Source 2], etc.
- ONLY cite sources that you actually reference in your answer
- Your inline citations (1), (2), (3) must EXACTLY match the numbers in your final Sources list
- If you cite (1) in text, then source 1 must be listed in Sources
- If you cite (3) in text, then source 3 must be listed in Sources
- NEVER cite a number that doesn't appear in your Sources section
- Start numbering your Sources from 1, regardless of the original source numbers

INLINE CITATION USAGE:
- Place citations (1), (2), (3) immediately after each factual claim
- Use citations for every regulatory requirement or specific obligation
- Multiple citations for claims supported by multiple sources: (1)(2)
- Do not cluster all citations at the end of paragraphs

EXAMPLE RESPONSE STYLE:
"SFDR Article 8 requires financial market participants to provide comprehensive disclosure about 
the environmental and social characteristics of their investment products. The core principle is 
transparency - investors need clear information to make informed sustainable investment decisions.

Pre-contractual documents must describe how the product meets its sustainability objectives (1). 
This includes detailed methodology for any sustainability indices used in the product documentation (1).

All participants must maintain current information on their websites describing each product's 
environmental or social characteristics (2). The disclosures must also indicate alignment with 
EU Taxonomy requirements, specifically detailing which underlying investments qualify as 
environmentally sustainable economic activities (3).

Ongoing obligations include regular updates on sustainability performance and annual reporting 
requirements for qualifying social entrepreneurship funds (4)(5).

Sources:
1. Sustainable Finance Disclosure Regulation (SFDR), Article 8
2. Sustainable Finance Disclosure Regulation (SFDR), Article 11
3. European Union Taxonomy Regulation (EU Taxonomy), Article 5
4. Sustainable Finance Disclosure Regulation (SFDR), Article 9"

CRITICAL: Write as continuous, professional prose. No bold formatting, no markdown headers, 
no bullet points unless absolutely essential for clarity.
"""


class RAGGenerator:
    """Unified RAG System focusing on retrieval and generation only"""

    # Constants
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_API_VERSION = "2024-02-01"
    DEFAULT_MODEL = "gpt-4.1"  # answer model; follows "only use the sources" rules better than gpt-4o-mini (~13x cost, ~$0.05 per answer). Router stays on gpt-4o-mini.

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
    ):
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
                raise ValueError(
                    "Azure OpenAI endpoint and API key are required. "
                    "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables "
                    "or pass them as parameters."
                )

            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=api_version,
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or GPT_API_KEY environment variable not set in .env")

            self.client = OpenAI(api_key=api_key)

        print(f"   LLM Provider: {'Azure OpenAI' if use_azure else 'OpenAI'}")

    def _format_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string for LLM."""
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            header = chunk.get("header_identifier") or "N/A"
            section = chunk.get("headers") or ""
            page = chunk.get("page")
            doc_id = chunk.get("doc_id") or chunk.get("filename") or "N/A"
            document = doc_names([doc_id])[0]
            content = chunk.get("content", "")
            label = f"[Source {i} - Document: {document} - Citation: {header}"
            if section:
                label += f" - Section: {section}"
            if chunk.get("is_reference"):
                label += f" - Referenced by: {chunk.get('referenced_from')}"
            if page:
                label += f" - Page {page}"
            label += "]"
            context_parts.append(f"{label}\n{content}")
        return "\n\n".join(context_parts)

    def _build_messages(
        self, query: str, context: str, target_documents: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Build messages for LLM API call."""
        scope_note = ""
        if target_documents:
            scope_note = (
                "\nThe question is about: " + "; ".join(target_documents) + ". "
                "If the sources from that document do not contain the answer, say so first.\n"
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Context:\n{context}

Question: {query}
{scope_note}
CRITICAL CITATION REQUIREMENTS:
- You will see sources numbered [Source 1], [Source 2], etc. in the context
- In your answer, you MUST renumber them starting from 1 for only the sources you actually use
- Your inline citations (1), (2), (3) must EXACTLY match the numbers in your final Sources section
- Example: If you only use [Source 2] and [Source 11] from context, cite them as (1) and (2) in your text, then list them as "1." and "2." in Sources
- NEVER cite a number that doesn't appear in your Sources section
- NEVER use the original source numbers from the context in your citations

Please provide a comprehensive answer using only the sources above, with proper citations.""",
            },
        ]

    def _generate_response_sync(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        target_documents: Optional[List[str]] = None,
    ) -> str:
        """Generate non-streaming response."""
        if not relevant_chunks:
            return "No relevant information found."

        context = self._format_context(relevant_chunks)
        messages = self._build_messages(query, context, target_documents)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.DEFAULT_TEMPERATURE,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                stream=True,
            )

            full_response = ""
            for part in response:
                if part.choices[0].delta.content:
                    content = part.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()
            return full_response

        except Exception as e:
            provider = "Azure OpenAI" if self.use_azure else "OpenAI"
            return f"Error generating response with {provider}: {e}"

    def _generate_response_stream(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        target_documents: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate streaming response."""
        if not relevant_chunks:
            yield "No relevant information found."
            return

        context = self._format_context(relevant_chunks)
        messages = self._build_messages(query, context, target_documents)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.DEFAULT_TEMPERATURE,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                stream=True,
            )

            for part in response:
                if part.choices[0].delta.content:
                    yield part.choices[0].delta.content

        except Exception as e:
            provider = "Azure OpenAI" if self.use_azure else "OpenAI"
            yield f"Error generating response with {provider}: {e}"

    def answer_query(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        target_documents: Optional[List[str]] = None,
    ) -> str:
        """Answer query using RAG with enhanced context."""
        return self._generate_response_sync(query, relevant_chunks, target_documents)

    def answer_query_stream(
        self,
        query: str,
        relevant_chunks: List[Dict[str, Any]],
        target_documents: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Stream answer using RAG with enhanced context."""
        return self._generate_response_stream(query, relevant_chunks, target_documents)

    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {"message": "Stats method needs implementation"}

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration info for debugging"""
        config = {
            "llm_provider": "Azure OpenAI" if self.use_azure else "OpenAI",
            "model": self.model,
        }

        if self.use_azure:
            config["azure_endpoint"] = getattr(self, "azure_endpoint", "N/A")

        return config
