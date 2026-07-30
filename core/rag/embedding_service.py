import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

logging.getLogger("faiss").setLevel(logging.ERROR)


class EmbeddingService:
    """Service for generating embeddings from text blocks"""

    def __init__(
        self,
        use_local: bool = True,
        local_model: str = "BAAI/bge-large-en-v1.5",  # all-mpnet-base-v2
        use_azure: bool = False,
        online_model: str = "text-embedding-3-large",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        use_cached_embeddings: bool = True,
        #  cached_local_model: bool = True,
        device: str = "cpu",
    ):  # cpu or cuda
        """
        Initialize embedding service

        Args:
            use_local: Use local sentence transformers model
            local_model: Local embedding model name
            use_azure: Use Azure OpenAI instead of OpenAI
            online_model: Online embedding model name (OpenAI or Azure deployment)
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            api_version: Azure API version
            use_cached_embeddings: Whether to cache embeddings to disk
            cached_local_model: Use cached local model files only
        """
        load_dotenv()

        self.use_local = use_local
        self.local_model_name = local_model
        self.use_azure = use_azure
        self.online_model = online_model
        self.use_cached_embeddings = use_cached_embeddings

        # Set final model name for clarity
        self.final_model = self._determine_final_model()

        print(f"🎯 Final embedding model: {self.final_model}")

        if use_local:
            from sentence_transformers import SentenceTransformer

            try:
                # Try loading from cache first
                self.local_model = SentenceTransformer(
                    local_model,
                    cache_folder="./model_cache",
                    local_files_only=True,
                    device=device,
                )
                print(f"✅ Loaded cached model: {local_model}")
            except:
                # If not cached, download it
                print(f"📥 Downloading model: {local_model}")
                self.local_model = SentenceTransformer(
                    local_model,
                    cache_folder="./model_cache",
                    local_files_only=False,
                    device=device,
                )
                print(f"✅ Model downloaded and ready: {local_model}")

        # Initialize online client if needed
        if not use_local:
            if use_azure:
                endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
                api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")

                if not endpoint or not api_key:
                    raise ValueError("Azure OpenAI endpoint and API key required")

                self.client = AzureOpenAI(
                    azure_endpoint=endpoint, api_key=api_key, api_version=api_version
                )
            else:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
                if not api_key or api_key == "your_openai_api_key_here":
                    raise ValueError("OPENAI_API_KEY or GPT_API_KEY environment variable not set in .env")
                self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    def _determine_final_model(self) -> str:
        """Determine the actual model being used"""
        if self.use_local:
            return f"Local: {self.local_model_name}"
        elif self.use_azure:
            return f"Azure OpenAI: {self.online_model}"
        else:
            return f"OpenAI: {self.online_model}"

    def get_provider_suffix(self) -> str:
        """Get suffix for cache file naming based on embedding provider"""
        if self.use_local:
            return f"local_{self.local_model_name.replace('/', '_').replace('.', '_')}"
        elif self.use_azure:
            return f"azure_{self.online_model}"
        else:
            return f"openai_{self.online_model}"

    def enrich_text_with_headers(
        self, chunk: Dict[str, Any], filename: str = None
    ) -> str:
        """Enrich block text with hierarchical headers and optionally filename"""
        enriched_parts = []

        # Add filename if provided
        if filename:
            enriched_parts.append(f"Document: {filename}")

        # Add headers if available
        headers = chunk.get("headers")
        if headers:
            enriched_parts.append(headers)

        # Add the main text
        enriched_parts.append(chunk["text"])

        return "\n\n".join(enriched_parts)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Truncate text to stay safely under OpenAI 8191 token limit (~30,000 chars)
        text = text[:28000] if text else ""
        if self.use_local:
            try:
                return self.local_model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")
                return []
        else:
            try:
                response = self.client.embeddings.create(
                    model=self.online_model, input=text, encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                provider = "Azure OpenAI" if self.use_azure else "OpenAI"
                print(f"{provider} embedding error: {e}")
                return []

    def embed_all_chunks(
        self,
        chunked_doc: Dict[str, Any],
        use_local_file: bool = True,
        cache_path: str = "data/data_processed",
        cache_filename: str = "",
    ) -> Dict[str, Any]:
        """
        Generate embeddings for blocks with caching

        Returns:
            nested dictionaries with structure:
            metadata: {...}
            'embedings': {
                'id': int,
                'content': str,
                'embedding': List[float],
                'block': Dict[str, Any]
             }
        """
        # Build cache file path
        file_name = chunked_doc["filename_without_ext"]
        if not cache_filename:
            cache_filename = file_name

        provider_suffix = self.get_provider_suffix()
        cache_file = f"{cache_path}/{cache_filename}_embds_{provider_suffix}.json"

        # Try loading from cache

        if self.use_cached_embeddings and use_local_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                    print(f"Loaded {len(cached)} embeddings from cache: {cache_file}")
                    return cached
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Generate embeddings
        print(f"Generating embeddings using {self.get_provider_name()}...")
        embeddings = []
        metadata = {
            k: v for k, v in chunked_doc.items() if k not in ["chunks", "blocks"]
        }
        chunks = chunked_doc.get("blocks") or chunked_doc.get("chunks", [])

        for i, chunk in enumerate(chunks):
            print(f"Embedding {i+1}/{len(chunks)}")

            # Enrich text with headers for better context
            content = self.enrich_text_with_headers(chunk, file_name)
            embedding = self.embed_text(content)

            if embedding:
                embedding_item = {
                    "id": i,
                    "content": chunk["text"],  # Store original text
                    "embedding": embedding,
                }
                # Add all other fields from chunk/block (except "text") to top level
                for k, v in chunk.items():
                    if k != "text":
                        embedding_item[k] = v

                embeddings.append(embedding_item)
            else:
                print(f"Failed to embed block {i}")

        # Save to cache
        # Combined structure for saving
        complete_data = {
            "metadata": metadata,  # Document info (filename, pages, etc.)
            "embeddings": embeddings,  # List of embedding objects
        }

        if self.use_cached_embeddings:
            try:
                os.makedirs(cache_path, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(complete_data, f)
                print(
                    f"Saved {len(embeddings)} embeddings to cache: {cache_file} along with metadata"
                )
            except Exception as e:
                print(f"Error saving cache: {e}")

        return complete_data

    def get_provider_name(self) -> str:
        """Get human-readable provider name"""
        if self.use_local:
            name = f"Local ({self.local_model_name})"
        elif self.use_azure:
            name = f"Azure OpenAI ({self.online_model})"
        else:
            name = f"OpenAI ({self.online_model})"

        print(f"🔧 Provider: {name}")
        return name

    def get_config(self) -> Dict[str, Any]:
        """Get configuration information"""
        config = {
            "provider": self.get_provider_name(),
            "use_local": self.use_local,
            "model": self.local_model_name if self.use_local else self.online_model,
            "use_azure": self.use_azure,
            "cache_enabled": self.use_cached_embeddings,
            "provider_suffix": self.get_provider_suffix(),
            "final_model": self.final_model,
        }

        print("📋 Embedding Service Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        return config


if __name__ == "__main__":

    import os
    import sys

    # Add src directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

    from pathlib import Path

    from document_chunker.block_chunker import RegulatoryChunkingSystem
    from document_processing.block_processor import block_processor

    from core.document_processing.readers.pymupdf_reader import PyMuPDFProcessor

    input_pdf = "data/raw/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
    output_dir = Path("data/data_processed")
    output_dir.mkdir(exist_ok=True)

    reader = PyMuPDFProcessor()

    raw_blocks = reader.extract_blocks(input_pdf)

    processor = block_processor(raw_blocks)
    processed_blocks = processor.process_blocks()

    chunker = RegulatoryChunkingSystem(processed_blocks)
    chunks_doc = chunker.chunk_blocks()

    embd_srv = EmbeddingService()
    embd_srv.embed_all_chunks(chunks_doc)
    embd_srv.get_provider_name()
    embd_srv.get_config()
