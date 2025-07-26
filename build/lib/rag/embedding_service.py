import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI


class EmbeddingService:
    """Service for generating embeddings from text blocks"""
    
    def __init__(self, 
                 use_local: bool = True,
                 local_model: str = "all-mpnet-base-v2",
                 use_azure: bool = False,
                 online_model: str = "text-embedding-3-large",
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 api_version: str = "2024-02-01",
                 use_cached_embeddings: bool = True,
                 cached_local_model: bool = True):
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
        
        # Initialize local model if needed
        if use_local:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(local_model, local_files_only=cached_local_model)
        else:
            self.local_model = None
            
        # Initialize online client if needed
        if not use_local:
            if use_azure:
                endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
                api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
                
                if not endpoint or not api_key:
                    raise ValueError("Azure OpenAI endpoint and API key required")
                
                self.client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
            else:
                api_key = os.getenv("GPT_API_KEY")
                if not api_key:
                    raise ValueError("GPT_API_KEY environment variable not set")
                self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def get_provider_suffix(self) -> str:
        """Get suffix for cache file naming based on embedding provider"""
        if self.use_local:
            return f"local_{self.local_model_name.replace('/', '_')}"
        elif self.use_azure:
            return f"azure_{self.online_model}"
        else:
            return f"openai_{self.online_model}"
    
    def enrich_text_with_headers(self, chunk: Dict[str, Any]) -> str:
        """Enrich block text with hierarchical headers"""
        enriched = chunk.get('headers')
        if enriched:
            return f"{enriched}\n\n{chunk['text']}"
        return chunk['text']
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.use_local:
            try:
                return self.local_model.encode(text).tolist()
            except Exception as e:
                print(f"Local embedding error: {e}")
                return []
        else:
            try:
                response = self.client.embeddings.create(
                    model=self.online_model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            except Exception as e:
                provider = "Azure OpenAI" if self.use_azure else "OpenAI"
                print(f"{provider} embedding error: {e}")
                return []
    
    def embed_all_chunks(self, 
                    chunked_doc: Dict[str, Any], 
                    cache_path: str = "data_processed",
                    cache_filename: str = "embeddings") -> List[Dict[str, Any]]:
        """
        Generate embeddings for blocks with caching
        
        Returns:
            List of dictionaries with structure:
            {
                'id': int,
                'content': str,
                'embedding': List[float],
                'block': Dict[str, Any]
            }
        """
        # Build cache file path
        provider_suffix = self.get_provider_suffix()
        cache_file = f"{cache_path}/{cache_filename}_{provider_suffix}.json"
        
        # Try loading from cache
        if self.use_cached_embeddings and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    print(f"Loaded {len(cached)} embeddings from cache: {cache_file}")
                    return cached
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Generate embeddings
        print(f"Generating embeddings using {self._get_provider_name()}...")
        embeddings = []
        embeddings = {k:v for k, v in chunked_doc.items if (v!='chunks' and v!="blocks")}
        chunks = chunked_doc.get('blocks') or chunked_doc.get('chunks', [])
        
        for i, chunk in enumerate(chunks):
            print(f"Embedding {i+1}/{len(chunks)}")
            
            # Enrich text with headers for better context
            content = self.enrich_text_with_headers(chunk)
            embedding = self.embed_text(content)
            
            if embedding:
                embeddings.append({
                    'id': i,
                    'content': chunk['text'],  # Store original text
                    'embedding': embedding,
                    'block': {k: v for k, v in chunk.items() if k != 'text'}  # Store full block metadata
                })
            else:
                print(f"Failed to embed block {i}")
        
        # Save to cache
        if self.use_cached_embeddings:
            try:
                os.makedirs(cache_path, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(embeddings, f)
                print(f"Saved {len(embeddings)} embeddings to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return embeddings
    
    def _get_provider_name(self) -> str:
        """Get human-readable provider name"""
        if self.use_local:
            return f"Local ({self.local_model_name})"
        elif self.use_azure:
            return f"Azure OpenAI ({self.online_model})"
        else:
            return f"OpenAI ({self.online_model})"
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration information"""
        return {
            "provider": self._get_provider_name(),
            "use_local": self.use_local,
            "model": self.local_model_name if self.use_local else self.online_model,
            "use_azure": self.use_azure,
            "cache_enabled": self.use_cached_embeddings,
            "provider_suffix": self.get_provider_suffix()
        }
    

if __name__ == "__main__":
    
    from pathlib import Path
    from efiras.document_readers.pymupdf_reader import PyMuPDFProcessor
    from efiras.document_processing.block_processor import block_processor
    from efiras.document_chunker.block_chunker import RegulatoryChunkingSystem



    input_pdf = "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
    output_dir = Path("data_processed")
    output_dir.mkdir(exist_ok=True)

      
    reader = PyMuPDFProcessor()
    processor = block_processor()
    chunker = RegulatoryChunkingSystem()

    reader.extract_blocks(input_pdf, output_dir)
    

  
    
    
    # Step 3: Clean and structure the text
    print("3. Cleaning and structuring text...")
    processor = block_processor()
    
    # Process and chunk blocks (includes TOC extraction and header assignment)
    processed_data = processor.process_and_chunk_blocks(raw_result)
    
    
    print(f"   - Extracted TOC entries: {len(processed_data['table_of_contents'])}")
    print(f"   - Processed blocks: {len(processed_data['blocks'])}")
    
    # Step 4: Create manageable chunks
    print("4. Creating manageable chunks...")
    chunker = RegulatoryChunkingSystem(max_chunk_size=512)
    chunked_blocks = chunker.chunk_blocks(processed_data)

    embd_serv = EmbeddingService()