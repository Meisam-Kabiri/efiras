import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from core.rag.search_service import SearchService
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.document_processing.readers.pymupdf_reader import PyMuPDFProcessor
from core.document_processing.processor import block_processor
from core.document_processing.chunker import RegulatoryChunkingSystem

def create_local_embeddings_for_all_pdf_in_directory(path:str):
    path = Path(path)
    files = path.glob("*.pdf")
    for file in files:
        print (file)

    embd_srv = EmbeddingService()
def main():
    # File paths
    input_pdf = "data/raw/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
    embeddings_path = 'data/data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
    output_dir = Path("data/data_processed")
    output_dir.mkdir(exist_ok=True)

    # Initialize services
    embedding_service = EmbeddingService()
    search_service = SearchService(index_dir="data/indexes")
    rag_system = RAGGenerator()

    # Step 1: Try to load existing embeddings
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            embeddings_data = json.load(f)
    else:
        print("Embeddings not found, generating new ones...")
        # Process PDF and generate embeddings
        reader = PyMuPDFProcessor()
        raw_blocks = reader.extract_blocks(input_pdf)
        
        processor = block_processor(raw_blocks)
        processed_blocks = processor.process_blocks()
        
        chunker = RegulatoryChunkingSystem(processed_blocks)
        chunks_doc = chunker.chunk_blocks()
        
        embeddings_data = embedding_service.embed_all_chunks(chunks_doc)

    # Step 2: Setup search service
    if not search_service.load_indexes():
        print("Building new search indexes...")
        search_service.build_indexes(embeddings_data)  # Build indexes from chunks
        search_service.save_indexes()  # Save for next time
    else:
        print("Search indexes loaded successfully!")
        search_service.set_chunks(embeddings_data)  # Set chunks first

    # Step 3: Process query
    query = "What monitoring elements must IFM implement for central administration delegation?"
    print(f"\nQuery: {query}")
    
    # Get query embedding
    query_embedding = embedding_service.embed_text(query)
    
    # Step 4: Search for relevant chunks
    print("\nSearching for relevant chunks...")
    relevant_chunks = search_service.search_documents(query, query_embedding, top_k=5)
    
    print(f"Found {len(relevant_chunks)} relevant chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        filename = chunk.get('filename', 'N/A')
        page = chunk.get('page', 'N/A')
        header = chunk.get('header_identifier', 'N/A')
        content_preview = chunk.get('content', '')[:100] + "..."
        print(f"{i}. {filename} - Page {page} - {header}")
        print(f"   {content_preview}")
    
    # Step 5: Generate answer using RAG
    print("\nGenerating answer...")
    print("=" * 80)
    answer = rag_system.answer_query(query, relevant_chunks)
    print("=" * 80)
    
    # Step 6: Print stats
    print(f"\nSearch Service Stats: {search_service.get_stats()}")
    print(f"RAG System Config: {rag_system.get_config_info()}")

if __name__ == "__main__":
    main()