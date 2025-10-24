import json
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from core.document_processing.chunker.block_chunker import \
    RegulatoryChunkingSystem
from core.document_processing.processor import block_processor
from core.document_processing.readers.pymupdf_reader import PyMuPDFProcessor
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.rag.search_service import SearchService


def main():
    # File paths
    emb_dir = Path("data/data_processed")
    embd_file_list = list(emb_dir.glob("*embds_local_BAAI_bge-large-en-v1_5*.json"))
    doc_list = []
    for file in embd_file_list:
        with open(file, "r") as f:
            doc_list.append(json.load(f))

    chunk_len_list = [len(doc["embeddings"]) for doc in doc_list]
    print(chunk_len_list)
    toatl_chunks = sum(chunk_len_list)
    print(toatl_chunks)

    # with open("data/data_processed/CRD_V_embds_local_BAAI_bge-large-en-v1.5.json", 'r') as f:
    #         doc_list.append(json.load(f))
    # Initialize services
    # Assume All the embeddings exists
    embedding_service = EmbeddingService()
    search_service = SearchService(index_dir="data/indexes")
    rag_system = RAGGenerator()

    # Step 2: Setup search service
    if not search_service.load_indexes():
        print("Building new search indexes...")
        search_service.build_indexes(doc_list)  # Build indexes from chunks
        search_service.save_indexes()  # Save for next time
    else:
        print("Search indexes loaded successfully!")
        search_service.set_chunks(doc_list)  # Set chunks first

    # Step 3: Process query
    query = "What is the exact minimum Common Equity Tier 1 capital ratio required under Basel III, and what additional capital conservation buffer must be maintained on top of this minimum? Please provide the specific percentages and the total effective minimum."
    # query = "According the Capital Requirements Directive (CRD), According to Article 111 (as replaced), who is responsible for supervising a parent credit institution on a consolidated basis if the parent is a parent credit institution in a Member State or an EU parent credit institution?"
    print(f"\nQuery: {query}")

    # Get query embedding
    query_embedding = embedding_service.embed_text(query)

    # Step 4: Search for relevant chunks
    print("\nSearching for relevant chunks...")
    relevant_chunks = search_service.search_documents(query, query_embedding, top_k=12)

    print(f"Found {len(relevant_chunks)} relevant chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        filename = chunk.get("filename", "N/A")
        page = chunk.get("page", "N/A")
        header = chunk.get("header_identifier", "N/A")
        content_preview = chunk.get("content", "")[:100] + "..."
        print(f"{i}. {filename} - Page {page} - {header}")
        print(f"   {content_preview}")

    # Step 5: Generate answer using RAG
    print("\nGenerating answer...")
    print("=" * 80)
    answer = rag_system.answer_query(query, relevant_chunks)
    print(answer)

    print("=" * 80)

    # Step 6: Print stats
    print(f"\nSearch Service Stats: {search_service.get_stats()}")
    print(f"RAG System Config: {rag_system.get_config_info()}")


if __name__ == "__main__":
    main()
