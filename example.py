"""
Document Processing System - Usage Example

This example demonstrates how to:
1. Process a PDF document using multiple engines
2. Clean and structure the extracted text
3. Create manageable chunks
4. Build a searchable knowledge base
5. Query documents using natural language
"""
import os, sys 
sys.path.append(os.path.abspath("src"))

import json
from pathlib import Path
from src.document_readers.base import DocumentProcessor, ProcessorConfig, ProcessorType

from src.document_processing.block_processor import block_processor
from src.document_chunker.block_chunker import RegulatoryChunkingSystem
from src.rag.rag_simple import RAGSystem
from src.document_processing.manager import *

def main():
    # Configuration
    # input_pdf = "data/regulatory_documents/eu/Basel  III.pdf"
    input_pdf = "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
    output_dir = Path("data_processed")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Document Processing System Example ===\n")
    
    # Step 1: Configure document processor
    print("1. Configuring document processor...")
    config = ProcessorConfig(
        chunk_size=1500,
        overlap=200,
        preserve_formatting=True,
        extract_tables=True,
        ocr_fallback=True
    )
    
    # Step 2: Process PDF document
    print("2. Processing PDF document...")
    manager = DocumentProcessorManager(config)
    
    # Process with automatic engine selection and fallback
    raw_result = manager.process_document(
        input_pdf,
        preferred_processor="PYMUPDF",
        fallback=True
    )
    
    print(f"   - Processed {raw_result['pages']} pages")
    print(f"   - Engine used: {raw_result['processor']}")
    print(f" file name is : {raw_result['filename_without_ext']}")

    
    # Step 3: Clean and structure the text
    print("3. Cleaning and structuring text...")
    processor = block_processor()
    
    # Process and chunk blocks (includes TOC extraction and header assignment)
    processed_data = processor.process_and_chunk_blocks(raw_result)
    
    
    print(f"   - Extracted TOC entries: {len(processed_data['table_of_contents'])}")
    print(f"   - Processed blocks: {len(processed_data['blocks'])}")
    
    # Step 4: Create manageable chunks
    print("4. Creating manageable chunks...")
    chunker = RegulatoryChunkingSystem(max_chunk_size=1500)
    chunked_blocks = chunker.chunk_blocks(processed_data['blocks'])
    
    # # Update processed data with chunked blocks
    # processed_data['blocks'] = chunked_blocks
    
 
    
    print(f"   - Total chunks created: {len(chunked_blocks)}")

    
    # Step 5: Build searchable knowledge base
    print("5. Building searchable knowledge base...")
    
    # Option A: Use local embeddings (faster, no API required)
    rag_local = RAGSystem(use_local_embeddings=True)
    rag_local.add_documents(
        chunked_blocks,
        cache_path=str(output_dir),
        cache_file_name=f"{Path(input_pdf).stem}_embeddings_local"
    )
    
    # Option B: Use OpenAI embeddings (requires API key)
    # rag_online = RAGSystem(use_local_embeddings=False)
    # rag_online.add_documents(
    #     chunked_blocks,
    #     cache_path=str(output_dir),
    #     cache_file_name=f"{Path(input_pdf).stem}_embeddings_online"
    # )
    
    
    # Step 6: Query the document
    print("6. Querying the document...")
    
    # Example queries
    queries = [
        "Can the same conducting officer in an Investment Fund Manager (IFM) be responsible for both the risk management function and the investment management function?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        print( "-" * 50)
        
        # Get answer with sources
        result = rag_local.answer_with_sources(query, top_k=3)
        
        print(f" \n   Answer: {result['answer']} \n")
        print( "." * 50)
        print(f"   Confidence: {result['confidence']:.2f}")
        print( "." * 50)
        print(f"   Sources found: {len(result['sources'])}")

        print( "-" * 50)
        
        # Show top source
        if result['sources']:
            top_source = result['sources'][0]
            print(f"   Top source: Page {top_source['page']}, Headers: {top_source['headers']}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Files saved in: {output_dir}")
    print(f"- Raw processed: {raw_result['filename_without_ext']}")
    print(f"- Embeddings: {Path(input_pdf).stem}_embeddings_local.json")

# def demonstrate_advanced_features():
#     """
#     Additional examples of advanced features
#     """
#     print("\n=== Advanced Features Demo ===")
    
#     # Load existing processed data
#     processed_file = "processed_data/sample_document_chunked.json"
#     if not Path(processed_file).exists():
#         print("Run main() first to create processed data")
#         return
    
#     with open(processed_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     blocks = data['blocks']
    
#     # Initialize RAG system
#     rag = RAGSystem(use_local_embeddings=True)
#     rag.add_documents(blocks, cache_path="processed_data", cache_file_name="sample_embeddings")
    
#     # Advanced query examples
#     print("\nAdvanced Query Examples:")
    
#     # 1. Complex analytical query
#     complex_query = "What are the relationships between risk management and compliance requirements?"
#     result = rag.answer_with_sources(complex_query, top_k=5)
#     print(f"\nComplex Analysis Query:")
#     print(f"Q: {complex_query}")
#     print(f"A: {result['answer'][:300]}...")
    
#     # 2. Specific information extraction
#     specific_query = "What are the specific percentages or numerical requirements mentioned?"
#     result = rag.answer_with_sources(specific_query, top_k=3)
#     print(f"\nSpecific Information Query:")
#     print(f"Q: {specific_query}")
#     print(f"A: {result['answer'][:300]}...")
    
#     # 3. Document structure query
#     structure_query = "What are the main sections or chapters in this document?"
#     result = rag.answer_with_sources(structure_query, top_k=5)
#     print(f"\nDocument Structure Query:")
#     print(f"Q: {structure_query}")
#     print(f"A: {result['answer'][:300]}...")

if __name__ == "__main__":
    # Basic usage example
    main()
    
    # Uncomment to run advanced features demo
    # demonstrate_advanced_features()