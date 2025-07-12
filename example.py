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
from src.rag.azure_openai_rag import AzureOpenAIRAGSystem
from src.rag.unified_rag import UnifiedRAGSystem

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
        chunk_size=2000,
        overlap=100,
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
    chunked_blocks = chunker.chunk_blocks(processed_data)
    
    # # Update processed data with chunked blocks
    # processed_data['blocks'] = chunked_blocks
    
 
    
    print(f"   - Total chunks created: {len(chunked_blocks)}")

    
    # Step 5: Build searchable knowledge base
    print("5. Building searchable knowledge base...")
    
    # Option A: Use OpenAI with local embeddings
    # rag_local = UnifiedRAGSystem(use_local_embeddings=True, use_azure=False)

    # Option B: Use Azure OpenAI with local embeddings  
    rag_local = UnifiedRAGSystem(use_local_embeddings=True, use_azure=True, model="gpt-35-turbo")

    # Option C: Use original Azure OpenAI class (for compatibility)
    # rag_local = AzureOpenAIRAGSystem(use_local_embeddings=True)
    
    rag_local.add_documents(
        chunked_blocks,
        cache_path=str(output_dir),
        cache_file_name=f"{Path(input_pdf).stem}_embeddings_local"
    )
    
    
    # Step 6: Query the document
    print("6. Querying the document...")
    
    # Example queries
    queries = [
        "What monitoring elements must IFM implement for central administration delegation?",
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

if __name__ == "__main__":
    # Basic usage example
    main()
    
    # Uncomment to run advanced features demo
    # demonstrate_advanced_features()