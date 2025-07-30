import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from rag.search_service import SearchService
from rag.embedding_service import EmbeddingService
from rag.rag_generator import RAGGenerator
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem

def create_local_embeddings_for_all_pdf_in_directory(path:str):
    path = Path(path)
    files = path.glob("*.pdf")
    embd_srv = EmbeddingService(device = 'cuda')  # Create once
    for file in files:
        # print (file.name)
        # print (file.stem)

        # Step 1: Try to load existing embeddings
        embd_dir = Path("data_processed")
        if list(embd_dir.glob(f"{file.stem}*embd*.json")):
          print("file exists")
        else:
           
        # Create Embedding for the file: 
          print("Embeddings not found, generating new ones...")
          # Process PDF and generate embeddings
          # removing the indexes
          index_path = Path("indexes")
          if index_path.exists() and index_path.is_dir():
            index_path.rmdir()  # Deletes the empty directory
            print(f"{index_path} has been deleted.")
          else:
             print(f"{index_path} does not exist or is not a directory.")
             
          reader = PyMuPDFProcessor(enable_save = False)
          raw_blocks = reader.extract_blocks(file)
          
          processor = block_processor(raw_blocks, enable_save = False)
          processed_blocks = processor.process_blocks()
          
          chunker = RegulatoryChunkingSystem(processed_blocks, enable_save = False)
          chunks_doc = chunker.chunk_blocks()
          
          # embd_srv = EmbeddingService()
          embeddings_data = embd_srv.embed_all_chunks(chunks_doc)

                  # Clear memory after each file
          import torch
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          # del embd_srv

def delete_chunk_block_files_pathlib(directory_path):
    directory = Path(directory_path)
    
    # Find files containing 'chunks' or 'blocks' with .json extension
    chunk_files = directory.glob("*chunks*.json")
    block_files = directory.glob("*blocks*.json")
    
    deleted_count = 0
    for file_path in list(chunk_files) + list(block_files):
        try:
            file_path.unlink()  # Delete the file
            print(f"Deleted: {file_path}")
            deleted_count += 1
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Total files deleted: {deleted_count}")



path = "data/regulatory_documents/eu"
create_local_embeddings_for_all_pdf_in_directory(path)
# delete_chunk_block_files_pathlib("data_processed")

