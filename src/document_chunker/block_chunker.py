import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
from pathlib import Path
from typing import Any
import logging
import json
from utils.text_utils import extract_paragraphs, extract_sentences, remove_newlines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    

class RegulatoryChunkingSystem:
    def __init__(self,
                 pdf_content: Dict[str, Any] = None,
                 min_chunk_size: int = 5,
                 max_chunk_size: int = 512,
                 skip_headers:bool = True,
                 model: str = "all-mpnet-base-v2"):
        
        self.pdf_content = pdf_content
        self.skip_headers = skip_headers
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tokenizer = SentenceTransformer(model, device="cpu").tokenizer
        


    def chunk_blocks(self, pdf_content: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text blocks into regulatory chunks based on size constraints.
        
        Args:
            blocks: List of text blocks with 'text' key and other metadata
            
        Returns:
            List of chunks preserving all original metadata plus chunk_id
        """
        if pdf_content:
            self.pdf_content = pdf_content

        blocks = self.pdf_content['blocks']
        chunked_document = {}
        chunked_document = {k: v for k, v in self.pdf_content.items() if k != "blocks"}


        chunks = []
        chunk_id = 0

        for block in blocks:
            if block["is_title"] and self.skip_headers:
                continue
            text = block.get('text', '').strip()
            tokens = self.tokenizer.tokenize(text)
            token_size = len(tokens)+2
            if not text or token_size < self.min_chunk_size:
                continue
                
            if len(text) <= self.max_chunk_size:
                # Preserve all metadata from original block
                chunk = block.copy()  # Copy all original keys
                chunk['chunk_id'] = chunk_id  # Single chunk from this block
                chunk_id += 1
                chunks.append(chunk)
            else:
                # Split oversized text into chunks with overlap
                paragraphs = extract_paragraphs(text)
                current_chunk_text = ""
                
                for paragraph in paragraphs:
                    # Check if adding this paragraph would exceed max_chunk_size
                    test_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + paragraph
                    test_tokens = self.tokenizer.tokenize(test_text)
                    
                    if len(test_tokens) + 2 <= self.max_chunk_size:
                        # Add paragraph to current chunk
                        current_chunk_text = test_text
                    else:
                        # Save current chunk if it has content
                        if current_chunk_text:
                            chunk = block.copy()
                            chunk['text'] = remove_newlines(current_chunk_text)
                            chunk['chunk_id'] = chunk_id
                            chunks.append(chunk)
                            chunk_id += 1
                            current_chunk_text = ""
                        
                        # Handle the current paragraph - split by sentences if too large
                        par_tokens = self.tokenizer.tokenize(paragraph)
                        if len(par_tokens) + 2 <= self.max_chunk_size:
                            # Paragraph fits as a single chunk
                            current_chunk_text = paragraph
                        else:
                            # Split paragraph into sentences
                            sentences = extract_sentences(paragraph)
                            for sentence in sentences:
                                test_text = current_chunk_text + (" " if current_chunk_text else "") + sentence
                                test_tokens = self.tokenizer.tokenize(test_text)
                                
                                if len(test_tokens) + 2 <= self.max_chunk_size:
                                    # Add sentence to current chunk
                                    current_chunk_text = test_text
                                else:
                                    # Save current chunk if it has content
                                    if current_chunk_text:
                                        chunk = block.copy()
                                        chunk['text'] = remove_newlines(current_chunk_text)
                                        chunk['chunk_id'] = chunk_id
                                        chunks.append(chunk)
                                        chunk_id += 1
                                    
                                    # Start new chunk with current sentence
                                    current_chunk_text = sentence
                
                # Save final chunk if it has content
                if current_chunk_text:
                    chunk = block.copy()
                    chunk['text'] = current_chunk_text
                    chunk['chunk_id'] = chunk_id
                    chunk_id += 1
                    chunks.append(chunk)



        chunked_document['chunks'] = chunks
        saving_path = f"data_processed/{self.pdf_content['filename_without_ext']}_chunks.json"
        file_path = Path(saving_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_document, f, indent=4, ensure_ascii=False)

        logger.info(f"Data saved to {file_path}")

        return chunked_document




# if __name__ == "__main__":
    
#     path  = "data_processed/Lux_cssf18_698eng_processed_blocks.json"
#     with open (path, 'r') as f:
#         blocks = json.load(f)
    
#     chunker = RegulatoryChunkingSystem()
#     chunks_doc = chunker.chunk_blocks(blocks)
#     chunks = chunks_doc["chunks"]
#     print(f"Created {len(chunks)} chunks")
#     # Save chunks to a JSON file
#     clean_chunks = [chunk['text'] for chunk in chunks]
#     # with open("data_processed/lux_cssf18_698eng_chunks.json", "w") as f:
#     #     json.dump(clean_chunks, f, indent=2)