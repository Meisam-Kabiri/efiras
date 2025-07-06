import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
from typing import Any

    

class RegulatoryChunkingSystem:
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 500,
                 overlap_percentage: float = 0.15,
                 semantic_model: str = "all-MiniLM-L6-v2"):
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap_size = int(max_chunk_size * overlap_percentage)
        self.overlap_percentage = overlap_percentage
        self.semantic_model = SentenceTransformer(semantic_model)
        


    def block_chunker(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Chunk text blocks into regulatory chunks based on size constraints.
        
        Args:
            blocks: List of text blocks with 'text' key
            
        Returns:
            List of chunks with 'content' key
        """
        chunks = []
        
        for block in blocks:
            text = block.get('text', '').strip()
            if not text or len(text) < self.min_chunk_size:
                continue
                
            if len(text) <= self.max_chunk_size:
                chunks.append({'content': text})
            else:
                # Split oversized text into chunks with overlap
                chunks.extend(self._split_large_text(text))
        
        return chunks

    def _split_large_text(self, text: str) -> List[Dict[str, str]]:
        """Split large text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append({'content': chunk})
            
            # Move start position with overlap consideration
            start = end - self.chunk_overlap_size if end < len(text) else len(text)
        
        return chunks
    


if __name__ == "__main__":
    # load the the list of blocks from a json File
    import json
    from pathlib import Path
    file_path = Path('src/cleaning/output_processed_text_pymupdf.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)
    for block in blocks:
        block["text"] = block["text"].replace("\n", " ")
    print(blocks[4]['text'])

    chunker = RegulatoryChunkingSystem()
    chunks = chunker.block_chunker(blocks)
    print(f"Created {len(chunks)} chunks")
    # Save chunks to a JSON file
    clean_chunks = [chunk['content'] for chunk in chunks]
    with open("data/chunks/lux_cssf18_698eng_chunks.json", "w") as f:
        json.dump(clean_chunks, f, indent=2)