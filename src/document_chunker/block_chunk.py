from typing import Dict, Any, List
import re

class BlockChunker:
    """Simple chunker that splits oversized blocks at natural boundaries."""
    
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        
    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main method: check each block and split if too big."""
        result = []
        
        for block in blocks:
            text = block.get('text', '')
            
            # If block is small enough, keep it as is
            if len(text) <= self.max_chunk_size:
                result.append(block)
            else:
                # Split big block into smaller pieces
                small_chunks = self._split_big_text(text)
                
                # Create new block for each piece
                for i, chunk_text in enumerate(small_chunks):
                    new_block = block.copy()
                    new_block['text'] = chunk_text
                    new_block['chunk_id'] = i
                    result.append(new_block)
        
        return result
    
    def _split_big_text(self, text: str) -> List[str]:
        """Try different ways to split text smartly."""
        
        # Method 1: Try splitting by paragraphs (best)
        if '\n\n' in text:
            chunks = self._split_by_paragraphs(text)
            if self._all_chunks_fit(chunks):
                return chunks
        
        # Method 2: Try splitting by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            chunks = self._split_by_sentences(sentences)
            if self._all_chunks_fit(chunks):
                return chunks
        
        # Method 3: Fallback to word splitting (always works)
        return self._split_by_words(text)
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Combine paragraphs into chunks that fit the size limit."""
        paragraphs = text.split('\n\n')
        return self._combine_pieces(paragraphs, separator='\n\n')
    
    def _split_by_sentences(self, sentences: List[str]) -> List[str]:
        """Combine sentences into chunks that fit the size limit."""
        return self._combine_pieces(sentences, separator=' ')
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split by words - this always works even for very long text."""
        words = text.split()
        return self._combine_pieces(words, separator=' ')
    
    def _combine_pieces(self, pieces: List[str], separator: str) -> List[str]:
        """
        Combine text pieces into chunks that don't exceed max size.
        This is the core logic used by all splitting methods.
        """
        chunks = []
        current_chunk_parts = []
        current_size = 0
        
        for piece in pieces:
            piece_size = len(piece)
            separator_size = len(separator) if current_chunk_parts else 0
            
            # Would this piece make the chunk too big?
            if current_size + separator_size + piece_size > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk_parts:
                    chunks.append(separator.join(current_chunk_parts))
                
                # Start new chunk with this piece
                current_chunk_parts = [piece]
                current_size = piece_size
            else:
                # Add piece to current chunk
                current_chunk_parts.append(piece)
                current_size += separator_size + piece_size
        
        # Don't forget the last chunk
        if current_chunk_parts:
            chunks.append(separator.join(current_chunk_parts))
        
        return chunks
    
    def _all_chunks_fit(self, chunks: List[str]) -> bool:
        """Check if all chunks are within size limit."""
        return all(len(chunk) <= self.max_chunk_size for chunk in chunks)

# Usage example
if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Load your processed blocks
    file_path = Path('data_processed/Lux_cssf18_698eng_processed_blocks.json')
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blocks = data.get('blocks', [])
        
        # Chunk the blocks
        chunker = BlockChunker(max_chunk_size=1000)
        chunked_blocks = chunker.chunk_blocks(blocks)
        
        print(f"Original blocks: {len(blocks)}")
        print(f"Chunked blocks: {len(chunked_blocks)}")
        
        # Show which blocks were split
        original_count = len(blocks)
        chunked_count = len(chunked_blocks)
        if chunked_count > original_count:
            print(f"Split {chunked_count - original_count} oversized blocks")
        
        # Save results
        data['blocks'] = chunked_blocks
        output_path = file_path.parent / f"{file_path.stem}_chunked.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to: {output_path}")
    else:
        print(f"File not found: {file_path}")