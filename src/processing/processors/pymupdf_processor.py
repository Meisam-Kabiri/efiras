# from processing.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
import re

from base import DocumentProcessor, ProcessorConfig, ProcessorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyMuPDFProcessor(DocumentProcessor):
    """Fast processor for text-based PDFs"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.processor_type = ProcessorType.PYMUPDF
    
    def is_available(self) -> bool:
        try:
            import fitz
            return True
        except ImportError:
            return False
    
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            import fitz
            
            doc = fitz.open(str(file_path))
            text = ""
            page_texts = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append(page_text)
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'processor': self.processor_type.value
            }
            
            doc.close()
            
            result = {
                'text': text,
                'page_texts': page_texts,
                'metadata': metadata,
                'extraction_method': 'text_layer'
                }
            
            file_path = Path('src/cleaning/output_processed_text_pymupdf.json')

            # 1. Create the parent directory if it doesn't exist
            # file_path.parent gives you the directory part of the path ('src/')
            # mkdir(parents=True) creates any missing parent directories (e.g., if 'src' itself didn't exist)
            # exist_ok=True prevents an error if the directory already exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            print(f"Data saved to {file_path}")

            return result
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise

    def extract_blocks(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
            """Extract text blocks from the PDF."""
            try:
                import fitz
                
                doc = fitz.open(str(file_path))
                page1 = doc[0]
                width = page1.rect.width
                height = page1.rect.height
                print (f"Page size: {width} x {height}")
                blocks = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    block_list = page.get_text("blocks")
                    
                    for block in block_list:
                        blocks.append({
                            'page': page_num + 1,
                            'bbox': block[:4],
                            'text': block[4]
                        })
                
                doc.close()
                blocks = self.extract_headers(blocks)
                blocks = [block for block in blocks 
                        if not (self.identify_header_footer_blocks(height, block) or len(block["text"]) < 5)
                        ]
                for i, block in enumerate(blocks):
                    
                    # Regex to match "Section/Chapter/Part" followed by numbers and optional text on the same line.
                    # The 'r"^\s*"' at the beginning allows for optional leading whitespace.
                    # The outer parentheses ( ) around the whole pattern make it a capturing group.


        

                    pattern = re.compile(
                        r"^\s*(Part|Chapter|Sub-chapter|Section)\s+"  # Prefix (e.g., "Part", "Chapter")
                        r"([IVXLCDM]+|\d+(?:\.\d+)*)"  # Roman numerals (IV) or digits (4.2.2)
                        r"(?:\.?)"  # Optional dot (e.g., "Chapter 1.")
                        r"\s*"  # Optional space
                        r"([^\n]*?(?=\s*\.{2,}\s*\d*|$))",  # Text until filler dots or end
                        re.IGNORECASE | re.MULTILINE
                    )   

                    match = pattern.match(block['text']) # No need to strip() here if using ^\s* in pattern

                    if match:
                        matched_string = match.group(0).strip() # .group(0) is the entire matched string
                        print(f"Block starts with: '{matched_string}'")


                    if block['text'].endswith(": \n"):
                        blocks[i]['text'] +=  "\n" + blocks[i + 1]['text']
                        del blocks[i + 1] 

                for i in range(200):
                    # Clean up text in each block
                    block = blocks[i]
                    # print(block['text'])
                    # print("\n-------------------------------------------------------------------\n")
                

                file_path = Path('src/cleaning/output_processed_text_pymupdf.json')

                # 1. Create the parent directory if it doesn't exist
                # file_path.parent gives you the directory part of the path ('src/')
                # mkdir(parents=True) creates any missing parent directories (e.g., if 'src' itself didn't exist)
                # exist_ok=True prevents an error if the directory already exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(blocks, f, indent=4, ensure_ascii=False)

                print(f"Data saved to {file_path}")
                return blocks
            
            except Exception as e:
                logger.error(f"PyMuPDF block extraction failed: {e}")
                raise
        
    def identify_header_footer_blocks(self, height, block: Dict[str, Any]) -> bool:
            """Remove header and footer blocks based on their position."""
            bbox = block['bbox']
            y1 = bbox[1]  # Top y-coordinate of the block
            y2 = bbox[3]
            # Check if the block is in the header or footer region
            if y1 < height * 0.08 or (y2 > height * 0.92 and y2-y1 < height * 0.1):
                return True
            
    def extract_headers(self, blocks):
        # Header patterns with hierarchy levels (lower number = higher priority)
        # ^ ensures pattern matches at start of text
        patterns = {
            1: r'(?i)^(?:part|chapter)\s+[ivx\d]+\b',
            2: r'(?i)^sub-?chapter\s+[\d.]+\b',
            3: r'(?i)^section\s+\d+\b',          # Section 4
            4: r'(?i)^section\s+\d+\.\d+\b',     # Section 4.1  
            5: r'(?i)^section\s+\d+\.\d+\.\d+\b', # Section 4.1.1
            6: r'(?i)^sub-?section\s+[\d.]+\b'
        }
        
        current_headers = {}  # level -> header text
        
        for block in blocks:
            text = block['text']
            
            # Check for headers in current block
            for level, pattern in patterns.items():
                if re.search(pattern, text):
                    # Clear lower priority headers and update current level
                    current_headers = {k: v for k, v in current_headers.items() if k < level}
                    current_headers[level] = re.search(pattern, text).group()
                    break
            
            # Assign accumulated headers to block
            block['headers'] = ', '.join(current_headers.values())
        
        return blocks
        
if __name__ == "__main__":
    config = ProcessorConfig(
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        ocr_fallback=True
    )
    
    processor = PyMuPDFProcessor(config)
    
    if processor.is_available():
        result = processor.extract_text("data/regulatory_documents/lu/Lux_cssf18_698eng.pdf")
        result = processor.extract_blocks("data/regulatory_documents/lu/Lux_cssf18_698eng.pdf")
        # print(f"Extraction successful with {processor.processor_type.value}")
        # print(f"Text length: {len(result['text'])} characters")
        # print(f"Pages: {result['metadata']['page_count']}")
    else:
        print("PyMuPDF is not available. Please install the required library.")