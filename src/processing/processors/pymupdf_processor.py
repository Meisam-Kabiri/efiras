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
                pages = doc.page_count
                path = Path(file_path)
                filename = path.name  # gets the full filename with extension
                filename_without_ext = path.stem  # gets filename without extension
                extension = path.suffix  # gets just the extension
                print (filename, filename_without_ext, extension)
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
                toc = self.extract_toc(blocks)  # Extract TOC if needed
                print(toc)
                [print(f"{key}: {value}") for d in toc for key, value in d.items()]
                blocks = self.extract_headers(blocks)
                block = self.clean_text_for_rag(blocks)
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
                        # print(f"Block starts with: '{matched_string}'")


                    if block['text'].endswith(": \n") or block['text'].endswith(":"):
                        blocks[i]['text'] +=  "\n" + blocks[i + 1]['text']
                        del blocks[i + 1] 

                for i in range(200):
                    # Clean up text in each block
                    block = blocks[i]
                    # print(block['text'])
                    # print("\n-------------------------------------------------------------------\n")
                

                self.enrich_blocks_with_titles(blocks, toc)
                saving_path = f"data_processed/{filename_without_ext}_extracted_blocks.json"
                file_path = Path(saving_path)  # Use Path to handle file paths

                # 1. Create the parent directory if it doesn't exist
                # file_path.parent gives you the directory part of the path ('src/')
                # mkdir(parents=True) creates any missing parent directories (e.g., if 'src' itself didn't exist)
                # exist_ok=True prevents an error if the directory already exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as single JSON with both blocks and TOC
                output = {
                    "document_info": {
                        "filename": filename,
                        "total_pages": pages,
                        "saved_path": str(file_path),
                        "processor": "PyMuPDF",
                        "filename_without_ext":filename_without_ext
                        # "extraction_date": today
                    },
                    "table_of_contents": toc,  # Your extracted TOC
                    "blocks": blocks  # Your text blocks
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=4, ensure_ascii=False)

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
    
    def extract_toc(self, blocks):
        toc = []
        in_toc = False
        
        for block in blocks:
            text = block['text'].strip()
            
            # Detect TOC start (only happens once)
            if not in_toc and 'table of contents' in text.lower():
                in_toc = True
                continue
                
            # Extract TOC entries once we're in TOC mode
            if in_toc:
                # Pattern: title + separators + page number
                # match = re.search(r'^(.*?)\s*[.\-_\s]{2,}\s*(\d+)$', text)
                match = re.search(r'^((?:Part|Chapter|Section|Sub-chapter)\s+[ivxlcdm\d.]+)\.?\s*(.*?)\s*[.\-_\s]{3,}\s*(\d+)\s*$', text, re.IGNORECASE | re.DOTALL)
                if match:
                    # print(f"\n: {match.group()}")
                    header, title, page_num = match.groups()
                    header = header.strip()
                    
                    # Determine level
                    if header.startswith('Part'):
                        level = 1
                    elif header.startswith('Chapter'):
                        level = 2
                    elif header.startswith('Sub-chapter'):
                        level = 3
                    elif header.startswith('Section'):
                        level = 4
                    elif header.startswith('Sub-section'):
                        level = 5
                    else:
                        level = 1
                    
                    
                    toc_dict = {
                        "page": int(page_num), 
                        "level": level,
                        "header": header.strip(),
                        "title": title
                    }
                    toc.append(toc_dict)
        # Sort TOC by page number                        
        return toc
    
    def clean_text_for_rag(self,blocks):
        for block in blocks:
            if 'text' in block:
                text = block['text']
                # Remove invisible/problematic characters
                text = re.sub(r'[\u200b\u200c\u200d\u00ad\u00a0\ufeff\u200f\u200e]', '', text)
                # Remove single \n but keep \n\n as paragraph breaks
                text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                # Clean up multiple spaces
                text = re.sub(r' +', ' ', text)
                # Strip leading/trailing whitespace
                block['text'] = text.strip()
        
        return blocks
    
    
    def find_toc_match(self, header: str, toc: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find matching TOC entry for a given header
        
        Args:
            header: Header string to match (e.g., "Chapter 6", "Sub-section 6.3.2.1")
            toc: Table of contents list
            
        Returns:
            Matching TOC entry or None
        """
        header_clean = header.strip().rstrip(".").strip().lower()
        
        for entry in toc:
            # Match against the 'header' field in TOC
            toc_header = entry.get('header', '').strip().rstrip(".").strip().lower()
            
            # Exact match
            if header_clean == toc_header:
                return entry
            
        return None  
    
    def enrich_blocks_with_titles(self, blocks: List[Dict[str, Any]], toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich blocks by appending TOC titles to each header segment.

        Args:
            blocks: List of parsed document blocks with 'headers'
            toc: List of TOC entries with 'header' and 'title'

        Returns:
            List of enriched blocks with 'enriched_headers' added
        """
        enriched_blocks = blocks.copy()
        for block in enriched_blocks:
            enriched_parts = []
            block['enriched_headers'] = ''  # Initialize enriched headers
            for header in block.get('headers', '').split(','):
                header = header.strip()
                if not header:
                    continue

                toc_entry = self.find_toc_match(header, toc)
                if toc_entry:
                    enriched = f"{header}: {toc_entry.get('title', '')}"
                else:
                    enriched = header  # fallback to original if no match

                enriched_parts.append(enriched)

            block['enriched_headers'] = ', '.join(enriched_parts)

        return enriched_blocks



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
        # result = processor.extract_blocks("data/regulatory_documents/eu/Basel  III.pdf")

        # print(f"Extraction successful with {processor.processor_type.value}")
        # print(f"Text length: {len(result['text'])} characters")
        # print(f"Pages: {result['metadata']['page_count']}")
    else:
        print("PyMuPDF is not available. Please install the required library.")