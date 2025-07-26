# from document_readers.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class block_processor():
    """Fast processor for text-based PDFs"""
    
    def __init__(self):
        pass

    def process_and_chunk_blocks(self, pdf_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and chunk the extracted blocks of text.
        Merges blocks with colons, identifies headers/footers, and chunks text.
        """
        if pdf_content is not None:
            blocks = pdf_content.get("blocks")
            height = pdf_content.get("height", 0)
            width = pdf_content.get("width", 0)
            filename = pdf_content.get("filename", "")
            filename_without_ext = pdf_content.get("filename_without_ext", "")
            pages = pdf_content.get("pages", 0)
            extension = pdf_content.get("extension", "")   

        if not blocks:
            logger.warning("No blocks found in the PDF content.")
            return []
        
        toc = self._extract_toc(blocks)  # Extract TOC if needed
        blocks = [block for block in blocks if not block.get('is_toc', False)] #remove the toc from blocks

        # [print(f"{key}: {value}") for d in toc for key, value in d.items()]

        blocks = self._extract_headers(blocks)
        block = self._clean_text(blocks)
        blocks = [block for block in blocks 
                if not (self._identify_header_footer_blocks(height, block) or len(block["text"]) < 5)
                ]
        


        blocks = self._merge_blocks_with_colon_pattern(blocks)

        
        

        self._enrich_blocks_with_titles(blocks, toc)
        self.reattach_split_paragraphs_across_pages(blocks)

        self._save_processed_blocks(
            filename=filename,
            filename_without_ext=filename_without_ext,
            pages=pages,
            toc=toc,
            blocks=blocks)
        
        return {
                "height": height,
                "width": width,
                "filename": filename,
                "filename_without_ext": filename_without_ext,
                "processor": "PyMuPDF",
                "extension": extension,
                "pages": pages,
                "blocks": blocks,
                "table_of_contents": toc
                    }
       
    def _merge_blocks_with_colon_pattern(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merges blocks that end with a colon (e.g., section headers) with the next block's text.
        Also detects section/chapter headers using regex.
        """
        i = 0
        pattern = re.compile(
            r"^\s*(Part|Chapter|Sub-chapter|Section)\s+"      # Prefix
            r"([IVXLCDM]+|\d+(?:\.\d+)*)"                     # Roman numerals or digits
            r"(?:\.?)\s*"                                     # Optional dot and space
            r"([^\n]*?(?=\s*\.{2,}\s*\d*|$))",                # Title content
            re.IGNORECASE | re.MULTILINE
        )

        while i < len(blocks) - 1:
            text = blocks[i]['text']
            
            # Check for pattern match (but don't do anything with it unless needed)
            match = pattern.match(text)
            if match:
                matched_string = match.group(0).strip()
                # Optionally store or log this if needed
                # logger.debug(f"Detected header: {matched_string}")

            # Merge if block ends with a colon
            if text.endswith(": \n") or text.endswith(":"):
                next_text = blocks[i + 1]['text']
                # Add \n\n if next block starts a new paragraph (numbered items, letters, capitals)
                if re.match(r'^(\d+[\.\)]|[a-z][\.\)]|\([a-z]\)|\(\d+\)|[A-Z])', next_text.strip()):
                    blocks[i]['text'] += "\n\n" + next_text
                else:
                    blocks[i]['text'] += next_text
                del blocks[i + 1]
            else:
                i += 1

        return blocks
    
    def _split_definition_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split blocks that contain multiple definitions into separate blocks"""
        new_blocks = []
        
        for block in blocks:
            text = block['text']
            
            # Split on numbered definitions, but only when they start with space/whitespace (not in "Article 101(4)")
            parts = re.split(r'(\s+\d+\)\s*")', text)
            
            if len(parts) > 1:  # Multiple definitions found
                combined_text = parts[0]  # Start with first part
                for i in range(1, len(parts), 2):  # Every other part starting from 1
                    if i + 1 < len(parts):
                        definition_start = parts[i]  # The space + number)
                        definition_content = parts[i + 1]  # The content
                        combined_text += '\n\n' + definition_start.strip() + definition_content
                
                new_block = block.copy()
                new_block['text'] = combined_text
                new_blocks.append(new_block)
            else:
                # Keep block as-is
                new_blocks.append(block)
        
        return new_blocks
   
    def _identify_header_footer_blocks(self, height, block: Dict[str, Any]) -> bool:
            """Remove header and footer blocks based on their position."""
            bbox = block['bbox']
            y1 = bbox[1]  # Top y-coordinate of the block
            y2 = bbox[3]
            # Check if the block is in the header or footer region
            if y1 < height * 0.05 or (y2 > height * 0.95 and y2-y1 < height * 0.1):
                return True
    def _extract_headers(self, blocks):
          """Extract headers and merge blocks when needed, return modified blocks list"""
          
          # Header patterns (lower number = higher hierarchy)
          patterns = {
              1: r'(?i)^(title|book|part)\s+([ivxlcdm\d]+)',
              2: r'(?i)^(chapter|division)\s+([ivxlcdm\d]+)', 
              3: r'(?i)^(section|sub-?part)\s+([ivxlcdm\d]+(?:\.\d+)*)',
              4: r'(?i)^(article|art\.?)\s+([\d\(\)]+)',
              5: r'(?i)^(sub-?section|subsection)\s+([\d.]+)',
              6: r'^(\(\d+\))',
              7: r'^(\d+)\.',
              8: r'^([a-z])\)',
          }
          
          # Non-hierarchical title patterns
          title_patterns = [
              r'^[A-Z][A-Z\s]{8,100}$',     # ALL CAPS TITLES
              r'^[A-Z][a-z\s]{8,50}$',      # Title Case Headers
          ]
          
          current_headers = {}
          result_blocks = []
          i = 0
          
          while i < len(blocks):
              block = blocks[i]
              text = block['text'].strip()
              header_found = False
              
              # Check hierarchical patterns
              for level, pattern in patterns.items():
                  match = re.match(pattern, text)
                  if match:
                      identifier = match.group().rstrip('.')
                      remaining = text[match.end():].strip('. ')
                      
                      # Determine header text and how many blocks to merge
                      if remaining and len(remaining) > 5:
                          # Single block: "Section 1. Title" -> "Section 1 : Title"
                          header_text = f"{identifier} : {remaining}"
                          blocks_to_merge = 1
                          
                      elif i + 1 < len(blocks) and self._is_title_block(blocks[i + 1]['text'].strip(), blocks[i + 1]):
                          # Multi-block: "Section 1" + "Title" -> "Section 1 : Title"
                          next_text = blocks[i + 1]['text'].strip()
                          header_text = f"{identifier} : {next_text}"
                          blocks_to_merge = 2
                          
                      else:
                          # Just identifier: "Section 1"
                          header_text = identifier
                          blocks_to_merge = 1
                      
                      # Update hierarchy
                      current_headers = {k: v for k, v in current_headers.items() if k < level}
                      current_headers[level] = header_text
                      
                      # Modify the first block with merged content
                      block['text'] = header_text
                      block['headers'] = " > ".join(current_headers.values())
                      block['header_identifier'] = " > ".join([self._extract_header_identifier(v) for v in current_headers.values()])
                      block['is_hierarchical'] = True
                      block['is_title'] = True
                      block['merged_from'] = blocks_to_merge
                      
                      result_blocks.append(block)
                      i += blocks_to_merge  # Skip merged blocks
                      header_found = True
                      break
              
              if not header_found:
                  # Check for non-hierarchical titles
                  is_title = (any(re.match(pattern, text) for pattern in title_patterns) or 
                            (hasattr(block, 'is_diff_format') and block.get('is_diff_format') == 'true'))
                  
                  # Add header fields to block (don't modify text)
                  block['headers'] = " > ".join(current_headers.values()) if current_headers else ""
                  block['header_identifier'] = " > ".join([self._extract_header_identifier(v) for v in current_headers.values()])
                  block['is_hierarchical'] = False
                  block['is_title'] = is_title
                  block['merged_from'] = 1
                  
                  result_blocks.append(block)
                  i += 1
          
          return result_blocks

    def _is_title_block(self, text, block):
      """Check if block should be treated as a title"""
      if len(text) < 5 or len(text) > 200:
          return False
      
      if hasattr(block, 'is_diff_format') and block.get('is_diff_format') == 'true':
          return True
      
      return (not re.match(r'^\d+\.', text) and 
              not re.match(r'^\([0-9]+\)', text) and
              not text.islower() and
              len(text.split()) >= 2)

    def _extract_header_identifier(self, header_text):
        """Extract just the identifier part from header text"""
        if ' : ' in header_text:
            return header_text.split(' : ')[0]
        return header_text
      
    def _build_header_path(self, current_headers):
        """Build meaningful header path avoiding overly verbose paths"""
        if not current_headers:
            return ''
        
        # Sort by hierarchy level and select most meaningful ones
        sorted_headers = sorted(current_headers.items())
        
        # For efficiency, limit to maximum 3 hierarchy levels
        selected_headers = []
        for level, header in sorted_headers:
            # Skip overly long descriptive headers in middle of path
            if len(selected_headers) > 0 and len(header) > 80:
                continue
            selected_headers.append(header)
            if len(selected_headers) >= 3:
                break
        
        return ' > '.join(selected_headers)
    
    def _extract_toc(self, blocks):
        toc = []
        in_toc = False
        
        for block in blocks:
            text = block['text'].strip()
            
            # Detect TOC start (only happens once)
            if not in_toc and 'table of contents' in text.lower():
                in_toc = True
                block['is_toc'] = True
                continue
                
            match = re.search(
                r'^((?:Part|Chapter|Section|Sub-chapter|Sub-section|ANNEX)\s+[ivxlcdm\d.]+)\.?\s*(.*?)\s*(?:[.\-_\s]{3,})?\s*(\d+)\s*$',
                text, re.IGNORECASE | re.DOTALL
            )

            if match:
                header, title, page_num = match.groups()

                # Check that after the page number there's only whitespace or end of string
                page_end_pos = text.rfind(page_num) + len(page_num)
                remaining_text = text[page_end_pos:].strip()

                if remaining_text == "":
                    header = header.strip()
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
                        "header": header,
                        "title": title
                    }
                    block['is_toc'] = True
                    toc.append(toc_dict)

            else:
                # Fallback: TOC-style entries like "1. General section.............. 87"
                match2 = re.search(
                    r'^(.*?)\s*[.\-_\s]{3,}\s*(\d+)\s*$',
                    text, re.IGNORECASE | re.DOTALL
                )

                if match2:
                    title, page_num = match2.groups()
                    # Minimal heuristic for level based on numbering
                    header = title.strip().split()[0]
                    if re.match(r'\d+(\.\d+)*', header):  # e.g., "2", "1.1"
                        level = 2
                    else:
                        level = 1

                    toc_dict = {
                        "page": int(page_num),
                        "level": level,
                        "header": header,
                        "title": title.strip()
                    }
                    block['is_toc'] = True
                    toc.append(toc_dict)
        return toc

    
    def _clean_text(self,blocks):
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
                """Simple bullet point cleaning"""
                # Replace bullet points with line breaks or spaces
                text = text.replace('•', '\n')  # or just ' ' for space
                text = text.replace(';', '.')   # Replace semicolons with periods
                
                block['text'] = text.strip()
        
        return blocks
    
    def _save_processed_blocks(
        self,
        filename: str,
        filename_without_ext: str,
        pages: int,
        toc: List[Dict[str, Any]],
        blocks: List[Dict[str, Any]]
        ) -> None:
        """
        Save extracted document blocks and TOC to a JSON file.

        Args:
            filename: Full file name with extension.
            filename_without_ext: File name without extension.
            pages: Total number of pages.
            toc: Table of contents entries.
            blocks: Extracted text blocks.
        """
        saving_path = f"data_processed/{filename_without_ext}_processed_blocks.json"
        file_path = Path(saving_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "document_info": {
                "filename": filename,
                "total_pages": pages,
                "saved_path": str(file_path),
                "processor": "PyMuPDF",
                "filename_without_ext": filename_without_ext
            },
            "filename_without_ext": filename_without_ext,
            "table_of_contents": toc,
            "blocks": blocks
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(f"Data saved to {file_path}")


    def _find_toc_match(self, header: str, toc: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    
    def _enrich_blocks_with_titles(self, blocks: List[Dict[str, Any]], toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich blocks by appending TOC titles to header segments found in TOC.

        Args:
            blocks: List of parsed document blocks with 'headers'
            toc: List of TOC entries with 'header' and 'title'

        Returns:
            List of enriched blocks with 'enriched_headers' added
        """
        enriched_blocks = blocks.copy()
        
        # If no TOC, just copy headers to enriched_headers
        if not toc:
            for block in enriched_blocks:
                block['enriched_headers'] = block.get('headers', '')
            return enriched_blocks
        
        for block in enriched_blocks:
            original_header = block.get('headers', '')
            if not original_header:
                block['enriched_headers'] = ''
                continue
            
            # Split hierarchical header by " > "
            header_parts = [part.strip() for part in original_header.split(' > ')]
            enriched_parts = []
            
            for part in header_parts:
                # Look up this part in TOC
                toc_entry = self._find_toc_match(part, toc)
                if toc_entry and toc_entry.get('title'):
                    # Enrich with TOC title and clean up newlines
                    title = toc_entry.get('title', '').replace('\n', ' ').strip()
                    enriched_part = f"{part}. {title}"
                else:
                    # Keep original if not found in TOC, but clean newlines
                    enriched_part = part.replace('\n', ' ').strip()
                
                enriched_parts.append(enriched_part)
            
            # Rebuild the hierarchical header
            block['enriched_headers'] = ' > '.join(enriched_parts)

        return enriched_blocks


    # src/processing/processors/pymupdf_processor.pydef reattach_split_paragraphs_across_pages(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def reattach_split_paragraphs_across_pages(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:   
        """
        Merge blocks across page boundaries that are likely part of the same paragraph.
        
        Args:
            blocks: List of extracted text blocks, each with 'text' and 'page' keys at minimum.
        Returns:
            List of blocks with likely split paragraphs reattached.
        """
        if len(blocks) <= 1:
            return blocks.copy()
        
        def starts_like_list_item(text: str) -> bool:
            """Check if text starts like a list item."""
            if not text.strip():
                return False
            # Match formats like "1.", "1.1)", "-", "•", "a)", "A.", etc.
            return bool(re.match(r'^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[•\-●‣▪▫⁃])\s+', text))
        
        def starts_with_special_word(text: str) -> bool:
            """Check if text starts with special formatting keywords."""
            keywords = {
                "note", "definition", "warning", "example", "conclusion", "summary",
                "important", "caution", "tip", "reminder", "chapter", "section",
                "figure", "table", "appendix", "reference", "bibliography"
            }
            words = text.strip().lower().split()
            if not words:
                return False
            return words[0] in keywords
        
        def is_heading_like(text: str) -> bool:
            """Check if text looks like a heading."""
            text = text.strip()
            if not text:
                return False
            
            # Check for common heading patterns
            heading_patterns = [
                r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
                r'^[A-Z][A-Z\s]+$',  # "CHAPTER ONE" (all caps)
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Chapter One" (title case)
            ]
            
            return any(re.match(pattern, text) for pattern in heading_patterns)
        
        def ends_with_hyphen(text: str) -> bool:
            """Check if text ends with a hyphen (word continuation)."""
            return text.rstrip().endswith('-')
        
        def starts_with_lowercase_or_continuation(text: str) -> bool:
            """Check if text starts with lowercase (likely continuation)."""
            text = text.strip()
            return text and text[0].islower()
        
        def should_merge_blocks(current: Dict[str, Any], next_block: Dict[str, Any]) -> bool:
            """Determine if two blocks should be merged."""
            current_text = current['text'].strip()
            next_text = next_block['text'].strip()
            
            # Don't merge empty blocks
            if not current_text or not next_text:
                return False
            
            # Only merge across consecutive pages
            if current['page'] + 1 != next_block['page']:
                return False
            
            # Don't merge if current block ends with strong punctuation
            if current_text.endswith(('.', '!', '?', ':', ';')):
                return False
            
            # Don't merge if next block starts like a list item
            if starts_like_list_item(next_text):
                return False
            
            # Don't merge if next block starts with special formatting words
            if starts_with_special_word(next_text):
                return False
            
            # Don't merge if next block looks like a heading
            if is_heading_like(next_text):
                return False
            
            # Merge if current block ends with hyphen (word continuation)
            if ends_with_hyphen(current_text):
                return True
            
            # Merge if next block starts with lowercase (likely continuation)
            if starts_with_lowercase_or_continuation(next_text):
                return True
            
            # Merge if current block doesn't end with typical sentence endings
            # and next block doesn't start with capital letter
            if (not current_text.endswith((',', ';', ':', '—', '–')) and 
                not next_text[0].isupper()):
                return True
            
            return False
        
        def merge_text_blocks(current_text: str, next_text: str) -> str:
            """Merge two text blocks with appropriate spacing."""
            current_text = current_text.strip()
            next_text = next_text.strip()
            
            # Handle hyphenated word continuation
            if current_text.endswith('-'):
                # Remove hyphen and join without space
                return current_text[:-1] + next_text
            
            # Add \n\n only for standalone numbered definitions (e.g., "17) definition...")
            if re.match(r'^\d+\)\s*"', next_text.strip()):
                return current_text + '\n\n' + next_text
            
            # Regular merge without any separator
            return current_text + ' ' + next_text
        
        merged_blocks = []
        i = 0
        
        while i < len(blocks) - 1:
            current = blocks[i].copy()  # Create a copy to avoid modifying original
            next_block = blocks[i + 1]
            
            if should_merge_blocks(current, next_block):
                # Merge the text appropriately
                current['text'] = merge_text_blocks(current['text'], next_block['text'])
                
                # Preserve other metadata (you might want to handle this differently)
                # For example, you might want to merge bounding boxes, keep both page numbers, etc.
                if 'bbox' in current and 'bbox' in next_block:
                    # This is a simple approach - you might want more sophisticated bbox merging
                    current['pages'] = [current['page'], next_block['page']]
                
                i += 2  # Skip the next block as it's merged
            else:
                merged_blocks.append(current)
                i += 1
        
        # Don't forget the last block if not merged
        if i == len(blocks) - 1:
            merged_blocks.append(blocks[-1].copy())
        
        return merged_blocks


if __name__ == "__main__":



    # Test your class here
    print("Quick test for PyMuPDFProcessor")
    # load raw blocks from a File 
    file_path = Path('data_processed/Lux_cssf18_698eng_raw_blocks.json')
    file_path = Path("data_processed/Basel_III_raw_blocks.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    processor  = block_processor()
    processor.process_and_chunk_blocks(result)