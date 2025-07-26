# from document_readers.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
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
    
    def extract_text(self):
        pass
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
                normal_font_size, normal_color = self.find_normal_font_size_and_color_whole_doc(doc)


                page1 = doc[0]
                width = page1.rect.width
                height = page1.rect.height
                print (f"Page size: {width} x {height}")
                blocks = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    block_list = page.get_text("blocks")
                    dict_data = page.get_text("dict")
                    
                    for i, block in enumerate(block_list):
                                              # Check if this block has bold text
                        is_diff_format = False
                        if i < len(dict_data["blocks"]) and "lines" in dict_data["blocks"][i]:
                            for line in dict_data["blocks"][i]["lines"]:
                                for span in line["spans"]:    
                                    if self.has_different_format(span, normal_font_size, normal_color):  # Bold flag
                                        is_diff_format = True
                                        break
                                if is_diff_format:
                                    break
                              
                        blocks.append({
                            'page': page_num + 1,
                            'bbox': block[:4],
                            'text': block[4],
                            'is_diff_format':is_diff_format 
                        })
                
                doc.close()

                
                logger.info(f"Extracted {len(blocks)} blocks from {filename} using PyMuPDF")
                output= {
                        "height": height,
                        "width": width,
                        "filename": filename,
                        "filename_without_ext": filename_without_ext,
                        "processor": "PyMuPDF",
                        "extension": extension,
                        "pages": pages,
                        "blocks": blocks,
                    }
                # save file to json file
                self._save_extracted_blocks(output)

                return output


            
            except Exception as e:
                logger.error(f"PyMuPDF block extraction failed: {e}")
                raise

    def _save_extracted_blocks(
            self,
            output: Dict[str, Any],
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
            saving_path = f"data_processed/{output['filename_without_ext']}_raw_blocks.json"
            file_path = Path(saving_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4, ensure_ascii=False)

            logger.info(f"Data saved to {file_path}")


    def has_different_format(self, span, normal_font_size, normal_color, size_threshold=1, color_tolerance=10):
      """Detect if text has different formatting from normal text"""
      
      # Bold formatting
      if span["flags"] & 16:
          return True
      
      # Significantly larger font size
      if span["size"] > normal_font_size + size_threshold:
          return True
      
      # Color significantly different from normal (with tolerance)
      color_diff = abs(span["color"] - normal_color)
      if color_diff > color_tolerance:
          return True

    def find_normal_font_size_and_color_whole_doc(self, doc):
          """Find normal font size and color across ALL pages of the document"""
          font_sizes = []
          colors = []
          
          # Loop through ALL pages
          for page_num in range(doc.page_count):
              page = doc[page_num]
              dict_data = page.get_text("dict")
              
              for block in dict_data["blocks"]:
                  if "lines" in block:
                      for line in block["lines"]:
                          for span in line["spans"]:
                              font_sizes.append(span["size"])
                              colors.append(span["color"])
          
          # Most common across entire document
          from collections import Counter
          most_common_size = Counter(font_sizes).most_common(1)
          normal_size = most_common_size[0][0] if most_common_size else 11
          
          most_common_color = Counter(colors).most_common(1)
          normal_color = most_common_color[0][0] if most_common_color else 0
          
          return normal_size, normal_color
if __name__ == "__main__":
    config = ProcessorConfig(
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        ocr_fallback=True
    )

    
    
    processor = PyMuPDFProcessor(config)

    # import fitz
    # path = "data/regulatory_documents/eu/Basel_III.pdf"
    # doc = fitz.open(path)
    # a, b = PyMuPDFProcessor.find_normal_font_size_and_color_whole_doc(doc)
    # print(a, '-----------', b)

    # normal_color = 1644572
    # r = (normal_color >> 16) & 255
    # g = (normal_color >> 8) & 255  
    # b = normal_color & 255
    # print(f"Normal color RGB: ({r}, {g}, {b})")

    
    if processor.is_available():
        result = processor.extract_blocks("data/regulatory_documents/eu/Basel_III.pdf")
    else:
        print("PyMuPDF is not available. Please install the required library.")





    # import fitz

    # doc = fitz.open("data/regulatory_documents/eu/Basel_III.pdf")
    # bold_blocks = []

    # for page in doc:
    #   for block in page.get_text("dict")["blocks"]:
    #       if "lines" in block:
    #           full_block_text = ""
    #           block_has_bold = False
              
    #           # Collect ALL text from ALL lines in this block
    #           for line in block["lines"]:
    #               for span in line["spans"]:
    #                   full_block_text += span["text"] + " "
    #                   if span["flags"] & 16:
    #                       block_has_bold = True
              
    #           # Only add if block has bold text
    #           if block_has_bold and len(full_block_text.strip()) > 10:
    #               # Clean up spacing and line breaks
    #               clean_text = " ".join(full_block_text.split())
    #               bold_blocks.append(clean_text)



    # open("bold.txt", "w").write("\n".join(bold_blocks))