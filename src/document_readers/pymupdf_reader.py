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

                
                logger.info(f"Extracted {len(blocks)} blocks from {filename} using PyMuPDF")
                output = {
                        "blocks": blocks,
                        "height": height,
                        "width": width,
                        "filename": filename,
                        "filename_without_ext": filename_without_ext,
                        "extension": extension,
                        "pages": pages
                    }
                # save file to json file
                self._save_extracted_blocks(filename, filename_without_ext, pages, [], blocks)

                return {
                        "blocks": blocks,
                        "height": height,
                        "width": width,
                        "filename": filename,
                        "filename_without_ext": filename_without_ext,
                        "extension": extension,
                        "pages": pages
                    }


            
            except Exception as e:
                logger.error(f"PyMuPDF block extraction failed: {e}")
                raise

    def _save_extracted_blocks(
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
            saving_path = f"data_processed/{filename_without_ext}_raw_blocks.json"
            file_path = Path(saving_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            output = {
                    "filename": filename,
                    "total_pages": pages,
                    "saved_path": str(file_path),
                    "processor": "PyMuPDF",
                    "filename_without_ext": filename_without_ext,
                    "blocks": blocks
                    }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4, ensure_ascii=False)

            logger.info(f"Data saved to {file_path}")

if __name__ == "__main__":
    config = ProcessorConfig(
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        ocr_fallback=True
    )
    
    processor = PyMuPDFProcessor(config)
    
    if processor.is_available():
        result = processor.extract_blocks("data/regulatory_documents/lu/Lux_cssf18_698eng.pdf")
    else:
        print("PyMuPDF is not available. Please install the required library.")