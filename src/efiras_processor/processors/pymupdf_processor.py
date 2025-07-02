from efiras_processor.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

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
            
            return {
                'text': text,
                'page_texts': page_texts,
                'metadata': metadata,
                'extraction_method': 'text_layer'
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise