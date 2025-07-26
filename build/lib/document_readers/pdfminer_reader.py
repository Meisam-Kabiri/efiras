from document_readers.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class PDFMinerProcessor(DocumentProcessor):
    """Detailed processor for complex PDF layouts"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.processor_type = ProcessorType.PDFMINER
    
    def is_available(self) -> bool:
        try:
            from pdfminer.high_level import extract_text
            return True
        except ImportError:
            return False
    
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.pdfpage import PDFPage
            from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
            from pdfminer.converter import TextConverter
            from pdfminer.layout import LAParams
            from io import StringIO
            
            # Extract full text
            text = extract_text(str(file_path))
            
            # Extract page by page for better control
            page_texts = []
            with open(file_path, 'rb') as file:
                resource_manager = PDFResourceManager()
                
                for page_num, page in enumerate(PDFPage.get_pages(file)):
                    output_string = StringIO()
                    device = TextConverter(resource_manager, output_string, laparams=LAParams())
                    interpreter = PDFPageInterpreter(resource_manager, device)
                    interpreter.process_page(page)
                    page_text = output_string.getvalue()
                    page_texts.append(page_text)
                    device.close()
                    output_string.close()
            
            metadata = {
                'page_count': len(page_texts),
                'processor': self.processor_type.value,
                'layout_analysis': True
            }
            
            return {
                'text': text,
                'page_texts': page_texts,
                'metadata': metadata,
                'extraction_method': 'layout_analysis'
            }
            
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {e}")
            raise