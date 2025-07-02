from efiras_processor.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class UnstructuredProcessor(DocumentProcessor):
    """Multi-format processor using unstructured.io"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.processor_type = ProcessorType.UNSTRUCTURED
    
    def is_available(self) -> bool:
        try:
            from unstructured.partition.auto import partition
            return True
        except ImportError:
            return False
    
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            from unstructured.partition.auto import partition
            
            elements = partition(filename=str(file_path))
            
            text = "\n".join([str(element) for element in elements])
            
            # Group elements by page if available
            page_texts = []
            current_page = []
            current_page_num = 1
            
            for element in elements:
                if hasattr(element, 'metadata') and element.metadata.page_number:
                    if element.metadata.page_number != current_page_num:
                        if current_page:
                            page_texts.append("\n".join([str(e) for e in current_page]))
                        current_page = [element]
                        current_page_num = element.metadata.page_number
                    else:
                        current_page.append(element)
                else:
                    current_page.append(element)
            
            if current_page:
                page_texts.append("\n".join([str(e) for e in current_page]))
            
            metadata = {
                'page_count': len(page_texts),
                'processor': self.processor_type.value,
                'element_count': len(elements),
                'element_types': list(set([type(e).__name__ for e in elements]))
            }
            
            return {
                'text': text,
                'page_texts': page_texts,
                'elements': elements,
                'metadata': metadata,
                'extraction_method': 'element_detection'
            }
            
        except Exception as e:
            logger.error(f"Unstructured extraction failed: {e}")
            raise