from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessorType(Enum):
    PYMUPDF = "pymupdf"
    AZURE_DI = "azure_document_intelligence" 
    PDFMINER = "pdfminer"
    UNSTRUCTURED = "unstructured"
    AUTO = "auto"




# class DocumentChunk:
#     def __init__(
#         self,
#         content: str,
#         metadata: Dict[str, Any],
#         chunk_id: str,
#         source_document: str,
#         page_numbers: List[int],
#         section_title: Optional[str] = None,
#         confidence_score: Optional[float] = None,
#     ):
#         self.content = content
#         self.metadata = metadata
#         self.chunk_id = chunk_id
#         self.source_document = source_document
#         self.page_numbers = page_numbers
#         self.section_title = section_title
#         self.confidence_score = confidence_score

#     def __repr__(self):
#         return f"DocumentChunk(content={self.content!r}, metadata={self.metadata!r}, ...)"

#     def __eq__(self, other):
#         return (
#             isinstance(other, DocumentChunk)
#             and self.content == other.content
#             and self.metadata == other.metadata
#             # ... (and all other fields)
#         )
#     # Plus other generated methods like `__hash__` if needed.

# equivalent to the simpler version like this: 

@dataclass
class DocumentChunk:
    """Standardized chunk format across all processors"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_document: str
    page_numbers: List[int]
    section_title: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class ProcessorConfig:
    """Configuration for different processors"""
    chunk_size: int = 1000
    overlap: int = 200
    preserve_formatting: bool = True
    extract_tables: bool = True
    ocr_fallback: bool = True
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None

pr_type = ProcessorType.PYMUPDF
print(f"Selected processor type: {pr_type.value}, {pr_type.name}")





# Subclasses must override this method to provide their own implementation.
# Enforces Subclass Responsibility
# If a subclass doesn’t override the method, Python will raise a TypeError when you try to instantiate it.
class DocumentProcessor(ABC):
    """Abstract base class for all document processors"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.processor_type = None
    
    @abstractmethod
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract raw text and metadata from document"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if processor dependencies are available"""
        pass
    
    def get_quality_score(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate extraction quality score (0-1)"""
        text = extracted_data.get('text', '')
        if not text:
            return 0.0
        
        # Basic quality metrics
        char_count = len(text)
        word_count = len(text.split())
        
        # Penalize if too much gibberish or too little content
        if char_count < 100:
            return 0.1
        if word_count < 50:
            return 0.3
        
        # Look for common PDF extraction artifacts
        artifact_score = 1.0
        artifacts = ['�', '□', '▢', 'ﬀ', 'ﬁ', 'ﬂ']
        for artifact in artifacts:
            if artifact in text:
                artifact_score -= 0.1
        
        return max(0.1, min(1.0, artifact_score))
    
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

