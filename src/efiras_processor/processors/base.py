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
    