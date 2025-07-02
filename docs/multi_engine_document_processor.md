# Multi-Engine Document Processor - Detailed Code Explanation

## Overview
This code implements a sophisticated document processing system called EFIRAS (appears to be a regulatory document processing system) that can handle multiple extraction engines and automatically select the best one for each document.

## Key Coding Concepts Used

### 1. **Abstract Base Classes (ABC)**
```python
from abc import ABC, abstractmethod
class DocumentProcessor(ABC):
```
- **What it is**: ABC stands for Abstract Base Class - a template that defines methods that must be implemented by subclasses
- **Why use it**: Ensures all document processors follow the same interface/contract
- **@abstractmethod**: Decorator that forces subclasses to implement specific methods

### 2. **Enums**
```python
class ProcessorType(Enum):
    PYMUPDF = "pymupdf"
    AZURE_DI = "azure_document_intelligence"
```
- **What it is**: Enumeration - a set of named constants
- **Why use it**: Prevents typos, provides autocomplete, makes code more readable and maintainable

### 3. **Dataclasses**
```python
@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
```
- **What it is**: Automatically generates `__init__`, `__repr__`, and other methods
- **Why use it**: Reduces boilerplate code for simple data containers

### 4. **Type Hints**
```python
def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
```
- **What it is**: Specifies expected input and output types
- **Why use it**: Improves code readability, enables IDE support, helps catch errors
- **Union[str, Path]**:  Accepts either a string OR a Path object (like "either this type or that type")
- **List[int]**:  A list that contains only integers (like [1, 2, 3, 4])
- **Any**:  Accepts any data type without restrictions (essentially disables type checking for that parameter)

## Class-by-Class Breakdown

### 1. **ProcessorType (Enum)**
```python
class ProcessorType(Enum):
    PYMUPDF = "pymupdf"
    AZURE_DI = "azure_document_intelligence" 
    PDFMINER = "pdfminer"
    UNSTRUCTURED = "unstructured"
    AUTO = "auto"
```
**Purpose**: Defines the available document processing engines as constants
**Why needed**: Prevents string typos and provides a centralized list of supported processors

### 2. **DocumentChunk (Dataclass)**
```python
@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_document: str
    page_numbers: List[int]
    section_title: Optional[str] = None
    confidence_score: Optional[float] = None
```
**Purpose**: Standardized format for document chunks after processing
**Key Features**:
- `content`: The actual text content
- `metadata`: Additional information about the chunk
- `chunk_id`: Unique identifier
- `page_numbers`: Which pages this chunk spans
- `Optional` fields: Can be None if not available

### 3. **ProcessorConfig (Dataclass)**
```python
@dataclass
class ProcessorConfig:
    chunk_size: int = 1000
    overlap: int = 200
    preserve_formatting: bool = True
    extract_tables: bool = True
    ocr_fallback: bool = True
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
```
**Purpose**: Configuration settings for all processors
**Key Features**:
- Default values for common settings
- Azure-specific configuration for cloud processing
- Boolean flags for different processing options

### 4. **DocumentProcessor (Abstract Base Class)**
```python
class DocumentProcessor(ABC):
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.processor_type = None
    
    @abstractmethod
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
```
**Purpose**: Base template that all processors must follow
**Key Methods**:
- `extract_text()`: Must be implemented by each processor
- `is_available()`: Checks if processor dependencies are installed
- `get_quality_score()`: Evaluates extraction quality (0-1 scale)

**Quality Scoring Logic**:
```python
def get_quality_score(self, extracted_data: Dict[str, Any]) -> float:
    # Penalize short content
    if char_count < 100: return 0.1
    if word_count < 50: return 0.3
    
    # Detect PDF artifacts (corrupted characters)
    artifacts = ['�', '□', '▢', 'ﬀ', 'ﬁ', 'ﬂ']
    for artifact in artifacts:
        if artifact in text:
            artifact_score -= 0.1
```

### 5. **PyMuPDFProcessor (Concrete Implementation)**
```python
class PyMuPDFProcessor(DocumentProcessor):
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)  # Call parent constructor
        self.processor_type = ProcessorType.PYMUPDF
```
**Purpose**: Fast processor for text-based PDFs using PyMuPDF library
**Strengths**: Very fast, good for PDFs with embedded text
**Process**:
1. Opens PDF with `fitz.open()`
2. Extracts text page by page
3. Collects metadata (title, author, creation date)
4. Returns structured data

### 6. **AzureDocumentProcessor (Cloud-based)**
```python
class AzureDocumentProcessor(DocumentProcessor):
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.processor_type = ProcessorType.AZURE_DI
        
        if not config.azure_endpoint or not config.azure_key:
            raise ValueError("Azure endpoint and key required")
```
**Purpose**: Uses Microsoft Azure's Document Intelligence service
**Strengths**: Excellent OCR, table extraction, handles complex layouts
**Process**:
1. Authenticates with Azure using credentials
2. Uploads document to Azure service
3. Receives structured analysis results
4. Extracts tables and confidence scores

### 7. **PDFMinerProcessor (Layout Analysis)**
```python
class PDFMinerProcessor(DocumentProcessor):
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
```
**Purpose**: Detailed processor for complex PDF layouts
**Strengths**: Better handling of complex layouts, font analysis
**Process**:
1. Uses `LAParams` for layout analysis parameters
2. Processes each page individually for better control
3. Maintains formatting and structure information

### 8. **UnstructuredProcessor (Multi-format)**
```python
class UnstructuredProcessor(DocumentProcessor):
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        from unstructured.partition.auto import partition
        elements = partition(filename=str(file_path))
```
**Purpose**: Handles multiple file formats using unstructured.io
**Strengths**: Supports many formats (PDF, DOCX, HTML, etc.)
**Process**:
1. Auto-detects file type and uses appropriate parser
2. Returns structured elements (headings, paragraphs, tables)
3. Groups elements by page when possible

### 9. **DocumentProcessorManager (Orchestrator)**
```python
class DocumentProcessorManager:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.processors = self._initialize_processors()
        self.fallback_order = [ProcessorType.PYMUPDF, ...]
```
**Purpose**: Main coordination class that manages all processors
**Key Features**:

#### Processor Initialization
```python
def _initialize_processors(self) -> Dict[ProcessorType, DocumentProcessor]:
    processors = {}
    if PyMuPDFProcessor(self.config).is_available():
        processors[ProcessorType.PYMUPDF] = PyMuPDFProcessor(self.config)
```
- Only initializes processors whose dependencies are available
- Gracefully handles missing libraries

#### Auto-Selection Logic
```python
def _get_auto_processor_order(self, file_path: Path) -> List[ProcessorType]:
    file_size = file_path.stat().st_size
    
    if file_size < 1_000_000:  # < 1MB
        return [ProcessorType.PYMUPDF, ...]  # Fast processors first
    elif file_size > 10_000_000:  # > 10MB
        return [ProcessorType.AZURE_DI, ...]  # Robust processors first
```
- Chooses processor order based on file size
- Smaller files → faster processors first
- Larger files → more robust processors first

#### Fallback Processing
```python
def process_document(self, file_path, preferred_processor=ProcessorType.AUTO, fallback=True):
    for processor_type in processor_order:
        try:
            result = processor.extract_text(file_path)
            quality_score = processor.get_quality_score(result)
            
            if quality_score >= 0.5:  # Acceptable quality
                return result
            elif not fallback:
                return result
            else:
                continue  # Try next processor
```
- Tries processors in order
- Checks quality score (0.5 threshold)
- Falls back to next processor if quality is poor
- Returns best available result

## Usage Pattern

### Basic Usage
```python
# 1. Configure
config = ProcessorConfig(
    chunk_size=1000,
    overlap=200,
    extract_tables=True
)

# 2. Initialize Manager
manager = DocumentProcessorManager(config)

# 3. Process Document
result = manager.process_document(
    "document.pdf",
    preferred_processor=ProcessorType.AUTO,
    fallback=True
)
```

### What You Get Back
```python
{
    'text': "Full document text...",
    'page_texts': ["Page 1 text", "Page 2 text", ...],
    'metadata': {
        'page_count': 5,
        'processor': 'pymupdf',
        'title': 'Document Title'
    },
    'quality_score': 0.85,
    'processor_used': 'pymupdf'
}
```

## Key Design Patterns

### 1. **Strategy Pattern**
- Different processors implement the same interface
- Manager can switch between processors dynamically

### 2. **Factory Pattern**
- Manager creates appropriate processor instances
- Handles dependency checking and initialization

### 3. **Chain of Responsibility**
- Tries processors in sequence until one succeeds
- Each processor can pass to the next if it fails

### 4. **Template Method**
- Base class defines the processing workflow
- Subclasses implement specific extraction logic

## Error Handling Strategy

### Graceful Degradation
```python
try:
    result = processor.extract_text(file_path)
    if quality_score >= 0.5:
        return result  # Good quality
    else:
        continue  # Try next processor
except Exception as e:
    logger.warning(f"Processor {processor_type.value} failed: {e}")
    if not fallback:
        raise  # Re-raise if no fallback allowed
    continue  # Try next processor
```

### Quality Assurance
- Each extraction gets a quality score
- Poor quality results trigger fallback
- Ensures best possible extraction quality

## Benefits of This Architecture

1. **Flexibility**: Easy to add new processors
2. **Reliability**: Multiple fallback options
3. **Efficiency**: Auto-selects best processor for each document
4. **Maintainability**: Clean separation of concerns
5. **Extensibility**: New processor types can be added easily
6. **Quality Control**: Built-in quality assessment and fallback

This design makes the system robust and adaptable to different document types and processing requirements.