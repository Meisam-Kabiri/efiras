# Regulatory Document Chunker - Detailed Code Explanation

## Overview
This code implements a specialized document chunker designed specifically for regulatory and financial documents in the EFIRAS system. Unlike generic text chunkers, this one understands the structure and patterns common in legal/financial documents like regulations, directives, and compliance documents.

## Key Coding Concepts Used

### 1. **Regular Expressions (Regex)**
```python
r'(Article\s+\d+[a-z]?)'
r'(Section\s+\d+(?:\.\d+)?)'
```
- **What it is**: Pattern matching language for finding specific text patterns
- **Why use it**: Perfect for finding regulatory structures like "Article 5", "Section 2.1"
- **`r''`**: Raw strings - backslashes don't need escaping
- **`\s+`**: One or more whitespace characters
- **`\d+`**: One or more digits
- **`(?:\.\d+)?`**: Optional decimal part (non-capturing group)

### 2. **Dataclasses (Reused)**
```python
@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
```
- Same concept as before - auto-generates boilerplate code for data containers

### 3. **UUID Generation**
```python
chunk_id = str(uuid.uuid4())
```
- **What it is**: Universally Unique Identifier
- **Why use it**: Ensures each chunk has a unique ID across all documents

### 4. **Set Operations**
```python
chunk_words = set(chunk_text.lower().split())
page_words = set(page_text.lower().split())
overlap_ratio = len(chunk_words & page_words) / len(chunk_words)
```
- **`&`**: Set intersection - finds common words between chunks and pages
- **Why use sets**: Fast lookups and mathematical operations on word collections

## Class-by-Class Breakdown

### 1. **DocumentChunk (Dataclass) - REUSED**
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
**Purpose**: Same as in the processor - standardized format for document chunks
**Key Point**: This is the output format that both the processor and chunker use

### 2. **RegulatoryChunker (Main Class)**
```python
class RegulatoryChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
```
**Purpose**: Specialized chunker that understands regulatory document structure
**Key Features**:
- Configurable chunk size and overlap
- Built-in knowledge of regulatory patterns
- Financial keyword extraction
- Jurisdiction detection

## Core Pattern Recognition Systems

### 1. **Section Patterns**
```python
self.section_patterns = [
    r'(Article\s+\d+[a-z]?)',        # "Article 5", "Article 12a"
    r'(Section\s+\d+(?:\.\d+)?)',    # "Section 3", "Section 2.1"
    r'(Chapter\s+\d+)',              # "Chapter 4"
    r'(Part\s+[IVXLC]+)',           # "Part IV" (Roman numerals)
    r'(Regulation\s+\d+)',           # "Regulation 123"
    r'(Directive\s+\d+)',            # "Directive 2018/123"
    r'(\d+\.\s+[A-Z][^.]*)',        # "1. DEFINITIONS"
    r'([A-Z][A-Z\s]{10,})',         # "CAPITAL REQUIREMENTS"
]
```
**Purpose**: Recognizes common regulatory document structures
**How it works**: Each pattern captures different ways regulations organize content
**Regex Breakdown**:
- `[a-z]?`: Optional lowercase letter (for "Article 5a")
- `[IVXLC]+`: Roman numerals using character class
- `[^.]*`: Any character except period (until sentence ends)

### 2. **Financial Keywords Classification**
```python
self.financial_keywords = {
    'risk_management': ['risk', 'capital', 'liquidity', 'leverage', 'exposure'],
    'compliance': ['compliance', 'reporting', 'disclosure', 'audit', 'supervision'],
    'market': ['market', 'trading', 'investment', 'securities', 'derivatives'],
    'banking': ['bank', 'credit', 'loan', 'deposit', 'basel'],
    'insurance': ['insurance', 'solvency', 'premium', 'underwriting'],
    'funds': ['fund', 'portfolio', 'asset management', 'ucits', 'aifmd']
}
```
**Purpose**: Categorizes document content by financial domain
**How it works**: Searches for keywords and tags chunks with relevant categories
**Benefit**: Enables targeted search and retrieval by financial topic

### 3. **Regulation Type Detection**
```python
self.regulation_patterns = {
    'MiFID II': r'(mifid|markets?\s+in\s+financial\s+instruments)',
    'Basel III': r'(basel|capital\s+requirements)',
    'GDPR': r'(gdpr|data\s+protection|personal\s+data)',
    'AMLD': r'(anti.?money\s+laundering|aml)',
    # ... more patterns
}
```
**Purpose**: Automatically identifies which regulation the document relates to
**Real-world context**: Each pattern matches common ways these regulations are referenced
**Pattern explanation**: `anti.?money` matches "anti-money" or "anti money laundering"

## Core Processing Methods

### 1. **Main Chunking Orchestrator**
```python
def chunk_document(self, extracted_data: Dict[str, Any], file_path: Union[str, Path]):
    # Clean the text first
    cleaned_text = self._clean_regulatory_text(text)
    
    # Try semantic chunking first (by sections/articles)
    semantic_chunks = self._semantic_chunk(cleaned_text, page_texts)
    
    if semantic_chunks and len(semantic_chunks) > 1:
        chunks = semantic_chunks  # Use intelligent chunking
    else:
        chunks = self._sliding_window_chunk(cleaned_text, page_texts)  # Fallback
```
**Purpose**: Main workflow that tries intelligent chunking first, falls back if needed
**Strategy**: 
1. Clean text (remove artifacts)
2. Try semantic chunking (by regulatory sections)
3. Fall back to sliding window if semantic fails
4. Enrich all chunks with metadata

### 2. **Text Cleaning for Regulatory Documents**
```python
def _clean_regulatory_text(self, text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF artifacts
    text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation across lines
    
    # Preserve important regulatory structure
    text = re.sub(r'(Article\s+\d+)', r'\n\n\1', text)
    
    # Clean up common regulatory document artifacts
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Page numbers
```
**Purpose**: Cleans PDF extraction artifacts while preserving regulatory structure
**Key Operations**:
- **Whitespace normalization**: `\s+` → single space
- **Hyphenation fix**: Removes line-break hyphens from PDF extraction
- **Structure preservation**: Adds line breaks before important sections
- **Artifact removal**: Removes page numbers and standalone numbers
- **Character encoding fix**: Replaces smart quotes with regular quotes

### 3. **Semantic Chunking (The Smart Part)**
```python
def _semantic_chunk(self, text: str, page_texts: List[str]) -> List[Dict[str, Any]]:
    # Combine all section patterns into one
    combined_pattern = '|'.join(self.section_patterns)
    sections = re.split(f'({combined_pattern})', text, flags=re.IGNORECASE)
    
    for i, section in enumerate(sections):
        # Check if this is a section header
        is_header = any(re.match(pattern, section.strip(), re.IGNORECASE) 
                      for pattern in self.section_patterns)
```
**Purpose**: Splits document by meaningful sections (Articles, Chapters, etc.)
**How it works**:
1. **Pattern Combination**: `|` creates OR pattern matching any section type
2. **`re.split()`**: Splits text but keeps the separators (the section headers)
3. **Header Detection**: Checks if each piece is a section header or content
4. **Chunk Assembly**: Groups headers with their following content

**Smart Features**:
- **Size Management**: Splits large sections if they exceed chunk size
- **Paragraph Boundaries**: When splitting, tries to break at paragraph boundaries
- **Section Preservation**: Keeps section headers with their content

### 4. **Sliding Window Fallback**
```python
def _sliding_window_chunk(self, text: str, page_texts: List[str]) -> List[Dict[str, Any]]:
    words = text.split()
    words_per_chunk = self.chunk_size // 4  # Estimate 4 chars per word
    overlap_words = self.overlap // 4
    
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk_words = words[i:i + words_per_chunk]
```
**Purpose**: Traditional chunking when semantic chunking fails
**How it works**:
- **Word-based**: Counts words instead of characters for better control
- **Overlap**: Each chunk shares some words with the next chunk
- **Step size**: `words_per_chunk - overlap_words` ensures proper overlap

## Advanced Features

### 1. **Page Number Detection**
```python
def _find_page_numbers(self, chunk_text: str, page_texts: List[str]) -> List[int]:
    for page_num, page_text in enumerate(page_texts, 1):
        chunk_words = set(chunk_text.lower().split())
        page_words = set(page_text.lower().split())
        
        overlap_ratio = len(chunk_words & page_words) / len(chunk_words)
        
        if overlap_ratio > 0.3:  # 30% word overlap threshold
            pages.append(page_num)
```
**Purpose**: Determines which pages each chunk spans
**Algorithm**:
1. **Word Sets**: Converts text to sets of unique words
2. **Intersection**: `&` finds common words between chunk and page
3. **Overlap Ratio**: Percentage of chunk words found in page
4. **Threshold**: 30% overlap required to associate chunk with page

### 2. **Metadata Enrichment**
```python
def _enrich_chunk_metadata(self, chunk_data, extracted_data, source_file):
    # Detect regulation type
    regulation_type = self._detect_regulation_type(content)
    
    # Extract financial keywords
    keywords = self._extract_keywords(content)
    
    # Detect jurisdiction
    jurisdiction = self._detect_jurisdiction(content)
    
    metadata = {
        'regulation_type': regulation_type,
        'jurisdiction': jurisdiction,
        'keywords': keywords,
        'chunk_method': 'semantic' if chunk_data.get('section_title') else 'sliding_window'
    }
```
**Purpose**: Adds rich metadata to each chunk for better searchability
**Metadata Types**:
- **Regulation Type**: MiFID II, Basel III, GDPR, etc.
- **Jurisdiction**: EU, US, UK, etc.
- **Keywords**: Categorized financial terms
- **Processing Info**: How the chunk was created

### 3. **Regulation Type Detection**
```python
def _detect_regulation_type(self, text: str) -> Optional[str]:
    text_lower = text.lower()
    
    for reg_type, pattern in self.regulation_patterns.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            return reg_type
    
    return None
```
**Purpose**: Automatically identifies which regulation the chunk discusses
**How it works**: Searches for regulation-specific keywords and phrases
**Returns**: Regulation acronym (like "MiFID II") or None if unidentified

## Chunking Strategy Comparison

### Semantic Chunking (Preferred)
```
Document: "Article 1 - Definitions... Article 2 - Scope... Article 3 - Requirements..."

Chunk 1: "Article 1 - Definitions [complete section content]"
Chunk 2: "Article 2 - Scope [complete section content]"  
Chunk 3: "Article 3 - Requirements [complete section content]"
```
**Advantages**:
- Chunks are logically complete
- Better for Q&A systems
- Preserves regulatory structure
- More meaningful for legal analysis

### Sliding Window (Fallback)
```
Document: "Long unstructured text without clear sections..."

Chunk 1: [Words 1-250 with overlap]
Chunk 2: [Words 200-450 with overlap]
Chunk 3: [Words 400-650 with overlap]
```
**Advantages**:
- Always works regardless of document structure
- Consistent chunk sizes
- Good for general text processing

## Real-World Example

### Input Document:
```
"Article 1 - Scope
This regulation applies to investment firms...

Article 2 - Definitions
For the purposes of this regulation:
(a) 'investment firm' means...

Section 3.1 - Capital Requirements
Investment firms shall maintain..."
```

### Processing Flow:
1. **Text Cleaning**: Removes page numbers, fixes encoding
2. **Pattern Matching**: Finds "Article 1", "Article 2", "Section 3.1"
3. **Semantic Splitting**: Creates 3 chunks, each with complete section
4. **Page Detection**: Maps each chunk to source pages
5. **Metadata Addition**: Tags with regulation type, keywords, jurisdiction

### Output Chunks:
```python
[
    DocumentChunk(
        content="Article 1 - Scope\nThis regulation applies to investment firms...",
        section_title="Article 1",
        metadata={
            'regulation_type': 'MiFID II',
            'keywords': {'market': ['investment'], 'compliance': ['regulation']},
            'chunk_method': 'semantic'
        }
    ),
    # ... more chunks
]
```

## Key Design Patterns

### 1. **Template Method Pattern**
- `chunk_document()` defines the overall workflow
- Individual methods handle specific steps
- Easy to modify or extend individual steps

### 2. **Strategy Pattern**
- Two chunking strategies: semantic vs sliding window
- Automatically selects best strategy
- Can easily add new chunking strategies

### 3. **Factory Pattern**
- Creates standardized DocumentChunk objects
- Handles all the metadata enrichment
- Consistent output format

## Error Handling & Robustness

### Graceful Degradation
```python
if semantic_chunks and len(semantic_chunks) > 1:
    chunks = semantic_chunks  # Preferred method
else:
    chunks = self._sliding_window_chunk(...)  # Always works
```

### Defensive Programming
```python
if not text:
    raise ValueError("No text content to chunk")

pages = pages if pages else [1]  # Default to page 1 if detection fails
```

## Benefits of This Specialized Approach

### 1. **Domain Awareness**
- Understands regulatory document structure
- Recognizes financial terminology
- Identifies specific regulations automatically

### 2. **Intelligent Chunking**
- Preserves logical document structure
- Better for legal/compliance use cases
- More meaningful chunks for AI processing

### 3. **Rich Metadata**
- Enables sophisticated search and filtering
- Supports compliance tracking
- Facilitates regulatory analysis

### 4. **Flexibility**
- Works with multiple document types
- Handles poorly structured documents
- Configurable for different needs

## Usage in EFIRAS System

This chunker is specifically designed for the EFIRAS (regulatory document processing) system where:
- **Legal precision** is critical
- **Regulatory structure** must be preserved
- **Compliance tracking** requires detailed metadata
- **Multi-language support** may be needed (EU regulations)
- **Cross-reference capability** is essential

The chunker bridges the gap between raw document extraction and intelligent document analysis, making regulatory documents searchable, analyzable, and actionable for compliance professionals.