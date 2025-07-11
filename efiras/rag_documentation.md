# Enhanced RAG System - Technical Documentation

## System Overview

The Enhanced RAG System is a retrieval-augmented generation framework designed for regulatory document Q&A. It combines semantic search with intelligent context assembly and regulatory-specific optimizations to provide comprehensive, well-cited answers from financial regulation documents.

## System Flow

### Phase 1: Document Ingestion & Embedding
```
Document Blocks → Embedding Generation → Vector Database Storage
```

1. **Input Processing**: Receives processed document blocks with enriched headers
2. **Embedding Text Creation**: Combines `enriched_headers` + `text` for optimal semantic representation
3. **Embedding Generation**: 
   - **If `use_local_embeddings=False`**: Uses OpenAI API (`text-embedding-3-large`)
   - **If `use_local_embeddings=True`**: Uses local SentenceTransformer model
4. **Vector Database Creation**: Stores embeddings with metadata in `self.vector_db`
5. **Cache Management**: Saves embeddings to disk for future use

### Phase 2: Query Processing & Retrieval
```
User Query → Query Embedding → Similarity Search → Top-K Retrieval
```

1. **Query Embedding**: Convert user question to vector using same embedding model
2. **Cosine Similarity**: Calculate similarity between query and all document embeddings
3. **Ranking**: Sort documents by similarity score
4. **Top-K Selection**: Return most relevant chunks (default: top 5)

### Phase 3: Context Enhancement
```
Retrieved Chunks → Cross-Reference Discovery → Enhanced Context Assembly
```

1. **Section Analysis**: Extract `enriched_headers` from retrieved chunks
2. **Cross-Reference Search**: Find additional blocks with matching section headers
3. **Context Expansion**: Add related content from same regulatory sections
4. **Deduplication**: Prevent adding already-retrieved content
5. **Context Assembly**: Combine all relevant content into unified context

### Phase 4: Regulatory Intelligence
```
Enhanced Context → Pattern Matching → Regulatory Citation Extraction
```

1. **Pattern Detection**: Apply regex patterns to identify regulatory references
2. **Citation Extraction**: Extract Articles, Sections, CSSF Regulations, Laws, etc.
3. **Reference Cataloging**: Create list of relevant regulatory citations
4. **Context Annotation**: Prepare context with identified regulations

### Phase 5: Answer Generation
```
Query + Enhanced Context + Regulations → Prompt Engineering → LLM Response
```

1. **Query Classification**: Determine if query is about compliance officers, appointments, etc.
2. **Prompt Selection**:
   - **If compliance-related**: Apply specialized compliance officer prompting
   - **Else**: Use general regulatory Q&A prompting
3. **Context Integration**: Combine enhanced context + regulatory references + query
4. **LLM Generation**: Use OpenAI Chat Completions with optimized parameters:
   - Temperature: 0.1 (consistency)
   - Max tokens: 1200 (comprehensiveness)
5. **Response Assembly**: Return formatted answer with citations

## Decision Trees & Conditional Logic

### Embedding Strategy Decision
```
Initialize RAGSystem
├─ use_local_embeddings = False
│  └─ Use OpenAI API (text-embedding-3-large)
│     ├─ Requires internet connection
│     ├─ Higher quality embeddings (3072-dim)
│     └─ API costs per embedding
└─ use_local_embeddings = True
   └─ Use SentenceTransformer (local)
      ├─ Offline operation
      ├─ Lower dimension embeddings (384-768-dim)
      └─ No API costs
```

### Query Processing Logic
```
User Query Input
├─ Generate query embedding (same model as documents)
├─ Perform cosine similarity search
├─ Retrieve top-k similar chunks
├─ Analyze query content
│  ├─ Contains ["compliance officer", "appointment", "requirements"]?
│  │  └─ YES: Apply specialized compliance prompting
│  └─ NO: Apply general regulatory prompting
├─ Enhance context with cross-references
├─ Extract regulatory citations
└─ Generate comprehensive answer
```

### Context Enhancement Flow
```
Retrieved Chunks
├─ Extract enriched_headers from each chunk
├─ For each unique header:
│  ├─ Search vector_db for matching headers
│  ├─ Filter out already-retrieved content
│  ├─ Add related blocks to context
│  └─ Limit to one additional block per section
├─ Combine all content
├─ Check length against max_context limit
│  ├─ If exceeds limit: Truncate to max_context
│  └─ If within limit: Use full enhanced context
└─ Return enhanced context string
```

### Cache Management Logic
```
add_documents() called
├─ Build cache file path:
│  ├─ use_local_embeddings = True
│  │  └─ cache_path/cache_file_name_local.json
│  └─ use_local_embeddings = False
│     └─ cache_path/cache_file_name_online.json
├─ Check if cache file exists
│  ├─ EXISTS: Load embeddings from cache
│  └─ NOT EXISTS: Generate new embeddings
│     ├─ Process each block
│     ├─ Create embedding text
│     ├─ Generate embedding
│     ├─ Store in memory
│     └─ Save to cache file
└─ Populate self.vector_db
```

## Core Functions

### Document Processing & Embedding

#### `create_embedding_text(block: Dict) -> str`
Combines enriched headers with block text to create optimal embedding content.

#### `embed_text(text: str) -> List[float]`
Generates vector embeddings using either OpenAI API or local SentenceTransformer models.

#### `embed_blocks(blocks: List[Dict], cache_path: str) -> List[Dict]`
Processes document blocks into embeddings with automatic caching for performance optimization.

#### `add_documents(blocks: List[Dict], cache_path: str, cache_file_name: str)`
Builds the vector database from processed document blocks with intelligent cache management.

### Retrieval & Search

#### `search(query: str, top_k: int) -> List[Dict]`
Performs cosine similarity search to find most relevant document chunks for a given query.

**Process:**
1. Generate query embedding
2. Calculate cosine similarity with all document embeddings
3. Rank and return top-k most similar chunks

### Enhanced Context Building

#### `_enhance_context_with_related_blocks(relevant_chunks: List[Dict], query: str) -> str`
**Purpose:** Builds richer context by discovering related content from the same regulatory sections.

**Algorithm:**
1. Extract section headers from retrieved chunks
2. Find additional blocks with matching `enriched_headers`
3. Add related content to enhance context completeness
4. Prevent duplication of already-retrieved content

### Regulatory Intelligence

#### `_extract_specific_regulations(context: str, query: str) -> List[str]`
**Purpose:** Automatically identifies and extracts regulatory citations from context.

**Detection Patterns:**
- Articles: `Article \d+(\([^)]+\))?(\s+of\s+[^.]+)?`
- Sections: `Section \d+(\.\d+)*(\.\d+)*`
- Sub-sections: `Sub-section \d+(\.\d+)*(\.\d+)*`
- CSSF Regulations: `CSSF\s+Regulation\s+\d+-\d+`
- Delegated Regulations: `Delegated\s+Regulation\s+\([^)]+\)\s+\d+/\d+`
- Circulars: `Circular\s+CSSF\s+\d+/\d+`
- Laws: `\d{4}\s+Law`

### Answer Generation

#### `answer_query(query: str, top_k: int, max_context: int) -> str`
**Purpose:** Generates comprehensive answers using enhanced RAG pipeline.

**Process Flow:**
1. Retrieve relevant chunks via semantic search
2. Enhance context with related regulatory blocks
3. Extract regulatory references from assembled context
4. Apply query-specific system prompting
5. Generate answer using OpenAI Chat Completions API

**Query-Specific Optimizations:**
- **Compliance Officer Queries**: Specialized prompting for appointment requirements, documentation, and procedures
- **General Regulatory Queries**: Standard regulatory Q&A prompting with emphasis on completeness and citations

#### `answer_with_sources(query: str, top_k: int) -> Dict`
**Purpose:** Provides detailed response with full source attribution and regulatory tracking.

**Returns:**
```python
{
    "answer": str,                      # Generated response
    "sources": List[Dict],              # Source chunks with metadata
    "regulatory_references": List[str], # Extracted citations
    "confidence": float                 # Retrieval confidence score
}
```

### Utility Functions

#### `stats() -> Dict[str, int]`
Returns database statistics including total document count.

## Key Enhancements Over Standard RAG

### 1. Regulatory Context Awareness
- **Cross-Section Linking**: Finds related content from same regulatory sections
- **Citation Extraction**: Automatically identifies and tracks regulatory references
- **Hierarchical Understanding**: Recognizes regulatory document structure (Articles > Sections > Sub-sections)

### 2. Query Intelligence
- **Domain-Specific Prompting**: Tailored system prompts for different regulatory query types
- **Completeness Optimization**: Enhanced prompts ensure comprehensive coverage of requirements and exceptions
- **Professional Formatting**: Structured outputs with proper organization and bullet points

### 3. Enhanced Retrieval
- **Enriched Embeddings**: Combines headers with content for better semantic matching
- **Context Expansion**: Builds richer context through cross-referencing
- **Relevance Optimization**: Lower temperature and increased token limits for regulatory precision

### 4. Source Attribution
- **Granular Tracking**: Links answers to specific document sections and pages
- **Regulatory Mapping**: Tracks which regulations and articles inform each answer
- **Confidence Scoring**: Provides retrieval confidence metrics

## Technical Specifications

### Embedding Models
- **Online**: OpenAI `text-embedding-3-large` (3072 dimensions)
- **Local Options**: SentenceTransformer models (384-768 dimensions)
  - `all-MiniLM-L6-v2`: Fast, good quality (384-dim)
  - `all-mpnet-base-v2`: High quality, slower (768-dim)

### Generation Models
- **Primary**: GPT-4 with optimized parameters
  - Temperature: 0.1 (regulatory consistency)
  - Max tokens: 1200 (comprehensive responses)

### Performance Characteristics
- **Caching**: Automatic embedding cache management
- **Scalability**: Handles large regulatory document sets
- **Accuracy**: Optimized for regulatory precision and completeness