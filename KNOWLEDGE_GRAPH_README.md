# Knowledge Graph RAG - LLM-First Approach

## Overview

This implementation uses **LLM to do EVERYTHING** - no traditional chunking, no manual processing.

### Key Principle

```
Raw Text (10 pages) → LLM → Chunks + Entities + Relationships + Summaries
```

The LLM:
1. **Sectionizes** the text into logical chunks
2. **Extracts** entities and relationships from each chunk
3. **Identifies** global entities that appear across chunks
4. **Creates** summaries at chunk and document level

## Core Components

### 1. LLMExtractor (`src/knowledge_graph/llm_extractor.py`)

**The heart of the system** - pure LLM extraction.

```python
from src.knowledge_graph.llm_extractor import LLMExtractor

# Initialize
extractor = LLMExtractor(model="gpt-4o-mini")

# Give it raw text (10 pages, 5 pages, whatever)
result = extractor.extract(
    text=raw_document_text,
    document_name="Basel III",
    context="Banking regulation"
)

# Get structured output
print(f"Chunks: {len(result.chunks)}")
print(f"Entities: {len(result.global_entities)}")
print(f"Relationships: {len(result.global_relationships)}")
print(f"Summary: {result.overall_summary}")
```

**Output Structure:**
```python
ExtractionResult(
    chunks=[
        Chunk(
            chunk_id="chunk_1",
            title="Capital Requirements",
            content="...",
            summary="...",
            key_points=["...", "..."],
            entities=[...],
            relationships=[...]
        ),
        ...
    ],
    global_entities=[...],      # Entities across multiple chunks
    global_relationships=[...],  # Cross-chunk relationships
    overall_summary="..."        # Document-level summary
)
```

### 2. HierarchicalKnowledgeGraph (`src/knowledge_graph/hierarchical_graph.py`)

Stores the extracted knowledge in a hierarchical graph structure:
- **Document level**: Overall document summaries
- **Chunk level**: Chunk summaries and content
- **Entity level**: Entities and relationships

```python
from src.knowledge_graph.hierarchical_graph import HierarchicalKnowledgeGraph

# Create graph
kg = HierarchicalKnowledgeGraph()

# Add extraction results
# Convert chunks to DocumentSection format first
kg.add_document_sections(sections)

# Search
sections = kg.search_summaries("capital requirements", level='chunk')
entities = kg.find_entities("Basel III", entity_type='REGULATION')
context = kg.get_entity_context(entity_id)

# Save/Load
kg.save("my_knowledge_graph.pkl")
kg = HierarchicalKnowledgeGraph.load("my_knowledge_graph.pkl")
```

### 3. GraphRAGRetriever (`src/knowledge_graph/graph_retrieval.py`)

Query the knowledge graph to answer questions.

```python
from src.knowledge_graph.graph_retrieval import GraphRAGRetriever

retriever = GraphRAGRetriever(
    knowledge_graph=kg,
    model="gpt-4o-mini"
)

# Answer questions
result = retriever.answer_question(
    "What are the capital requirements under Basel III?",
    max_sections=5,
    max_entities=10
)

print(result['answer'])
print(result['sources'])
```

## Quick Start

### Test the LLM Extractor

```bash
python test_llm_extractor.py
```

This will:
1. Take sample regulatory text
2. Extract chunks, entities, relationships
3. Show results
4. Save to `llm_extraction_result.json`

### Full Example (Coming Soon)

```bash
python example_graphrag.py
```

This will:
1. Read your PDF documents
2. Process with LLM extractor
3. Build knowledge graph
4. Answer questions

## How It Works

### Step 1: Give LLM Raw Text

```python
extractor = LLMExtractor()

# Just give it text - 10 pages, 5 pages, whatever
text = read_pdf_pages(pdf_file, pages=range(1, 11))  # 10 pages

result = extractor.extract(text, document_name="MiFID II")
```

### Step 2: LLM Does Everything

The LLM:
- Analyzes document structure
- Breaks into logical chunks
- Extracts entities (regulations, articles, requirements, etc.)
- Identifies relationships (REFERENCES, REQUIRES, DEFINES, etc.)
- Creates summaries

### Step 3: Build Knowledge Graph

```python
kg = HierarchicalKnowledgeGraph()

# Add each extraction result
for result in extraction_results:
    kg.add_document_sections(convert_to_sections(result))

kg.save("regulatory_kg.pkl")
```

### Step 4: Query the Graph

```python
retriever = GraphRAGRetriever(kg)

answer = retriever.answer_question(
    "What does Article 25 of MiFID II require?"
)
```

## Entity Types

The LLM extracts these entity types:

- **REGULATION**: MiFID II, Basel III, GDPR, etc.
- **ARTICLE**: Article 25, Section 4(1), etc.
- **REQUIREMENT**: Capital requirements, disclosure obligations
- **INSTITUTION**: Credit institutions, investment firms
- **FINANCIAL_INSTRUMENT**: Derivatives, bonds, shares
- **DEFINITION**: Defined terms
- **DATE**: Effective dates, deadlines
- **THRESHOLD**: 8%, €10M, etc.
- **AUTHORITY**: EBA, ESMA, ECB
- **PROCESS**: Regulatory processes
- **PENALTY**: Sanctions, fines

## Relationship Types

The LLM identifies these relationships:

- **REFERENCES**: One regulation references another
- **AMENDS**: One regulation amends another
- **SUPERSEDES**: Replaces another regulation
- **REQUIRES**: Imposes a requirement
- **DEFINES**: Defines a term
- **APPLIES_TO**: Applies to institution type
- **SPECIFIES**: Specifies details
- **EXEMPTS**: Provides exemption
- **IMPLEMENTS**: Implements another regulation
- **COMPLEMENTS**: Works with another regulation

## Advantages Over Traditional RAG

### Traditional RAG
❌ Fixed chunk sizes (lose context)
❌ Chunks don't understand document structure
❌ No entity relationships
❌ Purely keyword/semantic matching

### Knowledge Graph RAG
✅ LLM understands structure and creates logical chunks
✅ Extracts meaningful entities and relationships
✅ Can traverse relationships to find related information
✅ Summaries at multiple levels (chunk, document)
✅ Better for complex queries that span multiple sections

## Configuration

### Models

- **gpt-4o**: Best quality, more expensive (~$5-10 per 1M input tokens)
- **gpt-4o-mini**: Good quality, cheaper (~$0.15 per 1M input tokens) - **RECOMMENDED**

### Processing Settings

```python
# Extractor
extractor = LLMExtractor(
    model="gpt-4o-mini",  # or "gpt-4o" for best quality
)

# How much text to give LLM at once
text = get_pages(pdf, num_pages=10)  # You control this
```

## Cost Estimation

For 23 documents, ~200 pages each:

- **Input**: ~200 pages × 3000 chars/page × 23 docs = ~13.8M chars = ~3.5M tokens
- **Output**: ~16k tokens per extraction × 200 pages = ~3.2M tokens

**Cost (gpt-4o-mini)**:
- Input: 3.5M tokens × $0.15/1M = $0.53
- Output: 3.2M tokens × $0.60/1M = $1.92
- **Total: ~$2.50** for processing all 23 documents

**Cost (gpt-4o)**:
- Input: 3.5M tokens × $2.50/1M = $8.75
- Output: 3.2M tokens × $10/1M = $32
- **Total: ~$40** for processing all 23 documents

## Next Steps

1. **Test the extractor**: `python test_llm_extractor.py`
2. **Process your documents**: Feed 10-page sections to the extractor
3. **Build the graph**: Add all extraction results to HierarchicalKnowledgeGraph
4. **Query**: Use GraphRAGRetriever to answer questions

## Files

- `src/knowledge_graph/llm_extractor.py` - Core LLM extraction
- `src/knowledge_graph/hierarchical_graph.py` - Graph storage
- `src/knowledge_graph/graph_retrieval.py` - Query/retrieval
- `test_llm_extractor.py` - Simple test
- `example_graphrag.py` - Full pipeline (WIP)
