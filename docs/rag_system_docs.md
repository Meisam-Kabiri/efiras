## `RAGSystem` Class Documentation

The `RAGSystem` class implements a simple yet flexible Retrieval-Augmented Generation (RAG) pipeline. It supports **local and OpenAI-based embeddings**, basic vector similarity search, and GPT-powered answer generation based on the most relevant document blocks.

---

### Purpose

This class enables document-based question answering by:

- Generating dense embeddings from document chunks
- Caching embeddings for speed and reusability
- Performing similarity search with cosine distance
- Composing context from top-K similar chunks
- Using the OpenAI API to generate answers based on context
- Returning structured responses with optional source attribution

---

### Features

- 🔍 **Dual Embedding Support**:

  - Local (`sentence-transformers`)
  - Online (`OpenAI text-embedding-3-large`)

- 🧠 **Search & Retrieval**:

  - Cosine similarity scoring
  - Top-K ranking

- 💬 **Answer Generation**:

  - Uses GPT-4 (or other GPT models) for natural language response generation

- 🗾 **Source Attribution**:

  - Optionally returns document ID, preview, and headers for transparency

---

### Initialization

```python
RAGSystem(
    model="gpt-4",
    online_embedding_model="text-embedding-3-large",
    use_local_embeddings=False,
    local_embedding_model="all-mpnet-base-v2"
)
```

---

### Core Methods

#### `create_embedding_text(block: Dict[str, Any]) -> str`

Creates a string from a text block, optionally combining enriched headers and content.

---

#### `embed_text(text: str) -> List[float]`

Encodes text into dense vector using either:

- Local sentence transformer (`SentenceTransformer`)
- OpenAI embedding API (`openai.embeddings.create()`)

---

#### `embed_blocks(blocks: List[Dict], cache_path: str) -> List[Dict]`

Creates (or loads) vector embeddings for all blocks, using a cache file to avoid recomputation.

---

#### `add_documents(blocks, cache_path, cache_file_name)`

Embeds and stores documents for future search. Saves embeddings under:

- `filename_online.json` (if using OpenAI)
- `filename_local.json` (if using local embeddings)

---

#### `search(query: str, top_k: int = 5) -> List[Dict]`

Returns the top-K most relevant documents using cosine similarity between the query embedding and block embeddings.

---

#### `answer_query(query: str, top_k: int = 5, max_context: int = 6000) -> str`

Constructs a context from top-K similar chunks and sends it to OpenAI GPT for answer generation. Truncates context if necessary.

---

#### `answer_with_sources(query: str, top_k: int = 5) -> Dict`

Returns not only the answer, but also:

- Source headers
- Snippet preview (first 200 chars)
- Page number
- Confidence score based on number of hits

---

#### `stats() -> Dict[str, int]`

Returns the number of documents currently stored in memory.

---

### Example Usage

```python
file = "data_processed/Lux_cssf18_698eng_processed_blocks.json"
with open(file, 'r') as f:
    data = json.load(f)
blocks = data.get("blocks", [])

rag = RAGSystem(use_local_embeddings=True)
rag.add_documents(blocks, cache_path="data_processed", cache_file_name="Lux_cssf18_698eng_embeddings")

answer = rag.answer_query("What is the minimum number of senior management members?")
print(answer)
```

---

### Input Requirements

Input `blocks` must be a list of dictionaries containing at least:

```json
{
  "text": "...",
  "enriched_headers": "...",
  "page": 1
}
```

---

### Output Structure (for `answer_with_sources()`)

```json
{
  "answer": "The required number is three...",
  "sources": [
    {
      "id": 42,
      "preview": "The minimum number of senior management members required is three...",
      "headers": "Part II., Chapter 6., Section 6",
      "page": 17
    }
  ],
  "confidence": 1.0
}
```

---

### Notes

- Requires OpenAI key via `.env` as `GPT_API_KEY`
- Embedding cache is used to speed up repeated runs
- `sentence-transformers` package is only needed if using local models
- For OpenAI usage, ensure `.env` is configured with a valid API key

---

### TODO


 link to it from  root `README.md`:

```md
- [RAG System Documentation](docs/rag_system.md)
```

