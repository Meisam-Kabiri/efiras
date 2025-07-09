## `BlockChunker` Class Documentation

The `BlockChunker` class provides a simple yet flexible way to split oversized blocks of text into manageable chunks. It is particularly useful in natural language processing pipelines for **document chunking**, especially when preparing content for tasks such as RAG, summarization, or indexing.

---

### Purpose

This class inspects each input block and splits it into smaller parts if the text exceeds a defined maximum size. The splitting strategy is tiered:

1. **Paragraph Splitting** (preferred)
2. **Sentence Splitting** (fallback)
3. **Word-Based Splitting** (guaranteed fallback)

This ensures both performance and robustness when dealing with large chunks of text.

---

### Initialization

```python
BlockChunker(max_chunk_size=2000)
```

- `max_chunk_size`: Maximum number of characters allowed per chunk (default: `2000`)

---

### Core Method

#### `chunk_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]`

- Iterates over blocks.
- Splits oversized ones using the best available method.
- Returns a new list of chunks, preserving metadata like page and headers.

---

### Internal Splitting Methods

#### `_split_big_text(text: str) -> List[str]`

Main logic to determine how to split a long string, trying in order:

1. `_split_by_paragraphs()`
2. `_split_by_sentences()`
3. `_split_by_words()`

Only resorts to the next method if current one yields oversized chunks.

---

#### `_split_by_paragraphs(text: str)`

Splits the input by double newlines (`\n\n`) and combines resulting pieces without exceeding `max_chunk_size`.

---

#### `_split_by_sentences(sentences: List[str])`

Takes a list of sentences and merges them into compliant chunks.

---

#### `_split_by_words(text: str)`

Last-resort splitter that slices based on words. Always produces valid chunks.

---

#### `_combine_pieces(pieces: List[str], separator: str) -> List[str]`

Shared utility that combines a list of smaller strings into valid chunks.

---

#### `_all_chunks_fit(chunks: List[str]) -> bool`

Returns True if all chunks are within the allowed size.

---

### Usage Example

```python
from chunking.block_chunker import BlockChunker
import json
from pathlib import Path

file_path = Path('data_processed/Lux_cssf18_698eng_processed_blocks.json')
if file_path.exists():
    with open(file_path, 'r') as f:
        data = json.load(f)
    blocks = data.get('blocks', [])

    chunker = BlockChunker(max_chunk_size=1000)
    chunked_blocks = chunker.chunk_blocks(blocks)

    output_path = file_path.parent / f"{file_path.stem}_chunked.json"
    data['blocks'] = chunked_blocks
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
```

---

### Input Format

Each block must contain at least:

```json
{
  "text": "...",
  "page": 1
}
```

---

### Output Format

A list of blocks like:

```json
{
  "text": "...",
  "page": 1,
  "chunk_id": 0
}
```

---

### Notes

- Input blocks remain unchanged unless their text exceeds the limit.
- Chunk IDs are added to indicate splits.
- Works well in preprocessing for retrieval or QA systems.

---

### TODO


link it in  main `README.md` if needed:

