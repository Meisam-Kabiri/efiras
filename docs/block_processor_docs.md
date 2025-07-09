## `block_processor` Class Documentation

The `block_processor` class is responsible for performing structured post-processing and chunking on pre-extracted blocks of text from a PDF document. It refines raw text blocks into semantically meaningful segments suitable for downstream applications such as retrieval-augmented generation (RAG), information extraction, or document summarization.

### Purpose

This class works **after** a PDF has been processed by a low-level text extraction tool (like PyMuPDF). It expects pre-structured text blocks (containing bounding boxes and raw content) and enhances them by:

- Cleaning and normalizing text
- Removing header/footer noise
- Detecting and merging logically related blocks
- Annotating blocks with TOC-derived semantics (titles and headers)
- Saving enriched, structured data into JSON for persistent storage

### Workflow Overview

The main method `process_and_chunk_blocks()` executes a full transformation pipeline on the extracted document blocks. The pipeline involves:

1. **Metadata Extraction**: Extracts filename, extension, height, and page count from the input dictionary.
2. **TOC Detection**: Attempts to identify Table of Contents entries embedded in the document.
3. **Header Extraction**: Assigns hierarchical section headers to text blocks using rule-based regex patterns.
4. **Text Cleaning**: Removes special/invisible characters, fixes newlines, and strips excess whitespace.
5. **Header/Footer Removal**: Filters out structural noise based on bounding box coordinates.
6. **Colon-based Block Merging**: Merges blocks where text ends in a colon to combine titles with content.
7. **Semantic Enrichment**: Enriches block headers with their corresponding TOC titles.
8. **Paragraph Repair Across Pages**: Reattaches text fragments split between consecutive pages.
9. **Persistent Storage**: Saves all enhanced data (document metadata, TOC, processed blocks) to a JSON file.

### Key Methods

- `process_and_chunk_blocks()`: The orchestrator for the full pipeline.
- `_merge_blocks_with_colon_pattern()`: Merges section header blocks with their content.
- `_identify_header_footer_blocks()`: Flags blocks based on vertical positioning as likely noise.
- `_extract_headers()`: Uses regex rules to classify blocks into hierarchical headers.
- `_extract_toc()`: Detects and structures Table of Contents entries from text.
- `_clean_text()`: Applies rule-based text normalization.
- `_save_processed_blocks()`: Persists the final results into a JSON output.
- `_find_toc_match()` and `_enrich_blocks_with_titles()`: Align block headers with TOC titles for better semantic clarity.
- `reattach_split_paragraphs_across_pages()`: Detects and repairs paragraph fragments spread over page breaks.

### Input Format

`process_and_chunk_blocks()` expects a dictionary with at least the following keys:

```json
{
  "blocks": [
    {"page": 1, "bbox": [...], "text": "..."},
    ...
  ],
  "height": 792,
  "filename": "sample.pdf",
  "filename_without_ext": "sample",
  "pages": 10,
  "extension": ".pdf"
}
```

### Output Format

The processed data is saved in a structured JSON file containing:

- Document metadata (filename, page count, etc.)
- Table of Contents entries (if any)
- Cleaned and enriched text blocks

### Use Case

Typical usage involves invoking this processor **after** low-level text extraction using tools like `fitz` (PyMuPDF), especially when working with regulatory, academic, or structured documents that contain hierarchical organization.

---

This class promotes separation of concerns between raw extraction and structural text enrichment, allowing for modular, testable pipelines in document understanding systems.

