#!/usr/bin/env python3
"""
Text Processing Utilities
Common text processing functions used across the project
"""

import re
from typing import List


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text based on punctuation and formatting patterns.

    Rules:
    - Split on '.' followed by space and uppercase/list patterns
    - Split on '\n' followed by uppercase/list patterns
    - List patterns: (a), a), 1., 1), 1-, 1_

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Clean up extra whitespace but keep structure
    text = re.sub(r"[ \t]+", " ", text.strip())

    # Find all split positions
    sentences = []
    current_pos = 0

    # Pattern: . followed by space and sentence starter OR \n followed by sentence starter
    pattern = r"(\.\s+(?=[A-Z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\n\s*(?=[A-Za-z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\s+(?=\d+\))|\s+(?=\(\d+\)))"

    prev_sentence = ""
    for match in re.finditer(pattern, text):
        # Add text before the split
        sentence = prev_sentence + text[current_pos : match.start()].strip()
        if re.match(r"^\d+[.,:;!?]?\s*$", sentence):  ## only-number sentence
            prev_sentence += sentence + " "
        else:
            sentences.append(sentence)
            prev_sentence = ""
        current_pos = match.end() - 1  # Keep the starter character

    # Add remaining text
    if current_pos < len(text):
        sentence = text[current_pos:].strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def clean_whitespace(text: str) -> str:
    """
    Clean and normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_newlines(text: str) -> str:
    """
    Remove all newlines from text.

    Args:
        text: Input text

    Returns:
        Text without newlines
    """
    if not text:
        return ""

    return re.sub(r"\n+", " ", text)


def is_list_item(text: str) -> bool:
    """
    Check if text starts with a list pattern.

    Patterns: (a), a), 1., 1), 1-, 1_

    Args:
        text: Text to check

    Returns:
        True if text starts with list pattern
    """
    if not text:
        return False

    return bool(re.match(r"^(\([a-z]\)|[a-z]\)|\d+[\.\)\-_])", text.strip()))


def is_definition(text: str) -> bool:
    """
    Check if text looks like a definition.

    Pattern: number) "word"

    Args:
        text: Text to check

    Returns:
        True if text looks like a definition
    """
    if not text:
        return False

    return bool(re.match(r'^\d+\)\s*"', text.strip()))


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text by splitting on double newlines or newline-space-newline patterns.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    if not text:
        return []

    # Split on \n\n or \n \n patterns
    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Clean each paragraph and filter out empty ones
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    return cleaned_paragraphs


# HEADER_KEYWORDS = [
#     "Title", "Part", "Book", "Volume",
#     "Chapter", "Subchapter", "Sub-chapter", "Sub-Part",
#     "Section", "Subsection", "Sub-section", "Heading", "Division",
#     "Article", "Paragraph", "Clause", "Point", "Item",
#     "Annex", "Appendix", "Schedule", "Exhibit", "Module", "Standard"
# ]

HEADER_KEYWORDS = [
    # Primary structure
    "Title",
    "Part",
    "Chapter",
    "Sub-chapter",
    "Subchapter",
    "Section",
    "Sub-section",
    "Subsection",
    "Article",
    "Paragraph",
    # Common supplementary
    "Annex",
    "Appendix",
    "Schedule",
]

keyword_pattern = "|".join(HEADER_KEYWORDS)


def extract_header_title_from_block(text: str):
    main_header_patterns = [
        # Keyword-based headers (legal style)
        rf"^(?i)({keyword_pattern})\s+[A-Z\d.ivxlcdmIVXLCDM\-]+(?:\s*[-:\–])?\s+.+$",
        # # Multi-level numeric headings (1.2.3.4.5)
        r"^\d+\.\d+(?:\.\d+)*\s+[A-Z].*$",
        # # Optional: All-caps titles (some standards use them for hierarchy)
        r"^[A-Z][A-Z\s\-]{3,}$",  # INTRODUCTION, GENERAL PRINCIPLES
    ]

    for pattern in main_header_patterns:
        match = re.match(pattern, text)
        if match:
            return match.group()
    return None


def extract_header_identifier(header_text: str):
    """Extract just the structural part (Section 3.2.3, Part I, etc.) from headers"""

    identifier_patterns = [
        # Keyword + Roman numerals (Part I, Chapter IV)
        rf"^((?:{keyword_pattern})\s+[IVXLCDM]+)",
        # Keyword + numbers with dots (Section 3.2.3)
        rf"^((?:{keyword_pattern})\s+\d+(?:\.\d+)*)",
        # Keyword + simple number (Part 1, Annex 2)
        rf"^((?:{keyword_pattern})\s+\d+)",
        # Just numbers with dots (3.2.3)
        r"^(\d+(?:\.\d+)+)",
        # ANNEX with number
        r"^(ANNEX\s+\d+)",
        # Keyword + space then number (article 34)
        rf"(Article\s*\d+)",
    ]

    for pattern in identifier_patterns:
        match = re.search(pattern, header_text.strip(), re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def collect_unique_meaningful_headers_from_chunks(chunks: str):
    header_set_primary = set()
    cleaned_header_list = []
    for chunk in chunks:
        full_headers = chunk["enriched_headers"]
        headers_list = full_headers.split(">")
        header_set_primary.update(set(headers_list))

    for s in header_set_primary:
        s = s.strip()
        header = extract_header_identifier(s)
        if header:
            s = s.replace(header, " ")
        if len(s) > 5:
            cleaned = s.strip().lstrip(".")
            cleaned_header_list.append(cleaned)

    header_set = set(cleaned_header_list)

    return header_set


if __name__ == "__main__":
    import json

    path = "data/data_processed/Lux_cssf18_698eng_chunked_blocks.json"
    # # path = "data/data_processed/Basel_III_chunked_blocks.json"
    # with open(path, 'r') as f:

    #     chunks = json.load(f)

    chunks = []

    text = "Part II. Conditions for obtaining and maintaining the authorisation of an authorised  investment fund manager (IFM) who engages solely in the activity of management of UCIs as  laid down in Article 101(2) of the 2010 Law and Article 5(2) of the 2013 Law > Chapter 1. Basic principles > Section 5.3.2. Permanent compliance function > Sub-section 5.3.2.1. General principles > Sub-section 5.3.2.6. Obligations regarding the drawing-up of reports"
    a = {"enriched_headers": text}
    chunks.append(a)
    ss = collect_unique_meaningful_headers_from_chunks(chunks)
    for s in ss:
        print(s, "\n=======================\n")

    # for s  in ss:
    #     # if extract_header_identifier(s) and 'Article 358' in extract_header_identifier(s):
    #     if extract_header_identifier(s):
    #       print("============================================\n")
    #       print(extract_header_identifier(s), '\n+++++++++++++++++++++++++++++++++++\n')
    #       print(s.replace(extract_header_identifier(s), ''))
    # for chunk in chunks:
    #     if len(chunk['text']) > 200:
    #         s = extract_sentences(chunk['text'])
    #         for i, ss in enumerate(s):
    #             print(i)
    #             print(ss)
    #             print("-"*50)


#     Cross-Encoder Explained:
# A cross-encoder takes a query-document pair as input and outputs a relevance score.
# Unlike bi-encoders (regular embeddings) that encode query and document separately,
# cross-encoders process them together with attention mechanisms

#     Best way to fuse BM25 + Embeddings:
# Reciprocal Rank Fusion (RRF) is the most robust:
# re-ranking

# Complete pipeline:

# Stage 1: BM25 + embeddings → RRF → top 100 candidates
# Stage 2: Cross-encoder reranking → top 10 results
# Stage 3: Feed to LLM for answer generation


# Hypothetical Document Embeddings (HDE)
# HDE (Hypothetical Document Embeddings): Instead of embedding your raw query directly,
# use an LLM to first generate a hypothetical answer document
# (e.g., query "diabetes causes" → generate "Diabetes is caused by insufficient insulin production..."),
# then embed that generated text and use it for vector search - this works because the hypothetical answer is semantically
# richer and more likely to match relevant documents in your knowledge base,
# solving the problem of short/ambiguous queries that don't embed well for retrieval.
