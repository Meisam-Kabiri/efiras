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
    text = re.sub(r'[ \t]+', ' ', text.strip())
    
    # Find all split positions
    sentences = []
    current_pos = 0
    
    # Pattern: . followed by space and sentence starter OR \n followed by sentence starter
    pattern = r'(\.\s+(?=[A-Z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\n\s*(?=[A-Za-z]|\([a-z]\)|[a-z]\)|\d+[\.\)\-_])|\s+(?=\d+\))|\s+(?=\(\d+\)))'
    
    prev_sentence = ''
    for match in re.finditer(pattern, text):
        # Add text before the split
        sentence = prev_sentence+text[current_pos:match.start()].strip()
        if re.match(r'^\d+[.,:;!?]?\s*$', sentence): ## only-number sentence
            prev_sentence += sentence+' '
        else:
            sentences.append(sentence)
            prev_sentence = ''
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
    text = re.sub(r'\s+', ' ', text)
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
    
    return re.sub(r'\n+', ' ', text)


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
    
    return bool(re.match(r'^(\([a-z]\)|[a-z]\)|\d+[\.\)\-_])', text.strip()))


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
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
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
    "Title", "Part", "Chapter", "Sub-chapter", "Subchapter",
    "Section", "Sub-section", "Subsection", 
    "Article", "Paragraph",
    
    # Common supplementary
    "Annex", "Appendix", "Schedule"
]

keyword_pattern = "|".join(HEADER_KEYWORDS)

def extract_header_title_from_block(text:str):
    main_header_patterns = [
    # Keyword-based headers (legal style)
    rf"^(?i)({keyword_pattern})\s+[A-Z\d.ivxlcdmIVXLCDM\-]+(?:\s*[-:\–])?\s+.+$",

    # # Multi-level numeric headings (1.2.3.4.5)
    r'^\d+\.\d+(?:\.\d+)*\s+[A-Z].*$',

    # # Optional: All-caps titles (some standards use them for hierarchy)
    r"^[A-Z][A-Z\s\-]{3,}$",  # INTRODUCTION, GENERAL PRINCIPLES
    ]

    for pattern in main_header_patterns:
        match = re.match(pattern, text)
        if match:
            return match.group()
    return None


def extract_header_identifier(header_text:str):
      """Extract just the structural part (Section 3.2.3, Part I, etc.) from headers"""

      identifier_patterns = [
          # Keyword + Roman numerals (Part I, Chapter IV)
          rf'^((?:{keyword_pattern})\s+[IVXLCDM]+)',

          # Keyword + numbers with dots (Section 3.2.3) 
          rf'^((?:{keyword_pattern})\s+\d+(?:\.\d+)*)',

          # Keyword + simple number (Part 1, Annex 2)
          rf'^((?:{keyword_pattern})\s+\d+)',

          # Just numbers with dots (3.2.3)
          r'^(\d+(?:\.\d+)+)',

          # ANNEX with number
          r'^(ANNEX\s+\d+)',
      ]

      for pattern in identifier_patterns:
          match = re.search(pattern, header_text.strip(), re.IGNORECASE)
          if match:
              return match.group(1)
          

      return None

if __name__ == "__main__":
    import json
    with open("data_processed/Lux_cssf18_698eng_chunked_blocks.json", 'r') as f:
        chunks = json.load(f)

    for chunk in chunks:
        if len(chunk['text']) > 200:
            s = extract_sentences(chunk['text'])
            for i, ss in enumerate(s):
                print(i)
                print(ss)
                print("-"*50)