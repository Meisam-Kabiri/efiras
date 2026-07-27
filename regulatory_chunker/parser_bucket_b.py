#!/usr/bin/env python3
"""
parser_bucket_b.py

Generalized chunking script for Bucket B (flat text, PDFs, and flat HTML documents).
Reads BUCKET_B from buckets_config.py, fetches and caches files,
and processes them using a Stateful Stack Machine.
"""

import json
import re
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
try:
    from .buckets_config import BUCKET_B
except ImportError:
    from buckets_config import BUCKET_B

# Setup paths
CACHE_DIR = Path("flat_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("chunks_output")
OUT_DIR.mkdir(exist_ok=True)

# Try imports for PDF extraction
PDF_SUPPORT = False
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_SUPPORT = True
    except ImportError:
        pass

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())

def fetch_document(doc_id: str, url: str, fmt: str) -> Path:
    ext = "pdf" if fmt == "pdf" else "html"
    local_path = CACHE_DIR / f"{doc_id.lower()}.{ext}"
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    print(f"[{doc_id}] Downloading {fmt.upper()} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    
    local_path.write_bytes(r.content)
    print(f"[{doc_id}] Cached to {local_path} ({len(r.content)/1e6:.2f} MB)")
    return local_path

def extract_pdf_text_lines(filepath: Path) -> list:
    """Extracts lines from PDF using available PDF libraries."""
    lines = []
    if not PDF_SUPPORT:
        raise RuntimeError("PDF parsing libraries (pdfplumber or PyMuPDF/fitz) are not installed. "
                           "Please run 'pip install pdfplumber' or 'pip install pymupdf' to enable PDF support.")
    
    # Try pdfplumber first
    if "pdfplumber" in sys.modules:
        print("Using pdfplumber for PDF extraction...")
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines.extend(text.splitlines())
    # Try PyMuPDF (fitz) as fallback
    elif "fitz" in sys.modules:
        print("Using PyMuPDF (fitz) for PDF extraction...")
        doc = fitz.open(filepath)
        for page in doc:
            text = page.get_text()
            if text:
                lines.extend(text.splitlines())
    return lines

def extract_html_flat_lines(filepath: Path) -> list:
    """Extracts text lines from a flat HTML structure (e.g. Dodd-Frank pre/table structure)."""
    soup = BeautifulSoup(filepath.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    # Get all text blocks
    text = soup.get_text()
    return text.splitlines()

class StatefulStackParser:
    """
    Maintains a structural hierarchy stack (e.g. Division -> Title -> Section)
    processing text lines sequentially and emitting chunks.
    """
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        # Hierarchical level maps: (depth_priority, regex_pattern)
        self.patterns = {
            "division": (1, r"^(?:DIVISION|Division)\s+([A-Z0-9]+)\s*—?\s*(.*)"),
            "title": (2, r"^(?:TITLE|Title)\s+([A-Z0-9]+|I|V|X|L|C|D|M)\s*—?\s*(.*)"),
            "subtitle": (3, r"^(?:SUBTITLE|Subtitle)\s+([A-Z0-9]+)\s*—?\s*(.*)"),
            "part": (4, r"^(?:PART|Part)\s+([A-Z0-9]+)\s*—?\s*(.*)"),
            "section": (5, r"^(?:SEC\.|Section)\s+([0-9A-Za-z\-]+)\.?\s*(.*)"),
        }

    def process(self, lines: list) -> list:
        chunks = []
        stack = []  # Stack of {"type": str, "label": str, "heading": str, "level": int}
        current_chunk_text = []

        def emit_current_chunk():
            if not current_chunk_text:
                return
            
            path_metadata = [
                {"type": item["type"], "label": item["label"], "heading": item["heading"]}
                for item in stack
            ]
            
            if stack:
                deepest = stack[-1]
                t_type = deepest["type"]
                t_label = deepest["label"]
                chunk_id = f"{self.doc_id}/{t_type}_{t_label}"
                citation = f"{self.doc_id} {t_type.capitalize()} {t_label}"
            else:
                chunk_id = f"{self.doc_id}/body"
                citation = f"{self.doc_id}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": self.doc_id,
                "path": path_metadata,
                "citation": citation,
                "text": " ".join(current_chunk_text)
            })

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            matched_level = None
            matched_type = None
            matched_label = None
            matched_heading = None
            
            for key, (level, regex) in self.patterns.items():
                m = re.match(regex, line_stripped)
                if m:
                    matched_level = level
                    matched_type = key
                    matched_label = m.group(1)
                    matched_heading = m.group(2).strip() if len(m.groups()) > 1 else None
                    break
            
            if matched_level:
                emit_current_chunk()
                current_chunk_text = []
                
                # Pop anything equal to or deeper than matched level
                stack = [item for item in stack if item["level"] < matched_level]
                
                new_node = {
                    "type": matched_type,
                    "label": matched_label,
                    "heading": matched_heading,
                    "level": matched_level
                }
                stack.append(new_node)
                
                if matched_heading:
                    current_chunk_text.append(matched_heading)
            else:
                current_chunk_text.append(line_stripped)

        # Emit final trailing chunk
        emit_current_chunk()
        return chunks

def main():
    print("=== Starting Bucket B Parser ===")
    if not PDF_SUPPORT:
        print("WARNING: pdfplumber or PyMuPDF are not installed. PDF parsing will be skipped.", file=sys.stderr)
        print("To install PDF dependencies: pip install pdfplumber", file=sys.stderr)
        
    for doc_id, name, url, fmt in BUCKET_B:
        # Avoid processing DODD_FRANK_P2 twice as noted in buckets_config.py
        if doc_id == "DODD_FRANK_P2":
            print(f"[{doc_id}] Skipping secondary pass of DODD_FRANK to avoid duplication.")
            continue
            
        try:
            filepath = fetch_document(doc_id, url, fmt)
            
            # Extract lines
            if fmt == "pdf":
                if not PDF_SUPPORT:
                    print(f"[{doc_id}] Skipping PDF due to missing libraries.")
                    continue
                lines = extract_pdf_text_lines(filepath)
            else:
                lines = extract_html_flat_lines(filepath)
                
            parser = StatefulStackParser(doc_id)
            chunks = parser.process(lines)
            
            output_file = OUT_DIR / f"{doc_id.lower()}_chunks.json"
            output_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{doc_id}] Saved {len(chunks)} chunks to {output_file}\n")
            
        except Exception as e:
            print(f"Error processing {doc_id}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
