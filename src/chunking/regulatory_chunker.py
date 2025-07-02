"""
Regulatory Document Chunker for EFIRAS
Specialized chunking for financial/legal documents
"""

import re
import uuid
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Standardized chunk format"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_document: str
    page_numbers: List[int]
    section_title: Optional[str] = None
    confidence_score: Optional[float] = None

class RegulatoryChunker:
    """Specialized chunker for regulatory/financial documents"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Regulatory document patterns
        self.section_patterns = [
            r'(Article\s+\d+[a-z]?)',
            r'(Section\s+\d+(?:\.\d+)?)',
            r'(Chapter\s+\d+)',
            r'(Part\s+[IVXLC]+)',
            r'(Regulation\s+\d+)',
            r'(Directive\s+\d+)',
            r'(\d+\.\s+[A-Z][^.]*)',  # Numbered sections
            r'([A-Z][A-Z\s]{10,})',   # ALL CAPS headers
        ]
        
        # Financial keywords for metadata
        self.financial_keywords = {
            'risk_management': ['risk', 'capital', 'liquidity', 'leverage', 'exposure'],
            'compliance': ['compliance', 'reporting', 'disclosure', 'audit', 'supervision'],
            'market': ['market', 'trading', 'investment', 'securities', 'derivatives'],
            'banking': ['bank', 'credit', 'loan', 'deposit', 'basel'],
            'insurance': ['insurance', 'solvency', 'premium', 'underwriting'],
            'funds': ['fund', 'portfolio', 'asset management', 'ucits', 'aifmd']
        }
        
        # Regulation type patterns
        self.regulation_patterns = {
            'MiFID II': r'(mifid|markets?\s+in\s+financial\s+instruments)',
            'Basel III': r'(basel|capital\s+requirements)',
            'GDPR': r'(gdpr|data\s+protection|personal\s+data)',
            'AMLD': r'(anti.?money\s+laundering|aml)',
            'UCITS': r'(ucits|undertakings\s+for\s+collective)',
            'AIFMD': r'(aifmd|alternative\s+investment\s+fund)',
            'Solvency II': r'(solvency|insurance\s+capital)',
            'CRR/CRD': r'(crr|crd|capital\s+requirements\s+regulation)',
            'EMIR': r'(emir|european\s+market\s+infrastructure)',
            'PSD2': r'(psd2?|payment\s+services\s+directive)'
        }
    
    def chunk_document(self, extracted_data: Dict[str, Any], file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Main chunking method that handles different document types"""
        
        text = extracted_data.get('text', '')
        page_texts = extracted_data.get('page_texts', [])
        metadata = extracted_data.get('metadata', {})
        
        if not text:
            raise ValueError("No text content to chunk")
        
        # Clean the text first
        cleaned_text = self._clean_regulatory_text(text)
        
        # Try semantic chunking first (by sections/articles)
        semantic_chunks = self._semantic_chunk(cleaned_text, page_texts)
        
        if semantic_chunks and len(semantic_chunks) > 1:
            chunks = semantic_chunks
        else:
            # Fallback to sliding window chunking
            chunks = self._sliding_window_chunk(cleaned_text, page_texts)
        
        # Enrich chunks with metadata
        enriched_chunks = []
        for chunk_data in chunks:
            enriched_chunk = self._enrich_chunk_metadata(
                chunk_data, 
                extracted_data, 
                str(file_path)
            )
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _clean_regulatory_text(self, text: str) -> str:
        """Clean text specifically for regulatory documents"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF artifacts
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation across lines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        # Preserve important regulatory structure
        text = re.sub(r'(Article\s+\d+)', r'\n\n\1', text)
        text = re.sub(r'(Section\s+\d+)', r'\n\n\1', text)
        text = re.sub(r'(Chapter\s+\d+)', r'\n\n\1', text)
        
        # Clean up common regulatory document artifacts
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)  # Page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        
        # Fix common character encoding issues
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '-',  # Em/en dashes
            '…': '...',          # Ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _semantic_chunk(self, text: str, page_texts: List[str]) -> List[Dict[str, Any]]:
        """Chunk by regulatory sections (Articles, Sections, etc.)"""
        
        chunks = []
        
        # Try to split by major sections
        combined_pattern = '|'.join(self.section_patterns)
        sections = re.split(f'({combined_pattern})', text, flags=re.IGNORECASE)
        
        if len(sections) <= 2:  # No meaningful sections found
            return []
        
        current_chunk = ""
        current_section = ""
        current_pages = []
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Check if this is a section header
            is_header = any(re.match(pattern, section.strip(), re.IGNORECASE) 
                          for pattern in self.section_patterns)
            
            if is_header:
                # Save previous chunk if it exists
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'section_title': current_section,
                        'pages': current_pages.copy()
                    })
                
                # Start new chunk
                current_section = section.strip()
                current_chunk = section
                current_pages = self._find_page_numbers(section, page_texts)
            else:
                # Add content to current chunk
                current_chunk += " " + section
                current_pages.extend(self._find_page_numbers(section, page_texts))
                
                # Split if chunk gets too large
                if len(current_chunk) > self.chunk_size * 1.5:
                    # Try to split at paragraph boundaries
                    paragraphs = current_chunk.split('\n\n')
                    
                    temp_chunk = current_section
                    for para in paragraphs:
                        if len(temp_chunk + para) > self.chunk_size:
                            if temp_chunk.strip():
                                chunks.append({
                                    'content': temp_chunk.strip(),
                                    'section_title': current_section,
                                    'pages': list(set(current_pages))
                                })
                            temp_chunk = current_section + " " + para
                        else:
                            temp_chunk += "\n\n" + para
                    
                    current_chunk = temp_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'section_title': current_section,
                'pages': list(set(current_pages))
            })
        
        return chunks
    
    def _sliding_window_chunk(self, text: str, page_texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback sliding window chunking with overlap"""
        
        chunks = []
        words = text.split()
        
        # Estimate words per chunk (roughly 4 chars per word)
        words_per_chunk = self.chunk_size // 4
        overlap_words = self.overlap // 4
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            # Find which pages this chunk spans
            pages = self._find_page_numbers(chunk_text, page_texts)
            
            chunks.append({
                'content': chunk_text,
                'section_title': None,
                'pages': pages
            })
            
            if i + words_per_chunk >= len(words):
                break
        
        return chunks
    
    def _find_page_numbers(self, chunk_text: str, page_texts: List[str]) -> List[int]:
        """Find which page numbers a chunk spans"""
        if not page_texts:
            return [1]  # Default to page 1 if no page info
        
        pages = []
        
        # Simple approach: check if chunk text appears in page text
        for page_num, page_text in enumerate(page_texts, 1):
            # Check if significant portion of chunk appears in this page
            chunk_words = set(chunk_text.lower().split())
            page_words = set(page_text.lower().split())
            
            if len(chunk_words) == 0:
                continue
            
            overlap_ratio = len(chunk_words & page_words) / len(chunk_words)
            
            if overlap_ratio > 0.3:  # 30% word overlap threshold
                pages.append(page_num)
        
        return pages if pages else [1]
    
    def _enrich_chunk_metadata(
        self, 
        chunk_data: Dict[str, Any], 
        extracted_data: Dict[str, Any], 
        source_file: str
    ) -> DocumentChunk:
        """Add rich metadata to chunks"""
        
        content = chunk_data['content']
        
        # Generate unique chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Detect regulation type
        regulation_type = self._detect_regulation_type(content)
        
        # Extract financial keywords
        keywords = self._extract_keywords(content)
        
        # Detect jurisdiction
        jurisdiction = self._detect_jurisdiction(content)
        
        # Extract dates
        dates = self._extract_dates(content)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(chunk_data, extracted_data)
        
        metadata = {
            'source_document': Path(source_file).name,
            'full_path': source_file,
            'regulation_type': regulation_type,
            'jurisdiction': jurisdiction,
            'keywords': keywords,
            'dates': dates,
            'word_count': len(content.split()),
            'char_count': len(content),
            'processor_used': extracted_data.get('processor_used'),
            'extraction_method': extracted_data.get('extraction_method'),
            'chunk_method': 'semantic' if chunk_data.get('section_title') else 'sliding_window'
        }
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            chunk_id=chunk_id,
            source_document=Path(source_file).name,
            page_numbers=chunk_data.get('pages', [1]),
            section_title=chunk_data.get('section_title'),
            confidence_score=confidence
        )
    
    def _detect_regulation_type(self, text: str) -> Optional[str]:
        """Detect type of regulation from text content"""
        text_lower = text.lower()
        
        for reg_type, pattern in self.regulation_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return reg_type
        
        return None
    
    def _extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract financial/regulatory keywords by category"""
        text_lower = text.lower()