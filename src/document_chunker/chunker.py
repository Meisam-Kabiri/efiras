import re
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RegulationChunker:
    def __init__(self, raw_text: str, max_chunk_size: int = 1000):
        self.raw_text = raw_text
        self.chunks = []
        self.max_chunk_size = max_chunk_size
        
        # Add LangChain splitter for oversized chunks and fallback
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=70,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self) -> List[Dict]:
        # Try structured chunking first
        structured_chunks = self._try_structured_chunking()
        
        if structured_chunks:
            return self._process_chunks(structured_chunks)
        else:
            # Fall back to semantic chunking
            return self._fallback_chunking()

    def _try_structured_chunking(self) -> Optional[List[Dict]]:
        """Attempt to chunk using document structure"""
        # Enhanced heading patterns - more flexible
        heading_patterns = [
            # Original patterns
            r"^(Part\s+\w+.*?)$",
            r"^(Chapter\s+\d+.*?)$", 
            r"^(Sub-chapter\s+\d+\.\d+.*?)$",
            r"^(Section\s+\d+\.\d+\.\d+.*?)$",
            
            # # Additional common patterns
            # r"^(\d+\.\s+.+)$",  # 1. Title
            # r"^(\d+\.\d+\s+.+)$",  # 1.1 Subtitle  
            # r"^(\d+\.\d+\.\d+\s+.+)$",  # 1.1.1 Sub-subtitle
            # r"^([A-Z][A-Z\s]+)$",  # ALL CAPS HEADINGS
            # r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$",  # Title Case Headings
            # r"^(Article\s+\d+.*?)$",  # Article 1
            # r"^(Rule\s+\d+.*?)$",  # Rule 1
            # r"^([IVX]+\.\s+.+)$",  # Roman numerals
            # r"^([A-Z]\.\s+.+)$",  # A. Title
        ]
        
        combined_pattern = re.compile("|".join(heading_patterns), re.MULTILINE)
        matches = list(combined_pattern.finditer(self.raw_text))
        
        # Check if we found meaningful structure (at least 3 headings)
        if len(matches) < 3:
            return None
            
        # Check if headings are reasonably distributed
        text_length = len(self.raw_text)
        avg_section_length = text_length / len(matches)
        
        # If sections are too small (< 100 chars) or too large (> 10000 chars), 
        # the structure might not be reliable
        if avg_section_length < 100 or avg_section_length > 10000:
            return None
        
        temp_chunks = []
        for i, match in enumerate(matches):
            heading = next(g for g in match.groups() if g is not None)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            content = self.raw_text[start:end].strip()

            if content:
                temp_chunks.append({
                    "heading": heading.strip(),
                    "content": content,
                    "hierarchy": self._parse_hierarchy(heading)
                })
        
        return temp_chunks

    def _fallback_chunking(self) -> List[Dict]:
        """Fallback chunking when no clear structure is found"""
        print("No clear document structure found. Using semantic chunking...")
        
        # Try to identify natural break points
        chunks = self._semantic_chunking()
        
        if not chunks:
            # Last resort: simple text splitting
            chunks = self._simple_text_splitting()
        
        return chunks

    def _semantic_chunking(self) -> List[Dict]:
        """Attempt to chunk based on semantic breaks"""
        chunks = []
        
        # Look for paragraph breaks, numbered lists, bullet points
        semantic_breaks = [
            r'\n\n+',  # Double line breaks
            r'\n\d+\.\s+',  # Numbered lists
            r'\n[•\-\*]\s+',  # Bullet points
            r'\n[A-Z][a-z]+:',  # Labels like "Note:", "Important:"
            r'\n(?=\([a-z]\))',  # Lettered lists (a), (b), (c)
        ]
        
        # Split text on semantic breaks
        text = self.raw_text
        for pattern in semantic_breaks:
            parts = re.split(pattern, text, flags=re.MULTILINE)
            if len(parts) > 1:
                current_chunk = ""
                chunk_count = 1
                
                for part in parts:
                    if len(current_chunk) + len(part) <= self.max_chunk_size:
                        current_chunk += part + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "heading": f"Section {chunk_count}",
                                "content": current_chunk.strip(),
                                "hierarchy": [f"Section {chunk_count}"]
                            })
                            chunk_count += 1
                        current_chunk = part + "\n\n"
                
                # Add final chunk
                if current_chunk.strip():
                    chunks.append({
                        "heading": f"Section {chunk_count}",
                        "content": current_chunk.strip(),
                        "hierarchy": [f"Section {chunk_count}"]
                    })
                break
        
        return chunks

    def _simple_text_splitting(self) -> List[Dict]:
        """Simple text splitting as last resort"""
        print("Using simple text splitting as last resort...")
        
        sub_docs = self.text_splitter.create_documents([self.raw_text])
        chunks = []
        
        for i, sub_doc in enumerate(sub_docs):
            # Try to create meaningful headings from first line
            first_line = sub_doc.page_content.split('\n')[0][:50]
            if len(first_line) > 40:
                first_line = first_line[:40] + "..."
            
            chunks.append({
                "heading": f"Part {i+1}: {first_line}",
                "content": sub_doc.page_content,
                "hierarchy": [f"Part {i+1}"]
            })
        
        return chunks

    def _process_chunks(self, temp_chunks: List[Dict]) -> List[Dict]:
        """Process chunks to handle oversized ones"""
        final_chunks = []
        
        for chunk in temp_chunks:
            if len(chunk["content"]) > self.max_chunk_size:
                # Split large chunks
                sub_docs = self.text_splitter.create_documents([chunk["content"]])
                for i, sub_doc in enumerate(sub_docs):
                    final_chunks.append({
                        "heading": f"{chunk['heading']} (Part {i+1})",
                        "content": sub_doc.page_content,
                        "hierarchy": chunk["hierarchy"]
                    })
            else:
                final_chunks.append(chunk)
        
        self.chunks = final_chunks
        return self.chunks

    def _parse_hierarchy(self, heading: str) -> List[str]:
        """Enhanced hierarchy parsing"""
        levels = []
        
        # Original patterns
        if heading.startswith("Part"):
            levels.append(heading)
        elif heading.startswith("Chapter"):
            levels.append(heading)
        elif heading.startswith("Sub-chapter"):
            levels.append(heading)
        elif heading.startswith("Section"):
            levels.append(heading)
        elif heading.startswith("Article"):
            levels.append(heading)
        elif heading.startswith("Rule"):
            levels.append(heading)
        # Numbered patterns
        elif re.match(r'^\d+\.', heading):
            levels.append(heading)
        # Roman numerals
        elif re.match(r'^[IVX]+\.', heading):
            levels.append(heading)
        # Letter patterns
        elif re.match(r'^[A-Z]\.', heading):
            levels.append(heading)
        else:
            # Generic heading
            levels.append(heading)
            
        return levels

    def get_statistics(self) -> Dict:
        """Get chunking statistics"""
        if not self.chunks:
            return {}
        
        chunk_sizes = [len(chunk["content"]) for chunk in self.chunks]
        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes)
        }
    
if __name__ == "__main__":
    # Test with structured document
    structured_text = """
    Part I: Introduction
    This is the introduction section with some content. ldfadf asdf sfdsafsa fsdaf safdas fsf asfas fasdfsaf
    
    Chapter 1: Basic Principles
    These are the basic principles that govern...
    
    Section 1.1.1: Definitions
    Here are the key definitions...
    """
    
    # Test with unstructured document
    unstructured_text = """
    This is a long document without clear structure. It contains various paragraphs
    and information that needs to be chunked properly. The document discusses
    various topics and concepts without following a traditional hierarchical format.
    


    Sometimes there are natural breaks in the text that can be used for chunking.
    Other times, the text flows continuously without clear demarcation points.
    This is a long document without clear structure. It contains various paragraphs
    and information that needs to be chunked properly. The document discusses
    various topics and concepts without following a traditional hierarchical format.


    Other times, the text flows continuously without clear demarcation points.
    lab lab lab

Sub-section 5.3.3.3. Specific responsibilities and scope of the internal audit function
270. In general, the internal audit function must review and assess whether the IFM’s central
administration and internal governance arrangements are adequate and operate effectively. In this
respect, the internal audit function must assess, among others:
•
•
•
•
•
•
•
•
the monitoring of compliance with the laws and regulations and the prudential
requirements imposed by the CSSF;
the effectiveness and efficiency of the internal control;
the adequacy of the IFM’s organisation, including, in particular, the monitoring of
delegates as well as the implementation of the procedure for the approval of new
business relationships and new products;
the adequacy of the accounting and IT function;
the adequacy of the segregation of duties and of the exercise of activities;
the accurate and complete registration of the transactions and the compilation of
accurate, comprehensive, relevant and understandable financial and prudential
information which is available without delay to the management body/governing body,
to specialised committees, where appropriate, and to the senior management and the
CSSF;
the implementation of the decisions taken by the senior management and by the persons
acting by delegation and under their responsibility;
the compliance with the procedures governing capital adequacy as specified in Chapter
3 (Own funds) of this circular;
Circular CSSF 18/698
Page 42/96•
the operation and effectiveness of the compliance and risk management functions.
271. The internal audit must be independent from the other internal control functions which it audits.
Consequently, the risk management function or the compliance function cannot be performed by the
person responsible for the internal audit function of the IFM. However, these functions may take into
account the internal audit work as regards the verification of the correct application of the standards
in force to the exercise of the activities by the IFM.
272. The internal auditor must ensure that the department applies the international standards of the
Institute of Internal Auditors or equivalent international standards.
273. The multi-year internal audit plan must be provided to the CSSF upon request.
Sub-section 5.3.3.4. Person responsible for the permanent internal audit function
274. The IFM must communicate beforehand to the CSSF the name of the person responsible for the
internal audit function supplemented by the following pieces of information and any other document
which might be subsequently indicated by the CSSF:
•
•
•
•
a recent curriculum vitae, signed and dated;
a copy of the passport/identity card;
a declaration of honour, as may be downloaded on the CSSF website (www.cssf.lu);
and
a recent extract of the criminal record, if available, or any other equivalent document.
275. In the event of a change of the person responsible for internal audit, the IFM must communicate
beforehand to the CSSF the name of the person succeeding him/her in office supplemented by the
documents referred to in point 274 above.
276. The role of the person responsible for internal audit cannot be ensured by a member of the
management body/governing body of the IFM unless s/he is member of the IFM’s senior
management.

    """
    
    chunker1 = RegulationChunker(structured_text)
    chunks1 = chunker1.chunk()
    print("Structured document chunks:", len(chunks1))

    for i, chunk in enumerate(chunks1):
        print("chunk", i, ": ", chunk["content"], "\n---\n")
    
    chunker2 = RegulationChunker(unstructured_text)
    chunks2 = chunker2.chunk()
    print("Unstructured document chunks:", len(chunks2))
    print("Statistics:", chunker2.get_statistics())
    for i, chunk in enumerate(chunks2):
        print("chunk", i, ": ", chunk["content"], "\n---\n")