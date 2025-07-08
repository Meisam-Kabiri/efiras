import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

    

class RegulatoryChunkingSystem:
    def __init__(self, 
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 2000,
                 overlap_percentage: float = 0.15,
                 semantic_model: str = "all-MiniLM-L6-v2"):
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_percentage = overlap_percentage
        self.semantic_model = SentenceTransformer(semantic_model)
        
        # Regulatory patterns
        self.section_patterns = {
            'article': r'Article\s+\d+',
            'section': r'Section\s+\d+(?:\.\d+)*',
            'subsection': r'Subsection\s+[a-zA-Z]+',
            'paragraph': r'Paragraph\s+\d+',
            'clause': r'Clause\s+\d+',
            'definition': r'Definition\s+\d+|means:|defined as:|shall mean',
            'requirement': r'shall|must|required to|obligated to',
            'penalty': r'penalty|fine|sanction|violation|non-compliance',
            'procedure': r'procedure|process|steps|methodology'
        }
        
        self.citation_patterns = [
            r'\b\d+\s+U\.S\.C\.\s+§\s+\d+',  # US Code
            r'\b\d+\s+CFR\s+\d+\.\d+',       # Code of Federal Regulations
            r'Section\s+\d+(?:\.\d+)*',       # Internal sections
            r'Article\s+\d+',                 # Articles
            r'Regulation\s+\d+',              # Regulations
        ]

    def process_document(self, document_text: str) -> List[Dict[str, any]]:
        """Main method to process a regulatory document"""
        
        # Step 1: Hierarchical Structure Analysis
        hierarchical_chunks = self._hierarchical_chunking(document_text)
        
        # Step 2: Apply Semantic Grouping
        semantic_chunks = self._apply_semantic_chunking(hierarchical_chunks)
        
        # Step 3: Add Overlapping
        final_chunks = self._add_overlapping(semantic_chunks)
        
        # Step 4: Validate and Post-process
        validated_chunks = self._validate_chunks(final_chunks)
        
        return validated_chunks
    
    def _hierarchical_chunking(self, text: str) -> List[Dict[str, any]]:
        """First pass: Structure-aware chunking"""
        chunks = []
        current_hierarchy = []
        
        lines = text.split('\n')
        current_chunk = ""
        current_start = 0
        
        for i, line in enumerate(lines):
            # Detect hierarchical markers
            hierarchy_level = self._detect_hierarchy_level(line)
            
            if hierarchy_level:
                # Save previous chunk if exists
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        current_hierarchy.copy(),
                        current_start,
                        i
                    )
                    chunks.append(chunk)
                
                # Update hierarchy
                current_hierarchy = self._update_hierarchy(current_hierarchy, hierarchy_level)
                current_chunk = line + '\n'
                current_start = i
            else:
                current_chunk += line + '\n'
                
                # Check if chunk is getting too large
                if len(current_chunk) > self.max_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk.strip(),
                        current_hierarchy.copy(),
                        current_start,
                        i
                    )
                    chunks.append(chunk)
                    current_chunk = ""
                    current_start = i + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                current_hierarchy.copy(),
                current_start,
                len(lines)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _apply_semantic_chunking(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Second pass: Semantic grouping and refinement"""
        
        # Group chunks by semantic similarity
        embeddings = self.semantic_model.encode([chunk["content"] for chunk in chunks])
        
        # Cluster similar chunks
        n_clusters = min(len(chunks) // 3, 10)  # Adaptive cluster number
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            for i, chunk in enumerate(chunks):
                chunk["semantic_cluster"] = cluster_labels[i]
        
        # Merge small related chunks
        merged_chunks = self._merge_small_related_chunks(chunks, embeddings)
        
        return merged_chunks

    def _add_overlapping(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Third pass: Add overlapping content"""
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Calculate overlap with previous chunk
            if i > 0:
                overlap_size = int(len(chunk["content"]) * self.overlap_percentage)
                prev_chunk = chunks[i-1]
                
                # Add overlap from previous chunk
                if len(prev_chunk["content"]) > overlap_size:
                    overlap_content = prev_chunk["content"][-overlap_size:]
                    chunk["content"] = overlap_content + "\n" + chunk["content"]
                    
                    # Track overlap relationships
                    if chunk["overlap_with"] is None:
                        chunk["overlap_with"] = []
                    chunk["overlap_with"].append(prev_chunk["chunk_id"])
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks

    def _validate_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Final validation and cleanup"""
        
        validated_chunks = []
        
        for chunk in chunks:
            # Ensure minimum size
            if len(chunk["content"]) < self.min_chunk_size:
                # Try to merge with next chunk
                continue
            
            # Validate legal citations aren't broken
            chunk = self._fix_broken_citations(chunk)
            
            # Ensure cross-references are intact
            chunk = self._validate_cross_references(chunk)
            
            validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _detect_hierarchy_level(self, line: str) -> Optional[Dict[str, str]]:
        """Detect hierarchical structure markers"""
        
        for level, pattern in self.section_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                match = re.search(pattern, line, re.IGNORECASE)
                return {
                    'level': level,
                    'identifier': match.group(),
                    'full_line': line
                }
        return None
    
    def _create_chunk(self, content: str, hierarchy: List[str], start: int, end: int) -> List[Dict[str, any]]:
        """Create a regulatory chunk with metadata"""
        
        chunk_id = f"chunk_{start}_{end}"
        
        # Extract legal citations
        citations = []
        for pattern in self.citation_patterns:
            citations.extend(re.findall(pattern, content))
        
        # Extract cross-references
        cross_refs = re.findall(r'see\s+(?:Section|Article|Clause)\s+\d+(?:\.\d+)*', content, re.IGNORECASE)
        
        # Determine chunk type
        chunk_type = self._classify_chunk_type(content)
        
        # return RegulatoryChunk(
        #     content=content,
        #     chunk_id=chunk_id,
        #     section_hierarchy=hierarchy,
        #     chunk_type=chunk_type,
        #     legal_citations=citations,
        #     cross_references=cross_refs,
        #     start_position=start,
        #     end_position=end
        # )
        return {
            "content":content,
            "chunk_id":chunk_id,
            "section_hierarchy":hierarchy,
            "chunk_type":chunk_type,
            "legal_citations":citations,
            "cross_references":cross_refs,
            "start_position":start,
            "end_position":end,
            "overlap_with": None  # Track overlaps with other chunks
        }
    
    def _classify_chunk_type(self, content: str) -> str:
        """Classify the type of regulatory content"""
        
        content_lower = content.lower()
        
        if re.search(self.section_patterns['definition'], content_lower):
            return "definition"
        elif re.search(self.section_patterns['requirement'], content_lower):
            return "requirement"
        elif re.search(self.section_patterns['penalty'], content_lower):
            return "penalty"
        elif re.search(self.section_patterns['procedure'], content_lower):
            return "procedure"
        else:
            return "general"

    def _update_hierarchy(self, current_hierarchy: List[str], new_level: Dict[str, str]) -> List[str]:
        """Update hierarchical structure based on new level detected"""
        
        level_order = ['article', 'section', 'subsection', 'paragraph', 'clause']
        new_level_name = new_level['level']
        new_level_id = new_level['identifier']
        
        if new_level_name in level_order:
            level_index = level_order.index(new_level_name)
            # Keep hierarchy up to current level
            updated_hierarchy = current_hierarchy[:level_index]
            updated_hierarchy.append(new_level_id)
            return updated_hierarchy
        
        return current_hierarchy

    def _merge_small_related_chunks(self, chunks: List[Dict[str, any]], embeddings: np.ndarray) -> List[Dict[str, any]]:
        """Merge small chunks that are semantically related"""
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next similar chunk
            if len(current_chunk["content"]) < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # Calculate semantic similarity
                similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                
                if similarity > 0.7:  # High similarity threshold
                    # Merge chunks
                    merged_content = current_chunk["content"] + "\n\n" + next_chunk["content"]
                    # merged_chunk = RegulatoryChunk(
                    #     content=merged_content,
                    #     chunk_id=f"merged_{current_chunk.chunk_id}_{next_chunk.chunk_id}",
                    #     section_hierarchy=current_chunk.section_hierarchy,
                    #     chunk_type=current_chunk.chunk_type,
                    #     legal_citations=current_chunk.legal_citations + next_chunk.legal_citations,
                    #     cross_references=current_chunk.cross_references + next_chunk.cross_references,
                    #     start_position=current_chunk.start_position,
                    #     end_position=next_chunk.end_position
                    # )
                    merged_chunk = {
                        "content":merged_content,
                        "chunk_id":f"merged_{current_chunk['chunk_id']}_{next_chunk['chunk_id']}",
                        "section_hierarchy":current_chunk["section_hierarchy"],
                        "chunk_type":current_chunk["chunk_type"],
                        "legal_citations":current_chunk["legal_citations"] + next_chunk["legal_citations"],
                        "cross_references":current_chunk["cross_references"] + next_chunk["cross_references"],
                        "start_position":current_chunk["start_position"],
                        "end_position":next_chunk["end_position"],
                        "overlap_with": current_chunk["overlap_with"] if current_chunk["overlap_with"] else None
                    }
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's merged
                else:
                    merged_chunks.append(current_chunk)
                    i += 1
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks

    def _fix_broken_citations(self, chunk: Dict[str, any]) -> Dict[str, any]:
        """Ensure legal citations are not broken at chunk boundaries"""
        
        # Check if chunk ends with incomplete citation
        for pattern in self.citation_patterns:
            matches = list(re.finditer(pattern, chunk["content"]))
            if matches:
                last_match = matches[-1]
                # If citation is cut off at the end, mark for potential merging
                if last_match.end() >= len(chunk["content"]) - 10:
                    # Citation might be incomplete - this would need additional logic
                    # to check with next chunk
                    pass
        
        return chunk

    def _validate_cross_references(self, chunk: Dict[str, any]) -> Dict[str, any]:
        """Ensure cross-references are maintained"""
        
        # This is a placeholder for cross-reference validation
        # In practice, you'd maintain a reference map and ensure
        # referenced content is accessible
        
        return chunk
 
# if __name__ == "__main__":
#     # Test with structured document
#     structured_text = """
#     Part I: Introduction
#     This is the introduction section with some content. ldfadf asdf sfdsafsa fsdaf safdas fsf asfas fasdfsaf
    
#     Chapter 1: Basic Principles
#     These are the basic principles that govern...
    
#     Section 1.1.1: Definitions
#     Here are the key definitions...
#     """
    
#     # Test with unstructured document
#     unstructured_text = """
#     This is a long document without clear structure. It contains various paragraphs
#     and information that needs to be chunked properly. The document discusses
#     various topics and concepts without following a traditional hierarchical format.
    


#     Sometimes there are natural breaks in the text that can be used for chunking.
#     Other times, the text flows continuously without clear demarcation points.
#     This is a long document without clear structure. It contains various paragraphs
#     and information that needs to be chunked properly. The document discusses
#     various topics and concepts without following a traditional hierarchical format.


#     Other times, the text flows continuously without clear demarcation points.
#     lab lab lab

# Sub-section 5.3.3.3. Specific responsibilities and scope of the internal audit function
# 270. In general, the internal audit function must review and assess whether the IFM’s central administration and internal governance arrangements are adequate and operate effectively. In this respect, the internal audit function must assess, among others:
# the monitoring of compliance with the laws and regulations and the prudential
# requirements imposed by the CSSF;
# the effectiveness and efficiency of the internal control;
# the adequacy of the IFM’s organisation, including, in particular, the monitoring of
# delegates as well as the implementation of the procedure for the approval of new
# business relationships and new products;
# the adequacy of the accounting and IT function;
# the adequacy of the segregation of duties and of the exercise of activities;
# the accurate and complete registration of the transactions and the compilation of
# accurate, comprehensive, relevant and understandable financial and prudential
# information which is available without delay to the management body/governing body,
# to specialised committees, where appropriate, and to the senior management and the
# CSSF;
# the implementation of the decisions taken by the senior management and by the persons
# acting by delegation and under their responsibility;
# the compliance with the procedures governing capital adequacy as specified in Chapter
# 3 (Own funds) of this circular;
# Circular CSSF 18/698
# Page 42/96•
# the operation and effectiveness of the compliance and risk management functions.
# 271. The internal audit must be independent from the other internal control functions which it audits.
# Consequently, the risk management function or the compliance function cannot be performed by the
# person responsible for the internal audit function of the IFM. However, these functions may take into
# account the internal audit work as regards the verification of the correct application of the standards
# in force to the exercise of the activities by the IFM.
# 272. The internal auditor must ensure that the department applies the international standards of the
# Institute of Internal Auditors or equivalent international standards.
# 273. The multi-year internal audit plan must be provided to the CSSF upon request.
# Sub-section 5.3.3.4. Person responsible for the permanent internal audit function
# 274. The IFM must communicate beforehand to the CSSF the name of the person responsible for the
# internal audit function supplemented by the following pieces of information and any other document
# which might be subsequently indicated by the CSSF:
# •
# •
# •
# •
# a recent curriculum vitae, signed and dated;
# a copy of the passport/identity card;
# a declaration of honour, as may be downloaded on the CSSF website (www.cssf.lu);
# and
# a recent extract of the criminal record, if available, or any other equivalent document.
# 275. In the event of a change of the person responsible for internal audit, the IFM must communicate
# beforehand to the CSSF the name of the person succeeding him/her in office supplemented by the
# documents referred to in point 274 above.
# 276. The role of the person responsible for internal audit cannot be ensured by a member of the
# management body/governing body of the IFM unless s/he is member of the IFM’s senior
# management.

#     """
    
#     chunker = RegulatoryChunkingSystem()
#     # chunks = chunker._hierarchical_chunking(unstructured_text)
#     chunks = chunker.process_document(unstructured_text)
#     print("Structured document chunks:", len(chunks))

#     for i, chunk in enumerate(chunks):
#         print("chunk", i, ": ", chunk["content"], "\n---\n")
 