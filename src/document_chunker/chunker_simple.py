import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RegulationChunker:
    def __init__(self, raw_text: str, max_chunk_size: int = 2000):
        self.raw_text = raw_text
        self.chunks = []
        self.max_chunk_size = max_chunk_size
        
        # Add LangChain splitter for oversized chunks only
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            length_function=len
        )

    def chunk(self) -> List[Dict]:
        # Your original logic
        heading_pattern = re.compile(
            r"^(Part\s+\w+.*?)$|^(Chapter\s+\d+.*?)$|^(Sub-chapter\s+\d+\.\d+.*?)$|^(Section\s+\d+\.\d+\.\d+.*?)$",
            re.MULTILINE
        )
        matches = list(heading_pattern.finditer(self.raw_text))
        
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
        
        # NEW: Handle oversized chunks
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
        """Your original hierarchy parsing"""
        levels = []
        if heading.startswith("Part"):
            levels.append(heading)
        elif heading.startswith("Chapter"):
            levels.append(heading)
        elif heading.startswith("Sub-chapter"):
            levels.append(heading)
        elif heading.startswith("Section"):
            levels.append(heading)
        return levels