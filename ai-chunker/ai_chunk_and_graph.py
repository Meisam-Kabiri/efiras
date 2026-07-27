
from os import Path 
from typing import List, Optional
import fitz # PyMuPDF

graph_structure = """ {
  "chunks": [
    {
        "document": "<doc slug, copy from the user message>",
        "title":     "<number only, e.g. 'III', or null>",
        "chapter":   "<number only, or null>",
        "section":   "<number only, or null>",
        "article":   "<number only, e.g. '12', or null>",
        "paragraph": "<number only, e.g. '1', or null>",
        "labels": {
          "title":   "<display name without the number, or null>",
          "chapter": "<display name, or null>",
          "section": "<display name, or null>",
          "article": "<display name, or null>"
      },
      "content_type": "text | definition | table | recital | annex | list_item
      "raw_text": "<the exact verbatim text of this chunk, copied character-for-character>",
      "source_pages": [<page numbers this chunk's text appears on, read from th
      "defines": [
        {"term": "<defined term>", "definition": "<its definition as stated>"}
      ],
      "references": [
        "<every cross-reference exactly as written, e.g. 'Article 6(2)', 'Regulation (EU) 2019/1020'>"
      ]
    }
  ],
  "updated_state": {
    "title":   "<number or null>",
    "chapter": "<number or null>",
    "section": "<number or null>",
    "article": "<number or null>",
    "last_page_processed": <highest page number you fully processed>,
    "open_paragraph": "<null, OR the partial text of a paragraph that runs past the end of this slice and must be continued in the next slice>"
  }
} """


SYSTEM_PROMPT = f"""You are a regulatory document chunker. You receive a slice of a legal/regulatory document with page markers, plus the structural position where the previous slice ended. You split the text into chunks at the smallest meaningful unit (usually a paragraph or a defined point), attach structural metadata to each, and report where you ended.

Return ONE JSON object, nothing else. No markdown, no commentary. Exact shape:

{graph_structure}

RULES:
- raw_text MUST be the exact text from the document. Do not paraphrase, summarize, fix typos, or drop words.
- Use ONLY the numbers for title/chapter/section/article/paragraph. Put display with no display name still gets the number.
- If a structural level does not apply to a chunk, set it to null. Never invent a parent level that is not present in the text.
- Carry forward the structural position from the provided state. If this slice ng that article's numbers until a new heading appears.
- source_pages comes ONLY from the --- PAGE n --- markers in the text. Never guess a page number.
- references: capture every citation verbatim, as written. Do NOT resolve them,nt any not present in the text. Empty list if none.
- If a paragraph is cut off at the end of the slice, do NOT emit it as a chunk. Instead put its partial text in updated_state.open_paragraph so the next slice continues it.
- If the slice begins with text continuing a previous open_paragraph, prepend tete chunk.
- Output valid JSON only."""

USER_PROMPT = """Document: {document_slug}

Structural position where the previous slice ended (continue from here):
{current_state_json}

Document text for this slice (page markers included):
{page_text_with_markers}"""



class ai_chunker:
    def __init__(self, model, doc:Path):
        self.model = model
        self.doc = doc
        self.graph_structure = graph_structure

    @staticmethod
    def chunk_given_pages(model, doc:Path, pages:List):
        """
        Chunk a document given a list of pages to process.
        """
        # Open the document
        pdf_document = fitz.open(doc)
        
        # Extract text from the specified pages
        page_texts = []
        for page_num in pages:
            if page_num < 1 or page_num > pdf_document.page_count:
                raise ValueError(f"Page number {page_num} is out of range for document with {pdf_document.page_count} pages.")
            page = pdf_document[page_num - 1]  # fitz uses 0-based indexing
            page_texts.append(f"--- PAGE {page_num} ---\n{page.get_text()}")
        
        # Combine the extracted texts into a single string
        combined_text = "\n\n".join(page_texts)
        
        # Prepare the user prompt with the combined text
        user_prompt = USER_PROMPT.format(
            document_slug=doc.stem,
            current_state_json="{}",  # Assuming no previous state for this example
            page_text_with_markers=combined_text
        )
        
        # Call the model to chunk the text
        response = model.generate(user_prompt)
        
        return response
      
      
      
      def build_chunk_id(doc, chunk, seq):
    """Build a stable id from the address fields the LLM returned.
    Skips null levels; falls back to seq when there's no address at all."""
    parts = []
    for tok, key in [("t", "title"), ("c", "chapter"), ("s", "section"),
                     ("art", "article"), ("para", "paragraph")]:
        if chunk.get(key):                 # skip null / missing
            parts.append(f"{tok}{chunk[key]}")

    if not parts:                          # no structural address → positional id
        return f"{doc}::seq{seq}"
    return f"{doc}::" + ".".join(parts)


# in the loop, after parsing one window:
result = json.loads(response_text)
for chunk in result["chunks"]:
    chunk["chunk_id"] = build_chunk_id(doc_slug, chunk, global_seq)
    all_chunks.append(chunk)
    global_seq += 1
      
   
      
      
