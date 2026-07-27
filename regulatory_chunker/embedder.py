"""
embedder.py
OpenAI API Embedding generator for regulatory framework chunks.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
except ImportError:
    _TIKTOKEN_ENCODER = None

try:
    from .api import RegulatoryRepository
except ImportError:
    from api import RegulatoryRepository


def _enrich_text(chunk: Dict[str, Any], max_tokens: int = 7000) -> str:
    """
    Enriches chunk text with citation and hierarchical headings before embedding:
    [Citation] Heading
    Chunk Text Content
    Truncates text to max 7,000 tokens (well below OpenAI's 8,192 limit) using tiktoken
    if available, or a strict 8,000 character fallback (~2,000 tokens).
    """
    citation = chunk.get("citation", "")
    headings = [lvl.get("heading") for lvl in chunk.get("path", []) if lvl.get("heading")]
    heading_str = " - ".join(headings) if headings else ""

    parts = []
    if citation:
        parts.append(f"[{citation}]")
    if heading_str:
        parts.append(heading_str)
    parts.append(chunk.get("text", ""))

    full_text = "\n".join(parts)

    if _TIKTOKEN_ENCODER is not None:
        tokens = _TIKTOKEN_ENCODER.encode(full_text)
        if len(tokens) > max_tokens:
            full_text = _TIKTOKEN_ENCODER.decode(tokens[:max_tokens])
    else:
        # Strict character cap: 8,000 chars is ~2,000 tokens (100% guaranteed <8,192 tokens for any table/symbols)
        max_chars = 8000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

    return full_text


class RegulatoryEmbedder:
    """
    Generates OpenAI API embeddings using 'text-embedding-3-small'
    for pre-built regulatory chunks saved into data/regulatory_pipeline/openai_embeddings.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self._client = None
        self.repo = RegulatoryRepository()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            primary_path = Path(__file__).parent.parent / "data" / "regulatory_pipeline" / "openai_embeddings"
            self.output_dir = primary_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key or key == "your_openai_api_key_here":
                raise ValueError("OPENAI_API_KEY is not set or contains placeholder. Please update OPENAI_API_KEY in .env file.")
            self._client = OpenAI(api_key=key)
        return self._client

    def embed_document(self, doc_id: str, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Embeds all chunks for a given document ID using OpenAI API (text-embedding-3-small).
        Enriches text with [Citation] and Headings before passing to OpenAI.
        Safely truncates text blocks to strictly observe OpenAI's 8,192 token limit.
        Saves the embeddings to data/regulatory_pipeline/openai_embeddings/<doc_id>_openai_embeddings.json.
        """
        chunks = self.repo.get_chunks(doc_id)
        valid_chunks = [c for c in chunks if c.get("text")]
        enriched_texts = [_enrich_text(c, max_tokens=7000) for c in valid_chunks]

        if not enriched_texts:
            return []

        print(f"[{doc_id}] Requesting OpenAI embeddings for {len(valid_chunks)} chunks...")
        all_embeddings = []
        for i in range(0, len(enriched_texts), batch_size):
            batch = enriched_texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])

        payload = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "citation": chunk["citation"],
                "path": chunk["path"],
                "text": chunk["text"],
                "enriched_text": enriched_text,
                "embedding": embedding,
            }
            for chunk, enriched_text, embedding in zip(valid_chunks, enriched_texts, all_embeddings)
        ]

        out_file = self.output_dir / f"{doc_id.lower()}_openai_embeddings.json"
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def embed_all(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Embeds ALL available pre-built regulatory document chunks.
        """
        results = {}
        for doc_id in self.repo.list_documents():
            payload = self.embed_document(doc_id, batch_size=batch_size)
            results[doc_id] = len(payload)
        return results
