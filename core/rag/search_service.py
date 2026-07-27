import json
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np


def better_tokenize(text):
    import re

    # Keep important terms together
    text = re.sub(r"\n+", " ", text)  # Remove newlines
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single
    # Don't split "non-compliance" - keep as one term
    tokens = text.lower().split()
    return [token.strip(".,") for token in tokens if len(token) > 2]


# Columns stored on the "chunks" table, in addition to the FTS5-indexed "content".
CHUNK_COLUMNS = [
    "content",
    "page",
    "bbox",
    "is_diff_format",
    "headers",
    "header_identifier",
    "is_hierarchical",
    "is_title",
    "merged_from",
    "enriched_headers",
    "chunk_id",
    "doc_id",
    "filename",
    "doc_metadata",
]

# Fields that are stored as JSON text and need decoding back into Python objects.
JSON_COLUMNS = {"bbox", "doc_metadata"}
BOOL_COLUMNS = {"is_diff_format", "is_hierarchical", "is_title"}


class SearchService:
    def __init__(
        self,
        index_dir="data/indexes",
        documents_list: List[Dict] = None,
    ):
        """Initialize HybridSearch with optional index directory"""
        self.index_dir = index_dir
        self.faiss_index = None
        self.conn: Optional[sqlite3.Connection] = None
        self.num_chunks = 0
        # Only populated transiently while building indexes (see build_indexes());
        # query-time chunk lookups go through self.conn instead.
        self.chunks = []
        if documents_list:
            self.chunks = self.set_chunks(documents_list)

    def set_chunks(self, documents_list):
        """Set chunks data without building indexes"""
        all_chunks = []
        documents_list = (
            documents_list if isinstance(documents_list, list) else [documents_list]
        )

        for doc_idx, doc in enumerate(documents_list):
            filename = doc["metadata"].get("filename", f"document_{doc_idx}")

            for chunk_idx, embedded_chunk in enumerate(doc["embeddings"]):
                # Create new chunk without embedding (for memory efficiency)
                chunk_metadata = {
                    "content": embedded_chunk["content"],  # Text content
                    "id": embedded_chunk["id"],
                    "doc_id": doc_idx,
                    "filename": filename,
                    "chunk_id": chunk_idx,
                    "doc_metadata": doc.get("metadata", {}),
                    # Add all other fields from embedded_chunk except 'embedding'
                    **{
                        k: v
                        for k, v in embedded_chunk.items()
                        if k not in ["embedding", "content", "id"]
                    },
                }
                all_chunks.append(chunk_metadata)

        self.chunks = all_chunks
        print(f"✅ Set {len(self.chunks)} chunks")

    def build_indexes(self, documents_list):
        """Build the FAISS vector index. Chunk content/metadata is written to
        SQLite (with an FTS5 BM25 index) by save_indexes()."""
        """
        Each document in documents_list is a dictionary with:
          - "metadata": contains fields like "filename", "number of pages"
          - "embeddings": a list of chunks, each with:
              - "embedding": the vector representation
              - "content": the text content of the chunk
              - "headers"
              - "page_number"
              - "header_identifier" (the header text)
        """

        self.set_chunks(documents_list)
        all_embedded_chunks = []
        documents_list = (
            documents_list if isinstance(documents_list, list) else [documents_list]
        )
        for doc_idx, doc in enumerate(documents_list):
            # Get filename from metadata
            filename = doc["metadata"].get("filename", f"document_{doc_idx}")

            for chunk_idx, chunk in enumerate(doc["embeddings"]):
                chunk["doc_id"] = doc_idx
                chunk["filename"] = filename  # Use filename from metadata
                chunk["chunk_id"] = chunk_idx
                chunk["doc_metadata"] = doc.get("metadata", {})
                all_embedded_chunks.append(chunk)

        # Note: self.chunks is already set (embedding-stripped) by set_chunks() above.
        # all_embedded_chunks (with "embedding") is only used below to build FAISS.

        # Build FAISS
        embeddings = np.array([c["embedding"] for c in all_embedded_chunks]).astype(
            "float32"
        )
        faiss.normalize_L2(embeddings)

        if len(all_embedded_chunks) > 10000:
            # Large dataset - use IVF
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)
            self.faiss_index.train(embeddings)
            self.faiss_index.nprobe = 20
        elif len(all_embedded_chunks) > 1000:
            # Medium dataset - use HNSW
            self.faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 64)
            # Step 2: Set high-quality construction before adding vectors
            self.faiss_index.hnsw.efConstruction = 200

            # Step 4: Set high efSearch before querying
            self.faiss_index.hnsw.efSearch = 256
        else:
            # Small dataset - use flat
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])

        self.faiss_index.add(embeddings)

    def vector_search(self, query_embedding, top_k=100):
        """Fast vector search using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_indexes() first.")

        query_emb = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return scores[0], indices[0]

    def bm25_search(self, query, top_k=100):
        """BM25 search via SQLite FTS5 (ranking computed in SQL, nothing held in RAM)"""
        if self.conn is None:
            raise ValueError("Chunk database not loaded. Call load_indexes() first.")

        query_tokens = better_tokenize(query)
        if not query_tokens:
            return []

        # Quote each token so FTS5 query syntax characters in content don't break parsing
        match_query = " OR ".join(f'"{t}"' for t in query_tokens)

        rows = self.conn.execute(
            """
            SELECT rowid, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()

        # FTS5's bm25() is lower-is-better; negate so higher score = better match
        return [(row["rowid"], -row["score"]) for row in rows]

    def get_chunks_by_ids(self, ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        """Batch-fetch chunk content/metadata from SQLite for a set of global ids."""
        ids = list({int(i) for i in ids})
        if not ids or self.conn is None:
            return {}

        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {row["id"]: self._row_to_chunk(row) for row in rows}

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Dict[str, Any]:
        chunk = {"id": row["id"]}
        for col in CHUNK_COLUMNS:
            value = row[col]
            if col in JSON_COLUMNS:
                value = json.loads(value) if value else ({} if col == "doc_metadata" else None)
            elif col in BOOL_COLUMNS:
                value = bool(value)
            chunk[col] = value
        return chunk

    def rrf_combine(self, vector_results, bm25_results, query, chunks_lookup, k=0):
        """Reciprocal Rank Fusion combination"""
        """Enhanced RRF with document name boosting"""
        scores, indices = vector_results
        combined = {}

        # Extract document names/abbreviations from query
        doc_boost_keywords = self.extract_doc_keywords(query)

        # Vector rankings with document boost
        for rank, idx in enumerate(indices):
            idx = int(idx)
            if idx != -1 and idx in chunks_lookup:
                base_score = 1 / (k + rank + 1)

                # Apply document name boost
                doc_boost = self.calculate_doc_boost(
                    chunks_lookup[idx], doc_boost_keywords
                )
                combined[idx] = combined.get(idx, 0) + (base_score * doc_boost)

        # BM25 rankings with document boost
        for rank, (idx, bm25_score) in enumerate(bm25_results):
            if idx in chunks_lookup:
                base_score = 1 / (k + rank + 1)
                doc_boost = self.calculate_doc_boost(
                    chunks_lookup[idx], doc_boost_keywords
                )
                combined[idx] = combined.get(idx, 0) + (base_score * doc_boost)

        return combined

    def extract_doc_keywords(self, query):
        """Extract document name keywords from query"""
        doc_patterns = {
            "aifmd": [
                "aifmd",
                "alternative investment fund managers directive",
                "alternative_investment_fund_managers_directive",
            ],
            "aifmd_level_2": [
                "aifmd level 2",
                "aifmd_level_2",
                "aifmd level ii",
                "aifmd_level_ii",
            ],
            "basel_ii": ["basel ii", "basel_ii", "basel 2", "basel_2"],
            "basel_iii": ["basel iii", "basel_iii", "basel 3", "basel_3"],
            "crd_v": ["crd v", "crd_v", "crd 5", "crd_5"],
            "crr": [
                "crr",
                "capital requirements regulation",
                "capital_requirements_regulation",
            ],
            "dodd_frank": ["dodd frank", "dodd_frank", "dodd-frank"],
            "emir": [
                "emir",
                "european market infrastructure regulation",
                "european_market_infrastructure_regulation",
            ],
            "eu_taxonomy": [
                "eu taxonomy",
                "eu_taxonomy",
                "european union taxonomy regulation",
                "european_union_taxonomy_regulation",
            ],
            "5amld": [
                "5amld",
                "5_amld",
                "fifth anti money laundering directive",
                "fifth_anti_money_laundering_directive",
            ],
            "4amld": [
                "4amld",
                "4_amld",
                "fourth anti money laundering directive",
                "fourth_anti_money_laundering_directive",
            ],
            "fatf": [
                "fatf",
                "financial action task force",
                "financial_action_task_force",
            ],
            "gdpr": [
                "gdpr",
                "general data protection regulation",
                "general_data_protection_regulation",
            ],
            "cssf_18_698": ["cssf 18/698", "cssf_18_698", "cssf 18 698"],
            "mifid_ii": ["mifid ii", "mifid_ii", "mifid 2", "mifid_2"],
            "mifir": [
                "mifir",
                "markets in financial instruments regulation",
                "markets_in_financial_instruments_regulation",
            ],
            "psd2": [
                "psd2",
                "psd_2",
                "psd 2",
                "payment services directive 2",
                "payment_services_directive_2",
            ],
            "sftr": [
                "sftr",
                "securities financing transactions regulation",
                "securities_financing_transactions_regulation",
            ],
            "solvency_ii": ["solvency ii", "solvency_ii", "solvency 2", "solvency_2"],
            "solvency_ii_level_2": [
                "solvency ii level 2",
                "solvency_ii_level_2",
                "solvency 2 level 2",
                "solvency_2_level_2",
            ],
            "sfdr": [
                "sfdr",
                "sustainable finance disclosure regulation",
                "sustainable_finance_disclosure_regulation",
            ],
            "ucits": [
                "ucits",
                "undertakings for collective investment in transferable securities",
                "undertakings_for_collective_investment_in_transferable_securities",
            ],
        }

        query_lower = query.lower()
        found_keywords = []

        for doc_key, keywords in doc_patterns.items():
            if any(kw in query_lower for kw in keywords):
                found_keywords.append(doc_key)

        return found_keywords

    def calculate_doc_boost(self, chunk, boost_keywords):
        """Calculate boost multiplier based on document match"""
        if not boost_keywords:
            return 1.0  # No boost

        chunk_filename = chunk.get("filename", "").lower()

        for keyword in boost_keywords:
            if keyword in chunk_filename:
                return 1.5  # 50% boost for document name match

        return 1.0  # No boost

    def hybrid_search(self, query, query_embedding, top_k=12, expand_articles=True, doc_filter=None):
        """Hybrid search combining vector and BM25 results with optional document filtering"""
        # Get results from both search methods
        vector_results = self.vector_search(query_embedding, 100)
        bm25_results = self.bm25_search(query, 100)

        # Fetch candidate chunk data once, in a single batched SQLite query
        _, vector_indices = vector_results
        candidate_ids = {int(idx) for idx in vector_indices if int(idx) != -1}
        candidate_ids.update(idx for idx, _ in bm25_results)
        chunks_lookup = self.get_chunks_by_ids(candidate_ids)

        # Apply hard document filtering if doc_filter is provided
        if doc_filter:
            doc_filter_norms = [
                re.sub(r"[^a-z0-9]", "", str(d).lower()) for d in doc_filter if str(d).strip()
            ]
            filtered_lookup = {}
            for cid, chunk in chunks_lookup.items():
                chunk_doc_id = re.sub(r"[^a-z0-9]", "", str(chunk.get("doc_id", "")).lower())
                chunk_filename = re.sub(r"[^a-z0-9]", "", str(chunk.get("filename", "")).lower())

                if any(
                    df in chunk_doc_id or df in chunk_filename or chunk_doc_id in df
                    for df in doc_filter_norms
                ):
                    filtered_lookup[cid] = chunk

            # Only restrict if matches found; fallback to candidate pool if no match
            if filtered_lookup:
                chunks_lookup = filtered_lookup

        # Combine results using RRF
        combined = self.rrf_combine(vector_results, bm25_results, query, chunks_lookup)
        top_indices = sorted(combined.keys(), key=combined.get, reverse=True)[:top_k]

        top_chunks = [chunks_lookup[i] for i in top_indices if i in chunks_lookup]

        if expand_articles:
            top_chunks = self.expand_chunks_to_articles(top_chunks)

        return top_chunks

    def expand_chunks_to_articles(
        self, chunks: List[Dict[str, Any]], max_article_tokens: int = 3000
    ) -> List[Dict[str, Any]]:
        """
        Expands retrieved chunks to their full parent Article / Provision level
        using SQLite queries to fetch all sibling paragraphs in order.
        """
        if not chunks or self.conn is None:
            return chunks

        expanded_results = []
        seen_keys = set()

        for chunk in chunks:
            doc_id = chunk.get("doc_id") or chunk.get("filename")
            headers = chunk.get("headers") or ""
            header_identifier = chunk.get("header_identifier") or chunk.get("citation") or ""

            target_header = headers or header_identifier
            if not doc_id or not target_header:
                expanded_results.append(chunk)
                continue

            article_key = (doc_id, target_header)
            if article_key in seen_keys:
                continue
            seen_keys.add(article_key)

            # Query all sibling chunks belonging to this document and header in SQLite
            try:
                rows = self.conn.execute(
                    "SELECT * FROM chunks WHERE doc_id = ? AND (headers = ? OR header_identifier = ?) ORDER BY id",
                    (doc_id, target_header, target_header),
                ).fetchall()
            except Exception:
                rows = []

            if not rows or len(rows) <= 1:
                expanded_results.append(chunk)
                continue

            sibling_chunks = [self._row_to_chunk(row) for row in rows]
            merged_text = "\n\n".join(
                c.get("content", "") for c in sibling_chunks if c.get("content")
            )
            approx_tokens = len(merged_text.split()) * 1.3

            if approx_tokens <= max_article_tokens:
                expanded_chunk = dict(chunk)
                expanded_chunk["content"] = merged_text
                expanded_chunk["is_expanded"] = True
                expanded_chunk["sibling_count"] = len(sibling_chunks)
                expanded_results.append(expanded_chunk)
            else:
                expanded_results.append(chunk)

        return expanded_results

    def hybrid_search_with_cross_encoder(self, query, query_embedding, top_k=8):
        """Hybrid search with cross-encoder re-ranking"""
        # Initialize cross-encoder (you might want to do this in __init__)
        from sentence_transformers import CrossEncoder

        try:
            self.cross_encoder = CrossEncoder(
                "BAAI/bge-reranker-large", local_files_only=True
            )
        except:
            self.cross_encoder = CrossEncoder(
                "BAAI/bge-reranker-large", local_files_only=False
            )

        # Step 1: Get candidates from both search methods
        vector_results = self.vector_search(query_embedding, 15)  # Get top 15 from each
        bm25_results = self.bm25_search(query, 15)

        # Step 2: Combine candidates and remove duplicates
        _, vector_indices = vector_results
        candidate_ids = {int(idx) for idx in vector_indices if int(idx) != -1}
        candidate_ids.update(idx for idx, _ in bm25_results)

        chunks_lookup = self.get_chunks_by_ids(candidate_ids)
        if not chunks_lookup:
            return []

        candidate_chunks = list(chunks_lookup.items())  # [(id, chunk), ...]

        # Step 3: Re-rank using cross-encoder
        pairs = [(query, chunk["content"]) for _, chunk in candidate_chunks]
        scores = self.cross_encoder.predict(pairs)

        # Step 4: Sort by cross-encoder scores and return top_k
        scored_chunks = list(zip(candidate_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending

        # Return top_k chunks
        top_chunks = [chunk for (idx, chunk), score in scored_chunks[:top_k]]

        return top_chunks

    def search_documents(
        self, query, query_embedding, top_k=5, expand_articles=True, doc_filter=None
    ):
        return self.hybrid_search(
            query,
            query_embedding,
            top_k=top_k,
            expand_articles=expand_articles,
            doc_filter=doc_filter,
        )

    def save_indexes(self, faiss_path=None, db_path=None):
        """Save indexes to disk for persistence: FAISS for vectors, SQLite
        (with an FTS5 BM25 index) for chunk content/metadata."""
        faiss_path = faiss_path or os.path.join(self.index_dir, "faiss.index")
        db_path = db_path or os.path.join(self.index_dir, "chunks.db")

        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"✅ FAISS index saved to {faiss_path}")

        if self.chunks:
            self._write_chunks_db(db_path)
            print(f"✅ Chunks database saved to {db_path}")

    def _write_chunks_db(self, db_path):
        """(Re)build the chunks.db SQLite file from self.chunks. The row id is
        the chunk's position in self.chunks, which matches the FAISS vector
        order and is the index used everywhere at query time."""
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        try:
            columns_sql = ", ".join(f"{col} TEXT" for col in CHUNK_COLUMNS)
            conn.execute(f"CREATE TABLE chunks (id INTEGER PRIMARY KEY, {columns_sql})")
            conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "content, content='chunks', content_rowid='id')"
            )

            def row_values(chunk_id, chunk):
                values = []
                for col in CHUNK_COLUMNS:
                    value = chunk.get(col)
                    if col in JSON_COLUMNS:
                        value = json.dumps(value) if value is not None else None
                    values.append(value)
                return (chunk_id, *values)

            conn.executemany(
                f"INSERT INTO chunks (id, {', '.join(CHUNK_COLUMNS)}) "
                f"VALUES (?, {', '.join('?' * len(CHUNK_COLUMNS))})",
                (row_values(i, chunk) for i, chunk in enumerate(self.chunks)),
            )
            conn.execute(
                "INSERT INTO chunks_fts (rowid, content) SELECT id, content FROM chunks"
            )
            conn.commit()
        finally:
            conn.close()

    def load_indexes(self, faiss_path=None, db_path=None):
        """Load indexes from disk"""

        if not faiss_path:
            env_faiss = os.getenv("FAISS_PATH") or os.getenv("FAISS_FILENAME")
            if env_faiss:
                faiss_path = env_faiss if os.path.isabs(env_faiss) else os.path.join(self.index_dir, env_faiss)
            else:
                reg_faiss = os.path.join(self.index_dir, "regulatory_faiss.bin")
                faiss_path = reg_faiss if os.path.exists(reg_faiss) else os.path.join(self.index_dir, "faiss.index")

        if not db_path:
            env_db = os.getenv("DB_PATH") or os.getenv("DB_FILENAME")
            if env_db:
                db_path = env_db if os.path.isabs(env_db) else os.path.join(self.index_dir, env_db)
            else:
                reg_db = os.path.join(self.index_dir, "regulatory_chunks.db")
                db_path = reg_db if os.path.exists(reg_db) else os.path.join(self.index_dir, "chunks.db")

        success = True

        try:
            # Load FAISS index
            if os.path.exists(faiss_path):
                # Load with memory mapping instead of loading entirely into RAM
                # Uses memory-mapped I/O, meaning:
                # FAISS doesn’t load the full file into RAM at once.
                # It maps the index file on disk into memory and loads only the needed parts when required.
                self.faiss_index = faiss.read_index(faiss_path, faiss.IO_FLAG_MMAP)
                print(f"✅ FAISS index loaded from {faiss_path}")
            else:
                print(f"⚠️ FAISS index not found at {faiss_path}")
                success = False

            # Open the chunks database (read-only; content stays on disk, nothing
            # is loaded into RAM up front)
            if os.path.exists(db_path):
                self.conn = sqlite3.connect(
                    f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
                )
                self.conn.row_factory = sqlite3.Row
                self.num_chunks = self.conn.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
                print(f"✅ Chunks database loaded from {db_path} ({self.num_chunks} chunks)")
            else:
                print(f"⚠️ Chunks database not found at {db_path}")
                success = False

            return success
        except Exception as e:
            print(f"❌ Error loading indexes: {e}")
            return False

    def get_stats(self):
        """Get index statistics"""
        stats = {
            "num_chunks": self.num_chunks,
            "faiss_index_type": (
                type(self.faiss_index).__name__ if self.faiss_index else None
            ),
            "index_directory": self.index_dir,
        }
        return stats


if __name__ == "__main__":
    # asyncio.run(main())
    search = SearchService(index_dir="data/indexes")

    # Load embeddings
    path = (
        "data/data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json"
    )
    with open(path, "r") as f:
        load_embeddings = json.load(f)

    # Build indexes

    if not search.load_indexes():
        search.build_indexes(load_embeddings)
        # Optional: Save indexes for persistence
        search.save_indexes()

    # Query
    query = "What monitoring elements must IFM implement for central administration delegation?"

    # Get query embedding
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    query_embed = model.encode(query)

    results = search.search_documents(query, query_embed, top_k=5)

    print("Search Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content'][:100]}...")

    # Print stats
    print("\nIndex Stats:", search.get_stats())
