#!/usr/bin/env python3
"""
scripts/query_regulatory_rag.py
Interactive local CLI tool to query your pre-built regulatory indexes (FAISS + SQLite BM25 + OpenAI).
"""

import sys
import os
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.rag.embedding_service import EmbeddingService
from core.rag.search_service import SearchService
from core.rag.rag_generator import RAGGenerator
from app.config import INDEX_DIR, EMBEDDING_MODEL, FAISS_FILENAME, DB_FILENAME


def main():
    print("=================================================================")
    print("  EFIRAS Interactive Regulatory RAG Query Tool (OpenAI + FAISS)")
    print("=================================================================")

    index_dir = os.getenv("INDEX_DIR", INDEX_DIR)
    faiss_path = os.path.join(index_dir, FAISS_FILENAME)
    db_path = os.path.join(index_dir, DB_FILENAME)

    print(f"Loading indexes from: {index_dir}")
    print(f"  -> FAISS: {faiss_path}")
    print(f"  -> DB:    {db_path}")

    # Initialize services
    print("\nInitializing EmbeddingService (OpenAI text-embedding-3-small)...")
    emb_service = EmbeddingService(use_local=False, online_model=EMBEDDING_MODEL)

    print("Initializing SearchService...")
    search_service = SearchService(index_dir=index_dir)
    success = search_service.load_indexes(faiss_path=faiss_path, db_path=db_path)

    if not success:
        print("❌ Error loading indexes. Please run: python -m regulatory_chunker.run_pipeline --step index")
        sys.exit(1)

    print("Initializing RAG Generator...")
    rag_generator = RAGGenerator()

    print("\n✅ System Ready! Type your question below (or 'q' to quit):\n")

    while True:
        try:
            query = input("\n🔍 Ask a regulatory question > ").strip()
            if not query:
                continue
            if query.lower() in ["q", "quit", "exit"]:
                print("Exiting RAG Query Tool. Goodbye!")
                break

            print(f"\nProcessing query: '{query}'...")
            
            # Step 1: Generate query embedding via OpenAI
            query_embedding = emb_service.embed_text(query)

            # Step 2: Search hybrid FAISS + SQLite BM25 indexes
            results = search_service.search_documents(query, query_embedding, top_k=5)
            print(f"\nFound {len(results)} relevant regulatory chunks:")
            for idx, res in enumerate(results, 1):
                citation = res.get("header_identifier") or res.get("citation") or res.get("chunk_id", "N/A")
                doc_id = res.get("filename") or res.get("doc_id", "N/A")
                preview = res.get("content", "")[:120].replace("\n", " ")
                print(f"  [{idx}] {doc_id} - {citation}")
                print(f"      \"{preview}...\"")

            # Step 3: Generate RAG answer
            print("\nGenerating RAG Answer...")
            print("=" * 70)
            rag_generator.answer_query(query, results)
            print("=" * 70)

        except KeyboardInterrupt:
            print("\nExiting RAG Query Tool. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
