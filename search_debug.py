"""
RAG Search Debug Example

This script helps debug how the RAGSystem's search method works:
- Load preprocessed and chunked blocks
- Build the knowledge base
- Run a query
- Step into search() using ipdb
"""

import os, sys
sys.path.append(os.path.abspath("src"))

import json
from pathlib import Path
from src.rag.rag_simple import RAGSystem

from dotenv import load_dotenv
load_dotenv()

def main():
    print("=== Debugging RAG Search Function ===\n")

    # === Step 1: Load preprocessed chunked blocks ===
    input_path = "data_processed/Lux_cssf18_698eng_processed_blocks.json"
    
    if not Path(input_path).exists():
        print("❌ Processed chunk file not found. Please run the main pipeline first.")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    print(f"✔ Loaded {len(blocks)} blocks")

    # === Step 2: Initialize RAG System ===
    rag = RAGSystem(use_local_embeddings=True)

    # === Step 3: Embed blocks and build DB ===
    print("🔧 Embedding and building knowledge base...")
    rag.add_documents(
        blocks,
        cache_path="data_processed",
        cache_file_name="Lux_cssf18_698eng_debug"
    )

    # === Step 4: Run a query with ipdb trace ===
    query = "517. In addition to the elements referred to in Sub-sectionc"

    print(f"\n🔍 Debugging search for query:\n'{query}'\n")

    # import ipdb; ipdb.set_trace()  # 🔍 Enter debugging before search

    results = rag.search(query, top_k=5)

    print("\n=== Top Search Results ===")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Result {i}:")
        print(f"Headers: {doc['block'].get('enriched_headers', 'N/A')}")
        print(f"Content Preview: {doc['content']}...")

if __name__ == "__main__":
    main()
