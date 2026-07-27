#!/usr/bin/env python3
"""
run_pipeline.py
Master offline pipeline runner for regulatory framework documents.
Orchestrates: Step 1 (Chunking) -> Step 2 (OpenAI Embedding) -> Step 3 (FAISS + SQLite BM25 Indexing).
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path for importing core module
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .api import RegulatoryChunker, RegulatoryRepository
    from .embedder import RegulatoryEmbedder
    from .fetch_chunk import EurLexChunker
    from .buckets_config import BUCKET_A, BUCKET_B
except ImportError:
    from api import RegulatoryChunker, RegulatoryRepository
    from embedder import RegulatoryEmbedder
    from fetch_chunk import EurLexChunker
    from buckets_config import BUCKET_A, BUCKET_B


def run_chunking():
    print("\n=======================================================")
    print("  STEP 1: CHUNKING REGULATORY FRAMEWORK DOCUMENTS")
    print("=======================================================")
    chunker = RegulatoryChunker()
    repo = RegulatoryRepository()
    out_dir = repo.chunks_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, name, url, fmt in BUCKET_A:
        out_file = out_dir / f"{doc_id.lower()}_chunks.json"
        print(f"[{doc_id}] Chunking HTML: {name}...")
        html_cache_file = Path(__file__).parent / "html_cache" / f"{doc_id.lower()}.html"
        try:
            chunker_inst = EurLexChunker(
                doc_id=doc_id,
                source=str(html_cache_file) if html_cache_file.exists() else url,
                cache_file=html_cache_file,
                verbose=False,
            )
            chunks = chunker_inst.run(out_file=str(out_file))
            print(f"  -> Saved {len(chunks)} chunks to {out_file}")
        except Exception as e:
            print(f"  -> Error: {e}")

    for doc_id, name, url, fmt in BUCKET_B:
        if doc_id == "DODD_FRANK_P2":
            continue
        out_file = out_dir / f"{doc_id.lower()}_chunks.json"
        
        # Smart candidate matching for Bucket B HTML cache names (e.g. fatf_2012.html, basel_iii.html)
        html_candidates = [
            f"{doc_id.lower()}.html",
            f"{doc_id.lower().replace('fatf2012', 'fatf_2012')}.html",
            f"{doc_id.lower().replace('basel3', 'basel_iii')}.html",
        ]
        html_cache_file = None
        for cand in html_candidates:
            p = Path(__file__).parent / "html_cache" / cand
            if p.exists():
                html_cache_file = p
                break

        pdf_cache = Path(__file__).parent / "flat_cache" / f"{doc_id.lower()}.pdf"

        print(f"[{doc_id}] Chunking PDF/HTML: {name}...")
        try:
            if html_cache_file and html_cache_file.exists():
                from regulatory_chunker.tree_chunker import TreeChunker
                parser = TreeChunker(doc_id=doc_id)
                chunks = parser.chunk(html_cache_file.read_text(encoding="utf-8"))
            elif pdf_cache.exists():
                chunks = chunker.chunk(doc_id=doc_id, source=str(pdf_cache), fmt="pdf")
            else:
                print(f"[{doc_id}] Neither HTML nor PDF cache found, skipping.")
                continue
            out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  -> Saved {len(chunks)} chunks to {out_file}")
        except Exception as e:
            print(f"  -> Error: {e}")


def run_embedding():
    print("\n=======================================================")
    print("  STEP 2: GENERATING OPENAI API EMBEDDINGS")
    print("=======================================================")
    embedder = RegulatoryEmbedder(model="text-embedding-3-small")
    summary = embedder.embed_all()
    for doc_id, count in summary.items():
        print(f"[{doc_id}] Embedded {count} chunks")
    print(f"  -> Saved embeddings to {embedder.output_dir}")


def run_indexing():
    print("\n=======================================================")
    print("  STEP 3: BUILDING FAISS VECTOR & SQLITE BM25 INDEXES")
    print("=======================================================")
    try:
        from core.rag.search_service import SearchService
    except ImportError as e:
        print(f"  -> Error importing SearchService: {e}")
        return

    embeddings_dir = Path(__file__).parent.parent / "data" / "regulatory_pipeline" / "openai_embeddings"
    if not embeddings_dir.exists():
        embeddings_dir = Path(__file__).parent / "regulatory_embeddings"

    index_output_dir = Path(__file__).parent.parent / "data" / "regulatory_indexes"
    index_output_dir.mkdir(parents=True, exist_ok=True)

    documents_list = []
    for filepath in sorted(embeddings_dir.glob("*_openai_embeddings.json")):
        doc_id = filepath.stem.replace("_openai_embeddings", "").upper()
        payloads = json.loads(filepath.read_text(encoding="utf-8"))

        chunk_embeddings = []
        for item in payloads:
            heading_list = [lvl.get("heading") for lvl in item.get("path", []) if lvl.get("heading")]
            headers_str = " - ".join(heading_list) if heading_list else ""

            chunk_embeddings.append({
                "id": item["chunk_id"],
                "content": item.get("enriched_text", item.get("text", "")),
                "embedding": item["embedding"],
                "doc_id": item["doc_id"],
                "citation": item["citation"],
                "headers": headers_str,  # Must be a string for SQLite TEXT column
                "header_identifier": item["citation"],
            })

        documents_list.append({
            "metadata": {"filename": doc_id, "doc_id": doc_id},
            "embeddings": chunk_embeddings,
        })

    print(f"Loaded {len(documents_list)} document payloads for indexing.")
    if not documents_list:
        print("  -> No embedding files found in directory! Run Step 2 (--step embed) first.")
        return

    search_service = SearchService(index_dir=str(index_output_dir))
    search_service.build_indexes(documents_list)

    faiss_path = index_output_dir / "regulatory_faiss.bin"
    db_path = index_output_dir / "regulatory_chunks.db"
    search_service.save_indexes(faiss_path=str(faiss_path), db_path=str(db_path))

    print(f"  -> FAISS index saved: {faiss_path}")
    print(f"  -> SQLite BM25 index saved: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Master Regulatory Offline Pipeline")
    parser.add_argument(
        "--step",
        choices=["chunk", "embed", "index", "all"],
        default="all",
        help="Pipeline step to run: 'chunk', 'embed', 'index', or 'all' (default)",
    )
    args = parser.parse_args()

    if args.step in ["chunk", "all"]:
        run_chunking()
    if args.step in ["embed", "all"]:
        run_embedding()
    if args.step in ["index", "all"]:
        run_indexing()

    print("\n✅ Master Regulatory Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
