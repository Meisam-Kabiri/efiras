
import asyncio
import numpy as np
import faiss
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
import os
import json
from typing import List, Dict, Any, Optional


class SearchService:
    def __init__(self, index_dir="indexes",
                  use_async:bool = True,
                  # Azure Search parameters
                  use_azure_search: bool = False,
                  azure_search_endpoint: Optional[str] = None,
                  azure_search_key: Optional[str] = None,
                  azure_search_index: str = "documents",
                  documents_list: List[Dict] = None):
        

        """Initialize HybridSearch with optional index directory"""
        self.index_dir = index_dir
        self.faiss_index = None
        self.whoosh_index = None
        self.chunks = []
        self.use_async = use_async
        if documents_list:
            self.chunks = self.set_chunks(documents_list)
            if self.use_azure_search:
                self.add_embedings_to_azure(documents_list)

        # Add Azure Search option
        self.use_azure_search = use_azure_search
        if use_azure_search:
            try:
                from rag.azure_search_backend import AzureSearchBackend
                AZURE_SEARCH_AVAILABLE = True
            except ImportError:
                AZURE_SEARCH_AVAILABLE = False
            self.azure_backend = AzureSearchBackend(
                endpoint=azure_search_endpoint,
                index_name=azure_search_index,
                api_key=azure_search_key
            )

    def add_embedings_to_azure(self, document_list):
            print(f"Uploading {len(document_list)} chunks to Azure Search...")
            self.azure_backend.add_documents(document_list)
            print("✅ Azure Search upload complete")
            return  # Don't build FAISS/Whoosh if using Azure
    
    def set_chunks(self, documents_list):
        """Set chunks data without building indexes"""
        all_chunks = []
        documents_list = documents_list if isinstance(documents_list, list) else [documents_list]
        
        for doc_idx, doc in enumerate(documents_list):
            filename = doc["metadata"].get("filename", f"document_{doc_idx}")
            
            for chunk_idx, embedded_chunk in enumerate(doc["embeddings"]):
                # Create new chunk without embedding (for memory efficiency)
                chunk_metadata = {
                    'content': embedded_chunk['content'],  # Text content
                    'id': embedded_chunk['id'],
                    'doc_id': doc_idx,
                    'filename': filename,
                    'chunk_id': chunk_idx,
                    'doc_metadata': doc.get("metadata", {}),
                    # Add all other fields from embedded_chunk except 'embedding'
                    **{k: v for k, v in embedded_chunk.items() 
                      if k not in ['embedding', 'content', 'id']}
                }
                all_chunks.append(chunk_metadata)
    
        self.chunks = all_chunks
        print(f"✅ Set {len(self.chunks)} chunks")


    def build_indexes(self, documents_list):
        """Build FAISS and Whoosh indexes"""
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
        documents_list = documents_list if isinstance(documents_list, list) else [documents_list]
        for doc_idx, doc in enumerate(documents_list):
          # Get filename from metadata
          filename = doc["metadata"].get("filename", f"document_{doc_idx}")
          
          for chunk_idx, chunk in enumerate(doc["embeddings"]):
              chunk["doc_id"] = doc_idx
              chunk["filename"] = filename  # Use filename from metadata
              chunk["chunk_id"] = chunk_idx
              chunk["doc_metadata"] = doc.get("metadata", {})
              all_embedded_chunks.append(chunk)



        
        os.makedirs(self.index_dir + "/whoosh", exist_ok=True)
        self.chunks = all_embedded_chunks
        
        # Build FAISS
        embeddings = np.array([c["embedding"] for c in all_embedded_chunks]).astype('float32')
        faiss.normalize_L2(embeddings)
        
        if len(all_embedded_chunks) > 10000:
            # Large dataset - use IVF
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 100)
            self.faiss_index.train(embeddings)
            self.faiss_index.nprobe = 10
        elif len(all_embedded_chunks) > 1000:
            # Medium dataset - use HNSW
            self.faiss_index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        else:
            # Small dataset - use flat
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        
        self.faiss_index.add(embeddings)
        
        # Build Whoosh
        schema = Schema(id=ID(stored=True), content=TEXT())
        self.whoosh_index = create_in(self.index_dir + "/whoosh", schema)
        
        writer = self.whoosh_index.writer()
        for i, chunk in enumerate(all_embedded_chunks):
            writer.add_document(id=str(i), content=chunk["content"])
        writer.commit()
        
        print(f"✅ Indexes built for {len(all_embedded_chunks)} chunks")

    async def vector_search(self, query_embedding, top_k=100):
        """Fast vector search using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_indexes() first.")
            
        query_emb = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return scores[0], indices[0]

    async def bm25_search(self, query, top_k=100):
        """Fast BM25 search using Whoosh"""
        if self.whoosh_index is None:
            raise ValueError("Whoosh index not built. Call build_indexes() first.")
            
        with self.whoosh_index.searcher(weighting=scoring.BM25F()) as searcher:
            parser = QueryParser("content", self.whoosh_index.schema)
            try:
                parsed_query = parser.parse(query)
            except:
                # Fallback for malformed queries
                from whoosh.query import Every
                parsed_query = Every()
                
            results = searcher.search(parsed_query, limit=top_k)
            return [(int(r['id']), r.score) for r in results]

    def rrf_combine(self, vector_results, bm25_results, k=60):
        """Reciprocal Rank Fusion combination"""
        scores, indices = vector_results
        combined = {}
        
        # Vector rankings
        for rank, idx in enumerate(indices):
            idx = int(idx)
            if idx != -1:
                combined[idx] = combined.get(idx, 0) + 1/(k + rank + 1)
        
        # BM25 rankings  
        for rank, (idx, score) in enumerate(bm25_results):
            combined[idx] = combined.get(idx, 0) + 1/(k + rank + 1)
        
        return combined
    

    async def hybrid_search(self, query, query_embedding, top_k=5):
        """Sync method with concurrent async calls"""               
        if self.use_async:
            # Run both async methods concurrently from sync function
            vector_results, bm25_results = await asyncio.gather(
                self.vector_search(query_embedding, 100),
                self.bm25_search(query, 100)
            )
        else:
            vector_results = await self.vector_search(query_embedding, 100)
            bm25_results = await self.bm25_search(query, 100)

        
        # Rest is sync
        combined = self.rrf_combine(vector_results, bm25_results)
        top_indices = sorted(combined.keys(), key=combined.get, reverse=True)[:top_k]
        
        top_chunks = [self.chunks[i] for i in top_indices if i < len(self.chunks)]
        
        return top_chunks


    def search_documents(self, query, query_embedding, top_k=5):
        if self.use_azure_search:
                azure_results = self.azure_backend.search(query_embedding=query_embedding,
                                                          query_text=query,
                                                          top_k=top_k
                                                          )
                
                # # Convert to match your local format if needed
                # top_content = [doc["content"] for doc in azure_results]
                # top_indices = [i for i in range(len(azure_results))]  # Sequential indices
                return azure_results
        
        top_chunks = asyncio.run(self.hybrid_search(query, query_embedding, top_k))
        return top_chunks
    
    def save_indexes(self, faiss_path=None, whoosh_path=None):
        """Save indexes to disk for persistence"""
        if faiss_path is None:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
        if whoosh_path is None:
            whoosh_path = self.index_dir + "/whoosh"
            
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)
            print(f"✅ FAISS index saved to {faiss_path}")
            
        if self.whoosh_index is not None:
            print(f"✅ Whoosh index already saved to {whoosh_path}")

    def load_indexes(self, faiss_path=None, whoosh_path=None, chunks_data=None):
        """Load indexes from disk"""
        if faiss_path is None:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
        if whoosh_path is None:
            whoosh_path = self.index_dir + "/whoosh"

        success = True   
        try:
            # Load FAISS index
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                print(f"✅ FAISS index loaded from {faiss_path}")
            else:
              print(f"⚠️ FAISS index not found at {faiss_path}")
              success = False
            
            # Load Whoosh index
            if os.path.exists(whoosh_path):
                self.whoosh_index = open_dir(whoosh_path)
                print(f"✅ Whoosh index loaded from {whoosh_path}")
            else:
              print(f"⚠️ Whoosh index not found at {whoosh_path}")
              success = False
            
            # Load chunks data if provided
            if chunks_data is not None:
                self.chunks = chunks_data
                print(f"✅ Loaded {len(self.chunks)} chunks")
                
            return success
        except Exception as e:
            print(f"❌ Error loading indexes: {e}")
            return False

    def get_stats(self):
        """Get index statistics"""
        stats = {
            "num_chunks": len(self.chunks),
            "faiss_index_type": type(self.faiss_index).__name__ if self.faiss_index else None,
            "whoosh_index_exists": self.whoosh_index is not None,
            "index_directory": self.index_dir
        }
        return stats


if __name__ == "__main__":
    # asyncio.run(main())
    search = SearchService(index_dir="indexes")
    
    # Load embeddings
    path = 'data_processed/Lux_cssf18_698eng_embds_local_BAAI_bge-large-en-v1.5.json'
    with open(path, 'r') as f:
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
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu')
    query_embed = model.encode(query)
    

    # results, indices = asyncio.run(search.hybrid_search(query, query_embed, top_k=5))
    # or 
    results, indices = search.search_documents(query, query_embed, top_k=5)

    
    print("Search Results:")
    print(f"Indices: {indices}")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content'][:100]}...")
    
    # Print stats
    print("\nIndex Stats:", search.get_stats())


    

