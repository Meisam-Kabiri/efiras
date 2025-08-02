from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import tempfile
import json
from pathlib import Path
import requests


# Add your src path
sys.path.append(os.path.abspath("src"))
from rag.search_service import SearchService
from rag.embedding_service import EmbeddingService
from rag.rag_generator import RAGGenerator
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager



# Global variables
embedding_service = None
search_service = None
rag_generator = None

async def startup():
    global embedding_service, search_service, rag_generator  # ADD THIS LINE
    print("🚀 Loading models...")
    embedding_service = EmbeddingService()
    search_service = SearchService(index_dir="indexes")
    rag_generator = RAGGenerator()
    print("✅ Models loaded!")

    print("🚀 Starting RAG system initialization...")
    
    # Create indexes directory
    Path("indexes").mkdir(exist_ok=True)
    
    # Download files only if they don't exist
    files_to_download = [
        {
            "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/bm25_tokenized.pkl",
            "path": "indexes/bm25_tokenized.pkl"
        },
        {
            "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/faiss.index", 
            "path": "indexes/faiss.index"
        },
        {
            "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/chunks_metadata.json",
            "path": "indexes/chunks_metadata.json"
        }
    ]
    
    for file_info in files_to_download:
        file_path = Path(file_info["path"])
        if not file_path.exists():
            print(f"📥 Downloading {file_info['path']}...")
            try:
                response = requests.get(file_info["url"], timeout=300)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {file_info['path']} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
            except Exception as e:
                print(f"❌ Failed to download {file_info['path']}: {e}")
        else:
            print(f"✅ {file_info['path']} already exists, skipping download")
    
    print("🎉 File check/download complete!")

# This will run when the application shuts down
async def shutdown():
    print("Running shutdown tasks")
    # Clean up resources here
    # Example: close database connections, etc.

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield  # The application runs here
    await shutdown()

app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int





print("RAG system ready!")

@app.post("/query-stream")
async def query_documents_stream(request: QueryRequest):
    print("the query stream started")
    import time 
    
    def generate_response():
        print("the genrate response fucntion started")
        try:
            start = time.time()
            query_embedding = embedding_service.embed_text(request.question)
            print(f"Embedding: {time.time() - start:.2f}s")
            start = time.time()
            relevant_chunks = search_service.search_documents(
                request.question, query_embedding, top_k=12
            )
            print(f"Search: {time.time() - start:.2f}s")
            
            # This will now work because answer_query_stream exists
            for chunk in rag_generator.answer_query_stream(request.question, relevant_chunks):
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send sources and completion
            sources = [...]  # your existing sources code
            final_data = {"type": "sources", "sources": sources}
            yield f"data: {json.dumps(final_data)}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.get("/")
def home():
    return {"message": "Financial RAG API is running with 3-service architecture"}

@app.get("/status")
def get_status():
    """Get system status"""
    return {
        "status": "running",
        # "documents_loaded": len(doc_list),
        # "total_chunks": sum(len(doc["embeddings"]) for doc in doc_list),
        "search_stats": search_service.get_stats(),
        "rag_config": rag_generator.get_config_info()
    }

# @app.get("/documents")
# def list_documents():
#     """List all loaded documents"""
#     docs_info = []
#     for doc in doc_list:
#         docs_info.append({
#             "filename": doc["metadata"]["filename"],
#             "pages": doc["metadata"].get("pages", 0),
#             "chunks": len(doc["embeddings"])
#         })
#     return {"documents": docs_info}

if __name__ == "__main__":
    import uvicorn
    print("Starting modernized RAG API server...")
    port=int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="localhost", port=8080)
    uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=port)
