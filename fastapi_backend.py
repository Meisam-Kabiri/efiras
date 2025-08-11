from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import tempfile
import json
from pathlib import Path
import requests
from datetime import datetime


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
import time
import hashlib
from fastapi import Request, HTTPException
from collections import defaultdict
from threading import Lock

from visit_tracker import track_visit, get_visit_stats, get_recent_visits

class SimpleMemoryRateLimiter:
    def __init__(self):
        self.usage_data = defaultdict(lambda: {"minute": {}, "hour": {}, "day": {}})
        self.lock = Lock()  # Thread safety
        
        # Set your limits here
        self.LIMITS = {
            "minute": 1,   # 3 requests per minute
            "hour": 1,    # 20 requests per hour
            "day": 10     # 100 requests per day
        }
    def get_user_fingerprint(self, request: Request) -> str:
        """Create unique fingerprint for each user based on browser characteristics"""
        
        # Combine IP + User Agent + Accept Language for uniqueness
        # Safely get headers with fallbacks for None values
        fingerprint_parts = [
            str(request.client.host or "unknown"),  # IP address
            str(request.headers.get("user-agent") or "unknown"),  # Browser info
            str(request.headers.get("accept-language") or "unknown"),  # Language
            str(request.headers.get("accept-encoding") or "unknown"),   # Encoding
            str(request.headers.get("cache-control") or "unknown"),
            str(request.headers.get("sec-ch-ua") or "unknown"),
        ]
        
        # Join them and create a hash
        fingerprint_string = "|".join(fingerprint_parts)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        
        # Return first 16 characters (enough for uniqueness)
        return fingerprint_hash[:16]


    def clean_old_data(self, user_data, window_type, current_window):
        """Remove old time windows to prevent memory buildup"""
        to_remove = []
        for window_time in user_data[window_type]:
            if window_time < current_window - 2:  # Keep last 2 windows for safety
                to_remove.append(window_time)
        
        for old_window in to_remove:
            del user_data[window_type][old_window]
    
    def check_and_increment(self, request):
        """Check rate limit and increment if allowed"""
        user_id = self.get_user_fingerprint(request)
        current_time = int(time.time())

        print(f"🔍 Debug - User ID: {user_id}")
        print(f"🔍 Debug - Current time: {current_time}")
        
        # Calculate current time windows
        current_minute = current_time // 60
        current_hour = current_time // 3600
        current_day = current_time // 86400

        print(f"🔍 Debug - Current minute window: {current_minute}")
        
        with self.lock:
            user_data = self.usage_data[user_id]
            
            # Clean old data to prevent memory leak
            self.clean_old_data(user_data, "minute", current_minute)
            self.clean_old_data(user_data, "hour", current_hour)
            self.clean_old_data(user_data, "day", current_day)
            
            # Get current counts
            minute_count = user_data["minute"].get(current_minute, 0)
            hour_count = user_data["hour"].get(current_hour, 0)
            day_count = user_data["day"].get(current_day, 0)

            print(f"🔍 Debug - Current counts: minute={minute_count}, hour={hour_count}, day={day_count}")
            print(f"🔍 Debug - Limits: minute={self.LIMITS['minute']}, hour={self.LIMITS['hour']}, day={self.LIMITS['day']}")
            
            # # Check limits
            # if minute_count >= self.LIMITS["minute"]:
            #     return {
            #         "allowed": False,
            #         "reason": "Too many requests per minute. Please wait a moment.",
            #         "reset_in": 60 - (current_time % 60)
            #     }
            
            # if hour_count >= self.LIMITS["hour"]:
            #     return {
            #         "allowed": False,
            #         "reason": "Hourly limit reached. Please wait or consider our paid plans.",
            #         "reset_in": 3600 - (current_time % 3600)
            #     }
                
            if day_count >= self.LIMITS["day"]:
                return {
                    "allowed": False,
                    "reason": "Daily limit reached.!",
                    "reset_in": 86400 - (current_time % 86400)
                }
            
            print(f"✅ Debug - ALLOWED: incrementing counters")
            # All good - increment counters
            user_data["minute"][current_minute] = minute_count + 1
            user_data["hour"][current_hour] = hour_count + 1
            user_data["day"][current_day] = day_count + 1
            
            print(f"🔍 Debug - After increment: minute={minute_count + 1}")
            return {
                "allowed": True,
                "remaining": min(
                    self.LIMITS["minute"] - minute_count - 1,
                    self.LIMITS["hour"] - hour_count - 1,
                    self.LIMITS["day"] - day_count - 1
                ),
                "usage": {
                    "minute": f"{minute_count + 1}/{self.LIMITS['minute']}",
                    "hour": f"{hour_count + 1}/{self.LIMITS['hour']}",
                    "day": f"{day_count + 1}/{self.LIMITS['day']}"
                }
            }


# Global variables
embedding_service = None
search_service = None
rag_generator = None
rate_limiter = SimpleMemoryRateLimiter()

def check_rate_limit(request: Request):
    """Check if request should be rate limited"""
    
    result = rate_limiter.check_and_increment(request)
    
    if not result["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": result["reason"],
                "reset_in_seconds": result["reset_in"],
                "tip": "Please wait a moment, or contact us about higher limits for serious usage."
            }
        )
    
    return result

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

    print("🔧 Loading indexes...")
    success = search_service.load_indexes()
    if success:
        print("✅ All indexes loaded successfully!")
    else:
        print("❌ Failed to load some indexes")

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
async def query_documents_stream(request: QueryRequest, http_request: Request):
    
    rate_info = check_rate_limit(http_request)

    print("the query stream started")
     
    
    def generate_response():
        print("the genrate response fucntion started")
        full_response = "" 
        try:
            start = time.time()
            query_embedding = embedding_service.embed_text(request.question)
            print(f"Embedding: {time.time() - start:.2f}s")
            start = time.time()
            relevant_chunks = search_service.search_documents(
                request.question, query_embedding, top_k=12
            )
            print(f"Search: {time.time() - start:.2f}s")
            
            
            for chunk in rag_generator.answer_query_stream(request.question, relevant_chunks):
                full_response += chunk
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send sources and completion
            # sources = [...]  # your existing sources code
            # final_data = {"type": "sources", "sources": sources}
            # yield f"data: {json.dumps(final_data)}\n\n"
            yield f"data: [DONE]\n\n"


            # Save query and answer after streaming is complete
            query_log = {
                "timestamp": datetime.now().isoformat(),
                "query": request.question,
                "answer": full_response
            }

            # Save to file
            with open("query_logs.json", "a") as f:
                f.write(json.dumps(query_log) + "\n")
            print(f"Logged query: {request.question[:50]}...")
            
        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.get("/")
def home(request: Request):
    track_visit(request)
    return {"message": "Financial RAG API is running with 3-service architecture"}

@app.get("/admin/visits")
async def visit_statistics():
    """Get website visit statistics"""
    stats = get_visit_stats()
    recent = get_recent_visits(10)
    return {
        "statistics": stats,
        "recent_visits": recent
    }


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
