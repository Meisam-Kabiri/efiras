from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import tempfile
import json
from pathlib import Path
import requests
from datetime import datetime
from typing import Optional
import asyncio

# Add your src path
from core.rag.search_service import SearchService
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator


from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import time
import hashlib
from fastapi import Request, HTTPException
from collections import defaultdict
from threading import Lock

from core.database.operations.visit_tracker import track_visit, get_visit_stats, get_recent_visits
from core.services.rate_limiter import SimpleMemoryRateLimiter

# from auth.firebase_admin_config import verify_firebase_token
from auth.auth_middleware import get_current_user
from auth.firebase_user_tracker import usage_tracker, initialize_usage_tracker


# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Initialize all services"""
    global embedding_service, search_service, rag_generator
    
    try:
        logger.info("🚀 Loading models...")
        embedding_service = EmbeddingService()
        search_service = SearchService(index_dir="data/indexes")
        rag_generator = RAGGenerator()
        logger.info("✅ Models loaded!")

        logger.info("🚀 Starting RAG system initialization...")
        
        # Create indexes directory
        Path("data/indexes").mkdir(exist_ok=True)
        
        # Download files only if they don't exist
        files_to_download = [
            {
                "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/bm25_tokenized.pkl",
                "path": "data/indexes/bm25_tokenized.pkl"
            },
            {
                "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/faiss.index", 
                "path": "data/indexes/faiss.index"
            },
            {
                "url": "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/chunks_metadata.json",
                "path": "data/indexes/chunks_metadata.json"
            }
        ]
        
        for file_info in files_to_download:
            file_path = Path(file_info["path"])
            if not file_path.exists():
                logger.info(f"📥 Downloading {file_info['path']}...")
                try:
                    response = requests.get(file_info["url"], timeout=300)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Downloaded {file_info['path']} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                except Exception as e:
                    logger.error(f"❌ Failed to download {file_info['path']}: {e}")
            else:
                logger.info(f"✅ {file_info['path']} already exists, skipping download")
        
        logger.info("🎉 File check/download complete!")

        logger.info("🔧 Loading indexes...")
        success = search_service.load_indexes()
        if success:
            logger.info("✅ All indexes loaded successfully!")
        else:
            logger.error("❌ Failed to load some indexes")

        # NEW: Initialize authentication services
        logger.info("🔧 Initializing authentication services...")
        await initialize_usage_tracker()
        logger.info("✅ Authentication services initialized")
        
        logger.info("🎉 EFIRAS backend started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# This will run when the application shuts down
async def shutdown():
    """Cleanup when app shuts down"""
    try:
        logger.info("🔧 Running shutdown tasks...")
        
        # NEW: Close authentication services
        await usage_tracker.close()
        logger.info("✅ Authentication services closed")
        
        # Add any other cleanup here
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield  # The application runs here
    await shutdown()

app = FastAPI(
    title="EFIRAS API",
    description="Regulatory AI Assistant with Authentication",
    version="1.0.0",
    lifespan=lifespan
)

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
    usage_info: Optional[dict] = None  # NEW: Usage tracking info

class AuthenticatedQueryRequest(BaseModel):
    question: str
    document_filter: Optional[str] = None

class UsageResponse(BaseModel):
    daily_queries: int
    daily_limit: int
    remaining: int
    plan: str
    total_queries: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int




# PUBLIC ENDPOINTS (No authentication required)

@app.get("/")
def home(request: Request):
    track_visit(request)
    return {"message": "EFIRAS API is running", "version": "1.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "EFIRAS API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query-stream")
async def query_documents_stream(request: QueryRequest, http_request: Request, background_tasks: BackgroundTasks  ):
    
    """Public query endpoint with rate limiting (no auth required)"""
    rate_info = check_rate_limit(http_request)
    logger.info(f"Public query: {request.question[:50]}...")

     
    
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


            # # Save query and answer after streaming is complete
            # query_log = {
            #     "timestamp": datetime.now().isoformat(),
            #     "query": request.question,
            #     "answer": full_response
            # }

            # # Save to file
            # with open("logs/query_logs.json", "a") as f:
            #     f.write(json.dumps(query_log) + "\n")
            # print(f"Logged query: {request.question[:50]}...")

            # Schedule background task to record query
            background_tasks.add_task(
                record_anonymous_query,
                request.question,
                full_response
            )
            print(f"Logged query: {request.question[:50]}...")
            
        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")

async def record_anonymous_query(query_text: str, response_text: str):
    """Record anonymous query to database - with anonymous user creation"""
    try:
        async with usage_tracker.connection_pool.acquire() as conn:
            # First, ensure "anonymous" user exists
            await conn.execute("""
                INSERT INTO users (user_id, email, plan) 
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO NOTHING
            """, "anonymous", "anonymous@system.internal", "free")
            
            # Then insert query log
            await conn.execute("""
                INSERT INTO query_logs (user_id, query_text, success, created_at)
                VALUES ($1, $2, $3, $4)
            """, "anonymous", query_text[:1000], True, datetime.now())
            
        logger.info(f"✅ Anonymous query recorded to database")
    except Exception as e:
        logger.error(f"❌ Failed to record anonymous query: {e}")



@app.post("/auth/query-stream")
async def authenticated_query_stream(
    request: AuthenticatedQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Authenticated query endpoint with usage tracking"""
    start_time = time.time()
    
    try:
        user_id = current_user["user_id"]
        email = current_user["email"]
        
        logger.info(f"Authenticated query from {email}: {request.question[:50]}...")
        
        # Check usage limits
        can_query, usage_info = await usage_tracker.can_make_query(user_id, email)
        
        if not can_query:
            logger.warning(f"Rate limit exceeded for user {email}")
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit exceeded. You've used {usage_info.daily_queries}/{usage_info.daily_limit} queries today."
            )
        
        # Variables to track the response
        full_response = ""
        query_success = True
        error_message = None
        
        def generate_authenticated_response():
            nonlocal full_response, query_success, error_message
            
            try:
                # Process query with your RAG system
                query_embedding = embedding_service.embed_text(request.question)
                relevant_chunks = search_service.search_documents(
                    request.question, query_embedding, top_k=15
                )
                
                for chunk in rag_generator.answer_query_stream(request.question, relevant_chunks):
                    full_response += chunk
                    data = {"type": "content", "content": chunk}
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Send completion
                yield f"data: [DONE]\n\n"
                
                # Mark as successful
                query_success = True
                
            except Exception as e:
                logger.error(f"Authenticated query error for {email}: {e}")
                query_success = False
                error_message = str(e)
                
                error_data = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        # Create the streaming response
        response = StreamingResponse(generate_authenticated_response(), media_type="text/event-stream")
        
        # Record the query AFTER streaming is set up but BEFORE returning
        try:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # # Use asyncio.create_task to record in background
            # asyncio.create_task(record_query_async(
            #     user_id=user_id,
            #     email=email,
            #     query_text=request.question,
            #     response_time_ms=response_time_ms,
            #     success=query_success,
            #     error_message=error_message
            # ))

            background_tasks.add_task(
            record_authenticated_query,
            user_id, email, request.question, response_time_ms, True
        )
            
        except Exception as e:
            logger.error(f"Failed to create recording task: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


async def record_authenticated_query(user_id: str, email: str, query_text: str, 
                                         response_time_ms: int, success: bool, error_msg: str = None):
    """Record authenticated query to database - ASYNC VERSION"""
    try:
        await usage_tracker.record_query(
            user_id=user_id,
            email=email,
            query_text=query_text,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_msg
        )
        logger.info(f"✅ Query recorded to database for user {email}")
    except Exception as e:
        logger.error(f"❌ Failed to record authenticated query: {e}")

# # Helper function to record query asynchronously
# async def record_query_async(user_id: str, email: str, query_text: str, 
#                            response_time_ms: int, success: bool, error_message: str = None):
#     """Record query to database asynchronously"""
#     try:
#         await usage_tracker.record_query(
#             user_id=user_id,
#             email=email,
#             query_text=query_text,
#             response_time_ms=response_time_ms,
#             success=success,
#             error_message=error_message
#         )
#         logger.info(f"✅ Query recorded to database for user {email}")
        
#     except Exception as e:
#         logger.error(f"❌ Failed to record query to database: {e}")




@app.get("/auth/usage", response_model=UsageResponse)
async def get_user_usage(current_user: dict = Depends(get_current_user)):
    """Get current user's usage statistics"""
    try:
        user_id = current_user["user_id"]
        email = current_user["email"]
        
        can_query, usage_info = await usage_tracker.can_make_query(user_id, email)
        
        return UsageResponse(
            daily_queries=usage_info.daily_queries,
            daily_limit=usage_info.daily_limit,
            remaining=usage_info.remaining,
            plan=usage_info.plan,
            total_queries=usage_info.total_queries
        )
        
    except Exception as e:
        logger.error(f"Usage check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get usage information")

@app.get("/auth/analytics")
async def get_user_analytics(current_user: dict = Depends(get_current_user)):
    """Get detailed user analytics"""
    try:
        user_id = current_user["user_id"]
        analytics = await usage_tracker.get_user_analytics(user_id)
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

# ADMIN ENDPOINTS

@app.get("/admin/visits")
async def visit_statistics():
    """Get website visit statistics"""
    stats = get_visit_stats()
    recent = get_recent_visits(10)
    return {
        "statistics": stats,
        "recent_visits": recent
    }

@app.get("/admin/system-stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """Get system-wide statistics (requires authentication)"""
    try:
        stats = await usage_tracker.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

@app.get("/status")
def get_status():
    """Get system status"""
    return {
        "status": "running",
        "search_stats": search_service.get_stats(),
        "rag_config": rag_generator.get_config_info(),
        "authentication": "enabled"
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting modernized RAG API server...")
    port=int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="localhost", port=8080)
    uvicorn.run("efiras_backend:app", host="0.0.0.0", port=port)
