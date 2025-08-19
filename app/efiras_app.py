from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
import requests
from contextlib import asynccontextmanager

from core.rag.search_service import SearchService
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from auth.firebase_user_tracker import usage_tracker, initialize_usage_tracker

from endpoints import (
    set_services, home, health_check, query_documents_stream,
    authenticated_query_stream, get_user_usage, get_user_analytics,
    visit_statistics, get_system_stats, get_status, delete_user_account
)
from pydantic_models import QueryRequest, AuthenticatedQueryRequest, UsageResponse


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
embedding_service = None
search_service = None
rag_generator = None

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
        
        # Set services in endpoints module
        set_services(embedding_service, search_service, rag_generator)
        
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

# Import dependency injection for routes
from fastapi import Request, BackgroundTasks, Depends
from auth.auth_middleware import get_current_user

# Route registration
@app.get("/")
def route_home(request: Request):
    return home(request)

@app.get("/health")
def route_health_check():
    return health_check()

@app.post("/query-stream")
async def route_query_stream(request: QueryRequest, http_request: Request, background_tasks: BackgroundTasks):
    return await query_documents_stream(request, http_request, background_tasks)

@app.post("/auth/query-stream")
async def route_auth_query_stream(
    request: AuthenticatedQueryRequest, 
    background_tasks: BackgroundTasks, 
    current_user: dict = Depends(get_current_user)
):
    return await authenticated_query_stream(request, background_tasks, current_user)

@app.get("/auth/usage", response_model=UsageResponse)
async def route_user_usage(current_user: dict = Depends(get_current_user)):
    return await get_user_usage(current_user)

@app.get("/auth/analytics")
async def route_user_analytics(current_user: dict = Depends(get_current_user)):
    return await get_user_analytics(current_user)

@app.get("/admin/visits")
async def route_visit_stats():
    return await visit_statistics()

@app.get("/admin/system-stats")
async def route_system_stats(current_user: dict = Depends(get_current_user)):
    return await get_system_stats(current_user)

@app.delete("/auth/account")
async def route_delete_account(current_user: dict = Depends(get_current_user)):
    return await delete_user_account(current_user)

@app.get("/status")
def route_status():
    return get_status()


if __name__ == "__main__":
    import uvicorn
    print("Starting modernized RAG API server...")
    port=int(os.environ.get("PORT", 8080))
    # uvicorn.run(app, host="localhost", port=8080)
    uvicorn.run("efiras_app:app", host="0.0.0.0", port=port)
