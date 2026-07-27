"""
EFIRAS FastAPI Application.

This module provides the main FastAPI application for the EFIRAS
(Enhanced Financial Information Retrieval and Analysis System) API,
including RAG-based document querying with authentication and usage tracking.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional




import requests
from config import (
    ALLOW_CREDENTIALS,
    ALLOWED_HEADERS,
    ALLOWED_METHODS,
    ALLOWED_ORIGINS,
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    INDEX_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    REQUEST_TIMEOUT_SECONDS,
    REQUIRED_INDEX_FILES,
)
from endpoints import (
    authenticated_query_stream,
    get_status,
    health_check,
    home,
    query_documents_stream,
    set_services,
)
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic_models import (
    APIInfoResponse,
    AuthenticatedQueryRequest,
    QueryRequest,
    StatusResponse,
)

from auth.auth_middleware import get_current_user
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.rag.search_service import SearchService

from slowapi.errors import RateLimitExceeded
from app.rate_limit import limiter, custom_rate_limit_handler



# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global service instances
embedding_service: Optional[EmbeddingService] = None
search_service: Optional[SearchService] = None
rag_generator: Optional[RAGGenerator] = None


async def startup() -> None:
    """
    Initialize all application services on startup.

    This function:
    - Loads ML models (embeddings, RAG generator)
    - Downloads required index files from remote storage
    - Initializes search indexes

    Raises:
        Exception: If any critical service fails to initialize
    """
    global embedding_service, search_service, rag_generator

    try:
        logger.info("Loading ML models...")
        from app.config import USE_LOCAL_EMBEDDINGS, EMBEDDING_MODEL
        embedding_service = EmbeddingService(use_local=USE_LOCAL_EMBEDDINGS, online_model=EMBEDDING_MODEL)
        search_service = SearchService(index_dir=INDEX_DIR)
        rag_generator = RAGGenerator()
        logger.info("Models loaded successfully")

        logger.info("Starting RAG system initialization...")

        # Create indexes directory
        Path(INDEX_DIR).mkdir(exist_ok=True, parents=True)

        # Download required index files if they don't exist
        for file_info in REQUIRED_INDEX_FILES:
            file_path = Path(file_info["path"])
            if not file_path.exists():
                logger.info(f"Downloading {file_info['path']}...")
                try:
                    response = requests.get(
                        file_info["url"], timeout=REQUEST_TIMEOUT_SECONDS
                    )
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        f.write(response.content)

                    file_size_mb = file_path.stat().st_size / 1024 / 1024
                    logger.info(
                        f"Downloaded {file_info['path']} ({file_size_mb:.1f}MB)"
                    )
                except requests.RequestException as e:
                    logger.error(f"Failed to download {file_info['path']}: {e}")
                    raise
            else:
                logger.info(f"{file_info['path']} already exists, skipping download")

        logger.info("File check/download complete")

        logger.info("Loading search indexes...")
        success = search_service.load_indexes()
        if success:
            logger.info("All indexes loaded successfully")
        else:
            logger.error("Failed to load some indexes")
            raise RuntimeError("Index loading failed")

        # Set services in endpoints module
        set_services(embedding_service, search_service, rag_generator)

        logger.info("EFIRAS backend started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


async def shutdown() -> None:
    """
    Cleanup resources when application shuts down.

    This function performs any necessary cleanup operations.
    """
    try:
        logger.info("Running shutdown tasks...")
        logger.info("Shutdown completed successfully")

    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    await startup()
    yield
    await shutdown()


# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan
)

# Register limiter with app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)


# Route registration
@app.get("/", tags=["general"], response_model=APIInfoResponse, summary="API Information")
def route_home() -> APIInfoResponse:
    """Root endpoint with basic API information."""
    return home()


@app.get(
    "/health", tags=["general"], response_model=StatusResponse, summary="Health Check"
)
def route_health_check() -> StatusResponse:
    """Health check endpoint for monitoring and load balancers."""
    return health_check()


@app.post("/query-stream", tags=["query"], summary="Public Query Stream")
@limiter.limit("10/hour;15/day")
async def public_query(
    query_request: QueryRequest, request: Request
) -> StreamingResponse:
    """
    Public query endpoint with rate limiting.

    Stream responses for document queries without authentication.
    """
    return await query_documents_stream(query_request)


@app.post("/auth/query-stream",
    tags=["query", "authenticated"],
    summary="Authenticated Query Stream",
)
@limiter.limit("15/hour;30/day")
async def auth_query(
    query_request: AuthenticatedQueryRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Authenticated query endpoint with rate limiting.

    Requires valid authentication token. Higher rate limits than public endpoint.
    """
    return await authenticated_query_stream(query_request, current_user)



@app.get("/status", tags=["general"], summary="Get System Status")
def route_status() -> Dict[str, Any]:
    """Get detailed system status and configuration."""
    return get_status()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", DEFAULT_PORT))
    logger.info(f"Starting EFIRAS API server on {DEFAULT_HOST}:{port}")
    uvicorn.run("efiras_app:app", host=DEFAULT_HOST, port=port)
