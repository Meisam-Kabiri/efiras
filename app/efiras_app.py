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
from config import (ALLOW_CREDENTIALS, ALLOWED_HEADERS, ALLOWED_METHODS,
                    ALLOWED_ORIGINS, API_DESCRIPTION, API_TITLE, API_VERSION,
                    DEFAULT_HOST, DEFAULT_PORT, INDEX_DIR, LOG_FORMAT,
                    LOG_LEVEL, REQUEST_TIMEOUT_SECONDS, REQUIRED_INDEX_FILES)
from endpoints import (authenticated_query_stream, clear_session,
                       delete_user_account, get_status, get_system_stats,
                       get_user_analytics, get_user_usage, health_check, home,
                       query_documents_stream, set_services, visit_statistics)
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic_models import (AccountDeletionResponse, APIInfoResponse,
                             AuthenticatedQueryRequest, QueryRequest,
                             SessionClearResponse, StatusResponse,
                             UsageResponse)

from auth.auth_middleware import get_current_user
from auth.firebase_user_tracker import initialize_usage_tracker, usage_tracker
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.rag.search_service import SearchService

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
    - Sets up authentication services

    Raises:
        Exception: If any critical service fails to initialize
    """
    global embedding_service, search_service, rag_generator

    try:
        logger.info("Loading ML models...")
        embedding_service = EmbeddingService()
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

        logger.info("Initializing authentication services...")
        await initialize_usage_tracker()
        logger.info("Authentication services initialized")

        # Set services in endpoints module
        set_services(embedding_service, search_service, rag_generator)

        logger.info("EFIRAS backend started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


async def shutdown() -> None:
    """
    Cleanup resources when application shuts down.

    This function:
    - Closes database connections
    - Releases authentication service resources
    - Performs any necessary cleanup operations
    """
    try:
        logger.info("Running shutdown tasks...")

        await usage_tracker.close()
        logger.info("Authentication services closed")

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

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)


# Route registration
@app.get(
    "/", tags=["general"], response_model=APIInfoResponse, summary="API Information"
)
def route_home(request: Request) -> APIInfoResponse:
    """Root endpoint with basic API information."""
    return home(request)


@app.get(
    "/health", tags=["general"], response_model=StatusResponse, summary="Health Check"
)
def route_health_check() -> StatusResponse:
    """Health check endpoint for monitoring and load balancers."""
    return health_check()


@app.post("/query-stream", tags=["query"], summary="Public Query Stream")
async def route_query_stream(
    request: QueryRequest, http_request: Request, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Public query endpoint with rate limiting.

    Stream responses for document queries without authentication.
    """
    return await query_documents_stream(request, http_request, background_tasks)


@app.post(
    "/auth/query-stream",
    tags=["query", "authenticated"],
    summary="Authenticated Query Stream",
)
async def route_auth_query_stream(
    request: AuthenticatedQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Authenticated query endpoint with usage tracking.

    Requires valid authentication token and tracks query usage against user limits.
    """
    return await authenticated_query_stream(request, background_tasks, current_user)


@app.get(
    "/auth/usage",
    response_model=UsageResponse,
    tags=["authenticated"],
    summary="Get User Usage",
)
async def route_user_usage(
    current_user: dict = Depends(get_current_user),
) -> UsageResponse:
    """Get current user's usage statistics and remaining quota."""
    return await get_user_usage(current_user)


@app.get("/auth/analytics", tags=["authenticated"], summary="Get User Analytics")
async def route_user_analytics(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get detailed analytics for the authenticated user."""
    return await get_user_analytics(current_user)


@app.get("/admin/visits", tags=["admin"], summary="Get Visit Statistics")
async def route_visit_stats() -> Dict[str, Any]:
    """Get website visit statistics (admin endpoint)."""
    return await visit_statistics()


@app.get(
    "/admin/system-stats",
    tags=["admin", "authenticated"],
    summary="Get System Statistics",
)
async def route_system_stats(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get system-wide statistics (requires authentication)."""
    return await get_system_stats(current_user)


@app.delete(
    "/auth/account",
    response_model=AccountDeletionResponse,
    tags=["authenticated"],
    summary="Delete User Account",
)
async def route_delete_account(
    current_user: dict = Depends(get_current_user),
) -> AccountDeletionResponse:
    """Delete user account and all associated data."""
    return await delete_user_account(current_user)


@app.get("/status", tags=["general"], summary="Get System Status")
def route_status() -> Dict[str, Any]:
    """Get detailed system status and configuration."""
    return get_status()


@app.delete(
    "/session/clear",
    response_model=SessionClearResponse,
    tags=["session"],
    summary="Clear Session (DELETE)",
)
async def route_clear_session_delete(session_id: str) -> SessionClearResponse:
    """Clear session data (DELETE method)."""
    return await clear_session(session_id)


@app.post(
    "/session/clear",
    response_model=SessionClearResponse,
    tags=["session"],
    summary="Clear Session (POST)",
)
async def route_clear_session_post(session_id: str) -> SessionClearResponse:
    """Clear session data (POST method)."""
    return await clear_session(session_id)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", DEFAULT_PORT))
    logger.info(f"Starting EFIRAS API server on {DEFAULT_HOST}:{port}")
    uvicorn.run("efiras_app:app", host=DEFAULT_HOST, port=port)
