"""
API endpoint handlers for EFIRAS application.

This module contains all the endpoint handler functions for the EFIRAS API,
including public and authenticated query endpoints, analytics, and administration.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import (API_VERSION, DEFAULT_TOP_K_AUTHENTICATED,
                    DEFAULT_TOP_K_PUBLIC, LOGS_DIR)
from fastapi import BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_models import (AccountDeletionResponse, APIInfoResponse,
                             AuthenticatedQueryRequest, QueryRequest,
                             SessionClearResponse, StatusResponse,
                             UsageResponse, UserContext)
from services import (check_rate_limit, record_anonymous_query,
                      record_authenticated_query)

from auth.auth_middleware import get_current_user
from auth.firebase_user_tracker import usage_tracker
from core.database.operations.visit_tracker import (get_recent_visits,
                                                    get_visit_stats,
                                                    track_visit)
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.rag.search_service import SearchService

logger = logging.getLogger(__name__)

# Global service instances (set by main application)
embedding_service: Optional[EmbeddingService] = None
search_service: Optional[SearchService] = None
rag_generator: Optional[RAGGenerator] = None


def set_services(
    emb_service: EmbeddingService, search_svc: SearchService, rag_gen: RAGGenerator
) -> None:
    """
    Set the global service instances.

    Args:
        emb_service: Embedding service instance
        search_svc: Search service instance
        rag_gen: RAG generator instance
    """
    global embedding_service, search_service, rag_generator
    embedding_service = emb_service
    search_service = search_svc
    rag_generator = rag_gen


def home(request: Request) -> APIInfoResponse:
    """
    Root endpoint handler.

    Args:
        request: FastAPI request object

    Returns:
        Basic API information
    """
    track_visit(request)
    return APIInfoResponse(message="EFIRAS API is running", version=API_VERSION)


def health_check() -> StatusResponse:
    """
    Health check endpoint for monitoring.

    Returns:
        Service health status and metadata
    """
    return StatusResponse(
        status="healthy",
        service="EFIRAS API",
        version=API_VERSION,
        timestamp=datetime.now().isoformat(),
    )


async def query_documents_stream(
    request: QueryRequest, http_request: Request, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Public query endpoint with rate limiting.

    Streams responses for document queries without requiring authentication.
    Subject to rate limiting based on IP address.

    Args:
        request: Query request with question and optional session ID
        http_request: FastAPI HTTP request for rate limiting
        background_tasks: FastAPI background tasks handler

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If rate limit is exceeded
    """
    check_rate_limit(http_request)
    logger.info(f"Public query received: {request.question[:50]}...")

    if request.session_id:
        logger.debug(f"Anonymous session ID: {request.session_id}")
        _log_to_session_file(request.session_id, f"USER: {request.question}\n")

    def generate_response():
        full_response = ""
        try:
            start_time = time.time()
            query_embedding = embedding_service.embed_text(request.question)
            logger.debug(f"Embedding generated in {time.time() - start_time:.2f}s")

            start_time = time.time()
            relevant_chunks = search_service.search_documents(
                request.question, query_embedding, top_k=DEFAULT_TOP_K_PUBLIC
            )
            logger.debug(f"Search completed in {time.time() - start_time:.2f}s")

            for chunk in rag_generator.answer_query_stream(
                request.question, relevant_chunks
            ):
                full_response += chunk
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: [DONE]\n\n"

            if request.session_id:
                _log_to_session_file(request.session_id, f"AI: {full_response}\n\n")

            background_tasks.add_task(
                record_anonymous_query, request.question, full_response
            )
            logger.info(f"Query logged: {request.question[:50]}...")

        except Exception as e:
            logger.error(f"Error in query generation: {e}", exc_info=True)
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


async def authenticated_query_stream(
    request: AuthenticatedQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Authenticated query endpoint with usage tracking.

    Requires valid authentication and tracks queries against user limits.
    Provides higher top_k results for authenticated users.

    Args:
        request: Authenticated query request with question and optional filters
        background_tasks: FastAPI background tasks handler
        current_user: Current authenticated user from JWT token

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If rate limit exceeded or authentication fails
    """
    start_time = time.time()
    user_id = current_user["user_id"]
    email = current_user["email"]

    if request.session_id:
        logger.debug(f"Authenticated session ID: {request.session_id}, User: {email}")
        _log_to_session_file(
            request.session_id, f"USER ({email}): {request.question}\n"
        )

    try:
        logger.info(f"Authenticated query from {email}: {request.question[:50]}...")

        # Check usage limits
        can_query, usage_info = await usage_tracker.can_make_query(user_id, email)

        if not can_query:
            logger.warning(f"Rate limit exceeded for user {email}")
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit exceeded. You've used {usage_info.daily_queries}/{usage_info.daily_limit} queries today.",
            )

        full_response = ""
        query_success = True
        error_message = None

        def generate_authenticated_response():
            nonlocal full_response, query_success, error_message

            try:
                query_embedding = embedding_service.embed_text(request.question)
                relevant_chunks = search_service.search_documents(
                    request.question, query_embedding, top_k=DEFAULT_TOP_K_AUTHENTICATED
                )

                for chunk in rag_generator.answer_query_stream(
                    request.question, relevant_chunks
                ):
                    full_response += chunk
                    data = {"type": "content", "content": chunk}
                    yield f"data: {json.dumps(data)}\n\n"

                yield f"data: [DONE]\n\n"
                query_success = True

            except Exception as e:
                logger.error(
                    f"Authenticated query error for {email}: {e}", exc_info=True
                )
                query_success = False
                error_message = str(e)

                error_data = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        response = StreamingResponse(
            generate_authenticated_response(), media_type="text/event-stream"
        )

        if request.session_id:
            background_tasks.add_task(
                _log_authenticated_response_to_session,
                request.session_id,
                full_response,
            )

        response_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(
            record_authenticated_query,
            user_id,
            email,
            request.question,
            response_time_ms,
            True,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication failed")


async def get_user_usage(
    current_user: dict = Depends(get_current_user),
) -> UsageResponse:
    """
    Get current user's usage statistics.

    Args:
        current_user: Current authenticated user

    Returns:
        UsageResponse with query counts and limits

    Raises:
        HTTPException: If usage information cannot be retrieved
    """
    try:
        user_id = current_user["user_id"]
        email = current_user["email"]

        can_query, usage_info = await usage_tracker.can_make_query(user_id, email)

        return UsageResponse(
            daily_queries=usage_info.daily_queries,
            daily_limit=usage_info.daily_limit,
            remaining=usage_info.remaining,
            plan=usage_info.plan,
            total_queries=usage_info.total_queries,
        )

    except Exception as e:
        logger.error(f"Usage check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get usage information")


async def get_user_analytics(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get detailed analytics for authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        Dictionary containing user analytics data

    Raises:
        HTTPException: If analytics cannot be retrieved
    """
    try:
        user_id = current_user["user_id"]
        analytics = await usage_tracker.get_user_analytics(user_id)
        return analytics

    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get analytics")


async def visit_statistics() -> Dict[str, Any]:
    """
    Get website visit statistics.

    Returns:
        Dictionary with visit statistics and recent visits
    """
    stats = get_visit_stats()
    recent = get_recent_visits(10)
    return {"statistics": stats, "recent_visits": recent}


async def get_system_stats(
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get system-wide statistics.

    Requires authentication to prevent information disclosure.

    Args:
        current_user: Current authenticated user

    Returns:
        Dictionary with system statistics

    Raises:
        HTTPException: If statistics cannot be retrieved
    """
    try:
        stats = await usage_tracker.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"System stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


async def delete_user_account(
    current_user: UserContext = Depends(get_current_user),
) -> AccountDeletionResponse:
    """
    Delete user account and all related data.

    Permanently removes user data from the system.

    Args:
        current_user: Current authenticated user

    Returns:
        Deletion confirmation message

    Raises:
        HTTPException: If account deletion fails
    """
    try:
        user_id = (
            current_user.user_id
            if isinstance(current_user, UserContext)
            else current_user["user_id"]
        )
        email = (
            current_user.email
            if isinstance(current_user, UserContext)
            else current_user["email"]
        )

        logger.info(f"Deleting account for user: {email}")

        success = await usage_tracker.delete_user_account(user_id, email)

        if success:
            return AccountDeletionResponse(
                message="Account deleted successfully", deleted=True
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete account")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete account")


def get_status() -> Dict[str, Any]:
    """
    Get current system status and configuration.

    Note: Returns Dict instead of model due to dynamic nested structure
    from search_service and rag_generator.

    Returns:
        Dictionary with service status and configuration
    """
    return {
        "status": "running",
        "search_stats": search_service.get_stats(),
        "rag_config": rag_generator.get_config_info(),
        "authentication": "enabled",
    }


async def clear_session(session_id: str) -> SessionClearResponse:
    """
    Clear and log session cleanup.

    Args:
        session_id: ID of session to clear

    Returns:
        Success confirmation
    """
    logger.info(f"Clearing session: {session_id}")

    deleted_log_path = Path(LOGS_DIR) / "deleted_sessions.txt"
    deleted_log_path.parent.mkdir(exist_ok=True, parents=True)

    with open(deleted_log_path, "a") as f:
        f.write(f"DELETED: {session_id} at {datetime.now().isoformat()}\n")

    return SessionClearResponse(success=True)


# Helper functions
def _log_to_session_file(session_id: str, content: str) -> None:
    """
    Write content to session log file.

    Args:
        session_id: Session identifier
        content: Content to log
    """
    try:
        log_path = Path(LOGS_DIR) / f"session_{session_id}.txt"
        log_path.parent.mkdir(exist_ok=True, parents=True)

        with open(log_path, "a") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to write to session file: {e}")


async def _log_authenticated_response_to_session(
    session_id: str, full_response: str
) -> None:
    """
    Log authenticated AI response to session file.

    Args:
        session_id: Session identifier
        full_response: Complete AI response text
    """
    _log_to_session_file(session_id, f"AI: {full_response}\n\n")
