"""
API endpoint handlers for EFIRAS application.

This module contains all the endpoint handler functions for the EFIRAS API,
including public and authenticated query endpoints.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import API_VERSION
from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_models import (
    APIInfoResponse,
    AuthenticatedQueryRequest,
    QueryRequest,
    StatusResponse,
)

from auth.auth_middleware import get_current_user
from core.rag import agent as _agent
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


def home() -> APIInfoResponse:
    """
    Root endpoint handler.

    Returns:
        Basic API information
    """
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
    request: QueryRequest
) -> StreamingResponse:
    """
    Public query endpoint with rate limiting.

    Streams responses for document queries without requiring authentication.
    Subject to rate limiting via Redis (applied in main app).

    Args:
        request: Query request with question

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If rate limit is exceeded
    """
    logger.info(f"Public query received: {request.question[:50]}...")

    def generate_response():
        try:
            route_result, relevant_chunks, short_circuit = _agent.run(
                request.question, embedding_service, search_service
            )
            logger.info(
                f"Route: kind={route_result.get('kind')} scope={route_result.get('scope')} "
                f"docs={route_result.get('documents')} chunks={len(relevant_chunks)}"
            )

            if short_circuit is not None:
                yield f"data: {json.dumps({'type': 'content', 'content': short_circuit})}\n\n"
                yield f"data: [DONE]\n\n"
                return

            for chunk in rag_generator.answer_query_stream(
                route_result.get("expanded_query") or request.question,
                relevant_chunks,
            ):
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in query generation: {e}", exc_info=True)
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


async def authenticated_query_stream(
    request: AuthenticatedQueryRequest,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Authenticated query endpoint with rate limiting.

    Requires valid authentication. Rate limiting handled via Redis (applied in main app).
    Provides higher top_k results for authenticated users.

    Args:
        request: Authenticated query request with question
        current_user: Current authenticated user from JWT token

    Returns:
        StreamingResponse with Server-Sent Events

    Raises:
        HTTPException: If rate limit exceeded or authentication fails
    """
    email = current_user["email"]
    logger.info(f"Authenticated query from {email}: {request.question[:50]}...")

    def generate_authenticated_response():
        try:
            route_result, relevant_chunks, short_circuit = _agent.run(
                request.question, embedding_service, search_service
            )
            logger.info(
                f"Auth route [{email}]: kind={route_result.get('kind')} "
                f"scope={route_result.get('scope')} chunks={len(relevant_chunks)}"
            )

            if short_circuit is not None:
                yield f"data: {json.dumps({'type': 'content', 'content': short_circuit})}\n\n"
                yield f"data: [DONE]\n\n"
                return

            for chunk in rag_generator.answer_query_stream(
                route_result.get("expanded_query") or request.question,
                relevant_chunks,
            ):
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: [DONE]\n\n"

        except Exception as e:
            logger.error(
                f"Authenticated query error for {email}: {e}", exc_info=True
            )
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_authenticated_response(), media_type="text/event-stream"
    )






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




