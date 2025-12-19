"""
Service layer for EFIRAS application.

This module provides service-level business logic including rate limiting,
query recording, and user tracking operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.config import MAX_QUERY_LENGTH
from fastapi import HTTPException, Request

from auth.firebase_user_tracker import usage_tracker
from core.services.rate_limiter import SimpleMemoryRateLimiter

logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = SimpleMemoryRateLimiter()


def check_rate_limit(request: Request) -> Dict[str, Any]:
    """
    Check if request should be rate limited.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit check result dictionary

    Raises:
        HTTPException: If rate limit is exceeded (429 status)
    """
    result = rate_limiter.check_and_increment(request)

    if not result["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": result["reason"],
                "reset_in_seconds": result["reset_in"],
                "tip": "Please wait a moment, or contact us about higher limits for serious usage.",
            },
        )

    return result


async def record_anonymous_query(query_text: str, response_text: str) -> None:
    """
    Record anonymous query to database.

    Ensures anonymous user exists and logs the query for analytics.

    Args:
        query_text: User's query text
        response_text: System's response text
    """
    try:
        async with usage_tracker.connection_pool.acquire() as conn:
            # Ensure anonymous user exists
            await conn.execute(
                """
                INSERT INTO users (user_id, email, plan)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO NOTHING
                """,
                "anonymous",
                "anonymous@system.internal",
                "free",
            )

            # Insert query log with truncated text
            truncated_query = query_text[:MAX_QUERY_LENGTH]
            await conn.execute(
                """
                INSERT INTO query_logs (user_id, query_text, success, created_at)
                VALUES ($1, $2, $3, $4)
                """,
                "anonymous",
                truncated_query,
                True,
                datetime.now(),
            )

        logger.info("Anonymous query recorded to database")

    except Exception as e:
        logger.error(f"Failed to record anonymous query: {e}", exc_info=True)


async def record_authenticated_query(
    user_id: str,
    email: str,
    query_text: str,
    response_time_ms: int,
    success: bool,
    error_msg: Optional[str] = None,
) -> None:
    """
    Record authenticated query to database.

    Args:
        user_id: User's unique identifier
        email: User's email address
        query_text: User's query text
        response_time_ms: Response time in milliseconds
        success: Whether the query succeeded
        error_msg: Optional error message if query failed
    """
    try:
        await usage_tracker.record_query(
            user_id=user_id,
            email=email,
            query_text=query_text,
            response_time_ms=response_time_ms,
            success=success,
            error_message=error_msg,
        )
        logger.info(f"Query recorded to database for user {email}")

    except Exception as e:
        logger.error(
            f"Failed to record authenticated query for {email}: {e}", exc_info=True
        )
