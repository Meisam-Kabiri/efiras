import logging
import json
from datetime import datetime
from fastapi import Request, HTTPException
from core.services.rate_limiter import SimpleMemoryRateLimiter
from auth.firebase_user_tracker import usage_tracker

logger = logging.getLogger(__name__)

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