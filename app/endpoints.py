import logging
import json
import time
import asyncio
from datetime import datetime
from fastapi import Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from core.rag.search_service import SearchService
from core.rag.embedding_service import EmbeddingService
from core.rag.rag_generator import RAGGenerator
from core.database.operations.visit_tracker import track_visit, get_visit_stats, get_recent_visits
from auth.auth_middleware import get_current_user
from auth.firebase_user_tracker import usage_tracker

from pydantic_models import QueryRequest, AuthenticatedQueryRequest, UsageResponse
from services import check_rate_limit, record_anonymous_query, record_authenticated_query

logger = logging.getLogger(__name__)

# Global variables (will be set by main app)
embedding_service = None
search_service = None
rag_generator = None

def set_services(emb_service, search_svc, rag_gen):
    """Set the global service instances"""
    global embedding_service, search_service, rag_generator
    embedding_service = emb_service
    search_service = search_svc
    rag_generator = rag_gen

def home(request: Request):
    track_visit(request)
    return {"message": "EFIRAS API is running", "version": "1.0.0"}

def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "EFIRAS API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

async def query_documents_stream(request: QueryRequest, http_request: Request, background_tasks: BackgroundTasks):
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
            
            yield f"data: [DONE]\n\n"

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

async def get_user_analytics(current_user: dict = Depends(get_current_user)):
    """Get detailed user analytics"""
    try:
        user_id = current_user["user_id"]
        analytics = await usage_tracker.get_user_analytics(user_id)
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

async def visit_statistics():
    """Get website visit statistics"""
    stats = get_visit_stats()
    recent = get_recent_visits(10)
    return {
        "statistics": stats,
        "recent_visits": recent
    }

async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """Get system-wide statistics (requires authentication)"""
    try:
        stats = await usage_tracker.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

async def delete_user_account(current_user: dict = Depends(get_current_user)):
    """Delete user account and all related data"""
    try:
        user_id = current_user["user_id"]
        email = current_user["email"]
        
        logger.info(f"Deleting account for user: {email}")
        
        success = await usage_tracker.delete_user_account(user_id, email)
        
        if success:
            return {"message": "Account deleted successfully", "deleted": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete account")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")

def get_status():
    """Get system status"""
    return {
        "status": "running",
        "search_stats": search_service.get_stats(),
        "rag_config": rag_generator.get_config_info(),
        "authentication": "enabled"
    }