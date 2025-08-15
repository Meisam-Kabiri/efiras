from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    usage_info: Optional[dict] = None

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