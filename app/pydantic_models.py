"""
Pydantic models for request and response validation.

This module defines all data models used for API request validation
and response serialization in the EFIRAS application.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Public query request model."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "question": "What are the capital requirements for Basel III?",
                "session_id": "abc123-def456",
            }
        },
    )

    question: str = Field(
        ...,
        description="User's question to query the document database",
        min_length=1,
        max_length=30000,
        examples=["What are the capital requirements for Basel III?"],
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking conversation history",
        examples=["abc123-def456"],
    )


class QueryResponse(BaseModel):
    """Query response model with answer and sources."""

    question: str = Field(..., description="Original question from the user")
    answer: str = Field(..., description="Generated answer from the RAG system")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of source documents used to generate the answer",
    )
    usage_info: Optional[Dict[str, Any]] = Field(
        None, description="Usage information for authenticated users"
    )


class AuthenticatedQueryRequest(BaseModel):
    """Authenticated query request model with additional options."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "question": "What are the liquidity coverage requirements?",
                "document_filter": "Basel_III_Framework.pdf",
                "session_id": "xyz789-uvw012",
            }
        },
    )

    question: str = Field(
        ...,
        description="User's question to query the document database",
        min_length=1,
        max_length=30000,
        examples=["What are the liquidity coverage requirements?"],
    )
    document_filter: Optional[str] = Field(
        None,
        description="Optional filter to search within specific documents",
        examples=["Basel_III_Framework.pdf"],
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking conversation history",
        examples=["xyz789-uvw012"],
    )


class UsageResponse(BaseModel):
    """User usage statistics response model."""

    daily_queries: int = Field(..., description="Number of queries used today", ge=0)
    daily_limit: int = Field(..., description="Maximum allowed queries per day", gt=0)
    remaining: int = Field(..., description="Remaining queries for today", ge=0)
    plan: str = Field(..., description="User's subscription plan (free, premium, etc.)")
    total_queries: int = Field(
        ..., description="Total queries made by user all-time", ge=0
    )


class UploadResponse(BaseModel):
    """Document upload response model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "File uploaded successfully",
                "filename": "Basel_III_Framework.pdf",
                "chunks_created": 245,
            }
        }
    )

    message: str = Field(
        ...,
        description="Status message about the upload",
        examples=["File uploaded successfully"],
    )
    filename: str = Field(
        ...,
        description="Name of the uploaded file",
        examples=["Basel_III_Framework.pdf"],
    )
    chunks_created: int = Field(
        ...,
        description="Number of text chunks created from the document",
        ge=0,
        examples=[245],
    )


class StatusResponse(BaseModel):
    """System status response model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "service": "EFIRAS API",
                "version": "1.0.0",
                "timestamp": "2025-10-22T10:30:00",
            }
        }
    )

    status: str = Field(
        ..., description="Current system status", examples=["healthy", "running"]
    )
    service: str = Field(..., description="Service name", examples=["EFIRAS API"])
    version: str = Field(..., description="API version", examples=["1.0.0"])
    timestamp: str = Field(
        ...,
        description="Current timestamp in ISO format",
        examples=["2025-10-22T10:30:00"],
    )


class APIInfoResponse(BaseModel):
    """Root endpoint information response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "EFIRAS API is running", "version": "1.0.0"}
        }
    )

    message: str = Field(
        ..., description="Welcome message", examples=["EFIRAS API is running"]
    )
    version: str = Field(..., description="API version", examples=["1.0.0"])


class SessionClearResponse(BaseModel):
    """Session clear response model."""

    model_config = ConfigDict(json_schema_extra={"example": {"success": True}})

    success: bool = Field(
        ..., description="Whether session was cleared successfully", examples=[True]
    )


class AccountDeletionResponse(BaseModel):
    """Account deletion response model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "Account deleted successfully", "deleted": True}
        }
    )

    message: str = Field(
        ...,
        description="Deletion status message",
        examples=["Account deleted successfully"],
    )
    deleted: bool = Field(
        ..., description="Whether account was deleted", examples=[True]
    )


class UserContext(BaseModel):
    """User context from authentication token."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "auth0|123456789",
                "email": "user@example.com",
                "email_verified": True,
            }
        }
    )

    user_id: str = Field(
        ..., description="Unique user identifier", examples=["auth0|123456789"]
    )
    email: str = Field(
        ..., description="User's email address", examples=["user@example.com"]
    )
    email_verified: Optional[bool] = Field(
        None, description="Whether user's email is verified", examples=[True]
    )
