"""
Response Schemas.

Pydantic models for API response bodies.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Response from RAG query endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    tenant_id: str = Field(..., description="Tenant identifier")
    query: str = Field(..., description="Original query")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")


class TokenResponse(BaseModel):
    """Authentication token response."""
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(
        default=None,
        description="JWT refresh token"
    )
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(default=3600, description="Token expiry in seconds")


class UserInfoResponse(BaseModel):
    """User information response."""
    
    username: str
    tenant_id: str
    roles: List[str]
    role_group: str
    email: Optional[str] = None
    full_name: Optional[str] = None


class ReindexResponse(BaseModel):
    """Response from reindex operation."""
    
    documents: int = Field(..., description="Number of documents indexed")
    dim: int = Field(..., description="Embedding dimension")


class TraceResponse(BaseModel):
    """Trace/span response."""
    
    name: str
    trace_id: str
    span_id: str
    start_time: int
    end_time: int
    status: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]


class LogEntryResponse(BaseModel):
    """Log entry response."""
    
    id: str
    timestamp: str
    level: str
    service: str
    message: str
    tenant_id: str
    username: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
