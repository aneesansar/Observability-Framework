"""
Request Schemas.

Pydantic models for API request bodies.
"""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for RAG query endpoints."""
    
    query: str = Field(..., description="The query text")
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return"
    )


class LoginRequest(BaseModel):
    """Request body for login endpoint."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class ReindexRequest(BaseModel):
    """Request body for reindex endpoint."""
    
    documents_dir: Optional[str] = Field(
        default=None,
        description="Custom documents directory path"
    )
