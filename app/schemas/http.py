from pydantic import BaseModel
from typing import Optional, Dict


class QueryRequest(BaseModel):
    """Request body for text query endpoints."""
    query: str


class LogEntry(BaseModel):
    """Structured log entry stored in the in-memory buffer for live log views."""
    id: str
    timestamp: str
    level: str
    service: str
    tenant_id: str
    message: str
    metadata: Optional[Dict] = None
