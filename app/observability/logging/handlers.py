"""
Custom Log Handlers.

Provides specialized log handlers for observability features.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from ..config import observability_settings
from ..context import get_current_tenant_id, get_current_username

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry for API access."""
    
    id: str
    timestamp: str
    level: str
    service: str
    message: str
    tenant_id: str = "anonymous"
    username: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level,
            "service": self.service,
            "message": self.message,
            "tenant_id": self.tenant_id,
            "username": self.username,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata,
        }


class InMemoryLogHandler(logging.Handler):
    """
    Custom log handler that stores logs in memory with tenant/user metadata.
    
    Features:
    - Thread-safe log storage
    - Automatic tenant/user extraction from context
    - Trace ID correlation
    - Configurable buffer size
    """
    
    _instance: Optional["InMemoryLogHandler"] = None
    
    def __new__(cls, *args, **kwargs) -> "InMemoryLogHandler":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._buffer_initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: Optional[str] = None,
        max_entries: Optional[int] = None,
    ):
        """
        Initialize the handler.
        
        Args:
            service_name: Service name for log entries
            max_entries: Maximum log entries to store
        """
        if self._buffer_initialized:
            return
        
        super().__init__()
        
        self._buffer: List[LogEntry] = []
        self._lock = Lock()
        self._service_name = service_name or observability_settings.service_name
        self._max_entries = max_entries or observability_settings.max_log_buffer
        self._buffer_initialized = True
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the in-memory buffer.
        
        Args:
            record: Log record to store
        """
        try:
            # Extract tenant/user from context or record
            tenant_id = (
                getattr(record, "tenant_id", None)
                or get_current_tenant_id()
                or "anonymous"
            )
            username = (
                getattr(record, "username", None)
                or get_current_username()
            )
            
            # Extract trace correlation
            trace_id = getattr(record, "otelTraceID", None)
            span_id = getattr(record, "otelSpanID", None)
            
            # Build metadata
            metadata = {
                "filename": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
                "logger": record.name,
            }
            
            # Add exception info if present
            if record.exc_info:
                metadata["exception"] = self.format(record)
            
            # Create log entry
            entry = LogEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat() + "Z",
                level=record.levelname,
                service=self._service_name,
                message=record.getMessage(),
                tenant_id=str(tenant_id),
                username=str(username) if username else None,
                trace_id=trace_id,
                span_id=span_id,
                metadata=metadata,
            )
            
            # Add to buffer with size management
            with self._lock:
                self._buffer.append(entry)
                
                # Trim if over capacity
                if len(self._buffer) > self._max_entries:
                    excess = len(self._buffer) - self._max_entries
                    self._buffer = self._buffer[excess:]
                    
        except Exception:
            # Never fail logging due to handler errors
            self.handleError(record)
    
    def get_logs(
        self,
        level: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LogEntry]:
        """
        Get logs from the buffer.
        
        Args:
            level: Filter by log level
            tenant_id: Filter by tenant ID
            limit: Maximum entries to return
            
        Returns:
            List of LogEntry objects
        """
        with self._lock:
            logs = list(self._buffer)
        
        # Apply filters
        if level:
            logs = [log for log in logs if log.level.upper() == level.upper()]
        
        if tenant_id:
            logs = [log for log in logs if log.tenant_id == tenant_id]
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            logs = logs[:limit]
        
        return logs
    
    def clear(self) -> int:
        """
        Clear all logs from the buffer.
        
        Returns:
            Number of logs cleared
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count
    
    @property
    def count(self) -> int:
        """Get current number of stored logs."""
        with self._lock:
            return len(self._buffer)


# Module-level buffer access
_log_handler: Optional[InMemoryLogHandler] = None


def get_inmemory_handler() -> InMemoryLogHandler:
    """
    Get the singleton in-memory log handler.
    
    Returns:
        InMemoryLogHandler instance
    """
    global _log_handler
    if _log_handler is None:
        _log_handler = InMemoryLogHandler()
    return _log_handler


def get_log_buffer(
    level: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[LogEntry]:
    """
    Get logs from the in-memory buffer.
    
    Args:
        level: Filter by log level
        tenant_id: Filter by tenant ID
        limit: Maximum entries to return
        
    Returns:
        List of LogEntry objects
    """
    handler = get_inmemory_handler()
    return handler.get_logs(level=level, tenant_id=tenant_id, limit=limit)


def clear_log_buffer() -> int:
    """
    Clear all logs from the buffer.
    
    Returns:
        Number of logs cleared
    """
    handler = get_inmemory_handler()
    return handler.clear()
