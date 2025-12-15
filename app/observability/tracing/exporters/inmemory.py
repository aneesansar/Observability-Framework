"""
In-Memory Span Exporter.

Thread-safe span storage for API access and debugging.
Provides access to finished spans for observability endpoints.
"""

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ...config import observability_settings

logger = logging.getLogger(__name__)


@dataclass
class SpanData:
    """Simplified span data for API responses."""
    
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: int
    end_time: int
    attributes: Dict[str, Any]
    status: str
    events: List[Dict[str, Any]]


class InMemorySpanExporter(SpanExporter):
    """
    Thread-safe in-memory span exporter.
    
    Stores finished spans for use in API endpoints like /traces or /llm-metrics.
    Implements automatic cleanup when max capacity is reached.
    
    Attributes:
        max_spans: Maximum number of spans to store
    """
    
    def __init__(self, max_spans: Optional[int] = None):
        """
        Initialize the exporter.
        
        Args:
            max_spans: Maximum spans to store (defaults to settings)
        """
        self._spans: List[ReadableSpan] = []
        self._lock = Lock()
        self._max_spans = max_spans or observability_settings.max_stored_spans
        logger.info(f"InMemorySpanExporter initialized (max_spans={self._max_spans})")
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to in-memory storage.
        
        Args:
            spans: Sequence of spans to export
            
        Returns:
            SpanExportResult.SUCCESS
        """
        with self._lock:
            self._spans.extend(spans)
            
            # Trim if over capacity
            if len(self._spans) > self._max_spans:
                excess = len(self._spans) - self._max_spans
                self._spans = self._spans[excess:]
                logger.debug(f"Trimmed {excess} old spans from storage")
        
        if observability_settings.trace_debug:
            logger.debug(
                f"Exported {len(spans)} spans, total stored: {len(self._spans)}"
            )
        
        return SpanExportResult.SUCCESS
    
    def get_finished_spans(self) -> List[ReadableSpan]:
        """
        Get all stored spans.
        
        Returns:
            Copy of stored spans list
        """
        with self._lock:
            return list(self._spans)
    
    def get_spans_by_tenant(self, tenant_id: str) -> List[ReadableSpan]:
        """
        Get spans filtered by tenant_id.
        
        Args:
            tenant_id: Tenant identifier to filter by
            
        Returns:
            List of spans for the specified tenant
        """
        with self._lock:
            return [
                span for span in self._spans
                if span.attributes.get("tenant_id") == tenant_id
            ]
    
    def get_recent_spans(
        self,
        limit: int = 10,
        tenant_id: Optional[str] = None,
    ) -> List[SpanData]:
        """
        Get recent spans as simplified data objects.
        
        Args:
            limit: Maximum number of spans to return
            tenant_id: Optional tenant filter
            
        Returns:
            List of SpanData objects
        """
        with self._lock:
            spans = self._spans
            
            # Filter by tenant if specified
            if tenant_id:
                spans = [
                    s for s in spans
                    if s.attributes.get("tenant_id") == tenant_id
                ]
            
            # Get last N spans
            recent = spans[-limit:] if limit else spans
            
            result = []
            for span in recent:
                try:
                    result.append(self._span_to_data(span))
                except Exception as e:
                    logger.debug(f"Failed to convert span: {e}")
                    continue
            
            return result
    
    def _span_to_data(self, span: ReadableSpan) -> SpanData:
        """
        Convert a ReadableSpan to SpanData.
        
        Args:
            span: OpenTelemetry span
            
        Returns:
            SpanData object
        """
        # Extract events
        events = []
        if hasattr(span, "events") and span.events:
            for event in span.events:
                events.append({
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {},
                })
        
        # Get parent span ID
        parent_span_id = None
        if span.parent and hasattr(span.parent, "span_id"):
            parent_span_id = format(span.parent.span_id, "016x")
        
        return SpanData(
            name=span.name,
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            parent_span_id=parent_span_id,
            start_time=span.start_time,
            end_time=span.end_time,
            attributes=dict(span.attributes) if span.attributes else {},
            status=span.status.status_code.name if span.status else "UNSET",
            events=events,
        )
    
    def clear(self) -> int:
        """
        Clear all stored spans.
        
        Returns:
            Number of spans cleared
        """
        with self._lock:
            count = len(self._spans)
            self._spans.clear()
            logger.info(f"Cleared {count} spans from storage")
            return count
    
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self.clear()
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush (no-op for in-memory).
        
        Args:
            timeout_millis: Timeout in milliseconds
            
        Returns:
            True
        """
        return True
    
    @property
    def span_count(self) -> int:
        """Get current number of stored spans."""
        with self._lock:
            return len(self._spans)


# Singleton instance
_inmemory_exporter: Optional[InMemorySpanExporter] = None


def get_inmemory_exporter() -> InMemorySpanExporter:
    """
    Get the singleton in-memory exporter instance.
    
    Returns:
        InMemorySpanExporter instance
    """
    global _inmemory_exporter
    if _inmemory_exporter is None:
        _inmemory_exporter = InMemorySpanExporter()
    return _inmemory_exporter


def get_recent_spans(
    limit: int = 10,
    tenant_id: Optional[str] = None,
) -> List[SpanData]:
    """
    Convenience function to get recent spans.
    
    Args:
        limit: Maximum number of spans
        tenant_id: Optional tenant filter
        
    Returns:
        List of SpanData objects
    """
    exporter = get_inmemory_exporter()
    return exporter.get_recent_spans(limit=limit, tenant_id=tenant_id)


def clear_spans() -> int:
    """
    Clear all stored spans.
    
    Returns:
        Number of spans cleared
    """
    exporter = get_inmemory_exporter()
    return exporter.clear()
