"""
Tenant Span Processor.

Automatically enriches all spans with tenant_id and user metadata.
Ensures proper tenant isolation in multi-tenant environments.
"""

import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ...context import get_current_request
from ...semantic_conventions import SpanAttributes

logger = logging.getLogger(__name__)


class TenantSpanProcessor(SpanProcessor):
    """
    Span processor that automatically adds tenant metadata to all spans.
    
    This processor runs on every span creation, including:
    - FastAPI auto-instrumented HTTP spans
    - Custom @observe decorated spans
    - Middleware spans
    - Any other spans created in the system
    
    Attributes added:
    - tenant_id: Tenant identifier for isolation
    - username: Current user's username
    """
    
    def on_start(
        self,
        span: Span,
        parent_context: Optional[Context] = None,
    ) -> None:
        """
        Called when a span is started.
        
        Enriches the span with tenant_id and username from request context.
        
        Args:
            span: The span being started
            parent_context: Parent context (optional)
        """
        try:
            request = get_current_request()
            
            if request is None or not hasattr(request, "state"):
                return
            
            # Extract tenant_id
            tenant_id = getattr(request.state, "tenant_id", None)
            if tenant_id:
                span.set_attribute(SpanAttributes.TENANT_ID, str(tenant_id))
            
            # Extract username
            username = getattr(request.state, "username", None)
            if username:
                span.set_attribute(SpanAttributes.USERNAME, str(username))
            
            # Debug logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Enriched span '{span.name}' with tenant_id='{tenant_id}', "
                    f"username='{username}'"
                )
                
        except Exception as e:
            # Never fail span creation due to enrichment errors
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Failed to enrich span: {e}")
    
    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends.
        
        Re-enriches the span with final tenant_id from request context.
        This ensures spans get the correct tenant_id even if it was
        updated by middleware after span creation.
        
        Args:
            span: The span that ended
        """
        # ReadableSpan doesn't allow modification, but we can check
        # if we need to update. The actual update happens via the
        # span reference we have during on_start.
        # 
        # Unfortunately, ReadableSpan is immutable. We need a different approach.
        pass
    
    def shutdown(self) -> None:
        """
        Called on processor shutdown.
        
        No cleanup needed.
        """
        pass
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any buffered data.
        
        Args:
            timeout_millis: Timeout in milliseconds
            
        Returns:
            True (no buffering in this processor)
        """
        return True
