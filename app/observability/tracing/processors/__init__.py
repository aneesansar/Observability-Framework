"""
Span Processors.

Custom OpenTelemetry span processors for enrichment and filtering.
"""

from .tenant_processor import TenantSpanProcessor

__all__ = ["TenantSpanProcessor"]
