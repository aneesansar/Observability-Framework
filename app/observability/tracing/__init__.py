"""
Tracing Module.

Provides OpenTelemetry tracing setup, span processors, and exporters.
"""

from .provider import (
    TracingProvider,
    get_tracer,
    get_current_span,
    setup_tracing,
)

__all__ = [
    "TracingProvider",
    "get_tracer",
    "get_current_span",
    "setup_tracing",
]
