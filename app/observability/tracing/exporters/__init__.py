"""
Span Exporters.

Custom OpenTelemetry span exporters for various backends.
"""

from .inmemory import (
    InMemorySpanExporter,
    get_inmemory_exporter,
    get_recent_spans,
    clear_spans,
)

__all__ = [
    "InMemorySpanExporter",
    "get_inmemory_exporter",
    "get_recent_spans",
    "clear_spans",
]
