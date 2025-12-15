"""
Logging Module.

Provides structured logging with OpenTelemetry correlation.
"""

from .provider import (
    LoggingProvider,
    setup_logging,
    get_logger,
)
from .handlers import (
    InMemoryLogHandler,
    get_log_buffer,
    clear_log_buffer,
    get_inmemory_handler,
)

__all__ = [
    "LoggingProvider",
    "setup_logging",
    "get_logger",
    "InMemoryLogHandler",
    "get_log_buffer",
    "clear_log_buffer",
    "get_inmemory_handler",
]
