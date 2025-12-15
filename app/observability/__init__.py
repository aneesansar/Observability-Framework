"""
Observability Module.

Provides comprehensive observability for the application including:
- OpenTelemetry tracing with tenant isolation
- Prometheus metrics
- Structured logging with trace correlation
- Auto-instrumentation decorators

Usage:
    from app.observability import setup_observability, observe
    
    # Initialize observability (call once at startup)
    app = FastAPI()
    setup_observability(app)
    
    # Use @observe decorator for custom instrumentation
    @observe()
    def my_function():
        pass

Architecture:
    observability/
    ├── __init__.py              # Public API
    ├── config.py                # Settings
    ├── setup.py                 # Main entry point
    ├── semantic_conventions.py  # Attribute names
    ├── context/                 # Request context management
    ├── tracing/                 # OpenTelemetry tracing
    │   ├── provider.py          # TracerProvider setup
    │   ├── processors/          # Span processors
    │   └── exporters/           # Span exporters
    ├── metrics/                 # Prometheus metrics
    │   ├── provider.py          # Metrics setup
    │   └── collectors.py        # Custom collectors
    ├── logging/                 # Logging infrastructure
    │   ├── provider.py          # Logging setup
    │   └── handlers.py          # Custom handlers
    └── instrumentation/         # Decorators & auto-instrumentation
        ├── decorators.py        # @observe decorator
        └── auto_instrument.py   # Module instrumentation
"""

# Main setup
from .setup import (
    setup_observability,
    get_observability_manager,
    ObservabilityManager,
)

# Configuration
from .config import (
    observability_settings,
    get_observability_settings,
    ObservabilitySettings,
)

# Context management
from .context import (
    RequestContextManager,
    get_current_request,
    get_current_tenant_id,
    get_current_username,
    request_context,
)

# Tracing
from .tracing import (
    setup_tracing,
    get_tracer,
    get_current_span,
    TracingProvider,
)
from .tracing.exporters import (
    get_inmemory_exporter,
    get_recent_spans,
    clear_spans,
    InMemorySpanExporter,
)
from .tracing.processors import TenantSpanProcessor

# Metrics
from .metrics import (
    setup_metrics,
    get_metrics_provider,
    MetricsProvider,
    get_function_metrics,
    get_llm_metrics,
    get_rag_metrics,
    FunctionMetrics,
    LLMMetrics,
    RAGMetrics,
)

# Logging
from .logging import (
    setup_logging,
    get_logger,
    LoggingProvider,
    get_log_buffer,
    clear_log_buffer,
    InMemoryLogHandler,
)

# Instrumentation
from .instrumentation import (
    observe,
    observe_async,
    ObserveConfig,
    observe_all_methods,
    observe_all_functions_in_module,
    observe_all_in_file,
    instrument_all_in_package,
)

# Semantic conventions
from .semantic_conventions import (
    SpanAttributes,
    RoleBasedAttributes,
    role_attrs,
    AttributePrefix,
    EventNames,
    MetricNames,
    TracerNames,
)


__all__ = [
    # Setup
    "setup_observability",
    "get_observability_manager",
    "ObservabilityManager",
    # Config
    "observability_settings",
    "get_observability_settings",
    "ObservabilitySettings",
    # Context
    "RequestContextManager",
    "get_current_request",
    "get_current_tenant_id",
    "get_current_username",
    "request_context",
    # Tracing
    "setup_tracing",
    "get_tracer",
    "get_current_span",
    "TracingProvider",
    "get_inmemory_exporter",
    "get_recent_spans",
    "clear_spans",
    "InMemorySpanExporter",
    "TenantSpanProcessor",
    # Metrics
    "setup_metrics",
    "get_metrics_provider",
    "MetricsProvider",
    "get_function_metrics",
    "get_llm_metrics",
    "get_rag_metrics",
    "FunctionMetrics",
    "LLMMetrics",
    "RAGMetrics",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggingProvider",
    "get_log_buffer",
    "clear_log_buffer",
    "InMemoryLogHandler",
    # Instrumentation
    "observe",
    "observe_async",
    "ObserveConfig",
    "observe_all_methods",
    "observe_all_functions_in_module",
    "observe_all_in_file",
    "instrument_all_in_package",
    # Semantic conventions
    "SpanAttributes",
    "RoleBasedAttributes",
    "role_attrs",
    "AttributePrefix",
    "EventNames",
    "MetricNames",
    "TracerNames",
]
