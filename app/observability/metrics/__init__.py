"""
Metrics Module.

Provides Prometheus metrics setup and custom collectors.
"""

from .provider import (
    MetricsProvider,
    setup_metrics,
    get_metrics_provider,
)
from .collectors import (
    FunctionMetrics,
    LLMMetrics,
    RAGMetrics,
    get_function_metrics,
    get_llm_metrics,
    get_rag_metrics,
)

__all__ = [
    "MetricsProvider",
    "setup_metrics",
    "get_metrics_provider",
    "FunctionMetrics",
    "LLMMetrics",
    "RAGMetrics",
    "get_function_metrics",
    "get_llm_metrics",
    "get_rag_metrics",
]
