"""
Semantic Conventions for OpenTelemetry Attributes.

Centralized definitions for all span attribute names used throughout the application.
Following OpenTelemetry semantic conventions with custom extensions for role-based visibility.

Role Prefixes:
- Admin.*    : Visible to admin users (full access)
- Manager.*  : Visible to managers (operational metrics)
- Developer.*: Visible to developers (debugging info)
"""

from enum import Enum
from typing import Final


class AttributePrefix(str, Enum):
    """Role-based attribute prefixes for visibility control."""
    
    ADMIN = "Admin"
    MANAGER = "Manager"
    DEVELOPER = "Developer"


class SpanAttributes:
    """Standard span attribute names."""
    
    # Tenant & User Identity
    TENANT_ID: Final[str] = "tenant_id"
    USERNAME: Final[str] = "username"
    
    # Function Metadata
    FUNCTION_NAME: Final[str] = "function.name"
    FUNCTION_DEPTH: Final[str] = "function.depth"
    FUNCTION_IS_ROOT: Final[str] = "function.is_root"
    FUNCTION_PARENT: Final[str] = "function.parent"
    FUNCTION_DURATION: Final[str] = "function.duration"
    FUNCTION_HIERARCHY: Final[str] = "function.hierarchy"
    
    # Function Arguments & Results
    FUNCTION_ARGS: Final[str] = "function.args"
    FUNCTION_KWARGS: Final[str] = "function.kwargs"
    FUNCTION_RESULT: Final[str] = "function.result"
    FUNCTION_RESULT_LENGTH: Final[str] = "function.result_length"
    
    # Error Tracking
    ERROR: Final[str] = "error"
    ERROR_TYPE: Final[str] = "error.type"
    ERROR_MESSAGE: Final[str] = "error.message"
    
    # LLM Specific
    LLM_PROMPT: Final[str] = "llm.prompt"
    LLM_RESPONSE: Final[str] = "llm.response"
    LLM_MODEL: Final[str] = "llm.model"
    LLM_TOKENS_INPUT: Final[str] = "llm.tokens.input"
    LLM_TOKENS_OUTPUT: Final[str] = "llm.tokens.output"
    
    # RAG Specific
    RAG_QUERY: Final[str] = "rag.query"
    RAG_CONTEXT: Final[str] = "rag.context"
    RAG_RETRIEVED_DOCS: Final[str] = "rag.retrieved_docs"
    RAG_RELEVANCE_SCORE: Final[str] = "rag.relevance_score"
    
    # Quality Metrics
    METRIC_FAITHFULNESS: Final[str] = "metric.faithfulness"
    METRIC_RELEVANCY: Final[str] = "metric.relevancy"
    METRIC_HALLUCINATION: Final[str] = "metric.hallucination"
    METRIC_CONSISTENCY: Final[str] = "metric.consistency"


class RoleBasedAttributes:
    """
    Factory for creating role-prefixed attribute names.
    
    Usage:
        attrs = RoleBasedAttributes()
        attrs.admin("function.name")  # Returns "Admin.function.name"
    """
    
    @staticmethod
    def admin(attr: str) -> str:
        """Create admin-level attribute name."""
        return f"{AttributePrefix.ADMIN.value}.{attr}"
    
    @staticmethod
    def manager(attr: str) -> str:
        """Create manager-level attribute name."""
        return f"{AttributePrefix.MANAGER.value}.{attr}"
    
    @staticmethod
    def developer(attr: str) -> str:
        """Create developer-level attribute name."""
        return f"{AttributePrefix.DEVELOPER.value}.{attr}"
    
    @staticmethod
    def for_role(role: AttributePrefix, attr: str) -> str:
        """Create attribute name for a specific role."""
        return f"{role.value}.{attr}"


# Singleton instance for convenience
role_attrs = RoleBasedAttributes()


class EventNames:
    """Standard event names for span events."""
    
    REASONING_STEP: Final[str] = "Reasoning Step"
    LLM_CALL_START: Final[str] = "LLM Call Started"
    LLM_CALL_END: Final[str] = "LLM Call Completed"
    RETRIEVAL_START: Final[str] = "Retrieval Started"
    RETRIEVAL_END: Final[str] = "Retrieval Completed"
    ERROR_OCCURRED: Final[str] = "Error Occurred"


class MetricNames:
    """Standard metric names for Prometheus."""
    
    # Counters
    FUNCTION_CALLS_TOTAL: Final[str] = "function_calls_total"
    LLM_REQUESTS_TOTAL: Final[str] = "llm_requests_total"
    RAG_QUERIES_TOTAL: Final[str] = "rag_queries_total"
    ERRORS_TOTAL: Final[str] = "errors_total"
    
    # Histograms
    FUNCTION_DURATION_SECONDS: Final[str] = "function_duration_seconds"
    LLM_LATENCY_SECONDS: Final[str] = "llm_latency_seconds"
    RAG_LATENCY_SECONDS: Final[str] = "rag_latency_seconds"
    
    # Gauges
    ACTIVE_REQUESTS: Final[str] = "active_requests"


class TracerNames:
    """Standard tracer names."""
    
    CUSTOM_FUNCTIONS: Final[str] = "custom.functions"
    LLM_OPERATIONS: Final[str] = "llm.operations"
    RAG_OPERATIONS: Final[str] = "rag.operations"
    HTTP_SERVER: Final[str] = "http.server"
