"""
Custom Metric Collectors.

Provides domain-specific metrics for functions, LLM operations, and RAG pipelines.
"""

import logging
from typing import Dict, List, Optional

from prometheus_client import Counter, Histogram, Gauge, Info

from ..semantic_conventions import MetricNames

logger = logging.getLogger(__name__)


class FunctionMetrics:
    """
    Metrics for function execution tracking.
    
    Tracks:
    - Total function calls (by function name)
    - Function duration distribution
    - Error counts
    """
    
    _instance: Optional["FunctionMetrics"] = None
    
    def __new__(cls) -> "FunctionMetrics":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize function metrics."""
        if self._initialized:
            return
        
        self.calls_total = Counter(
            MetricNames.FUNCTION_CALLS_TOTAL,
            "Total number of function calls",
            ["function", "status"],
        )
        
        self.duration_seconds = Histogram(
            MetricNames.FUNCTION_DURATION_SECONDS,
            "Function execution duration in seconds",
            ["function"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        
        self.errors_total = Counter(
            MetricNames.ERRORS_TOTAL,
            "Total number of errors",
            ["function", "error_type"],
        )
        
        self._initialized = True
        logger.debug("FunctionMetrics initialized")
    
    def record_call(
        self,
        function_name: str,
        duration: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record a function call.
        
        Args:
            function_name: Name of the function
            duration: Execution duration in seconds
            success: Whether the call succeeded
            error_type: Type of error if failed
        """
        status = "success" if success else "error"
        self.calls_total.labels(function=function_name, status=status).inc()
        self.duration_seconds.labels(function=function_name).observe(duration)
        
        if not success and error_type:
            self.errors_total.labels(
                function=function_name,
                error_type=error_type,
            ).inc()


class LLMMetrics:
    """
    Metrics for LLM operations.
    
    Tracks:
    - LLM request counts (by model)
    - Latency distribution
    - Token usage
    - Quality metrics (faithfulness, relevancy)
    """
    
    _instance: Optional["LLMMetrics"] = None
    
    def __new__(cls) -> "LLMMetrics":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize LLM metrics."""
        if self._initialized:
            return
        
        self.requests_total = Counter(
            MetricNames.LLM_REQUESTS_TOTAL,
            "Total number of LLM requests",
            ["model", "status"],
        )
        
        self.latency_seconds = Histogram(
            MetricNames.LLM_LATENCY_SECONDS,
            "LLM request latency in seconds",
            ["model"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        
        self.tokens_input = Counter(
            "llm_tokens_input_total",
            "Total input tokens processed",
            ["model"],
        )
        
        self.tokens_output = Counter(
            "llm_tokens_output_total",
            "Total output tokens generated",
            ["model"],
        )
        
        self.faithfulness_score = Histogram(
            "llm_faithfulness_score",
            "Faithfulness score distribution",
            ["model"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        
        self.relevancy_score = Histogram(
            "llm_relevancy_score",
            "Answer relevancy score distribution",
            ["model"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        
        self._initialized = True
        logger.debug("LLMMetrics initialized")
    
    def record_request(
        self,
        model: str,
        latency: float,
        success: bool = True,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Record an LLM request.
        
        Args:
            model: Model name
            latency: Request latency in seconds
            success: Whether the request succeeded
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        status = "success" if success else "error"
        self.requests_total.labels(model=model, status=status).inc()
        self.latency_seconds.labels(model=model).observe(latency)
        
        if input_tokens > 0:
            self.tokens_input.labels(model=model).inc(input_tokens)
        if output_tokens > 0:
            self.tokens_output.labels(model=model).inc(output_tokens)
    
    def record_quality_metrics(
        self,
        model: str,
        faithfulness: Optional[float] = None,
        relevancy: Optional[float] = None,
    ) -> None:
        """
        Record quality metrics for an LLM response.
        
        Args:
            model: Model name
            faithfulness: Faithfulness score (0-1)
            relevancy: Relevancy score (0-1)
        """
        if faithfulness is not None:
            self.faithfulness_score.labels(model=model).observe(faithfulness)
        if relevancy is not None:
            self.relevancy_score.labels(model=model).observe(relevancy)


class RAGMetrics:
    """
    Metrics for RAG pipeline operations.
    
    Tracks:
    - Query counts
    - Retrieval latency
    - Document counts
    - Relevance scores
    """
    
    _instance: Optional["RAGMetrics"] = None
    
    def __new__(cls) -> "RAGMetrics":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize RAG metrics."""
        if self._initialized:
            return
        
        self.queries_total = Counter(
            MetricNames.RAG_QUERIES_TOTAL,
            "Total number of RAG queries",
            ["status"],
        )
        
        self.latency_seconds = Histogram(
            MetricNames.RAG_LATENCY_SECONDS,
            "RAG query latency in seconds",
            [],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        
        self.retrieved_docs = Histogram(
            "rag_retrieved_documents",
            "Number of documents retrieved per query",
            [],
            buckets=(1, 2, 3, 5, 10, 20, 50),
        )
        
        self.relevance_score = Histogram(
            "rag_relevance_score",
            "Retrieval relevance score distribution",
            [],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        
        self._initialized = True
        logger.debug("RAGMetrics initialized")
    
    def record_query(
        self,
        latency: float,
        success: bool = True,
        num_docs: int = 0,
        relevance: Optional[float] = None,
    ) -> None:
        """
        Record a RAG query.
        
        Args:
            latency: Query latency in seconds
            success: Whether the query succeeded
            num_docs: Number of documents retrieved
            relevance: Average relevance score
        """
        status = "success" if success else "error"
        self.queries_total.labels(status=status).inc()
        self.latency_seconds.observe(latency)
        
        if num_docs > 0:
            self.retrieved_docs.observe(num_docs)
        if relevance is not None:
            self.relevance_score.observe(relevance)


# Singleton accessors

def get_function_metrics() -> FunctionMetrics:
    """Get the function metrics instance."""
    return FunctionMetrics()


def get_llm_metrics() -> LLMMetrics:
    """Get the LLM metrics instance."""
    return LLMMetrics()


def get_rag_metrics() -> RAGMetrics:
    """Get the RAG metrics instance."""
    return RAGMetrics()
