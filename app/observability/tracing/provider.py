"""
Tracing Provider Setup.

Configures OpenTelemetry TracerProvider with appropriate processors and exporters.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Tracer, get_current_span as _get_current_span

from ..config import observability_settings
from .exporters.inmemory import InMemorySpanExporter, get_inmemory_exporter
from .exporters.mongodb import MongoDBSpanExporter, get_mongodb_exporter
from .processors.tenant_processor import TenantSpanProcessor

logger = logging.getLogger(__name__)


class TracingProvider:
    """
    Manages OpenTelemetry tracing configuration.
    
    Provides a centralized setup for TracerProvider with:
    - Service resource configuration
    - Span processors (tenant enrichment, export)
    - Multiple exporter support (in-memory, OTLP)
    """
    
    _instance: Optional["TracingProvider"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "TracingProvider":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize tracing provider (only runs once due to singleton)."""
        if TracingProvider._initialized:
            return
        
        self._provider: Optional[TracerProvider] = None
        self._inmemory_exporter: Optional[InMemorySpanExporter] = None
        self._mongodb_exporter: Optional[MongoDBSpanExporter] = None
        TracingProvider._initialized = True
    
    def setup(
        self,
        service_name: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        enable_inmemory: bool = True,
        enable_mongodb: bool = True,
    ) -> TracerProvider:
        """
        Configure and set up the TracerProvider.
        
        Args:
            service_name: Override service name from settings
            otlp_endpoint: OTLP exporter endpoint (optional)
            enable_inmemory: Enable in-memory span storage
            enable_mongodb: Enable MongoDB span persistence
            
        Returns:
            Configured TracerProvider
        """
        service_name = service_name or observability_settings.service_name
        
        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: service_name})
        
        # Create provider
        self._provider = TracerProvider(resource=resource)
        
        # Add tenant enrichment processor first
        self._provider.add_span_processor(TenantSpanProcessor())
        logger.info("TenantSpanProcessor registered")
        
        # Add in-memory exporter for API access
        if enable_inmemory:
            self._inmemory_exporter = get_inmemory_exporter()
            self._provider.add_span_processor(
                SimpleSpanProcessor(self._inmemory_exporter)
            )
            logger.info("InMemorySpanExporter registered")
        
        # Add MongoDB exporter for persistent storage
        if enable_mongodb:
            self._setup_mongodb_exporter()
        
        # Add OTLP exporter if configured
        otlp_endpoint = otlp_endpoint or observability_settings.otlp_endpoint
        if otlp_endpoint:
            self._setup_otlp_exporter(otlp_endpoint)
        
        # Set as global provider
        trace.set_tracer_provider(self._provider)
        logger.info(f"TracerProvider configured for service: {service_name}")
        
        return self._provider
    
    def _setup_mongodb_exporter(self) -> None:
        """Configure MongoDB exporter for persistent trace storage."""
        try:
            self._mongodb_exporter = get_mongodb_exporter()
            self._provider.add_span_processor(
                BatchSpanProcessor(self._mongodb_exporter)
            )
            logger.info("MongoDBSpanExporter registered")
        except Exception as e:
            logger.warning(f"Failed to configure MongoDB exporter: {e}")
    
    def _setup_otlp_exporter(self, endpoint: str) -> None:
        """
        Configure OTLP exporter.
        
        Args:
            endpoint: OTLP endpoint URL
        """
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            
            exporter = OTLPSpanExporter(endpoint=endpoint)
            self._provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"OTLP exporter configured: {endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not available. Install opentelemetry-exporter-otlp")
        except Exception as e:
            logger.error(f"Failed to configure OTLP exporter: {e}")
    
    def add_exporter(
        self,
        exporter: SpanExporter,
        use_batch: bool = True,
    ) -> None:
        """
        Add a custom span exporter.
        
        Args:
            exporter: SpanExporter instance
            use_batch: Use BatchSpanProcessor (True) or SimpleSpanProcessor (False)
        """
        if not self._provider:
            raise RuntimeError("TracingProvider not initialized. Call setup() first.")
        
        processor = (
            BatchSpanProcessor(exporter) if use_batch
            else SimpleSpanProcessor(exporter)
        )
        self._provider.add_span_processor(processor)
    
    @property
    def provider(self) -> Optional[TracerProvider]:
        """Get the TracerProvider instance."""
        return self._provider
    
    @property
    def inmemory_exporter(self) -> Optional[InMemorySpanExporter]:
        """Get the in-memory exporter for span access."""
        return self._inmemory_exporter
    
    def get_tracer(self, name: str) -> Tracer:
        """
        Get a tracer instance.
        
        Args:
            name: Tracer name (usually module name)
            
        Returns:
            Tracer instance
        """
        return trace.get_tracer(name)


# Module-level convenience functions

_tracing_provider: Optional[TracingProvider] = None


def setup_tracing(
    app: Optional[FastAPI] = None,
    service_name: Optional[str] = None,
) -> TracerProvider:
    """
    Set up tracing for the application.
    
    Args:
        app: FastAPI application to instrument (optional)
        service_name: Override service name
        
    Returns:
        Configured TracerProvider
    """
    global _tracing_provider
    
    _tracing_provider = TracingProvider()
    provider = _tracing_provider.setup(service_name=service_name)
    
    # Instrument FastAPI if provided
    if app:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument_app(app)
            logger.info("FastAPI instrumented for tracing")
        except ImportError:
            logger.warning("FastAPI instrumentor not available")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    return provider


def get_tracer(name: str) -> Tracer:
    """
    Get a tracer instance.
    
    Args:
        name: Tracer name
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_current_span() -> Span:
    """
    Get the current active span.
    
    Returns:
        Current Span instance
    """
    return _get_current_span()
