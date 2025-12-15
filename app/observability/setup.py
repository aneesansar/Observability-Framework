"""
Observability Setup.

Single entry point for initializing all observability components.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI

from .config import observability_settings
from .tracing import setup_tracing, TracingProvider
from .metrics import setup_metrics, MetricsProvider
from .logging import setup_logging, LoggingProvider, get_inmemory_handler
from .instrumentation import instrument_all_in_package

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """
    Centralized manager for all observability components.
    
    Provides a single entry point for initializing:
    - Tracing (OpenTelemetry)
    - Metrics (Prometheus)
    - Logging (with OTEL correlation)
    - Auto-instrumentation
    
    Example:
        manager = ObservabilityManager()
        manager.setup(app)
        
        # Or use the convenience function:
        setup_observability(app)
    """
    
    _instance: Optional["ObservabilityManager"] = None
    
    def __new__(cls) -> "ObservabilityManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the manager."""
        if self._initialized:
            return
        
        self._tracing_provider: Optional[TracingProvider] = None
        self._metrics_provider: Optional[MetricsProvider] = None
        self._logging_provider: Optional[LoggingProvider] = None
        self._app: Optional[FastAPI] = None
        self._initialized = True
    
    def setup(
        self,
        app: Optional[FastAPI] = None,
        service_name: Optional[str] = None,
        auto_instrument: bool = True,
        instrument_path: Optional[str] = None,
        instrument_root: str = "app",
    ) -> "ObservabilityManager":
        """
        Initialize all observability components.
        
        Args:
            app: FastAPI application to instrument
            service_name: Override service name from settings
            auto_instrument: Enable auto-instrumentation of modules
            instrument_path: Path to instrument (defaults to app directory)
            instrument_root: Import root for instrumentation
            
        Returns:
            Self for chaining
        """
        self._app = app
        service_name = service_name or observability_settings.service_name
        
        # Set OTEL resource attributes early
        self._configure_otel_resources(service_name)
        
        # Setup logging first (for subsequent log output)
        self._logging_provider = setup_logging()
        
        # Setup tracing
        self._tracing_provider = TracingProvider()
        self._tracing_provider.setup(service_name=service_name)
        
        # Instrument FastAPI if provided
        if app:
            self._instrument_fastapi(app)
            
            # Setup metrics
            self._metrics_provider = setup_metrics(app)
            
            # Add in-memory log handler to root logger
            self._setup_log_handler()
        
        # Auto-instrument modules
        if auto_instrument:
            self._auto_instrument(instrument_path, instrument_root)
        
        logger.info(
            f"Observability initialized for service: {service_name}"
        )
        
        return self
    
    def _configure_otel_resources(self, service_name: str) -> None:
        """Set OTEL resource attributes."""
        if not os.getenv("OTEL_RESOURCE_ATTRIBUTES"):
            os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"service.name={service_name}"
    
    def _instrument_fastapi(self, app: FastAPI) -> None:
        """Instrument FastAPI with OpenTelemetry."""
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument_app(app)
            logger.info("FastAPI instrumented for tracing")
        except ImportError:
            logger.warning("FastAPI instrumentor not available")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def _setup_log_handler(self) -> None:
        """Add in-memory log handler to root logger."""
        handler = get_inmemory_handler()
        root_logger = logging.getLogger()
        
        # Avoid duplicate handlers
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)
            logger.debug("In-memory log handler added to root logger")
    
    def _auto_instrument(
        self,
        instrument_path: Optional[str],
        instrument_root: str,
    ) -> None:
        """Auto-instrument modules in the package."""
        if instrument_path is None:
            # Default to app directory
            instrument_path = os.path.dirname(os.path.dirname(__file__))
        
        try:
            instrument_all_in_package(instrument_path, instrument_root)
            logger.info(f"Auto-instrumented modules in: {instrument_root}")
        except Exception as e:
            logger.warning(f"Auto-instrumentation failed: {e}")
    
    @property
    def tracing(self) -> Optional[TracingProvider]:
        """Get the tracing provider."""
        return self._tracing_provider
    
    @property
    def metrics(self) -> Optional[MetricsProvider]:
        """Get the metrics provider."""
        return self._metrics_provider
    
    @property
    def logging(self) -> Optional[LoggingProvider]:
        """Get the logging provider."""
        return self._logging_provider


# Module-level convenience function

_manager: Optional[ObservabilityManager] = None


def setup_observability(
    app: Optional[FastAPI] = None,
    service_name: Optional[str] = None,
    auto_instrument: bool = True,
) -> ObservabilityManager:
    """
    Initialize all observability components.
    
    This is the main entry point for setting up observability.
    Call this once during application startup.
    
    Args:
        app: FastAPI application to instrument
        service_name: Override service name from settings
        auto_instrument: Enable auto-instrumentation
        
    Returns:
        ObservabilityManager instance
        
    Example:
        from app.observability import setup_observability
        
        app = FastAPI()
        setup_observability(app)
    """
    global _manager
    
    _manager = ObservabilityManager()
    _manager.setup(
        app=app,
        service_name=service_name,
        auto_instrument=auto_instrument,
    )
    
    return _manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """
    Get the observability manager instance.
    
    Returns:
        ObservabilityManager or None if not initialized
    """
    return _manager
