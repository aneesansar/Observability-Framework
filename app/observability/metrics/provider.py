"""
Metrics Provider Setup.

Configures Prometheus metrics with FastAPI integration.
"""

import logging
from typing import Optional

from fastapi import FastAPI

from ..config import observability_settings

logger = logging.getLogger(__name__)


class MetricsProvider:
    """
    Manages Prometheus metrics configuration.
    
    Provides centralized setup for:
    - FastAPI metrics instrumentation
    - Custom metric registration
    - Metrics endpoint exposure
    """
    
    _instance: Optional["MetricsProvider"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "MetricsProvider":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize metrics provider."""
        if MetricsProvider._initialized:
            return
        
        self._instrumentator = None
        self._enabled = observability_settings.enable_metrics
        MetricsProvider._initialized = True
    
    def setup(self, app: FastAPI) -> None:
        """
        Configure and expose metrics for FastAPI.
        
        Args:
            app: FastAPI application instance
        """
        if not self._enabled:
            logger.info("Metrics disabled via configuration")
            return
        
        try:
            from prometheus_fastapi_instrumentator import Instrumentator
            
            self._instrumentator = Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
                inprogress_name="http_requests_inprogress",
                inprogress_labels=True,
            )
            
            # Instrument and expose
            self._instrumentator.instrument(app)
            self._instrumentator.expose(
                app,
                endpoint=observability_settings.metrics_endpoint,
                include_in_schema=False,
            )
            
            logger.info(
                f"Metrics enabled at {observability_settings.metrics_endpoint}"
            )
            
        except ImportError:
            logger.warning(
                "prometheus-fastapi-instrumentator not installed. "
                "Install with: pip install prometheus-fastapi-instrumentator"
            )
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
    
    @property
    def enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self._enabled
    
    @property
    def instrumentator(self):
        """Get the Prometheus instrumentator."""
        return self._instrumentator


# Module-level singleton
_metrics_provider: Optional[MetricsProvider] = None


def setup_metrics(app: FastAPI) -> MetricsProvider:
    """
    Set up metrics for the application.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        MetricsProvider instance
    """
    global _metrics_provider
    
    _metrics_provider = MetricsProvider()
    _metrics_provider.setup(app)
    
    return _metrics_provider


def get_metrics_provider() -> Optional[MetricsProvider]:
    """
    Get the metrics provider instance.
    
    Returns:
        MetricsProvider or None if not initialized
    """
    return _metrics_provider
