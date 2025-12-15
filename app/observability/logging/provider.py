"""
Logging Provider Setup.

Configures Python logging with OpenTelemetry instrumentation.
"""

import logging
import os
from typing import Optional

from ..config import observability_settings

logger = logging.getLogger(__name__)


class LoggingProvider:
    """
    Manages logging configuration with OpenTelemetry integration.
    
    Provides:
    - OpenTelemetry logging instrumentation
    - Trace ID correlation in logs
    - Configurable log levels and formats
    """
    
    _instance: Optional["LoggingProvider"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "LoggingProvider":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging provider."""
        if LoggingProvider._initialized:
            return
        
        self._root_logger: Optional[logging.Logger] = None
        LoggingProvider._initialized = True
    
    def setup(self) -> None:
        """
        Configure logging with OpenTelemetry instrumentation.
        
        Sets up:
        - OTEL resource attributes for service name
        - Logging instrumentation for trace correlation
        - Root logger configuration
        """
        # Set OTEL resource attributes for logging
        self._configure_otel_resources()
        
        # Instrument logging
        self._instrument_logging()
        
        # Configure root logger
        self._configure_root_logger()
        
        logger.info(
            f"Logging configured with level={observability_settings.log_level}"
        )
    
    def _configure_otel_resources(self) -> None:
        """Set OpenTelemetry resource attributes for logging."""
        if not os.getenv("OTEL_RESOURCE_ATTRIBUTES"):
            service_name = observability_settings.service_name
            os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"service.name={service_name}"
    
    def _instrument_logging(self) -> None:
        """Apply OpenTelemetry logging instrumentation."""
        try:
            from opentelemetry.instrumentation.logging import LoggingInstrumentor
            
            LoggingInstrumentor().instrument(set_logging_format=True)
            logger.debug("OpenTelemetry logging instrumentation applied")
            
        except ImportError:
            logger.warning(
                "opentelemetry-instrumentation-logging not installed. "
                "Log correlation will not be available."
            )
        except Exception as e:
            logger.error(f"Failed to instrument logging: {e}")
    
    def _configure_root_logger(self) -> None:
        """Configure the root logger."""
        level_name = observability_settings.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        
        # Configure basic logging
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured Logger instance
        """
        return logging.getLogger(name)


# Module-level singleton
_logging_provider: Optional[LoggingProvider] = None


def setup_logging() -> LoggingProvider:
    """
    Set up logging for the application.
    
    Returns:
        LoggingProvider instance
    """
    global _logging_provider
    
    _logging_provider = LoggingProvider()
    _logging_provider.setup()
    
    return _logging_provider


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
