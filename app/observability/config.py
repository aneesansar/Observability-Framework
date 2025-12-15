"""
Observability Configuration.

Centralized settings for OpenTelemetry, metrics, and logging configuration.
Uses Pydantic for validation and environment variable loading.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ObservabilitySettings(BaseSettings):
    """
    Configuration settings for the observability stack.
    
    All settings can be overridden via environment variables.
    """
    
    # Service Identity
    service_name: str = Field(
        default="Speedy-Observability",
        alias="SERVICE_NAME",
        description="Name of the service for tracing and metrics"
    )
    
    # OTLP Exporter
    otlp_endpoint: Optional[str] = Field(
        default=None,
        alias="OTLP_ENDPOINT",
        description="OTLP exporter endpoint (e.g., http://localhost:4318)"
    )
    otlp_headers: Optional[str] = Field(
        default=None,
        alias="OTLP_HEADERS",
        description="OTLP headers as comma-separated key=value pairs"
    )
    
    # Metrics
    enable_metrics: bool = Field(
        default=True,
        alias="ENABLE_METRICS",
        description="Enable Prometheus metrics endpoint"
    )
    metrics_endpoint: str = Field(
        default="/metrics",
        alias="METRICS_ENDPOINT",
        description="Path for metrics endpoint"
    )
    metrics_port: Optional[int] = Field(
        default=None,
        alias="METRICS_PORT",
        description="Separate port for metrics (if not using main app)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="json",
        alias="LOG_FORMAT",
        description="Log format: 'json' or 'text'"
    )
    
    # Tracing
    trace_debug: bool = Field(
        default=False,
        alias="TRACE_DEBUG",
        description="Enable trace debugging output"
    )
    trace_sample_rate: float = Field(
        default=1.0,
        alias="TRACE_SAMPLE_RATE",
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 to 1.0)"
    )
    
    # In-Memory Storage
    max_stored_spans: int = Field(
        default=10000,
        alias="MAX_STORED_SPANS",
        description="Maximum number of spans to store in memory"
    )
    max_log_buffer: int = Field(
        default=1000,
        alias="MAX_LOG_BUFFER",
        description="Maximum number of log entries to buffer"
    )
    
    # Instrumentation Control
    observe_exclude_patterns: List[str] = Field(
        default_factory=list,
        alias="OBSERVE_EXCLUDE",
        description="Comma-separated regex patterns for modules to exclude from auto-instrumentation"
    )
    skip_paths: List[str] = Field(
        default_factory=lambda: ["/metrics", "/health", "/docs", "/openapi.json", "/observability"],
        alias="OBSERVE_SKIP_PATHS",
        description="URL paths to skip from tracing"
    )
    
    @field_validator("observe_exclude_patterns", mode="before")
    @classmethod
    def parse_exclude_patterns(cls, v):
        """Parse comma-separated patterns into a list."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v or []
    
    @field_validator("skip_paths", mode="before")
    @classmethod
    def parse_skip_paths(cls, v):
        """Parse comma-separated paths into a list."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v or ["/metrics", "/health", "/docs", "/openapi.json", "/observability"]
    
    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is uppercase and valid."""
        if v:
            v = v.upper()
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v not in valid_levels:
                raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_observability_settings() -> ObservabilitySettings:
    """
    Get cached observability settings instance.
    
    Returns:
        ObservabilitySettings: Singleton settings instance
    """
    return ObservabilitySettings()


# Convenience alias
observability_settings = get_observability_settings()
