import logging
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from .config import settings

def setup_logging():
    LoggingInstrumentor().instrument(set_logging_format=True)
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), "INFO"))
