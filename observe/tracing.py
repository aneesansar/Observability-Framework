from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import get_current_span as _get_current_span

from .inmemory_exporter import in_memory_exporter
from .config import settings


def setup_tracing(app=None):
    """Configure tracing for FastAPI with OTLP + In-Memory exporters."""
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: settings.SERVICE_NAME}))
    trace.set_tracer_provider(provider)

    # Exporter 1: Remote (OTLP Collector / Jaeger / Tempo)
#    otlp_exporter = OTLPSpanExporter(endpoint=settings.OTLP_ENDPOINT)
 #   provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Exporter 2: Local In-Memory Exporter for Debug / API access
    provider.add_span_processor(BatchSpanProcessor(in_memory_exporter))

    if app:
        FastAPIInstrumentor().instrument_app(app)

    print(f"[Tracing Initialized] Service: {settings.SERVICE_NAME} | OTLP: {settings.OTLP_ENDPOINT}")


def get_current_span():
    """Helper to get the current active OpenTelemetry span."""
    return _get_current_span()
