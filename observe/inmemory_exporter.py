from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from threading import Lock


class CustomInMemorySpanExporter(SpanExporter):
    """
    A thread-safe custom in-memory span exporter.
    Stores all finished spans for use in API endpoints like /traces or /llm-metrics.
    """

    def __init__(self):
        self._spans = []
        self._lock = Lock()

    def export(self, spans):
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        with self._lock:
            return list(self._spans)

    def clear(self):
        with self._lock:
            self._spans.clear()


# ✅ Create a single global exporter instance
in_memory_exporter = CustomInMemorySpanExporter()
