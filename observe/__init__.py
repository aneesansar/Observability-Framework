from .tracing import setup_tracing
from .metrics import setup_metrics
from .logging import setup_logging

def setup_observability(app=None):
    setup_tracing(app)
    setup_metrics(app)
    setup_logging()
