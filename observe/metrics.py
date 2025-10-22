from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
from .config import settings

instrumentator = Instrumentator()

def setup_metrics(app: FastAPI):
    if settings.ENABLE_METRICS:
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
