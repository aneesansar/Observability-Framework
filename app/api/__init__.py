"""
API Module.

FastAPI routes and controllers organized by version.

Structure:
    api/
    ├── __init__.py
    ├── v1/
    │   ├── __init__.py
    │   ├── router.py           # Main v1 router
    │   ├── endpoints/
    │   │   ├── auth.py         # Authentication endpoints
    │   │   ├── rag.py          # RAG query endpoints
    │   │   ├── observability.py # Telemetry endpoints
    │   │   └── evaluation.py   # Evaluation endpoints
    │   └── schemas/
    │       ├── requests.py     # Request models
    │       └── responses.py    # Response models
    └── dependencies.py         # Shared dependencies
"""

from .v1.router import router as v1_router

__all__ = ["v1_router"]
