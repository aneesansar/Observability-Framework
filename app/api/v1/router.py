"""
API v1 Router.

Main router that includes all v1 endpoints.
"""

from fastapi import APIRouter

from .endpoints import auth, rag, observability

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Include endpoint routers
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(rag.router, prefix="/rag", tags=["rag"])
router.include_router(observability.router, prefix="/observability", tags=["observability"])
