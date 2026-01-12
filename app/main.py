"""
Application Entry Point.

FastAPI application with modular architecture.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables first
load_dotenv()

# Import core configuration
from app.core import settings, AppError, ResourceQuotaMiddleware, get_resource_manager, configure_tenant_quotas

# Import observability (sets up tracing before other imports)
from app.observability import (
    setup_observability,
    RequestContextManager,
    get_log_buffer,
)

# Import auth
from app.auth import (
    TenantMiddleware,
    auth_settings,
)

# Import API routers
from app.api import v1_router

# Import RAG for startup warmup
from app.rag import reindex, get_embedding_service

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================
# === Application Lifespan ===
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Startup:
    - Warm up embedding model
    - Build/load vector index
    
    Shutdown:
    - Cleanup resources
    """
    # Startup
    logger.info("Starting application...")
    
    try:
        # Warm up embedding model
        logger.info("Warming up embedding model...")
        embedding_service = get_embedding_service()
        _ = embedding_service.dimension  # Force model load
        logger.info(f"Embedding model loaded: {embedding_service.model_name}")
        
        # Build/load index
        logger.info("Loading vector index...")
        result = reindex()
        logger.info(f"Index ready: {result['documents']} documents, dim={result['dim']}")
        
    except Exception as e:
        logger.warning(f"Startup warmup failed (non-fatal): {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# ============================================================
# === Create FastAPI Application ===
# ============================================================
app = FastAPI(
    title=settings.api_title,
    description="""
## RAG Observability Demo API

A production-ready RAG (Retrieval-Augmented Generation) API with comprehensive observability.

### Features

- **RAG Pipeline**: Query travel data with semantic search and LLM generation
- **Authentication**: Keycloak-based OAuth2/OIDC authentication
- **Observability**: OpenTelemetry tracing, Prometheus metrics, structured logging
- **Multi-tenancy**: Tenant isolation for all data and telemetry

### Modules

- `/api/v1/auth` - Authentication endpoints
- `/api/v1/rag` - RAG query endpoints
- `/api/v1/observability` - Traces and logs access
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================
# === Middleware ===
# ============================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resource Quota middleware (added BEFORE TenantMiddleware so it runs AFTER in the chain)
# FastAPI middleware order is reversed: last added = first executed
app.add_middleware(ResourceQuotaMiddleware)

# Tenant/Auth middleware (runs first to set tenant_id)
app.add_middleware(TenantMiddleware)

# Configure tenant quotas for fair resource distribution
# Lower limits to make rate limiting more visible in testing
configure_tenant_quotas({
    "tenant_a": {"max_concurrent_requests": 3, "requests_per_minute": 15, "tokens_per_minute": 30000},
    "tenant_b": {"max_concurrent_requests": 3, "requests_per_minute": 15, "tokens_per_minute": 30000},
    "tenant_c": {"max_concurrent_requests": 3, "requests_per_minute": 15, "tokens_per_minute": 30000},
    "tenant_d": {"max_concurrent_requests": 3, "requests_per_minute": 15, "tokens_per_minute": 30000},
    "tenant_e": {"max_concurrent_requests": 3, "requests_per_minute": 15, "tokens_per_minute": 30000},
})


# ============================================================
# === Exception Handlers ===
# ============================================================
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Handle application errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
        },
    )


# ============================================================
# === Setup Observability ===
# ============================================================
setup_observability(app)


# ============================================================
# === Include Routers ===
# ============================================================
app.include_router(v1_router)


# ============================================================
# === Root Endpoints ===
# ============================================================
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.service_name,
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.service_name,
    }


# ============================================================
# === Legacy Compatibility Endpoints ===
# ============================================================
# These endpoints maintain backwards compatibility with existing clients

from fastapi import Depends, Form, HTTPException, Query
from app.auth import get_current_user, auth_service, UserIdentity
from app.rag import get_rag_pipeline
from app.api.v1.endpoints.rag import format_results_as_json, format_as_fenced_json, generate_text_explanation
from pydantic import BaseModel
import json


class QueryRequest(BaseModel):
    """Legacy query request."""
    query: str


@app.post("/login", tags=["legacy"])
async def legacy_login(
    username: str = Form(...),
    password: str = Form(...),
):
    """Legacy login endpoint."""
    try:
        token_info = await auth_service.login(username, password)
        return token_info.to_dict()
    except Exception as e:
        logger.warning(f"Login failed for {username}: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


@app.post("/query", tags=["legacy"])
async def legacy_query(
    request: Request,
    query_data: QueryRequest,
    user: UserIdentity = Depends(get_current_user),
):
    """
    Legacy query endpoint.
    
    Returns response in format expected by frontend:
    - answer: fenced JSON with results array
    - results: direct results array
    """
    try:
        pipeline = get_rag_pipeline()
        response = pipeline.query(
            question=query_data.query,
            tenant_id=user.tenant_id,
        )
        
        # Format results for frontend
        results = format_results_as_json(response.retrieved_docs)
        answer = format_as_fenced_json(results)
        text_explanation = generate_text_explanation(query_data.query, results)
        
        # Generate conversation ID for feedback tracking
        import uuid
        conversation_id = str(uuid.uuid4())
        
        return {
            "conversation_id": conversation_id,
            "text_explanation": text_explanation,
            "answer": answer,
            "results": results,
            "reasoning": response.generation_result.reasoning_trace if response.generation_result else [],
            "query_result": {
                "tenant_id": user.tenant_id,
                "query": query_data.query,
                "retrieved_count": len(response.retrieved_docs),
            },
        }
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query processing failed")


@app.get("/logs", tags=["legacy"])
async def legacy_logs(
    level: str = Query(None),
    user: UserIdentity = Depends(get_current_user),
):
    """Legacy logs endpoint."""
    logs = get_log_buffer(
        level=level,
        tenant_id=None if user.is_admin else user.tenant_id,
    )
    return sorted(
        [log.to_dict() for log in logs],
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )


@app.get("/auth/debug", tags=["legacy"])
async def legacy_auth_debug(
    user: UserIdentity = Depends(get_current_user),
):
    """Legacy auth debug endpoint."""
    return {
        "username": user.username,
        "tenant_id": user.tenant_id,
        "roles": user.roles,
        "role_group": user.role_group.value,
    }


# ============================================================
# === Main Entry Point ===
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
