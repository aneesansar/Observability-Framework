"""
Tenant Resource Quota Management.

Ensures fair resource distribution across tenants by limiting:
- Concurrent requests per tenant
- Requests per minute (rate limiting)
- Token usage per minute
- Memory allocation hints

Prevents a single tenant from consuming all system resources.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class TenantQuota:
    """Resource quota configuration for a tenant."""
    max_concurrent_requests: int = 5
    requests_per_minute: int = 30
    tokens_per_minute: int = 50000
    max_query_length: int = 1000
    priority: int = 1  # 1=normal, 2=premium, 3=enterprise


@dataclass
class TenantResourceState:
    """Current resource usage state for a tenant."""
    concurrent_requests: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    minute_start: float = field(default_factory=time.time)
    total_requests: int = 0
    total_tokens: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    
    def reset_if_new_minute(self):
        """Reset per-minute counters if a new minute has started."""
        now = time.time()
        if now - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = now


class TenantResourceManager:
    """
    Manages resource allocation across tenants.
    
    Ensures fair distribution by:
    1. Limiting concurrent requests per tenant
    2. Rate limiting requests per minute
    3. Tracking token usage
    4. Providing queue-based fairness for burst traffic
    """
    
    # Default quotas for different tenant tiers
    DEFAULT_QUOTAS = {
        "default": TenantQuota(max_concurrent_requests=5, requests_per_minute=30, tokens_per_minute=50000),
        "premium": TenantQuota(max_concurrent_requests=10, requests_per_minute=60, tokens_per_minute=100000, priority=2),
        "enterprise": TenantQuota(max_concurrent_requests=20, requests_per_minute=120, tokens_per_minute=200000, priority=3),
    }
    
    def __init__(self, global_max_concurrent: int = 50):
        """
        Initialize the resource manager.
        
        Args:
            global_max_concurrent: Maximum concurrent requests across all tenants
        """
        self.global_max_concurrent = global_max_concurrent
        self.global_concurrent = 0
        self._tenant_states: Dict[str, TenantResourceState] = defaultdict(TenantResourceState)
        self._tenant_quotas: Dict[str, TenantQuota] = {}
        # Use threading.Lock for thread-safety across async requests
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._global_lock = threading.Lock()
        self._request_queue: asyncio.Queue = asyncio.Queue()
        
    def set_tenant_quota(self, tenant_id: str, quota: TenantQuota):
        """Set custom quota for a tenant."""
        self._tenant_quotas[tenant_id] = quota
        logger.info(f"Set quota for {tenant_id}: {quota}")
    
    def get_tenant_quota(self, tenant_id: str) -> TenantQuota:
        """Get quota for a tenant (default if not set)."""
        return self._tenant_quotas.get(tenant_id, self.DEFAULT_QUOTAS["default"])
    
    def get_tenant_state(self, tenant_id: str) -> TenantResourceState:
        """Get current resource state for a tenant."""
        state = self._tenant_states[tenant_id]
        state.reset_if_new_minute()
        return state
    
    def acquire_resources(self, tenant_id: str, estimated_tokens: int = 0) -> Dict[str, Any]:
        """
        Attempt to acquire resources for a request.
        
        Args:
            tenant_id: Tenant identifier
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            Dict with acquisition result and wait info
            
        Raises:
            HTTPException: If resources cannot be acquired
        """
        quota = self.get_tenant_quota(tenant_id)
        
        logger.info(f"QUOTA CHECK: tenant={tenant_id}, quota={quota.max_concurrent_requests} concurrent, {quota.requests_per_minute}/min")
        
        with self._locks[tenant_id]:
            state = self.get_tenant_state(tenant_id)
            
            logger.info(f"QUOTA STATE: tenant={tenant_id}, current_concurrent={state.concurrent_requests}, requests_this_min={state.requests_this_minute}")
            
            # Check concurrent request limit
            if state.concurrent_requests >= quota.max_concurrent_requests:
                state.rejected_requests += 1
                logger.warning(f"Tenant {tenant_id} exceeded concurrent limit: {state.concurrent_requests}/{quota.max_concurrent_requests}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "CONCURRENT_LIMIT_EXCEEDED",
                        "message": f"Maximum concurrent requests ({quota.max_concurrent_requests}) exceeded",
                        "current": state.concurrent_requests,
                        "limit": quota.max_concurrent_requests,
                        "retry_after": 5,
                    },
                    headers={"Retry-After": "5", "X-Quota-Concurrent-Limit": str(quota.max_concurrent_requests)},
                )
            
            # Check rate limit
            if state.requests_this_minute >= quota.requests_per_minute:
                state.rejected_requests += 1
                wait_time = 60 - (time.time() - state.minute_start)
                logger.warning(f"Tenant {tenant_id} rate limited: {state.requests_this_minute}/{quota.requests_per_minute}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit ({quota.requests_per_minute}/min) exceeded",
                        "current": state.requests_this_minute,
                        "limit": quota.requests_per_minute,
                        "retry_after": int(wait_time) + 1,
                    },
                    headers={"Retry-After": str(int(wait_time) + 1), "X-Quota-Rate-Limit": str(quota.requests_per_minute)},
                )
            
            # Check token limit
            if state.tokens_this_minute + estimated_tokens > quota.tokens_per_minute:
                state.rejected_requests += 1
                wait_time = 60 - (time.time() - state.minute_start)
                logger.warning(f"Tenant {tenant_id} token limited: {state.tokens_this_minute}/{quota.tokens_per_minute}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "TOKEN_LIMIT_EXCEEDED",
                        "message": f"Token limit ({quota.tokens_per_minute}/min) exceeded",
                        "current": state.tokens_this_minute,
                        "limit": quota.tokens_per_minute,
                        "retry_after": int(wait_time) + 1,
                    },
                    headers={"Retry-After": str(int(wait_time) + 1), "X-Quota-Token-Limit": str(quota.tokens_per_minute)},
                )
            
            # Check global limit
            with self._global_lock:
                if self.global_concurrent >= self.global_max_concurrent:
                    state.rejected_requests += 1
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "SYSTEM_OVERLOADED",
                            "message": "System at capacity, please retry",
                            "retry_after": 10,
                        },
                        headers={"Retry-After": "10"},
                    )
                self.global_concurrent += 1
            
            # Acquire resources
            state.concurrent_requests += 1
            state.requests_this_minute += 1
            state.total_requests += 1
            
            return {
                "acquired": True,
                "tenant_id": tenant_id,
                "concurrent": state.concurrent_requests,
                "rate": state.requests_this_minute,
                "quota": quota,
            }
    
    def release_resources(self, tenant_id: str, tokens_used: int = 0):
        """
        Release resources after request completion.
        
        Args:
            tenant_id: Tenant identifier
            tokens_used: Actual tokens used
        """
        with self._locks[tenant_id]:
            state = self._tenant_states[tenant_id]
            state.concurrent_requests = max(0, state.concurrent_requests - 1)
            state.tokens_this_minute += tokens_used
            state.total_tokens += tokens_used
        
        with self._global_lock:
            self.global_concurrent = max(0, self.global_concurrent - 1)
    
    @asynccontextmanager
    async def request_context(self, tenant_id: str, estimated_tokens: int = 0):
        """
        Context manager for resource acquisition.
        
        Usage:
            async with resource_manager.request_context(tenant_id) as ctx:
                # Process request
                pass
        """
        result = self.acquire_resources(tenant_id, estimated_tokens)
        try:
            yield result
        finally:
            self.release_resources(tenant_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        stats = {
            "global_concurrent": self.global_concurrent,
            "global_max_concurrent": self.global_max_concurrent,
            "tenants": {},
        }
        
        for tenant_id, state in self._tenant_states.items():
            state.reset_if_new_minute()
            quota = self.get_tenant_quota(tenant_id)
            stats["tenants"][tenant_id] = {
                "concurrent": state.concurrent_requests,
                "concurrent_limit": quota.max_concurrent_requests,
                "requests_this_minute": state.requests_this_minute,
                "rate_limit": quota.requests_per_minute,
                "tokens_this_minute": state.tokens_this_minute,
                "token_limit": quota.tokens_per_minute,
                "total_requests": state.total_requests,
                "total_tokens": state.total_tokens,
                "rejected_requests": state.rejected_requests,
                "utilization_pct": (state.concurrent_requests / quota.max_concurrent_requests) * 100,
            }
        
        return stats
    
    def get_fairness_metrics(self) -> Dict[str, Any]:
        """Calculate fairness metrics across tenants."""
        if not self._tenant_states:
            return {"fairness_index": 1.0, "tenants": 0}
        
        # Jain's fairness index
        requests = [s.total_requests for s in self._tenant_states.values()]
        if not requests or sum(requests) == 0:
            return {"fairness_index": 1.0, "tenants": len(self._tenant_states)}
        
        n = len(requests)
        sum_x = sum(requests)
        sum_x2 = sum(x**2 for x in requests)
        
        fairness_index = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
        
        return {
            "fairness_index": fairness_index,
            "tenants": n,
            "total_requests": sum_x,
            "requests_per_tenant": {tid: s.total_requests for tid, s in self._tenant_states.items()},
            "rejected_per_tenant": {tid: s.rejected_requests for tid, s in self._tenant_states.items()},
        }


# Global resource manager instance
_resource_manager: Optional[TenantResourceManager] = None


def get_resource_manager() -> TenantResourceManager:
    """Get or create the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = TenantResourceManager()
    return _resource_manager


class ResourceQuotaMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enforcing tenant resource quotas.
    
    Applies to RAG query endpoints to ensure fair resource distribution.
    """
    
    def __init__(
        self,
        app,
        protected_paths: Optional[list] = None,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.resource_manager = get_resource_manager()
        self.protected_paths = set(protected_paths or ["/api/v1/rag/query"])
        self.exclude_paths = set(exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with resource quota enforcement."""
        path = request.url.path
        
        # Skip excluded paths
        if path in self.exclude_paths:
            return await call_next(request)
        
        # Check if path is protected
        is_protected = any(path.startswith(p) for p in self.protected_paths)
        if not is_protected:
            return await call_next(request)
        
        # Get tenant ID from request state (set by TenantMiddleware)
        tenant_id = getattr(request.state, "tenant_id", None) or "anonymous"
        
        logger.info(f"ResourceQuota: Processing request for tenant={tenant_id}, path={path}")
        
        # Acquire resources (this will raise HTTPException if limits exceeded)
        try:
            ctx = self.resource_manager.acquire_resources(tenant_id)
            logger.info(f"ResourceQuota: Acquired for {tenant_id}, concurrent={ctx['concurrent']}, rate={ctx['rate']}")
        except HTTPException as e:
            logger.warning(f"ResourceQuota: REJECTED {tenant_id} - {e.detail}")
            # Return JSONResponse directly instead of raising (middleware can't handle raised exceptions well)
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
                headers=dict(e.headers) if e.headers else {"Retry-After": "5"},
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add quota headers
            quota = ctx["quota"]
            state = self.resource_manager.get_tenant_state(tenant_id)
            response.headers["X-Quota-Concurrent-Used"] = str(state.concurrent_requests)
            response.headers["X-Quota-Concurrent-Limit"] = str(quota.max_concurrent_requests)
            response.headers["X-Quota-Rate-Used"] = str(state.requests_this_minute)
            response.headers["X-Quota-Rate-Limit"] = str(quota.requests_per_minute)
            
            return response
        finally:
            # Always release resources
            self.resource_manager.release_resources(tenant_id)


def configure_tenant_quotas(quotas: Dict[str, Dict[str, int]]):
    """
    Configure quotas for multiple tenants.
    
    Args:
        quotas: Dict mapping tenant_id to quota settings
        
    Example:
        configure_tenant_quotas({
            "tenant_a": {"max_concurrent_requests": 3, "requests_per_minute": 20},
            "tenant_b": {"max_concurrent_requests": 5, "requests_per_minute": 30},
        })
    """
    manager = get_resource_manager()
    for tenant_id, settings in quotas.items():
        quota = TenantQuota(**settings)
        manager.set_tenant_quota(tenant_id, quota)
