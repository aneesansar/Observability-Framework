"""
Rate Limiting Middleware.

Token bucket rate limiting for API endpoints.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Tokens are added at a fixed rate up to a maximum capacity.
    Each request consumes one token.
    """
    
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize with full bucket."""
        self.tokens = float(self.capacity)
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if rate limited
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now
    
    @property
    def wait_time(self) -> float:
        """Time to wait for next token."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.refill_rate


class RateLimiter:
    """
    Rate limiter with per-key buckets.
    
    Supports different limits for different keys (e.g., per tenant, per IP).
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst size (default: 2x rate)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or (requests_per_minute * 2)
        self.refill_rate = requests_per_minute / 60.0  # per second
        
        self._buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.refill_rate,
            )
        )
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., tenant_id, IP)
            
        Returns:
            True if allowed, False if rate limited
        """
        bucket = self._buckets[key]
        return bucket.consume()
    
    def get_wait_time(self, key: str) -> float:
        """
        Get wait time for next allowed request.
        
        Args:
            key: Rate limit key
            
        Returns:
            Seconds to wait
        """
        bucket = self._buckets[key]
        return bucket.wait_time
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests.
        
        Args:
            key: Rate limit key
            
        Returns:
            Number of remaining requests
        """
        bucket = self._buckets[key]
        bucket._refill()
        return int(bucket.tokens)
    
    def reset(self, key: str) -> None:
        """
        Reset a bucket.
        
        Args:
            key: Rate limit key
        """
        if key in self._buckets:
            del self._buckets[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Limits requests per tenant/IP with configurable rates.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        exclude_paths: Optional[list] = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Rate limit
            burst_size: Burst size
            key_func: Function to extract rate limit key from request
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )
        self.key_func = key_func or self._default_key_func
        self.exclude_paths = set(exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"])
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next handler
            
        Returns:
            Response
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get rate limit key
        key = self.key_func(request)
        
        # Check rate limit
        if not self.limiter.is_allowed(key):
            wait_time = self.limiter.get_wait_time(key)
            logger.warning(
                f"Rate limit exceeded for {key}",
                extra={"key": key, "wait_time": wait_time}
            )
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "retry_after": int(wait_time) + 1,
                },
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Remaining": "0",
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.limiter.get_remaining(key)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _default_key_func(self, request: Request) -> str:
        """
        Default key function: use tenant_id or IP.
        
        Args:
            request: Incoming request
            
        Returns:
            Rate limit key
        """
        # Try tenant_id from state
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id and tenant_id != "anonymous":
            return f"tenant:{tenant_id}"
        
        # Fall back to IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"


# ============================================================
# === Rate Limit Decorator ===
# ============================================================

# Global rate limiters for different purposes
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(
    name: str,
    requests_per_minute: int = 60,
) -> RateLimiter:
    """
    Get or create a named rate limiter.
    
    Args:
        name: Limiter name
        requests_per_minute: Rate limit
        
    Returns:
        RateLimiter instance
    """
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(requests_per_minute)
    return _rate_limiters[name]


def rate_limit(
    requests_per_minute: int = 60,
    key: Optional[str] = None,
):
    """
    Decorator to rate limit a function.
    
    Args:
        requests_per_minute: Rate limit
        key: Static key (if None, uses function name)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        limiter = get_rate_limiter(
            func.__name__,
            requests_per_minute,
        )
        
        def wrapper(*args, **kwargs):
            limit_key = key or func.__name__
            
            if not limiter.is_allowed(limit_key):
                wait_time = limiter.get_wait_time(limit_key)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                        "retry_after": int(wait_time) + 1,
                    },
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
