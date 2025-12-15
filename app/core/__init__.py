"""
Core Module.

Shared infrastructure components for the application.

Contains:
- Configuration management
- Exception definitions
- Dependency injection
- Caching
- Rate limiting
"""

from .config import (
    Settings,
    settings,
    get_settings,
)
from .exceptions import (
    AppError,
    NotFoundError,
    ValidationError,
    ServiceUnavailableError,
    RateLimitError,
    ConfigurationError,
)
from .container import (
    Container,
    get_container,
    setup_container,
    inject,
    get_dependency,
)
from .cache import (
    Cache,
    EmbeddingCache,
    LLMResponseCache,
    get_embedding_cache,
    get_llm_cache,
    get_all_cache_stats,
    cached,
)
from .rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    get_rate_limiter,
    rate_limit,
)

__all__ = [
    # Config
    "Settings",
    "settings",
    "get_settings",
    # Exceptions
    "AppError",
    "NotFoundError",
    "ValidationError",
    "ServiceUnavailableError",
    "RateLimitError",
    "ConfigurationError",
    # Container
    "Container",
    "get_container",
    "setup_container",
    "inject",
    "get_dependency",
    # Cache
    "Cache",
    "EmbeddingCache",
    "LLMResponseCache",
    "get_embedding_cache",
    "get_llm_cache",
    "get_all_cache_stats",
    "cached",
    # Rate Limit
    "RateLimiter",
    "RateLimitMiddleware",
    "get_rate_limiter",
    "rate_limit",
]
