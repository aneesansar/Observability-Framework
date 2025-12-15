"""
Caching Layer.

In-memory caching with TTL support for embeddings and LLM responses.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    
    value: T
    created_at: float
    ttl: float
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl <= 0:
            return False  # No expiration
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


class Cache(Generic[T]):
    """
    Generic in-memory cache with TTL support.
    
    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Hit/miss statistics
    - Thread-safe operations
    """
    
    def __init__(
        self,
        name: str = "cache",
        max_size: int = 1000,
        default_ttl: float = 3600.0,
    ):
        """
        Initialize the cache.
        
        Args:
            name: Cache name for logging
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (0 = no expiration)
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        
        if entry is None:
            self._misses += 1
            return None
        
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        
        entry.touch()
        self._hits += 1
        return entry.value
    
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None = use default)
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        self._cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        
        # Find entry with lowest hits (simple LRU approximation)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].hits,
        )
        del self._cache[lru_key]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


# ============================================================
# === Specialized Caches ===
# ============================================================

class EmbeddingCache(Cache[list]):
    """
    Cache for text embeddings.
    
    Keys are hashed text content.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: float = 86400.0,  # 24 hours
    ):
        super().__init__(
            name="embeddings",
            max_size=max_size,
            default_ttl=ttl,
        )
    
    @staticmethod
    def make_key(text: str, model: str = "") -> str:
        """
        Create cache key from text.
        
        Args:
            text: Text to embed
            model: Model name
            
        Returns:
            Cache key
        """
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get_embedding(
        self,
        text: str,
        model: str = "",
    ) -> Optional[list]:
        """
        Get cached embedding.
        
        Args:
            text: Text that was embedded
            model: Model name
            
        Returns:
            Embedding vector or None
        """
        key = self.make_key(text, model)
        return self.get(key)
    
    def set_embedding(
        self,
        text: str,
        embedding: list,
        model: str = "",
    ) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model name
        """
        key = self.make_key(text, model)
        self.set(key, embedding)


class LLMResponseCache(Cache[str]):
    """
    Cache for LLM responses.
    
    Keys are hashed prompts.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600.0,  # 1 hour
    ):
        super().__init__(
            name="llm_responses",
            max_size=max_size,
            default_ttl=ttl,
        )
    
    @staticmethod
    def make_key(
        prompt: str,
        model: str = "",
        tenant_id: str = "",
    ) -> str:
        """
        Create cache key from prompt.
        
        Args:
            prompt: LLM prompt
            model: Model name
            tenant_id: Tenant identifier
            
        Returns:
            Cache key
        """
        content = f"{tenant_id}:{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get_response(
        self,
        prompt: str,
        model: str = "",
        tenant_id: str = "",
    ) -> Optional[str]:
        """
        Get cached LLM response.
        
        Args:
            prompt: LLM prompt
            model: Model name
            tenant_id: Tenant identifier
            
        Returns:
            Cached response or None
        """
        key = self.make_key(prompt, model, tenant_id)
        return self.get(key)
    
    def set_response(
        self,
        prompt: str,
        response: str,
        model: str = "",
        tenant_id: str = "",
    ) -> None:
        """
        Cache an LLM response.
        
        Args:
            prompt: LLM prompt
            response: LLM response
            model: Model name
            tenant_id: Tenant identifier
        """
        key = self.make_key(prompt, model, tenant_id)
        self.set(key, response)


# ============================================================
# === Global Cache Instances ===
# ============================================================

_embedding_cache: Optional[EmbeddingCache] = None
_llm_cache: Optional[LLMResponseCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_llm_cache() -> LLMResponseCache:
    """Get the global LLM response cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache()
    return _llm_cache


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        "embeddings": get_embedding_cache().stats,
        "llm_responses": get_llm_cache().stats,
    }


# ============================================================
# === Caching Decorator ===
# ============================================================

def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use ("embeddings" or "llm_responses")
        ttl: TTL in seconds
        key_func: Custom key generation function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Create a dedicated cache for this function
        func_cache: Cache[Any] = Cache(
            name=f"{cache_name}:{func.__name__}",
            default_ttl=ttl or 3600.0,
        )
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                key = hashlib.sha256(key_data.encode()).hexdigest()[:32]
            
            # Check cache
            cached_value = func_cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            func_cache.set(key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Attach cache for inspection
        wrapper.cache = func_cache
        return wrapper
    
    return decorator
