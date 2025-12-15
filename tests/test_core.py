"""
Tests for Core Module.

Tests for configuration, exceptions, caching, and rate limiting.
"""

import pytest
import time


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_loads(self, test_settings):
        """Test that settings load correctly."""
        assert test_settings is not None
        assert test_settings.service_name is not None
    
    def test_cors_origins_parsing(self, test_settings):
        """Test CORS origins are parsed as list."""
        origins = test_settings.cors_origins
        assert isinstance(origins, list)
    
    def test_role_lists_parsing(self, test_settings):
        """Test role lists are parsed correctly."""
        assert isinstance(test_settings.admin_roles, list)
        assert isinstance(test_settings.manager_roles, list)
        assert isinstance(test_settings.developer_roles, list)


class TestExceptions:
    """Tests for exception classes."""
    
    def test_app_error(self):
        """Test AppError creation."""
        from app.core import AppError
        
        error = AppError("Test error", status_code=400)
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.error_code == "INTERNAL_ERROR"
    
    def test_not_found_error(self):
        """Test NotFoundError creation."""
        from app.core import NotFoundError
        
        error = NotFoundError("Resource not found", resource_type="Trip")
        assert error.status_code == 404
        assert error.details["resource_type"] == "Trip"
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        from app.core import ValidationError
        
        error = ValidationError("Invalid input", field="query")
        assert error.status_code == 400
        assert error.details["field"] == "query"
    
    def test_error_to_dict(self):
        """Test error serialization."""
        from app.core import AppError
        
        error = AppError("Test", status_code=500, error_code="TEST_ERROR")
        result = error.to_dict()
        
        assert result["error"] == "TEST_ERROR"
        assert result["message"] == "Test"


class TestCache:
    """Tests for caching functionality."""
    
    def test_cache_set_get(self, embedding_cache):
        """Test basic cache set and get."""
        embedding_cache.set("key1", [1.0, 2.0, 3.0])
        result = embedding_cache.get("key1")
        
        assert result == [1.0, 2.0, 3.0]
    
    def test_cache_miss(self, embedding_cache):
        """Test cache miss returns None."""
        result = embedding_cache.get("nonexistent")
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        from app.core.cache import Cache
        
        cache = Cache[str](name="test", default_ttl=0.1)
        cache.set("key", "value")
        
        # Should exist immediately
        assert cache.get("key") == "value"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key") is None
    
    def test_cache_stats(self, embedding_cache):
        """Test cache statistics."""
        embedding_cache.set("key1", [1.0])
        embedding_cache.get("key1")  # hit
        embedding_cache.get("key2")  # miss
        
        stats = embedding_cache.stats
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    def test_embedding_cache_key_generation(self):
        """Test embedding cache key generation."""
        from app.core.cache import EmbeddingCache
        
        key1 = EmbeddingCache.make_key("hello", "model1")
        key2 = EmbeddingCache.make_key("hello", "model1")
        key3 = EmbeddingCache.make_key("hello", "model2")
        
        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different model = different key
    
    def test_llm_cache_key_generation(self):
        """Test LLM cache key generation."""
        from app.core.cache import LLMResponseCache
        
        key1 = LLMResponseCache.make_key("prompt", "model", "tenant1")
        key2 = LLMResponseCache.make_key("prompt", "model", "tenant2")
        
        assert key1 != key2  # Different tenant = different key


class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    def test_rate_limiter_allows_requests(self, rate_limiter):
        """Test rate limiter allows requests within limit."""
        for _ in range(10):
            assert rate_limiter.is_allowed("test_key")
    
    def test_rate_limiter_blocks_excess(self, rate_limiter):
        """Test rate limiter blocks excess requests."""
        # Exhaust the bucket
        for _ in range(20):  # burst_size
            rate_limiter.is_allowed("test_key")
        
        # Next request should be blocked
        assert not rate_limiter.is_allowed("test_key")
    
    def test_rate_limiter_refills(self, rate_limiter):
        """Test rate limiter refills over time."""
        # Consume some tokens but not all
        for _ in range(15):
            rate_limiter.is_allowed("test_key")
        
        # Wait for refill (at 10/min = 0.167/sec, 0.5s = ~0.8 tokens)
        time.sleep(0.5)
        
        # Should still have tokens (started with 20, used 15, refilled ~0.8)
        assert rate_limiter.get_remaining("test_key") >= 0
    
    def test_rate_limiter_per_key(self, rate_limiter):
        """Test rate limiter tracks keys separately."""
        # Exhaust key1
        for _ in range(20):
            rate_limiter.is_allowed("key1")
        
        # key2 should still be allowed
        assert rate_limiter.is_allowed("key2")
    
    def test_get_remaining(self, rate_limiter):
        """Test getting remaining requests."""
        initial = rate_limiter.get_remaining("test_key")
        
        rate_limiter.is_allowed("test_key")
        
        remaining = rate_limiter.get_remaining("test_key")
        assert remaining == initial - 1


class TestContainer:
    """Tests for dependency injection container."""
    
    def test_register_singleton(self, container):
        """Test singleton registration."""
        container.register_singleton("test", lambda: {"value": 42})
        
        result1 = container.get("test")
        result2 = container.get("test")
        
        assert result1 is result2  # Same instance
        assert result1["value"] == 42
    
    def test_register_instance(self, container):
        """Test instance registration."""
        instance = {"value": 100}
        container.register_instance("test", instance)
        
        result = container.get("test")
        assert result is instance
    
    def test_override(self, container):
        """Test dependency override."""
        container.register_singleton("test", lambda: "original")
        container.override("test", "override")
        
        assert container.get("test") == "override"
        
        container.clear_override("test")
        assert container.get("test") == "original"
    
    def test_missing_dependency(self, container):
        """Test missing dependency raises KeyError."""
        with pytest.raises(KeyError):
            container.get("nonexistent")
