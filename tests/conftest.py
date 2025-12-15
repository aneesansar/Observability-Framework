"""
Pytest Configuration and Fixtures.

Shared fixtures for all tests.
"""

import os
import pytest
from typing import Generator

# Set test environment
os.environ["DISABLE_AUTH"] = "true"
os.environ["SERVICE_NAME"] = "test-service"


@pytest.fixture(scope="session")
def test_settings():
    """Get test settings."""
    from app.core import get_settings
    return get_settings()


@pytest.fixture
def mock_user():
    """Create a mock user identity."""
    from app.auth import UserIdentity, RoleGroup
    
    return UserIdentity(
        username="test_user",
        tenant_id="test_tenant",
        roles=["Developer"],
        role_group=RoleGroup.DEVELOPER,
    )


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user identity."""
    from app.auth import UserIdentity, RoleGroup
    
    return UserIdentity(
        username="admin_user",
        tenant_id="admin_tenant",
        roles=["Admin"],
        role_group=RoleGroup.ADMIN,
    )


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from app.main_new import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def container():
    """Get a fresh container for testing."""
    from app.core import get_container
    
    container = get_container()
    container.reset()
    yield container
    container.reset()


@pytest.fixture
def embedding_cache():
    """Get a fresh embedding cache."""
    from app.core.cache import EmbeddingCache
    
    cache = EmbeddingCache(max_size=100, ttl=60.0)
    yield cache
    cache.clear()


@pytest.fixture
def llm_cache():
    """Get a fresh LLM cache."""
    from app.core.cache import LLMResponseCache
    
    cache = LLMResponseCache(max_size=100, ttl=60.0)
    yield cache
    cache.clear()


@pytest.fixture
def rate_limiter():
    """Get a fresh rate limiter."""
    from app.core.rate_limit import RateLimiter
    
    return RateLimiter(requests_per_minute=10, burst_size=20)
