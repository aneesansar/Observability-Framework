"""
Dependency Injection Container.

Centralized dependency management using dependency-injector pattern.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Container:
    """
    Simple dependency injection container.
    
    Provides:
    - Singleton registration
    - Factory registration
    - Lazy initialization
    - Override support for testing
    """
    
    _instance: Optional["Container"] = None
    
    def __new__(cls) -> "Container":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the container."""
        if self._initialized:
            return
        
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._overrides: Dict[str, Any] = {}
        self._initialized = True
    
    def register_singleton(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> None:
        """
        Register a singleton dependency.
        
        The factory is called once on first access.
        
        Args:
            name: Dependency name
            factory: Factory function to create the instance
        """
        self._factories[name] = factory
    
    def register_instance(
        self,
        name: str,
        instance: Any,
    ) -> None:
        """
        Register an existing instance.
        
        Args:
            name: Dependency name
            instance: The instance to register
        """
        self._singletons[name] = instance
    
    def get(self, name: str) -> Any:
        """
        Get a dependency by name.
        
        Args:
            name: Dependency name
            
        Returns:
            The dependency instance
            
        Raises:
            KeyError: If dependency not found
        """
        # Check overrides first (for testing)
        if name in self._overrides:
            return self._overrides[name]
        
        # Check existing singletons
        if name in self._singletons:
            return self._singletons[name]
        
        # Create from factory
        if name in self._factories:
            instance = self._factories[name]()
            self._singletons[name] = instance
            return instance
        
        raise KeyError(f"Dependency not found: {name}")
    
    def override(self, name: str, instance: Any) -> None:
        """
        Override a dependency (for testing).
        
        Args:
            name: Dependency name
            instance: Override instance
        """
        self._overrides[name] = instance
    
    def clear_override(self, name: str) -> None:
        """
        Clear an override.
        
        Args:
            name: Dependency name
        """
        self._overrides.pop(name, None)
    
    def clear_all_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()
    
    def reset(self) -> None:
        """Reset the container (for testing)."""
        self._singletons.clear()
        self._factories.clear()
        self._overrides.clear()


@lru_cache()
def get_container() -> Container:
    """Get the container instance."""
    return Container()


# ============================================================
# === Service Registration ===
# ============================================================

def setup_container() -> Container:
    """
    Set up the dependency injection container with all services.
    
    Returns:
        Configured container
    """
    container = get_container()
    
    # Register settings
    from app.core.config import get_settings
    container.register_singleton("settings", get_settings)
    
    # Register auth services
    from app.auth.config import get_auth_settings
    from app.auth.keycloak_client import get_keycloak_client
    from app.auth.service import get_auth_service
    
    container.register_singleton("auth_settings", get_auth_settings)
    container.register_singleton("keycloak_client", get_keycloak_client)
    container.register_singleton("auth_service", get_auth_service)
    
    # Register RAG services
    from app.rag.config import get_rag_settings
    from app.rag.core.embeddings import get_embedding_service
    from app.rag.core.retriever import get_retriever
    from app.rag.core.generator import get_generator
    from app.rag.core.pipeline import get_rag_pipeline
    from app.rag.storage.faiss_store import get_faiss_store
    from app.rag.storage.mongo_store import get_mongo_store
    
    container.register_singleton("rag_settings", get_rag_settings)
    container.register_singleton("embedding_service", get_embedding_service)
    container.register_singleton("retriever", get_retriever)
    container.register_singleton("generator", get_generator)
    container.register_singleton("rag_pipeline", get_rag_pipeline)
    container.register_singleton("faiss_store", get_faiss_store)
    container.register_singleton("mongo_store", get_mongo_store)
    
    # Register observability services
    from app.observability.tracing.provider import get_tracing_provider
    from app.observability.metrics.provider import get_metrics_provider
    from app.observability.logging.provider import get_logging_provider
    
    container.register_singleton("tracing_provider", get_tracing_provider)
    container.register_singleton("metrics_provider", get_metrics_provider)
    container.register_singleton("logging_provider", get_logging_provider)
    
    return container


# ============================================================
# === Dependency Injection Helpers ===
# ============================================================

def inject(name: str) -> Callable:
    """
    Decorator to inject a dependency.
    
    Usage:
        @inject("auth_service")
        def my_function(auth_service: AuthService):
            pass
    
    Args:
        name: Dependency name
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            container = get_container()
            kwargs[name] = container.get(name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_dependency(name: str) -> Any:
    """
    Get a dependency from the container.
    
    Args:
        name: Dependency name
        
    Returns:
        Dependency instance
    """
    container = get_container()
    return container.get(name)
