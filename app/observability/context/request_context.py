"""
Request Context Management.

Thread-safe context variables for request-scoped data propagation.
Provides utilities for managing request context across async boundaries.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, List, Optional

from fastapi import Request


# Thread-safe context variables
request_context: ContextVar[Optional[Request]] = ContextVar("request_context", default=None)
# Note: Using factory pattern to avoid mutable default issues
_function_stack: ContextVar[Optional[List[str]]] = ContextVar("function_stack", default=None)


@dataclass
class RequestMetadata:
    """Extracted metadata from a request for observability."""
    
    tenant_id: Optional[str] = None
    username: Optional[str] = None
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None


class RequestContextManager:
    """
    Manager for request context lifecycle.
    
    Usage:
        async with RequestContextManager(request):
            # Request context is available here
            tenant_id = get_current_tenant_id()
    """
    
    def __init__(self, request: Request):
        """
        Initialize context manager with a request.
        
        Args:
            request: FastAPI Request object
        """
        self._request = request
        self._token: Optional[Token] = None
    
    def __enter__(self) -> "RequestContextManager":
        """Set the request context."""
        self._token = request_context.set(self._request)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clear the request context."""
        if self._token is not None:
            try:
                request_context.reset(self._token)
            except ValueError:
                # Context already reset
                pass
    
    async def __aenter__(self) -> "RequestContextManager":
        """Async context manager entry."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.__exit__(exc_type, exc_val, exc_tb)


def get_current_request() -> Optional[Request]:
    """
    Get the current request from context.
    
    Returns:
        The current FastAPI Request or None if not in request context
    """
    try:
        return request_context.get()
    except LookupError:
        return None


def get_current_tenant_id() -> Optional[str]:
    """
    Get the current tenant ID from request context.
    
    Returns:
        Tenant ID string or None if not available
    """
    request = get_current_request()
    if request and hasattr(request, "state"):
        return getattr(request.state, "tenant_id", None)
    return None


def get_current_username() -> Optional[str]:
    """
    Get the current username from request context.
    
    Returns:
        Username string or None if not available
    """
    request = get_current_request()
    if request and hasattr(request, "state"):
        return getattr(request.state, "username", None)
    return None


def get_request_metadata() -> RequestMetadata:
    """
    Extract all relevant metadata from current request.
    
    Returns:
        RequestMetadata with available fields populated
    """
    request = get_current_request()
    if not request:
        return RequestMetadata()
    
    metadata = RequestMetadata()
    
    if hasattr(request, "state"):
        metadata.tenant_id = getattr(request.state, "tenant_id", None)
        metadata.username = getattr(request.state, "username", None)
        metadata.request_id = getattr(request.state, "request_id", None)
    
    if hasattr(request, "url"):
        metadata.path = str(request.url.path)
    
    if hasattr(request, "method"):
        metadata.method = request.method
    
    return metadata


def get_function_stack() -> List[str]:
    """
    Get the current function call stack.
    
    Returns:
        List of function names in the current call stack
    """
    try:
        stack = _function_stack.get()
        return stack if stack is not None else []
    except LookupError:
        return []


def push_function(func_name: str) -> Optional[Token]:
    """
    Push a function onto the call stack.
    
    Args:
        func_name: Name of the function being entered
        
    Returns:
        Token for resetting the stack later, or None if failed
    """
    try:
        current_stack = _function_stack.get()
        if current_stack is None:
            current_stack = []
        new_stack = current_stack + [func_name]
        return _function_stack.set(new_stack)
    except Exception:
        return None


def pop_function(token: Optional[Token]) -> None:
    """
    Pop a function from the call stack.
    
    Args:
        token: Token returned from push_function
    """
    if token is None:
        return
    try:
        _function_stack.reset(token)
    except ValueError:
        # Already reset
        pass


def is_in_request_context() -> bool:
    """
    Check if we're currently in a request context.
    
    Returns:
        True if a request context is active
    """
    return get_current_request() is not None


def should_skip_path(path: str, skip_paths: List[str]) -> bool:
    """
    Check if a path should be skipped from tracing.
    
    Args:
        path: Request path to check
        skip_paths: List of path prefixes to skip
        
    Returns:
        True if the path should be skipped
    """
    path_lower = path.lower()
    return any(path_lower.startswith(skip.lower()) for skip in skip_paths)
