"""
Authentication Middleware.

HTTP middleware for tenant/user context extraction and tracing.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability import (
    RequestContextManager,
    get_tracer,
    SpanAttributes,
)

from .config import auth_settings
from .models import RoleGroup, UserIdentity
from .service import auth_service

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tenant context extraction and tracing.
    
    Extracts tenant_id, username, and roles from the authentication token
    and stores them in request.state for downstream use.
    
    Also creates a tracing span with tenant metadata.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request and extract tenant context.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from downstream handlers
        """
        # Set up observability context
        async with RequestContextManager(request):
            # Extract user identity
            user = await self._extract_user_identity(request)
            
            # Store in request state
            self._store_in_request_state(request, user)
            
            # Create tracing span
            tracer = get_tracer(__name__)
            with tracer.start_as_current_span("tenant_request") as span:
                span.set_attribute(SpanAttributes.TENANT_ID, user.tenant_id)
                span.set_attribute(SpanAttributes.USERNAME, user.username)
                span.set_attribute("http.path", request.url.path)
                span.set_attribute("http.method", request.method)
                span.set_attribute("user.role_group", user.role_group.value)
                
                # Process request
                response = await call_next(request)
                
                # Add response info to span
                span.set_attribute("http.status_code", response.status_code)
                
                return response
    
    async def _extract_user_identity(self, request: Request) -> UserIdentity:
        """
        Extract user identity from request.
        
        Args:
            request: Incoming request
            
        Returns:
            UserIdentity (anonymous if no valid token)
        """
        # Dev mode
        if auth_settings.disable_auth:
            return UserIdentity.dev_user()
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return UserIdentity.anonymous()
        
        token = auth_header.split("Bearer ")[1]
        
        try:
            return auth_service.validate_token(token)
        except Exception as e:
            logger.debug(f"Token validation failed in middleware: {e}")
            return UserIdentity.anonymous()
    
    def _store_in_request_state(
        self,
        request: Request,
        user: UserIdentity,
    ) -> None:
        """
        Store user identity in request state.
        
        Args:
            request: Request object
            user: User identity
        """
        request.state.tenant_id = user.tenant_id
        request.state.username = user.username
        request.state.user_roles = user.roles
        request.state.role_group = user.role_group.value
        request.state.user_identity = user


async def tenant_telemetry_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """
    Functional middleware for tenant telemetry.
    
    Alternative to class-based middleware for use with app.middleware decorator.
    
    Args:
        request: Incoming request
        call_next: Next handler
        
    Returns:
        Response
    """
    # Set up observability context
    async with RequestContextManager(request):
        # Extract user identity
        user = await _extract_user_from_request(request)
        
        # Store in request state
        request.state.tenant_id = user.tenant_id
        request.state.username = user.username
        request.state.user_roles = user.roles
        request.state.role_group = user.role_group.value
        
        # Create tracing span
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("tenant_request") as span:
            span.set_attribute(SpanAttributes.TENANT_ID, user.tenant_id)
            span.set_attribute(SpanAttributes.USERNAME, user.username)
            span.set_attribute("http.path", request.url.path)
            
            response = await call_next(request)
            return response


async def _extract_user_from_request(request: Request) -> UserIdentity:
    """
    Extract user identity from request headers.
    
    Args:
        request: Incoming request
        
    Returns:
        UserIdentity
    """
    if auth_settings.disable_auth:
        return UserIdentity.dev_user()
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return UserIdentity.anonymous()
    
    token = auth_header.split("Bearer ")[1]
    
    try:
        return auth_service.validate_token(token)
    except Exception:
        return UserIdentity.anonymous()
