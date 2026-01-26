"""
Authentication Middleware.

HTTP middleware for tenant/user context extraction and tracing.
Supports role-based tenant validation.
"""

import logging
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
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
    
    Extracts tenant_id, username, and roles from:
    1. X-User-Roles header (format: TenantA:Admin,TenantB:User,User:john_doe)
    2. Authentication token (fallback)
    
    When X-Tenant-ID header is provided, validates user has access to that tenant.
    
    Also creates a tracing span with tenant metadata.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request and extract tenant context.
        
        Validates tenant access when X-Tenant-ID header is provided.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from downstream handlers
        """
        # Set up observability context
        async with RequestContextManager(request):
            # First, try to parse roles from X-User-Roles header
            roles_header_value = request.headers.get(auth_settings.roles_header)
            header_parsed = self._parse_roles_header(roles_header_value)
            
            # Extract user identity from token (for fallback values)
            user = await self._extract_user_identity(request)
            
            # Override user identity with header-parsed values if available
            if header_parsed["first_tenant"] or header_parsed["username"]:
                user = self._merge_header_roles(user, header_parsed)
            
            # Check if a specific tenant is requested via X-Tenant-ID header
            # This ALWAYS overrides the tenant_id when provided
            requested_tenant = request.headers.get(auth_settings.tenant_header)
            if requested_tenant:
                # Validate access if enabled and user has tenant restrictions
                if auth_settings.validate_tenant_access and user.accessible_tenants:
                    if not user.has_access_to_tenant(requested_tenant):
                        logger.warning(
                            f"User '{user.username}' does not have access to tenant '{requested_tenant}'. "
                            f"Accessible tenants: {user.accessible_tenants}"
                        )
                        raise HTTPException(
                            status_code=403,
                            detail=f"Access denied to tenant '{requested_tenant}'"
                        )
                # Override tenant_id with requested tenant
                user = self._override_tenant(user, requested_tenant)
            
            # Debug logging for tenant extraction
            logger.info(
                f"[TENANT_DEBUG] header_name='{auth_settings.tenant_header}', "
                f"X-Tenant-ID='{requested_tenant}', "
                f"X-User-Roles='{roles_header_value}', "
                f"final tenant_id='{user.tenant_id}', username='{user.username}'"
            )
            
            # Store in request state
            self._store_in_request_state(request, user)
            
            # Update the current active span (parent HTTP span) with correct tenant_id
            # This is crucial because the TenantSpanProcessor may have set wrong values
            # before the middleware determined the correct tenant from headers
            from opentelemetry import trace
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute(SpanAttributes.TENANT_ID, user.tenant_id)
                current_span.set_attribute(SpanAttributes.USERNAME, user.username)
            
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
    
    def _override_tenant(self, user: UserIdentity, tenant_id: str) -> UserIdentity:
        """
        Create a new UserIdentity with overridden tenant_id.
        
        Args:
            user: Original user identity
            tenant_id: New tenant ID
            
        Returns:
            New UserIdentity with updated tenant_id and role_group
        """
        # Get role group for the new tenant
        tenant_roles = user.get_roles_for_tenant(tenant_id)
        role_group = user.role_group
        
        if tenant_roles:
            # Re-determine role group based on tenant-specific roles
            role_group = auth_service._determine_role_group(tenant_roles)
        
        return UserIdentity(
            username=user.username,
            tenant_id=tenant_id,
            roles=user.roles,
            role_group=role_group,
            email=user.email,
            full_name=user.full_name,
            raw_claims=user.raw_claims,
            accessible_tenants=user.accessible_tenants,
            tenant_roles=user.tenant_roles,
        )
    
    def _parse_roles_header(self, roles_header_value: Optional[str]) -> dict:
        """
        Parse roles header value into structured data.
        
        Expected format: "TenantA:Admin,TenantB:User,User:john_doe"
        - TenantA:Admin -> tenant_id=TenantA, role=Admin
        - User:john_doe -> username=john_doe
        
        Args:
            roles_header_value: Comma-separated roles string
            
        Returns:
            Dict with username, accessible_tenants, tenant_roles, first_tenant
        """
        result = {
            "username": None,
            "accessible_tenants": set(),
            "tenant_roles": {},
            "first_tenant": None,
        }
        
        if not roles_header_value:
            return result
        
        # Split by comma and process each role
        roles = [r.strip() for r in roles_header_value.split(",") if r.strip()]
        
        for role_entry in roles:
            if ":" not in role_entry:
                continue
            
            parts = role_entry.split(":", 1)
            if len(parts) != 2:
                continue
            
            key, value = parts[0].strip(), parts[1].strip()
            
            if not key or not value:
                continue
            
            # Check if this is a username entry
            if key.lower() == "user":
                result["username"] = value
            else:
                # This is a tenant:role entry
                tenant_id = key
                tenant_role = value
                
                result["accessible_tenants"].add(tenant_id)
                
                if tenant_id not in result["tenant_roles"]:
                    result["tenant_roles"][tenant_id] = []
                result["tenant_roles"][tenant_id].append(tenant_role)
                
                # Track first tenant for default
                if result["first_tenant"] is None:
                    result["first_tenant"] = tenant_id
        
        return result
    
    def _merge_header_roles(self, user: UserIdentity, header_parsed: dict) -> UserIdentity:
        """
        Merge header-parsed roles into user identity.
        
        Args:
            user: Original user identity from token
            header_parsed: Parsed roles from X-User-Roles header
            
        Returns:
            New UserIdentity with merged data
        """
        # Use header values if available, otherwise fall back to user values
        username = header_parsed["username"] or user.username
        tenant_id = header_parsed["first_tenant"] or user.tenant_id
        accessible_tenants = header_parsed["accessible_tenants"] or user.accessible_tenants
        tenant_roles = header_parsed["tenant_roles"] or user.tenant_roles
        
        # Determine role group based on tenant-specific roles
        current_tenant_roles = tenant_roles.get(tenant_id, [])
        role_group = auth_service._determine_role_group(current_tenant_roles) if current_tenant_roles else user.role_group
        
        return UserIdentity(
            username=username,
            tenant_id=tenant_id,
            roles=user.roles,
            role_group=role_group,
            email=user.email,
            full_name=user.full_name,
            raw_claims=user.raw_claims,
            accessible_tenants=accessible_tenants,
            tenant_roles=tenant_roles,
        )
    
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
