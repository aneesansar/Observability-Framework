"""
Authentication Dependencies.

FastAPI dependencies for authentication and authorization.
"""

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from .config import auth_settings
from .exceptions import AuthError, InsufficientPermissionsError
from .models import RoleGroup, UserIdentity
from .service import auth_service

logger = logging.getLogger(__name__)

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserIdentity:
    """
    FastAPI dependency to get the current authenticated user.
    
    Validates the bearer token and returns user identity.
    If the middleware has already set tenant_id from x-tenant-id header,
    that value takes precedence over the token-derived tenant_id.
    
    Args:
        request: FastAPI request object
        token: Bearer token from Authorization header
        
    Returns:
        UserIdentity of the authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if middleware already set user_identity with correct tenant
    if hasattr(request.state, "user_identity"):
        return request.state.user_identity
    
    # Dev mode bypass
    if auth_settings.disable_auth:
        user = UserIdentity.dev_user()
        user = _apply_tenant_override(request, user)
        _store_in_request_state(request, user)
        return user
    
    # Check for token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = auth_service.validate_token(token)
        # Apply tenant override from x-tenant-id header if present
        user = _apply_tenant_override(request, user)
        _store_in_request_state(request, user)
        return user
    except AuthError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserIdentity]:
    """
    FastAPI dependency to optionally get the current user.
    
    Returns None if no token is provided instead of raising an error.
    Useful for endpoints that work with or without authentication.
    
    Args:
        request: FastAPI request object
        token: Bearer token (optional)
        
    Returns:
        UserIdentity or None
    """
    if not token and not auth_settings.disable_auth:
        return None
    
    try:
        return await get_current_user(request, token)
    except HTTPException:
        return None


def require_role(*required_roles: str):
    """
    Dependency factory for role-based access control.
    
    Args:
        required_roles: Role names that are allowed access
        
    Returns:
        FastAPI dependency function
        
    Example:
        @app.get("/admin")
        async def admin_endpoint(user: UserIdentity = Depends(require_role("admin"))):
            pass
    """
    async def role_checker(
        user: UserIdentity = Depends(get_current_user),
    ) -> UserIdentity:
        if not any(user.has_role(role) for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(required_roles)}",
            )
        return user
    
    return role_checker


def require_role_group(*allowed_groups: RoleGroup):
    """
    Dependency factory for role group-based access control.
    
    Args:
        allowed_groups: RoleGroup values that are allowed access
        
    Returns:
        FastAPI dependency function
        
    Example:
        @app.get("/managers")
        async def manager_endpoint(
            user: UserIdentity = Depends(require_role_group(RoleGroup.ADMIN, RoleGroup.MANAGER))
        ):
            pass
    """
    async def group_checker(
        user: UserIdentity = Depends(get_current_user),
    ) -> UserIdentity:
        if user.role_group not in allowed_groups:
            allowed_names = [g.value for g in allowed_groups]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role group: {', '.join(allowed_names)}",
            )
        return user
    
    return group_checker


def require_admin():
    """Dependency that requires admin role."""
    return require_role_group(RoleGroup.ADMIN)


def require_manager_or_above():
    """Dependency that requires manager or admin role."""
    return require_role_group(RoleGroup.ADMIN, RoleGroup.MANAGER)


def require_same_tenant(tenant_id_param: str = "tenant_id"):
    """
    Dependency factory for tenant isolation.
    
    Ensures the user can only access their own tenant's resources.
    Admins can access any tenant.
    
    Args:
        tenant_id_param: Name of the path/query parameter containing tenant_id
        
    Returns:
        FastAPI dependency function
    """
    async def tenant_checker(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> UserIdentity:
        # Admins can access any tenant
        if user.is_admin:
            return user
        
        # Get tenant_id from path or query params
        requested_tenant = (
            request.path_params.get(tenant_id_param)
            or request.query_params.get(tenant_id_param)
        )
        
        if requested_tenant and requested_tenant != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to this tenant is denied",
            )
        
        return user
    
    return tenant_checker


def _apply_tenant_override(request: Request, user: UserIdentity) -> UserIdentity:
    """
    Apply tenant override from x-tenant-id header if present.
    
    Parses roles in TenantA:Admin format to validate access.
    
    Args:
        request: FastAPI request object
        user: Original user identity
        
    Returns:
        UserIdentity with tenant_id from header if valid
    """
    from .config import auth_settings
    
    # Get tenant from header
    requested_tenant = request.headers.get(auth_settings.tenant_header)
    if not requested_tenant:
        return user
    
    # If user has accessible_tenants, validate access
    if user.accessible_tenants and requested_tenant not in user.accessible_tenants:
        # User doesn't have access to this tenant
        # But if validate_tenant_access is False, allow it anyway
        if auth_settings.validate_tenant_access:
            logger.warning(
                f"User '{user.username}' requested tenant '{requested_tenant}' "
                f"but only has access to: {user.accessible_tenants}"
            )
            # Don't override, keep original tenant
            return user
    
    # Override tenant_id
    if requested_tenant != user.tenant_id:
        # Get role for the new tenant
        tenant_roles = user.get_roles_for_tenant(requested_tenant)
        role_group = user.role_group
        if tenant_roles:
            role_group = auth_service._determine_role_group(tenant_roles)
        
        return UserIdentity(
            username=user.username,
            tenant_id=requested_tenant,
            roles=user.roles,
            role_group=role_group,
            email=user.email,
            full_name=user.full_name,
            raw_claims=user.raw_claims,
            accessible_tenants=user.accessible_tenants,
            tenant_roles=user.tenant_roles,
        )
    
    return user


def _store_in_request_state(request: Request, user: UserIdentity) -> None:
    """
    Store user identity in request state for middleware access.
    
    Args:
        request: FastAPI request object
        user: User identity to store
    """
    request.state.tenant_id = user.tenant_id
    request.state.username = user.username
    request.state.user_roles = user.roles
    request.state.role_group = user.role_group.value
    request.state.user_identity = user
