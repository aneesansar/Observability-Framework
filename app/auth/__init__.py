"""
Authentication Module.

Provides authentication and authorization for the application.

Features:
- Keycloak integration for OAuth2/OIDC
- JWT token validation
- Role-based access control (RBAC)
- Tenant isolation
- FastAPI dependencies for protected routes

Architecture:
    auth/
    ├── __init__.py          # Public API
    ├── config.py            # AuthSettings
    ├── exceptions.py        # Custom exceptions
    ├── models.py            # UserIdentity, TokenInfo, RoleGroup
    ├── keycloak_client.py   # Keycloak wrapper
    ├── service.py           # AuthService
    ├── dependencies.py      # FastAPI dependencies
    └── middleware.py        # HTTP middleware

Usage:
    from app.auth import get_current_user, require_role, UserIdentity
    
    @app.get("/protected")
    async def protected_endpoint(user: UserIdentity = Depends(get_current_user)):
        return {"tenant": user.tenant_id}
    
    @app.get("/admin-only")
    async def admin_endpoint(user: UserIdentity = Depends(require_role("admin"))):
        return {"admin": True}
"""

# Configuration
from .config import (
    auth_settings,
    get_auth_settings,
    AuthSettings,
)

# Exceptions
from .exceptions import (
    AuthError,
    InvalidTokenError,
    TokenExpiredError,
    InvalidCredentialsError,
    AuthServiceUnavailableError,
    InsufficientPermissionsError,
    TenantAccessDeniedError,
)

# Models
from .models import (
    UserIdentity,
    TokenInfo,
    RoleGroup,
)

# Keycloak Client
from .keycloak_client import (
    KeycloakClient,
    keycloak_client,
    get_keycloak_client,
)

# Service
from .service import (
    AuthService,
    auth_service,
    get_auth_service,
)

# Dependencies
from .dependencies import (
    oauth2_scheme,
    get_current_user,
    get_current_user_optional,
    require_role,
    require_role_group,
    require_admin,
    require_manager_or_above,
    require_same_tenant,
)

# Middleware
from .middleware import (
    TenantMiddleware,
    tenant_telemetry_middleware,
)


__all__ = [
    # Config
    "auth_settings",
    "get_auth_settings",
    "AuthSettings",
    # Exceptions
    "AuthError",
    "InvalidTokenError",
    "TokenExpiredError",
    "InvalidCredentialsError",
    "AuthServiceUnavailableError",
    "InsufficientPermissionsError",
    "TenantAccessDeniedError",
    # Models
    "UserIdentity",
    "TokenInfo",
    "RoleGroup",
    # Keycloak
    "KeycloakClient",
    "keycloak_client",
    "get_keycloak_client",
    # Service
    "AuthService",
    "auth_service",
    "get_auth_service",
    # Dependencies
    "oauth2_scheme",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "require_role_group",
    "require_admin",
    "require_manager_or_above",
    "require_same_tenant",
    # Middleware
    "TenantMiddleware",
    "tenant_telemetry_middleware",
]
