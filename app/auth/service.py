"""
Authentication Service.

Core authentication logic and user identity management.
"""

import logging
from typing import Any, Dict, List, Optional

from .config import auth_settings
from .exceptions import InvalidTokenError
from .keycloak_client import keycloak_client
from .models import RoleGroup, TokenInfo, UserIdentity

logger = logging.getLogger(__name__)


class AuthService:
    """
    Authentication service for user identity management.
    
    Provides:
    - User authentication (login)
    - Token validation
    - User identity extraction
    - Role-based access control helpers
    """
    
    _instance: Optional["AuthService"] = None
    
    def __new__(cls) -> "AuthService":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the auth service."""
        self._keycloak = keycloak_client
    
    async def login(self, username: str, password: str) -> TokenInfo:
        """
        Authenticate user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            TokenInfo with access and refresh tokens
        """
        # Dev mode bypass
        if auth_settings.disable_auth:
            logger.info(f"Dev mode login for {username}")
            return TokenInfo(
                access_token="dev-token",
                refresh_token="dev-refresh",
                token_type="bearer",
                expires_in=3600,
            )
        
        logger.debug(f"Attempting login for {username}")
        token_info = self._keycloak.authenticate(username, password)
        logger.info(f"User {username} logged in successfully")
        return token_info
    
    def validate_token(self, token: str) -> UserIdentity:
        """
        Validate a token and extract user identity.
        
        Args:
            token: JWT access token
            
        Returns:
            UserIdentity with user information
            
        Raises:
            InvalidTokenError: If token is invalid
        """
        # Dev mode bypass
        if auth_settings.disable_auth:
            return UserIdentity.dev_user()
        
        # Try to decode token first (faster, has full claims)
        try:
            claims = self._keycloak.decode_token(token)
            return self._extract_identity(claims)
        except InvalidTokenError:
            pass
        
        # Fallback to userinfo endpoint
        try:
            claims = self._keycloak.get_userinfo(token)
            return self._extract_identity(claims)
        except Exception as e:
            raise InvalidTokenError(f"Invalid or expired token: {e}")
    
    def _extract_identity(self, claims: Dict[str, Any]) -> UserIdentity:
        """
        Extract UserIdentity from token claims.
        
        Args:
            claims: Token claims dictionary
            
        Returns:
            UserIdentity object
        """
        username = claims.get("preferred_username", "unknown")
        tenant_id = claims.get("tenant_id") or username
        email = claims.get("email")
        full_name = claims.get("name")
        
        # Extract roles
        roles = self._extract_roles(claims)
        role_group = self._determine_role_group(roles)
        
        # Debug logging
        if auth_settings.role_debug:
            logger.info(
                f"[ROLE_DEBUG] tenant={tenant_id} roles={roles} "
                f"role_group={role_group.value}"
            )
        
        return UserIdentity(
            username=username,
            tenant_id=tenant_id,
            roles=roles,
            role_group=role_group,
            email=email,
            full_name=full_name,
            raw_claims=claims,
        )
    
    def _extract_roles(self, claims: Dict[str, Any]) -> List[str]:
        """
        Extract all roles from token claims.
        
        Args:
            claims: Token claims dictionary
            
        Returns:
            List of role names
        """
        roles = []
        
        # Realm roles
        realm_access = claims.get("realm_access", {})
        realm_roles = realm_access.get("roles", [])
        roles.extend(realm_roles)
        
        # Resource/client roles
        resource_access = claims.get("resource_access", {})
        for resource in resource_access.values():
            resource_roles = resource.get("roles", [])
            roles.extend(resource_roles)
        
        # Deduplicate
        return list(set(roles))
    
    def _determine_role_group(self, roles: List[str]) -> RoleGroup:
        """
        Determine the role group from a list of roles.
        
        Args:
            roles: List of role names
            
        Returns:
            RoleGroup enum value
        """
        roles_lower = [r.lower() for r in roles]
        
        # Check admin roles
        admin_roles = [r.lower() for r in auth_settings.admin_roles]
        if any(r in admin_roles for r in roles_lower):
            return RoleGroup.ADMIN
        
        # Check manager roles
        manager_roles = [r.lower() for r in auth_settings.manager_roles]
        if any(r in manager_roles for r in roles_lower):
            return RoleGroup.MANAGER
        
        # Check developer roles
        developer_roles = [r.lower() for r in auth_settings.developer_roles]
        if any(r in developer_roles for r in roles_lower):
            return RoleGroup.DEVELOPER
        
        # Default to developer
        return RoleGroup.DEVELOPER
    
    def get_allowed_attr_prefixes(self, role_group: RoleGroup) -> List[str]:
        """
        Get allowed attribute prefixes for a role group.
        
        Used for filtering telemetry data based on user role.
        
        Args:
            role_group: User's role group
            
        Returns:
            List of allowed attribute prefixes
        """
        if role_group == RoleGroup.ADMIN:
            return auth_settings.admin_attr_prefixes
        elif role_group == RoleGroup.MANAGER:
            return auth_settings.manager_attr_prefixes
        else:
            return auth_settings.developer_attr_prefixes


def get_auth_service() -> AuthService:
    """
    Get the auth service instance.
    
    Returns:
        AuthService instance
    """
    return AuthService()


# Convenience alias
auth_service = get_auth_service()
