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
        
        Parses roles in format "TenantA:Admin,TenantB:User,User:john_doe":
        - TenantA:Admin -> tenant_id=TenantA, role=Admin
        - User:john_doe -> username=john_doe
        
        Args:
            claims: Token claims dictionary
            
        Returns:
            UserIdentity object
        """
        email = claims.get("email")
        full_name = claims.get("name")
        
        # Extract raw roles from claims
        raw_roles = self._extract_roles(claims)
        
        # Parse roles to extract tenant_id, username, and tenant-specific roles
        parsed = self._parse_tenant_roles(raw_roles)
        
        # Username: from User:xxx role, or preferred_username, or "unknown"
        username = (
            parsed["username"] 
            or claims.get("preferred_username") 
            or "unknown"
        )
        
        # Tenant ID: from first tenant in roles, or explicit claim, or username
        tenant_id = (
            parsed["first_tenant"]
            or claims.get("tenant_id")
            or username
        )
        
        # Determine role group from the role for current tenant
        current_tenant_roles = parsed["tenant_roles"].get(tenant_id, [])
        role_group = self._determine_role_group(current_tenant_roles or raw_roles)
        
        # Debug logging
        if auth_settings.role_debug:
            logger.info(
                f"[ROLE_DEBUG] tenant={tenant_id} username={username} "
                f"raw_roles={raw_roles} tenant_roles={parsed['tenant_roles']} "
                f"role_group={role_group.value}"
            )
        
        return UserIdentity(
            username=username,
            tenant_id=tenant_id,
            roles=raw_roles,
            role_group=role_group,
            email=email,
            full_name=full_name,
            raw_claims=claims,
            accessible_tenants=parsed["accessible_tenants"],
            tenant_roles=parsed["tenant_roles"],
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
    
    def _parse_tenant_roles(self, roles: List[str]) -> Dict[str, Any]:
        """
        Parse roles to extract tenant information.
        
        Expected format: "TenantA:Admin", "TenantB:User", "User:john_doe"
        - TenantA:Admin -> tenant_id=TenantA, role=Admin
        - User:john_doe -> username=john_doe
        
        Args:
            roles: List of role strings
            
        Returns:
            Dict with:
            - username: Extracted username (from User:xxx)
            - accessible_tenants: Set of tenant IDs user has access to
            - tenant_roles: Dict mapping tenant_id to list of roles
            - first_tenant: First tenant found (for default)
        """
        result = {
            "username": None,
            "accessible_tenants": set(),
            "tenant_roles": {},
            "first_tenant": None,
        }
        
        for role in roles:
            if ":" not in role:
                continue
            
            parts = role.split(":", 1)
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
