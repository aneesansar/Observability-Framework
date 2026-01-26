"""
Authentication Configuration.

Centralized settings for Keycloak and authentication.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    """
    Authentication configuration settings.
    
    All settings can be overridden via environment variables.
    """
    
    # Keycloak Server
    keycloak_server_url: Optional[str] = Field(
        default=None,
        alias="KEYCLOAK_SERVER_URL",
        description="Keycloak server URL"
    )
    realm_name: Optional[str] = Field(
        default=None,
        alias="REALM_NAME",
        description="Keycloak realm name"
    )
    client_id: Optional[str] = Field(
        default=None,
        alias="CLIENT_ID",
        description="Keycloak client ID"
    )
    client_secret_key: Optional[str] = Field(
        default=None,
        alias="CLIENT_SECRET_KEY",
        description="Keycloak client secret"
    )
    callback_uri: str = Field(
        default="http://localhost:8000/callback",
        alias="CALLBACK_URI",
        description="OAuth callback URI"
    )
    
    # Auth Behavior
    disable_auth: bool = Field(
        default=False,
        alias="DISABLE_AUTH",
        description="Disable authentication (dev mode)"
    )
    role_debug: bool = Field(
        default=False,
        alias="ROLE_DEBUG",
        description="Enable role debugging output"
    )
    
    # Tenant/Roles Headers
    roles_header: str = Field(
        default="X-User-Roles",
        alias="ROLES_HEADER",
        description="Header name for user roles (format: TenantA:Admin,TenantB:User,User:username)"
    )
    tenant_header: str = Field(
        default="x-tenant-id",
        alias="TENANT_HEADER",
        description="Header name for explicit tenant ID override"
    )
    validate_tenant_access: bool = Field(
        default=True,
        alias="VALIDATE_TENANT_ACCESS",
        description="Validate that user has access to requested tenant via roles"
    )
    
    # Role Mappings (stored as comma-separated strings, parsed on access)
    admin_roles_str: str = Field(
        default="admin,Admin",
        alias="ADMIN_ROLES",
        description="Roles that grant admin access (comma-separated)"
    )
    manager_roles_str: str = Field(
        default="manager,Manager",
        alias="MANAGER_ROLES",
        description="Roles that grant manager access (comma-separated)"
    )
    developer_roles_str: str = Field(
        default="developer,Developer",
        alias="DEVELOPER_ROLES",
        description="Roles that grant developer access (comma-separated)"
    )
    
    # Attribute Prefixes for Role-Based Visibility
    admin_attr_prefixes_str: str = Field(
        default="Admin,Manager,Developer",
        alias="ADMIN_ATTR_PREFIXES",
        description="Attribute prefixes visible to admins (comma-separated)"
    )
    manager_attr_prefixes_str: str = Field(
        default="Manager,Developer",
        alias="MANAGER_ATTR_PREFIXES",
        description="Attribute prefixes visible to managers (comma-separated)"
    )
    developer_attr_prefixes_str: str = Field(
        default="Developer",
        alias="DEVELOPER_ATTR_PREFIXES",
        description="Attribute prefixes visible to developers (comma-separated)"
    )
    
    @property
    def admin_roles(self) -> List[str]:
        """Get admin roles as list."""
        return [r.strip() for r in self.admin_roles_str.split(",") if r.strip()]
    
    @property
    def manager_roles(self) -> List[str]:
        """Get manager roles as list."""
        return [r.strip() for r in self.manager_roles_str.split(",") if r.strip()]
    
    @property
    def developer_roles(self) -> List[str]:
        """Get developer roles as list."""
        return [r.strip() for r in self.developer_roles_str.split(",") if r.strip()]
    
    @property
    def admin_attr_prefixes(self) -> List[str]:
        """Get admin attribute prefixes as list."""
        return [p.strip() for p in self.admin_attr_prefixes_str.split(",") if p.strip()]
    
    @property
    def manager_attr_prefixes(self) -> List[str]:
        """Get manager attribute prefixes as list."""
        return [p.strip() for p in self.manager_attr_prefixes_str.split(",") if p.strip()]
    
    @property
    def developer_attr_prefixes(self) -> List[str]:
        """Get developer attribute prefixes as list."""
        return [p.strip() for p in self.developer_attr_prefixes_str.split(",") if p.strip()]
    
    @field_validator("disable_auth", "role_debug", mode="before")
    @classmethod
    def parse_bool(cls, v):
        """Parse string booleans."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)
    
    @property
    def normalized_server_url(self) -> Optional[str]:
        """Get server URL with trailing slash."""
        if self.keycloak_server_url:
            return self.keycloak_server_url.rstrip("/") + "/"
        return None
    
    @property
    def is_configured(self) -> bool:
        """Check if Keycloak is properly configured."""
        return all([
            self.keycloak_server_url,
            self.realm_name,
            self.client_id,
        ])
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_auth_settings() -> AuthSettings:
    """
    Get cached auth settings instance.
    
    Returns:
        AuthSettings: Singleton settings instance
    """
    return AuthSettings()


# Convenience alias
auth_settings = get_auth_settings()
