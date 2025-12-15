"""
Authentication Models.

Data models for user identity and authentication.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RoleGroup(str, Enum):
    """User role groups for access control."""
    
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    ANONYMOUS = "anonymous"


@dataclass
class UserIdentity:
    """
    Represents an authenticated user's identity.
    
    Contains all relevant information extracted from the authentication token.
    """
    
    username: str
    tenant_id: str
    roles: List[str] = field(default_factory=list)
    role_group: RoleGroup = RoleGroup.DEVELOPER
    email: Optional[str] = None
    full_name: Optional[str] = None
    raw_claims: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role_group == RoleGroup.ADMIN
    
    @property
    def is_manager(self) -> bool:
        """Check if user has manager or higher role."""
        return self.role_group in (RoleGroup.ADMIN, RoleGroup.MANAGER)
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated (not anonymous)."""
        return self.role_group != RoleGroup.ANONYMOUS
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role.lower() in [r.lower() for r in self.roles]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "username": self.username,
            "tenant_id": self.tenant_id,
            "roles": self.roles,
            "role_group": self.role_group.value,
            "email": self.email,
            "full_name": self.full_name,
        }
    
    @classmethod
    def anonymous(cls) -> "UserIdentity":
        """Create an anonymous user identity."""
        return cls(
            username="anonymous",
            tenant_id="anonymous",
            roles=[],
            role_group=RoleGroup.ANONYMOUS,
        )
    
    @classmethod
    def dev_user(cls) -> "UserIdentity":
        """Create a dev mode user identity."""
        return cls(
            username="dev",
            tenant_id="dev",
            roles=["Developer"],
            role_group=RoleGroup.DEVELOPER,
        )


@dataclass
class TokenInfo:
    """
    Information about an authentication token.
    """
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }
        if self.refresh_token:
            result["refresh_token"] = self.refresh_token
        return result
