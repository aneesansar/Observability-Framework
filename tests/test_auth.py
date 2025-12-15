"""
Tests for Authentication Module.

Tests for auth models, service, and dependencies.
"""

import pytest


class TestUserIdentity:
    """Tests for UserIdentity model."""
    
    def test_user_identity_creation(self, mock_user):
        """Test UserIdentity creation."""
        assert mock_user.username == "test_user"
        assert mock_user.tenant_id == "test_tenant"
        assert "Developer" in mock_user.roles
    
    def test_is_admin(self, mock_user, mock_admin_user):
        """Test is_admin property."""
        assert not mock_user.is_admin
        assert mock_admin_user.is_admin
    
    def test_is_manager(self, mock_user, mock_admin_user):
        """Test is_manager property."""
        assert not mock_user.is_manager
        assert mock_admin_user.is_manager  # Admin is also manager
    
    def test_has_role(self, mock_user):
        """Test has_role method."""
        assert mock_user.has_role("Developer")
        assert mock_user.has_role("developer")  # Case insensitive
        assert not mock_user.has_role("Admin")
    
    def test_to_dict(self, mock_user):
        """Test serialization."""
        result = mock_user.to_dict()
        
        assert result["username"] == "test_user"
        assert result["tenant_id"] == "test_tenant"
        assert "Developer" in result["roles"]
    
    def test_anonymous_user(self):
        """Test anonymous user creation."""
        from app.auth import UserIdentity, RoleGroup
        
        user = UserIdentity.anonymous()
        
        assert user.username == "anonymous"
        assert user.role_group == RoleGroup.ANONYMOUS
        assert not user.is_authenticated
    
    def test_dev_user(self):
        """Test dev user creation."""
        from app.auth import UserIdentity
        
        user = UserIdentity.dev_user()
        
        assert user.username == "dev"
        assert user.tenant_id == "dev"


class TestRoleGroup:
    """Tests for RoleGroup enum."""
    
    def test_role_group_values(self):
        """Test RoleGroup enum values."""
        from app.auth import RoleGroup
        
        assert RoleGroup.ADMIN.value == "admin"
        assert RoleGroup.MANAGER.value == "manager"
        assert RoleGroup.DEVELOPER.value == "developer"
        assert RoleGroup.ANONYMOUS.value == "anonymous"


class TestTokenInfo:
    """Tests for TokenInfo model."""
    
    def test_token_info_creation(self):
        """Test TokenInfo creation."""
        from app.auth import TokenInfo
        
        token = TokenInfo(
            access_token="test_token",
            refresh_token="refresh_token",
        )
        
        assert token.access_token == "test_token"
        assert token.token_type == "bearer"
        assert token.expires_in == 3600
    
    def test_token_info_to_dict(self):
        """Test TokenInfo serialization."""
        from app.auth import TokenInfo
        
        token = TokenInfo(access_token="test")
        result = token.to_dict()
        
        assert "access_token" in result
        assert "token_type" in result


class TestAuthExceptions:
    """Tests for auth exceptions."""
    
    def test_invalid_token_error(self):
        """Test InvalidTokenError."""
        from app.auth import InvalidTokenError
        
        error = InvalidTokenError("Token expired")
        assert error.status_code == 401
        assert "expired" in error.message
    
    def test_insufficient_permissions_error(self):
        """Test InsufficientPermissionsError."""
        from app.auth import InsufficientPermissionsError
        
        error = InsufficientPermissionsError()
        assert error.status_code == 403


class TestAuthSettings:
    """Tests for auth settings."""
    
    def test_auth_settings_loads(self):
        """Test auth settings load."""
        from app.auth import auth_settings
        
        assert auth_settings is not None
    
    def test_role_lists(self):
        """Test role lists are parsed."""
        from app.auth import auth_settings
        
        assert isinstance(auth_settings.admin_roles, list)
        assert len(auth_settings.admin_roles) > 0
