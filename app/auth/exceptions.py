"""
Authentication Exceptions.

Custom exceptions for authentication and authorization errors.
"""

from typing import Any, Dict, Optional


class AuthError(Exception):
    """Base exception for authentication errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 401,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidTokenError(AuthError):
    """Raised when a token is invalid or expired."""
    
    def __init__(self, message: str = "Invalid or expired token"):
        super().__init__(message, status_code=401)


class TokenExpiredError(AuthError):
    """Raised when a token has expired."""
    
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, status_code=401)


class InvalidCredentialsError(AuthError):
    """Raised when username/password is incorrect."""
    
    def __init__(self, message: str = "Invalid username or password"):
        super().__init__(message, status_code=401)


class AuthServiceUnavailableError(AuthError):
    """Raised when the authentication service is unavailable."""
    
    def __init__(self, message: str = "Authentication service unavailable"):
        super().__init__(message, status_code=503)


class InsufficientPermissionsError(AuthError):
    """Raised when user lacks required permissions."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)


class TenantAccessDeniedError(AuthError):
    """Raised when user tries to access another tenant's resources."""
    
    def __init__(self, message: str = "Access to this tenant is denied"):
        super().__init__(message, status_code=403)
