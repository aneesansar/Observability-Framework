"""
Authentication Endpoints.

Login, token refresh, and user info endpoints.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from app.auth import (
    auth_service,
    get_current_user,
    UserIdentity,
    AuthError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
) -> Dict[str, Any]:
    """
    Authenticate with username and password.
    
    Returns access and refresh tokens.
    """
    try:
        token_info = await auth_service.login(username, password)
        logger.info(f"User {username} logged in successfully")
        return token_info.to_dict()
    except AuthError as e:
        logger.warning(f"Login failed for {username}: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Login error for {username}: {e}")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")


@router.get("/me")
async def get_current_user_info(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get current user information.
    
    Returns user identity from the access token.
    """
    return user.to_dict()


@router.get("/debug")
async def auth_debug(
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Debug endpoint for authentication.
    
    Returns detailed user info including raw claims.
    """
    return {
        "username": user.username,
        "tenant_id": user.tenant_id,
        "roles": user.roles,
        "role_group": user.role_group.value,
        "is_admin": user.is_admin,
        "is_manager": user.is_manager,
    }
