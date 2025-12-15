import logging
from typing import Any, Dict, Optional

from keycloak import KeycloakOpenID

from .config import auth_settings
from .exceptions import AuthServiceUnavailableError, InvalidTokenError
from .models import TokenInfo

logger = logging.getLogger(__name__)


class KeycloakClient:
    """Keycloak client wrapper used by the authentication layer.

    This client uses python-keycloak's ``KeycloakOpenID`` to perform:
    - username/password authentication (token endpoint)
    - token verification/decoding
    - userinfo lookup

    When ``DISABLE_AUTH=true`` the higher-level AuthService bypasses this
    client entirely, so these methods are only used when auth is enabled
    and the Keycloak settings are correctly configured.
    """

    def __init__(self) -> None:
        if not auth_settings.is_configured:
            # If auth is enabled but Keycloak is not configured, this is a setup error.
            if not auth_settings.disable_auth:
                logger.error(
                    "Keycloak is not properly configured. Set DISABLE_AUTH=true for "
                    "local dev or provide KEYCLOAK_* env vars."
                )
                raise AuthServiceUnavailableError(
                    "Keycloak configuration is missing. Check KEYCLOAK_* env vars."
                )

            # Auth is disabled; we will never be called in this mode.
            self._client = None
            return

        server_url = auth_settings.normalized_server_url
        realm_name = auth_settings.realm_name
        client_id = auth_settings.client_id
        client_secret = auth_settings.client_secret_key

        if not (server_url and realm_name and client_id):
            logger.error("Incomplete Keycloak configuration: server_url/realm/client_id missing")
            raise AuthServiceUnavailableError(
                "Incomplete Keycloak configuration. Check KEYCLOAK_SERVER_URL, REALM_NAME, "
                "and CLIENT_ID."
            )

        logger.info(
            "Initializing KeycloakOpenID client for realm '%s' at '%s'", realm_name, server_url
        )

        self._client = KeycloakOpenID(
            server_url=server_url,
            realm_name=realm_name,
            client_id=client_id,
            client_secret_key=client_secret,
        )

    def authenticate(self, username: str, password: str) -> TokenInfo:
        """Authenticate a user against Keycloak and return TokenInfo.

        Raises AuthServiceUnavailableError if communication with Keycloak fails.
        """
        if not self._client:
            raise AuthServiceUnavailableError("Keycloak client not initialized")

        try:
            token_data = self._client.token(username, password)
        except Exception as exc:  # noqa: BLE001
            logger.error("Keycloak authentication failed for %s: %s", username, exc)
            raise AuthServiceUnavailableError(f"Keycloak authentication failed: {exc}") from exc

        # token_data typically contains: access_token, refresh_token, token_type, expires_in, ...
        return TokenInfo(
            access_token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_type=token_data.get("token_type", "bearer"),
            expires_in=token_data.get("expires_in", 0),
        )

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate an access token using Keycloak.

        Returns the token claims on success, or raises InvalidTokenError.
        """
        if not self._client:
            raise AuthServiceUnavailableError("Keycloak client not initialized")

        try:
            # verify=True ensures signature and expiry checks
            claims: Dict[str, Any] = self._client.decode_token(token, verify=True)
            return claims
        except Exception as exc:  # noqa: BLE001
            logger.warning("Token validation failed: %s", exc)
            raise InvalidTokenError(f"Token validation failed: {exc}") from exc

    def get_userinfo(self, token: str) -> Dict[str, Any]:
        """Fetch userinfo for a token from Keycloak.

        Used as a fallback if direct token decoding fails.
        """
        if not self._client:
            raise AuthServiceUnavailableError("Keycloak client not initialized")

        try:
            userinfo: Dict[str, Any] = self._client.userinfo(token)
            return userinfo
        except Exception as exc:  # noqa: BLE001
            logger.warning("Userinfo retrieval failed: %s", exc)
            raise InvalidTokenError(f"Userinfo retrieval failed: {exc}") from exc


_keycloak_client: Optional[KeycloakClient] = None


def get_keycloak_client() -> KeycloakClient:
    """Return a singleton KeycloakClient instance."""
    global _keycloak_client
    if _keycloak_client is None:
        _keycloak_client = KeycloakClient()
    return _keycloak_client


# Convenience alias used across the codebase
keycloak_client: KeycloakClient = get_keycloak_client()
