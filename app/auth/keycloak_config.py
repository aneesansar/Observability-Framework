from dotenv import load_dotenv
from keycloak import KeycloakOpenID, KeycloakAuthenticationError
from fastapi import Depends, HTTPException, status, Request, Form
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from opentelemetry import trace
import os

# === Load environment variables ===
load_dotenv()

# === Settings ===
class Settings:
    KEYCLOAK_SERVER_URL: str = os.getenv("KEYCLOAK_SERVER_URL")
    REALM_NAME: str = os.getenv("REALM_NAME")
    CLIENT_ID: str = os.getenv("CLIENT_ID")
    CLIENT_SECRET_KEY: str = os.getenv("CLIENT_SECRET_KEY", "")
    CALLBACK_URI: str = os.getenv("CALLBACK_URI", "http://localhost:8000/callback")

settings = Settings()

# === Keycloak client ===
keycloak_openid = KeycloakOpenID(
    server_url=settings.KEYCLOAK_SERVER_URL,
    realm_name=settings.REALM_NAME,
    client_id=settings.CLIENT_ID,
    client_secret_key=settings.CLIENT_SECRET_KEY
)

# === Token scheme for protected endpoints ===
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# === Login endpoint ===
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Authenticate with Keycloak using username/password
    and return access and refresh tokens.
    """
    try:
        token = keycloak_openid.token(username, password)
        return token
    except KeycloakAuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid username or password")


# === Verify token and extract user ===
def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Decode and validate the token, return user identity and claims.
    """
    try:
        # Verify signature & expiration
        options = {"verify_aud": False, "verify_exp": True}
        user_info = keycloak_openid.userinfo(token)
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# === Middleware for tenant/user tracing ===
async def tenant_telemetry_middleware(request: Request, call_next):
    tracer = trace.get_tracer(__name__)
    tenant_id = "anonymous"
    username = "guest"

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
        try:
            user_info = keycloak_openid.userinfo(token)
            username = user_info.get("preferred_username")
            tenant_id = user_info.get("tenant_id", username)
        except Exception:
            pass

    with tracer.start_as_current_span("tenant_request") as span:
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("user.name", username)
        span.set_attribute("http.path", request.url.path)
        response = await call_next(request)
        return response
