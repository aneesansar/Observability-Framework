from fastapi import FastAPI, Depends, Form, HTTPException, Request, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from keycloak import KeycloakOpenID, KeycloakAuthenticationError
from dotenv import load_dotenv
from opentelemetry import trace
from datetime import datetime
import logging
import uuid
import os

# Import routers
from app.routes.llm_routes import router as llm_router
from app.routes.observability_routes import router as observability_router
from app.rag_pipeline import generate_response
from observe import setup_observability
from observe.instrumentation import instrument_all_in_package

# ============================================================
# === Load environment variables ===
# ============================================================
load_dotenv()

KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
REALM_NAME = os.getenv("REALM_NAME")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET_KEY = os.getenv("CLIENT_SECRET_KEY")
CALLBACK_URI = os.getenv("CALLBACK_URI", "http://localhost:8000/callback")

# ============================================================
# === Initialize FastAPI App ===
# ============================================================
app = FastAPI(title="RAG Observability Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# === Observability Setup ===
# ============================================================
setup_observability(app)
instrument_all_in_package(os.path.join(os.path.dirname(__file__), ""), "app")

# ============================================================
# === Keycloak Setup ===
# ============================================================
keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_SERVER_URL,
    realm_name=REALM_NAME,
    client_id=CLIENT_ID,
    client_secret_key=CLIENT_SECRET_KEY
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# ============================================================
# === MODELS ===
# ============================================================
class QueryRequest(BaseModel):
    query: str


class LogEntry(BaseModel):
    id: str
    timestamp: str
    level: str
    service: str
    tenant_id: str
    message: str
    metadata: dict | None = None


# ============================================================
# === In-Memory Log Storage ===
# ============================================================
log_buffer: list[LogEntry] = []


class InMemoryLogHandler(logging.Handler):
    """
    Custom log handler that stores logs in memory with tenant/user metadata.
    """
    def emit(self, record: logging.LogRecord):
        try:
            log_entry = LogEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat() + "Z",
                level=record.levelname,
                service="backend",
                tenant_id=getattr(record, "tenant_id", "anonymous"),
                message=record.getMessage(),
                metadata={
                    "filename": record.pathname,
                    "line": record.lineno,
                    "function": record.funcName,
                    "exception": getattr(record, "exc_text", None),
                },
            )
            log_buffer.append(log_entry)
            if len(log_buffer) > 500:  # limit history
                log_buffer.pop(0)
        except Exception:
            pass


# Setup logger
logger = logging.getLogger("backend")
logger.setLevel(logging.DEBUG)
logger.addHandler(InMemoryLogHandler())
logger.propagate = True

# ============================================================
# === TOKEN VALIDATION DEPENDENCY ===
# ============================================================
def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validate JWT token and return Keycloak user info.
    """
    try:
        user_info = keycloak_openid.userinfo(token)
        return user_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
        )

# ============================================================
# === TENANT TELEMETRY + LOGGING MIDDLEWARE ===
# ============================================================
@app.middleware("http")
async def tenant_telemetry_middleware(request: Request, call_next):
    """
    Middleware:
      - Extract tenant (username) from Keycloak token
      - Tag logs and traces with tenant info
      - Capture request/response lifecycle
    """
    tracer = trace.get_tracer(__name__)
    tenant_id = "anonymous"
    username = "guest"

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
        try:
            user_info = keycloak_openid.userinfo(token)
            username = user_info.get("preferred_username", "guest")
            tenant_id = username
        except Exception:
            pass

    request.state.tenant_id = tenant_id
    request.state.username = username

    logger.info(f"Incoming request {request.method} {request.url.path}", extra={"tenant_id": tenant_id})

    try:
        response = await call_next(request)
        logger.info(
            f"Completed request {request.method} {request.url.path} with status {response.status_code}",
            extra={"tenant_id": tenant_id}
        )
        return response
    except Exception as exc:
        logger.error(
            f"Unhandled error in {request.url.path}: {exc}",
            exc_info=True,
            extra={"tenant_id": tenant_id}
        )
        raise

# ============================================================
# === LOGIN ENDPOINT ===
# ============================================================
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Log in to Keycloak using username/password (Direct Access Grant).
    Returns access_token and refresh_token.
    """
    try:
        token = keycloak_openid.token(username, password)
        logger.info(f"User {username} logged in successfully", extra={"tenant_id": username})
        return token
    except KeycloakAuthenticationError as e:
        logger.warning(f"Failed login attempt for {username}", extra={"tenant_id": username})
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {e.response_body}")

# ============================================================
# === QUERY ENDPOINT (PROTECTED) ===
# ============================================================
@app.post("/query")
async def handle_query(request: Request, query_data: QueryRequest, user=Depends(get_current_user)):
    """
    Query endpoint that generates an LLM response.
    Uses the Keycloak username as tenant_id automatically.
    """
    tenant_id = getattr(request.state, "tenant_id", "anonymous")
    try:
        logger.info(f"Processing query for tenant {tenant_id}", extra={"tenant_id": tenant_id})
        response = generate_response(query_data.query, tenant_id)
        logger.info(f"Response generated for tenant {tenant_id}", extra={"tenant_id": tenant_id})
        return {"tenant": tenant_id, "response": response}
    except Exception as exc:
        logger.error(f"Error while generating response: {exc}", exc_info=True, extra={"tenant_id": tenant_id})
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ============================================================
# === LOGS ENDPOINT ===
# ============================================================
@app.get("/logs")
async def get_logs(level: str = Query(None, description="Filter logs by level: info, warning, error")):
    """
    Return live in-memory logs.
    Optional filter: ?level=info|warning|error
    """
    if level:
        filtered = [log for log in log_buffer if log.level.lower() == level.lower()]
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)
    return sorted(log_buffer, key=lambda x: x.timestamp, reverse=True)

# ============================================================
# === AUTH DEBUG (OPTIONAL) ===
# ============================================================
@app.get("/auth/debug")
async def auth_debug(request: Request, user=Depends(get_current_user)):
    """
    Returns user info decoded from access token, including tenant_id.
    """
    return {
        "username": getattr(request.state, "username", "guest"),
        "tenant_id": getattr(request.state, "tenant_id", "anonymous"),
        "keycloak_userinfo": user
    }

# ============================================================
# === INCLUDE ROUTERS ===
# ============================================================
app.include_router(llm_router)
app.include_router(observability_router)
