# SpeedyGPT Observe Backend

FastAPI backend showcasing tenant-aware LLM observability with tracing, metrics, and logging. It instruments LLM calls, exposes in-memory traces for quick inspection, and integrates with Prometheus metrics. Authentication is handled via Keycloak (Direct Access Grant flow).

## Features
- **FastAPI service** with CORS for a local frontend (`http://localhost:3000`).
- **LLM reasoning telemetry** captured per request with attributes like latency, confidence, hallucination score, and reasoning steps.
- **In-memory traces** exported for immediate inspection via API.
- **Prometheus metrics** exposed at `/metrics` (toggle via env).
- **Keycloak auth** for protected endpoints.

## Tech Stack
- **API**: FastAPI + Uvicorn
- **Observability**: OpenTelemetry SDK, FastAPI instrumentation, custom in-memory span exporter, Prometheus client + instrumentator
- **LLM / NLP**: `sentence-transformers` (+ `torch`), simple similarity/consistency checks
- **Auth**: Keycloak (via `keycloak` Python client)

## Directory
- `app/`
  - `main.py` — FastAPI app, middleware, routes include, Keycloak integration
  - `routes/llm_routes.py` — LLM metrics views from spans
  - `routes/observability_routes.py` — traces, stats, reset endpoints
  - `rag_pipeline.py` — calls HF endpoint, records rich span attributes/events
  - `auth/keycloak_config.py` — reusable Keycloak helpers (not wired by default in `main.py`)
- `observe/`
  - `__init__.py` — `setup_observability()` (tracing, metrics, logging)
  - `tracing.py` — tracer provider and in-memory exporter hookup
  - `metrics.py` — Prometheus `/metrics` exposure
  - `logging.py` — OpenTelemetry logging instrumentation
  - `instrumentation.py` — general-purpose function decorator and auto-instrumentation
  - `inmemory_exporter.py` — thread-safe span buffer
  - `config.py` — `pydantic-settings` env loader
- `.env` — example environment variables
- `requirements.txt` — Python dependencies

## Requirements
- Python 3.10+ (developed with 3.12)
- Keycloak server (for protected endpoints)
- Optional: Prometheus to scrape `/metrics`

## Environment
Environment variables are loaded from `.env` via `pydantic-settings` and `python-dotenv`.

Key variables used by the app:
- `SERVICE_NAME` — OpenTelemetry service name (e.g., `Speedy-Observability`)
- `ENABLE_METRICS` — `True` to expose `/metrics`
- `METRICS_PORT` — not used by the server directly; server binds to the Uvicorn port
- `LOG_LEVEL` — e.g., `INFO`
- `KEYCLOAK_SERVER_URL` — e.g., `http://localhost:90/`
- `REALM_NAME`
- `FRONTEND_CLIENT_ID` — not required by the backend
- `CLIENT_ID`
- `CLIENT_SECRET_KEY`
- `CALLBACK_URI` — default `http://localhost:8000/callback`
- `OTLP_ENDPOINT` — optional (currently commented out in tracing setup)

A sample `.env` already exists in the repo. Update values for your Keycloak.

## Install
Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: The code imports a few packages not listed in `requirements.txt`. If you see ImportError, install:

```bash
pip install requests python-keycloak python-jose psutil
```

## Run
Start the API with Uvicorn from the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json
- Metrics (if enabled): http://localhost:8000/metrics

## Authentication
Protected endpoints use Keycloak. Obtain a token via the login endpoint:

- `POST /login` (form-encoded): `username`, `password`
- Response includes `access_token`. Use it as `Authorization: Bearer <token>` for protected routes.

Environment variables for Keycloak must be set (`KEYCLOAK_SERVER_URL`, `REALM_NAME`, `CLIENT_ID`, `CLIENT_SECRET_KEY`).

## API Overview
- `POST /login` — Keycloak login, returns tokens
- `POST /query` — Protected. Body: `{ "query": "your question" }`.
  - Produces LLM reasoning spans and returns only the final answer.
- `GET /logs` — In-memory app log buffer (optionally filter with `?level=info|warning|error`)
- `GET /observability/traces` — Paginated tree of recent traces (excludes system/internal paths)
- `GET /observability/stats` — Aggregated app/system/LLM stats
- `POST /observability/reset` — Clear in-memory spans
- `GET /llm-metrics/` — Raw LLM span attributes for each call
- `GET /llm-metrics/summary` — Aggregated LLM stats

CORS is enabled for `http://localhost:3000` by default in `app/main.py`.

## How it works
- `observe.setup_observability(app)` configures tracing, metrics, and logging. Traces are exported to an in-memory buffer and (optionally) to OTLP when enabled.
- `app/rag_pipeline.py` executes an LLM request to a Hugging Face Inference Endpoint and writes detailed span attributes and events. The endpoint URL/model defaults are in code (`OLLAMA_URL`, `DEFAULT_MODEL`).
- Additional programmatic instrumentation is applied via `observe.instrumentation.instrument_all_in_package("app", ...)`.

## Troubleshooting
- **ImportError (keycloak, jose, requests, psutil)**: run the extra `pip install` command above.
- **401 Unauthorized on `/query`**: ensure Keycloak env vars are correct and the token from `/login` is sent as `Authorization: Bearer <token>`.
- **No metrics at `/metrics`**: set `ENABLE_METRICS=True` in `.env` and restart.
- **No traces**: generate traffic (e.g., call `/login`, `/query`). Use `/observability/reset` to clear state.
- **External LLM errors**: `app/rag_pipeline.py` posts to the configured HF endpoint; verify the URL/credentials and allowed models.

## Notes
- OTLP exporter lines in `observe/tracing.py` are commented out. Provide `OTLP_ENDPOINT` and uncomment to ship traces to a collector (Jaeger/Tempo/etc.).

