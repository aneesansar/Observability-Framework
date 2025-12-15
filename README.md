# SpeedyGPT Observe Backend

FastAPI backend for tenant-aware LLM observability and a Mongo-backed RAG API. It instruments LLM calls, exposes in-memory traces/metrics, and provides deterministic RAG answers grounded on your JSON documents stored in MongoDB.

## Features
- **Mongo-backed RAG**: `/rag/query`, `/rag/reindex`.
- **FastAPI service** with CORS (configurable via env).
- **LLM reasoning telemetry**: latency, scores, events.
- **In-memory traces and Prometheus metrics** (toggle via env).
- **Keycloak auth** for protected endpoints.

## Tech Stack
- **API**: FastAPI + Uvicorn
- **Observability**: OpenTelemetry SDK, FastAPI instrumentation, custom in-memory span exporter, Prometheus client + instrumentator
- **LLM / NLP**: `sentence-transformers` (+ `torch`), simple similarity/consistency checks
- **Auth**: Keycloak (via `keycloak` Python client)

## Directory
- `app/`
  - `main.py` — app wiring (Keycloak, CORS, routers), observability setup
  - `routes/llm_routes.py` — LLM metrics views
  - `routes/observability_routes.py` — traces, stats, reset
  - `routes/rag_routes.py` — Mongo-backed RAG endpoints
  - `mongo_ingest.py` — ingest and embed `documents/*.json` into Mongo
  - `rag_pipeline.py` — LLM pipeline and query parsing
  - `config.py` — centralized env-driven settings
- `observe/` — observability setup, tracing, instrumentation, exporter
- `documents/` — your source JSONs to ingest
- `.env` — environment configuration
- `requirements.txt` — dependencies

## Requirements
- Python 3.10+ (developed with 3.12)
- Keycloak server (for protected endpoints)
- Optional: Prometheus to scrape `/metrics`

## Configuration (.env)
Settings are centralized in `app/config.py` and loaded from `.env`.

Core
- `SERVICE_NAME=Speedy-Observability`
- `API_TITLE=RAG Observability Demo`
- `CORS_ORIGINS=http://localhost:3000`
- `LOG_LEVEL=INFO`

Keycloak
- `KEYCLOAK_SERVER_URL=http://localhost:90/`
- `REALM_NAME=...`
- `CLIENT_ID=...`
- `CLIENT_SECRET_KEY=...`
- `CALLBACK_URI=http://localhost:8001/callback`

Mongo
- `MONGO_URI=mongodb://localhost:27017`
- `MONGO_DB=rag`
- `MONGO_COLL=trips`

Models
- `EMBED_MODEL=all-MiniLM-L6-v2`
- `OLLAMA_URL=http://localhost:11434`
- `LLM_MODEL=gemma3:latest`

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
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

- API docs: http://localhost:8001/docs
- OpenAPI: http://localhost:8001/openapi.json
- Metrics (if enabled): http://localhost:8001/metrics

## Authentication
Protected endpoints use Keycloak. Obtain a token via the login endpoint:

- `POST /login` (form-encoded): `username`, `password`
- Response includes `access_token`. Use it as `Authorization: Bearer <token>` for protected routes.

Environment variables for Keycloak must be set (`KEYCLOAK_SERVER_URL`, `REALM_NAME`, `CLIENT_ID`, `CLIENT_SECRET_KEY`).

## API Overview
- Auth
  - `POST /login` — Keycloak login, returns tokens
  - `POST /query` — Protected. Body: `{ "query": "..." }`
- Observability
  - `GET /logs`
  - `GET /observability/traces`
  - `GET /observability/stats`
  - `POST /observability/reset`
  - `GET /llm-metrics/`, `GET /llm-metrics/summary`
- RAG (Mongo-backed)
  - `POST /rag/reindex` — warm in-memory retriever
  - `POST /rag/query` — query JSON docs (country, dates, prices, airports)

CORS is enabled for `http://localhost:3000` by default in `app/main.py`.

## RAG ingestion and queries

1) Ingest documents into Mongo

```bash
export $(grep -v '^#' .env | xargs)
python app/mongo_ingest.py
```

2) Warm the in-memory index

```bash
curl -X POST http://localhost:8001/rag/reindex
```

3) Query examples

```bash
curl -X POST http://localhost:8001/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Trips to Nepal","top_k":5}'

curl -X POST http://localhost:8001/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Single Cheapest Trip to India from Frankfurt Airport","top_k":1}'

curl -X POST http://localhost:8001/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Trips to India in 2026-05","top_k":5}'
```

## Troubleshooting
- ImportError (matplotlib, keycloak, jose, requests, psutil): install missing deps and restart.
- 401 Unauthorized on `/query`: configure Keycloak envs and use `Authorization: Bearer <token>`.
- No metrics: set `ENABLE_METRICS=True` and restart.
- No RAG results: verify Mongo is running, run `app/mongo_ingest.py`, confirm `.env` Mongo settings.

## Notes
- OTLP exporter lines in `observe/tracing.py` are commented out. Provide `OTLP_ENDPOINT` and enable to ship traces to a collector.
- The deprecated Local RAG console endpoints were removed; only Mongo-backed `/rag/*` endpoints remain.

