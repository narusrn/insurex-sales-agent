"""FastAPI app. Standardized against the reference API (support-ticket-triage-
service)'s own api/main.py: one shared checkpointer connection for the app's
whole lifetime (lifespan, not opened per-request), X-Request-ID middleware,
structured {code, message} error responses on every HTTPException/validation
error, a GET /health liveness check, and docs/openapi/every real route under
/api/v1 — including /dashboard and /chat-test, which stay unauthenticated
(a plain browser navigation can't attach X-API-Key) but are still versioned
pages, not an unversioned side door. Only /health stays outside /api/v1,
same exception the reference makes: infra/monitoring checks shouldn't need to
track an API version. CORS is open (allow_origins=["*"]) specifically so
api/static/chat_test.html works when opened directly via file:// — confirmed
necessary the hard way: without it, the browser blocks the fetch with "No
'Access-Control-Allow-Origin' header" before the request even reaches the API
key check. No multi-tenant concept. See implementation-plan.md §1.1.
Run with: uvicorn api.main:app --reload
"""
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # every other module's .env loading lives in its own __main__ smoke test, which
# never runs under uvicorn — confirmed the hard way (Phase 6.3): a live request failed with
# "Missing credentials... OPENAI_API_KEY" because nothing had loaded .env for the server process

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.orchestrator import CHECKPOINT_DB_PATH, build_graph

from .auth import require_api_key
from .routes.chat import router as chat_router
from .routes.runs import router as runs_router
from .schemas.error import ErrorDetail, ErrorResponse

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # AsyncSqliteSaver is itself an async context manager -- open ONE connection for the
    # app's whole lifetime, not a new one per request (agents/runner.py used to; moved here
    # to match the reference API's own lifespan pattern). setup() runs once, not per turn.
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph().compile(checkpointer=checkpointer)
        yield


app = FastAPI(
    title="InsureX Sales Agent",
    version="1.0.0",
    description=(
        "LangGraph-based insurance sales agent: answers product questions over a RAG "
        "knowledge base and collects lead info during the conversation. Every route under "
        "`/api/v1` requires the `X-API-Key` header except `/api/v1/dashboard` and "
        "`/api/v1/chat-test` (plain HTML pages a browser navigates to directly — no way to "
        "attach a custom header to that) — click **Authorize** above and paste the key once "
        "to unlock **Try it out** on the rest."
    ),
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "chat", "description": "Send messages and read conversation history."},
        {"name": "runs", "description": "Per-turn usage, cost, and trace logs."},
        {"name": "pages", "description": "Static HTML pages — unauthenticated, not JSON APIs."},
        {"name": "meta", "description": "Service-level checks, outside the API version."},
    ],
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Request-ID"] = os.urandom(4).hex()
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=ErrorDetail(**detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=ErrorDetail(code="validation_error", message=str(exc.errors()))).model_dump(),
    )


@app.get("/health", tags=["meta"], summary="Liveness check")
async def health(request: Request) -> JSONResponse:
    checks = {
        "graph_ready": hasattr(request.app.state, "graph"),
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")),
        "api_key_set": bool(os.environ.get("API_KEY")),
    }
    healthy = all(checks.values())
    return JSONResponse(status_code=200 if healthy else 503, content={"status": "ok" if healthy else "degraded", "checks": checks})


app.include_router(chat_router, prefix="/api/v1", dependencies=[Depends(require_api_key)])
app.include_router(runs_router, prefix="/api/v1", dependencies=[Depends(require_api_key)])


@app.get("/api/v1/dashboard", tags=["pages"], summary="Usage dashboard (HTML page)")
async def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIR / "dashboard.html")


@app.get("/api/v1/chat-test", tags=["pages"], summary="Live chat test page (HTML page)")
async def chat_test() -> FileResponse:
    return FileResponse(_STATIC_DIR / "chat_test.html")
