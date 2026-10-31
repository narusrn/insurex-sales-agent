"""FastAPI app — /chat and /runs behind X-API-Key auth, /dashboard public (a
static page with no data baked in; it asks for the key client-side and sends
it as X-API-Key on its own /runs fetch — same split as the reference
dashboard this was checked against). No CORS, no multi-tenant concept. See
implementation-plan.md §1.1. Run with: uvicorn api.main:app --reload
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # every other module's .env loading lives in its own __main__ smoke test, which
# never runs under uvicorn — confirmed the hard way (Phase 6.3): a live request failed with
# "Missing credentials... OPENAI_API_KEY" because nothing had loaded .env for the server process

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .routes.chat import router as chat_router
from .routes.runs import router as runs_router

_DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"

app = FastAPI(title="InsureX Sales Agent")
app.include_router(chat_router)
app.include_router(runs_router)


@app.get("/dashboard")
async def dashboard() -> FileResponse:
    return FileResponse(_DASHBOARD_PATH)
