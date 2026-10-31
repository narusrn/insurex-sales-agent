"""FastAPI app — /chat and /runs behind X-API-Key auth, /dashboard and
/chat-test public (static pages with no data baked in; they ask for the key
client-side and send it as X-API-Key on their own fetches). CORS is open
(allow_origins=["*"]) specifically so api/static/chat_test.html works when
opened directly via file:// — confirmed necessary the hard way: without it,
the browser blocks the fetch with "No 'Access-Control-Allow-Origin' header"
before the request even reaches the API key check. No multi-tenant concept.
See implementation-plan.md §1.1. Run with: uvicorn api.main:app --reload
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # every other module's .env loading lives in its own __main__ smoke test, which
# never runs under uvicorn — confirmed the hard way (Phase 6.3): a live request failed with
# "Missing credentials... OPENAI_API_KEY" because nothing had loaded .env for the server process

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .routes.chat import router as chat_router
from .routes.runs import router as runs_router

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="InsureX Sales Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(runs_router)


@app.get("/dashboard")
async def dashboard() -> FileResponse:
    return FileResponse(_STATIC_DIR / "dashboard.html")


@app.get("/chat-test")
async def chat_test() -> FileResponse:
    return FileResponse(_STATIC_DIR / "chat_test.html")
