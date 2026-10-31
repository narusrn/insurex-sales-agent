"""FastAPI app, one router included — no auth, no CORS, no multi-router
registration (single demo agent, no multi-tenant concept). See
implementation-plan.md §1.1. Run with: uvicorn api.main:app --reload
"""
from dotenv import load_dotenv

load_dotenv()  # every other module's .env loading lives in its own __main__ smoke test, which
# never runs under uvicorn — confirmed the hard way (Phase 6.3): a live request failed with
# "Missing credentials... OPENAI_API_KEY" because nothing had loaded .env for the server process

from fastapi import FastAPI

from .routes.chat import router as chat_router

app = FastAPI(title="InsureX Sales Agent")
app.include_router(chat_router)
