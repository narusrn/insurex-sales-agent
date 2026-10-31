"""require_api_key — FastAPI dependency gating every data endpoint behind a single
shared `X-API-Key` header, checked against the `API_KEY` env var. No per-user
keys, no scopes — matches this project's single-demo-agent scope (§1.1).
"""
import os

from fastapi import Header, HTTPException


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
