"""require_api_key — FastAPI dependency gating every data endpoint behind a single
shared `X-API-Key` header, checked against the `API_KEY` env var. No per-user
keys, no scopes — matches this project's single-demo-agent scope (§1.1).

Uses fastapi.security.APIKeyHeader (not a plain Header() param) specifically so
X-API-Key registers as a real OpenAPI security scheme — Swagger UI shows a proper
lock icon / "Authorize" button instead of an undocumented header field. Raises
with a {code, message} detail dict — api/main.py's http_exception_handler wraps
that straight into ErrorResponse, same shape on every error response.
"""
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Shared demo API key, checked against the server's API_KEY env var.",
)


def require_api_key(api_key: str = Depends(api_key_header)) -> str:
    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            {"code": "server_misconfigured", "message": "API_KEY not configured on server"},
        )
    if api_key != expected:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            {"code": "unauthorized", "message": "Invalid or missing X-API-Key header"},
        )
    return api_key
