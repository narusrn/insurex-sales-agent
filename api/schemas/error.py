"""Structured error shape every handler in api/main.py's exception handlers
responds with — matches the reference API's own ErrorDetail/ErrorResponse.
"""
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code, e.g. 'unauthorized'.", examples=["unauthorized"])
    message: str = Field(..., description="Human-readable explanation.")


class ErrorResponse(BaseModel):
    error: ErrorDetail
