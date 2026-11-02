"""Response shape for GET /leads — mirrors agents/tools.py's `leads` table
schema (_init_leads_db()) one-to-one.
"""
from pydantic import BaseModel, Field


class Lead(BaseModel):
    id: int
    session_id: str = Field(..., description="Also the checkpointer thread_id — look up the full conversation via GET /api/v1/chat/{session_id}.")
    name: str
    occupation: str
    income: str
    phone: str
    interested_product: str | None = Field(None, description="Best-effort, nullable — the product the customer named interest in, if any.")
    inquiry_summary: str | None = Field(None, description="Best-effort, nullable — a short summary of what the customer was asking about.")
    conversation_log: str | None = Field(None, description="render_dialogue(trace) snapshot taken at the moment this lead was saved.")
    created_at: str = Field(..., description="UTC timestamp (SQLite's own datetime('now')) of when the lead was saved.")
