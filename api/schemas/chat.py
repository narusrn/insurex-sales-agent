"""SessionChatRequest — the one request body POST /chat accepts. See
implementation-plan.md §1.1.
"""
from pydantic import BaseModel


class SessionChatRequest(BaseModel):
    message: str
    session_id: str
