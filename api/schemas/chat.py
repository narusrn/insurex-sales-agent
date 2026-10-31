"""SessionChatRequest + response shapes for POST /chat and GET /chat/{session_id}.
See implementation-plan.md §1.1.
"""
from pydantic import BaseModel, Field


class SessionChatRequest(BaseModel):
    message: str = Field(..., description="The customer's message for this turn.", examples=["Health Plus มีแผนอะไรบ้าง"])
    session_id: str = Field(
        ...,
        description="Client-generated conversation id (e.g. a UUID). Reused across turns to "
        "keep the same LangGraph checkpointer thread; a new value starts a fresh conversation.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'.", examples=["assistant"])
    content: str = Field(..., examples=["คุณสนใจประกันสุขภาพ Health Plus ครับ ..."])
