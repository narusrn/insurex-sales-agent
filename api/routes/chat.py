"""POST /chat — thin HTTP layer only, no shaping of its own. See
implementation-plan.md §1.1.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.runner import stream_events

from ..schemas.chat import SessionChatRequest

router = APIRouter()


@router.post("/chat")
async def chat(request: SessionChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_events(request.session_id, request.message),
        media_type="text/event-stream",
    )
