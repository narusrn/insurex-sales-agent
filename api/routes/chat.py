"""POST /chat + GET /chat/{session_id} — thin HTTP layer only, no shaping of
its own. See implementation-plan.md §1.1.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agents.runner import get_history, stream_events

from ..auth import require_api_key
from ..schemas.chat import SessionChatRequest

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/chat")
async def chat(request: SessionChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_events(request.session_id, request.message),
        media_type="text/event-stream",
    )


@router.get("/chat/{session_id}")
async def chat_history(session_id: str) -> list[dict]:
    return await get_history(session_id)
