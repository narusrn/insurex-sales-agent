"""POST /chat + GET /chat/{session_id} — thin HTTP layer only, no shaping of
its own. See implementation-plan.md §1.1.
"""
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agents.runner import get_history, stream_events

from ..schemas.chat import ChatMessage, SessionChatRequest

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    summary="Send a chat message, stream the reply",
    description=(
        "Sends one customer message for `session_id` and streams the agent's reply as "
        "Server-Sent Events (`text/event-stream`), not JSON. Reuse the same `session_id` on "
        "every turn of a conversation — it's the LangGraph checkpointer thread key; a new "
        "value starts a fresh conversation.\n\n"
        "Frame types (`event:` field):\n"
        "- `token` (default, no `event:` line) — `{\"type\": \"token\", \"content\": \"...\"}`, one "
        "per streamed chunk of the reply text. Concatenate `content` across all of these for "
        "the full reply.\n"
        "- `event: status` — `{\"stage\": \"working\", \"tool\": \"<tool name>\"}`, sent when the "
        "agent calls a tool (e.g. `retrieve`, `save_lead`).\n"
        "- `event: usage` — `{\"node\": \"<langgraph node>\", \"input_tokens\": ..., "
        "\"output_tokens\": ..., \"total_tokens\": ...}`, one per LLM call made during the turn "
        "(there can be several — retrieval grading, lead extraction, the final reply).\n"
        "- `event: done` — `{\"session_id\": \"...\"}`, always the last frame, sent even if the "
        "turn errored partway through.\n\n"
        "Each call also appends one record to the usage log, visible via `GET /api/v1/runs`."
    ),
    responses={
        200: {
            "description": "SSE stream of the frame types listed above.",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type": "token", "content": "คุณ"}\n\n'
                        'data: {"type": "token", "content": "สนใจ"}\n\n'
                        'event: status\n'
                        'data: {"stage": "working", "tool": "retrieve"}\n\n'
                        'event: usage\n'
                        'data: {"node": "supervisor", "input_tokens": 512, "output_tokens": 48, "total_tokens": 560}\n\n'
                        'event: done\n'
                        'data: {"session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}\n\n'
                    )
                }
            },
        },
    },
)
async def chat(body: SessionChatRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(
        stream_events(request.app.state.graph, body.session_id, body.message),
        media_type="text/event-stream",
    )


@router.get(
    "/chat/{session_id}",
    response_model=list[ChatMessage],
    summary="Get chat history for a session",
    description=(
        "Returns the user-facing conversation for `session_id`, chronological, straight from "
        "the LangGraph checkpointer — no separate chat-history storage. Internal messages "
        "(tool calls, the retrieval bridge note) are filtered out; only what the customer sent "
        "and what the agent actually replied are included. An unknown or not-yet-used "
        "`session_id` returns an empty list, not a 404."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {"role": "user", "content": "Health Plus มีแผนอะไรบ้าง"},
                        {"role": "assistant", "content": "Health Plus มี 3 แผนหลักครับ ..."},
                    ]
                }
            }
        }
    },
)
async def chat_history(session_id: str, request: Request) -> list[dict]:
    return await get_history(request.app.state.graph, session_id)
