"""RUNS the graph — astream_turn() + stream_events() for api/routes/chat.py's SSE
endpoint. orchestrator.py only DEFINES the graph; this module is what actually
runs it, same split as the reference repo it's checked against. See
implementation-plan.md §1.1.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .orchestrator import CHECKPOINT_DB_PATH, build_graph

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent / "db" / "runs.jsonl"


def _log_run(session_id: str, message: str, reply: str, tools_used: list[str], latency: float, error: str | None) -> None:
    """One JSON line per turn — powers the /runs endpoint + dashboard. Best-effort:
    a logging failure must never break the actual chat response."""
    try:
        RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "message": message,
            "reply": reply,
            "tools_used": tools_used,
            "latency": round(latency, 2),
            "success": error is None,
            "error": error,
        }
        with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def astream_turn(session_id: str, user_message: str):
    """Raw (namespace, mode, chunk) tuples for one turn — opens its own checkpointer
    connection scoped to this call only (simpler than a long-lived global singleton,
    no explicit close bookkeeping)."""
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        await checkpointer.setup()
        graph = build_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        state = {
            "session_id": session_id,  # save_lead reads state["session_id"] directly (not .get) —
            # omitting it here would KeyError the first time a fresh thread reaches save_lead
            "messages": [HumanMessage(content=user_message)],
            "query": None, "snippets": None, "step": 0,
            "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
        }
        async for namespace, mode, chunk in graph.astream(
            state, config, stream_mode=["messages", "updates"], subgraphs=True,  # §8 — not optional
        ):
            yield namespace, mode, chunk


async def stream_structured_events(session_id: str, user_message: str):
    """{"type": "token", "content": ...} / {"type": "status", "tool": ...} — the
    filtered, UI-agnostic event stream both stream_events() (SSE, below) and
    streamlit_app.py build on, so the namespace/mode filtering logic exists in
    exactly one place.

    namespace == () restricts streamed tokens to supervisor's own LLM call —
    without it, rag_subgraph's internal calls leak through too: confirmed
    empirically (this module's own smoke test) that grade_documents' structured-
    output call streams its raw JSON ({"binary_score":"yes"}) as if it were reply
    text, right before the real answer. supervisor is the only node that ever
    composes a user-facing reply (§2) — nothing from inside the subgraph should
    ever reach the client as a token."""
    async for namespace, mode, chunk in astream_turn(session_id, user_message):
        if mode == "messages" and namespace == ():
            token_chunk, _metadata = chunk
            # §8: "messages" mode also passes complete ToolMessages through as they're
            # added to state — only AIMessageChunk is real streamed reply text.
            if isinstance(token_chunk, AIMessageChunk) and token_chunk.content:
                yield {"type": "token", "content": token_chunk.content}
        elif mode == "updates" and namespace == ():
            for node_name, node_output in chunk.items():
                if node_name != "supervisor" or not node_output:
                    continue
                last = (node_output.get("messages") or [None])[-1]
                for tc in getattr(last, "tool_calls", None) or []:
                    yield {"type": "status", "tool": tc["name"]}


def _sse(data: dict, event: str | None = None) -> str:
    frame = f"event: {event}\n" if event else ""
    return frame + f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def stream_events(session_id: str, user_message: str):
    """Ready-to-send SSE frames for one turn. Ends with event: done regardless of how
    the stream ended (finally: block). Also logs one run record (db/runs.jsonl) for
    the /runs + /dashboard endpoints — a basic usage-tracking addition, separate from
    render_dialogue(trace) (§3.8), which is the detailed human-readable log, not a
    queryable one."""
    start = time.monotonic()
    reply_parts: list[str] = []
    tools_used: list[str] = []
    error: str | None = None
    try:
        async for event in stream_structured_events(session_id, user_message):
            if event["type"] == "token":
                reply_parts.append(event["content"])
                yield _sse({"type": "token", "content": event["content"]})
            elif event["type"] == "status":
                tools_used.append(event["tool"])
                yield _sse({"stage": "working", "tool": event["tool"]}, event="status")
    except Exception as e:
        error = str(e)
        raise
    finally:
        _log_run(session_id, user_message, "".join(reply_parts), tools_used, time.monotonic() - start, error)
        yield _sse({"session_id": session_id}, event="done")


if __name__ == "__main__":
    # smoke test — Phase 6.1: drain stream_events() for one turn, print every frame,
    # confirm token frames arrive incrementally (not all at once) and done frame carries session_id
    import asyncio
    import time

    from dotenv import load_dotenv
    load_dotenv()

    async def _run():
        frame_times = []
        async for frame in stream_events("runner-smoke-test", "Health Plus มีแผนอะไรบ้าง"):
            frame_times.append(time.monotonic())
            print(frame.strip())
        assert len(frame_times) > 2, "expected multiple frames, not one bulk response"
        gaps = [b - a for a, b in zip(frame_times, frame_times[1:])]
        assert any(g > 0.01 for g in gaps), "frames arrived all at once — not real streaming"
        print(f"\nrunner.py smoke test OK: {len(frame_times)} frames, incremental arrival confirmed")

    asyncio.run(_run())
