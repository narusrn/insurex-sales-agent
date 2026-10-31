"""RUNS the graph — astream_turn() + stream_events() for api/routes/chat.py's SSE
endpoint. orchestrator.py only DEFINES the graph; this module is what actually
runs it, same split as the reference repo it's checked against. See
implementation-plan.md §1.1.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .orchestrator import CHECKPOINT_DB_PATH, build_graph

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent / "db" / "runs.jsonl"

# USD per 1M tokens, (input, output) — hand-maintained, NOT fetched live, so treat
# estimated_cost_usd as approximate and possibly stale; prices OpenAI published as of
# this project's build (2026). Unknown models get cost=None rather than a wrong guess.
_PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _PRICING_PER_1M_TOKENS.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return round(input_tokens / 1_000_000 * in_rate + output_tokens / 1_000_000 * out_rate, 6)


def _log_run(
    session_id: str, message: str, reply: str, tools_used: list[str], latency: float,
    error: str | None, usage_by_node: list[dict], trace: list[str],
) -> None:
    """One JSON line per turn — powers the /runs endpoint + dashboard. Best-effort:
    a logging failure must never break the actual chat response."""
    try:
        RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        input_tokens = sum(u["input_tokens"] for u in usage_by_node)
        output_tokens = sum(u["output_tokens"] for u in usage_by_node)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "message": message,
            "reply": reply,
            "tools_used": tools_used,
            "latency": round(latency, 2),
            "success": error is None,
            "error": error,
            "llm_calls": len(usage_by_node),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "by_node": usage_by_node,
            },
            "estimated_cost_usd": _estimate_cost_usd(model, input_tokens, output_tokens),
            "cost_note": f"approximate — hardcoded {model} per-token rates, may not reflect current pricing",
            "trace": trace,
        }
        with RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


async def _get_trace(session_id: str) -> list[str]:
    """Final render_dialogue()-ready trace for a session, straight from the checkpointer —
    same read pattern as get_history(), just a different state field."""
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        return (snapshot.values or {}).get("trace", [])


async def get_history(session_id: str) -> list[dict]:
    """User-facing chat history for a session — [{"role": "user"/"assistant", "content": ...}],
    chronological. Reads straight from the checkpointer, no separate storage. Filters to
    HumanMessage and supervisor's own AIMessages only: a tool-call-only AIMessage has no content
    (already excluded by the `m.content` check), and after_retrieve_node's bridge note is
    excluded by its `name` tag (orchestrator.py) — neither is something a customer ever saw,
    so neither belongs in a chat-history view. An unused session_id returns [] (confirmed
    directly: aget_state on a fresh thread_id returns an empty StateSnapshot, not an error)."""
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        messages = (snapshot.values or {}).get("messages", [])
        history = []
        for m in messages:
            if isinstance(m, HumanMessage) and m.content:
                history.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage) and m.content and m.name != "after_retrieve_bridge":
                history.append({"role": "assistant", "content": m.content})
        return history


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
    filtered, UI-agnostic event stream both stream_events() (SSE, below) and the
    live chat test page (api/static/chat_test.html, via /chat) build on, so the
    filtering logic exists in exactly one place.

    namespace == () alone is NOT enough to isolate supervisor's own reply tokens
    — confirmed the hard way (live chat_test.html testing): `extract_lead_info`'s
    own internal `.with_structured_output()` call runs inside the top-level
    `extract_lead_info` ToolNode, not inside a subgraph, so it ALSO has
    namespace == (), same as supervisor's own LLM call — the raw JSON it
    generates streamed straight into the reply text, glued onto the front of
    supervisor's actual answer with no separator. namespace == () still matters
    for excluding rag_subgraph's internal calls (e.g. grade_documents, which DOES
    have a non-empty namespace), but the chunk's own metadata is what actually
    identifies which node produced it: metadata["langgraph_node"] — checked
    directly against the installed LangGraph version rather than assumed to
    exist. supervisor is the only node that should ever reach the client as a
    token (§2) — gated on node identity now, not just nesting depth."""
    async for namespace, mode, chunk in astream_turn(session_id, user_message):
        if mode == "messages":
            token_chunk, metadata = chunk
            # usage_metadata: NOT scoped to namespace == ()/supervisor like the token check below
            # — every LLM call anywhere in the turn (supervisor's own, extract_lead_info's
            # structured-output call, every call inside rag_subgraph) contributes to real cost,
            # confirmed empirically that usage_metadata appears on all of them once
            # get_llm()'s stream_usage=True is set, regardless of node or nesting.
            if isinstance(token_chunk, AIMessageChunk) and token_chunk.usage_metadata:
                yield {
                    "type": "usage",
                    "node": metadata.get("langgraph_node"),
                    "usage_metadata": dict(token_chunk.usage_metadata),
                }
            # §8: "messages" mode also passes complete ToolMessages through as they're added to
            # state — only AIMessageChunk is real streamed reply text, and ONLY from supervisor's
            # own top-level call — see this function's own docstring for why both checks matter.
            if (
                namespace == ()
                and isinstance(token_chunk, AIMessageChunk)
                and token_chunk.content
                and metadata.get("langgraph_node") == "supervisor"
            ):
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
    the /runs + /dashboard endpoints: token usage per LLM call, an approximate cost
    estimate, and the full render_dialogue()-ready trace (§3.8) — the human-readable
    log now doubles as a queryable one instead of living only in `state["trace"]`."""
    start = time.monotonic()
    reply_parts: list[str] = []
    tools_used: list[str] = []
    usage_by_node: list[dict] = []
    error: str | None = None
    try:
        async for event in stream_structured_events(session_id, user_message):
            if event["type"] == "token":
                reply_parts.append(event["content"])
                yield _sse({"type": "token", "content": event["content"]})
            elif event["type"] == "status":
                tools_used.append(event["tool"])
                yield _sse({"stage": "working", "tool": event["tool"]}, event="status")
            elif event["type"] == "usage":
                usage_by_node.append({"node": event["node"], **event["usage_metadata"]})
                yield _sse({"node": event["node"], **event["usage_metadata"]}, event="usage")
    except Exception as e:
        error = str(e)
        raise
    finally:
        trace = await _get_trace(session_id) if error is None else []
        _log_run(
            session_id, user_message, "".join(reply_parts), tools_used,
            time.monotonic() - start, error, usage_by_node, trace,
        )
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
