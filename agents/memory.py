"""sanitize_dangling_tool_calls() + _maybe_trim_and_summarize() — both run at the
top of supervisor_node, before it assembles context for its own LLM call.
See implementation-plan.md §5.1/§5.2/§3.1, agent-architecture-patterns.md §5/§8.
"""
import os

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from .llm import get_llm
from .state import GraphState

CONTEXT_TOKEN_THRESHOLD = int(os.environ.get("CONTEXT_TOKEN_THRESHOLD", "2000"))
CONTEXT_KEEP_TOKENS = int(os.environ.get("CONTEXT_KEEP_TOKENS", "800"))


def sanitize_dangling_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """A crash/disconnect can leave an AIMessage(tool_calls=[...]) checkpointed with
    no ToolMessage ever written for it — the next LLM call then gets hard-rejected
    ("tool_calls must be followed by tool messages"), permanently, since the bad
    history is now persisted. Verbatim pattern from agent-architecture-patterns.md §8:
    never drop the original AIMessage, never auto-retry — let the model re-request
    the tool call if it's still needed."""
    answered_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    result = []
    for m in messages:
        result.append(m)
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc["id"] not in answered_ids:
                    result.append(ToolMessage(
                        content="This tool call was interrupted and did not complete.",
                        tool_call_id=tc["id"],
                    ))
    return result


def _summarize(existing_summary: str, dropped_messages: list[BaseMessage]) -> str:
    """Merges into the EXISTING summary, not just the newest overflow in isolation —
    otherwise a later trim would silently drop whatever an earlier trim already
    compressed. §5's own warning."""
    raw = "\n".join(f"{m.type}: {m.content}" for m in dropped_messages)[:4000]
    prompt = (
        "สรุปข้อเท็จจริง/ความต้องการที่สำคัญจากบทสนทนาส่วนนี้ เป็นภาษาไทย 2-3 ประโยค "
        "ผสานเข้ากับสรุปเดิมด้านล่าง (ถ้ามี) ไม่ใช่สรุปแยกต่างหาก:\n\n"
        f"สรุปเดิม: {existing_summary or '(ไม่มี)'}\n\n"
        f"บทสนทนาส่วนที่จะถูกตัดออกจาก context:\n{raw}"
    )
    response = get_llm(temperature=0).invoke(prompt)
    return str(response.content).strip()


def _maybe_trim_and_summarize(state: GraphState) -> dict:
    messages = state["messages"]
    if count_tokens_approximately(messages) <= CONTEXT_TOKEN_THRESHOLD:
        return {}
    kept = trim_messages(
        messages, max_tokens=CONTEXT_KEEP_TOKENS, token_counter="approximate",
        strategy="last", start_on="human", include_system=True,
    )
    kept_ids = {m.id for m in kept}
    dropped = [m for m in messages if m.id not in kept_ids]
    if not dropped:
        return {}
    new_summary = _summarize(state.get("conversation_summary") or "", dropped)
    return {
        "conversation_summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in dropped],
    }


if __name__ == "__main__":
    # smoke test — Phase 5.1/5.2: dangling-tool-call repair + trim/summarize trigger
    from langchain_core.messages import HumanMessage

    dangling = [HumanMessage(content="hi"), AIMessage(content="", tool_calls=[
        {"name": "x", "args": {}, "id": "call_1"},
    ])]
    fixed = sanitize_dangling_tool_calls(dangling)
    assert len(fixed) == 3 and isinstance(fixed[-1], ToolMessage) and fixed[-1].tool_call_id == "call_1"

    already_answered = [HumanMessage(content="hi"), AIMessage(content="", tool_calls=[
        {"name": "x", "args": {}, "id": "call_2"},
    ]), ToolMessage(content="ok", tool_call_id="call_2")]
    assert sanitize_dangling_tool_calls(already_answered) == already_answered

    from dotenv import load_dotenv
    load_dotenv()
    long_text = "ลูกค้าถามเรื่องประกันสุขภาพหลายคำถามมาก " * 400
    long_history = [HumanMessage(content=long_text, id=f"m{i}") for i in range(5)]
    for i, m in enumerate(long_history):
        m.id = f"m{i}"
    result = _maybe_trim_and_summarize({"messages": long_history, "conversation_summary": None})
    assert result, "expected a trim+summarize update for an over-threshold history"
    assert result["conversation_summary"].strip()
    assert all(isinstance(m, RemoveMessage) for m in result["messages"])
    print("memory.py smoke test OK")
