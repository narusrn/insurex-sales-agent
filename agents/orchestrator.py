"""Top-level graph — DEFINES it only (compiles nothing itself). agents/runner.py
RUNS it (Phase 6). See implementation-plan.md §1/§2.
"""
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import get_llm
from .memory import _maybe_trim_and_summarize, sanitize_dangling_tool_calls
from .state import GraphState
from .subgraphs.rag import graph as rag_subgraph
from .tools import extract_lead_info, handoff_to_kb_retriever, save_lead

CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "checkpoints.sqlite"
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # db/ is gitignored, never committed —
# on a fresh clone (confirmed the hard way on Streamlit Cloud) nothing else creates it before
# SqliteSaver/AsyncSqliteSaver.from_conn_string() tries to open a file inside a directory that
# doesn't exist yet, which sqlite3 can't do. Worked locally only because db/ already existed from
# earlier test runs. Same mkdir-before-connect pattern tools.py's _init_leads_db() already uses.
GRAPH_MAX_STEPS = int(os.environ.get("GRAPH_MAX_STEPS", "3"))
FALLBACK_MESSAGE = "ขออภัยค่ะ ระบบไม่สามารถประมวลผลคำถามนี้ได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"

_SUPERVISOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "_supervisor.md"
SUPERVISOR_TOOLS = [handoff_to_kb_retriever, extract_lead_info, save_lead]


def _supervisor_prompt() -> str:
    return _SUPERVISOR_PROMPT_PATH.read_text(encoding="utf-8")


def supervisor_node(state: GraphState) -> dict:
    raw_messages = state["messages"]
    # loop guard, checked first — reset on a fresh HumanMessage, not accumulated across turns [§3.1]
    step = 0 if (raw_messages and isinstance(raw_messages[-1], HumanMessage)) else state.get("step", 0)
    if step >= GRAPH_MAX_STEPS:
        return {
            "messages": [AIMessage(content=FALLBACK_MESSAGE)],
            "trace": [f"Supervisor: {FALLBACK_MESSAGE}"],
            "step": step,
        }

    # Repairs a checkpointed-but-never-answered tool call (crash/disconnect, §5.1) before anything
    # else touches `messages` — a dangling tool_calls entry hard-rejects every subsequent LLM call.
    sanitized = sanitize_dangling_tool_calls(raw_messages)
    repaired = [m for m in sanitized if m not in raw_messages]

    # Trims + summarizes once the real conversation (not demo-sized) crosses the threshold (§5.2) —
    # operates on the sanitized list so a repair ToolMessage can't itself get seen as "dangling".
    trim_update = _maybe_trim_and_summarize({**state, "messages": sanitized})
    removed_ids = {rm.id for rm in trim_update.get("messages", [])}
    context_messages = [m for m in sanitized if m.id not in removed_ids]
    conversation_summary = trim_update.get("conversation_summary") or state.get("conversation_summary")

    system_messages = [SystemMessage(content=_supervisor_prompt())]
    if conversation_summary:
        system_messages.append(SystemMessage(content=f"สรุปบทสนทนาก่อนหน้า (ถูกตัดออกจาก context แล้ว): {conversation_summary}"))

    llm = get_llm(temperature=0).bind_tools(SUPERVISOR_TOOLS)
    response = llm.invoke(system_messages + context_messages)

    update = {"messages": repaired + trim_update.get("messages", []) + [response], "step": step + 1}
    if trim_update.get("conversation_summary"):
        update["conversation_summary"] = trim_update["conversation_summary"]
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        update["trace"] = [f"Supervisor: {response.content}"]
    else:
        # Only handoff_to_kb_retriever's trace line is written HERE — it narrates supervisor's own
        # DECISION to call it, before the call happens. extract_lead_info/save_lead write their own
        # trace lines instead (their Command.update IS the result, there's no decision moment here
        # to narrate) — see tools.py.
        for tc in tool_calls:
            if tc["name"] == "handoff_to_kb_retriever":
                update.setdefault("trace", []).append(f"Supervisor: {tc['args']['reason']} → ค้นข้อมูลในฐานความรู้")
    return update


def supervisor_router(state: GraphState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    # Routes straight to the node named after the tool that was actually called — not a single
    # shared "tools" dispatcher — so each tool is its own node in the graph (and in the diagram).
    # Only the first call matters: every bound tool returns Command(goto=...), and this design's
    # prompt (§3.2) always asks for exactly one tool call per turn.
    return tool_calls[0]["name"] if tool_calls else END


def after_retrieve_node(state: GraphState) -> dict:
    """Bridge, OUTSIDE rag_subgraph — the only place retrieved snippets become
    a note supervisor's messages can see. See state.py's `snippets` comment
    for why it's a shared field despite [§2]'s generic-bucket warning.

    Phase 5.4 finding (qa-12): retries can exhaust with grade_documents still saying "no" —
    the subgraph still exits (route_after_grade's own fallback), handing back snippets that were
    never actually confirmed relevant. Without a warning here, supervisor had no way to tell the
    difference and answered a specific, likely-wrong number with full confidence instead of
    hedging the way it correctly did for qa-06. `last_retrieval_relevant` (state.py) is read here,
    not inside the subgraph, since only the FINAL grade (after the last rewrite, if any) matters."""
    snippets = state.get("snippets") or []
    header = "Retrieved {} snippet(s):".format(len(snippets))
    if state.get("last_retrieval_relevant") is False:
        header = (
            "⚠️ ค้นหาซ้ำจนครบจำนวนครั้งแล้วแต่ยังไม่พบข้อมูลที่เกี่ยวข้องโดยตรงกับคำถาม "
            "(ข้อความด้านล่างนี้คือสิ่งที่ค้นเจอ แต่ถูกประเมินว่าอาจไม่ตรงประเด็น — "
            "ห้ามสรุปตัวเลข/เงื่อนไขเฉพาะเจาะจงจากข้อมูลนี้อย่างมั่นใจ ให้บอกลูกค้าตรงๆ ว่าไม่พบข้อมูลที่ชัดเจนแทน):\n" + header
        )
    # name="after_retrieve_bridge" tags this as an internal note, not a real reply — needed so
    # anything reading `messages` for user-facing chat history (agents.runner.get_history) can
    # tell it apart from supervisor's own AIMessages, which are the only ones ever shown to a
    # customer. Found necessary the same way the token-stream leak was: this message has real
    # content, so a naive "AIMessage with content = an assistant turn" filter would wrongly surface
    # "Retrieved N snippet(s): ..." as something the agent said to the user.
    note = AIMessage(content="{}\n{}".format(header, "\n---\n".join(snippets)), name="after_retrieve_bridge")
    return {"messages": [note]}


builder = StateGraph(GraphState)
builder.add_node("supervisor", supervisor_node)
# Each tool is its OWN node (named after the tool itself), not one shared "tools" dispatcher —
# ToolNode([single_tool]) still handles arg validation / InjectedState / InjectedToolCallId the
# normal way, it's just given one tool instead of all three. destinations: render-only metadata
# (no effect on execution — LangGraph's own add_node docstring, reconfirmed by rerunning the smoke
# test after adding it) so draw_mermaid_png() shows each tool's own Command(goto=...) target —
# ToolNode has nothing to infer that from automatically otherwise.
builder.add_node("handoff_to_kb_retriever", ToolNode([handoff_to_kb_retriever]), destinations=("rag_subgraph",))
builder.add_node("extract_lead_info", ToolNode([extract_lead_info]), destinations=("supervisor",))
builder.add_node("save_lead", ToolNode([save_lead]), destinations=("supervisor",))
builder.add_node("rag_subgraph", rag_subgraph)
builder.add_node("after_retrieve", after_retrieve_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", supervisor_router, {
    "handoff_to_kb_retriever": "handoff_to_kb_retriever",
    "extract_lead_info": "extract_lead_info",
    "save_lead": "save_lead",
    END: END,
})
# NO static edge out of any tool node — confirmed empirically (Phase 1.4 smoke test) that a static
# edge here fires ALONGSIDE a tool's own Command(goto=...), not instead of it, even when they
# target different nodes. Every tool node relies solely on its own tool's Command(goto=...)
# (handoff_to_kb_retriever -> "rag_subgraph"; extract_lead_info/save_lead -> "supervisor").
builder.add_edge("rag_subgraph", "after_retrieve")
builder.add_edge("after_retrieve", "supervisor")


def build_graph() -> StateGraph:
    return builder


if __name__ == "__main__":
    # smoke test — Phase 1.4: qa-only flow, no checkpointer (plain .compile())
    from dotenv import load_dotenv
    load_dotenv()
    graph = build_graph().compile()
    result = graph.invoke({
        "session_id": "smoke-test",
        "messages": [HumanMessage(content="Health Plus มีแผนอะไรบ้าง")],
        "query": None, "snippets": None, "step": 0, "conversation_summary": None, "trace": [],
        "last_retrieval_relevant": None,
    })
    reply = result["messages"][-1].content
    assert reply.strip(), "empty final reply"
    print("TRACE:\n" + "\n".join(result["trace"]))
    print("\nREPLY:", reply)
    print("\norchestrator.py smoke test OK")
