"""supervisor's bound tools. Phase 1: handoff_to_kb_retriever.
Phase 3: extract_lead_info + save_lead. See implementation-plan.md §3.3/§3.5.
"""
import sqlite3
from pathlib import Path
from typing import Annotated

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import GraphState
from .trace import render_dialogue

_EXTRACT_LEAD_PROMPT_PATH = Path(__file__).parent / "prompts" / "_extract_lead.md"
LEADS_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "leads.sqlite"


@tool
def handoff_to_kb_retriever(query: str, reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Hand off to the knowledge-base retriever when answering needs information from the product documents."""
    # NOTE: this tool's trace line is written by supervisor_node BEFORE the call (it's narrating a
    # *decision*, not a *result*) — see orchestrator.py. extract_lead_info/save_lead below write
    # their own trace lines instead, since there's no "decision" moment for supervisor to narrate;
    # their Command.update is the first and only place the result exists. Both are safe from Bug 1's
    # double-apply (no static edge competes with either tool's own goto="supervisor").
    return Command(update={
        "query": query,
        "messages": [ToolMessage(content=f"Searching knowledge base: {reason}", tool_call_id=tool_call_id)],
    }, goto="rag_subgraph")


class SaveLeadInput(BaseModel):
    """Persist a customer lead. Only succeeds once name, occupation, income,
    and phone are ALL present and non-empty, AND the customer has explicitly
    confirmed those exact values — never guess, fill a placeholder, or set
    confirmed=True without having actually asked."""
    name: str | None = Field(None, description="Customer's full name, exactly as stated")
    occupation: str | None = Field(None, description="Customer's occupation, exactly as stated")
    income: str | None = Field(None, description="Customer's monthly income, exactly as stated")
    phone: str | None = Field(None, description="Customer's contact phone number, exactly as stated")
    interested_product: str | None = Field(None, description="Product name the customer named interest in, if any")
    inquiry_summary: str | None = Field(None, description="One-sentence summary of what the customer wants, for staff follow-up")
    confirmed: bool = Field(False, description="True ONLY if the customer's most recent message explicitly confirmed the restated name/occupation/income/phone are correct — never default this to True")


@tool
def extract_lead_info(
    state: Annotated[GraphState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Extract whatever lead information (name, occupation, income, phone,
    interested product, inquiry summary) is present in the conversation so far.
    Leave a field blank rather than guess — this only reports what's actually
    been said, it never persists anything."""
    llm = get_llm(temperature=0)
    prompt = _EXTRACT_LEAD_PROMPT_PATH.read_text(encoding="utf-8")
    # state["messages"][-1] is always the AIMessage that's CURRENTLY calling this tool — its
    # tool_calls has no ToolMessage response yet (we're still building it), so OpenAI rejects the
    # history as-is ("assistant message with tool_calls must be followed by tool messages").
    # Confirmed empirically (Phase 3 smoke test). Dropping it is safe: it never carries lead data,
    # only the decision to call extract_lead_info.
    result = llm.with_structured_output(SaveLeadInput).invoke(
        [SystemMessage(content=prompt)] + state["messages"][:-1]
    )
    payload = result.model_dump_json(exclude={"confirmed"})  # confirmed is never this tool's call to make
    return Command(update={
        "messages": [ToolMessage(content=payload, tool_call_id=tool_call_id)],
        "trace": [f"Lead Extractor: {payload}"],
    }, goto="supervisor")


def _is_blank(v: str | None) -> bool:
    return v is None or not v.strip()  # catches None AND "", " ", etc.


REQUIRED_LEAD_FIELDS = ("name", "occupation", "income", "phone")


def _init_leads_db() -> None:
    LEADS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LEADS_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL,
                name                TEXT NOT NULL,
                occupation          TEXT NOT NULL,
                income              TEXT NOT NULL,
                phone               TEXT NOT NULL,
                interested_product  TEXT,
                inquiry_summary     TEXT,
                conversation_log    TEXT,
                created_at          TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)


@tool(args_schema=SaveLeadInput)
def save_lead(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[GraphState, InjectedState],
    confirmed: bool = False,
    **fields,
) -> Command:
    """Persist a customer lead. Only succeeds once name, occupation, income,
    and phone are ALL present and non-empty, AND the customer has explicitly
    confirmed those exact values."""
    missing = [f for f in REQUIRED_LEAD_FIELDS if _is_blank(fields.get(f))]
    if missing:
        content = f"Missing: {', '.join(missing)}. Ask the customer for these before saving."
        return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                                "trace": [f"Lead Saver: {content}"]}, goto="supervisor")
    if not confirmed:
        restated = ", ".join(f"{f}={fields[f]}" for f in REQUIRED_LEAD_FIELDS)
        content = (f"Data complete but not yet confirmed ({restated}). Restate these exact "
                   f"values to the customer and ask them to confirm before calling save_lead "
                   f"again with confirmed=True.")
        return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                                "trace": [f"Lead Saver: {content}"]}, goto="supervisor")

    conversation_log = render_dialogue(state["trace"])  # §3.8 — snapshot at save time
    _init_leads_db()
    with sqlite3.connect(LEADS_DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO leads (session_id, name, occupation, income, phone, "
            "interested_product, inquiry_summary, conversation_log) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (state["session_id"], fields["name"], fields["occupation"], fields["income"], fields["phone"],
             fields.get("interested_product"), fields.get("inquiry_summary"), conversation_log),
        )
        row_id = cur.lastrowid
    content = f"Lead saved (id={row_id})"
    return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                            "trace": [f"Lead Saver: {content}"]}, goto="supervisor")


if __name__ == "__main__":
    # smoke test — Phase 3: completeness gate, confirmation gate, then a real insert, no LLM needed
    import tempfile
    LEADS_DB_PATH_BACKUP = LEADS_DB_PATH
    globals()["LEADS_DB_PATH"] = Path(tempfile.mktemp(suffix=".sqlite"))

    fake_state = {"session_id": "smoke-test", "trace": ["Supervisor: test"]}

    r1 = save_lead.func(tool_call_id="t1", state=fake_state, name="สมชาย")
    assert "Missing:" in r1.update["messages"][0].content, r1.update

    r2 = save_lead.func(tool_call_id="t2", state=fake_state, name="สมชาย", occupation="วิศวกร",
                         income="50000", phone="0812345678", confirmed=False)
    assert "not yet confirmed" in r2.update["messages"][0].content, r2.update

    r3 = save_lead.func(tool_call_id="t3", state=fake_state, name="สมชาย", occupation="วิศวกร",
                         income="50000", phone="0812345678", confirmed=True)
    assert "Lead saved" in r3.update["messages"][0].content, r3.update
    assert r3.goto == "supervisor"

    with sqlite3.connect(LEADS_DB_PATH) as conn:
        row = conn.execute("SELECT name, phone FROM leads").fetchone()
    assert row == ("สมชาย", "0812345678"), row

    globals()["LEADS_DB_PATH"] = LEADS_DB_PATH_BACKUP
    print("tools.py smoke test OK")
