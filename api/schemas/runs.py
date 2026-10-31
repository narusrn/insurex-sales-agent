"""Response shape for GET /runs — mirrors the dict agents/runner.py's
_log_run() writes to db/runs.jsonl, one-to-one, so Swagger documents exactly
what the dashboard consumes.
"""
from pydantic import BaseModel, Field


class NodeUsage(BaseModel):
    node: str | None = Field(None, description="LangGraph node that made this LLM call, e.g. 'supervisor'.")
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RunUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    by_node: list[NodeUsage] = Field(..., description="One entry per LLM call made during the turn.")


class RunRecord(BaseModel):
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the turn.")
    session_id: str
    message: str = Field(..., description="The customer's message for this turn.")
    reply: str = Field(..., description="The agent's full reply text (empty if the turn errored).")
    tools_used: list[str] = Field(..., description="Names of tools the supervisor called this turn, in order.")
    latency: float = Field(..., description="Wall-clock seconds for the whole turn.")
    success: bool
    error: str | None = Field(None, description="Exception message if the turn errored, else null.")
    llm_calls: int = Field(..., description="Total LLM calls made this turn, across every node.")
    usage: RunUsage
    estimated_cost_usd: float | None = Field(
        None, description="Approximate cost from hardcoded per-token rates; null for an unrecognized model."
    )
    cost_note: str = Field(..., description="Disclaimer: rates are hand-maintained, not fetched live.")
    trace: list[str] = Field(..., description="Human-readable step-by-step trace of the turn, in order.")
