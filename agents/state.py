"""GraphState + RetrieverState. See implementation-plan.md §3.1."""
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def add_trace(existing: list[str], new: list[str]) -> list[str]:
    """Like operator.add, except it skips whatever prefix of `new` already
    overlaps the tail of `existing`. CONFIRMED EMPIRICALLY (Phase 1.4 smoke
    test, LangGraph 1.2.16): because `trace` is a key `RetrieverState`
    shares with `GraphState` (needed so the subgraph's lines reach the
    parent at all, same mechanism `query`/`snippets` rely on — [§2]'s
    auto-merge), the subgraph's own final cumulative trace gets merged into
    the parent a second time on top of the same lines already propagated
    incrementally as its internal steps ran — e.g. existing=["A"],
    new=["A","B","C"] arrives once `after_retrieve` already has "A" from a
    live per-step merge. A plain `operator.add` would double "A"; this finds
    the real overlap and appends only the genuinely new suffix. `query` and
    `snippets` don't show this because they're plain overwrite fields, not
    reducer-accumulated — the bug is specific to sharing an *accumulator*
    field by name across a parent/subgraph boundary."""
    for overlap in range(min(len(existing), len(new)), 0, -1):
        if existing[-overlap:] == new[:overlap]:
            return existing + new[overlap:]
    return existing + new


class GraphState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    query: str | None  # seeds RetrieverState on handoff — must exist on both schemas [§2 auto-merge]
    snippets: list[str] | None  # report-OUT channel, symmetric with `query`'s report-IN: shared with
    # RetrieverState so after_retrieve (outside the subgraph) can read what retrieve found and write
    # it into `messages`. Safe as a plain overwrite field (no reducer) despite [§2]'s stale-data
    # warning about generic buckets — that warning is about *concurrent* writers landing before a
    # reader sees the first write; here there's exactly one writer (the subgraph, once per handoff)
    # and exactly one reader (after_retrieve, immediately next step, same turn), never read later
    step: int  # loop guard, reset to 0 on a fresh HumanMessage — see supervisor_node
    conversation_summary: str | None  # [§5] merge-updated by _maybe_trim_and_summarize
    trace: Annotated[list[str], add_trace]  # human-readable dialogue lines — §3.8
    last_retrieval_relevant: bool | None  # Phase 5.4 finding: grade_documents' own final verdict,
    # now shared with GraphState (plain overwrite field, same safe pattern as query/snippets — NOT
    # an accumulator, so none of add_trace's cross-boundary double-merge risk applies) so
    # after_retrieve can tell supervisor when retries were exhausted on UNGRADED-relevant snippets.
    # Without this, qa-12 showed supervisor stating a specific, likely-wrong number with confidence
    # instead of hedging, because it had no way to know the snippets it was given were graded "no"


class RetrieverState(TypedDict):  # rag_subgraph — isolated, see §2's table
    query: str
    retriever_messages: Annotated[list[BaseMessage], add_messages]
    trace: Annotated[list[str], add_trace]  # same name+reducer as parent — auto-merges up [§2]
    retries: int
    snippets: list[str]
    last_retrieval_relevant: bool  # grade_documents' own verdict — named to match GraphState's
    # shared field directly (was `relevant`, Phase 5.4 rename) instead of mirroring into a 2nd field
