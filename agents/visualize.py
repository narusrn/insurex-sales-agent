"""get_graph(xray=True) — run after any node/edge change. See implementation-plan.md
§2/Phase 3.6. Two functions, run by hand, not imported elsewhere.

draw_mermaid_png() renders correctly end-to-end (all Command branches, the fully
expanded rag_subgraph, the loop back to supervisor) because orchestrator.py's
"tools" node declares destinations={...} — render-only metadata LangGraph uses
to know where a Command-returning tool can go, since a bare ToolNode gives it
nothing to infer that from. Without it, those edges render as disconnected boxes
(Phase 2.3's original finding, corrected in Phase 3.6).
"""
from pathlib import Path

from .orchestrator import build_graph
from .subgraphs.rag import graph as rag_subgraph

PNG_PATH = Path(__file__).resolve().parent.parent / "assets" / "graph.png"
RETRIEVER_PNG_PATH = Path(__file__).resolve().parent.parent / "assets" / "graph-retriever.png"


def print_mermaid() -> None:
    print(build_graph().compile().get_graph(xray=True).draw_mermaid())


def save_png(path: Path = PNG_PATH) -> None:
    build_graph().compile().get_graph(xray=True).draw_mermaid_png(output_file_path=str(path))
    print(f"saved {path}")


def save_retriever_png(path: Path = RETRIEVER_PNG_PATH) -> None:
    """Standalone render of just the RAG subgraph (retrieve -> grade -> rewrite
    loop) — the same nested box visible inside graph.png's xray view, isolated
    for a slide that's specifically about the agentic retriever, not the whole
    agent."""
    rag_subgraph.get_graph().draw_mermaid_png(output_file_path=str(path))
    print(f"saved {path}")


if __name__ == "__main__":
    save_png()
    save_retriever_png()
