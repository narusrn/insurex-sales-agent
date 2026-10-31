"""One assert: graph compiles, a trivial qa turn returns a non-empty reply.
See implementation-plan.md §5.3. Run with `pytest tests/` (from the project root)
or directly with `python -m tests.test_graph_smoke`.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from agents.orchestrator import build_graph


def test_graph_smoke():
    graph = build_graph().compile()
    result = graph.invoke({
        "session_id": "test-smoke",
        "messages": [HumanMessage(content="สวัสดีค่ะ")],
        "query": None, "snippets": None, "step": 0,
        "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
    })
    assert result["messages"][-1].content.strip()


if __name__ == "__main__":
    test_graph_smoke()
    print("test_graph_smoke.py OK")
