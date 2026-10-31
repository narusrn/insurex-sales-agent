"""NODE_DISPLAY_NAMES + render_dialogue(). See implementation-plan.md §3.8.

Covers plain node functions only (the rag subgraph's internals, after_retrieve).
supervisor's own tool-bound functions (extract_lead_info, save_lead,
handoff_to_kb_retriever — added in Phase 3) write their own hardcoded
"Lead Extractor:"/"Lead Saver:" trace lines directly, not via this dict —
see §3.3's "why Command" note for why.
"""

NODE_DISPLAY_NAMES = {
    "supervisor": "Supervisor",  # every visit — routing chatter and every kind of final reply alike
    "generate_query_or_respond": "RAG Agent",
    "retrieve": "RAG Agent",
    "grade_documents": "RAG Agent (ตรวจสอบ)",
    "rewrite_question": "RAG Agent",
    "after_retrieve": "RAG Agent",
}


def render_dialogue(trace: list[str]) -> str:
    return "\n".join(trace)


if __name__ == "__main__":
    demo = [f"{NODE_DISPLAY_NAMES['supervisor']}: สวัสดีครับ", f"{NODE_DISPLAY_NAMES['retrieve']}: ค้นพบ 1 ข้อความ"]
    assert render_dialogue(demo) == "Supervisor: สวัสดีครับ\nRAG Agent: ค้นพบ 1 ข้อความ"
    print("trace.py smoke test OK")
