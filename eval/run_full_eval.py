"""Phase 5.4 — full pass over all 21 golden cases (13 qa + 6 lead + 2 session).
Writes eval/demo_log.txt: pass/fail per case + render_dialogue(trace) for each run —
this log IS the brief's "Demo/Test Case" deliverable. Retrieval metrics (Hit Rate@k,
MRR) computed via eval.metrics.aggregate() for qa_cases with expected_source_docs.
See implementation-plan.md §5.4.
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.orchestrator import CHECKPOINT_DB_PATH, build_graph
from agents.tools import LEADS_DB_PATH
from agents.trace import render_dialogue
from eval.metrics import aggregate, extract_sources

DATASET_PATH = ROOT / "eval" / "golden_dataset.json"
LOG_PATH = ROOT / "eval" / "demo_log.txt"


def run_qa_cases(graph, cases, log):
    results, retrieval_pairs = [], []
    for case in cases:
        state = graph.invoke({
            "session_id": f"demo-{case['id']}",
            "messages": [HumanMessage(content=case["query"])],
            "query": None, "snippets": None, "step": 0,
            "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
        })
        reply = state["messages"][-1].content
        keypoints_ok = all(kp in reply for kp in case["expected_answer_keypoints"])
        note = ""
        passed = keypoints_ok
        if case["id"] == "qa-13":
            no_tool_call = not any("RAG Agent" in line for line in state["trace"])
            passed = keypoints_ok and no_tool_call
            note = f" (+ no-tool-call check: {'ok' if no_tool_call else 'FAILED'})"

        retrieved = extract_sources(state.get("snippets") or [])
        if case["expected_source_docs"]:
            retrieval_pairs.append((case["expected_source_docs"], retrieved))

        results.append((case["id"], passed))
        print(f"\n{'=' * 70}", file=log)
        print(f"{case['id']} [{case['difficulty']}] — {'PASS' if passed else 'FAIL'}{note}", file=log)
        print(f"Query: {case['query']}", file=log)
        print(f"Expected keypoints: {case['expected_answer_keypoints']}", file=log)
        print(f"Expected source docs: {case['expected_source_docs']}  |  Actually retrieved: {retrieved}", file=log)
        print("--- trace ---", file=log)
        print(render_dialogue(state["trace"]), file=log)
        print("--- final reply ---", file=log)
        print(reply, file=log)
    return results, retrieval_pairs


def run_lead_cases(graph, cases, log):
    results = []
    for case in cases:
        session_id = f"demo-{case['id']}"
        state = {
            "session_id": session_id, "messages": [], "query": None,
            "snippets": None, "step": 0, "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
        }
        steps = case.get("steps") or [{"turn": 1, "conversation": case["conversation"]}]
        for step in steps:
            state["messages"].append(HumanMessage(content=step["conversation"]))
            state = graph.invoke(state)

        with sqlite3.connect(LEADS_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT name, occupation, income, phone, interested_product FROM leads WHERE session_id=?",
                (session_id,),
            ).fetchall()
        actual_saved = len(rows) > 0
        passed = actual_saved == case["expected_save_lead_called"]

        results.append((case["id"], passed))
        print(f"\n{'=' * 70}", file=log)
        print(f"{case['id']} [{case['scenario']}] — {'PASS' if passed else 'FAIL'}", file=log)
        for step in steps:
            print(f"  turn {step['turn']}: {step['conversation']}", file=log)
        print(f"expected_save_lead_called: {case['expected_save_lead_called']}  actual: {actual_saved}", file=log)
        print(f"DB row(s): {rows}", file=log)
        print("--- trace ---", file=log)
        print(render_dialogue(state["trace"]), file=log)
        print("--- final reply ---", file=log)
        print(state["messages"][-1].content, file=log)
    return results


# Semantic pass/fail per session_cases' own written `expectation` — 2 fixed golden
# cases, not a generic framework; each check is literally that case's expectation.
_SESSION_CHECKS = {
    "session-01": lambda transcripts: "สมหญิง" not in transcripts["sess-B"][-1],
    "session-02": lambda transcripts: any(kw in transcripts["sess-C"][-1] for kw in ("Bronze", "8,500")),
}


def run_session_cases(cases, log):
    results = []
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        graph = build_graph().compile(checkpointer=checkpointer)
        seen = set()
        for case in cases:
            transcripts: dict[str, list[str]] = {}
            print(f"\n{'=' * 70}", file=log)
            print(f"{case['id']} [{case['scenario']}]", file=log)
            for step in case["steps"]:
                sid = step["session_id"]
                config = {"configurable": {"thread_id": sid}}
                if sid not in seen:
                    payload = {
                        "session_id": sid, "messages": [HumanMessage(content=step["turn"])],
                        "query": None, "snippets": None, "step": 0,
                        "conversation_summary": None, "trace": [], "last_retrieval_relevant": None,
                    }
                    seen.add(sid)
                else:
                    payload = {"messages": [HumanMessage(content=step["turn"])]}
                result = graph.invoke(payload, config=config)
                reply = result["messages"][-1].content
                transcripts.setdefault(sid, []).append(reply)
                print(f"  [{sid}] USER: {step['turn']}", file=log)
                print(f"  [{sid}] REPLY: {reply}", file=log)

            passed = _SESSION_CHECKS[case["id"]](transcripts)
            results.append((case["id"], passed))
            print(f"Expectation: {case['expectation']}", file=log)
            print(f"Result: {'PASS' if passed else 'FAIL'}", file=log)
    return results


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    graph = build_graph().compile()  # qa_cases + lead_cases: no checkpointer needed, one turn/case-local state

    with LOG_PATH.open("w", encoding="utf-8") as log:
        print("InsureX Sales Agent — full golden-dataset run", file=log)
        print(f"{len(dataset['qa_cases'])} qa_cases + {len(dataset['lead_cases'])} lead_cases + "
              f"{len(dataset['session_cases'])} session_cases = 21 total\n", file=log)

        qa_results, retrieval_pairs = run_qa_cases(graph, dataset["qa_cases"], log)
        lead_results = run_lead_cases(graph, dataset["lead_cases"], log)
        session_results = run_session_cases(dataset["session_cases"], log)

        with sqlite3.connect(LEADS_DB_PATH) as conn:
            leads_proof = conn.execute(
                "SELECT id, session_id, name, occupation, income, phone, interested_product FROM leads ORDER BY id"
            ).fetchall()

        metrics = aggregate(retrieval_pairs)

        print(f"\n{'=' * 70}", file=log)
        print("SUMMARY", file=log)
        print(f"{'=' * 70}", file=log)
        all_results = qa_results + lead_results + session_results
        for case_id, passed in all_results:
            print(f"  {'PASS' if passed else 'FAIL'}  {case_id}", file=log)
        n_pass = sum(1 for _, p in all_results if p)
        print(f"\n{n_pass}/{len(all_results)} cases passed", file=log)
        print(f"\nRetrieval metrics ({metrics['n_cases']} qa_cases with expected_source_docs): "
              f"Hit Rate@k = {metrics['hit_rate_at_k']:.2f}, MRR = {metrics['mrr']:.2f}", file=log)

        print(f"\n{'=' * 70}", file=log)
        print("leads.sqlite — SELECT * proof (Bonus 1 output; file itself is gitignored)", file=log)
        print(f"{'=' * 70}", file=log)
        for row in leads_proof:
            print(f"  {row}", file=log)

    print(f"done — {n_pass}/{len(all_results)} passed, log written to {LOG_PATH}")
    print(f"retrieval: hit_rate@k={metrics['hit_rate_at_k']:.2f} mrr={metrics['mrr']:.2f}")


if __name__ == "__main__":
    main()
