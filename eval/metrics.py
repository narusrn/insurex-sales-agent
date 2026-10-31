"""Standard IR/RAG retrieval metrics — Hit Rate@k and MRR. See
implementation-plan.md §5. Deliberately not a framework (RAGAS/DeepEval):
these two formulas are all a 5-doc corpus needs, computed against the
`expected_source_docs` already in eval/golden_dataset.json.
"""
import re

_SOURCE_TAG = re.compile(r"^\[([^\]]+)\]")


def extract_sources(snippets: list[str]) -> list[str]:
    """Pull the "[doc.pdf]" tag retrieve_knowledge_base prefixes each chunk
    with (agents/subgraphs/rag.py) back out, in retrieval order, for Hit
    Rate@k/MRR. A chunk with no tag (shouldn't happen) is just skipped."""
    sources = []
    for s in snippets:
        m = _SOURCE_TAG.match(s)
        if m:
            sources.append(m.group(1))
    return sources


def hit_rate(expected_docs: list[str], retrieved_docs: list[str]) -> bool:
    """True if any expected doc appears anywhere in the retrieved top-k."""
    return bool(set(expected_docs) & set(retrieved_docs))


def mrr(expected_docs: list[str], retrieved_docs: list[str]) -> float:
    """1/rank of the first expected doc in retrieved_docs, 0.0 if none hit."""
    expected = set(expected_docs)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            return 1 / rank
    return 0.0


def aggregate(per_case_results: list[tuple[list[str], list[str]]]) -> dict:
    """per_case_results: [(expected_docs, retrieved_docs), ...] — only pass
    cases with a non-empty expected_docs (skip small_talk/out_of_scope; those
    test fallback behavior, not retrieval, and have none by design)."""
    hits = [hit_rate(exp, ret) for exp, ret in per_case_results]
    rrs = [mrr(exp, ret) for exp, ret in per_case_results]
    n = len(per_case_results)
    return {
        "n_cases": n,
        "hit_rate_at_k": sum(hits) / n if n else 0.0,
        "mrr": sum(rrs) / n if n else 0.0,
    }


if __name__ == "__main__":
    # self-check
    assert extract_sources(["[a.pdf] hello", "[b.pdf] world"]) == ["a.pdf", "b.pdf"]
    assert extract_sources(["no tag here"]) == []
    assert hit_rate(["a.pdf"], ["b.pdf", "a.pdf"]) is True
    assert hit_rate(["a.pdf"], ["b.pdf", "c.pdf"]) is False
    assert mrr(["a.pdf"], ["a.pdf", "b.pdf"]) == 1.0
    assert mrr(["a.pdf"], ["b.pdf", "a.pdf"]) == 0.5
    assert mrr(["a.pdf"], ["b.pdf", "c.pdf"]) == 0.0
    result = aggregate([(["a.pdf"], ["a.pdf"]), (["a.pdf"], ["b.pdf", "a.pdf"])])
    assert result == {"n_cases": 2, "hit_rate_at_k": 1.0, "mrr": 0.75}
    print("eval/metrics.py self-check OK")
