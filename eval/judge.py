"""LLM-as-judge groundedness check — a second, independent LLM call grades
whether a qa_case's final reply is faithful to the snippets actually
retrieved for it, with no pre-written expected answer to match against (a
good reply can be phrased many ways, so the real retrieved text is the
ground truth, judged fresh each run — same reasoning the reference eval
harness this project is checked against uses). Complements
eval/metrics.py's retrieval-only Hit Rate@k/MRR and run_full_eval.py's
strict keypoint substring match, neither of which can tell "right answer,
different phrasing" apart from "wrong answer" — confirmed a real example of
this gap: qa-02 states the correct premium figure but fails the strict
keypoint check on a secondary phrase.
"""
from pydantic import BaseModel, Field

from agents.llm import get_llm


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(
        description="True only if every specific fact/number/condition in the reply is "
        "actually supported by the snippets. False if it states anything the snippets "
        "don't support. A reply that honestly says it couldn't find clear information is "
        "always grounded."
    )
    rationale: str = Field(description="One sentence: what in the reply is or isn't supported.")


_JUDGE_PROMPT = """You are grading whether an insurance sales agent's reply is faithful to \
the knowledge-base snippets it retrieved -- not whether it's a good or complete answer, only \
whether it avoids stating anything the snippets don't actually support.

Retrieved snippets:
{snippets}

Agent's reply:
{reply}
"""


def judge_groundedness(reply: str, snippets: list[str]) -> GroundednessVerdict:
    judge = get_llm(temperature=0).with_structured_output(GroundednessVerdict)
    prompt = _JUDGE_PROMPT.format(snippets="\n---\n".join(snippets), reply=reply)
    return judge.invoke(prompt)


if __name__ == "__main__":
    # smoke test — one obviously-grounded reply, one obviously-fabricated one
    from dotenv import load_dotenv
    load_dotenv()

    snippet = ["[02_premium_coverage_table.pdf] แผน Bronze: วงเงิน 500,000 บาท, "
               "อายุ 0-20 ปี เบี้ยประกัน 8,500 บาท"]

    grounded = judge_groundedness(
        "แผน Bronze วงเงิน 500,000 บาท เบี้ยประกัน 8,500 บาทสำหรับอายุ 0-20 ปี", snippet,
    )
    assert grounded.grounded is True, grounded

    ungrounded = judge_groundedness(
        "แผน Bronze วงเงิน 999,999,999 บาท ไม่มีเงื่อนไขใดๆ ทั้งสิ้น", snippet,
    )
    assert ungrounded.grounded is False, ungrounded
    print("eval/judge.py smoke test OK")
