"""GET /runs — JSON dump of db/runs.jsonl, newest first, capped. Powers the
usage dashboard (GET /dashboard). Requires X-API-Key, same as /chat.
"""
import json

from fastapi import APIRouter

from agents.runner import RUNS_LOG_PATH

from ..schemas.runs import RunRecord

router = APIRouter(tags=["runs"])

MAX_RUNS = 500


@router.get(
    "/runs",
    response_model=list[RunRecord],
    summary="List recent chat turns with cost/usage/trace",
    description=(
        f"Returns up to {MAX_RUNS} most recent chat turns (one record per `POST /chat` call), "
        "newest first. Each record has the full turn: the message and reply, which tools ran, "
        "latency, token usage broken down per LLM call, an approximate USD cost estimate, and "
        "the step-by-step trace. This is a direct read of `db/runs.jsonl` — nothing is "
        "aggregated or paginated server-side."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {
                            "timestamp": "2026-11-01T03:12:45.102+00:00",
                            "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "message": "Health Plus มีแผนอะไรบ้าง",
                            "reply": "Health Plus มี 3 แผนหลักครับ ...",
                            "tools_used": ["retrieve"],
                            "latency": 2.31,
                            "success": True,
                            "error": None,
                            "llm_calls": 3,
                            "usage": {
                                "input_tokens": 1540,
                                "output_tokens": 210,
                                "total_tokens": 1750,
                                "by_node": [
                                    {"node": "grade_documents", "input_tokens": 420, "output_tokens": 12, "total_tokens": 432},
                                    {"node": "supervisor", "input_tokens": 1120, "output_tokens": 198, "total_tokens": 1318},
                                ],
                            },
                            "estimated_cost_usd": 0.000357,
                            "cost_note": "approximate — hardcoded gpt-4o-mini per-token rates, may not reflect current pricing",
                            "trace": ["retrieve: 4 snippets", "grade_documents: relevant", "supervisor: replied"],
                        }
                    ]
                }
            }
        }
    },
)
async def runs() -> list[dict]:
    if not RUNS_LOG_PATH.exists():
        return []
    records = []
    with RUNS_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return list(reversed(records))[:MAX_RUNS]
