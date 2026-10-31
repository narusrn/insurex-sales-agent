"""GET /runs — JSON dump of db/runs.jsonl, newest first, capped. Powers the
usage dashboard (GET /dashboard). Requires X-API-Key, same as /chat.
"""
import json

from fastapi import APIRouter, Depends

from agents.runner import RUNS_LOG_PATH

from ..auth import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])

MAX_RUNS = 500


@router.get("/runs")
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
