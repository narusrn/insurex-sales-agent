"""GET /leads — every customer lead saved so far (db/leads.sqlite), newest
first. Requires X-API-Key, same as /chat and /runs.
"""
import sqlite3

from fastapi import APIRouter

from agents.tools import LEADS_DB_PATH

from ..schemas.leads import Lead

router = APIRouter(tags=["leads"])


@router.get(
    "/leads",
    response_model=list[Lead],
    summary="List customers who've shown buying interest",
    description=(
        "Every lead `save_lead` has persisted so far — one row per confirmed, complete "
        "submission (name, occupation, income, phone all present and explicitly confirmed "
        "by the customer; see `README.md` \"Lead collection\"). Newest first. Returns `[]` "
        "before the first lead is ever saved (the table is created lazily, on first save)."
    ),
)
async def leads() -> list[dict]:
    if not LEADS_DB_PATH.exists():
        return []
    with sqlite3.connect(LEADS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, session_id, name, occupation, income, phone, "
            "interested_product, inquiry_summary, conversation_log, created_at "
            "FROM leads ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]
