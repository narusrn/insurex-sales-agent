"""Structured JSON stdout logging — [§9] Layer 1 only, no Mongo/cost layer.
See implementation-plan.md §1's "Kept, not dropped" note.
"""
import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


if __name__ == "__main__":
    # smoke test — Phase 0.4: one log call prints valid JSON
    setup_logging()
    logging.getLogger(__name__).info("logging_conf smoke test", extra={"session_id": "demo-1"})
