"""Debug session 97ca38 — NDJSON to workspace debug-97ca38.log (remove after verify)."""
from __future__ import annotations

import json
import time
from pathlib import Path

_LOG = Path(__file__).resolve().parents[4] / "debug-97ca38.log"
_SESSION = "97ca38"


def dbg97(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    *,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": _SESSION,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
            "runId": run_id,
        }
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
