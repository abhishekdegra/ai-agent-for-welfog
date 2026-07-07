"""
Step 4 Qdrant infrastructure verification.

Does not ingest vectors. Verifies config, health, and collection readiness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.qdrant_service import (
    ensure_qdrant_collection,
    init_qdrant_on_startup,
    qdrant_config,
    qdrant_health_check,
)
from support_paths import ENV_FILE


def run() -> None:
    load_dotenv(ENV_FILE)
    cfg = qdrant_config()
    print("Qdrant config:", json.dumps(cfg or {"enabled": False}, ensure_ascii=False))
    print("Health:", json.dumps(qdrant_health_check(), ensure_ascii=False))
    print("Collection:", json.dumps(ensure_qdrant_collection(), ensure_ascii=False))
    print("Startup:", json.dumps(init_qdrant_on_startup(), ensure_ascii=False))


if __name__ == "__main__":
    run()
