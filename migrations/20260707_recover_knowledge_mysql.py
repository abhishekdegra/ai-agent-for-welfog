"""
Recover knowledge_documents / knowledge_document_chunks after MySQL #1932 ghost tables.

Reuses existing repair + migration SQL + import + chunking + optional Qdrant index.
Does not redefine schema. Safe to run multiple times.

Run from support/:
  python migrations/20260707_recover_knowledge_mysql.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.knowledge_embedding_indexer import (  # noqa: E402
    reconcile_knowledge_vectors_after_mysql_recovery,
)
from services.mysql_service import (  # noqa: E402
    ensure_knowledge_mysql_ready,
    get_mysql_connection,
    repair_mysql_knowledge_tables_if_broken,
)
from services.qdrant_service import qdrant_config, qdrant_health_check  # noqa: E402
from support_paths import ENV_FILE  # noqa: E402


def _verify() -> dict:
    out: dict = {
        "knowledge_documents": None,
        "knowledge_document_chunks": None,
        "document_count": 0,
        "chunk_count": 0,
        "qdrant": {},
    }
    conn = get_mysql_connection()
    if not conn:
        out["error"] = "mysql_unreachable"
        return out
    try:
        with conn.cursor() as cur:
            for table in ("knowledge_documents", "knowledge_document_chunks"):
                try:
                    cur.execute(f"SELECT 1 FROM `{table}` LIMIT 1")
                    out[table] = "ok"
                except Exception as e:
                    out[table] = f"error: {e}"
            if out["knowledge_documents"] == "ok":
                cur.execute("SELECT COUNT(*) AS c FROM knowledge_documents")
                out["document_count"] = int((cur.fetchone() or {}).get("c") or 0)
            if out["knowledge_document_chunks"] == "ok":
                cur.execute("SELECT COUNT(*) AS c FROM knowledge_document_chunks")
                out["chunk_count"] = int((cur.fetchone() or {}).get("c") or 0)
    finally:
        conn.close()

    health = qdrant_health_check()
    cfg = qdrant_config() or {}
    out["qdrant"] = {
        "health": health,
        "collection": cfg.get("collection"),
    }
    return out


def run() -> None:
    load_dotenv(ENV_FILE)
    repaired = repair_mysql_knowledge_tables_if_broken()
    print(f"Knowledge tables repaired: {repaired}")
    ready = ensure_knowledge_mysql_ready()
    print("MySQL ready:", json.dumps(ready, ensure_ascii=False, indent=2))
    vectors = reconcile_knowledge_vectors_after_mysql_recovery()
    print("Vectors:", json.dumps(vectors, ensure_ascii=False, indent=2))
    print("Verify:", json.dumps(_verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
