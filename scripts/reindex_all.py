"""
Fresh-machine / production bootstrap: chunk + embed all active knowledge into Qdrant.

Safe to re-run (idempotent per document). Does not change Brain/routing/APIs.

Usage (from repo root, with .env configured and Qdrant up):
  python scripts/reindex_all.py

Requires:
  - MySQL reachable (knowledge_documents populated via Admin or SQL/import)
  - QDRANT_ENABLED=1 and QDRANT_URL pointing at docker compose Qdrant
  - OPENAI_API_KEY for embeddings
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_paths import ENV_FILE

load_dotenv(ENV_FILE)


def run() -> int:
    from services.knowledge_reindex_service import reindex_knowledge_document_on_create
    from services.mysql_service import (
        get_mysql_connection,
        init_mysql_knowledge_chunks_schema,
        init_mysql_knowledge_documents_schema,
    )
    from services.qdrant_service import init_qdrant_on_startup, is_qdrant_configured, qdrant_health_check

    print("[reindex_all] ensuring MySQL knowledge schemas...", flush=True)
    init_mysql_knowledge_documents_schema()
    init_mysql_knowledge_chunks_schema()

    if not is_qdrant_configured():
        print(
            "[reindex_all] ERROR: Qdrant not configured. "
            "Set QDRANT_ENABLED=1 and QDRANT_URL=http://127.0.0.1:6333 in .env",
            flush=True,
        )
        return 1

    print("[reindex_all] connecting to Qdrant...", flush=True)
    startup = init_qdrant_on_startup()
    health = qdrant_health_check()
    print(json.dumps({"startup": startup, "health": health}, ensure_ascii=False), flush=True)
    if not health.get("reachable"):
        print(
            "[reindex_all] ERROR: Qdrant unreachable. Run: docker compose up -d",
            flush=True,
        )
        return 1

    conn = get_mysql_connection()
    if not conn:
        print("[reindex_all] ERROR: MySQL unreachable. Check MYSQL_* in .env", flush=True)
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, index_status, status
                FROM knowledge_documents
                WHERE status = 'active'
                ORDER BY id ASC
                """
            )
            docs = list(cur.fetchall() or [])
    finally:
        conn.close()

    if not docs:
        print(
            "[reindex_all] No active knowledge_documents rows. "
            "Create docs in Admin, or import SQL / run "
            "python migrations/20260707_import_knowledge_txt_to_mysql.py",
            flush=True,
        )
        return 0

    print(f"[reindex_all] reindexing {len(docs)} active document(s)...", flush=True)
    ok_n = 0
    fail_n = 0
    results: list[dict] = []
    for doc in docs:
        doc_id = int(doc["id"])
        title = (doc.get("title") or "").strip()
        print(f"[reindex_all] -> id={doc_id} title={title!r}", flush=True)
        try:
            result = reindex_knowledge_document_on_create(doc_id)
        except Exception as exc:
            result = {"ok": False, "doc_id": doc_id, "error": str(exc)}
        results.append(
            {
                "doc_id": doc_id,
                "title": title,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
            }
        )
        if result.get("ok"):
            ok_n += 1
        else:
            fail_n += 1
            print(f"[reindex_all] FAIL id={doc_id}: {result.get('error') or result}", flush=True)

    summary = {
        "ok": fail_n == 0,
        "documents_total": len(docs),
        "documents_ok": ok_n,
        "documents_failed": fail_n,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    raise SystemExit(run())
