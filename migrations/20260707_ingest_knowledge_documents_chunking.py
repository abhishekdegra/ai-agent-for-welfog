"""
Step 3 ingestion-only pipeline.

- Cleans knowledge_documents content
- Deterministically chunks documents
- Stores chunk metadata/content in knowledge_document_chunks
- Updates index_status: pending -> processing -> pending_ready
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mysql_service import (
    get_mysql_connection,
    ingest_knowledge_documents_for_chunking,
)
from support_paths import ENV_FILE


def run(chunk_size: int = 900, overlap: int = 120) -> None:
    load_dotenv(ENV_FILE)
    result = ingest_knowledge_documents_for_chunking(chunk_size=chunk_size, overlap=overlap)

    conn = get_mysql_connection()
    total_docs = 0
    total_chunks = 0
    avg_chunk_size = 0.0
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM knowledge_documents")
                total_docs = int((cur.fetchone() or {}).get("c") or 0)
                cur.execute("SELECT COUNT(*) AS c FROM knowledge_document_chunks")
                total_chunks = int((cur.fetchone() or {}).get("c") or 0)
                cur.execute("SELECT AVG(CHAR_LENGTH(content)) AS avg_len FROM knowledge_document_chunks")
                avg_chunk_size = float((cur.fetchone() or {}).get("avg_len") or 0.0)
        finally:
            conn.close()

    print(f"Processed documents: {result['documents']}")
    print(f"Generated chunks: {result['chunks']}")
    print(f"Pipeline average chunk size: {result['avg_chunk_size']:.2f}")
    print(f"Total documents: {total_docs}")
    print(f"Total chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.2f}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    run()
