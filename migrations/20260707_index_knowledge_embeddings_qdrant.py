"""
Step 5: OpenAI embedding indexer -> Qdrant.

Run:
  python migrations/20260707_index_knowledge_embeddings_qdrant.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.knowledge_embedding_indexer import index_pending_ready_knowledge_chunks
from services.qdrant_service import qdrant_health_check
from support_paths import ENV_FILE


def run() -> None:
    load_dotenv(ENV_FILE)
    result = index_pending_ready_knowledge_chunks()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Qdrant health:", json.dumps(qdrant_health_check(), ensure_ascii=False))


if __name__ == "__main__":
    run()
