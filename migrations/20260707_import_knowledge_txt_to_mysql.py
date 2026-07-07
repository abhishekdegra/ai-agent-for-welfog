"""
Step 2 one-time migration/import pipeline.

Reads support/knowledge/*.txt and imports one row per file into MySQL
knowledge_documents table.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mysql_service import import_knowledge_txt_files_to_mysql
from support_paths import ENV_FILE, KNOWLEDGE_DIR


def run() -> None:
    load_dotenv(ENV_FILE)
    result = import_knowledge_txt_files_to_mysql(KNOWLEDGE_DIR)
    print(f"Imported files: {result['imported']}")
    print(f"Skipped duplicates: {result['skipped_duplicates']}")
    print(f"Total rows in knowledge_documents: {result['total_rows']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    run()
