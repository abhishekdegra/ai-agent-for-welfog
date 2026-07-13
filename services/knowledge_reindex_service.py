"""
Step 8: Automatic knowledge re-indexing pipeline.

Orchestrates clean -> chunk -> embed -> Qdrant upsert on document create/update/delete.
"""
from __future__ import annotations

import os
from typing import Any

from services.knowledge_embedding_indexer import index_knowledge_document
from services.mysql_service import (
    _infer_knowledge_category_from_title,
    bump_knowledge_document_version,
    create_knowledge_document_record,
    delete_knowledge_document_chunks,
    delete_knowledge_document_chunks_except_version,
    delete_knowledge_document_record,
    get_knowledge_document_by_id,
    get_knowledge_document_by_title,
    process_knowledge_document_chunks,
    set_knowledge_document_index_status,
)
from services.qdrant_service import (
    delete_knowledge_vectors_by_doc_id,
    delete_knowledge_vectors_by_doc_id_except_version,
    ensure_qdrant_collection,
    is_qdrant_configured,
    qdrant_config,
)


def _chunk_size() -> int:
    try:
        return max(200, int(os.getenv("KNOWLEDGE_CHUNK_SIZE") or "900"))
    except (TypeError, ValueError):
        return 900


def _chunk_overlap() -> int:
    try:
        return max(0, int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP") or "120"))
    except (TypeError, ValueError):
        return 120


def _qdrant_collection() -> str | None:
    cfg = qdrant_config() or {}
    return cfg.get("collection")


def _delete_qdrant_vectors_for_doc(doc_id: int) -> dict[str, Any]:
    if not is_qdrant_configured():
        return {"ok": True, "skipped": True, "reason": "qdrant_disabled"}
    collection = _qdrant_collection()
    if not collection:
        return {"ok": False, "error": "qdrant_collection_missing"}
    ensure_qdrant_collection()
    try:
        delete_knowledge_vectors_by_doc_id(collection, int(doc_id))
        return {"ok": True, "doc_id": int(doc_id), "collection": collection}
    except Exception as exc:
        return {"ok": False, "doc_id": int(doc_id), "error": str(exc)}


def _chunk_and_embed_document(
    doc_id: int,
    *,
    replace_all_versions: bool = False,
) -> dict[str, Any]:
    chunk_result = process_knowledge_document_chunks(
        doc_id,
        chunk_size=_chunk_size(),
        overlap=_chunk_overlap(),
        replace_all_versions=replace_all_versions,
    )
    if not chunk_result.get("ok"):
        return {
            "ok": False,
            "doc_id": int(doc_id),
            "stage": "chunk",
            "error": chunk_result.get("error"),
            "chunk_result": chunk_result,
        }

    index_result = index_knowledge_document(doc_id)
    ok = bool(index_result.get("ok"))
    return {
        "ok": ok,
        "doc_id": int(doc_id),
        "version": chunk_result.get("version"),
        "chunks": chunk_result.get("chunks", 0),
        "chunks_indexed": index_result.get("chunks_indexed", 0),
        "chunks_skipped": index_result.get("chunks_skipped", 0),
        "chunks_failed": index_result.get("chunks_failed", 0),
        "index_status": "indexed" if ok else "failed",
        "chunk_result": chunk_result,
        "index_result": index_result,
    }


def reindex_knowledge_document_on_create(doc_id: int) -> dict[str, Any]:
    """
    Create flow: pending -> processing -> pending_ready -> indexed/failed.
    Clean -> Chunk -> Embed -> Upsert.
    """
    doc = get_knowledge_document_by_id(doc_id)
    if not doc:
        return {"ok": False, "operation": "create", "doc_id": doc_id, "error": "document_not_found"}

    print(f"[kb-reindex] create doc_id={doc_id} title={doc.get('title')}", flush=True)
    set_knowledge_document_index_status(doc_id, "pending")
    result = _chunk_and_embed_document(doc_id, replace_all_versions=False)
    result["operation"] = "create"
    print(f"[kb-reindex] create done doc_id={doc_id} ok={result.get('ok')}", flush=True)
    return result


def _delete_qdrant_vectors_except_version(doc_id: int, keep_version: int) -> dict[str, Any]:
    if not is_qdrant_configured():
        return {"ok": True, "skipped": True, "reason": "qdrant_disabled"}
    collection = _qdrant_collection()
    if not collection:
        return {"ok": False, "error": "qdrant_collection_missing"}
    ensure_qdrant_collection()
    try:
        delete_knowledge_vectors_by_doc_id_except_version(
            collection, int(doc_id), int(keep_version)
        )
        return {
            "ok": True,
            "doc_id": int(doc_id),
            "keep_version": int(keep_version),
            "collection": collection,
        }
    except Exception as exc:
        return {"ok": False, "doc_id": int(doc_id), "error": str(exc)}


def reindex_knowledge_document_on_update(doc_id: int, content: str) -> dict[str, Any]:
    """
    Update flow (atomic for chat):
    - bump version + mark index_status=processing (excluded from retrieval)
    - keep previous Qdrant vectors until new version is upserted
    - regenerate chunks + embeddings for the new version
    - upsert new vectors, then delete older version vectors
    - drop older MySQL chunks
    - indexer sets index_status=indexed (document becomes retrievable again)
    """
    doc = get_knowledge_document_by_id(doc_id)
    if not doc:
        return {"ok": False, "operation": "update", "doc_id": doc_id, "error": "document_not_found"}

    print(f"[kb-reindex] update doc_id={doc_id} title={doc.get('title')}", flush=True)

    bump = bump_knowledge_document_version(doc_id, content)
    if not bump.get("ok"):
        return {
            "ok": False,
            "operation": "update",
            "doc_id": doc_id,
            "error": bump.get("error"),
            "stage": "version_bump",
        }

    old_version = bump.get("old_version")
    new_version = bump.get("new_version")

    # Do NOT delete Qdrant first — processing status already hides this doc from retrieval.
    result = _chunk_and_embed_document(doc_id, replace_all_versions=False)
    if not result.get("ok"):
        result.update(
            {
                "operation": "update",
                "old_version": old_version,
                "new_version": new_version,
            }
        )
        print(
            f"[kb-reindex] update failed doc_id={doc_id} v{old_version}->v{new_version}",
            flush=True,
        )
        return result

    qdrant_del = _delete_qdrant_vectors_except_version(doc_id, int(new_version or 1))
    if not qdrant_del.get("ok") and not qdrant_del.get("skipped"):
        set_knowledge_document_index_status(doc_id, "failed")
        return {
            "ok": False,
            "operation": "update",
            "doc_id": doc_id,
            "stage": "qdrant_prune_old",
            "error": qdrant_del.get("error"),
            "old_version": old_version,
            "new_version": new_version,
            "chunk_result": result.get("chunk_result"),
            "index_result": result.get("index_result"),
        }

    delete_knowledge_document_chunks_except_version(doc_id, int(new_version or 1))

    result.update(
        {
            "operation": "update",
            "old_version": old_version,
            "new_version": new_version,
            "qdrant_delete": qdrant_del,
        }
    )
    print(
        f"[kb-reindex] update done doc_id={doc_id} v{old_version}->v{new_version} ok={result.get('ok')}",
        flush=True,
    )
    return result


def reindex_knowledge_document_on_delete(doc_id: int) -> dict[str, Any]:
    """Delete flow: hide from retrieval first, then remove Qdrant + MySQL."""
    doc = get_knowledge_document_by_id(doc_id)
    if not doc:
        return {"ok": True, "operation": "delete", "doc_id": doc_id, "already_deleted": True}

    print(f"[kb-reindex] delete doc_id={doc_id} title={doc.get('title')}", flush=True)

    # Exclude from grounded retrieval before vectors disappear.
    set_knowledge_document_index_status(doc_id, "processing")

    qdrant_del = _delete_qdrant_vectors_for_doc(doc_id)
    chunks_deleted = delete_knowledge_document_chunks(doc_id)
    mysql_deleted = delete_knowledge_document_record(doc_id)

    ok = mysql_deleted and (qdrant_del.get("ok") or qdrant_del.get("skipped"))
    result = {
        "ok": ok,
        "operation": "delete",
        "doc_id": int(doc_id),
        "chunks_deleted": chunks_deleted,
        "mysql_deleted": mysql_deleted,
        "qdrant_delete": qdrant_del,
    }
    print(f"[kb-reindex] delete done doc_id={doc_id} ok={ok}", flush=True)
    return result


def sync_admin_txt_create(title: str, content: str) -> dict[str, Any]:
    """Admin .txt create -> MySQL insert + full reindex pipeline."""
    t = (title or "").strip()
    if not t:
        return {"ok": False, "error": "empty_title"}

    existing = get_knowledge_document_by_title(t)
    if existing:
        return sync_admin_txt_update(t, content)

    doc_id = create_knowledge_document_record(
        t,
        content or "",
        category=_infer_knowledge_category_from_title(t),
    )
    if not doc_id:
        return {"ok": False, "error": "mysql_insert_failed", "title": t}

    reindex = reindex_knowledge_document_on_create(doc_id)
    return {"ok": reindex.get("ok"), "title": t, "doc_id": doc_id, "reindex": reindex}


def sync_admin_txt_update(title: str, content: str) -> dict[str, Any]:
    """Admin .txt update -> version bump + reindex pipeline."""
    t = (title or "").strip()
    doc = get_knowledge_document_by_title(t)
    if not doc:
        return sync_admin_txt_create(t, content)

    doc_id = int(doc["id"])
    reindex = reindex_knowledge_document_on_update(doc_id, content or "")
    return {"ok": reindex.get("ok"), "title": t, "doc_id": doc_id, "reindex": reindex}


def sync_admin_txt_delete(title: str) -> dict[str, Any]:
    """Admin .txt delete -> remove chunks, vectors, MySQL row."""
    t = (title or "").strip()
    doc = get_knowledge_document_by_title(t)
    if not doc:
        return {"ok": True, "title": t, "already_deleted": True}

    doc_id = int(doc["id"])
    reindex = reindex_knowledge_document_on_delete(doc_id)
    return {"ok": reindex.get("ok"), "title": t, "doc_id": doc_id, "reindex": reindex}


def resume_failed_knowledge_reindex(doc_id: int) -> dict[str, Any]:
    """
    Idempotent resume:
    - failed/pending -> full create pipeline
    - pending_ready -> embed only
  - processing with chunks -> embed only
    """
    doc = get_knowledge_document_by_id(doc_id)
    if not doc:
        return {"ok": False, "doc_id": doc_id, "error": "document_not_found"}

    status = (doc.get("index_status") or "").strip().lower()
    if status == "pending_ready":
        index_result = index_knowledge_document(doc_id)
        return {"ok": index_result.get("ok"), "doc_id": doc_id, "resume": "embed", "index_result": index_result}
    if status in ("pending", "failed", "processing"):
        return reindex_knowledge_document_on_create(doc_id)
    if status == "indexed":
        return {"ok": True, "doc_id": doc_id, "resume": "already_indexed"}
    return reindex_knowledge_document_on_create(doc_id)
