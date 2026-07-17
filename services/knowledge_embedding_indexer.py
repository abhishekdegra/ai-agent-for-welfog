"""
Step 5: OpenAI embedding indexer for knowledge_document_chunks -> Qdrant.

Does not modify existing RAG/retrieval/chat flows.
"""
from __future__ import annotations

import os
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import requests

from services.mysql_service import (
    get_mysql_connection,
    init_mysql_knowledge_chunks_schema,
    set_knowledge_document_index_status,
)
from services.qdrant_service import (
    ensure_qdrant_collection,
    get_qdrant_client,
    is_qdrant_configured,
    qdrant_config,
    qdrant_health_check,
    retrieve_knowledge_point_hashes,
    upsert_knowledge_vectors,
)

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_DIM = 1536


def _embedding_model() -> str:
    return (os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_EMBED_MODEL).strip()


def _embedding_dimensions() -> int:
    try:
        return int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or os.getenv("QDRANT_VECTOR_SIZE") or DEFAULT_EMBED_DIM)
    except (TypeError, ValueError):
        return DEFAULT_EMBED_DIM


def _batch_size() -> int:
    try:
        return max(1, min(128, int(os.getenv("KNOWLEDGE_EMBED_BATCH_SIZE") or "32")))
    except (TypeError, ValueError):
        return 32


def _openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def qdrant_point_id(chunk_key: str) -> str:
    """Deterministic UUID from doc_id:version:chunk_no."""
    return str(uuid5(NAMESPACE_URL, f"welfog:kb:{chunk_key}"))


def _chunk_payload(row: dict[str, Any]) -> dict[str, Any]:
    from services.knowledge_keys import canonical_knowledge_key

    raw_title = (row.get("title") or "").strip()
    return {
        "doc_id": int(row["doc_id"]),
        "version": int(row["version"]),
        "chunk_no": int(row["chunk_no"]),
        "title": canonical_knowledge_key(raw_title) or raw_title,
        "category": (row.get("category") or "general").strip() or "general",
        "language": (row.get("language") or "auto").strip() or "auto",
        "content": row.get("content") or "",
        "content_hash": (row.get("content_hash") or "").strip(),
    }


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    try:
        from services.chat_resilience import chat_turn_abandoned

        if chat_turn_abandoned():
            raise RuntimeError("chat_turn_abandoned")
    except ImportError:
        pass
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    if not texts:
        return []

    model = _embedding_model()
    dimensions = _embedding_dimensions()
    timeout = float(os.getenv("OPENAI_EMBED_TIMEOUT_SEC") or "5")

    resp = requests.post(
        OPENAI_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": texts,
            "dimensions": dimensions,
        },
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI embeddings HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    items = sorted(data.get("data") or [], key=lambda x: int(x.get("index") or 0))
    vectors: list[list[float]] = []
    for item in items:
        vec = item.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("OpenAI embeddings response missing vector")
        vectors.append(vec)
    if len(vectors) != len(texts):
        raise RuntimeError(f"OpenAI embeddings count mismatch ({len(vectors)} != {len(texts)})")
    return vectors


def _fetch_pending_ready_documents(cur) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, title, category, language, version
        FROM knowledge_documents
        WHERE status = 'active' AND index_status = 'pending_ready'
        ORDER BY id ASC
        """
    )
    return list(cur.fetchall() or [])


def _fetch_document_chunks(cur, doc_id: int, version: int) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT chunk_id, doc_id, version, chunk_no, title, category, language, content, content_hash
        FROM knowledge_document_chunks
        WHERE doc_id = %s AND version = %s
        ORDER BY chunk_no ASC
        """,
        (doc_id, version),
    )
    return list(cur.fetchall() or [])


def _set_document_index_status(cur, doc_id: int, status: str) -> None:
    cur.execute(
        "UPDATE knowledge_documents SET index_status = %s WHERE id = %s",
        (status, doc_id),
    )


def index_knowledge_document(doc_id: int) -> dict[str, Any]:
    """
    Embed + upsert one document's current-version chunks into Qdrant.
    Expects index_status=pending_ready (or resumable failed with chunks present).
    """
    if not is_qdrant_configured():
        set_knowledge_document_index_status(doc_id, "failed")
        return {"ok": False, "doc_id": doc_id, "error": "qdrant_not_configured"}

    health = qdrant_health_check()
    if not health.get("reachable"):
        set_knowledge_document_index_status(doc_id, "failed")
        return {"ok": False, "doc_id": doc_id, "error": f"qdrant_unreachable: {health.get('detail')}"}

    collection_init = ensure_qdrant_collection()
    if not collection_init.get("ok"):
        set_knowledge_document_index_status(doc_id, "failed")
        return {
            "ok": False,
            "doc_id": doc_id,
            "error": f"qdrant_collection_init_failed: {collection_init.get('detail')}",
        }

    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        set_knowledge_document_index_status(doc_id, "failed")
        return {"ok": False, "doc_id": doc_id, "error": "mysql_unreachable"}

    cfg = qdrant_config() or {}
    collection = cfg.get("collection")
    batch_size = _batch_size()
    chunks_indexed = 0
    chunks_skipped = 0
    chunks_failed = 0
    failed_chunks: list[str] = []

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, version, index_status FROM knowledge_documents WHERE id = %s LIMIT 1",
                (int(doc_id),),
            )
            doc = cur.fetchone()
            if not doc:
                return {"ok": False, "doc_id": doc_id, "error": "document_not_found"}

            version = int(doc.get("version") or 1)
            chunks = _fetch_document_chunks(cur, doc_id, version)
            if not chunks:
                _set_document_index_status(cur, doc_id, "failed")
                conn.commit()
                return {"ok": False, "doc_id": doc_id, "error": "no_chunks", "chunks_indexed": 0}

            point_ids = [qdrant_point_id(c["chunk_id"]) for c in chunks]
            existing_hashes = retrieve_knowledge_point_hashes(collection, point_ids)

            todo: list[dict[str, Any]] = []
            for chunk, pid in zip(chunks, point_ids):
                expected_hash = (chunk.get("content_hash") or "").strip()
                if existing_hashes.get(pid) == expected_hash and expected_hash:
                    chunks_skipped += 1
                    continue
                todo.append({**chunk, "_point_id": pid})

            if not todo:
                _set_document_index_status(cur, doc_id, "indexed")
                conn.commit()
                return {
                    "ok": True,
                    "doc_id": doc_id,
                    "version": version,
                    "chunks_indexed": 0,
                    "chunks_skipped": chunks_skipped,
                    "chunks_failed": 0,
                }

            doc_failed = False
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                texts = [b.get("content") or "" for b in batch]
                try:
                    vectors = embed_texts_openai(texts)
                    points = []
                    for row, vector in zip(batch, vectors):
                        points.append(
                            {
                                "point_id": row["_point_id"],
                                "vector": vector,
                                "payload": _chunk_payload(row),
                            }
                        )
                    upsert_knowledge_vectors(collection, points)
                    chunks_indexed += len(batch)
                    time.sleep(0.05)
                except Exception as batch_exc:
                    doc_failed = True
                    for row in batch:
                        cid = row.get("chunk_id") or f"{doc_id}:{version}:{row.get('chunk_no')}"
                        failed_chunks.append(str(cid))
                        chunks_failed += 1
                    print(f"[kb-index] doc_id={doc_id} batch failed: {batch_exc}", flush=True)
                    break

            if doc_failed:
                _set_document_index_status(cur, doc_id, "failed")
            else:
                _set_document_index_status(cur, doc_id, "indexed")
            conn.commit()
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        set_knowledge_document_index_status(doc_id, "failed")
        return {
            "ok": False,
            "doc_id": doc_id,
            "error": str(exc),
            "chunks_indexed": chunks_indexed,
            "chunks_failed": chunks_failed,
            "failed_chunks": failed_chunks,
        }
    finally:
        conn.close()

    return {
        "ok": chunks_failed == 0,
        "doc_id": doc_id,
        "chunks_indexed": chunks_indexed,
        "chunks_skipped": chunks_skipped,
        "chunks_failed": chunks_failed,
        "failed_chunks": failed_chunks,
        "collection": collection,
    }


def index_pending_ready_knowledge_chunks() -> dict[str, Any]:
    """
    Batch embedding indexer:
    - reads pending_ready documents
    - embeds their MySQL chunks with OpenAI
    - upserts vectors into Qdrant
    - marks documents indexed/failed
    """
    if not is_qdrant_configured():
        return {
            "ok": False,
            "error": "qdrant_not_configured",
            "chunks_indexed": 0,
            "chunks_failed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
        }

    health = qdrant_health_check()
    if not health.get("reachable"):
        return {
            "ok": False,
            "error": f"qdrant_unreachable: {health.get('detail')}",
            "chunks_indexed": 0,
            "chunks_failed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
        }

    collection_init = ensure_qdrant_collection()
    if not collection_init.get("ok"):
        return {
            "ok": False,
            "error": f"qdrant_collection_init_failed: {collection_init.get('detail')}",
            "chunks_indexed": 0,
            "chunks_failed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
        }

    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return {
            "ok": False,
            "error": "mysql_unreachable",
            "chunks_indexed": 0,
            "chunks_failed": 0,
            "documents_indexed": 0,
            "documents_failed": 0,
        }

    cfg = qdrant_config() or {}
    collection = cfg.get("collection")
    batch_size = _batch_size()
    chunks_indexed = 0
    chunks_skipped = 0
    chunks_failed = 0
    documents_indexed = 0
    documents_failed = 0
    failed_chunks: list[str] = []

    try:
        with conn.cursor() as cur:
            docs = _fetch_pending_ready_documents(cur)
            print(f"[kb-index] pending_ready documents: {len(docs)}", flush=True)

            for doc in docs:
                doc_id = int(doc["id"])
                version = int(doc.get("version") or 1)
                chunks = _fetch_document_chunks(cur, doc_id, version)
                if not chunks:
                    print(f"[kb-index] doc_id={doc_id} has no chunks — marking failed", flush=True)
                    _set_document_index_status(cur, doc_id, "failed")
                    documents_failed += 1
                    conn.commit()
                    continue

                point_ids = [qdrant_point_id(c["chunk_id"]) for c in chunks]
                existing_hashes = retrieve_knowledge_point_hashes(collection, point_ids)

                todo: list[dict[str, Any]] = []
                for chunk, pid in zip(chunks, point_ids):
                    expected_hash = (chunk.get("content_hash") or "").strip()
                    if existing_hashes.get(pid) == expected_hash and expected_hash:
                        chunks_skipped += 1
                        continue
                    todo.append({**chunk, "_point_id": pid})

                if not todo:
                    _set_document_index_status(cur, doc_id, "indexed")
                    documents_indexed += 1
                    conn.commit()
                    print(f"[kb-index] doc_id={doc_id} already indexed ({len(chunks)} chunks skipped)", flush=True)
                    continue

                doc_failed = False
                for start in range(0, len(todo), batch_size):
                    batch = todo[start : start + batch_size]
                    texts = [b.get("content") or "" for b in batch]
                    try:
                        vectors = embed_texts_openai(texts)
                        points = []
                        for row, vector in zip(batch, vectors):
                            points.append(
                                {
                                    "point_id": row["_point_id"],
                                    "vector": vector,
                                    "payload": _chunk_payload(row),
                                }
                            )
                        upsert_knowledge_vectors(collection, points)
                        chunks_indexed += len(batch)
                        print(
                            f"[kb-index] doc_id={doc_id} upserted {len(batch)} chunks "
                            f"({start + len(batch)}/{len(todo)})",
                            flush=True,
                        )
                        time.sleep(0.05)
                    except Exception as batch_exc:
                        doc_failed = True
                        for row in batch:
                            cid = row.get("chunk_id") or f"{doc_id}:{version}:{row.get('chunk_no')}"
                            failed_chunks.append(str(cid))
                            chunks_failed += 1
                        print(f"[kb-index] batch failed doc_id={doc_id}: {batch_exc}", flush=True)
                        break

                if doc_failed:
                    _set_document_index_status(cur, doc_id, "failed")
                    documents_failed += 1
                else:
                    _set_document_index_status(cur, doc_id, "indexed")
                    documents_indexed += 1
                    print(f"[kb-index] doc_id={doc_id} indexed successfully", flush=True)
                conn.commit()
    except Exception as exc:
        print(f"[kb-index] pipeline error: {exc}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass
        return {
            "ok": False,
            "error": str(exc),
            "chunks_indexed": chunks_indexed,
            "chunks_skipped": chunks_skipped,
            "chunks_failed": chunks_failed,
            "documents_indexed": documents_indexed,
            "documents_failed": documents_failed,
            "failed_chunks": failed_chunks,
        }
    finally:
        conn.close()

    client = get_qdrant_client()
    qdrant_points = None
    if client and collection:
        try:
            info = client.get_collection(collection)
            qdrant_points = getattr(info, "points_count", None)
        except Exception:
            qdrant_points = None

    return {
        "ok": chunks_failed == 0 and documents_failed == 0,
        "embedding_model": _embedding_model(),
        "embedding_dimensions": _embedding_dimensions(),
        "collection": collection,
        "chunks_indexed": chunks_indexed,
        "chunks_skipped": chunks_skipped,
        "chunks_failed": chunks_failed,
        "documents_indexed": documents_indexed,
        "documents_failed": documents_failed,
        "failed_chunks": failed_chunks,
        "qdrant_points_count": qdrant_points,
    }


def _qdrant_knowledge_points_count() -> int | None:
    if not is_qdrant_configured():
        return None
    cfg = qdrant_config() or {}
    collection = cfg.get("collection")
    client = get_qdrant_client()
    if not client or not collection:
        return None
    try:
        info = client.get_collection(collection)
        return int(getattr(info, "points_count", 0) or 0)
    except Exception:
        return None


def reconcile_knowledge_vectors_after_mysql_recovery() -> dict[str, Any]:
    """
    After MySQL knowledge rebuild: index vectors only if the Qdrant collection is empty.
    If vectors already exist, mark pending_ready docs as indexed (no OpenAI re-embed).
    """
    points = _qdrant_knowledge_points_count()
    if points is None:
        return {"ok": False, "action": "skipped", "detail": "qdrant_unavailable", "qdrant_points": None}
    if points > 0:
        conn = get_mysql_connection()
        if not conn:
            return {"ok": False, "action": "skipped", "detail": "mysql_unreachable", "qdrant_points": points}
        marked = 0
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE knowledge_documents
                    SET index_status = 'indexed'
                    WHERE status = 'active'
                      AND index_status IN ('pending_ready', 'pending', 'processing')
                    """
                )
                marked = int(cur.rowcount or 0)
            conn.commit()
        except Exception as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            return {
                "ok": False,
                "action": "mark_indexed",
                "detail": str(exc),
                "qdrant_points": points,
                "documents_marked": 0,
            }
        finally:
            conn.close()
        print(
            f"[kb-index] Qdrant already has {points} points — marked {marked} docs indexed (no re-embed)",
            flush=True,
        )
        return {
            "ok": True,
            "action": "mark_indexed_existing_vectors",
            "qdrant_points": points,
            "documents_marked": marked,
        }

    print("[kb-index] Qdrant collection empty — indexing pending_ready chunks", flush=True)
    indexed = index_pending_ready_knowledge_chunks()
    return {
        "ok": bool(indexed.get("ok")),
        "action": "index_pending_ready",
        "qdrant_points": points,
        "index_result": indexed,
    }
