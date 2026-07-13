"""
Qdrant infrastructure service (Step 4).

Prepares collection + connection management only.
No embeddings or vector ingestion in this step.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Optional

_client: Any = None
_client_lock = threading.Lock()


def _env_bool(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ensure_project_dotenv_loaded() -> None:
    if os.getenv("QDRANT_URL", "").strip() or _env_bool("QDRANT_ENABLED", "0"):
        return
    try:
        from dotenv import load_dotenv

        from support_paths import ENV_FILE

        if os.path.isfile(ENV_FILE):
            load_dotenv(ENV_FILE)
    except Exception:
        pass


def qdrant_config() -> Optional[dict[str, Any]]:
    """Return Qdrant config from env, or None when disabled/unconfigured."""
    _ensure_project_dotenv_loaded()
    if not _env_bool("QDRANT_ENABLED", "0"):
        return None
    url = (os.getenv("QDRANT_URL") or "").strip()
    if not url:
        return None
    distance = (os.getenv("QDRANT_DISTANCE") or "Cosine").strip()
    collection = (os.getenv("QDRANT_COLLECTION") or "welfog_knowledge_chunks").strip()
    try:
        vector_size = int(os.getenv("QDRANT_VECTOR_SIZE") or "1536")
    except (TypeError, ValueError):
        vector_size = 1536
    try:
        timeout_sec = float(os.getenv("QDRANT_TIMEOUT_SEC") or "5")
    except (TypeError, ValueError):
        timeout_sec = 5.0
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
    return {
        "url": url,
        "api_key": api_key,
        "collection": collection,
        "vector_size": max(1, vector_size),
        "distance": distance,
        "timeout_sec": max(1.0, timeout_sec),
    }


def is_qdrant_configured() -> bool:
    return qdrant_config() is not None


def _distance_enum(name: str):
    from qdrant_client.models import Distance

    key = (name or "Cosine").strip().lower()
    if key == "euclid":
        return Distance.EUCLID
    if key == "dot":
        return Distance.DOT
    return Distance.COSINE


def get_qdrant_client():
    """Lazy singleton Qdrant client with basic connection settings."""
    global _client
    cfg = qdrant_config()
    if not cfg:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        from qdrant_client import QdrantClient

        _client = QdrantClient(
            url=cfg["url"],
            api_key=cfg.get("api_key"),
            timeout=cfg["timeout_sec"],
        )
        return _client


def qdrant_health_check() -> dict[str, Any]:
    """
    Lightweight health probe.
    Returns: {ok, configured, reachable, collection, detail}
    """
    cfg = qdrant_config()
    if not cfg:
        return {
            "ok": True,
            "configured": False,
            "reachable": False,
            "collection": None,
            "detail": "Qdrant disabled or QDRANT_URL missing",
        }
    client = get_qdrant_client()
    if client is None:
        return {
            "ok": False,
            "configured": True,
            "reachable": False,
            "collection": cfg["collection"],
            "detail": "client_init_failed",
        }
    try:
        collections = client.get_collections()
        names = [c.name for c in (collections.collections or [])]
        return {
            "ok": True,
            "configured": True,
            "reachable": True,
            "collection": cfg["collection"],
            "collection_exists": cfg["collection"] in names,
            "detail": "connected",
        }
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "reachable": False,
            "collection": cfg["collection"],
            "detail": str(exc),
        }


def ensure_qdrant_collection() -> dict[str, Any]:
    """
    Ensure target collection exists with expected vector params.
    Does not upsert any points.
    """
    cfg = qdrant_config()
    if not cfg:
        return {"ok": True, "configured": False, "created": False, "detail": "disabled"}
    client = get_qdrant_client()
    if client is None:
        return {"ok": False, "configured": True, "created": False, "detail": "client_init_failed"}

    collection = cfg["collection"]
    vector_size = int(cfg["vector_size"])
    distance = _distance_enum(cfg["distance"])

    try:
        from qdrant_client.models import PayloadSchemaType, VectorParams

        existing = {c.name for c in (client.get_collections().collections or [])}
        created = False
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            created = True

        # Payload indexes for filtered + hybrid (dense + lexical) retrieval.
        for field_name, schema in (
            ("doc_id", PayloadSchemaType.INTEGER),
            ("version", PayloadSchemaType.INTEGER),
            ("chunk_no", PayloadSchemaType.INTEGER),
            ("category", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("title", PayloadSchemaType.KEYWORD),
            ("content_hash", PayloadSchemaType.KEYWORD),
            ("content", PayloadSchemaType.TEXT),
        ):
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                # Index may already exist; safe to ignore in idempotent startup.
                pass

        return {
            "ok": True,
            "configured": True,
            "created": created,
            "collection": collection,
            "vector_size": vector_size,
            "distance": cfg["distance"],
            "detail": "ready",
        }
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "created": False,
            "collection": collection,
            "detail": str(exc),
        }


def init_qdrant_on_startup() -> dict[str, Any]:
    """Startup hook: health probe + ensure collection (no vector writes)."""
    health = qdrant_health_check()
    if not health.get("configured"):
        print("[startup] Qdrant disabled (set QDRANT_ENABLED=1 and QDRANT_URL)", flush=True)
        return {"health": health, "collection": {"ok": True, "configured": False}}

    if not health.get("reachable"):
        print(f"[startup] Qdrant unreachable: {health.get('detail')}", flush=True)
        return {"health": health, "collection": {"ok": False, "configured": True}}

    collection = ensure_qdrant_collection()
    if collection.get("ok"):
        state = "created" if collection.get("created") else "exists"
        print(
            f"[startup] Qdrant ready collection={collection.get('collection')} "
            f"vector_size={collection.get('vector_size')} state={state}",
            flush=True,
        )
    else:
        print(f"[startup] Qdrant collection init failed: {collection.get('detail')}", flush=True)
    return {"health": health, "collection": collection}


def retrieve_knowledge_point_hashes(collection: str, point_ids: list[str]) -> dict[str, str]:
    """Return {point_id: content_hash} for existing points (skip re-index)."""
    if not point_ids:
        return {}
    client = get_qdrant_client()
    if client is None:
        return {}
    out: dict[str, str] = {}
    batch = 128
    for i in range(0, len(point_ids), batch):
        ids = point_ids[i : i + batch]
        try:
            records = client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for rec in records or []:
                pid = str(rec.id)
                payload = rec.payload or {}
                out[pid] = str(payload.get("content_hash") or "").strip()
        except Exception:
            continue
    return out


def upsert_knowledge_vectors(collection: str, points: list[dict]) -> int:
    """
    Upsert knowledge vectors into Qdrant.
    Each point: {point_id, vector, payload}
    """
    if not points:
        return 0
    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")
    from qdrant_client.models import PointStruct

    qpoints = [
        PointStruct(
            id=p["point_id"],
            vector=p["vector"],
            payload=p.get("payload") or {},
        )
        for p in points
    ]
    client.upsert(collection_name=collection, points=qpoints, wait=True)
    return len(qpoints)


def search_knowledge_vectors(
    collection: str,
    query_vector: list[float],
    *,
    top_k: int = 5,
    min_score: float | None = None,
    category_filter: list[str] | None = None,
    language_filter: str | None = None,
    doc_id_filter: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search in the knowledge collection.
    Returns list of {score, payload, point_id}.
    """
    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")

    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must: list[Any] = []
    if category_filter:
        scope_keys = [c.strip().lower() for c in category_filter if (c or "").strip()]
        if scope_keys:
            # kb_keys are document titles (file stems); category is a separate taxonomy field.
            # Match either payload.title OR payload.category so brain hints scope correctly.
            must.append(
                Filter(
                    should=[
                        FieldCondition(key="title", match=MatchAny(any=scope_keys)),
                        FieldCondition(key="category", match=MatchAny(any=scope_keys)),
                    ]
                )
            )
    if language_filter:
        must.append(
            FieldCondition(
                key="language",
                match=MatchAny(any=[language_filter.lower(), "auto"]),
            )
        )
    if doc_id_filter:
        must.append(FieldCondition(key="doc_id", match=MatchAny(any=[int(x) for x in doc_id_filter])))

    query_filter = Filter(must=must) if must else None
    limit = max(1, min(20, int(top_k)))
    score_threshold = float(min_score) if min_score is not None else None

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )

    out: list[dict[str, Any]] = []
    for rec in results or []:
        payload = dict(rec.payload or {})
        out.append(
            {
                "point_id": str(rec.id),
                "score": float(rec.score or 0.0),
                "payload": payload,
            }
        )
    return out


def lexical_knowledge_search(
    collection: str,
    query: str,
    *,
    top_k: int = 8,
    doc_id_filter: list[int] | None = None,
    language_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full-text hybrid leg via Qdrant TEXT index on payload.content.

    Uses the analyzer on the raw query first; if that is too strict (AND over
    multilingual filler), falls back to OR over whitespace tokens filtered only
    by length/shape (no curated keyword/stopword lists). Merged with dense via RRF.
    """
    import re

    q = (query or "").strip()
    if len(q) < 2:
        return []

    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")

    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText

    def _base_must() -> list[Any]:
        must: list[Any] = []
        if language_filter:
            must.append(
                FieldCondition(
                    key="language",
                    match=MatchAny(any=[language_filter.lower(), "auto"]),
                )
            )
        if doc_id_filter:
            must.append(
                FieldCondition(
                    key="doc_id", match=MatchAny(any=[int(x) for x in doc_id_filter])
                )
            )
        return must

    def _scroll(text_condition: FieldCondition | Filter) -> list[Any]:
        must = _base_must()
        must.append(text_condition)
        try:
            records, _next = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=must),
                limit=max(1, min(20, int(top_k))),
                with_payload=True,
                with_vectors=False,
            )
            return list(records or [])
        except Exception:
            return []

    # 1) Whole-query MatchText (best when query terms all appear in a chunk).
    records = _scroll(FieldCondition(key="content", match=MatchText(text=q[:500])))

    # 2) If empty: OR over content-shaped tokens (length/shape only — not a stop list).
    if not records:
        raw_toks = re.findall(r"[A-Za-z0-9]+", q)
        shaped: list[str] = []
        seen: set[str] = set()
        for tok in raw_toks:
            key = tok.lower()
            if key in seen:
                continue
            # IR heuristic: keep content-like tokens; drop 1–2 char noise.
            if len(tok) >= 4 or (tok.isupper() and len(tok) >= 2) or any(ch.isdigit() for ch in tok):
                seen.add(key)
                shaped.append(tok)
        if shaped:
            records = _scroll(
                Filter(
                    should=[
                        FieldCondition(key="content", match=MatchText(text=t))
                        for t in shaped[:8]
                    ]
                )
            )

    limit = max(1, min(20, int(top_k)))
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records[:limit]):
        payload = dict(rec.payload or {})
        score = max(0.22, 0.48 - (0.02 * i))
        out.append(
            {
                "point_id": str(rec.id),
                "score": float(score),
                "payload": payload,
                "match_type": "fulltext",
            }
        )
    return out


def delete_knowledge_vectors_by_point_ids(collection: str, point_ids: list[str]) -> int:
    """Delete Qdrant points by deterministic IDs."""
    if not point_ids:
        return 0
    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")
    batch = 128
    deleted = 0
    for i in range(0, len(point_ids), batch):
        ids = point_ids[i : i + batch]
        client.delete(collection_name=collection, points_selector=ids, wait=True)
        deleted += len(ids)
    return deleted


def delete_knowledge_vectors_by_doc_id(collection: str, doc_id: int) -> int:
    """Delete all Qdrant vectors for a knowledge document (all versions)."""
    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=int(doc_id)))])
        ),
        wait=True,
    )
    return 1


def delete_knowledge_vectors_by_doc_id_except_version(
    collection: str,
    doc_id: int,
    keep_version: int,
) -> int:
    """
    Delete Qdrant vectors for a document except the keep_version points.
    Used after a successful reindex upsert so old vectors are removed atomically.
    """
    client = get_qdrant_client()
    if client is None:
        raise RuntimeError("Qdrant client unavailable")
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        FilterSelector,
        MatchValue,
    )

    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=int(doc_id)))],
                must_not=[
                    FieldCondition(
                        key="version",
                        match=MatchValue(value=int(keep_version)),
                    )
                ],
            )
        ),
        wait=True,
    )
    return 1
