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

        # Payload indexes for future filtered retrieval (metadata from chunk pipeline).
        for field_name, schema in (
            ("doc_id", PayloadSchemaType.INTEGER),
            ("version", PayloadSchemaType.INTEGER),
            ("chunk_no", PayloadSchemaType.INTEGER),
            ("category", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("title", PayloadSchemaType.KEYWORD),
            ("content_hash", PayloadSchemaType.KEYWORD),
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
