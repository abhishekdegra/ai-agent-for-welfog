"""
Step 6: Qdrant semantic knowledge retrieval (context only — no answer generation).

Reuses OpenAI text-embedding-3-small (same as Step 5 indexer) for query vectors.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from services.knowledge_embedding_indexer import embed_texts_openai
from services.qdrant_service import (
    is_qdrant_configured,
    qdrant_config,
    qdrant_health_check,
    search_knowledge_vectors,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


DEFAULT_TOP_K = _env_int("KNOWLEDGE_RETRIEVAL_TOP_K", 8)
DEFAULT_MIN_SCORE = _env_float("KNOWLEDGE_RETRIEVAL_MIN_SCORE", 0.22)
DEFAULT_ANSWER_TOP_K = _env_int("KNOWLEDGE_ANSWER_TOP_K", 12)


def kb_retrieval_backend() -> str:
    return (os.getenv("KB_RETRIEVAL_BACKEND") or "memory").strip().lower()


def is_qdrant_kb_retrieval_enabled() -> bool:
    if kb_retrieval_backend() != "qdrant":
        return False
    if not is_qdrant_configured():
        return False
    health = qdrant_health_check()
    return bool(health.get("reachable"))


def is_kb_intent(ai_route: dict | None) -> bool:
    """True when brain routing locked Knowledge Base answer path."""
    if not isinstance(ai_route, dict):
        return False
    try:
        from services.ai_route_semantics import brain_route_prefers_kb_answer

        return brain_route_prefers_kb_answer(ai_route)
    except ImportError:
        ch = (ai_route.get("data_channel") or "").strip().lower()
        return ch == "kb" or bool(ai_route.get("kb_keys"))


def should_use_qdrant_retrieval(*, ai_route: dict | None = None, kb_intent: bool = False) -> bool:
    """
    Qdrant retrieval runs only for Knowledge Base intent when backend=qdrant.
    Callers without ai_route may pass kb_intent=True on confirmed KB paths.
    """
    if not is_qdrant_kb_retrieval_enabled():
        return False
    if kb_intent or is_kb_intent(ai_route):
        return True
    return False


def _normalize_query(query: str) -> str:
    import re

    t = re.sub(r"<br\s*/?>", "\n", query or "", flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:900]


def _normalize_categories(keys: list[str] | None) -> list[str] | None:
    if not keys:
        return None
    out: list[str] = []
    for k in keys:
        c = (k or "").strip().lower()
        if c and c not in out:
            out.append(c)
    return out or None


def _normalize_language(language: str | None) -> str | None:
    lang = (language or "").strip().lower()
    if not lang or lang in ("auto", "any", "*"):
        return None
    return lang


def _fetch_active_doc_ids() -> list[int] | None:
    """MySQL status filter — only chunks from active knowledge documents."""
    try:
        from services.mysql_service import get_mysql_connection

        conn = get_mysql_connection()
        if not conn:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM knowledge_documents WHERE status = %s",
                    ("active",),
                )
                rows = cur.fetchall() or []
            return [int(r["id"]) for r in rows if r.get("id") is not None]
        finally:
            conn.close()
    except Exception:
        return None


def _cached_embed_query(query: str) -> list[float]:
    try:
        from services.chat_flow_telemetry import _TLS

        if getattr(_TLS, "kb_openai_embed_key", None) == query:
            cached = getattr(_TLS, "kb_openai_embed_vec", None)
            if cached is not None:
                return list(cached)
    except ImportError:
        pass
    vectors = embed_texts_openai([query])
    if not vectors:
        raise RuntimeError("OpenAI query embedding failed")
    vec = list(vectors[0])
    try:
        from services.chat_flow_telemetry import _TLS

        _TLS.kb_openai_embed_key = query
        _TLS.kb_openai_embed_vec = vec
    except ImportError:
        pass
    return vec


def _payload_to_hit(payload: dict[str, Any], score: float) -> dict[str, Any]:
    category = (payload.get("category") or payload.get("title") or "general").strip().lower()
    content = payload.get("content") or ""
    return {
        "source": category or "general",
        "chunk": content,
        "score": float(score),
        "match_type": "semantic",
        "doc_id": payload.get("doc_id"),
        "version": payload.get("version"),
        "chunk_no": payload.get("chunk_no"),
        "title": payload.get("title"),
        "category": payload.get("category"),
        "language": payload.get("language"),
        "content_hash": payload.get("content_hash"),
    }


@dataclass
class KnowledgeRetrievalResult:
    query: str = ""
    hits: list[dict[str, Any]] = field(default_factory=list)
    top_score: float = 0.0
    query_time_ms: float = 0.0
    chunks_returned: int = 0
    backend: str = "qdrant"
    collection: str = ""
    filters: dict[str, Any] = field(default_factory=dict)


def retrieve_knowledge_context(
    query: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    categories: list[str] | None = None,
    language: str | None = None,
    status: str = "active",
    ai_route: dict | None = None,
    kb_intent: bool = False,
    log: bool = True,
) -> KnowledgeRetrievalResult:
    """
    Semantic retrieval from Qdrant — returns ranked chunk hits only (no LLM answer).
  If top score is below min_score threshold, returns empty hits.
    """
    empty = KnowledgeRetrievalResult(query=query or "", backend="qdrant")
    if not should_use_qdrant_retrieval(ai_route=ai_route, kb_intent=kb_intent):
        return empty

    q = _normalize_query(query)
    if not q:
        return empty

    cfg = qdrant_config() or {}
    collection = cfg.get("collection") or ""
    if not collection:
        return empty

    limit = max(1, min(20, int(top_k if top_k is not None else DEFAULT_TOP_K)))
    floor = float(min_score if min_score is not None else DEFAULT_MIN_SCORE)

    route_keys = list((ai_route or {}).get("kb_keys") or [])
    cat_filter = _normalize_categories(categories or route_keys or None)
    lang_filter = _normalize_language(
        language or (ai_route or {}).get("reply_lang") or (ai_route or {}).get("language")
    )

    doc_ids: list[int] | None = None
    if (status or "active").strip().lower() == "active":
        doc_ids = _fetch_active_doc_ids()

    filters_applied = {
        "categories": cat_filter,
        "language": lang_filter,
        "status": status,
        "active_doc_ids": len(doc_ids) if doc_ids else None,
    }

    t0 = time.perf_counter()
    used_scoped_backoff = False
    try:
        query_vector = _cached_embed_query(q)
        raw_hits = search_knowledge_vectors(
            collection,
            query_vector,
            top_k=limit,
            min_score=floor,
            category_filter=cat_filter,
            language_filter=lang_filter,
            doc_id_filter=doc_ids,
        )
        # Scoped backoff: if brain kb_keys over-constrain (zero hits), retry unscoped within active docs.
        if not raw_hits and cat_filter:
            used_scoped_backoff = True
            raw_hits = search_knowledge_vectors(
                collection,
                query_vector,
                top_k=limit,
                min_score=floor,
                category_filter=None,
                language_filter=lang_filter,
                doc_id_filter=doc_ids,
            )
            if raw_hits and log:
                print(
                    f"[kb-retrieve] scoped backoff: 0 hits with scope={cat_filter} "
                    f"-> {len(raw_hits)} unscoped hits",
                    flush=True,
                )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        try:
            from services.knowledge_rag_debug import (
                is_rag_debug_enabled,
                record_embedding_debug,
                set_answer_source_debug,
                flush_rag_debug_report,
            )
            from services.knowledge_embedding_indexer import _embedding_model

            if is_rag_debug_enabled():
                record_embedding_debug(model=_embedding_model(), ok=False, error=str(exc))
                set_answer_source_debug("THRESHOLD_FAILED", fallback_reason=f"Retrieval error: {exc}")
                flush_rag_debug_report()
        except ImportError:
            pass
        if log:
            print(
                f"[kb-retrieve] query_time_ms={elapsed_ms:.1f} top_score=0.00 "
                f"chunks_returned=0 error={exc}",
                flush=True,
            )
        return KnowledgeRetrievalResult(
            query=q,
            query_time_ms=elapsed_ms,
            backend="qdrant",
            collection=collection,
            filters=filters_applied,
        )

    hits = [_payload_to_hit(h["payload"], h["score"]) for h in raw_hits]
    top_score = float(hits[0]["score"]) if hits else 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    try:
        from services.knowledge_rag_debug import (
            is_rag_debug_enabled,
            record_embedding_debug,
            record_qdrant_request_debug,
            run_shadow_retrieval_debug,
        )

        if is_rag_debug_enabled():
            from services.knowledge_embedding_indexer import _embedding_model

            record_embedding_debug(
                model=_embedding_model(),
                ok=True,
                dim=len(query_vector) if query_vector else None,
            )
            record_qdrant_request_debug(
                {
                    "collection": collection,
                    "top_k": limit,
                    "score_threshold": floor,
                    "metadata_filters": filters_applied,
                    "category_filter": cat_filter,
                    "language_filter": lang_filter,
                    "active_doc_ids_count": len(doc_ids) if doc_ids else None,
                    "scoped_backoff_used": used_scoped_backoff,
                }
            )
            run_shadow_retrieval_debug(
                collection=collection,
                query_vector=query_vector,
                score_threshold=floor,
                top_k=limit,
                category_filter=cat_filter,
                language_filter=lang_filter,
                doc_id_filter=doc_ids,
                production_hits=hits,
                scoped_backoff=used_scoped_backoff,
            )
    except ImportError:
        pass

    if log:
        print(
            f"[kb-retrieve] query_time_ms={elapsed_ms:.1f} top_score={top_score:.4f} "
            f"chunks_returned={len(hits)} collection={collection} "
            f"filters={filters_applied}",
            flush=True,
        )

    return KnowledgeRetrievalResult(
        query=q,
        hits=hits,
        top_score=top_score,
        query_time_ms=elapsed_ms,
        chunks_returned=len(hits),
        backend="qdrant",
        collection=collection,
        filters=filters_applied,
    )


def retrieve_knowledge_hits(
    query: str,
    *,
    keys: list[str] | None = None,
    top_n: int = 4,
    min_score: float | None = None,
    ai_route: dict | None = None,
    kb_intent: bool = False,
    log_retrieval: bool = True,
) -> list[dict[str, Any]]:
    """Adapter for kb_service.semantic_kb_search hit shape."""
    result = retrieve_knowledge_context(
        query,
        top_k=top_n,
        min_score=min_score,
        categories=keys,
        ai_route=ai_route,
        kb_intent=kb_intent,
        log=log_retrieval,
    )
    if not result.hits:
        return []

    if log_retrieval:
        try:
            from services.kb_service import log_kb_retrieval

            top = result.hits[0]
            log_kb_retrieval(
                query_intent=str(top.get("source") or "general"),
                query_meaning=((ai_route or {}).get("user_meaning") or "").strip(),
                retrieval_query=result.query,
                matched_file=str(top.get("source") or ""),
                selected_category=str(top.get("category") or top.get("source") or "general"),
                similarity_score=float(top.get("score") or 0),
                selected_chunks=result.hits,
            )
        except ImportError:
            pass

    return result.hits


def format_knowledge_context_string(hits: list[dict[str, Any]]) -> str:
    """HTML context block for downstream grounding (no answer generation)."""
    if not hits:
        return ""
    out: list[str] = []
    for h in hits:
        src = h.get("source") or "?"
        sc = float(h.get("score") or 0)
        chunk = h.get("chunk") or ""
        out.append(f"[source={src} score={sc:.2f}] {chunk}")
    return "<br><br>".join(out)
