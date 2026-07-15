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
    lexical_knowledge_search,
    qdrant_config,
    qdrant_health_check,
    search_knowledge_vectors,
)
from utils.reasoning_log import log_reasoning


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
# Soft floor for recall — gray-band (0.14–0.22) still reaches answer / extractive path.
DEFAULT_MIN_SCORE = _env_float("KNOWLEDGE_RETRIEVAL_MIN_SCORE", 0.14)
DEFAULT_ANSWER_TOP_K = _env_int("KNOWLEDGE_ANSWER_TOP_K", 8)
# Soft kb_keys boost — must stay below typical semantic gaps so #1 unscoped wins.
KB_KEY_HINT_BOOST = _env_float("KNOWLEDGE_HINT_BOOST", 0.025)
# Force-include unscoped hits this far above the best hint-matched raw score.
KB_KEY_HINT_MARGIN = _env_float("KNOWLEDGE_HINT_MARGIN", 0.03)

# Process-local query embedding cache (OpenAI latency dominates retrieval).
_QUERY_EMBED_CACHE: dict[str, list[float]] = {}
_QUERY_EMBED_CACHE_MAX = 64


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


def _hit_matches_kb_hints(payload_or_hit: dict[str, Any], hints: list[str] | None) -> bool:
    """True when chunk title matches a Brain kb_key (canonical form). Category is not identity."""
    if not hints:
        return False
    try:
        from services.knowledge_keys import canonical_knowledge_key
    except ImportError:
        canonical_knowledge_key = lambda t: (t or "").strip().lower()  # noqa: E731

    title = canonical_knowledge_key(payload_or_hit.get("title") or "")
    for h in hints:
        hk = canonical_knowledge_key(str(h or ""))
        if hk and hk == title:
            return True
    return False


def _is_internal_knowledge_title(title: str) -> bool:
    """Agent/internal docs must not enter customer grounded retrieval."""
    try:
        from services.knowledge_keys import is_internal_agent_knowledge_key

        return is_internal_agent_knowledge_key(title)
    except ImportError:
        t = (title or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not t:
            return False
        if t.startswith("welfog_api"):
            return True
        if t in ("system_messages", "system_messages_2", "system_message"):
            return True
        return False


def _fetch_active_doc_ids() -> list[int] | None:
    """
    MySQL filter for retrieval SoT: active + fully indexed customer docs only.

    Eligibility is ONLY status+index_status (+ internal agent title exclusion).
    Category labels never exclude customer Admin knowledge.
    """
    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        if get_kb_turn_cache("kb_active_doc_ids_ready", False):
            cached = get_kb_turn_cache("kb_active_doc_ids", None)
            return list(cached) if cached is not None else None
    except ImportError:
        pass

    out: list[int] | None = None
    try:
        from services.mysql_service import get_mysql_connection

        conn = get_mysql_connection()
        if not conn:
            out = None
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, category FROM knowledge_documents
                        WHERE status = %s AND index_status = %s
                        """,
                        ("active", "indexed"),
                    )
                    rows = cur.fetchall() or []
                ids: list[int] = []
                for r in rows:
                    if r.get("id") is None:
                        continue
                    title = (r.get("title") or "").strip()
                    # Never exclude by category — only true agent/internal titles.
                    if _is_internal_knowledge_title(title):
                        continue
                    ids.append(int(r["id"]))
                out = ids
            finally:
                conn.close()
    except Exception:
        out = None

    try:
        from services.chat_flow_telemetry import set_kb_turn_cache

        set_kb_turn_cache("kb_active_doc_ids", list(out) if out is not None else None)
        set_kb_turn_cache("kb_active_doc_ids_ready", True)
    except ImportError:
        pass
    return out


def _kb_retrieval_base_cache_key(
    query: str,
    *,
    floor: float,
    lang_filter: str | None,
    status: str,
    doc_ids: list[int] | None,
    candidate_k: int,
) -> str:
    docs = ",".join(str(i) for i in (doc_ids or [])[:200])
    return (
        f"base|{query}|{floor:.4f}|{lang_filter or '-'}|{status}|"
        f"k={candidate_k}|docs={docs}"
    )


def _kb_retrieval_result_cache_key(
    query: str,
    *,
    floor: float,
    lang_filter: str | None,
    status: str,
    hint_keys: list[str] | None,
    limit: int,
    doc_ids: list[int] | None,
) -> str:
    hints = ",".join(hint_keys or [])
    docs = ",".join(str(i) for i in (doc_ids or [])[:200])
    return (
        f"res|{query}|{floor:.4f}|{lang_filter or '-'}|{status}|"
        f"limit={limit}|hints={hints}|docs={docs}"
    )


def _store_query_embed_cache(qkey: str, vec: list[float]) -> None:
    global _QUERY_EMBED_CACHE
    if len(_QUERY_EMBED_CACHE) >= _QUERY_EMBED_CACHE_MAX:
        try:
            _QUERY_EMBED_CACHE.pop(next(iter(_QUERY_EMBED_CACHE)))
        except StopIteration:
            _QUERY_EMBED_CACHE.clear()
    _QUERY_EMBED_CACHE[qkey] = vec
    try:
        from services.chat_flow_telemetry import _TLS

        _TLS.kb_openai_embed_key = qkey
        _TLS.kb_openai_embed_vec = list(vec)
    except ImportError:
        pass


def _cached_embed_query(query: str) -> list[float]:
    """
    Query embedding with per-turn TLS reuse + small process LRU.
    OpenAI embed is the dominant KB retrieval latency (~1s+); cache avoids repeat cost.
    """
    qkey = (query or "").strip().lower()[:900]
    try:
        from services.chat_flow_telemetry import _TLS

        if getattr(_TLS, "kb_openai_embed_key", None) == qkey:
            cached = getattr(_TLS, "kb_openai_embed_vec", None)
            if cached is not None:
                return list(cached)
    except ImportError:
        pass

    hit = _QUERY_EMBED_CACHE.get(qkey)
    if hit is not None:
        _store_query_embed_cache(qkey, list(hit))
        return list(hit)

    vectors = embed_texts_openai([query])
    if not vectors:
        raise RuntimeError("OpenAI query embedding failed")
    vec = list(vectors[0])
    _store_query_embed_cache(qkey, vec)
    return vec


def _cached_embed_queries(queries: list[str]) -> list[list[float]]:
    """Batch-embed query variants (one OpenAI call for all cache misses)."""
    cleaned = [_normalize_query(q) for q in queries if _normalize_query(q)]
    if not cleaned:
        return []
    keys = [q.strip().lower()[:900] for q in cleaned]
    vectors: list[list[float] | None] = [None] * len(cleaned)
    missing_idx: list[int] = []
    missing_texts: list[str] = []

    for i, (q, qkey) in enumerate(zip(cleaned, keys)):
        hit = _QUERY_EMBED_CACHE.get(qkey)
        if hit is not None:
            vectors[i] = list(hit)
            continue
        try:
            from services.chat_flow_telemetry import _TLS

            if getattr(_TLS, "kb_openai_embed_key", None) == qkey:
                cached = getattr(_TLS, "kb_openai_embed_vec", None)
                if cached is not None:
                    vectors[i] = list(cached)
                    continue
        except ImportError:
            pass
        missing_idx.append(i)
        missing_texts.append(q)

    if missing_texts:
        fresh = embed_texts_openai(missing_texts)
        if len(fresh) != len(missing_texts):
            raise RuntimeError("OpenAI query embedding failed")
        for i, vec in zip(missing_idx, fresh):
            v = list(vec)
            vectors[i] = v
            _store_query_embed_cache(keys[i], v)

    return [list(v) for v in vectors if v is not None]


def _retrieval_query_variants(query: str) -> list[str]:
    """
    Distinct embed/fulltext variants from a joined retrieval query.

    build_kb_retrieval_query joins English gloss + raw with ' — '.
    Never re-embed the full joined string (that caused query_variants=2 for X—X).
    """
    q = _normalize_query(query)
    if not q:
        return []
    variants: list[str] = []
    seen: set[str] = set()
    for p in q.split(" — "):
        part = p.strip()
        if not part:
            continue
        low = part.lower()
        if low.startswith("topic:"):
            continue
        if low in seen:
            continue
        seen.add(low)
        variants.append(part)
    return variants[:2]


def _rrf_merge_raw_hits(
    hit_lists: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across dense (and optional lexical) hit lists."""
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, h in enumerate(hits or []):
            pid = str(h.get("point_id") or "")
            if not pid:
                continue
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
            prev = best.get(pid)
            if prev is None or float(h.get("score") or 0) > float(prev.get("score") or 0):
                best[pid] = h
    merged: list[dict[str, Any]] = []
    for pid, rrf in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        h = dict(best[pid])
        # Keep dense similarity as primary score; RRF only for ordering among lists.
        h["_rrf"] = rrf
        merged.append(h)
    return merged


def _payload_to_hit(payload: dict[str, Any], score: float, *, match_type: str = "semantic") -> dict[str, Any]:
    """
    Map Qdrant payload → retrieval hit.

    Document identity (`source`) MUST be the canonical Admin title key
    (e.g. short_videopolicy), NEVER the soft taxonomy category (e.g. policy).
    Category is retained only as a label for logging / soft hints.
    """
    try:
        from services.knowledge_keys import canonical_knowledge_key
    except ImportError:
        canonical_knowledge_key = lambda t: (t or "").strip().lower().replace("-", "_").replace(" ", "_")  # noqa: E731

    raw_title = (payload.get("title") or "").strip()
    title_key = canonical_knowledge_key(raw_title) or raw_title.lower()
    category = (payload.get("category") or "").strip().lower() or "general"
    content = payload.get("content") or ""
    # Identity = title. Fall back to doc_id lookup only if title missing (legacy points).
    source = title_key
    if not source and payload.get("doc_id") is not None:
        try:
            from services.mysql_service import get_mysql_connection

            conn = get_mysql_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT title FROM knowledge_documents WHERE id = %s LIMIT 1",
                            (int(payload["doc_id"]),),
                        )
                        row = cur.fetchone() or {}
                    source = canonical_knowledge_key(row.get("title") or "") or ""
                finally:
                    conn.close()
        except Exception:
            source = ""
    if not source:
        source = "general"
    return {
        "source": source,
        "chunk": content,
        "score": float(score),
        "match_type": match_type or "semantic",
        "doc_id": payload.get("doc_id"),
        "version": payload.get("version"),
        "chunk_no": payload.get("chunk_no"),
        "title": title_key or raw_title,
        "category": category,
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


def _soft_rerank_with_kb_hints(
    raw_hits: list[dict[str, Any]],
    *,
    hints: list[str] | None,
    limit: int,
    boost: float | None = None,
    margin: float | None = None,
) -> list[dict[str, Any]]:
    """
    Soft Brain kb_keys preference.

    Raw semantic score stays primary. When Brain locks kb_keys, a hint-matched
    chunk within ``margin`` of the top raw score is preferred — this stops
    lexical near-misses (e.g. seller FAQ) from beating the keyed fact chunk.
    """
    if not raw_hits:
        return []
    hint_list = list(hints or [])
    b = float(boost if boost is not None else KB_KEY_HINT_BOOST)
    b = max(0.0, min(0.08, b))
    m = float(margin if margin is not None else KB_KEY_HINT_MARGIN)
    # Brain-locked keys need enough room to beat lexical false friends.
    m = max(0.03, min(0.18, m if hint_list else m))
    if hint_list:
        m = max(m, 0.12)
    limit = max(1, min(20, int(limit)))

    enriched: list[dict[str, Any]] = []
    for h in raw_hits:
        payload = h.get("payload") or {}
        score = float(h.get("score") or 0.0)
        matched = _hit_matches_kb_hints(payload, hint_list)
        soft = score + (b if matched else 0.0)
        enriched.append({**h, "_hint_matched": matched, "_soft_score": soft, "_raw": score})

    best_raw = max((float(x.get("_raw") or 0.0) for x in enriched), default=0.0)
    best_hint_raw = max(
        (float(x.get("_raw") or 0.0) for x in enriched if x.get("_hint_matched")),
        default=-1.0,
    )
    prefer_hints = bool(hint_list) and best_hint_raw >= 0.0 and best_hint_raw >= (
        best_raw - m
    )

    # Primary: raw semantic. When a keyed hit is close enough, promote it.
    by_rank = sorted(
        enriched,
        key=lambda x: (
            1 if (prefer_hints and x.get("_hint_matched")) else 0,
            float(x.get("_raw") or 0.0),
            float(x.get("_soft_score") or 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _pid(item: dict[str, Any]) -> str:
        return str(item.get("point_id") or id(item))

    for h in by_rank:
        if len(selected) >= limit:
            break
        pid = _pid(h)
        if pid in seen:
            continue
        seen.add(pid)
        selected.append(h)

    out: list[dict[str, Any]] = []
    for h in selected:
        out.append({k: v for k, v in h.items() if not str(k).startswith("_")})
    return out


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

    Brain kb_keys / categories are soft hints (boost + rerank), never hard Qdrant
    title/category filters. Hard filters remain: active+indexed customer doc_ids.
    Reply language is NOT used as a Qdrant filter (KB payloads are language=auto);
    language belongs to answer generation only.

    Within one /chat turn: identical Qdrant+embed work is reused; soft kb_keys
    rerank may re-run on cached raw hits (deterministic, same answers).
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
    hint_keys = _normalize_categories(categories or route_keys or None)
    lang_filter = _normalize_language(language)

    doc_ids: list[int] | None = None
    if (status or "active").strip().lower() == "active":
        doc_ids = _fetch_active_doc_ids()

    variants = _retrieval_query_variants(q)

    filters_applied = {
        "kb_key_hints": hint_keys,
        "kb_keys_mode": "soft_rerank",
        "language": lang_filter,
        "status": status,
        "active_doc_ids": len(doc_ids) if doc_ids else None,
        "category_hard_filter": None,
        "query_variants": len(variants),
        "hybrid": "dense_rrf+fulltext",
    }

    t0 = time.perf_counter()
    candidate_k = max(limit * 3, min(24, limit + 10))
    query_vector: list[float] = []
    reused_base = False

    result_key = _kb_retrieval_result_cache_key(
        q,
        floor=floor,
        lang_filter=lang_filter,
        status=status,
        hint_keys=hint_keys,
        limit=limit,
        doc_ids=doc_ids,
    )
    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        result_cache = get_kb_turn_cache("kb_retrieval_result_cache") or {}
        cached_result = result_cache.get(result_key)
        if isinstance(cached_result, KnowledgeRetrievalResult):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if log:
                print(
                    f"[kb-retrieve] query_time_ms={elapsed_ms:.1f} top_score={cached_result.top_score:.4f} "
                    f"chunks_returned={cached_result.chunks_returned} collection={collection} "
                    f"filters={filters_applied} reuse=result_cache",
                    flush=True,
                )
            return KnowledgeRetrievalResult(
                query=cached_result.query,
                hits=list(cached_result.hits or []),
                top_score=float(cached_result.top_score or 0.0),
                query_time_ms=elapsed_ms,
                chunks_returned=int(cached_result.chunks_returned or 0),
                backend=cached_result.backend,
                collection=cached_result.collection,
                filters=dict(filters_applied),
            )
        for prior in result_cache.values():
            if not isinstance(prior, KnowledgeRetrievalResult):
                continue
            pf = (prior.filters or {}) if isinstance(prior.filters, dict) else {}
            if (
                prior.query == q
                and float(pf.get("_floor") or -1) == floor
                and pf.get("language") == lang_filter
                and pf.get("status") == status
                and pf.get("kb_key_hints") == hint_keys
                and int(prior.chunks_returned or 0) >= limit
            ):
                sliced = list(prior.hits or [])[:limit]
                top_score = float(sliced[0]["score"]) if sliced else 0.0
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if log:
                    print(
                        f"[kb-retrieve] query_time_ms={elapsed_ms:.1f} top_score={top_score:.4f} "
                        f"chunks_returned={len(sliced)} collection={collection} "
                        f"filters={filters_applied} reuse=result_slice",
                        flush=True,
                    )
                return KnowledgeRetrievalResult(
                    query=q,
                    hits=sliced,
                    top_score=top_score,
                    query_time_ms=elapsed_ms,
                    chunks_returned=len(sliced),
                    backend="qdrant",
                    collection=collection,
                    filters=dict(filters_applied),
                )
    except ImportError:
        pass

    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        base_cache = get_kb_turn_cache("kb_retrieval_base_cache") or {}
        base_hit = None
        doc_sig = ",".join(str(i) for i in (doc_ids or [])[:200])
        for bval in list(base_cache.values()):
            if not isinstance(bval, dict):
                continue
            if (
                bval.get("query") == q
                and float(bval.get("floor") or -1) == floor
                and bval.get("lang_filter") == lang_filter
                and bval.get("status") == status
                and bval.get("doc_sig") == doc_sig
                and int(bval.get("candidate_k") or 0) >= candidate_k
            ):
                base_hit = bval
                break

        if base_hit is not None:
            reused_base = True
            raw_hits = list(base_hit.get("raw_hits") or [])
            query_vector = list(base_hit.get("query_vector") or [])
            filters_applied["query_variants"] = int(base_hit.get("query_variants") or 0)
            filters_applied["reuse"] = "base_cache"
        else:
            variant_list = variants or [q]
            try:
                from services.translation_service import text_usable_as_english_retrieval

                dense_queries = [
                    v for v in variant_list if text_usable_as_english_retrieval(v)
                ]
            except ImportError:
                dense_queries = []
            if not dense_queries:
                dense_queries = variant_list[:1]
            # One English dense query is enough — raw Hinglish is for fulltext only.
            # Extra dense variants double OpenAI embed latency with little recall gain.
            dense_queries = dense_queries[:1]

            vectors = _cached_embed_queries(dense_queries)
            if not vectors:
                raise RuntimeError("OpenAI query embedding failed")
            query_vector = list(vectors[0])

            dense_lists: list[list[dict[str, Any]]] = []
            fulltext_lists: list[list[dict[str, Any]]] = []
            ft_q = (variants or [q])[0]

            def _run_dense() -> list[list[dict[str, Any]]]:
                out_lists: list[list[dict[str, Any]]] = []
                for vec in vectors:
                    out_lists.append(
                        search_knowledge_vectors(
                            collection,
                            vec,
                            top_k=candidate_k,
                            min_score=floor,
                            category_filter=None,
                            language_filter=lang_filter,
                            doc_id_filter=doc_ids,
                        )
                    )
                return out_lists

            def _run_fulltext() -> list[list[dict[str, Any]]]:
                try:
                    ft = lexical_knowledge_search(
                        collection,
                        ft_q,
                        top_k=max(6, limit),
                        doc_id_filter=doc_ids,
                        language_filter=lang_filter,
                    )
                    return [ft] if ft else []
                except Exception:
                    return []

            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

                # Prefer sequential under /chat worker — nested pools + async deadline
                # caused 35s deadlocks (parent join waiting, child pool starved).
                use_pool = (os.getenv("KB_RETRIEVAL_PARALLEL") or "0").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if use_pool:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        dense_fut = pool.submit(_run_dense)
                        ft_fut = pool.submit(_run_fulltext)
                        try:
                            dense_lists = dense_fut.result(timeout=8.0)
                        except FuturesTimeout:
                            dense_lists = []
                            log_reasoning("Qdrant dense search timed out (8s) — skip.")
                        try:
                            fulltext_lists = ft_fut.result(timeout=8.0)
                        except FuturesTimeout:
                            fulltext_lists = []
                            log_reasoning("Qdrant fulltext search timed out (8s) — skip.")
                else:
                    dense_lists = _run_dense()
                    fulltext_lists = _run_fulltext()
            except Exception:
                dense_lists = _run_dense()
                fulltext_lists = _run_fulltext()

            merge_lists = list(dense_lists) + fulltext_lists
            raw_merged = _rrf_merge_raw_hits(merge_lists)
            raw_merged.sort(
                key=lambda h: (
                    float(h.get("_rrf") or 0.0),
                    float(h.get("score") or 0.0),
                ),
                reverse=True,
            )
            raw_hits = raw_merged
            base_key = _kb_retrieval_base_cache_key(
                q,
                floor=floor,
                lang_filter=lang_filter,
                status=status,
                doc_ids=doc_ids,
                candidate_k=candidate_k,
            )
            base_cache[base_key] = {
                "query": q,
                "floor": floor,
                "lang_filter": lang_filter,
                "status": status,
                "doc_sig": doc_sig,
                "candidate_k": candidate_k,
                "query_variants": len(variants),
                "raw_hits": list(raw_hits),
                "query_vector": list(query_vector),
            }
            set_kb_turn_cache("kb_retrieval_base_cache", base_cache)

        raw_hits = _soft_rerank_with_kb_hints(
            raw_hits,
            hints=hint_keys,
            limit=limit,
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
                set_answer_source_debug(
                    "THRESHOLD_FAILED", fallback_reason=f"Retrieval error: {exc}"
                )
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

    hits = [
        _payload_to_hit(
            h["payload"],
            h["score"],
            match_type=str(h.get("match_type") or "semantic"),
        )
        for h in raw_hits
    ]
    top_score = float(hits[0]["score"]) if hits else 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    try:
        from services.knowledge_rag_debug import (
            is_rag_debug_enabled,
            record_embedding_debug,
            record_qdrant_request_debug,
            run_shadow_retrieval_debug,
        )

        if is_rag_debug_enabled() and not reused_base:
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
                    "candidate_k": candidate_k,
                    "score_threshold": floor,
                    "metadata_filters": filters_applied,
                    "kb_key_hints": hint_keys,
                    "kb_keys_mode": "soft_rerank",
                    "language_filter": lang_filter,
                    "active_doc_ids_count": len(doc_ids) if doc_ids else None,
                    "scoped_backoff_used": False,
                }
            )
            run_shadow_retrieval_debug(
                collection=collection,
                query_vector=query_vector,
                score_threshold=floor,
                top_k=limit,
                category_filter=None,
                language_filter=lang_filter,
                doc_id_filter=doc_ids,
                production_hits=hits,
                scoped_backoff=False,
            )
    except ImportError:
        pass

    reuse_tag = " reuse=base_cache" if reused_base else ""
    if log:
        print(
            f"[kb-retrieve] query_time_ms={elapsed_ms:.1f} top_score={top_score:.4f} "
            f"chunks_returned={len(hits)} collection={collection} "
            f"filters={filters_applied}{reuse_tag}",
            flush=True,
        )

    filters_out = dict(filters_applied)
    filters_out["_floor"] = floor

    out = KnowledgeRetrievalResult(
        query=q,
        hits=hits,
        top_score=top_score,
        query_time_ms=elapsed_ms,
        chunks_returned=len(hits),
        backend="qdrant",
        collection=collection,
        filters=filters_out,
    )
    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        result_cache = get_kb_turn_cache("kb_retrieval_result_cache") or {}
        result_cache[result_key] = out
        set_kb_turn_cache("kb_retrieval_result_cache", result_cache)
    except ImportError:
        pass
    return out



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
