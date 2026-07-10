"""
Temporary Knowledge Retrieval Debug Mode (observability only).

Enable with RAG_DEBUG=true — does not alter retrieval, routing, or prompts.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

_TLS = threading.local()

DEBUG_TOP_N = 10


def is_rag_debug_enabled() -> bool:
    return (os.getenv("RAG_DEBUG") or "").strip().lower() in ("1", "true", "yes", "on")


def _chunk_key(hit: dict[str, Any]) -> str:
    return "|".join(
        [
            str(hit.get("doc_id") or ""),
            str(hit.get("version") or ""),
            str(hit.get("chunk_no") or ""),
            str(hit.get("content_hash") or ""),
        ]
    )


def _hit_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    p = raw.get("payload") or {}
    return {
        "score": float(raw.get("score") or 0),
        "doc_id": p.get("doc_id"),
        "version": p.get("version"),
        "chunk_no": p.get("chunk_no"),
        "title": p.get("title"),
        "category": p.get("category"),
        "language": p.get("language"),
        "content_hash": p.get("content_hash"),
        "chunk": p.get("content") or "",
        "point_id": raw.get("point_id"),
    }


def _preview(text: str, n: int = 250) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


@dataclass
class RagDebugSession:
    original_query: str = ""
    normalized_query: str = ""
    preprocessing_query: str = ""
    ai_route: dict[str, Any] = field(default_factory=dict)
    reply_lang: str = ""
    embedding_model: str = ""
    embedding_ok: bool | None = None
    embedding_dim: int | None = None
    embedding_error: str = ""
    qdrant_request: dict[str, Any] = field(default_factory=dict)
    top_chunks: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[dict[str, str]] = field(default_factory=list)
    final_chunks: list[dict[str, Any]] = field(default_factory=list)
    grounding_context: str = ""
    grounding_summary: str = ""
    user_question: str = ""
    answer_source: str = "UNKNOWN"
    fallback_reason: str = ""
    used_qdrant_path: bool = False
    used_scoped_backoff: bool = False

    def to_report(self) -> str:
        route = self.ai_route or {}
        lines: list[str] = [
            "=" * 72,
            "RAG DEBUG REPORT",
            "=" * 72,
            "",
            "1. Original user query",
            f"   {self.original_query or '-'}",
            "",
            "2. Query after preprocessing/normalization",
            f"   preprocessing: {self.preprocessing_query or '-'}",
            f"   normalized:    {self.normalized_query or '-'}",
            "",
            "3. Brain routing result",
            f"   intent:       {route.get('intent', '-')}",
            f"   data_channel: {route.get('data_channel', '-')}",
            f"   kb_keys:      {route.get('kb_keys', [])}",
            f"   language:     {self.reply_lang or route.get('reply_lang') or route.get('language', '-')}",
            f"   confidence:   {route.get('confidence', route.get('route_confidence', '-'))}",
            f"   user_meaning: {(route.get('user_meaning') or '-')[:200]}",
            "",
            "4. Embedding information",
            f"   model:      {self.embedding_model or '-'}",
            f"   success:    {self.embedding_ok}",
            f"   dimension:  {self.embedding_dim if self.embedding_dim is not None else '-'}",
        ]
        if self.embedding_error:
            lines.append(f"   error:      {self.embedding_error}")
        lines.extend(
            [
                "",
                "5. Qdrant search request",
                json.dumps(self.qdrant_request, ensure_ascii=False, indent=2),
                "",
                f"6. Top {DEBUG_TOP_N} retrieved chunks (debug shadow search, no threshold)",
            ]
        )
        if not self.top_chunks:
            lines.append("   (none)")
        for i, ch in enumerate(self.top_chunks[:DEBUG_TOP_N], start=1):
            lines.extend(
                [
                    f"   --- chunk #{i} ---",
                    f"   score:        {ch.get('score')}",
                    f"   doc_id:       {ch.get('doc_id')}",
                    f"   version:      {ch.get('version')}",
                    f"   chunk_no:     {ch.get('chunk_no')}",
                    f"   title:        {ch.get('title')}",
                    f"   category:     {ch.get('category')}",
                    f"   language:     {ch.get('language')}",
                    f"   content_hash: {ch.get('content_hash')}",
                    f"   preview:      {_preview(ch.get('chunk') or '')}",
                ]
            )
        lines.extend(["", "7. Discarded chunks (why not in production result)"])
        if not self.discarded:
            lines.append("   (none — all shadow candidates accounted for)")
        for d in self.discarded[:25]:
            lines.append(f"   - doc={d.get('doc_id')} chunk={d.get('chunk_no')} score={d.get('score')}")
            lines.append(f"     Discarded because: {d.get('reason')}")
        fc = self.final_chunks if isinstance(self.final_chunks, list) else []
        ctx_len = len(self.grounding_context or "")
        titles = list(dict.fromkeys(str(c.get("title") or c.get("category") or "?") for c in fc))
        lines.extend(
            [
                "",
                "8. Final chunks passed to LLM (production path)",
                f"   number of chunks:    {len(fc)}",
                f"   total context length: {ctx_len}",
                f"   chunk titles:        {titles}",
                "",
                "9. LLM grounding prompt (no API keys)",
                f"   system prompt summary: {self.grounding_summary or '-'}",
                "   retrieved context:",
            ]
        )
        if self.grounding_context:
            for ln in (self.grounding_context[:4000]).splitlines()[:40]:
                lines.append(f"     {ln}")
            if len(self.grounding_context) > 4000:
                lines.append("     … (truncated)")
        else:
            lines.append("     (empty)")
        lines.extend(
            [
                "   user question:",
                f"     {self.user_question or self.original_query or '-'}",
                "",
                "10. Final answer source",
                f"   {self.answer_source}",
                "",
                "11. Fallback detail",
            ]
        )
        if self.fallback_reason:
            lines.append(f"   Fallback because:")
            lines.append(f"   {self.fallback_reason}")
        else:
            lines.append("   (no fallback)")
        lines.append("=" * 72)
        return "\n".join(lines)


def get_rag_debug_session() -> RagDebugSession | None:
    if not is_rag_debug_enabled():
        return None
    sess = getattr(_TLS, "rag_debug_session", None)
    if sess is None:
        sess = RagDebugSession()
        _TLS.rag_debug_session = sess
    return sess


def clear_rag_debug_session() -> None:
    if hasattr(_TLS, "rag_debug_session"):
        delattr(_TLS, "rag_debug_session")


def begin_rag_debug_turn(
    original_query: str,
    *,
    preprocessing_query: str = "",
    normalized_query: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> None:
    if not is_rag_debug_enabled():
        return
    clear_rag_debug_session()
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.original_query = original_query or ""
    sess.preprocessing_query = preprocessing_query or ""
    sess.normalized_query = normalized_query or ""
    sess.ai_route = dict(ai_route or {})
    sess.reply_lang = reply_lang or ""
    sess.user_question = original_query or ""


def record_embedding_debug(*, model: str, ok: bool, dim: int | None = None, error: str = "") -> None:
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.embedding_model = model
    sess.embedding_ok = ok
    sess.embedding_dim = dim
    sess.embedding_error = error or ""


def record_qdrant_request_debug(request: dict[str, Any]) -> None:
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.qdrant_request = dict(request)


def analyze_discarded_chunks(
    *,
    shadow_top: list[dict[str, Any]],
    production_hits: list[dict[str, Any]],
    score_threshold: float,
    top_k: int,
    category_filter: list[str] | None,
    language_filter: str | None,
    active_doc_ids: list[int] | None,
    scoped_backoff: bool,
) -> list[dict[str, str]]:
    prod_keys = {_chunk_key(h) for h in production_hits}
    scope = [c.lower() for c in (category_filter or []) if c]
    lang = (language_filter or "").strip().lower()
    active = set(int(x) for x in (active_doc_ids or []))
    out: list[dict[str, str]] = []

    for rank, ch in enumerate(shadow_top, start=1):
        key = _chunk_key(ch)
        if key in prod_keys:
            continue
        reasons: list[str] = []
        score = float(ch.get("score") or 0)
        doc_id = ch.get("doc_id")
        title = str(ch.get("title") or "").lower()
        category = str(ch.get("category") or "").lower()
        chunk_lang = str(ch.get("language") or "").lower()

        if score < score_threshold:
            reasons.append(f"score below threshold ({score:.4f} < {score_threshold})")
        if active and doc_id is not None and int(doc_id) not in active:
            reasons.append("inactive document (doc_id not in active set)")
        if lang and chunk_lang not in (lang, "auto"):
            reasons.append(f"language mismatch (chunk={chunk_lang}, filter={lang})")
        if scope and title not in scope and category not in scope:
            reasons.append(f"category/title scope mismatch (title={title}, category={category}, filter={scope})")
        if rank > top_k:
            reasons.append(f"top_k cutoff (rank {rank} > {top_k})")
        if scoped_backoff and scope and title not in scope and category not in scope:
            reasons.append("scoped filter removed chunk; production used scoped backoff")

        if not reasons:
            reasons.append("filtered by metadata or not selected in production path")

        out.append(
            {
                "doc_id": str(doc_id),
                "chunk_no": str(ch.get("chunk_no")),
                "score": f"{score:.4f}",
                "reason": "; ".join(reasons),
            }
        )
    return out


def run_shadow_retrieval_debug(
    *,
    collection: str,
    query_vector: list[float],
    score_threshold: float,
    top_k: int,
    category_filter: list[str] | None,
    language_filter: str | None,
    doc_id_filter: list[int] | None,
    production_hits: list[dict[str, Any]],
    scoped_backoff: bool,
) -> None:
    if not is_rag_debug_enabled():
        return
    sess = get_rag_debug_session()
    if not sess:
        return
    try:
        from services.qdrant_service import search_knowledge_vectors

        raw = search_knowledge_vectors(
            collection,
            query_vector,
            top_k=DEBUG_TOP_N,
            min_score=None,
            category_filter=None,
            language_filter=None,
            doc_id_filter=None,
        )
        sess.top_chunks = [_hit_from_raw(r) for r in raw]
        sess.discarded = analyze_discarded_chunks(
            shadow_top=sess.top_chunks,
            production_hits=production_hits,
            score_threshold=score_threshold,
            top_k=top_k,
            category_filter=category_filter,
            language_filter=language_filter,
            active_doc_ids=doc_id_filter,
            scoped_backoff=scoped_backoff,
        )
    except Exception as exc:
        sess.discarded.append({"doc_id": "-", "chunk_no": "-", "score": "-", "reason": f"shadow search failed: {exc}"})


def record_final_chunks_debug(hits: list[dict[str, Any]], grounding_context: str = "") -> None:
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.final_chunks = list(hits or [])
    if grounding_context:
        sess.grounding_context = grounding_context


def record_grounding_prompt_debug(
    *,
    context: str,
    summary: str,
    user_question: str,
) -> None:
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.grounding_context = context or ""
    sess.grounding_summary = summary or ""
    sess.user_question = user_question or sess.original_query


def set_answer_source_debug(source: str, *, fallback_reason: str = "") -> None:
    sess = get_rag_debug_session()
    if not sess:
        return
    sess.answer_source = source or "UNKNOWN"
    sess.fallback_reason = fallback_reason or ""


def flush_rag_debug_report() -> None:
    if not is_rag_debug_enabled():
        return
    sess = get_rag_debug_session()
    if not sess:
        return
    print(sess.to_report(), flush=True)
    clear_rag_debug_session()
