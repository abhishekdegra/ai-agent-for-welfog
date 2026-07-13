"""
Step 7: Grounded Knowledge Base answer generation via Qdrant retrieval + LLM.

Retrieves context from Qdrant, passes only that context to the existing answer LLM,
and never answers outside retrieved chunks.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from services.knowledge_retriever import (
    DEFAULT_ANSWER_TOP_K,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    format_knowledge_context_string,
    is_kb_intent,
    retrieve_knowledge_context,
    should_use_qdrant_retrieval,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


ANSWER_TOP_K = _env_int("KNOWLEDGE_ANSWER_TOP_K", DEFAULT_ANSWER_TOP_K)
# Chunks passed to the answer LLM (smaller = faster, less noise). Retrieval may fetch more.
ANSWER_CONTEXT_K = max(3, min(12, _env_int("KNOWLEDGE_ANSWER_CONTEXT_K", 6)))


def should_generate_qdrant_kb_answer(ai_route: dict | None) -> bool:
    """Knowledge Base intent + Qdrant retrieval backend."""
    return should_use_qdrant_retrieval(ai_route=ai_route, kb_intent=True) and is_kb_intent(
        ai_route
    )


def _trim_hits_for_answer_llm(hits: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Keep highest-scoring chunks for the grounding prompt."""
    if not hits:
        return []
    cap = max(1, int(limit if limit is not None else ANSWER_CONTEXT_K))
    ranked = sorted(hits, key=lambda h: float(h.get("score") or 0), reverse=True)
    return ranked[:cap]


def _build_retrieval_query(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    *,
    user_meaning_en: str = "",
    ai_route: dict | None = None,
) -> str:
    try:
        from services.kb_service import build_kb_retrieval_query

        q = build_kb_retrieval_query(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
        )
        if q.strip():
            return q
    except ImportError:
        pass
    parts = [
        (user_meaning_en or "").strip(),
        ((ai_route or {}).get("user_meaning") or "").strip(),
        (msg_en or "").strip(),
        (original_msg or "").strip(),
    ]
    merged = " — ".join(dict.fromkeys(p for p in parts if p))
    return merged[:900]


def _build_strict_llm_context(hits: list[dict[str, Any]]) -> str:
    """Context block containing ONLY retrieved chunk text."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        src = (hit.get("source") or hit.get("category") or "general").strip()
        title = (hit.get("title") or "").strip()
        score = float(hit.get("score") or 0)
        content = re.sub(r"\s+", " ", (hit.get("chunk") or "").strip())
        if not content:
            continue
        header = f"CHUNK {i} [category={src}"
        if title:
            header += f" title={title}"
        header += f" score={score:.3f}]"
        blocks.append(f"{header}\n{content}")
    if not blocks:
        return ""
    return (
        "AUTHORITATIVE KNOWLEDGE BASE EXCERPTS (ONLY permitted source of facts):\n\n"
        + "\n\n".join(blocks)
        + "\n\nSTRICT RULES:\n"
        "- Answer ONLY using facts present in the chunks above.\n"
        "- If the chunks do not contain the answer, say it is not in the available knowledge.\n"
        "- Do NOT invent policies, fees, timelines, phone numbers, or emails.\n"
        "- Answer ONLY what the user asked — concise, no unrelated sections.\n"
        "- Do NOT repeat or restate the user's question.\n"
        "- Copy phone/email/addresses exactly from the chunks.\n"
    )


def log_kb_answer_grounding(
    *,
    query: str = "",
    chunks_used: int = 0,
    chunk_sources: list[str] | None = None,
    top_score: float = 0.0,
    answer_source: str = "",
    retrieval_time_ms: float = 0.0,
    answer_time_ms: float = 0.0,
) -> None:
    from utils.reasoning_log import chat_log, log_reasoning

    sources = ",".join(chunk_sources or []) or "-"
    msg = (
        f"[kb-grounding] query={(query or '-')[:120]!r} "
        f"chunks_used={chunks_used} chunk_sources={sources} "
        f"top_score={top_score:.4f} answer_source={answer_source or '-'} "
        f"retrieval_time_ms={retrieval_time_ms:.1f} answer_time_ms={answer_time_ms:.1f}"
    )
    log_reasoning(msg)
    chat_log(msg)
    print(msg, flush=True)


def _kb_fallback_reply(
    original_msg: str,
    msg_en: str = "",
    *,
    reply_lang: str = "",
    answer_source: str = "fallback_no_context",
) -> str:
    try:
        from services.kb_service import _format_kb_brain_gap_reply

        return _format_kb_brain_gap_reply(original_msg, msg_en, reply_lang=reply_lang)
    except ImportError:
        from services.kb_service import format_kb_no_information_reply

        return format_kb_no_information_reply(original_msg, reply_lang=reply_lang)


def _extractive_answer_from_hits(
    hits: list[dict[str, Any]],
    original_msg: str,
    *,
    reply_lang: str = "",
    min_score: float = 0.14,
) -> str:
    """
    When grounded LLM is empty/refuses but retrieval found usable chunks,
    answer from the best chunk text (no extra LLM). Industry RAG fallback.
    """
    if not hits:
        return ""
    ranked = sorted(hits, key=lambda h: float(h.get("score") or 0), reverse=True)
    best = ranked[0]
    if float(best.get("score") or 0) < min_score:
        return ""
    try:
        from services.kb_service import format_direct_reply_from_kb_hit

        body = format_direct_reply_from_kb_hit(
            best,
            original_msg,
            reply_lang=reply_lang,
            retrieval_query=original_msg,
            fast_lane=True,
        )
        return (body or "").strip()
    except ImportError:
        chunk = re.sub(r"\s+", " ", (best.get("chunk") or "").strip())
        return chunk[:900] if chunk else ""


def _extract_llm_answer_text(ai: dict | None) -> str:
    if not isinstance(ai, dict):
        return ""
    resp = (ai.get("response") or "").strip()
    if resp:
        return resp
    for key in ("answer", "text", "reply"):
        val = (ai.get(key) or "").strip()
        if val:
            return val
    skip = {
        "reasoning",
        "intent",
        "search_query",
        "extracted_pincode",
        "needs_order_id",
        "is_welfog_related",
    }
    parts: list[str] = []
    for key, val in ai.items():
        if key in skip:
            continue
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts).strip()


def _format_llm_answer(
    response_text: str,
    original_msg: str,
    reply_lang: str,
) -> str:
    from services.kb_service import (
        _ai_response_looks_like_placeholder,
        _plain_text_to_html_body,
        polish_faq_reply_for_customer,
    )
    from services.translation_service import finalize_customer_reply

    resp = (response_text or "").strip()
    if not resp or _ai_response_looks_like_placeholder(resp):
        return ""
    resp = polish_faq_reply_for_customer(resp, original_msg)
    # Grounded answer LLM already received reply_lang instructions — skip a second
    # English↔Hinglish rewrite (often 1–7s + Groq rate-limit). Native script still localizes.
    if resp.strip().startswith("<"):
        return (
            finalize_customer_reply(
                resp, original_msg, reply_lang, allow_llm_style_rewrite=False
            )
            or ""
        )
    body = _plain_text_to_html_body(resp) or resp
    return (
        finalize_customer_reply(
            body, original_msg, reply_lang, allow_llm_style_rewrite=False
        )
        or ""
    )


def generate_grounded_kb_answer_from_qdrant(
    original_msg: str,
    *,
    msg_en: str = "",
    keys: list[str] | None = None,
    reply_lang: str = "",
    conversation_context: str = "",
    user_meaning_en: str = "",
    ai_route: dict | None = None,
) -> str | None:
    """
    Qdrant retrieval → strict-context LLM answer.

    Returns:
      - str: grounded answer or existing fallback when no context / LLM miss
      - None: Qdrant KB answer path not applicable (caller uses legacy flow)
    """
    if not should_generate_qdrant_kb_answer(ai_route):
        try:
            from services.knowledge_rag_debug import (
                begin_rag_debug_turn,
                flush_rag_debug_report,
                is_rag_debug_enabled,
                set_answer_source_debug,
            )

            if is_rag_debug_enabled():
                begin_rag_debug_turn(
                    original_msg,
                    preprocessing_query=msg_en,
                    ai_route=ai_route,
                    reply_lang=reply_lang,
                )
                set_answer_source_debug(
                    "ROUTING_FALLBACK",
                    fallback_reason="Qdrant grounded KB path not applicable (brain route or backend).",
                )
                flush_rag_debug_report()
        except ImportError:
            pass
        return None

    from services.translation_service import customer_reply_language, resolve_customer_reply_lang

    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return None

    rl = resolve_customer_reply_lang(original_msg, reply_lang or customer_reply_language(original_msg))
    retrieval_query = _build_retrieval_query(
        original_msg,
        msg_en,
        conversation_context,
        user_meaning_en=user_meaning_en,
        ai_route=ai_route,
    )

    try:
        from services.knowledge_rag_debug import begin_rag_debug_turn, is_rag_debug_enabled
        from services.knowledge_retriever import _normalize_query

        if is_rag_debug_enabled():
            begin_rag_debug_turn(
                original_msg,
                preprocessing_query=retrieval_query,
                normalized_query=_normalize_query(retrieval_query),
                ai_route=ai_route,
                reply_lang=rl,
            )
    except ImportError:
        pass

    route_keys = list(keys or (ai_route or {}).get("kb_keys") or [])
    t0 = time.perf_counter()

    # Reuse same-turn Qdrant hits when authoritative corpus was already retrieved.
    reused = False
    retrieval_hits: list[dict[str, Any]] = []
    top_score = 0.0
    try:
        from services.chat_flow_telemetry import get_kb_grounding_context

        prior_hits, _prior_corpus = get_kb_grounding_context()
        if prior_hits:
            retrieval_hits = list(prior_hits)
            top_score = max((float(h.get("score") or 0) for h in retrieval_hits), default=0.0)
            reused = True
    except ImportError:
        pass

    if not reused:
        # Soft recall floor — gray-band scores still reach extractive/LLM answer.
        soft_floor = min(float(DEFAULT_MIN_SCORE), 0.14)
        retrieval = retrieve_knowledge_context(
            retrieval_query,
            top_k=max(ANSWER_TOP_K, ANSWER_CONTEXT_K),
            min_score=soft_floor,
            categories=route_keys or None,
            language=None,
            ai_route=ai_route,
            kb_intent=True,
            log=True,
        )
        retrieval_hits = list(retrieval.hits or [])
        top_score = float(retrieval.top_score or 0.0)

    retrieval_ms = (time.perf_counter() - t0) * 1000.0
    answer_hits = _trim_hits_for_answer_llm(retrieval_hits)

    if answer_hits:
        try:
            from services.chat_flow_telemetry import (
                lock_authoritative_kb_route_from_retrieval,
                set_kb_grounding_context,
            )

            lock_authoritative_kb_route_from_retrieval(
                chunks=len(answer_hits),
                top_score=top_score,
                ai_route=ai_route,
            )
            set_kb_grounding_context(answer_hits)
        except ImportError:
            pass

    if not answer_hits:
        try:
            from services.knowledge_rag_debug import (
                flush_rag_debug_report,
                record_final_chunks_debug,
                set_answer_source_debug,
            )

            record_final_chunks_debug([])
            set_answer_source_debug(
                "NO_CONTEXT",
                fallback_reason="No chunks retrieved above threshold.",
            )
            flush_rag_debug_report()
        except ImportError:
            pass
        log_kb_answer_grounding(
            query=retrieval_query,
            chunks_used=0,
            top_score=0.0,
            answer_source="fallback_no_context",
            retrieval_time_ms=retrieval_ms,
            answer_time_ms=0.0,
        )
        try:
            from services.kb_service import log_kb_pipeline_complete

            log_kb_pipeline_complete(
                query=retrieval_query,
                language=rl,
                intent="kb",
                route="qdrant_grounded",
                retrieved_chunks=0,
                similarity_score=0.0,
                response_time_ms=retrieval_ms,
                source="fallback_no_context",
            )
        except ImportError:
            pass
        return _kb_fallback_reply(original_msg, msg_en, reply_lang=rl)

    strict_context = _build_strict_llm_context(answer_hits)
    if not strict_context.strip():
        try:
            from services.knowledge_rag_debug import (
                flush_rag_debug_report,
                record_final_chunks_debug,
                set_answer_source_debug,
            )

            record_final_chunks_debug(answer_hits)
            set_answer_source_debug(
                "NO_CONTEXT",
                fallback_reason="Retrieved chunks produced empty grounding context.",
            )
            flush_rag_debug_report()
        except ImportError:
            pass
        log_kb_answer_grounding(
            query=retrieval_query,
            chunks_used=0,
            top_score=top_score,
            answer_source="fallback_no_context",
            retrieval_time_ms=retrieval_ms,
            answer_time_ms=0.0,
        )
        return _kb_fallback_reply(original_msg, msg_en, reply_lang=rl)

    chunk_sources = list(
        dict.fromkeys(
            str(h.get("title") or h.get("source") or h.get("category") or "general")
            for h in answer_hits
        )
    )

    try:
        from services.knowledge_rag_debug import (
            record_final_chunks_debug,
            record_grounding_prompt_debug,
        )

        record_final_chunks_debug(answer_hits, strict_context)
        record_grounding_prompt_debug(
            context=strict_context,
            summary="ai_brain_answer with strict KB-only grounding rules (Welfog assistant JSON response).",
            user_question=original_msg,
        )
    except ImportError:
        pass

    try:
        from services.chat_flow_telemetry import (
            clear_kb_grounding_operation,
            ensure_kb_grounding_llm_slot,
            mark_kb_grounding_operation,
        )

        mark_kb_grounding_operation()
        ensure_kb_grounding_llm_slot()
    except ImportError:
        pass

    t1 = time.perf_counter()
    answer_text = ""
    try:
        from services.ai_service import ai_brain_answer

        ai = ai_brain_answer(
            original_msg,
            strict_context,
            conversation_context=conversation_context,
            reply_lang=rl,
        )
        answer_text = _extract_llm_answer_text(ai)
    except Exception as exc:
        print(f"[kb-grounding] LLM error: {exc}", flush=True)
        answer_text = ""
    finally:
        try:
            from services.chat_flow_telemetry import clear_kb_grounding_operation

            clear_kb_grounding_operation()
        except ImportError:
            pass

    try:
        from services.knowledge_grounding_validator import (
            enforce_kb_fact_grounding,
            rebuild_contact_answer_from_chunks,
        )

        answer_text = enforce_kb_fact_grounding(
            answer_text,
            answer_hits,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if not (answer_text or "").strip():
            answer_text = rebuild_contact_answer_from_chunks(
                answer_hits,
                original_msg=original_msg,
                msg_en=msg_en,
            )
    except ImportError:
        pass

    answer_ms = (time.perf_counter() - t1) * 1000.0
    formatted = _format_llm_answer(answer_text, original_msg, rl)

    if formatted:
        try:
            from services.knowledge_grounding_validator import enforce_kb_fact_grounding

            formatted = enforce_kb_fact_grounding(
                formatted,
                answer_hits,
                original_msg=original_msg,
                msg_en=msg_en,
            ) or formatted
        except ImportError:
            pass

    if not formatted:
        try:
            from services.knowledge_grounding_validator import rebuild_contact_answer_from_chunks

            chunk_reply = rebuild_contact_answer_from_chunks(
                answer_hits,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            if chunk_reply:
                formatted = chunk_reply
        except ImportError:
            pass

    # LLM refused / empty but retrieval succeeded — serve chunk extractively
    # instead of false "not in knowledge base" (common on Hinglish FAQ asks).
    if not formatted and answer_hits and top_score >= 0.14:
        extractive = _extractive_answer_from_hits(
            answer_hits,
            original_msg,
            reply_lang=rl,
            min_score=0.14,
        )
        if extractive:
            log_kb_answer_grounding(
                query=retrieval_query,
                chunks_used=len(answer_hits),
                chunk_sources=chunk_sources,
                top_score=top_score,
                answer_source="extractive_chunk_fallback",
                retrieval_time_ms=retrieval_ms,
                answer_time_ms=answer_ms,
            )
            try:
                from services.kb_service import log_kb_pipeline_complete

                log_kb_pipeline_complete(
                    query=retrieval_query,
                    language=rl,
                    intent="kb",
                    route="qdrant_extractive",
                    retrieved_chunks=len(answer_hits),
                    similarity_score=top_score,
                    response_time_ms=retrieval_ms + answer_ms,
                    source="extractive_chunk_fallback",
                )
            except ImportError:
                pass
            return extractive

    if not formatted:
        try:
            from services.knowledge_rag_debug import flush_rag_debug_report, set_answer_source_debug

            set_answer_source_debug(
                "LLM_FALLBACK",
                fallback_reason="LLM returned empty response or placeholder text.",
            )
            flush_rag_debug_report()
        except ImportError:
            pass
        log_kb_answer_grounding(
            query=retrieval_query,
            chunks_used=len(answer_hits),
            chunk_sources=chunk_sources,
            top_score=top_score,
            answer_source="fallback_llm_empty",
            retrieval_time_ms=retrieval_ms,
            answer_time_ms=answer_ms,
        )
        try:
            from services.kb_service import log_kb_pipeline_complete

            log_kb_pipeline_complete(
                query=retrieval_query,
                language=rl,
                intent="kb",
                route="qdrant_grounded",
                retrieved_chunks=len(answer_hits),
                similarity_score=top_score,
                response_time_ms=retrieval_ms + answer_ms,
                source="fallback_llm_empty",
            )
        except ImportError:
            pass
        return _kb_fallback_reply(original_msg, msg_en, reply_lang=rl)

    try:
        from services.knowledge_rag_debug import flush_rag_debug_report, set_answer_source_debug

        set_answer_source_debug("QDRANT_GROUNDED")
        flush_rag_debug_report()
    except ImportError:
        pass
    log_kb_answer_grounding(
        query=retrieval_query,
        chunks_used=len(answer_hits),
        chunk_sources=chunk_sources,
        top_score=top_score,
        answer_source="qdrant_grounded_llm",
        retrieval_time_ms=retrieval_ms,
        answer_time_ms=answer_ms,
    )
    try:
        from services.kb_service import log_kb_pipeline_complete

        log_kb_pipeline_complete(
            query=retrieval_query,
            language=rl,
            intent="kb",
            route="qdrant_grounded",
            retrieved_chunks=len(answer_hits),
            similarity_score=top_score,
            response_time_ms=retrieval_ms + answer_ms,
            source="qdrant_grounded_llm",
        )
    except ImportError:
        pass
    return formatted


def ensure_kb_grounding_corpus_for_turn(
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Guarantee per-turn grounding corpus for factual KB enforcement.
    Reuses stored retrieval hits or performs one Qdrant retrieval when needed.
    """
    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            get_kb_grounding_context,
            is_authoritative_kb_route_locked,
            lock_authoritative_kb_route_from_retrieval,
            set_kb_grounding_context,
        )
        from services.knowledge_grounding_validator import _chunk_corpus
    except ImportError:
        return [], ""

    hits, corpus = get_kb_grounding_context()
    if corpus.strip() or hits:
        return hits, corpus

    route = ai_route if isinstance(ai_route, dict) else get_cached_brain_route()
    route = route if isinstance(route, dict) else {}

    needs_corpus = is_authoritative_kb_route_locked()
    if not needs_corpus:
        try:
            from utils.helpers import (
                _text_asks_customer_care_contact,
                message_asks_welfog_social_media,
                message_is_knowledge_information_request,
            )

            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            needs_corpus = (
                _text_asks_customer_care_contact(comb)
                or message_asks_welfog_social_media(comb)
                or message_is_knowledge_information_request(comb)
                or (route.get("data_channel") or "").strip().lower() == "kb"
            )
        except ImportError:
            needs_corpus = (route.get("data_channel") or "").strip().lower() == "kb"

    if not needs_corpus:
        return [], ""

    retrieval_query = _build_retrieval_query(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route,
    )
    keys = list(route.get("kb_keys") or [])
    try:
        retrieval = retrieve_knowledge_context(
            retrieval_query or f"{original_msg} {msg_en}".strip(),
            top_k=ANSWER_TOP_K,
            min_score=DEFAULT_MIN_SCORE,
            categories=keys or None,
            ai_route=route,
            kb_intent=True,
            log=False,
        )
        if retrieval.hits:
            chunk_corpus = _chunk_corpus(retrieval.hits)
            set_kb_grounding_context(retrieval.hits, corpus=chunk_corpus)
            lock_authoritative_kb_route_from_retrieval(
                chunks=len(retrieval.hits),
                top_score=retrieval.top_score,
                ai_route=route,
            )
            return retrieval.hits, chunk_corpus
    except Exception:
        pass

    if keys:
        try:
            from services.kb_service import read_concatenated_kb_file_contents

            blob = read_concatenated_kb_file_contents(keys)
            if blob.strip():
                set_kb_grounding_context([], corpus=blob)
                return [], blob
        except ImportError:
            pass

    try:
        from services.kb_service import get_customer_kb_keys, read_concatenated_kb_file_contents

        # Last-resort corpus from dynamic Admin catalog (no hardcoded contact/company keys).
        all_keys = get_customer_kb_keys()[:8]
        if all_keys:
            blob = read_concatenated_kb_file_contents(all_keys)
            if blob.strip():
                set_kb_grounding_context([], corpus=blob)
                return [], blob
    except ImportError:
        pass

    return [], ""


def try_structured_kb_answer_from_corpus(
    corpus: str,
    hits: list[dict[str, Any]],
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> str:
    """Deterministic KB answer built only from grounding corpus (no LLM)."""
    try:
        from services.knowledge_grounding_validator import (
            _chunk_corpus,
            extract_grounded_facts,
            rebuild_contact_answer_from_corpus,
        )
    except ImportError:
        return ""

    blob = (corpus or _chunk_corpus(hits)).strip()
    if not blob:
        return ""

    facts = extract_grounded_facts(blob)
    try:
        from utils.helpers import (
            _text_asks_customer_care_contact,
            message_asks_welfog_social_media,
        )

        contact_turn = _text_asks_customer_care_contact(f"{original_msg} {msg_en}")
        social_turn = message_asks_welfog_social_media(f"{original_msg} {msg_en}")
    except ImportError:
        contact_turn = facts.has_contact_facts()
        social_turn = bool(facts.urls)

    if contact_turn and facts.has_contact_facts():
        cc = rebuild_contact_answer_from_corpus(
            blob, original_msg=original_msg, msg_en=msg_en
        )
        if cc:
            return cc

    if social_turn and facts.urls:
        try:
            from services.kb_service import format_welfog_social_media_reply_from_kb

            social = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=reply_lang,
                ai_confirmed=True,
            )
            if social:
                return social
        except ImportError:
            pass

    return ""


def resolve_authoritative_kb_answer(
    original_msg: str,
    *,
    msg_en: str = "",
    keys: list[str] | None = None,
    reply_lang: str = "",
    conversation_context: str = "",
    user_meaning_en: str = "",
    ai_route: dict | None = None,
) -> str | None:
    """
    Single authoritative KB answer pipeline:
      retrieve → structured facts → grounded LLM → fact contract.
    """
    route = dict(ai_route or {})
    if keys:
        route.setdefault("kb_keys", keys)

    hits, corpus = ensure_kb_grounding_corpus_for_turn(
        original_msg,
        msg_en,
        conversation_context=conversation_context,
        ai_route=route,
    )

    structured = try_structured_kb_answer_from_corpus(
        corpus,
        hits,
        original_msg,
        msg_en,
        ai_route=route,
        reply_lang=reply_lang,
    )
    if structured:
        return structured

    if should_generate_qdrant_kb_answer(route):
        qdrant_answer = generate_grounded_kb_answer_from_qdrant(
            original_msg,
            msg_en=msg_en,
            keys=keys,
            reply_lang=reply_lang,
            conversation_context=conversation_context,
            user_meaning_en=user_meaning_en,
            ai_route=route,
        )
        if qdrant_answer and qdrant_answer.strip():
            return qdrant_answer

    if corpus.strip() or hits:
        try:
            from services.knowledge_grounding_validator import enforce_kb_fact_grounding

            rebuilt = enforce_kb_fact_grounding(
                "",
                hits,
                corpus=corpus,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            if rebuilt:
                return rebuilt
        except ImportError:
            pass
        if structured:
            return structured

    return None


def get_qdrant_kb_context_for_llm(
    query: str,
    *,
    keys: list[str] | None = None,
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> str:
    """Return formatted retrieved context only (no answer generation)."""
    if not should_generate_qdrant_kb_answer(ai_route):
        return ""
    result = retrieve_knowledge_context(
        query,
        top_k=ANSWER_TOP_K,
        min_score=DEFAULT_MIN_SCORE,
        categories=keys,
        language=reply_lang or None,
        ai_route=ai_route,
        kb_intent=True,
        log=False,
    )
    return format_knowledge_context_string(result.hits)
