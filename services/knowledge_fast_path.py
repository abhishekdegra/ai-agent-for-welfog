"""
Welfog FAQ / policy fast path — AI meaning → vector KB (all docs) → complete answer.

ONE cached micro-LLM per turn (any language) decides informational KB vs live API.
No fixed Hindi/English keyword routing gates. No faqs-only silo.
"""
from __future__ import annotations

from typing import Optional

from utils.reasoning_log import log_reasoning


def try_knowledge_fast_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
) -> Optional[str]:
    """
    AI-first informational answer: meaning → embeddings over ALL admin KB docs
    → grounded / localized reply in the customer's language.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    from services.turn_intent_coordinator import resolve_kb_turn_ai_first

    resolved = resolve_kb_turn_ai_first(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ctx=None,
        preflight=True,
    )
    if resolved.get("action") != "answer_kb":
        return None

    from services.kb_service import (
        _kb_answer_body_is_usable,
        ensure_knowledge_cache_fresh,
        format_dynamic_kb_answer,
    )
    from services.translation_service import customer_reply_language

    ensure_knowledge_cache_fresh()
    rl = reply_lang or customer_reply_language(original_msg)
    kb_keys = list(resolved.get("kb_keys") or [])
    user_meaning = (resolved.get("user_meaning") or "").strip()
    ai_route_hint = {
        "user_meaning": user_meaning,
        "data_channel": "kb",
        "kb_keys": kb_keys,
        "intent": (resolved.get("kb_topic") or "general"),
    }
    topic = (resolved.get("kb_topic") or "general_faq").strip()

    # Full-corpus grounded path first (faqs/shipping/owner/seller/… by vectors).
    try:
        from services.knowledge_answer_service import (
            generate_grounded_kb_answer_from_qdrant,
            should_generate_qdrant_kb_answer,
        )

        if should_generate_qdrant_kb_answer(ai_route_hint):
            grounded = generate_grounded_kb_answer_from_qdrant(
                original_msg,
                msg_en=msg_en,
                keys=None,  # soft: search all active customer docs
                reply_lang=rl,
                conversation_context=conversation_context,
                user_meaning_en=user_meaning,
                ai_route=ai_route_hint,
            )
            if grounded and _kb_answer_body_is_usable(grounded, min_chars=20):
                log_reasoning(
                    f"Knowledge fast path (AI): topic={topic} qdrant_grounded "
                    f"source={resolved.get('source')}"
                )
                _record_kb_fast_route(topic)
                return grounded
    except Exception as exc:
        log_reasoning(f"Knowledge fast path grounded skipped: {exc}")

    body = format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=rl,
        conversation_context=conversation_context,
        suggested_keys=kb_keys or None,
        ai_route=ai_route_hint,
    )
    if body and _kb_answer_body_is_usable(body, min_chars=20):
        log_reasoning(
            f"Knowledge fast path (AI): topic={topic} dynamic KB "
            f"source={resolved.get('source')}"
        )
        _record_kb_fast_route(topic)
        return body

    return None


def _record_kb_fast_route(topic: str) -> None:
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        record_route_step("knowledge_fast")
        record_route(intent="general", source=f"knowledge_fast_{topic}")
    except ImportError:
        pass
