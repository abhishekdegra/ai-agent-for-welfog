"""
Welfog FAQ / policy fast path — AI meaning → vector KB → concise answer.

ONE cached micro-LLM per turn (any language) decides informational KB vs live API.
No fixed Hindi/English keyword routing gates.
"""
from __future__ import annotations

from typing import Optional

from utils.reasoning_log import log_reasoning

_FAQ_SCORE_MIN = 0.34


def try_knowledge_fast_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
) -> Optional[str]:
    """
    AI-first informational answer: meaning → KB keys → embedding match → localize.
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
    )
    if resolved.get("action") != "answer_kb":
        return None

    from services.kb_service import (
        _customer_needs_kb_localization,
        _faq_answer_text_from_chunk,
        _finalize_kb_customer_reply,
        _ground_kb_excerpt_for_customer,
        _infer_kb_answer_budget,
        _kb_plain_excerpt,
        _plain_text_to_html_body,
        ensure_knowledge_cache_fresh,
        format_dynamic_kb_answer,
        polish_faq_reply_for_customer,
        resolve_best_faq_chunk_for_question,
    )
    from services.translation_service import customer_reply_language

    ensure_knowledge_cache_fresh()
    rl = reply_lang or customer_reply_language(original_msg)
    kb_keys = list(resolved.get("kb_keys") or ["faqs"])
    user_meaning = (resolved.get("user_meaning") or "").strip()
    ai_route_hint = {"user_meaning": user_meaning} if user_meaning else None
    topic = (resolved.get("kb_topic") or "general_faq").strip()

    faq = resolve_best_faq_chunk_for_question(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route_hint,
    )
    if faq and float(faq.get("score") or 0) >= _FAQ_SCORE_MIN:
        excerpt = _faq_answer_text_from_chunk(faq.get("chunk") or "")
        if excerpt.strip():
            budget, _, _ = _infer_kb_answer_budget(user_meaning or comb)
            budget = min(budget, 440)
            plain = _kb_plain_excerpt(excerpt, max_chars=budget)
            if _customer_needs_kb_localization(original_msg, rl):
                body = _ground_kb_excerpt_for_customer(
                    original_msg,
                    plain,
                    conversation_context=conversation_context,
                    reply_lang=rl,
                    kb_sources=kb_keys[:2],
                )
            else:
                html = polish_faq_reply_for_customer(
                    _plain_text_to_html_body(plain) or plain, original_msg
                )
                body = _finalize_kb_customer_reply(html, original_msg, rl)
            if body:
                log_reasoning(
                    f"Knowledge fast path (AI): topic={topic} "
                    f"faq_score={float(faq.get('score') or 0):.2f} "
                    f"source={resolved.get('source')}"
                )
                _record_kb_fast_route(topic)
                return body

    body = format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=rl,
        conversation_context=conversation_context,
        suggested_keys=kb_keys,
        ai_route=ai_route_hint,
    )
    if body:
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
