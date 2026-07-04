"""
Zero-LLM recovery for KB / chitchat / out-of-domain — before infra busy fallback.

Used when deadline, LLM budget, or locked-route tiers would otherwise show
'high traffic' while embeddings/KB already have the answer.
"""
from __future__ import annotations

from typing import Any, Optional

from utils.reasoning_log import log_reasoning


def _zero_llm_kb_turn_blocked(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> bool:
    """
    Fast structural guards for zero-LLM KB — avoids turn_blocks_kb_pre_scope
    (pulls heavy catalog pipelines; can block chat 90s+ on informational turns).
    """
    import re

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return True
    try:
        from utils.helpers import (
            _is_plausible_order_id,
            _text_has_product_shopping_intent_core,
            _text_wants_order_history_list_in_chat,
            extract_order_id,
            extract_product_id,
        )

        comb_low = comb.lower()
        if extract_product_id(comb_low) or extract_order_id(comb_low):
            return True
        if _is_plausible_order_id(comb_low) or re.search(r"\b[1-9]\d{5}\b", comb):
            return True
        try:
            from utils.helpers import (
                message_asks_welfog_social_media,
                message_is_knowledge_information_request,
                message_is_welfog_about_request,
            )

            if (
                message_is_knowledge_information_request(comb, conversation_context)
                or message_is_welfog_about_request(comb)
                or message_asks_welfog_social_media(comb, conversation_context)
            ):
                return False
        except ImportError:
            pass
        if _text_has_product_shopping_intent_core(comb):
            return True
        if _text_wants_order_history_list_in_chat(comb, conversation_context):
            return True
        try:
            from utils.helpers import (
                _text_has_refund_or_return_intent,
                _text_is_live_order_lookup_intent,
                _text_is_order_delivery_issue,
                _text_is_order_tracking_intent,
            )

            if (
                _text_is_order_tracking_intent(comb)
                or _text_is_live_order_lookup_intent(comb, conversation_context)
                or _text_is_order_delivery_issue(comb)
                or _text_has_refund_or_return_intent(comb)
            ):
                return True
        except ImportError:
            pass
    except ImportError:
        pass
    return False


def try_pure_embedding_kb_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
) -> Optional[str]:
    """
    AI KB preflight alias — one ai_classify_kb_turn + vector retrieval.
    No keyword-list routing (any language / typo / style).
    """
    try:
        from services.knowledge_query_pipeline import try_ai_first_kb_early_reply

        return try_ai_first_kb_early_reply(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            preflight=True,
        )
    except ImportError:
        return None


def try_vector_kb_only_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
) -> Optional[str]:
    """Embedding KB only — no extra classifier LLM (brain already ran)."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    if _zero_llm_kb_turn_blocked(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return None
    try:
        from services.knowledge_query_pipeline import try_ai_first_kb_early_reply

        return try_ai_first_kb_early_reply(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            ai_route=ai_route,
        )
    except ImportError:
        pass
    return None


def try_embedding_kb_only_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
) -> Optional[str]:
    """AI intent + vector KB — any language; embedding fallback when classifier unavailable."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    if _zero_llm_kb_turn_blocked(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return None

    if isinstance(ai_route, dict):
        try:
            from services.early_live_dispatch import turn_blocks_kb_pre_scope

            if turn_blocks_kb_pre_scope(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
            ):
                return None
        except ImportError:
            pass
    else:
        try:
            from services.conversation_scope import _has_definite_welfog_shopping_signal

            if _has_definite_welfog_shopping_signal(comb):
                return None
        except ImportError:
            pass

    try:
        from services.chat_flow_telemetry import get_cached_brain_route

        brain = get_cached_brain_route()
        if isinstance(brain, dict):
            ch = (brain.get("data_channel") or "").strip().lower()
            if ch in ("live_api", "catalog"):
                return None
    except ImportError:
        pass

    try:
        from services.knowledge_query_pipeline import try_ai_first_kb_early_reply
        from services.kb_service import get_knowledge_version

        body = try_ai_first_kb_early_reply(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            ai_route=ai_route,
        )
        if body:
            _log_intel(
                original_msg,
                reply_lang,
                intent="general",
                route="ai_first_kb",
                source="kb_ai_vector",
                knowledge_version=get_knowledge_version(),
            )
            return body
    except ImportError:
        pass
    return None


def try_lightweight_conversational_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ctx: dict | None = None,
) -> Optional[str]:
    """
    Scope AI reply when KB/API miss — any language/spelling, no keyword lists.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    try:
        from services.conversation_scope import _has_definite_welfog_shopping_signal

        if _has_definite_welfog_shopping_signal(comb):
            return None
    except ImportError:
        pass

    try:
        from services.chitchat_resolver import try_scope_ai_early_reply

        scope_hit = try_scope_ai_early_reply(
            original_msg,
            msg_en,
            conversation_context or "",
            reply_lang=reply_lang,
            preflight=False,
        )
        if scope_hit:
            body, _route = scope_hit
            if body and body.strip():
                log_reasoning(
                    "Lightweight conversational reply — scope AI (deadline/KB miss)."
                )
                return body
    except ImportError:
        pass

    return None


def try_zero_llm_customer_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
    ctx: dict | None = None,
) -> Optional[str]:
    """Semantic KB + chitchat + polite OOD — no routing LLM."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    conv_body = try_lightweight_conversational_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ctx=ctx,
    )
    if conv_body:
        return conv_body

    if _zero_llm_kb_turn_blocked(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return None

    # 1) Structural greeting / thanks — brain + AI chitchat only (no template fast path)
    try:
        from services.conversation_zero_llm_fallback import try_pure_embedding_kb_reply

        body = try_pure_embedding_kb_reply(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
        )
        if body:
            return body
    except ImportError:
        pass

    # 3) Embedding KB + optional classifier
    body = try_embedding_kb_only_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ai_route=ai_route,
    )
    if body:
        return body

    # 3) Scope AI for chitchat / OOD when KB miss
    try:
        from services.chitchat_resolver import try_scope_ai_early_reply

        scope_hit = try_scope_ai_early_reply(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
        )
        if scope_hit:
            scope_body, _ = scope_hit
            if scope_body:
                return scope_body
    except ImportError:
        pass

    # 4) Semantic knowledge pipeline (embedding-first; LLM only if budget left)
    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if not should_skip_micro_classifier_llm():
            from services.knowledge_query_pipeline import (
                try_knowledge_reply_before_interference,
            )

            kb_sem = try_knowledge_reply_before_interference(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang=reply_lang,
                ai_route=ai_route,
            )
            if kb_sem:
                _log_intel(
                    original_msg,
                    reply_lang,
                    intent="general",
                    route="knowledge_pre_scope",
                    source="semantic_kb",
                    knowledge_version=_kv(),
                )
                return kb_sem
    except ImportError:
        pass

    # 4) Polite out-of-domain (template — no LLM when budget tight)
    try:
        from services.conversation_scope import (
            build_off_topic_polite_reply,
            message_is_obvious_off_topic_outside_welfog,
        )

        if message_is_obvious_off_topic_outside_welfog(comb):
            ood = build_off_topic_polite_reply(
                original_msg,
                msg_en,
                reply_lang=reply_lang,
                ai_route=ai_route,
                conversation_context=conversation_context,
                prefer_llm=True,
            )
            if ood:
                _log_intel(
                    original_msg,
                    reply_lang,
                    intent="out_of_domain",
                    route="off_topic_polite",
                    source="scope_template",
                )
                return ood
    except ImportError:
        pass

    return None


def try_semantic_kb_preflight_fast_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
) -> Optional[tuple[str, dict]]:
    """
    Early chat path: embedding-matched KB before universal brain LLM.
    Returns (html, ai_route_snapshot) or None.
    """
    body = try_embedding_kb_only_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
    )
    if not body:
        return None
    return body, {
        "intent": "general",
        "data_channel": "kb",
        "route_handler": "kb_ai_first_preflight",
        "conversation_scope": "welfog_support",
        "is_welfog_related": True,
        "run_catalog_search": False,
        "needs_order_id": False,
    }


def _kv() -> str:
    try:
        from services.kb_service import get_knowledge_version

        return get_knowledge_version()
    except ImportError:
        return "-"


def _log_intel(
    original_msg: str,
    reply_lang: str,
    *,
    intent: str,
    route: str,
    source: str,
    rag_source_file: str = "",
    chunk_score: float = 0.0,
    knowledge_version: str = "",
) -> None:
    try:
        from services.chat_flow_telemetry import log_conversation_intel
        from services.translation_service import customer_reply_language

        log_conversation_intel(
            query=original_msg,
            language=reply_lang or customer_reply_language(original_msg),
            intent=intent,
            route=route,
            rag_source_file=rag_source_file,
            chunk_score=chunk_score,
            knowledge_version=knowledge_version or _kv(),
            source=source,
        )
    except ImportError:
        log_reasoning(
            f"[conversation-intel] intent={intent} route={route} source={source} "
            f"rag={rag_source_file or '-'} kv={knowledge_version or '-'}"
        )
