"""
AI-first general chitchat — any language, any phrasing.

Priority: brain conversation_scope → scope micro-LLM → keyword fallback (LLM down only).
"""
from __future__ import annotations

import re
import threading
from typing import Any, Optional

from utils.reasoning_log import log_reasoning

_RESOLVE_CACHE = threading.local()
_IN_RESOLVE = threading.local()


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _brain_scope_decision(ai_route: dict | None):
    from services.conversation_scope import SCOPE_CHITCHAT, SCOPE_OUT, scope_from_ai_route

    if not isinstance(ai_route, dict):
        return None
    dec = scope_from_ai_route(ai_route)
    if not dec:
        return None
    if dec.scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
        return dec
    return None


def _brain_misrouted_chitchat_as_product(ai_route: dict | None) -> bool:
    """Brain locked product search but message reads like casual talk to the bot."""
    if not isinstance(ai_route, dict):
        return False
    intent = (ai_route.get("intent") or "").strip().lower()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if intent != "product" or channel != "catalog":
        return False
    if not ai_route.get("run_catalog_search"):
        return False
    sq = (ai_route.get("search_query") or "").strip().lower()
    um = (ai_route.get("user_meaning") or "").lower()
    menu_words = (
        "free",
        "busy",
        "hello",
        "hi ",
        "thank",
        "thanks",
        "bye",
        "greet",
        "small talk",
        "chitchat",
        "how are you",
        "what are you doing",
    )
    if sq in menu_words or any(w in sq for w in ("thank", "bye", "hello", "free", "busy")):
        return True
    if any(
        x in um
        for x in (
            "thank",
            "greet",
            "goodbye",
            "bye",
            "how are you",
            "what are you doing",
            "free",
            "busy",
            "small talk",
            "chitchat",
            "gratitude",
        )
    ):
        return True
    return False


def _should_invoke_chitchat_classifier(
    ai_route: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Run scope LLM when brain did not already lock chitchat — not keyword-gated."""
    from services.conversation_scope import (
        SCOPE_WELFOG,
        _has_definite_welfog_shopping_signal,
        _substantive_welfog_intent,
        turn_requests_catalog_menu,
    )

    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    if turn_requests_catalog_menu(
        original_msg, msg_en, ai_route=ai_route, conversation_context=conversation_context, allow_llm=False
    ):
        return False
    try:
        from services.account_list_semantics import (
            detect_account_list_followup_in_chat,
            message_requests_account_list_data,
        )

        if message_requests_account_list_data(
            original_msg, msg_en, conversation_context
        ) or detect_account_list_followup_in_chat(
            original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass
    if _has_definite_welfog_shopping_signal(comb):
        return False

    brain = _brain_scope_decision(ai_route)
    if brain:
        return False

    if isinstance(ai_route, dict):
        if _substantive_welfog_intent(ai_route):
            intent = (ai_route.get("intent") or "").strip().lower()
            channel = (ai_route.get("data_channel") or "").strip().lower()
            if intent == "product" and channel == "catalog":
                return _brain_misrouted_chitchat_as_product(ai_route)
            if channel in ("live_api", "catalog") and intent not in ("general", ""):
                return False
        if _brain_misrouted_chitchat_as_product(ai_route):
            return True
        scope = (ai_route.get("conversation_scope") or "").strip().lower()
        if scope == SCOPE_WELFOG and _substantive_welfog_intent(ai_route):
            return False

    return True


def _keyword_fallback_scope(original_msg: str, msg_en: str = "", conversation_context: str = ""):
    """LLM unavailable only — never primary path."""
    from services.conversation_scope import SCOPE_CHITCHAT, ScopeDecision

    try:
        from utils.helpers import (
            message_is_bot_availability_chitchat,
            message_is_casual_farewell_or_closing,
            message_is_user_feedback_or_closing,
            _is_light_smalltalk_fast,
            _is_short_pure_greeting,
        )

        comb = _combined(original_msg, msg_en)
        if (
            message_is_bot_availability_chitchat(comb)
            or message_is_user_feedback_or_closing(comb)
            or message_is_casual_farewell_or_closing(comb)
            or _is_short_pure_greeting(comb)
            or _is_light_smalltalk_fast(original_msg, msg_en)
        ):
            return ScopeDecision(
                scope=SCOPE_CHITCHAT,
                source="keyword_fallback",
                confidence=0.75,
            )
    except ImportError:
        pass
    return None


def resolve_chitchat_scope(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
):
    """
    Returns ScopeDecision — welfog_support when not chitchat/out_of_domain.
    """
    from services.conversation_scope import SCOPE_WELFOG, ScopeDecision

    if getattr(_IN_RESOLVE, "active", False):
        return ScopeDecision(scope=SCOPE_WELFOG, source="recursion_guard", confidence=0.0)

    comb = _combined(original_msg, msg_en)
    if not comb:
        return ScopeDecision(scope=SCOPE_WELFOG, source="empty", confidence=0.0)

    cache_key = (
        f"{hash(comb)}|{hash((conversation_context or '')[-200:])}|"
        f"{hash(str((ai_route or {}).get('intent')))}|{allow_llm}"
    )
    if getattr(_RESOLVE_CACHE, "key", None) == cache_key:
        cached = getattr(_RESOLVE_CACHE, "result", None)
        if cached is not None:
            return cached

    _IN_RESOLVE.active = True
    try:
        brain = _brain_scope_decision(ai_route)
        if brain:
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = brain
            return brain

        if allow_llm and _should_invoke_chitchat_classifier(
            ai_route, original_msg, msg_en, conversation_context
        ):
            from services.conversation_scope import ai_classify_scope_and_reply

            classified = ai_classify_scope_and_reply(
                original_msg, msg_en, conversation_context, reply_lang
            )
            if classified and classified.scope != SCOPE_WELFOG:
                log_reasoning(
                    f"Chitchat AI scope: {classified.scope} "
                    f"(conf={classified.confidence:.2f}) — "
                    f"{(classified.user_meaning or '')[:80]}"
                )
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = classified
                return classified

        if not allow_llm:
            fb = _keyword_fallback_scope(original_msg, msg_en, conversation_context)
            if fb:
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = fb
                return fb

        none = ScopeDecision(scope=SCOPE_WELFOG, source="not_chitchat", confidence=0.0)
        _RESOLVE_CACHE.key = cache_key
        _RESOLVE_CACHE.result = none
        return none
    finally:
        _IN_RESOLVE.active = False


def turn_is_chitchat_not_shopping(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    from services.conversation_scope import SCOPE_CHITCHAT

    dec = resolve_chitchat_scope(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    return dec.scope == SCOPE_CHITCHAT


def try_chitchat_routing_decision(
    route_data: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[tuple[Any, dict]]:
    """
    After ai_brain_route: lock chitchat with AI scope + reply in customer language.
    """
    from services.answer_router import AnswerRouteDecision
    from services.conversation_scope import (
        SCOPE_CHITCHAT,
        SCOPE_OUT,
        log_scope_routing_telemetry,
        turn_requests_catalog_menu,
    )

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None
    if turn_requests_catalog_menu(
        original_msg, msg_en, ai_route=route_data, conversation_context=conversation_context
    ):
        return None
    try:
        from services.product_catalog_resolver import (
            product_catalog_route_is_locked,
            turn_requests_product_catalog,
        )
        from utils.helpers import (
            _message_has_catalog_product_signal,
            _message_has_generic_shopping_item_signal,
            _text_has_product_shopping_intent,
        )

        if product_catalog_route_is_locked(route_data):
            return None
        if turn_requests_product_catalog(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=route_data,
            allow_llm=False,
        ):
            return None
        if (
            _message_has_catalog_product_signal(comb)
            or _message_has_generic_shopping_item_signal(comb)
            or _text_has_product_shopping_intent(comb)
        ):
            return None
    except ImportError:
        pass

    dec = resolve_chitchat_scope(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route_data,
        reply_lang=reply_lang,
        allow_llm=True,
    )
    if dec.scope not in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
        return None
    if dec.confidence < 0.52 and dec.source not in (
        "ai_route",
        "ai_route_intent",
        "ai_route_meta",
        "scope_llm",
    ):
        return None

    out = dict(route_data or {})
    out["conversation_scope"] = dec.scope
    out["user_meaning"] = dec.user_meaning or out.get("user_meaning") or ""
    if dec.reply:
        out["scope_reply"] = dec.reply
    out["data_channel"] = "none"
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out.pop("_product_catalog_locked", None)
    out.pop("route_handler", None)

    if dec.scope == SCOPE_CHITCHAT:
        out["intent"] = "general"
        out["meta_kind"] = "conversational"
        out["is_welfog_related"] = True
        handler = "warm_feedback"
        intent = "general"
        source = "chitchat_ai"
    elif dec.scope == "harm_sensitive":
        out["intent"] = "out_of_domain"
        out["is_welfog_related"] = False
        handler = "off_topic"
        intent = "out_of_domain"
        source = "reject"
    else:
        out["intent"] = "out_of_domain"
        out["is_welfog_related"] = False
        handler = "off_topic"
        intent = "out_of_domain"
        source = "reject"

    if not out.get("scope_reply"):
        try:
            from services.conversational_ack_flow import ai_chitchat_reply
            import re

            html = ai_chitchat_reply(
                original_msg, msg_en, conversation_context, reply_lang=reply_lang
            )
            if html:
                plain = re.sub(r"<[^>]+>", " ", html).strip()
                if plain:
                    out["scope_reply"] = plain
        except ImportError:
            pass

    log_scope_routing_telemetry(
        scope=dec.scope,
        route=handler,
        confidence=dec.confidence,
        source=dec.source,
    )
    return (
        AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            is_welfog_related=out.get("is_welfog_related", True),
            reason=f"Chitchat AI ({dec.scope}) — {dec.user_meaning[:80]}",
        ),
        out,
    )


def try_chitchat_ai_preflight(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[tuple[Any, dict]]:
    """
    Short casual turns — scope LLM before heavy brain route (AI-only, no keyword gate).
    """
    from services.conversation_scope import _has_definite_welfog_shopping_signal

    comb = _combined(original_msg, msg_en)
    if not comb or len(comb) > 180:
        return None
    if _has_definite_welfog_shopping_signal(comb):
        return None
    low = comb.lower()
    if re.search(
        r"\b(?:chahiye|chiye|dikha|dikhao|milega|milta|buy|need|want|order|track|refund|"
        r"pincode|delivery|wishlist|invoice|payment)\b",
        low,
    ):
        return None
    return try_chitchat_routing_decision(
        None, original_msg, msg_en, conversation_context, reply_lang
    )
