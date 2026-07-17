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
    o = (original_msg or "").strip()
    e = (msg_en or "").strip()
    if not o:
        return e
    if not e or e.lower() == o.lower():
        return o
    return f"{o} {e}".strip()


def _chitchat_lane_skips_transactional_guards(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Greetings / casual openers — skip KB-turn, refund, pincode, thread LLM stack.
    Uses semantic helpers only (not phrase-list routing).
    """
    try:
        from utils.helpers import (
            _is_light_smalltalk_fast,
            _is_short_pure_greeting,
            _looks_like_greeting_message,
            message_is_bot_availability_chitchat,
            message_is_casual_farewell_or_closing,
            message_is_user_feedback_or_closing,
        )

        comb = _combined(original_msg, msg_en)
        if not comb:
            return False
        # Ultra-light openers first — avoid deep helper graphs on "hi" / "thanks".
        if _is_short_pure_greeting(comb) or _is_light_smalltalk_fast(original_msg, msg_en):
            return True
        if message_is_bot_availability_chitchat(comb):
            return True
        if message_is_user_feedback_or_closing(comb) or message_is_casual_farewell_or_closing(comb):
            return True
        if _looks_like_greeting_message(original_msg or comb):
            return True
        return False
    except ImportError:
        pass
    return False


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
    try:
        from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path

        if turn_is_pincode_delivery_fast_path(
            original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass
    try:
        from services.refund_intent_fast_path import turn_has_refund_topic

        if turn_has_refund_topic(original_msg, msg_en):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _text_is_pincode_serviceability_question,
            message_has_live_pincode_check_intent,
        )

        if message_has_live_pincode_check_intent(
            original_msg, conversation_context, msg_en
        ):
            return False
        if _text_is_pincode_serviceability_question(comb, conversation_context):
            return False
    except ImportError:
        pass
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


def _try_chitchat_scope_only_routing(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[tuple[Any, dict]]:
    """One scope LLM — chitchat / out_of_domain / welfog_support (no KB-turn / thread stack)."""
    from services.answer_router import AnswerRouteDecision
    from services.conversation_scope import (
        SCOPE_CHITCHAT,
        SCOPE_OUT,
        SCOPE_WELFOG,
        ai_classify_scope_and_reply,
        log_scope_routing_telemetry,
    )

    dec = ai_classify_scope_and_reply(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if not dec:
        return None
    if dec.scope == SCOPE_WELFOG:
        return None

    out: dict = {
        "user_meaning": dec.user_meaning or "",
        "reasoning": f"Conversational lane — {dec.scope}",
        "intent": "general" if dec.scope == SCOPE_CHITCHAT else "out_of_domain",
        "data_channel": "none",
        "meta_kind": "conversational",
        "conversation_scope": dec.scope,
        "is_welfog_related": dec.scope == SCOPE_CHITCHAT,
        "needs_order_id": False,
        "run_catalog_search": False,
        "numeric_context": "none",
    }
    if dec.reply:
        out["scope_reply"] = dec.reply

    if dec.scope == SCOPE_CHITCHAT:
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

    log_scope_routing_telemetry(
        scope=dec.scope,
        route=handler,
        confidence=dec.confidence,
        source=dec.source or "scope_llm_fast_lane",
    )
    return (
        AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            is_welfog_related=out.get("is_welfog_related", True),
            reason=f"Conversational fast lane ({dec.scope}) — {(dec.user_meaning or '')[:80]}",
        ),
        out,
    )


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
    if _chitchat_lane_skips_transactional_guards(
        original_msg, msg_en, conversation_context
    ):
        return _try_chitchat_scope_only_routing(
            original_msg, msg_en, conversation_context, reply_lang
        )
    try:
        from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path

        if turn_is_pincode_delivery_fast_path(
            original_msg, msg_en, conversation_context
        ):
            return None
    except ImportError:
        pass
    try:
        from services.refund_intent_fast_path import turn_has_refund_topic

        if turn_has_refund_topic(original_msg, msg_en):
            return None
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _text_is_pincode_serviceability_question,
            message_has_live_pincode_check_intent,
        )

        if message_has_live_pincode_check_intent(
            original_msg, conversation_context, msg_en
        ):
            return None
        if _text_is_pincode_serviceability_question(comb, conversation_context):
            return None
    except ImportError:
        pass
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
    Deprecated pre-brain path — micro-LLM stack removed; brain route classifies once.
    """
    try:
        from services.chat_flow_telemetry import should_defer_micro_classifiers_to_brain

        if should_defer_micro_classifiers_to_brain():
            return None
    except ImportError:
        pass
    return None


def _is_instant_greeting_thanks_lane(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Ultra-short hello / thanks / bye — served by fast AI chitchat (user language).
    Anything longer or substantive goes to ai_brain_route (no keyword shopping lists).
    """
    comb = _combined(original_msg, msg_en)
    if not comb or len(comb) > 48:
        return False
    try:
        from utils.helpers import _native_script_message_looks_informational

        if _native_script_message_looks_informational(original_msg or comb):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _is_short_pure_greeting,
            _looks_like_native_script_short_greeting,
            _looks_like_greeting_message,
            message_is_bot_availability_chitchat,
            message_is_casual_farewell_or_closing,
            message_is_user_feedback_or_closing,
        )

        if _is_short_pure_greeting(original_msg or comb):
            return True
        if _looks_like_native_script_short_greeting(original_msg):
            return True
        if _looks_like_greeting_message(original_msg or comb):
            return True
        if message_is_user_feedback_or_closing(comb):
            return True
        if message_is_casual_farewell_or_closing(comb):
            return True
        if message_is_bot_availability_chitchat(comb):
            return True
    except ImportError:
        pass
    return False


def try_conversational_turn_fast_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[tuple[str, dict]]:
    """
    Deprecated — all turns use ai_brain_route + AI scope/chitchat replies (no templates).
    """
    return None


def try_scope_ai_early_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    preflight: bool = False,
) -> Optional[tuple[str, dict]]:
    """
    One scope LLM for chitchat / out-of-domain before universal brain — natural reply
    in the customer's language and style (not static templates).
    """
    from services.translation_service import resolve_customer_reply_lang

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    if preflight:
        return None

    # Early social scope path skips heavy KB/orderguards (those load embeds +
    # order-intent LLM and can burn 60s+ before the actual scope LLM).
    # Shopping/order structural turns are filtered by the chat_routes caller.

    if not preflight:
        try:
            from services.chat_flow_telemetry import get_cached_brain_route

            brain = get_cached_brain_route()
            if isinstance(brain, dict):
                ch = (brain.get("data_channel") or "").strip().lower()
                if ch in ("live_api", "catalog", "kb"):
                    return None
        except ImportError:
            pass

    if not preflight:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                return None
        except ImportError:
            pass

    from services.conversation_scope import (
        SCOPE_CHITCHAT,
        SCOPE_OUT,
        SCOPE_WELFOG,
        ai_classify_scope_and_reply,
        build_scope_reply,
        finalize_scope_reply_html,
    )

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    dec = ai_classify_scope_and_reply(
        original_msg, msg_en, conversation_context, rl, preflight=preflight
    )
    if not dec or dec.scope == SCOPE_WELFOG:
        return None
    if dec.scope not in (SCOPE_CHITCHAT, SCOPE_OUT):
        return None
    # Without RECENT chat, OOD mislabels shopping typos/corrections ("fame" for frame).
    # Require high confidence so ambiguous turns fall through to Brain product routing.
    if dec.scope == SCOPE_OUT:
        has_ctx = bool((conversation_context or "").strip())
        min_ood = 0.92 if not has_ctx else 0.72
        if dec.confidence < min_ood:
            log_reasoning(
                f"Early scope OOD deferred to Brain "
                f"(conf={dec.confidence:.2f} < {min_ood:.2f}, ctx={'yes' if has_ctx else 'no'})."
            )
            return None
    if dec.confidence < 0.48 and not (dec.reply or "").strip():
        return None

    # Prefer scope LLM authored reply — never call a second chitchat LLM here.
    reply_raw = (dec.reply or "").strip()
    if not reply_raw:
        body = build_scope_reply(
            dec, original_msg, msg_en, rl, prefer_llm=False
        )
    else:
        body = finalize_scope_reply_html(reply_raw, original_msg, reply_lang=rl)
    if not body:
        return None

    is_chitchat = dec.scope == SCOPE_CHITCHAT
    route_data = {
        "intent": "general" if is_chitchat else "out_of_domain",
        "conversation_scope": dec.scope,
        "data_channel": "none",
        "meta_kind": "conversational",
        "is_welfog_related": is_chitchat,
        "run_catalog_search": False,
        "needs_order_id": False,
        "user_meaning": (dec.user_meaning or "")[:200],
        "route_handler": "warm_feedback" if is_chitchat else "off_topic",
        "scope_reply": reply_raw[:500],
    }
    log_reasoning(
        f"Early scope AI ({dec.scope}) — natural reply before brain."
    )
    return body, route_data
