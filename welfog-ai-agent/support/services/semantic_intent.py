"""
AI-primary semantic intent — LLM understands meaning in any language/script.
Keyword helpers are fallback only when LLM is unavailable or for numeric safety (PIN vs Order ID).
"""
from __future__ import annotations

import os
from typing import Optional

from utils.reasoning_log import log_reasoning


def strict_ai_semantic_mode() -> bool:
    return (os.getenv("STRICT_AI_INTENT_ROUTING", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def llm_semantic_route_available(route: dict | None) -> bool:
    if not route or not isinstance(route, dict):
        return False
    if route.get("llm_unavailable"):
        return False
    return bool((route.get("intent") or "").strip())


def skip_keyword_intent_routes(ai_route: dict | None) -> bool:
    """When True, skip phrase-based order-history / placement / ctx.last shortcuts."""
    return strict_ai_semantic_mode() and llm_semantic_route_available(ai_route)


def ai_route_requests_refund_status_lookup(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Personal refund/return status for one order — Groq JSON first, any language."""
    from services.refund_status_semantics import ai_route_requests_refund_status_lookup as _lookup

    return _lookup(route, original_msg, msg_en, conversation_context)


def ai_route_requests_live_order_lookup(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """AI classified ONE order live lookup (track/refund/payment) — any language."""
    if not route:
        return False
    from utils.helpers import (
        _text_is_order_id_help_request,
        _text_is_refund_return_policy_howto,
        turn_is_catalog_product_lookup,
        user_turn_qualifies_for_live_order_api,
    )

    comb = f"{original_msg} {msg_en}".strip()
    from utils.helpers import _text_is_pincode_serviceability_question

    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if turn_is_catalog_product_lookup(original_msg, msg_en, route):
        return False
    if _text_is_refund_return_policy_howto(comb):
        return False
    if _text_is_order_id_help_request(comb):
        return False
    if ai_route_requests_refund_status_lookup(
        route, original_msg, msg_en, conversation_context
    ):
        return False
    from services.order_details_flow import message_wants_order_details_or_invoice

    if message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=route
    ):
        return False
    if not user_turn_qualifies_for_live_order_api(
        original_msg, msg_en, conversation_context, ai_route=route
    ):
        return False
    intent = (route.get("intent") or "").strip().lower()
    numeric = (route.get("numeric_context") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent == "order_history":
        return False
    if intent in ("order", "refund", "payment") and channel == "live_api":
        return True
    if intent in ("order", "refund", "payment") and (
        numeric == "order_id" or route.get("needs_order_id")
    ):
        from utils.helpers import extract_order_id

        if extract_order_id(original_msg, "") or extract_order_id(msg_en, ""):
            return True
        from utils.helpers import _text_is_refund_return_status_lookup

        if _text_is_refund_return_status_lookup(comb, conversation_context):
            return True
    return False


def ai_route_requests_order_history_list(route: dict | None) -> bool:
    if not route:
        return False
    return (
        (route.get("intent") or "").strip().lower() == "order_history"
        and (route.get("data_channel") or "").strip().lower() == "live_api"
    )


def should_skip_ctx_last_pinning(
    ai_route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Do not repeat wishlist/order-history from ctx.last when AI chose a specific new intent.
    Works across Tamil/Telugu/Hindi/Punjabi/etc. because AI reads meaning, not keywords.
    """
    if not llm_semantic_route_available(ai_route):
        return False
    if ai_route_requests_live_order_lookup(
        ai_route, original_msg, msg_en, conversation_context
    ):
        log_reasoning("Skip ctx.last pin — AI chose live single-order lookup.")
        return True
    if ai_route_requests_order_history_list(ai_route):
        log_reasoning("Skip ctx.last pin — AI chose order_history list (fresh API).")
        return True
    intent = (ai_route.get("intent") or "").strip().lower()
    if intent and intent not in ("general", ""):
        if not ai_route.get("continue_previous_topic"):
            log_reasoning(f"Skip ctx.last pin — AI intent={intent}, continue_topic=false.")
            return True
    return False


def ai_route_is_pincode_intent(route: dict | None) -> bool:
    """Groq classified delivery/serviceability — city name or PIN, any phrasing."""
    if not route:
        return False
    return (route.get("intent") or "").strip().lower() == "pincode_check"


def ai_route_requests_pincode_delivery(route: dict | None) -> bool:
    if not route:
        return False
    if ai_route_is_pincode_intent(route):
        return True
    return (
        (route.get("intent") or "").strip().lower() == "pincode_check"
        and (route.get("data_channel") or "").strip().lower() == "live_api"
    )


def resolve_pincode_serviceability_from_ai_or_heuristics(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """
    Delivery / service at PIN or city — AI route first; keyword heuristics only if LLM unavailable.
    """
    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            ai_route_requests_wishlist_howto,
        )

        if account_list_route_is_locked(ai_route) or ai_route_requests_wishlist_howto(
            ai_route, original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass

    if ai_route_requests_pincode_delivery(ai_route):
        log_reasoning(
            "[delivery-intent] serviceability_turn=true source=ai_route "
            "selected_source=pincode_pipeline"
        )
        return True
    from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

    if ai_meaning_describes_delivery_serviceability(ai_route):
        log_reasoning(
            "[delivery-intent] serviceability_turn=true source=ai_meaning "
            "selected_source=pincode_pipeline"
        )
        return True
    try:
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        if turn_requests_delivery_serviceability(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=True,
        ):
            log_reasoning(
                "[delivery-intent] serviceability_turn=true source=location_resolver "
                "selected_source=pincode_pipeline"
            )
            return True
    except ImportError:
        pass
    if skip_keyword_intent_routes(ai_route):
        return False
    from utils.helpers import _text_is_pincode_serviceability_question

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        log_reasoning(
            "[delivery-intent] serviceability_turn=true source=keyword_failsafe "
            "selected_source=pincode_pipeline"
        )
        return True
    return False


def resolve_live_lookup_from_ai_or_heuristics(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Prefer AI semantic classification; keyword heuristics only if LLM missing."""
    from utils.helpers import _text_is_refund_return_policy_howto

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if _text_is_refund_return_policy_howto(comb):
        return False
    if ai_route_requests_pincode_delivery(ai_route):
        return False
    if ai_route_requests_live_order_lookup(
        ai_route, original_msg, msg_en, conversation_context
    ):
        return True
    if skip_keyword_intent_routes(ai_route):
        return False
    from utils.helpers import message_needs_live_single_order_lookup_heuristic

    return message_needs_live_single_order_lookup_heuristic(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
