"""
Industry-level intent execution: Groq/AI detects meaning → run matching handler reply.
Generic shortcuts (warm thanks, policy KB dump) cannot override a substantive detected intent.
"""
from __future__ import annotations

from typing import Any, Optional

from services.answer_router import AnswerRouteDecision, dispatch_early_answer
from services.entity_first_handlers import try_pincode_delivery_reply
from utils.reasoning_log import log_reasoning

SUBSTANTIVE_INTENTS = frozenset(
    {
        "pincode_check",
        "order",
        "order_history",
        "refund",
        "payment",
        "product",
        "wishlist",
        "seller",
        "deals",
        "categories",
        "category_feed",
    }
)

API_HANDLERS = frozenset(
    {
        "pincode_delivery_api",
        "order_details_api",
        "order_tracking_api",
        "refund_status_api",
        "order_ai_flow",
        "product_ai_flow",
        "wishlist_api",
        "deals_api",
        "categories_api",
        "category_feed_api",
        "catalog_pro_id",
        "order_id_ai_flow",
    }
)

STRUCTURED_KB_HANDLERS = frozenset(
    {
        "order_placement_kb",
        "order_history_howto_kb",
        "wishlist_howto_kb",
        "order_tracking_howto_kb",
        "order_id_help_kb",
        "assistant_capability_kb",
        "assistant_intro",
        "policy_structured_kb",
        "bot_latency_apology",
        "bot_topic_correction",
        "bot_insult_calm",
        "bot_search_behavior_help",
    }
)


def substantive_detected_intent(
    route_decision: AnswerRouteDecision | None,
    ai_route: dict | None,
) -> str:
    if route_decision:
        intent = (route_decision.intent or "").strip().lower()
        if intent in SUBSTANTIVE_INTENTS:
            return intent
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        if intent in SUBSTANTIVE_INTENTS:
            return intent
    return ""


def ai_route_blocks_generic_shortcuts(
    route_decision: AnswerRouteDecision | None,
    ai_route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    When True, block Direct KB / warm-feedback / dynamic_kb hijack for this turn.
    """
    intent = substantive_detected_intent(route_decision, ai_route)
    if intent:
        return True
    if route_decision and (route_decision.handler or "") in API_HANDLERS:
        return True
    if isinstance(ai_route, dict):
        ch = (ai_route.get("data_channel") or "").strip().lower()
        if ch in ("live_api", "catalog", "ai_order", "kb_ai"):
            return True
    from utils.helpers import (
        extract_embedded_query_identifiers,
        message_has_live_pincode_check_intent,
    )

    if message_has_live_pincode_check_intent(
        original_msg, conversation_context, msg_en
    ):
        return True
    ids = extract_embedded_query_identifiers(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if (ids.get("pincode") or "").strip() or (ids.get("order_id") or "").strip():
        return True
    return False


def execute_detected_intent_reply(
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    try_live_order_callback=None,
) -> Optional[str]:
    """
    Run handler for AI-detected intent. Returns reply HTML/text or None.
    Order: entity-first PIN → entity-first order → dispatch_early_answer handlers.
    """
    intent = substantive_detected_intent(route_decision, ai_route)
    handler = (route_decision.handler or "").strip()

    from services.semantic_intent import (
        ai_route_requests_live_order_lookup,
        ai_route_requests_pincode_delivery,
        resolve_pincode_serviceability_from_ai_or_heuristics,
    )

    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            ai_route_requests_wishlist_howto,
        )

        if account_list_route_is_locked(ai_route) or ai_route_requests_wishlist_howto(
            ai_route, original_msg, msg_en, conv_for_llm
        ):
            return None
    except ImportError:
        pass

    serviceability_turn = resolve_pincode_serviceability_from_ai_or_heuristics(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    )

    order_first = (
        not serviceability_turn
        and (
            handler in ("order_tracking_api", "order_details_api")
            or intent in ("order", "refund", "payment")
            or ai_route_requests_live_order_lookup(
                ai_route, original_msg, msg_en, conv_for_llm
            )
        )
    )
    pin_first = serviceability_turn or ai_route_requests_pincode_delivery(ai_route) or (
        intent == "pincode_check" or handler == "pincode_delivery_api"
    )

    if pin_first:
        pin_reply = try_pincode_delivery_reply(
            original_msg, msg_en, conv_for_llm, lang, ctx, ai_route=ai_route
        )
        if pin_reply:
            log_reasoning(
                f"Intent executor: pincode first (serviceability={serviceability_turn}, "
                f"intent={intent}, handler={handler})."
            )
            return pin_reply

    if order_first and not pin_first:
        if handler != "order_details_api" and try_live_order_callback:
            live_reply = try_live_order_callback()
            if live_reply:
                log_reasoning(
                    f"Intent executor: live order reply (intent={intent}, handler={handler})."
                )
                return live_reply
        if handler == "order_details_api":
            early_details = dispatch_early_answer(
                route_decision,
                original_msg,
                msg_en,
                reply_lang=lang,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                ai_route=ai_route,
            )
            if early_details:
                log_reasoning(
                    f"Intent executor: order details/invoice handler={handler} intent={intent}."
                )
                return early_details
        early_order = dispatch_early_answer(
            route_decision,
            original_msg,
            msg_en,
            reply_lang=lang,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            ai_route=ai_route,
        )
        if early_order:
            log_reasoning(
                f"Intent executor: dispatch order handler={route_decision.handler} "
                f"intent={route_decision.intent}."
            )
            return early_order

    if try_live_order_callback:
        live_reply = try_live_order_callback()
        if live_reply:
            log_reasoning(
                f"Intent executor: live order reply (intent={intent}, handler={handler})."
            )
            return live_reply

    if handler == "deals_api" or intent == "deals":
        from services.catalog_menu_replies import build_today_deals_reply_html

        body = build_today_deals_reply_html(original_msg, reply_lang=lang)
        if body:
            log_reasoning("Intent executor: today deals API cards.")
            return body

    if handler == "categories_api" or intent == "categories":
        from services.welfog_api import resolve_category_product_browse_route, ensure_expanded_categories_map_for_ctx
        from services.product_search_flow import run_product_search_ai_flow

        ensure_expanded_categories_map_for_ctx(ctx)
        cat_browse = resolve_category_product_browse_route(f"{original_msg} {msg_en}", ctx=ctx)
        if cat_browse:
            cid, sq = cat_browse
            ctx.setdefault("data", {})["selected_category_id"] = cid
            ps = run_product_search_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
                search_query=sq,
                ai_route=ai_route,
            )
            if ps.handled and ps.reply_html:
                log_reasoning(f"Intent executor: category-filtered products (category_id={cid}).")
                return ps.reply_html

        from services.catalog_menu_replies import build_categories_list_reply_html

        body = build_categories_list_reply_html(ctx, original_msg, reply_lang=lang)
        if body:
            log_reasoning("Intent executor: categories list API.")
            return body

    if (
        intent
        or handler in API_HANDLERS
        or handler in STRUCTURED_KB_HANDLERS
        or ai_route_blocks_generic_shortcuts(
            route_decision, ai_route, original_msg, msg_en, conv_for_llm
        )
    ):
        early = dispatch_early_answer(
            route_decision,
            original_msg,
            msg_en,
            reply_lang=lang,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            ai_route=ai_route,
        )
        if early:
            log_reasoning(
                f"Intent executor: dispatch_early_answer handler={route_decision.handler} "
                f"intent={route_decision.intent}."
            )
            return early

        if intent == "pincode_check" or handler == "pincode_delivery_api":
            pin_reply = try_pincode_delivery_reply(
                original_msg, msg_en, conv_for_llm, lang, ctx, ai_route=ai_route
            )
            if pin_reply:
                log_reasoning("Intent executor: pincode retry after dispatch miss.")
                return pin_reply

    return None
