"""
Entity-first handlers: if the user message contains a PIN / Order ID + clear intent,
run the live API immediately — before warm feedback, KB dumps, or generic templates.
"""
from __future__ import annotations

from typing import Optional

from utils.reasoning_log import log_reasoning


def try_pincode_delivery_reply(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
    ctx: dict,
    ai_route: dict | None = None,
) -> str | None:
    """Live pincode API (city geocode or PIN), AI ask-PIN reply, or invalid-PIN message."""
    from services.location_delivery_resolver import turn_requests_delivery_serviceability
    from services.pincode_delivery_flow import run_delivery_location_check

    if not turn_requests_delivery_serviceability(
        original_msg,
        msg_en,
        conv_for_llm,
        ai_route=ai_route,
        allow_llm=True,
    ):
        return None

    result = run_delivery_location_check(
        original_msg,
        msg_en,
        conversation_context=conv_for_llm,
        reply_lang=lang,
        ai_route=ai_route,
    )
    if not result.handled or not result.reply_html:
        return None

    log_reasoning("Entity-first pincode delivery — location resolver + live API.")
    ctx["last"] = "pincode"
    ctx["awaiting"] = None
    ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
    return result.reply_html


def try_live_order_reply(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    ai_route: dict | None = None,
    *,
    reply_for_live_order_id_lookup,
    resolve_order_id_for_tracking,
    resolve_live_api_intent_from_conversation,
) -> str | None:
    """Live order track/refund/payment — injected callbacks avoid circular imports with chat_routes."""
    from utils.helpers import (
        extract_latest_order_id_from_user_conversation,
        extract_order_id,
        should_attempt_live_order_api_reply,
    )

    if not should_attempt_live_order_api_reply(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    ):
        return None

    from services.order_details_flow import message_wants_order_details_or_invoice
    from utils.helpers import (
        _message_is_order_id_followup_submission,
        user_turn_qualifies_for_live_order_api,
    )

    od_goal = message_wants_order_details_or_invoice(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    )
    if od_goal:
        log_reasoning(
            f"Skip live order early path — customer wants {od_goal} (details/invoice API)."
        )
        return None
    if not user_turn_qualifies_for_live_order_api(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    ):
        return None

    pending_oid = resolve_order_id_for_tracking(
        original_msg.strip() or msg_en.strip(),
        conv_for_llm,
        bot_awaiting_order_id=ctx.get("awaiting") == "order_id"
        or _message_is_order_id_followup_submission(original_msg, conv_for_llm),
    )
    if not pending_oid:
        pending_oid = extract_order_id(original_msg, conv_for_llm) or extract_order_id(
            msg_en, conv_for_llm
        )
    if not pending_oid:
        pending_oid = extract_latest_order_id_from_user_conversation(conv_for_llm, original_msg)
    if not pending_oid:
        return None

    live_intent = resolve_live_api_intent_from_conversation(
        conv_for_llm,
        ctx.get("last"),
        original_msg,
        msg_en,
        ai_route=ai_route if isinstance(ai_route, dict) else None,
    )
    log_reasoning(
        f"Entity-first live {live_intent} for Order ID {pending_oid}."
    )
    ctx["last"] = live_intent if live_intent in ("refund", "payment", "order") else "order"
    ctx["order_id"] = pending_oid
    ctx["awaiting"] = None
    return reply_for_live_order_id_lookup(
        live_intent, pending_oid, user_id, original_msg, lang
    )
