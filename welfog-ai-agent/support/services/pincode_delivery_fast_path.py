"""
Pincode delivery fast path — explicit PIN + serviceability intent → one live API call.

Skips chitchat preflight, brain route, and delivery micro-LLM stack when the user
already named a 6-digit PIN (or continues a pincode thread).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def _pincode_thread_nudge_without_digits(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> bool:
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb or re.search(r"\b[1-9]\d{5}\b", comb):
        return False
    tl = f" {comb.lower()} "
    in_thread = False
    try:
        from utils.helpers import _conversation_in_pincode_delivery_flow

        in_thread = _conversation_in_pincode_delivery_flow(conversation_context)
    except ImportError:
        pass
    if isinstance(ctx, dict) and (ctx.get("last") or "").strip().lower() == "pincode":
        in_thread = True
    if not in_thread:
        return False
    if re.search(r"\b(?:is|usi|wahi|same)\s+(?:pin|pincode|pincod)\b", tl):
        return True
    if any(
        x in tl
        for x in (
            "bta de", "bata de", "bta na", "bata na", "btao", "bta do", "bata do",
            "tu bta", "tum bta", "bta ", "bata ", "btao ", "check kr", "check kar",
        )
    ):
        return True
    if "pincode" in tl or "pincod" in tl or "pin code" in tl:
        return True
    return False


def turn_is_pincode_delivery_fast_path(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False

    try:
        from utils.helpers import (
            _conversation_in_pincode_delivery_flow,
            _naive_six_digit_pin_from_text,
            _text_is_pincode_serviceability_question,
            message_has_live_pincode_check_intent,
            turn_is_catalog_product_lookup,
        )
    except ImportError:
        return False

    if turn_is_catalog_product_lookup(original_msg, msg_en):
        return False

    if message_has_live_pincode_check_intent(
        original_msg, conversation_context, msg_en
    ):
        return True
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return True
    if _pincode_thread_nudge_without_digits(
        original_msg, msg_en, conversation_context, ctx
    ):
        return True
    if isinstance(ctx, dict) and (ctx.get("last") or "").strip().lower() == "pincode":
        if _naive_six_digit_pin_from_text(comb) or _conversation_in_pincode_delivery_flow(
            conversation_context
        ):
            return True
    if _conversation_in_pincode_delivery_flow(conversation_context):
        if _naive_six_digit_pin_from_text(comb):
            return True
        try:
            from services.location_delivery_resolver import turn_continues_pincode_area_check

            if turn_continues_pincode_area_check(comb, conversation_context):
                return True
        except ImportError:
            pass
    return False


def resolve_pin_for_fast_path(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> str:
    try:
        from utils.helpers import (
            _naive_six_digit_pin_from_text,
            extract_latest_pincode_from_user_conversation,
            resolve_pincode_for_check,
            should_reuse_pincode_from_conversation_history,
        )
    except ImportError:
        return ""

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    pin = resolve_pincode_for_check(
        original_msg,
        conversation_context,
        msg_en=msg_en,
        ai_route=None,
    )
    if pin:
        return pin

    pin = _naive_six_digit_pin_from_text(comb)
    if pin:
        return pin

    if _pincode_thread_nudge_without_digits(
        original_msg, msg_en, conversation_context, ctx
    ) or should_reuse_pincode_from_conversation_history(
        original_msg, msg_en, conversation_context
    ):
        latest = extract_latest_pincode_from_user_conversation(
            conversation_context, original_msg
        )
        if latest:
            return latest

    if isinstance(ctx, dict):
        stored = (ctx.get("data") or {}).get("last_pincode") or ctx.get("last_pincode")
        if stored and str(stored).strip():
            return str(stored).strip()

    return ""


def try_pincode_delivery_fast_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    reset_context_fn: Callable[[dict], None] | None = None,
) -> Optional[str]:
    if not turn_is_pincode_delivery_fast_path(
        original_msg, msg_en, conversation_context, ctx
    ):
        return None

    pin = resolve_pin_for_fast_path(original_msg, msg_en, conversation_context, ctx)
    if not pin:
        return None

    from services.pincode_delivery_flow import format_pincode_check_reply, validate_pincode_before_api
    from services.welfog_api import check_pincode_delivery

    lang = reply_lang or "en"
    ok, err_key, fmt = validate_pincode_before_api(pin, original_msg)
    if not ok:
        from services.pincode_delivery_flow import _pin_localized

        return _pin_localized(err_key, original_msg, lang, **fmt)

    log_reasoning(
        f"Pincode fast path: live API for PIN {pin} (skip routing/chitchat LLMs)."
    )
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        record_route_step("pincode_delivery_fast")
        record_route(intent="pincode_check", source="pincode_delivery_fast")
    except ImportError:
        pass

    api_res = check_pincode_delivery(pin)
    body = format_pincode_check_reply(pin, api_res, original_msg, lang)

    if isinstance(ctx, dict):
        if reset_context_fn:
            reset_context_fn(ctx)
        ctx["last"] = "pincode"
        ctx["awaiting"] = None
        ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
        ctx.setdefault("data", {})["last_pincode"] = pin

    return body


def try_pincode_delivery_fast_route(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: dict | None = None,
) -> Optional[tuple[Any, dict]]:
    if not turn_is_pincode_delivery_fast_path(
        original_msg, msg_en, conv_for_llm, ctx
    ):
        return None

    pin = resolve_pin_for_fast_path(original_msg, msg_en, conv_for_llm, ctx)
    if not pin:
        return None

    from services.answer_router import AnswerRouteDecision

    route_data = {
        "user_meaning": f"Check Welfog delivery/service for PIN {pin}",
        "reasoning": "Pincode fast path — direct live API for named PIN.",
        "intent": "pincode_check",
        "data_channel": "live_api",
        "needs_order_id": False,
        "run_catalog_search": False,
        "numeric_context": "pincode",
        "order_lookup_kind": "none",
        "extracted_pincode": pin,
        "_preflight_api": True,
        "_pincode_delivery_fast": True,
    }
    decision = AnswerRouteDecision(
        source="api",
        intent="pincode_check",
        handler="pincode_delivery_api",
        is_welfog_related=True,
        reason=f"Pincode delivery fast path — PIN {pin}",
    )
    log_reasoning(f"Pincode delivery route lock: PIN {pin}")
    return decision, route_data
