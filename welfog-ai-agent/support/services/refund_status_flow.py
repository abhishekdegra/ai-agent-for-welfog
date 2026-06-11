"""
Personal refund/return status — Groq intent → return-request API (ownership gated).
General refund policy/how-to stays on KB (not this module).
"""
from __future__ import annotations

from services.order_history_flow import OrderFlowResult, _localized_sysmsg
from utils.reasoning_log import log_reasoning


def _refund_status_error_reply(
    err: str,
    order_id: str,
    original_msg: str,
    reply_lang: str,
) -> str:
    lang = reply_lang or "en"
    if err == "login_required":
        return _localized_sysmsg("order_track_login_required", original_msg, reply_lang=lang) or (
            "Please log in to Welfog and open chat from your account to see refund status."
        )
    if err == "not_owned":
        return _localized_sysmsg("order_track_not_owned", original_msg, reply_lang=lang) or (
            "This Order ID does not match your account. Use the ID from your SMS/email or My Orders."
        )
    if err == "unverified":
        return _localized_sysmsg("order_track_unverified", original_msg, reply_lang=lang) or (
            "Could not verify this Order ID with your account. Try the account that placed the order."
        )
    hint = _localized_sysmsg("order_track_not_found", original_msg, reply_lang=lang, order_id=order_id)
    if hint:
        return hint
    return (
        f"We could not fetch refund status for Order ID <b>{order_id}</b>. "
        "Check My Orders or try again shortly."
    )


def _fetch_and_format_refund_status(
    order_id: str,
    user_id: str,
    original_msg: str,
    reply_lang: str,
    *,
    source: str = "refund_status_flow",
) -> str:
    from services.welfog_api import (
        fetch_welfog_return_request_for_user,
        format_refund_status_reply,
    )

    log_reasoning(
        f"Live refund status API for id={order_id} (user_id={user_id} ownership check)"
    )
    filtered, err = fetch_welfog_return_request_for_user(order_id, user_id)
    if filtered is not None:
        body = format_refund_status_reply(filtered, order_id, lang=reply_lang, filtered=filtered)
        if body:
            return body
    return _refund_status_error_reply(err or "not_found", order_id, original_msg, reply_lang)


def run_refund_status_ai_flow(
    original_msg: str,
    msg_en: str,
    user_id: str,
    conversation_context: str = "",
    reply_lang: str = "en",
    ai_route: dict | None = None,
) -> OrderFlowResult:
    from utils.helpers import (
        _text_is_refund_return_policy_howto,
        _user_announcing_will_provide_order_id,
        resolve_order_id_for_tracking,
    )

    rl = reply_lang or "en"
    comb = f"{original_msg or ''} {msg_en or ''}".strip()

    try:
        from services.refund_status_semantics import (
            KIND_PERSONAL_STATUS,
            KIND_POLICY_HOWTO,
            resolve_refund_turn,
        )

        resolved = resolve_refund_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            reply_lang=rl,
            allow_llm=True,
        )
        if resolved.kind == KIND_POLICY_HOWTO:
            log_reasoning("Refund status flow skipped — policy/how-to (KB path).")
            return OrderFlowResult(handled=False)
        if resolved.kind != KIND_PERSONAL_STATUS:
            if _text_is_refund_return_policy_howto(comb):
                log_reasoning("Refund status flow skipped — policy/how-to (KB path).")
                return OrderFlowResult(handled=False)
            return OrderFlowResult(handled=False)
    except ImportError:
        if _text_is_refund_return_policy_howto(comb):
            log_reasoning("Refund status flow skipped — policy/how-to (KB path).")
            return OrderFlowResult(handled=False)

    try:
        from services.order_tracking_semantics import ai_route_requests_order_tracking_lookup

        if ai_route_requests_order_tracking_lookup(
            ai_route, original_msg, msg_en, conversation_context
        ):
            from services.refund_status_semantics import current_turn_wants_personal_refund_status

            if not current_turn_wants_personal_refund_status(
                original_msg, msg_en, conversation_context, ai_route=ai_route, allow_llm=False
            ):
                log_reasoning("Refund status flow skipped — order tracking path.")
                return OrderFlowResult(handled=False)
    except ImportError:
        pass

    user_line = (original_msg or msg_en or "").strip()
    if _user_announcing_will_provide_order_id(user_line):
        body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="refund") or ""
        return OrderFlowResult(
            handled=True,
            reply_html=body,
            intent="refund",
            needs_order_id=True,
        )

    oid = resolve_order_id_for_tracking(user_line, conversation_context)
    if not oid:
        body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="refund") or ""
        log_reasoning("Refund status — need Order ID from customer.")
        return OrderFlowResult(
            handled=True,
            reply_html=body,
            intent="refund",
            needs_order_id=True,
        )

    html = _fetch_and_format_refund_status(
        oid,
        user_id,
        original_msg,
        rl,
        source="refund_status_ai_flow",
    )
    return OrderFlowResult(
        handled=True,
        reply_html=html,
        intent="refund",
        order_id=oid,
    )
