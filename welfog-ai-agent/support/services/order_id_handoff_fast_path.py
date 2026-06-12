"""
Order-ID handoff fast path — bot asked for Order ID, user pasted it (or says track now).

Skips ai_brain_route and specialist LLMs; one live API call for that single order only.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def _bot_awaiting_order_id(
    conversation_context: str,
    ctx: dict | None,
) -> bool:
    if isinstance(ctx, dict) and ctx.get("awaiting") == "order_id":
        return True
    try:
        from utils.helpers import _conversation_awaiting_order_id

        return bool(_conversation_awaiting_order_id(conversation_context))
    except ImportError:
        return False


def _is_order_id_handoff_turn(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> bool:
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False

    try:
        from utils.helpers import (
            _conversation_bot_asked_for_pincode,
            _message_is_order_id_followup_submission,
            _text_is_pincode_serviceability_question,
            turn_is_catalog_product_lookup,
        )
    except ImportError:
        return False

    if turn_is_catalog_product_lookup(original_msg, msg_en):
        return False
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False

    if _bot_awaiting_order_id(conversation_context, ctx):
        return True
    if _message_is_order_id_followup_submission(original_msg, conversation_context):
        return True

    try:
        from services.conversation_thread_semantics import (
            message_needs_thread_continuation,
            resolve_explicit_turn_goal_from_message,
        )

        if message_needs_thread_continuation(original_msg, conversation_context):
            return True
        if _bot_awaiting_order_id(conversation_context, ctx):
            explicit = resolve_explicit_turn_goal_from_message(
                original_msg, msg_en, conversation_context, None, allow_llm=False
            )
            if explicit in (
                "track",
                "refund_status",
                "payment",
                "order_invoice",
                "order_details",
            ):
                return True
    except ImportError:
        pass

    return False


def _resolve_handoff_order_id(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> str:
    try:
        from utils.helpers import (
            _message_is_order_id_followup_submission,
            resolve_order_id_for_tracking,
        )
    except ImportError:
        return ""

    bot_awaiting = _bot_awaiting_order_id(conversation_context, ctx) or _message_is_order_id_followup_submission(
        original_msg, conversation_context
    )
    oid = resolve_order_id_for_tracking(
        original_msg.strip() or msg_en.strip(),
        conversation_context,
        bot_awaiting_order_id=bot_awaiting,
    )
    return (oid or "").strip()


def _resolve_handoff_thread_goal(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> str:
    try:
        from services.conversation_thread_semantics import (
            infer_order_thread_goal,
            resolve_explicit_turn_goal_from_message,
        )
    except ImportError:
        infer_order_thread_goal = None  # type: ignore
        resolve_explicit_turn_goal_from_message = None  # type: ignore

    ctx_last = ctx.get("last") if isinstance(ctx, dict) else None

    if resolve_explicit_turn_goal_from_message:
        explicit = resolve_explicit_turn_goal_from_message(
            original_msg,
            msg_en,
            conversation_context,
            None,
            allow_llm=False,
        )
        if explicit:
            return explicit

    if infer_order_thread_goal:
        thread = infer_order_thread_goal(
            conversation_context,
            f"{original_msg} {msg_en}".strip(),
            ctx_last=ctx_last,
            ai_route=None,
            allow_llm=False,
        )
        if thread:
            return thread

    cl = (ctx_last or "").strip().lower()
    if cl == "refund":
        return "refund_status"
    if cl == "payment":
        return "payment"
    return "track"


def _fetch_details_handoff_reply(
    goal: str,
    order_id: str,
    user_id: str,
    original_msg: str,
    reply_lang: str,
) -> str:
    from services.order_details_flow import (
        _details_error_reply,
        detect_order_details_focus,
        fields_included_for_focus,
        format_order_invoice_reply_html,
        log_invoice_flow,
        log_order_data_pipeline,
    )
    from services.welfog_api import (
        fetch_purchase_history_details_for_user,
        format_order_details_reply,
    )

    rl = reply_lang or "en"
    action = "order_invoice" if goal == "order_invoice" else "order_details"
    api_name = "purchase-history-details"
    log_reasoning(f"Order-ID handoff fast path: {action} for id={order_id}")

    if action == "order_invoice":
        log_invoice_flow(
            intent="order_invoice",
            order_id=order_id,
            selected_flow="order_details_api",
            invoice_status="ownership_check",
        )

    row, err = fetch_purchase_history_details_for_user(order_id, user_id)
    if err or not row:
        if action == "order_invoice":
            log_invoice_flow(
                intent="order_invoice",
                order_id=order_id,
                selected_flow="order_details_api",
                invoice_status=err or "not_found",
            )
        return _details_error_reply(err or "not_found", order_id, original_msg, rl)

    if action == "order_invoice":
        body = format_order_invoice_reply_html(order_id, lang=rl)
        log_invoice_flow(
            intent="order_invoice",
            order_id=order_id,
            selected_flow="order_details_api",
            invoice_status="ready",
        )
        log_order_data_pipeline(
            action=action,
            source="live_api",
            api=f"{api_name}+invoice_url",
            focus="invoice",
            order_id=order_id,
            fields=fields_included_for_focus("order_invoice"),
        )
        return body

    focus = detect_order_details_focus(original_msg, action=action, ai_focus="")
    body = format_order_details_reply(row, order_id, focus=focus, lang=rl)
    log_order_data_pipeline(
        action=action,
        source="live_api",
        api=api_name,
        focus=focus,
        order_id=order_id,
        fields=fields_included_for_focus(focus),
    )
    return body


def try_order_id_handoff_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    user_id: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    reply_for_live_order_id_lookup: Callable[..., str],
    reset_context_fn: Callable[[dict], None] | None = None,
) -> Optional[str]:
    """
    Direct live API for a single handed-over Order ID — zero routing LLM calls.
    """
    if not _is_order_id_handoff_turn(original_msg, msg_en, conversation_context, ctx):
        return None

    order_id = _resolve_handoff_order_id(
        original_msg, msg_en, conversation_context, ctx
    )
    if not order_id:
        return None

    goal = _resolve_handoff_thread_goal(
        original_msg, msg_en, conversation_context, ctx
    )
    lang = reply_lang or "en"

    log_reasoning(
        f"Order-ID handoff fast path: goal={goal} id={order_id} "
        "(skip brain route — single live API)."
    )
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        record_route_step("order_id_handoff_fast")
        record_route(intent=goal, source="order_id_handoff_fast")
    except ImportError:
        pass

    if isinstance(ctx, dict):
        if reset_context_fn:
            reset_context_fn(ctx)
        ctx["order_id"] = order_id
        ctx["awaiting"] = None
        if goal == "refund_status":
            ctx["last"] = "refund"
        elif goal == "payment":
            ctx["last"] = "payment"
        else:
            ctx["last"] = "order"

    if goal in ("order_invoice", "order_details"):
        return _fetch_details_handoff_reply(
            goal, order_id, user_id, original_msg, lang
        )

    live_intent = "refund" if goal == "refund_status" else goal
    if live_intent not in ("refund", "payment", "order"):
        live_intent = "order"

    return reply_for_live_order_id_lookup(
        live_intent, order_id, user_id, original_msg, lang
    )


def try_order_id_handoff_route(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: dict | None = None,
) -> Optional[tuple[Any, dict]]:
    """Router-level fast path — lock route without brain LLM when handoff applies."""
    if not _is_order_id_handoff_turn(original_msg, msg_en, conv_for_llm, ctx):
        return None

    order_id = _resolve_handoff_order_id(original_msg, msg_en, conv_for_llm, ctx)
    if not order_id:
        return None

    from services.answer_router import AnswerRouteDecision

    goal = _resolve_handoff_thread_goal(original_msg, msg_en, conv_for_llm, ctx)
    handler = "order_tracking_api"
    intent = "order"
    olk = "track"
    if goal == "refund_status":
        handler = "refund_status_api"
        intent = "refund"
        olk = "refund_status"
    elif goal == "payment":
        handler = "payment_api"
        intent = "payment"
        olk = "payment"
    elif goal == "order_invoice":
        handler = "order_details_api"
        intent = "order"
        olk = "invoice"
    elif goal == "order_details":
        handler = "order_details_api"
        intent = "order"
        olk = "details"

    route_data = {
        "user_meaning": f"Continue {goal} for order {order_id}",
        "reasoning": "Order-ID handoff — direct live API for submitted id.",
        "intent": intent,
        "data_channel": "live_api",
        "needs_order_id": False,
        "run_catalog_search": False,
        "numeric_context": "order_id",
        "order_lookup_kind": olk,
        "extracted_order_id": order_id,
        "_preflight_api": True,
        "_order_id_handoff": True,
    }
    decision = AnswerRouteDecision(
        source="api",
        intent=intent,
        handler=handler,
        is_welfog_related=True,
        reason=f"Order-ID handoff fast path ({goal})",
    )
    log_reasoning(f"Order-ID handoff route lock: {handler} id={order_id}")
    return decision, route_data
