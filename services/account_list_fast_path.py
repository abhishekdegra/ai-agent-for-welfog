"""
Order history list + wishlist — AI-first (one micro-LLM per turn, cached).

Any language / any phrasing → account_list_kind + user_meaning → one API or how-to KB.
Keyword helpers: LLM-unavailable failsafe only (see turn_intent_coordinator).
"""
from __future__ import annotations

from typing import Callable, Optional

from utils.reasoning_log import log_reasoning


def try_account_list_fast_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    user_id: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    format_purchase_history_reply: Callable[..., str],
    format_wishlist_reply: Callable[..., str],
    localized_sysmsg: Callable[..., str],
    sysmsg: Callable[[str], str],
    reset_context_fn=None,
) -> Optional[str]:
    from services.turn_intent_coordinator import resolve_account_list_action_ai_first

    action = resolve_account_list_action_ai_first(
        original_msg, msg_en, conversation_context, reply_lang, ctx,
        ai_route=(ctx or {}).get("data", {}).get("ai_route") if isinstance(ctx, dict) else None,
    )
    kind = (action.get("kind") or "").strip()
    act = (action.get("action") or "").strip()
    if not kind or act == "none":
        return None

    from services.account_list_semantics import (
        ACTION_WANTS_DATA,
        ACTION_WANTS_STEPS,
        KIND_PURCHASE_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_WISHLIST_IN_CHAT,
    )

    lang = reply_lang or "en"
    log_reasoning(
        f"Account-list fast path: kind={kind} action={act} "
        f"source={action.get('source') or '-'}"
    )
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        topic = (action.get("topic") or kind).strip().lower()
        record_route_step("account_list_fast")
        record_route(intent=topic, source=f"account_list_fast_{kind}")
    except ImportError:
        pass

    if isinstance(ctx, dict):
        if reset_context_fn:
            reset_context_fn(ctx)
        if kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
            ctx["last"] = "wishlist"
            ctx.setdefault("data", {})["topic_mode"] = (
                "wishlist_howto" if kind == KIND_WISHLIST_HOWTO else "wishlist_list"
            )
        elif kind in (KIND_PURCHASE_IN_CHAT, KIND_PURCHASE_HOWTO):
            ctx["last"] = "order_history"
            ctx.setdefault("data", {})["topic_mode"] = (
                "order_history_howto" if kind == KIND_PURCHASE_HOWTO else "order_history_list"
            )
        ctx["awaiting"] = None
        ctx["order_id"] = None

    if kind == KIND_WISHLIST_IN_CHAT and act == ACTION_WANTS_DATA:
        return format_wishlist_reply(user_id, page=1, append_only=False)
    if kind == KIND_WISHLIST_HOWTO and act == ACTION_WANTS_STEPS:
        return (
            localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
            or sysmsg("wishlist_help")
            or None
        )
    if kind == KIND_PURCHASE_IN_CHAT and act == ACTION_WANTS_DATA:
        return format_purchase_history_reply(user_id, page=1, append_only=False)
    if kind == KIND_PURCHASE_HOWTO and act == ACTION_WANTS_STEPS:
        return (
            localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
            or sysmsg("order_history_help")
            or None
        )
    return None


def try_account_list_fast_route(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> Optional[tuple]:
    from services.turn_intent_coordinator import resolve_account_list_action_ai_first

    action = resolve_account_list_action_ai_first(
        original_msg, msg_en, conv_for_llm, reply_lang, ctx,
        ai_route=(ctx or {}).get("data", {}).get("ai_route") if isinstance(ctx, dict) else None,
    )
    kind = (action.get("kind") or "").strip()
    act = (action.get("action") or "").strip()
    if not kind or act == "none":
        return None

    from services.account_list_semantics import (
        ACTION_WANTS_DATA,
        ACTION_WANTS_STEPS,
        KIND_PURCHASE_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_WISHLIST_IN_CHAT,
        _apply_kind_to_route,
    )
    from services.answer_router import AnswerRouteDecision

    um = (action.get("user_meaning") or "").strip() or f"Account list — {kind}"
    route_data = _apply_kind_to_route(
        {
            "user_meaning": um,
            "reasoning": "Account-list AI-first — one classification per turn.",
        },
        kind,
        action.get("source") or "account_list_fast",
    )
    route_data["_account_list_fast"] = True

    if kind == KIND_WISHLIST_IN_CHAT and act == ACTION_WANTS_DATA:
        handler, intent, source = "wishlist_api", "wishlist", "api"
    elif kind == KIND_WISHLIST_HOWTO and act == ACTION_WANTS_STEPS:
        handler, intent, source = "wishlist_howto_kb", "general", "kb"
    elif kind == KIND_PURCHASE_IN_CHAT and act == ACTION_WANTS_DATA:
        handler, intent, source = "order_ai_flow", "order_history", "api"
    elif kind == KIND_PURCHASE_HOWTO and act == ACTION_WANTS_STEPS:
        handler, intent, source = "order_history_howto_kb", "general", "kb"
    else:
        return None

    decision = AnswerRouteDecision(
        source=source,
        intent=intent,
        handler=handler,
        is_welfog_related=True,
        reason=f"Account-list AI-first — {kind}",
        kb_keys=["faqs", "welfog_api_wishlist"] if "wishlist_howto" in handler else None,
    )
    return decision, route_data
