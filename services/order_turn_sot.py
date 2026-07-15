"""
Order / refund / tracking turn Single Source of Truth (SoT).

Locks personal live-API turns so product catalog rescue cannot steal
order-id handoffs or delivery-status asks.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def order_session_blocks_product_rescue(
    ctx: dict | None,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """Session lock or order-id follow-up — never product rescue."""
    try:
        from utils.helpers import ctx_has_order_thread_lock, extract_order_id

        if ctx_has_order_thread_lock(ctx):
            return True
    except ImportError:
        if isinstance(ctx, dict) and ctx.get("awaiting") == "order_id":
            return True

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb or not re.search(r"\b[0-9]{4,20}\b", comb):
        return False

    if isinstance(ctx, dict):
        last = (ctx.get("last") or "").strip().lower()
        if last in ("refund", "order", "invoice", "payment", "track"):
            return True
        pending = (
            (ctx.get("data") or {}).get("pending_action") or ""
        ).strip().lower()
        if pending in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            return True

    try:
        from utils.helpers import extract_order_id

        if extract_order_id(comb, "") and isinstance(ctx, dict):
            ai_route = (ctx.get("data") or {}).get("ai_route") or {}
            if isinstance(ai_route, dict) and ai_route.get("needs_order_id"):
                return True
    except ImportError:
        pass
    return False


def brain_route_blocks_product_rescue(
    route: dict | None,
    *,
    ctx: dict | None = None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    if order_session_blocks_product_rescue(ctx, original_msg, msg_en):
        return True
    if not isinstance(route, dict):
        return False
    try:
        from services.brain_direct_dispatch import _brain_route_is_personal_order_live

        if _brain_route_is_personal_order_live(route):
            return True
    except ImportError:
        pass
    if route.get("needs_order_id"):
        return True
    channel = (route.get("data_channel") or "").strip().lower()
    intent = (route.get("intent") or "").strip().lower()
    if channel == "live_api" and intent in ("order", "refund", "payment"):
        return True
    try:
        from services.ai_route_semantics import (
            _message_is_personal_order_tracking_without_id,
            brain_route_to_live_goal,
            ensure_brain_order_route_locked,
        )

        locked = ensure_brain_order_route_locked(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
        if brain_route_to_live_goal(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        ):
            return True
        if _message_is_personal_order_tracking_without_id(original_msg, msg_en):
            return True
    except ImportError:
        pass
    return False


def fast_finalize_personal_order_live(
    route: dict,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> dict | None:
    """
    Promote brain JSON to locked personal order/refund/track before product rescue.
    Uses brain fields + structural/AI guardrails — no product keyword lists.
    """
    if not isinstance(route, dict):
        return None
    try:
        from services.ai_route_semantics import (
            brain_route_is_order_nav_howto,
            reconcile_order_nav_howto_from_brain_meaning,
        )

        route = reconcile_order_nav_howto_from_brain_meaning(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
        if brain_route_is_order_nav_howto(route):
            return None
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import (
            _message_is_personal_order_tracking_without_id,
            ensure_brain_order_route_locked,
            reconcile_structural_order_sub_intent_from_tracking_message,
        )
        from services.brain_direct_dispatch import _brain_route_is_personal_order_live
        from services.refund_status_semantics import (
            promote_refund_status_on_route,
            refund_status_route_is_locked,
        )

        out = ensure_brain_order_route_locked(
            dict(route),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
        # Personal refund wins over track/KB drift BEFORE track promotion.
        out = promote_refund_status_on_route(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
            allow_llm=False,
        )
        if refund_status_route_is_locked(out):
            out["run_catalog_search"] = False
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["_order_live_route_locked"] = True
            out.pop("search_query", None)
            log_reasoning("Order SoT — personal refund_status locked (before track).")
            return out

        out = reconcile_structural_order_sub_intent_from_tracking_message(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if _message_is_personal_order_tracking_without_id(original_msg, msg_en):
            out["intent"] = "order"
            out["data_channel"] = "live_api"
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["order_lookup_kind"] = "track"
            out["route_handler"] = "order_tracking_api"
            out["run_catalog_search"] = False
            out["kb_keys"] = []
        if _brain_route_is_personal_order_live(out):
            out["run_catalog_search"] = False
            out["_order_live_route_locked"] = True
            out.pop("search_query", None)
            log_reasoning(
                "Order SoT — personal live API locked "
                f"(olk={out.get('order_lookup_kind')!r})."
            )
            return out
    except ImportError:
        pass
    if order_session_blocks_product_rescue(ctx, original_msg, msg_en):
        out = dict(route)
        out["run_catalog_search"] = False
        out["_order_live_route_locked"] = True
        return out
    return None


def try_force_order_id_handoff_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    user_id: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    reply_for_live_order_id_lookup: Callable[..., str],
) -> Optional[str]:
    """
    Last-resort zero-LLM handoff when awaiting=order_id but normal handoff missed.

    Must honor locked invoice/details/refund goals — never collapse invoice → track API.
    """
    if not isinstance(ctx, dict) or ctx.get("awaiting") != "order_id":
        return None
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not re.search(r"\b[0-9]{4,20}\b", comb):
        return None
    try:
        from services.order_id_handoff_fast_path import (
            _fetch_details_handoff_reply,
            _infer_handoff_goal_zero_llm,
            _locked_handoff_goal_from_session,
            _prefer_live_handoff_goal,
            _resolve_handoff_order_id,
        )

        oid = _resolve_handoff_order_id(
            original_msg, msg_en, conversation_context, ctx
        )
        if not oid:
            return None
        goal = _prefer_live_handoff_goal(
            _locked_handoff_goal_from_session(
                ctx,
                conversation_context,
                original_msg=original_msg,
                msg_en=msg_en,
            ),
            _infer_handoff_goal_zero_llm(
                conversation_context,
                ctx,
                original_msg=original_msg,
                msg_en=msg_en,
            ),
        )
        if not goal:
            last = (ctx.get("last") or "").strip().lower()
            goal = {
                "refund": "refund_status",
                "invoice": "order_invoice",
                "payment": "payment",
            }.get(last, "")
            if not goal:
                ai_route = (ctx.get("data") or {}).get("ai_route") or {}
                if isinstance(ai_route, dict):
                    try:
                        from services.ai_route_semantics import brain_route_to_live_goal

                        goal = brain_route_to_live_goal(ai_route) or ""
                    except ImportError:
                        olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
                        if olk == "invoice":
                            goal = "order_invoice"
                        elif olk in ("details", "order_details"):
                            goal = "order_details"
                        elif olk == "refund_status":
                            goal = "refund_status"
            if not goal:
                goal = "track"
        lang = reply_lang or "en"
        log_reasoning(
            f"Order SoT forced handoff — goal={goal} id={oid} (awaiting lock)."
        )
        if isinstance(ctx, dict):
            ctx["order_id"] = oid
            ctx["awaiting"] = None
            if goal == "refund_status":
                ctx["last"] = "refund"
            elif goal == "order_invoice":
                ctx["last"] = "invoice"
            elif goal == "payment":
                ctx["last"] = "payment"
            else:
                ctx["last"] = "order"
            ctx.setdefault("data", {})
            ctx["data"].pop("pending_action", None)
            ctx["data"]["topic_mode"] = f"order_{goal}"

        if goal in ("order_invoice", "order_details", "payment"):
            ai_focus = ""
            ai_route = (ctx.get("data") or {}).get("ai_route") or {}
            if isinstance(ai_route, dict):
                ai_focus = (ai_route.get("field_focus") or "").strip()
            if goal == "payment":
                ai_focus = ai_focus or "payment"
            details_goal = "order_details" if goal == "payment" else goal
            return _fetch_details_handoff_reply(
                details_goal, oid, user_id, original_msg, lang, ai_focus=ai_focus
            )

        live_intent = "refund" if goal == "refund_status" else "order"
        return reply_for_live_order_id_lookup(
            live_intent, oid, user_id, original_msg, lang
        )
    except ImportError:
        return None
