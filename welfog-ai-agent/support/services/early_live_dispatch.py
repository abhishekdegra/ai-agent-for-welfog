"""
Unified early live-API dispatch for ALL personal-data intents.

After the main AI router runs once, call try_early_live_api_reply() before KB
pre-scope or duplicate micro-classifiers (track, refund, details, invoice,
wishlist, order history, pincode).
"""
from __future__ import annotations

from typing import Any, Optional

from utils.reasoning_log import log_reasoning

_LIVE_API_HANDLERS = frozenset(
    {
        "wishlist_api",
        "order_ai_flow",
        "order_tracking_api",
        "order_details_api",
        "refund_status_api",
        "pincode_delivery_api",
    }
)

_LIVE_ORDER_GOALS = frozenset(
    {
        "refund_status",
        "order_invoice",
        "order_details",
        "track",
        "payment",
    }
)


def _brain_locked_single_order_turn(
    ai_route: dict | None,
    *,
    handler: str = "",
    olk: str = "",
) -> bool:
    """True when ai_brain_route already locked a single-order live API this turn."""
    try:
        from services.semantic_intent import llm_semantic_route_available
    except ImportError:
        return False
    if not llm_semantic_route_available(ai_route):
        return False
    r = ai_route or {}
    rh = (handler or r.get("route_handler") or "").strip().lower()
    olk_v = (olk or r.get("order_lookup_kind") or "").strip().lower()
    if rh in ("order_tracking_api", "order_details_api", "refund_status_api"):
        return True
    if olk_v in (
        "track",
        "tracking",
        "details",
        "invoice",
        "order_details",
        "refund_status",
    ):
        return True
    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    return (
        intent in ("order", "refund", "payment")
        and channel == "live_api"
        and bool(r.get("needs_order_id"))
    )


def _stamp_order_id_pending_from_brain(
    ctx: dict,
    *,
    ai_route: dict | None,
    handler: str = "",
    olk: str = "",
) -> str:
    goal = ""
    try:
        from services.ai_route_semantics import brain_route_to_live_goal

        goal = brain_route_to_live_goal(ai_route or {}) or ""
    except ImportError:
        pass
    if goal not in _LIVE_ORDER_GOALS:
        olk_v = (olk or (ai_route or {}).get("order_lookup_kind") or "").strip().lower()
        rh = (handler or (ai_route or {}).get("route_handler") or "").strip().lower()
        by_olk = {
            "track": "track",
            "tracking": "track",
            "invoice": "order_invoice",
            "details": "order_details",
            "order_details": "order_details",
            "refund_status": "refund_status",
        }
        by_rh = {
            "order_tracking_api": "track",
            "order_details_api": "order_details",
            "refund_status_api": "refund_status",
        }
        goal = by_olk.get(olk_v) or by_rh.get(rh) or "track"
    ctx.setdefault("data", {})["pending_action"] = goal
    ctx["data"]["topic_mode"] = f"order_{goal}"
    if isinstance(ai_route, dict):
        ctx["data"]["ai_route"] = dict(ai_route)
    return goal


def ai_route_is_live_api_turn(
    ai_route: dict | None,
    route_decision: Any = None,
) -> bool:
    """True when this turn needs a live API — never informational KB pre-scope."""
    r = ai_route or {}
    handler = (
        (getattr(route_decision, "handler", None) if route_decision else None)
        or r.get("route_handler")
        or ""
    ).strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    intent = (
        (getattr(route_decision, "intent", None) if route_decision else None)
        or r.get("intent")
        or ""
    ).strip().lower()
    if handler in _LIVE_API_HANDLERS:
        return True
    if channel == "live_api" and intent in (
        "order",
        "refund",
        "payment",
        "order_history",
        "wishlist",
        "pincode_check",
    ):
        return True
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    if olk in ("track", "tracking", "details", "invoice", "refund_status"):
        return True
    return False


def turn_blocks_kb_pre_scope(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    route_decision: Any = None,
) -> str:
    """Non-empty reason ⇒ skip KB pre-scope (live API or personal data)."""
    if ai_route_is_live_api_turn(ai_route, route_decision):
        return "live_api_route"

    try:
        from services.account_list_semantics import (
            turn_requests_purchase_history_in_chat,
            turn_requests_wishlist_in_chat,
        )

        if turn_requests_wishlist_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return "wishlist_in_chat"
        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return "order_history_in_chat"
    except ImportError:
        pass

    try:
        from services.refund_status_semantics import ai_route_requests_refund_status_lookup
        from services.order_details_flow import message_wants_order_details_or_invoice
        from utils.helpers import user_turn_qualifies_for_live_order_api

        if ai_route_requests_refund_status_lookup(
            ai_route, original_msg, msg_en, conversation_context
        ):
            return "refund_status"
        if message_wants_order_details_or_invoice(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return "order_details_invoice"
        if user_turn_qualifies_for_live_order_api(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return "live_order_lookup"
    except ImportError:
        pass

    return ""


def _record_live(intent: str, source: str) -> None:
    try:
        from services.chat_flow_telemetry import record_route

        record_route(intent=intent, source=source)
    except ImportError:
        pass


def try_early_live_api_reply(
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    route_decision: Any,
    ai_route: dict | None,
    reply_for_live_order_id_lookup,
    reset_context_fn,
) -> Optional[str]:
    """
    Dispatch live APIs immediately after main route.
    Returns reply HTML or None (caller continues normal flow).
    """
    handler = (getattr(route_decision, "handler", None) or "").strip().lower()
    intent = (
        (getattr(route_decision, "intent", None) or "")
        or ((ai_route or {}).get("intent") or "")
    ).strip().lower()
    olk = ((ai_route or {}).get("order_lookup_kind") or "").strip().lower()

    # --- Order history list + wishlist before single-order APIs ---
    try:
        from services.account_list_fast_path import try_account_list_fast_reply
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer
        from services.welfog_api import format_purchase_history_reply, format_wishlist_reply

        def _loc_sys(key: str, user_msg: str, reply_lang: str = "en") -> str:
            return localized_sysmsg_for_customer(key, user_msg, reply_lang=reply_lang)

        account_handoff = try_account_list_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            localized_sysmsg=_loc_sys,
            sysmsg=sysmsg,
            reset_context_fn=reset_context_fn,
        )
        if account_handoff:
            log_reasoning("Early live dispatch: account-list fast path.")
            return account_handoff
    except ImportError:
        pass

    # --- Other company / person — decline before pincode or order fast paths ---
    try:
        from services.support_scope import try_external_scope_fast_decline

        external = try_external_scope_fast_decline(
            original_msg, msg_en, conv_for_llm, reply_lang=lang
        )
        if external:
            log_reasoning("Early live dispatch: external scope decline.")
            return external
    except ImportError:
        pass

    # --- Pincode delivery: direct API when PIN is already in the message ---
    try:
        from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

        pin_handoff = try_pincode_delivery_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            lang,
            ctx,
            reset_context_fn=reset_context_fn,
        )
        if pin_handoff:
            log_reasoning("Early live dispatch: pincode delivery fast path.")
            return pin_handoff
    except ImportError:
        pass

    # --- Latest-turn order goal (details / invoice / track / refund) before stale refund thread ---
    try:
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

        live_handoff = try_order_live_intent_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
        )
        if live_handoff:
            log_reasoning("Early live dispatch: order live intent fast path.")
            return live_handoff
    except ImportError:
        pass

    # Enriched brain JSON may resolve goal when resolve_current_live_goal did not.
    try:
        from services.semantic_intent import llm_semantic_route_available
        from services.ai_route_semantics import (
            brain_route_to_live_goal,
            ensure_brain_order_route_locked,
        )
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

        if llm_semantic_route_available(ai_route):
            locked = ensure_brain_order_route_locked(
                ai_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
            brain_goal = brain_route_to_live_goal(
                locked,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
            if brain_goal in _LIVE_ORDER_GOALS:
                brain_fast = try_order_live_intent_fast_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm,
                    user_id,
                    lang,
                    ctx,
                    reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context_fn,
                    preset_goal=brain_goal,
                )
                if brain_fast:
                    log_reasoning(
                        f"Early live dispatch: brain-locked order goal={brain_goal} "
                        "(no specialist order LLM)."
                    )
                    return brain_fast
    except ImportError:
        pass

    # --- FAQ / policy KB (vector) before refund misroute ---
    try:
        from services.knowledge_fast_path import try_knowledge_fast_reply

        kb_handoff = try_knowledge_fast_reply(
            original_msg, msg_en, conv_for_llm, reply_lang=lang
        )
        if kb_handoff:
            log_reasoning("Early live dispatch: knowledge fast path.")
            return kb_handoff
    except ImportError:
        pass

    # --- Refund: policy KB / API / ask Order ID (locked route, no cascade) ---
    try:
        from services.refund_intent_fast_path import try_refund_intent_fast_reply

        refund_handoff = try_refund_intent_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
        )
        if refund_handoff:
            log_reasoning("Early live dispatch: refund intent fast path.")
            return refund_handoff
    except ImportError:
        pass

    # --- Order-ID handoff: one API for the submitted id (no routing LLM stack) ---
    try:
        from services.order_id_handoff_fast_path import try_order_id_handoff_reply

        handoff = try_order_id_handoff_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
        )
        if handoff:
            log_reasoning("Early live dispatch: order-ID handoff fast path.")
            return handoff
    except ImportError:
        pass

    # --- Pincode delivery (handler OR semantic delivery turn) ---
    try:
        from services.entity_first_handlers import try_pincode_delivery_reply
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        pincode_turn = (
            handler == "pincode_delivery_api"
            or intent == "pincode_check"
            or turn_requests_delivery_serviceability(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=ai_route,
                allow_llm=True,
            )
        )
        if pincode_turn:
            pin = try_pincode_delivery_reply(
                original_msg, msg_en, conv_for_llm, lang, ctx, ai_route=ai_route
            )
            if pin:
                log_reasoning("Early live dispatch: pincode delivery API.")
                _record_live("pincode_check", "pincode_delivery_api")
                return pin
    except ImportError:
        pass

    # --- Wishlist list in chat ---
    try:
        from services.account_list_semantics import turn_requests_wishlist_in_chat
        from services.welfog_api import format_wishlist_reply

        if (
            handler == "wishlist_api"
            or intent == "wishlist"
            or turn_requests_wishlist_in_chat(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route
            )
        ):
            from services.account_list_semantics import ai_route_requests_wishlist_howto

            if not ai_route_requests_wishlist_howto(ai_route):
                log_reasoning("Early live dispatch: wishlist API.")
                _record_live("wishlist", "wishlist_api")
                reset_context_fn(ctx)
                ctx["last"] = "wishlist"
                ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
                return format_wishlist_reply(user_id, page=1, append_only=False)
    except ImportError:
        pass

    # --- Purchase / order history list in chat ---
    try:
        from services.account_list_semantics import (
            ai_route_requests_order_history_howto,
            turn_requests_purchase_history_in_chat,
        )
        from services.location_delivery_resolver import turn_requests_delivery_serviceability
        from services.welfog_api import format_purchase_history_reply

        pincode_turn = (
            handler == "pincode_delivery_api"
            or intent == "pincode_check"
            or turn_requests_delivery_serviceability(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=ai_route,
                allow_llm=True,
            )
        )
        single_order_intent = False
        try:
            from services.order_details_flow import (
                message_has_non_tracking_order_id_intent,
                message_wants_order_details_or_invoice,
            )
            from services.refund_status_semantics import current_turn_wants_personal_refund_status

            if message_has_non_tracking_order_id_intent(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route
            ):
                single_order_intent = True
            elif message_wants_order_details_or_invoice(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route
            ):
                single_order_intent = True
            elif current_turn_wants_personal_refund_status(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route, allow_llm=False
            ):
                single_order_intent = True
        except ImportError:
            pass

        refund_topic = False
        try:
            from services.refund_intent_fast_path import turn_has_refund_topic

            refund_topic = turn_has_refund_topic(original_msg, msg_en)
        except ImportError:
            pass
        if (
            not pincode_turn
            and not single_order_intent
            and not refund_topic
            and (
                handler == "order_ai_flow"
                or intent == "order_history"
                or turn_requests_purchase_history_in_chat(
                    original_msg, msg_en, conv_for_llm, ai_route=ai_route
                )
            )
        ):
            if not ai_route_requests_order_history_howto(ai_route):
                log_reasoning("Early live dispatch: purchase-history API.")
                _record_live("order_history", "order_ai_flow")
                reset_context_fn(ctx)
                ctx["last"] = "order_history"
                ctx.setdefault("data", {})["topic_mode"] = "order_history_list"
                return format_purchase_history_reply(user_id, page=1, append_only=False)
    except ImportError:
        pass

    # --- Refund status (before details/invoice — refund beats stale locks) ---
    try:
        from services.refund_status_flow import run_refund_status_ai_flow
        from services.refund_status_semantics import (
            KIND_PERSONAL_STATUS,
            resolve_refund_turn,
        )

        thread_refund = False
        try:
            from services.conversation_thread_semantics import (
                infer_order_thread_goal,
                message_needs_thread_continuation,
            )

            if message_needs_thread_continuation(
                f"{original_msg} {msg_en}".strip(), conv_for_llm
            ):
                thread_refund = (
                    infer_order_thread_goal(
                        conv_for_llm,
                        f"{original_msg} {msg_en}".strip(),
                        ctx_last=ctx.get("last") if isinstance(ctx, dict) else None,
                    )
                    == "refund_status"
                )
        except ImportError:
            pass
        resolved = resolve_refund_turn(
            original_msg,
            msg_en,
            conv_for_llm,
            ai_route=ai_route,
            reply_lang=lang,
            allow_llm=True,
        )
        wants_refund = resolved.kind == KIND_PERSONAL_STATUS or thread_refund
        if wants_refund or handler == "refund_status_api" or olk == "refund_status":
            if not _brain_locked_single_order_turn(ai_route, handler=handler, olk=olk):
                result = run_refund_status_ai_flow(
                    original_msg,
                    msg_en,
                    user_id,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ai_route=ai_route,
                )
                if result.handled and result.reply_html:
                    log_reasoning(
                        f"Early live dispatch: refund status API ({wants_refund or olk or handler})."
                    )
                    _record_live("refund", "refund_status_api")
                    if result.needs_order_id and isinstance(ctx, dict):
                        goal = _stamp_order_id_pending_from_brain(
                            ctx, ai_route=ai_route, handler=handler, olk=olk
                        )
                        ctx["last"] = "refund"
                        ctx["awaiting"] = "order_id"
                    else:
                        reset_context_fn(ctx)
                        ctx["last"] = "refund"
                        if result.order_id:
                            ctx["order_id"] = result.order_id
                        ctx["awaiting"] = None
                    return result.reply_html
            else:
                log_reasoning(
                    "Skip refund specialist LLM — brain already locked single-order turn."
                )
    except ImportError:
        pass

    # --- Order details / invoice ---
    try:
        from services.location_delivery_resolver import turn_requests_delivery_serviceability
        from services.order_details_flow import (
            message_wants_order_details_or_invoice,
            run_order_details_ai_flow,
        )

        pincode_blocks_order = (
            handler == "pincode_delivery_api"
            or intent == "pincode_check"
            or turn_requests_delivery_serviceability(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=ai_route,
                allow_llm=True,
            )
        )
        wants_od = (
            ""
            if pincode_blocks_order
            else message_wants_order_details_or_invoice(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route
            )
        )
        if (
            not pincode_blocks_order
            and (
                wants_od
                or handler == "order_details_api"
                or olk in ("details", "invoice", "order_details", "order_invoice")
            )
        ):
            if not _brain_locked_single_order_turn(ai_route, handler=handler, olk=olk):
                result = run_order_details_ai_flow(
                    original_msg,
                    msg_en,
                    user_id,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ai_route=ai_route,
                )
                if result.handled and result.reply_html:
                    log_reasoning(
                        f"Early live dispatch: order details API ({wants_od or olk or handler})."
                    )
                    _record_live("order", "order_details_api")
                    if result.needs_order_id and isinstance(ctx, dict):
                        _stamp_order_id_pending_from_brain(
                            ctx, ai_route=ai_route, handler=handler, olk=olk
                        )
                        ctx["last"] = "order"
                        ctx["awaiting"] = "order_id"
                    else:
                        reset_context_fn(ctx)
                        ctx["last"] = "order"
                        if result.order_id:
                            ctx["order_id"] = result.order_id
                        ctx["awaiting"] = None
                    return result.reply_html
            else:
                log_reasoning(
                    "Skip order-details specialist LLM — brain already locked single-order turn."
                )
    except ImportError:
        pass

    # --- Account list follow-up (self-view in chat) before wrong tracking/KB ---
    try:
        from services.account_list_semantics import detect_account_list_followup_in_chat
        from services.welfog_api import format_purchase_history_reply, format_wishlist_reply

        follow = detect_account_list_followup_in_chat(
            original_msg,
            msg_en,
            conv_for_llm,
            ctx=ctx,
            ai_route=ai_route,
            reply_lang=lang,
        )
        if follow == "wishlist":
            log_reasoning("Early live dispatch: wishlist follow-up in chat.")
            _record_live("wishlist", "wishlist_api")
            reset_context_fn(ctx)
            ctx["last"] = "wishlist"
            ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
            return format_wishlist_reply(user_id, page=1, append_only=False)
        if follow == "order_history":
            log_reasoning("Early live dispatch: order-history follow-up in chat.")
            _record_live("order_history", "order_ai_flow")
            reset_context_fn(ctx)
            ctx["last"] = "order_history"
            ctx.setdefault("data", {})["topic_mode"] = "order_history_list"
            return format_purchase_history_reply(user_id, page=1, append_only=False)
    except ImportError:
        pass

    # --- Order tracking ---
    try:
        from services.conversation_thread_semantics import (
            infer_order_thread_goal,
            message_needs_thread_continuation,
        )

        _comb_track = f"{original_msg} {msg_en}".strip()
        if message_needs_thread_continuation(_comb_track, conv_for_llm):
            _thread = infer_order_thread_goal(
                conv_for_llm,
                _comb_track,
                ctx_last=ctx.get("last") if isinstance(ctx, dict) else None,
            )
            if _thread == "refund_status":
                handler = ""
                olk = ""
    except ImportError:
        pass
    if handler == "order_tracking_api" or olk in ("track", "tracking"):
        try:
            from utils.helpers import extract_order_id, resolve_order_id_for_tracking

            oid = extract_order_id(f"{original_msg} {msg_en}", conv_for_llm)
            if not oid:
                oid = resolve_order_id_for_tracking(
                    original_msg.strip() or msg_en.strip(), conv_for_llm
                )
            if not oid:
                from services.account_list_semantics import (
                    detect_account_list_followup_in_chat,
                    turn_requests_purchase_history_in_chat,
                )

                um = ((ai_route or {}).get("user_meaning") or "").lower()
                if (
                    detect_account_list_followup_in_chat(
                        original_msg,
                        msg_en,
                        conv_for_llm,
                        ctx=ctx,
                        ai_route=ai_route,
                        reply_lang=lang,
                    )
                    or turn_requests_purchase_history_in_chat(
                        original_msg, msg_en, conv_for_llm, ai_route=ai_route
                    )
                    or (
                        "order history" in um
                        and "track" not in um
                        and "shipment" not in um
                    )
                ):
                    log_reasoning(
                        "Early live dispatch: skip tracking — account list / history turn."
                    )
                    return None
        except ImportError:
            pass
        if not _brain_locked_single_order_turn(ai_route, handler=handler, olk=olk):
            try:
                from services.order_tracking_flow import run_order_tracking_ai_flow

                result = run_order_tracking_ai_flow(
                    original_msg,
                    msg_en,
                    user_id,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                )
                if result.handled and result.reply_html:
                    log_reasoning("Early live dispatch: order tracking API.")
                    _record_live("order", "order_tracking_api")
                    if result.needs_order_id and isinstance(ctx, dict):
                        _stamp_order_id_pending_from_brain(
                            ctx, ai_route=ai_route, handler=handler, olk=olk
                        )
                        ctx["last"] = "order"
                        ctx["awaiting"] = "order_id"
                    else:
                        reset_context_fn(ctx)
                        ctx["last"] = "order"
                        if result.order_id:
                            ctx["order_id"] = result.order_id
                        ctx["awaiting"] = None
                    return result.reply_html
            except ImportError:
                pass
        else:
            log_reasoning(
                "Skip order-tracking specialist LLM — brain already locked single-order turn."
            )

    # --- Live order-id lookup (track / refund / payment) when ID present ---
    try:
        from services.order_details_flow import message_wants_order_details_or_invoice
        from utils.helpers import (
            _message_is_order_id_followup_submission,
            resolve_order_id_for_tracking,
            resolve_live_api_intent_from_conversation,
            should_attempt_live_order_api_reply,
            user_turn_qualifies_for_live_order_api,
        )

        try:
            from services.conversation_thread_semantics import (
                resolve_explicit_turn_goal_from_message,
            )

            explicit = resolve_explicit_turn_goal_from_message(
                original_msg, msg_en, conv_for_llm, ai_route, allow_llm=False
            )
            if explicit in ("order_invoice", "order_details"):
                return None
        except ImportError:
            pass
        if message_wants_order_details_or_invoice(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route
        ):
            return None
        if user_turn_qualifies_for_live_order_api(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route
        ) and should_attempt_live_order_api_reply(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route
        ):
            pending_oid = resolve_order_id_for_tracking(
                original_msg.strip() or msg_en.strip(),
                conv_for_llm,
                bot_awaiting_order_id=ctx.get("awaiting") == "order_id"
                or _message_is_order_id_followup_submission(original_msg, conv_for_llm),
            )
            if pending_oid:
                live_intent = resolve_live_api_intent_from_conversation(
                    conv_for_llm,
                    ctx.get("last"),
                    original_msg,
                    msg_en,
                    ai_route=ai_route if isinstance(ai_route, dict) else None,
                )
                log_reasoning(
                    f"Early live dispatch: {live_intent} for order {pending_oid}."
                )
                _record_live(live_intent, f"live_{live_intent}_api")
                ctx["last"] = (
                    live_intent if live_intent in ("refund", "payment", "order") else "order"
                )
                ctx["order_id"] = pending_oid
                ctx["awaiting"] = None
                return reply_for_live_order_id_lookup(
                    live_intent, pending_oid, user_id, original_msg, lang
                )
    except ImportError:
        pass

    return None
