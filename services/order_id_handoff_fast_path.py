"""
Order-ID handoff fast path — bot asked for Order ID, user pasted it (or says track now).

Skips ai_brain_route and specialist LLMs; one live API call for that single order only.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _is_bare_id_token(text: str) -> bool:
    """Bare pasted token — must look like a real Order ID, not random chat ('ehat', 'bye')."""
    comb = (text or "").strip()
    if not comb or not re.fullmatch(r"[A-Za-z0-9]{4,20}", comb):
        return False
    try:
        from utils.helpers import _is_plausible_order_id

        return _is_plausible_order_id(comb, shallow=True)
    except ImportError:
        return bool(re.fullmatch(r"[0-9]{4,20}", comb))


def _message_is_bare_order_id_submission(original_msg: str, msg_en: str = "") -> bool:
    """Bare pasted Order ID — do not treat duplicated msg_en as non-bare."""
    raw = (original_msg or "").strip()
    if _is_bare_id_token(raw):
        return True
    comb = _combined(original_msg, msg_en)
    return _is_bare_id_token(comb)


_INTENT_LABEL_TO_GOAL = {
    "refund": "refund_status",
    "order": "track",
    "invoice": "order_invoice",
    "payment": "payment",
    "track": "track",
    "tracking": "track",
}


def _normalize_live_goal(goal: str) -> str:
    g = (goal or "").strip().lower()
    if g in ("refund",):
        return "refund_status"
    if g in ("invoice", "bill"):
        return "order_invoice"
    if g in ("details", "order_details"):
        return "order_details"
    if g in ("tracking",):
        return "track"
    return g


def lock_order_id_ask_session(
    ctx: dict,
    goal: str,
    *,
    intent_label: str = "",
    ai_route: dict | None = None,
) -> None:
    """
    Lock session so the next bare Order ID hits the correct live API (any language).
    Source of truth: brain ai_route + pending_action — not user-message keywords.
    """
    if not isinstance(ctx, dict):
        return
    goal = _normalize_live_goal(goal)
    if goal not in (
        "track",
        "order_invoice",
        "order_details",
        "refund_status",
        "payment",
    ):
        goal = "track"
    label = (intent_label or "").strip().lower() or {
        "refund_status": "refund",
        "order_invoice": "invoice",
        "payment": "payment",
    }.get(goal, "order")
    ctx["order_id"] = None
    ctx["awaiting"] = "order_id"
    ctx["last"] = label
    ctx.setdefault("data", {})
    ctx["data"]["pending_action"] = goal
    ctx["data"]["topic_mode"] = f"order_{goal}"
    if isinstance(ai_route, dict) and ai_route:
        route_snapshot = dict(ai_route)
        route_snapshot["needs_order_id"] = True
        route_snapshot["numeric_context"] = "order_id"
        route_snapshot["data_channel"] = route_snapshot.get("data_channel") or "live_api"
        ctx["data"]["ai_route"] = route_snapshot
    else:
        try:
            from services.ai_route_semantics import LIVE_API_FROM_GOAL

            olk_map = {
                "track": "track",
                "order_invoice": "invoice",
                "order_details": "details",
                "payment": "details",
                "refund_status": "refund_status",
            }
            ctx["data"]["ai_route"] = {
                "intent": "refund" if goal == "refund_status" else "order",
                "data_channel": "live_api",
                "needs_order_id": True,
                "numeric_context": "order_id",
                "order_lookup_kind": olk_map.get(goal, goal),
                "route_handler": LIVE_API_FROM_GOAL.get(goal, ""),
            }
        except ImportError:
            ctx["data"]["ai_route"] = {
                "intent": "refund" if goal == "refund_status" else "order",
                "data_channel": "live_api",
                "needs_order_id": True,
                "order_lookup_kind": goal,
            }
    log_reasoning(f"Order-ID ask session locked: goal={goal} label={label}")


def lock_order_id_ask_from_intent_label(
    ctx: dict,
    intent_label: str,
    *,
    ai_route: dict | None = None,
) -> None:
    goal = _INTENT_LABEL_TO_GOAL.get((intent_label or "").strip().lower(), "track")
    lock_order_id_ask_session(
        ctx, goal, intent_label=intent_label, ai_route=ai_route
    )


def resolve_bare_order_id_handoff_goal(
    ctx: dict | None,
    conversation_context: str = "",
    *,
    original_msg: str = "",
    msg_en: str = "",
    reply_lang: str = "en",
) -> str:
    """Resolve live API goal for bare Order ID — session lock first, then thread."""
    goal = _resolve_handoff_thread_goal(
        original_msg,
        msg_en,
        conversation_context,
        ctx,
        reply_lang,
    )
    if goal:
        return _normalize_live_goal(goal)
    locked = _locked_handoff_goal_from_session(
        ctx,
        conversation_context,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    return _normalize_live_goal(locked)


def _prefer_live_handoff_goal(*goals: str) -> str:
    """
    Bare Order-ID follow-up: keep track/invoice/details/refund distinct.
    Refund wins over a mistaken track lock when any signal says refund.
    """
    cleaned = []
    for g in goals:
        g = _normalize_live_goal(g or "")
        if g in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            cleaned.append(g)
    if not cleaned:
        return ""
    if "refund_status" in cleaned:
        return "refund_status"
    if "order_invoice" in cleaned:
        return "order_invoice"
    if "payment" in cleaned:
        return "payment"
    if "order_details" in cleaned:
        return "order_details"
    return cleaned[0]


def _infer_handoff_goal_zero_llm(
    conversation_context: str,
    ctx: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """ctx + last bot line + current turn markers — bare ID handoff only."""
    comb = _combined(original_msg, msg_en)
    ctx_last = ""
    pending_action = ""
    awaiting = None
    topic_mode = ""
    if isinstance(ctx, dict):
        ctx_last = (ctx.get("last") or "").strip().lower()
        pending_action = (
            (ctx.get("data") or {}).get("pending_action") or ""
        ).strip().lower()
        awaiting = ctx.get("awaiting")
        topic_mode = ((ctx.get("data") or {}).get("topic_mode") or "").strip().lower()

    session_goal = ""
    if pending_action in (
        "track",
        "order_invoice",
        "order_details",
        "refund_status",
        "payment",
    ):
        session_goal = pending_action
        log_reasoning(f"Handoff goal from pending_action: {pending_action}")
    elif awaiting == "order_id" and topic_mode.startswith("order_"):
        topic_goal = topic_mode.replace("order_", "", 1)
        if topic_goal in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            session_goal = topic_goal
            log_reasoning(f"Handoff goal from topic_mode: {topic_goal}")

    route_goal = ""
    if isinstance(ctx, dict) and awaiting == "order_id":
        ai_route = (ctx.get("data") or {}).get("ai_route") or {}
        if isinstance(ai_route, dict):
            try:
                from services.ai_route_semantics import brain_route_to_live_goal

                live = brain_route_to_live_goal(ai_route)
                if live:
                    route_goal = live
                    log_reasoning(f"Handoff goal from ctx ai_route (awaiting): {live}")
            except ImportError:
                pass
            if not route_goal:
                olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
                rh = (ai_route.get("route_handler") or "").strip().lower()
                if olk == "refund_status" or rh == "refund_status_api":
                    route_goal = "refund_status"
                elif olk == "invoice":
                    route_goal = "order_invoice"
                elif olk in ("details", "order_details") or rh == "order_details_api":
                    route_goal = "order_details"
                elif olk in ("track", "tracking") or rh == "order_tracking_api":
                    route_goal = "track"

    if ctx_last == "refund":
        session_goal = session_goal or "refund_status"
    elif ctx_last == "payment":
        session_goal = session_goal or "payment"
    elif ctx_last == "invoice":
        session_goal = session_goal or "order_invoice"
    elif ctx_last == "track":
        session_goal = session_goal or "track"

    thread_goal = ""
    if conversation_context:
        try:
            from services.conversation_thread_semantics import infer_order_thread_goal

            # When session already locked (invoice ask → OID), skip slow LLM.
            # Otherwise AI classifies thread meaning (any language) — no phrase lists.
            ai_route_ctx = (
                ((ctx.get("data") or {}).get("ai_route") or {})
                if isinstance(ctx, dict)
                else {}
            )
            thread_goal = infer_order_thread_goal(
                conversation_context,
                comb or " ",
                ctx_last=ctx_last,
                ai_route=ai_route_ctx if isinstance(ai_route_ctx, dict) else None,
                allow_llm=not bool(session_goal or route_goal),
            ) or ""
            if thread_goal in (
                "refund_status",
                "order_invoice",
                "order_details",
                "track",
                "payment",
            ):
                log_reasoning(f"Handoff goal from conversation thread: {thread_goal}")
        except ImportError:
            pass

    # Structured session label only — never scrape assistant HTML with keyword lists.
    # Ask paths must call lock_order_id_ask_session so last/pending_action are SoT.
    label_goal = ""
    if ctx_last == "invoice":
        label_goal = "order_invoice"
    elif ctx_last == "refund":
        label_goal = "refund_status"
    elif ctx_last == "payment":
        label_goal = "payment"
    elif ctx_last == "track":
        label_goal = "track"

    preferred = _prefer_live_handoff_goal(
        session_goal, route_goal, label_goal, thread_goal
    )

    # Current-turn AI/explicit meaning beats a stale track lock (any language/style).
    # Bare Order-ID-only lines leave this empty → session/route wins.
    current_turn_goal = ""
    if comb and not re.fullmatch(
        r"(?:order\s*id\s*[:\-]?\s*)?[A-Za-z0-9]{4,20}",
        comb,
        flags=re.I,
    ):
        try:
            from services.conversation_thread_semantics import (
                resolve_explicit_turn_goal_from_message,
            )

            ai_route_ctx = (
                ((ctx.get("data") or {}).get("ai_route") or {})
                if isinstance(ctx, dict)
                else {}
            )
            # Prefer LLM meaning when structural cache is empty — industry path.
            current_turn_goal = resolve_explicit_turn_goal_from_message(
                original_msg,
                msg_en,
                conversation_context,
                ai_route_ctx if isinstance(ai_route_ctx, dict) else None,
                allow_llm=True,
            ) or ""
            if current_turn_goal not in (
                "refund_status",
                "order_invoice",
                "order_details",
                "track",
                "payment",
            ):
                current_turn_goal = ""
            if current_turn_goal:
                log_reasoning(
                    f"Handoff goal from current-turn AI/explicit: {current_turn_goal}"
                )
        except ImportError:
            current_turn_goal = ""

    if current_turn_goal:
        preferred = _prefer_live_handoff_goal(current_turn_goal, preferred)

    if preferred:
        log_reasoning(
            f"Handoff goal preferred={preferred} "
            f"(session={session_goal or '-'} route={route_goal or '-'} "
            f"label={label_goal or '-'} thread={thread_goal or '-'} "
            f"current={current_turn_goal or '-'})"
        )
        return preferred

    if comb:
        try:
            from services.semantic_intent import zero_llm_intent_guess_allowed

            allow_phrase_guess = zero_llm_intent_guess_allowed()
        except ImportError:
            allow_phrase_guess = True
        if allow_phrase_guess:
            try:
                from services.refund_status_semantics import _message_has_refund_topic

                if _message_has_refund_topic(comb):
                    return "refund_status"
            except ImportError:
                pass
            try:
                from services.conversation_thread_semantics import (
                    resolve_explicit_turn_goal_from_message,
                )

                # Second chance with LLM when preferred was empty.
                explicit = resolve_explicit_turn_goal_from_message(
                    original_msg,
                    msg_en,
                    conversation_context,
                    None,
                    allow_llm=True,
                )
                if explicit in (
                    "refund_status",
                    "order_invoice",
                    "order_details",
                    "track",
                    "payment",
                ):
                    return explicit
            except ImportError:
                pass
            try:
                from utils.helpers import _text_is_refund_return_status_lookup

                if _text_is_refund_return_status_lookup(comb, conversation_context):
                    return "refund_status"
            except ImportError:
                pass

    if ctx_last in ("order", "track", "order_history", "invoice"):
        if isinstance(ctx, dict) and awaiting != "order_id":
            ai_route = (ctx.get("data") or {}).get("ai_route") or {}
            if isinstance(ai_route, dict):
                try:
                    from services.ai_route_semantics import brain_route_to_live_goal

                    live = brain_route_to_live_goal(ai_route)
                    if live:
                        log_reasoning(f"Handoff goal from ctx ai_route: {live}")
                        return live
                except ImportError:
                    pass
                try:
                    from services.order_details_flow import _goal_from_brain_route

                    bg = _goal_from_brain_route(ai_route)
                    if bg == "order_details":
                        return "order_details"
                    if bg == "order_invoice":
                        return "order_invoice"
                    if bg == "track_single_order":
                        return "track"
                except ImportError:
                    pass
        if ctx_last == "track":
            return "track"
        if ctx_last == "invoice":
            return "order_invoice"
    if isinstance(ctx, dict):
        ai_route = (ctx.get("data") or {}).get("ai_route") or {}
        if isinstance(ai_route, dict):
            try:
                from services.ai_route_semantics import brain_route_to_live_goal

                live = brain_route_to_live_goal(ai_route)
                if live:
                    log_reasoning(f"Handoff goal fallback from ctx ai_route: {live}")
                    return live
            except ImportError:
                pass
    try:
        from services.semantic_intent import strict_ai_semantic_mode

        if strict_ai_semantic_mode():
            return ""
    except ImportError:
        pass
    return ""


def _locked_handoff_goal_from_session(
    ctx: dict | None,
    conversation_context: str = "",
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """Session lock for bare Order ID — pending_action / topic_mode / ai_route (zero LLM)."""
    if not isinstance(ctx, dict):
        return ""
    last = (ctx.get("last") or "").strip().lower()
    if last == "refund":
        return "refund_status"
    if last == "invoice":
        return "order_invoice"
    if last == "payment":
        return "payment"
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
        return pending
    topic = ((ctx.get("data") or {}).get("topic_mode") or "").strip().lower()
    if topic.startswith("order_"):
        topic_goal = topic.replace("order_", "", 1)
        if topic_goal in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            return topic_goal
    if ctx.get("awaiting") == "order_id":
        goal = _infer_handoff_goal_zero_llm(
            conversation_context,
            ctx,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if goal:
            return goal
        ai_route = (ctx.get("data") or {}).get("ai_route") or {}
        if isinstance(ai_route, dict):
            try:
                from services.ai_route_semantics import brain_route_to_live_goal

                live = brain_route_to_live_goal(ai_route)
                if live:
                    return live
            except ImportError:
                pass
            olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
            rh = (ai_route.get("route_handler") or "").strip().lower()
            if olk in ("track", "tracking") or rh == "order_tracking_api":
                return "track"
            if olk == "refund_status" or rh == "refund_status_api":
                return "refund_status"
            if olk == "invoice":
                return "order_invoice"
            if olk in ("details", "order_details") or rh == "order_details_api":
                return "order_details"
    return ""


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
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False

    try:
        from services.chitchat_resolver import _is_instant_greeting_thanks_lane

        if _is_instant_greeting_thanks_lane(
            original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass

    # Session lock / pending_action wins — order IDs often embed 6-digit PIN substrings.
    if _bot_awaiting_order_id(conversation_context, ctx):
        if re.search(r"\b[0-9]{4,20}\b", comb):
            return True
        return False
    if isinstance(ctx, dict):
        last = (ctx.get("last") or "").strip().lower()
        if last in ("refund", "order", "invoice", "payment", "track"):
            if re.search(r"\b[0-9]{4,20}\b", comb):
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
        ) and re.search(r"\b[0-9]{4,20}\b", comb):
            return True

    if _message_is_bare_order_id_submission(original_msg, msg_en) and _bot_awaiting_order_id(conversation_context, ctx):
        return True

    try:
        from utils.helpers import (
            _conversation_bot_asked_for_pincode,
            _conversation_bot_offered_order_id_or_tracking,
            _message_is_order_id_followup_submission,
            _text_is_pincode_serviceability_question,
            turn_is_catalog_product_lookup,
        )
    except ImportError:
        return False

    if turn_is_catalog_product_lookup(original_msg, msg_en):
        return False
    if _message_is_order_id_followup_submission(original_msg, conversation_context):
        return True
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False

    if not _message_is_bare_order_id_submission(original_msg, msg_en):
        try:
            from utils.helpers import extract_order_id

            if extract_order_id(comb, conversation_context):
                return True
        except ImportError:
            pass
        try:
            from services.refund_status_semantics import _message_has_refund_topic
            from services.order_details_flow import _lightweight_details_or_invoice_signal

            if _message_has_refund_topic(comb):
                return False
            if _lightweight_details_or_invoice_signal(comb) in (
                "order_invoice",
                "order_details",
            ):
                return False
        except ImportError:
            pass

    if _message_is_bare_order_id_submission(original_msg, msg_en) and _conversation_bot_offered_order_id_or_tracking(
        conversation_context
    ):
        return True

    return False


def _resolve_handoff_order_id(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> str:
    comb = _combined(original_msg, msg_en)
    raw = (original_msg or "").strip()
    for candidate in (raw, comb):
        if _is_bare_id_token(candidate):
            return candidate.strip()

    bot_awaiting = _bot_awaiting_order_id(conversation_context, ctx)
    if bot_awaiting or (
        isinstance(ctx, dict)
        and ((ctx.get("data") or {}).get("pending_action") or "").strip()
    ):
        m = re.search(r"\b([0-9]{4,20})\b", comb)
        if m:
            return m.group(1)

    try:
        from utils.helpers import (
            _message_is_order_id_followup_submission,
            resolve_order_id_for_tracking,
        )
    except ImportError:
        return ""

    bot_awaiting = bot_awaiting or _message_is_order_id_followup_submission(
        original_msg, conversation_context
    )
    oid = resolve_order_id_for_tracking(
        comb,
        conversation_context,
        bot_awaiting_order_id=bot_awaiting,
    )
    return (oid or "").strip()


def _resolve_handoff_thread_goal(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
    reply_lang: str = "en",
) -> str:
    comb = _combined(original_msg, msg_en)
    bare = _is_bare_id_token(comb)
    awaiting = _bot_awaiting_order_id(conversation_context, ctx)

    if bare or awaiting:
        goal = _infer_handoff_goal_zero_llm(
            conversation_context, ctx, original_msg=original_msg, msg_en=msg_en
        )
        log_reasoning(f"Handoff goal (zero-LLM): {goal}")
        return goal

    allow_semantic_llm = bool(
        re.search(r"\S+\s+\S+", comb)
        and not re.fullmatch(r"[0-9]{4,20}", comb.strip())
    )

    try:
        from services.conversation_thread_semantics import (
            resolve_explicit_turn_goal_from_message,
        )

        explicit = resolve_explicit_turn_goal_from_message(
            original_msg,
            msg_en,
            conversation_context,
            None,
            allow_llm=allow_semantic_llm,
            reply_lang=reply_lang,
        )
        if explicit:
            return explicit
    except ImportError:
        pass

    return _infer_handoff_goal_zero_llm(
        conversation_context, ctx, original_msg=original_msg, msg_en=msg_en
    )


def _fetch_details_handoff_reply(
    goal: str,
    order_id: str,
    user_id: str,
    original_msg: str,
    reply_lang: str,
    *,
    ai_focus: str = "",
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

    row, err = fetch_purchase_history_details_for_user(order_id, user_id, fast=True)
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

    focus = detect_order_details_focus(
        original_msg, action=action, ai_focus=ai_focus or ""
    )
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


def _handoff_requires_brain_first(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: dict | None,
) -> bool:
    """
    Semantic order turns need ai_brain_route before handoff (any language).
    Handoff stays zero-LLM only for bare id / awaiting locked pending_action.
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    try:
        from utils.helpers import extract_order_id

        if extract_order_id(comb, conversation_context):
            return False
    except ImportError:
        pass
    if _message_is_bare_order_id_submission(original_msg, msg_en):
        return False
    if _bot_awaiting_order_id(conversation_context, ctx):
        return False
    if isinstance(ctx, dict):
        pending = (
            (ctx.get("data") or {}).get("pending_action") or ""
        ).strip().lower()
        if pending in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ) and re.search(r"\b[0-9]{4,20}\b", comb):
            return False
    try:
        from services.semantic_intent import strict_ai_semantic_mode

        return strict_ai_semantic_mode()
    except ImportError:
        return True


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
    if _handoff_requires_brain_first(
        original_msg, msg_en, conversation_context, ctx
    ):
        log_reasoning(
            "Order-ID handoff skipped — semantic turn needs ai_brain_route first."
        )
        return None
    try:
        from services.catalog_turn_semantics import should_skip_catalog_for_conversational_turn

        if should_skip_catalog_for_conversational_turn(
            original_msg, msg_en, conversation_context
        ):
            return None
    except ImportError:
        pass
    if not _is_order_id_handoff_turn(original_msg, msg_en, conversation_context, ctx):
        return None

    order_id = _resolve_handoff_order_id(
        original_msg, msg_en, conversation_context, ctx
    )
    if not order_id:
        return None

    goal = _resolve_handoff_thread_goal(
        original_msg, msg_en, conversation_context, ctx, reply_lang
    )
    if not goal:
        goal = _locked_handoff_goal_from_session(
            ctx,
            conversation_context,
            original_msg=original_msg,
            msg_en=msg_en,
        )
    if not goal:
        return None
    lang = reply_lang or "en"

    log_reasoning(
        f"Order-ID handoff fast path: goal={goal} id={order_id} "
        "(skip brain route — single live API)."
    )
    try:
        from services.chat_flow_telemetry import log_order_dispatch, record_route, record_route_step

        record_route_step("order_id_handoff_fast")
        record_route(intent=goal, source="order_id_handoff_fast")
        tool = (
            "order_details_api"
            if goal in ("order_invoice", "order_details", "payment")
            else "refund_status_api"
            if goal == "refund_status"
            else "order_tracking_api"
            if goal == "track"
            else ""
        )
        log_order_dispatch(
            detected_intent=goal,
            message=_combined(original_msg, msg_en),
            previous_context=conversation_context,
            pending_action=goal,
            order_id_found=order_id,
            selected_tool=tool,
            api_called=True,
        )
    except ImportError:
        try:
            from services.chat_flow_telemetry import record_route, record_route_step

            record_route_step("order_id_handoff_fast")
            record_route(intent=goal, source="order_id_handoff_fast")
        except ImportError:
            pass

    if isinstance(ctx, dict):
        ctx["order_id"] = order_id
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
        if isinstance(ctx, dict):
            ai_route = (ctx.get("data") or {}).get("ai_route") or {}
            if isinstance(ai_route, dict):
                ai_focus = (ai_route.get("field_focus") or "").strip()
        if goal == "payment":
            ai_focus = ai_focus or "payment"
        details_goal = "order_details" if goal == "payment" else goal
        return _fetch_details_handoff_reply(
            details_goal, order_id, user_id, original_msg, lang, ai_focus=ai_focus
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
    if not goal:
        goal = _locked_handoff_goal_from_session(
            ctx,
            conv_for_llm,
            original_msg=original_msg,
            msg_en=msg_en,
        )
    if not goal:
        return None

    try:
        from services.ai_route_semantics import LIVE_API_FROM_GOAL

        handler = LIVE_API_FROM_GOAL.get(goal, "")
    except ImportError:
        handler = ""
    intent = "refund" if goal == "refund_status" else "payment" if goal == "payment" else "order"
    olk_map = {
        "track": "track",
        "order_invoice": "invoice",
        "order_details": "details",
        "payment": "details",
        "refund_status": "refund_status",
    }
    olk = olk_map.get(goal, "")
    if not handler:
        return None

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
