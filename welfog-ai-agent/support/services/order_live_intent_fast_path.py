"""
Explicit per-turn order intent — latest message wins over stale ctx.last / refund thread.

One goal resolution → one live API (details / invoice / track / refund / payment).
"""
from __future__ import annotations

import re
import threading
from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning

_LIVE_GOAL_CACHE = threading.local()

_LIVE_GOALS = frozenset(
    (
        "refund_status",
        "track",
        "order_details",
        "order_invoice",
        "payment",
    )
)

_PENDING_ACTION_TO_GOAL = {
    "order_invoice": "order_invoice",
    "order_details": "order_details",
    "track": "track",
    "track_single_order": "track",
    "refund_status": "refund_status",
    "payment": "payment",
}


def resolve_structural_locked_order_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> str:
    """
    Zero-LLM order goal only when session intent is already locked.
    Never guess invoice vs track from customer text — that is ai_brain_route's job.
    """
    if not isinstance(ctx, dict):
        return ""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return ""
    pending = (
        (ctx.get("data") or {}).get("pending_action") or ""
    ).strip().lower()
    awaiting = ctx.get("awaiting")
    has_oid = bool(re.search(r"\b[0-9]{4,20}\b", comb))
    bare_oid = bool(re.fullmatch(r"[0-9]{4,20}", comb.strip()))

    message_goal = _structural_message_live_goal_no_id(
        original_msg, msg_en, conversation_context
    )
    if message_goal:
        if pending and pending in _PENDING_ACTION_TO_GOAL:
            pending_goal = _PENDING_ACTION_TO_GOAL[pending]
            if pending_goal != message_goal:
                try:
                    from utils.helpers import clear_order_session_for_new_lookup

                    clear_order_session_for_new_lookup(ctx)
                except ImportError:
                    ctx["order_id"] = None
                    ctx["awaiting"] = None
                    ctx.pop("last", None)
                    if isinstance(ctx.get("data"), dict):
                        ctx["data"].pop("pending_action", None)
                        ctx["data"].pop("topic_mode", None)
        return message_goal

    if pending in _PENDING_ACTION_TO_GOAL and (bare_oid or (has_oid and awaiting == "order_id")):
        return _PENDING_ACTION_TO_GOAL[pending]

    if has_oid:
        try:
            from services.ai_route_semantics import _structural_refund_goal_from_message

            refund_goal = _structural_refund_goal_from_message(
                original_msg, msg_en
            )
            if refund_goal == "refund_status":
                return "refund_status"
        except ImportError:
            pass

    if re.fullmatch(r"[0-9]{4,20}", comb.strip()):
        try:
            from services.order_id_handoff_fast_path import _infer_handoff_goal_zero_llm

            handoff = (
                _infer_handoff_goal_zero_llm(
                    conversation_context,
                    ctx,
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
                or ""
            ).strip()
            if handoff in _PENDING_ACTION_TO_GOAL:
                return _PENDING_ACTION_TO_GOAL[handoff]
            if handoff in _LIVE_GOALS:
                return handoff
        except ImportError:
            pass
        if awaiting == "order_id":
            try:
                from services.order_id_handoff_fast_path import (
                    _locked_handoff_goal_from_session,
                )

                locked = _locked_handoff_goal_from_session(
                    ctx,
                    conversation_context,
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
                if locked in _LIVE_GOALS:
                    return locked
            except ImportError:
                pass
        return ""
    return _structural_message_live_goal_no_id(
        original_msg, msg_en, conversation_context
    )


def _structural_message_live_goal_no_id(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    """Zero-LLM single-order goal from customer text when no order id in message."""
    comb = _combined(original_msg, msg_en)
    if not comb or re.search(r"\b[0-9]{4,20}\b", comb):
        return ""
    if re.search(
        r"\b(?:invoice|invoic\w*|bill|receipt|gst|chalan|challan)\b",
        comb,
        re.I,
    ):
        tl = f" {comb.lower()} "
        track_markers = (
            "track",
            "tracking",
            "kab aa",
            "kab tak",
            "kb tk",
            "pahunch",
            "delivery status",
            "nhi aa",
            "nahi aa",
            "nhi aaya",
            "nahi aaya",
        )
        if not any(m in tl for m in track_markers):
            return "order_invoice"
    try:
        from utils.helpers import (
            _text_is_order_tracking_intent_leaf,
            _text_is_undelivered_order_complaint,
            message_is_general_delivery_policy_question,
        )

        if not message_is_general_delivery_policy_question(comb):
            if _text_is_order_tracking_intent_leaf(
                comb
            ) or _text_is_undelivered_order_complaint(comb):
                return "track"
    except ImportError:
        pass
    try:
        from services.order_details_flow import _lightweight_details_or_invoice_signal

        light = (_lightweight_details_or_invoice_signal(comb) or "").strip()
        if light == "order_invoice":
            return "order_invoice"
        if light == "order_details":
            return "order_details"
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import _structural_track_goal_from_message
        from utils.helpers import message_is_general_delivery_policy_question

        if not message_is_general_delivery_policy_question(comb):
            if _structural_track_goal_from_message(original_msg, msg_en) == "track":
                return "track"
    except ImportError:
        pass
    return ""


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _message_references_prior_order_id(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    return bool(
        re.search(r"\b(?:is|usi|wahi|same|iski|iska|ye|yeh)\s*(?:id|order)\b", tl)
        or re.search(r"\bisid\b", tl)
        or re.search(r"\bis\s*id\b", tl)
        or "is order" in tl
        or "is id" in tl
    )


def _message_needs_order_context_carryover(
    text: str,
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """
    Follow-up on same order without repeating id — entity carryover, not keyword intent.
    Intent comes from ai_brain_route; this only recovers order_id from thread/session.
    """
    if not (text or "").strip():
        return False
    if re.search(r"\b[0-9]{4,20}\b", text):
        return False
    try:
        from utils.helpers import (
            _text_is_order_tracking_intent_leaf,
            _text_is_undelivered_order_complaint,
            message_user_switches_order_scope,
        )

        if message_user_switches_order_scope(text):
            return False
        if _text_is_order_tracking_intent_leaf(text) or _text_is_undelivered_order_complaint(
            text
        ):
            if not _message_references_prior_order_id(text):
                return False
    except ImportError:
        pass
    if _message_references_prior_order_id(text):
        return True
    if not (conversation_context or "").strip():
        return False
    try:
        from utils.helpers import extract_latest_order_id_from_user_conversation

        if not extract_latest_order_id_from_user_conversation(conversation_context, text):
            return False
    except ImportError:
        return False
    try:
        from services.semantic_intent import llm_semantic_route_available
        from services.turn_intent_coordinator import _brain_route_for_turn
        from services.ai_route_semantics import infer_semantic_goal_from_ai_route

        brain = _brain_route_for_turn()
        if llm_semantic_route_available(brain):
            semantic = infer_semantic_goal_from_ai_route(brain)
            if semantic in (
                "order_details",
                "order_invoice",
                "refund_status",
                "track_single_order",
            ):
                return _message_references_prior_order_id(text)
    except ImportError:
        pass
    return False


def _turn_is_pincode_delivery_not_single_order(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """PIN + serviceability — must not steal 6-digit PIN as Order ID (recursion-safe)."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    try:
        from utils.helpers import _leaf_non_tracking_order_id_intent

        if _leaf_non_tracking_order_id_intent(comb):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _text_has_pincode_delivery_intent,
            _text_is_pincode_serviceability_question_light,
            message_has_live_pincode_check_intent,
        )

        if message_has_live_pincode_check_intent(
            original_msg, conversation_context, msg_en
        ):
            return True
        if _text_is_pincode_serviceability_question_light(comb):
            return True
        if _text_has_pincode_delivery_intent(comb, conversation_context):
            return True
    except ImportError:
        pass
    if isinstance(ctx, dict) and (ctx.get("last") or "").strip().lower() == "pincode":
        if re.search(r"\b[1-9]\d{5}\b", comb):
            return True
    return False


def _turn_is_account_list_not_single_order(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """Skip single-order API when AI already classified an account-list intent."""
    try:
        from services.account_list_semantics import (
            KIND_NONE,
            _norm_account_list_kind,
            account_list_action_from_brain_route,
        )
        from services.turn_intent_coordinator import (
            _brain_route_for_turn,
            structural_skip_account_list_classifier,
        )

        if structural_skip_account_list_classifier(
            original_msg, msg_en, conversation_context, None
        ):
            return False
        brain = _brain_route_for_turn()
        if isinstance(brain, dict):
            try:
                from services.ai_route_semantics import (
                    brain_route_to_live_goal,
                    ensure_brain_order_route_locked,
                )

                locked = ensure_brain_order_route_locked(
                    brain,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                live_goal = brain_route_to_live_goal(
                    locked,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                if live_goal in (
                    "refund_status",
                    "order_invoice",
                    "order_details",
                    "track",
                    "payment",
                ):
                    return False
            except ImportError:
                pass
            if brain.get("needs_order_id"):
                olk = (brain.get("order_lookup_kind") or "").strip().lower()
                rh = (brain.get("route_handler") or "").strip().lower()
                if olk not in ("none", "") or rh in (
                    "order_tracking_api",
                    "order_details_api",
                    "refund_status_api",
                ):
                    return False
        action = account_list_action_from_brain_route(brain)
        kind = _norm_account_list_kind(action.get("kind") or KIND_NONE)
        if kind != KIND_NONE and float(action.get("confidence") or 0) >= 0.72:
            return True
        if isinstance(brain, dict):
            bi = (brain.get("intent") or "").strip().lower()
            if bi in ("deals", "categories", "category_feed", "wishlist"):
                return True
            if bi == "order_history":
                try:
                    from services.ai_route_semantics import resolve_order_live_goal_for_turn

                    struct_goal = resolve_order_live_goal_for_turn(
                        brain,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conversation_context=conversation_context,
                    )
                    if struct_goal in (
                        "refund_status",
                        "order_invoice",
                        "order_details",
                        "track",
                        "payment",
                    ):
                        return False
                except ImportError:
                    pass
                if re.search(r"\b\d{4,20}\b", f"{original_msg} {msg_en}"):
                    return False
                if re.search(
                    r"\b(?:ek|one|mera|meri|usi|uski|uska|iska|iski)\b",
                    f" {(original_msg or '').lower()} {(msg_en or '').lower()} ",
                ):
                    return False
                return True
    except ImportError:
        pass
    return False


def _keyword_intent_fallback_enabled() -> bool:
    """When strict AI routing is on, customer text keywords must not pick the API."""
    try:
        from services.semantic_intent import strict_ai_semantic_mode

        return not strict_ai_semantic_mode()
    except ImportError:
        return True


def _goal_from_brain_route(
    reply_lang: str = "en",
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """Live goal from ai_brain_route JSON (any customer language → English user_meaning)."""
    try:
        from services.turn_intent_coordinator import _brain_route_for_turn
        from services.ai_route_semantics import resolve_order_live_goal_for_turn
        from services.semantic_intent import llm_semantic_route_available

        brain = ai_route if isinstance(ai_route, dict) else _brain_route_for_turn()
        if not isinstance(brain, dict) or not llm_semantic_route_available(brain):
            return ""
        return resolve_order_live_goal_for_turn(
            brain,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
    except ImportError:
        pass
    return ""


# Back-compat alias
_goal_from_brain_route_fallback = _goal_from_brain_route


def _resolve_message_live_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """
    Legacy keyword fallback when STRICT_AI_INTENT_ROUTING=0 or brain LLM unavailable.
    Default: intent from ai_brain_route only (any language/style).
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return ""
    if _turn_is_pincode_delivery_not_single_order(
        original_msg, msg_en, conversation_context, None
    ):
        return ""
    if _turn_is_account_list_not_single_order(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return ""

    brain_goal = _goal_from_brain_route(
        reply_lang,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    if brain_goal in _LIVE_GOALS:
        return brain_goal

    if not _keyword_intent_fallback_enabled():
        return ""

    try:
        from services.order_details_flow import (
            _conversation_hints_invoice_followup,
            _text_wants_order_details_not_tracking,
            _text_wants_order_invoice,
        )

        if _text_wants_order_invoice(comb, conversation_context) or _conversation_hints_invoice_followup(
            comb, conversation_context
        ):
            return "order_invoice"
    except ImportError:
        pass

    try:
        from utils.helpers import _text_is_refund_return_status_lookup

        if _text_is_refund_return_status_lookup(comb, conversation_context):
            return "refund_status"
    except ImportError:
        pass

    try:
        from services.order_details_flow import _text_wants_order_details_not_tracking

        if _text_wants_order_details_not_tracking(comb, conversation_context):
            return "order_details"
    except ImportError:
        pass

    try:
        from services.order_details_flow import _PAYMENT_FOCUS_RE
        from utils.helpers import extract_order_id

        if _PAYMENT_FOCUS_RE.search(comb) and (
            extract_order_id(comb, conversation_context)
            or _message_references_prior_order_id(comb)
        ):
            return "order_details"
    except ImportError:
        pass

    try:
        from utils.helpers import _text_is_order_tracking_intent_leaf

        if _text_is_order_tracking_intent_leaf(comb):
            return "track"
    except ImportError:
        pass

    if _message_references_prior_order_id(comb):
        try:
            from services.order_details_flow import (
                _conversation_hints_invoice_followup,
                _text_wants_order_details_not_tracking,
                _text_wants_order_invoice,
            )
            from utils.helpers import _text_is_refund_return_status_lookup, _text_is_order_tracking_intent_leaf

            if _text_wants_order_invoice(comb, conversation_context) or _conversation_hints_invoice_followup(
                comb, conversation_context
            ):
                return "order_invoice"
            if _text_is_refund_return_status_lookup(comb, conversation_context):
                return "refund_status"
            if _text_wants_order_details_not_tracking(comb, conversation_context):
                return "order_details"
            if _text_is_order_tracking_intent_leaf(comb):
                return "track"
        except ImportError:
            pass

    return ""


def resolve_live_goal_lightweight(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """Live goal from ai_brain_route when available; keyword fallback only if disabled."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return ""
    if _turn_is_pincode_delivery_not_single_order(
        original_msg, msg_en, conversation_context, None
    ):
        return ""
    if _turn_is_account_list_not_single_order(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return ""
    brain_goal = _goal_from_brain_route(
        reply_lang,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    if brain_goal in _LIVE_GOALS:
        return brain_goal
    if _keyword_intent_fallback_enabled():
        return _resolve_message_live_goal(
            original_msg, msg_en, conversation_context, reply_lang
        )
    return ""


def resolve_order_id_for_live_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> str:
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import message_user_switches_order_scope

        if message_user_switches_order_scope(comb):
            if isinstance(ctx, dict):
                ctx["order_id"] = None
            return ""
    except ImportError:
        pass
    if _turn_is_pincode_delivery_not_single_order(
        original_msg, msg_en, conversation_context, ctx
    ):
        return ""
    if _turn_is_account_list_not_single_order(
        original_msg, msg_en, conversation_context, ""
    ):
        return ""
    try:
        from utils.helpers import extract_order_id

        oid_fast = extract_order_id(comb, conversation_context)
        if oid_fast:
            return str(oid_fast).strip()
    except ImportError:
        pass
    if _message_needs_order_context_carryover(comb, conversation_context, ctx):
        if isinstance(ctx, dict):
            ctx_oid = (ctx.get("order_id") or "").strip()
            if ctx_oid:
                return ctx_oid
        try:
            from utils.helpers import extract_latest_order_id_from_user_conversation

            oid_ctx = extract_latest_order_id_from_user_conversation(
                conversation_context, comb
            )
            if oid_ctx:
                return str(oid_ctx).strip()
        except ImportError:
            pass
    try:
        from services.refund_intent_fast_path import resolve_refund_order_id
        from utils.helpers import extract_latest_order_id_from_user_conversation
    except ImportError:
        return ""

    oid = resolve_refund_order_id(original_msg, msg_en, conversation_context)
    if not oid and _message_references_prior_order_id(comb):
        oid = extract_latest_order_id_from_user_conversation(
            conversation_context, comb
        ) or ""
    return (oid or "").strip()


def resolve_current_live_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """What the LATEST user message wants — not stale session thread."""
    cache_key = (
        f"{hash(_combined(original_msg, msg_en))}|"
        f"{hash((conversation_context or '')[-500:])}|{reply_lang}"
    )
    if getattr(_LIVE_GOAL_CACHE, "key", None) == cache_key:
        return str(getattr(_LIVE_GOAL_CACHE, "result", "") or "")

    def _finish(goal: str) -> str:
        _LIVE_GOAL_CACHE.key = cache_key
        _LIVE_GOAL_CACHE.result = goal or ""
        return goal or ""

    comb = _combined(original_msg, msg_en)
    if _turn_is_pincode_delivery_not_single_order(
        original_msg, msg_en, conversation_context, None
    ):
        return _finish("")
    if _turn_is_account_list_not_single_order(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return _finish("")

    brain_goal = _goal_from_brain_route(
        reply_lang,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    if brain_goal in _LIVE_GOALS:
        log_reasoning(f"Order live goal from ai_brain_route: {brain_goal}.")
        return _finish(brain_goal)

    if _keyword_intent_fallback_enabled():
        msg_goal = _resolve_message_live_goal(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if msg_goal in _LIVE_GOALS:
            log_reasoning(f"Order live goal keyword fallback: {msg_goal}.")
            return _finish(msg_goal)

    try:
        from utils.helpers import extract_order_id

        if extract_order_id(comb, conversation_context):
            log_reasoning(
                "Order id present but no live goal — skip refund/semantic LLM classifiers."
            )
            return _finish("")
    except ImportError:
        pass

    try:
        from services.turn_intent_coordinator import _brain_route_for_turn

        brain = _brain_route_for_turn()
        skip_refund_llm = bool(
            isinstance(brain, dict)
            and brain.get("_universal_brain_route")
            and (
                (brain.get("route_handler") or "").strip().lower() == "refund_status_api"
                or (brain.get("order_lookup_kind") or "").strip().lower() == "refund_status"
            )
        )
    except ImportError:
        skip_refund_llm = True

    if not skip_refund_llm:
        try:
            from services.refund_intent_fast_path import _get_refund_ai_classification
            from services.refund_status_semantics import _message_has_refund_topic
            from utils.helpers import _text_is_refund_return_status_lookup

            has_refund_signal = bool(
                _message_has_refund_topic(comb)
                or _text_is_refund_return_status_lookup(comb, conversation_context)
            )
            rf = _get_refund_ai_classification(
                original_msg, msg_en, conversation_context, reply_lang
            )
            if rf:
                rk = (rf.get("refund_turn_kind") or "none").strip().lower()
                conf = float(rf.get("confidence") or 0.0)
                if rk == "policy_howto" and conf >= 0.5:
                    return _finish("")
                if rk == "personal_status" and conf >= 0.5 and has_refund_signal:
                    return _finish("refund_status")
        except ImportError:
            pass

    return _finish("")


def _brain_field_focus_for_details() -> str:
    try:
        from services.turn_intent_coordinator import _brain_route_for_turn

        brain = _brain_route_for_turn()
        if isinstance(brain, dict):
            return (brain.get("field_focus") or "").strip()
    except ImportError:
        pass
    return ""


def try_order_live_intent_fast_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    user_id: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    reply_for_live_order_id_lookup: Callable[..., str],
    reset_context_fn=None,
    preset_goal: str = "",
) -> Optional[str]:
    import time as _time

    _t_total = _time.perf_counter()
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import (
            clear_order_session_for_new_lookup,
            message_user_switches_order_scope,
        )

        if message_user_switches_order_scope(comb):
            clear_order_session_for_new_lookup(ctx)
    except ImportError:
        pass

    goal = (preset_goal or "").strip().lower()
    if goal not in _LIVE_GOALS:
        goal = resolve_current_live_goal(
            original_msg, msg_en, conversation_context, reply_lang
        )
    if goal not in _LIVE_GOALS:
        return None

    tool_map = {
        "refund_status": "refund_status_api",
        "order_invoice": "order_details_api",
        "order_details": "order_details_api",
        "track": "order_tracking_api",
        "payment": "order_details_api",
    }
    selected_tool = tool_map.get(goal, "")
    if not selected_tool:
        return None
    comb = _combined(original_msg, msg_en)
    prev_ctx_used = bool((conversation_context or "").strip()) and (
        _message_references_prior_order_id(comb)
        or not re.search(r"\b[0-9]{4,20}\b", comb)
    )

    oid = resolve_order_id_for_live_turn(
        original_msg, msg_en, conversation_context, ctx
    )
    if not oid:
        if goal in ("order_details", "order_invoice", "refund_status", "track", "payment"):
            from services.order_history_flow import _localized_sysmsg

            intent = (
                "refund"
                if goal == "refund_status"
                else "invoice"
                if goal == "order_invoice"
                else "order"
            )
            body = _localized_sysmsg(
                "ask_order_id_for_intent",
                original_msg,
                reply_lang=reply_lang or "en",
                intent=intent,
            )
            if body and isinstance(ctx, dict):
                ctx["order_id"] = None
                ctx["awaiting"] = "order_id"
                ctx["last"] = intent
                ctx.setdefault("data", {})["pending_action"] = goal
                ctx["data"]["topic_mode"] = f"order_{goal}"
                try:
                    from services.ai_route_semantics import LIVE_API_FROM_GOAL

                    handler = LIVE_API_FROM_GOAL.get(goal, "")
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
                        "order_lookup_kind": olk_map.get(goal, ""),
                        "route_handler": handler,
                    }
                except ImportError:
                    pass
            try:
                from services.chat_flow_telemetry import log_order_dispatch

                log_order_dispatch(
                    detected_intent=goal,
                    detected_language=reply_lang or "en",
                    message=comb,
                    previous_context=(conversation_context or "")[:200],
                    previous_context_used=prev_ctx_used,
                    pending_action=goal,
                    order_id_found="",
                    selected_tool=selected_tool,
                    api_called=False,
                    api_time_ms=0.0,
                    entities={"needs_order_id": "true"},
                )
            except ImportError:
                pass
            return body or None
        return None

    lang = reply_lang or "en"
    log_reasoning(
        f"Order live intent fast path: goal={goal} id={oid or '-'} (latest turn wins)."
    )
    try:
        from services.chat_flow_telemetry import record_route, record_route_step, store_turn_analysis
        from services.answer_router import AnswerRouteDecision

        handler_map = {
            "refund_status": ("refund_status_api", "refund"),
            "order_details": ("order_details_api", "order"),
            "order_invoice": ("order_details_api", "order"),
            "track": ("order_tracking_api", "order"),
            "payment": ("order_details_api", "payment"),
        }
        mapped = handler_map.get(goal)
        if not mapped:
            return None
        h, intent = mapped
        record_route_step("order_live_intent_fast")
        record_route(intent=intent, source=h)
        store_turn_analysis(
            {
                "intent": intent,
                "data_channel": "live_api",
                "route_handler": h,
                "order_lookup_kind": goal,
                "extracted_order_id": oid,
                "needs_order_id": not bool(oid),
                "numeric_context": "order_id" if oid else "none",
            },
            AnswerRouteDecision(
                source="api",
                intent=intent,
                handler=h,
                is_welfog_related=True,
                reason=f"Order live intent fast — {goal}",
            ),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
    except ImportError:
        pass

    if isinstance(ctx, dict):
        ctx["order_id"] = oid
        ctx["awaiting"] = None
        ctx.setdefault("data", {}).pop("pending_action", None)
        if goal == "refund_status":
            ctx["last"] = "refund"
        elif goal == "order_invoice":
            ctx["last"] = "invoice"
        elif goal == "payment":
            ctx["last"] = "payment"
        else:
            ctx["last"] = "order"

    _t_api = _time.perf_counter()
    reply_html = ""
    if goal in ("order_details", "order_invoice", "payment"):
        from services.order_id_handoff_fast_path import _fetch_details_handoff_reply

        details_focus = (
            "payment"
            if goal == "payment"
            else _brain_field_focus_for_details()
        )
        reply_html = _fetch_details_handoff_reply(
            "order_details" if goal == "payment" else goal,
            oid,
            user_id,
            original_msg,
            lang,
            ai_focus=details_focus,
        ) or ""
    else:
        live_intent = "refund" if goal == "refund_status" else goal
        if live_intent == "track":
            live_intent = "order"
        if live_intent not in ("refund", "payment", "order"):
            live_intent = "order"
        reply_html = reply_for_live_order_id_lookup(
            live_intent, oid, user_id, original_msg, lang
        ) or ""

    api_time_ms = (_time.perf_counter() - _t_api) * 1000.0
    try:
        from services.chat_flow_telemetry import log_order_dispatch, record_api_time

        record_api_time(api_time_ms / 1000.0)
        log_order_dispatch(
            detected_intent=goal,
            detected_language=lang,
            message=comb,
            previous_context=(conversation_context or "")[:200],
            previous_context_used=prev_ctx_used,
            pending_action=goal,
            order_id_found=oid or "",
            selected_tool=selected_tool,
            api_called=bool(reply_html),
            api_time_ms=api_time_ms,
            entities={"order_id": oid},
        )
    except ImportError:
        pass
    log_reasoning(
        f"[order-flow] structural_fast_path total_time="
        f"{(_time.perf_counter() - _t_total) * 1000.0:.0f}ms goal={goal}"
    )
    return reply_html or None


def try_order_live_intent_fast_route(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: dict | None = None,
    reply_lang: str = "en",
) -> Optional[tuple[Any, dict]]:
    """Router lock when latest turn has explicit live goal (+ optional order id)."""
    goal = resolve_current_live_goal(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    if goal not in _LIVE_GOALS:
        return None

    oid = resolve_order_id_for_live_turn(original_msg, msg_en, conv_for_llm, ctx)
    from services.answer_router import AnswerRouteDecision

    user_meaning = f"Live {goal}" + (f" for order {oid}" if oid else "")

    if goal in ("order_details", "order_invoice"):
        from services.order_details_flow import _apply_order_details_to_route

        route_data = _apply_order_details_to_route(
            {
                "user_meaning": user_meaning,
                "reasoning": "Order live intent — latest message goal.",
                "extracted_order_id": oid,
                "needs_order_id": not bool(oid),
            },
            goal,
            "order_live_intent_fast",
        )
        decision = AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            is_welfog_related=True,
            reason=f"Order live intent — {goal}",
        )
        return decision, route_data

    if goal == "refund_status":
        from services.refund_status_semantics import _apply_refund_status_to_route

        if not oid:
            route_data = {
                "user_meaning": user_meaning,
                "reasoning": "Order live intent — refund status, need Order ID.",
                "intent": "refund",
                "data_channel": "live_api",
                "needs_order_id": True,
                "numeric_context": "order_id",
                "order_lookup_kind": "refund_status",
                "route_handler": "refund_status_api",
            }
        else:
            route_data = _apply_refund_status_to_route(
                {
                    "user_meaning": user_meaning,
                    "reasoning": "Order live intent — live return-request API.",
                    "extracted_order_id": oid,
                    "needs_order_id": False,
                },
                "order_live_intent_fast",
            )
        return (
            AnswerRouteDecision(
                source="api",
                intent="refund",
                handler="refund_status_api",
                is_welfog_related=True,
                reason="Order live intent — refund_status",
            ),
            route_data,
        )

    handler = "order_tracking_api"
    intent = "order"
    olk = "track"
    if goal == "payment":
        handler = "payment_api"
        intent = "payment"
        olk = "payment"

    route_data = {
        "user_meaning": user_meaning,
        "reasoning": f"Order live intent — {goal}.",
        "intent": intent,
        "data_channel": "live_api",
        "extracted_order_id": oid,
        "needs_order_id": not bool(oid),
        "numeric_context": "order_id",
        "order_lookup_kind": olk,
        "route_handler": handler,
        "_order_live_intent_fast": True,
    }
    return (
        AnswerRouteDecision(
            source="api",
            intent=intent,
            handler=handler,
            is_welfog_related=True,
            reason=f"Order live intent — {goal}",
        ),
        route_data,
    )
