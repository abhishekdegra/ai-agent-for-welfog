"""
AI-first live ORDER TRACKING vs refund status vs order details — any language.

Tracking = shipment/ETA/courier/live status.
Refund = return-request API only when user means money-back/return approval.
"""
from __future__ import annotations

from typing import Optional

from services.ai_route_semantics import coerce_route_str
from utils.reasoning_log import log_reasoning

_TRACKING_MEANING = (
    "track order",
    "order tracking",
    "track shipment",
    "shipment status",
    "delivery status",
    "where is my order",
    "when will order arrive",
    "courier status",
    "live tracking",
    "order update",
    "kab aayega",
    "kab milega",
)

_TRACKING_EXCLUDE_FROM_REFUND = (
    "track",
    "tracking",
    "trck",
    "trak",
    "shipment",
    "courier",
    "parcel",
    "delivery status",
    "where is",
    "kab aayega",
    "kab aaega",
    "kab milega",
    "kab tak",
    "live status",
    "out for delivery",
    "shipped",
    "dispatch",
)


def _combined(original_msg: str, msg_en: str) -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def message_user_wants_order_tracking(
    text: str,
    conversation_context: str = "",
) -> bool:
    """Live shipment/ETA tracking — not refund, not payment/address details."""
    from utils.helpers import (
        _text_has_refund_or_return_intent,
        _text_is_order_tracking_intent,
        _text_is_refund_return_status_lookup,
    )

    if not (text or "").strip():
        return False
    if _text_is_refund_return_status_lookup(text, conversation_context):
        if not _text_is_order_tracking_intent(text):
            return False
    if _text_has_refund_or_return_intent(text) and "refund" in f" {text.lower()} ":
        if not _text_is_order_tracking_intent(text):
            return False
    return _text_is_order_tracking_intent(text)


def message_user_rejects_refund_wants_tracking(text: str) -> bool:
    """User correcting bot: not refund — they want tracking."""
    if not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    wants_track = any(
        x in tl
        for x in (
            "track",
            "tracking",
            "trck",
            "trak",
            "order track",
            "track kr",
            "track kar",
        )
    )
    if not wants_track:
        return False
    rejects_refund = any(
        x in tl
        for x in (
            "refund kyu",
            "refund kyo",
            "refund nhi",
            "refund nahi",
            "refund status nhi",
            "return nhi",
            "not refund",
            "why refund",
            "galat",
            "wrong",
        )
    )
    return rejects_refund or ("track" in tl and "refund" in tl)


def order_tracking_route_is_locked(route: dict | None) -> bool:
    if not route:
        return False
    try:
        from services.ai_route_semantics import (
            ai_meaning_describes_order_details,
            correct_order_details_vs_tracking_from_ai_meaning,
        )

        corrected = correct_order_details_vs_tracking_from_ai_meaning(dict(route))
        if ai_meaning_describes_order_details(corrected):
            olk = (corrected.get("order_lookup_kind") or "").strip().lower()
            if olk in ("details", "invoice"):
                return False
    except ImportError:
        pass
    olk = (route.get("order_lookup_kind") or "").strip().lower()
    if olk in ("track", "tracking", "track_single_order"):
        return True
    return (route.get("route_handler") or "").strip().lower() == "order_tracking_api"


def _apply_order_tracking_to_route(out: dict, source: str) -> dict:
    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["order_lookup_kind"] = "track"
    out["route_handler"] = "order_tracking_api"
    out["answer_strategy"] = "live_api_only"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    log_reasoning(f"Order-tracking ({source}): live welfog_track API (not refund/details).")
    return out


def ai_route_requests_order_tracking_lookup(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Shipment/ETA live tracking for one order — message + Groq JSON (any language)."""
    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(
            route,
            original_msg,
            msg_en,
            conversation_context,
            allow_llm=True,
        ):
            return False
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if ai_route_is_kb_read(route):
            return False
    except ImportError:
        pass
    comb = _combined(original_msg, msg_en)
    try:
        from services.refund_status_semantics import current_turn_wants_personal_refund_status

        if current_turn_wants_personal_refund_status(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=route,
            allow_llm=False,
        ):
            return False
    except ImportError:
        pass
    try:
        from services.order_details_flow import (
            message_user_wants_order_invoice,
            order_details_route_is_locked,
        )

        if message_user_wants_order_invoice(comb, conversation_context, ai_route=route):
            return False
        if order_details_route_is_locked(route):
            olk = (route or {}).get("order_lookup_kind") or ""
            if str(olk).strip().lower() in ("invoice", "order_invoice"):
                return False
    except ImportError:
        pass
    if message_user_rejects_refund_wants_tracking(comb):
        return True
    if message_user_wants_order_tracking(comb, conversation_context):
        return True
    if order_tracking_route_is_locked(route):
        return True

    r = route or {}
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    rh = (r.get("route_handler") or "").strip().lower()
    if olk in ("track", "tracking") or rh == "order_tracking_api":
        return True

    um = f" {(r.get('user_meaning') or '').lower()} "
    if any(x in um for x in _TRACKING_MEANING):
        return True

    try:
        from services.order_details_flow import understand_single_order_request

        sub = understand_single_order_request(
            original_msg, msg_en, conversation_context, ai_route=route
        )
        if (sub.get("goal") or "").strip() == "track_single_order":
            return True
    except ImportError:
        pass
    return False


def promote_order_tracking_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> dict:
    """Align Groq route with live order tracking (before refund/details promotion)."""
    out = dict(route or {})
    comb = _combined(original_msg, msg_en)

    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(
            out,
            original_msg,
            msg_en,
            conversation_context,
            allow_llm=True,
        ):
            return out
    except ImportError:
        pass

    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if ai_route_is_kb_read(out):
            return out
    except ImportError:
        pass

    try:
        from services.refund_status_semantics import (
            current_turn_wants_personal_refund_status,
            refund_status_route_is_locked,
        )

        if (
            current_turn_wants_personal_refund_status(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=out,
                allow_llm=False,
            )
            or refund_status_route_is_locked(out)
        ):
            return out
    except ImportError:
        pass

    try:
        from services.order_details_flow import (
            message_has_non_tracking_order_id_intent,
            message_user_wants_order_invoice,
            order_details_route_is_locked,
            _infer_followup_goal_from_conversation,
            _message_explicitly_wants_tracking,
        )

        try:
            from services.conversation_thread_semantics import infer_order_thread_goal

            if infer_order_thread_goal(conversation_context, _combined(original_msg, msg_en)) in (
                "refund_status",
                "order_details",
                "order_invoice",
            ):
                return out
        except ImportError:
            pass
        try:
            from utils.helpers import _leaf_non_tracking_order_id_intent

            if _leaf_non_tracking_order_id_intent(_combined(original_msg, msg_en)):
                return out
        except ImportError:
            if message_has_non_tracking_order_id_intent(
                original_msg, msg_en, conversation_context, ai_route=out
            ):
                return out

        wants_track_now = (
            _message_explicitly_wants_tracking(comb, conversation_context)
            or _infer_followup_goal_from_conversation(comb, conversation_context)
            == "track_single_order"
        )
        if order_details_route_is_locked(out) and not wants_track_now:
            olk = (out.get("order_lookup_kind") or "").strip().lower()
            if olk in ("invoice", "order_invoice", "details") or (
                out.get("route_handler") or ""
            ).strip().lower() == "order_details_api":
                return out
        if message_user_wants_order_invoice(comb, conversation_context, ai_route=out):
            return out
    except ImportError:
        pass

    if message_user_rejects_refund_wants_tracking(comb):
        return _apply_order_tracking_to_route(out, "user_correction")

    try:
        from services.account_list_semantics import (
            detect_account_list_followup_in_chat,
            turn_requests_purchase_history_in_chat,
        )

        if detect_account_list_followup_in_chat(
            original_msg, msg_en, conversation_context
        ) or turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            return out
    except ImportError:
        pass

    olk = (out.get("order_lookup_kind") or "").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    if olk in ("track", "tracking") or rh == "order_tracking_api":
        return _apply_order_tracking_to_route(out, "ai_route_field")

    if message_user_wants_order_tracking(comb, conversation_context):
        return _apply_order_tracking_to_route(out, "text_signal")

    try:
        from services.order_details_flow import understand_single_order_request

        sub = understand_single_order_request(
            original_msg, msg_en, conversation_context, ai_route=out
        )
        if (sub.get("goal") or "").strip() == "track_single_order":
            return _apply_order_tracking_to_route(out, "order_intent")
    except ImportError:
        pass

    um = f" {(out.get('user_meaning') or '').lower()} "
    if any(x in um for x in _TRACKING_MEANING):
        return _apply_order_tracking_to_route(out, "ai_meaning")

    return out


def tracking_blocks_refund_meaning(blob: str) -> bool:
    """True when meaning is shipment track — must not promote refund_status."""
    if not blob.strip():
        return False
    if any(m in blob for m in _TRACKING_EXCLUDE_FROM_REFUND):
        return True
    if "track" in blob and "order" in blob:
        return True
    return False
