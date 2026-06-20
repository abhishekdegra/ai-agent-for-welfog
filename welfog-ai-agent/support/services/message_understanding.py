"""
Semantic message understanding — Groq ai_brain_route is the primary classifier.

Code only fixes clear conflicts (PIN vs order id, wishlist vs order history, other company).
Do NOT re-route with large phrase lists — trust AI intent when set.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from utils.reasoning_log import log_reasoning


def apply_embedded_identifiers_from_message(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
) -> dict:
    """
    When pincode / order id / product id is already in the user's message, use it immediately.
    Do not route to 'ask again' flows.
    """
    if not route_data:
        return route_data or {}

    from utils.helpers import (
        _text_has_delivery_or_order_area_intent,
        _text_has_delivery_serviceability_intent,
        _text_has_pincode_delivery_intent,
        _text_has_refund_or_return_intent,
        _text_is_order_tracking_intent,
        _text_is_refund_return_policy_howto,
        message_needs_policy_answer,
        _user_denies_pincode_insists_order_id,
        _user_explicitly_asks_payment_status,
        extract_embedded_query_identifiers,
    )

    out = dict(route_data)
    comb = f"{original_msg} {msg_en}".strip()
    if _text_is_refund_return_policy_howto(comb) or message_needs_policy_answer(comb):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: return/refund how-to or policy — KB (not stale live refund API).")
        return out

    ids = extract_embedded_query_identifiers(
        original_msg, msg_en, conversation_context, ai_route=out
    )

    oid = (ids.get("order_id") or "").strip()
    if oid and ids.get("numeric_context") == "order_id":
        live_intent = (out.get("intent") or "order").strip().lower()
        if live_intent not in ("order", "refund", "payment"):
            if _text_has_refund_or_return_intent(comb):
                live_intent = "refund"
            elif _user_explicitly_asks_payment_status(comb):
                live_intent = "payment"
            else:
                live_intent = "order"
        try:
            from services.order_details_flow import _fast_order_lookup_goal

            fast_goal = _fast_order_lookup_goal(
                original_msg, msg_en, conversation_context, out
            )
        except ImportError:
            fast_goal = ""
        if fast_goal == "order_invoice":
            out["intent"] = "order"
            out["order_lookup_kind"] = "invoice"
            out["route_handler"] = "order_details_api"
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["data_channel"] = "live_api"
            out["extracted_pincode"] = ""
            log_reasoning(
                f"Query has Order ID {oid} — invoice/details API (not generic track)."
            )
            return out
        if fast_goal == "order_details":
            out["intent"] = "order"
            out["order_lookup_kind"] = "details"
            out["route_handler"] = "order_details_api"
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["data_channel"] = "live_api"
            out["extracted_pincode"] = ""
            log_reasoning(
                f"Query has Order ID {oid} — order details API (address/amount/payment)."
            )
            return out
        if _text_has_refund_or_return_intent(comb):
            try:
                from services.refund_status_semantics import (
                    current_turn_wants_personal_refund_status,
                )

                if current_turn_wants_personal_refund_status(
                    original_msg, msg_en, conversation_context, ai_route=out, allow_llm=False
                ):
                    out["intent"] = "refund"
                    out["order_lookup_kind"] = "refund_status"
                    out["route_handler"] = "refund_status_api"
                    out["needs_order_id"] = True
                    out["numeric_context"] = "order_id"
                    out["data_channel"] = "live_api"
                    out["extracted_pincode"] = ""
                    log_reasoning(
                        f"Query has Order ID {oid} — refund status API (not order list/track)."
                    )
                    return out
            except ImportError:
                pass
        if (
            _text_is_order_tracking_intent(comb)
            or _user_explicitly_asks_payment_status(comb)
            or _user_denies_pincode_insists_order_id(comb)
            or live_intent in ("order", "refund", "payment")
        ):
            if _text_has_refund_or_return_intent(comb) and live_intent != "refund":
                live_intent = "refund"
            out["intent"] = live_intent
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["data_channel"] = "live_api"
            out["extracted_pincode"] = ""
            out.pop("route_handler", None)
            log_reasoning(f"Query already has Order ID {oid} — live {live_intent} (no separate ask).")
            return out

    from utils.helpers import message_requests_new_area_without_pin

    pin = (ids.get("pincode") or "").strip()
    if pin and message_requests_new_area_without_pin(comb, conversation_context=conversation_context):
        pin = ""
        ids["pincode"] = ""
        ids["numeric_context"] = "none"
    if pin:
        delivery_q = (
            _text_has_pincode_delivery_intent(comb, conversation_context)
            or _text_has_delivery_serviceability_intent(comb, conversation_context)
            or _text_has_delivery_or_order_area_intent(comb)
            or (out.get("intent") or "") == "pincode_check"
            or ids.get("numeric_context") == "pincode"
        )
        if delivery_q and (out.get("intent") or "") not in ("product", "wishlist", "order_history"):
            out["intent"] = "pincode_check"
            out["extracted_pincode"] = pin
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["numeric_context"] = "pincode"
            out["data_channel"] = "live_api"
            out["is_welfog_related"] = True
            out.pop("route_handler", None)
            log_reasoning(f"Query already has PIN {pin} — delivery API (no separate ask).")
            return out

    pid = (ids.get("product_id") or "").strip()
    if pid and ids.get("numeric_context") == "product_id":
        out["intent"] = "product"
        out["search_query"] = f"pro_id {pid}"
        out["needs_order_id"] = False
        out["data_channel"] = "catalog"
        out["is_welfog_related"] = True
        log_reasoning(f"Query already has product id {pid} — catalog lookup.")
        return out

    return out


_NON_CATALOG_INTENTS = frozenset(
    {
        "wishlist",
        "order_history",
        "order",
        "refund",
        "payment",
        "seller",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
        "out_of_domain",
    }
)


def route_semantic_lock_active(route: dict | None) -> bool:
    """
    True when Groq + corrections chose a specific channel/intent.
    Keyword layers must not re-route to product catalog in that case.
    """
    if not route:
        return False
    channel = (route.get("data_channel") or "").strip().lower()
    if channel in ("live_api", "kb", "none"):
        return True
    intent = (route.get("intent") or "").strip().lower()
    if intent in _NON_CATALOG_INTENTS:
        return True
    if (route.get("route_handler") or "").strip():
        return True
    return False


def apply_ai_route_corrections(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
) -> dict:
    """Trust ai_brain_route output; apply only universal safety rules."""
    result = _apply_ai_route_corrections_body(
        route_data, original_msg, msg_en, conversation_context
    )
    if isinstance(result, dict):
        try:
            from services.account_list_semantics import account_list_route_is_locked
            from services.conversation_scope import turn_blocks_product_catalog
            from services.product_catalog_resolver import (
                apply_product_catalog_to_route,
                product_catalog_route_is_locked,
            )

            skip_catalog_refresh = bool(
                result.get("_universal_brain_route")
                or account_list_route_is_locked(result)
                or product_catalog_route_is_locked(result)
            )
            if not skip_catalog_refresh and not turn_blocks_product_catalog(
                original_msg, msg_en, conversation_context, ai_route=result
            ):
                refreshed = apply_product_catalog_to_route(
                    result, original_msg, msg_en, conversation_context=conversation_context
                )
                if product_catalog_route_is_locked(refreshed):
                    result = refreshed
        except ImportError:
            pass
    if isinstance(result, dict) and not result.get("_routing_corrections_done"):
        result = dict(result)
        result["_routing_corrections_done"] = True
    return result


def _apply_ai_route_corrections_body(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
) -> dict:
    """
    Internal route correction pass — use apply_ai_route_corrections() so
    _routing_corrections_done is set once per HTTP turn.
    """
    if not route_data:
        return route_data or {}

    from services.ai_route_semantics import coerce_route_str, _normalize_llm_route

    out = _normalize_llm_route(dict(route_data))
    if out.get("_routing_corrections_done"):
        try:
            from services.chat_flow_telemetry import skip_step

            skip_step("apply_ai_route_corrections", "already applied")
        except ImportError:
            pass
        return out
    try:
        from services.chat_flow_telemetry import record_route_step

        record_route_step("apply_ai_route_corrections")
    except ImportError:
        pass
    intent = coerce_route_str(out.get("intent"), "general")
    numeric = coerce_route_str(out.get("numeric_context"), "none").lower()
    extracted_pin = coerce_route_str(out.get("extracted_pincode"), "")
    reuse = coerce_route_str(out.get("reuse_user_value_from_chat"), "").lower()

    from utils.helpers import (
        _conversation_awaiting_order_id,
        _conversation_bot_asked_for_pincode,
        _conversation_in_pincode_delivery_flow,
        _user_switches_pincode_subject,
        extract_latest_order_id_from_user_conversation,
        extract_latest_pincode_from_user_conversation,
        extract_order_id,
        extract_pincode_preferred_from_message,
        resolve_pincode_for_check,
        message_is_user_feedback_or_closing,
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        message_is_wishlist_like_request,
        _text_asks_how_to_view_order_history,
        _user_asks_order_history_navigation_help,
        message_is_past_purchase_list_request,
    )

    comb = f"{original_msg} {msg_en}"

    if out.get("_universal_brain_route"):
        rh_pre = (out.get("route_handler") or "").strip().lower()
        if rh_pre == "refund_status_api" or (out.get("order_lookup_kind") or "").strip().lower() == "refund_status":
            out["_routing_corrections_done"] = True
            log_reasoning("Universal brain: refund API locked — skip safety LLM stack.")
            return out
        try:
            from services.account_list_semantics import account_list_route_is_locked

            if account_list_route_is_locked(out):
                out["_routing_corrections_done"] = True
                log_reasoning(
                    f"Universal brain: account-list locked "
                    f"({out.get('account_list_kind')}) — skip safety LLM stack."
                )
                return out
        except ImportError:
            pass
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            if product_catalog_route_is_locked(out):
                out["_routing_corrections_done"] = True
                log_reasoning("Universal brain: product catalog locked — skip safety LLM stack.")
                return out
        except ImportError:
            pass
        if (out.get("intent") or "").strip().lower() == "refund":
            try:
                from utils.helpers import _text_is_refund_return_status_lookup

                if _text_is_refund_return_status_lookup(comb, conversation_context) and (
                    extract_order_id(comb) or re.search(r"\b\d{6,}\b", comb)
                ):
                    out["data_channel"] = "live_api"
                    out["route_handler"] = "refund_status_api"
                    out["order_lookup_kind"] = "refund_status"
                    out["needs_order_id"] = True
                    out["numeric_context"] = "order_id"
                    out["run_catalog_search"] = False
                    out["_routing_corrections_done"] = True
                    log_reasoning("Universal brain: refund + order id → live refund API.")
                    return out
            except ImportError:
                pass

    promotions_done = bool(out.get("_turn_promotions_done"))
    try:
        from services.chat_flow_telemetry import ai_route_already_decided, skip_step

        if promotions_done:
            skip_step("route_promotions", "enrich_route_from_llm already ran")
    except ImportError:
        pass

    if not promotions_done:
        from services.product_browse_semantics import promote_product_browse_on_route
        from services.meta_turn_semantics import promote_assistant_intro_on_route
        from services.account_list_semantics import promote_account_list_on_route

        out = promote_account_list_on_route(out, original_msg, msg_en, conversation_context)
        try:
            from services.account_list_semantics import account_list_route_is_locked

            if account_list_route_is_locked(out):
                log_reasoning(
                    f"Account-list locked ({out.get('account_list_kind') or out.get('route_handler')}) "
                    "— skip catalog/pincode safety overrides."
                )
                return out
        except ImportError:
            pass

        out = promote_product_browse_on_route(
            out, original_msg, msg_en, conversation_context=conversation_context
        )
        out = promote_assistant_intro_on_route(out, original_msg, msg_en)
        try:
            from services.account_list_semantics import KIND_NONE, _norm_account_list_kind
            from services.ai_route_semantics import coerce_route_str

            if _norm_account_list_kind(coerce_route_str(out.get("account_list_kind"), KIND_NONE)) == KIND_NONE:
                out = promote_account_list_on_route(out, original_msg, msg_en, conversation_context)
        except ImportError:
            out = promote_account_list_on_route(out, original_msg, msg_en, conversation_context)

        try:
            from services.account_list_semantics import account_list_route_is_locked

            if account_list_route_is_locked(out):
                return out
        except ImportError:
            pass

        from services.semantic_intent import skip_keyword_intent_routes

        if skip_keyword_intent_routes(out):
            from services.ai_route_semantics import (
                correct_delivery_vs_tracking_from_ai_meaning,
                promote_informational_kb_from_ai_meaning,
            )

            out = promote_informational_kb_from_ai_meaning(out)
            out = correct_delivery_vs_tracking_from_ai_meaning(out)
            try:
                from services.conversation_scope import turn_blocks_product_catalog
                from services.product_catalog_resolver import (
                    apply_product_catalog_to_route,
                    product_catalog_route_is_locked,
                )

                if not turn_blocks_product_catalog(
                    original_msg, msg_en, conversation_context, ai_route=out
                ):
                    refreshed = apply_product_catalog_to_route(
                        out, original_msg, msg_en, conversation_context=conversation_context
                    )
                    if product_catalog_route_is_locked(refreshed):
                        out = refreshed
            except ImportError:
                pass

    from services.turn_intent_gate import apply_meta_turn_to_route
    from services.ai_route_semantics import meta_turn_from_route
    from utils.helpers import message_is_welfog_about_request

    if (out.get("meta_kind") or "").strip().lower() == "assistant_intro" and message_is_welfog_about_request(
        comb
    ):
        out["meta_kind"] = "none"
        out.pop("route_handler", None)
        out["data_channel"] = "kb"
        keys = list(out.get("kb_keys") or [])
        for k in ("company", "faqs"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: company/about question — clear mistaken assistant_intro meta.")

    meta_turn = meta_turn_from_route(out)
    if meta_turn:
        if meta_turn.kind == "assistant_intro" and message_is_welfog_about_request(comb):
            log_reasoning("Safety: skip assistant_intro meta — company KB route.")
        else:
            out = apply_meta_turn_to_route(out, meta_turn)
            log_reasoning(
                f"Safety: meta-turn {meta_turn.kind} — structured reply (no catalog/order hijack)."
            )
            return out

    from utils.helpers import (
        _text_is_refund_return_policy_howto,
        message_needs_policy_answer,
        user_turn_qualifies_for_live_order_api,
        message_is_bot_search_complaint,
        message_is_bot_capability_question,
        message_is_user_confused_or_rephrasing_bot,
        _text_has_past_order_complaint_context,
    )

    if message_is_bot_capability_question(comb):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["is_welfog_related"] = True
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "assistant_capability_kb"
        out["continue_previous_topic"] = False
        out.pop("handler", None)
        log_reasoning("Safety: assistant capability question — not catalog search.")
        return out

    if message_is_bot_search_complaint(comb, conversation_context) or (
        message_is_user_confused_or_rephrasing_bot(comb, conversation_context)
        and (
            _text_has_past_order_complaint_context(comb)
            or _text_has_past_order_complaint_context(conversation_context)
            or message_is_bot_search_complaint(comb, conversation_context)
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["is_welfog_related"] = True
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "policy_structured_kb"
        out["continue_previous_topic"] = False
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: user correcting bot / wrong-item thread — refund KB (not product search).")
        return out

    from utils.helpers import _text_is_refund_return_status_lookup

    pincode_delivery_locked = False
    allow_pin_llm = not bool(out.get("_universal_brain_route"))
    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(out):
            allow_pin_llm = False
    except ImportError:
        pass
    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        pincode_delivery_locked = pincode_delivery_route_is_locked(
            out, original_msg, msg_en, conversation_context, allow_llm=allow_pin_llm
        )
    except ImportError:
        pass

    try:
        from services.order_details_flow import (
            ai_route_requests_order_details_lookup,
            promote_order_details_on_route,
            order_details_route_is_locked,
        )

        if pincode_delivery_locked:
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("promote_order_details", "pincode delivery locked")
            except ImportError:
                pass
        elif order_details_route_is_locked(out):
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("promote_order_details", "route already locked")
            except ImportError:
                pass
            return out
        elif ai_route_requests_order_details_lookup(
            out, original_msg, msg_en, conversation_context
        ):
            out = promote_order_details_on_route(
                out, original_msg, msg_en, conversation_context
            )
            olk = (out.get("order_lookup_kind") or "").strip().lower()
            if olk in ("invoice", "order_invoice"):
                log_reasoning(
                    "Safety: personal order invoice — purchase-history-details + download (not KB/catalog)."
                )
            else:
                log_reasoning(
                    "Safety: personal order details — purchase-history-details API (not refund)."
                )
            return out
        if order_details_route_is_locked(out):
            return out
    except ImportError:
        pass

    try:
        from services.order_tracking_semantics import (
            ai_route_requests_order_tracking_lookup,
            promote_order_tracking_on_route,
            order_tracking_route_is_locked,
        )

        if pincode_delivery_locked:
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("promote_order_tracking", "pincode delivery locked")
            except ImportError:
                pass
        elif order_tracking_route_is_locked(out):
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("promote_order_tracking", "route already locked")
            except ImportError:
                pass
            return out
        elif ai_route_requests_order_tracking_lookup(
            out, original_msg, msg_en, conversation_context
        ):
            out = promote_order_tracking_on_route(
                out, original_msg, msg_en, conversation_context
            )
            log_reasoning(
                "Safety: live order tracking — welfog_track API (not refund/details)."
            )
            return out
        if order_tracking_route_is_locked(out):
            return out
    except ImportError:
        pass

    try:
        from services.semantic_intent import (
            ai_route_requests_refund_status_lookup,
            llm_semantic_route_available,
        )

        refund_status_turn = ai_route_requests_refund_status_lookup(
            out, original_msg, msg_en, conversation_context
        )
        if not refund_status_turn and not llm_semantic_route_available(out):
            refund_status_turn = _text_is_refund_return_status_lookup(comb, conversation_context)
    except ImportError:
        refund_status_turn = _text_is_refund_return_status_lookup(comb, conversation_context)

    if refund_status_turn:
        out["intent"] = "refund"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["order_lookup_kind"] = "refund_status"
        out["route_handler"] = "refund_status_api"
        out["answer_strategy"] = "api_then_ai"
        out["run_catalog_search"] = False
        out["search_query"] = ""
        log_reasoning(
            "Safety: personal refund status — live return-request API (not policy KB)."
        )
        return out

    try:
        from services.refund_status_semantics import refund_status_route_is_locked

        if refund_status_route_is_locked(out):
            return out
    except ImportError:
        pass

    policy_override_ok = True
    if route_semantic_lock_active(out) and (out.get("data_channel") or "").strip().lower() == "kb":
        if not _text_is_refund_return_policy_howto(comb):
            um_low = f" {(out.get('user_meaning') or '').lower()} "
            keys_set = {k.lower() for k in (out.get("kb_keys") or [])}
            if keys_set & {"company", "payment", "seller", "privacy", "terms"} and not keys_set <= {
                "refund",
                "faqs",
                "shipping",
                "terms",
            }:
                policy_override_ok = False
            elif not any(
                x in um_low
                for x in ("refund", "return", "damaged", "defective", "wrong item", "replacement")
            ):
                policy_override_ok = False

    if policy_override_ok and (
        _text_is_refund_return_policy_howto(comb)
        or (
            message_needs_policy_answer(comb)
            and not user_turn_qualifies_for_live_order_api(
                original_msg, msg_en, conversation_context, ai_route=out
            )
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: return/refund/wrong-item — KB (before tracking API).")
        return out

    from utils.helpers import (
        _text_is_order_tracking_intent,
        _text_is_undelivered_order_complaint,
        _text_is_order_tracking_only_issue,
        _user_rejects_order_history_wants_tracking,
    )

    if (
        _text_is_order_tracking_only_issue(comb)
        or _user_rejects_order_history_wants_tracking(comb, conversation_context)
    ):
        out["intent"] = "order"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["data_channel"] = "live_api"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning(
            "Safety: undelivered / tracking — live order API (before catalog product)."
        )
        return out

    from utils.helpers import (
        turn_is_catalog_product_lookup,
        extract_product_id,
        _text_is_sku_product_lookup_context,
    )

    try:
        from services.account_list_semantics import account_list_route_is_locked

        account_list_locked = account_list_route_is_locked(out)
    except ImportError:
        account_list_locked = False

    if not account_list_locked and turn_is_catalog_product_lookup(original_msg, msg_en, out):
        pid = extract_product_id(comb)
        out["intent"] = "product"
        out["needs_order_id"] = False
        out["is_welfog_related"] = True
        out["data_channel"] = "catalog"
        out["numeric_context"] = "product_id" if pid else "none"
        out["continue_previous_topic"] = False
        out["route_handler"] = "catalog_pro_id" if pid else "product_ai_flow"
        if pid:
            out["search_query"] = f"pro_id {pid}"
        elif _text_is_sku_product_lookup_context(comb):
            try:
                from services.opensearch_products import _extract_sku_from_text

                sku_val = _extract_sku_from_text(comb)
                if sku_val:
                    out["search_query"] = sku_val
            except ImportError:
                pass
        log_reasoning(
            "Safety: catalog pro_id/SKU/product-id lookup — not Order ID tracking."
        )
        return out

    from utils.helpers import (
        message_is_user_confused_or_rephrasing_bot,
        _text_has_delivery_or_order_area_intent,
        _text_is_pincode_serviceability_question,
    )

    if message_is_user_confused_or_rephrasing_bot(comb, conversation_context):
        try:
            from services.account_list_semantics import (
                account_list_route_is_locked,
                ai_route_requests_wishlist_howto,
            )
            from utils.helpers import message_mentions_wishlist_topic

            skip_pin_rephrase = (
                account_list_locked
                or ai_route_requests_wishlist_howto(
                    out, original_msg, msg_en, conversation_context
                )
                or message_mentions_wishlist_topic(comb)
            )
        except ImportError:
            skip_pin_rephrase = False

        if not skip_pin_rephrase and (
            _conversation_in_pincode_delivery_flow(conversation_context)
            or _text_has_delivery_or_order_area_intent(comb)
            or _text_is_pincode_serviceability_question(comb, conversation_context)
            or (out.get("intent") or "").strip().lower() == "pincode_check"
        ):
            out["intent"] = "pincode_check"
            out["is_welfog_related"] = True
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["numeric_context"] = "none"
            out["data_channel"] = "live_api"
            out.pop("route_handler", None)
            log_reasoning("Safety: user rephrasing delivery question — pincode_check (not payment/KB).")
            return out

    out = apply_embedded_identifiers_from_message(out, original_msg, msg_en, conversation_context)

    from services.semantic_route_guard import reconcile_ai_route_with_semantic_goal

    try:
        from services.chat_flow_telemetry import ai_route_already_decided, skip_step

        if ai_route_already_decided(out):
            skip_step("reconcile_semantic_goal", "main router locked intent")
        else:
            out = reconcile_ai_route_with_semantic_goal(
                out, original_msg, msg_en, conversation_context
            )
    except ImportError:
        out = reconcile_ai_route_with_semantic_goal(
            out, original_msg, msg_en, conversation_context
        )
    try:
        from services.account_list_semantics import promote_account_list_on_route

        if not promotions_done and not out.get("_semantic_override"):
            out = promote_account_list_on_route(
                out, original_msg, msg_en, conversation_context
            )
        elif promotions_done:
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("account_list_repromote", "enrich already ran")
            except ImportError:
                pass
        if (out.get("intent") or "").strip().lower() in ("wishlist", "order_history"):
            return out
    except ImportError:
        pass
    if out.get("_semantic_override"):
        return out

    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list

    if is_deals_request_message(original_msg, msg_en):
        out["intent"] = "deals"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["answer_strategy"] = "live_api_only"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning("Safety: deals/offers request → deals API (not product search).")
        return out

    if message_asks_welfog_categories_list(comb):
        from utils.helpers import _text_requests_category_product_browse

        if _text_requests_category_product_browse(comb):
            from services.welfog_api import get_category_id_from_text

            cid = get_category_id_from_text(comb)
            if cid:
                out["intent"] = "product"
                out["data_channel"] = "catalog"
                out["needs_order_id"] = False
                out["numeric_context"] = "none"
                out["search_query"] = ""
                out["answer_strategy"] = "catalog_only"
                out["continue_previous_topic"] = False
                out.pop("route_handler", None)
                log_reasoning(f"Safety: category product browse → catalog (category_id={cid}).")
                return out
        out["intent"] = "categories"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["answer_strategy"] = "live_api_only"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning("Safety: Welfog categories list/count → categories API.")
        return out

    from utils.helpers import (
        _text_has_refund_or_return_intent,
        _user_denies_pincode_insists_order_id,
        _user_explicitly_asks_payment_status,
        resolve_live_api_intent_from_conversation,
    )

    if _user_denies_pincode_insists_order_id(comb):
        live_intent = resolve_live_api_intent_from_conversation(
            conversation_context,
            ctx_last=None,
            original_msg=original_msg,
            msg_en=msg_en,
            ai_route=out,
        )
        if _text_has_refund_or_return_intent(comb):
            live_intent = "refund"
        elif _user_explicitly_asks_payment_status(comb):
            live_intent = "payment"
        oid = extract_order_id(original_msg, conversation_context) or extract_latest_order_id_from_user_conversation(
            conversation_context, original_msg
        )
        out["intent"] = live_intent
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["extracted_pincode"] = ""
        out["data_channel"] = "live_api"
        out["is_welfog_related"] = True
        out.pop("route_handler", None)
        log_reasoning(
            f"Safety: user corrected not-pincode — live {live_intent}"
            + (f" for Order ID {oid}." if oid else " (await id).")
        )
        return out

    if (out.get("intent") or "") == "pincode_check" and (out.get("extracted_pincode") or "").strip():
        return out

    if _conversation_bot_asked_for_pincode(conversation_context):
        pin_ctx = (
            extract_pincode_preferred_from_message(comb)
            or resolve_pincode_for_check(original_msg, conversation_context, msg_en=msg_en)
            or (original_msg.strip()[:6] if re.fullmatch(r"[1-9]\d{5}", original_msg.strip()) else "")
        )
        if pin_ctx:
            out["intent"] = "pincode_check"
            out["is_welfog_related"] = True
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["extracted_pincode"] = pin_ctx
            out["numeric_context"] = "pincode"
            out["data_channel"] = "live_api"
            out.pop("route_handler", None)
            log_reasoning(
                f"Safety: bot asked for PIN — {pin_ctx} is delivery check (not order ID)."
            )
            return out

    try:
        from services.support_scope import message_mentions_other_company_support

        if message_mentions_other_company_support(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["data_channel"] = "none"
            out["route_handler"] = "other_company_decline"
            out.pop("handler", None)
            log_reasoning("Safety: non-Welfog query — polite decline (no assume→Welfog).")
            return out
    except ImportError:
        pass

    groq_intent = (out.get("intent") or "").strip().lower()
    strict_semantic_mode = (os.getenv("STRICT_AI_INTENT_ROUTING", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    semantic_locked = route_semantic_lock_active(out)

    from utils.helpers import (
        message_clarifies_wishlist_not_order_history,
        message_is_wishlist_like_request,
        _text_asks_wishlist,
        _user_denies_order_history_wants_saved_or_liked,
    )

    from utils.helpers import _text_is_order_id_help_request, _text_is_tracking_howto_request

    if _text_is_order_id_help_request(comb):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["route_handler"] = "order_id_help_kb"
        out["continue_previous_topic"] = False
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "shipping", "welfog_api_order_id"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: where to find Order ID — KB help (not live tracking).")
        return out

    if groq_intent == "order" and _text_is_order_id_help_request(comb):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["route_handler"] = "order_id_help_kb"
        out.pop("handler", None)
        log_reasoning("Safety: order intent + find-ID wording → order_id_help_kb.")
        return out

    from utils.helpers import (
        _text_is_refund_return_policy_howto,
        message_needs_policy_answer,
        user_turn_qualifies_for_live_order_api,
    )

    if _text_is_refund_return_policy_howto(comb) or (
        message_needs_policy_answer(comb)
        and not user_turn_qualifies_for_live_order_api(
            original_msg, msg_en, conversation_context, ai_route=out
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: refund/return policy how-to — KB (overrides stale refund API lock).")
        return out

    from utils.helpers import (
        _text_is_order_tracking_intent,
        _text_is_undelivered_order_complaint,
        _user_rejects_order_history_wants_tracking,
    )

    if (
        _text_is_order_tracking_intent(comb)
        or _text_is_undelivered_order_complaint(comb)
        or _user_rejects_order_history_wants_tracking(comb, conversation_context)
    ):
        out["intent"] = "order"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["data_channel"] = "live_api"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning(
            "Safety: undelivered order / wants tracking — live order API (not purchase-history list)."
        )
        return out

    from utils.helpers import _current_turn_has_order_id, turn_is_catalog_product_lookup

    bare_turn = (original_msg or msg_en or "").strip()
    if (
        _current_turn_has_order_id(original_msg, msg_en)
        and re.fullmatch(r"\s*[0-9]{4,20}\s*", bare_turn)
        and not turn_is_catalog_product_lookup(original_msg, msg_en, out)
    ):
        out["intent"] = "order"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["data_channel"] = "live_api"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning("Safety: bare Order ID line — live tracking API.")
        return out

    reasoning_low_lock = f" {(out.get('reasoning') or '').lower()} "
    if groq_intent == "order_history" and (
        message_is_wishlist_like_request(comb)
        or message_clarifies_wishlist_not_order_history(comb)
        or _user_denies_order_history_wants_saved_or_liked(comb)
        or any(
            x in reasoning_low_lock
            for x in (
                "saved product", "saved products", "liked product", "liked products",
                "not order history", "not purchase history", "wishlist",
            )
        )
    ):
        out["intent"] = "wishlist"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "live_api"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning(
            "Safety: saved/liked products — wishlist API (overrides mistaken order_history AI lock)."
        )
        return out

    if (
        strict_semantic_mode
        and semantic_locked
        and not out.get("_semantic_override")
        and groq_intent in (
            "product",
            "order",
            "order_history",
            "wishlist",
            "refund",
            "payment",
            "seller",
            "pincode_check",
            "deals",
            "categories",
            "category_feed",
            "out_of_domain",
        )
    ):
        log_reasoning(f"Strict semantic lock: trusting AI intent={groq_intent} (minimal heuristic overrides).")
        return out

    reasoning_low_early = f" {(out.get('reasoning') or '').lower()} "
    if "assume" in reasoning_low_early and any(
        b in reasoning_low_early
        for b in ("amazon", "flipkart", "myntra", "meesho", "other company", "another company")
    ):
        out["intent"] = "out_of_domain"
        out["is_welfog_related"] = False
        out["search_query"] = ""
        out["data_channel"] = "none"
        out["route_handler"] = "other_company_decline"
        log_reasoning("Safety: AI reasoning tried to assume external→Welfog — blocked.")
        return out

    if message_is_user_feedback_or_closing(comb):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["is_welfog_related"] = True
        out["search_query"] = ""
        out["numeric_context"] = "none"
        log_reasoning("Safety: user feedback / thanks — not Order ID or tracking.")
        return out

    from utils.helpers import (
        _turn_is_catalog_product_request,
        _text_has_product_shopping_intent,
        extract_product_search_query,
    )

    if (not semantic_locked) and (_text_has_product_shopping_intent(comb) or _turn_is_catalog_product_request(comb)):
        out["intent"] = "product"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = extract_product_search_query(original_msg, msg_en, "") or msg_en
        out["data_channel"] = "catalog"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning(
            "Safety: product shopping (incl. conversational browse) — catalog before KB/placement lock."
        )
        return out

    from utils.helpers import (
        _message_has_catalog_product_signal,
        _message_overrides_placement_followup,
        _text_has_delivery_or_order_area_intent,
        _text_has_delivery_serviceability_intent,
        _text_has_explicit_how_to_place_order,
        _text_has_order_placement_intent,
        _text_has_pincode_delivery_intent,
        _text_has_product_shopping_intent,
        _text_needs_order_id_for_refund_or_payment,
        message_is_casual_offtopic_not_shopping,
    )

    groq_intent = (out.get("intent") or "").strip().lower()

    from utils.helpers import _text_is_pincode_serviceability_question
    from services.semantic_intent import skip_keyword_intent_routes
    from services.ai_route_semantics import correct_delivery_vs_tracking_from_ai_meaning

    if skip_keyword_intent_routes(out):
        out = correct_delivery_vs_tracking_from_ai_meaning(out)
        if (out.get("intent") or "").strip().lower() == "pincode_check":
            return out
    elif _text_is_pincode_serviceability_question(comb, conversation_context):
        out["intent"] = "pincode_check"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["run_catalog_search"] = False
        out["numeric_context"] = "pincode"
        out["data_channel"] = "live_api"
        out.pop("route_handler", None)
        pin_turn = extract_pincode_preferred_from_message(comb)
        if pin_turn:
            out["extracted_pincode"] = pin_turn
        log_reasoning(
            "Keyword failsafe: delivery/serviceability — pincode_check (LLM unavailable)."
        )
        return out

    pin_turn = extract_pincode_preferred_from_message(comb)
    if pin_turn and (
        _text_has_pincode_delivery_intent(comb, conversation_context)
        or _text_has_delivery_serviceability_intent(comb, conversation_context)
        or _conversation_in_pincode_delivery_flow(conversation_context)
        or re.search(
            r"\b(ispe|isme|is\s+par|is\s+per|uspe|us\s+par|yahan|yaha|idhar)\b",
            comb.lower(),
        )
    ):
        out["intent"] = "pincode_check"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["extracted_pincode"] = pin_turn
        out["numeric_context"] = "pincode"
        out["data_channel"] = "live_api"
        out.pop("route_handler", None)
        log_reasoning(f"Safety: PIN {pin_turn} delivery follow-up — not product search.")
        return out

    from utils.helpers import (
        _text_asks_welfog_fees_or_charges,
        _text_has_seller_login_problem_intent,
        _user_complains_bot_gave_wrong_topic,
        _user_seller_issue_still_unresolved,
        _conversation_in_seller_support_flow,
    )

    if _text_asks_welfog_fees_or_charges(comb):
        out["intent"] = "general"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "dynamic_kb"
        out["kb_keys"] = ["payment", "faqs"]
        out.pop("handler", None)
        log_reasoning("Safety: service/platform fees — payment KB (not greeting).")
        return out

    from utils.helpers import (
        _text_asks_customer_care_contact,
        _text_asks_short_video_content_rules,
    )

    if _text_asks_customer_care_contact(comb):
        out["intent"] = "general"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "customer_care_kb"
        out["kb_keys"] = ["support"]
        out.pop("handler", None)
        log_reasoning("Safety: customer-care contact — support KB (not short-video thread).")
        return out

    if _text_asks_short_video_content_rules(comb, conversation_context):
        out["intent"] = "general"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "dynamic_kb"
        out["kb_keys"] = ["terms", "seller", "privacy"]
        out.pop("handler", None)
        log_reasoning("Safety: short video / shorts rules — terms+seller KB (KB-only, concise).")
        return out

    if (
        _text_has_seller_login_problem_intent(comb, conversation_context)
        or _user_seller_issue_still_unresolved(comb, conversation_context)
        or _user_complains_bot_gave_wrong_topic(comb)
        or (
            _conversation_in_seller_support_flow(conversation_context)
            and groq_intent == "seller"
        )
    ):
        out["intent"] = "seller"
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "dynamic_kb"
        out.pop("handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("seller", "support", "faqs"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: seller login/support thread — seller KB (not order history).")
        return out

    stale_placement_lock = (
        (out.get("route_handler") or "") == "order_placement_kb"
        or (
            out.get("continue_previous_topic")
            and _message_overrides_placement_followup(comb)
            and not _text_has_explicit_how_to_place_order(comb)
        )
    )
    if stale_placement_lock and _message_overrides_placement_followup(comb) and not _text_has_explicit_how_to_place_order(comb):
        out["intent"] = "refund"
        out["needs_order_id"] = bool(_text_needs_order_id_for_refund_or_payment(comb))
        out["is_welfog_related"] = True
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("refund", "faqs", "shipping"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: return/refund after placement thread — AI+KB (not stale placement card).")
        return out

    from utils.helpers import (
        _message_has_app_navigation_intent,
        _text_asks_to_view_purchase_or_order_history,
        message_denies_wishlist_wants_order_history,
        message_wants_order_history_app_navigation,
    )

    if (
        message_wants_order_history_app_navigation(comb, conversation_context)
        or _text_asks_to_view_purchase_or_order_history(comb)
        or (
            message_denies_wishlist_wants_order_history(comb)
            and _message_has_app_navigation_intent(comb)
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "order_history_howto_kb"
        out["continue_previous_topic"] = False
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "welfog_api_order_history"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: order history view/how-to — KB (before placement override).")
        return out

    if (not semantic_locked) and (
        _text_has_explicit_how_to_place_order(comb) or (
        _text_has_order_placement_intent(comb)
        and groq_intent not in ("product", "order_history", "wishlist")
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["is_welfog_related"] = True
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "order_placement_kb"
        out.pop("handler", None)
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "shipping"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning(
            "Safety: how to place order (explicit) — KB steps before product/tracking overrides."
        )
        return out

    # --- Live API intents BEFORE product catalog (wishlist ≠ product search) ---
    try:
        from services.support_scope import message_mentions_other_company_support

        if message_mentions_other_company_support(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["search_query"] = ""
            out["data_channel"] = "none"
            out["route_handler"] = "other_company_decline"
            log_reasoning("Safety: external app/shop feature — not Welfog wishlist/API.")
            return out
    except ImportError:
        pass

    reasoning_low = f" {(out.get('reasoning') or '').lower()} "
    ai_means_wishlist = any(
        x in reasoning_low
        for x in (
            " saved ", " liked ", " wishlist", " heart ", " favourite", " favorite",
            " pasand ", " saved or liked", " liked products", " saved products",
        )
    ) and not any(
        x in reasoning_low
        for x in (" purchased ", " bought ", " order history", " placed order", " mangaya")
    )

    try:
        from services.opensearch_products import is_price_or_rating_browse_turn
        from services.catalog_spec_semantics import user_mentions_rating_this_turn

        rating_or_price_browse = (
            is_price_or_rating_browse_turn(comb)
            or user_mentions_rating_this_turn(comb)
        )
    except ImportError:
        rating_or_price_browse = False

    if (
        not rating_or_price_browse
        and (
            intent == "wishlist"
            or _text_asks_wishlist(comb)
            or message_is_wishlist_like_request(comb)
            or (groq_intent == "product" and ai_means_wishlist)
        )
    ):
        if not _text_asks_how_to_view_wishlist(comb):
            out["intent"] = "wishlist"
            out["needs_order_id"] = False
            out["is_welfog_related"] = True
            out["search_query"] = ""
            out["numeric_context"] = "none"
            out["data_channel"] = "live_api"
            out["continue_previous_topic"] = False
            out.pop("route_handler", None)
            log_reasoning("Safety: wishlist / liked items — wishlists API (before product search).")
            return out
    from utils.helpers import (
        _message_has_app_navigation_intent,
        message_asks_my_welfog_purchases,
        message_clarifies_wishlist_not_order_history,
        message_denies_wishlist_wants_order_history,
        message_mentions_wishlist_topic,
    )

    from utils.helpers import _turn_blocks_wishlist_howto_routing

    if not _turn_blocks_wishlist_howto_routing(comb) and (
        not message_denies_wishlist_wants_order_history(comb)
        and (
            _text_asks_how_to_view_wishlist(comb, conversation_context)
            or message_clarifies_wishlist_not_order_history(comb)
            or (
                message_is_wishlist_like_request(comb)
                and _message_has_app_navigation_intent(comb)
            )
            or (
                message_mentions_wishlist_topic(comb)
                and not _text_asks_wishlist(comb)
                and not message_asks_my_welfog_purchases(comb)
            )
        )
    ):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["data_channel"] = "kb"
        out["route_handler"] = "wishlist_howto_kb"
        out["continue_previous_topic"] = False
        out["kb_keys"] = list(out.get("kb_keys") or []) + ["faqs"]
        log_reasoning("Safety: wishlist how-to — KB steps (before order-history overrides).")
        return out

    from utils.helpers import (
        _text_wants_order_history_list_in_chat,
        _user_asks_order_history_navigation_help,
        message_needs_live_single_order_lookup,
    )

    if (
        _text_wants_order_history_list_in_chat(comb, conversation_context)
        and not message_needs_live_single_order_lookup(
            original_msg, msg_en, conversation_context, ai_route=out
        )
    ):
        out["intent"] = "order_history"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "live_api"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning("Safety: order history in chat — purchase-history API (before product search).")
        return out

    if _user_asks_order_history_navigation_help(comb, conversation_context):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["route_handler"] = "order_history_howto_kb"
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "welfog_api_order_history"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: order history how-to — KB steps (before product search).")
        return out

    if intent == "order_history" or (
        message_is_past_purchase_list_request(comb)
        and not _text_asks_how_to_view_order_history(comb)
        and not message_needs_live_single_order_lookup(
            original_msg, msg_en, conversation_context, ai_route=out
        )
    ):
        out["intent"] = "order_history"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "live_api"
        out["continue_previous_topic"] = False
        out.pop("route_handler", None)
        log_reasoning("Safety: past purchases — order history API (before product search).")
        return out

    if groq_intent == "product" and out.get("is_welfog_related", True) and not route_semantic_lock_active(out):
        sq = (out.get("search_query") or "").strip()
        if sq or _message_has_catalog_product_signal(comb) or _text_has_product_shopping_intent(comb):
            out["data_channel"] = "catalog"
            out["needs_order_id"] = False
            out.pop("route_handler", None)
            log_reasoning("Trust Groq product intent — skip keyword placement/off-topic overrides.")
            return out

    if message_is_casual_offtopic_not_shopping(comb):
        if not (
            _message_has_catalog_product_signal(comb)
            or _text_has_product_shopping_intent(comb)
        ):
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["data_channel"] = "none"
            log_reasoning("Safety: personal/off-topic — not Welfog shopping.")
            return out

    from utils.helpers import message_is_seller_on_welfog_request

    if groq_intent == "seller" or message_is_seller_on_welfog_request(comb):
        out["intent"] = "seller"
        out["data_channel"] = "kb"
        out["needs_order_id"] = False
        out["search_query"] = ""
        keys = list(out.get("kb_keys") or [])
        for k in ("seller", "faqs", "support"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Trust Groq seller intent — seller KB.")
        return out

    from utils.helpers import (
        _text_has_product_shopping_intent,
        message_asks_other_company_social_media,
        message_asks_welfog_social_media,
    )

    if not _text_has_product_shopping_intent(comb):
        if message_asks_other_company_social_media(comb, conversation_context=conversation_context):
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["data_channel"] = "none"
            out["route_handler"] = "other_company_social_decline"
            log_reasoning("Safety: other company social media — polite decline.")
            return out
        if message_asks_welfog_social_media(comb, conversation_context=conversation_context):
            out["intent"] = "general"
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["data_channel"] = "kb"
            out["route_handler"] = "welfog_social_kb"
            keys = list(out.get("kb_keys") or [])
            if "company" not in keys:
                keys.append("company")
            out["kb_keys"] = keys
            log_reasoning("Safety: Welfog official social links from company KB.")
            return out

    if intent in ("order_history", "wishlist", "order", "order_id"):
        out["search_query"] = ""

    if _text_wants_order_history_list_in_chat(comb, conversation_context):
        if not (
            (out.get("intent") or "").strip().lower() == "wishlist"
            or message_is_wishlist_like_request(comb)
            or _text_asks_wishlist(comb)
        ):
            out["intent"] = "order_history"
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["numeric_context"] = "none"
            out["data_channel"] = "live_api"
            out.pop("route_handler", None)
            log_reasoning("Safety: order list in chat overrides awaiting-order-id lock.")
            return out

    if _user_asks_order_history_navigation_help(comb, conversation_context):
        out["intent"] = "general"
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "kb"
        out["route_handler"] = "order_history_howto_kb"
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "welfog_api_order_history"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning("Safety: history how-to overrides awaiting-order-id lock.")
        return out

    try:
        from services.conversation_thread_semantics import (
            apply_thread_goal_to_route,
            resolve_explicit_turn_goal_from_message,
        )

        explicit_goal = resolve_explicit_turn_goal_from_message(
            original_msg, msg_en, conversation_context, out, allow_llm=True
        )
        if explicit_goal in (
            "refund_status",
            "track",
            "order_invoice",
            "order_details",
            "payment",
        ):
            out = apply_thread_goal_to_route(out, explicit_goal, "explicit_turn_goal")
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            oid = extract_order_id(original_msg, conversation_context) or extract_latest_order_id_from_user_conversation(
                conversation_context, original_msg
            )
            log_reasoning(
                f"Safety: explicit turn goal={explicit_goal}"
                + (f" order_id={oid}." if oid else " (await id).")
            )
            return out
    except ImportError:
        pass

    if _conversation_awaiting_order_id(conversation_context) and not message_is_user_feedback_or_closing(
        comb
    ):
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            out["intent"] = "wishlist"
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["numeric_context"] = "none"
            out["data_channel"] = "live_api"
            log_reasoning("Safety: wishlist overrides stale awaiting-order-id state.")
            return out
        if message_is_past_purchase_list_request(comb):
            out["intent"] = "order_history"
            out["needs_order_id"] = False
            out["search_query"] = ""
            out["numeric_context"] = "none"
            out["data_channel"] = "live_api"
            log_reasoning("Safety: purchase list overrides stale awaiting-order-id state.")
            return out
        thread = ""
        try:
            from services.conversation_thread_semantics import (
                apply_thread_goal_to_route,
                infer_order_thread_goal,
            )

            thread = infer_order_thread_goal(
                conversation_context,
                comb,
                ai_route=out,
                allow_llm=True,
            )
            if thread in ("refund_status", "track", "order_invoice", "order_details", "payment"):
                out = apply_thread_goal_to_route(out, thread, "awaiting_order_id_thread")
        except ImportError:
            thread = ""
        if not thread:
            tail = (conversation_context or "")[-3500:].lower()
            if any(
                x in tail
                for x in (
                    "refund", "return daal", "return daale", "refund nhi", "refund nahi",
                    "paise wapas", "paise nahi", "money back",
                )
            ):
                out["intent"] = "refund"
                out["order_lookup_kind"] = "refund_status"
                out["route_handler"] = "refund_status_api"
                out["data_channel"] = "live_api"
            elif "payment" in tail and "refund" not in tail:
                out["intent"] = "payment"
            else:
                out["intent"] = "order"
                out["order_lookup_kind"] = "track"
                out["route_handler"] = "order_tracking_api"
                out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["extracted_pincode"] = ""
        oid = extract_order_id(original_msg, conversation_context) or extract_latest_order_id_from_user_conversation(
            conversation_context, original_msg
        )
        if oid:
            log_reasoning(
                f"Safety: bot asked Order ID — live {out.get('intent')} "
                f"({out.get('order_lookup_kind') or '-'}) for {oid}."
            )
        else:
            log_reasoning("Safety: bot asked Order ID — wait for id.")
        return out

    if numeric == "pincode" or intent == "pincode_check":
        from utils.helpers import _digits_in_message_are_order_id_not_pincode

        if _digits_in_message_are_order_id_not_pincode(comb, conversation_context):
            live_intent = resolve_live_api_intent_from_conversation(
                conversation_context,
                ctx_last=None,
                original_msg=original_msg,
                msg_en=msg_en,
                ai_route=out,
            )
            if _text_has_refund_or_return_intent(comb):
                live_intent = "refund"
            out["intent"] = live_intent
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["extracted_pincode"] = ""
            out["data_channel"] = "live_api"
            log_reasoning(f"Safety: order-id context overrides pincode_check → live {live_intent}.")
            return out
        out["intent"] = "pincode_check"
        out["needs_order_id"] = False
        out["search_query"] = ""
        pin = resolve_pincode_for_check(
            original_msg, conversation_context, ai_extracted=extracted_pin, msg_en=msg_en
        )
        if not pin:
            pin = extract_pincode_preferred_from_message(f"{original_msg} {msg_en}")
        if pin:
            out["extracted_pincode"] = pin
        if intent == "order":
            log_reasoning("Safety: AI numeric_context=pincode → pincode_check not order.")
        return out

    if intent == "order" and numeric == "pincode":
        out["intent"] = "pincode_check"
        out["needs_order_id"] = False
        log_reasoning("Safety: order+pincode numeric → pincode_check.")
        return out

    from utils.helpers import message_requests_new_area_without_pin

    if _user_switches_pincode_subject(f"{original_msg} {msg_en}"):
        out["continue_previous_topic"] = False
        out["intent"] = "pincode_check"
        out["needs_order_id"] = False
        pin = extract_pincode_preferred_from_message(f"{original_msg} {msg_en}")
        if pin:
            out["extracted_pincode"] = pin
        log_reasoning(f"Safety: user corrected PIN → {pin or 'ask'}.")
        return out

    if message_requests_new_area_without_pin(comb, conversation_context=conversation_context):
        out["intent"] = "pincode_check"
        out["extracted_pincode"] = ""
        out["needs_order_id"] = False
        out["search_query"] = ""
        out["numeric_context"] = "none"
        out["data_channel"] = "live_api"
        out["is_welfog_related"] = True
        out.pop("route_handler", None)
        log_reasoning("New area/city in message — do not reuse previous PIN from history.")
        return out

    if reuse == "pincode":
        latest_pin = extract_latest_pincode_from_user_conversation(conversation_context, original_msg)
        if latest_pin:
            out["intent"] = "pincode_check"
            out["extracted_pincode"] = latest_pin
            out["needs_order_id"] = False
            out["search_query"] = ""
            log_reasoning(f"AI reuse_user_value_from_chat → PIN {latest_pin}.")
        return out

    if reuse == "order_id":
        latest_oid = extract_latest_order_id_from_user_conversation(conversation_context, original_msg)
        if latest_oid:
            out["intent"] = "order"
            out["needs_order_id"] = True
            log_reasoning(f"AI reuse_user_value_from_chat → Order ID {latest_oid}.")
        return out

    from utils.helpers import _user_references_prior_submission

    if (
        _user_references_prior_submission(comb)
        and not message_requests_new_area_without_pin(comb, conversation_context=conversation_context)
        and (
            intent in ("pincode_check", "general", "order")
            or numeric == "pincode"
            or _conversation_in_pincode_delivery_flow(conversation_context)
        )
    ):
        latest_pin = extract_latest_pincode_from_user_conversation(conversation_context, original_msg)
        if latest_pin:
            out["intent"] = "pincode_check"
            out["extracted_pincode"] = latest_pin
            out["needs_order_id"] = False
            out["search_query"] = ""
            log_reasoning(f"User said PIN already sent — reuse {latest_pin}.")
            return out

    pin_embedded = extract_pincode_preferred_from_message(comb)
    from utils.helpers import _digits_in_message_are_order_id_not_pincode

    if pin_embedded and not _digits_in_message_are_order_id_not_pincode(comb, conversation_context) and (
        _text_has_pincode_delivery_intent(comb, conversation_context)
        or _text_has_delivery_serviceability_intent(comb, conversation_context)
        or _text_has_delivery_or_order_area_intent(comb)
        or re.search(r"\b(per|par|pe)\b", comb.lower())
    ) and intent not in ("product", "order_history", "wishlist"):
        out["intent"] = "pincode_check"
        out["extracted_pincode"] = pin_embedded
        out["needs_order_id"] = False
        out["numeric_context"] = "pincode"
        out["data_channel"] = "live_api"
        log_reasoning(f"PIN {pin_embedded} in query → pincode_check (no re-ask).")
        return out

    return out


def merge_ai_route_into_ai_data(ai_route: dict, ai_data: dict) -> None:
    """Copy Groq routing into execution dict — do not let keyword layers override intent."""
    if not ai_route or not ai_data:
        return
    ai_data["_ai_routed"] = True
    for key in (
        "intent",
        "is_welfog_related",
        "search_query",
        "needs_order_id",
        "extracted_pincode",
        "data_channel",
        "route_handler",
        "kb_keys",
        "numeric_context",
        "continue_previous_topic",
        "reasoning",
    ):
        if key in ai_route and ai_route[key] is not None:
            ai_data[key] = ai_route[key]


def should_skip_response_cache_for_message(
    route_data: Optional[dict],
    *,
    computed_fresh: bool = False,
) -> bool:
    """Every user message gets fresh AI routing; API data is always live."""
    return True
