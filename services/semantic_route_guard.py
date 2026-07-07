"""
Semantic user goal vs AI route — meaning wins over mistaken Groq intent / strict lock.

Works across languages via multilingual_intent + lightweight complaint detectors.
Not a phrase-list router for every message: only overrides when goal clearly conflicts with AI JSON.
"""
from __future__ import annotations

import os
import re

from utils.reasoning_log import log_reasoning


def semantic_guard_conflicts_only() -> bool:
    """When True (default with strict AI routing), keyword guard only overrides clear AI mistakes."""
    mode = (os.getenv("SEMANTIC_GUARD_MODE", "") or "").strip().lower()
    if mode in ("always", "full", "keyword"):
        return False
    if mode in ("conflicts_only", "conflicts", "ai_first"):
        return True
    from services.semantic_intent import strict_ai_semantic_mode

    return strict_ai_semantic_mode()


def ai_route_aligns_with_semantic_goal(goal: str, route: dict) -> bool:
    """True when Groq routing already matches inferred customer goal — no keyword override."""
    if not goal:
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    rh = (route.get("route_handler") or "").strip().lower()
    needs_oid = bool(route.get("needs_order_id"))

    if goal in ("order_invoice", "order_details"):
        return intent in ("order", "payment") and channel == "live_api"
    if goal == "track_single_order":
        return intent in ("order", "refund", "payment") and channel in ("live_api", "kb", "none")
    if goal == "order_history_list":
        return intent == "order_history" and channel == "live_api" and not needs_oid
    if goal == "wishlist_list":
        return intent == "wishlist" and channel == "live_api" and not needs_oid
    if goal == "wishlist_howto":
        return intent in ("general", "wishlist") and (
            rh == "wishlist_howto_kb" or channel == "kb"
        )
    if goal == "refund_policy":
        return intent in ("general", "refund", "payment", "seller") and channel == "kb"
    if goal == "refund_status":
        return intent in ("refund", "payment", "order") and channel == "live_api"
    if goal == "order_id_help":
        return rh == "order_id_help_kb" or (
            intent == "general" and channel == "kb" and not needs_oid
        )
    if goal == "pincode_delivery":
        return intent == "pincode_check" and channel == "live_api" and not needs_oid
    if goal == "order_history_list":
        if intent in ("product", "deals", "categories", "category_feed"):
            return True
        if channel == "catalog":
            return True
    return True


def infer_customer_semantic_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """What the customer wants THIS turn (empty if unclear — trust AI)."""
    if isinstance(ai_route, dict) and ai_route.get("_universal_brain_route"):
        try:
            from services.chat_flow_telemetry import is_routing_complete

            if is_routing_complete():
                return ""
        except ImportError:
            pass
        scope = (ai_route.get("conversation_scope") or "").strip().lower()
        intent_ub = (ai_route.get("intent") or "").strip().lower()
        channel_ub = (ai_route.get("data_channel") or "").strip().lower()
        if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
            return ""
        if intent_ub in ("general", "out_of_domain", "product"):
            if channel_ub in ("catalog", "none", "kb", ""):
                return ""
        if channel_ub in ("catalog", "live_api"):
            return ""

    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        product_locked = product_catalog_route_is_locked(ai_route)
    except ImportError:
        product_locked = False
    if product_locked:
        return ""
    if not product_locked:
        try:
            from services.location_delivery_resolver import turn_requests_delivery_serviceability

            if turn_requests_delivery_serviceability(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                allow_llm=False,
            ):
                return "pincode_delivery"
        except ImportError:
            pass

    try:
        from services.conversation_scope import turn_blocks_product_catalog
        from services.product_catalog_resolver import (
            KIND_PRODUCT_SEARCH,
            product_catalog_route_is_locked,
            resolve_product_search_turn,
        )

        if product_catalog_route_is_locked(ai_route):
            return "product_catalog"
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm
            from services.ai_route_semantics import brain_route_indicates_product_catalog

            if should_skip_micro_classifier_llm():
                if brain_route_indicates_product_catalog(ai_route):
                    return "product_catalog"
            elif not turn_blocks_product_catalog(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            ):
                resolved = resolve_product_search_turn(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route=ai_route,
                    allow_llm=True,
                )
                if resolved.kind == KIND_PRODUCT_SEARCH:
                    return "product_catalog"
        except ImportError:
            if not turn_blocks_product_catalog(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            ):
                resolved = resolve_product_search_turn(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route=ai_route,
                    allow_llm=True,
                )
                if resolved.kind == KIND_PRODUCT_SEARCH:
                    return "product_catalog"
    except ImportError:
        pass

    try:
        from services.refund_status_semantics import (
            KIND_PERSONAL_STATUS,
            KIND_POLICY_HOWTO,
            _message_has_refund_topic,
            resolve_refund_turn,
        )

        comb_rf = f"{original_msg or ''} {msg_en or ''}".strip()
        if _message_has_refund_topic(comb_rf):
            resolved = resolve_refund_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                allow_llm=True,
            )
            if resolved.kind == KIND_PERSONAL_STATUS:
                return "refund_status"
            if resolved.kind == KIND_POLICY_HOWTO:
                return "refund_policy"
    except ImportError:
        pass

    if ai_route:
        try:
            from services.account_list_semantics import (
                infer_account_list_semantic_goal_from_route,
            )

            route_goal = infer_account_list_semantic_goal_from_route(
                ai_route, original_msg, msg_en, conversation_context
            )
            if route_goal:
                return route_goal
        except ImportError:
            pass

        from services.semantic_intent import llm_semantic_route_available
        from services.ai_route_semantics import infer_semantic_goal_from_ai_route

        if llm_semantic_route_available(ai_route):
            ai_goal = infer_semantic_goal_from_ai_route(ai_route)
            if ai_goal == "order_details":
                try:
                    from services.refund_status_semantics import (
                        current_turn_wants_personal_refund_status,
                    )

                    comb_early = f"{original_msg or ''} {msg_en or ''}".strip()
                    if current_turn_wants_personal_refund_status(
                        original_msg,
                        msg_en,
                        conversation_context,
                        ai_route=ai_route,
                        allow_llm=False,
                    ):
                        ai_goal = "refund_status"
                except ImportError:
                    pass
            if ai_goal in ("order_invoice", "order_details", "track_single_order"):
                try:
                    from services.order_details_flow import (
                        _current_turn_wants_invoice,
                        _current_turn_wants_tracking,
                        _user_rejects_invoice_wants_details,
                        message_is_catalog_product_browse_not_order_details,
                    )
                    from services.refund_status_semantics import (
                        current_turn_wants_personal_refund_status,
                    )

                    comb_early = f"{original_msg or ''} {msg_en or ''}".strip()
                    if current_turn_wants_personal_refund_status(
                        original_msg,
                        msg_en,
                        conversation_context,
                        ai_route=ai_route,
                        allow_llm=False,
                    ):
                        ai_goal = "refund_status"
                    elif message_is_catalog_product_browse_not_order_details(comb_early):
                        ai_goal = ""
                    elif _current_turn_wants_tracking(comb_early, conversation_context):
                        ai_goal = "track_single_order"
                    elif _user_rejects_invoice_wants_details(comb_early, conversation_context):
                        ai_goal = "order_details"
                    elif ai_goal == "order_details" and _current_turn_wants_invoice(
                        comb_early, conversation_context
                    ):
                        ai_goal = "order_invoice"
                    elif ai_goal == "order_invoice" and _user_rejects_invoice_wants_details(
                        comb_early, conversation_context
                    ):
                        ai_goal = "order_details"
                except ImportError:
                    pass
            if ai_goal:
                return ai_goal

    from services.semantic_intent import skip_keyword_intent_routes

    if skip_keyword_intent_routes(ai_route):
        return ""

    from utils.helpers import (
        _text_is_order_id_help_request,
        _text_is_order_tracking_intent,
        _text_is_refund_return_policy_howto,
        _text_is_refund_return_status_lookup,
        _text_is_tracking_howto_request,
        _text_is_undelivered_order_complaint,
        _text_requests_purchase_order_list_in_chat,
        _text_wants_order_history_list_in_chat,
        _user_asks_hypothetical_tracking_capability,
        _user_rejects_order_history_wants_tracking,
        message_is_past_purchase_list_request,
        message_needs_policy_answer,
    )
    from utils.multilingual_intent import (
        intent_combined_text,
        multilingual_order_history_match,
        multilingual_order_tracking_match,
    )

    comb = intent_combined_text(original_msg, msg_en)

    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
    )

    if not message_denies_wishlist_wants_order_history(comb):
        if _text_asks_how_to_view_wishlist(comb, conversation_context):
            return "wishlist_howto"
        if _text_asks_wishlist(comb) or message_is_wishlist_like_request(comb):
            return "wishlist_list"

    try:
        from services.order_tracking_semantics import (
            message_user_wants_order_tracking,
            message_user_rejects_refund_wants_tracking,
        )

        if (
            message_user_rejects_refund_wants_tracking(comb)
            or message_user_wants_order_tracking(comb, conversation_context)
        ):
            return "track_single_order"
    except ImportError:
        pass

    if _text_is_refund_return_status_lookup(comb, conversation_context):
        try:
            from services.order_tracking_semantics import message_user_wants_order_tracking
            from services.order_details_flow import ai_route_requests_order_details_lookup

            if message_user_wants_order_tracking(comb, conversation_context):
                return "track_single_order"
            if not ai_route_requests_order_details_lookup(
                ai_route, original_msg, msg_en, conversation_context
            ):
                return "refund_status"
        except ImportError:
            return "refund_status"

    try:
        from services.order_details_flow import _current_turn_wants_tracking
        from utils.helpers import extract_order_id, _text_is_order_tracking_intent_leaf

        if extract_order_id(comb, conversation_context) and (
            _text_is_order_tracking_intent_leaf(comb)
            or _current_turn_wants_tracking(comb, conversation_context)
        ):
            return "track_single_order"
    except ImportError:
        pass

    if (
        message_is_past_purchase_list_request(comb)
        or _text_wants_order_history_list_in_chat(comb, conversation_context)
        or _text_requests_purchase_order_list_in_chat(comb, conversation_context)
    ):
        if not (message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb)):
            return "order_history_list"

    if _text_is_refund_return_policy_howto(comb) or (
        message_needs_policy_answer(comb)
        and not _text_is_refund_return_status_lookup(comb, conversation_context)
    ):
        return "refund_policy"

    from utils.helpers import (
        _text_has_past_order_complaint_context,
        _text_is_order_tracking_only_issue,
        _user_rejects_order_history_wants_tracking,
    )

    from services.semantic_intent import llm_semantic_route_available

    ai_intent_early = ((ai_route or {}).get("intent") or "").strip().lower()
    skip_order_specialist = False
    try:
        from utils.helpers import turn_skips_order_micro_classifiers

        skip_order_specialist = turn_skips_order_micro_classifiers(
            original_msg, msg_en, conversation_context, ai_route
        )
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if ai_route_is_kb_read(ai_route):
            return ""
    except ImportError:
        pass

    if not skip_order_specialist and not (
        llm_semantic_route_available(ai_route) and ai_intent_early == "pincode_check"
    ):
        if not _text_is_refund_return_status_lookup(comb, conversation_context):
            try:
                from services.order_details_flow import (
                    _fast_order_lookup_goal,
                    order_details_route_is_locked,
                    understand_single_order_request,
                )

                if order_details_route_is_locked(ai_route):
                    olk = ((ai_route or {}).get("order_lookup_kind") or "").strip().lower()
                    if olk == "invoice":
                        return "order_invoice"
                    if olk == "details":
                        return "order_details"
                fast_od = _fast_order_lookup_goal(
                    original_msg, msg_en, conversation_context, ai_route=ai_route
                )
                if fast_od in ("order_invoice", "order_details", "track_single_order"):
                    return fast_od
                sub = understand_single_order_request(
                    original_msg, msg_en, conversation_context, ai_route=ai_route
                )
                od_goal = (sub.get("goal") or "").strip()
                if od_goal in ("order_invoice", "order_details", "track_single_order"):
                    return od_goal
            except ImportError:
                pass

    if (
        _text_is_order_tracking_only_issue(comb)
        or _user_rejects_order_history_wants_tracking(comb, conversation_context)
    ):
        return "track_single_order"

    from utils.helpers import (
        _text_has_product_shopping_intent,
        _turn_is_catalog_product_request,
        _looks_like_browse_all_categories_message,
        message_asks_welfog_categories_list,
        message_is_bot_search_complaint,
        message_is_bot_capability_question,
        message_needs_support_not_product,
    )
    from services.conversation_followup import is_deals_request_message

    if (
        message_is_bot_search_complaint(comb, conversation_context)
        or message_is_bot_capability_question(comb)
        or message_needs_support_not_product(comb)
    ):
        return "refund_policy" if (
            _text_has_past_order_complaint_context(comb)
            or _text_has_past_order_complaint_context(conversation_context)
        ) else ""

    if (
        _text_has_product_shopping_intent(comb)
        or _turn_is_catalog_product_request(comb)
        or is_deals_request_message(original_msg, msg_en)
        or message_asks_welfog_categories_list(comb)
        or _looks_like_browse_all_categories_message(comb)
    ):
        return ""

    if (
        _text_requests_purchase_order_list_in_chat(comb, conversation_context)
        or _text_wants_order_history_list_in_chat(comb, conversation_context)
    ):
        if not (message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb)):
            return "order_history_list"

    if _text_is_order_id_help_request(comb) or _text_is_tracking_howto_request(comb):
        return "order_id_help"

    if _user_asks_hypothetical_tracking_capability(comb):
        return "track_single_order"

    if _text_is_refund_return_status_lookup(comb, conversation_context):
        return "refund_status"

    if (
        _text_requests_purchase_order_list_in_chat(comb, conversation_context)
        or _text_wants_order_history_list_in_chat(comb, conversation_context)
    ):
        if not (message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb)):
            return "order_history_list"

    from services.semantic_intent import llm_semantic_route_available

    if not llm_semantic_route_available(ai_route):
        from utils.helpers import (
            _text_is_delivery_serviceability_hypothetical,
            _text_is_pincode_serviceability_question,
        )

        if _text_is_delivery_serviceability_hypothetical(comb) or _text_is_pincode_serviceability_question(
            comb, conversation_context
        ):
            return "pincode_delivery"

        if (
            _user_rejects_order_history_wants_tracking(comb, conversation_context)
            or _text_is_undelivered_order_complaint(comb)
            or multilingual_order_tracking_match(comb, original_msg)
            or _text_is_order_tracking_intent(comb)
        ):
            return "track_single_order"

    if multilingual_order_history_match(comb, original_msg):
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            return "wishlist_list"
        if not (
            _text_is_undelivered_order_complaint(comb)
            or _user_rejects_order_history_wants_tracking(comb, conversation_context)
        ):
            return "order_history_list"

    if message_is_past_purchase_list_request(comb):
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            return "wishlist_list"
        if not (
            _text_is_undelivered_order_complaint(comb)
            or multilingual_order_tracking_match(comb, original_msg)
        ):
            return "order_history_list"

    return ""


def _ai_intent_conflicts_with_goal(goal: str, intent: str, route: dict) -> bool:
    intent = (intent or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if not goal:
        return False
    if goal in ("order_invoice", "order_details"):
        return intent in ("order_history", "product", "wishlist") or (
            intent == "general" and channel == "kb"
        )
    if goal == "track_single_order":
        return intent in ("order_history", "product", "wishlist") or (
            intent == "general" and channel == "kb"
        )
    if goal == "order_history_list":
        return intent in ("order", "product", "general", "wishlist") and (
            channel == "live_api" or (intent == "order" and route.get("needs_order_id"))
        )
    if goal == "wishlist_list":
        return intent in ("order_history", "order", "product", "general") and (
            channel == "live_api" or channel == "catalog"
        )
    if goal == "wishlist_howto":
        return intent in ("order_history", "wishlist") and channel == "live_api"
    if goal == "refund_policy":
        return intent in ("refund", "payment", "order") and channel == "live_api"
    if goal == "refund_status":
        return intent == "order_history"
    if goal == "order_id_help":
        return intent in ("order", "order_history") and channel == "live_api"
    if goal == "pincode_delivery":
        return intent in ("order", "order_history", "product", "wishlist") or (
            intent == "general" and channel == "kb"
        )
    return False


def _route_patch_for_semantic_goal(
    route: dict,
    goal: str,
    original_msg: str,
    msg_en: str,
    conversation_context: str,
) -> dict:
    from utils.helpers import _current_turn_has_order_id

    out = dict(route)
    keys = list(out.get("kb_keys") or [])
    bare_turn = (original_msg or msg_en or "").strip()

    if goal in ("order_invoice", "order_details"):
        out["intent"] = "order"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "live_api_only"
        out["route_handler"] = "order_details_api"
        for k in ("welfog_api_order_history", "welfog_api", "faqs"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        return out

    if goal == "track_single_order":
        out["intent"] = "order"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "kb_then_ai"
        for k in ("welfog_api_order_tracking", "shipping", "faqs"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        if _current_turn_has_order_id(original_msg, msg_en) and re.fullmatch(
            r"\s*[0-9]{4,20}\s*", bare_turn
        ):
            out["needs_order_id"] = True
        out.pop("route_handler", None)
        return out

    if goal == "pincode_delivery":
        out["intent"] = "pincode_check"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "pincode"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "live_api_only"
        out["route_handler"] = "pincode_delivery_api"
        for k in ("welfog_api_pincode_delivery", "shipping", "faqs"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        return out

    if goal == "order_history_list":
        out["intent"] = "order_history"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "live_api_only"
        out.pop("route_handler", None)
        return out

    if goal == "wishlist_list":
        out["intent"] = "wishlist"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "live_api_only"
        out.pop("route_handler", None)
        return out

    if goal == "wishlist_howto":
        out["intent"] = "general"
        out["data_channel"] = "kb"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "structured_handler"
        out["route_handler"] = "wishlist_howto_kb"
        for k in ("faqs", "welfog_api_wishlist"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        return out

    if goal == "refund_policy":
        out["intent"] = "general"
        out["data_channel"] = "kb"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["search_query"] = ""
        out["continue_previous_topic"] = False
        out["answer_strategy"] = "kb_then_ai"
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        out.pop("route_handler", None)
        return out

    if goal == "refund_status":
        out["intent"] = "refund"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["order_lookup_kind"] = "refund_status"
        out["search_query"] = ""
        out["answer_strategy"] = "api_then_ai"
        out["route_handler"] = "refund_status_api"
        return out

    if goal == "order_id_help":
        out["intent"] = "general"
        out["data_channel"] = "kb"
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["route_handler"] = "order_id_help_kb"
        out["answer_strategy"] = "structured_handler"
        for k in ("faqs", "shipping", "welfog_api_order_id"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        return out

    return out


def reconcile_ai_route_with_semantic_goal(
    route: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """
    Align Groq JSON with inferred customer goal when they clearly disagree.
    When LLM routing is available, trust AI unless conflict is obvious (conflicts_only mode).
    Sets _semantic_goal and _semantic_override on the route dict for strict-lock bypass.
    """
    from services.semantic_intent import llm_semantic_route_available

    out = dict(route or {})
    goal = infer_customer_semantic_goal(
        original_msg, msg_en, conversation_context, ai_route=out
    )
    if not goal:
        out.pop("_semantic_goal", None)
        out.pop("_semantic_override", None)
        return out

    ai_intent = (out.get("intent") or "").strip().lower()
    needs_patch = _ai_intent_conflicts_with_goal(goal, ai_intent, out) or (
        goal == "order_history_list"
        and ai_intent == "order_history"
        and out.get("needs_order_id")
    ) or (
        goal == "track_single_order"
        and ai_intent in ("general", "order_history")
        and (out.get("data_channel") or "").strip().lower() == "kb"
    ) or (
        goal == "wishlist_list"
        and ai_intent == "order_history"
    ) or (
        goal == "order_history_list"
        and ai_intent == "wishlist"
    )

    if semantic_guard_conflicts_only() and llm_semantic_route_available(out):
        if not needs_patch or ai_route_aligns_with_semantic_goal(goal, out):
            out["_semantic_goal"] = goal
            out.pop("_semantic_override", None)
            log_reasoning(
                f"Semantic goal '{goal}' aligns with AI intent={ai_intent} — trust LLM routing."
            )
            return out
        if needs_patch:
            patched = _route_patch_for_semantic_goal(
                out, goal, original_msg, msg_en, conversation_context
            )
            patched["_semantic_goal"] = goal
            patched["_semantic_override"] = True
            log_reasoning(
                f"Semantic conflict: goal '{goal}' overrides AI intent={ai_intent} "
                f"(LLM available but clearly wrong)."
            )
            return patched
        out["_semantic_goal"] = goal
        out.pop("_semantic_override", None)
        return out

    if not needs_patch:
        out["_semantic_goal"] = goal
        out.pop("_semantic_override", None)
        return out

    patched = _route_patch_for_semantic_goal(
        out, goal, original_msg, msg_en, conversation_context
    )
    patched["_semantic_goal"] = goal
    patched["_semantic_override"] = True
    log_reasoning(
        f"Semantic goal '{goal}' overrides AI intent={ai_intent} "
        f"(keyword guard — LLM unavailable or full guard mode)."
    )
    return patched
