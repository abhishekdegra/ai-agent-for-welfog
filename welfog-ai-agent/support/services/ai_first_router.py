"""
AI-first routing: every customer message is understood by Groq BEFORE picking API / KB / cache.

Deterministic shortcuts ONLY for unambiguous tokens (catalog pro_id in message) — not phrase lists.
"""
from __future__ import annotations

import os
from typing import Optional

from services.answer_router import AnswerRouteDecision
from services.ai_service import ai_brain_route
from services.message_understanding import apply_ai_route_corrections
from utils.reasoning_log import log_reasoning


def _wishlist_answer_route_decision(
    route_data: dict,
    comb_hist: str,
    conv_for_llm: str,
    reasoning: str,
) -> Optional[AnswerRouteDecision]:
    """
    Saved/liked products in chat — must win over purchase-history semantic goal
    when AI JSON or translated text drifts to order_history.
    """
    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        _wants_wishlist_list_in_chat,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
    )

    if message_denies_wishlist_wants_order_history(comb_hist):
        return None

    from services.account_list_semantics import (
        KIND_PURCHASE_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_WISHLIST_IN_CHAT,
        _norm_account_list_kind,
    )

    kind = _norm_account_list_kind(route_data.get("account_list_kind") or "")
    if kind == KIND_WISHLIST_HOWTO:
        log_reasoning("AI route → wishlist how-to (account_list_kind).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs", "welfog_api_wishlist"],
            is_welfog_related=True,
            reason=f"Wishlist navigation — {reasoning}",
        )
    if kind == KIND_WISHLIST_IN_CHAT:
        log_reasoning("AI route → wishlist API (account_list_kind).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Wishlist API — {reasoning}",
        )
    if kind in (KIND_PURCHASE_IN_CHAT, KIND_PURCHASE_HOWTO):
        return None

    ai_i = (route_data.get("intent") or "").strip().lower()
    meaning_blob = (
        f" {(route_data.get('user_meaning') or '').lower()} "
        f" {(route_data.get('reasoning') or '').lower()} "
    )
    ai_describes_wishlist = any(
        x in meaning_blob
        for x in (
            "wishlist",
            "wish list",
            "saved product",
            "saved products",
            "liked product",
            "liked products",
            "heart ",
            "favourite",
            "favorite",
        )
    ) and not any(
        x in meaning_blob
        for x in ("order history", "purchase history", "past order", "orders placed")
    )

    is_wishlist_turn = (
        ai_i == "wishlist"
        or message_is_wishlist_like_request(comb_hist)
        or _text_asks_wishlist(comb_hist)
        or (ai_describes_wishlist and ai_i != "order_history")
    )
    if not is_wishlist_turn:
        return None

    if _text_asks_how_to_view_wishlist(comb_hist, conv_for_llm):
        log_reasoning("AI route → wishlist how-to (saved/liked topic, not purchase list).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs", "welfog_api_wishlist"],
            is_welfog_related=True,
            reason=f"Wishlist navigation — {reasoning}",
        )

    if (
        _text_asks_wishlist(comb_hist)
        or _wants_wishlist_list_in_chat(comb_hist)
        or message_is_wishlist_like_request(comb_hist)
        or ai_i == "wishlist"
    ):
        log_reasoning("AI route → wishlist list in chat (trust saved/liked topic over purchase list).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Wishlist API — {reasoning}",
        )
    return None


def _kb_channel_brain_decision(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    comb_hist: str,
    kb_keys: list,
    reasoning: str,
) -> Optional[AnswerRouteDecision]:
    """
    Trust Groq data_channel=kb before semantic-goal overrides.
    Prevents company/payment FAQ turns from being hijacked to refund policy templates.
    """
    channel = (route_data.get("data_channel") or "").strip().lower()
    intent = (route_data.get("intent") or "general").strip().lower()
    if channel != "kb" or intent not in ("general", "refund", "payment", "seller"):
        return None
    try:
        from services.product_catalog_resolver import turn_requests_product_catalog

        if turn_requests_product_catalog(
            original_msg, msg_en, "", ai_route=route_data, allow_llm=True
        ):
            return None
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if not ai_route_is_kb_read(route_data):
            return None
    except ImportError:
        return None

    rh = (route_data.get("route_handler") or "").strip()
    from utils.helpers import (
        message_asks_welfog_categories_list,
        message_is_welfog_about_request,
        should_use_warm_conversational_reply,
    )
    from services.conversation_followup import is_deals_request_message
    from services.semantic_answer_plan import build_semantic_answer_plan

    if is_deals_request_message(original_msg, msg_en):
        log_reasoning("AI route → today's deals API (KB channel blocked — deals menu).")
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            is_welfog_related=True,
            reason=f"Deals / offers — {reasoning}",
        )
    if message_asks_welfog_categories_list(comb_hist):
        log_reasoning("AI route → categories API (KB channel blocked — category list).")
        return AnswerRouteDecision(
            source="api",
            intent="categories",
            handler="categories_api",
            is_welfog_related=True,
            reason=f"Welfog categories — {reasoning}",
        )

    if message_is_welfog_about_request(comb_hist):
        log_reasoning("AI route → Welfog company/about KB (trust brain channel=kb).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="welfog_about_kb",
            kb_keys=kb_keys or ["company", "faqs"],
            is_welfog_related=True,
            reason=f"Welfog company/platform — {reasoning}",
        )

    if intent == "general" and should_use_warm_conversational_reply(
        original_msg, msg_en, comb_hist, route_data
    ):
        log_reasoning("AI route: general talk/thanks — warm feedback (trust brain channel=kb).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="warm_feedback",
            is_welfog_related=True,
            reason=f"Conversational thanks/praise — {reasoning}",
        )

    plan = build_semantic_answer_plan(route_data, handler=rh or "")
    if plan.answer_strategy in ("kb_then_ai", "api_kb_ai"):
        rh_kb = rh or "ai_route_and_answer"
        src = "kb_ai"
    elif plan.answer_strategy == "kb_only":
        rh_kb = rh or "dynamic_kb"
        src = "kb"
    else:
        rh_kb = rh or "dynamic_kb"
        src = "kb"
    log_reasoning(
        f"AI route: trust brain channel=kb strategy={plan.answer_strategy} → {rh_kb}."
    )
    return AnswerRouteDecision(
        source=src,
        intent=intent,
        handler=rh_kb,
        kb_keys=kb_keys or list(route_data.get("kb_keys") or ["faqs"]),
        is_welfog_related=True,
        reason=f"AI KB answer ({plan.answer_strategy}) — {reasoning}",
    )


def _decision_from_ai_route(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: Optional[dict] = None,
) -> AnswerRouteDecision:
    """Map Groq routing JSON → handler. Trust AI intent; no keyword list overrides here."""
    intent = (route_data.get("intent") or "general").strip()
    is_welfog = bool(route_data.get("is_welfog_related", True))
    kb_keys = list(route_data.get("kb_keys") or [])
    search_query = (route_data.get("search_query") or "").strip()
    needs_order_id = bool(route_data.get("needs_order_id", False))
    reasoning = (route_data.get("reasoning") or "")[:200]

    from services.product_search_flow import message_eligible_for_product_ai_flow
    from services.ai_route_semantics import ai_route_allows_catalog_search
    from utils.helpers import extract_product_search_query

    comb_hist = f"{original_msg} {msg_en}".strip()

    from services.semantic_intent import ai_route_requests_pincode_delivery
    from utils.helpers import message_has_live_pincode_check_intent

    if ai_route_requests_pincode_delivery(route_data):
        pin = (route_data.get("extracted_pincode") or "").strip()
        log_reasoning(f"AI route → pincode delivery API (semantic){f' PIN {pin}' if pin else ''}.")
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            is_welfog_related=True,
            reason=f"AI semantic delivery/serviceability check — {reasoning}",
        )

    from utils.helpers import should_use_warm_conversational_reply

    if should_use_warm_conversational_reply(original_msg, msg_en, comb_hist, route_data):
        if (
            intent in ("general", "out_of_domain")
            and not needs_order_id
            and not search_query
            and not message_has_live_pincode_check_intent(original_msg, comb_hist, msg_en)
            and intent not in ("pincode_check", "order", "order_history", "refund", "payment", "product")
        ):
            log_reasoning("AI+heuristic: conversational thanks/praise — warm reply (not KB).")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational talk — {reasoning}",
            )

    rh = (route_data.get("route_handler") or "").strip()
    channel = (route_data.get("data_channel") or "").strip().lower()

    strict_semantic_mode = (os.getenv("STRICT_AI_INTENT_ROUTING", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    # Legacy phrase-based structured procedures are disabled in strict semantic mode.
    if not rh and not strict_semantic_mode:
        from services.answer_router import resolve_structured_procedure_route

        structured = resolve_structured_procedure_route(
            original_msg, msg_en, conversation_context=comb_hist
        )
        if structured:
            return structured

    from utils.helpers import _text_wants_order_history_list_in_chat, message_needs_live_single_order_lookup
    from services.semantic_intent import (
        ai_route_requests_live_order_lookup,
        ai_route_requests_order_history_list,
        ai_route_requests_pincode_delivery,
        ai_route_requests_refund_status_lookup,
    )

    ai_intent = (route_data.get("intent") or "").strip().lower()
    channel = (route_data.get("data_channel") or "").strip().lower()

    from utils.helpers import (
        _text_is_order_tracking_intent,
        _text_is_undelivered_order_complaint,
        _user_rejects_order_history_wants_tracking,
    )

    from services.semantic_route_guard import infer_customer_semantic_goal

    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list, _text_requests_category_product_browse

    ai_intent_pre = (route_data.get("intent") or "").strip().lower()
    channel_pre = (route_data.get("data_channel") or "").strip().lower()

    if is_deals_request_message(original_msg, msg_en) or ai_intent_pre == "deals":
        log_reasoning("AI route → today's deals API (semantic deals/offers).")
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            is_welfog_related=True,
            reason=f"Deals / offers — {reasoning}",
        )

    from services.welfog_api import resolve_category_product_browse_route, ensure_expanded_categories_map_for_ctx

    if ctx:
        ensure_expanded_categories_map_for_ctx(ctx)
    cat_browse = resolve_category_product_browse_route(comb_hist, ctx=ctx)
    if cat_browse:
        cid, sq = cat_browse
        if ctx is not None:
            ctx.setdefault("data", {})["selected_category_id"] = cid
            ctx["awaiting"] = None
        log_reasoning(f"AI route → category-filtered products (category_id={cid}).")
        return AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason=f"Products in category {cid} — {reasoning}",
        )

    if (
        message_asks_welfog_categories_list(comb_hist)
        or ai_intent_pre == "category_feed"
        or (
            ai_intent_pre in ("categories", "general")
            and "categor" in comb_hist.lower()
            and channel_pre == "live_api"
            and not _text_requests_category_product_browse(comb_hist, ctx)
        )
    ):
        log_reasoning("AI route → categories API (browse/list/count).")
        return AnswerRouteDecision(
            source="api",
            intent="categories",
            handler="categories_api",
            is_welfog_related=True,
            reason=f"Welfog categories — {reasoning}",
        )

    wishlist_dec = _wishlist_answer_route_decision(route_data, comb_hist, conv_for_llm, reasoning)
    if wishlist_dec is not None:
        return wishlist_dec

    try:
        from services.conversation_scope import (
            SCOPE_CHITCHAT,
            SCOPE_OUT,
            _turn_requests_catalog_menu,
            scope_from_ai_route,
            turn_blocks_product_catalog,
        )

        if _turn_requests_catalog_menu(original_msg, msg_en, ai_route=route_data):
            if is_deals_request_message(original_msg, msg_en):
                log_reasoning("AI route → today's deals API (catalog menu scope guard).")
                return AnswerRouteDecision(
                    source="api",
                    intent="deals",
                    handler="deals_api",
                    is_welfog_related=True,
                    reason=f"Deals / offers — {reasoning}",
                )
            if message_asks_welfog_categories_list(comb_hist):
                log_reasoning("AI route → categories API (catalog menu scope guard).")
                return AnswerRouteDecision(
                    source="api",
                    intent="categories",
                    handler="categories_api",
                    is_welfog_related=True,
                    reason=f"Welfog categories — {reasoning}",
                )
        elif turn_blocks_product_catalog(
            original_msg, msg_en, conv_for_llm, ai_route=route_data
        ):
            scope_dec = scope_from_ai_route(route_data)
            if scope_dec and scope_dec.scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
                if scope_dec.scope == SCOPE_CHITCHAT:
                    log_reasoning("AI route → conversational scope (beats product/KB).")
                    return AnswerRouteDecision(
                        source="scope",
                        intent="general",
                        handler="warm_feedback",
                        is_welfog_related=True,
                        reason=f"Chitchat — {scope_dec.user_meaning[:80]}",
                    )
                log_reasoning("AI route → out_of_domain scope (beats product/KB).")
                return AnswerRouteDecision(
                    source="reject",
                    intent="out_of_domain",
                    handler="off_topic",
                    is_welfog_related=False,
                    reason=f"Out of domain — {scope_dec.user_meaning[:80]}",
                )
    except ImportError:
        pass

    try:
        from services.conversation_scope import turn_blocks_product_catalog
        from services.product_catalog_resolver import product_catalog_route_decision

        if not turn_blocks_product_catalog(
            original_msg, msg_en, conv_for_llm, ai_route=route_data
        ):
            pc_dec, route_data = product_catalog_route_decision(
                route_data, original_msg, msg_en, conv_for_llm, reasoning=reasoning
            )
            if pc_dec is not None:
                return pc_dec
    except ImportError:
        pass

    kb_brain = _kb_channel_brain_decision(
        route_data, original_msg, msg_en, comb_hist, kb_keys, reasoning
    )
    if kb_brain is not None:
        return kb_brain

    sem_goal = infer_customer_semantic_goal(
        original_msg, msg_en, conv_for_llm, ai_route=route_data
    )
    # When Groq already chose pincode_check, do not let keyword semantic goals override.
    if ai_route_requests_pincode_delivery(route_data):
        sem_goal = sem_goal if sem_goal == "pincode_delivery" else "pincode_delivery"
    if sem_goal == "refund_policy":
        log_reasoning("AI route → return/refund/wrong-item KB (semantic goal).")
        return AnswerRouteDecision(
            source="kb",
            intent="refund",
            handler="policy_structured_kb",
            kb_keys=["faqs", "refund", "shipping"],
            is_welfog_related=True,
            reason=f"Return/refund policy — {reasoning}",
        )
    if sem_goal == "refund_status":
        log_reasoning("AI route → personal refund status API (semantic goal).")
        return AnswerRouteDecision(
            source="api",
            intent="refund",
            handler="refund_status_api",
            is_welfog_related=True,
            reason=f"Personal refund/return status — {reasoning}",
        )
    if sem_goal == "track_single_order":
        log_reasoning("AI route → live order tracking (semantic goal, any language).")
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            is_welfog_related=True,
            reason=f"Semantic track order — {reasoning}",
        )
    if sem_goal in ("order_invoice", "order_details"):
        try:
            from services.location_delivery_resolver import pincode_delivery_route_is_locked

            if pincode_delivery_route_is_locked(
                route_data,
                original_msg,
                msg_en,
                comb_hist,
                allow_llm=True,
            ):
                log_reasoning(
                    "Semantic order_details skipped — delivery/serviceability (pincode API)."
                )
                sem_goal = "pincode_delivery"
        except ImportError:
            pass
    if sem_goal in ("order_invoice", "order_details"):
        try:
            from services.order_details_flow import (
                message_is_catalog_product_browse_not_order_details,
            )
            from utils.helpers import (
                _text_has_product_shopping_intent,
                _turn_is_catalog_product_request,
            )

            if (
                message_is_catalog_product_browse_not_order_details(comb_hist)
                or _text_has_product_shopping_intent(comb_hist)
                or _turn_is_catalog_product_request(comb_hist)
            ):
                log_reasoning(
                    "Semantic order_details skipped — product browse/catalog search."
                )
                sem_goal = ""
        except ImportError:
            pass
    if sem_goal in ("order_invoice", "order_details"):
        log_reasoning(f"AI route → order details/invoice API (semantic goal={sem_goal}).")
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            is_welfog_related=True,
            reason=f"Semantic {sem_goal} — {reasoning}",
        )
    if sem_goal == "pincode_delivery":
        log_reasoning("AI route → pincode delivery API (semantic goal, area serviceability).")
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            is_welfog_related=True,
            reason=f"Semantic pincode delivery — {reasoning}",
        )
    if sem_goal == "product_catalog":
        try:
            from services.product_catalog_resolver import (
                apply_product_catalog_to_route,
                log_product_catalog_routing,
                resolve_product_search_turn,
            )

            resolved = resolve_product_search_turn(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=route_data,
                allow_llm=True,
            )
            route_data = apply_product_catalog_to_route(
                route_data,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
            )
            sq = (
                resolved.search_query
                or (route_data.get("search_query") or "").strip()
                or extract_product_search_query(comb_hist, comb_hist, conv_for_llm, ai_route=route_data)
                or ""
            )
            log_product_catalog_routing(
                detected_intent="product_search",
                product_entities=resolved.entities or route_data.get("_product_entities"),
                selected_route="product_ai_flow",
                filters=resolved.entities or {},
                source=resolved.source or "semantic_goal",
            )
        except ImportError:
            sq = (route_data.get("search_query") or "").strip()
        log_reasoning(f"AI route → product catalog API (semantic goal, sq={sq!r}).")
        return AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason=f"Semantic product catalog — {reasoning}",
        )
    if sem_goal == "wishlist_howto":
        log_reasoning("AI route → wishlist how-to KB (semantic goal).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs", "welfog_api_wishlist"],
            is_welfog_related=True,
            reason=f"Wishlist where/how in app — {reasoning}",
        )
    if sem_goal == "wishlist_list":
        log_reasoning("AI route → wishlist list API (semantic goal).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Semantic wishlist — {reasoning}",
        )

    if sem_goal == "order_history_list":
        from utils.helpers import (
            _text_has_product_shopping_intent,
            _turn_is_catalog_product_request,
            message_is_wishlist_like_request,
            _text_asks_wishlist,
        )

        if (
            ai_intent_pre == "wishlist"
            or message_is_wishlist_like_request(comb_hist)
            or _text_asks_wishlist(comb_hist)
        ):
            log_reasoning(
                "Semantic order_history_list skipped — saved/liked wishlist topic."
            )
        elif ai_intent_pre in ("product", "deals", "categories", "category_feed") or channel_pre == "catalog":
            log_reasoning(
                "Semantic order_history_list skipped — AI/catalog intent is product browse, not purchase list."
            )
        elif _text_has_product_shopping_intent(comb_hist) or _turn_is_catalog_product_request(comb_hist):
            log_reasoning(
                "Semantic order_history_list skipped — product shopping (show/buy item), not order list."
            )
        else:
            log_reasoning("AI route → order history list (semantic goal).")
            return AnswerRouteDecision(
                source="ai_order",
                intent="order_history",
                handler="order_ai_flow",
                is_welfog_related=True,
                reason=f"Semantic purchase list — {reasoning}",
            )

    try:
        from services.order_tracking_semantics import (
            ai_route_requests_order_tracking_lookup,
        )

        if ai_route_requests_order_tracking_lookup(
            route_data, original_msg, msg_en, comb_hist
        ):
            log_reasoning("AI route → order tracking API (disambiguated from refund/details).")
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_tracking_api",
                is_welfog_related=True,
                reason=f"Live order tracking — {reasoning}",
            )
    except ImportError:
        pass

    try:
        from services.order_details_flow import (
            ai_route_requests_order_details_lookup,
            message_wants_order_details_or_invoice,
        )

        if ai_route_requests_order_details_lookup(
            route_data, original_msg, msg_en, comb_hist
        ):
            od_goal = message_wants_order_details_or_invoice(
                original_msg, msg_en, comb_hist, ai_route=route_data
            ) or "order_details"
            log_reasoning(
                f"AI route → order details API (disambiguated, goal={od_goal})."
            )
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_details_api",
                is_welfog_related=True,
                reason=f"Order details/invoice — {reasoning}",
            )
    except ImportError:
        pass

    if ai_route_requests_refund_status_lookup(route_data, original_msg, msg_en, comb_hist):
        log_reasoning("AI route → refund status API (semantic, any language).")
        return AnswerRouteDecision(
            source="api",
            intent="refund",
            handler="refund_status_api",
            is_welfog_related=True,
            reason=f"Personal refund/return status — {reasoning}",
        )

    if ai_route_requests_live_order_lookup(route_data, original_msg, msg_en, comb_hist):
        live = ai_intent if ai_intent in ("refund", "payment", "order") else "order"
        handler = "order_tracking_api" if live == "order" else "ai_route_and_answer"
        log_reasoning(f"AI route → live {live} lookup (semantic, any language).")
        return AnswerRouteDecision(
            source="api",
            intent=live,
            handler=handler,
            is_welfog_related=True,
            reason=f"AI semantic live {live} — {reasoning}",
        )

    from utils.helpers import (
        message_is_wishlist_like_request,
        resolve_navigation_help_topic,
    )

    from utils.helpers import _text_is_order_id_help_request, _text_is_tracking_howto_request

    ai_intent_early = (route_data.get("intent") or "").strip().lower()
    if _text_is_order_id_help_request(comb_hist):
        log_reasoning("AI route → order_id help (where to find ID, not tracking/history).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_id_help_kb",
            kb_keys=["faqs", "shipping", "welfog_api_order_id"],
            is_welfog_related=True,
            reason=f"Order ID location help — {reasoning}",
        )
    nav_early = resolve_navigation_help_topic(comb_hist, conv_for_llm, ai_route=route_data)
    if ai_intent_early == "wishlist" or message_is_wishlist_like_request(comb_hist):
        from utils.helpers import _text_asks_wishlist, _wants_wishlist_list_in_chat

        if _text_asks_wishlist(comb_hist) or _wants_wishlist_list_in_chat(comb_hist):
            log_reasoning("AI route → wishlist list in chat (trust AI + saved/liked topic).")
            return AnswerRouteDecision(
                source="api",
                intent="wishlist",
                handler="wishlist_api",
                is_welfog_related=True,
                reason=f"Wishlist API — {reasoning}",
            )
    if (
        nav_early == "order_history_howto"
        and ai_intent_early != "wishlist"
        and not message_is_wishlist_like_request(comb_hist)
        and not _text_wants_order_history_list_in_chat(comb_hist, conv_for_llm)
    ):
        log_reasoning("AI route → order_history how-to (app navigation, not list API).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=["welfog_api_order_history", "faqs"],
            is_welfog_related=True,
            reason=f"Order history where/how in app — {reasoning}",
        )

    if ai_route_requests_order_history_list(route_data) and nav_early != "order_history_howto":
        log_reasoning("AI route → order_history list (semantic purchase-history API).")
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason=f"AI semantic order list — {reasoning}",
        )

    if not message_needs_live_single_order_lookup(
        original_msg, msg_en, comb_hist, ai_route=route_data
    ) and _text_wants_order_history_list_in_chat(comb_hist, conv_for_llm):
        log_reasoning("AI route → order_history list in chat (purchase-history API).")
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason=f"Show purchase history in chat — {reasoning}",
        )

    if rh == "order_history_howto_kb" and _text_wants_order_history_list_in_chat(comb_hist):
        rh = ""
    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        message_clarifies_wishlist_not_order_history,
        message_mentions_wishlist_topic,
    )

    from utils.helpers import (
        _message_has_app_navigation_intent,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
        message_wants_order_history_app_navigation,
        resolve_navigation_help_topic,
    )

    nav_topic = resolve_navigation_help_topic(comb_hist, conv_for_llm, ai_route=route_data)
    if nav_topic == "order_history_howto" and rh in ("wishlist_howto_kb", ""):
        rh = "order_history_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_history_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: wishlist → order_history_howto (semantic navigation).")
    elif nav_topic == "wishlist_howto" and rh == "order_history_howto_kb":
        rh = "wishlist_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "wishlist_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto → wishlist_howto (semantic navigation).")

    if rh == "order_history_howto_kb" and _text_is_order_id_help_request(comb_hist):
        rh = "order_id_help_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_id_help_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto → order_id_help_kb.")
    elif rh == "order_history_howto_kb" and (
        _text_asks_how_to_view_wishlist(comb_hist, conv_for_llm)
        or message_clarifies_wishlist_not_order_history(comb_hist)
        or message_mentions_wishlist_topic(comb_hist)
        or (
            message_is_wishlist_like_request(comb_hist)
            and _message_has_app_navigation_intent(comb_hist)
        )
    ):
        rh = "wishlist_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "wishlist_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto -> wishlist_howto (wishlist topic).")
    from utils.helpers import (
        _message_overrides_placement_followup,
        _text_has_explicit_how_to_place_order,
        _text_has_order_placement_intent,
    )

    if rh == "order_placement_kb" and _message_overrides_placement_followup(comb_hist):
        rh = ""
        route_data = dict(route_data)
        route_data["continue_previous_topic"] = False
        route_data.pop("route_handler", None)
        if (route_data.get("intent") or "general") in ("general", "order"):
            route_data["intent"] = "refund"
        log_reasoning("AI route: return/refund overrides stale order_placement_kb — AI+KB answer.")
    elif rh == "order_placement_kb" and (
        nav_topic == "order_history_howto"
        or message_wants_order_history_app_navigation(comb_hist, conv_for_llm)
    ):
        rh = "order_history_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_history_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_placement_kb → order_history_howto (view history).")
    elif rh == "order_placement_kb" and not (
        _text_has_explicit_how_to_place_order(comb_hist)
        or _text_has_order_placement_intent(comb_hist)
    ):
        rh = ""
    from utils.helpers import _text_has_product_shopping_intent, _turn_is_catalog_product_request

    if rh == "order_placement_kb" and (
        _text_has_product_shopping_intent(comb_hist) or _turn_is_catalog_product_request(comb_hist)
    ):
        rh = ""
        route_data = dict(route_data)
        route_data["intent"] = "product"
        route_data["data_channel"] = "catalog"
        route_data.pop("route_handler", None)
        log_reasoning("AI route: product browse overrides mistaken order_placement_kb.")
    if rh in ("dynamic_kb", "knowledge_topic_kb", "welfog_fees_kb", "short_video_rules_kb"):
        ai_intent_now = (route_data.get("intent") or intent or "").strip().lower()
        if should_use_warm_conversational_reply(
            original_msg, msg_en, comb_hist, route_data
        ) and not message_has_live_pincode_check_intent(
            original_msg, comb_hist, msg_en
        ) and ai_intent_now not in (
            "pincode_check",
            "order",
            "order_history",
            "refund",
            "payment",
            "product",
            "wishlist",
        ):
            log_reasoning(f"AI handler={rh} overridden — conversational talk, warm reply.")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational talk — {reasoning}",
            )
    if rh == "order_details_api":
        try:
            from services.order_details_flow import (
                message_is_catalog_product_browse_not_order_details,
            )
            from utils.helpers import (
                _text_has_product_shopping_intent,
                _turn_is_catalog_product_request,
            )

            if (
                message_is_catalog_product_browse_not_order_details(comb_hist)
                or _text_has_product_shopping_intent(comb_hist)
                or _turn_is_catalog_product_request(comb_hist)
            ):
                log_reasoning(
                    "Handler order_details_api skipped — product browse/catalog search."
                )
                rh = ""
                route_data = dict(route_data)
                route_data["intent"] = "product"
                route_data["data_channel"] = "catalog"
                route_data.pop("route_handler", None)
                route_data["order_lookup_kind"] = "none"
                route_data["needs_order_id"] = False
        except ImportError:
            pass
    if rh == "order_details_api":
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            kb_keys=list(route_data.get("kb_keys") or []),
            is_welfog_related=True,
            reason=f"Order details/invoice API — {reasoning}",
        )
    if rh:
        return AnswerRouteDecision(
            source="kb",
            intent=(route_data.get("intent") or "general"),
            handler=rh,
            kb_keys=list(route_data.get("kb_keys") or []),
            is_welfog_related=True,
            reason=f"Structured procedure handler from AI corrections — {reasoning}",
        )

    from services.support_scope import message_mentions_other_company_support

    if not is_welfog or intent == "out_of_domain":
        if message_mentions_other_company_support(
            original_msg, msg_en, conversation_context=comb_hist, ai_route=route_data
        ):
            log_reasoning("AI route: external company — decline (no Welfog remap).")
            return AnswerRouteDecision(
                source="reject",
                intent="out_of_domain",
                handler="other_company_decline",
                is_welfog_related=False,
                reason=f"Not a Welfog request — {reasoning}",
            )
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="off_topic",
            is_welfog_related=False,
            reason=f"AI: off-topic — {reasoning}",
        )

    api_intents = {
        "order_id": ("ai_order_id", "order_id_ai_flow"),
        "order_history": ("ai_order", "order_ai_flow"),
        "wishlist": ("api", "wishlist_api"),
        "product": ("ai_product", "product_ai_flow"),
        "deals": ("api", "deals_api"),
        "categories": ("api", "categories_api"),
        "category_feed": ("api", "category_feed_api"),
        "pincode_check": ("api", "pincode_delivery_api"),
    }

    # Trust AI data_channel when it conflicts with a mis-labelled intent string.
    if channel == "live_api" and intent in api_intents:
        source, handler = api_intents[intent]
        log_reasoning(f"AI route: data_channel=live_api → {handler}.")
        return AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            search_query="",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI live API — {reasoning}",
        )

    if channel == "catalog" and intent == "product":
        sq = extract_product_search_query(original_msg, msg_en, search_query) or search_query
        if ai_route_allows_catalog_search(route_data) and message_eligible_for_product_ai_flow(
            comb_hist, msg_en, original_msg, ai_route=route_data
        ):
            log_reasoning("AI route: data_channel=catalog → product_ai_flow.")
            return AnswerRouteDecision(
                source="ai_product",
                intent="product",
                handler="product_ai_flow",
                search_query=sq,
                is_welfog_related=True,
                reason=f"AI catalog search — {reasoning}",
            )

    if channel == "kb" and intent in ("general", "refund", "payment", "seller"):
        try:
            from services.conversation_scope import turn_blocks_product_catalog
            from services.product_catalog_resolver import product_catalog_route_decision

            if not turn_blocks_product_catalog(
                original_msg, msg_en, conv_for_llm, ai_route=route_data
            ):
                pc_dec, route_data = product_catalog_route_decision(
                    route_data, original_msg, msg_en, conv_for_llm, reasoning=reasoning
                )
                if pc_dec is not None:
                    return pc_dec
        except ImportError:
            pass
        from utils.helpers import should_use_warm_conversational_reply
        from services.semantic_answer_plan import build_semantic_answer_plan

        if intent == "general" and should_use_warm_conversational_reply(
            original_msg, msg_en, comb_hist, route_data
        ):
            log_reasoning("AI route: general talk/thanks — warm feedback (skip dynamic_kb).")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational thanks/praise — {reasoning}",
            )
        plan = build_semantic_answer_plan(route_data, handler=rh or "")
        if plan.answer_strategy in ("kb_then_ai", "api_kb_ai"):
            rh_kb = rh or "ai_route_and_answer"
            src = "kb_ai"
        elif plan.answer_strategy == "kb_only":
            rh_kb = rh or "dynamic_kb"
            src = "kb"
        else:
            rh_kb = rh or "dynamic_kb"
            src = "kb"
        log_reasoning(f"AI route: data_channel=kb strategy={plan.answer_strategy} → {rh_kb}.")
        return AnswerRouteDecision(
            source=src,
            intent=intent,
            handler=rh_kb,
            kb_keys=kb_keys or ["faqs"],
            is_welfog_related=True,
            reason=f"AI KB answer ({plan.answer_strategy}) — {reasoning}",
        )

    if intent in api_intents:
        if intent == "product":
            sq = extract_product_search_query(original_msg, msg_en, search_query) or search_query
            if not (
                ai_route_allows_catalog_search(route_data)
                and message_eligible_for_product_ai_flow(
                    comb_hist, msg_en, original_msg, ai_route=route_data
                )
            ):
                log_reasoning(
                    "AI route intent=product downgraded — no reliable catalog signal in latest turn."
                )
                return AnswerRouteDecision(
                    source="kb_ai",
                    intent="general",
                    handler="ai_route_and_answer",
                    kb_keys=kb_keys or ["faqs", "support"],
                    is_welfog_related=True,
                    reason=f"Ambiguous product intent; ask user to clarify exact need — {reasoning}",
                )
            search_query = sq
        resolved_intent = (route_data.get("intent") or intent).strip()
        if resolved_intent in api_intents and resolved_intent != intent:
            intent = resolved_intent
        source, handler = api_intents[intent]
        return AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            search_query=search_query if intent == "product" else "",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI route → {intent} — {reasoning}",
        )

    if intent == "seller":
        return AnswerRouteDecision(
            source="kb",
            intent="seller",
            handler="seller_kb",
            kb_keys=kb_keys or ["seller", "faqs", "support"],
            is_welfog_related=True,
            reason=f"AI route → seller KB — {reasoning}",
        )

    if intent == "order":
        from utils.helpers import _text_has_order_placement_intent

        comb_route = f"{original_msg} {msg_en}".strip()
        if _text_has_order_placement_intent(comb_route):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="order_placement_kb",
                is_welfog_related=True,
                kb_keys=kb_keys or ["faqs", "shipping"],
                reason=f"How to place order (not tracking) — {reasoning}",
            )
        if needs_order_id:
            od_goal = (route_data.get("_semantic_goal") or "").strip()
            if od_goal not in ("order_invoice", "order_details"):
                from services.order_details_flow import infer_order_details_semantic_goal

                od_goal = infer_order_details_semantic_goal(
                    original_msg, msg_en, comb_hist, ai_route=route_data
                )
            if od_goal in ("order_invoice", "order_details"):
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI: order details/invoice — {reasoning}",
                )
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_tracking_api",
                is_welfog_related=True,
                reason=f"AI: live order tracking — {reasoning}",
            )
        try:
            from services.order_tracking_semantics import message_user_wants_order_tracking
            from utils.helpers import _text_is_order_tracking_intent_leaf

            if message_user_wants_order_tracking(
                comb_route, comb_hist
            ) or _text_is_order_tracking_intent_leaf(comb_route):
                log_reasoning(
                    "Order intent without needs_order_id — live track API (not how-to KB)."
                )
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_tracking_api",
                    is_welfog_related=True,
                    reason=f"Live order tracking — {reasoning}",
                )
        except ImportError:
            pass
        if _text_is_tracking_howto_request(comb_route):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="order_tracking_howto_kb",
                is_welfog_related=True,
                kb_keys=kb_keys or ["shipping", "faqs"],
                reason=f"AI: order help / how-to track — {reasoning}",
            )
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            is_welfog_related=True,
            reason=f"Live order tracking (ask order id) — {reasoning}",
        )

    if intent in ("refund", "payment", "seller"):
        return AnswerRouteDecision(
            source="ai",
            intent=intent,
            handler="ai_route_and_answer",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI: {intent} — {reasoning}",
        )

    return AnswerRouteDecision(
        source="kb_ai",
        intent="general",
        handler="ai_route_and_answer",
        search_query=search_query,
        kb_keys=kb_keys or ["faqs", "company"],
        is_welfog_related=True,
        reason=f"AI: general / KB-grounded — {reasoning}",
    )


def _try_conversational_fast_path(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """AI scope preflight — any-language chitchat (no keyword gate)."""
    try:
        from services.chitchat_resolver import try_chitchat_ai_preflight

        preflight = try_chitchat_ai_preflight(
            original_msg, msg_en, conv_for_llm, reply_lang
        )
        if preflight:
            log_reasoning("Conversational AI preflight — scope LLM (any language).")
            return preflight
    except ImportError:
        pass
    return None


def _try_conversational_keyword_fallback(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """LLM unavailable only — deterministic chitchat fallback."""
    from utils.helpers import message_is_conversational_general_talk

    combined = f"{original_msg} {msg_en}".strip()
    if not combined or not message_is_conversational_general_talk(
        original_msg, msg_en, conv_for_llm
    ):
        return None
    log_reasoning("Conversational keyword fallback (LLM down).")
    route_data = {
        "user_meaning": combined[:200],
        "reasoning": "Chitchat — keyword fallback while LLM unavailable.",
        "intent": "general",
        "data_channel": "none",
        "meta_kind": "conversational",
        "conversation_scope": "general_chitchat",
        "is_welfog_related": True,
        "needs_order_id": False,
        "run_catalog_search": False,
        "_preflight_conversational": True,
    }
    return (
        AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="warm_feedback",
            is_welfog_related=True,
            reason="Conversational — LLM-down fallback",
        ),
        route_data,
    )


def _try_account_list_fast_path(
    original_msg: str,
    msg_en: str,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """Wishlist / order history in chat — skip routing LLM when intent is unambiguous."""
    from utils.helpers import (
        _text_asks_how_to_view_order_history,
        _text_asks_how_to_view_wishlist,
        _text_asks_order_history,
        _text_asks_wishlist,
        _wants_wishlist_list_in_chat,
        message_is_wishlist_like_request,
    )

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None

    if (
        _text_asks_wishlist(combined)
        or _wants_wishlist_list_in_chat(combined)
        or message_is_wishlist_like_request(combined)
    ) and not _text_asks_how_to_view_wishlist(combined):
        log_reasoning("Account-list fast-path → wishlist API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show saved/liked products in chat.",
            "intent": "wishlist",
            "data_channel": "live_api",
            "account_list_kind": "wishlist_in_chat",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_api": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="wishlist",
                handler="wishlist_api",
                is_welfog_related=True,
                reason="Wishlist list in chat — fast-path",
            ),
            route_data,
        )

    if _text_asks_order_history(combined) and not _text_asks_how_to_view_order_history(
        combined
    ):
        log_reasoning("Account-list fast-path → order history API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show purchase history in chat.",
            "intent": "order_history",
            "data_channel": "live_api",
            "account_list_kind": "purchase_history_in_chat",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_api": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="order_history",
                handler="order_ai_flow",
                is_welfog_related=True,
                reason="Order history in chat — fast-path",
            ),
            route_data,
        )

    return None


def _try_catalog_menu_fast_path(
    original_msg: str,
    msg_en: str,
    ctx: Optional[dict] = None,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """Categories list / today's deals — skip brain routing LLM (fast, deterministic)."""
    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None

    if is_deals_request_message(original_msg, msg_en):
        log_reasoning("Catalog-menu fast-path → today's deals API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show today's deals / offers on Welfog.",
            "intent": "deals",
            "data_channel": "live_api",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_catalog_menu": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="deals",
                handler="deals_api",
                is_welfog_related=True,
                reason="Today's deals — fast-path",
            ),
            route_data,
        )

    if message_asks_welfog_categories_list(combined):
        try:
            from services.welfog_api import (
                ensure_expanded_categories_map_for_ctx,
                resolve_category_product_browse_route,
            )

            if ctx:
                ensure_expanded_categories_map_for_ctx(ctx)
            cat_browse = resolve_category_product_browse_route(combined, ctx=ctx)
            if cat_browse:
                cid, sq = cat_browse
                if ctx is not None:
                    ctx.setdefault("data", {})["selected_category_id"] = cid
                    ctx["awaiting"] = None
                log_reasoning(
                    f"Catalog-menu fast-path → category browse (category_id={cid})."
                )
                route_data = {
                    "user_meaning": combined[:200],
                    "reasoning": f"Products in category {cid}.",
                    "intent": "product",
                    "data_channel": "catalog",
                    "run_catalog_search": True,
                    "search_query": sq,
                    "needs_order_id": False,
                    "numeric_context": "none",
                    "_preflight_catalog_menu": True,
                }
                return (
                    AnswerRouteDecision(
                        source="ai_product",
                        intent="product",
                        handler="product_ai_flow",
                        search_query=sq,
                        is_welfog_related=True,
                        reason=f"Category browse {cid} — fast-path",
                    ),
                    route_data,
                )
        except ImportError:
            pass

        log_reasoning("Catalog-menu fast-path → categories API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "List Welfog shopping categories.",
            "intent": "categories",
            "data_channel": "live_api",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_catalog_menu": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="categories",
                handler="categories_api",
                is_welfog_related=True,
                reason="Welfog categories list — fast-path",
            ),
            route_data,
        )

    return None


def _try_policy_kb_preflight(
    original_msg: str,
    msg_en: str,
    reply_lang: str,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """
    Informational KB turns skip the routing LLM when embeddings match a policy file.
    Works for any language and phrasing — no keyword lists.
    """
    import re

    from services.kb_service import resolve_kb_keys_for_question
    from services.query_understanding import (
        infer_kb_query_category,
        top_customer_kb_file_match,
    )
    from utils.helpers import (
        _is_plausible_order_id,
        extract_product_id,
        message_is_conversational_general_talk,
        should_send_warm_greeting_reply,
    )

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None
    comb_low = combined.lower()

    if should_send_warm_greeting_reply(
        original_msg, msg_en
    ) or message_is_conversational_general_talk(original_msg, msg_en):
        return None

    # Structural only: catalog pro_id, order id, PIN → need full LLM routing / live API
    if extract_product_id(comb_low):
        return None
    if _is_plausible_order_id(comb_low) or re.search(r"\b\d{6,}\b", comb_low):
        return None
    if re.search(r"\b[1-9]\d{5}\b", combined):
        return None

    top_key, top_score = top_customer_kb_file_match(original_msg, msg_en)
    _MIN_PREFLIGHT = 0.28
    _MIN_COMPANY_PREFLIGHT = 0.36
    if not top_key or top_score < _MIN_PREFLIGHT:
        return None

    informational = frozenset(
        {"company", "faqs", "payment", "refund", "shipping", "seller", "privacy", "terms", "support"}
    )
    if top_key not in informational:
        return None
    if top_key == "company" and top_score < _MIN_COMPANY_PREFLIGHT:
        return None

    keys = resolve_kb_keys_for_question(original_msg, msg_en, max_files=3)
    if not keys:
        keys = [top_key]
    cat = infer_kb_query_category(original_msg, msg_en)
    if cat == "payment" and "payment" in keys:
        keys = ["payment"]
    elif cat == "refund" and "refund" in keys:
        keys = [k for k in ("refund", "faqs", "shipping") if k in keys][:3]

    if top_key == "company" and top_score >= _MIN_COMPANY_PREFLIGHT:
        log_reasoning(
            f"KB preflight: semantic match company (score={top_score:.2f}) — skip routing LLM."
        )
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Welfog company/platform — embedding KB preflight.",
            "intent": "general",
            "data_channel": "kb",
            "kb_keys": ["company", "faqs"],
            "answer_strategy": "kb_then_ai",
            "conversation_scope": "welfog_support",
            "is_welfog_related": True,
            "meta_kind": "none",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_kb": True,
        }
        return (
            AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="welfog_about_kb",
                kb_keys=["company", "faqs"],
                is_welfog_related=True,
                reason="Welfog company/about — semantic preflight",
            ),
            route_data,
        )

    if top_key in ("faqs", "shipping", "support") and top_score >= 0.38:
        keys = [top_key]
        if top_score < 0.50:
            extra = resolve_kb_keys_for_question(original_msg, msg_en, max_files=2)
            keys = list(dict.fromkeys(keys + (extra or [])))[:2]
        intent = "shipping" if top_key == "shipping" else "general"
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": f"FAQ/policy KB ({top_key}, score={top_score:.2f}) — routing LLM skipped.",
            "intent": intent,
            "data_channel": "kb",
            "kb_keys": keys,
            "answer_strategy": "kb_then_ai",
            "conversation_scope": "welfog_support",
            "is_welfog_related": True,
            "meta_kind": "none",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_kb": True,
        }
        log_reasoning(
            f"KB preflight: FAQ match {top_key} (score={top_score:.2f}) — skip routing LLM."
        )
        return (
            AnswerRouteDecision(
                source="kb_ai",
                intent=intent,
                handler="ai_route_and_answer",
                kb_keys=keys,
                is_welfog_related=True,
                reason=f"FAQ KB ({top_key}) — semantic preflight",
            ),
            route_data,
        )

    if cat not in ("payment", "refund", "shipping", "privacy", "terms", "seller"):
        return None

    intent = cat
    route_data = {
        "user_meaning": combined[:200],
        "reasoning": f"Informational KB ({cat}, score={top_score:.2f}) — routing LLM skipped.",
        "intent": intent,
        "data_channel": "kb",
        "kb_keys": keys,
        "answer_strategy": "kb_then_ai",
        "conversation_scope": "welfog_support",
        "is_welfog_related": True,
        "meta_kind": "none",
        "needs_order_id": False,
        "run_catalog_search": False,
        "numeric_context": "none",
        "order_lookup_kind": "none",
        "continue_previous_topic": False,
        "_preflight_kb": True,
    }
    log_reasoning(
        f"KB preflight: semantic match {top_key} (score={top_score:.2f}) "
        f"keys={','.join(keys)} — skip routing LLM."
    )
    return (
        AnswerRouteDecision(
            source="kb_ai",
            intent=intent,
            handler="ai_route_and_answer",
            kb_keys=keys,
            is_welfog_related=True,
            reason=f"Policy KB ({cat}) — semantic preflight",
        ),
        route_data,
    )


def resolve_answer_route_ai_first(
    original_msg: str,
    msg_en: str,
    retrieval_query: str = "",
    conv_for_llm: str = "",
    reply_lang: str = "en",
    ctx: Optional[dict] = None,
) -> tuple[AnswerRouteDecision, Optional[dict]]:
    """
    1) Optional tiny shortcuts (explicit catalog pro_id)
    2) Groq ai_brain_route with full conversation — primary path for ALL phrasing
    3) Safety corrections (PIN vs order id, wishlist vs orders) — never re-route via keyword lists
    """
    from services.answer_router import resolve_answer_route
    from utils.helpers import (
        _is_conversation_acknowledgment,
        _normalize_order_chat_text,
        extract_product_id,
        _text_is_product_id_lookup_context,
    )

    combined = f"{original_msg} {msg_en}".strip()
    comb_norm = _normalize_order_chat_text(combined)
    comb_low = comb_norm.lower()
    strict_llm_failsafe = (os.getenv("STRICT_LLM_FAILSAFE", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    if extract_product_id(comb_low) and _text_is_product_id_lookup_context(comb_low):
        pid = extract_product_id(comb_low)
        return (
            AnswerRouteDecision(
                source="api",
                intent="product",
                handler="catalog_pro_id",
                search_query=f"pro_id {pid}" if pid else "",
                reason="Catalog pro_id lookup (deterministic).",
            ),
            None,
        )

    try:
        from services.pincode_delivery_fast_path import try_pincode_delivery_fast_route

        pin_fast = try_pincode_delivery_fast_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx
        )
        if pin_fast:
            decision, route_data = pin_fast
            try:
                from services.chat_flow_telemetry import record_route_step, store_turn_analysis

                record_route_step("pincode_delivery_fast")
                store_turn_analysis(route_data, decision)
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    conv_fast = _try_conversational_fast_path(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    if conv_fast:
        decision, route_data = conv_fast
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("conversational_fast")
            store_turn_analysis(route_data, decision)
        except ImportError:
            pass
        return decision, route_data

    api_fast = _try_account_list_fast_path(original_msg, msg_en)
    if api_fast:
        decision, route_data = api_fast
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("account_list_fast")
            store_turn_analysis(route_data, decision)
        except ImportError:
            pass
        return decision, route_data

    try:
        from services.order_id_handoff_fast_path import try_order_id_handoff_route

        handoff_route = try_order_id_handoff_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx
        )
        if handoff_route:
            decision, route_data = handoff_route
            try:
                from services.chat_flow_telemetry import record_route_step, store_turn_analysis

                record_route_step("order_id_handoff_fast")
                store_turn_analysis(route_data, decision)
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    preflight = _try_policy_kb_preflight(original_msg, msg_en, reply_lang)
    if preflight:
        decision, route_data = preflight
        try:
            from services.query_understanding import apply_query_understanding

            route_data = apply_query_understanding(
                route_data, original_msg, msg_en, conv_for_llm, reply_lang=reply_lang
            )
        except Exception as exc:
            log_reasoning(f"query_understanding skipped on preflight (non-fatal): {exc}")
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("kb_preflight")
            store_turn_analysis(route_data, decision)
        except ImportError:
            pass
        return decision, route_data

    # === LLM routing for API/catalog/ambiguous turns ===
    from utils.debug_session_log import dbg

    dbg("H1", "ai_first_router.py:pre_brain", "before ai_brain_route", {"msg_len": len(original_msg or "")})
    route_data = ai_brain_route(original_msg, conv_for_llm, reply_lang=reply_lang)
    if route_data and isinstance(ctx, dict) and ctx.get("last"):
        route_data = dict(route_data)
        route_data["_ctx_last"] = ctx.get("last")
    dbg(
        "H1",
        "ai_first_router.py:post_brain",
        "after ai_brain_route",
        {"has_route": bool(route_data), "intent": (route_data or {}).get("intent")},
    )

    if route_data:
        try:
            from services.catalog_menu_resolver import try_catalog_menu_routing_decision

            menu_dec = try_catalog_menu_routing_decision(
                route_data,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=reply_lang,
                ctx=ctx,
            )
            if menu_dec:
                decision, route_data = menu_dec
                try:
                    from services.chat_flow_telemetry import (
                        record_route_step,
                        store_turn_analysis,
                    )

                    record_route_step("catalog_menu_ai_route")
                    store_turn_analysis(route_data, decision)
                except ImportError:
                    pass
                return decision, route_data
        except ImportError:
            pass
        try:
            from services.product_catalog_resolver import apply_product_catalog_to_route

            route_data = apply_product_catalog_to_route(
                route_data,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=reply_lang,
            )
        except ImportError:
            pass
        try:
            from services.chitchat_resolver import try_chitchat_routing_decision
            from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path

            if not turn_is_pincode_delivery_fast_path(
                original_msg, msg_en, conv_for_llm, ctx
            ):
                scope_fast = try_chitchat_routing_decision(
                    route_data, original_msg, msg_en, conv_for_llm, reply_lang
                )
                if scope_fast:
                    decision, route_data = scope_fast
                    try:
                        from services.chat_flow_telemetry import (
                            record_route_step,
                            store_turn_analysis,
                        )

                        record_route_step("scope_routing_gate")
                        store_turn_analysis(route_data, decision)
                    except ImportError:
                        pass
                    return decision, route_data
        except ImportError:
            pass
        try:
            from services.ai_route_semantics import enrich_route_from_llm

            dbg("H1", "ai_first_router.py:pre_enrich", "before enrich_route_from_llm", {})
            route_data = enrich_route_from_llm(
                route_data, original_msg, msg_en, conv_for_llm
            )
            dbg(
                "H1",
                "ai_first_router.py:post_enrich",
                "after enrich_route_from_llm",
                {"intent": route_data.get("intent"), "scope": route_data.get("conversation_scope")},
            )
        except ImportError:
            pass
        try:
            route_data = apply_ai_route_corrections(route_data, original_msg, msg_en, conv_for_llm)
        except Exception as exc:
            from services.ai_route_semantics import _normalize_llm_route

            log_reasoning(f"apply_ai_route_corrections failed (using normalized route): {exc}")
            route_data = _normalize_llm_route(route_data)
        try:
            from services.query_understanding import apply_query_understanding

            route_data = apply_query_understanding(
                route_data, original_msg, msg_en, conv_for_llm, reply_lang=reply_lang
            )
        except Exception as exc:
            log_reasoning(f"query_understanding skipped (non-fatal): {exc}")
        log_reasoning(
            f"AI-first route: intent={route_data.get('intent')} "
            f"numeric={route_data.get('numeric_context')} "
            f"continue_topic={route_data.get('continue_previous_topic')} — "
            f"{(route_data.get('reasoning') or '')[:100]}"
        )
        decision = _decision_from_ai_route(route_data, original_msg, msg_en, conv_for_llm, ctx=ctx)
        dbg(
            "H1",
            "ai_first_router.py:return",
            "route decision ready",
            {"intent": decision.intent, "handler": decision.handler},
        )
        try:
            from services.chat_flow_telemetry import store_turn_analysis

            store_turn_analysis(route_data, decision)
        except ImportError:
            pass
        return decision, route_data

    # === LLM unavailable — deterministic fallbacks only ===
    conv_fallback = _try_conversational_keyword_fallback(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    if conv_fallback:
        return conv_fallback

    catalog_fallback = _try_catalog_menu_fast_path(original_msg, msg_en, ctx=ctx)
    if catalog_fallback:
        decision, route_data = catalog_fallback
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("catalog_menu_fast_fallback")
            store_turn_analysis(route_data, decision)
        except ImportError:
            pass
        return decision, route_data

    if _is_conversation_acknowledgment(combined) and conv_for_llm:
        if strict_llm_failsafe:
            from services.conversation_scope import (
                SCOPE_OUT,
                resolve_conversation_scope,
            )

            scope_dec = resolve_conversation_scope(
                original_msg, msg_en, conv_for_llm, reply_lang=reply_lang, ai_route=None
            )
            if scope_dec.scope == SCOPE_OUT:
                log_reasoning(
                    "AI-first: LLM down on ack turn — scope fallback out_of_domain."
                )
                return (
                    AnswerRouteDecision(
                        source="reject",
                        intent="out_of_domain",
                        handler="off_topic",
                        is_welfog_related=False,
                        reason="Off-topic while LLM unavailable.",
                    ),
                    {"llm_unavailable": True, "intent": "out_of_domain", "data_channel": "none"},
                )
            log_reasoning("AI-first: LLM providers unavailable on ack turn — deterministic temporary-load reply.")
            return (
                AnswerRouteDecision(
                    source="reject",
                    intent="general",
                    handler="temporary_load",
                    is_welfog_related=True,
                    reason="LLM providers unavailable; skip heuristic fallback.",
                ),
                {"llm_unavailable": True, "intent": "general", "data_channel": "none"},
            )
        from services.answer_router import try_product_shopping_route_decision

        product_dec = try_product_shopping_route_decision(
            original_msg, msg_en, conv_for_llm=conv_for_llm, ai_route=None
        )
        if product_dec:
            log_reasoning("AI-first: ack + Groq down — heuristic product route.")
            sq = (product_dec.search_query or "").strip()
            return product_dec, {
                "intent": "product",
                "is_welfog_related": True,
                "data_channel": "catalog",
                "search_query": sq,
                "continue_previous_topic": True,
            }

    from utils.helpers import (
        _text_has_delivery_or_order_area_intent,
        _text_has_delivery_serviceability_intent,
        _text_has_pincode_delivery_intent,
        _text_has_refund_or_return_intent,
        _text_is_live_order_lookup_intent,
        _user_denies_pincode_insists_order_id,
        _user_explicitly_asks_payment_status,
        extract_embedded_query_identifiers,
        resolve_live_api_intent_from_conversation,
    )

    from utils.helpers import user_turn_qualifies_for_live_order_api

    ids = extract_embedded_query_identifiers(original_msg, msg_en, conv_for_llm)
    oid_early = (ids.get("order_id") or "").strip()
    if (
        oid_early
        and ids.get("numeric_context") == "order_id"
        and user_turn_qualifies_for_live_order_api(
            original_msg, msg_en, conv_for_llm, ai_route=None
        )
    ):
        live_intent = resolve_live_api_intent_from_conversation(
            conv_for_llm,
            ctx_last=(ctx or {}).get("last") if isinstance(ctx, dict) else None,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if _text_has_refund_or_return_intent(combined):
            live_intent = "refund"
        elif _user_explicitly_asks_payment_status(combined):
            live_intent = "payment"
        log_reasoning(f"AI-first: Order ID {oid_early} in query — live {live_intent} API (not pincode).")
        handler = "order_tracking_api" if live_intent == "order" else "ai_route_and_answer"
        return (
            AnswerRouteDecision(
                source="api",
                intent=live_intent,
                handler=handler,
                is_welfog_related=True,
                reason=f"Live {live_intent} lookup for Order ID {oid_early} already in user message.",
            ),
            {
                "intent": live_intent,
                "needs_order_id": True,
                "numeric_context": "order_id",
                "extracted_pincode": "",
                "data_channel": "live_api",
                "is_welfog_related": True,
                "search_query": "",
            },
        )

    if (ids.get("pincode") or "").strip() and (
        _text_has_pincode_delivery_intent(combined, conv_for_llm)
        or _text_has_delivery_serviceability_intent(combined, conv_for_llm)
        or _text_has_delivery_or_order_area_intent(combined)
    ) and not (
        _text_is_live_order_lookup_intent(combined, conv_for_llm)
        or _user_denies_pincode_insists_order_id(combined)
    ):
        pin = ids["pincode"]
        log_reasoning(f"AI-first: PIN {pin} embedded in query — pincode API.")
        return (
            AnswerRouteDecision(
                source="api",
                intent="pincode_check",
                handler="pincode_delivery_api",
                is_welfog_related=True,
                reason=f"Delivery check for PIN {pin} already in user message.",
            ),
            {
                "intent": "pincode_check",
                "extracted_pincode": pin,
                "numeric_context": "pincode",
                "needs_order_id": False,
                "data_channel": "live_api",
                "is_welfog_related": True,
                "search_query": "",
            },
        )

    from services.answer_router import resolve_structured_procedure_route

    structured_pre = resolve_structured_procedure_route(
        original_msg, msg_en, conversation_context=conv_for_llm
    )
    if structured_pre:
        log_reasoning(f"LLM down — structured procedure fallback: {structured_pre.handler}")
        return structured_pre, None

    try:
        from services.account_list_semantics import turn_requests_purchase_history_in_chat

        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conv_for_llm, ai_route=None
        ):
            log_reasoning("AI-first: LLM down — order history list → purchase-history API.")
            return (
                AnswerRouteDecision(
                    source="api",
                    intent="order_history",
                    handler="order_ai_flow",
                    is_welfog_related=True,
                    reason="Purchase history list in chat (heuristic while LLM unavailable).",
                ),
                {
                    "intent": "order_history",
                    "data_channel": "live_api",
                    "needs_order_id": False,
                    "is_welfog_related": True,
                    "search_query": "",
                },
            )
    except ImportError:
        pass

    legacy = resolve_answer_route(
        original_msg, msg_en, retrieval_query=retrieval_query, conv_for_llm=conv_for_llm, ctx=ctx
    )
    if legacy and (legacy.intent or legacy.handler):
        log_reasoning(
            f"LLM down — legacy route fallback: intent={legacy.intent} handler={legacy.handler}"
        )
        return legacy, None

    if strict_llm_failsafe:
        from services.conversation_scope import SCOPE_OUT, resolve_conversation_scope

        scope_dec = resolve_conversation_scope(
            original_msg, msg_en, conv_for_llm, reply_lang=reply_lang, ai_route=None
        )
        if scope_dec.scope == SCOPE_OUT:
            log_reasoning(
                "AI-first: LLM unavailable — scope fallback out_of_domain (no temporary-load)."
            )
            return (
                AnswerRouteDecision(
                    source="reject",
                    intent="out_of_domain",
                    handler="off_topic",
                    is_welfog_related=False,
                    reason="Off-topic while LLM unavailable.",
                ),
                {"llm_unavailable": True, "intent": "out_of_domain", "data_channel": "none"},
            )
        log_reasoning("AI-first: LLM providers unavailable — deterministic temporary-load reply.")
        return (
            AnswerRouteDecision(
                source="reject",
                intent="general",
                handler="temporary_load",
                is_welfog_related=True,
                reason="LLM providers unavailable; skip legacy/random fallback.",
            ),
            {"llm_unavailable": True, "intent": "general", "data_channel": "none"},
        )

    from services.answer_router import try_product_shopping_route_decision

    product_dec = try_product_shopping_route_decision(
        original_msg, msg_en, conv_for_llm=conv_for_llm, ai_route=None
    )
    if product_dec:
        log_reasoning("AI-first: Groq unavailable — heuristic product route.")
        sq = (product_dec.search_query or "").strip()
        return product_dec, {
            "intent": "product",
            "is_welfog_related": True,
            "data_channel": "catalog",
            "search_query": sq,
        }
    log_reasoning("AI-first: all routes exhausted — generic legacy router.")
    return legacy, None
