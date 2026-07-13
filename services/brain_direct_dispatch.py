"""
One brain LLM → immediate handler (no second routing pass).

Product: brain intent=catalog → OpenSearch (one pass).
Account list / refund / chitchat: same pattern.
"""
from __future__ import annotations

import re

from typing import Callable, Optional

from services.product_search_flow import run_product_search_ai_flow
from utils.reasoning_log import log_reasoning


def _brain_route_is_personal_order_live(brain_route: dict) -> bool:
    """
    True when brain JSON locks a single-order live API turn.
    Uses semantic route fields only — no customer-message keyword lists.
    """
    if not isinstance(brain_route, dict):
        return False
    olk = (brain_route.get("order_lookup_kind") or "").strip().lower()
    rh = (brain_route.get("route_handler") or "").strip().lower()
    channel = (brain_route.get("data_channel") or "").strip().lower()
    intent = (brain_route.get("intent") or "").strip().lower()
    needs_oid = bool(brain_route.get("needs_order_id"))

    if rh in ("order_tracking_api", "order_details_api", "refund_status_api"):
        return True
    if olk in (
        "track",
        "tracking",
        "invoice",
        "details",
        "order_details",
        "refund_status",
        "payment",
    ):
        return True
    if intent in ("order", "refund", "payment") and channel in ("live_api", "order"):
        return True
    if needs_oid and intent in ("order", "refund", "payment"):
        return True
    if needs_oid and olk in (
        "track",
        "tracking",
        "invoice",
        "details",
        "order_details",
        "refund_status",
    ):
        return True
    return False


def _scope_reply_is_placeholder(text: str) -> bool:
    """Brain sometimes echoes prompt instructions instead of a real customer reply."""
    low = (text or "").strip().lower()
    if not low:
        return True
    if low in (
        "warm",
        "warm reply",
        "warm natural reply",
        "natural reply",
        "scope reply",
        "customer language",
        "product_search",
        "product",
        "catalog",
        "catalog_search",
        "opensearch",
        "order_tracking",
        "order_details",
        "wishlist",
        "kb",
        "live_api",
        "out_of_domain",
        "general_chitchat",
        "welfog_support",
    ):
        return True
    if len(low) < 12:
        return low in ("warm", "thanks", "ok", "okay", "hi", "hello") or (
            len(low) <= 4 and not any(c.isalpha() for c in low[1:])
        )
    markers = (
        "warm natural reply",
        "customer language",
        "scope_reply",
        "in the customer",
        "1-3 sentences",
        "2-3 sentences",
        "2-5 sentences",
        "mirroring the user",
        "exact language",
        "script, and slang",
        "empty when",
        "required when",
    )
    return any(m in low for m in markers)


def _try_brain_immediate_scope_or_kb_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable | None = None,
    reply_for_live_order_id_lookup: Callable | None = None,
) -> Optional[str]:
    """
    Brain already classified this turn — answer OOD/chitchat/KB immediately.
    Skips pincode/product/order stacks and duplicate micro-LLM classifiers.
    """
    if not isinstance(brain_route, dict) or brain_route.get("llm_unavailable"):
        return None

    try:
        from services.chat_flow_telemetry import should_skip_post_pincode_route_steal

        if should_skip_post_pincode_route_steal("_try_brain_immediate_scope_or_kb_reply"):
            return None
    except ImportError:
        pass

    scope = (brain_route.get("conversation_scope") or "").strip().lower()
    intent = (brain_route.get("intent") or "").strip().lower()
    channel = (brain_route.get("data_channel") or "").strip().lower()
    kb_keys = [str(k).strip() for k in (brain_route.get("kb_keys") or []) if str(k).strip()]
    authoritative_kb_json = (
        channel == "kb"
        and bool(kb_keys)
        and not brain_route.get("needs_order_id")
        and brain_route.get("is_welfog_related", True) is not False
    )
    try:
        from services.chat_flow_telemetry import (
            brain_route_authoritative_kb_lock,
            is_authoritative_kb_route_locked,
        )

        kb_authoritative = (
            is_authoritative_kb_route_locked()
            or brain_route_authoritative_kb_lock(brain_route)
        )
    except ImportError:
        kb_authoritative = channel == "kb"

    if kb_authoritative and channel == "kb" and not brain_route.get("needs_order_id"):
        if intent not in (
            "order",
            "order_history",
            "wishlist",
            "product",
            "pincode_check",
            "deals",
            "categories",
        ):
            kb_body = _try_brain_kb_locked_reply(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=str((ctx or {}).get("user_id") or ""),
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn or (lambda _u: None),
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup
                or (lambda *a, **k: None),
            )
            if kb_body:
                log_reasoning(
                    "Brain immediate — authoritative KB route (OOD guards bypassed)."
                )
                return kb_body
            meaning = (brain_route.get("user_meaning") or "").strip()
            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            query = meaning or comb
            keys = list(brain_route.get("kb_keys") or [])
            rh = (brain_route.get("route_handler") or "").strip().lower()
            try:
                from services.kb_service import (
                    KB_DIRECT_MIN_SCORE,
                    direct_kb_search,
                    format_kb_answer_from_brain_keys,
                    format_welfog_social_media_reply_from_kb,
                )

                if keys:
                    file_body = format_kb_answer_from_brain_keys(
                        original_msg,
                        msg_en,
                        keys,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        user_meaning_en=meaning,
                        ai_route=brain_route,
                    )
                    if file_body:
                        log_reasoning(
                            "Brain immediate — authoritative KB grounded answer."
                        )
                        return file_body
                if "social" in rh:
                    social = format_welfog_social_media_reply_from_kb(
                        original_msg,
                        msg_en,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        user_meaning_en=meaning,
                        ai_confirmed=True,
                    )
                    if social:
                        return social
                if not keys:
                    body = direct_kb_search(
                        query,
                        keys=keys or None,
                        min_score=KB_DIRECT_MIN_SCORE,
                        zero_llm_fast=True,
                    )
                    if body:
                        return body
            except ImportError:
                pass

    if intent == "out_of_domain":
        scope = "out_of_domain"
    elif not scope and intent in ("general",):
        scope = "general_chitchat"

    # --- Pure chitchat — never KB vector / catalog ---
    if scope == "general_chitchat" and not authoritative_kb_json:
        if channel == "kb":
            channel = "none"
        if not _brain_is_pincode_delivery_turn(brain_route) and not _brain_is_product_catalog_turn(
            brain_route, original_msg, msg_en
        ):
            scope_reply = (brain_route.get("scope_reply") or "").strip()
            if _scope_reply_is_placeholder(scope_reply):
                scope_reply = ""
            if not scope_reply:
                try:
                    from services.conversational_ack_flow import resolve_brain_scope_customer_reply

                    scope_reply = resolve_brain_scope_customer_reply(
                        brain_route,
                        scope,
                        original_msg,
                        msg_en=msg_en,
                        conversation_context=conv_for_llm,
                        reply_lang=lang,
                        ctx=ctx,
                    )
                except ImportError:
                    pass
            if scope_reply:
                log_reasoning(
                    "Brain immediate — chitchat scope_reply (skip KB from user_meaning)."
                )
                try:
                    from services.chat_flow_telemetry import mark_routing_complete

                    mark_routing_complete()
                except ImportError:
                    pass
                return scope_reply

    # --- Admin KB semantic check BEFORE OOD refusal ---
    # Indexed Admin knowledge is SoT for whether a topic is answerable. Brain may
    # mislabel platform policies (reels, shorts, etc.) as OOD "not shopping".
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog
        from utils.helpers import turn_is_obvious_product_shopping_turn

        skip_kb_probe = (
            brain_route_indicates_product_catalog(brain_route)
            or turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conv_for_llm
            )
            or intent
            in (
                "order",
                "order_history",
                "wishlist",
                "product",
                "pincode_check",
                "deals",
                "categories",
            )
        )
    except ImportError:
        skip_kb_probe = False

    if not skip_kb_probe and (
        scope in ("out_of_domain", "harm_sensitive")
        or not brain_route.get("is_welfog_related", True)
        or (brain_route.get("data_channel") or "").strip().lower() != "kb"
    ):
        try:
            from services.kb_service import promote_route_from_semantic_kb_match

            rescued = promote_route_from_semantic_kb_match(
                brain_route, original_msg, msg_en=msg_en
            )
            if rescued:
                brain_route = rescued
                scope = (brain_route.get("conversation_scope") or "").strip().lower()
                intent = (brain_route.get("intent") or "").strip().lower()
                try:
                    from services.chat_flow_telemetry import store_brain_route_result

                    store_brain_route_result(brain_route)
                except ImportError:
                    pass
        except ImportError:
            pass

    # --- OOD / chitchat before transactional detours (after Admin KB rescue) ---
    try:
        from services.chat_flow_telemetry import should_skip_post_kb_ood_guard

        skip_ood = should_skip_post_kb_ood_guard("_try_brain_immediate_scope_or_kb_reply")
    except ImportError:
        skip_ood = kb_authoritative

    if not skip_ood and (
        scope in ("out_of_domain", "harm_sensitive")
        or not brain_route.get("is_welfog_related", True)
    ):
        scope = scope or "out_of_domain"
        scope_reply = (brain_route.get("scope_reply") or "").strip()
        if _scope_reply_is_placeholder(scope_reply):
            scope_reply = ""
        if not scope_reply:
            try:
                from services.conversational_ack_flow import (
                    ai_ood_reply,
                    resolve_brain_scope_customer_reply,
                )

                scope_reply = resolve_brain_scope_customer_reply(
                    brain_route,
                    scope,
                    original_msg,
                    msg_en=msg_en,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ctx=ctx,
                ) or ai_ood_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm,
                    lang,
                )
            except ImportError:
                pass
        if scope_reply:
            log_reasoning(f"Brain immediate — OOD/scope ({scope}).")
            try:
                from services.chat_flow_telemetry import mark_routing_complete

                mark_routing_complete()
            except ImportError:
                pass
            return scope_reply

    # --- KB vector for Welfog support turns (Admin catalog — unscoped semantic) ---
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog
        from utils.helpers import turn_is_obvious_product_shopping_turn

        if not (
            brain_route_indicates_product_catalog(brain_route)
            or turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conv_for_llm
            )
        ):
            from services.knowledge_query_pipeline import try_knowledge_vector_only_reply

            kb_vec = try_knowledge_vector_only_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=lang,
                ai_route=brain_route,
                embedding_only=True,
            )
            if kb_vec and kb_vec.strip():
                log_reasoning("Brain immediate — admin KB vector (Welfog support turn).")
                return kb_vec
    except ImportError:
        pass

    # --- KB policy (refund/shipping/faqs) — must run before product/order stacks ---
    if channel == "kb" and not brain_route.get("needs_order_id"):
        if intent not in (
            "order",
            "order_history",
            "wishlist",
            "product",
            "pincode_check",
            "deals",
            "categories",
        ):
            kb_body = _try_brain_kb_locked_reply(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=str((ctx or {}).get("user_id") or ""),
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn or (lambda _u: None),
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup
                or (lambda *a, **k: None),
            )
            if kb_body:
                return kb_body
            meaning = (brain_route.get("user_meaning") or "").strip()
            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            query = meaning or comb
            keys = list(brain_route.get("kb_keys") or [])
            rh = (brain_route.get("route_handler") or "").strip().lower()
            try:
                from services.kb_service import (
                    KB_DIRECT_MIN_SCORE,
                    direct_kb_search,
                    format_kb_answer_from_brain_keys,
                    format_welfog_social_media_reply_from_kb,
                )

                if keys:
                    file_body = format_kb_answer_from_brain_keys(
                        original_msg,
                        msg_en,
                        keys,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        user_meaning_en=meaning,
                        ai_route=brain_route,
                    )
                    if file_body:
                        log_reasoning(
                            "Brain immediate — KB file from brain keys (no vector scan)."
                        )
                        return file_body
                if "social" in rh:
                    social = format_welfog_social_media_reply_from_kb(
                        original_msg,
                        msg_en,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        user_meaning_en=meaning,
                        ai_confirmed=True,
                    )
                    if social:
                        log_reasoning(
                            "Brain immediate — Welfog social from KB (no extra LLM)."
                        )
                        return social
                if not keys:
                    body = direct_kb_search(
                        query,
                        keys=keys or None,
                        min_score=KB_DIRECT_MIN_SCORE,
                        zero_llm_fast=True,
                    )
                    if body:
                        log_reasoning(
                            "Brain immediate — vector KB from brain keys (no extra LLM)."
                        )
                        return body
            except ImportError:
                pass

    # --- Welfog informational KB (welfog_support only — never general_chitchat) ---
    if (
        scope in ("welfog_support", "")
        and brain_route.get("is_welfog_related", True)
        and (brain_route.get("user_meaning") or "").strip()
    ):
        if not _brain_is_pincode_delivery_turn(brain_route) and not _brain_is_product_catalog_turn(
            brain_route, original_msg, msg_en
        ):
            try:
                from services.kb_service import (
                    format_kb_answer_from_brain_keys,
                    resolve_brain_kb_keys,
                )

                kb_keys = resolve_brain_kb_keys(
                    brain_route,
                    original_msg,
                    msg_en,
                    conversation_context=conv_for_llm,
                )
                if kb_keys:
                    kb_from_meaning = format_kb_answer_from_brain_keys(
                        original_msg,
                        msg_en,
                        kb_keys,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        user_meaning_en=(brain_route.get("user_meaning") or "").strip(),
                        ai_route=brain_route,
                    )
                    if kb_from_meaning:
                        log_reasoning(
                            "Brain immediate — KB from user_meaning (skip chitchat LLM)."
                        )
                        try:
                            from services.chat_flow_telemetry import mark_routing_complete

                            mark_routing_complete()
                        except ImportError:
                            pass
                        return kb_from_meaning
            except ImportError:
                pass

    # --- OOD / chitchat (brain scope_reply — no catalog/pincode detours) ---
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        if _brain_is_pincode_delivery_turn(brain_route) or _brain_is_product_catalog_turn(
            brain_route, original_msg, msg_en
        ):
            return None
        scope_reply = (brain_route.get("scope_reply") or "").strip()
        if _scope_reply_is_placeholder(scope_reply):
            scope_reply = ""
        if not scope_reply:
            try:
                from services.conversational_ack_flow import resolve_brain_scope_customer_reply

                scope_reply = resolve_brain_scope_customer_reply(
                    brain_route,
                    scope,
                    original_msg,
                    msg_en=msg_en,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ctx=ctx,
                )
            except ImportError:
                pass
        if scope_reply:
            log_reasoning(
                f"Brain immediate — scope_reply ({scope}), no duplicate scope LLM."
            )
            try:
                from services.chat_flow_telemetry import mark_routing_complete

                mark_routing_complete()
            except ImportError:
                pass
            return scope_reply

    return None


def _try_brain_kb_locked_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable,
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """Brain channel=kb → locked KB handler (no second routing LLM stack)."""
    if _brain_route_is_personal_order_live(brain_route):
        return None
    try:
        from services.refund_status_semantics import (
            current_turn_wants_personal_refund_status,
            refund_status_route_is_locked,
        )

        if refund_status_route_is_locked(brain_route) or current_turn_wants_personal_refund_status(
            original_msg,
            msg_en,
            conv_for_llm,
            brain_route,
            allow_llm=False,
        ):
            return None
    except ImportError:
        pass
    try:
        from services.order_details_flow import order_details_route_is_locked

        olk = (brain_route.get("order_lookup_kind") or "").strip().lower()
        if order_details_route_is_locked(brain_route) or olk == "invoice":
            return None
    except ImportError:
        pass
    channel = (brain_route.get("data_channel") or "").strip().lower()
    if channel != "kb" or brain_route.get("needs_order_id"):
        return None
    intent = (brain_route.get("intent") or "").strip().lower()
    if intent in (
        "order",
        "order_history",
        "wishlist",
        "product",
        "pincode_check",
        "deals",
        "categories",
    ):
        return None
    kb_keys = list(brain_route.get("kb_keys") or [])
    if not kb_keys:
        try:
            from services.kb_service import resolve_brain_kb_keys

            kb_keys = resolve_brain_kb_keys(
                brain_route,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
            )
        except ImportError:
            kb_keys = []
    try:
        from services.kb_service import format_kb_answer_from_brain_keys

        direct = format_kb_answer_from_brain_keys(
            original_msg,
            msg_en,
            kb_keys,
            reply_lang=lang,
            conversation_context=(
                ""
                if brain_route.get("_preflight_kb")
                else conv_for_llm
            ),
            user_meaning_en=(brain_route.get("user_meaning") or "").strip(),
            ai_route=brain_route,
        )
        if direct:
            log_reasoning(
                f"Brain direct dispatch: KB keys={','.join(kb_keys[:3])} (no routing stack)."
            )
            try:
                from services.chat_flow_telemetry import mark_routing_complete

                mark_routing_complete()
            except ImportError:
                pass
            reset_context_fn(ctx)
            return direct
    except Exception as exc:
        log_reasoning(f"Brain KB direct dispatch skipped: {exc}")
    return None


def _mark_product_routing_complete(brain_route: dict, sq: str) -> None:
    try:
        from services.answer_router import AnswerRouteDecision
        from services.chat_flow_telemetry import store_turn_analysis

        decision = AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason="Brain direct dispatch — product catalog.",
        )
        store_turn_analysis(brain_route, decision)
    except ImportError:
        pass


def _brain_is_product_catalog_turn(brain: dict, original_msg: str, msg_en: str) -> bool:
    if _brain_is_pincode_delivery_turn(brain):
        return False
    try:
        from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

        if ai_meaning_describes_delivery_serviceability(brain):
            return False
    except ImportError:
        pass
    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            reconcile_account_list_from_brain_meaning,
            _kind_from_meaning_blob,
            _meaning_blob,
            KIND_PURCHASE_HOWTO,
            KIND_PURCHASE_IN_CHAT,
            KIND_WISHLIST_HOWTO,
            KIND_WISHLIST_IN_CHAT,
        )

        fixed = reconcile_account_list_from_brain_meaning(dict(brain))
        if account_list_route_is_locked(fixed):
            return False
        mk = _kind_from_meaning_blob(_meaning_blob(fixed))
        if mk in (
            KIND_WISHLIST_IN_CHAT,
            KIND_WISHLIST_HOWTO,
            KIND_PURCHASE_IN_CHAT,
            KIND_PURCHASE_HOWTO,
        ):
            return False
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(brain) or brain.get("_product_catalog_locked"):
            return True
        if brain_route_indicates_product_catalog(brain):
            return True
    except ImportError:
        pass
    intent = (brain.get("intent") or "").strip().lower()
    channel = (brain.get("data_channel") or "").strip().lower()
    if intent in ("categories", "category_feed"):
        return False
    if intent == "category_browse" and not (brain.get("category_browse") or "").strip():
        if not (brain.get("search_query") or "").strip() and not brain.get("run_catalog_search"):
            return False
    if brain.get("category_only_browse") or (brain.get("category_browse") or "").strip():
        return True
    meta = (brain.get("meta_kind") or "none").strip().lower()
    if meta not in ("", "none"):
        return False
    if intent == "product" and channel == "catalog":
        return True
    if channel == "catalog" and brain.get("run_catalog_search"):
        return True
    if intent == "product" and (brain.get("search_query") or brain.get("run_catalog_search")):
        return True
    return intent == "product"


def _prepare_brain_product_route(
    brain: dict,
    original_msg: str,
    msg_en: str,
) -> tuple[dict, str]:
    route = dict(brain)
    route["_universal_brain_route"] = True
    route["intent"] = "product"
    route["data_channel"] = "catalog"
    route["run_catalog_search"] = True
    route["_product_catalog_locked"] = True
    route["needs_order_id"] = False
    route["numeric_context"] = route.get("numeric_context") or "none"
    route["meta_kind"] = "none"

    sq = ""
    entities: dict = {}
    try:
        from services.ai_route_semantics import _brain_product_entities_from_route

        entities = _brain_product_entities_from_route(route, original_msg=original_msg, msg_en=msg_en)
        if entities:
            route["_product_entities"] = entities
    except ImportError:
        pass

    try:
        from services.product_filter_pipeline import _enrich_brain_entities_structural

        entities = _enrich_brain_entities_structural(
            entities or route.get("_product_entities") or {},
            original_msg,
            msg_en,
            brain_route=route,
        )
        if entities:
            route["_product_entities"] = entities
    except ImportError:
        pass

    try:
        from services.ai_route_semantics import resolve_catalog_search_phrase

        sq = resolve_catalog_search_phrase(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
        )
    except ImportError:
        sq = (route.get("search_query") or "").strip()

    if not sq:
        from utils.reasoning_log import log_reasoning

        log_reasoning(
            "Brain product turn — no structured product_name entity; "
            "Product Entity Extraction will own title_query (never Brain user_meaning)."
        )

    # Keep Brain entity hint only. Never stamp user_meaning / paraphrases into search_query
    # as OpenSearch title_query — extract_semantic_product_entities is the SoT.
    route["search_query"] = sq
    route["_ai_single_pass"] = True
    # Title always needs semantic product entity extraction (unless SKU / pro_id).
    pe = route.get("_product_entities") or {}
    if pe.get("sku") or pe.get("pro_id") or pe.get("product_id"):
        route["_needs_product_nlu_llm"] = False
    elif route.get("category_only_browse") and not sq:
        route["_needs_product_nlu_llm"] = False
    else:
        route["_needs_product_nlu_llm"] = True
    return route, sq


def _resolve_browse_category_id_fast(
    brain_route: dict,
    original_msg: str,
    msg_en: str,
    *,
    ctx: dict | None = None,
) -> Optional[str]:
    """Top-level nav departments only — never inner_categories fanout."""
    try:
        from services.welfog_api import resolve_nav_category_id_fast
    except ImportError:
        return None

    cat_id_raw = brain_route.get("category_id") or brain_route.get("extracted_category_id")
    if cat_id_raw is not None and str(cat_id_raw).strip():
        try:
            return str(int(str(cat_id_raw).strip()))
        except (TypeError, ValueError):
            pass

    texts: list[str] = []
    for raw in (
        brain_route.get("category_browse"),
        brain_route.get("search_query"),
        brain_route.get("user_meaning"),
        original_msg,
        msg_en,
        f"{original_msg} {msg_en}".strip(),
    ):
        t = (raw or "").strip()
        if t and t not in texts:
            texts.append(t)
    for t in texts:
        cid = resolve_nav_category_id_fast(t, ctx=ctx)
        if cid:
            return cid
    return None


def _brain_route_is_category_only_browse(
    brain_route: dict,
    original_msg: str,
    msg_en: str,
) -> bool:
    # Concrete product search terms always win over department browse.
    sq = (brain_route.get("search_query") or "").strip()
    if sq:
        return False
    entities = brain_route.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}
    pe = brain_route.get("product_entities") or brain_route.get("_product_entities") or {}
    if not isinstance(pe, dict):
        pe = {}
    if any(
        (entities.get(k) or pe.get(k))
        for k in (
            "product_name",
            "sku",
            "pro_id",
            "product_id",
            "brand",
            "color",
            "size",
            "model",
            "product_type",
        )
    ):
        return False
    try:
        from services.welfog_api import (
            _message_has_product_search_filters,
            _message_targets_specific_product,
            _user_explicitly_browses_category,
            resolve_nav_category_id_fast,
        )

        comb = f"{original_msg} {msg_en}".strip()
        if comb and _message_has_product_search_filters(comb):
            return False
        if comb and _message_targets_specific_product(comb):
            return False
        if bool(brain_route.get("category_only_browse")) or (
            brain_route.get("category_browse") or ""
        ).strip():
            if comb and _user_explicitly_browses_category(comb):
                return bool(resolve_nav_category_id_fast(comb))
            # Brain said category browse but message has no specific product —
            # only treat as category-only when it is a bare department ask.
            if comb and not _message_targets_specific_product(comb):
                return True
            return False
        if comb and _user_explicitly_browses_category(comb):
            return bool(resolve_nav_category_id_fast(comb))
    except ImportError:
        pass
    return False


def _try_brain_category_browse_direct_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    Brain already classified category browse → skip heavy product pipeline and do
    category-only catalog search directly (OpenSearch + REST fallback).
    """
    if not isinstance(brain_route, dict):
        return None

    cat_browse = (brain_route.get("category_browse") or "").strip()
    cat_only = _brain_route_is_category_only_browse(brain_route, original_msg, msg_en)
    candidate = (
        cat_browse
        or (brain_route.get("search_query") or "").strip()
        or (brain_route.get("user_meaning") or "").strip()
    )

    try:
        from services.welfog_api import (
            category_name_for_id,
            resolve_nav_category_id_fast,
        )
        from services.opensearch_products import (
            catalog_search_live,
            product_search_show_view_more,
            sanitize_product_search_spec,
            build_product_rail_with_pagination,
        )
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer
        from services.welfog_api import build_welfog_product_browse_url
    except ImportError:
        return None

    # Heuristic safety: only treat as category-only when we don't see product-specific filters.
    entities = brain_route.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}
    has_specific_product_filters = any(
        entities.get(k)
        for k in (
            "product_name",
            "sku",
            "pro_id",
            "product_id",
            "brand",
            "color",
            "size",
            "model",
            "search_terms",
        )
    )

    if has_specific_product_filters:
        return None
    if not cat_only and not cat_browse:
        return None

    cid = _resolve_browse_category_id_fast(
        brain_route, original_msg, msg_en, ctx=ctx
    )
    if not cid:
        return None

    cat_label = (cat_browse or "").strip().title()
    if not cat_label:
        cat_label = category_name_for_id(str(cid), ctx=ctx) or f"Category {cid}"

    import time as _time

    _t_cat = _time.perf_counter()
    os_spec = sanitize_product_search_spec(
        {
            "category_id": int(cid),
            "title_query": "",
            "_category_only_browse": True,
            # Fast mode: no tiered relax; still allows REST fallback for category-only.
            "_ai_single_pass": True,
        }
    )
    products, os_spec, _os_total, os_has_more = catalog_search_live(
        os_spec,
        original_msg=original_msg,
        msg_en=msg_en,
        ctx=ctx,
    )
    _catalog_ms = (_time.perf_counter() - _t_cat) * 1000.0

    if not products:
        body = localized_sysmsg_for_customer(
            "product_not_found",
            original_msg,
            reply_lang=lang,
            query=cat_label,
            fallback_en=sysmsg("product_not_found", query=cat_label) or "",
        ) or sysmsg("product_not_found", query=cat_label)
        reset_context_fn(ctx)
        return body

    response_text = sysmsg("products_title_query", query=cat_label) or ""
    if cat_browse:
        from services.welfog_api import WELFOG_SITE_BASE, _slugify_category_label

        slug = _slugify_category_label(cat_browse)
        browse_url = (
            f"{WELFOG_SITE_BASE}/category/{slug}" if slug else build_welfog_product_browse_url(os_spec, ctx=ctx)
        )
    else:
        browse_url = build_welfog_product_browse_url(os_spec, ctx=ctx)
    response_text += build_product_rail_with_pagination(
        products,
        sysmsg,
        has_more=product_search_show_view_more(products, os_has_more),
        next_page=2,
        browse_more_url=browse_url,
    )
    _total_ms = (_time.perf_counter() - _t_cat) * 1000.0
    log_reasoning(
        f"Category browse direct: catalog={_catalog_ms:.0f}ms total={_total_ms:.0f}ms "
        f"({len(products)} cards, category_id={cid})."
    )
    reset_context_fn(ctx)
    return response_text


def _try_brain_product_catalog_fallback_dispatch(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    One ai_brain_route already ran — OpenSearch catalog (no specialist LLM stack).
    """
    if not isinstance(brain_route, dict) or brain_route.get("llm_unavailable"):
        return None
    try:
        from services.chat_flow_telemetry import should_skip_post_pincode_route_steal

        if should_skip_post_pincode_route_steal("product_catalog_fallback"):
            return None
    except ImportError:
        pass
    if brain_route.get("_pincode_delivery_locked") or (
        (brain_route.get("intent") or "").strip().lower() == "pincode_check"
    ):
        return None
    intent_ub = (brain_route.get("intent") or "").strip().lower()
    if intent_ub in ("categories", "category_feed"):
        return None
    if intent_ub == "category_browse" and not (brain_route.get("category_browse") or "").strip():
        if not (brain_route.get("search_query") or "").strip():
            return None
    channel = (brain_route.get("data_channel") or "").strip().lower()
    if channel == "kb":
        return None
    if not _brain_is_product_catalog_turn(brain_route, original_msg, msg_en):
        return None
    product_route, sq = _prepare_brain_product_route(
        brain_route, original_msg, msg_en
    )
    body = _run_product_catalog_flow(
        product_route,
        sq,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )
    if body:
        log_reasoning(
            f"Brain product catalog fallback: sq={sq!r} (zero extra LLM)."
        )
        try:
            from services.chat_flow_telemetry import mark_routing_complete

            mark_routing_complete()
        except ImportError:
            pass
    return body


def _run_product_catalog_flow(
    product_route: dict,
    sq: str,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    try:
        from utils.helpers import (
            clear_order_session_for_new_lookup,
            turn_is_obvious_product_shopping_turn,
        )

        product_turn = bool(
            product_route.get("_product_catalog_locked")
            or (product_route.get("intent") or "").strip().lower() == "product"
        )
        if not product_turn:
            product_turn = turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conv_for_llm
            )
        if product_turn and isinstance(ctx, dict) and (
            ctx.get("awaiting")
            or (ctx.get("data") or {}).get("pending_action")
        ):
            log_reasoning(
                "Product catalog — sanitize stale order session before OpenSearch."
            )
            clear_order_session_for_new_lookup(ctx)
            ctx["awaiting"] = None
            ctx["last"] = None
    except ImportError:
        pass
    ctx.setdefault("data", {})["ai_route"] = product_route
    try:
        ps = run_product_search_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
            search_query=sq,
            ai_route=product_route,
        )
        if ps.handled and ps.reply_html:
            try:
                import threading

                threading.Thread(
                    target=lambda: _mark_product_routing_complete(product_route, sq),
                    daemon=True,
                ).start()
            except Exception:
                _mark_product_routing_complete(product_route, sq)
            reset_context_fn(ctx)
            return ps.reply_html
        _mark_product_routing_complete(product_route, sq)
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer

        body = localized_sysmsg_for_customer(
            "product_not_found", original_msg, reply_lang=lang, query=sq
        ) or sysmsg("product_not_found", query=sq)
        reset_context_fn(ctx)
        return body
    except ImportError:
        return None


def try_explicit_sku_catalog_reply(
    original_msg: str,
    msg_en: str,
    *,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    Warehouse SKU in message → OpenSearch only (zero LLM).
    Handles explicit 'sku' mentions and standalone catalog codes (e.g. SSKKOO).
    """
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return None
    sku: str | None = None
    bare = (original_msg or "").strip()
    try:
        from services.catalog_spec_semantics import user_mentions_sku_this_turn
        from services.opensearch_products import (
            _extract_sku_from_text,
            _sku_token_acceptable,
            build_product_rail_with_pagination,
            catalog_search_live,
            product_search_show_view_more,
            sanitize_product_search_spec,
        )
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer
        from services.welfog_api import build_welfog_product_browse_url

        if re.search(r"\bsku\b", comb, re.I):
            sku = _extract_sku_from_text(original_msg) or _extract_sku_from_text(msg_en)
            if sku and not _sku_token_acceptable(sku, explicit_sku_mention=True):
                sku = None
        elif bare and " " not in bare and user_mentions_sku_this_turn(bare):
            if _sku_token_acceptable(bare):
                sku = bare
        if not sku:
            return None
    except ImportError:
        return None

    log_reasoning(f"Structural SKU fast path ({sku!r}) → OpenSearch (zero LLM).")
    product_route: dict = {
        "intent": "product",
        "data_channel": "catalog",
        "run_catalog_search": True,
        "_product_catalog_locked": True,
        "_product_entities": {"sku": sku},
        "_ai_single_pass": True,
        "search_query": "",
        "user_meaning": f"Product for SKU {sku}",
        "needs_order_id": False,
        "numeric_context": "none",
        "meta_kind": "none",
        "is_welfog_related": True,
        "route_handler": "sku_structural_fast",
    }
    ctx.setdefault("data", {})["ai_route"] = product_route

    os_spec = sanitize_product_search_spec(
        {"sku": sku, "title_query": "", "_ai_single_pass": True, "strict_sku_match": False}
    )
    products, os_spec, _os_total, os_has_more = catalog_search_live(
        os_spec,
        original_msg=original_msg,
        msg_en=msg_en,
        ctx=ctx,
    )
    display_query = f"SKU {sku}"
    if not products:
        body = localized_sysmsg_for_customer(
            "product_not_found",
            original_msg,
            lang,
            query=display_query,
            fallback_en=sysmsg("product_not_found", query=display_query) or "",
        )
        reset_context_fn(ctx)
        return body

    response_text = sysmsg("products_title_query", query=display_query) or ""
    browse_url = build_welfog_product_browse_url(os_spec, ctx=ctx)
    response_text += build_product_rail_with_pagination(
        products,
        sysmsg,
        has_more=product_search_show_view_more(products, os_has_more),
        next_page=2,
        browse_more_url=browse_url,
    )
    try:
        from services.chat_flow_telemetry import mark_routing_complete, record_route

        record_route(intent="product", source="sku_structural_fast")
        mark_routing_complete()
    except ImportError:
        pass
    reset_context_fn(ctx)
    return response_text


def try_category_browse_catalog_reply(
    original_msg: str,
    msg_en: str,
    *,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    Named category browse (electronics ke products) → OpenSearch + REST (zero LLM).
    """
    browse_text = (original_msg or msg_en or "").strip()
    if not browse_text:
        return None
    try:
        from services.welfog_api import (
            _message_has_product_search_filters,
            build_welfog_product_browse_url,
            category_name_for_id,
            message_requests_category_browse,
            query_should_use_category_id_only,
            resolve_category_browse_for_catalog,
        )
        from services.opensearch_products import (
            build_product_rail_with_pagination,
            catalog_search_live,
            product_search_show_view_more,
            sanitize_product_search_spec,
        )
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer

        if _message_has_product_search_filters(browse_text):
            return None
        if not message_requests_category_browse(browse_text):
            return None

        route = resolve_category_browse_for_catalog(
            browse_text, ctx=ctx, allow_inner_lookup=True
        )
        if not route:
            return None
        cid, sq = route
        if (sq or "").strip() and not query_should_use_category_id_only(
            cid, browse_text, ctx
        ):
            return None
    except ImportError:
        return None

    cat_label = category_name_for_id(str(cid), ctx=ctx) or f"category {cid}"
    log_reasoning(
        f"Structural category browse fast path ({cat_label!r}, id={cid}) → catalog (zero LLM)."
    )
    product_route: dict = {
        "intent": "product",
        "data_channel": "catalog",
        "run_catalog_search": True,
        "_product_catalog_locked": True,
        "_product_entities": {"category": cat_label, "category_id": cid},
        "search_query": "",
        "user_meaning": f"Products in {cat_label}",
        "needs_order_id": False,
        "numeric_context": "none",
        "meta_kind": "none",
        "is_welfog_related": True,
        "route_handler": "category_browse_structural_fast",
    }
    ctx.setdefault("data", {})["ai_route"] = product_route
    ctx["data"]["selected_category_id"] = str(cid)

    os_spec = sanitize_product_search_spec(
        {"category_id": int(cid), "title_query": "", "_category_only_browse": True, "_ai_single_pass": True}
    )
    products, os_spec, _os_total, os_has_more = catalog_search_live(
        os_spec,
        original_msg=original_msg,
        msg_en=msg_en,
        ctx=ctx,
    )
    display_query = cat_label
    if not products:
        body = localized_sysmsg_for_customer(
            "product_not_found",
            original_msg,
            lang,
            query=display_query,
            fallback_en=sysmsg("product_not_found", query=display_query) or "",
        )
        reset_context_fn(ctx)
        return body

    response_text = sysmsg("products_title_query", query=display_query) or ""
    browse_url = build_welfog_product_browse_url(os_spec, ctx=ctx)
    response_text += build_product_rail_with_pagination(
        products,
        sysmsg,
        has_more=product_search_show_view_more(products, os_has_more),
        next_page=2,
        browse_more_url=browse_url,
    )
    try:
        from services.chat_flow_telemetry import mark_routing_complete, record_route

        record_route(intent="product", source="category_browse_structural_fast")
        mark_routing_complete()
    except ImportError:
        pass
    reset_context_fn(ctx)
    return response_text


def try_llm_down_product_catalog_recovery(
    original_msg: str,
    msg_en: str,
    *,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    Brain routing LLM failed — one compact product classify (AI) then OpenSearch.
    Uses translated msg_en so Hinglish/Tamil recover like English.
    """
    try:
        from services.chat_flow_telemetry import reset_llm_budget_for_recovery

        reset_llm_budget_for_recovery()
    except ImportError:
        pass
    try:
        from services.product_catalog_resolver import (
            KIND_PRODUCT_SEARCH,
            ai_classify_product_search_turn,
        )

        classified = ai_classify_product_search_turn(
            original_msg,
            msg_en,
            conv_for_llm,
            ai_route=None,
            reply_lang=lang,
        )
        if not classified or classified.get("turn_kind") != KIND_PRODUCT_SEARCH:
            return None
        sq = (classified.get("search_query") or "").strip()
        entities = dict(classified.get("entities") or {})
        if not sq:
            sq = (entities.get("product_name") or "").strip()
        if not sq or len(sq) < 2:
            return None
        product_route = {
            "intent": "product",
            "data_channel": "catalog",
            "run_catalog_search": True,
            "_product_catalog_locked": True,
            "_universal_brain_route": True,
            "_ai_single_pass": True,
            "_needs_product_nlu_llm": False,
            "_product_nlu_from_ai": True,
            "_product_entities": entities,
            "search_query": sq,
            "user_meaning": classified.get("user_meaning") or f"Show {sq}",
        }
        log_reasoning(
            f"LLM-down recovery — compact product classify sq={sq!r} → OpenSearch."
        )
        return _run_product_catalog_flow(
            product_route,
            sq,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
        )
    except ImportError:
        return None


def try_structural_product_catalog_reply(
    original_msg: str,
    msg_en: str,
    *,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    Clear product browse (iphone dikha, samsung cover) — OpenSearch without brain LLM.
    Saves ~30–45s on obvious catalog turns.
    """
    try:
        from services.semantic_intent import zero_llm_intent_guess_allowed

        if not zero_llm_intent_guess_allowed():
            return None
    except ImportError:
        pass
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return None
    _skip_delivery_block = False
    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            kb_classified_as_product_catalog,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(original_msg, msg_en, conv_for_llm, "")
        if peeked is not _KB_CACHE_UNSET and kb_classified_as_product_catalog(peeked):
            _skip_delivery_block = True
    except ImportError:
        pass
    if not _skip_delivery_block:
        try:
            from services.location_delivery_resolver import (
                turn_requests_delivery_serviceability,
            )

            if turn_requests_delivery_serviceability(
                original_msg, msg_en, conv_for_llm, allow_llm=False
            ):
                return None
        except ImportError:
            pass
    try:
        from utils.helpers import (
            _message_looks_like_shopping_query,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )
        from services.product_query_understanding import (
            resolve_catalog_search_terms_for_message,
        )

        if not _message_looks_like_shopping_query(comb):
            return None
        if message_is_wishlist_like_request(comb) or message_is_past_purchase_list_request(comb):
            return None

        product_route = {
            "intent": "product",
            "data_channel": "catalog",
            "run_catalog_search": True,
            "_product_catalog_locked": True,
            "_universal_brain_route": True,
            "_ai_single_pass": True,
            # resolve_catalog_search_terms_for_message = Product Entity Extraction.
            "_needs_product_nlu_llm": False,
            "_product_nlu_from_ai": True,
        }
        sq = resolve_catalog_search_terms_for_message(
            original_msg, msg_en, ai_route=product_route
        )
        if not sq or len(sq) < 2:
            return None
    except ImportError:
        return None

    product_route["search_query"] = sq
    product_route["_product_entities"] = {
        **(product_route.get("_product_entities") or {}),
        "product_name": sq,
    }
    product_route["user_meaning"] = f"Show {sq}"
    log_reasoning(f"Structural product fast path sq={sq!r} → OpenSearch (entity extract).")
    return _run_product_catalog_flow(
        product_route,
        sq,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )


def _brain_is_order_live_turn(
    brain_route: dict,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conv_for_llm: str = "",
) -> bool:
    if not isinstance(brain_route, dict):
        return False
    alk = (brain_route.get("account_list_kind") or "").strip().lower()
    if alk not in ("", "none"):
        return False
    if _brain_route_is_personal_order_live(brain_route):
        return True
    try:
        from services.ai_route_semantics import (
            brain_route_to_live_goal,
            ensure_brain_order_route_locked,
            resolve_order_live_goal_for_turn,
        )

        locked = ensure_brain_order_route_locked(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm or "",
        )
        turn_goal = resolve_order_live_goal_for_turn(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm or "",
        )
        if turn_goal in (
            "refund_status",
            "order_invoice",
            "order_details",
            "track",
            "payment",
        ):
            return True
        enriched_goal = brain_route_to_live_goal(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm or "",
        )
        if enriched_goal in (
            "refund_status",
            "order_invoice",
            "order_details",
            "track",
            "payment",
        ):
            return True
    except ImportError:
        pass
    try:
        from services.order_live_intent_fast_path import resolve_live_goal_lightweight

        goal = resolve_live_goal_lightweight(
            original_msg, msg_en, conv_for_llm, reply_lang="en"
        )
        if goal in (
            "refund_status",
            "order_invoice",
            "order_details",
            "track",
            "payment",
        ):
            return True
    except ImportError:
        pass
    intent = (brain_route.get("intent") or "").strip().lower()
    if intent in ("wishlist", "product", "pincode_check", "deals", "categories"):
        return False
    if intent == "order_history":
        if brain_route.get("needs_order_id"):
            rh = (brain_route.get("route_handler") or "").strip().lower()
            olk = (brain_route.get("order_lookup_kind") or "").strip().lower()
            if rh in (
                "order_tracking_api",
                "order_details_api",
                "refund_status_api",
            ) or olk in ("track", "invoice", "details", "refund_status"):
                return True
        return False
    return False


def _brain_order_goal_from_route(
    brain_route: dict,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conv_for_llm: str = "",
) -> str:
    """Map brain JSON → live goal. ai_brain_route is source of truth (any language)."""
    try:
        from services.ai_route_semantics import resolve_order_live_goal_for_turn

        brain_goal = resolve_order_live_goal_for_turn(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm or "",
        )
        if brain_goal:
            return brain_goal
    except ImportError:
        pass

    try:
        from services.order_live_intent_fast_path import (
            _keyword_intent_fallback_enabled,
            _resolve_message_live_goal,
        )

        if _keyword_intent_fallback_enabled():
            message_goal = _resolve_message_live_goal(
                original_msg, msg_en, conv_for_llm or "", reply_lang="en"
            )
            if message_goal:
                return message_goal
    except ImportError:
        pass

    return ""


def _brain_order_tool_for_goal(goal: str) -> str:
    try:
        from services.ai_route_semantics import LIVE_API_FROM_GOAL

        return LIVE_API_FROM_GOAL.get((goal or "").strip().lower(), "")
    except ImportError:
        tools = {
            "order_invoice": "order_details_api",
            "order_details": "order_details_api",
            "track": "order_tracking_api",
            "payment": "order_details_api",
            "refund_status": "refund_status_api",
        }
        return tools.get((goal or "").strip().lower(), "")


def _try_brain_order_live_fallback_dispatch(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reply_for_live_order_id_lookup: Callable,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """
    One brain LLM already ran — ask Order ID or call live API (no specialist LLM stack).
    """
    if not isinstance(brain_route, dict) or brain_route.get("llm_unavailable"):
        return None
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog

        if brain_route_indicates_product_catalog(brain_route):
            return None
    except ImportError:
        pass
    if not _brain_is_order_live_turn(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
    ):
        channel = (brain_route.get("data_channel") or "").strip().lower()
        if not (
            brain_route.get("needs_order_id")
            and channel in ("live_api", "order", "")
        ):
            return None
    try:
        from services.ai_route_semantics import (
            brain_route_to_live_goal,
            ensure_brain_order_route_locked,
        )
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

        locked = ensure_brain_order_route_locked(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm or "",
        )
        goal = _brain_order_goal_from_route(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
        )
        if not goal:
            goal = brain_route_to_live_goal(
                locked,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm or "",
            )
        if not goal:
            return None
        ctx.setdefault("data", {})["ai_route"] = locked
        live_reply = try_order_live_intent_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
            preset_goal=goal,
        )
        if live_reply:
            log_reasoning(
                f"Brain order live fallback: goal={goal} (ask ID or live API, zero extra LLM)."
            )
            try:
                from services.chat_flow_telemetry import mark_routing_complete

                mark_routing_complete()
            except ImportError:
                pass
        return live_reply
    except ImportError:
        return None


def _try_brain_order_live_direct_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reply_for_live_order_id_lookup: Callable[..., str],
) -> Optional[str]:
    """Brain locked order/refund/invoice/track → existing live API (one brain LLM)."""
    if not _brain_is_order_live_turn(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
    ):
        return None

    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(brain_route):
            return None
    except ImportError:
        pass

    try:
        from utils.helpers import (
            clear_order_session_for_new_lookup,
            message_user_switches_order_scope,
        )

        if message_user_switches_order_scope(f"{original_msg} {msg_en}".strip()):
            clear_order_session_for_new_lookup(ctx)
    except ImportError:
        pass

    goal = _brain_order_goal_from_route(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
    )
    if not goal:
        try:
            from services.ai_route_semantics import (
                brain_route_to_live_goal,
                ensure_brain_order_route_locked,
            )

            locked = ensure_brain_order_route_locked(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm or "",
            )
            goal = brain_route_to_live_goal(
                locked,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm or "",
            )
        except ImportError:
            pass
    if not goal:
        return None
    try:
        from services.ai_route_semantics import correct_order_details_vs_tracking_from_ai_meaning

        route_for_api = correct_order_details_vs_tracking_from_ai_meaning(dict(brain_route))
    except ImportError:
        route_for_api = dict(brain_route)
    route_for_api.setdefault("intent", "order" if goal != "refund_status" else "refund")
    route_for_api.setdefault("data_channel", "live_api")
    route_for_api["route_handler"] = _brain_order_tool_for_goal(goal)
    route_for_api["order_lookup_kind"] = (
        "invoice"
        if goal == "order_invoice"
        else "details"
        if goal in ("order_details", "payment")
        else "refund_status"
        if goal == "refund_status"
        else "track"
    )
    route_for_api["needs_order_id"] = True
    route_for_api["numeric_context"] = "order_id"

    oid = (brain_route.get("extracted_order_id") or "").strip()
    if not oid:
        try:
            from utils.helpers import resolve_order_id_for_tracking

            oid = (
                resolve_order_id_for_tracking(
                    f"{original_msg} {msg_en}".strip(),
                    conv_for_llm,
                    bot_awaiting_order_id=ctx.get("awaiting") == "order_id",
                    ai_extracted=brain_route.get("extracted_order_id"),
                )
                or ""
            ).strip()
        except ImportError:
            oid = ""
    else:
        try:
            from utils.helpers import coerce_valid_order_id

            oid = coerce_valid_order_id(
                oid, context=f"{original_msg} {msg_en} {conv_for_llm or ''}"
            )
        except ImportError:
            oid = oid.strip()

    tool = _brain_order_tool_for_goal(goal)
    if not tool:
        return None

    if not oid:
        from services.order_history_flow import _localized_sysmsg

        intent_label = (
            "refund"
            if goal == "refund_status"
            else "invoice"
            if goal == "order_invoice"
            else "order"
        )
        body = _localized_sysmsg(
            "ask_order_id_for_intent",
            original_msg,
            reply_lang=lang,
            intent=intent_label,
        )
        if not body:
            return None
        try:
            from services.order_id_handoff_fast_path import lock_order_id_ask_session

            lock_order_id_ask_session(
                ctx, goal, intent_label=intent_label, ai_route=route_for_api
            )
        except ImportError:
            ctx["order_id"] = None
            ctx["awaiting"] = "order_id"
            ctx["last"] = intent_label
            ctx.setdefault("data", {})["pending_action"] = goal
            ctx["data"]["topic_mode"] = f"order_{goal}"
            ctx["data"]["ai_route"] = route_for_api
        try:
            from services.chat_flow_telemetry import log_order_dispatch

            log_order_dispatch(
                detected_intent=goal,
                detected_language=lang,
                message=f"{original_msg} {msg_en}".strip(),
                previous_context=conv_for_llm,
                pending_action=goal,
                order_id_found="",
                selected_tool=tool,
                api_called=False,
            )
        except ImportError:
            pass
        log_reasoning(
            f"Brain direct order: ask Order ID for goal={goal} (pending_action stored)."
        )
        return body

    route_for_api["extracted_order_id"] = oid
    route_for_api["needs_order_id"] = False
    api_called = True
    reply_html = ""

    if goal in ("order_invoice", "order_details", "payment"):
        try:
            from services.order_id_handoff_fast_path import _fetch_details_handoff_reply
            import time as _time

            _t_api = _time.perf_counter()
            details_goal = "order_details" if goal == "payment" else goal
            details_focus = (
                "payment"
                if goal == "payment"
                else (route_for_api.get("field_focus") or "").strip()
            )
            reply_html = _fetch_details_handoff_reply(
                details_goal,
                oid,
                user_id,
                original_msg,
                lang,
                ai_focus=details_focus,
            )
            api_time_ms = (_time.perf_counter() - _t_api) * 1000.0
        except ImportError:
            return None
    else:
        live_intent = (
            "refund"
            if goal == "refund_status"
            else "payment"
            if goal == "payment"
            else "order"
        )
        import time as _time

        _t_api = _time.perf_counter()
        reply_html = reply_for_live_order_id_lookup(
            live_intent, oid, user_id, original_msg, lang
        )
        api_time_ms = (_time.perf_counter() - _t_api) * 1000.0

    if not reply_html:
        return None

    try:
        from services.answer_router import AnswerRouteDecision
        from services.chat_flow_telemetry import log_order_dispatch, store_turn_analysis

        log_order_dispatch(
            detected_intent=goal,
            detected_language=lang,
            message=f"{original_msg} {msg_en}".strip(),
            previous_context=conv_for_llm,
            pending_action=goal,
            order_id_found=oid,
            selected_tool=tool,
            api_called=api_called,
            api_time_ms=api_time_ms,
        )
        store_turn_analysis(
            route_for_api,
            AnswerRouteDecision(
                source="api",
                intent=route_for_api.get("intent") or "order",
                handler=tool,
                is_welfog_related=True,
                reason=f"Brain direct order dispatch — {goal}",
            ),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
    except ImportError:
        pass

    ctx["order_id"] = oid
    ctx["awaiting"] = None
    ctx["last"] = (
        "refund"
        if goal == "refund_status"
        else "payment"
        if goal == "payment"
        else "order"
    )
    ctx.setdefault("data", {})["ai_route"] = route_for_api
    ctx["data"].pop("pending_action", None)
    log_reasoning(f"Brain direct dispatch: order live API goal={goal} id={oid}.")
    return reply_html


def _brain_is_pincode_delivery_turn(brain_route: dict) -> bool:
    if not isinstance(brain_route, dict):
        return False
    if brain_route.get("_pincode_delivery_locked"):
        return True
    intent = (brain_route.get("intent") or "").strip().lower()
    handler = (brain_route.get("route_handler") or "").strip().lower()
    channel = (brain_route.get("data_channel") or "").strip().lower()
    if intent == "pincode_check" or handler == "pincode_delivery_api":
        return True
    if channel == "live_api" and intent == "pincode_check":
        return True
    try:
        from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

        if ai_meaning_describes_delivery_serviceability(brain_route):
            return True
    except ImportError:
        pass
    try:
        from services.chat_flow_telemetry import is_authoritative_pincode_route_locked

        if is_authoritative_pincode_route_locked():
            return True
    except ImportError:
        pass
    return False


def _extract_pincode_from_brain_route(brain_route: dict, original_msg: str, msg_en: str) -> str:
    """PIN only from THIS message (or Brain when it matches typed digits). Never history."""
    import re

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    msg_pin = ""
    m = re.search(r"\b([1-9]\d{5})\b", comb)
    if m:
        msg_pin = m.group(1)

    # No 6-digit PIN in this turn → city/ask-PIN path (geocode or prompt).
    if not msg_pin:
        return ""

    brain_pin = re.sub(r"\D", "", str(brain_route.get("extracted_pincode") or ""))
    if len(brain_pin) == 6 and brain_pin[0] != "0" and brain_pin == msg_pin:
        return brain_pin
    try:
        from utils.helpers import resolve_pincode_for_check

        pin = resolve_pincode_for_check(
            original_msg,
            "",
            msg_en=msg_en,
            ai_extracted=msg_pin,
            ai_route=brain_route,
        )
        if pin:
            return pin
    except ImportError:
        pass
    return msg_pin


def _extract_location_from_brain_route(brain_route: dict) -> str:
    try:
        from services.location_delivery_resolver import _city_label_from_ai_route

        return _city_label_from_ai_route(brain_route)
    except ImportError:
        for key in ("extracted_location", "extracted_city", "search_query"):
            val = str(brain_route.get(key) or "").strip()
            if val and len(val) >= 2:
                return val
        return ""


def _log_pincode_dispatch(
    *,
    detected_intent: str,
    extracted_pincode: str = "",
    extracted_location: str = "",
    selected_tool: str,
    api_called: bool,
) -> None:
    try:
        from services.chat_flow_telemetry import (
            llm_calls_count,
            log_intent_routing,
            mark_routing_complete,
            record_route,
            record_route_step,
            response_time_sec,
        )

        record_route_step("brain_direct_pincode")
        record_route(intent=detected_intent, source="brain_direct_pincode")
        mark_routing_complete()
        entities: dict[str, str] = {}
        if extracted_pincode:
            entities["pincode"] = extracted_pincode
        if extracted_location:
            entities["location"] = extracted_location
        extra = (
            f"extracted_pincode={extracted_pincode or '-'} "
            f"extracted_location={extracted_location or '-'} "
            f"api_called={api_called} llm_call_count={llm_calls_count()}"
        )
        log_intent_routing(
            detected_intent=detected_intent,
            selected_existing_tool=selected_tool,
            entities=entities,
            source_used="brain_direct_pincode",
            response_time=response_time_sec(),
            reason="Brain locked pincode — direct API (no routing stack)",
            extra=extra,
        )
        log_reasoning(
            f"[pincode-flow] detected_intent={detected_intent} "
            f"extracted_pincode={extracted_pincode or '-'} "
            f"extracted_location={extracted_location or '-'} "
            f"selected_tool={selected_tool} api_called={api_called} "
            f"llm_call_count={llm_calls_count()} "
            f"response_time={response_time_sec():.2f}s"
        )
    except ImportError:
        pass


def _try_brain_account_list_live_api_reply(
    brain_route: dict,
    *,
    user_id: str,
    format_purchase_history_reply: Callable[..., str],
    format_wishlist_reply: Callable[..., str],
) -> Optional[str]:
    """Dispatch account-list live API from brain JSON only — before pincode/product stacks."""
    if not isinstance(brain_route, dict):
        return None
    try:
        from services.account_list_semantics import reconcile_account_list_from_brain_meaning

        brain_route = reconcile_account_list_from_brain_meaning(dict(brain_route))
    except ImportError:
        pass
    try:
        from services.account_list_semantics import account_list_route_is_locked
        from services.ai_route_semantics import brain_route_indicates_account_list_live
    except ImportError:
        return None
    if not (
        account_list_route_is_locked(brain_route)
        or brain_route_indicates_account_list_live(brain_route)
    ):
        return None
    if brain_route.get("needs_order_id"):
        return None
    alk = (brain_route.get("account_list_kind") or "").strip().lower()
    intent = (brain_route.get("intent") or "").strip().lower()
    if alk == "purchase_history_in_chat" or intent == "order_history":
        log_reasoning("Brain dispatch — order history API from AI JSON.")
        return format_purchase_history_reply(user_id, page=1, append_only=False)
    if alk == "wishlist_in_chat" or intent == "wishlist":
        log_reasoning("Brain dispatch — wishlist API from AI JSON.")
        return format_wishlist_reply(user_id, page=1, append_only=False)
    return None


def _try_brain_pincode_direct_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """Brain intent=pincode_check → existing pincode API (one brain LLM + live check)."""
    if not _brain_is_pincode_delivery_turn(brain_route):
        return None

    import time

    t_all = time.perf_counter()
    try:
        from services.chat_flow_telemetry import (
            lock_authoritative_pincode_route,
            record_phase,
        )

        lock_authoritative_pincode_route(source="brain_dispatch", ai_route=brain_route)
    except ImportError:
        record_phase = None  # type: ignore

    try:
        from utils.helpers import clear_order_session_for_new_lookup

        if isinstance(ctx, dict) and (
            ctx.get("awaiting") or (ctx.get("data") or {}).get("pending_action")
        ):
            log_reasoning(
                "Pincode dispatch — sanitize stale order session before live API."
            )
            clear_order_session_for_new_lookup(ctx)
            ctx["awaiting"] = None
            ctx["last"] = None
    except ImportError:
        pass

    try:
        from services.support_scope import (
            build_other_company_support_decline,
            message_mentions_other_company_support,
        )

        if message_mentions_other_company_support(
            original_msg,
            msg_en,
            conv_for_llm,
            ai_route=brain_route,
        ):
            log_reasoning(
                "Brain pincode dispatch: other-company scope — skip Welfog pincode API."
            )
            _log_pincode_dispatch(
                detected_intent="out_of_domain",
                selected_tool="other_company_decline",
                api_called=False,
            )
            return build_other_company_support_decline(original_msg, reply_lang=lang)
    except ImportError:
        pass

    pin = _extract_pincode_from_brain_route(brain_route, original_msg, msg_en)
    location = _extract_location_from_brain_route(brain_route)
    route_for_api = dict(brain_route)
    route_for_api.setdefault("intent", "pincode_check")
    route_for_api.setdefault("data_channel", "live_api")
    route_for_api.setdefault("route_handler", "pincode_delivery_api")
    route_for_api["_pincode_delivery_locked"] = True
    if pin:
        route_for_api["extracted_pincode"] = pin
    if location and not (route_for_api.get("search_query") or "").strip():
        route_for_api["search_query"] = location

    # Brain locked delivery but PIN/city missing → one semantic location fill (force_llm).
    # No static city-list scan; geocode uses Brain/meaning city_name only.
    if not pin:
        try:
            from services.location_delivery_resolver import (
                ai_understand_delivery_turn,
                geocode_city_to_pincode,
            )

            t_loc = time.perf_counter()
            city = (location or "").strip()
            if not city:
                meaning = (brain_route.get("user_meaning") or "").strip()
                understood = ai_understand_delivery_turn(
                    original_msg,
                    msg_en or meaning,
                    conv_for_llm,
                    ai_route=route_for_api,
                    reply_lang=lang,
                    force_llm=True,
                )
                if understood:
                    pin_u = re.sub(r"\D", "", str(understood.pincode or ""))
                    if len(pin_u) == 6 and pin_u[0] != "0":
                        pin = pin_u
                    city = (understood.city_name or "").strip() or city
            if city and not pin:
                geo = geocode_city_to_pincode(city)
                resolved_pin = str(geo.get("pincode") or "")
                if resolved_pin:
                    pin = resolved_pin
                    location = str(geo.get("city_label") or city)
                    route_for_api["extracted_pincode"] = pin
                    route_for_api["search_query"] = location
                    log_reasoning(
                        f"Brain pincode: city {city!r} → PIN {pin} (geocode, skip ask-PIN)."
                    )
            if record_phase:
                record_phase(
                    "pincode_location_resolve",
                    (time.perf_counter() - t_loc) * 1000.0,
                )
        except ImportError:
            pass

    api_called = False
    reply_html = ""

    if pin:
        try:
            from services.pincode_delivery_flow import (
                format_pincode_check_reply,
                validate_pincode_before_api,
                _pin_localized,
            )
            from services.welfog_api import check_pincode_delivery

            ok, err_key, fmt = validate_pincode_before_api(pin, original_msg)
            if not ok:
                reply_html = _pin_localized(err_key, original_msg, lang, **fmt)
            else:
                log_reasoning(
                    f"Brain direct pincode: live API for PIN {pin} (skip routing stack)."
                )
                t_api = time.perf_counter()
                api_res = check_pincode_delivery(pin)
                if record_phase:
                    record_phase(
                        "pincode_live_api",
                        (time.perf_counter() - t_api) * 1000.0,
                    )
                api_called = True
                reply_html = format_pincode_check_reply(
                    pin,
                    api_res,
                    original_msg,
                    lang,
                    city_label=location or "",
                )
        except ImportError:
            reply_html = ""

    if not reply_html:
        try:
            from services.pincode_delivery_flow import run_delivery_location_check

            result = run_delivery_location_check(
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ai_route=route_for_api,
                allow_llm=False,
            )
            if result.handled and result.reply_html:
                reply_html = result.reply_html
                api_called = bool(pin) or (
                    "available" in (reply_html or "").lower()
                    or "not available" in (reply_html or "").lower()
                    or "service" in (reply_html or "").lower()
                )
        except ImportError:
            return None

    if not reply_html:
        try:
            from services.pincode_delivery_flow import build_pincode_missing_or_invalid_reply

            reply_html = build_pincode_missing_or_invalid_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=lang,
                allow_ai=False,
            )
        except ImportError:
            reply_html = ""

    if not reply_html:
        return None

    _log_pincode_dispatch(
        detected_intent="pincode_check",
        extracted_pincode=pin,
        extracted_location=location,
        selected_tool="pincode_delivery_api",
        api_called=api_called,
    )
    if record_phase:
        record_phase("pincode_dispatch_total", (time.perf_counter() - t_all) * 1000.0)
    try:
        from services.answer_router import AnswerRouteDecision
        from services.chat_flow_telemetry import store_turn_analysis

        store_turn_analysis(
            route_for_api,
            AnswerRouteDecision(
                source="api",
                intent="pincode_check",
                handler="pincode_delivery_api",
                is_welfog_related=True,
                reason="Brain direct pincode dispatch.",
            ),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
    except ImportError:
        pass

    if pin:
        reset_context_fn(ctx)
        try:
            from utils.helpers import mark_pincode_delivery_completed

            mark_pincode_delivery_completed(ctx, pin=pin, ai_route=route_for_api)
        except ImportError:
            ctx["awaiting"] = None
            ctx["last"] = None
            ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
            ctx.setdefault("data", {})["last_pincode"] = pin
            ctx.setdefault("data", {})["ai_route"] = route_for_api
    else:
        reset_context_fn(ctx)
        try:
            from utils.helpers import set_pincode_await_context

            set_pincode_await_context(ctx, route_for_api)
        except ImportError:
            ctx["awaiting"] = "pincode"
            ctx["last"] = "pincode"
            ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
            ctx.setdefault("data", {})["ai_route"] = route_for_api
    log_reasoning("Brain direct dispatch: pincode delivery API.")
    return reply_html


def try_structural_order_live_reply(
    original_msg: str,
    msg_en: str,
    *,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """
    Clear order live turns (2606010 invoice, track, refund) — live API without brain LLM.
    """
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return None
    try:
        from utils.helpers import (
            _message_looks_like_shopping_query,
            turn_is_obvious_product_shopping_turn,
        )

        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conv_for_llm
        ) or _message_looks_like_shopping_query(comb):
            return None
    except ImportError:
        pass
    try:
        from utils.helpers import turn_is_catalog_product_lookup

        if turn_is_catalog_product_lookup(original_msg, msg_en):
            return None
    except ImportError:
        pass

    try:
        from services.order_live_intent_fast_path import (
            resolve_structural_locked_order_goal,
            try_order_live_intent_fast_reply,
        )

        goal = resolve_structural_locked_order_goal(
            original_msg, msg_en, conv_for_llm, ctx
        )
        if not goal:
            return None
        log_reasoning(
            f"Structural order fast path: locked goal={goal} (skip brain LLM)."
        )
        return try_order_live_intent_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
            preset_goal=goal,
        )
    except ImportError:
        return None


def _try_brain_catalog_menu_direct_reply(
    brain_route: dict,
    *,
    original_msg: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """Brain locked deals / categories — one API call, no account-list or order LLM stack."""
    intent = (brain_route.get("intent") or "").strip().lower()
    if intent not in ("deals", "categories", "category_feed"):
        return None
    try:
        from services.catalog_menu_replies import (
            build_categories_list_reply_html,
            build_today_deals_reply_html,
        )
        from services.chat_flow_telemetry import mark_routing_complete

        reset_context_fn(ctx)
        ctx.setdefault("data", {})["topic_mode"] = intent
        if intent == "deals":
            log_reasoning("Brain direct dispatch: today's deals API (brain-locked, zero extra LLM).")
            body = build_today_deals_reply_html(original_msg, reply_lang=lang)
            ctx["data"]["pending_offer"] = "deals"
        else:
            log_reasoning("Brain direct dispatch: categories API (brain-locked, zero extra LLM).")
            body = build_categories_list_reply_html(ctx, original_msg, reply_lang=lang)
            ctx["data"]["pending_offer"] = "categories"
        mark_routing_complete()
        return body or None
    except ImportError:
        return None


def _try_brain_account_list_direct_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    format_purchase_history_reply: Callable,
    format_wishlist_reply: Callable,
    reset_context_fn: Callable[[dict], None],
) -> Optional[str]:
    """Account-list API or how-to KB — reuses ai_brain_route account_list_kind (no keyword gate)."""
    try:
        from services.account_list_fast_path import try_account_list_fast_reply
        from services.kb_service import sysmsg
        from services.translation_service import localized_sysmsg_for_customer

        account_reply = try_account_list_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            localized_sysmsg=lambda k, m, reply_lang="en": localized_sysmsg_for_customer(
                k, m, reply_lang=reply_lang
            ),
            sysmsg=sysmsg,
            reset_context_fn=reset_context_fn,
        )
        if account_reply:
            log_reasoning("Brain direct dispatch: account-list (AI-locked route).")
            try:
                from services.chat_flow_telemetry import mark_routing_complete

                mark_routing_complete()
            except ImportError:
                pass
            return account_reply
    except ImportError:
        pass
    return None


def try_brain_direct_dispatch(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    format_purchase_history_reply: Callable[..., str],
    format_wishlist_reply: Callable[..., str],
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """
    Execute handler immediately from ai_brain_route JSON.
    Returns reply HTML or None (caller continues resolve_answer_route).
    """
    if not isinstance(brain_route, dict) or brain_route.get("llm_unavailable"):
        return None

    # Delivery semantic plan first — before KB informational / early KB steal.
    try:
        from services.ai_route_semantics import reconcile_pincode_delivery_from_brain_meaning
        from services.chat_flow_telemetry import (
            is_authoritative_pincode_route_locked,
            should_skip_post_pincode_route_steal,
        )

        brain_route = reconcile_pincode_delivery_from_brain_meaning(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        if (
            brain_route.get("_pincode_delivery_locked")
            or (brain_route.get("intent") or "").strip().lower() == "pincode_check"
            or is_authoritative_pincode_route_locked()
        ):
            pin_first = _try_brain_pincode_direct_reply(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
            )
            if pin_first:
                return pin_first
        if should_skip_post_pincode_route_steal("brain_direct_dispatch"):
            return None
    except ImportError:
        pass

    try:
        from services.knowledge_query_pipeline import try_kb_informational_locked_reply
        from services.turn_intent_coordinator import kb_turn_blocks_product_catalog

        kb_lock = try_kb_informational_locked_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            reply_lang=lang,
            brain_route=brain_route,
        )
        if kb_lock and kb_lock.strip():
            log_reasoning("Brain dispatch — KB informational lock (no catalog).")
            return kb_lock
        classified = None
        try:
            from services.turn_intent_coordinator import peek_or_classify_kb_turn

            classified = peek_or_classify_kb_turn(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                preflight=True,
            )
        except ImportError:
            pass
        if kb_turn_blocks_product_catalog(classified):
            brain_route = dict(brain_route)
            brain_route["data_channel"] = "kb"
            brain_route["intent"] = "general"
            brain_route["run_catalog_search"] = False
    except ImportError:
        pass

    channel0 = (brain_route.get("data_channel") or "").strip().lower()
    intent0 = (brain_route.get("intent") or "").strip().lower()
    try:
        from services.ai_route_semantics import (
            brain_route_indicates_informational_kb,
            reconcile_welfog_kb_from_brain_meaning,
        )

        if brain_route_indicates_informational_kb(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        ):
            kb_route = reconcile_welfog_kb_from_brain_meaning(
                dict(brain_route),
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
            kb_sem = _try_brain_kb_locked_reply(
                kb_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            )
            if kb_sem:
                log_reasoning("Brain dispatch — semantic KB from brain JSON.")
                return kb_sem
    except ImportError:
        pass
    if channel0 == "kb" and not brain_route.get("needs_order_id"):
        if intent0 in ("refund", "payment", "general", "seller", ""):
            kb_early = _try_brain_kb_locked_reply(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            )
            if kb_early:
                return kb_early

    try:
        from services.ai_route_semantics import (
            brain_route_indicates_informational_kb,
            brain_route_indicates_product_catalog,
            reconcile_product_catalog_from_brain_meaning,
        )

        try:
            from services.account_list_semantics import (
                account_list_route_is_locked,
                reconcile_account_list_from_brain_meaning,
            )

            brain_route = reconcile_account_list_from_brain_meaning(brain_route)
            _alk_locked = account_list_route_is_locked(brain_route)
        except ImportError:
            _alk_locked = False
        if not (
            brain_route.get("_pincode_delivery_locked")
            or (brain_route.get("intent") or "").strip().lower() == "pincode_check"
        ) and not _alk_locked:
            brain_route = reconcile_product_catalog_from_brain_meaning(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
            )
        if brain_route_indicates_product_catalog(brain_route):
            brain_route["conversation_scope"] = "welfog_support"
            brain_route["scope_reply"] = ""
            brain_route["meta_kind"] = "none"
    except ImportError:
        pass

    brain_intent = (brain_route.get("intent") or "").strip().lower()
    meta_kind = (brain_route.get("meta_kind") or "").strip().lower()
    brain_olk = (brain_route.get("order_lookup_kind") or "").strip().lower()
    brain_rh = (brain_route.get("route_handler") or "").strip().lower()
    scope = (brain_route.get("conversation_scope") or "").strip().lower()
    if brain_intent == "out_of_domain" or scope == "out_of_domain":
        scope = "out_of_domain"
    elif not scope and meta_kind in ("conversational", "assistant_intro"):
        scope = "general_chitchat"
    elif not scope and brain_intent in ("general",):
        scope = "general_chitchat"
    scope_reply = (brain_route.get("scope_reply") or "").strip()
    if _scope_reply_is_placeholder(scope_reply):
        scope_reply = ""

    menu_early = _try_brain_catalog_menu_direct_reply(
        brain_route,
        original_msg=original_msg,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )
    if menu_early:
        return menu_early

    # Pincode / delivery serviceability before catalog — city/PIN checks are live API.
    pincode_early = _try_brain_pincode_direct_reply(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )
    if pincode_early:
        return pincode_early

    # Product catalog when brain locked shopping — beats order/KB detours.
    if _brain_is_product_catalog_turn(brain_route, original_msg, msg_en):
        direct_cat = _try_brain_category_browse_direct_reply(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
        )
        if direct_cat:
            log_reasoning("Brain category direct dispatch — category-only catalog (skip heavy pipeline).")
            return direct_cat
        product_route, sq = _prepare_brain_product_route(
            brain_route, original_msg, msg_en
        )
        log_reasoning(
            f"Brain direct dispatch: product catalog sq={sq!r} → OpenSearch."
        )
        return _run_product_catalog_flow(
            product_route,
            sq,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
        )

    # Account-list API/KB first when brain classified it — before single-order LLM stack.
    _account_list_early = brain_intent in ("order_history", "wishlist")
    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            _kind_from_meaning_blob,
            _meaning_blob,
            KIND_PURCHASE_HOWTO,
            KIND_PURCHASE_IN_CHAT,
            KIND_WISHLIST_HOWTO,
            KIND_WISHLIST_IN_CHAT,
        )

        if account_list_route_is_locked(brain_route):
            _account_list_early = True
        else:
            mk = _kind_from_meaning_blob(_meaning_blob(brain_route))
            if mk in (
                KIND_WISHLIST_IN_CHAT,
                KIND_WISHLIST_HOWTO,
                KIND_PURCHASE_IN_CHAT,
                KIND_PURCHASE_HOWTO,
            ):
                _account_list_early = True
    except ImportError:
        pass
    if _account_list_early:
        if _brain_is_order_live_turn(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
        ) or brain_route.get("needs_order_id"):
            log_reasoning(
                "Skip account-list — brain locked single-order live (invoice/track/refund/details)."
            )
            _account_list_early = False
    if _account_list_early:
        account_early = _try_brain_account_list_direct_reply(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            reset_context_fn=reset_context_fn,
        )
        if account_early:
            return account_early
        # Brain already classified account-list — never fall through to single-order LLM stack.
        log_reasoning(
            f"Brain account-list direct API — intent={brain_intent} (no specialist LLM)."
        )
        reset_context_fn(ctx)
        if brain_intent == "wishlist":
            return format_wishlist_reply(user_id, page=1, append_only=False)
        return format_purchase_history_reply(user_id, page=1, append_only=False)

    order_direct = _try_brain_order_live_direct_reply(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if order_direct:
        return order_direct

    fallback_live = _try_brain_order_live_fallback_dispatch(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        reset_context_fn=reset_context_fn,
    )
    if fallback_live:
        return fallback_live

    kb_direct = _try_brain_kb_locked_reply(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if kb_direct:
        return kb_direct

    # --- Product catalog handled above (after pincode) ---

    # Scope chitchat only after actionable routes — never when brain locked shopping/delivery.
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        if _brain_is_pincode_delivery_turn(brain_route):
            pin_scope = _try_brain_pincode_direct_reply(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
            )
            if pin_scope:
                log_reasoning(
                    "Brain scope override — pincode delivery (scope_reply ignored)."
                )
                return pin_scope
        try:
            from services.ai_route_semantics import brain_route_indicates_product_catalog
        except ImportError:
            brain_route_indicates_product_catalog = lambda _r: False  # type: ignore
        if _brain_is_product_catalog_turn(
            brain_route, original_msg, msg_en
        ) or brain_route_indicates_product_catalog(brain_route):
            product_route, sq = _prepare_brain_product_route(
                brain_route, original_msg, msg_en
            )
            log_reasoning(
                f"Brain scope override — product catalog sq={sq!r} (scope_reply ignored)."
            )
            return _run_product_catalog_flow(
                product_route,
                sq,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
            )
        try:
            from services.order_live_intent_fast_path import (
                resolve_live_goal_lightweight,
                try_order_live_intent_fast_reply,
            )

            live_goal = resolve_live_goal_lightweight(
                original_msg, msg_en, conv_for_llm, reply_lang=lang
            )
            if live_goal in (
                "refund_status",
                "order_invoice",
                "order_details",
                "track",
                "payment",
            ):
                live_reply = try_order_live_intent_fast_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm,
                    user_id,
                    lang,
                    ctx,
                    reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context_fn,
                    preset_goal=live_goal,
                )
                if live_reply:
                    log_reasoning(
                        f"Brain scope bypass — message live goal={live_goal} (not chitchat)."
                    )
                    return live_reply
        except ImportError:
            pass
        if _scope_reply_is_placeholder(scope_reply):
            scope_reply = ""
        if not scope_reply:
            try:
                from services.ai_route_semantics import (
                    brain_route_indicates_product_catalog,
                    reconcile_product_catalog_from_brain_meaning,
                )
                from services.product_catalog_resolver import product_catalog_route_is_locked

                catalog_route = reconcile_product_catalog_from_brain_meaning(
                    dict(brain_route),
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
                if product_catalog_route_is_locked(catalog_route) or brain_route_indicates_product_catalog(
                    catalog_route
                ):
                    product_route, sq = _prepare_brain_product_route(
                        catalog_route, original_msg, msg_en
                    )
                    log_reasoning(
                        f"Brain scope bypass — AI catalog sq={sq!r} (not keyword routing)."
                    )
                    return _run_product_catalog_flow(
                        product_route,
                        sq,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conv_for_llm=conv_for_llm,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context_fn,
                    )
            except ImportError:
                pass
            try:
                from services.conversational_ack_flow import resolve_brain_scope_customer_reply

                scope_reply = resolve_brain_scope_customer_reply(
                    brain_route,
                    scope,
                    original_msg,
                    msg_en=msg_en,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ctx=ctx,
                )
            except ImportError:
                pass
        if scope_reply:
            log_reasoning(f"Brain direct dispatch: scope_reply ({scope}).")
            try:
                from services.chat_flow_telemetry import mark_routing_complete, store_turn_analysis
                from services.answer_router import AnswerRouteDecision

                mark_routing_complete()
                store_turn_analysis(
                    brain_route,
                    AnswerRouteDecision(
                        source="scope",
                        intent="general" if scope == "general_chitchat" else "out_of_domain",
                        handler="warm_feedback" if scope == "general_chitchat" else "off_topic",
                        is_welfog_related=scope != "out_of_domain",
                        reason=f"Brain scope {scope}",
                    ),
                )
            except ImportError:
                pass
            try:
                from utils.helpers import (
                    message_is_casual_farewell_or_closing,
                    message_is_user_feedback_or_closing,
                )

                comb = f"{original_msg} {msg_en}".strip()
                if not (
                    message_is_user_feedback_or_closing(comb)
                    or message_is_casual_farewell_or_closing(comb)
                ):
                    reset_context_fn(ctx)
            except ImportError:
                reset_context_fn(ctx)
            return scope_reply

    skip_account = scope in ("general_chitchat", "out_of_domain", "harm_sensitive") or meta_kind in (
        "conversational",
        "assistant_intro",
    ) or brain_intent in (
        "general",
        "out_of_domain",
        "product",
        "refund",
        "order",
        "order_invoice",
        "payment",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
    ) or brain_olk in ("invoice", "details", "track", "tracking", "refund_status") or brain_rh in (
        "order_details_api",
        "order_tracking_api",
        "refund_status_api",
    )

    # --- Wishlist / order history fallback (micro-LLM when brain did not lock kind) ---
    if not skip_account:
        account_reply = _try_brain_account_list_direct_reply(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            reset_context_fn=reset_context_fn,
        )
        if account_reply:
            return account_reply

    # --- Refund / track / order details / invoice with id ---
    try:
        import re

        from utils.helpers import _text_is_refund_return_status_lookup, extract_order_id

        comb_live = f"{original_msg} {msg_en}".strip()
        live_route = dict(brain_route)
        try:
            from services.ai_route_semantics import (
                correct_order_details_vs_tracking_from_ai_meaning,
            )

            live_route = correct_order_details_vs_tracking_from_ai_meaning(live_route)
        except ImportError:
            pass
        intent = (live_route.get("intent") or "").strip().lower()
        olk = (live_route.get("order_lookup_kind") or "").strip().lower()
        rh = (live_route.get("route_handler") or "").strip().lower()

        if olk == "invoice":
            live_route["data_channel"] = "live_api"
            live_route["route_handler"] = "order_details_api"
            live_route["order_lookup_kind"] = "invoice"
            live_route["needs_order_id"] = True
            live_route["numeric_context"] = "order_id"
            live_route["run_catalog_search"] = False
            ctx.setdefault("data", {})["ai_route"] = live_route
        elif olk in ("details", "order_details") or (
            rh == "order_details_api" and intent in ("order", "payment")
        ):
            live_route["data_channel"] = "live_api"
            live_route["route_handler"] = "order_details_api"
            live_route["order_lookup_kind"] = "details"
            live_route["needs_order_id"] = True
            live_route["numeric_context"] = "order_id"
            live_route["run_catalog_search"] = False
            ctx.setdefault("data", {})["ai_route"] = live_route
        elif olk in ("track", "tracking") or rh == "order_tracking_api":
            live_route["data_channel"] = "live_api"
            live_route["route_handler"] = "order_tracking_api"
            live_route["order_lookup_kind"] = "track"
            live_route["needs_order_id"] = True
            live_route["numeric_context"] = "order_id"
            live_route["run_catalog_search"] = False
            ctx.setdefault("data", {})["ai_route"] = live_route
        elif intent == "refund" or olk == "refund_status":
            if _text_is_refund_return_status_lookup(comb_live, conv_for_llm) and (
                extract_order_id(comb_live) or re.search(r"\b\d{6,}\b", comb_live)
            ):
                live_route["data_channel"] = "live_api"
                live_route["route_handler"] = "refund_status_api"
                live_route["order_lookup_kind"] = "refund_status"
                live_route["needs_order_id"] = True
                live_route["numeric_context"] = "order_id"
                live_route["run_catalog_search"] = False
                ctx.setdefault("data", {})["ai_route"] = live_route
    except ImportError:
        pass

    try:
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

        skip_live = brain_intent in (
            "deals",
            "categories",
            "category_feed",
            "wishlist",
            "out_of_domain",
        )
        if brain_intent == "order_history" and not (
            brain_route.get("needs_order_id")
            or _brain_is_order_live_turn(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
            )
        ):
            skip_live = True
        if brain_intent == "general" and not brain_route.get("needs_order_id"):
            skip_live = True
        if not skip_live:
            live_reply = _try_brain_order_live_fallback_dispatch(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
                reset_context_fn=reset_context_fn,
            )
            if live_reply:
                return live_reply
    except ImportError:
        pass

    product_fb = _try_brain_product_catalog_fallback_dispatch(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )
    if product_fb:
        return product_fb

    return None


def try_finish_brain_classified_turn(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    format_purchase_history_reply: Callable[..., str],
    format_wishlist_reply: Callable[..., str],
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """
    Cached brain already classified this turn — dispatch once, never legacy LLM stack.
    """
    if not isinstance(brain_route, dict) or brain_route.get("llm_unavailable"):
        return None

    # Single execution plan: delivery/pincode BEFORE KB and product.
    # Brain may mislabel city serviceability as kb/catalog; semantic reconcile corrects.
    try:
        from services.ai_route_semantics import reconcile_pincode_delivery_from_brain_meaning
        from services.chat_flow_telemetry import (
            is_authoritative_pincode_route_locked,
            should_skip_post_pincode_route_steal,
        )

        pin_route = reconcile_pincode_delivery_from_brain_meaning(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        if (
            pin_route.get("_pincode_delivery_locked")
            or (pin_route.get("intent") or "").strip().lower() == "pincode_check"
            or is_authoritative_pincode_route_locked()
        ):
            pin_first = _try_brain_pincode_direct_reply(
                pin_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
            )
            if pin_first:
                log_reasoning(
                    "Brain finish — pincode delivery first (single execution plan)."
                )
                return pin_first
        if should_skip_post_pincode_route_steal("try_finish_post_pin"):
            return None
    except ImportError:
        pass

    channel_finish = (brain_route.get("data_channel") or "").strip().lower()
    if channel_finish == "kb" and not brain_route.get("needs_order_id"):
        kb_first = _try_brain_immediate_scope_or_kb_reply(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        )
        if kb_first:
            log_reasoning("Brain finish — KB first (channel=kb, skip product stack).")
            return kb_first
        kb_direct = try_brain_direct_dispatch(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            reset_context_fn=reset_context_fn,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        )
        if kb_direct:
            log_reasoning("Brain finish — KB direct dispatch (channel=kb).")
            return kb_direct

    try:
        from services.chat_flow_telemetry import should_skip_post_pincode_route_steal
        from services.ai_route_semantics import (
            brain_route_indicates_product_catalog,
            reconcile_product_catalog_from_brain_meaning,
        )
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if should_skip_post_pincode_route_steal("try_finish_product"):
            pass
        else:
            catalog_route = reconcile_product_catalog_from_brain_meaning(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            if product_catalog_route_is_locked(
                catalog_route
            ) or brain_route_indicates_product_catalog(catalog_route):
                product_first = _try_brain_product_catalog_fallback_dispatch(
                    catalog_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context_fn,
                )
                if product_first:
                    log_reasoning(
                        "Brain finish — product OpenSearch first (skip KB embed queue)."
                    )
                    return product_first
    except ImportError:
        pass

    immediate = _try_brain_immediate_scope_or_kb_reply(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if immediate:
        log_reasoning("Brain finish — immediate scope/KB from cached route.")
        return immediate

    direct = try_brain_direct_dispatch(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        format_purchase_history_reply=format_purchase_history_reply,
        format_wishlist_reply=format_wishlist_reply,
        reset_context_fn=reset_context_fn,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if direct:
        log_reasoning("Brain finish — direct dispatch from cached route.")
        return direct

    scope = (brain_route.get("conversation_scope") or "").strip().lower()
    intent = (brain_route.get("intent") or "").strip().lower()
    if intent == "out_of_domain":
        scope = "out_of_domain"
    elif not scope and intent in ("general",):
        scope = "general_chitchat"
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        try:
            from services.conversational_ack_flow import resolve_brain_scope_customer_reply

            body = resolve_brain_scope_customer_reply(
                brain_route,
                scope,
                original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
            )
            if body:
                log_reasoning(f"Brain finish — scope AI reply ({scope}).")
                return body
        except ImportError:
            pass

    # Product catalog / structural OpenSearch — brain classified but early dispatch missed.
    try:
        from services.ai_route_semantics import (
            brain_route_indicates_product_catalog,
            reconcile_product_catalog_from_brain_meaning,
        )
        from services.product_catalog_resolver import product_catalog_route_is_locked

        catalog_route = reconcile_product_catalog_from_brain_meaning(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if product_catalog_route_is_locked(catalog_route) or brain_route_indicates_product_catalog(
            catalog_route
        ):
            product_fb = _try_brain_product_catalog_fallback_dispatch(
                catalog_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context_fn,
            )
            if product_fb:
                log_reasoning("Brain finish — product catalog OpenSearch from cached route.")
                return product_fb
    except ImportError:
        pass

    try:
        structural = try_structural_product_catalog_reply(
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
        )
        if structural:
            log_reasoning("Brain finish — structural product OpenSearch (msg_en fallback).")
            return structural
    except ImportError:
        pass

    try:
        order_finish = _try_brain_order_live_direct_reply(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        )
        if not order_finish:
            order_finish = _try_brain_order_live_fallback_dispatch(
                brain_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
                reset_context_fn=reset_context_fn,
            )
        if order_finish:
            log_reasoning("Brain finish — order track/invoice ask-ID or live API.")
            return order_finish
    except ImportError:
        pass
    return None
