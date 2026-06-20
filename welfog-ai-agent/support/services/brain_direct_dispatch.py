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
        return low in ("warm", "thanks", "ok", "okay", "hi", "hello")
    markers = (
        "warm natural reply",
        "customer language",
        "scope_reply",
        "in the customer",
        "1-3 sentences",
        "2-5 sentences",
        "empty when",
        "required when",
    )
    return any(m in low for m in markers)


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
        kb_keys = (
            ["refund", "faqs"]
            if intent == "refund"
            else ["faqs", "refund"]
            if "refund" in f"{original_msg} {msg_en}".lower()
            else ["faqs"]
        )
    try:
        from services.kb_service import format_kb_answer_from_brain_keys

        direct = format_kb_answer_from_brain_keys(
            original_msg,
            msg_en,
            kb_keys,
            reply_lang=lang,
            conversation_context=conv_for_llm,
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
    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            reconcile_wishlist_from_brain_meaning,
            _kind_from_meaning_blob,
            _meaning_blob,
            KIND_WISHLIST_HOWTO,
            KIND_WISHLIST_IN_CHAT,
        )

        fixed = reconcile_wishlist_from_brain_meaning(dict(brain))
        if account_list_route_is_locked(fixed):
            return False
        mk = _kind_from_meaning_blob(_meaning_blob(fixed))
        if mk in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
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
        from services.ai_route_semantics import (
            _brain_product_entities_from_route,
            _catalog_search_query_from_brain_route,
        )

        entities = _brain_product_entities_from_route(route, original_msg=original_msg, msg_en=msg_en)
        if entities:
            route["_product_entities"] = entities
        sq = _catalog_search_query_from_brain_route(route)
    except ImportError:
        sq = (route.get("search_query") or "").strip()

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

    if not sq:
        sq = (entities.get("product_name") or route.get("search_query") or "").strip()
    if not sq:
        from utils.reasoning_log import log_reasoning

        sq = "products"
        log_reasoning(
            "Brain product turn missing product_name/search_query — generic browse fallback."
        )

    route["search_query"] = sq
    route["_ai_single_pass"] = True
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
    if bool(brain_route.get("category_only_browse")):
        return True
    if (brain_route.get("category_browse") or "").strip():
        return True
    entities = brain_route.get("entities") or {}
    if not isinstance(entities, dict):
        entities = {}
    if any(
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
        )
    ):
        return False
    try:
        from services.welfog_api import (
            _message_has_product_search_filters,
            _user_explicitly_browses_category,
            resolve_nav_category_id_fast,
        )

        comb = f"{original_msg} {msg_en}".strip()
        if comb and _message_has_product_search_filters(comb):
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
    Explicit warehouse SKU in message → OpenSearch only (zero LLM).
    e.g. 'is sku ka product bta INFINIX HOT 10 PLAY-EGL-SP'
    """
    comb = f"{original_msg} {msg_en}".strip()
    if not comb or not re.search(r"\bsku\b", comb, re.I):
        return None
    try:
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

        sku = _extract_sku_from_text(original_msg) or _extract_sku_from_text(msg_en)
        if not sku or not _sku_token_acceptable(sku, explicit_sku_mention=True):
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
            _user_explicitly_browses_category,
            build_welfog_product_browse_url,
            category_name_for_id,
            query_should_use_category_id_only,
            resolve_category_browse_for_catalog,
            resolve_nav_category_id_fast,
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
        if not _user_explicitly_browses_category(browse_text) and not resolve_nav_category_id_fast(
            browse_text, ctx
        ):
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
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return None
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
        }
        sq = resolve_catalog_search_terms_for_message(
            original_msg, msg_en, ai_route=product_route
        )
        if not sq or len(sq) < 2:
            return None
    except ImportError:
        return None

    product_route["search_query"] = sq
    product_route["user_meaning"] = f"Show {sq}"
    log_reasoning(f"Structural product fast path sq={sq!r} → OpenSearch (skip brain LLM).")
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
    return False


def _extract_pincode_from_brain_route(brain_route: dict, original_msg: str, msg_en: str) -> str:
    import re

    pin = re.sub(r"\D", "", str(brain_route.get("extracted_pincode") or ""))
    if len(pin) == 6 and pin[0] != "0":
        return pin
    try:
        from utils.helpers import resolve_pincode_for_check

        pin = resolve_pincode_for_check(
            original_msg,
            "",
            msg_en=msg_en,
            ai_extracted=brain_route.get("extracted_pincode"),
            ai_route=brain_route,
        )
        if pin:
            return pin
    except ImportError:
        pass
    comb = f"{original_msg} {msg_en}".strip()
    m = re.search(r"\b([1-9]\d{5})\b", comb)
    return m.group(1) if m else ""


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
    if pin:
        route_for_api["extracted_pincode"] = pin
    if location and not (route_for_api.get("search_query") or "").strip():
        route_for_api["search_query"] = location

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
                api_res = check_pincode_delivery(pin)
                api_called = True
                reply_html = format_pincode_check_reply(
                    pin, api_res, original_msg, lang
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
                allow_llm=not bool(pin),
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
        ctx["last"] = "pincode"
        ctx["awaiting"] = None
        ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
        ctx.setdefault("data", {})["last_pincode"] = pin
        ctx.setdefault("data", {})["ai_route"] = route_for_api
    else:
        from utils.helpers import set_pincode_await_context

        set_pincode_await_context(ctx, route_for_api)
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

    # Product catalog first when brain already locked shopping — beats order/KB detours.
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
            KIND_WISHLIST_HOWTO,
            KIND_WISHLIST_IN_CHAT,
        )

        if account_list_route_is_locked(brain_route):
            _account_list_early = True
        else:
            mk = _kind_from_meaning_blob(_meaning_blob(brain_route))
            if mk in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
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

    pincode_direct = _try_brain_pincode_direct_reply(
        brain_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
    )
    if pincode_direct:
        return pincode_direct

    # --- Product catalog block moved to top of dispatch (after catalog menu) ---

    # Account-list before chitchat — brain already classified meaning (any language).
    _account_list_early = False
    try:
        from services.account_list_semantics import account_list_route_is_locked

        _account_list_early = account_list_route_is_locked(brain_route)
    except ImportError:
        pass
    if brain_intent in ("order_history", "wishlist"):
        _account_list_early = True
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

    # Scope chitchat only after actionable routes — avoids product turns mislabeled as general.
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
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
                from services.conversation_scope import (
                    SCOPE_OUT,
                    _generic_scope_fallback,
                    finalize_scope_reply_html,
                )
                from services.translation_service import is_hinglish_message
                from utils.helpers import build_warm_conversation_reply

                if scope == "out_of_domain":
                    use_h = lang == "hinglish" or is_hinglish_message(
                        original_msg or msg_en
                    )
                    sr = _generic_scope_fallback(SCOPE_OUT, use_h)
                    scope_reply = finalize_scope_reply_html(
                        sr, original_msg, reply_lang=lang
                    )
                else:
                    scope_reply = (
                        build_warm_conversation_reply(
                            original_msg, msg_en, reply_lang=lang
                        )
                        or ""
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

    return None
