"""
Product Turn Single Source of Truth (SoT).

Industry pattern:
  ONE AI understanding (Brain JSON or one Product Classifier pass)
    → locked ProductTurnUnderstanding on route
    → CatalogSpec (structural gap-fill only)
    → ONE OpenSearch query
    → Response

No product keyword lists. No downstream reinterpretation of customer text.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def route_is_locked_product_turn(route: dict | None) -> bool:
    if not isinstance(route, dict):
        return False
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route):
            return True
    except ImportError:
        pass
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    return bool(
        route.get("_product_catalog_locked")
        or (intent == "product" and channel == "catalog" and route.get("run_catalog_search"))
    )


def finalize_product_lock(route: dict, *, search_query: str = "", confidence: float = 0.0) -> dict:
    """
    Stamp immutable product-turn flags. Downstream must not downgrade or re-parse user text.
    """
    out = dict(route or {})
    sq = (search_query or out.get("search_query") or "").strip()
    pe = dict(out.get("_product_entities") or {})
    if not isinstance(pe, dict):
        pe = {}
    if sq and not (pe.get("product_name") or "").strip():
        pe["product_name"] = sq
        out["_product_entities"] = pe
        if isinstance(out.get("product_entities"), dict):
            pe2 = dict(out["product_entities"])
            pe2.setdefault("product_name", sq)
            out["product_entities"] = pe2

    out["intent"] = "product"
    out["data_channel"] = "catalog"
    out["run_catalog_search"] = True
    out["_product_catalog_locked"] = True
    out["is_welfog_related"] = True
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["meta_kind"] = "none"
    out["conversation_scope"] = "welfog_support"
    out["kb_keys"] = []
    out["_ai_single_pass"] = True
    out.pop("scope_reply", None)
    out.pop("route_handler", None)

    if sq:
        out["search_query"] = sq
        out["_needs_product_nlu_llm"] = False
        out["_product_nlu_from_ai"] = True
        out["_product_understanding_confidence"] = float(confidence or 0.0)
        out.pop("category_only_browse", None)
        if not (pe.get("category") or "").strip():
            out.pop("category_browse", None)
    else:
        out["_needs_product_nlu_llm"] = True

    return out


def brain_json_already_shopping(route: dict | None) -> bool:
    """Brain ai_brain_route JSON already locked shopping — no rescue LLM."""
    if not isinstance(route, dict):
        return False
    try:
        from services.ai_route_semantics import (
            _brain_route_has_shopping_entities,
            brain_route_indicates_product_catalog,
        )

        if brain_route_indicates_product_catalog(route):
            return True
        if _brain_route_has_shopping_entities(route):
            return True
    except ImportError:
        pass
    intent = (route.get("intent") or "").strip().lower()
    if intent in ("product", "product_search") and route.get("run_catalog_search"):
        return True
    pe = route.get("_product_entities") or route.get("product_entities") or {}
    if isinstance(pe, dict) and (pe.get("product_name") or pe.get("brand") or pe.get("sku")):
        return True
    sq = (route.get("search_query") or "").strip()
    return bool(sq and route.get("run_catalog_search") is True)


def lock_product_turn_from_ai(
    route: dict,
    *,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    allow_rescue_llm: bool = True,
) -> dict | None:
    """
    Promote route to locked product catalog when AI says shopping.
    At most ONE extra classifier LLM when Brain misrouted (not when Brain already product).
    """
    if not isinstance(route, dict):
        return None

    if route_is_locked_product_turn(route):
        return finalize_product_lock(route)

    if brain_json_already_shopping(route):
        try:
            from services.ai_route_semantics import reconcile_product_catalog_from_brain_meaning

            locked = reconcile_product_catalog_from_brain_meaning(
                route, original_msg=original_msg, msg_en=msg_en
            )
            if route_is_locked_product_turn(locked):
                log_reasoning(
                    "Product SoT — brain JSON locked catalog "
                    f"(sq={(locked.get('search_query') or '')!r})."
                )
                return finalize_product_lock(locked)
        except ImportError:
            pass
        return finalize_product_lock(route)

    if not allow_rescue_llm:
        return None

    try:
        from services.order_turn_sot import brain_route_blocks_product_rescue

        if brain_route_blocks_product_rescue(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
        ):
            return None
    except ImportError:
        pass

    try:
        from services.ai_first_router import _try_brain_misroute_product_rescue_via_ai

        rescued = _try_brain_misroute_product_rescue_via_ai(
            route,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
        )
        if isinstance(rescued, dict) and route_is_locked_product_turn(rescued):
            conf = float(rescued.get("_product_understanding_confidence") or 0.0)
            return finalize_product_lock(
                rescued,
                search_query=(rescued.get("search_query") or "").strip(),
                confidence=conf,
            )
    except ImportError:
        pass
    return None


def understanding_from_locked_route(
    route: dict,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict[str, Any]:
    """
    Build catalog understanding ONLY from locked route SoT — never re-read customer text.
    """
    try:
        from services.product_catalog_resolver import understanding_from_locked_product_route

        u = understanding_from_locked_product_route(
            route, "", original_msg=original_msg, msg_en=msg_en
        )
        if u:
            u["_ai_first"] = True
            u["_product_nlu_from_ai"] = True
            return u
    except ImportError:
        pass

    sq = (route.get("search_query") or "").strip()
    pe = dict(route.get("_product_entities") or {})
    u: dict[str, Any] = {
        "action": "search_products",
        "is_shopping": True,
        "_ai_first": True,
        "_product_nlu_from_ai": True,
    }
    if sq:
        u["search_terms"] = sq
    if pe:
        for k, v in pe.items():
            if v not in (None, "", [], {}):
                u[k] = v
    return u


def try_run_locked_product_catalog(
    route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
    allow_rescue_llm: bool = True,
) -> tuple[Optional[str], Optional[dict]]:
    """
    Lock product turn (Brain or one classifier) → prepare route → ONE OpenSearch pass.
    Returns (reply_html, locked_route) or (None, None).
    """
    authoritative_product = bool(
        (route.get("intent") or "").strip().lower() in ("product", "product_search")
        and (route.get("data_channel") or "").strip().lower() == "catalog"
        and route.get("_product_catalog_locked")
    )
    if authoritative_product:
        # Universal Brain already resolved conflicts and locked this turn. Running
        # order/refund/KB blockers (and lock rescue) again traverses legacy intent
        # graphs and can touch MySQL before OpenSearch.
        locked = dict(route)
        log_reasoning(
            "Product SoT — authoritative Brain lock; skip blocker/rescue graphs."
        )
    else:
        try:
            from services.order_turn_sot import brain_route_blocks_product_rescue

            if brain_route_blocks_product_rescue(
                route,
                ctx=ctx,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            ):
                log_reasoning("Product SoT skipped — order/refund session or live API locked.")
                return None, None
            from services.kb_turn_sot import kb_turn_blocks_product_rescue

            if kb_turn_blocks_product_rescue(route):
                log_reasoning("Product SoT skipped — KB route locked.")
                return None, None
        except ImportError:
            pass

        locked = lock_product_turn_from_ai(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
            allow_rescue_llm=allow_rescue_llm,
        )
    if not locked:
        return None, None

    try:
        from services.brain_direct_dispatch import (
            _prepare_brain_product_route,
            _run_product_catalog_flow,
        )

        _t_prepare = time.perf_counter()
        product_route, sq = _prepare_brain_product_route(
            locked, original_msg, msg_en
        )
        log_reasoning(
            f"Product SoT phase prepare={(time.perf_counter() - _t_prepare):.3f}s."
        )
        _t_catalog = time.perf_counter()
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
        log_reasoning(
            f"Product SoT phase catalog={(time.perf_counter() - _t_catalog):.3f}s."
        )
        if body:
            log_reasoning(
                f"Product SoT dispatch — sq={sq!r} (1 understanding → 1 OpenSearch)."
            )
            return body, product_route
    except ImportError:
        pass
    return None, locked


def product_turn_blocks_kb_or_ood(route: dict | None) -> bool:
    """When True, KB/OOD/chitchat dispatch must not run."""
    return route_is_locked_product_turn(route)
