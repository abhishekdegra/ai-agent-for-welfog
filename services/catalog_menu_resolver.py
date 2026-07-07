"""
AI-first Welfog catalog menu routing — deals / categories list / category browse.

Any language: brain route is primary; micro-classifier is safety net when brain
misroutes to product search or KB. Deterministic signals are fallback when LLM is down.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from utils.reasoning_log import log_reasoning

_RESOLVE_CACHE = threading.local()

KIND_NONE = "none"
KIND_DEALS = "deals"
KIND_CATEGORIES_LIST = "categories_list"
KIND_CATEGORY_BROWSE = "category_browse"

_CATALOG_MENU_INTENTS = frozenset({"deals", "categories", "category_feed"})


@dataclass
class ResolvedCatalogMenuTurn:
    kind: str = KIND_NONE
    category_name: str = ""
    search_query: str = ""
    source: str = ""
    user_meaning: str = ""
    confidence: float = 0.0


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _suspicious_product_search_for_menu(ai_route: dict | None) -> bool:
    """Brain locked product search but query looks like a menu word, not a SKU."""
    if not isinstance(ai_route, dict):
        return False
    sq = (ai_route.get("search_query") or "").strip().lower()
    if not sq:
        return False
    menu_tokens = (
        "categor",
        "catagori",
        "department",
        "section",
        "deal",
        "offer",
        "discount",
        "promo",
        "sale",
    )
    return any(tok in sq for tok in menu_tokens)


def _brain_catalog_menu_resolution(ai_route: dict | None) -> Optional[ResolvedCatalogMenuTurn]:
    """Trust ai_brain_route when it already chose deals / categories."""
    if not isinstance(ai_route, dict):
        return None
    intent = (ai_route.get("intent") or "").strip().lower()
    um = (ai_route.get("user_meaning") or "").strip()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if intent == "deals":
        return ResolvedCatalogMenuTurn(
            kind=KIND_DEALS,
            source="brain_route",
            user_meaning=um,
            confidence=0.92,
        )
    if intent in ("categories", "category_feed"):
        return ResolvedCatalogMenuTurn(
            kind=KIND_CATEGORIES_LIST,
            source="brain_route",
            user_meaning=um,
            confidence=0.92,
        )
    if intent == "product" and channel == "catalog" and _suspicious_product_search_for_menu(
        ai_route
    ):
        return None
    return None


def _semantic_signal_catalog_menu(
    original_msg: str,
    msg_en: str = "",
) -> Optional[ResolvedCatalogMenuTurn]:
    """Fast semantic signals — typo-tolerant, not a substitute for brain/LLM."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return None
    try:
        from services.conversation_followup import is_deals_request_message
        from utils.helpers import message_asks_welfog_categories_list

        if is_deals_request_message(original_msg, msg_en):
            return ResolvedCatalogMenuTurn(
                kind=KIND_DEALS,
                source="semantic_signal",
                confidence=0.88,
            )
        if message_asks_welfog_categories_list(comb):
            return ResolvedCatalogMenuTurn(
                kind=KIND_CATEGORIES_LIST,
                source="semantic_signal",
                confidence=0.88,
            )
    except ImportError:
        pass
    return None


def _user_meaning_suggests_catalog_menu(ai_route: dict | None) -> bool:
    """Brain English meaning hints at deals/categories but intent may be wrong."""
    if not isinstance(ai_route, dict):
        return False
    um = (ai_route.get("user_meaning") or "").lower()
    if not um:
        return False
    menu_topic = re.search(
        r"\b(categor|catagori|department|section|deal|offer|discount|promo|flash\s+sale)\b",
        um,
    )
    menu_action = re.search(
        r"\b(list|show|today|todays|all|menu|browse|display|top)\b",
        um,
    )
    return bool(menu_topic and (menu_action or "categor" in um or "deal" in um))


def _should_invoke_catalog_menu_classifier(
    ai_route: dict | None,
    original_msg: str,
    msg_en: str = "",
) -> bool:
    """One micro-LLM call only when brain likely misrouted catalog menu."""
    if _brain_catalog_menu_resolution(ai_route):
        return False
    if _semantic_signal_catalog_menu(original_msg, msg_en):
        return False
    if not isinstance(ai_route, dict):
        return True
    intent = (ai_route.get("intent") or "").strip().lower()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if intent in _CATALOG_MENU_INTENTS:
        return False
    try:
        from services.account_list_semantics import (
            detect_account_list_followup_in_chat,
            message_requests_account_list_data,
        )

        comb = _combined(original_msg, msg_en)
        if message_requests_account_list_data(comb):
            return False
        if detect_account_list_followup_in_chat(original_msg, msg_en):
            return False
    except ImportError:
        pass
    if intent in (
        "order",
        "order_history",
        "wishlist",
        "refund",
        "payment",
        "pincode_check",
        "seller",
        "out_of_domain",
    ):
        return False
    if _suspicious_product_search_for_menu(ai_route):
        return True
    if _user_meaning_suggests_catalog_menu(ai_route):
        return True
    mk = (ai_route.get("meta_kind") or "none").strip().lower()
    if mk in ("conversational", "assistant_intro", "hostile"):
        return False
    scope = (ai_route.get("conversation_scope") or "").strip().lower()
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        if intent in ("general", "product", "category_browse") and ai_route.get(
            "is_welfog_related", True
        ) is not False:
            return True
        return False
    if intent == "product" and channel == "catalog" and ai_route.get("run_catalog_search"):
        return True
    return False


def ai_classify_catalog_menu_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> Optional[ResolvedCatalogMenuTurn]:
    """Micro-classifier: deals / categories list / category browse vs none — any language."""
    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None
    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1200)
    user_line = _trim_text_mid(comb, 600)
    route_hint = ""
    if isinstance(ai_route, dict):
        route_hint = (
            f"brain_intent={ai_route.get('intent') or '-'} "
            f"meaning={(ai_route.get('user_meaning') or '')[:120]}"
        )

    system_prompt = f"""You classify Welfog CATALOG MENU requests for the LATEST user message.

The customer may write in ANY language, script, dialect, slang, typos, or Hinglish.
Understand MEANING — never match fixed English/Hindi keywords only.

Return ONLY JSON:
{{
  "user_meaning": "one English sentence",
  "turn_kind": "none" | "deals" | "categories_list" | "category_browse",
  "category_name": "English category name if category_browse else empty",
  "confidence": 0.0 to 1.0
}}

turn_kind rules:
- deals: today's deals / offers / discounts / flash sale on Welfog — NOT a product literally named "deals"
- categories_list: full list of shopping departments/categories on Welfog (what can I shop, show all sections)
- category_browse: show products INSIDE one named category (beauty, electronics, grocery, fashion)
- none: find/buy a specific product, orders, refund, KB policy, greeting, off-topic

Examples (same intent across languages):
- "aaj ki top deals" / "show today's offers" / "இன்றைய சலுகைகள்" / "आज के ऑफर" → deals
- "categories list" / "kya kya category hai" / "विभाग दिखाओ" / "show all departments" → categories_list
- "beauty products dikhao" / "electronics ke items" → category_browse, category_name=beauty/electronics
- "lal mirch chahiye" / "iPhone 15 cover" → none (product search)

{language_reply_instruction(rl)}
JSON only."""

    user_payload = f"ROUTER: {route_hint}\nLATEST:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CHAT:\n{compact_ctx}\n\n{user_payload}"

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=220,
        timeout_sec=14,
        max_attempts=1,
    )
    if not data:
        return None

    kind = (data.get("turn_kind") or KIND_NONE).strip().lower()
    if kind not in (KIND_DEALS, KIND_CATEGORIES_LIST, KIND_CATEGORY_BROWSE, KIND_NONE):
        kind = KIND_NONE
    conf = float(data.get("confidence") or 0.0)
    um = (data.get("user_meaning") or "").strip()
    cat = (data.get("category_name") or "").strip()
    log_reasoning(
        f"Catalog-menu LLM: kind={kind} conf={conf:.2f} cat={cat!r} — {um[:80] or '-'}"
    )
    if kind == KIND_NONE or conf < 0.52:
        return None
    return ResolvedCatalogMenuTurn(
        kind=kind,
        category_name=cat,
        search_query=cat,
        source="catalog_menu_llm",
        user_meaning=um,
        confidence=conf,
    )


def resolve_catalog_menu_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
    force_llm: bool = False,
) -> ResolvedCatalogMenuTurn:
    """
    Priority: brain route → semantic signals → AI micro-classifier → none.
    force_llm bypasses micro-classifier defer after universal brain routing.
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return ResolvedCatalogMenuTurn(kind=KIND_NONE)

    if force_llm:
        try:
            from services.chat_flow_telemetry import ensure_product_rescue_llm_slot

            ensure_product_rescue_llm_slot()
        except ImportError:
            pass
    else:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                brain = _brain_catalog_menu_resolution(ai_route)
                if brain:
                    _RESOLVE_CACHE.key = (
                        f"{hash(comb)}|{hash((conversation_context or '')[-200:])}|"
                        f"{hash(str((ai_route or {}).get('intent')))}|{allow_llm}|{force_llm}"
                    )
                    _RESOLVE_CACHE.result = brain
                    return brain
                log_reasoning(
                    "Catalog-menu: defer/skip — universal brain route owns classification."
                )
                return ResolvedCatalogMenuTurn(kind=KIND_NONE)
        except ImportError:
            pass

    cache_key = (
        f"{hash(comb)}|{hash((conversation_context or '')[-200:])}|"
        f"{hash(str((ai_route or {}).get('intent')))}|{allow_llm}|{force_llm}"
    )
    if getattr(_RESOLVE_CACHE, "key", None) == cache_key:
        cached = getattr(_RESOLVE_CACHE, "result", None)
        if isinstance(cached, ResolvedCatalogMenuTurn):
            return cached

    brain = _brain_catalog_menu_resolution(ai_route)
    if brain:
        _RESOLVE_CACHE.key = cache_key
        _RESOLVE_CACHE.result = brain
        return brain

    if not force_llm:
        signal = _semantic_signal_catalog_menu(original_msg, msg_en)
        if signal:
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = signal
            return signal

    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(ai_route) and not force_llm:
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = ResolvedCatalogMenuTurn(kind=KIND_NONE)
            return ResolvedCatalogMenuTurn(kind=KIND_NONE)
    except ImportError:
        pass

    invoke_llm = force_llm or (
        allow_llm and _should_invoke_catalog_menu_classifier(ai_route, original_msg, msg_en)
    )
    if invoke_llm:
        classified = ai_classify_catalog_menu_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            reply_lang=reply_lang,
        )
        if classified:
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = classified
            return classified

    none = ResolvedCatalogMenuTurn(kind=KIND_NONE)
    _RESOLVE_CACHE.key = cache_key
    _RESOLVE_CACHE.result = none
    return none


def turn_requests_catalog_menu(
    original_msg: str,
    msg_en: str = "",
    ai_route: dict | None = None,
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    resolved = resolve_catalog_menu_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    return resolved.kind != KIND_NONE


def _build_route_data_from_resolution(
    resolved: ResolvedCatalogMenuTurn,
    combined: str,
    ai_route: dict | None,
) -> dict:
    out = dict(ai_route or {})
    out["user_meaning"] = resolved.user_meaning or out.get("user_meaning") or combined[:200]
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out.pop("_product_catalog_locked", None)
    out.pop("run_catalog_search", None)
    out["_catalog_menu_locked"] = True
    out["_catalog_menu_source"] = resolved.source
    if resolved.kind == KIND_DEALS:
        out.update(
            intent="deals",
            data_channel="live_api",
            reasoning="Today's deals / offers on Welfog.",
        )
        out["run_catalog_search"] = False
    elif resolved.kind == KIND_CATEGORIES_LIST:
        out.update(
            intent="categories",
            data_channel="live_api",
            reasoning="List Welfog shopping categories.",
        )
        out["run_catalog_search"] = False
    elif resolved.kind == KIND_CATEGORY_BROWSE:
        sq = resolved.search_query or resolved.category_name
        out.update(
            intent="product",
            data_channel="catalog",
            run_catalog_search=True,
            search_query=sq,
            reasoning=f"Products in category {resolved.category_name or sq}.",
        )
    out.pop("kb_keys", None)
    out["is_welfog_related"] = True
    out["meta_kind"] = "none"
    out["conversation_scope"] = "welfog_support"
    out["_turn_promotions_done"] = True
    return out


def guard_reconcile_catalog_menu_route(
    route_data: dict,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> dict:
    """
    Guard: brain JSON first; ONE catalog-menu LLM when misrouted to KB/chitchat/product.
    Any language — not customer keyword lists.
    """
    if not isinstance(route_data, dict):
        return route_data
    comb = _combined(original_msg, msg_en)
    brain_res = _brain_catalog_menu_resolution(route_data)
    if brain_res:
        log_reasoning(f"Guard catalog-menu: brain locked {brain_res.kind}.")
        return _build_route_data_from_resolution(brain_res, comb, route_data)

    needs_ai = (
        _should_invoke_catalog_menu_classifier(route_data, original_msg, msg_en)
        or _user_meaning_suggests_catalog_menu(route_data)
        or _suspicious_product_search_for_menu(route_data)
    )
    intent = (route_data.get("intent") or "").strip().lower()
    channel = (route_data.get("data_channel") or "").strip().lower()
    if (
        not needs_ai
        and channel == "kb"
        and intent in ("general", "payment", "seller", "refund")
        and route_data.get("is_welfog_related", True) is not False
    ):
        needs_ai = True
        log_reasoning(
            "Guard catalog-menu: brain KB channel — verify with catalog-menu LLM."
        )
    if (
        not needs_ai
        and intent == "general"
        and (route_data.get("scope_reply") or "").strip()
        and route_data.get("is_welfog_related", True) is not False
    ):
        needs_ai = True
        log_reasoning(
            "Guard catalog-menu: brain chitchat reply — verify deals/categories intent."
        )
    um_blob = (route_data.get("user_meaning") or "").lower()
    if (
        not needs_ai
        and "categor" in um_blob
        and intent in ("general", "category_browse", "product", "category_feed")
    ):
        needs_ai = True
        log_reasoning(
            "Guard catalog-menu: brain meaning mentions categories — verify list vs browse."
        )
    if not needs_ai and intent == "category_browse":
        needs_ai = True
        log_reasoning(
            "Guard catalog-menu: category_browse without locked menu — verify via LLM."
        )
    if not needs_ai:
        return route_data

    resolved = resolve_catalog_menu_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route_data,
        reply_lang=reply_lang,
        allow_llm=True,
        force_llm=True,
    )
    if resolved.kind != KIND_NONE:
        log_reasoning(
            f"Guard catalog-menu: AI corrected route → {resolved.kind} "
            f"(was intent={(route_data.get('intent') or '-')})."
        )
        return _build_route_data_from_resolution(resolved, comb, route_data)
    return route_data


def catalog_menu_route_decision(
    resolved: ResolvedCatalogMenuTurn,
    original_msg: str,
    msg_en: str,
    ai_route: dict | None = None,
    ctx: dict | None = None,
    reasoning: str = "",
) -> Optional[tuple[Any, dict]]:
    """Build AnswerRouteDecision for a resolved catalog menu turn."""
    from services.answer_router import AnswerRouteDecision

    if resolved.kind == KIND_NONE:
        return None

    combined = _combined(original_msg, msg_en)
    route_data = _build_route_data_from_resolution(resolved, combined, ai_route)
    why = reasoning or resolved.source or "catalog menu"

    if resolved.kind == KIND_DEALS:
        log_reasoning(f"Catalog-menu route → deals API ({resolved.source}).")
        return (
            AnswerRouteDecision(
                source="api",
                intent="deals",
                handler="deals_api",
                is_welfog_related=True,
                reason=f"Deals / offers — {why}",
            ),
            route_data,
        )

    if resolved.kind == KIND_CATEGORIES_LIST:
        log_reasoning(f"Catalog-menu route → categories API ({resolved.source}).")
        return (
            AnswerRouteDecision(
                source="api",
                intent="categories",
                handler="categories_api",
                is_welfog_related=True,
                reason=f"Welfog categories — {why}",
            ),
            route_data,
        )

    if resolved.kind == KIND_CATEGORY_BROWSE:
        try:
            from services.welfog_api import (
                ensure_expanded_categories_map_for_ctx,
                resolve_category_product_browse_route,
            )

            if ctx:
                ensure_expanded_categories_map_for_ctx(ctx)
            browse_text = f"{combined} {resolved.category_name}".strip()
            cat_browse = resolve_category_product_browse_route(browse_text, ctx=ctx)
            if cat_browse:
                cid, sq = cat_browse
                if ctx is not None:
                    ctx.setdefault("data", {})["selected_category_id"] = cid
                    ctx["awaiting"] = None
                route_data["search_query"] = sq
                log_reasoning(
                    f"Catalog-menu route → category browse (category_id={cid}, {resolved.source})."
                )
                return (
                    AnswerRouteDecision(
                        source="ai_product",
                        intent="product",
                        handler="product_ai_flow",
                        search_query=sq,
                        is_welfog_related=True,
                        reason=f"Category browse {cid} — {why}",
                    ),
                    route_data,
                )
        except ImportError:
            pass
        sq = resolved.search_query or resolved.category_name
        route_data["search_query"] = sq
        route_data["run_catalog_search"] = True
        log_reasoning(f"Catalog-menu route → category product search ({resolved.source}).")
        return (
            AnswerRouteDecision(
                source="ai_product",
                intent="product",
                handler="product_ai_flow",
                search_query=sq,
                is_welfog_related=True,
                reason=f"Category browse — {why}",
            ),
            route_data,
        )

    return None


def try_catalog_menu_routing_decision(
    route_data: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
) -> Optional[tuple[Any, dict]]:
    """
    After ai_brain_route: lock deals/categories when brain or micro-classifier agrees.
    """
    resolved = resolve_catalog_menu_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route_data,
        reply_lang=reply_lang,
        allow_llm=True,
    )
    if resolved.kind == KIND_NONE:
        return None
    reasoning = (route_data or {}).get("reasoning") or resolved.user_meaning or ""
    out = catalog_menu_route_decision(
        resolved,
        original_msg,
        msg_en,
        ai_route=route_data,
        ctx=ctx,
        reasoning=str(reasoning)[:200],
    )
    return out
