"""
Pre-API product query understanding: merge AI + heuristics, map filters → catalog spec, debug logs.
Does not change catalog REST/OpenSearch endpoints — only enriches the spec layer.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

_FILTER_KEYS = (
    "title_query",
    "color",
    "size",
    "brand",
    "sku",
    "pro_id",
    "category_id",
    "purchase_price_min",
    "purchase_price_max",
    "rating_min",
    "rating_max",
    "sort",
    "in_stock_only",
    "brand_aliases",
    "mandatory_match_tokens",
    "match_mode",
)

_API_PARAM_KEYS = (
    "name",
    "categories",
    "color",
    "size",
    "brand",
    "sku",
    "pro_id",
    "min_price",
    "max_price",
    "rating_min",
    "rating_max",
    "sort",
    "in_stock",
)


def detected_product_from_turn(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_understanding: Optional[dict] = None,
    spec: Optional[dict] = None,
) -> str:
    """Human-readable product focus for logs (not used for API)."""
    ai = ai_understanding or {}
    if ai.get("pro_id") or (spec or {}).get("pro_id"):
        return f"pro_id:{ai.get('pro_id') or (spec or {}).get('pro_id')}"
    if ai.get("sku") or (spec or {}).get("sku"):
        return f"sku:{ai.get('sku') or (spec or {}).get('sku')}"
    reqs = ai.get("product_requests") or []
    if isinstance(reqs, list) and len(reqs) >= 2:
        labels = [
            (r.get("label") or r.get("search_terms") or "").strip()
            for r in reqs
            if isinstance(r, dict)
        ]
        labels = [x for x in labels if x]
        if labels:
            return " + ".join(labels[:4])
    terms = (ai.get("search_terms") or "").strip()
    if terms:
        return terms
    pt = (ai.get("product_type") or "").strip()
    if pt:
        return pt
    tq = (spec or {}).get("title_query") or ""
    if tq:
        return str(tq).strip()
    try:
        from services.opensearch_products import extract_color_and_product_title

        _, title = extract_color_and_product_title(f"{original_msg} {msg_en}".strip())
        if title:
            return title
    except ImportError:
        pass
    return ""


def filters_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Active catalog filters only (for debug)."""
    if not spec:
        return {}
    out: dict[str, Any] = {}
    for k in _FILTER_KEYS:
        v = spec.get(k)
        if v is None or v == "" or v == []:
            continue
        out[k] = v
    return out


def spec_to_api_params(spec: dict[str, Any]) -> dict[str, Any]:
    """Map internal OpenSearch/REST spec → outward API param names (for logs)."""
    if not spec:
        return {}
    params: dict[str, Any] = {}
    tq = (spec.get("title_query") or "").strip()
    if tq:
        params["name"] = tq
    if spec.get("category_id"):
        params["categories"] = spec["category_id"]
    for src, dst in (
        ("color", "color"),
        ("size", "size"),
        ("brand", "brand"),
        ("sku", "sku"),
        ("pro_id", "pro_id"),
        ("purchase_price_min", "min_price"),
        ("purchase_price_max", "max_price"),
        ("rating_min", "rating_min"),
        ("rating_max", "rating_max"),
        ("sort", "sort"),
    ):
        v = spec.get(src)
        if v is not None and v != "":
            params[dst] = v
    if spec.get("in_stock_only"):
        params["in_stock"] = True
    return {k: params[k] for k in _API_PARAM_KEYS if k in params}


def _user_turn_mentions_price(comb: str) -> bool:
    try:
        from services.opensearch_products import _user_mentions_price_this_turn

        return _user_mentions_price_this_turn(comb)
    except ImportError:
        return bool(re.search(r"\b(?:rs|₹|rupee|inr|price|budget|under|sasta)\b", comb, re.I))


def finalize_catalog_spec_for_api(
    spec: dict[str, Any],
    original_msg: str,
    msg_en: str = "",
    *,
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Last mile before OpenSearch/REST: only clean product keywords + explicit filters.
    Never send full user sentences or conversational filler as title_query.
    """
    if not spec:
        spec = {}
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    spec = enrich_catalog_spec_from_user_turn(
        spec, original_msg, msg_en, ai_understanding=ai_understanding
    )

    ai = ai_understanding or {}
    if _user_turn_mentions_price(comb) and not (
        ai.get("max_price") is not None
        or ai.get("min_price") is not None
        or spec.get("purchase_price_max") is not None
        or spec.get("purchase_price_min") is not None
    ):
        try:
            from services.opensearch_products import _extract_price_bounds

            pmax, pmin = _extract_price_bounds(original_msg or comb, comb.lower())
            if pmax is not None:
                spec["purchase_price_max"] = pmax
            if pmin is not None and (pmax is None or float(pmin) < float(pmax)):
                spec["purchase_price_min"] = pmin
        except ImportError:
            pass

    if spec.get("pro_id"):
        spec["title_query"] = ""
        spec.pop("mandatory_match_tokens", None)
        spec["title_match_strict"] = False
        return spec

    rating_only_browse = bool(
        ai.get("_ai_first")
        and ai.get("rating_min") is not None
        and not (ai.get("search_terms") or "").strip()
        and not (ai.get("brand") or "").strip()
    )
    if rating_only_browse:
        spec["title_query"] = ""
        spec.pop("mandatory_match_tokens", None)
        spec["title_match_strict"] = False
        try:
            from services.opensearch_products import sanitize_product_search_spec

            return sanitize_product_search_spec(spec)
        except ImportError:
            return spec

    try:
        from utils.helpers import _text_is_sim_ejector_pin_product_request

        if _text_is_sim_ejector_pin_product_request(comb):
            spec["title_query"] = "sim ejector pin"
            spec["title_match_strict"] = False
            spec.pop("mandatory_match_tokens", None)
    except ImportError:
        pass

    sku = (spec.get("sku") or "").strip()
    if sku:
        try:
            from services.opensearch_products import _extract_sku_from_text

            extracted = _extract_sku_from_text(comb)
            if extracted:
                spec["sku"] = extracted
        except ImportError:
            pass
        spec["title_query"] = ""
        spec.pop("mandatory_match_tokens", None)
        spec["title_match_strict"] = False

    try:
        from services.product_query_understanding import (
            extract_focused_product_query,
            is_noisy_search_query,
            polish_search_terms,
            scrub_conversational_tail_from_terms,
        )

        raw_tq = scrub_conversational_tail_from_terms((spec.get("title_query") or "").strip())
        polished = polish_search_terms(raw_tq, original_msg) if raw_tq else ""
        if not polished or is_noisy_search_query(polished):
            focused = extract_focused_product_query(original_msg, msg_en)
            if focused:
                polished = polish_search_terms(focused, original_msg)
        if polished and not is_noisy_search_query(polished):
            spec["title_query"] = polished
        elif spec.get("product_type"):
            spec["title_query"] = polish_search_terms(
                str(spec.get("product_type") or ""), original_msg
            )
        else:
            spec.pop("title_query", None)

        if spec.get("purchase_price_max") is not None or spec.get("purchase_price_min") is not None:
            try:
                from services.opensearch_products import (
                    _extract_product_keywords,
                    extract_color_and_product_title,
                )

                _, parsed_title = extract_color_and_product_title(comb)
                kw = _extract_product_keywords(comb.lower()) or parsed_title
                if kw:
                    clean_kw = polish_search_terms(kw, original_msg)
                    if clean_kw and not is_noisy_search_query(clean_kw):
                        spec["title_query"] = clean_kw
            except ImportError:
                pass

        words = (spec.get("title_query") or "").split()
        if len(words) > 6:
            spec["title_query"] = " ".join(words[:6])
    except ImportError:
        pass

    mode = (ai.get("match_mode") or spec.get("match_mode") or "").strip().lower()
    tq_len = len((spec.get("title_query") or "").split())
    if mode == "universal" or tq_len <= 3:
        spec["title_match_strict"] = False
        mandatory = spec.get("mandatory_match_tokens") or []
        if mandatory and tq_len <= 2:
            spec["mandatory_match_tokens"] = [
                m
                for m in mandatory
                if m not in ("product", "products", "item", "items")
            ]
            if not spec["mandatory_match_tokens"]:
                spec.pop("mandatory_match_tokens", None)

    try:
        from services.opensearch_products import sanitize_product_search_spec

        return sanitize_product_search_spec(spec)
    except ImportError:
        return spec


def enrich_catalog_spec_from_user_turn(
    spec: dict[str, Any],
    original_msg: str,
    msg_en: str = "",
    *,
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Gap-fill spec from deterministic parse (price, rating, colour, SKU, pro_id)
    when AI missed fields — any language via opensearch_products heuristics.
    """
    if not spec:
        spec = {}
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return spec

    try:
        from services.opensearch_products import parse_product_filters_from_text

        h = parse_product_filters_from_text(original_msg or comb)
        if msg_en:
            h_en = parse_product_filters_from_text(msg_en)
            for k in _FILTER_KEYS:
                if h.get(k) in (None, "") and h_en.get(k) not in (None, ""):
                    h[k] = h_en[k]
    except ImportError:
        return spec

    ai = ai_understanding or {}
    trust_ai = bool(
        ai.get("search_terms")
        or ai.get("product_requests")
        or ai.get("is_shopping")
        or spec.get("_catalog_ai")
    )

    price_turn = _user_turn_mentions_price(comb)
    ai_has_price = ai.get("max_price") is not None or ai.get("min_price") is not None
    ai_has_rating = ai.get("rating_min") is not None or ai.get("rating_max") is not None
    for key in (
        "purchase_price_max",
        "purchase_price_min",
        "rating_min",
        "rating_max",
        "size",
        "color",
        "sku",
        "pro_id",
        "sort",
    ):
        hv = h.get(key)
        if hv in (None, ""):
            continue
        if key in ("purchase_price_max", "purchase_price_min") and (
            ai_has_price or ai.get("_ai_first")
        ):
            continue
        if key in ("rating_min", "rating_max") and (ai_has_rating or ai.get("_ai_first")):
            continue
        if trust_ai and spec.get(key) not in (None, "") and not (
            price_turn and key in ("purchase_price_max", "purchase_price_min")
        ):
            continue
        spec[key] = hv

    if h.get("title_query") and not (spec.get("title_query") or "").strip():
        if not spec.get("pro_id") and not spec.get("sku"):
            spec["title_query"] = h["title_query"]

    if not spec.get("brand") and h.get("brand"):
        spec["brand"] = h["brand"]

    try:
        from utils.helpers import extract_product_id

        pid = extract_product_id(comb)
        if pid and not spec.get("pro_id"):
            spec["pro_id"] = int(pid)
            spec["title_query"] = ""
    except ImportError:
        pass

    try:
        from services.opensearch_products import sanitize_product_search_spec

        return sanitize_product_search_spec(spec)
    except ImportError:
        return spec


def log_product_search_pipeline(
    *,
    original_msg: str = "",
    msg_en: str = "",
    ai_understanding: Optional[dict] = None,
    spec: Optional[dict] = None,
    products: Optional[list] = None,
    reply_lang: str = "",
    detected_intent: str = "product_search",
    product_entities: Optional[dict] = None,
    selected_route: str = "product_ai_flow",
) -> None:
    detected = detected_product_from_turn(
        original_msg, msg_en, ai_understanding=ai_understanding, spec=spec
    )
    flt = filters_from_spec(spec or {})
    api_params = spec_to_api_params(spec or {})
    count = len(products) if products is not None else None
    lang_hint = (reply_lang or "").strip() or "auto"
    entities = dict(product_entities or {})
    if not entities and ai_understanding:
        for k in (
            "brand", "color", "size", "sku", "pro_id", "product_type",
            "purchase_price_min", "purchase_price_max", "rating_min",
        ):
            v = ai_understanding.get(k)
            if v is not None and v != "":
                entities[k] = v
    line = (
        f"[product-search] detected_intent={detected_intent} "
        f"product_entities={entities!r} selected_route={selected_route!r} "
        f"filters={flt!r} result_count={count} "
        f"detected_product={detected!r} api_params={api_params!r} lang={lang_hint}"
    )
    log_reasoning(line)
    chat_log(line)


def apply_understanding_sku_pro_id_fixes(
    understanding: Optional[dict],
    original_msg: str,
    msg_en: str = "",
) -> Optional[dict]:
    """Promote explicit SKU / pro_id from text when LLM left them in search_terms."""
    if not understanding:
        understanding = {}
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return understanding

    try:
        from services.opensearch_products import _extract_sku_from_text

        sku = _extract_sku_from_text(original_msg or "")
        if not sku and (msg_en or "").strip().lower() != (original_msg or "").strip().lower():
            sku = _extract_sku_from_text(msg_en)
        if sku and not (understanding.get("sku") or "").strip():
            understanding["sku"] = sku
            understanding["is_shopping"] = True
            understanding["action"] = understanding.get("action") or "search_products"
    except ImportError:
        pass

    try:
        from utils.helpers import extract_product_id

        pid = extract_product_id(comb)
        if pid and not understanding.get("pro_id"):
            understanding["pro_id"] = pid
            understanding["is_shopping"] = True
    except ImportError:
        pass

    st = (understanding.get("search_terms") or "").strip()
    if understanding.get("sku") and st.lower() == str(understanding["sku"]).lower():
        understanding["search_terms"] = ""
    return understanding
