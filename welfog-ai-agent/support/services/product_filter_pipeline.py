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


def extracted_filters_payload(
    spec: Optional[dict],
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """Structured filters for debug — maps to user-facing filter schema."""
    s = spec or {}
    ai = ai_understanding or {}
    out: dict[str, Any] = {
        "product_query": (s.get("title_query") or ai.get("search_terms") or "").strip(),
        "category": s.get("category_id"),
        "brand": s.get("brand") or ai.get("brand"),
        "color": s.get("color") or ai.get("color"),
        "price_range": {},
        "rating": {},
        "sku": s.get("sku") or ai.get("sku"),
        "product_id": s.get("pro_id") or ai.get("pro_id"),
        "other_filters": {},
    }
    if s.get("purchase_price_min") is not None:
        out["price_range"]["min"] = s["purchase_price_min"]
    if s.get("purchase_price_max") is not None:
        out["price_range"]["max"] = s["purchase_price_max"]
    if s.get("rating_min") is not None:
        out["rating"]["min"] = s["rating_min"]
    if s.get("rating_max") is not None:
        out["rating"]["max"] = s["rating_max"]
    for k in ("size", "brand_aliases", "mandatory_match_tokens", "product_type", "sort"):
        if s.get(k) not in (None, "", []):
            out["other_filters"][k] = s.get(k)
    if not out["price_range"]:
        out.pop("price_range")
    if not out["rating"]:
        out.pop("rating")
    if not out["other_filters"]:
        out.pop("other_filters")
    return out


def brain_search_query_is_noisy(sq: str) -> bool:
    """
    Brain search_query must not become catalog title_query when it embeds filters
    (e.g. 'mobile cover under 190', 'mobile covers rating >= 2').
    """
    low = (sq or "").strip().lower()
    if not low:
        return False
    if re.search(r"\b(under|below|above|over|upto|uptil|between)\b", low):
        return True
    if re.search(r"\b(rating|stars?|rated)\b", low):
        return True
    if re.search(r"(>=|<=|>|<)", low):
        return True
    if re.search(r"\b\d{2,6}\b", low) and re.search(
        r"\b(rs|₹|rupee|rupees|inr|budget|price|kam|sasta)\b", low
    ):
        return True
    return False


def _enrich_brain_entities_structural(
    entities: dict | None,
    original_msg: str,
    msg_en: str = "",
    *,
    brain_route: Optional[dict] = None,
) -> dict:
    """Gap-fill SKU/pro_id/price from message when brain JSON missed obvious signals."""
    e = dict(entities or {})
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    route = brain_route or {}

    if route.get("numeric_context") == "product_id" and not e.get("product_id"):
        try:
            from utils.helpers import extract_product_id

            pid = extract_product_id(comb)
            if pid:
                e["product_id"] = int(pid)
        except ImportError:
            pass

    if not e.get("product_id"):
        try:
            from utils.helpers import extract_product_id

            if re.search(r"\b(?:product\s*id|pro\s*id|pro_id)\b", comb, re.I):
                pid = extract_product_id(comb)
                if pid:
                    e["product_id"] = int(pid)
        except ImportError:
            pass

    if not e.get("sku") and re.search(r"\bsku\b", comb, re.I):
        try:
            from services.opensearch_products import _extract_sku_from_text

            sk = _extract_sku_from_text(comb)
            if sk:
                e["sku"] = sk
        except ImportError:
            pass

    if e.get("price_max") is None and e.get("price_min") is None:
        try:
            from services.opensearch_products import _extract_price_bounds

            comb_low = (original_msg or comb).lower()
            pmax, pmin = _extract_price_bounds(original_msg or comb, comb_low)
            if pmax is not None:
                e["price_max"] = pmax
            if pmin is not None:
                e["price_min"] = pmin
        except ImportError:
            pass

    return e


def _structural_gap_ai_from_message(
    original_msg: str,
    msg_en: str = "",
    locked_understanding: Optional[dict] = None,
    brain_search_query: str = "",
) -> dict[str, Any]:
    """
    Structural gap-fill only — pro_id, SKU, brain entity JSON.
    No keyword/regex product parsing; product terms come from product NLU.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    ai: dict[str, Any] = {"action": "search_products", "is_shopping": True}

    try:
        ai = apply_understanding_sku_pro_id_fixes(ai, original_msg, msg_en) or ai
    except Exception:
        pass

    locked = dict(locked_understanding or {})
    for src, dst in (
        ("pro_id", "pro_id"),
        ("sku", "sku"),
        ("color", "color"),
        ("brand", "brand"),
        ("max_price", "max_price"),
        ("min_price", "min_price"),
        ("rating_min", "rating_min"),
        ("rating_max", "rating_max"),
        ("category_browse", "category_browse"),
        ("category_only_browse", "category_only_browse"),
        ("search_terms", "search_terms"),
        ("product_type", "product_type"),
        ("match_mode", "match_mode"),
        ("mandatory_match_tokens", "mandatory_match_tokens"),
        ("exclude_title_tokens", "exclude_title_tokens"),
        ("device_browse", "device_browse"),
        ("specific_accessory", "specific_accessory"),
        ("related_search_terms", "related_search_terms"),
        ("allow_related_fallback", "allow_related_fallback"),
    ):
        v = locked.get(src)
        if v not in (None, "", []):
            ai[dst] = v

    sq = (brain_search_query or locked.get("search_terms") or "").strip()
    if sq and not brain_search_query_is_noisy(sq) and not ai.get("search_terms"):
        ai["search_terms"] = sq

    try:
        from utils.helpers import extract_product_id

        pid = extract_product_id(comb)
        if pid and not ai.get("pro_id"):
            ai["pro_id"] = int(pid)
            ai["search_terms"] = ""
    except ImportError:
        pass

    if ai.get("pro_id") or ai.get("sku"):
        ai["_ai_first"] = True

    if locked.get("max_price") is not None and ai.get("max_price") is None:
        ai["max_price"] = locked["max_price"]
    if locked.get("min_price") is not None and ai.get("min_price") is None:
        ai["min_price"] = locked["min_price"]
    if locked.get("exclude_title_tokens") and not ai.get("exclude_title_tokens"):
        ai["exclude_title_tokens"] = locked["exclude_title_tokens"]
    if locked.get("mandatory_match_tokens") and not ai.get("mandatory_match_tokens"):
        ai["mandatory_match_tokens"] = locked["mandatory_match_tokens"]

    return ai


def _spec_has_catalog_signal(spec: dict) -> bool:
    if not spec:
        return False
    return bool(
        (spec.get("title_query") or "").strip()
        or spec.get("pro_id")
        or spec.get("sku")
        or spec.get("category_id")
        or spec.get("color")
        or spec.get("brand")
        or spec.get("purchase_price_max") is not None
        or spec.get("purchase_price_min") is not None
        or spec.get("rating_min") is not None
        or spec.get("rating_max") is not None
    )


def _locked_catalog_spec_ready(
    locked_understanding: Optional[dict],
    ai_route: Optional[dict],
    *,
    brain_search_query: str = "",
) -> bool:
    """True when AI already extracted product filters — skip duplicate product NLU LLM."""
    route = ai_route or {}
    if route.get("_product_catalog_locked"):
        entities = route.get("_product_entities") or {}
        if isinstance(entities, dict) and entities:
            return True
        locked = dict(locked_understanding or {})
        if locked.get("pro_id") or locked.get("sku"):
            return True
        if (locked.get("search_terms") or "").strip():
            return True
        if (route.get("search_query") or brain_search_query or "").strip():
            return True
    locked = dict(locked_understanding or {})
    if locked.get("_ai_first") and (
        locked.get("search_terms")
        or locked.get("pro_id")
        or locked.get("sku")
        or locked.get("brand")
        or locked.get("color")
        or locked.get("max_price") is not None
        or locked.get("min_price") is not None
        or locked.get("rating_min") is not None
    ):
        return True
    return False


def _build_catalog_spec_from_locked_entities(
    original_msg: str,
    msg_en: str,
    *,
    gap: dict[str, Any],
    ctx=None,
    ai_route: Optional[dict] = None,
    locked_understanding: Optional[dict] = None,
    brain_search_query: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map locked brain / product-AI-first entities → OpenSearch spec (no NLU LLM)."""
    from services.opensearch_products import build_catalog_search_spec

    merge_ai = dict(gap)
    locked = dict(locked_understanding or {})
    for k, v in locked.items():
        if v not in (None, "", []) and k not in merge_ai:
            merge_ai[k] = v

    entities = dict((ai_route or {}).get("_product_entities") or {})
    sq = (
        (brain_search_query or "").strip()
        or ((ai_route or {}).get("search_query") or "").strip()
    )
    try:
        from services.product_catalog_resolver import entities_to_understanding

        eu = entities_to_understanding(
            entities,
            search_query=sq,
            original_msg=original_msg or msg_en,
        )
        if eu:
            for k, v in eu.items():
                if v not in (None, "", []) and k not in merge_ai:
                    merge_ai[k] = v
    except ImportError:
        pass

    if sq and not merge_ai.get("search_terms") and not merge_ai.get("pro_id"):
        try:
            if not brain_search_query_is_noisy(sq):
                merge_ai["search_terms"] = sq
        except Exception:
            merge_ai["search_terms"] = sq

    try:
        merge_ai = apply_understanding_sku_pro_id_fixes(
            merge_ai, original_msg, msg_en
        ) or merge_ai
    except Exception:
        pass

    merge_ai["_ai_first"] = True
    merge_ai["is_shopping"] = True
    merge_ai.setdefault("action", "search_products")
    spec = build_catalog_search_spec(
        original_msg,
        msg_en,
        ai=merge_ai,
        ctx=ctx,
        ai_route=ai_route,
    )
    return spec, merge_ai


def _build_related_accessory_understanding(
    locked_understanding: dict,
    entities: dict,
) -> dict | None:
    """
    Device not in catalog — retry using AI-provided related_search_terms only.
    No hardcoded product-type keyword lists.
    """
    locked = dict(locked_understanding or {})
    ent = dict(entities or {})
    if locked.get("specific_accessory"):
        return None
    if coerce_related_false(locked.get("allow_related_fallback")) or coerce_related_false(
        ent.get("allow_related_fallback")
    ):
        return None
    if not locked.get("device_browse"):
        return None
    related = (
        (locked.get("related_search_terms") or ent.get("related_search_terms") or "").strip()
    )
    if not related:
        return None
    out = dict(locked)
    out.pop("device_browse", None)
    out.pop("exclude_title_tokens", None)
    out["_related_fallback"] = True
    out["search_terms"] = related
    out["_ai_first"] = True
    return out


def coerce_related_false(val) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val is False
    return str(val).strip().lower() in ("false", "0", "no", "n")


def build_catalog_spec_for_product_turn(
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx=None,
    ai_route: Optional[dict] = None,
    locked_understanding: Optional[dict] = None,
    brain_search_query: str = "",
    allow_product_nlu_llm: bool = True,
) -> tuple[dict[str, Any], Optional[dict]]:
    """
    AI-first catalog spec — one product NLU understands filters in any language;
    structural gap-fill only for pro_id/SKU/brain entities (no keyword product lists).
    """
    from services.opensearch_products import build_catalog_search_spec

    gap = _structural_gap_ai_from_message(
        original_msg,
        msg_en,
        locked_understanding=locked_understanding,
        brain_search_query=brain_search_query,
    )
    nlu: Optional[dict] = None
    spec: dict[str, Any] = {}

    skip_nlu = not allow_product_nlu_llm or _locked_catalog_spec_ready(
        locked_understanding,
        ai_route,
        brain_search_query=brain_search_query,
    )

    if allow_product_nlu_llm and not skip_nlu:
        try:
            from services.product_query_understanding import understand_product_query

            spec, nlu = understand_product_query(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang,
            )
            if nlu:
                nlu["_ai_first"] = True
        except ImportError:
            spec = {}
    elif skip_nlu:
        spec, nlu = _build_catalog_spec_from_locked_entities(
            original_msg,
            msg_en,
            gap=gap,
            ctx=ctx,
            ai_route=ai_route,
            locked_understanding=locked_understanding,
            brain_search_query=brain_search_query,
        )
        log_reasoning(
            "Catalog spec from locked AI entities — skip duplicate product NLU LLM."
        )

    if not _spec_has_catalog_signal(spec):
        merge_ai = dict(gap)
        if nlu:
            for k, v in nlu.items():
                if v not in (None, "", []) and k not in merge_ai:
                    merge_ai[k] = v
        merge_ai["_ai_first"] = True
        spec = build_catalog_search_spec(
            original_msg,
            msg_en,
            ai=merge_ai,
            ctx=ctx,
            ai_route=ai_route,
        )
        if not nlu:
            nlu = merge_ai
    elif gap.get("pro_id") or gap.get("sku"):
        try:
            from services.catalog_spec_from_ai import merge_ai_into_catalog_spec

            gap_merge = dict(gap)
            gap_merge["_ai_first"] = True
            spec = merge_ai_into_catalog_spec(spec, gap_merge, original_msg=original_msg)
        except ImportError:
            if gap.get("pro_id"):
                spec["pro_id"] = gap["pro_id"]
                spec["title_query"] = ""

    catalog_locked = bool((ai_route or {}).get("_product_catalog_locked"))
    ai_first = bool((nlu or {}).get("_ai_first"))
    if (
        spec.get("pro_id")
        or (
            spec.get("sku")
            and not brain_search_query_is_noisy(brain_search_query or "")
        )
        or catalog_locked
        or ai_first
    ):
        spec["_ai_single_pass"] = True
    else:
        spec.pop("_ai_single_pass", None)

    if spec.get("category_id") and not spec.get("_ai_single_pass"):
        try:
            from services.welfog_api import (
                _user_explicitly_browses_category,
                query_should_use_category_id_only,
            )
            from utils.helpers import _text_requests_category_product_browse

            comb_cat = f"{original_msg or ''} {msg_en or ''}".strip()
            category_browse = (
                _text_requests_category_product_browse(comb_cat, ctx=ctx)
                or _user_explicitly_browses_category(comb_cat)
                or query_should_use_category_id_only(
                    str(spec["category_id"]), comb_cat, ctx=ctx
                )
            )
            if category_browse:
                spec["title_query"] = ""
                spec.pop("brand_aliases", None)
                spec.pop("brand_name_match_only", None)
        except ImportError:
            pass

    return spec, nlu or gap


def apply_catalog_result_filters(products: list, spec: dict) -> list:
    """Post-filter + rank OpenSearch hits to match explicit user filters."""
    if not products or not spec:
        return products or []
    from services.opensearch_products import (
        apply_catalog_post_filters,
        filter_products_by_purchase_price,
        filter_products_by_rating_range,
        filter_products_by_requested_color,
        _post_filter_mode_for_spec,
    )

    before = len(products)
    out = list(products)

    if spec.get("color"):
        color_filtered = filter_products_by_requested_color(out, spec["color"])
        if color_filtered:
            out = color_filtered

    out = apply_catalog_post_filters(
        out,
        spec,
        post_filter_mode=_post_filter_mode_for_spec(spec),
    )
    out = filter_products_by_purchase_price(out, spec)
    out = filter_products_by_rating_range(out, spec)

    removed = before - len(out)
    if removed > 0:
        log_reasoning(
            f"[product-search] post_filter removed={removed} "
            f"kept={len(out)} reason=filter_mismatch"
        )
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
        params["category"] = spec["category_id"]
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


def resolve_category_browse_into_spec(
    spec: dict[str, Any],
    original_msg: str,
    msg_en: str = "",
    *,
    ctx=None,
) -> dict[str, Any]:
    """Map category browse messages → category_id (API nav), not a vague title_query."""
    if not spec:
        return spec
    if spec.get("_ai_single_pass"):
        return spec
    if spec.get("pro_id") or spec.get("sku"):
        return spec
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return spec
    try:
        from services.welfog_api import (
            _message_has_product_search_filters,
            _user_explicitly_browses_category,
            get_category_id_from_text,
            resolve_category_product_browse_route,
        )
        from utils.helpers import _text_requests_category_product_browse
    except ImportError:
        return spec

    if _message_has_product_search_filters(comb):
        spec = dict(spec or {})
        spec.pop("category_id", None)
        return spec

    browses = _text_requests_category_product_browse(comb, ctx=ctx) or _user_explicitly_browses_category(
        comb
    )
    route = resolve_category_product_browse_route(comb, ctx=ctx) if browses else None
    cid = None
    sq = ""
    if route:
        cid, sq = route
    elif browses:
        resolved = get_category_id_from_text(comb, ctx=ctx)
        if resolved:
            cid, sq = str(resolved), ""

    if not cid:
        spec = dict(spec or {})
        if spec.get("category_id"):
            try:
                from services.welfog_api import query_should_use_category_id_only

                if query_should_use_category_id_only(spec["category_id"], comb, ctx=ctx):
                    spec["title_query"] = ""
                    spec.pop("brand", None)
                    spec.pop("brand_aliases", None)
                    spec.pop("brand_name_match_only", None)
            except ImportError:
                pass
        return spec

    spec = dict(spec)
    spec["category_id"] = int(cid)
    try:
        from services.welfog_api import query_should_use_category_id_only

        category_only = query_should_use_category_id_only(cid, comb, ctx=ctx) or not (sq or "").strip()
    except ImportError:
        category_only = browses and not (sq or "").strip()

    if category_only:
        spec["title_query"] = ""
    elif sq:
        spec["title_query"] = sq.strip()
    elif browses:
        spec["title_query"] = ""
    return spec


def finalize_catalog_spec_for_api(
    spec: dict[str, Any],
    original_msg: str,
    msg_en: str = "",
    *,
    ai_understanding: Optional[dict] = None,
    ctx=None,
) -> dict[str, Any]:
    """
    Last mile before OpenSearch/REST: only clean product keywords + explicit filters.
    Never send full user sentences or conversational filler as title_query.
    """
    if not spec:
        spec = {}
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    ai = ai_understanding or {}
    if not (ai.get("_ai_first") or spec.get("_ai_single_pass")):
        spec = enrich_catalog_spec_from_user_turn(
            spec, original_msg, msg_en, ai_understanding=ai_understanding
        )
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

        if not spec.get("_ai_single_pass"):
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

    spec = resolve_category_browse_into_spec(spec, original_msg, msg_en, ctx=ctx)

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
    Legacy gap-fill hook — skipped when product NLU already produced the spec.
    """
    if not spec:
        spec = {}
    ai = ai_understanding or {}
    if ai.get("_ai_first") or spec.get("_ai_single_pass") or spec.get("_catalog_ai"):
        return spec
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
    total_results: Optional[int] = None,
    removed_reason: str = "",
) -> None:
    detected = detected_product_from_turn(
        original_msg, msg_en, ai_understanding=ai_understanding, spec=spec
    )
    extracted = extracted_filters_payload(spec, ai_understanding)
    flt = filters_from_spec(spec or {})
    api_params = spec_to_api_params(spec or {})
    filtered_count = len(products) if products is not None else None
    lang_hint = (reply_lang or "").strip() or "auto"
    entities = dict(product_entities or {})
    if not entities and ai_understanding:
        for k in (
            "brand", "color", "size", "sku", "pro_id", "product_type", "product_name",
            "purchase_price_min", "purchase_price_max", "rating_min", "category", "category_id",
        ):
            v = ai_understanding.get(k)
            if v is not None and v != "":
                entities[k] = v
    if spec:
        for k in (
            "title_query", "brand", "color", "size", "sku", "pro_id", "category_id",
            "purchase_price_min", "purchase_price_max", "rating_min", "product_type",
        ):
            v = spec.get(k)
            if v is not None and v != "" and k not in entities:
                entities[k] = v
    try:
        from services.opensearch_products import is_opensearch_configured as _os_ok
    except ImportError:
        def _os_ok() -> bool:
            return False
    selected_api = "opensearch_catalog"
    if spec and spec.get("_category_only_browse"):
        selected_api = "category_browse"
    elif spec and spec.get("sku"):
        selected_api = "sku_lookup"
    elif not _os_ok():
        selected_api = "rest_catalog"
    debug_line = (
        f"[product-search] detected_entities={entities!r} "
        f"selected_existing_api={selected_api!r} "
        f"request_payload={api_params!r} filters_sent={flt!r} "
        f"result_count={filtered_count}"
    )
    line = (
        f"[product-search] original_query={original_msg[:120]!r} "
        f"extracted_filters={extracted!r} open_search_params={api_params!r} "
        f"total_results={total_results} filtered_results={filtered_count} "
        f"removed_reason={removed_reason!r} "
        f"detected_intent={detected_intent} selected_route={selected_route!r} "
        f"filters={flt!r} detected_product={detected!r} lang={lang_hint}"
    )
    log_reasoning(debug_line)
    chat_log(debug_line)
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
        from services.catalog_spec_semantics import resolve_is_sku_label_turn

        label_mode, label_named = resolve_is_sku_label_turn(comb)
    except ImportError:
        label_mode, label_named = "", ""
    if label_mode == "sku" and label_named:
        understanding["sku"] = label_named
        understanding["search_terms"] = ""
        understanding.pop("brand", None)
        understanding.pop("color", None)
        understanding.pop("mandatory_match_tokens", None)
        understanding["is_shopping"] = True
        understanding["action"] = understanding.get("action") or "search_products"
    elif label_mode == "title" and label_named:
        understanding["search_terms"] = label_named
        understanding.pop("sku", None)
        understanding["match_mode"] = "strict"
        understanding["is_shopping"] = True
        understanding["action"] = understanding.get("action") or "search_products"
        tokens = [
            t.lower()
            for t in re.findall(r"[a-z0-9]{2,}", label_named.lower())
            if t not in ("back", "cover", "case", "mobile", "phone")
        ]
        mandatory = list(understanding.get("mandatory_match_tokens") or [])
        for tok in tokens[:4]:
            if tok not in mandatory:
                mandatory.append(tok)
        if mandatory:
            understanding["mandatory_match_tokens"] = mandatory

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
