"""
Map Groq product-search JSON → OpenSearch/API spec. No per-brand hardcoding in Python.
"""
from __future__ import annotations

import re
from typing import Any, Optional


def _norm_list(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        val = [val]
    out = []
    seen: set[str] = set()
    for item in val:
        s = re.sub(r"\s+", " ", str(item or "").strip().lower())
        if len(s) < 2 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def merge_ai_into_catalog_spec(
    spec: dict[str, Any],   
    ai: Optional[dict[str, Any]],
    *,
    original_msg: str = "",
) -> dict[str, Any]:
    """AI fields are authoritative for brand, aliases, strict tokens, match_mode."""
    from services.product_query_understanding import polish_search_terms, sanitize_llm_color

    spec = dict(spec or {})
    if not ai:
        return spec

    try:
        from services.catalog_spec_semantics import resolve_is_sku_label_turn

        label_mode, label_named = resolve_is_sku_label_turn(original_msg or "")
    except ImportError:
        label_mode, label_named = "", ""
    if label_mode == "sku":
        ai = dict(ai)
        ai["sku"] = (ai.get("sku") or label_named or "").strip()
        ai["search_terms"] = ""
        ai.pop("brand", None)
        ai.pop("color", None)
        ai.pop("mandatory_match_tokens", None)
    elif label_mode == "title":
        ai = dict(ai)
        ai["search_terms"] = label_named
        ai.pop("sku", None)
        ai["match_mode"] = "strict"
        mandatory = list(ai.get("mandatory_match_tokens") or [])
        for tok in re.findall(r"[a-z0-9]{2,}", label_named.lower()):
            if tok in ("back", "cover", "case", "mobile", "phone"):
                continue
            if tok not in mandatory:
                mandatory.append(tok)
        if mandatory:
            ai["mandatory_match_tokens"] = mandatory

    terms = polish_search_terms((ai.get("search_terms") or "").strip(), original_msg)
    if terms:
        spec["title_query"] = terms

    skip_cat_lookup = bool(spec.get("_ai_single_pass") or ai.get("_ai_first"))
    if terms and not skip_cat_lookup:
        try:
            from services.welfog_api import get_category_id_from_text

            cat_id = spec.get("category_id")
            if cat_id and get_category_id_from_text(terms) == str(cat_id):
                spec.pop("title_query", None)
                terms = ""
        except ImportError:
            pass

    brand = (ai.get("brand") or "").strip()
    if brand:
        try:
            from services.product_catalog_resolver import sanitize_catalog_brand

            brand = sanitize_catalog_brand(
                brand,
                product_name=terms or (ai.get("search_terms") or ""),
                explicit_from_brain=True,
            )
        except ImportError:
            pass
    if brand:
        spec["brand"] = brand

    aliases = _norm_list(ai.get("brand_aliases"))
    if brand:
        bl = brand.lower()
        if bl not in aliases:
            aliases.insert(0, bl)
        for part in re.findall(r"[a-z0-9]+", bl):
            if len(part) >= 2 and part not in aliases:
                aliases.append(part)
    llm_color = sanitize_llm_color(
        str(ai.get("color") or ""),
        terms,
        original_msg,
        trust_llm=bool(ai.get("_ai_first")),
    )
    if llm_color:
        spec["color"] = llm_color  # AI colour beats regex/heuristic parse
    elif ai.get("color"):
        spec.pop("color", None)

    size_raw = (ai.get("size") or "").strip()
    if not size_raw and original_msg:
        from services.opensearch_products import _extract_size_from_text

        size_raw = _extract_size_from_text(original_msg)
    if size_raw:
        from services.opensearch_products import (
            _normalize_catalog_size_value,
            _strip_size_from_title_query,
        )

        spec["size"] = _normalize_catalog_size_value(size_raw)
        if spec.get("title_query"):
            spec["title_query"] = _strip_size_from_title_query(
                str(spec["title_query"]), spec["size"]
            )

    # Catalog often has brand="No Brand" — match model words in title, never catalogue colours.
    try:
        from services.opensearch_products import _COLOR_NAME_TOKENS, _strip_color_from_title_query

        color_tokens = _COLOR_NAME_TOKENS
        if terms and spec.get("color"):
            stripped = _strip_color_from_title_query(terms, str(spec["color"]))
            if stripped:
                spec["title_query"] = stripped
    except ImportError:
        color_tokens = frozenset()

    if terms and not ai.get("_ai_first"):
        for tok in re.findall(r"[a-z0-9]{3,}", terms.lower()):
            if tok in ("cover", "covers", "case", "mobile", "phone", "bumper", "products", "product"):
                continue
            if color_tokens and tok in color_tokens:
                continue
            if tok.isdigit() or re.fullmatch(r"\d{2,6}", tok):
                continue
            if tok in (
                "under", "below", "above", "over", "upto", "rating", "rated", "stars", "star",
                "rs", "rupee", "rupees", "inr", "budget", "price", "kam", "sasta",
            ):
                continue
            if tok not in aliases:
                aliases.append(tok)
    aliases = [a for a in aliases if not color_tokens or a not in color_tokens]
    if brand and brand.lower() in (color_tokens or ()):
        spec.pop("brand", None)
        brand = ""
    if aliases:
        spec["brand_aliases"] = aliases
        spec["brand_name_match_only"] = True
    else:
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)

    ptype = (ai.get("product_type") or "").strip().lower()
    if ptype:
        spec["product_type"] = ptype

    mode = (ai.get("match_mode") or "").strip().lower()
    if not mode:
        mode = "strict" if brand else "universal"

    mandatory = _norm_list(ai.get("mandatory_match_tokens"))
    if ptype and ptype not in mandatory:
        mandatory.append(ptype)
    try:
        from services.opensearch_products import extract_material_tokens

        blob = f"{terms} {original_msg}"
        mats = extract_material_tokens(blob)
        if mats:
            for m in mats:
                if m not in mandatory:
                    mandatory.append(m)
            spec["material_tokens"] = mats
    except ImportError:
        pass
    try:
        from services.opensearch_products import _COLOR_NAME_TOKENS

        mandatory = [m for m in mandatory if m not in _COLOR_NAME_TOKENS]
    except ImportError:
        pass

    if mode == "universal":
        spec["title_match_strict"] = False
        if ptype:
            spec["mandatory_match_tokens"] = [ptype]
        else:
            spec.pop("mandatory_match_tokens", None)
    else:
        spec["title_match_strict"] = True
        if mandatory:
            spec["mandatory_match_tokens"] = mandatory

    try:
        from services.opensearch_products import (
            _normalize_generic_title_query,
            sanitize_product_search_spec,
        )

        _normalize_generic_title_query(spec, original_msg)
        sanitize_product_search_spec(spec)
    except ImportError:
        pass

    try:
        from services.catalog_spec_semantics import resolve_is_sku_label_turn

        label_mode, label_named = resolve_is_sku_label_turn(original_msg or "")
    except ImportError:
        label_mode, label_named = "", ""
    ai_sku = str(ai.get("sku") or "").strip()
    if label_mode == "title":
        spec.pop("sku", None)
        spec.pop("strict_sku_match", None)
    elif label_mode == "sku" or ai_sku or (
        original_msg and re.search(r"\bsku\b", original_msg, re.I)
    ):
        try:
            from services.catalog_spec_semantics import user_mentions_sku_this_turn
            from services.opensearch_products import (
                _extract_sku_from_text,
                _sku_token_acceptable,
            )

            raw_sku = (
                ai_sku
                or label_named
                or (_extract_sku_from_text(original_msg or "") or "").strip()
            )
            sku_ok = bool(
                raw_sku
                and _sku_token_acceptable(raw_sku, explicit_sku_mention=True)
                and (
                    ai_sku
                    or ai.get("_ai_first")
                    or user_mentions_sku_this_turn(original_msg or "")
                )
            )
            if sku_ok:
                spec["sku"] = raw_sku.upper() if raw_sku.isupper() else raw_sku
                spec["strict_sku_match"] = False
                spec["title_query"] = ""
                spec.pop("category_id", None)
                spec.pop("brand", None)
                spec.pop("brand_aliases", None)
                spec.pop("brand_name_match_only", None)
        except ImportError:
            if ai_sku:
                spec["sku"] = ai_sku
                spec["strict_sku_match"] = True

    exclude = _norm_list(ai.get("exclude_title_tokens"))
    if exclude:
        spec["exclude_title_tokens"] = exclude

    try:
        pid = ai.get("pro_id") or ai.get("product_id")
        if pid is not None and str(pid).strip().isdigit():
            spec["pro_id"] = int(str(pid).strip())
            spec["title_query"] = ""
            spec["strict_no_relax"] = True
            spec.pop("brand", None)
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)
    except (TypeError, ValueError):
        pass

    try:
        from services.catalog_spec_semantics import (
            user_mentions_rating_this_turn,
        )
        from services.opensearch_products import _user_mentions_price_this_turn

        allow_price = _user_mentions_price_this_turn(original_msg or "")
        allow_rating = user_mentions_rating_this_turn(original_msg or "")
    except ImportError:
        allow_price = allow_rating = True

    if allow_price or ai.get("_ai_first"):
        for src, dst in (
            ("max_price", "purchase_price_max"),
            ("min_price", "purchase_price_min"),
            ("purchase_price_max", "purchase_price_max"),
            ("purchase_price_min", "purchase_price_min"),
        ):
            if ai.get(src) is not None:
                try:
                    spec[dst] = float(ai[src])
                except (TypeError, ValueError):
                    pass

    if allow_rating or ai.get("_ai_first"):
        for key in ("rating_min", "rating_max"):
            if ai.get(key) is not None:
                try:
                    val = float(ai[key])
                    if key == "rating_min" and val <= 0:
                        val = 0.01
                    spec[key] = val
                except (TypeError, ValueError):
                    pass
        if not ai.get("_ai_first"):
            try:
                from services.catalog_spec_semantics import normalize_rating_filters_from_message

                normalize_rating_filters_from_message(original_msg or "", spec)
            except ImportError:
                pass

    spec["_catalog_ai"] = True
    if ai.get("_ai_first"):
        spec["_ai_single_pass"] = True
    if ai.get("strict_no_relax"):
        spec["strict_no_relax"] = True
    if ai.get("strict_sku_match"):
        spec["strict_sku_match"] = True
    if ai.get("purchase_price_min") is not None and spec.get("purchase_price_min") is None:
        try:
            spec["purchase_price_min"] = float(ai["purchase_price_min"])
        except (TypeError, ValueError):
            pass
    if ai.get("purchase_price_max") is not None and spec.get("purchase_price_max") is None:
        try:
            spec["purchase_price_max"] = float(ai["purchase_price_max"])
        except (TypeError, ValueError):
            pass

    cat_browse = (ai.get("category_browse") or "").strip()
    if ai.get("category_id") is not None and not spec.get("category_id"):
        try:
            spec["category_id"] = int(ai["category_id"])
        except (TypeError, ValueError):
            pass
    if cat_browse:
        try:
            from services.welfog_api import (
                get_category_id_from_text,
                query_should_use_category_id_only,
            )

            cid = get_category_id_from_text(cat_browse)
            if cid:
                spec["category_id"] = int(cid)
                msg_for_gate = original_msg or cat_browse
                category_only = bool(ai.get("category_only_browse")) or query_should_use_category_id_only(
                    cid, msg_for_gate, ctx=None
                )
                # search_terms that are just the department name are NOT a product title.
                st = (
                    (ai.get("search_terms") or ai.get("product_name") or ai.get("product_type") or "")
                ).strip().lower()
                cat_norm = re.sub(r"[^a-z0-9]+", " ", cat_browse.lower()).strip()
                st_norm = re.sub(r"[^a-z0-9]+", " ", st).strip()
                st_is_dept = bool(st_norm) and (
                    st_norm == cat_norm
                    or st_norm in cat_norm
                    or cat_norm in st_norm
                )
                if category_only or st_is_dept:
                    spec["title_query"] = ""
                    spec["_category_only_browse"] = True
                    spec.pop("brand", None)
                    spec.pop("brand_aliases", None)
                    spec.pop("brand_name_match_only", None)
                elif not (spec.get("title_query") or "").strip() and st and not st_is_dept:
                    from services.product_query_understanding import polish_search_terms

                    spec["title_query"] = polish_search_terms(st, original_msg)
        except (TypeError, ValueError, ImportError):
            pass

    if spec.get("category_id") and original_msg:
        try:
            from services.welfog_api import query_should_use_category_id_only

            if query_should_use_category_id_only(
                spec["category_id"], original_msg, ctx=None
            ):
                spec["title_query"] = ""
                spec["_category_only_browse"] = True
                spec.pop("brand", None)
                spec.pop("brand_aliases", None)
                spec.pop("brand_name_match_only", None)
        except ImportError:
            pass

    try:
        from services.opensearch_products import _scrub_price_filters_on_rating_turn
        from services.catalog_spec_semantics import (
            ai_set_price_filter,
            reconcile_ai_first_catalog_spec,
        )

        _scrub_price_filters_on_rating_turn(spec, original_msg or "")
        from services.catalog_spec_semantics import enforce_explicit_user_filters_only

        spec = enforce_explicit_user_filters_only(
            spec, original_msg or "", "", ai_understanding=ai
        )
        spec = reconcile_ai_first_catalog_spec(
            spec, original_msg or "", "", ai_understanding=ai
        )
    except ImportError:
        pass

    return spec
