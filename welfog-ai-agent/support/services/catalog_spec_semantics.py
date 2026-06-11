"""
AI-first catalog spec — Groq product JSON drives filters; regex only gaps SKU/pro_id.
"""
from __future__ import annotations

import re
from typing import Any, Optional


def ai_understanding_has_shopping(ai: Optional[dict]) -> bool:
    if not ai:
        return False
    if ai.get("is_shopping") is False:
        return False
    if (ai.get("search_terms") or "").strip():
        return True
    if ai.get("product_requests"):
        return True
    if ai.get("sku") or ai.get("pro_id"):
        return True
    if ai.get("action") == "search_products":
        return True
    return bool(ai.get("color") or ai.get("brand") or ai.get("product_type"))


def ai_set_price_filter(ai: Optional[dict]) -> bool:
    if not ai:
        return False
    return ai.get("max_price") is not None or ai.get("min_price") is not None


def ai_set_rating_filter(ai: Optional[dict]) -> bool:
    if not ai:
        return False
    return ai.get("rating_min") is not None or ai.get("rating_max") is not None


def user_mentions_sku_this_turn(text: str) -> bool:
    """SKU filter when user said SKU/code or pasted a warehouse-style code (not prose words)."""
    low = (text or "").lower()
    if re.search(r"\bsku\b", low):
        return True
    if re.search(r"\b(?:product|item|warehouse)\s+code\b", low):
        return True
    if re.search(r"\bsku\s*[:=\-#]", low):
        return True
    try:
        from services.opensearch_products import (
            _is_valid_sku_token,
            _looks_like_warehouse_sku,
        )

        for m in re.finditer(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{4,80}\b", text or ""):
            tok = m.group(0).strip()
            if _looks_like_warehouse_sku(tok) and _is_valid_sku_token(tok):
                return True
    except ImportError:
        pass
    return False


def scrub_ai_product_understanding(
    ai: Optional[dict],
    original_msg: str = "",
    msg_en: str = "",
) -> Optional[dict]:
    """
    Clean Groq product JSON before catalog merge — drop hallucinated sku/price/rating.
    Works for any language; does not rely on growing keyword lists.
    """
    if not ai or not isinstance(ai, dict):
        return ai
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    out = dict(ai)
    try:
        from services.opensearch_products import (
            _is_valid_sku_token,
            _looks_like_warehouse_sku,
            _user_mentions_price_this_turn,
        )
    except ImportError:
        return out

    sku = str(out.get("sku") or "").strip()
    if sku:
        if not user_mentions_sku_this_turn(comb):
            out.pop("sku", None)
        elif not _looks_like_warehouse_sku(sku) or not _is_valid_sku_token(sku):
            out.pop("sku", None)

    if not user_mentions_rating_this_turn(comb):
        out.pop("rating_min", None)
        out.pop("rating_max", None)
    else:
        spec_probe: dict[str, Any] = {
            "rating_min": out.get("rating_min"),
            "rating_max": out.get("rating_max"),
        }
        normalize_rating_filters_from_message(comb, spec_probe)
        if spec_probe.get("rating_min") is not None:
            out["rating_min"] = spec_probe["rating_min"]
        else:
            out.pop("rating_min", None)
        if spec_probe.get("rating_max") is not None:
            out["rating_max"] = spec_probe["rating_max"]
        else:
            out.pop("rating_max", None)

    if not _user_mentions_price_this_turn(comb):
        out.pop("max_price", None)
        out.pop("min_price", None)

    st = (out.get("search_terms") or "").strip()
    if st:
        try:
            from services.product_query_understanding import (
                is_noisy_search_query,
                polish_search_terms,
                scrub_conversational_tail_from_terms,
            )

            cleaned = polish_search_terms(
                scrub_conversational_tail_from_terms(st), original_msg
            )
            if cleaned and not is_noisy_search_query(cleaned):
                out["search_terms"] = cleaned
            elif is_noisy_search_query(st):
                out.pop("search_terms", None)
        except ImportError:
            pass

    reqs = out.get("product_requests")
    if isinstance(reqs, list):
        cleaned = []
        for item in reqs:
            if isinstance(item, dict):
                cleaned.append(
                    scrub_ai_product_understanding(item, original_msg, msg_en) or item
                )
            else:
                cleaned.append(item)
        out["product_requests"] = cleaned
    return out


def normalize_rating_filters_from_message(comb: str, spec: dict[str, Any]) -> None:
    """Map 'rating more than 0' → rating_min=0.01; fix LLM using 1.0 instead of 0."""
    if not comb or not spec:
        return
    if not user_mentions_rating_this_turn(comb):
        return
    low = comb.lower()
    wants_above_zero = bool(
        re.search(
            r"(?:more|greater)\s+than\s+0|above\s+0|over\s+0|>\s*0|"
            r"(?:rating|stars?)\s*(?:is\s+)?(?:more|greater)\s+than\s+0|"
            r"(?:rating|stars?)\s*(?:above|over)\s+0|"
            r"0\s*se\s+(?:jyada|zyada|zada|upar)|"
            r"(?:jin|jinki|jo)\b.*\b(?:rating|stars?)\b.*\b(?:above|over|upar|jyada)\s+0\b",
            low,
            re.I,
        )
    )
    if wants_above_zero:
        spec["rating_min"] = 0.01
        return
    rmin = spec.get("rating_min")
    if rmin is None:
        return
    try:
        rv = float(rmin)
    except (TypeError, ValueError):
        return
    if rv == 1.0 and re.search(r"\b0\b", low) and re.search(
        r"(?:more|greater)\s+than|above|over|jyada|zyada", low, re.I
    ):
        spec["rating_min"] = 0.01


def rating_min_display_label(rmin: Any) -> str:
    try:
        v = float(rmin)
    except (TypeError, ValueError):
        return f"rating {rmin}+ stars"
    if v <= 0.01:
        return "rating above 0"
    if v == int(v):
        return f"rating {int(v)}+ stars"
    return f"rating {v}+ stars"


def user_mentions_rating_this_turn(text: str) -> bool:
    """True only when the user explicitly asked for a star/rating filter this turn."""
    try:
        from services.opensearch_products import _turn_mentions_rating_filter

        if _turn_mentions_rating_filter(text or ""):
            return True
    except ImportError:
        pass
    low = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:star|stars|rating|rated|rated\s+products?|best\s+rated|top\s+rated|"
            r"acha\s+rating|acchi\s+rating|achha\s+rating)\b",
            low,
            re.I,
        )
    )


def enforce_explicit_user_filters_only(
    spec: dict[str, Any],
    original_msg: str = "",
    msg_en: str = "",
    *,
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Drop price/rating filters unless the LATEST user message asked for them.
    Stops LLM/context from adding under Rs 150, rating 0.01+, rating under 5, etc.
    """
    if not spec:
        return spec
    if spec.get("_catalog_ai") or (isinstance(ai_understanding, dict) and ai_understanding.get("_ai_first")):
        return spec
    try:
        from services.opensearch_products import _user_mentions_price_this_turn
    except ImportError:
        _user_mentions_price_this_turn = lambda _t: False  # type: ignore

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    spec = dict(spec)

    if not user_mentions_rating_this_turn(comb):
        spec.pop("rating_min", None)
        spec.pop("rating_max", None)
    else:
        normalize_rating_filters_from_message(comb, spec)
        low = comb.lower()
        rmin = spec.get("rating_min")
        if rmin is not None and float(rmin) <= 0.01:
            if not re.search(
                r"(?:rating|stars?).*(?:above|over|upar|jyada|zyada|\b0\b|zero|more\s+than)|"
                r"(?:above|over|more\s+than|greater\s+than).*(?:rating|stars?|\b0\b)",
                low,
                re.I,
            ):
                spec.pop("rating_min", None)
        rmax = spec.get("rating_max")
        if rmax is not None:
            try:
                rv = float(rmax)
            except (TypeError, ValueError):
                rv = None
            if rv is not None and rv >= 4.5:
                if not re.search(
                    r"\b(?:under|below|kam|se\s+kam|se\s+niche|low)\b.*"
                    r"(?:\d(?:\.\d)?\s*)?(?:star|stars?|rating)|"
                    r"(?:rating|stars?)\s*(?:under|below|kam)",
                    low,
                    re.I,
                ):
                    spec.pop("rating_max", None)

    if not _user_mentions_price_this_turn(comb):
        for k in (
            "purchase_price_max",
            "purchase_price_min",
            "unit_price_max",
            "unit_price_min",
        ):
            spec.pop(k, None)
    else:
        pmax = spec.get("purchase_price_max")
        pmin = spec.get("purchase_price_min")
        nums_in_msg = [int(x) for x in re.findall(r"\d{2,7}", comb)]
        if pmax is not None and nums_in_msg and int(float(pmax)) not in nums_in_msg:
            if not any(abs(int(float(pmax)) - n) <= 5 for n in nums_in_msg):
                spec.pop("purchase_price_max", None)
        if pmin is not None and nums_in_msg and int(float(pmin)) not in nums_in_msg:
            if not any(abs(int(float(pmin)) - n) <= 5 for n in nums_in_msg):
                spec.pop("purchase_price_min", None)

    if ai_understanding and not user_mentions_rating_this_turn(comb):
        ai_understanding.pop("rating_min", None) if isinstance(ai_understanding, dict) else None
    return spec


def extract_gap_fill_ids_from_text(
    text: str,
    *,
    pro_id: Optional[int] = None,
) -> dict[str, Any]:
    """SKU / pro_id only — never title, price, rating, colour (LLM owns those)."""
    from services.opensearch_products import (
        _extract_first,
        _extract_sku_from_text,
        _PRO_ID_PATTERNS,
    )

    out: dict[str, Any] = {}
    if pro_id:
        out["pro_id"] = pro_id
    if user_mentions_sku_this_turn(text or ""):
        sku = _extract_sku_from_text(text or "")
        if sku:
            out["sku"] = sku
    if not out.get("pro_id"):
        pid = _extract_first(_PRO_ID_PATTERNS, (text or "").lower())
        if pid and str(pid).isdigit():
            out["pro_id"] = int(pid)
    return out


def reconcile_ai_first_catalog_spec(
    spec: dict[str, Any],
    original_msg: str = "",
    msg_en: str = "",
    *,
    ctx: Optional[dict] = None,
    ai_route: Optional[dict] = None,
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Post-merge cleanup when Groq product AI drove the spec — no regex title/price/rating parse.
    """
    from services.opensearch_products import (
        _scrub_bogus_brand_from_spec,
        _scrub_price_filters_on_rating_turn,
        _user_mentions_price_this_turn,
    )

    if not spec:
        return spec
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    spec = dict(spec)
    spec["_catalog_ai"] = True

    gap = extract_gap_fill_ids_from_text(comb)
    if gap.get("pro_id"):
        spec["pro_id"] = gap["pro_id"]
        spec["title_query"] = ""
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        spec["title_match_strict"] = False
        spec.pop("_catalog_ai", None)
        return spec
    if gap.get("sku") and not spec.get("sku"):
        spec["sku"] = gap["sku"]

    if ai_set_rating_filter(ai_understanding):
        _scrub_price_filters_on_rating_turn(spec, comb)

    _scrub_bogus_brand_from_spec(spec, comb)
    normalize_rating_filters_from_message(comb, spec)
    spec = enforce_explicit_user_filters_only(
        spec, original_msg, msg_en, ai_understanding=ai_understanding
    )

    try:
        from services.opensearch_products import scrub_filter_only_catalog_title

        scrub_filter_only_catalog_title(spec, comb)
    except ImportError:
        pass

    spec.pop("_catalog_ai", None)
    return spec
