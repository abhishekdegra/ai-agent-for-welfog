"""
AI-first catalog spec — Groq product JSON drives filters; regex only gaps SKU/pro_id.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_CATALOG_ENTITY_KEYS = (
    "product_name",
    "brand",
    "color",
    "size",
    "sku",
    "pro_id",
    "product_id",
    "category",
    "model",
    "category_id",
    "product_type",
    "search_terms",
)
_CATALOG_NUMERIC_KEYS = (
    "price_min",
    "price_max",
    "min_price",
    "max_price",
    "rating_min",
    "rating_max",
    "purchase_price_min",
    "purchase_price_max",
)


def coerce_catalog_entity_map(val) -> dict:
    """Brain may return product_entities as dict or single-element list — normalize safely."""
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict):
                return dict(item)
    return {}


def _dict_has_catalog_signal(d: dict | None) -> bool:
    if not d:
        return False
    for k in _CATALOG_ENTITY_KEYS:
        v = d.get(k)
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, str) and not str(v).strip():
            continue
        return True
    for k in _CATALOG_NUMERIC_KEYS:
        if d.get(k) is not None:
            return True
    return False


def catalog_has_search_substance(
    text: str = "",
    *,
    entities: dict | None = None,
    understanding: dict | None = None,
    ai_route: dict | None = None,
) -> bool:
    """
    True when brain JSON or structural ids give enough signal to search.
    No language-specific phrase lists — trusts AI extraction + SKU/pro_id patterns.
    """
    route = ai_route if isinstance(ai_route, dict) else {}
    pe = coerce_catalog_entity_map(route.get("_product_entities"))
    if not pe:
        pe = coerce_catalog_entity_map(route.get("product_entities"))
    merged_ent = {**pe, **coerce_catalog_entity_map(entities)}

    if _dict_has_catalog_signal(merged_ent) or _dict_has_catalog_signal(
        coerce_catalog_entity_map(understanding)
    ):
        return True

    sq = (
        (route.get("search_query") or "")
        or ((understanding or {}).get("search_terms") or "")
    ).strip()
    if sq:
        try:
            from services.product_filter_pipeline import brain_search_query_is_noisy

            if not brain_search_query_is_noisy(sq):
                return True
        except ImportError:
            return True

    comb = (text or "").strip()
    if comb:
        try:
            from utils.helpers import extract_product_id
            from services.opensearch_products import _extract_sku_from_text

            if extract_product_id(comb) or _extract_sku_from_text(comb):
                return True
        except ImportError:
            pass
    return False


def catalog_title_unusable(
    candidate: str = "",
    *,
    entities: dict | None = None,
    understanding: dict | None = None,
    ai_route: dict | None = None,
) -> bool:
    """
    Reject a string as OpenSearch title_query when AI already has catalog entities,
    or when the candidate is structurally non-catalog (filters embedded, id-meta, empty).
    """
    t = (candidate or "").strip()
    route = ai_route if isinstance(ai_route, dict) else {}

    try:
        from services.product_filter_pipeline import brain_search_query_is_noisy
    except ImportError:
        brain_search_query_is_noisy = lambda _s: False  # noqa: E731

    if t and brain_search_query_is_noisy(t):
        return True
    if t and re.search(r"\bdetails\s+of\b", t, re.I) and re.search(
        r"\b(id|sku|product)\b", t, re.I
    ):
        return True

    pe = coerce_catalog_entity_map(route.get("_product_entities"))
    if not pe:
        pe = coerce_catalog_entity_map(route.get("product_entities"))
    ent = {**pe, **coerce_catalog_entity_map(entities)}
    ai_name = (ent.get("product_name") or "").strip()
    if t and ai_name and ai_name.lower() != t.lower():
        g_tokens = {w for w in re.findall(r"[\w]+", ai_name.lower()) if len(w) >= 3}
        c_tokens = {w for w in re.findall(r"[\w]+", t.lower()) if len(w) >= 3}
        if g_tokens and c_tokens and not (g_tokens & c_tokens):
            return True

    ent_without_title = {
        k: v for k, v in ent.items() if k not in ("product_name", "search_terms")
    }

    if catalog_has_search_substance(
        "",
        entities=ent_without_title,
        understanding=understanding,
        ai_route=route,
    ):
        if not t:
            return True
        good = (ent.get("product_name") or (understanding or {}).get("search_terms") or "").strip()
        if good and good.lower() != t.lower():
            g, c = good.lower(), t.lower()
            if g not in c and c not in g:
                g_tokens = {w for w in re.findall(r"[\w]+", g) if len(w) >= 3}
                c_tokens = {w for w in re.findall(r"[\w]+", c) if len(w) >= 3}
                if g_tokens and c_tokens and not (g_tokens & c_tokens):
                    return True
        return False

    if not t or len(t) < 2:
        return True
    if route.get("continue_previous_topic"):
        return True
    words = re.findall(r"[\w]+", t, re.UNICODE)
    if len(words) <= 2:
        return True
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(t):
            return True
    except ImportError:
        pass
    return False


def should_merge_session_catalog(
    understanding: dict | None,
    ai_route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """Reuse last browse spec when brain says continue OR product turn lacks catalog signal."""
    route = ai_route if isinstance(ai_route, dict) else {}
    if route.get("continue_previous_topic"):
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent != "product" and channel != "catalog":
        return False
    return not catalog_has_search_substance(
        f"{original_msg or ''} {msg_en or ''}".strip(),
        entities=coerce_catalog_entity_map(
            route.get("_product_entities") or route.get("product_entities")
        ),
        understanding=understanding,
        ai_route=route,
    )


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


def _is_sku_named_label(named: str) -> bool:
    """True when the token before 'is sku' is a warehouse/catalog code, not a product title."""
    tok = (named or "").strip(" \"'")
    if len(tok) < 4:
        return False
    try:
        from services.opensearch_products import (
            _looks_like_warehouse_sku,
            _sku_token_acceptable,
        )

        return bool(
            _looks_like_warehouse_sku(tok)
            or _sku_token_acceptable(tok, explicit_sku_mention=True)
        )
    except ImportError:
        return bool(re.search(r"[\-_]", tok) and re.search(r"\d", tok))


def resolve_is_sku_label_turn(text: str) -> tuple[str, str]:
    """
  For '{name} is sku' turns — return ('sku', code) or ('title', name) or ('', '').
    """
    m = re.search(
        r"^(.+?)\s+is\s+sku(?:\s+ka)?\b",
        (text or "").strip(),
        re.IGNORECASE,
    )
    if not m:
        return "", ""
    named = m.group(1).strip(" \"'")
    if _is_sku_named_label(named):
        return "sku", named
    return "title", named


def user_labels_product_title_as_sku(text: str) -> bool:
    """'Samsung S22 Back Cover is sku ka product' — product title, not a warehouse code."""
    mode, _ = resolve_is_sku_label_turn(text or "")
    return mode == "title"


def user_mentions_sku_this_turn(text: str) -> bool:
    """SKU filter when user said SKU/code or pasted a warehouse-style code (not prose words)."""
    low = (text or "").lower()
    if user_labels_product_title_as_sku(text or ""):
        return False
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


_ACCESSORY_RE = re.compile(
    r"\b(cover|covers|case|cases|bumper|backcover|back\s*cover|tempered|protector)\b",
    re.I,
)
_ONLY_PRODUCT_RE = re.compile(
    r"\b(?:only|sirf|bas)\s+(?:the\s+)?([a-z0-9][a-z0-9\s\-]{0,40}?)(?:\s+(?:chahiye|chahie|chaiye|want|wanted|need|needed))?\b",
    re.I,
)
_NEGATE_ACCESSORY_RE = re.compile(
    r"(?:"
    r"\b(?:cover|case|covers|cases)\s+(?:nahi|nai|nahin|no|not)\b|"
    r"\b(?:nahi|nai|nahin|not|no)\s+(?:\w+\s+){0,4}(?:cover|case|covers|cases)\b|"
    r"\bwithout\s+(?:cover|case)\b|"
    r"\bnot\s+(?:a\s+)?(?:cover|case)\b"
    r")",
    re.I,
)


def _message_mentions_accessory(text: str) -> bool:
    return bool(_ACCESSORY_RE.search(text or ""))


def _user_negates_accessory_search(text: str) -> bool:
    return bool(_NEGATE_ACCESSORY_RE.search(text or ""))


def _strip_accessory_tokens_from_terms(terms: str) -> str:
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (terms or "").lower())
    drop = {
        "cover", "covers", "case", "cases", "bumper", "backcover", "back",
        "mobile", "phone", "tempered", "protector", "glass",
    }
    kept = [w for w in words if w not in drop]
    return " ".join(kept).strip()


def _only_product_phrase_from_message(text: str) -> str:
    m = _ONLY_PRODUCT_RE.search(text or "")
    if not m:
        return ""
    phrase = re.sub(r"\s+", " ", (m.group(1) or "").strip().lower())
    if _message_mentions_accessory(phrase):
        return ""
    return _strip_accessory_tokens_from_terms(phrase)


def align_catalog_terms_to_user_message(
    ai: dict[str, Any],
    original_msg: str = "",
    msg_en: str = "",
) -> dict[str, Any]:
    """
    Never search accessory words the customer did not ask for (e.g. iphone → not iphone cover).
    """
    out = dict(ai)
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return out

    user_wants_accessory = _message_mentions_accessory(comb)
    user_negates_accessory = _user_negates_accessory_search(comb)
    only_phrase = _only_product_phrase_from_message(comb)

    if user_negates_accessory or only_phrase:
        out["allow_related_fallback"] = False
        out["specific_accessory"] = True
        out.pop("device_browse", None)
        out.pop("related_search_terms", None)
        out.pop("exclude_title_tokens", None)
        if only_phrase:
            out["search_terms"] = only_phrase
        elif user_negates_accessory:
            st = (out.get("search_terms") or "").strip()
            if st:
                cleaned = _strip_accessory_tokens_from_terms(st)
                if cleaned:
                    out["search_terms"] = cleaned
        ptype = (out.get("product_type") or "").strip().lower()
        if ptype in ("cover", "case", "cases", "covers", "bumper"):
            out.pop("product_type", None)
        mmt = [
            t
            for t in (out.get("mandatory_match_tokens") or [])
            if str(t).strip().lower() not in ("cover", "covers", "case", "cases", "bumper")
        ]
        if mmt:
            out["mandatory_match_tokens"] = mmt[:4]
        else:
            out.pop("mandatory_match_tokens", None)
        return out

    if not user_wants_accessory:
        st = (out.get("search_terms") or "").strip()
        if st and _ACCESSORY_RE.search(st):
            cleaned = _strip_accessory_tokens_from_terms(st)
            if cleaned:
                out["search_terms"] = cleaned
            elif not _ACCESSORY_RE.search(comb):
                out["search_terms"] = st
        related = (out.get("related_search_terms") or "").strip()
        if related and _ACCESSORY_RE.search(related) and not user_wants_accessory:
            out.pop("related_search_terms", None)
            out["allow_related_fallback"] = False
        ptype = (out.get("product_type") or "").strip().lower()
        if ptype in ("cover", "case", "cases", "covers", "bumper") and not user_wants_accessory:
            out.pop("product_type", None)
        mmt = [
            t
            for t in (out.get("mandatory_match_tokens") or [])
            if str(t).strip().lower() not in ("cover", "covers", "case", "cases", "bumper")
        ]
        if mmt:
            out["mandatory_match_tokens"] = mmt[:4]
        elif out.get("mandatory_match_tokens"):
            out.pop("mandatory_match_tokens", None)

    return out


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
    label_mode, label_named = resolve_is_sku_label_turn(comb)
    if label_mode == "sku":
        out["sku"] = sku or label_named
        out["search_terms"] = ""
        out.pop("brand", None)
        out.pop("brand_aliases", None)
        out.pop("color", None)
        out.pop("model", None)
        out.pop("product_type", None)
        out.pop("mandatory_match_tokens", None)
        sku = str(out.get("sku") or "").strip()
    elif label_mode == "title":
        out.pop("sku", None)
        out["search_terms"] = label_named
        sku = ""
    if sku:
        if not user_mentions_sku_this_turn(comb):
            out.pop("sku", None)
        elif not (
            _looks_like_warehouse_sku(sku)
            or _is_valid_sku_token(sku, explicit_sku_mention=True)
        ):
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
                polish_search_terms,
                scrub_conversational_tail_from_terms,
            )

            cleaned = polish_search_terms(
                scrub_conversational_tail_from_terms(st), original_msg
            )
            if cleaned and not catalog_title_unusable(cleaned, understanding=out):
                out["search_terms"] = cleaned
            elif catalog_title_unusable(st, understanding=out):
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
    return align_catalog_terms_to_user_message(out, original_msg, msg_en)


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

    if isinstance(ai_understanding, dict) and ai_understanding.get("_ai_first"):
        spec["_ai_single_pass"] = True
    spec.pop("_catalog_ai", None)
    return spec
