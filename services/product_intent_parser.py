"""
Strict ecommerce NLU before OpenSearch — word-boundary extraction, no substring colour bleed.

USER MESSAGE → ProductIntentParser → validated catalog spec → OpenSearch
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

# Brands whose names contain colour substrings — never treat as colours.
_BRAND_BLOCK_COLOR = frozenset(
    {
        "redmi",
        "reddit",
        "redbull",
        "greenland",
        "blackberry",
        "blackdecker",
    }
)

# Alias hints only — AI may return ANY brand; this map does not limit detection.
_KNOWN_BRANDS: dict[str, tuple[str, list[str]]] = {
    "iphone": ("iPhone", ["iphone", "apple"]),
    "apple": ("Apple", ["apple", "iphone"]),
    "samsung": ("Samsung", ["samsung"]),
    "xiaomi": ("Xiaomi", ["xiaomi", "mi"]),
    "redmi": ("Redmi", ["redmi", "xiaomi"]),
    "oneplus": ("OnePlus", ["oneplus"]),
    "oppo": ("Oppo", ["oppo"]),
    "vivo": ("Vivo", ["vivo"]),
    "realme": ("Realme", ["realme"]),
    "poco": ("Poco", ["poco"]),
    "motorola": ("Motorola", ["motorola"]),
    "nokia": ("Nokia", ["nokia"]),
    "honor": ("Honor", ["honor"]),
    "infinix": ("Infinix", ["infinix"]),
    "lg": ("LG", ["lg"]),
    "tecno": ("Tecno", ["tecno"]),
    "itel": ("Itel", ["itel"]),
    "jio": ("Jio", ["jio"]),
    "jioo": ("Jio", ["jio"]),
}

# Fallback colour tokens when LLM unavailable — AI path accepts any catalog colour.
_COLOR_WORD_MAP: dict[str, str] = {
    "black": "Black",
    "white": "White",
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "yellow": "Yellow",
    "pink": "Pink",
    "purple": "Purple",
    "grey": "Grey",
    "gray": "Grey",
    "orange": "Orange",
    "brown": "Brown",
    "silver": "Silver",
    "gold": "Gold",
    "navy": "Navy",
    "maroon": "Maroon",
    "beige": "Beige",
    "kala": "Black",
    "safed": "White",
    "laal": "Red",
    "lal": "Red",
    "neela": "Blue",
    "hara": "Green",
    "hare": "Green",
    "haraa": "Green",
    "peela": "Yellow",
    "हरा": "Green",
    "हरे": "Green",
    "लाल": "Red",
    "काला": "Black",
    "नीला": "Blue",
}

_PRODUCT_NOUNS = frozenset(
    {
        "cover",
        "covers",
        "case",
        "mobile",
        "phone",
        "shirt",
        "bottle",
        "charger",
        "cable",
        "watch",
        "earphone",
        "headphone",
        "product",
        "products",
    }
)

_SKU_RE = re.compile(
    r"\b(?:sku\s*[:\-#]?\s*)?([A-Z]{1,4}[-_][A-Za-z0-9][A-Za-z0-9_\-]{3,60})\b",
    re.IGNORECASE,
)

_PRO_ID_RE = re.compile(
    r"\b(?:pro[_\s-]?id|product[_\s-]?id|pid)\s*[:\-#]?\s*(\d{4,12})\b",
    re.IGNORECASE,
)

_ID_PRODUCT_RE = re.compile(
    r"\b(\d{4,12})\s+(?:is\s+)?(?:product\s+)?id\b|\b(?:product\s+)?id\s+(\d{4,12})\b|"
    r"\b(\d{4,12})\s+is\s+id\s+ka\b|\bid\s+(\d{4,12})\s+ka\s+product\b|"
    r"\byeh\s+product\s+id\b.*\b(\d{4,12})\b|\b(\d{4,12})\s+.*\bproduct\s+id\b",
    re.IGNORECASE,
)

_PRICE_RANGE_RE = re.compile(
    r"\b(?:price\s+range\s+)?(\d{2,7})\s*(?:rs|₹|rupees?|inr)?\s*(?:se|to|-|–)\s*(\d{2,7})\b|"
    r"\b(?:between|from)\s+(\d{2,7})\s+(?:and|to|-|–)\s+(\d{2,7})\b|"
    r"\b(\d{2,7})\s+se\s+(\d{2,7})\s+ke\b",
    re.IGNORECASE,
)

_PRICE_UNDER_RE = re.compile(
    r"(?:under|below|less\s+than|max|upto|up\s+to|within|andar|ke\s+andar|ke\s+neeche|se\s+kam)\s*"
    r"(?:rs\.?|₹|inr)?\s*(\d{2,7})|"
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)?\s*(?:ke\s+andar|ke\s+neeche|se\s+kam|tak|wale|wali)|"
    r"\bunder\s+(\d{2,7})\b",
    re.IGNORECASE,
)

_PRICE_OVER_RE = re.compile(
    r"(?:above|over|more\s+than|min|at\s+least|se\s+upar|se\s+zyada)\s*(?:rs\.?|₹|inr)?\s*(\d{2,7})|"
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)?\s*(?:se\s+upar|se\s+zyada)",
    re.IGNORECASE,
)

_RATING_MIN_RE = re.compile(
    r"(?:rating|rated|stars?)\s*(?:above|over|min|minimum|>=?|se\s+upar)?\s*(\d(?:\.\d)?)|"
    r"(\d(?:\.\d)?)\s*(?:star|stars|rating)\s*(?:se\s+upar|above|\+)|"
    r"\b(\d(?:\.\d)?)\s*\+\s*(?:star|stars|rating)",
    re.IGNORECASE,
)

_RATING_MAX_RE = re.compile(
    r"(?:under|below|less\s+than|se\s+kam|se\s+niche)\s*(\d(?:\.\d)?)\s*(?:star|stars|rating)|"
    r"(?:rating|stars?)\s*(?:under|below|se\s+kam)\s*(\d(?:\.\d)?)",
    re.IGNORECASE,
)

_DEVICE_MODEL_RE = re.compile(
    r"\b(redmi\s+note\s+\d+\s*(?:\d?g|pro|plus|max)?|"
    r"iphone\s+\d+\s*(?:pro|max|plus)?|"
    r"galaxy\s+[a-z]\d+|samsung\s+s\d+|"
    r"oneplus\s+\d+|poco\s+[a-z0-9]+)\b",
    re.IGNORECASE,
)


@dataclass
class ProductIntentFilters:
    brand: Optional[str] = None
    color: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    rating_min: Optional[float] = None
    rating_max: Optional[float] = None
    sku: Optional[str] = None
    product_id: Optional[int] = None
    category: Optional[str] = None
    device_model: Optional[str] = None


@dataclass
class ProductIntent:
    intent: str = "product_search"
    product_query: str = ""
    filters: ProductIntentFilters = field(default_factory=ProductIntentFilters)
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_prompt: str = ""
    source: str = "parser"
    brand_aliases: list[str] = field(default_factory=list)
    mandatory_match_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["filters"] = asdict(self.filters)
        return d

    def to_understanding_dict(self) -> dict[str, Any]:
        """Shape expected by product_search_flow / build_catalog_search_spec."""
        f = self.filters
        out: dict[str, Any] = {
            "action": "search_products",
            "is_shopping": True,
            "search_terms": self.product_query,
            "color": f.color or "",
            "brand": f.brand or "",
            "brand_aliases": list(self.brand_aliases or []),
            "sku": f.sku or "",
            "pro_id": f.product_id,
            "rating_min": f.rating_min,
            "rating_max": f.rating_max,
            "min_price": f.min_price,
            "max_price": f.max_price,
            "purchase_price_min": f.min_price,
            "purchase_price_max": f.max_price,
            "device_model": f.device_model or "",
            "mandatory_match_tokens": list(self.mandatory_match_tokens or []),
            "match_mode": "strict" if self._has_strict_filters() else "universal",
            "intent_kind": self.intent,
            "confidence": self.confidence,
            "strict_no_relax": self._has_strict_filters(),
            "strict_sku_match": bool(f.sku),
        }
        if f.device_model:
            out["title_match_strict"] = True
        if f.brand and f.color:
            out["title_match_strict"] = True
        return out

    def to_catalog_spec(self) -> dict[str, Any]:
        """Direct OpenSearch spec."""
        f = self.filters
        spec: dict[str, Any] = {
            "title_query": self.product_query,
            "color": f.color,
            "brand": f.brand,
            "brand_aliases": self.brand_aliases or ([f.brand] if f.brand else []),
            "sku": f.sku,
            "pro_id": f.product_id,
            "purchase_price_min": f.min_price,
            "purchase_price_max": f.max_price,
            "rating_min": f.rating_min,
            "rating_max": f.rating_max,
            "mandatory_match_tokens": self.mandatory_match_tokens,
            "strict_no_relax": self._has_strict_filters(),
            "strict_sku_match": bool(f.sku),
        }
        if self._has_strict_filters():
            spec["title_match_strict"] = True
        if f.device_model:
            spec["device_model"] = f.device_model
            if f.device_model.lower() not in [a.lower() for a in spec.get("brand_aliases") or []]:
                spec.setdefault("brand_aliases", []).append(f.device_model)
        return {k: v for k, v in spec.items() if v not in (None, "", [])}

    def _has_strict_filters(self) -> bool:
        f = self.filters
        return bool(
            f.sku
            or f.product_id
            or f.color
            or f.brand
            or f.device_model
            or f.min_price is not None
            or f.max_price is not None
            or f.rating_min is not None
            or f.rating_max is not None
        )


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def extract_color_word_boundary(text: str) -> Optional[str]:
    """Colour from independent words only — Redmi ≠ red."""
    if not (text or "").strip():
        return None
    low = text.lower()
    for phrase, canon in (
        ("sky blue", "Sky Blue"),
        ("light blue", "Light Blue"),
        ("dark green", "Dark Green"),
    ):
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            return canon
    for word, canon in _COLOR_WORD_MAP.items():
        if word in _BRAND_BLOCK_COLOR:
            continue
        if re.search(rf"\b{re.escape(word)}\b", low):
            return canon
    return None


def extract_brand_and_aliases(text: str) -> tuple[Optional[str], list[str]]:
    low = f" {text.lower()} "
    for key, (display, aliases) in _KNOWN_BRANDS.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return display, list(aliases)
    m = re.search(
        r"\bbrand\s+(?:jioo|jio|[a-z0-9][a-z0-9\-]{1,24})\b|\b([a-z0-9][a-z0-9\-]{1,24})\s+brand\s+ke\b",
        low,
    )
    if m:
        raw = (m.group(1) or m.group(2) or "").strip()
        if raw and raw not in _PRODUCT_NOUNS:
            return raw.title(), [raw.lower()]
    return None, []


def extract_device_model(text: str) -> Optional[str]:
    m = _DEVICE_MODEL_RE.search(text or "")
    if m:
        return m.group(0).strip()
    return None


def extract_sku(text: str) -> Optional[str]:
    low = text or ""
    m = _SKU_RE.search(low)
    if m:
        tok = m.group(1).strip()
        if re.search(r"\d", tok) or "-" in tok or "_" in tok:
            return tok
    return None


def _looks_like_product_id_turn(text: str) -> bool:
    if not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    if any(
        x in tl
        for x in (
            "product id",
            "pro id",
            "id ka product",
            "yeh product id",
            "product id h",
            "iska product",
            "product dikha",
            "product dikhao",
        )
    ):
        return True
    if re.search(r"\b\d{6,12}\b", text) and re.search(
        r"\b(?:product|sku|dikha|dikhao|catalog|item)\b", tl
    ):
        return True
    return False


def extract_product_id_strict(text: str) -> Optional[int]:
    if not _looks_like_product_id_turn(text):
        return None
    try:
        from utils.helpers import extract_product_id

        pid = extract_product_id(text)
    except RecursionError:
        pid = None
    if pid:
        return int(pid)
    for pat in (
        _PRO_ID_RE,
        _ID_PRODUCT_RE,
        re.compile(r"\b(\d{6,12})\s+iska\s+product\b", re.I),
        re.compile(r"\b(\d{6,12})\s+.*\bproduct\s+dikha", re.I),
    ):
        m = pat.search(text)
        if m:
            for g in m.groups():
                if g and str(g).isdigit():
                    return int(g)
    return None


def extract_price_bounds(text: str) -> tuple[Optional[float], Optional[float]]:
    low = (text or "").lower()
    mn = mx = None
    m = _PRICE_RANGE_RE.search(low)
    if m:
        groups = [g for g in m.groups() if g and str(g).isdigit()]
        if len(groups) >= 2:
            a, b = float(groups[0]), float(groups[1])
            mn, mx = (min(a, b), max(a, b))
            return mn, mx
    m = _PRICE_UNDER_RE.search(low)
    if m:
        for g in m.groups():
            if g and str(g).isdigit():
                mx = float(g)
                break
    m = _PRICE_OVER_RE.search(low)
    if m:
        for g in m.groups():
            if g and str(g).isdigit():
                mn = float(g)
                break
    return mn, mx


def extract_rating_bounds(text: str) -> tuple[Optional[float], Optional[float]]:
    low = text or ""
    rmin = rmax = None
    m = _RATING_MIN_RE.search(low)
    if m:
        for g in m.groups():
            if g:
                rmin = float(g)
                break
    m = _RATING_MAX_RE.search(low)
    if m:
        for g in m.groups():
            if g:
                rmax = float(g)
                break
    return rmin, rmax


def _build_product_query(text: str, brand: Optional[str], device: Optional[str], color: Optional[str]) -> str:
    from services.opensearch_products import extract_color_and_product_title

    _, title = extract_color_and_product_title(text)
    parts: list[str] = []
    if device:
        parts.append(device)
    elif brand:
        parts.append(brand)
    if title:
        parts.append(title)
    elif not parts:
        for noun in _PRODUCT_NOUNS:
            if re.search(rf"\b{re.escape(noun)}\b", text.lower()):
                parts.append(noun)
                break
    q = " ".join(parts).strip()
    if color and q:
        for hue in (color, color.lower()):
            q = re.sub(rf"\b{re.escape(hue)}\b", "", q, flags=re.I).strip()
    return q or (brand or device or "").strip()


def _score_confidence(intent: ProductIntent, text: str) -> float:
    f = intent.filters
    score = 0.45
    if f.sku or f.product_id:
        score = 0.95
    elif f.brand or f.device_model:
        score += 0.2
    if f.color:
        score += 0.1
    if intent.product_query:
        score += 0.15
    if f.min_price is not None or f.max_price is not None:
        score += 0.1
    if f.rating_min is not None or f.rating_max is not None:
        score += 0.1
    if re.search(r"\b(?:dikha|dikhao|show|chahiye|products?)\b", text.lower()):
        score += 0.05
    return min(1.0, score)


def _normalize_brand_aliases(brand: str) -> list[str]:
    """Optional alias hints — does NOT limit which brands AI may return."""
    if not brand:
        return []
    low = brand.strip().lower()
    for key, (_display, aliases) in _KNOWN_BRANDS.items():
        if low == key or low in [a.lower() for a in aliases]:
            return list(aliases)
    return [low]


def validate_ai_color(ai_color: str, user_text: str, brand: str = "") -> Optional[str]:
    """
    Accept any catalog colour the AI inferred — only block known false positives.
    Redmi must not become Red; brand substrings must not become colours.
    """
    if not (ai_color or "").strip():
        return None
    canon = (ai_color or "").strip()
    low = (user_text or "").lower()
    c_low = canon.lower()

    if c_low == "red" and re.search(r"\bredmi\b", low) and not re.search(r"\bred\b", low):
        return None
    if brand and len(c_low) >= 2 and c_low in brand.lower() and c_low != brand.lower():
        return None

    if re.search(rf"\b{re.escape(c_low)}\b", low):
        return canon.title() if canon.islower() else canon

    det = extract_color_word_boundary(user_text)
    if det and det.lower() == c_low:
        return det

    try:
        from services.opensearch_products import color_hue_mentioned_in_text
        from services.welfog_api import _normalize_color

        if color_hue_mentioned_in_text(canon, user_text):
            return _normalize_color(c_low) or canon
        normed = _normalize_color(c_low)
        if normed and color_hue_mentioned_in_text(normed, user_text):
            return normed
    except ImportError:
        pass

    if re.search(r"\b(?:color|colour|rang|रंग|रंगे|रंग का)\b", low):
        return canon.title() if canon.islower() else canon
    return None


class ProductIntentParser:
    """
    AI-first pre-OpenSearch NLU (industry pattern).

    Flow: exact SKU/pro_id (deterministic) → Groq meaning JSON → validate/sanitize → OpenSearch.
    Static brand/colour lists are guardrails only — NOT the primary detector.
    """

    CONFIDENCE_CLARIFY_THRESHOLD = 0.7

    @classmethod
    def parse(
        cls,
        original_msg: str,
        msg_en: str = "",
        conversation_context: str = "",
        *,
        ai_route: Optional[dict] = None,
    ) -> ProductIntent:
        comb = _combined(original_msg, msg_en)

        sku = extract_sku(comb)
        if sku:
            intent = ProductIntent(
                intent="sku_lookup",
                filters=ProductIntentFilters(sku=sku),
                confidence=0.95,
                source="sku_exact",
            )
            cls._log_pipeline(original_msg, intent)
            return intent

        pid = extract_product_id_strict(comb)
        if pid:
            intent = ProductIntent(
                intent="product_detail",
                filters=ProductIntentFilters(product_id=pid),
                confidence=0.95,
                source="product_id_exact",
            )
            cls._log_pipeline(original_msg, intent)
            return intent

        intent = cls._parse_llm_primary(original_msg, msg_en, conversation_context)
        if intent and intent.confidence >= 0.5:
            intent = cls._validate_llm_intent(intent, comb, original_msg, msg_en)
            intent.confidence = _score_confidence(intent, comb)
        else:
            intent = cls._parse_deterministic(comb, original_msg, msg_en)
            intent.source = "deterministic_failsafe"
            intent.confidence = _score_confidence(intent, comb)
        if intent.confidence < cls.CONFIDENCE_CLARIFY_THRESHOLD and not intent.filters.sku and not intent.filters.product_id:
            has_price_or_rating = (
                intent.filters.min_price is not None
                or intent.filters.max_price is not None
                or intent.filters.rating_min is not None
                or intent.filters.rating_max is not None
            )
            if not intent.product_query and not intent.filters.brand and not has_price_or_rating:
                intent.needs_clarification = True
                intent.clarification_prompt = (
                    "Which product are you looking for — brand, type (cover, bottle), or colour?"
                )

        cls._log_pipeline(original_msg, intent)
        return intent

    @classmethod
    def _parse_deterministic(
        cls, comb: str, original_msg: str, msg_en: str
    ) -> ProductIntent:
        f = ProductIntentFilters()
        intent_kind = "product_search"

        sku = extract_sku(comb)
        if sku:
            f.sku = sku
            intent_kind = "sku_lookup"
            return ProductIntent(
                intent=intent_kind,
                product_query="",
                filters=f,
                confidence=0.95,
                source="sku_exact",
            )

        pid = extract_product_id_strict(comb)
        if pid:
            f.product_id = pid
            intent_kind = "product_detail"
            return ProductIntent(
                intent=intent_kind,
                product_query="",
                filters=f,
                confidence=0.95,
                source="product_id_exact",
            )

        brand, aliases = extract_brand_and_aliases(comb)
        f.brand = brand
        device = extract_device_model(comb)
        f.device_model = device
        f.color = extract_color_word_boundary(comb)

        pmin, pmax = extract_price_bounds(comb)
        f.min_price, f.max_price = pmin, pmax
        if pmin is not None or pmax is not None:
            intent_kind = "price_filter"

        rmin, rmax = extract_rating_bounds(comb)
        f.rating_min, f.rating_max = rmin, rmax
        if rmin is not None or rmax is not None:
            intent_kind = "rating_filter"

        product_query = _build_product_query(comb, brand, device, f.color)
        mandatory: list[str] = []
        if device:
            for tok in re.findall(r"[a-z0-9]+", device.lower()):
                if len(tok) >= 3:
                    mandatory.append(tok)
        if brand and brand.lower() not in (m.lower() for m in mandatory):
            mandatory.append(brand.lower())
        for noun in _PRODUCT_NOUNS:
            if re.search(rf"\b{re.escape(noun)}\b", comb.lower()):
                mandatory.append(noun.rstrip("s") if noun.endswith("s") else noun)
                break

        return ProductIntent(
            intent=intent_kind,
            product_query=product_query,
            filters=f,
            brand_aliases=aliases,
            mandatory_match_tokens=mandatory[:4],
            confidence=_score_confidence(
                ProductIntent(product_query=product_query, filters=f), comb
            ),
            source="deterministic",
        )

    @classmethod
    def _parse_llm_primary(
        cls,
        original_msg: str,
        msg_en: str,
        conversation_context: str,
    ) -> Optional[ProductIntent]:
        """Groq semantic JSON — any product, any colour, any language."""
        try:
            from services.product_search_flow import ai_understand_product_search
            from services.catalog_spec_semantics import scrub_ai_product_understanding

            u = ai_understand_product_search(
                original_msg.strip() or msg_en.strip(),
                conversation_context=conversation_context,
            )
            if not u:
                return None
            u = scrub_ai_product_understanding(u, original_msg, msg_en) or u
            if not u.get("is_shopping", True) or (u.get("action") or "") == "not_shopping":
                return None

            reqs = u.get("product_requests") or []
            if isinstance(reqs, list) and len(reqs) == 1:
                u = {**u, **{k: v for k, v in reqs[0].items() if v not in (None, "", [])}}

            f = ProductIntentFilters()
            brand = (u.get("brand") or "").strip() or None
            f.brand = brand
            f.color = (u.get("color") or "").strip() or None
            f.sku = (u.get("sku") or "").strip() or None
            try:
                if u.get("pro_id") is not None:
                    f.product_id = int(str(u["pro_id"]).strip())
            except (TypeError, ValueError):
                pass
            for key, attr in (
                ("min_price", "min_price"),
                ("max_price", "max_price"),
                ("rating_min", "rating_min"),
                ("rating_max", "rating_max"),
            ):
                if u.get(key) is not None:
                    try:
                        setattr(f, attr, float(u[key]))
                    except (TypeError, ValueError):
                        pass

            intent_kind = "product_search"
            if f.sku:
                intent_kind = "sku_lookup"
            elif f.product_id:
                intent_kind = "product_detail"
            elif f.rating_min is not None or f.rating_max is not None:
                intent_kind = "rating_filter"
            elif f.min_price is not None or f.max_price is not None:
                intent_kind = "price_filter"

            product_query = (u.get("search_terms") or "").strip()
            aliases = list(u.get("brand_aliases") or []) or _normalize_brand_aliases(brand or "")
            mandatory = list(u.get("mandatory_match_tokens") or [])

            conf = 0.82 if product_query or brand or f.color else 0.55
            if (u.get("reasoning") or "").strip():
                conf = min(0.95, conf + 0.05)

            return ProductIntent(
                intent=intent_kind,
                product_query=product_query,
                filters=f,
                brand_aliases=aliases,
                mandatory_match_tokens=mandatory[:4],
                confidence=conf,
                source="ai_semantic",
            )
        except Exception:
            return None

    @classmethod
    def _validate_llm_intent(
        cls,
        intent: ProductIntent,
        comb: str,
        original_msg: str,
        msg_en: str,
    ) -> ProductIntent:
        """Sanitize AI output — block colour bleed, merge numeric guards."""
        f = intent.filters
        blob = _combined(original_msg, msg_en)

        if f.color:
            validated = validate_ai_color(f.color, blob, f.brand or "")
            f.color = validated

        if not f.color:
            det = extract_color_word_boundary(blob)
            if det:
                f.color = det

        if not f.brand:
            b, al = extract_brand_and_aliases(blob)
            if b:
                f.brand = b
                intent.brand_aliases = al

        if f.min_price is None and f.max_price is None:
            pmin, pmax = extract_price_bounds(blob)
            f.min_price, f.max_price = pmin, pmax
        if f.rating_min is None and f.rating_max is None:
            rmin, rmax = extract_rating_bounds(blob)
            f.rating_min, f.rating_max = rmin, rmax

        if not f.sku:
            f.sku = extract_sku(blob) or f.sku
        if not f.product_id:
            pid = extract_product_id_strict(blob)
            if pid:
                f.product_id = pid
                intent.intent = "product_detail"

        if not intent.product_query:
            intent.product_query = _build_product_query(
                blob, f.brand, f.device_model, f.color
            )

        if f.brand and not intent.brand_aliases:
            intent.brand_aliases = _normalize_brand_aliases(f.brand)

        intent.filters = f
        intent.source = "ai_semantic+validated"
        return intent

    @classmethod
    def _log_pipeline(cls, user_query: str, intent: ProductIntent) -> None:
        payload = intent.to_dict()
        msg = (
            f"[product-nlu] USER_QUERY={user_query[:200]!r} "
            f"EXTRACTED_INTENT_JSON={json.dumps(payload, ensure_ascii=False)[:1200]}"
        )
        log_reasoning(msg)
        chat_log(msg)


def log_opensearch_query(spec: dict[str, Any], result_count: int) -> None:
    try:
        from services.opensearch_products import _build_opensearch_body

        body = _build_opensearch_body(spec, size=8, offset=0)
        q = json.dumps(body.get("query") or {}, ensure_ascii=False)[:2000]
    except Exception:
        q = str(spec)[:500]
    msg = f"[product-nlu] FINAL_OPENSEARCH_QUERY={q} RESULT_COUNT={result_count}"
    log_reasoning(msg)
    chat_log(msg)
