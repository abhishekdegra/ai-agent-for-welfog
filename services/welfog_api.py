import difflib
import os
import re
import threading
import time
from datetime import datetime, timedelta
from html import escape as html_escape
from typing import Optional

import requests

WELFOG_TRACK_API_URL = "https://welfogapi.welfog.com/api/onedelivery/welfog_track"
WELFOG_RETURN_REQUEST_API_URL = "https://welfogapi.welfog.com/api/v2/return-request/{order_id}"


def _record_api_elapsed(started: float) -> None:
    try:
        from services.chat_flow_telemetry import record_api_time

        record_api_time(time.perf_counter() - float(started))
    except ImportError:
        pass

_RETURN_REQUEST_CUSTOMER_FIELDS = (
    "order_id",
    "refund_status",
    "refund_amount",
    "message",
    "isbankvalid",
)
TRACK_PRODUCT_IMG_BASE = "https://d1f02fefkbso7w.cloudfront.net/"

_ORDER_FLOW_STEP_ORDER = (
    "placed",
    "pending",
    "confirmed",
    "processing",
    "shipped",
    "out_for_delivery",
    "delivered",
)

PH_API_BASE = "https://welfogapi.welfog.com/api/v2/purchase-history"
PH_DETAILS_API_BASE = "https://welfogapi.welfog.com/api/v2/purchase-history-details"
INVOICE_API_BASE = "https://supplierservice.welfog.com/get_invoice"
_LIVE_ORDER_API_TIMEOUT_SEC = 6


def _is_transient_network_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
        ),
    ):
        return True
    msg = str(exc).lower()
    return (
        "failed to resolve" in msg
        or "name resolution" in msg
        or "getaddrinfo failed" in msg
        or "timed out" in msg
        or "max retries exceeded" in msg
    )
PH_IMG_BASE = "https://d1f02fefkbso7w.cloudfront.net/"
PH_PER_PAGE = 10
# In-memory order list per logged-in user (same-process); grows by offset/page until API repeats or ends.
_PH_ACCUM_LOCK = threading.Lock()
_PH_ACCUM = {}  # uid -> {"rows": [...], "ts": float, "exhausted": bool}
_PH_ACCUM_TTL_SEC = 300

WL_API_TEMPLATE = "https://welfogapi.welfog.com/api/v2/wishlists/{user_id}"
WL_PER_PAGE = 8
WL_API_PAGE_SIZE = 50
_WL_CACHE_LOCK = threading.Lock()
_WL_CACHE = {}  # uid -> {"items": [...], "ts": float}
_WL_CACHE_TTL_SEC = 300

_CATALOG_SEARCH_URL = "https://welfogapi.welfog.com/api/v2/products/search"


def _catalog_search_get(params: dict, *, timeout: int = 12) -> list:
    """Raw products/search GET — returns data[] or []."""
    p = {"latitude": "", "longitude": ""}
    p.update(params or {})
    try:
        res = requests.get(_CATALOG_SEARCH_URL, params=p, timeout=timeout)
        if res.status_code != 200:
            return []
        return res.json().get("data") or []
    except Exception:
        return []


_REST_CATEGORY_BASELINE_SLUGS: frozenset[str] | None = None


def _product_slugs_from_rows(rows: list) -> frozenset[str]:
    """Stable fingerprints for REST rows (cards often lack slug/id — use name)."""
    out: set[str] = set()
    for row in (rows or [])[:20]:
        if not isinstance(row, dict):
            continue
        slug = row.get("slug") or row.get("id") or row.get("pro_id")
        if slug is not None and str(slug).strip():
            out.add(str(slug).strip().lower())
            continue
        name = (row.get("name") or "").strip().lower()
        if name:
            out.add(re.sub(r"\s+", " ", name)[:80])
    return frozenset(out)


def _rest_category_filter_trustworthy(category_id, rows: list) -> bool:
    """
    Welfog products/search often ignores category= and returns the default home pool.
    Reject near-identical unfiltered dumps; allow smaller real category pages even if
    some SKUs also appear on the home page.
    """
    global _REST_CATEGORY_BASELINE_SLUGS
    if not rows:
        return False
    slugs = _product_slugs_from_rows(rows)
    if not slugs:
        return False
    if _REST_CATEGORY_BASELINE_SLUGS is None:
        baseline = _catalog_search_get({"page": 1})
        _REST_CATEGORY_BASELINE_SLUGS = _product_slugs_from_rows(baseline or [])
    baseline = _REST_CATEGORY_BASELINE_SLUGS or frozenset()
    if not baseline:
        return True
    if slugs == baseline:
        return False
    overlap = len(slugs & baseline)
    # Same-sized (or nearly) dump with mostly the same items → ignored category filter.
    if len(slugs) >= max(12, len(baseline) - 2) and overlap / max(len(slugs), 1) >= 0.8:
        return False
    if abs(len(slugs) - len(baseline)) <= 2 and overlap / max(len(slugs), 1) >= 0.85:
        return False
    return True


def _catalog_search_with_category(base_params: dict, category_id) -> list:
    """
    Welfog search API: `categories=` is the reliable department filter for many ids;
    `category` / `category_id` often return the default home pool.
    """
    if not category_id:
        return _catalog_search_get(base_params)
    cid = str(category_id).strip()
    for key in ("categories", "category", "category_id"):
        params = dict(base_params)
        params[key] = cid
        rows = _catalog_search_get(params)
        if rows and _rest_category_filter_trustworthy(cid, rows):
            return rows
    return []


def _parse_price_number(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _shipping_cost_from_product(product: dict) -> float:
    if not isinstance(product, dict):
        return 0.0
    pools: list[dict] = [product]
    nested = product.get("data")
    if isinstance(nested, dict):
        pools.append(nested)
    for pool in pools:
        v = _parse_price_number(pool.get("shipping_cost"))
        if v is not None and v >= 0:
            return float(v)
    return 0.0


def customer_landed_price(product: dict) -> Optional[float]:
    """
    Total price the customer pays at checkout: purchase_price + shipping_cost.
    OpenSearch and REST catalog both expose these fields separately.
    """
    if not isinstance(product, dict):
        return None
    pools: list[dict] = [product]
    nested = product.get("data")
    if isinstance(nested, dict):
        pools.append(nested)
    base: Optional[float] = None
    shipping = 0.0
    for pool in pools:
        purchase = _parse_price_number(pool.get("purchase_price"))
        if purchase is not None and purchase >= 0:
            base = float(purchase)
            shipping = _shipping_cost_from_product(pool)
            return base + shipping
    sale_keys = (
        "main_price",
        "selling_price",
        "new_price",
        "sale_price",
        "discounted_price",
        "final_price",
        "price",
    )
    for pool in pools:
        for key in sale_keys:
            v = _parse_price_number(pool.get(key))
            if v is not None and v > 0:
                return float(v) + _shipping_cost_from_product(pool)
    return None


def customer_sale_price(product: dict) -> Optional[float]:
    """
    Price the customer pays (purchase + shipping), not MRP (stroked_price, unit_price).
    """
    landed = customer_landed_price(product)
    if landed is not None:
        return landed
    if not isinstance(product, dict):
        return None
    pools: list[dict] = [product]
    nested = product.get("data")
    if isinstance(nested, dict):
        pools.append(nested)
    mrp_keys = ("stroked_price", "unit_price", "base_price", "mrp", "old_price")
    for pool in pools:
        for key in mrp_keys:
            v = _parse_price_number(pool.get(key))
            if v is not None and v > 0:
                return float(v)
    return None


def format_customer_price_display(product: dict, fallback=""):
    """Int when whole rupees — for product cards."""
    p = customer_sale_price(product)
    if p is None:
        return fallback
    return int(p) if p == int(p) else p


def slug_from_welfog_product_link(link: str) -> str:
    m = re.search(r"/product_details/([^/?#]+)", link or "", flags=re.IGNORECASE)
    return (m.group(1) or "").strip() if m else ""


def sync_product_cards_prices_from_rest_api(
    cards: list,
    query: str,
    *,
    color=None,
    category_id=None,
) -> list:
    """OpenSearch often has MRP in unit_price — refresh cards from live API main_price."""
    if not cards or not (query or "").strip():
        return cards
    try:
        params = {"page": 1, "latitude": "", "longitude": "", "name": (query or "").strip()}
        if color:
            params["color"] = color
        if category_id:
            params["category"] = str(category_id)
        res = requests.get(_CATALOG_SEARCH_URL, params=params, timeout=12)
        if res.status_code != 200:
            return cards
        by_slug: dict[str, object] = {}
        by_name: dict[str, object] = {}
        for raw in res.json().get("data") or []:
            if not isinstance(raw, dict):
                continue
            price = format_customer_price_display(raw, "")
            if price == "":
                continue
            slug = (raw.get("slug") or "").strip()
            name = (raw.get("name") or "").strip().lower()
            if slug:
                by_slug[slug] = price
            if name:
                by_name[name] = price
        out = []
        for card in cards:
            c = dict(card)
            slug = slug_from_welfog_product_link(c.get("link") or "")
            if slug and slug in by_slug:
                c["price"] = by_slug[slug]
            else:
                nm = (c.get("name") or "").strip().lower()
                if nm in by_name:
                    c["price"] = by_name[nm]
            out.append(c)
        return out
    except Exception:
        return cards


def fetch_api(endpoint, order_id, user_id=None):
    _t0 = time.perf_counter()
    try:
        url = f"http://localhost:5000/{endpoint}/{order_id}"
        params = {}
        if user_id:
            params["user_id"] = str(user_id)
        res = requests.get(url, params=params, timeout=10)
        return res.json() if res.status_code == 200 else None
    except:
        return None
    finally:
        _record_api_elapsed(_t0)


def _normalize_color(text: str):
    """Map colour words using word boundaries only — Redmi must not become Red."""
    if not text:
        return None
    t = text.lower().strip()
    if re.search(r"\bmulticolou?r\b", t) or "multi color" in t:
        return "Multicolor"
    phrase_map = (
        ("sky blue", "Sky Blue"),
        ("sky-blue", "Sky Blue"),
        ("light blue", "Light Blue"),
        ("aasmani", "Sky Blue"),
        ("asmani", "Sky Blue"),
        ("aasmaani", "Sky Blue"),
    )
    for phrase, canon in phrase_map:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            return canon
    word_map = (
        ("orange", "Orange"),
        ("black", "Black"),
        ("white", "White"),
        ("red", "Red"),
        ("blue", "Blue"),
        ("green", "Green"),
        ("yellow", "Yellow"),
        ("pink", "Pink"),
        ("purple", "Purple"),
        ("brown", "Brown"),
        ("grey", "Grey"),
        ("gray", "Grey"),
        ("navy", "Navy"),
        ("maroon", "Maroon"),
        ("beige", "Beige"),
        ("cream", "Beige"),
        ("silver", "Silver"),
        ("gold", "Gold"),
        ("kala", "Black"),
        ("safed", "White"),
        ("laal", "Red"),
        ("lal", "Red"),
        ("neela", "Blue"),
        ("hara", "Green"),
        ("hare", "Green"),
        ("peela", "Yellow"),
    )
    for word, canon in word_map:
        if re.search(rf"\b{re.escape(word)}\b", t):
            return canon
    return None


def _strip_color_words(query: str) -> str:
    if not query:
        return query
    query_lower = query.lower()
    color_tokens = {
        "red", "blue", "black", "white", "green", "yellow", "pink", "purple",
        "brown", "grey", "gray", "multicolor", "orange", "maroon", "silver",
        "gold", "navy", "beige", "cream",
    }
    stop_tokens = {
        "color", "colour", "rang", "ki", "ke", "ka", "liye", "ke liye", "se",
        "dikha", "dikhao", "dikho", "dikhaa", "dikhaan", "dikhado", "de", "do", "please",
        "items", "item", "products", "product", "kuch", "koi", "cheez", "cheeze", "dikhana",
    }
    words = re.findall(r"[a-z0-9]+", query_lower)
    cleaned = [w for w in words if w not in color_tokens and w not in stop_tokens]
    return _normalize_product_query(" ".join(cleaned).strip() or query_lower)


def _normalize_product_query(query: str) -> str:
    if not query:
        return query
    plural_map = {
        "covers": "cover",
        "cases": "case",
        "chargers": "charger",
        "cables": "cable",
        "shirts": "shirt",
        "pants": "pant",
        "phones": "phone",
        "mobiles": "mobile",
        "accessories": "accessory",
        "glasses": "glass",
        "jeans": "jeans",
        "shoes": "shoe",
        "earphones": "earphone",
        "headphones": "headphone",
        "loafers": "loafer",
        "tshirts": "tshirt",
    }
    no_strip_s = frozenset(
        {"tshirt", "tshirts", "shorts", "jeans", "glass", "dress", "class", "loafer", "loafers"}
    )
    words = query.split()
    if not words:
        return query
    last = words[-1]
    if last in plural_map:
        words[-1] = plural_map[last]
    elif len(last) > 3 and last.endswith("s") and last not in no_strip_s:
        words[-1] = last[:-1]
    return " ".join(words)

def _normalize_query_part(part: str) -> str:
    if not part:
        return part
    try:
        from services.product_query_understanding import clean_product_part_label
    except ImportError:
        clean_product_part_label = None
    tokens = []
    for chunk in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", part.lower()):
        tokens.extend(chunk.replace("-", " ").split())
    cleaned_words = [t for t in tokens if t and t not in GENERIC_QUERY_TOKENS]
    normalized = _normalize_product_query(" ".join(cleaned_words).strip())
    out = normalized if normalized else part.strip()
    if clean_product_part_label:
        cleaned = clean_product_part_label(out)
        if cleaned:
            return cleaned
    return out

def _generate_search_variants(query: str) -> list[str]:
    if not query:
        return []
    query = _normalize_query_part(query)
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if not tokens:
        return []

    variants = []
    product_noun_groups = {
        "cover", "case", "bumper", "glass", "protector",
        "charger", "cable", "adapter", "usb",
        "earphone", "earbud", "headphone", "headphones",
    }
    generic_inserts = ["mobile", "phone"]
    if any(tok in product_noun_groups for tok in tokens):
        has_generic = any(tok in generic_inserts for tok in tokens)
        if not has_generic:
            brand_tokens = [tok for tok in tokens if tok in PHONE_BRANDS or tok in FASHION_BRANDS]
            if brand_tokens:
                # Insert generic noun before product type for brand-specific queries.
                for noun_idx, tok in enumerate(tokens):
                    if tok in product_noun_groups:
                        for insert in generic_inserts:
                            candidate = tokens[:noun_idx] + [insert] + tokens[noun_idx:]
                            variants.append(" ".join(candidate))
                        break
            else:
                for insert in generic_inserts:
                    variants.append(f"{insert} {query}")
                    variants.append(f"{query} {insert}")

    if query not in variants:
        variants.insert(0, query)
    return [v for v in dict.fromkeys(variants) if v]

# ================= EXTERNAL APIs =================
PHONE_BRANDS = frozenset(
    {
        "samsung", "iphone", "apple", "vivo", "oppo", "realme", "redmi", "mi", "xiaomi", "oneplus",
        "poco", "nothing", "google", "motorola", "nokia", "honor", "infinix", "durex",
    }
)
FASHION_BRANDS = frozenset(
    {
        "raymond", "nike", "adidas", "puma", "woodland", "levis", "levi", "zara", "tommy",
        "gucci", "fossil", "bintage", "casio", "wood", "peter", "england",
    }
)
MATERIAL_KEYWORDS = frozenset(
    {
        "transparent", "crystal", "clear", "velvet", "leather", "nylon", "rubber", "silicone",
        "magsafe", "matte", "glossy", "gloss", "metallic", "wooden", "metal",
    }
)
GENERIC_QUERY_TOKENS = frozenset(
    {
        "all", "show", "me", "the", "a", "an", "buy", "need", "want", "dikha", "chahiye",
        "saare", "mere", "ko", "bhai", "please", "pls", "sirf", "bas", "only",
        "ke", "ki", "ka", "se", "par", "pe", "me", "mai", "main",
        "items", "item", "products", "product", "kuch", "koi", "cheez", "cheeze",
        "dikhao", "dikho", "dikhana", "dikhaa", "dikhaan", "dikhado",
    }
)
COLOR_HUE_WORDS = frozenset(
    {
        "red", "blue", "black", "white", "green", "yellow", "pink", "purple",
        "brown", "grey", "gray", "multicolor", "orange", "maroon", "silver",
        "gold", "navy", "beige", "cream",
    }
)
COLOR_META_WORDS = frozenset({"color", "colour", "rang"})


def _lg_velvet_model_noise(query_lower: str, name_lower: str) -> bool:
    """User said 'velvet' as material but product is LG Velvet phone — skip."""
    if "velvet" not in query_lower or "lg" in query_lower:
        return False
    return bool(re.search(r"\blg\s*[-]?\s*velvet\b", name_lower))


def _required_title_keywords(q_words: list, query_lower: str, color_requested) -> list:
    """Tokens that must appear as substrings in product title (brand / material / long tokens)."""
    structural = {
        "cover", "case", "cases", "covers", "bumper", "mobile", "phone", "cell", "glass", "protector",
    }
    brands = PHONE_BRANDS | FASHION_BRANDS
    req = []
    for w in q_words:
        if len(w) < 3:
            continue
        if w in GENERIC_QUERY_TOKENS:
            continue
        if color_requested and w in COLOR_HUE_WORDS:
            continue
        if w in COLOR_META_WORDS:
            continue
        if w in MATERIAL_KEYWORDS or w in brands:
            req.append(w)
        elif len(w) >= 5 and w not in structural:
            req.append(w)
    # de-dupe preserve order
    seen = set()
    out = []
    for w in req:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _keyword_in_product_title(kw: str, title_lower: str) -> bool:
    if kw in title_lower:
        return True
    if len(kw) > 3 and kw.endswith("s") and kw[:-1] in title_lower:
        return True
    return False


PRODUCT_TYPE_NOUNS = frozenset(
    {
        "cover", "case", "bumper", "glass", "protector", "charger", "cable", "adapter", "usb",
        "earphone", "earphones", "earbud", "headphone", "headphones", "jeans", "pant", "pants",
        "shirt", "tshirt", "mobile", "phone", "tablet", "watch", "wallet", "bag", "shoe", "shoes",
        "saree", "dress", "sunglasses", "spectacles", "book", "books", "accessory", "accessories",
        "lipstick", "perfume", "belt", "cap", "hat", "socks", "sandal", "sandals", "slipper", "slippers",
        "rice", "wheat", "jug", "bottle", "flour", "atta", "dal", "lentil", "oil", "milk", "bread",
        "cooker", "pan", "pot", "utensil", "toy", "toys", "sofa", "chair", "table", "bedsheet",
        "towel", "soap", "shampoo", "cream", "lotion", "fan", "bulb", "lamp", "keyboard", "mouse",
        "speaker", "camera", "trimmer", "blender", "mixer", "grinder", "mask", "diaper", "notebook",
    }
)

_MULTI_JOINER_RE = re.compile(
    r"\s+(?:and|or|aur|plus|with|along\s+with|as\s+well\s+as|"
    r"sath|saath|aur\s+bhi|matlab|meaning)\s+|[,;&]+",
    flags=re.IGNORECASE,
)


def repair_multi_product_joiners(text: str) -> str:
    """Fix common typos so 'mobile covers an d shoes' splits like '... and ...'."""
    if not text:
        return text
    t = text
    t = re.sub(r"\s+an\s+d\s+", " and ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+a\s+n\s+d\s+", " and ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+nd\s+", " and ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+&\s+", " and ", t)
    t = re.sub(r"\s+also\s+", " and ", t, flags=re.IGNORECASE)
    return t


def _token_is_product_noun(token: str) -> bool:
    if not token:
        return False
    t = token.lower()
    if t in PRODUCT_TYPE_NOUNS:
        return True
    if len(t) > 3 and t.endswith("s") and t[:-1] in PRODUCT_TYPE_NOUNS:
        return True
    return False


def _split_implicit_multi_products(query: str) -> list:
    """
    Split 'mobile covers white shoes' (no 'and') into separate product searches.
    """
    if not query:
        return []
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in GENERIC_QUERY_TOKENS and t not in {"d", "an"}
    ]
    if len(tokens) < 4:
        return []
    modifier_nouns = frozenset({"mobile", "phone", "cell", "tablet"})
    noun_indices = []
    for i, t in enumerate(tokens):
        if not _token_is_product_noun(t):
            continue
        if t in modifier_nouns and i + 1 < len(tokens) and _token_is_product_noun(tokens[i + 1]):
            continue
        noun_indices.append(i)
    if len(noun_indices) < 2:
        return []
    first_end = noun_indices[0] + 1
    split_start = noun_indices[1]
    while split_start > first_end and tokens[split_start - 1] in COLOR_HUE_WORDS:
        split_start -= 1
    while split_start > first_end and tokens[split_start - 1] in ("mobile", "phone", "cell"):
        split_start -= 1
    part1 = _normalize_query_part(" ".join(tokens[:first_end]))
    part2 = _normalize_query_part(" ".join(tokens[split_start:]))
    parts = [p for p in (part1, part2) if p]
    return parts if len(parts) >= 2 else []


_MULTI_PART_FILLER = frozenset(
    {
        "to", "yrr", "yr", "yrrr", "bro", "bta", "de", "bhai", "sir", "pls", "please",
        "batana", "batao", "bata", "chahiye", "dikhao", "dikha", "dikhado", "dikhana",
        "mereko", "mujhe", "ko", "ke", "ki", "ka", "liye", "aur", "and", "or",
        "mean", "i", "a", "an", "the", "show", "me", "also", "kaam", "kr", "karo",
        "help", "find", "one", "na", "ek", "dono", "donon", "both", "lena", "leni",
    }
)

# Device words in "cover for my mobile" — not a second product in the same part.
_DEVICE_MODIFIER_NOUNS = frozenset({"mobile", "phone", "cell", "tablet"})


def _strip_filler_tokens_from_part(part: str) -> str:
    """Language-neutral: drop conversational filler tokens (works on msg_en / Latin)."""
    if not part:
        return part
    raw = str(part).strip()
    if not re.search(r"[a-z0-9]", raw, re.I):
        return raw
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    kept = [
        t
        for t in tokens
        if t not in GENERIC_QUERY_TOKENS and t not in _MULTI_PART_FILLER
    ]
    if not kept:
        return raw
    return " ".join(kept)


def _clean_multi_product_part(part: str) -> str:
    """Strip trailing device/for-me tails and filler words before validating a segment."""
    if not part:
        return part
    p = str(part).strip()
    p = re.sub(
        r"\s+for\s+(?:me|my(?:\s+mobile)?|mobile|phone)\s*$",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"\s+for\s*$", "", p, flags=re.IGNORECASE).strip()
  # Roman Hinglish extras (msg_en path reduces need; kept for hinglish-only chats)
    p = re.sub(
        r"\b(?:dono|donon|dikha\s*de|dikhao|batao|batana|bta\s*de)\b",
        " ",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"\s+", " ", p).strip()
    p = _strip_filler_tokens_from_part(p) or p
    return _normalize_query_part(p) if p else ""


def _part_contains_multiple_product_nouns(part: str) -> bool:
    """Reject merged junk like 'red tshirt for iphone cover my mobile'."""
    if not part:
        return False
    found = []
    for tok in re.findall(r"[a-z0-9]+", part.lower()):
        if _token_is_product_noun(tok):
            found.append(tok)
    uniq = []
    for n in found:
        if n not in uniq:
            uniq.append(n)
    primary = [n for n in uniq if n not in _DEVICE_MODIFIER_NOUNS]
    if len(primary) >= 2:
        return True
    if len(primary) >= 1:
        return False
    return len(uniq) >= 2


def _part_looks_like_product_query(part: str) -> bool:
    if not part or not str(part).strip():
        return False
    if _part_contains_multiple_product_nouns(part):
        return False
    tokens = [t.replace("-", "") for t in re.findall(r"[a-z0-9\-]+", str(part).lower())]
    non_filler = [t for t in tokens if t not in GENERIC_QUERY_TOKENS and t not in _MULTI_PART_FILLER]
    if not non_filler:
        return False
    if len(non_filler) > 9:
        return False
    if any(_token_is_product_noun(t) for t in non_filler):
        return True
    if len(non_filler) >= 2:
        return len([t for t in non_filler if len(t) >= 3]) >= 2
    return len(non_filler[0]) >= 4


def multi_product_parts_are_valid(parts: list) -> bool:
    """True only when 2+ parts each look like a real product request (not 'to bta de')."""
    if not parts or len(parts) < 2:
        return False
    try:
        from services.product_query_understanding import is_noisy_search_query
    except ImportError:
        is_noisy_search_query = lambda _t: False
    good = 0
    for p in parts:
        if not p or not _part_looks_like_product_query(p):
            continue
        if is_noisy_search_query(p):
            continue
        good += 1
    return good >= 2


def _split_aur_mere_dual(text: str) -> list:
    """'white shirt chahiye aur mere pass infinix cover' → two clauses."""
    if not text:
        return []
    m = re.search(
        r"^(.+?)\s+aur\s+mere(?:\s+pass)?\b\s*(.+)$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if not m:
        return []
    a = _normalize_query_part(m.group(1))
    b = _normalize_query_part(m.group(2))
    if a and b and a != b:
        return [a, b]
    return []


def _split_brand_ke_liye_dual(text: str) -> list:
    """
    'samsung ke liye aur infinix ke liye mobile covers' → two parts sharing the tail noun.
    """
    if not text:
        return []
    low = text.lower()
    m = re.search(
        r"(.+?\s+ke\s+liye)\s+aur\s+(.+?\s+ke\s+liye)"
        r"(?:\s+(.+?))?(?:\s+(?:dikh|dikha|dikhao|bta|btao|de|do|lena|chahiye|manga|mang|h)\b|$)",
        low,
        flags=re.IGNORECASE,
    )
    if not m:
        return []
    shared = (m.group(3) or "").strip()
    a = _normalize_query_part(f"{m.group(1)} {shared}".strip())
    b = _normalize_query_part(f"{m.group(2)} {shared}".strip())
    if a and b and a != b:
        return [a, b]
    return []


def _split_hinglish_ek_aur_dual(text: str) -> list:
    """'ek to iphone cover dikha aur ek na samsung cover' → two parts."""
    if not text:
        return []
    low = text.lower()
    # ek samsung ka aur ek iphone ka cover → samsung cover, iphone cover
    m_ek_ka = re.search(
        r"ek\s+(.+?)\s+ka\s+aur\s+ek\s+(.+?)\s+ka\s+(.+?)(?:\s+(?:lena|lenaa|chahiye|dikhao|dikha|bta|btao|de|do|h)\b|$)",
        low,
        flags=re.IGNORECASE,
    )
    if m_ek_ka:
        a_raw, b_raw, noun = m_ek_ka.group(1).strip(), m_ek_ka.group(2).strip(), m_ek_ka.group(3).strip()
        noun = _normalize_query_part(noun)
        a = _normalize_query_part(f"{a_raw} {noun}".strip())
        b = _normalize_query_part(f"{b_raw} {noun}".strip())
        if a and b and a != b:
            return [a, b]
    for pat in (
        r"ek\s+to\s+(.+?)\s+aur\s+ek\s+na\s+(.+?)(?:\s+(?:dikha|dikah|dikhao|bta|btao|de|do)\b|$)",
        r"ek\s+to\s+(.+?)\s+aur\s+(.+?)(?:\s+(?:dikha|dikah|dikhao|bta|btao|de|do)\b|$)",
        r"(.+?)\s+aur\s+ek\s+na\s+(.+?)(?:\s+(?:dikha|dikah|dikhao|bta|btao)\b|$)",
        r"ek\s+(.+?)\s+aur\s+ek\s+(.+?)(?:\s+(?:dikha|dikah|dikhao|bta|btao|de|do|lena)\b|$)",
    ):
        m = re.search(pat, low, flags=re.IGNORECASE)
        if m:
            a = _normalize_query_part(m.group(1))
            b = _normalize_query_part(m.group(2))
            if a and b and a != b:
                return [a, b]
    return []


def collect_multi_product_parts(*queries: str) -> list:
    """
    Deduplicated product sub-queries from user text.
    Pass msg_en first when available (Marathi/Tamil/etc. → English via to_en).
    """
    seen: set[str] = set()
    out: list[str] = []
    candidates: list[str] = []
    for raw in queries:
        if raw and str(raw).strip():
            candidates.append(repair_multi_product_joiners(str(raw).strip()))
    if not candidates:
        return []
    # Prefer English-normalized text for splitting when caller passes (msg_en, original).
    if len(candidates) >= 2:
        candidates = [candidates[0], candidates[1]] + sorted(candidates[2:], key=len, reverse=True)
    else:
        candidates.sort(key=len, reverse=True)
    tried: set[str] = set()
    for q in candidates:
        if q in tried:
            continue
        tried.add(q)
        parts = _split_aur_mere_dual(q)
        if len(parts) < 2:
            parts = _split_brand_ke_liye_dual(q)
        if len(parts) < 2:
            parts = _split_hinglish_ek_aur_dual(q)
        if len(parts) < 2:
            parts = _split_product_query(q)
        if len(parts) < 2:
            parts = _split_implicit_multi_products(q)
        if len(parts) < 2 and _MULTI_JOINER_RE.search(q):
            parts = [_normalize_query_part(p) for p in _MULTI_JOINER_RE.split(q) if p.strip()]
        for p in parts:
            p = _clean_multi_product_part((p or "").strip())
            if not p or p in seen:
                continue
            if not _part_looks_like_product_query(p):
                continue
            seen.add(p)
            out.append(p)
        if len(out) >= 2:
            return out
    return out


def _color_for_product_part(part: str, message_color=None):
    """Use color only when it belongs to this part (avoid 'white' on every split query)."""
    part_color = _normalize_color(part or "")
    if part_color:
        return part_color
    if message_color and message_color.lower() in (part or "").lower():
        return message_color
    return None


def search_multi_product_parts(
    parts: list,
    category_id=None,
    message_color=None,
    page: int = 1,
) -> tuple[list, list]:
    """
    Search each part separately. Returns (products deduped, missing_part_labels).
    """
    products: list = []
    missing: list = []
    seen_keys: set = set()
    try:
        from services.product_query_understanding import clean_product_part_label
    except ImportError:
        clean_product_part_label = lambda t, _m="": (t or "").strip()

    for part in parts:
        label = clean_product_part_label(part) or part
        part_color = _color_for_product_part(label, message_color)
        part_results = fetch_products_from_api(
            label, category_id=category_id, color=part_color, page=page
        )
        if part_results:
            for item in part_results:
                key = (item.get("link"), item.get("name"), item.get("price"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                products.append(item)
        else:
            missing.append(label)
    return products, missing


def _split_product_query(query: str) -> list:
    if not query:
        return []
    query = repair_multi_product_joiners(query)
    parts = re.split(r"\s+(?:and|or|aur|plus|with|sath|saath)\s+|[,/&]+", query, flags=re.IGNORECASE)
    parts = [_normalize_query_part(part) for part in parts if part.strip()]

    if len(parts) <= 1:
        return parts

    product_nouns = PRODUCT_TYPE_NOUNS

    def _has_product_noun(text: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return any(_token_is_product_noun(tok) for tok in tokens)

    if len(parts) == 2 and _has_product_noun(parts[-1]) and not _has_product_noun(parts[0]):
        tail_tokens = re.findall(r"[a-z0-9]+", parts[-1].lower())
        tail = []
        generic_prefixed = {
            "mobile",
            "phone",
            "tablet",
            "watch",
            "bag",
            "wallet",
            "shoe",
            "shoes",
            "case",
            "cover",
            "charger",
            "cable",
            "adapter",
            "earphone",
            "earbud",
            "glass",
            "protector",
            "shirt",
            "pant",
            "pants",
            "jeans",
            "sunglasses",
            "spectacles",
        }
        for idx in range(len(tail_tokens) - 1, -1, -1):
            tok = tail_tokens[idx]
            if tok in product_nouns:
                if idx > 0 and tail_tokens[idx - 1] in generic_prefixed:
                    tail = tail_tokens[idx - 1 :]
                else:
                    tail = tail_tokens[idx:]
                break
        if tail:
            tail_text = " ".join(tail)
            parts[0] = _normalize_query_part(f"{parts[0]} {tail_text}".strip())
    return [part for part in parts if part]


def fetch_catalog_browse_products(*, max_pages: int = 5) -> list[dict]:
    """Live REST catalog pages without a name filter — for price/rating-only post-filter browse."""
    from services import kb_service as _kb

    sysmsg = _kb.sysmsg
    IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
    out: list[dict] = []
    seen: set[str] = set()
    try:
        for pg in range(1, max(1, int(max_pages)) + 1):
            r = requests.get(
                _CATALOG_SEARCH_URL,
                params={"page": pg, "latitude": "", "longitude": ""},
                timeout=12,
            )
            if r.status_code != 200:
                continue
            for p in (r.json() or {}).get("data") or []:
                if not isinstance(p, dict):
                    continue
                key = str(p.get("slug") or p.get("id") or p.get("name") or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                name = p.get("name") or sysmsg("default_product_card_title")
                thumb = (p.get("thumbnail_image") or "").lstrip("/")
                slug = (p.get("slug") or "").strip()
                out.append(
                    {
                        "name": name,
                        "price": format_customer_price_display(p, sysmsg("na_price")),
                        "image": f"{IMAGE_BASE_URL}{thumb}" if thumb else "",
                        "link": f"https://welfog.com/product_details/{slug}" if slug else "https://welfog.com",
                        "rating": p.get("rating"),
                        "brand": p.get("brand") or "",
                        "sku": p.get("sku") or "",
                        "color_name": p.get("color_name") or p.get("color") or "",
                    }
                )
    except Exception as e:
        print(f"Catalog browse fetch error: {e}")
    return out


def fetch_product_by_pro_id(pro_id: int) -> Optional[dict]:
    """Lookup one catalog product by pro_id — OpenSearch first, then live API."""
    if not pro_id:
        return None
    try:
        from services.opensearch_products import search_opensearch_products

        cards, _total, _hm = search_opensearch_products(
            {"pro_id": int(pro_id), "title_query": ""},
            page_size=1,
            post_filter_mode="light",
        )
        if cards:
            return cards[0]
    except Exception:
        pass
    try:
        for pg in (1, 2, 3):
            batch = fetch_products_from_api(str(pro_id), page=pg)
            for item in batch or []:
                link = (item.get("link") or "").lower()
                if str(pro_id) in link or str(pro_id) in (item.get("name") or ""):
                    return item
    except Exception:
        pass
    return None


def fetch_products_from_api(query, category_id=None, color=None, page=1):
    from services import kb_service as _kb
    sysmsg = _kb.sysmsg
    if (not query or not query.strip()) and not category_id and not color:
        return []

    query = (query or "").lower()

    q_words = [w for w in query.split() if w not in GENERIC_QUERY_TOKENS]
    clean_query = " ".join(q_words)
    if not clean_query:
        clean_query = query

    if color:
        clean_query = _strip_color_words(clean_query)

    anti_words = []
    cover_terms = ["cover", "case", "bumper", "glass", "protector"]
    charge_terms = ["charger", "cable", "adapter", "usb"]
    has_cover_terms = any(term in clean_query for term in cover_terms)
    has_charge_terms = any(term in clean_query for term in charge_terms)

    if has_cover_terms and not has_charge_terms:
        anti_words.extend(charge_terms)
    elif has_charge_terms and not has_cover_terms:
        anti_words.extend(cover_terms)

    all_brands = PHONE_BRANDS | FASHION_BRANDS
    brands = list(all_brands)
    query_brands = [w for w in q_words if w in all_brands]

    meaningful_for_color_browse = [
        w
        for w in q_words
        if len(w) >= 2 and w not in COLOR_HUE_WORDS and w not in COLOR_META_WORDS
    ]
    color_browse_only = bool(color and not meaningful_for_color_browse)

    try:
        products_list = []

        def _do_search(extra_params):
            p = {"page": page or 1, "latitude": "", "longitude": ""}
            p.update(extra_params or {})
            cat = p.pop("categories", None) or p.pop("category", None) or p.pop("category_id", None)
            if cat is not None:
                return _catalog_search_with_category(p, cat)
            return _catalog_search_get(p)

        original_query_brands: list = []
        product_brand_match = False

        if color_browse_only:
            for pg in (1, 2, 3):
                products_list.extend(_do_search({"color": color, "page": pg}))
                if len(products_list) >= 60:
                    break
            # de-dupe by slug/id
            seen = set()
            uniq = []
            for it in products_list:
                if not isinstance(it, dict):
                    continue
                k = it.get("slug") or it.get("id")
                if k is not None:
                    if k in seen:
                        continue
                    seen.add(k)
                uniq.append(it)
            products_list = uniq
        else:
            params = {"page": page or 1}
            if clean_query.strip():
                params["name"] = clean_query
            if color:
                params["color"] = color

            if category_id:
                products_list = _catalog_search_with_category(params, category_id)
            else:
                products_list = _catalog_search_get(params)

            original_query_brands = list(query_brands)
            if query_brands and products_list:
                for item in products_list:
                    name_lower = (item.get("name") or "").lower()
                    if any(qb in name_lower for qb in query_brands):
                        product_brand_match = True
                        break
                if not product_brand_match:
                    query_brands = []

            if not products_list and len(q_words) > 1 and not category_id:
                fallback_word = q_words[0] if any(b in q_words[0] for b in brands) else q_words[-1]
                products_list = _do_search({"page": 1, "name": fallback_word})

            if not products_list:
                variants = _generate_search_variants(clean_query)
                for variant in variants:
                    if variant == clean_query:
                        continue
                    products_list = _do_search({"page": page or 1, "name": variant})
                    if products_list:
                        break

            if not products_list:
                parts = _split_product_query(query)
                if len(parts) > 1:
                    combined = []
                    seen = set()
                    for part in parts:
                        part_results = fetch_products_from_api(part, category_id=category_id, color=color, page=page)
                        for item in part_results:
                            key = item.get("link") or item.get("name") or str(item.get("price"))
                            if key not in seen:
                                seen.add(key)
                                combined.append(item)
                    return combined[:15]
                return []

        required = _required_title_keywords(q_words, query, bool(color))
        if original_query_brands and not product_brand_match:
            required = [w for w in required if w not in original_query_brands]
        query_lower_full = query

        IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
        scored_products = []

        def _matches_color(product: dict, color_name: str) -> bool:
            if not color_name:
                return True
            color_name = color_name.lower()
            fields = []
            if isinstance(product.get("name"), str):
                fields.append(product.get("name"))
            if isinstance(product.get("color"), str):
                fields.append(product.get("color"))
            if isinstance(product.get("colors"), str):
                fields.append(product.get("colors"))
            if isinstance(product.get("colors"), list):
                fields.extend([str(x) for x in product.get("colors") if x])
            if isinstance(product.get("description"), str):
                fields.append(product.get("description"))
            pdata = product.get("data")
            if isinstance(pdata, dict):
                if isinstance(pdata.get("color_name"), str):
                    fields.append(pdata.get("color_name"))
                if isinstance(pdata.get("color"), str):
                    fields.append(pdata.get("color"))
                if isinstance(pdata.get("colors"), str):
                    fields.append(pdata.get("colors"))
                if isinstance(pdata.get("colors"), list):
                    fields.extend([str(x) for x in pdata.get("colors") if x])
            all_text = " ".join(fields).lower()
            name_only = (product.get("name") or "").lower()
            if color_name in all_text or color_name in name_only:
                return True
            from services.opensearch_products import _product_name_matches_color

            return _product_name_matches_color(name_only or all_text, color_name)

        for p in products_list:
            name = p.get("name") or sysmsg("default_product_card_title")
            name_lower = name.lower()

            if query_brands and not any(qb in name_lower for qb in query_brands):
                continue

            if color and not _matches_color(p, color):
                continue

            product_words = name_lower.split()
            if any(aw in product_words for aw in anti_words):
                continue

            if _lg_velvet_model_noise(query_lower_full, name_lower):
                continue

            if required:
                missing = [kw for kw in required if not _keyword_in_product_title(kw, name_lower)]
                if missing:
                    continue

            score = 0
            for word in q_words:
                singular = word[:-1] if word.endswith("s") else word
                if word in name_lower or singular in name_lower:
                    score += 2
                elif len(word) >= 4:
                    for n_word in product_words:
                        if abs(len(word) - len(n_word)) <= 1:
                            match_count = sum(1 for a, b in zip(word, n_word) if a == b)
                            if match_count / max(len(word), 1) >= 0.75:
                                score += 1
                                break

            if color and color.lower() in name_lower:
                score += 2

            if score > 0 or not q_words:
                scored_products.append(
                    {
                        "name": name,
                        "price": format_customer_price_display(p, sysmsg("na_price")),
                        "image": (IMAGE_BASE_URL + p.get("thumbnail_image", "").lstrip("/"))
                        if p.get("thumbnail_image")
                        else "",
                        "link": f"https://welfog.com/product_details/{p.get('slug', '')}" if p.get("slug") else "https://welfog.com",
                        "score": score,
                    }
                )

        scored_products.sort(key=lambda x: x["score"], reverse=True)
        if not scored_products:
            parts = _split_product_query(query)
            if len(parts) > 1:
                combined = []
                seen = set()
                for part in parts:
                    part_results = fetch_products_from_api(part, category_id=category_id, color=color, page=page)
                    for item in part_results:
                        key = item.get("link") or item.get("name") or str(item.get("price"))
                        if key not in seen:
                            seen.add(key)
                            combined.append(item)
                return combined[:15]
        return scored_products[:15]

    except Exception as e:
        print(f"Product Fetch Error: {e}")
        return []


def check_pincode_delivery(pincode):
    """
    Live delivery serviceability for a 6-digit Indian PIN.
    Returns dict with result true/false, or error metadata (never bare None on HTTP errors).
    """
    pin = re.sub(r"\D", "", str(pincode or ""))
    if len(pin) != 6 or pin[0] == "0":
        return {
            "result": None,
            "message": "Invalid pincode format",
            "error": "invalid_format",
            "pincode": pin or str(pincode or ""),
        }
    try:
        url = "https://welfogapi.welfog.com/api/v2/pincode/check_pincode"
        payload = {"pincode": pin}
        res = requests.post(url, data=payload, timeout=12)
        if res.status_code == 200:
            data = res.json() if res.content else {}
            if isinstance(data, dict):
                data.setdefault("pincode", pin)
                return data
            return {"result": None, "message": "Unexpected API response", "error": "bad_json", "pincode": pin}
        print(f"⚠️ Pincode API status {res.status_code} for {pin}")
        return {
            "result": None,
            "message": f"Pincode API HTTP {res.status_code}",
            "error": "http_error",
            "status_code": res.status_code,
            "pincode": pin,
        }
    except requests.Timeout:
        print(f"❌ Pincode API timeout for {pin}")
        return {"result": None, "message": "Pincode API timeout", "error": "timeout", "pincode": pin}
    except Exception as e:
        print(f"❌ Pincode API Exception: {e}")
        return {"result": None, "message": str(e), "error": "exception", "pincode": pin}


def fetch_nav_categories():
    try:
        url = "https://welfogapi.welfog.com/api/nav_cat_data"
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Categories API Exception: {e}")
        return None


# -------- Category name -> id resolver (cached) --------
_navcat_cache = {"ts": 0.0, "map": {}}
_navcat_slug_cache = {"ts": 0.0, "id_to_slug": {}}
_navcat_ttl_sec = 600  # 10 minutes

WELFOG_SITE_BASE = "https://welfog.com"

_innercat_cache = {"ts": 0.0, "flat": {}, "id_meta": {}, "by_main": {}}


def _normalize_cat_name(s: str):
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CATEGORY_BROWSE_FILLER = frozenset(
    {
        "mereko", "mujhe", "meko", "mere", "mera", "meri", "please", "plz",
        "product", "products", "item", "items", "show", "list", "dikhao", "dikha",
        "dikhe", "dikhaa", "dekho", "dekh", "batao", "btao", "bata", "bta", "btana", "batana",
        "chahiye", "chiye", "category", "categories", "ke", "ki", "ka", "ko", "se",
        "me", "in", "all", "saare", "saari", "sab", "want", "need", "give", "dena",
        "lao", "la", "de", "do", "wale", "wali", "some", "any", "bhi", "also", "too",
        "k", "pls", "yar", "yrr", "bhai", "na",
    }
)

# Common customer phrasing → normalized nav department names (top-level only).
_NAV_DEPT_PHRASE_ALIASES: tuple[tuple[str, str], ...] = (
    ("mens fashion", "men fashion"),
    ("men s fashion", "men fashion"),
    ("womens fashion", "women fashion"),
    ("women s fashion", "women fashion"),
    ("groceries", "home kitchen"),
    ("grocery", "home kitchen"),
    ("mobile phones", "electronics"),
    ("mobile phone", "electronics"),
    ("mobiles", "electronics"),
    ("mobile", "electronics"),
    ("home and kitchen", "home kitchen"),
    ("home & kitchen", "home kitchen"),
    ("men grooming", "men s grooming"),
    ("mens grooming", "men s grooming"),
)


def _expand_category_query_text(text: str) -> str:
    """Typo / Hinglish aliases before nav lookup — no API calls."""
    t = _normalize_cat_name(text)
    if not t:
        return ""
    t = re.sub(r"\bmens\b", "men", t)
    t = re.sub(r"\bwomens\b", "women", t)
    for src, dst in sorted(_NAV_DEPT_PHRASE_ALIASES, key=lambda x: -len(x[0])):
        if src in t:
            t = t.replace(src, dst)
    return t


def _message_has_product_search_filters(text: str) -> bool:
    """True when the turn targets a specific product (brand, type, color, price…) — not dept browse."""
    if not (text or "").strip():
        return False
    if _message_targets_specific_product(text):
        return True
    low = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]{2,}", low))
    if words & PHONE_BRANDS:
        return True
    try:
        from services.opensearch_products import (
            _extract_price_bounds,
            color_hue_mentioned_in_text,
            normalize_color_fuzzy,
        )

        pmax, pmin = _extract_price_bounds(text, low)
        if pmax is not None or pmin is not None:
            return True
        hue = normalize_color_fuzzy(low)
        if hue and color_hue_mentioned_in_text(hue, low):
            return True
    except ImportError:
        pass
    if re.search(r"\bsku\b", low):
        return True
    return False


_CATEGORY_BROWSE_SIGNALS = (
    "dikhao", "dikha", "dikhe", "dikhaa", "dikha de", "dikha do", "dikhado",
    "dekho", "dekh", "show", "list", "display", "browse", "explore",
    "batao", "btao", "bata", "bta", "btana", "batana", "bta de", "bata de", "bta do",
    "bhi bta", "bhi dikha", "bhi bata", "bhi show", "k bhi", "ke bhi",
    "chahiye", "chiye", "dena", "de do", "dede", "de de",
    "lao", "la do", "bhejo", "send", "view", "see",
    "puchh", "pooch", "puch", "bol", "bolo",
    "product", "products", "item", "items", "samaan", "cheez",
)


def _message_has_browse_signals(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    return any(s in tl for s in _CATEGORY_BROWSE_SIGNALS)


def message_requests_category_browse(text: str) -> bool:
    """True when user wants products from a named department (any language/style)."""
    if not (text or "").strip():
        return False
    try:
        from utils.helpers import _looks_like_browse_all_categories_message

        if _looks_like_browse_all_categories_message(text):
            return False
    except ImportError:
        pass
    if _message_has_product_search_filters(text):
        return False
    if _user_explicitly_browses_category(text):
        return True
    if not _message_has_browse_signals(text):
        return False
    return bool(resolve_nav_category_id_fast(text))


def resolve_nav_category_id_fast(text: str, ctx=None) -> Optional[str]:
    """
    Top-level Welfog nav departments only — one cached nav_cat API, no inner_categories fanout.
  """
    if not (text or "").strip():
        return None
    expanded = _expand_category_query_text(text)
    explicit = _extract_explicit_category_id(expanded, ctx)
    if explicit:
        return explicit
    nav_map = _get_nav_categories_map(ctx)
    main_hint = _main_category_hint_from_text(expanded, ctx)
    if main_hint and (
        _user_explicitly_browses_category(text) or _message_has_browse_signals(text)
    ):
        return str(main_hint)
    for candidate in _category_lookup_candidates(expanded):
        if not candidate:
            continue
        for name, cid in sorted(nav_map.items(), key=lambda x: len(x[0]), reverse=True):
            if name and name in candidate:
                if not main_hint or _category_matches_main(str(cid), main_hint, {}, nav_map):
                    return str(cid)
        hit = _category_fuzzy_lookup(
            candidate, nav_map, main_hint=main_hint, id_meta={}, nav_map=nav_map
        )
        if hit:
            return str(hit)
    if main_hint:
        return str(main_hint)
    return None


def _resolve_inner_category_id(text: str, ctx=None) -> Optional[str]:
    """Subcategory / inner tree — loads inner_categories (slower; cached)."""
    t = _expand_category_query_text(text)
    if not t:
        return None
    _, id_meta, _ = _get_inner_categories_flat_map(ctx)
    nav_map = _get_nav_categories_map(ctx)
    main_hint = _main_category_hint_from_text(t, ctx)
    mapping = _get_combined_categories_map(ctx)
    for candidate in _category_lookup_candidates(text):
        hit = _category_fuzzy_lookup(
            candidate,
            mapping,
            main_hint=main_hint,
            id_meta=id_meta,
            nav_map=nav_map,
        )
        if hit and not _should_ignore_wrong_covers_category(text, hit):
            return hit
    return None


def resolve_category_browse_for_catalog(
    text: str,
    ctx=None,
    *,
    allow_inner_lookup: bool = True,
) -> Optional[tuple[str, str]]:
    """
    Category-only or category-first browse — nav-fast first, inner tree only when needed.
    Returns (category_id, product_search_query); empty query = whole department.
    """
    if not message_requests_category_browse(text):
        return None
    if _message_has_product_search_filters(text):
        return None
    cid = resolve_nav_category_id_fast(text, ctx)
    if not cid and allow_inner_lookup and _user_explicitly_browses_category(text):
        if ctx:
            ensure_expanded_categories_map_for_ctx(ctx)
        cid = _resolve_inner_category_id(text, ctx)
    if not cid:
        return None
    if not is_top_level_main_category(cid, ctx) and not _user_explicitly_browses_category(text):
        return None
    sq = ""
    if _message_targets_specific_product(text):
        from utils.helpers import extract_product_search_query

        sq = (extract_product_search_query(text, text, "") or "").strip()
        if query_should_use_category_id_only(cid, sq or text, ctx):
            sq = category_browse_search_name(cid, sq or text, ctx) or sq
    else:
        sq = category_browse_search_name(cid, text, ctx)
    return str(cid), sq


def _get_nav_categories_map(ctx=None, force_refresh: bool = False) -> dict:
    """normalized category name -> id (cached)."""
    if ctx:
        cat_map = (ctx.get("data") or {}).get("categories_map") or {}
        if cat_map:
            out = {}
            for name, cid in cat_map.items():
                nn = _normalize_cat_name(name)
                if nn:
                    out[nn] = str(cid)
            if out:
                return out

    now_ts = datetime.now().timestamp()
    cached_map = _navcat_cache.get("map") or {}
    if (
        not force_refresh
        and cached_map
        and (now_ts - float(_navcat_cache.get("ts") or 0)) < _navcat_ttl_sec
    ):
        return dict(cached_map)

    cats = fetch_nav_categories()
    items = []
    if isinstance(cats, dict):
        if isinstance(cats.get("categories"), list):
            items = cats.get("categories")
        else:
            for key in ("data", "categories", "result"):
                if isinstance(cats.get(key), list):
                    items = cats.get(key)
                    break
    elif isinstance(cats, list):
        items = cats

    new_map = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("id") or it.get("category_id") or it.get("cat_id")
        name = it.get("name") or it.get("title") or it.get("category_name")
        nn = _normalize_cat_name(name)
        if cid and nn:
            new_map[nn] = str(cid)

    _navcat_cache["ts"] = now_ts
    _navcat_cache["map"] = new_map
    return new_map


def _slugify_category_label(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", _normalize_cat_name(name or ""))
    return s.strip("-")


def _get_category_id_to_slug_map(ctx=None, force_refresh: bool = False) -> dict:
    """Welfog.com category page slug per nav category id (from nav_cat_data)."""
    if ctx:
        extra = (ctx.get("data") or {}).get("category_slugs") or {}
        if extra:
            return {str(k): str(v) for k, v in extra.items() if v}

    now_ts = datetime.now().timestamp()
    cached = _navcat_slug_cache.get("id_to_slug") or {}
    if (
        not force_refresh
        and cached
        and (now_ts - float(_navcat_slug_cache.get("ts") or 0)) < _navcat_ttl_sec
    ):
        return dict(cached)

    cats = fetch_nav_categories()
    items = []
    if isinstance(cats, dict):
        if isinstance(cats.get("categories"), list):
            items = cats.get("categories")
        else:
            for key in ("data", "categories", "result"):
                if isinstance(cats.get(key), list):
                    items = cats.get(key)
                    break
    elif isinstance(cats, list):
        items = cats

    id_to_slug: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("id") or it.get("category_id") or it.get("cat_id")
        slug = (it.get("slug") or "").strip()
        if cid and slug:
            id_to_slug[str(cid)] = slug

    _navcat_slug_cache["ts"] = now_ts
    _navcat_slug_cache["id_to_slug"] = id_to_slug
    return id_to_slug


def category_slug_for_id(category_id, ctx=None) -> str:
    """Slug for https://welfog.com/category/<slug> — nav slug first, else slugified display name."""
    cid = str(category_id or "").strip()
    if not cid:
        return ""
    slug_map = _get_category_id_to_slug_map(ctx)
    if cid in slug_map:
        return slug_map[cid]
    main_id = resolve_search_category_id(cid, ctx)
    if main_id and main_id in slug_map:
        return slug_map[main_id]
    name = category_name_for_id(cid, ctx)
    if name:
        leaf = name.split("—")[-1].strip() if "—" in name else name
        return _slugify_category_label(leaf)
    return ""


def build_welfog_product_browse_url(spec: dict, ctx=None) -> str:
    """
    Deep link for "View more products" — category page or site search (same as View Product pattern).
    Category-only browse → /category/<slug>; named search → /search/?q=...
    """
    from urllib.parse import quote, quote_plus

    spec = spec or {}
    cat_id = str(spec.get("category_id") or "").strip()
    title = (spec.get("title_query") or "").strip()
    color = (spec.get("color") or "").strip()
    brand = (spec.get("brand") or "").strip()

    search_parts: list[str] = []
    if color and color.lower() not in title.lower():
        search_parts.append(color)
    if title:
        search_parts.append(title)
    elif brand:
        search_parts.append(brand)
    search_q = " ".join(search_parts).strip()

    if cat_id and not search_q:
        slug = category_slug_for_id(cat_id, ctx)
        if slug:
            return f"{WELFOG_SITE_BASE}/category/{quote(slug, safe='')}"
        return f"{WELFOG_SITE_BASE}/category/{quote(cat_id, safe='')}"

    if search_q:
        url = f"{WELFOG_SITE_BASE}/search/?q={quote_plus(search_q)}"
        if cat_id:
            slug = category_slug_for_id(cat_id, ctx)
            if slug:
                url += f"&category={quote_plus(slug)}"
            else:
                url += f"&categories={quote_plus(cat_id)}"
        return url

    if cat_id:
        slug = category_slug_for_id(cat_id, ctx)
        if slug:
            return f"{WELFOG_SITE_BASE}/category/{quote(slug, safe='')}"
        return f"{WELFOG_SITE_BASE}/category/{quote(cat_id, safe='')}"

    return WELFOG_SITE_BASE


def fetch_inner_categories(main_category_id):
    """GET inner_categories — parent groups + children subcategories for a main nav department."""
    try:
        mid = str(main_category_id or "").strip()
        if not mid:
            return None
        url = "https://welfogapi.welfog.com/api/inner_categories"
        res = requests.get(url, params={"main_category_id": mid}, timeout=12)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Inner Categories API Exception: {e}")
        return None


def _flatten_inner_categories_payload(payload, main_category_id: str, main_display_name: str = ""):
    """Build name->id aliases and id->meta from one inner_categories response."""
    flat = {}
    id_meta = {}
    by_main = []
    if not isinstance(payload, dict):
        return flat, id_meta, by_main

    main_id = str(payload.get("main_category_id") or main_category_id or "").strip()
    main_label = (main_display_name or "").strip()
    groups = payload.get("categories") or []
    if not isinstance(groups, list):
        return flat, id_meta, by_main

    def _register(name: str, cid, level: str, parent_name: str = ""):
        if not name or cid is None:
            return
        cid_s = str(cid)
        nn = _normalize_cat_name(name)
        if not nn:
            return
        id_meta[cid_s] = {
            "id": cid_s,
            "name": str(name).strip(),
            "level": level,
            "main_category_id": main_id,
            "parent_name": parent_name or "",
        }

        def _set_alias(alias: str, force: bool = False):
            if not alias:
                return
            if alias in flat and flat[alias] != cid_s and not force:
                # Same label in another department — drop bare alias; use qualified names only
                del flat[alias]
            elif alias not in flat or force:
                flat[alias] = cid_s

        if level == "child":
            if nn in flat and flat[nn] != cid_s:
                del flat[nn]
            elif nn not in flat:
                flat[nn] = cid_s
        elif nn not in flat:
            flat[nn] = cid_s

        short_main = main_label.split()[0] if main_label else ""
        for stem in _category_stem_aliases(name):
            _set_alias(stem)
            if main_label and len(short_main) >= 3:
                _set_alias(_normalize_cat_name(f"{short_main} {stem}"))

        if main_label:
            q_main = _normalize_cat_name(f"{main_label} {name}")
            _set_alias(q_main)
            if len(short_main) >= 3:
                _set_alias(_normalize_cat_name(f"{short_main} {name}"))
        if parent_name:
            q_parent = _normalize_cat_name(f"{parent_name} {name}")
            _set_alias(q_parent)
            if main_label:
                q_all = _normalize_cat_name(f"{main_label} {parent_name} {name}")
                _set_alias(q_all)
                short_main = main_label.split()[0]
                if len(short_main) >= 3:
                    _set_alias(_normalize_cat_name(f"{short_main} {parent_name} {name}"))

    for parent in groups:
        if not isinstance(parent, dict):
            continue
        pid = parent.get("id")
        pname = (parent.get("name") or "").strip()
        if pid is not None and pname:
            _register(pname, pid, "parent")
            by_main.append(
                {
                    "id": str(pid),
                    "name": pname,
                    "children": [],
                }
            )
        children_out = by_main[-1]["children"] if by_main else []
        for child in parent.get("children") or []:
            if not isinstance(child, dict):
                continue
            cid = child.get("id")
            cname = (child.get("name") or "").strip()
            if cid is None or not cname:
                continue
            _register(cname, cid, "child", parent_name=pname)
            children_out.append({"id": str(cid), "name": cname})

    return flat, id_meta, by_main


def _get_inner_categories_flat_map(ctx=None, force_refresh: bool = False):
    """
    Cached flat map: normalized category/subcategory name -> id, plus id->meta.
    Loads inner_categories for every top-level nav department.
    """
    if ctx:
        data = ctx.get("data") or {}
        inner_map = data.get("inner_categories_flat") or {}
        inner_meta = data.get("category_id_meta") or {}
        if inner_map and inner_meta:
            return dict(inner_map), dict(inner_meta), data.get("inner_categories_by_main") or {}

    now_ts = datetime.now().timestamp()
    cached_flat = _innercat_cache.get("flat") or {}
    cached_meta = _innercat_cache.get("id_meta") or {}
    if (
        not force_refresh
        and cached_flat
        and (now_ts - float(_innercat_cache.get("ts") or 0)) < _navcat_ttl_sec
    ):
        return dict(cached_flat), dict(cached_meta), dict(_innercat_cache.get("by_main") or {})

    nav_map = _get_nav_categories_map(ctx, force_refresh=force_refresh)
    main_ids = sorted({str(v) for v in nav_map.values() if v})
    id_to_main_name = {}
    for name, mid in nav_map.items():
        id_to_main_name.setdefault(str(mid), name)

    flat = {}
    id_meta = {}
    by_main = {}
    if len(main_ids) <= 1:
        for mid in main_ids:
            payload = fetch_inner_categories(mid)
            main_name = id_to_main_name.get(mid, "")
            f, m, tree = _flatten_inner_categories_payload(payload, mid, main_name)
            flat.update(f)
            id_meta.update(m)
            if tree:
                by_main[mid] = tree
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = min(6, max(2, len(main_ids)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_inner_categories, mid): mid for mid in main_ids
            }
            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    payload = fut.result()
                except Exception:
                    payload = None
                main_name = id_to_main_name.get(mid, "")
                f, m, tree = _flatten_inner_categories_payload(payload, mid, main_name)
                flat.update(f)
                id_meta.update(m)
                if tree:
                    by_main[mid] = tree

    _innercat_cache["ts"] = now_ts
    _innercat_cache["flat"] = flat
    _innercat_cache["id_meta"] = id_meta
    _innercat_cache["by_main"] = by_main
    return flat, id_meta, by_main


def warmup_welfog_category_caches(*, load_inner: bool = True) -> None:
    """Preload nav + slug (+ optional inner tree) so first /chat category browse stays fast."""
    import time as _time

    from utils.reasoning_log import log_reasoning

    t0 = _time.perf_counter()
    _get_nav_categories_map()
    _get_category_id_to_slug_map()
    nav_ms = (_time.perf_counter() - t0) * 1000.0
    inner_ms = 0.0
    if load_inner:
        t1 = _time.perf_counter()
        _get_inner_categories_flat_map()
        inner_ms = (_time.perf_counter() - t1) * 1000.0
    log_reasoning(
        f"Category cache warmup: nav+slug={nav_ms:.0f}ms"
        + (f", inner={inner_ms:.0f}ms" if load_inner else " (inner skipped)")
    )


def _get_combined_categories_map(ctx=None, force_refresh: bool = False) -> dict:
    """Top-level nav names + inner parent/child aliases (longest-match lookup uses this)."""
    nav = _get_nav_categories_map(ctx, force_refresh=force_refresh)
    inner_flat, _, _ = _get_inner_categories_flat_map(ctx, force_refresh=force_refresh)
    combined = dict(nav)
    for name, cid in inner_flat.items():
        if name not in combined:
            combined[name] = cid
    if ctx:
        extra = (ctx.get("data") or {}).get("categories_map") or {}
        for name, cid in extra.items():
            nn = _normalize_cat_name(name)
            if nn and cid:
                combined[nn] = str(cid)
    return combined


def _category_stem_aliases(display_name: str):
    """Short tokens customers use (kurta, flipflops) from longer catalog labels."""
    n = _normalize_cat_name(display_name)
    if not n:
        return
    if "kurta" in n:
        yield "kurta"
        yield "kurtas"
    if "flipflop" in n or "slipper" in n:
        yield "flipflops"
        yield "slippers"
        yield "flip flop"
    if "sandal" in n or "floater" in n:
        yield "sandals"
        yield "floaters"


def _category_lookup_candidates(text: str) -> list:
    """Normalized phrases to try, including 'men footwear' from 'footwear ... men'."""
    out = []
    seen = set()

    def _add(s: str):
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    _add(_normalize_cat_name(text))
    stripped = _strip_message_for_category_lookup(text)
    _add(stripped)
    words = [w for w in stripped.split() if len(w) >= 3]
    dept_words = {"men", "women", "kids", "kid", "baby", "home", "beauty"}
    dept = [w for w in words if w in dept_words]
    rest = [w for w in words if w not in dept_words]
    for d in dept:
        for r in rest:
            _add(_normalize_cat_name(f"{d} {r}"))
            _add(_normalize_cat_name(f"{r} {d}"))
    return out


def _main_category_hint_from_text(text: str, ctx=None):
    """If message names a top-level department (e.g. men fashion, or just 'men'), return its main category id."""
    t = _normalize_cat_name(text)
    if not t:
        return None
    nav = _get_nav_categories_map(ctx)
    words = set(t.split())
    grooming_words = frozenset(
        {"grooming", "groom", "beard", "shave", "shaver", "razor", "trimmer", "aftershave", "facial"}
    )
    # "men"/"women" alone → prefer fashion departments (before short nav name "women" / "men")
    dept_priority = (
        ("men", ("men fashion", "men s grooming")),
        ("women", ("women fashion", "women")),
    )
    for dept_word, preferred_names in dept_priority:
        if dept_word not in words:
            continue
        for pref in preferred_names:
            if pref in nav and (pref != "men s grooming" or words.intersection(grooming_words)):
                return str(nav[pref])
    for name in sorted(nav.keys(), key=len, reverse=True):
        if name and name in t:
            return str(nav[name])
    for name, mid in sorted(nav.items(), key=lambda x: len(x[0]), reverse=True):
        if not name:
            continue
        parts = name.split()
        if len(parts) >= 2 and parts[0] in words:
            return str(mid)
    return None


def _known_category_ids(ctx=None):
    nav = _get_nav_categories_map(ctx)
    _, id_meta, _ = _get_inner_categories_flat_map(ctx)
    return set(nav.values()) | set(id_meta.keys())


def _extract_explicit_category_id(text: str, ctx=None):
    t = _normalize_cat_name(text)
    if not t:
        return None
    known = _known_category_ids(ctx)
    for m in re.finditer(r"\b(?:category|cat|subcategory|subcat|id)\s*[:=]?\s*(\d{1,5})\b", t):
        cid = m.group(1)
        if cid in known:
            return cid
    for m in re.finditer(r"\b(\d{1,5})\b", t):
        cid = m.group(1)
        if cid in known:
            return cid
    return None


def category_meta_for_id(category_id: str, ctx=None):
    cid = str(category_id or "").strip()
    if not cid:
        return {}
    _, id_meta, _ = _get_inner_categories_flat_map(ctx)
    return dict(id_meta.get(cid) or {})


def is_top_level_main_category(category_id, ctx=None) -> bool:
    cid = str(category_id or "").strip()
    if not cid:
        return False
    return cid in set(_get_nav_categories_map(ctx).values())


def main_category_has_inner_children(main_category_id, ctx=None) -> bool:
    mid = str(main_category_id or "").strip()
    if not mid:
        return False
    _, _, by_main = _get_inner_categories_flat_map(ctx)
    return bool(by_main.get(mid))


def ensure_expanded_categories_map_for_ctx(ctx):
    """Store full name->id map on ctx for follow-up category selection in chat."""
    if not ctx:
        return
    ctx.setdefault("data", {})
    combined = _get_combined_categories_map(ctx)
    ctx["data"]["categories_map"] = dict(combined)
    _, id_meta, by_main = _get_inner_categories_flat_map(ctx)
    ctx["data"]["category_id_meta"] = dict(id_meta)
    ctx["data"]["inner_categories_by_main"] = dict(by_main)


def format_inner_categories_reply(main_category_id, ctx=None):
    from services import kb_service as _kb

    mid = str(main_category_id or "").strip()
    if not mid:
        return ""
    payload = fetch_inner_categories(mid)
    if not payload:
        return _kb.sysmsg("categories_unavailable") or ""
    groups = payload.get("categories") or []
    if not groups:
        return _kb.sysmsg("categories_parse_failed") or ""

    dept = category_name_for_id(mid, ctx) or f"Category {mid}"
    ensure_expanded_categories_map_for_ctx(ctx)
    ctx.setdefault("data", {})
    ctx["data"]["selected_main_category_id"] = mid

    out = _kb.sysmsg("inner_categories_title", department=dept) or (
        f"<div style='margin-bottom:8px; color:#333;'><b>{html_escape(dept)} — subcategories:</b></div>"
    )
    out += _kb.sysmsg("categories_list_wrap_start") or "<div style='font-size:13px; color:#333; line-height:1.6;'>"
    for parent in groups:
        if not isinstance(parent, dict):
            continue
        pname = parent.get("name") or ""
        pid = parent.get("id")
        if pname and pid is not None:
            out += f"• <b>{html_escape(str(pname))}</b> (id: {pid})<br>"
        for child in parent.get("children") or []:
            if not isinstance(child, dict):
                continue
            cname = child.get("name") or ""
            cid = child.get("id")
            if cname and cid is not None:
                out += f"&nbsp;&nbsp;◦ {html_escape(str(cname))} (id: {cid})<br>"
    out += (_kb.sysmsg("categories_list_wrap_end") or "</div>") + (
        _kb.sysmsg("inner_categories_footer") or _kb.sysmsg("categories_footer") or ""
    )
    return out


def _category_matches_main(cid: str, main_hint: str, id_meta: dict, nav_map: dict) -> bool:
    if not main_hint:
        return True
    cid_s = str(cid)
    if cid_s == str(main_hint):
        return True
    meta = (id_meta or {}).get(cid_s) or {}
    meta_main = str(meta.get("main_category_id") or "")
    if meta_main:
        return meta_main == str(main_hint)
    # Top-level nav id with no inner meta
    return cid_s in set(nav_map.values()) and cid_s == str(main_hint)


def _category_match_score(name: str, cid: str, main_hint: str, id_meta: dict, nav_map: dict) -> int:
    score = len(name or "")
    meta = (id_meta or {}).get(str(cid)) or {}
    if meta.get("level") == "child":
        score += 80
    elif meta.get("level") == "parent":
        score += 20
    if main_hint and _category_matches_main(cid, main_hint, id_meta, nav_map):
        score += 120
    return score


def _category_fuzzy_lookup(input_text: str, mapping: dict, main_hint: str = None, id_meta: dict = None, nav_map: dict = None):
    if not mapping or not input_text:
        return None
    nav_map = nav_map or {}
    id_meta = id_meta or {}

    substring_hits = []
    for name in mapping.keys():
        if name and name in input_text:
            cid = str(mapping[name])
            substring_hits.append((name, cid, _category_match_score(name, cid, main_hint, id_meta, nav_map)))

    if substring_hits:
        substring_hits.sort(key=lambda x: x[2], reverse=True)
        if main_hint:
            for _name, cid, _sc in substring_hits:
                if _category_matches_main(cid, main_hint, id_meta, nav_map):
                    return cid
            return None
        return substring_hits[0][1]

    words = [w for w in input_text.split() if len(w) >= 4]
    names = list(mapping.keys())
    for w in words:
        if w in PHONE_BRANDS or w in FASHION_BRANDS:
            continue
        best = difflib.get_close_matches(w, names, n=5, cutoff=0.82)
        for hit in best:
            cid = str(mapping[hit])
            if _category_matches_main(cid, main_hint, id_meta, nav_map):
                return cid
    phrase = difflib.get_close_matches(input_text, names, n=5, cutoff=0.75)
    for hit in phrase:
        cid = str(mapping[hit])
        if _category_matches_main(cid, main_hint, id_meta, nav_map):
            return cid
    if not main_hint:
        for hit in phrase:
            return str(mapping[hit])
    return None


def _is_category_browse_filler_token(tok: str) -> bool:
    """True when token is browse filler or a verb/morph variant of one (dekhne←dekh, bta←bata)."""
    t = (tok or "").strip().lower()
    if not t:
        return True
    if t in _CATEGORY_BROWSE_FILLER:
        return True
    # Morphological extensions of filler stems (≥4 chars): dekh→dekhne, dikha→dikhao.
    for f in _CATEGORY_BROWSE_FILLER:
        if len(f) >= 4 and len(t) > len(f) and t.startswith(f):
            return True
    # Truncated verb stems (bta←bata/batao) — token is a prefix of a filler word.
    if len(t) >= 3:
        for f in _CATEGORY_BROWSE_FILLER:
            if len(f) > len(t) and f.startswith(t):
                return True
    return False


def _strip_message_for_category_lookup(text: str) -> str:
    words = [
        w
        for w in _normalize_cat_name(text).split()
        if w and len(w) >= 2 and not _is_category_browse_filler_token(w)
    ]
    return " ".join(words).strip()


def category_name_for_id(category_id: str, ctx=None) -> str:
    cid = str(category_id or "").strip()
    if not cid:
        return ""
    for name, mapped in _get_nav_categories_map(ctx).items():
        if mapped == cid:
            return name.title() if name.islower() else name
    meta = category_meta_for_id(cid, ctx)
    if meta.get("name"):
        label = meta["name"]
        if meta.get("parent_name"):
            return f"{meta['parent_name']} — {label}"
        return label
    return ""


def query_should_use_category_id_only(category_id, query: str, ctx=None) -> bool:
    """
    Welfog products/search: `categories=<id>` must be used without a conflicting `name`
    filter — e.g. name=electronics + categories=16 returns empty while categories=16 alone works.
    """
    if not category_id:
        return False
    q = (query or "").strip()
    if not q:
        return True
    # Explicit department browse beats leftover verb noise — still id-only.
    if _user_explicitly_browses_category(q):
        resolved = (
            resolve_nav_category_id_fast(q, ctx=ctx)
            or resolve_nav_category_id_fast(_expand_category_query_text(q), ctx=ctx)
            or get_category_id_from_text(q, ctx=ctx)
        )
        if resolved and str(resolved) == str(category_id):
            if not _message_targets_specific_product(q):
                return True
    # Named product (vest, slippers…) always needs title_query — never id-only.
    if _message_targets_specific_product(q):
        return False
    stripped = _strip_message_for_category_lookup(q)
    if not stripped:
        return True
    # Leftover resolves to THIS category (or is the category name) → id-only browse.
    fast = resolve_nav_category_id_fast(stripped, ctx=ctx)
    if fast and str(fast) == str(category_id):
        return True
    if get_category_id_from_text(stripped, ctx=ctx) == str(category_id):
        return True
    # Typo / morphology left non-category tokens but full message still resolves
    # to this department — prefer category-id browse over polluted title_query.
    full_fast = resolve_nav_category_id_fast(q, ctx=ctx) or resolve_nav_category_id_fast(
        _expand_category_query_text(q), ctx=ctx
    )
    if full_fast and str(full_fast) == str(category_id):
        leftover = set(re.findall(r"[a-z0-9]{3,}", (stripped or "").lower()))
        try:
            leftover -= _category_name_tokens_for_id(category_id, ctx)
        except Exception:
            pass
        leftover = {t for t in leftover if not _is_category_browse_filler_token(t)}
        if not leftover:
            return True
    return False


def category_browse_search_name(category_id, query: str, ctx=None) -> str:
    """Return '' for category-only browse queries so the API returns category products."""
    if query_should_use_category_id_only(category_id, query, ctx):
        return ""
    # Never pass the raw conversational sentence as a product name filter.
    stripped = _strip_message_for_category_lookup(query)
    if not stripped:
        return ""
    if resolve_nav_category_id_fast(stripped, ctx=ctx) or get_category_id_from_text(
        stripped, ctx=ctx
    ):
        return ""
    return stripped


def _category_name_tokens_for_id(category_id, ctx=None) -> set[str]:
    """Normalized labels for a category id (nav + inner display names)."""
    out: set[str] = set()
    cid = str(category_id or "").strip()
    if not cid:
        return out
    label = category_name_for_id(cid, ctx) or ""
    for part in (label, label.split("—")[-1] if "—" in label else ""):
        n = _normalize_cat_name(part)
        if n:
            out.add(n)
    nav = _get_nav_categories_map(ctx)
    for name, mapped in nav.items():
        if str(mapped) == cid:
            n = _normalize_cat_name(name)
            if n:
                out.add(n)
    return out


def strip_category_browse_conflicts_from_spec(spec: dict, ctx=None) -> dict:
    """
    Category-only browse: drop bogus SKU and brand_aliases parsed from the category name
    (e.g. 'Electronics' mistaken as SKU; AI search_terms → brand_aliases=['electronics']).
    """
    spec = dict(spec or {})
    cat_id = str(spec.get("category_id") or "").strip()
    if not cat_id or (spec.get("title_query") or "").strip():
        return spec

    cat_names = _category_name_tokens_for_id(cat_id, ctx)

    def _is_category_token(val: str) -> bool:
        if not val:
            return False
        n = _normalize_cat_name(val)
        if n in cat_names:
            return True
        resolved = get_category_id_from_text(val, ctx=ctx)
        return resolved == cat_id

    sku = (spec.get("sku") or "").strip()
    if sku and _is_category_token(sku):
        spec.pop("sku", None)

    brand = (spec.get("brand") or "").strip()
    if brand and _is_category_token(brand):
        spec.pop("brand", None)

    aliases = spec.get("brand_aliases")
    if isinstance(aliases, list) and aliases:
        kept = [a for a in aliases if not _is_category_token(str(a))]
        if kept:
            spec["brand_aliases"] = kept
        else:
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)

    return spec


# Nav "Covers" under Home Utility — not phone cases (OpenSearch uses Mobile Cases & Covers).
_HOME_UTILITY_COVERS_CAT_ID = "1070"
_PHONE_COVER_CONTEXT_WORDS = frozenset(
    {
        "mobile", "phone", "iphone", "samsung", "vivo", "oppo", "realme", "redmi", "mi",
        "xiaomi", "oneplus", "infinix", "poco", "motorola", "nokia", "apple", "galaxy",
        "bumper", "case", "cases", "backcover", "tempered",
    }
)
_AMBIGUOUS_CATEGORY_PRODUCT_WORDS = frozenset(
    {"cover", "covers", "case", "cases", "bag", "bags", "wallet", "charger", "cable"}
)


def _should_ignore_wrong_covers_category(text: str, category_id: str) -> bool:
    """'covers' alone maps to Home Utility Covers (1070) but users mean mobile cases."""
    if str(category_id) != _HOME_UTILITY_COVERS_CAT_ID:
        return False
    low = (text or "").lower()
    if any(w in low for w in _PHONE_COVER_CONTEXT_WORDS):
        return True
    stripped = _strip_message_for_category_lookup(text)
    words = {w for w in stripped.split() if len(w) >= 3}
    filler = GENERIC_QUERY_TOKENS | COLOR_HUE_WORDS | COLOR_META_WORDS | {"color", "colour", "rang", "product", "products"}
    meaningful = words - filler
    if meaningful and meaningful <= _AMBIGUOUS_CATEGORY_PRODUCT_WORDS:
        return True
    return False


_CATEGORY_BROWSE_RE = re.compile(
    r"(?:"
    r"\b(?:category|subcategory|department|section)\b|"
    r"\bke\s+(?:saare\s+)?products?\b|"
    r"\b(?:me|mai|men)\s+(?:dikha|dikhao|dikhado|products?|items?|sab|saare)\b|"
    r"\bbrowse\s+(?:all\s+)?categories\b|"
    r"\b(?:saari|sari)\s+categories\b"
    r")",
    re.IGNORECASE,
)


def _user_explicitly_browses_category(text: str) -> bool:
    """User wants a department browse, not a single SKU keyword (shirt, charger…)."""
    low = (text or "").lower()
    if _CATEGORY_BROWSE_RE.search(low):
        return True
    dept_words = (
        "fashion", "beauty", "electronics", "grocery", "groceries", "kitchen",
        "home", "sports", "footwear", "appliance", "appliances", "mobiles",
    )
    if any(d in low for d in dept_words) and re.search(
        r"\b(?:products?|items?|sab|saare|browse|explore|dikha|dikhao)\b", low
    ):
        try:
            from services.opensearch_products import _extract_product_keywords

            if not _extract_product_keywords(low):
                return True
        except ImportError:
            return True
    return False


def _message_targets_specific_product(text: str) -> bool:
    """
    True when the message names a product type beyond a bare department browse.

    Uses leftover tokens after stripping category labels / browse filler — not a
    product synonym dictionary — so vest/baniyan/flipflops/etc. stay product search.
    """
    low = (text or "").lower()
    if not low.strip():
        return False
    try:
        from services.opensearch_products import (
            _GENERIC_TITLE_MODIFIERS,
            _extract_product_keywords,
            _PRODUCT_NOUNS,
        )

        if _extract_product_keywords(low):
            return True
        words = set(re.findall(r"[a-z0-9]{3,}", low))
        if words & set(_PRODUCT_NOUNS):
            return True
    except ImportError:
        _GENERIC_TITLE_MODIFIERS = frozenset()  # noqa: N806

    stripped = _strip_message_for_category_lookup(text)
    leftover = set(re.findall(r"[a-z0-9]{3,}", (stripped or "").lower()))
    try:
        nav = _get_nav_categories_map(None) or {}
        cat_toks: set[str] = set()
        for name in nav.keys():
            cat_toks.update(re.findall(r"[a-z0-9]{3,}", (name or "").lower()))
        leftover -= cat_toks
    except Exception:
        pass
    leftover = {t for t in leftover if not _is_category_browse_filler_token(t)}
    try:
        leftover -= set(_GENERIC_TITLE_MODIFIERS)
    except Exception:
        pass
    # Any remaining token (vest, baniyan, flipflops, innerwear, …) ⇒ product search.
    if leftover:
        # If message resolves to a department, only ignore leftover that is the
        # category name / typo noise — keep real product nouns (vest, slippers…).
        try:
            cid = resolve_nav_category_id_fast(text) or resolve_nav_category_id_fast(
                _expand_category_query_text(text)
            )
            if cid:
                leftover -= _category_name_tokens_for_id(cid)
                leftover = {t for t in leftover if not _is_category_browse_filler_token(t)}
                try:
                    leftover -= set(_GENERIC_TITLE_MODIFIERS)
                except Exception:
                    pass
                if not leftover:
                    return False
        except Exception:
            pass
        return True
    return False


def resolve_category_id_for_product_search(text: str, ctx=None) -> Optional[str]:
    """
    Attach categories= for department browse or when a category name is resolved.
    SKU-style queries without a category name search the full catalog by title only.
    """
    from utils.helpers import _text_requests_category_product_browse

    ctx = ctx or {}
    prior = (ctx.get("data") or {}).get("selected_category_id")
    if prior:
        return str(prior)

    if _message_has_product_search_filters(text):
        return None

    cid = get_category_id_from_text(text, ctx)
    if not cid:
        return None

    if _text_requests_category_product_browse(text, ctx) or _user_explicitly_browses_category(text):
        if _message_has_product_search_filters(text):
            return None
        return cid

    if _message_targets_specific_product(text):
        return None

    return cid


def resolve_category_product_browse_route(
    text: str,
    ctx=None,
) -> Optional[tuple[str, str]]:
    """
    User wants products filtered by a Welfog category name (not the full category list).
    Returns (category_id, product_search_query) — empty query means category-only browse.
    """
    return resolve_category_browse_for_catalog(text, ctx, allow_inner_lookup=True)


def resolve_search_category_id(category_id: str, ctx=None) -> str:
    """Map inner/nav category id to the id that works with products/search categories= filter."""
    cid = str(category_id or "").strip()
    if not cid:
        return cid
    nav_vals = set(_get_nav_categories_map(ctx).values())
    if cid in nav_vals:
        return cid
    meta = category_meta_for_id(cid, ctx)
    main_id = str(meta.get("main_category_id") or "").strip()
    if main_id and main_id in nav_vals:
        return main_id
    return cid


def department_catalog_label_names(category_id, ctx=None) -> list[str]:
    """
    Live nav/inner display names under a department — used to match OpenSearch
    category_name (leaf labels). No product keyword lists.
    """
    cid = str(category_id or "").strip()
    if not cid:
        return []
    main = resolve_search_category_id(cid, ctx) or cid
    names: list[str] = []
    seen: set[str] = set()

    def _add(label: str) -> None:
        raw = (label or "").strip()
        if not raw:
            return
        leaf = raw.split("—")[-1].strip() if "—" in raw else raw
        for cand in (raw, leaf):
            key = _normalize_cat_name(cand)
            if key and key not in seen:
                seen.add(key)
                names.append(cand.strip())

    _add(category_name_for_id(main, ctx))
    _add(category_name_for_id(cid, ctx))
    try:
        _, meta, _ = _get_inner_categories_flat_map(ctx)
        for _child_id, m in (meta or {}).items():
            if not isinstance(m, dict):
                continue
            if str(m.get("main_category_id") or "").strip() != str(main):
                continue
            _add(str(m.get("name") or ""))
    except Exception:
        pass
    return names


def fetch_products_by_category_browse(category_id, ctx=None, page=1, color=None):
    """
    Category-only product browse via search API with resolved main category id.
    Never falls back to unfiltered name-only search (that mixes other departments).
    Returns (products_list, effective_category_id).
    """
    from utils.reasoning_log import log_reasoning

    cid = str(category_id or "").strip()
    if not cid:
        return [], cid

    search_cid = resolve_search_category_id(cid, ctx)
    try_ids = []
    for c in (search_cid, cid):
        if c and c not in try_ids:
            try_ids.append(c)

    for try_cid in try_ids:
        products = fetch_products_from_api("", category_id=try_cid, color=color, page=page)
        if products and _rest_category_filter_trustworthy(try_cid, products):
            log_reasoning(
                f"Category browse API hit category={try_cid} ({len(products)} items)."
            )
            return products, try_cid
        if products:
            log_reasoning(
                f"Category browse REST ignored category={try_cid} "
                f"(default catalog dump — skipped)."
            )

    # Trusted name+category only (same fingerprint gate). Never name-only.
    label = category_name_for_id(cid, ctx) or category_name_for_id(search_cid, ctx)
    if label:
        kw = (label.split("—")[-1] if "—" in label else label).strip()
        products = fetch_products_from_api(kw, category_id=search_cid, color=color, page=page)
        if products and _rest_category_filter_trustworthy(search_cid, products):
            log_reasoning(f"Category browse name+filter categories={search_cid} name={kw!r}.")
            return products, search_cid

    return [], search_cid


def get_category_id_from_text(text: str, ctx=None):
    """
    Returns category_id (string) if any top-level, inner parent, or subcategory name matches.
    Also accepts explicit ids present in nav / inner_categories (e.g. 1014, 10).
    """
    fast = resolve_nav_category_id_fast(text, ctx)
    if fast:
        return fast

    explicit = _extract_explicit_category_id(text, ctx)
    if explicit:
        return explicit

    main_hint = _main_category_hint_from_text(text, ctx)
    if main_hint and _user_explicitly_browses_category(text):
        return main_hint

    inner = _resolve_inner_category_id(text, ctx)
    if inner:
        return inner

    t = _expand_category_query_text(text)
    if not t:
        return None

    main_hint = _main_category_hint_from_text(t, ctx)
    _, id_meta, _ = _get_inner_categories_flat_map(ctx)
    nav_map = _get_nav_categories_map(ctx)
    mapping = _get_combined_categories_map(ctx)
    for candidate in _category_lookup_candidates(text):
        hit = _category_fuzzy_lookup(
            candidate,
            mapping,
            main_hint=main_hint,
            id_meta=id_meta,
            nav_map=nav_map,
        )
        if hit and not _should_ignore_wrong_covers_category(text, hit):
            return hit
    return None


def fetch_today_deals(latitude="34.04505157470703", longitude="78.38422393798828"):
    try:
        url = "https://welfogapi.welfog.com/api/today_deal"
        res = requests.get(url, params={"latitude": latitude, "longitude": longitude}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Deals API Exception: {e}")
        return None


def fetch_category_wise_feed(page=1, latitude="34.04505157470703", longitude="78.38422393798828"):
    try:
        url = "https://welfogapi.welfog.com/api/cat_wise_product_show"
        res = requests.get(url, params={"latitude": latitude, "longitude": longitude, "page": page}, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception as e:
        print(f"❌ Cat-wise API Exception: {e}")
        return None


def fetch_purchase_history(user_id, page=1):
    """GET purchase-history — tries page + offset variants (many backends ignore `page` alone)."""
    uid = str(user_id).strip()
    if not uid:
        return None
    url = f"{PH_API_BASE}/{uid}"
    page = max(1, int(page))
    offset = (page - 1) * PH_PER_PAGE
    param_sets = (
        {"user_id": uid, "page": page, "per_page": PH_PER_PAGE, "limit": PH_PER_PAGE, "offset": offset},
        {"user_id": uid, "page": page, "per_page": PH_PER_PAGE},
        {"user_id": uid, "limit": PH_PER_PAGE, "offset": offset},
        {"user_id": uid, "page": page},
    )
    last_err = None
    timeout = _LIVE_ORDER_API_TIMEOUT_SEC
    for params in param_sets:
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            last_err = res.status_code
        except Exception as e:
            last_err = e
            if _is_transient_network_error(e):
                break
            continue
    print(f"purchase-history failed last_err={last_err}", flush=True)
    return None


def _ph_fetch_deadline() -> float:
    return time.time() + float(os.getenv("PH_FETCH_BUDGET_SEC", "10") or "10")


def _fetch_purchase_history_at_offset(uid: str, offset: int, *, deadline: float | None = None):
    """Single API call with strongest offset/limit hints — returns (rows, reported_total)."""
    url = f"{PH_API_BASE}/{uid}"
    page_num = offset // PH_PER_PAGE + 1 if PH_PER_PAGE else 1
    param_sets = (
        {"user_id": uid, "limit": PH_PER_PAGE, "offset": offset, "page": page_num, "per_page": PH_PER_PAGE},
        {"user_id": uid, "offset": offset, "limit": PH_PER_PAGE},
        {"user_id": uid, "page": page_num, "per_page": PH_PER_PAGE},
    )
    timeout = min(_LIVE_ORDER_API_TIMEOUT_SEC, 5)
    for params in param_sets:
        if deadline is not None and time.time() >= deadline:
            return None
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code != 200:
                continue
            data = res.json()
            rows = data.get("data") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                continue
            reported_total = None
            if isinstance(data, dict):
                meta = data.get("meta")
                if isinstance(meta, dict):
                    try:
                        reported_total = int(meta.get("total"))
                    except (TypeError, ValueError):
                        reported_total = None
            return rows, reported_total
        except Exception:
            continue
    return None


def _ph_accum_snapshot(uid: str) -> dict:
    with _PH_ACCUM_LOCK:
        slot = _PH_ACCUM.get(str(uid).strip())
        return dict(slot) if isinstance(slot, dict) else {}


def _ph_infer_has_more(
    rows: list,
    chunk: list,
    *,
    exhausted: bool,
    reported_total: int | None,
    page: int,
) -> bool:
    if reported_total is not None and reported_total > 0:
        return reported_total > page * PH_PER_PAGE
    if len(rows) > page * PH_PER_PAGE:
        return True
    return len(chunk) >= PH_PER_PAGE and not exhausted


def _purchase_history_first_page_rows(uid: str) -> list:
    """
    Page-1 guard/chat fast path — one fetch round, strict budget, warm accum cache.
    Avoids multi-round offset probing that could exceed client timeouts.
    """
    uid = str(uid).strip()
    if not uid:
        return []
    deadline = _ph_fetch_deadline()
    now = time.time()
    with _PH_ACCUM_LOCK:
        slot = _PH_ACCUM.get(uid)
        if slot and (now - slot["ts"]) <= _PH_ACCUM_TTL_SEC:
            rows = slot.get("rows") or []
            if rows:
                return list(rows)
    batch_pack = _fetch_purchase_history_at_offset(uid, 0, deadline=deadline)
    if not batch_pack:
        return []
    batch, reported_total = batch_pack
    if not batch:
        return []
    with _PH_ACCUM_LOCK:
        slot = {
            "rows": list(batch),
            "ts": time.time(),
            "exhausted": len(batch) < PH_PER_PAGE,
            "reported_total": reported_total,
        }
        _PH_ACCUM[uid] = slot
        return list(batch)


def _row_id(row):
    if not isinstance(row, dict):
        return None
    return row.get("oid") or row.get("id")


def _ensure_accumulated_rows(uid: str, min_length: int):
    """
    Append batches using offset until we have at least min_length rows or API repeats / ends.
    Fixes backends that return the same first page for every `page` value.
    HTTP calls run outside _PH_ACCUM_LOCK so /chat greetings are not blocked.
    """
    uid = str(uid).strip()
    if min_length <= PH_PER_PAGE + 1:
        rows = _purchase_history_first_page_rows(uid)
        if rows:
            return rows
    max_fetch_rounds = 4
    rounds = 0
    deadline = _ph_fetch_deadline()
    while rounds < max_fetch_rounds and time.time() < deadline:
        rounds += 1
        with _PH_ACCUM_LOCK:
            now = time.time()
            slot = _PH_ACCUM.get(uid)
            if slot and (now - slot["ts"]) > _PH_ACCUM_TTL_SEC:
                _PH_ACCUM.pop(uid, None)
                slot = None
            if slot is None:
                slot = {"rows": [], "ts": now, "exhausted": False}
                _PH_ACCUM[uid] = slot
            rows = slot["rows"]
            if slot["exhausted"] and len(rows) >= min_length:
                return rows
            if slot["exhausted"]:
                return rows
            if len(rows) >= min_length:
                return rows
            offset = len(rows)
            first_row_id = _row_id(rows[0]) if rows else None

        if time.time() >= deadline:
            break
        batch_pack = _fetch_purchase_history_at_offset(uid, offset, deadline=deadline)
        if not batch_pack:
            with _PH_ACCUM_LOCK:
                slot = _PH_ACCUM.get(uid)
                if slot:
                    slot["exhausted"] = True
                    slot["ts"] = time.time()
                return (slot or {}).get("rows") or []
        batch, reported_total = batch_pack
        if not batch:
            with _PH_ACCUM_LOCK:
                slot = _PH_ACCUM.get(uid)
                if slot:
                    slot["exhausted"] = True
                    slot["ts"] = time.time()
                return (slot or {}).get("rows") or []

        first_new = _row_id(batch[0])
        with _PH_ACCUM_LOCK:
            slot = _PH_ACCUM.get(uid)
            if not slot:
                return batch[:min_length] if batch else []
            rows = slot["rows"]
            if reported_total is not None:
                slot["reported_total"] = reported_total
            if rows and first_new is not None and first_new == first_row_id:
                slot["exhausted"] = True
                slot["ts"] = time.time()
                return rows
            if rows:
                existing_ids = {_row_id(r) for r in rows if _row_id(r) is not None}
                new_rows = []
                for r in batch:
                    rid = _row_id(r)
                    if rid is None:
                        new_rows.append(r)
                    elif rid not in existing_ids:
                        existing_ids.add(rid)
                        new_rows.append(r)
                if not new_rows:
                    slot["exhausted"] = True
                    slot["ts"] = time.time()
                    return rows
                rows.extend(new_rows)
            else:
                rows.extend(batch)
            slot["ts"] = time.time()
            if len(batch) < PH_PER_PAGE:
                slot["exhausted"] = True
            if len(rows) >= min_length:
                return rows
    with _PH_ACCUM_LOCK:
        slot = _PH_ACCUM.get(uid)
        return (slot or {}).get("rows") or []


def purchase_history_slice(uid: str, page: int):
    """
    Returns (display_rows, has_more) for 1-based page using accumulated server-side list.
    """
    uid = str(uid).strip()
    page = max(1, int(page))
    if page == 1:
        rows = _purchase_history_first_page_rows(uid)
        chunk = rows[:PH_PER_PAGE]
        snap = _ph_accum_snapshot(uid)
        has_more = _ph_infer_has_more(
            rows,
            chunk,
            exhausted=bool(snap.get("exhausted")),
            reported_total=snap.get("reported_total"),
            page=page,
        )
        return chunk, has_more
    end = page * PH_PER_PAGE
    # Fetch one extra row past the current page boundary so we can accurately
    # tell whether there is a next page, even when the current page is exactly
    # full (e.g. 10, 20, 30 rows).
    rows = _ensure_accumulated_rows(uid, end + 1)
    start = (page - 1) * PH_PER_PAGE
    chunk = rows[start:end]
    snap = _ph_accum_snapshot(uid)
    has_more = _ph_infer_has_more(
        rows,
        chunk,
        exhausted=bool(snap.get("exhausted")),
        reported_total=snap.get("reported_total"),
        page=page,
    )
    if not has_more and len(rows) > end:
        has_more = True
    return chunk, has_more


def _ph_order_flow_from_row(row: dict) -> dict:
    detail = row.get("order_status_detail") if isinstance(row, dict) else None
    if not isinstance(detail, dict):
        return {}
    order_status = detail.get("order_status")
    if not isinstance(order_status, dict):
        return {}
    flow = order_status.get("order_flow")
    return flow if isinstance(flow, dict) else {}


def _ph_status_badge_class(current_status: str, delivery_string: str) -> str:
    s = f"{current_status or ''} {delivery_string or ''}".lower()
    if "cancel" in s:
        return "wf-ph-status-badge--cancelled"
    if "deliver" in s and "out for" not in s:
        return "wf-ph-status-badge--delivered"
    if "ship" in s or "out for" in s:
        return "wf-ph-status-badge--shipped"
    return "wf-ph-status-badge--active"


def _ph_order_flow_timeline_html(row: dict) -> str:
    """Vertical step flow from purchase-history order_status_detail.order_flow."""
    flow = _ph_order_flow_from_row(row)
    if not flow:
        return "<div class='wf-ph-track-empty'>Tracking steps are not available for this order yet.</div>"

    current_lower = str(row.get("current_order_status") or "").lower()
    cancelled_step = flow.get("cancelled") if isinstance(flow.get("cancelled"), dict) else {}
    cancelled = "cancel" in current_lower or bool(cancelled_step.get("status"))

    keys = list(_ORDER_FLOW_STEP_ORDER)
    if cancelled and "cancelled" in flow and "cancelled" not in keys:
        keys.append("cancelled")

    parts = ["<div class='wf-ph-flow' role='list'>"]
    for key in keys:
        step = flow.get(key)
        if not isinstance(step, dict):
            continue
        done = bool(step.get("status"))
        title = html_escape(str(step.get("title") or key.replace("_", " ").title()))
        when = _format_track_step_datetime(step.get("date"))
        reason = step.get("reason")
        state = "done" if done else "pending"
        parts.append(f"<div class='wf-ph-flow-step wf-ph-flow-step--{state}' role='listitem'>")
        parts.append("<span class='wf-ph-flow-dot' aria-hidden='true'></span>")
        parts.append("<div class='wf-ph-flow-body'>")
        parts.append(f"<div class='wf-ph-flow-title'>{title}</div>")
        if when:
            parts.append(f"<div class='wf-ph-flow-when'>{html_escape(when)}</div>")
        if reason:
            parts.append(f"<div class='wf-ph-flow-reason'>{html_escape(str(reason))}</div>")
        parts.append("</div></div>")
    parts.append("</div>")
    return "".join(parts)


def _ph_render_card(row):
    if not isinstance(row, dict):
        return ""
    title = html_escape(str(row.get("product_title") or "Product"))
    oid = row.get("oid") or row.get("id") or ""
    pay = html_escape(str(row.get("payment_status_string") or row.get("payment_status") or "—"))
    current_status = html_escape(
        str(row.get("current_order_status") or row.get("delivery_status_string") or "—")
    )
    order_date = html_escape(str(row.get("date") or ""))
    grand_total = html_escape(str(row.get("grand_total") or ""))
    badge_cls = _ph_status_badge_class(
        str(row.get("current_order_status") or ""),
        str(row.get("delivery_status_string") or ""),
    )
    slug = row.get("product_slug") or ""
    img_raw = row.get("product_img") or ""
    img_url = ""
    if img_raw:
        img_url = PH_IMG_BASE + str(img_raw).lstrip("/")
    link = f"https://welfog.com/product_details/{slug}" if slug else "https://welfog.com"

    if img_url:
        img_html = f"<img class='wf-ph-img' src='{html_escape(img_url)}' alt=''>"
    else:
        img_html = "<div class='wf-ph-img wf-ph-img--placeholder' aria-hidden='true'></div>"

    oid_raw = html_escape(str(oid)) if oid != "" else ""
    oid_attr = f" data-ph-oid=\"{oid_raw}\"" if oid_raw else ""
    panel_id = f"ph-track-{oid_raw}" if oid_raw else "ph-track-unknown"
    flow_html = _ph_order_flow_timeline_html(row)

    meta_extra = ""
    if order_date:
        meta_extra += f"<div class='wf-ph-meta'>Ordered: {order_date}</div>"
    if grand_total:
        meta_extra += f"<div class='wf-ph-meta'>Total: <b>{grand_total}</b></div>"

    return (
        f"<div class='wf-ph-card'{oid_attr}>"
        "<div class='wf-ph-card-head'>"
        f"<button type='button' class='wf-ph-track-btn' aria-expanded='false' "
        f"aria-controls='{panel_id}'>Track order</button>"
        "<div class='wf-ph-card-row'>"
        f"{img_html}"
        "<div class='wf-ph-card-body'>"
        f"<div class='wf-ph-name'>{title}</div>"
        f"<div class='wf-ph-meta'>Order ID: <b>{html_escape(str(oid))}</b></div>"
        f"<div class='wf-ph-meta'>Payment: {pay}</div>"
        f"<div class='wf-ph-meta wf-ph-meta--status'>"
        "<span class='wf-ph-status-label'>Order status:</span> "
        f"<span class='wf-ph-status-badge {badge_cls}'>{current_status}</span>"
        "</div>"
        f"{meta_extra}"
        f"<a class='wf-ph-link' href='{html_escape(link)}' target='_blank' rel='noopener noreferrer'>View product</a>"
        "</div></div></div>"
        f"<div class='wf-ph-track-panel' id='{panel_id}' hidden>"
        "<div class='wf-ph-track-panel-title'>Order progress</div>"
        f"{flow_html}"
        "</div>"
        "</div>"
    )


def _ph_tail_html(has_more, next_page):
    """Footer below the scroll area — always visible without scrolling past all cards."""
    if not has_more:
        return ""
    return (
        "<div class='wf-ph-tail'>"
        f"<button type='button' class='wf-ph-more' data-next-page=\"{next_page}\">"
        "<span class='wf-ph-more__label'>View more</span></button>"
        "</div>"
    )


def format_purchase_history_reply(user_id, page=1, append_only=False):
    """
    HTML block for chat: up to PH_PER_PAGE orders + optional View more.
    Wrapped in .wf-ph-root so "View more" can append into the same bubble.
    append_only: kept for compatibility; prefer format_purchase_history_append_payload for pagination.
    """
    uid = str(user_id).strip()
    if uid == "STATIC_USER_001":
        inner = (
            "<div class='wf-ph-msg'>Please open this chat from the Welfog app while logged in, "
            "or add <b>?user_id=YOUR_ID</b> to this page URL, so we can load your order history.</div>"
        )
        return f"<div class='wf-ph-root'>{inner}</div>"

    display_rows, has_more = purchase_history_slice(uid, page)

    if not display_rows:
        if append_only:
            inner = "<div class='wf-ph-msg'>No more orders to show.</div>"
        else:
            inner = "<div class='wf-ph-msg'>No orders found in your history yet.</div>"
        return f"<div class='wf-ph-root'>{inner}</div>"

    parts = []
    if not append_only:
        parts.append("<div class='wf-ph-title'><b>Your order history</b></div>")
    parts.append("<div class='wf-ph-scroll'>")
    for row in display_rows:
        parts.append(_ph_render_card(row))
    parts.append("</div>")
    parts.append(_ph_tail_html(has_more, page + 1))

    inner = "".join(parts)
    return f"<div class='wf-ph-root'>{inner}</div>"


def _oid_render_row(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    oid = row.get("oid") or row.get("id") or ""
    title = html_escape(str(row.get("product_title") or "Order")[:80])
    if len(str(row.get("product_title") or "")) > 80:
        title += "…"
    status = html_escape(str(row.get("current_order_status") or row.get("delivery_status_string") or "—"))
    date = html_escape(str(row.get("date") or ""))
    oid_s = html_escape(str(oid))
    meta = f"{title}"
    if date:
        meta += f" · {date}"
    meta += f" · {status}"
    return (
        f"<div class='wf-oid-row' data-oid=\"{oid_s}\">"
        f"<span class='wf-oid-id'>{oid_s}</span>"
        f"<span class='wf-oid-meta'>{meta}</span>"
        "</div>"
    )


def format_order_ids_reply(user_id, page: int = 1) -> str:
    """Compact order-ID list from purchase-history API (for order-id chat intent)."""
    uid = str(user_id).strip()
    if uid == "STATIC_USER_001":
        inner = (
            "<div class='wf-oid-msg'>Please open this chat while logged in, "
            "or add <b>?user_id=YOUR_ID</b> to the URL.</div>"
        )
        return f"<div class='wf-oid-root'>{inner}</div>"

    display_rows, has_more = purchase_history_slice(uid, page)
    if not display_rows:
        inner = "<div class='wf-oid-msg'>No orders found — place an order on Welfog first.</div>"
        return f"<div class='wf-oid-root'>{inner}</div>"

    parts = [
        "<div class='wf-oid-title'><b>Your order IDs</b></div>",
        "<div class='wf-oid-hint'>Tap an ID when tracking — or paste it here for live status.</div>",
        "<div class='wf-oid-scroll'>",
    ]
    for row in display_rows:
        parts.append(_oid_render_row(row))
    if has_more:
        parts.append(
            f"<div class='wf-oid-tail'>"
            f"<button type='button' class='wf-ph-more wf-oid-more' data-next-page=\"{page + 1}\">"
            "<span class='wf-ph-more__label'>View more</span></button></div>"
        )
    parts.append("</div>")
    return f"<div class='wf-oid-root'>{''.join(parts)}</div>"


def format_purchase_history_append_payload(user_id, page):
    """
    JSON-friendly pieces for in-place append (same chat bubble): new cards + footer button.
    page should be >= 2 when loading the next API page after the first screen.
    """
    uid = str(user_id).strip()
    if uid == "STATIC_USER_001":
        return {
            "cards_html": "<div class='wf-ph-msg'>Please open this chat while logged in or pass user_id in the URL.</div>",
            "tail_html": "",
        }

    display_rows, has_more = purchase_history_slice(uid, page)

    if not display_rows:
        return {
            "cards_html": "<div class='wf-ph-msg'>No more orders to show.</div>",
            "tail_html": "",
        }

    cards_html = "".join(_ph_render_card(row) for row in display_rows)
    tail_html = _ph_tail_html(has_more, page + 1)
    return {"cards_html": cards_html, "tail_html": tail_html}


def fetch_wishlist(user_id, page=1, per_page=None):
    """GET /api/v2/wishlists/{user_id} — one API page."""
    uid = str(user_id or "").strip()
    if not uid:
        return None
    url = WL_API_TEMPLATE.format(user_id=uid)
    params = {"page": max(1, int(page)), "per_page": per_page or WL_API_PAGE_SIZE}
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return None


def _wishlist_rows_from_payload(payload):
    """Extract list of wishlist row dicts from assorted API shapes."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "wishlist", "products", "list", "rows"):
            block = data.get(key)
            if isinstance(block, list):
                return block
    for key in ("wishlist", "items", "products"):
        block = payload.get(key)
        if isinstance(block, list):
            return block
    return []


def _normalize_wishlist_row(row):
    if not isinstance(row, dict):
        return None
    prod = row.get("product") if isinstance(row.get("product"), dict) else {}
    if not prod and (row.get("link") or row.get("name") or row.get("thumbnail_image")):
        prod = row
    if not prod:
        return None
    link = str(prod.get("link") or prod.get("slug") or "").strip()
    thumb = prod.get("thumbnail_image") or prod.get("image") or prod.get("img") or ""
    return {
        "wish_id": row.get("id") or row.get("wish_id"),
        "product_id": prod.get("id") or prod.get("product_id"),
        "link": link,
        "name": str(prod.get("name") or prod.get("title") or "Product").strip(),
        "thumbnail_image": str(thumb).strip(),
        "base_price": str(prod.get("base_price") or prod.get("mrp") or "").strip(),
        "selling_price": str(prod.get("selling_price") or prod.get("price") or "").strip(),
        "stock": prod.get("stock"),
        "rating": prod.get("rating"),
    }


def _load_all_wishlist_items(uid: str):
    """Fetch every page from wishlists API until exhausted."""
    uid = str(uid).strip()
    seen_pids = set()
    items = []
    page = 1
    while page <= 30:
        payload = fetch_wishlist(uid, page=page, per_page=WL_API_PAGE_SIZE)
        if not isinstance(payload, dict) or not payload.get("success"):
            break
        rows = _wishlist_rows_from_payload(payload)
        if not rows:
            break
        added = 0
        for row in rows:
            norm = _normalize_wishlist_row(row)
            if not norm:
                continue
            pid = norm.get("product_id") or norm.get("wish_id") or norm.get("link")
            key = str(pid)
            if key in seen_pids:
                continue
            seen_pids.add(key)
            items.append(norm)
            added += 1
        if len(rows) < WL_API_PAGE_SIZE or added == 0:
            break
        page += 1
    return items


def _ensure_wishlist_items(uid: str, force_refresh: bool = False):
    """Load and cache normalized wishlist rows for this user."""
    uid = str(uid).strip()
    now = time.time()
    with _WL_CACHE_LOCK:
        if force_refresh:
            _WL_CACHE.pop(uid, None)
        slot = _WL_CACHE.get(uid)
        if slot and (now - slot["ts"]) > _WL_CACHE_TTL_SEC:
            _WL_CACHE.pop(uid, None)
            slot = None
        if slot and isinstance(slot.get("items"), list):
            return slot["items"]
    items = _load_all_wishlist_items(uid)
    with _WL_CACHE_LOCK:
        _WL_CACHE[uid] = {"items": items, "ts": now}
    return items


def wishlist_slice(uid: str, page: int):
    """Returns (display_rows, has_more) for 1-based page."""
    uid = str(uid).strip()
    page = max(1, int(page))
    rows = _ensure_wishlist_items(uid)
    start = (page - 1) * WL_PER_PAGE
    end = page * WL_PER_PAGE
    chunk = rows[start:end]
    has_more = len(rows) > end
    return chunk, has_more


def _wl_render_card(row):
    if not isinstance(row, dict):
        return ""
    title = html_escape(str(row.get("name") or "Product"))
    slug = row.get("link") or ""
    pid = row.get("product_id") or ""
    rating = row.get("rating")
    stock = row.get("stock")
    img_raw = row.get("thumbnail_image") or ""
    img_url = PH_IMG_BASE + str(img_raw).lstrip("/") if img_raw else ""
    link = f"https://welfog.com/product_details/{slug}" if slug else "https://welfog.com"
    stock_txt = "In stock" if stock not in (0, "0", None, False) else "Out of stock"
    rating_txt = f"★ {rating}" if rating not in (None, "", 0) else ""

    if img_url:
        img_html = f"<img class='wf-wl-img' src='{html_escape(img_url)}' alt=''>"
    else:
        img_html = "<div class='wf-wl-img wf-wl-img--placeholder' aria-hidden='true'></div>"

    pid_raw = html_escape(str(pid)) if pid != "" else ""
    pid_attr = f' data-wl-pid="{pid_raw}"' if pid_raw else ""
    sell_esc = html_escape(str(row.get("selling_price") or "—"))
    base_esc = html_escape(str(row.get("base_price") or ""))
    if base_esc and base_esc != sell_esc:
        price_line = (
            f"<span class='wf-wl-price--sale'>{sell_esc}</span> "
            f"<span class='wf-wl-price--base'>{base_esc}</span>"
        )
    else:
        price_line = sell_esc

    meta_bits = [f"<div class='wf-wl-meta'>{price_line}</div>"]
    if rating_txt:
        meta_bits.append(f"<div class='wf-wl-meta'>{html_escape(rating_txt)}</div>")
    meta_bits.append(f"<div class='wf-wl-meta wf-wl-meta--stock'>{html_escape(stock_txt)}</div>")

    return (
        f"<div class='wf-wl-card'{pid_attr}>"
        "<div class='wf-wl-card-row'>"
        f"{img_html}"
        "<div class='wf-wl-card-body'>"
        f"<div class='wf-wl-name'>{title}</div>"
        + "".join(meta_bits)
        + f"<a class='wf-wl-link' href='{html_escape(link)}' target='_blank' rel='noopener noreferrer'>View product</a>"
        "</div></div></div>"
    )


def _wl_tail_html(has_more, next_page):
    if not has_more:
        return ""
    return (
        "<div class='wf-wl-tail'>"
        f"<button type='button' class='wf-wl-more' data-next-page=\"{next_page}\">"
        "<span class='wf-wl-more__label'>View more</span></button>"
        "</div>"
    )


def format_wishlist_reply(user_id, page=1, append_only=False):
    """HTML block for chat wishlist list + View more."""
    uid = str(user_id).strip()
    if uid == "STATIC_USER_001":
        inner = (
            "<div class='wf-wl-msg'>Please open this chat from the Welfog app while logged in, "
            "or add <b>?user_id=YOUR_ID</b> to this page URL so we can load your wishlist.</div>"
        )
        return f"<div class='wf-wl-root'>{inner}</div>"

    all_rows = _ensure_wishlist_items(uid, force_refresh=not append_only)
    display_rows, has_more = wishlist_slice(uid, page)
    if not display_rows:
        if append_only:
            inner = "<div class='wf-wl-msg'>No more saved items to show.</div>"
        else:
            inner = "<div class='wf-wl-msg'>Your wishlist is empty — save products you like from the app.</div>"
        return f"<div class='wf-wl-root'>{inner}</div>"

    total = len(all_rows)
    parts = []
    if not append_only:
        count_lbl = f" ({total} items)" if total != 1 else " (1 item)"
        parts.append(f"<div class='wf-wl-title'><b>Your wishlist</b>{html_escape(count_lbl)}</div>")
    parts.append("<div class='wf-wl-scroll'>")
    for row in display_rows:
        parts.append(_wl_render_card(row))
    parts.append(_wl_tail_html(has_more, page + 1))
    parts.append("</div>")
    return f"<div class='wf-wl-root'>{''.join(parts)}</div>"


def format_wishlist_append_payload(user_id, page):
    """Fragments for in-bubble wishlist pagination (page >= 2)."""
    uid = str(user_id).strip()
    if uid == "STATIC_USER_001":
        return {
            "cards_html": "<div class='wf-wl-msg'>Please open this chat while logged in or pass user_id in the URL.</div>",
            "tail_html": "",
        }
    display_rows, has_more = wishlist_slice(uid, page)
    if not display_rows:
        return {
            "cards_html": "<div class='wf-wl-msg'>No more saved items to show.</div>",
            "tail_html": "",
        }
    cards_html = "".join(_wl_render_card(row) for row in display_rows)
    tail_html = _wl_tail_html(has_more, page + 1)
    return {"cards_html": cards_html, "tail_html": tail_html}


def format_wishlist_page_html(user_id, page=1):
    """Full-page wishlist markup (standalone /wishlist route)."""
    uid = str(user_id).strip()
    block = format_wishlist_reply(uid, page=page, append_only=False)
    return block.replace("wf-wl-more", "wf-wl-more wf-wl-more--page")


def _normalize_track_order_id(order_id: str):
    raw = str(order_id or "").strip()
    if re.fullmatch(r"\d+", raw):
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw.upper()


def _is_logged_in_customer_id(user_id) -> bool:
    uid = str(user_id or "").strip()
    return bool(uid) and uid != "STATIC_USER_001"


def _order_id_match_variants(order_id: str) -> set:
    """Normalized order-id tokens for comparison against purchase-history rows."""
    raw = str(order_id or "").strip()
    if not raw:
        return set()
    variants = {raw, raw.upper(), raw.lower()}
    if re.fullmatch(r"\d+", raw):
        try:
            variants.add(str(int(raw)))
        except ValueError:
            pass
    return variants


def _row_order_id_values(row) -> set:
    if not isinstance(row, dict):
        return set()
    out = set()
    for key in ("oid", "id", "order_id", "orderId", "order_code", "code"):
        val = row.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        out.add(s)
        out.add(s.upper())
        out.add(s.lower())
        if re.fullmatch(r"\d+", s):
            try:
                out.add(str(int(s)))
            except ValueError:
                pass
    return out


def _purchase_history_reported_total(uid: str):
    """Best-effort order count from purchase-history meta (None if unknown)."""
    data = fetch_purchase_history(uid, page=1)
    if not isinstance(data, dict):
        return None
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return None
    try:
        return int(meta.get("total"))
    except (TypeError, ValueError):
        return None


def _track_api_denies_owner(data: dict) -> bool:
    """True when track API explicitly says this user cannot view the order."""
    if not isinstance(data, dict):
        return False
    result = str(data.get("result") or "").lower()
    if result in {"forbidden", "unauthorized", "denied", "not_allowed", "not_owner", "not_owned"}:
        return True
    if data.get("owned") is False or data.get("is_owner") is False:
        return True
    msg = str(data.get("message") or data.get("error") or "").lower()
    return any(
        token in msg
        for token in (
            "not belong",
            "does not belong",
            "not your order",
            "not linked",
            "unauthorized",
            "forbidden",
            "access denied",
        )
    )


def _track_api_confirms_owner(data: dict) -> bool:
    """True when track API explicitly confirms ownership (future backends)."""
    if not isinstance(data, dict) or data.get("result") != "ok":
        return False
    if data.get("owned") is True or data.get("is_owner") is True:
        return True
    owner = data.get("user_id") or data.get("customer_id") or data.get("buyer_id")
    return owner is not None


def check_order_ownership(user_id: str, order_id: str, max_batches: int = 8) -> str:
    """
    Purchase-history ownership check.

    Returns:
      - 'owned' — order id found in this user's purchase history
      - 'not_owned' — user has purchase history but this order id is not in it
      - 'unverified' — no purchase-history rows (cannot prove ownership client-side)
    """
    uid = str(user_id or "").strip()
    if not _is_logged_in_customer_id(uid):
        return "unverified"
    targets = _order_id_match_variants(order_id)
    if not targets:
        return "unverified"

    reported_total = _purchase_history_reported_total(uid)
    offset = 0
    batches = 0
    seen_first_at_offset = None
    saw_any_row = False

    while batches < max_batches:
        batch_pack = _fetch_purchase_history_at_offset(uid, offset)
        if not batch_pack:
            break
        batch, _reported = batch_pack
        if not batch:
            break
        saw_any_row = True
        first_rid = _row_id(batch[0])
        if offset > 0 and first_rid is not None and first_rid == seen_first_at_offset:
            break
        if offset == 0:
            seen_first_at_offset = first_rid

        for row in batch:
            if targets & _row_order_id_values(row):
                return "owned"

        if len(batch) < PH_PER_PAGE:
            break
        offset += len(batch)
        batches += 1

    if saw_any_row:
        return "not_owned"
    if reported_total is not None and reported_total > 0:
        return "not_owned"
    return "unverified"


def order_belongs_to_user(user_id: str, order_id: str, max_batches: int = 8) -> bool:
    return check_order_ownership(user_id, order_id, max_batches=max_batches) == "owned"


def fetch_welfog_order_tracking(order_id: str, user_id=None):
    """Live order status from Welfog OneDelivery track API (POST JSON)."""
    _t0 = time.perf_counter()
    try:
        oid = _normalize_track_order_id(order_id)
        if not oid:
            return None
        payload = {"order_id": oid}
        uid = str(user_id or "").strip()
        if _is_logged_in_customer_id(uid):
            try:
                payload["user_id"] = int(uid) if uid.isdigit() else uid
            except ValueError:
                payload["user_id"] = uid
        try:
            res = requests.post(
                WELFOG_TRACK_API_URL,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            if res.status_code != 200:
                return None
            data = res.json()
            if isinstance(data, dict) and data.get("result") == "ok":
                return data
        except Exception as e:
            from utils.reasoning_log import log_reasoning

            log_reasoning(f"Order track API error: {e}")
        return None
    finally:
        _record_api_elapsed(_t0)


def fetch_welfog_order_tracking_for_user(order_id: str, user_id: str):
    """
    Fetch tracking only when the order belongs to the logged-in customer.

    Returns (data, error_code):
      - data: API payload when allowed
      - error_code: None | 'login_required' | 'not_owned' | 'not_found'
    """
    uid = str(user_id or "").strip()
    if not _is_logged_in_customer_id(uid):
        return None, "login_required"

    ownership = check_order_ownership(uid, order_id)
    data = fetch_welfog_order_tracking(order_id, user_id=uid)

    if ownership == "owned":
        if not data:
            return None, "not_found"
        return data, None

    if data and _track_api_denies_owner(data):
        return None, "not_owned"

    if ownership == "not_owned":
        if data:
            return None, "not_owned"
        return None, "not_found"

    # unverified: purchase-history empty — rely on track API ownership signals when added
    if data and _track_api_confirms_owner(data):
        return data, None
    if data:
        return None, "unverified"
    return None, "not_found"


def filter_return_request_for_customer(data: dict | None) -> dict:
    """Expose only customer-relevant refund/return fields (no internal ids)."""
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for key in _RETURN_REQUEST_CUSTOMER_FIELDS:
        val = data.get(key)
        if val is not None and val != "":
            out[key] = val
    if data.get("result") is False and data.get("message") and "message" not in out:
        out["message"] = data.get("message")
    return out


def log_refund_status_pipeline(
    *,
    intent: str,
    order_id: str,
    source: str,
    filtered_response: dict | None,
) -> None:
    import json

    from utils.reasoning_log import chat_log, log_reasoning

    fr = json.dumps(filtered_response or {}, ensure_ascii=False, default=str)[:600]
    msg = (
        f"[refund-status] intent={intent} order_id={order_id} "
        f"source={source} filtered_response={fr}"
    )
    log_reasoning(msg)
    chat_log(msg)


def fetch_welfog_return_request(order_id: str):
    """GET return-request / refund status for one order."""
    oid = _normalize_track_order_id(order_id)
    if not oid:
        return None
    url = WELFOG_RETURN_REQUEST_API_URL.format(order_id=oid)
    try:
        res = requests.get(
            url,
            timeout=_LIVE_ORDER_API_TIMEOUT_SEC,
            headers={"Accept": "application/json"},
        )
        if res.status_code != 200:
            return None
        data = res.json()
        return data if isinstance(data, dict) else None
    except Exception as e:
        from utils.reasoning_log import log_reasoning

        log_reasoning(f"Return-request API error: {e}")
        return None


def fetch_welfog_return_request_for_user(order_id: str, user_id: str):
    """
    Refund/return status only when order belongs to the logged-in customer.

    Returns (filtered_payload, error_code):
      - filtered_payload: customer-safe fields when allowed
      - error_code: None | 'login_required' | 'not_owned' | 'unverified' | 'not_found'
    """
    uid = str(user_id or "").strip()
    if not _is_logged_in_customer_id(uid):
        return None, "login_required"

    ownership = check_order_ownership(uid, order_id)
    if ownership == "not_owned":
        return None, "not_owned"
    if ownership != "owned":
        return None, "unverified"

    raw = fetch_welfog_return_request(order_id)
    if raw is None:
        return None, "not_found"

    filtered = filter_return_request_for_customer(raw)
    log_refund_status_pipeline(
        intent="refund",
        order_id=str(_normalize_track_order_id(order_id) or order_id),
        source="return_request_api",
        filtered_response=filtered,
    )
    return filtered, None


def format_refund_status_reply(
    data: dict,
    order_id: str,
    lang: str = "en",
    *,
    filtered: dict | None = None,
) -> str:
    """HTML card from return-request API (filtered customer fields only)."""
    payload = filtered if isinstance(filtered, dict) else filter_return_request_for_customer(data)
    if not payload and isinstance(data, dict):
        payload = filter_return_request_for_customer(data)
    if not payload:
        return ""

    lang = (lang or "en").lower().strip()
    use_hinglish = lang == "hinglish"
    oid = html_escape(str(payload.get("order_id") or order_id))
    status_raw = (payload.get("refund_status") or "").strip()
    message = (payload.get("message") or "").strip()
    amount = payload.get("refund_amount")
    bank_valid = payload.get("isbankvalid")

    labels = {
        "title": "Refund / return status",
        "order_id": "Order ID",
        "status": "Refund status",
        "amount": "Refund amount",
        "bank": "Bank details",
        "none": "No refund record found for this order yet.",
    }
    if use_hinglish:
        labels.update(
            {
                "title": "Refund / return status",
                "order_id": "Order ID",
                "status": "Refund status",
                "amount": "Refund amount",
                "bank": "Bank details",
                "none": "Is order ke liye abhi refund record nahi mila.",
            }
        )

    wrap = "color:#333;line-height:1.55;font-size:14px;"
    parts = [
        f"<div style='{wrap}'>",
        f"<div style='font-size:16px;font-weight:700;margin-bottom:10px;'>{labels['title']}</div>",
        f"<div style='background:#f8f9fa;border-radius:10px;padding:12px 14px;border:1px solid #eee;'>",
        f"<div><b>{labels['order_id']}:</b> {oid}</div>",
    ]

    if status_raw:
        parts.append(
            f"<div style='margin-top:6px;'><b>{labels['status']}:</b> "
            f"<span style='color:#ff7a00;font-weight:600;'>{html_escape(status_raw)}</span></div>"
        )
    elif message:
        parts.append(
            f"<div style='margin-top:8px;'>{html_escape(message)}</div>"
        )
    else:
        parts.append(f"<div style='margin-top:8px;'>{labels['none']}</div>")

    if amount is not None and str(amount).strip() not in ("", "0", "0.0"):
        amt_disp = html_escape(str(amount))
        if not str(amt_disp).startswith("Rs") and not str(amt_disp).startswith("₹"):
            amt_disp = f"Rs {amt_disp}"
        parts.append(f"<div style='margin-top:6px;'><b>{labels['amount']}:</b> {amt_disp}</div>")

    if bank_valid is not None:
        bank_txt = "Valid" if bank_valid else "Needs update"
        if use_hinglish:
            bank_txt = "Sahi hain" if bank_valid else "Update chahiye"
        parts.append(
            f"<div style='margin-top:6px;'><b>{labels['bank']}:</b> {html_escape(str(bank_txt))}</div>"
        )

    parts.append("</div></div>")
    return "".join(parts)


def _parse_track_date_only(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _compute_expected_delivery_display(order_date_str: str, minutes_raw, cancelled: bool, lang: str = "en"):
    """
    API `expected_delivery` is minutes from order date — convert to calendar ETA.
    """
    if cancelled:
        if lang == "hinglish":
            return "Order cancel ho chuka hai — expected delivery apply nahi hoti."
        return "Not applicable — order was cancelled."

    try:
        minutes = int(minutes_raw)
    except (TypeError, ValueError):
        return ""

    if minutes <= 0:
        return ""

    order_day = _parse_track_date_only(order_date_str)
    if not order_day:
        return ""

    eta_day = order_day + timedelta(minutes=minutes)
    days_approx = minutes / 1440.0
    if days_approx >= 1:
        days_label = f"~{days_approx:.1f} days" if days_approx < 2 else f"~{int(round(days_approx))} days"
    else:
        hours = max(1, int(round(minutes / 60)))
        days_label = f"~{hours} hour(s)"

    eta_fmt = eta_day.strftime("%d %b %Y")
    if lang == "hinglish":
        return f"<b>{eta_fmt}</b> tak (order date + {days_label}, total {minutes:,} minutes)"
    return f"<b>{eta_fmt}</b> (about {days_label} from order date)"

def _format_track_step_datetime(iso_s: str) -> str:
    if not iso_s:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso_s).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return str(iso_s)[:16].replace("T", " ")


def _status_label(status: str, lang: str) -> str:
    s = (status or "").replace("_", " ").strip().title()
    if lang != "hinglish":
        return s  # English base; send_reply localizes for pa/ta/hi/mr/…
    mapping = {
        "Cancelled": "Cancel ho gaya",
        "Delivered": "Deliver ho gaya",
        "Shipped": "Ship ho gaya",
        "Processing": "Process ho raha hai",
        "Pending": "Pending",
        "Confirmed": "Confirm ho gaya",
    }
    return mapping.get(s, s)


def _format_track_payment_display(data: dict) -> str:
    """Payment line for track card — never strip closing ')' from (cash_on_delivery)."""
    status = str(data.get("payment_status") or "").strip()
    ptype = str(data.get("payment_type") or "").strip().replace("_", " ")
    if status and ptype:
        return f"{status} ({ptype})"
    if status:
        return status
    if ptype:
        return ptype
    return "—"


def format_order_tracking_reply(data: dict, order_id: str, lang: str = "en") -> str:
    """HTML summary card from welfog_track API payload."""
    if not data or data.get("result") != "ok":
        return ""

    lang = (lang or "en").lower().strip()
    use_hinglish = lang == "hinglish"

    oid = html_escape(str(data.get("order_id") or order_id))
    status_raw = (data.get("current_order_status") or data.get("status") or "").strip()
    status_lower = status_raw.lower()
    cancelled = status_lower == "cancelled" or status_lower == "canceled"

    labels = {
        "title": "Live order tracking",
        "order_id": "Order ID",
        "status": "Current status",
        "order_date": "Order date",
        "expected": "Expected delivery by",
        "payment": "Payment",
        "product": "Product",
        "timeline": "Order timeline",
    }
    if use_hinglish:
        labels.update(
            {
                "title": "Aapke order ka live status",
                "order_id": "Order ID",
                "status": "Abhi status",
                "order_date": "Order date",
                "expected": "Expected delivery",
                "payment": "Payment",
                "product": "Product",
                "timeline": "Order steps",
            }
        )

    order_date = html_escape(str(data.get("order_date") or "—"))
    eta_html = _compute_expected_delivery_display(
        data.get("order_date"), data.get("expected_delivery"), cancelled, lang
    )
    status_html = html_escape(_status_label(status_raw, lang))
    payment = html_escape(_format_track_payment_display(data))
    product_title = html_escape(str(data.get("product_title") or "—"))
    img_path = (data.get("product_img") or "").strip()
    img_url = f"{TRACK_PRODUCT_IMG_BASE}{img_path.lstrip('/')}" if img_path else ""

    wrap = "color:#333;line-height:1.55;font-size:14px;"
    html_parts = [
        f"<div style='{wrap}'>",
        f"<div style='font-size:16px;font-weight:700;margin-bottom:10px;'>{labels['title']}</div>",
        f"<div style='background:#f8f9fa;border-radius:10px;padding:12px 14px;border:1px solid #eee;'>",
        f"<div><b>{labels['order_id']}:</b> {oid}</div>",
        f"<div style='margin-top:6px;'><b>{labels['status']}:</b> <span style='color:#ff7a00;font-weight:600;'>{status_html}</span></div>",
        f"<div style='margin-top:6px;'><b>{labels['order_date']}:</b> {order_date}</div>",
    ]
    if eta_html:
        html_parts.append(f"<div style='margin-top:6px;'><b>{labels['expected']}:</b> {eta_html}</div>")
    html_parts.append(f"<div style='margin-top:6px;'><b>{labels['payment']}:</b> {payment}</div>")
    html_parts.append("</div>")

    if product_title and product_title != "—":
        html_parts.append(f"<div style='margin-top:12px;'><b>{labels['product']}:</b> {product_title}</div>")
        if img_url:
            html_parts.append(
                f"<div style='margin-top:8px;'><img src='{html_escape(img_url)}' alt='' "
                f"style='max-width:120px;max-height:120px;border-radius:8px;border:1px solid #eee;'/></div>"
            )

    flow = (
        (data.get("order_status_detail") or {})
        .get("order_status", {})
        .get("order_flow", {})
    )
    if isinstance(flow, dict) and flow:
        html_parts.append(
            f"<div style='margin-top:14px;font-weight:600;margin-bottom:8px;'>{labels['timeline']}</div>"
        )
        html_parts.append("<div style='border-left:3px solid #ff7a00;padding-left:12px;margin-left:4px;'>")
        keys = list(_ORDER_FLOW_STEP_ORDER)
        if cancelled and "cancelled" in flow and "cancelled" not in keys:
            keys.append("cancelled")
        for key in keys:
            step = flow.get(key)
            if not isinstance(step, dict):
                continue
            done = bool(step.get("status"))
            title = html_escape(str(step.get("title") or key.replace("_", " ").title()))
            when = _format_track_step_datetime(step.get("date"))
            reason = step.get("reason")
            dot_color = "#22c55e" if done else "#cbd5e1"
            text_color = "#333" if done else "#888"
            html_parts.append(
                f"<div style='margin-bottom:10px;color:{text_color};'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                f"background:{dot_color};margin-right:8px;vertical-align:middle;'></span>"
                f"<b>{title}</b>"
            )
            if when:
                html_parts.append(f"<span style='font-size:12px;color:#666;'> — {html_escape(when)}</span>")
            if reason:
                html_parts.append(
                    f"<div style='font-size:12px;color:#666;margin-left:18px;margin-top:2px;'>"
                    f"{html_escape(str(reason))}</div>"
                )
            html_parts.append("</div>")
        html_parts.append("</div>")

    track_code = data.get("tracking_code")
    if track_code:
        html_parts.append(
            f"<div style='margin-top:10px;font-size:13px;'><b>Tracking code:</b> "
            f"{html_escape(str(track_code))}</div>"
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def _details_api_denies(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    result = str(data.get("result") or "").lower()
    if result in {"forbidden", "unauthorized", "denied", "not_allowed", "not_owner", "not_owned", "error", "fail"}:
        return True
    if data.get("success") is False:
        return True
    if data.get("owned") is False or data.get("is_owner") is False:
        return True
    msg = str(data.get("message") or data.get("error") or data.get("msg") or "").lower()
    return any(
        token in msg
        for token in (
            "not belong",
            "does not belong",
            "not your order",
            "not linked",
            "unauthorized",
            "forbidden",
            "access denied",
            "not found",
        )
    )


def _extract_ph_details_row(raw) -> Optional[dict]:
    """Normalize purchase-history-details payload to a single order row dict."""
    if not isinstance(raw, dict):
        return None
    if _details_api_denies(raw):
        return None
    for key in ("data", "order", "order_detail", "order_details", "details"):
        block = raw.get(key)
        if isinstance(block, dict) and (
            block.get("product_title")
            or block.get("oid")
            or block.get("id")
            or block.get("order_id")
        ):
            return block
    if raw.get("product_title") or raw.get("oid") or raw.get("id") or raw.get("order_id"):
        return raw
    rows = raw.get("data")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return None


def fetch_purchase_history_details(order_id: str, user_id=None, *, fast: bool = False):
    """GET purchase-history-details for one order (ownership enforced server-side via user_id)."""
    oid = str(order_id or "").strip()
    uid = str(user_id or "").strip()
    if not oid or not uid:
        return None
    url = f"{PH_DETAILS_API_BASE}/{oid}"
    param_sets = (
        {"user_id": uid},
    ) if fast else (
        {"user_id": uid},
        {"user_id": int(uid)} if uid.isdigit() else {"user_id": uid},
    )
    timeout = _LIVE_ORDER_API_TIMEOUT_SEC if fast else 10
    for params in param_sets:
        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code in (401, 403, 404):
                return None
            if res.status_code != 200:
                continue
            raw = res.json()
            if _details_api_denies(raw):
                return None
            row = _extract_ph_details_row(raw)
            if row:
                return row
        except Exception as e:
            from utils.reasoning_log import log_reasoning

            log_reasoning(f"Order details API error: {e}")
            if fast or _is_transient_network_error(e):
                return None
            continue
    return None


def fetch_purchase_history_details_for_user(order_id: str, user_id: str, *, fast: bool = False):
    """
    Returns (row, error_code):
      error_code: None | 'login_required' | 'not_owned' | 'not_found'
    """
    _t0 = time.perf_counter()
    try:
        uid = str(user_id or "").strip()
        if not _is_logged_in_customer_id(uid):
            return None, "login_required"
        row = fetch_purchase_history_details(order_id, user_id=uid, fast=fast)
        if row:
            if not fast:
                row = _enrich_order_details_row(row, uid)
            return row, None
        if fast:
            return None, "not_found"
        if check_order_ownership(uid, order_id, max_batches=3) == "not_owned":
            return None, "not_owned"
        return None, "not_found"
    finally:
        _record_api_elapsed(_t0)


def _enrich_order_details_row(row: dict, user_id: str) -> dict:
    """Fill product title/slug from purchase-history when details API omits them."""
    if not isinstance(row, dict):
        return row
    if _ph_details_product_title(row) != "Your order item":
        return row
    oid = str(row.get("oid") or row.get("id") or "").strip()
    if not oid:
        return row
    targets = _order_id_match_variants(oid)
    offset = 0
    for _ in range(12):
        batch_pack = _fetch_purchase_history_at_offset(str(user_id).strip(), offset)
        if not batch_pack:
            break
        batch, _reported = batch_pack
        if not batch:
            break
        for ph_row in batch:
            if targets & _row_order_id_values(ph_row):
                out = dict(row)
                if ph_row.get("product_title"):
                    out["product_title"] = ph_row["product_title"]
                if ph_row.get("product_slug"):
                    out["product_slug"] = ph_row["product_slug"]
                return out
        if len(batch) < PH_PER_PAGE:
            break
        offset += len(batch)
    return row


def build_order_invoice_url(order_id: str) -> str:
    from urllib.parse import urlencode

    oid = str(order_id or "").strip()
    return f"{INVOICE_API_BASE}?{urlencode({'order_id': oid})}"


def _ph_details_img_url(row: dict) -> str:
    img_raw = (row or {}).get("product_img") or ""
    if not img_raw:
        return ""
    return PH_IMG_BASE + str(img_raw).lstrip("/")


def _ph_details_payment_line(row: dict) -> str:
    pay = str(row.get("payment_status_string") or row.get("payment_status") or "").strip()
    ptype = str(row.get("payment_type") or row.get("payment_method") or row.get("payment_mode") or "").strip().replace("_", " ")
    if pay and ptype:
        return f"{pay} · {ptype}"
    return pay or ptype or "—"


def _ph_details_product_title(row: dict) -> str:
    for key in ("product_title", "product_name", "name", "title"):
        val = str((row or {}).get(key) or "").strip()
        if val and val.lower() not in ("—", "-", "product"):
            return val
    return "Your order item"


def _normalize_shipping_address(row: dict) -> dict:
    """Parse shipping_address dict from API (never str(dict) in UI)."""
    import ast
    import json

    raw = (row or {}).get("shipping_address") or (row or {}).get("delivery_address")
    if raw is None:
        plain = str((row or {}).get("address") or "").strip()
        return {"address": plain} if plain else {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        if s.startswith("{"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"address": s}
    return {}


def _format_shipping_address_html(addr: dict, *, multiline: bool = True) -> str:
    if not isinstance(addr, dict) or not addr:
        return "—"
    name = str(addr.get("name") or "").strip()
    phone = str(addr.get("phone") or "").strip()
    email = str(addr.get("email") or "").strip()
    line = str(addr.get("address") or addr.get("address_line") or "").strip()
    city = str(addr.get("city") or "").strip()
    state = str(addr.get("state") or "").strip()
    postal = str(addr.get("postal_code") or addr.get("pincode") or "").strip()
    country = str(addr.get("country") or "").strip()

    city_state = ", ".join(x for x in (city, state) if x)
    if postal:
        city_state = f"{city_state} {postal}".strip() if city_state else postal
    loc = ", ".join(x for x in (line, city_state, country) if x)

    bits = []
    if name:
        bits.append(f"<div class='wf-od-address__name'>{html_escape(name)}</div>")
    if phone:
        bits.append(f"<div class='wf-od-address__line'><b>Phone:</b> {html_escape(phone)}</div>")
    if email:
        bits.append(f"<div class='wf-od-address__line'><b>Email:</b> {html_escape(email)}</div>")
    if loc:
        bits.append(f"<div class='wf-od-address__line'><b>Address:</b> {html_escape(loc)}</div>")
    if not bits:
        return "—"
    return f"<div class='wf-od-address'>{''.join(bits)}</div>"


def format_order_details_reply(
    row: dict,
    order_id: str,
    focus: str = "summary",
    lang: str = "en",
) -> str:
    """HTML card from purchase-history-details — app-style sections, focus-filtered."""
    if not isinstance(row, dict):
        return ""
    focus = (focus or "summary").lower().strip()
    if focus == "address":
        focus = "delivery"
    lang = (lang or "en").lower().strip()
    use_hinglish = lang == "hinglish"

    labels = {
        "title": "Order details",
        "order_info": "Order information",
        "shipping": "Shipping address",
        "product_sec": "Product",
        "order_id": "Order ID",
        "date": "Order date",
        "status": "Status",
        "total": "Total amount",
        "payment_method": "Payment method",
        "payment_status": "Payment status",
        "delivery": "Delivery status",
        "size": "Size",
        "color": "Color",
        "qty": "Qty",
        "view_product": "View product",
    }
    if use_hinglish:
        labels.update(
            {
                "title": "Aapke order ki details",
                "order_info": "Order ki jaankari",
                "shipping": "Delivery address",
                "product_sec": "Product",
            }
        )

    oid = html_escape(str(row.get("oid") or row.get("id") or order_id or "—"))
    product_title = html_escape(_ph_details_product_title(row))
    status = html_escape(str(row.get("current_order_status") or "—"))
    delivery_status = html_escape(str(row.get("delivery_status_string") or row.get("delivery_status") or ""))
    pay_status = html_escape(str(row.get("payment_status_string") or row.get("payment_status") or "—"))
    pay_method = html_escape(
        str(row.get("payment_type") or row.get("payment_mode") or row.get("payment_method") or "—").replace("_", " ")
    )
    order_date = html_escape(str(row.get("date") or row.get("order_date") or "—"))
    grand_total = html_escape(str(row.get("grand_total") or row.get("subtotal") or ""))
    size = html_escape(str(row.get("size") or "—"))
    color = html_escape(str(row.get("color") or "—"))
    shipping_type = html_escape(str(row.get("shipping_type_string") or row.get("shipping_type") or ""))
    addr_html = _format_shipping_address_html(_normalize_shipping_address(row))
    img_url = _ph_details_img_url(row)
    slug = row.get("product_slug") or ""
    product_link = f"https://welfog.com/product_details/{slug}" if slug else ""

    show_order = focus in ("summary", "payment", "status", "all")
    show_shipping = focus in ("summary", "delivery", "all")
    show_product = focus in ("summary", "product", "all")
    show_payment_only = focus == "payment"
    show_delivery_only = focus == "delivery"
    show_product_only = focus == "product"
    show_status_only = focus == "status"

    parts = [
        "<div class='wf-od-root' data-wf-live-api='order_details'>",
        f"<div class='wf-od-title'>{labels['title']}</div>",
        "<div class='wf-od-layout'>",
    ]

    if show_order or show_payment_only or show_status_only:
        parts.append("<div class='wf-od-section'>")
        parts.append(f"<div class='wf-od-section__head'>{labels['order_info']}</div>")
        parts.append("<div class='wf-od-grid'>")
        parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['order_id']}</span><span class='wf-od-kv__v'>{oid}</span></div>")
        if order_date and order_date != "—" and not show_payment_only:
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['date']}</span><span class='wf-od-kv__v'>{order_date}</span></div>")
        if status and status != "—":
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['status']}</span><span class='wf-od-kv__v wf-od-kv__v--status'>{status}</span></div>")
        if delivery_status and delivery_status.lower() not in ("—", status.lower()) and focus in ("summary", "delivery", "status", "all"):
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['delivery']}</span><span class='wf-od-kv__v'>{delivery_status}</span></div>")
        if grand_total and focus in ("summary", "payment", "all"):
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['total']}</span><span class='wf-od-kv__v wf-od-kv__v--total'>{grand_total}</span></div>")
        if focus in ("summary", "payment", "all"):
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['payment_method']}</span><span class='wf-od-kv__v'>{pay_method}</span></div>")
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>{labels['payment_status']}</span><span class='wf-od-kv__v'>{pay_status}</span></div>")
        if shipping_type and shipping_type != "—" and focus in ("summary", "delivery", "all"):
            parts.append(f"<div class='wf-od-kv'><span class='wf-od-kv__k'>Shipping</span><span class='wf-od-kv__v'>{shipping_type}</span></div>")
        parts.append("</div></div>")

    if show_shipping or show_delivery_only:
        parts.append("<div class='wf-od-section'>")
        head = labels["shipping"]
        if show_delivery_only:
            head = f"{head} · {labels['order_id']} {oid}"
        parts.append(f"<div class='wf-od-section__head'>{head}</div>")
        parts.append(addr_html if addr_html != "—" else "<div class='wf-od-meta'>—</div>")
        parts.append("</div>")

    if show_product or show_product_only:
        parts.append("<div class='wf-od-section wf-od-section--product'>")
        parts.append(f"<div class='wf-od-section__head'>{labels['product_sec']}</div>")
        parts.append("<div class='wf-od-product-row'>")
        if img_url:
            parts.append(f"<img class='wf-od-product-row__img' src='{html_escape(img_url)}' alt=''>")
        else:
            parts.append("<div class='wf-od-product-row__img wf-od-product-row__img--ph' aria-hidden='true'></div>")
        parts.append("<div class='wf-od-product-row__body'>")
        parts.append(f"<div class='wf-od-product-row__name'>{product_title}</div>")
        if size and size != "—":
            parts.append(f"<div class='wf-od-meta'>{labels['size']}: {size}</div>")
        if color and color != "—":
            parts.append(f"<div class='wf-od-meta'>{labels['color']}: {color}</div>")
        if grand_total:
            parts.append(f"<div class='wf-od-meta wf-od-meta--price'>{labels['total']}: <b>{grand_total}</b></div>")
        if product_link:
            parts.append(
                f"<a class='wf-od-link' href='{html_escape(product_link)}' target='_blank' rel='noopener noreferrer'>"
                f"{labels['view_product']}</a>"
            )
        parts.append("</div></div></div>")

    parts.append("</div></div>")
    return "".join(parts)


def format_order_invoice_reply_html(order_id: str, lang: str = "en") -> str:
    """Verified invoice download button (caller must confirm ownership via details API first)."""
    oid = html_escape(str(order_id or "").strip())
    url = html_escape(build_order_invoice_url(order_id))
    lang = (lang or "en").lower().strip()
    if lang == "hinglish":
        title = "Aapka invoice"
        hint = "Neeche button se invoice download karein (sirf aapke order ke liye)."
        btn = "Invoice download karein"
    else:
        title = "Your invoice"
        hint = "Use the button below to download your invoice for this order."
        btn = "Download invoice"
    return (
        f"<div class='wf-od-root wf-od-root--invoice' data-wf-live-api='order_invoice'>"
        "<div class='wf-invoice-card'>"
        "<div class='wf-invoice-card__icon' aria-hidden='true'>"
        "<svg viewBox='0 0 24 24' width='28' height='28' fill='none' "
        "stroke='currentColor' stroke-width='2'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/>"
        "<polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/>"
        "<line x1='16' y1='17' x2='8' y2='17'/></svg></div>"
        f"<div class='wf-od-title'>{title}</div>"
        f"<p class='wf-od-hint'>{hint}</p>"
        f"<p class='wf-od-meta wf-invoice-card__oid'><b>Order ID:</b> {oid}</p>"
        f"<a class='wf-invoice-btn' href='{url}' target='_blank' rel='noopener noreferrer' "
        f"download role='button'><span class='wf-invoice-btn__label'>{html_escape(btn)}</span></a>"
        "</div></div>"
    )
