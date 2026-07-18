"""OpenSearch product search — filters by size, color, brand, price, SKU, pro_id, pagination."""
import os
import re
import threading
from html import escape as html_escape
from typing import Any, Optional

import requests

from services.welfog_api import _normalize_color, fetch_products_from_api

_OS_REQUEST_TLS = threading.local()


def reset_opensearch_request_count() -> None:
    _OS_REQUEST_TLS.count = 0


def get_opensearch_request_count() -> int:
    return int(getattr(_OS_REQUEST_TLS, "count", 0) or 0)

IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
PAGE_SIZE = 8
MAX_PAGE_SIZE = 20
# Hide "View more" when the first page already shows fewer than this (avoids dead pagination).
MIN_PRODUCTS_FOR_VIEW_MORE = 3

_FILTER_STOP = frozenset(
    {
        "show", "dikha", "dikhao", "dikho", "dikhado", "de", "do", "please", "product", "products",
        "item", "items", "kuch", "koi", "mujhe", "mere", "ko", "me", "mai", "main", "wala", "wali",
        "wale", "according", "acc", "ke", "ki", "ka", "se", "par", "pe", "liye", "only", "sirf",
        "bas", "sort", "sorted", "order", "by", "filter", "filtered", "with", "having", "h", "hai",
        "kya", "ky", "hain", "the", "a", "an", "is", "are", "ka", "ke",
        "neeche", "niche", "upar", "tak", "bhai", "sir", "dikana", "dikhao",
        "batana", "batanao", "mereko", "mujhe", "dikjan", "dikjan", "maang", "mang",
        "rha", "rhe", "fir", "firse", "phir", "phirse", "hisb", "hisaab", "hisab",
        "accord", "options", "option", "available", "mil", "mile", "mila", "milna",
        "kr", "krna", "karna", "bol", "bolo", "sun", "suno", "bta", "btana", "btado",
        "coverz", "wala", "wali", "wale", "hue", "huye", "hain", "tha",
        "the", "unka", "unki", "unke", "uske", "uska", "iski",
        "color", "colour", "rang", "hu", "hun", "hoon", "hain", "rah", "raha", "rahe",
        "rahi", "dikha", "dikhe", "dikhen", "chahiye", "chahie", "chaahiye",
        "jara", "zara", "thoda", "thodi", "zaraa", "jarra", "dekho", "dekhna",
        "bata", "batana", "batanaa", "dikana", "dikhana", "please", "plz",
        "kesa", "kesa", "kaise", "kese", "kaisa", "kaisi", "theek", "thik",
        "achha", "acha", "accha", "badhiya", "badiya", "shukriya", "thanks",
        "puchh", "puch", "puchna", "bolna", "bolo", "bol", "sunna", "sunn",
        "search", "krke", "krke", "bta", "btao", "dega", "karega", "krega",
    }
)

_SIZE_PATTERNS = (
    r"\bsize\s*[:=\-]?\s*(free\s*size|xxl|xl|xs|s|m|l|\d+(?:\.\d+)?\s*(?:inch|in|cm)?)\b",
    r"\bsize\s+(free\s*size|xxl|xl|xs|s|m|l|\d+(?:\.\d+)?)\b",
    # Hinglish / natural word order: "XXL size", "L size ki tshirt"
    r"\b(free\s*size|xxl|xl|xs|s|m|l)\s+size\b",
    r"\b(free\s*size)\b",
    r"\b(\d+(?:\.\d+)?)\s*(?:inch|in|cm)\s+size\b",
)
_BRAND_STOP = frozenset(
    {
        "cheapest", "sasta", "saste", "expensive", "mehnga", "rating", "price", "sort",
        "products", "product", "items", "item", "mobile", "phone", "case", "cover",
        "ka", "ke", "ki", "ko", "hai", "h", "kya",
    }
)
_BRAND_PATTERNS = (
    r"\b([a-z0-9][a-z0-9\-]{1,30})\s+brand\s+ka\b",
    r"\b([a-z0-9][a-z0-9\-]{1,30})\s+brand\s+ke\b",
    r"\b([a-z0-9][a-z0-9\-]{1,30})\s+brand\b",
    r"\bbrand\s*[:\-]?\s*([a-z0-9][a-z0-9\-]{1,30})(?!\s+ka\b)(?!\s+ke\b)",
)
# Hinglish: samsung ke liye cover, iphone ka case
_BRAND_KE_KI_KA_PATTERNS = (
    r"\b(samsung|iphone|apple|vivo|oppo|realme|redmi|xiaomi|oneplus|poco|motorola|nokia|honor|infinix|lg|tecno|itel)\s+(?:mobile|phone)?\s*(?:ke|ki|ka)\s+liye\b",
    r"\b(samsung|iphone|apple|vivo|oppo|realme|redmi|xiaomi|oneplus|poco|motorola|nokia|honor|infinix|lg|tecno|itel)\s+(?:ke|ki|ka)\b",
    r"\b(?:ke|ki|ka)\s+(samsung|iphone|apple|vivo|oppo|realme|redmi|xiaomi|oneplus|poco|motorola|nokia|honor|infinix|lg|tecno|itel)\b",
)
_PRICE_UNDER = (
    r"(?:under|below|less\s+than|max|upto|up\s+to|within)\s*(?:rs\.?|₹|inr)?\s*(\d{2,7})",
    r"(?:under|below|less\s+than|max|upto|up\s+to|within)\s*(\d{2,7})\s*(?:rs\.?|₹|inr)?",
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)?\s*(?:se\s+kam|tak|ya\s+kam|ya\s+niche|se\s+niche|se\s+under|wale|wali|walon)",
    r"(?:sasta|saste|cheap|cheapest|kam\s+price|low\s+price|minimum\s+price)",
)
_PRICE_OVER = (
    r"(?:above|over|more\s+than|min|at\s+least)\s*(?:rs\.?|₹|inr)?\s*(\d{2,7})",
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)?\s*(?:se\s+(?:zyada|upar|jyada|jyaada)|ya\s+(?:zyada|upar))",
    r"(?:mehnga|mehenga|expensive|costly|high\s+price|maximum\s+price)",
)
_RATING_PATTERNS = (
    r"(?:rating|rated|stars?)\s*(?:of\s+)?(?:at\s+least|min|minimum|>=?|above)?\s*(\d(?:\.\d)?)",
    r"(\d(?:\.\d)?)\s*(?:star|stars|rating)",
    r"\b(best|highest|top|acha|acchi|achha)\s+(?:rated|rating|stars?)\b",
    r"\b(high|good)\s+rating\b",
)
_RATING_UNDER_PATTERNS = (
    r"(?:under|below|less\s+than|low|kam|se\s+kam|se\s+niche)\s*(?:(\d(?:\.\d)?)\s*)?(?:star|stars|rating)",
    r"(?:rating|stars?)\s*(?:under|below|low|kam|se\s+kam|se\s+niche)\s*(\d(?:\.\d)?)",
    r"(\d(?:\.\d)?)\s*(?:star|stars|rating)\s*(?:se\s+kam|se\s+niche|ya\s+kam|under|below|tak)",
    r"\blow\s+rating\b",
)
_GENERIC_BRAND_WORDS = frozenset(
    {
        "welfog", "no brand", "generic", "unbranded", "mobile", "phone", "product", "products",
        "item", "items", "cover", "case", "your", "search", "brand",
    }
)
_SKU_PATTERNS = (r"\bsku\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9_\-]{2,60})",)
_PRO_ID_PATTERNS = (
    r"\b(?:pro[_\s-]?id|product[_\s-]?id|pid)\s*[:\-#]?\s*(\d{4,12})",
    r"\bproduct\s+id\s+(?:de\s+rha\s+hu|de\s+raha\s+hu|de\s+rahi\s+hu)\s+(\d{4,12})",
    r"\b(\d{4,12})\s+iska\s+product\b",
    r"\b(\d{6,12})\s+is\s+id\s+ke",
    r"\bid\s+(\d{6,12})\s+ke\s+products?",
    r"\bproducts?\s+(?:for|of)\s+(?:id\s+)?(\d{6,12})",
)
_SORT_PRICE_ASC = (
    "cheapest", "sasta", "saste", "kam price", "low price", "price low", "price ascending",
    "sort by price", "price sort low", "sabse sasta",
)
_SORT_PRICE_DESC = (
    "expensive", "mehnga", "mehenga", "high price", "price high", "price descending",
    "sabse mehnga",
)
_SORT_RATING = (
    "best rating", "highest rating", "top rated", "rating high", "rating desc", "best rated",
    "acha rating", "acchi rating",
)
_SORT_PURCHASE_ASC = ("low purchase", "purchase price low", "kam purchase")

_COLOR_TYPO_MAP = {
    "aasmani": "sky blue",
    "asmani": "sky blue",
    "aasmaani": "sky blue",
    "neela": "blue",
    "neeli": "blue",
    "kala": "black",
    "kaala": "black",
    "safed": "white",
    "laal": "red",
    "lal": "red",
    "hara": "green",
    "peela": "yellow",
    "blaack": "black",
    "blak": "black",
    "blck": "black",
    "blk": "black",
    "whit": "white",
    "whte": "white",
    "greeen": "green",
    "gren": "green",
    "grean": "green",
    "grenn": "green",
    "grenen": "green",
    "blew": "blue",
    "blu": "blue",
    "rd": "red",
    "yelow": "yellow",
    "purpel": "purple",
    "greay": "grey",
    "gray": "grey",
    "grey": "grey",
    "orang": "orange",
    "multicolour": "multicolor",
}
# American/British spellings stripped from title_query when a colour filter is active.
_COLOR_SPELLING_VARIANTS: dict[str, tuple[str, ...]] = {
    "grey": ("grey", "gray"),
    "gray": ("grey", "gray"),
    "black": ("black",),
    "white": ("white",),
    "green": ("green",),
}
# Title tokens that mean the same product type (cover vs bumper vs case).
_PRODUCT_TYPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cover": ("cover", "covers", "case", "cases", "bumper", "back cover", "backcover", "skin"),
    "case": ("case", "cases", "cover", "covers", "bumper", "back cover", "backcover"),
    "bumper": ("bumper", "cover", "covers", "case", "cases", "back cover", "backcover"),
    "charger": ("charger", "chargers", "charging", "adapter", "adaptor", "power bank", "powerbank"),
}

# Title phrases to drop when product_type is X (stops charger query showing covers).
_TYPE_CONFLICT_EXCLUDES: dict[str, tuple[str, ...]] = {
    "charger": ("cover", "case", "cases", "bumper", "back cover", "screen guard", "tempered glass", "protector glass"),
    "adapter": ("cover", "case", "bumper", "back cover"),
    "cable": ("cover", "case", "bumper", "back cover"),
    "cover": ("charger", "charging cable", "usb cable", "power adapter", "wall charger", "car charger", "power bank"),
    "case": ("charger", "charging cable", "power adapter", "wall charger", "car charger"),
    "bumper": ("charger", "charging cable", "power adapter", "wall charger"),
}
_PRODUCT_NOUNS = frozenset(
    {
        "cover", "covers", "case", "cases", "bumper", "mobile", "phone", "charger",
        "cable", "adapter", "earphone", "headphone", "earbuds", "watch", "shirt",
        "shirts", "tshirt", "jeans", "jean", "shoe", "shoes", "sandal", "sandals",
        "rice", "wheat", "jug", "bottle", "flour", "atta", "dal", "lentil", "oil",
        "milk", "bread", "soap", "cream", "laptop", "tv", "fan", "bag", "bags",
        "wallet", "belt", "kurta", "saree", "dress", "hoodie", "socks", "sock",
        "pant", "pants", "trouser", "shorts", "toy", "book", "pen", "notebook",
        "keyboard", "mouse", "speaker", "trimmer", "razor", "perfume", "lipstick",
        "foundation", "serum", "shampoo", "conditioner", "towel", "bedsheet",
        "pillow", "curtain", "mug", "glass", "spoon", "knife", "pan", "pot",
    }
)
_COLOR_NAME_TOKENS = frozenset(
    {
        "black", "white", "red", "green", "blue", "yellow", "pink", "purple", "grey",
        "gray", "orange", "brown", "silver", "gold", "navy", "maroon", "beige",
        "kala", "safed", "laal", "lal", "neela", "hara", "peela",
    }
)
# Material/finish in product title — NOT catalog colour (transparent cover ≠ color Transparent).
_MATERIAL_TITLE_TOKENS = frozenset(
    {
        "transparent", "translucent", "crystal", "clear", "matte", "glossy", "frosted",
        "silicone", "leather", "velvet", "rubber", "hybrid", "tempered",
    }
)
_NOUN_TYPO_MAP = {
    "covr": "cover", "covar": "cover", "cvr": "cover", "coverz": "cover", "coverzs": "cover",
    "moblie": "mobile", "mobail": "mobile",
    "tshrt": "tshirt", "tshirt": "tshirt", "jean": "jeans", "chargr": "charger", "botle": "bottle",
    "shrt": "shirt", "kurtaa": "kurta",
}
# NOTE: typo map is OpenSearch recall only — never use it for intent / category-browse routing.
# Words that must NEVER be treated as warehouse SKU codes (colours, product types, brands).
_SKU_ATTRIBUTE_STOP = _COLOR_NAME_TOKENS | _PRODUCT_NOUNS | frozenset(
    {
        "multicolor", "multicolour", "multicoloured", "multicolored",
        "oneplus", "iphone", "samsung", "redmi", "infinix", "vivo", "oppo", "realme",
        "xiaomi", "poco", "motorola", "nokia", "apple", "google", "nothing", "honor",
        "transparent", "crystal", "bumper", "tempered", "glass", "protector",
    }
)
_SKU_STOP = frozenset(
    {
        "product", "products", "dikhao", "dikha", "dikho", "dikhado", "batao", "bata", "batana",
        "batanao", "please", "welfog", "yah", "yh", "ye", "yeh", "he", "hai", "h", "iska", "iski",
        "isko", "iske", "mereko", "mujhe", "mera", "mere", "dikana", "dikhaa", "search", "query",
        "iska", "iski", "isko", "sku", "code", "number", "id",
        "chahiye", "chahie", "chaahiye", "chahiyee", "chahiyye", "chahiyya", "chahiy",
        "milega", "milegi", "chahiy", "lena", "dena", "lao", "laao", "dikhe", "dikhen",
        "mast", "shaadi", "pehan", "pehn", "pehnna", "pehanne", "wedding", "liye", "ke",
    }
) | _SKU_ATTRIBUTE_STOP
_CONFLICTING_COLOR_HINTS = {
    "black": ("green", "red", "blue", "yellow", "pink", "orange", "purple", "multicolor", "multi color"),
    "white": ("black", "green", "red", "blue", "multicolor"),
    "green": ("red", "blue", "pink", "orange", "purple", "black"),
    "red": ("green", "blue", "black"),
    "blue": ("green", "red", "orange"),
    "yellow": ("black", "blue", "green"),
    "multicolor": (),
}
_UNDER_PRICE_MARKERS = re.compile(
    r"(?:\bunder\b|\bbelow\b|less\s+than|\bupto\b|up\s+to|\bwithin\b|\bmax\b|"
    r"\bse\s+kam\b|\bse\s+niche\b|\bke\s+neeche\b|\bke\s+niche\b|\bke\s+andar\b|"
    r"\bandar\s+andar\b|kam\s+price|kam\s+me|"
    r"\bin\s+this\s+range\b|\bthis\s+range\b|\bwithin\s+(?:my|their|the)?\s*(?:budget|range)\b|"
    r"\bi\s+have\s+(?:only\s+)?(?:rs|₹|inr|\d)|\bbudget\s+of\b|"
    r"\b\d+(?:\.\d+)?\s*k\s+tak\b|"
    r"இன்\s*கீழ்|கீழ்|க்கு\s*கீழ்|"
    r"से\s*कम|से\s*नीचे|"
    r"below\s+rs|rs\s+se\s+kam|\btak\b)",
    re.IGNORECASE,
)
_BUDGET_AMOUNT_PATTERNS = (
    r"\bi\s+have\s+(?:only\s+)?(?:rs\.?|₹|inr)?\s*(\d{2,7})\b",
    r"\bi\s+have\s+(\d{2,7})\s*(?:rs\.?|₹|inr)\b",
    r"\bbudget\s+(?:of|is)?\s*(?:rs\.?|₹|inr)?\s*(\d{2,7})\b",
    r"(?:within|in)\s+(?:my|their|this|the)?\s*(?:budget|range)\s+(?:of\s+)?(?:rs\.?|₹|inr)?\s*(\d{2,7})\b",
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)\s+(?:budget|range|only|max)\b",
    r"(?:rs\.?|₹|inr)\s*(\d{2,7})\s+(?:budget|range|only|max)\b",
    r"(?:show|dikha\w*|get)\s+(?:me\s+)?(?:\w+\s+){0,6}(?:under|within|in)\s+(?:rs\.?|₹|inr)?\s*(\d{2,7})\b",
    r"\b(?:mere|meri)\s+(?:pass|paas)\s+(?:\w+\s+){0,6}?(\d{2,7})\s*(?:rs\.?|₹|inr)?\b",
    r"\b(?:pass|paas)\s+(?:\w+\s+){0,4}?(\d{2,7})\s*(?:rs\.?|₹|inr)\b",
    r"(\d{2,7})\s*(?:rs\.?|₹|inr)\s+(?:h|hai|he|ha)\b",
)
_OVER_PRICE_MARKERS = re.compile(
    r"(?:\babove\b|\bover\b|more\s+than|\bmin\b|at\s+least|\bse\s+upar\b|\bse\s+jyada\b|"
    r"से\s*ज्यादा|இன்\s*மேல்|மேல்)",
    re.IGNORECASE,
)
_RATING_CONTEXT = re.compile(
    r"\b(?:rating|rated|stars?|star)\b|"
    r"\b(?:jin|jinki|jo)\b[^\n]{0,80}\b(?:rating|stars?)\b",
    re.IGNORECASE,
)
_RATING_ABOVE_PATTERNS = (
    r"\b(?:rating|stars?)\s*(?:above|over|>\s*)\s*(\d(?:\.\d)?)\b",
    r"\b(?:above|over)\s*(\d(?:\.\d)?)\s*(?:star|stars?|rating)\b",
    r"\b(?:rating|stars?)\s+(?:more|greater|higher)\s+than\s+(\d(?:\.\d)?)\b",
    r"\bhaving\s+rating\s+(?:more|greater|higher)\s+than\s+(\d(?:\.\d)?)\b",
    r"\b(?:jin|jinki|jo)\b.*\b(?:rating|stars?)\b.*\b(?:above|over|upar|jyada|zyada|zada)\b.*?(\d(?:\.\d)?)",
    r"\b(?:rating|stars?)\s*(\d(?:\.\d)?)\s*se\s+(?:jyada|zyada|zada|upar|zyaada)\b",
)


def _collapse_repeated_chars(word: str) -> str:
    if not word:
        return word
    prev = None
    w = word.lower()
    while prev != w:
        prev = w
        w = re.sub(r"(.)\1+", r"\1", w)
    return w


def _catalog_colors_equivalent(card_color: str, requested: str) -> bool:
    """Strict match on catalog color_name field (API uses e.g. Black, Green, Sky Blue)."""
    if not card_color or not requested:
        return False
    a = card_color.strip().lower()
    b = requested.strip().lower()
    if a == b:
        return True
    if a in ("grey", "gray") and b in ("grey", "gray"):
        return True
    if a in ("sky blue", "light blue") and b in ("sky blue", "light blue"):
        return True
    na = _normalize_color(a) or a.title()
    nb = _normalize_color(b) or b.title()
    return na.lower() == nb.lower()


def extract_color_and_product_title(text: str) -> tuple[Optional[str], str]:
    """
    Split user message into catalog colour + product-type title (cover, shirt…).
    Avoids searching the whole sentence as product name.
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""
    low = raw.lower()
    color: Optional[str] = None

    short_hue = re.search(
        r"\b(black|white|green|red|blue|yellow|pink|purple|orange|brown|grey|gray|navy|maroon|beige|silver|gold|linen)\s+(?:color|colour|rang)\b",
        low,
        re.IGNORECASE,
    )
    if short_hue:
        hue = normalize_color_fuzzy(short_hue.group(1))
        if hue:
            color = hue

    compound_color = re.search(
        r"\b([A-Za-z]{4,32})\s+(?:color|colour|rang)\b",
        raw,
        re.IGNORECASE,
    )
    if compound_color and not color:
        cword = compound_color.group(1).strip()
        if cword.lower() not in _FILTER_STOP and cword.lower() not in _PRODUCT_NOUNS:
            hue = normalize_color_fuzzy(cword)
            color = hue or (cword.title() if cword.islower() else cword)
        elif not color:
            color = normalize_color_fuzzy(raw)
    for pat in (
        r"\b(black|white|green|red|blue|yellow|pink|purple|orange|brown|grey|gray|navy|maroon|beige|silver|gold)\s+(?:color|colour|rang)\b",
        r"\b(?:color|colour|rang)\s+(?:ke|ki|ka|me|mein)?\s*(black|white|green|red|blue|yellow|pink|purple|orange|brown|grey|gray|navy|maroon|beige|silver|gold)\b",
        r"\b(black|white|green|red|blue|yellow|pink|purple|orange|brown|grey|gray)\s+(?:ke|ki|ka)\s+(?:cover|covers|case|shirt|mobile|phone)\b",
    ):
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            hue = normalize_color_fuzzy(m.group(1))
            if hue:
                color = hue
                break

    title_parts: list[str] = []
    for tok in re.findall(r"[a-z]+", low):
        if tok in _FILTER_STOP or tok in _COLOR_NAME_TOKENS:
            continue
        mapped = _NOUN_TYPO_MAP.get(tok, tok)
        collapsed = _collapse_repeated_chars(mapped)
        mapped = _NOUN_TYPO_MAP.get(collapsed, collapsed)
        if re.fullmatch(r"cover\w{0,4}", mapped) and mapped not in _PRODUCT_NOUNS:
            mapped = "cover"
        noun = mapped.rstrip("s") if mapped.endswith("s") and mapped[:-1] in _PRODUCT_NOUNS else mapped
        if noun in _PRODUCT_NOUNS:
            title_parts.append(noun)
        elif mapped in _PRODUCT_NOUNS:
            title_parts.append(mapped)
    seen: set[str] = set()
    nouns: list[str] = []
    for n in title_parts:
        if n not in seen:
            seen.add(n)
            nouns.append(n)
    title = " ".join(nouns[:4])
    if not title:
        title = _extract_product_keywords(low)
    if color and title:
        title = _strip_color_from_title_query(title, color)
    elif color:
        scrub = low
        for hue in _color_spelling_variants(color):
            scrub = re.sub(rf"\b{re.escape(hue)}\b", " ", scrub, flags=re.IGNORECASE)
        for w in ("color", "colour", "rang", "ke", "ki", "ka", "se", "me", "mein"):
            scrub = re.sub(rf"\b{w}\b", " ", scrub)
        title = _extract_product_keywords(scrub) or ""
    if color:
        if re.search(r"[a-z][A-Z]", color) or (len(color) >= 8 and color[0].isupper()):
            pass
        else:
            color = _normalize_color(color) or color
    return color, (title or "").strip()


def normalize_color_fuzzy(text: str) -> Optional[str]:
    """Map Hinglish / typo color words to catalog color_name (e.g. blaack → Black)."""
    if not text:
        return None
    low = text.lower()
    for token in re.findall(r"[a-z]+", low):
        collapsed = _collapse_repeated_chars(token)
        for candidate in (token, collapsed, _COLOR_TYPO_MAP.get(token), _COLOR_TYPO_MAP.get(collapsed)):
            if not candidate:
                continue
            c = _normalize_color(candidate)
            if c:
                return c
        if collapsed.startswith("blac") or token.startswith("blac"):
            return "Black"
        if collapsed.startswith("gre") or token.startswith("gre"):
            return "Green"
    return _normalize_color(low)


def _looks_like_warehouse_sku(tok: str) -> bool:
    """
    Warehouse / catalog codes — not conversational words (any language).
    Must contain a digit, or underscore/hyphen with alnum mix; not plain lowercase prose.
    """
    if not tok:
        return False
    t = tok.strip()
    if len(t) < 4 or len(t) > 80:
        return False
    if t.isalpha() and t.islower():
        return False
    if re.fullmatch(r"(?i)sku", t):
        return False
    if t.isupper() and t.isalpha() and len(t) >= 4:
        return True
    if re.search(r"\d", t):
        if "_" in t or "-" in t:
            return True
        if sum(1 for c in t if c.isupper()) >= 1 and sum(1 for c in t if c.isdigit()) >= 1:
            return True
        if t.isupper() and len(t) >= 5:
            return True
        if len(t) >= 6:
            return True
    if "_" in t and re.search(r"[A-Za-z0-9]", t):
        return True
    if "-" in t and len(t) >= 5:
        parts = [p for p in t.split("-") if p]
        if len(parts) >= 2 and any(len(p) >= 2 for p in parts):
            if any(c.isupper() for c in t) or any(re.search(r"\d", p) for p in parts):
                return True
    return False


def _is_valid_sku_token(tok: str, *, explicit_sku_mention: bool = False) -> bool:
    """Safety net only — shape + stop lists; meaning comes from Groq product AI."""
    if not tok:
        return False
    t = tok.strip()
    if explicit_sku_mention and 4 <= len(t) <= 80 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", t):
        tl = t.lower()
        if tl not in _SKU_STOP and tl not in _FILTER_STOP and tl not in _SKU_ATTRIBUTE_STOP:
            if t.isupper() and sum(1 for c in t if c.isalpha()) >= 3:
                return True
            if re.search(r"[\-_]", t) and sum(1 for c in t if c.isalpha()) >= 3:
                return True
    if not _looks_like_warehouse_sku(tok):
        return False
    tl = tok.strip().lower()
    norm = _collapse_repeated_chars(tl)
    if (
        tl in _SKU_STOP
        or norm in _SKU_STOP
        or tl in _FILTER_STOP
        or norm in _FILTER_STOP
        or tl in _SKU_ATTRIBUTE_STOP
        or norm in _SKU_ATTRIBUTE_STOP
    ):
        return False
    if re.fullmatch(r"(?i)sku", tok):
        return False
    if tok[0].isupper() and tok[1:].islower() and tok.isalpha():
        try:
            from services.welfog_api import get_category_id_from_text

            if get_category_id_from_text(tok):
                return False
        except ImportError:
            pass
    return True


def _sku_token_acceptable(tok: str, *, explicit_sku_mention: bool = False) -> bool:
    if not tok:
        return False
    if _is_valid_sku_token(tok, explicit_sku_mention=explicit_sku_mention):
        return True
    return explicit_sku_mention and _looks_like_spaced_catalog_sku(tok)


def _normalize_explicit_sku_capture(raw_tok: str) -> str:
    tok = re.sub(r"\s+", " ", (raw_tok or "").strip())
    if not tok:
        return ""
    tok = re.sub(r"^(?:de|do)\s+", "", tok, flags=re.I)
    tok = re.split(
        r"\s+(?:yeh|ye|this)\s+(?:sku|product)\b",
        tok,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    tok = _strip_sku_leading_fillers(tok)
    refined = _best_warehouse_sku_span(tok)
    return refined or tok


_SKU_LEADING_FILLERS = frozenset(
    {
        "jara", "zara", "bata", "btana", "btao", "btado", "batao", "bataiye", "batao",
        "dikha", "dikhao", "dikha do", "de", "do", "show", "find", "please", "plz",
        "bhai", "yar", "yrr", "na", "hi", "haan", "bol", "bolo", "bta", "bata",
        "btana", "btao", "thoda", "abhi", "mujhe", "mere", "ko", "ka", "ki", "ke",
    }
)


def _strip_sku_leading_fillers(text: str) -> str:
    words = re.split(r"\s+", (text or "").strip())
    while words and words[0].lower() in _SKU_LEADING_FILLERS:
        words.pop(0)
    return " ".join(words).strip()


def _best_warehouse_sku_span(text: str) -> str:
    """
    Pick the warehouse SKU span from noisy captures like
    'jara INFINIX HOT 10 PLAY-EGL-SP' → 'INFINIX HOT 10 PLAY-EGL-SP'.
    """
    parts = re.split(r"\s+", (text or "").strip())
    if not parts:
        return ""
    best = ""
    for i in range(len(parts)):
        candidate = " ".join(parts[i:])
        if not candidate:
            continue
        if _looks_like_spaced_catalog_sku(candidate):
            return candidate
        if _sku_token_acceptable(candidate, explicit_sku_mention=True):
            if len(candidate) > len(best):
                best = candidate
    return best


def _refine_extracted_sku(raw: str) -> str:
    """Final cleanup on an extracted SKU token."""
    tok = _normalize_explicit_sku_capture(raw)
    if not tok:
        return ""
    if _sku_token_acceptable(tok, explicit_sku_mention=True):
        return tok
    refined = _best_warehouse_sku_span(tok)
    if refined and _sku_token_acceptable(refined, explicit_sku_mention=True):
        return refined
    return tok


def _looks_like_spaced_catalog_sku(tok: str) -> bool:
    """Warehouse SKUs like 'NFINIX HOT 10 PLAY-EGL-SP' after explicit sku ka product."""
    if not tok or len(tok) < 6 or len(tok) > 80:
        return False
    t = re.sub(r"\s+", " ", tok.strip())
    if not re.search(r"[A-Za-z]", t) or not re.search(r"[\-_]", t):
        return False
    if sum(1 for c in t if c.isupper()) < 3:
        return False
    tl = t.lower()
    if tl in _SKU_STOP or tl in _FILTER_STOP:
        return False
    return True


def _extract_sku_from_text(raw: str) -> Optional[str]:
    """
    Explicit SKU mentions only — never guess SKU from random long words (any language).
    """
    if not raw:
        return None
    candidates: list[tuple[int, str]] = []

    for m in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9_\-]{3,80})\s+(?:is\s+|ka\s+)?sku\b",
        raw,
        re.IGNORECASE,
    ):
        tok = m.group(1).strip()
        if _is_valid_sku_token(tok, explicit_sku_mention=True):
            candidates.append((100, tok))
    for m in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9_\-]{3,80})\s+is\s+sku\s+ka\b",
        raw,
        re.IGNORECASE,
    ):
        tok = m.group(1).strip()
        if _sku_token_acceptable(tok, explicit_sku_mention=True):
            candidates.append((101, tok))
    for m in re.finditer(
        r"\b([A-Za-z0-9][A-Za-z0-9_\-]{3,80})\s+sku\b",
        raw,
        re.IGNORECASE,
    ):
        tok = m.group(1).strip()
        if _is_valid_sku_token(tok, explicit_sku_mention=True):
            candidates.append((100, tok))

    for m in re.finditer(
        r"\bsku\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9_\-]{3,80})",
        raw,
        re.IGNORECASE,
    ):
        tok = m.group(1).strip()
        if _is_valid_sku_token(tok, explicit_sku_mention=True):
            candidates.append((90, tok))

    for m in re.finditer(
        r"\b(?:product|item|warehouse)\s+code\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9_\-]{3,80})",
        raw,
        re.IGNORECASE,
    ):
        tok = m.group(1).strip()
        if _is_valid_sku_token(tok):
            candidates.append((85, tok))

    for m in re.finditer(
        r"\b(?:is\s+)?sku\s+ka\s+(?:bta\w*\s+de|bata\w*\s+de|dikha\w*|show|find|de)\s+(.+?)\s*$",
        raw,
        re.IGNORECASE,
    ):
        tok = _normalize_explicit_sku_capture(m.group(1))
        if _sku_token_acceptable(tok, explicit_sku_mention=True):
            candidates.append((97, tok))
    for m in re.finditer(
        r"\b(?:is\s+)?sku\s+ka\s+product\s+(?:bta\w*|bata\w*|btana|dikha\w*|de|show|find)\s+(.+?)\s*$",
        raw,
        re.IGNORECASE,
    ):
        tok = _normalize_explicit_sku_capture(m.group(1))
        if _sku_token_acceptable(tok, explicit_sku_mention=True):
            candidates.append((98, tok))
    for m in re.finditer(
        r"\bsku\s+ka\s+product\b[^\n]{0,50}?(?:bta\w*|btana|dikha\w*|show|find|de|bata\w*)\s+(.+?)\s*$",
        raw,
        re.IGNORECASE,
    ):
        tok = _normalize_explicit_sku_capture(m.group(1))
        if _sku_token_acceptable(tok, explicit_sku_mention=True):
            candidates.append((95, tok))
    for m in re.finditer(
        r"\b(?:yeh|ye|this)\s+sku\s+ka\s+product\b[^\n]{0,50}?(?:dikha\w*|show|find|de)\s+(.+?)\s*$",
        raw,
        re.IGNORECASE,
    ):
        tok = _normalize_explicit_sku_capture(m.group(1))
        if _sku_token_acceptable(tok, explicit_sku_mention=True):
            candidates.append((96, tok))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], -len(x[1])))
    return _refine_extracted_sku(candidates[0][1])


def _turn_mentions_rating_filter(text: str) -> bool:
    return bool(_RATING_CONTEXT.search(text or ""))


def _user_mentions_price_this_turn(
    text: str,
    *,
    ai_understanding: Optional[dict] = None,
    spec: Optional[dict] = None,
) -> bool:
    """
    Budget/price intent — Brain + Product Entity Extraction own this in any language.
    No Hinglish/English keyword lists; trust AI fields when present.
    """
    try:
        from services.catalog_spec_semantics import ai_set_price_filter, spec_has_price_filter

        if ai_understanding and ai_set_price_filter(ai_understanding):
            return True
        if spec and spec_has_price_filter(spec):
            return True
    except ImportError:
        pass
    return False


def _extract_price_bounds(raw: str, low: str) -> tuple[Optional[float], Optional[float]]:
    """Price bounds come from Brain / Product NLU JSON — not regex keyword maps."""
    return None, None


# Universal budget parser — keys off NUMBERS + currency/comparator context, so it
# works for ANY product in ANY language/script (₹, rs, rupee, "under", "se kam",
# "ke andar", "tak", "range", "budget"…). This is a deterministic safety net for
# when the routing/entity LLM drops the price filter — NOT a product-keyword list.
_PRICE_CURRENCY_RE = re.compile(
    r"(?:₹|\brs\.?\b|\brs\b|rupees?|\binr\b|/-|\bprice\b|\bbudget\b|\brange\b|\bwithin\b|"
    r"\bunder\b|\bbelow\b|\bupto\b|\bup\s*to\b|\bmax(?:imum)?\b|\bmin(?:imum)?\b|"
    r"\babove\b|\bover\b|\bmore\s+than\b|\bat\s*least\b|\bbetween\b|\bbe?ech\b|"
    r"se\s*kam|se\s*jyada|se\s*zyada|"
    r"se\s*ni?ch[ae]|se\s*upar|ke\s*andar|\bandar\b|\btak\b|ke\s*ni?ch[ae]|ke\s*upar)",
    re.IGNORECASE,
)
_PRICE_MIN_RE = re.compile(
    r"(?:\babove\b|\bover\b|\bmore\s+than\b|\bat\s*least\b|\bminimum\b|\bmin\b|"
    r"se\s*jyada|se\s*zyada|se\s*upar|se\s*adhik|ke\s*upar)",
    re.IGNORECASE,
)
_PRICE_MAX_RE = re.compile(
    r"(?:\bunder\b|\bbelow\b|\bupto\b|\bup\s*to\b|\bwithin\b|\bless\s+than\b|"
    r"\bmax(?:imum)?\b|se\s*kam|se\s*ni?ch[ae]|ke\s*andar|\bandar\b|\btak\b|ke\s*ni?ch[ae])",
    re.IGNORECASE,
)


def _price_number_tokens(low: str) -> list[float]:
    """Numbers (2-7 digits, optional comma / k-suffix) that look like money amounts."""
    out: list[float] = []
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{2,3})+|\d{2,7})\s*(k\b)?", low):
        num = m.group(1).replace(",", "")
        try:
            val = float(num)
        except ValueError:
            continue
        if m.group(2):  # "2k" → 2000
            val *= 1000.0
        out.append(val)
    return out


def parse_price_bounds_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """Return (min_price, max_price) from a shopping message. None when no budget."""
    raw = (text or "").strip()
    if not raw:
        return None, None
    low = raw.lower()
    if not _PRICE_CURRENCY_RE.search(low):
        return None, None
    nums = _price_number_tokens(low)
    if not nums:
        return None, None

    # Explicit range: "300 to 500", "300-500", "300 se 500".
    rng = re.search(
        r"\b(\d{2,7})\s*(?:-|–|to|se|and|aur)\s*(\d{2,7})\b", low
    )
    if rng:
        a, b = float(rng.group(1)), float(rng.group(2))
        return (min(a, b), max(a, b))

    has_min = bool(_PRICE_MIN_RE.search(low))
    has_max = bool(_PRICE_MAX_RE.search(low))
    amount = max(nums)  # the budget figure, not stray digits like "combo 5"
    if has_min and not has_max:
        return (amount, None)
    if has_max and not has_min:
        return (None, amount)
    # Currency present, no direction word → treat as budget ceiling (ecommerce default).
    return (None, amount)


def strip_price_tokens_from_title(title: str) -> str:
    """Remove budget phrases from a title so OpenSearch matches the product noun only.

    Structural (number + currency/comparator + connective filler) — never a
    product-keyword list. "baniyan under 400 rs" → "baniyan"; "500 ki range me
    baniyan" → "baniyan".
    """
    t = (title or "").strip()
    if not t or not _PRICE_CURRENCY_RE.search(t):
        # No budget context in the title — leave the product noun untouched.
        return t
    # Drop currency/comparator words, standalone money numbers, and the small set of
    # connective fillers that glue a budget phrase to the product noun.
    t = _PRICE_CURRENCY_RE.sub(" ", t)
    t = re.sub(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d{2,7}\s*k?\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b(?:ki|ka|ke|me|mein|mera|meri|mere|wali|wala|se|aur|and|to|tak|liye)\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(t.split()).strip()
    # Never return an empty title from stripping — keep original if we nuked it all.
    return cleaned or title.strip()


def _scrub_price_filters_on_rating_turn(
    spec: dict[str, Any],
    text: str,
    *,
    ai_understanding: Optional[dict] = None,
) -> None:
    """Remove purchase_price_* on rating-only turns unless AI also set a budget."""
    if not spec or not _turn_mentions_rating_filter(text):
        return
    try:
        from services.catalog_spec_semantics import ai_set_price_filter, spec_has_price_filter

        if ai_understanding and ai_set_price_filter(ai_understanding):
            return
        if spec_has_price_filter(spec) and (
            spec.get("_catalog_ai") or spec.get("_ai_single_pass")
        ):
            return
    except ImportError:
        pass
    spec.pop("purchase_price_min", None)
    spec.pop("purchase_price_max", None)
    spec.pop("unit_price_min", None)
    spec.pop("unit_price_max", None)


def _extract_rating_bounds_from_text(low: str) -> tuple[Optional[float], Optional[float]]:
    """Parse rating_min / rating_max without treating stars as rupees."""
    rmin = rmax = None
    for pat in _RATING_UNDER_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            star_cap = None
            if m.lastindex and m.group(1):
                try:
                    star_cap = float(m.group(1))
                except ValueError:
                    star_cap = None
            if star_cap is None and re.search(r"\blow\s+rating\b", low):
                star_cap = 3.0
            if star_cap is not None:
                rmax = star_cap
            break
    m_under_rating = re.search(
        r"\b(?:rating|stars?)\s+(?:under|below|kam|se\s+kam|se\s+niche)\s+(\d(?:\.\d)?)\b",
        low,
        re.IGNORECASE,
    )
    if m_under_rating and rmax is None:
        try:
            rmax = float(m_under_rating.group(1))
        except ValueError:
            pass
    m_jinki_under = re.search(
        r"\b(?:jin|jinki|jo)\b.*\b(?:rating|stars?)\b.*\b(?:under|below|kam|niche|se\s+kam)\b",
        low,
        re.IGNORECASE,
    )
    if m_jinki_under and rmax is None:
        m_num = re.search(r"\b(\d(?:\.\d)?)\b", low[m_jinki_under.end() : m_jinki_under.end() + 50])
        if m_num:
            try:
                rmax = float(m_num.group(1))
            except ValueError:
                pass
    for pat in _RATING_ABOVE_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m and m.lastindex:
            try:
                rmin = float(m.group(1))
            except ValueError:
                pass
            break
    m_jinki = re.search(
        r"\b(?:jin|jinki|jo)\b.*\b(?:rating|stars?)\b.*\b(?:above|over|upar|jyada|zyada|zada|se)\b",
        low,
        re.IGNORECASE,
    )
    if m_jinki and rmin is None:
        m_num = re.search(r"\b(\d(?:\.\d)?)\b", low[m_jinki.end() : m_jinki.end() + 40])
        if not m_num:
            m_num = re.search(r"\b(\d(?:\.\d)?)\b", low)
        if m_num:
            try:
                rmin = float(m_num.group(1))
            except ValueError:
                pass
        if rmin is None and re.search(r"\b(?:above|over|upar|jyada)\s*0\b", low):
            rmin = 0.01
    return rmin, rmax


def _merge_title_queries(left: str, right: str) -> str:
    """Merge EN + original title hints without duplicating the same product noun."""
    a = (left or "").strip()
    b = (right or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a.lower() == b.lower():
        return a
    seen: set[str] = set()
    out: list[str] = []
    for tok in (a + " " + b).split():
        tl = tok.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(tok)
    return " ".join(out[:6])


def _merge_filter_specs(*specs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "title_query": "",
        "color": None,
        "size": None,
        "brand": None,
        "sku": None,
        "pro_id": None,
        "category_id": None,
        "unit_price_min": None,
        "unit_price_max": None,
        "purchase_price_min": None,
        "purchase_price_max": None,
        "rating_min": None,
        "rating_max": None,
        "in_stock_only": False,
        "sort": None,
    }
    for spec in specs:
        if not spec:
            continue
        for key, val in spec.items():
            if val is None or val == "" or val is False:
                continue
            if key == "sku" and merged.get("sku"):
                if _is_valid_sku_token(str(val)) and not _is_valid_sku_token(str(merged["sku"])):
                    merged["sku"] = val
                elif _is_valid_sku_token(str(merged["sku"])):
                    pass
                elif len(str(val)) > len(str(merged["sku"])):
                    merged["sku"] = val
            elif key == "brand" and merged.get("brand"):
                pass
            elif key == "title_query" and merged.get("title_query"):
                merged["title_query"] = _merge_title_queries(str(merged["title_query"]), str(val))
            else:
                merged[key] = val
    if merged.get("sku") and not _sku_token_acceptable(
        str(merged["sku"]), explicit_sku_mention=True
    ):
        merged["sku"] = None
    return merged


def _clean_title_hint(hint: str) -> str:
    from services.product_query_understanding import dedupe_search_terms, is_noisy_search_query

    h = dedupe_search_terms((hint or "").strip())
    if is_noisy_search_query(h):
        return ""
    h = _strip_size_from_title_query(h, _extract_size_from_text(h))
    return " ".join(h.split()).strip()


def build_product_search_spec(
    original_msg: str,
    msg_en: str = "",
    *,
    category_id: Optional[int] = None,
    color: Optional[str] = None,
    title_hint: str = "",
    pro_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Parse filters from original + translated text so SKU/price survive translation.
    Original message is preferred for SKU codes and numeric price limits.
    """
    hint = _clean_title_hint(title_hint)
    spec_orig = parse_product_filters_from_text(
        original_msg or "",
        category_id=category_id,
        color=color,
        title_hint="",
        pro_id=pro_id,
    )
    spec_en = parse_product_filters_from_text(
        f"{msg_en} {hint}".strip(),
        category_id=category_id,
        color=color or spec_orig.get("color"),
        title_hint="",
        pro_id=pro_id,
    )
    merged = _merge_filter_specs(spec_orig, spec_en)
    kw = _extract_product_keywords((msg_en or original_msg or "").lower())
    if merged.get("sku"):
        merged["title_query"] = kw or ""
    elif kw and not merged.get("title_query"):
        merged["title_query"] = kw
    elif merged.get("title_query"):
        cleaned = [
            w
            for w in merged["title_query"].split()
            if w.lower() not in _FILTER_STOP and not w.isdigit()
        ]
        merged["title_query"] = " ".join(cleaned[:6])
    literal_brand = _extract_brand_literal_from_text(original_msg or "")
    if literal_brand:
        merged["brand"] = literal_brand
    merged = finalize_catalog_search_spec(merged, original_msg, msg_en)
    merged = reconcile_catalog_spec_with_user_turn(
        merged, original_msg, msg_en, ctx=None, ai_route=None
    )
    if merged.get("pro_id"):
        merged["title_query"] = ""
        merged.pop("brand", None)
        merged.pop("brand_aliases", None)
        merged["title_match_strict"] = False
    return sanitize_product_search_spec(merged)


def _extract_product_keywords(low: str) -> str:
    if is_price_or_rating_browse_turn(low):
        return ""
    if re.search(r"\bsim\b", low) and re.search(
        r"\b(?:pin\b|nikal|ejector|tray|tool|opener|remover)",
        low,
        re.I,
    ):
        if not re.search(r"\b(?:pincode|pin\s*code)\b", low):
            return "sim ejector pin"
    words = []
    if re.search(r"\b(?:shoe|shwo|sho)\s+me\b", low):
        low = re.sub(r"\b(?:shoe|shwo|sho)\s+me\b", " ", low)
    for tok in re.findall(r"[a-z]+", low):
        if tok in ("shoe", "shwo", "sho") and re.search(r"\b(?:dikha|dikhao|cover|color|colour)\b", low):
            continue
        if len(tok) <= 2 and tok not in _PRODUCT_NOUNS:
            continue
        if tok in _FILTER_STOP:
            continue
        mapped = _NOUN_TYPO_MAP.get(tok, tok)
        collapsed = _collapse_repeated_chars(mapped)
        mapped = _NOUN_TYPO_MAP.get(collapsed, collapsed)
        if mapped in _PRODUCT_NOUNS:
            words.append(mapped.rstrip("s") if mapped.endswith("s") and mapped[:-1] in _PRODUCT_NOUNS else mapped)
        elif mapped.endswith("s") and mapped[:-1] in _PRODUCT_NOUNS:
            words.append(mapped[:-1])
        elif mapped in ("tshirt", "tee"):
            words.append("tshirt")
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out[:6])


def _post_filter_mode_for_spec(spec: dict[str, Any]) -> str:
    """
    Strict only when brand / mandatory tokens / explicit title_match_strict.
    Otherwise os_filters_only: light title relevance + brand/colour — do NOT force
    strict all-token wipe on every locked `_ai_single_pass` noun search.
    """
    if spec.get("title_match_strict"):
        return "strict"
    if spec.get("mandatory_match_tokens"):
        return "strict"
    # Brand-named searches stay strict for title+brand alignment.
    if (spec.get("brand") or "").strip() and spec.get("title_match_strict") is not False:
        if spec.get("title_match_strict") or (spec.get("match_mode") or "").strip().lower() == "strict":
            return "strict"
    # Soft noun searches: OpenSearch ranks; post-filter keeps colour/brand lightly.
    return "os_filters_only"


def has_structured_product_filters(spec: dict[str, Any]) -> bool:
    return any(
        [
            spec.get("sku"),
            spec.get("color"),
            spec.get("pro_id"),
            spec.get("brand"),
            spec.get("size"),
            spec.get("category_id"),
            spec.get("rating_min") is not None,
            spec.get("rating_max") is not None,
            spec.get("unit_price_max") is not None,
            spec.get("unit_price_min") is not None,
            spec.get("purchase_price_max") is not None,
            spec.get("purchase_price_min") is not None,
        ]
    )


def format_filter_display_label(spec: dict[str, Any]) -> str:
    """Human label for what the user asked — avoid 'Black colour black shirt'."""
    title = (spec.get("title_query") or "").strip()
    if title:
        extras = []
        if spec.get("sku") and _is_valid_sku_token(str(spec["sku"])):
            extras.append(f"SKU {spec['sku']}")
        if spec.get("purchase_price_max") is not None:
            extras.append(f"under Rs {int(spec['purchase_price_max'])}")
        elif spec.get("unit_price_max") is not None:
            extras.append(f"under Rs {int(spec['unit_price_max'])}")
        if spec.get("purchase_price_min") is not None:
            extras.append(f"above Rs {int(spec['purchase_price_min'])}")
        elif spec.get("unit_price_min") is not None:
            extras.append(f"above Rs {int(spec['unit_price_min'])}")
        if spec.get("rating_min") is not None:
            try:
                from services.catalog_spec_semantics import rating_min_display_label

                extras.append(rating_min_display_label(spec["rating_min"]))
            except ImportError:
                extras.append(f"rating {spec['rating_min']}+ stars")
        if spec.get("rating_max") is not None:
            extras.append(f"rating under {spec['rating_max']} stars")
        if spec.get("brand"):
            extras.append(f"{spec['brand']} brand")
        color = (spec.get("color") or "").strip()
        title_low = title.lower()
        if color and color.lower() not in title_low:
            return f"{color} {title}" + (f" ({', '.join(extras)})" if extras else "")
        if extras:
            return f"{title} ({', '.join(extras)})"
        return title
    parts = []
    if spec.get("sku") and _is_valid_sku_token(str(spec["sku"])):
        parts.append(f"SKU {spec['sku']}")
    if spec.get("color"):
        parts.append(f"{spec['color']} colour")
    if spec.get("purchase_price_max") is not None:
        parts.append(f"under Rs {int(spec['purchase_price_max'])}")
    elif spec.get("unit_price_max") is not None:
        parts.append(f"under Rs {int(spec['unit_price_max'])}")
    if spec.get("purchase_price_min") is not None:
        parts.append(f"above Rs {int(spec['purchase_price_min'])}")
    elif spec.get("unit_price_min") is not None:
        parts.append(f"above Rs {int(spec['unit_price_min'])}")
    if spec.get("rating_min") is not None:
        try:
            from services.catalog_spec_semantics import rating_min_display_label

            parts.append(rating_min_display_label(spec["rating_min"]))
        except ImportError:
            parts.append(f"rating {spec['rating_min']}+ stars")
    if spec.get("rating_max") is not None:
        parts.append(f"rating under {spec['rating_max']} stars")
    if spec.get("brand"):
        parts.append(f"{spec['brand']} brand")
    if spec.get("size"):
        parts.append(f"size {spec['size']}")
    return " ".join(parts).strip() or "your filters"


_GENERIC_TITLE_MODIFIERS = frozenset(
    {
        "men",
        "man",
        "mens",
        "women",
        "woman",
        "womens",
        "kids",
        "boy",
        "boys",
        "girl",
        "girls",
        "male",
        "female",
        "unisex",
        "for",
        "adult",
        "adults",
    }
)


def _title_match_tokens(title_query: str) -> list[str]:
    """Distinctive searchable tokens from the product title query (any catalog language/spelling)."""
    if not title_query:
        return []
    tokens = []
    for w in re.findall(r"[a-z0-9]{3,}", title_query.lower()):
        if w in _COLOR_NAME_TOKENS or w in _FILTER_STOP or w in _GENERIC_TITLE_MODIFIERS:
            continue
        if w in _MATERIAL_TITLE_TOKENS:
            continue
        if w in _PRODUCT_NOUNS or (w.endswith("s") and w[:-1] in _PRODUCT_NOUNS):
            tokens.append(w.rstrip("s") if w.endswith("s") and w[:-1] in _PRODUCT_NOUNS else w)
        elif len(w) >= 3:
            tokens.append(w)
    seen: set[str] = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:6]


def _compound_space_variants(token: str) -> list[str]:
    """
    Generate spaced splits for compound tokens (flipflops → flip flops).
    No synonym dictionary — all alphabetic split points; OpenSearch BM25 ranks the useful ones.
    """
    t = re.sub(r"[^a-z0-9]", "", (token or "").lower())
    # Short tokens (leather, samsung) produce junk splits (lea ther) that slow OS.
    if len(t) < 9:
        return []
    out: list[str] = []
    for i in range(3, len(t) - 2):
        left, right = t[:i], t[i:]
        if left.isalpha() and right.isalpha():
            out.append(f"{left} {right}")
        if len(out) >= 4:
            break
    return out


def _compound_token_surface_forms(tok: str) -> list[str]:
    """
    Surface forms for compact catalog tokens: tshirt ↔ t shirt / t-shirt.
    Algorithmic splits only — no per-product synonym tables.
    """
    raw = (tok or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", raw)
    if not compact:
        return []
    forms: list[str] = [compact]
    if re.search(r"[\s-]", raw):
        spaced = re.sub(r"[\s-]+", " ", raw).strip()
        if spaced:
            forms.append(spaced)
        forms.append(compact)
    for i in range(1, len(compact)):
        left, right = compact[:i], compact[i:]
        if left.isalpha() and right.isalpha() and len(left) >= 1 and len(right) >= 2:
            forms.extend([f"{left} {right}", f"{left}-{right}"])
    seen: set[str] = set()
    out: list[str] = []
    for f in forms:
        f = f.strip()
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _focus_title_query_on_product_noun(spec: dict[str, Any]) -> None:
    """
    OpenSearch BM25 on the product noun; audience words (women/men/kids) stay for post-filter.
    """
    tq = (spec.get("title_query") or "").strip()
    ptype = (spec.get("product_type") or "").strip().lower()
    if not tq or not ptype:
        return
    words = tq.lower().split()
    if len(words) < 2:
        return
    ptype_base = ptype.rstrip("s")
    audience: list[str] = []
    product_words: list[str] = []
    for w in words:
        wl = w.lower()
        if wl in _GENERIC_TITLE_MODIFIERS:
            audience.append(wl)
            continue
        if wl in (ptype, ptype_base) or wl.rstrip("s") == ptype_base:
            product_words.append(ptype)
        else:
            product_words.append(wl)
    if not audience:
        return
    if audience and not spec.get("audience_tokens"):
        spec["audience_tokens"] = list(dict.fromkeys(audience))
    if product_words:
        spec["title_query"] = " ".join(dict.fromkeys(product_words))
    else:
        spec["title_query"] = ptype


def _soft_collapse_repeated_letters(text: str) -> str:
    try:
        from services.product_query_understanding import soft_collapse_repeated_letters

        return soft_collapse_repeated_letters(text)
    except ImportError:
        t = (text or "").lower()
        # Match product_query_understanding: keep normal doubles (moose, google).
        t = re.sub(r"([aeiou])\1{2,}", r"\1", t)
        t = re.sub(r"([b-df-hj-np-tv-z])\1{2,}", r"\1", t)
        return t


def _expand_title_query_variants(title: str) -> list[str]:
    """
    Typo-softened primary + compound-split variants for robust BM25 matching.
    No product keyword lists — algorithmic repeats/splits only.

    Prefer soft-collapsed form first: raw keysmash + fuzziness:2 can make OpenSearch
    reject/empty the whole bool query even when a good alt variant is present.
    """
    base = re.sub(r"\s+", " ", (title or "").strip().lower())
    if not base:
        return []
    collapsed = _soft_collapse_repeated_letters(base)
    # Prefer ORIGINAL orthography as primary — soft-collapse is only a secondary
    # variant (moosewala must not become mosewala as the main OpenSearch query).
    variants: list[str] = []
    if base:
        variants.append(base)
    if collapsed and collapsed not in variants:
        variants.append(collapsed)

    def _add(alt: str) -> None:
        alt = re.sub(r"\s+", " ", (alt or "").strip().lower())
        if alt and alt not in variants:
            variants.append(alt)

    def _simple_singular_token(tok: str) -> str:
        """Structural English plural soft-form (caps→cap). Not a product dictionary."""
        t = (tok or "").lower()
        if len(t) <= 3 or not t.endswith("s"):
            return t
        if t.endswith(("ss", "us", "is", "oes", "xes", "ches", "shes")):
            return t
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        return t[:-1]

    # caps / sneakers → also query singular catalog titles
    for source in list(variants[:2]):
        toks = source.split()
        if not toks:
            continue
        sing = " ".join(_simple_singular_token(t) for t in toks)
        if sing != source:
            _add(sing)

    # Compound splits only for glued single tokens (flipflops → flip flops).
    # Multi-word titles like "samsung cover" already match BM25 — mid-word
    # splits (sam sung / sams ung) blow up the should-clause and slow OpenSearch.
    for source in list(variants[:2]):
        toks = source.split()
        for tok in toks:
            soft = _soft_collapse_repeated_letters(tok)
            if soft != tok:
                _add(source.replace(tok, soft, 1))
        if len(toks) >= 2:
            continue
        tok = toks[0] if toks else ""
        soft = _soft_collapse_repeated_letters(tok) if tok else ""
        splits = _compound_space_variants(soft if soft != tok else tok)
        ranked = sorted(
            splits,
            key=lambda s: abs(len(s.split()[0]) - len(s.split()[-1])) if " " in s else 99,
        )
        for split in ranked[:2]:
            _add(split)
    # Cap alts — primary + singular + 1–2 compound forms is enough for BM25.
    return variants[:4]


def _build_title_match_clause(title: str) -> dict[str, Any]:
    """
    Precision-first OpenSearch text query.

    - Searches product name / sku / brand (NOT category_name — that caused Men Fashion pollution).
    - Primary = typo-softened form when available (avoids fuzzy blow-ups on keysmash).
    - Uses compound-split variants as soft should clauses.
    """
    variants = _expand_title_query_variants(title)
    if not variants:
        return {"match_all": {}}
    primary = variants[0]
    token_count = len(re.findall(r"[a-z0-9]+", primary))
    # Soft-collapsed long tokens: mild AUTO fuzz is enough; avoid fuzziness:2
    # on raw keysmash (can empty the whole query).
    fuzz: Any = "AUTO"
    prefix_len = 0 if token_count == 1 else 1
    if token_count <= 1:
        msm: Any = "1"
    elif token_count == 2:
        msm = "2"
    else:
        msm = "2<75%"

    should: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": primary,
                "fields": ["name^5", "sku^2", "brand^1.5"],
                "type": "best_fields",
                "fuzziness": fuzz,
                "prefix_length": prefix_len,
                "minimum_should_match": msm,
            }
        },
        {
            "match_phrase": {
                "name": {
                    "query": primary,
                    "boost": 4.0,
                    "slop": 2,
                }
            }
        },
    ]
    for alt in variants[1:]:
        # Exact/phrase + light AUTO only — never expensive fuzziness:2 on alts.
        should.append(
            {
                "multi_match": {
                    "query": alt,
                    "fields": ["name^5", "sku^2", "brand"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "boost": 1.4,
                }
            }
        )
        should.append(
            {
                "match_phrase": {
                    "name": {
                        "query": alt,
                        "boost": 2.8,
                        "slop": 2,
                    }
                }
            }
        )
    return {"bool": {"should": should, "minimum_should_match": 1}}


def color_hue_mentioned_in_text(color_name: str, text: str) -> bool:
    """True only when this exact text mentions the colour (not a sibling item in the sentence)."""
    if not color_name or not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    for hue in _color_spelling_variants(color_name):
        if re.search(rf"\b{re.escape(hue.lower())}\b", tl):
            return True
    return False


def resolve_color_for_part_text(part_text: str, suggested_color: str = "") -> str:
    """
    Per-part catalog colour — any product type, any supported hue.
    Colour applies only when THIS part's words mention it (shirt=white, cover=no white bleed).
    """
    scoped = (part_text or "").strip()
    if not scoped:
        return ""
    explicit, _title = extract_color_and_product_title(scoped)
    if explicit:
        return explicit
    fuzzy = normalize_color_fuzzy(scoped)
    if fuzzy and color_hue_mentioned_in_text(fuzzy, scoped):
        return fuzzy
    sug = (suggested_color or "").strip()
    if sug:
        norm = normalize_color_fuzzy(sug) or sug
        if color_hue_mentioned_in_text(norm, scoped):
            return norm
    return ""


def _color_spelling_variants(color_name: str) -> list[str]:
    c = (color_name or "").strip().lower()
    if not c:
        return []
    variants = list(_COLOR_SPELLING_VARIANTS.get(c, (c,)))
    if c in _COLOR_NAME_TOKENS and c not in variants:
        variants.append(c)
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _normalize_catalog_size_value(size_val: str) -> str:
    """Map parsed size token to catalog size field value."""
    s = (size_val or "").strip()
    if not s:
        return ""
    if re.fullmatch(r"free\s*size", s, re.I):
        return "Free Size"
    if re.fullmatch(r"[xsmlXL]{1,3}", s, re.I):
        return s.upper()
    if re.search(r"\d", s):
        return s.strip()
    return s.title()


def _extract_size_from_text(text: str) -> str:
    """Pull size filter from message — size=10, size 10, 10 inch, free size."""
    raw = (text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    for pat in _SIZE_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            return _normalize_catalog_size_value(m.group(1).strip())
    return ""


def _strip_size_from_title_query(title_query: str, size_value: str = "") -> str:
    """OpenSearch uses size= filter; title_query must be product type only (cover, not size 10 cover)."""
    tq = (title_query or "").strip()
    if not tq:
        return ""
    for pat in _SIZE_PATTERNS:
        tq = re.sub(pat, " ", tq, flags=re.IGNORECASE)
    tq = re.sub(
        r"\b(?:size|sizes|sized|number|no\.?|num)\b",
        " ",
        tq,
        flags=re.IGNORECASE,
    )
    if size_value:
        sv = re.escape(str(size_value).strip())
        if sv:
            tq = re.sub(rf"\b{sv}\b", " ", tq, flags=re.IGNORECASE)
    return " ".join(tq.split()).strip()


def _strip_color_from_title_query(title_query: str, color_name: str) -> str:
    """API uses color= param; title should be product type only (cover not black cover)."""
    if not title_query or not color_name:
        return (title_query or "").strip()
    tq = title_query
    for token in _color_spelling_variants(color_name):
        tq = re.sub(rf"\b{re.escape(token)}\b", " ", tq, flags=re.IGNORECASE)
    for token in _COLOR_NAME_TOKENS:
        if token in color_name.lower() or color_name.lower() in token:
            tq = re.sub(rf"\b{re.escape(token)}\b", " ", tq, flags=re.IGNORECASE)
    return " ".join(tq.split()).strip()


_GENERIC_TITLE_WORDS = frozenset(
    {"products", "product", "items", "item", "goods", "stuff", "things", "cheez", "cheeze"}
)
_PRODUCT_MODIFIER_WORDS = frozenset({"mobile", "phone", "smart", "wireless", "cell"})


def _strip_generic_from_mandatory(spec: dict[str, Any]) -> None:
    mandatory = spec.get("mandatory_match_tokens")
    if not mandatory:
        return
    if not isinstance(mandatory, list):
        mandatory = [str(mandatory)]
    cleaned = [
        t.strip().lower()
        for t in mandatory
        if t
        and str(t).strip().lower() not in _GENERIC_TITLE_WORDS
        and str(t).strip().lower() not in _PRODUCT_MODIFIER_WORDS
    ]
    ptype = (spec.get("product_type") or "").strip().lower()
    product_nouns = [t for t in cleaned if t in _PRODUCT_NOUNS or t in _PRODUCT_TYPE_SYNONYMS]
    if spec.get("_semantic_constraints_from_ai"):
        # Product NLU explicitly supplied compatibility/features. Preserve those
        # semantic constraints instead of collapsing everything to the generic
        # product noun (cover/shirt), which used to lose model/waterproof/etc.
        spec["mandatory_match_tokens"] = cleaned[:6]
    elif product_nouns:
        spec["mandatory_match_tokens"] = [product_nouns[0]]
    elif ptype and ptype in _PRODUCT_TYPE_SYNONYMS:
        spec["mandatory_match_tokens"] = [ptype]
    elif cleaned:
        spec["mandatory_match_tokens"] = cleaned[:2]
    else:
        spec.pop("mandatory_match_tokens", None)


def _normalize_generic_title_query(spec: dict[str, Any], original_msg: str = "") -> None:
    tq = (spec.get("title_query") or "").strip().lower()
    if tq and tq not in _GENERIC_TITLE_WORDS:
        return
    ptype = (spec.get("product_type") or "").strip().lower()
    if ptype:
        spec["title_query"] = ptype
        return
    try:
        kw = _extract_product_keywords((original_msg or "").lower())
        if kw:
            spec["title_query"] = kw
            return
    except Exception:
        pass
    if spec.get("color"):
        spec["title_query"] = ""
    else:
        spec.pop("title_query", None)


def _strip_color_tokens_from_mandatory(spec: dict[str, Any]) -> None:
    """Colour is enforced via color= + color_name — not as a title keyword."""
    mandatory = spec.get("mandatory_match_tokens")
    if not mandatory:
        return
    if not isinstance(mandatory, list):
        mandatory = [str(mandatory)]
    cleaned = [
        t.strip().lower()
        for t in mandatory
        if t and str(t).strip().lower() not in _COLOR_NAME_TOKENS
    ]
    if cleaned:
        spec["mandatory_match_tokens"] = cleaned
    else:
        spec.pop("mandatory_match_tokens", None)


def _strip_color_tokens_from_brand_aliases(spec: dict[str, Any]) -> None:
    """Colours/generic words in brand_aliases wrongly filter out products."""
    aliases = spec.get("brand_aliases")
    if not aliases:
        return
    if not isinstance(aliases, list):
        aliases = [str(aliases)]
    cleaned = [
        a
        for a in aliases
        if str(a).strip().lower() not in _COLOR_NAME_TOKENS
        and str(a).strip().lower() not in _GENERIC_TITLE_WORDS
    ]
    if cleaned:
        spec["brand_aliases"] = cleaned
    else:
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
    brand = (spec.get("brand") or "").strip().lower()
    if brand and brand in _COLOR_NAME_TOKENS:
        spec.pop("brand", None)


def _scrub_invalid_price_rating_filters(spec: dict[str, Any]) -> None:
    """Remove Rs 0 / inverted budgets that zero out catalog results."""
    for key in ("purchase_price_min", "purchase_price_max", "unit_price_min", "unit_price_max"):
        val = spec.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            spec.pop(key, None)
            continue
        if f <= 0:
            spec.pop(key, None)
    pmin = spec.get("purchase_price_min")
    pmax = spec.get("purchase_price_max")
    if pmin is not None and pmax is not None:
        try:
            pmin_f, pmax_f = float(pmin), float(pmax)
            if pmin_f > pmax_f:
                spec.pop("purchase_price_min", None)
                spec.pop("purchase_price_max", None)
            elif pmin_f == pmax_f:
                spec["purchase_price_max"] = pmax_f
                spec.pop("purchase_price_min", None)
        except (TypeError, ValueError):
            pass
    rmin = spec.get("rating_min")
    rmax = spec.get("rating_max")
    if rmin is not None and rmax is not None:
        try:
            if float(rmin) >= float(rmax):
                spec.pop("rating_min", None)
                spec.pop("rating_max", None)
        except (TypeError, ValueError):
            pass


def sanitize_product_search_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Drop bogus SKU; strip catalogue colour from title_query. Brand/strictness come from AI spec."""
    if not spec:
        return spec
    sku = spec.get("sku")
    if sku:
        if not _sku_token_acceptable(str(sku), explicit_sku_mention=True):
            spec.pop("sku", None)
        else:
            spec.pop("brand", None)
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)
    color = spec.get("color")
    if color and spec.get("sku"):
        if str(spec["sku"]).strip().lower() == str(color).strip().lower():
            spec.pop("sku", None)
        elif re.search(r"[a-z][A-Z]", str(color)):
            spec.pop("sku", None)
    if spec.get("category_id") and not (spec.get("title_query") or "").strip():
        needs_strip = bool(
            spec.get("sku")
            or spec.get("brand")
            or spec.get("brand_aliases")
        )
        if needs_strip or not spec.get("_category_only_browse"):
            try:
                from services.welfog_api import strip_category_browse_conflicts_from_spec

                spec = strip_category_browse_conflicts_from_spec(spec)
            except ImportError:
                pass
    color = spec.get("color")
    if color and spec.get("title_query"):
        stripped = _strip_color_from_title_query(str(spec["title_query"]), str(color))
        if stripped:
            spec["title_query"] = stripped
    _strip_color_tokens_from_brand_aliases(spec)
    _strip_color_tokens_from_mandatory(spec)
    _strip_generic_from_mandatory(spec)
    if spec.get("brand_aliases") and not isinstance(spec["brand_aliases"], list):
        spec["brand_aliases"] = [str(spec["brand_aliases"])]
    if spec.get("mandatory_match_tokens") and not isinstance(spec["mandatory_match_tokens"], list):
        spec["mandatory_match_tokens"] = [
            t.strip().lower()
            for t in str(spec["mandatory_match_tokens"]).split()
            if t.strip()
        ]
    _scrub_invalid_price_rating_filters(spec)
    try:
        from services.catalog_spec_semantics import enforce_explicit_user_filters_only

        user_blob = str(spec.pop("_filter_user_msg", "") or "")
        if user_blob:
            spec = enforce_explicit_user_filters_only(spec, user_blob, "")
    except ImportError:
        pass
    tq = (spec.get("title_query") or "").strip().lower()
    if tq in _FILTER_STOP or tq in ("your", "search", "kesa", "kaise", "hello"):
        spec["title_query"] = ""
    # Defense: category_id + title that is only the department label → id-only browse.
    # (name=electronics&categories=16 returns empty; categories=16 alone works.)
    cid = spec.get("category_id")
    tq_raw = (spec.get("title_query") or "").strip()
    # Keep product noun searches (women tshirt, sneakers, iphone) — never promote
    # them to department-only browse via substring category-name overlap.
    has_product_signal = bool(
        (spec.get("product_type") or "").strip()
        or (spec.get("mandatory_match_tokens") or [])
        or (spec.get("brand") or "").strip()
        or (spec.get("pro_id") or spec.get("sku"))
    )
    if cid and tq_raw and not has_product_signal:
        try:
            from services.welfog_api import (
                _normalize_cat_name,
                category_name_for_id,
                query_should_use_category_id_only,
            )

            if query_should_use_category_id_only(cid, tq_raw):
                spec["title_query"] = ""
                spec["_category_only_browse"] = True
            else:
                cat_n = _normalize_cat_name(category_name_for_id(str(cid)) or "")
                tq_n = _normalize_cat_name(tq_raw)
                # Exact department label only — "women tshirt" must NOT become
                # Women Fashion browse because "women" overlaps the category name.
                if cat_n and tq_n and tq_n == cat_n:
                    spec["title_query"] = ""
                    spec["_category_only_browse"] = True
        except ImportError:
            pass
    # Algorithmic typo soften on the way into catalog (echoed keysmash → matchable).
    tq = (spec.get("title_query") or "").strip()
    if tq and re.search(r"(.)\1", tq.lower()):
        try:
            from services.product_query_understanding import soften_echoed_typo_search_terms

            soft = soften_echoed_typo_search_terms(tq)
            if soft:
                spec["title_query"] = soft
        except ImportError:
            soft = _soft_collapse_repeated_letters(tq)
            if soft:
                spec["title_query"] = soft
    if spec.get("purchase_price_max") is not None or spec.get("purchase_price_min") is not None:
        spec["_landed_price_filter"] = True
    _focus_title_query_on_product_noun(spec)
    return spec


def _product_name_matches_color(name_lower: str, color_name: str) -> bool:
    """Title/description colour check — reject wrong hue (green cover when user asked black)."""
    if not color_name:
        return True
    c = color_name.strip().lower()
    if c == "multicolor":
        return any(
            x in name_lower
            for x in ("multicolor", "multi color", "multi-color", "multicolour", "multi colour")
        )
    if c in name_lower:
        return True
    if c == "black" and ("black" in name_lower or "blk" in name_lower):
        return True
    if c in ("grey", "gray") and any(x in name_lower for x in ("grey", "gray")):
        return True
    if c == "sky blue" and any(x in name_lower for x in ("sky blue", "sky-blue", "aasmani", "asmani")):
        return True
    conflicts = _CONFLICTING_COLOR_HINTS.get(c, ())
    for bad in conflicts:
        if bad in name_lower:
            return False
    if c in ("black", "white", "green", "red", "blue", "yellow", "pink", "purple", "orange"):
        return c in name_lower
    return True


def extract_material_tokens(text: str) -> list[str]:
    """Words like transparent/crystal that must appear in title when user asked for them."""
    low = (text or "").lower()
    found: list[str] = []
    for tok in sorted(_MATERIAL_TITLE_TOKENS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(tok)}\b", low):
            found.append(tok)
    return found


def _apply_material_tokens_to_spec(spec: dict[str, Any], original_msg: str = "") -> None:
    mats = extract_material_tokens(original_msg)
    if not mats:
        return
    mandatory = list(spec.get("mandatory_match_tokens") or [])
    for m in mats:
        if m not in mandatory:
            mandatory.append(m)
    spec["mandatory_match_tokens"] = mandatory[:4]
    spec["material_tokens"] = mats


def filter_products_by_material_tokens(
    products: list[dict],
    material_tokens: list[str],
) -> list[dict]:
    if not material_tokens or not products:
        return products
    need = [t.strip().lower() for t in material_tokens if t and len(str(t).strip()) >= 3]
    if not need:
        return products
    out = []
    for p in products:
        name = (p.get("name") or "").lower()
        if any(t in name for t in need):
            out.append(p)
    return out


def filter_products_by_requested_color(
    products: list[dict],
    color_name: Optional[str],
) -> list[dict]:
    if not color_name or not products:
        return products
    out = []
    for p in products:
        card_color = (p.get("color_name") or p.get("color") or "").strip()
        if card_color:
            if _catalog_colors_equivalent(card_color, color_name):
                out.append(p)
            continue
        name = (p.get("name") or "").lower()
        if _product_name_matches_color(name, color_name):
            out.append(p)
    return out


def _product_type_token_in_name(tok: str, name_lower: str) -> bool:
    """cover/case/bumper are equivalent in catalog titles (Samsung Bumper, iPhone case)."""
    if _title_token_in_name(tok, name_lower):
        return True
    synonyms = _PRODUCT_TYPE_SYNONYMS.get(tok)
    if synonyms:
        return any(s in name_lower for s in synonyms)
    return False


def filter_products_by_audience_tokens(
    products: list[dict],
    audience_tokens: list[str],
) -> list[dict]:
    """
    Keep listings whose title reflects requested audience (women/men/kids) from AI terms.
    Hard filter — never soft-return the full unfiltered list (that mixed watches/socks
    into 'women tshirt' results).
    """
    tokens = [t.strip().lower() for t in (audience_tokens or []) if t and len(str(t).strip()) >= 3]
    if not tokens or not products:
        return products
    expanded: set[str] = set()
    for t in tokens:
        expanded.add(t)
        if not t.endswith("s"):
            expanded.add(f"{t}s")
        if t.endswith("s") and len(t) > 3:
            expanded.add(t[:-1])
    matched = []
    for p in products:
        name = (p.get("name") or "").lower()
        if any(at in name for at in expanded):
            matched.append(p)
    return matched


def filter_products_by_ai_mandatory_tokens(
    products: list[dict],
    mandatory_tokens: list[str],
    *,
    product_type: Optional[str] = None,
    skip_color_tokens: bool = False,
) -> list[dict]:
    """All AI-provided tokens must appear in title (works for any brand/product, no hardcoded list)."""
    tokens = [t.strip().lower() for t in (mandatory_tokens or []) if t and len(str(t).strip()) >= 2]
    if skip_color_tokens:
        tokens = [t for t in tokens if t not in _COLOR_NAME_TOKENS]
    if not tokens or not products:
        return products
    ptype = (product_type or "").strip().lower()
    out = []
    for p in products:
        name = (p.get("name") or "").lower()
        if not all(_product_type_token_in_name(tok, name) for tok in tokens):
            continue
        if ptype and not _product_type_token_in_name(ptype, name):
            nouns = ptype.rstrip("s") if ptype.endswith("s") else ptype
            if not (
                _product_type_token_in_name(ptype, name)
                or _product_type_token_in_name(nouns, name)
            ):
                continue
        out.append(p)
    return out


def _phone_brand_vocab() -> frozenset:
    """Phone + lifestyle brands for message infer (nike shoes, samsung cover, …)."""
    try:
        from services.product_query_understanding import _ALL_KNOWN_BRANDS

        return _ALL_KNOWN_BRANDS
    except ImportError:
        return frozenset(
            {
                "samsung", "iphone", "apple", "vivo", "oppo", "realme", "redmi", "mi", "xiaomi",
                "oneplus", "poco", "nothing", "google", "motorola", "nokia", "honor", "infinix",
                "lg", "tecno", "itel", "nike", "adidas", "puma", "reebok", "bata", "skechers",
            }
        )


def filter_products_strict_brand_match(
    products: list[dict],
    brand: Optional[str],
    brand_aliases: Optional[list[str]] = None,
) -> list[dict]:
    """Samsung part → titles with samsung only; drop iPhone/Infinix/LG-only listings."""
    if not brand or not products:
        return products
    bl = brand.strip().lower()
    if bl in _COLOR_NAME_TOKENS:
        return products
    aliases = [
        a.strip().lower()
        for a in (brand_aliases or [])
        if a and str(a).strip().lower() not in _COLOR_NAME_TOKENS
    ]
    if bl not in aliases:
        aliases.insert(0, bl)
    others = {b for b in _phone_brand_vocab() if b != bl and len(b) >= 3}
    out = []
    for p in products:
        name = (p.get("name") or "").lower()
        if not any(a in name for a in aliases):
            continue
        if others:
            competing = [ob for ob in others if ob in name]
            if competing and bl not in name:
                continue
        out.append(p)
    return out


def filter_products_by_requested_brand(
    products: list[dict],
    brand: Optional[str],
    brand_aliases: Optional[list[str]] = None,
    *,
    strict_title: bool = False,
) -> list[dict]:
    """Match brand/model in product title (catalog brand field is often 'No Brand')."""
    if strict_title and brand:
        return filter_products_strict_brand_match(products, brand, brand_aliases)
    if not products:
        return products
    aliases = [
        a.strip().lower()
        for a in (brand_aliases or [])
        if a and str(a).strip().lower() not in _COLOR_NAME_TOKENS
    ]
    if brand:
        bl = brand.strip().lower()
        if bl in _COLOR_NAME_TOKENS:
            return products
        if bl and bl not in aliases:
            aliases.insert(0, bl)
    if not aliases:
        return products
    generic_brands = frozenset({"no brand", "generic", "unbranded", "na", "n/a", ""})
    out = []
    for p in products:
        name = (p.get("name") or "").lower()
        card_brand = (p.get("brand") or "").strip().lower()
        if any(a in name for a in aliases):
            out.append(p)
            continue
        if card_brand and card_brand not in generic_brands and any(a in card_brand for a in aliases):
            out.append(p)
    return out


def conflict_exclude_tokens_for_product_type(product_type: str) -> list[str]:
    p = (product_type or "").strip().lower()
    if p.endswith("s") and len(p) > 3:
        singular = p[:-1]
        if singular in _TYPE_CONFLICT_EXCLUDES:
            p = singular
    return list(_TYPE_CONFLICT_EXCLUDES.get(p, ()))


def filter_products_by_exclude_tokens(
    products: list[dict],
    exclude_tokens: list[str],
) -> list[dict]:
    """Drop titles containing excluded phrases (e.g. 'lg velvet' phone when user wants velvet material)."""
    bad = [e.strip().lower() for e in (exclude_tokens or []) if e and len(str(e).strip()) >= 3]
    if not bad or not products:
        return products
    out = []
    for p in products:
        name = (p.get("name") or "").lower()
        if any(phrase in name for phrase in bad):
            continue
        out.append(p)
    return out


def filter_products_by_requested_sku(products: list[dict], sku: str) -> list[dict]:
    """Strict SKU match on catalog cards (wildcard already applied in OpenSearch)."""
    needle = (sku or "").strip().lower()
    if not needle or not products:
        return products
    out = []
    for p in products:
        psku = (p.get("sku") or "").strip().lower()
        if not psku:
            continue
        if psku == needle or needle in psku or psku in needle:
            out.append(p)
    return out


def _card_purchase_price(p: dict) -> Optional[float]:
    display = p.get("price")
    if display not in (None, ""):
        try:
            dv = float(str(display).replace(",", "").replace("₹", "").strip())
            if dv > 0:
                return dv
        except (TypeError, ValueError):
            pass
    try:
        from services.welfog_api import customer_landed_price

        landed = customer_landed_price(p)
        if landed is not None:
            return landed
    except ImportError:
        pass
    for key in ("purchase_price", "unit_price", "sale_price", "mrp"):
        raw = p.get(key)
        if raw is None or raw == "":
            continue
        try:
            base = float(str(raw).replace(",", "").replace("₹", "").strip())
        except (TypeError, ValueError):
            continue
        if base <= 0:
            continue
        try:
            ship = float(str(p.get("shipping_cost") or 0).replace(",", "").strip())
        except (TypeError, ValueError):
            ship = 0.0
        if key == "purchase_price":
            return base + max(0.0, ship)
        return base
    return None


def filter_products_by_purchase_price(products: list[dict], spec: dict[str, Any]) -> list[dict]:
    pmax = spec.get("purchase_price_max")
    pmin = spec.get("purchase_price_min")
    if pmax is None and pmin is None:
        return products
    if products:
        try:
            from services.welfog_api import refresh_product_cards_landed_prices

            products = refresh_product_cards_landed_prices(
                products,
                title_query=(spec.get("title_query") or ""),
                color=spec.get("color"),
                category_id=spec.get("category_id"),
            )
        except ImportError:
            pass
    out = []
    for p in products:
        val = _card_purchase_price(p)
        if val is None:
            continue
        if pmax is not None and val > float(pmax):
            continue
        if pmin is not None and val < float(pmin):
            continue
        out.append(p)
    return out


def filter_products_by_rating_range(products: list[dict], spec: dict[str, Any]) -> list[dict]:
    rmin = spec.get("rating_min")
    rmax = spec.get("rating_max")
    if rmin is None and rmax is None:
        return products
    try:
        rmin_f = float(rmin) if rmin is not None else None
    except (TypeError, ValueError):
        rmin_f = None
    try:
        rmax_f = float(rmax) if rmax is not None else None
    except (TypeError, ValueError):
        rmax_f = None
    out = []
    for p in products:
        raw = p.get("rating")
        if raw is None or raw == "":
            if rmin_f is not None and rmin_f <= 0.01:
                out.append(p)
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            if rmin_f is not None and rmin_f <= 0.01:
                out.append(p)
            continue
        if rmin_f is not None:
            if rmin_f <= 0.01:
                if val > 0:
                    out.append(p)
                continue
            elif val < rmin_f:
                continue
        if rmax_f is not None and val >= rmax_f:
            continue
        out.append(p)
    return out


def _brand_mentioned_in_text(brand: str, text: str) -> bool:
    bl = (brand or "").strip().lower()
    if not bl or bl in _GENERIC_BRAND_WORDS:
        return False
    tl = f" {(text or '').lower()} "
    return bool(re.search(rf"\b{re.escape(bl)}\b", tl))


def _message_has_specific_product_type(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    for noun in _PRODUCT_NOUNS:
        if re.search(rf"\b{re.escape(noun)}\b", tl):
            return True
    return False


def scrub_filter_only_catalog_title(spec: dict[str, Any], comb: str = "") -> None:
    """Drop junk title_query on price/rating-only browse (e.g. 'jinki', 'price range 100150')."""
    if not spec or not comb:
        return
    has_price = (
        spec.get("purchase_price_min") is not None
        or spec.get("purchase_price_max") is not None
    )
    has_rating = spec.get("rating_min") is not None or spec.get("rating_max") is not None
    if not (has_price or has_rating):
        return
    if spec.get("pro_id") or spec.get("sku"):
        return
    tq = (spec.get("title_query") or "").strip().lower()
    if not tq:
        return
    if is_price_or_rating_browse_turn(comb):
        spec["title_query"] = ""
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        spec["title_match_strict"] = False
        return
    if has_rating and _turn_mentions_rating_filter(comb):
        junk = {
            "jinki", "jin", "jo", "products", "product", "items", "item",
            "produts", "produt", "dikha", "dikhao", "dikho", "dikhado", "bta", "btao", "bata",
            "rating", "above", "ke", "more", "than",
        }
        tokens = set(tq.split())
        if tokens <= junk or tq in junk or any(t in junk for t in tokens):
            spec["title_query"] = ""
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)


def is_price_or_rating_browse_turn(text: str) -> bool:
    """User browses by budget/rating without naming a new brand/product type."""
    low = (text or "").lower()
    if not low.strip():
        return False
    has_rating = _turn_mentions_rating_filter(low) or any(
        re.search(p, low, re.I) for p in _RATING_UNDER_PATTERNS + _RATING_PATTERNS
    )
    has_budget = _user_mentions_price_this_turn(low)
    if has_rating and not has_budget:
        has_budget = False
    elif not has_budget:
        has_budget = any(re.search(p, low, re.I) for p in _PRICE_UNDER + _PRICE_OVER)
        if has_budget and has_rating and not re.search(
            r"\b(?:rs|₹|rupee|inr|budget|price)\b", low, re.I
        ):
            has_budget = False
    if not (has_budget or has_rating):
        return False
    if _extract_brand_literal_from_text(text):
        return False
    if _message_has_specific_product_type(text):
        return False
    return True


def _scrub_bogus_brand_from_spec(spec: dict[str, Any], original_msg: str = "") -> None:
    if spec.get("_catalog_ai") or spec.get("_ai_single_pass"):
        if not (spec.get("brand") or "").strip():
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)
        return
    aliases = spec.get("brand_aliases")
    if aliases:
        if not isinstance(aliases, list):
            aliases = [str(aliases)]
        cleaned_aliases = []
        for a in aliases:
            al = str(a).strip().lower()
            if not al or al in _GENERIC_BRAND_WORDS or al in _FILTER_STOP:
                continue
            if al.isdigit() or al in {"rs", "rupee", "rupees", "inr", "under", "below", "above", "over"}:
                continue
            cleaned_aliases.append(a)
        if cleaned_aliases:
            spec["brand_aliases"] = cleaned_aliases
        else:
            spec.pop("brand_aliases", None)
            spec.pop("brand_name_match_only", None)
    brand = (spec.get("brand") or "").strip()
    if not brand:
        return
    bl = brand.lower()
    if bl in _GENERIC_BRAND_WORDS or bl in _BRAND_STOP:
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        return
    if original_msg and not _brand_mentioned_in_text(brand, original_msg):
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        spec.pop("title_match_strict", None)


def reconcile_catalog_spec_with_user_turn(
    spec: dict[str, Any],
    original_msg: str = "",
    msg_en: str = "",
    *,
    ctx: Optional[dict] = None,
    ai_route: Optional[dict] = None,
    ai_understanding: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Align AI/heuristic spec with what the user actually asked — stop Samsung/Welfog bleed
    on price/rating-only turns; pro_id lookups must not search 'dikhoa' text.
    """
    if not spec:
        return spec
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    heuristic = parse_product_filters_from_text(original_msg or comb)

    if spec.get("pro_id") or heuristic.get("pro_id"):
        pid = spec.get("pro_id") or heuristic.get("pro_id")
        spec["pro_id"] = int(pid)
        spec["title_query"] = ""
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        spec["title_match_strict"] = False
        return spec

    trust_ai = bool(spec.get("_catalog_ai")) or bool(
        ai_understanding and (ai_understanding.get("search_terms") or ai_understanding.get("product_requests"))
    )
    if trust_ai:
        try:
            from services.catalog_spec_semantics import reconcile_ai_first_catalog_spec

            return reconcile_ai_first_catalog_spec(
                spec,
                original_msg,
                msg_en,
                ctx=ctx,
                ai_route=ai_route,
                ai_understanding=ai_understanding,
            )
        except ImportError:
            pass

    for key in ("purchase_price_max", "purchase_price_min", "rating_min", "rating_max", "size", "color"):
        if heuristic.get(key) is not None and heuristic.get(key) != "":
            if trust_ai and spec.get(key) not in (None, ""):
                continue
            spec[key] = heuristic[key]

    rmin_h, rmax_h = _extract_rating_bounds_from_text((comb or "").lower())
    if rmin_h is not None and spec.get("rating_min") is None:
        spec["rating_min"] = rmin_h
    if rmax_h is not None and spec.get("rating_max") is None:
        spec["rating_max"] = rmax_h

    _scrub_price_filters_on_rating_turn(spec, comb, ai_understanding=ai_understanding)

    try:
        from services.catalog_spec_semantics import spec_has_price_filter
    except ImportError:
        spec_has_price_filter = lambda _s: False  # type: ignore

    if not spec_has_price_filter(spec) and not (
        isinstance(ai_understanding, dict)
        and ai_understanding.get("_ai_first")
    ):
        spec.pop("purchase_price_max", None)
        spec.pop("purchase_price_min", None)
        spec.pop("unit_price_max", None)
        spec.pop("unit_price_min", None)

    _scrub_bogus_brand_from_spec(spec, comb)

    continue_topic = bool(isinstance(ai_route, dict) and ai_route.get("continue_previous_topic"))
    price_rating_browse = is_price_or_rating_browse_turn(comb)
    has_product_type = _message_has_specific_product_type(comb)
    has_budget = spec.get("purchase_price_max") is not None or spec.get("purchase_price_min") is not None

    if price_rating_browse and not continue_topic and not has_product_type:
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
        spec["title_query"] = ""
        spec["title_match_strict"] = False
    elif (
        has_budget
        and ctx
        and not _brand_mentioned_in_text(str(spec.get("brand") or ""), comb)
    ):
        last = ((ctx.get("data") or {}).get("last_os_spec") or {})
        if not has_product_type and last.get("title_query"):
            spec["title_query"] = last["title_query"]
        if not spec.get("color") and last.get("color"):
            spec["color"] = last["color"]
        spec.pop("brand", None)
        spec.pop("brand_aliases", None)
        spec.pop("brand_name_match_only", None)
    elif price_rating_browse and continue_topic and ctx:
        last = ((ctx.get("data") or {}).get("last_os_spec") or {})
        if last.get("title_query") and not _message_has_specific_product_type(comb):
            spec["title_query"] = last["title_query"]
        if last.get("color") and not spec.get("color"):
            spec["color"] = last["color"]

    if not trust_ai:
        parsed_color, parsed_title = extract_color_and_product_title(comb)
        if parsed_color:
            spec["color"] = parsed_color
        if parsed_title and not price_rating_browse:
            spec["title_query"] = parsed_title

    _scrub_bogus_brand_from_spec(spec, comb)
    _scrub_price_filters_on_rating_turn(spec, comb)
    try:
        from services.catalog_spec_semantics import enforce_explicit_user_filters_only

        spec = enforce_explicit_user_filters_only(
            spec, original_msg, msg_en, ai_understanding=ai_understanding
        )
    except ImportError:
        pass
    scrub_filter_only_catalog_title(spec, comb)
    single_pass = spec.pop("_ai_single_pass", None)
    spec.pop("_catalog_ai", None)
    if single_pass:
        spec["_ai_single_pass"] = True
    return spec


def apply_catalog_post_filters(
    products: list[dict],
    spec: dict[str, Any],
    *,
    post_filter_mode: str = "strict",
) -> list[dict]:
    """
    Post-filter after OpenSearch hits.
    strict: mandatory + brand + colour
    os_filters_only: colour + brand only (OpenSearch already narrowed by title)
    light: colour only
    """
    if not products:
        return products
    mode = (post_filter_mode or "strict").strip().lower()
    exclude = spec.get("exclude_title_tokens") or []
    if exclude:
        products = filter_products_by_exclude_tokens(products, exclude)

    if mode == "strict":
        mandatory = spec.get("mandatory_match_tokens") or []
        if mandatory:
            products = filter_products_by_ai_mandatory_tokens(
                products,
                mandatory,
                product_type=spec.get("product_type"),
                skip_color_tokens=bool(spec.get("color")),
            )
        else:
            title_q = (spec.get("title_query") or "").strip()
            strict = bool(spec.get("title_match_strict"))
            if title_q:
                products = filter_products_by_title_relevance(products, title_q, strict=strict)
                products = filter_products_by_relevance_score_gap(products)

    if mode in ("os_filters_only", "light"):
        title_q = (spec.get("title_query") or "").strip()
        if title_q and not _is_category_only_browse_spec(spec):
            products = filter_products_by_title_relevance(products, title_q, strict=False)
            products = filter_products_by_relevance_score_gap(products, min_ratio=0.28)

    req_brand = (spec.get("_requested_brand") or spec.get("brand") or "").strip()
    req_aliases = spec.get("_requested_brand_aliases") or spec.get("brand_aliases") or []
    if mode in ("strict", "os_filters_only") and (req_brand or req_aliases):
        strict_brand = bool(spec.get("title_match_strict")) and not spec.get("brand_name_match_only")
        products = filter_products_by_requested_brand(
            products,
            req_brand or None,
            brand_aliases=req_aliases if isinstance(req_aliases, list) else None,
            strict_title=strict_brand,
        )

    if mode in ("strict", "os_filters_only", "light") and spec.get("color"):
        products = filter_products_by_requested_color(products, spec.get("color"))

    if spec.get("sku"):
        products = filter_products_by_requested_sku(products, str(spec["sku"]))

    if spec.get("purchase_price_max") is not None or spec.get("purchase_price_min") is not None:
        products = filter_products_by_purchase_price(products, spec)

    if spec.get("rating_min") is not None or spec.get("rating_max") is not None:
        products = filter_products_by_rating_range(products, spec)

    # product_type from AI is authoritative for any language/style ask — always enforce
    # when present (including light/category paths) so sneakers ≠ fashion misc.
    ptype = (spec.get("product_type") or "").strip()
    if ptype and not _is_category_only_browse_spec(spec):
        products = filter_products_by_ai_mandatory_tokens(
            products,
            [ptype],
            product_type=ptype,
            skip_color_tokens=bool(spec.get("color")),
        )

    audience = spec.get("audience_tokens") or []
    if audience:
        products = filter_products_by_audience_tokens(products, audience)

    mats = spec.get("material_tokens") or []
    if mats:
        products = filter_products_by_material_tokens(products, mats)

    if _is_category_only_browse_spec(spec) and spec.get("category_id"):
        products = _filter_products_to_department(products, spec["category_id"])

    return products


def _filter_products_to_department(products: list[dict], category_id) -> list[dict]:
    """
    Drop cards that clearly belong to another department after a category browse.
    Membership is by live leaf category_id (or exact unique department label).
    Keeps rows with unknown/missing category metadata (REST often omits it).
    """
    if not products or not category_id:
        return products
    try:
        from services.welfog_api import (
            department_leaf_category_ids,
            department_unique_catalog_label_names,
            _normalize_cat_name,
        )

        leaf_ids = {str(x) for x in department_leaf_category_ids(category_id)}
        unique_labels = {
            _normalize_cat_name(n)
            for n in department_unique_catalog_label_names(category_id)
            if n
        }
    except Exception:
        return products
    if not leaf_ids and not unique_labels:
        return products

    kept: list[dict] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        raw_cid = p.get("category_id")
        if raw_cid is None and isinstance(p.get("data"), dict):
            raw_cid = p["data"].get("category_id")
        if raw_cid is not None and str(raw_cid).strip():
            if str(raw_cid).strip() in leaf_ids:
                kept.append(p)
            continue
        cname = (p.get("category_name") or "").strip()
        if not cname and isinstance(p.get("data"), dict):
            cname = (p["data"].get("category_name") or "").strip()
        if cname:
            if _normalize_cat_name(cname) in unique_labels:
                kept.append(p)
            continue
        # No category metadata on card — keep (REST rows often lack it; trust gate already ran).
        kept.append(p)
    return kept


def _edit_distance(a: str, b: str) -> int:
    """Small Levenshtein distance — used to mirror OpenSearch fuzziness in post-filters."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    # Two-row DP — enough for short catalog tokens.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _title_token_in_name(tok: str, name_lower: str) -> bool:
    """True when token (or a compound / near-typo form of it) appears in the product name.

    OpenSearch already fuzzy-matches titles; this post-filter must not undo that by
    requiring exact spelling. Short edit-distance tolerance (AUTO-like) covers
    typos such as bnaiyan↔baniyan without any product keyword lists.
    """
    tok = (tok or "").strip().lower()
    if not tok or not name_lower:
        return False
    if tok in name_lower:
        return True
    if tok.endswith("s") and len(tok) > 3 and tok[:-1] in name_lower:
        return True
    if not tok.endswith("s") and f"{tok}s" in name_lower:
        return True
    for form in _compound_token_surface_forms(tok):
        if form in name_lower:
            return True
        if form.endswith("s") and len(form) > 3 and form[:-1] in name_lower:
            return True
        if not form.endswith("s") and f"{form}s" in name_lower:
            return True
    # Compound without spaces: flipflops ↔ "flip flops"
    if " " not in tok and len(tok) >= 7:
        for split in _compound_space_variants(tok):
            parts = split.split()
            if len(parts) == 2 and parts[0] in name_lower and parts[1] in name_lower:
                return True
    # Typo-tolerant: mirror OpenSearch fuzziness AUTO
    # (0 for ≤2, 1 for 3–5, 2 for ≥6) so "bnaiyan" still matches "baniyan".
    if len(tok) >= 4:
        if len(tok) <= 5:
            max_ed = 1
        else:
            max_ed = 2
        for word in re.findall(r"[a-z0-9]{3,}", name_lower):
            if abs(len(word) - len(tok)) > max_ed:
                continue
            if _edit_distance(tok, word) <= max_ed:
                return True
    return False


def filter_products_by_title_relevance(
    products: list[dict],
    title_query: str,
    *,
    strict: bool = False,
    mandatory_tokens: Optional[list[str]] = None,
) -> list[dict]:
    """
    Keep products whose names cover the distinctive query tokens.

    Soft mode no longer keeps hits that only share a gender/department word
    (e.g. "men") — that caused Men Fashion flip-flops to pollute vest/baniyan results.
    """
    if mandatory_tokens:
        return filter_products_by_ai_mandatory_tokens(products, mandatory_tokens)

    need = _title_match_tokens(title_query)
    need = [t for t in need if t not in _COLOR_NAME_TOKENS and t not in _MATERIAL_TITLE_TOKENS]
    if not need or not products:
        return products

    # Prefer tokens that are not ultra-generic single adjectives.
    distinctive = [t for t in need if t not in _GENERIC_TITLE_MODIFIERS]
    require = distinctive if distinctive else need

    matched = []
    for p in products:
        name = (p.get("name") or "").lower()
        if not name:
            continue
        if strict:
            if all(_title_token_in_name(tok, name) for tok in require):
                matched.append(p)
        elif len(require) >= 2:
            # Soft multi-token: all distinctive tokens (gender already stripped).
            if all(_title_token_in_name(tok, name) for tok in require):
                matched.append(p)
        else:
            # Soft single noun (baniyan, flipflops, vest): compound-aware match.
            if any(_title_token_in_name(tok, name) for tok in require):
                matched.append(p)
    return matched


def filter_products_by_relevance_score_gap(
    products: list[dict],
    *,
    min_ratio: float = 0.35,
    absolute_floor: float = 1.0,
) -> list[dict]:
    """
    Drop weak BM25 outliers relative to the top hit.

    Scalable for any catalog size — no product-type rules. When the best vest
    scores ~20 and flip-flops score ~3 from a leftover category leak, flip-flops go.
    """
    if not products or len(products) <= 1:
        return products
    scores = [float(p.get("score") or 0) for p in products]
    top = max(scores)
    if top <= 0:
        return products
    floor = max(absolute_floor, top * float(min_ratio))
    kept = [p for p in products if float(p.get("score") or 0) >= floor]
    return kept if kept else products[:1]


def _ensure_project_dotenv_loaded() -> None:
    """Load parent .env when modules run outside app.py (tests, scripts)."""
    try:
        from services.env_loader import ensure_dotenv_loaded

        ensure_dotenv_loaded()
    except Exception:
        pass


def _env_opensearch_config():
    _ensure_project_dotenv_loaded()
    url = (os.getenv("OPENSEARCH_URL") or "").strip()
    user = (os.getenv("OPENSEARCH_USER") or "").strip()
    password = (os.getenv("OPENSEARCH_PASS") or "").strip()
    if not url:
        return None
    # Auth only from .env — never hardcode OpenSearch user/password in source.
    return {"url": url, "auth": (user, password) if user else None}


def is_opensearch_configured() -> bool:
    return _env_opensearch_config() is not None


def _scrub_color_meta_from_title(title: str) -> str:
    """Remove filler colour words from title_query (never search 'color cover')."""
    if not title:
        return ""
    tq = re.sub(r"\b(color|colour|rang)\b", " ", title, flags=re.IGNORECASE)
    return " ".join(tq.split()).strip()


def _extract_brand_literal_from_text(text: str) -> Optional[str]:
    """Brand from user text with original spelling (Gyld not gyid from translation)."""
    raw = (text or "").strip()
    if not raw:
        return None
    for pat in _BRAND_PATTERNS:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            brand = m.group(1).strip()
            if brand.lower() not in _BRAND_STOP and brand.lower() not in _FILTER_STOP:
                return brand if len(brand) > 4 else brand.title()
    return None


def _normalize_brand_label(token: str) -> str:
    t = (token or "").strip().lower()
    if t == "iphone":
        return "iPhone"
    if t in ("oneplus", "redmi", "oppo", "vivo", "realme", "infinix"):
        return t.capitalize()
    return t.title() if t else ""


def _infer_brand_from_message(text: str) -> Optional[str]:
    """Detect phone/device brand in Hinglish/English (samsung ke liye cover, sam sung)."""
    low = (text or "").lower()
    if not low:
        return None
    for pat in _BRAND_KE_KI_KA_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    return _normalize_brand_label(g)
    vocab = sorted(_phone_brand_vocab(), key=len, reverse=True)
    earliest: Optional[tuple[int, str]] = None
    for b in vocab:
        m = re.search(rf"\b{re.escape(b)}\b", low)
        if m and (earliest is None or m.start() < earliest[0]):
            earliest = (m.start(), b)
    if earliest:
        return _normalize_brand_label(earliest[1])
    # Space/typo tolerant: "sam sung", "one plus", "red mi" → brand vocab.
    compact = re.sub(r"[^a-z0-9]", "", low)
    for b in vocab:
        bc = re.sub(r"[^a-z0-9]", "", b.lower())
        if len(bc) >= 4 and bc in compact:
            return _normalize_brand_label(b)
    return None


def _catalog_title_for_search(spec: dict[str, Any]) -> str:
    """REST/OpenSearch name= must include brand (API search box uses 'samsung' not 'cover' alone)."""
    brand = (spec.get("brand") or "").strip()
    tq = (spec.get("title_query") or "").strip()
    if brand:
        bl = brand.lower()
        if not tq or bl not in tq.lower():
            return f"{brand} {tq}".strip() if tq else brand
    return tq


def extract_structured_catalog_filters(
    text: str,
    *,
    category_id: Optional[int] = None,
    color: Optional[str] = None,
    pro_id: Optional[int] = None,
) -> dict[str, Any]:
    """SKU, price, rating, size, pro_id from text — no title_query/brand (AI owns those)."""
    full = parse_product_filters_from_text(
        text,
        category_id=category_id,
        color=color,
        pro_id=pro_id,
    )
    out: dict[str, Any] = {}
    for key in (
        "sku",
        "pro_id",
        "purchase_price_max",
        "purchase_price_min",
        "rating_min",
        "rating_max",
        "size",
        "category_id",
        "color",
        "in_stock_only",
        "sort",
    ):
        val = full.get(key)
        if val is not None and val != "":
            out[key] = val
    return out


def _stamp_requested_brand(spec: dict[str, Any]) -> None:
    """Preserve user-requested brand for post-filter even when OS relax drops brand field."""
    brand = (spec.get("brand") or "").strip()
    if not brand:
        return
    spec["_requested_brand"] = brand
    aliases = [
        str(a).strip()
        for a in (spec.get("brand_aliases") or [])
        if a and str(a).strip()
    ]
    if brand.lower() not in [a.lower() for a in aliases]:
        aliases.insert(0, brand)
    spec["_requested_brand_aliases"] = aliases[:8]


def finalize_catalog_search_spec(
    spec: dict[str, Any],
    original_msg: str = "",
    msg_en: str = "",
) -> dict[str, Any]:
    """Colour + brand + title_query aligned with what the user actually asked."""
    # Per-part catalog search passes only that part as original_msg — do not widen blob.
    blob = f"{original_msg or ''} {msg_en or ''}".strip()
    trust_ai = bool(spec.get("_catalog_ai") or spec.get("_ai_single_pass"))
    user_title = ""
    if blob:
        if not trust_ai:
            user_color, user_title = extract_color_and_product_title(blob)
            if not user_color:
                fuzzy = normalize_color_fuzzy(blob)
                if fuzzy and color_hue_mentioned_in_text(fuzzy, blob):
                    user_color = fuzzy
            if user_color:
                spec["color"] = user_color
            elif spec.get("color") and not color_hue_mentioned_in_text(
                str(spec.get("color")), blob
            ):
                spec["color"] = ""
            literal_brand = _extract_brand_literal_from_text(original_msg or "")
            if literal_brand:
                spec["brand"] = literal_brand
            else:
                inferred_brand = _infer_brand_from_message(blob)
                if inferred_brand and not spec.get("brand"):
                    spec["brand"] = inferred_brand
            if user_title and not spec.get("title_query"):
                spec["title_query"] = user_title
        elif spec.get("color") and not color_hue_mentioned_in_text(
            str(spec.get("color")), blob
        ):
            spec["color"] = ""

    brand_only_browse = False
    brand = (spec.get("brand") or "").strip()
    if brand:
        bl = brand.lower()
        aliases = list(spec.get("brand_aliases") or [])
        if bl not in [str(a).lower() for a in aliases]:
            aliases.insert(0, brand)
        en_brand = _extract_brand_literal_from_text(msg_en or "")
        if not en_brand and msg_en:
            en_brand = parse_product_filters_from_text(msg_en).get("brand")
        if en_brand and en_brand.lower() not in [str(a).lower() for a in aliases]:
            aliases.append(en_brand)
        spec["brand_aliases"] = aliases[:8]
        spec["brand_name_match_only"] = True
        blob_low = blob.lower()
        brand_only = bool(
            re.search(rf"\b{re.escape(bl)}\s+brand\b", blob_low)
            and not (user_title or "").strip()
        )
        if brand_only or (
            (user_title or "").strip().lower() in ("product", "products")
            and re.search(r"\bbrand\b", blob_low)
        ):
            spec["title_query"] = ""
            spec["title_match_strict"] = False
            brand_only_browse = True
        else:
            spec["title_match_strict"] = True
        mode = (spec.get("match_mode") or "").strip().lower()
        if not mode or mode == "universal":
            spec["match_mode"] = "strict"

    tq = (spec.get("title_query") or "").strip()
    if spec.get("color"):
        tq = _strip_color_from_title_query(tq, str(spec["color"]))
    tq = _scrub_color_meta_from_title(tq)
    if brand_only_browse:
        spec["title_query"] = ""
    else:
        merged_title = _catalog_title_for_search(spec)
        if merged_title:
            spec["title_query"] = _scrub_color_meta_from_title(merged_title)
        elif tq:
            spec["title_query"] = tq
    if spec.get("brand"):
        _stamp_requested_brand(spec)
    return spec


def _enforce_user_color_product_spec(
    spec: dict[str, Any],
    original_msg: str,
    msg_en: str = "",
) -> dict[str, Any]:
    """Backward-compatible alias — use finalize_catalog_search_spec."""
    return finalize_catalog_search_spec(spec, original_msg, msg_en)


def is_single_color_product_query(text: str) -> bool:
    """One product ask with a colour filter — must not split into multi searches."""
    c, t = extract_color_and_product_title(text or "")
    return bool(c and t)


def build_catalog_search_spec(
    original_msg: str,
    msg_en: str = "",
    *,
    ai: Optional[dict[str, Any]] = None,
    category_id: Optional[int] = None,
    color: Optional[str] = None,
    pro_id: Optional[int] = None,
    ctx: Optional[dict] = None,
    ai_route: Optional[dict] = None,
) -> dict[str, Any]:
    """Merge message heuristics + AI JSON → OpenSearch filter spec (colour, price, brand, SKU, pro_id)."""
    ai_has_shopping = bool(
        ai
        and (
            ai.get("is_shopping")
            or ai.get("_ai_first")
            or (ai.get("search_terms") or "").strip()
            or ai.get("product_requests")
            or ai.get("category_browse")
            or ai.get("sku")
            or ai.get("pro_id")
            or ai.get("rating_min") is not None
            or ai.get("rating_max") is not None
            or ai.get("max_price") is not None
            or ai.get("min_price") is not None
            or ai.get("brand")
        )
    )
    if ai_has_shopping:
        try:
            from services.catalog_spec_semantics import extract_gap_fill_ids_from_text

            spec = extract_gap_fill_ids_from_text(
                original_msg or "", pro_id=pro_id
            )
            if msg_en:
                spec_en = extract_gap_fill_ids_from_text(msg_en, pro_id=pro_id)
                if spec_en.get("sku") and not spec.get("sku"):
                    spec["sku"] = spec_en["sku"]
                if spec_en.get("pro_id") and not spec.get("pro_id"):
                    spec["pro_id"] = spec_en["pro_id"]
        except ImportError:
            spec = extract_structured_catalog_filters(
                original_msg or "",
                category_id=category_id,
                color=color,
                pro_id=pro_id,
            )
        if category_id and not spec.get("category_id"):
            spec["category_id"] = category_id
        if color and not spec.get("color"):
            spec["color"] = color
    else:
        spec = parse_product_filters_from_text(
            original_msg or "",
            category_id=category_id,
            color=color,
            pro_id=pro_id,
        )
        if msg_en:
            spec_en = parse_product_filters_from_text(
                msg_en,
                category_id=category_id,
                color=color or spec.get("color"),
                pro_id=pro_id,
            )
            spec = _merge_filter_specs(spec, spec_en)
    if ai:
        heuristic_color = spec.get("color")
        try:
            from services.catalog_spec_from_ai import merge_ai_into_catalog_spec

            if ai.get("_ai_first"):
                spec["_ai_single_pass"] = True
            spec = merge_ai_into_catalog_spec(spec, ai, original_msg=original_msg)
            if not spec.get("color") and heuristic_color:
                spec["color"] = heuristic_color
        except ImportError:
            pass
    elif not spec.get("title_query"):
        spec = build_product_search_spec(
            original_msg,
            msg_en,
            category_id=category_id,
            color=color or spec.get("color"),
            pro_id=pro_id,
        )
    if pro_id and not spec.get("pro_id"):
        spec["pro_id"] = pro_id
    if category_id and not spec.get("category_id"):
        spec["category_id"] = category_id
    spec["_filter_user_msg"] = original_msg or ""
    spec = sanitize_product_search_spec(spec)
    _normalize_generic_title_query(spec, original_msg)
    _apply_material_tokens_to_spec(spec, original_msg)
    if msg_en:
        _apply_material_tokens_to_spec(spec, msg_en)
    spec = finalize_catalog_search_spec(spec, original_msg, msg_en)
    spec = sanitize_product_search_spec(spec)
    spec = reconcile_catalog_spec_with_user_turn(
        spec,
        original_msg,
        msg_en,
        ctx=ctx,
        ai_route=ai_route,
        ai_understanding=ai if ai_has_shopping else None,
    )
    if ai_has_shopping and isinstance(ai, dict) and ai.get("_ai_first"):
        spec["_ai_single_pass"] = True
    if isinstance(ai_route, dict) and ai_route.get("_ai_single_pass"):
        spec["_ai_single_pass"] = True
    try:
        from services.product_filter_pipeline import finalize_catalog_spec_for_api

        spec = finalize_catalog_spec_for_api(
            spec,
            original_msg,
            msg_en,
            ai_understanding=ai if ai_has_shopping else None,
            ctx=ctx,
        )
    except ImportError:
        pass
    return spec


def _to_number(val) -> Optional[float]:
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


def _extract_first(patterns: tuple, text: str, flags=re.IGNORECASE) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return (m.group(1) if m.lastindex else m.group(0)).strip()
    return None


def _brand_name_should_clauses(brand_or_aliases) -> list[dict]:
    """
    Welfog index often has brand='No Brand' — match phone/model words in name + category, not brand field alone.
    """
    tokens: list[str] = []
    if isinstance(brand_or_aliases, str):
        tokens = [brand_or_aliases.strip()]
    elif isinstance(brand_or_aliases, list):
        tokens = [str(x).strip() for x in brand_or_aliases if str(x).strip()]
    should: list[dict] = []
    seen: set[str] = set()
    for raw in tokens:
        low = raw.lower()
        if not low or low in seen or low in ("no brand", "brand", "generic", "unbranded"):
            continue
        seen.add(low)
        should.append(
            {
                "multi_match": {
                    "query": low,
                    "fields": ["name^5", "sku^2", "category_name", "brand"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            }
        )
    return should


def parse_product_filters_from_text(
    text: str,
    *,
    category_id: Optional[int] = None,
    color: Optional[str] = None,
    title_hint: str = "",
    pro_id: Optional[int] = None,
) -> dict[str, Any]:
    """Parse natural-language product filter/sort request (English + Hinglish)."""
    raw = (text or "").strip()
    low = raw.lower()
    spec: dict[str, Any] = {
        "title_query": "",
        "color": color,
        "size": None,
        "brand": None,
        "sku": None,
        "pro_id": pro_id,
        "category_id": category_id,
        "unit_price_min": None,
        "unit_price_max": None,
        "purchase_price_min": None,
        "purchase_price_max": None,
        "rating_min": None,
        "rating_max": None,
        "in_stock_only": False,
        "sort": None,
    }

    parsed_color, parsed_title = extract_color_and_product_title(raw)
    if parsed_color:
        spec["color"] = parsed_color
    elif not spec.get("color"):
        try:
            from services.product_intent_parser import extract_color_word_boundary

            spec["color"] = extract_color_word_boundary(raw)
        except ImportError:
            hue = normalize_color_fuzzy(raw)
            if hue and color_hue_mentioned_in_text(hue, raw):
                spec["color"] = hue
    if parsed_title:
        spec["title_query"] = parsed_title

    sku_inline = _extract_sku_from_text(raw)
    if sku_inline:
        spec["sku"] = sku_inline

    for pat in _SIZE_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            spec["size"] = _normalize_catalog_size_value(m.group(1).strip())
            break

    brand_m = _extract_brand_literal_from_text(raw)
    if brand_m:
        spec["brand"] = brand_m
    if not spec.get("brand"):
        inferred = _infer_brand_from_message(low)
        if inferred:
            spec["brand"] = inferred

    for pat in _RATING_UNDER_PATTERNS:
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            star_cap = None
            if m.lastindex and m.group(1):
                try:
                    star_cap = float(m.group(1))
                except ValueError:
                    star_cap = None
            if star_cap is None and re.search(r"\blow\s+rating\b", low):
                star_cap = 3.0
            if star_cap is not None:
                spec["rating_max"] = star_cap
            break

    if spec.get("rating_min") is None:
        for pat in _RATING_ABOVE_PATTERNS:
            m = re.search(pat, low, re.IGNORECASE)
            if m and m.lastindex:
                try:
                    spec["rating_min"] = float(m.group(1))
                except ValueError:
                    pass
                break
    m_rating_above = re.search(
        r"\b(?:rating|stars?)\s*(\d(?:\.\d)?)\s*se\s+(?:jyada|zyada|zada|upar|zyaada)\b",
        low,
        re.IGNORECASE,
    )
    if m_rating_above:
        try:
            spec["rating_min"] = float(m_rating_above.group(1))
        except ValueError:
            pass
    m_rating_products = re.search(
        r"\b(?:jin|jinki|jo)\s+products?\s+.*\b(?:rating|stars?)\b.*\b(?:jyada|zyada|upar|above|se)\b",
        low,
        re.IGNORECASE,
    )
    if m_rating_products and spec.get("rating_min") is None:
        spec["rating_min"] = 0.01

    if spec.get("rating_max") is None:
        for pat in _RATING_PATTERNS:
            m = re.search(pat, low, re.IGNORECASE)
            if m:
                wants_min = bool(re.search(r"\b(?:at\s+least|min|minimum|>=|above|se\s+upar)\b", low))
                filter_by_rating = bool(
                    re.search(r"\b(?:wale|wali|products?|dikhao|dikha|filter|only)\b", low)
                )
                if m.lastindex and m.group(1):
                    try:
                        star_val = float(m.group(1))
                    except ValueError:
                        star_val = None
                    if star_val is not None and (
                        wants_min
                        or filter_by_rating
                        or re.search(rf"\b{re.escape(m.group(1))}\s*(?:star|stars)\b", low)
                    ):
                        spec["rating_min"] = star_val
                    elif star_val is None:
                        spec["sort"] = spec["sort"] or "rating_desc"
                    else:
                        spec["sort"] = spec["sort"] or "rating_desc"
                else:
                    spec["sort"] = spec["sort"] or "rating_desc"
                break

    if not spec.get("sku"):
        sku = _extract_first(_SKU_PATTERNS, raw)
        if sku:
            spec["sku"] = sku.upper() if sku.isupper() else sku

    if re.search(r"\b(best|top|sabse\s+achi|sabse\s+badiya)\b", low):
        spec["sort"] = spec["sort"] or "rating_desc"

    if not spec.get("pro_id"):
        pro_id_val = _extract_first(_PRO_ID_PATTERNS, low)
        if pro_id_val and pro_id_val.isdigit():
            spec["pro_id"] = int(pro_id_val)

    if re.search(r"\b(?:in\s+stock|available|stock\s+me|stock\s+hai)\b", low):
        spec["in_stock_only"] = True

    if any(k in low for k in _SORT_PRICE_ASC):
        spec["sort"] = "purchase_price_asc"
    elif any(k in low for k in _SORT_PRICE_DESC):
        spec["sort"] = "purchase_price_desc"
    elif any(k in low for k in _SORT_RATING):
        spec["sort"] = "rating_desc"
    elif any(k in low for k in _SORT_PURCHASE_ASC):
        spec["sort"] = "purchase_price_asc"

    if re.search(r"\bsort\s+by\s+(price|rating|purchase)", low):
        kind = re.search(r"\bsort\s+by\s+(price|rating|purchase)", low).group(1)
        if kind == "price":
            spec["sort"] = "purchase_price_asc" if "low" in low or "asc" in low else "purchase_price_desc"
        elif kind == "rating":
            spec["sort"] = "rating_desc"
        else:
            spec["sort"] = "purchase_price_asc"

    clean_hint = (title_hint or "").strip()
    if clean_hint:
        if not spec.get("size"):
            spec["size"] = _extract_size_from_text(raw) or _extract_size_from_text(clean_hint)
        spec["title_query"] = _strip_size_from_title_query(clean_hint, spec.get("size") or "")
        if spec.get("color"):
            spec["title_query"] = _strip_color_from_title_query(
                spec["title_query"], str(spec["color"])
            )
        spec["title_query"] = re.sub(
            r"\b(?:product|products|item|items|ka|ke|ki|dikhao|dikha|dikao|dikhe|show|chahiye)\b",
            " ",
            spec["title_query"],
            flags=re.IGNORECASE,
        )
        spec["title_query"] = " ".join(spec["title_query"].split()).strip()
        if len(spec["title_query"]) < 2 and spec.get("size"):
            spec["title_query"] = _extract_product_keywords(low) or "cover"
        return spec

    if parsed_title:
        if spec.get("color"):
            spec["title_query"] = _strip_color_from_title_query(parsed_title, str(spec["color"]))
        if spec.get("size"):
            spec["title_query"] = _strip_size_from_title_query(spec["title_query"], str(spec["size"]))
        return spec

    title = raw
    scrub = title
    for pat in (
        _SIZE_PATTERNS + _BRAND_PATTERNS + _SKU_PATTERNS + _PRO_ID_PATTERNS
        + _PRICE_UNDER + _PRICE_OVER + _RATING_PATTERNS
    ):
        scrub = re.sub(pat, " ", scrub, flags=re.IGNORECASE)
    scrub = re.sub(
        r"\b(color|colour|rang|brand|size|sku|pro\s*id|product\s*id|rating|sort|filter|"
        r"under|below|above|cheapest|sasta|mehnga|stock|purchase\s+price|hai|kya)\b",
        " ",
        scrub,
        flags=re.IGNORECASE,
    )
    if spec["color"]:
        scrub = re.sub(re.escape(spec["color"]), " ", scrub, flags=re.IGNORECASE)
    if spec.get("brand"):
        scrub = re.sub(re.escape(str(spec["brand"])), " ", scrub, flags=re.IGNORECASE)
    product_kw = _extract_product_keywords(low)
    words = [
        w
        for w in re.findall(r"[a-z0-9]+", scrub.lower())
        if w not in _FILTER_STOP and w not in _BRAND_STOP and len(w) > 1
    ]
    mapped_words = []
    for w in words:
        if spec.get("sku") and w == spec["sku"].lower():
            continue
        mw = _NOUN_TYPO_MAP.get(w, w)
        if mw not in mapped_words:
            mapped_words.append(mw)
    spec["title_query"] = product_kw or " ".join(mapped_words[:8]).strip()
    if spec.get("sku"):
        spec["title_query"] = product_kw or ""
    if len(spec["title_query"]) < 2 and any(
        [
            spec.get("brand"),
            spec.get("sku"),
            spec.get("pro_id"),
            category_id,
            spec.get("color"),
            spec.get("size"),
            spec.get("sort"),
            spec.get("unit_price_max") is not None,
        ]
    ):
        spec["title_query"] = product_kw or ""

    return spec


_OS_CATEGORY_NAME_CACHE: dict[str, Any] = {"ts": 0.0, "names": []}


def _os_indexed_category_names() -> list[str]:
    """Distinct category_name values currently in the OpenSearch index (small agg)."""
    import time as _time

    now = _time.time()
    if _OS_CATEGORY_NAME_CACHE["names"] and now - float(_OS_CATEGORY_NAME_CACHE["ts"] or 0) < 300:
        return list(_OS_CATEGORY_NAME_CACHE["names"] or [])
    cfg = _env_opensearch_config()
    if not cfg:
        return []
    body = {
        "size": 0,
        "aggs": {"cats": {"terms": {"field": "category_name", "size": 200}}},
    }
    try:
        r = requests.post(
            cfg["url"],
            auth=cfg.get("auth"),
            json=body,
            timeout=8,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code >= 400:
            return list(_OS_CATEGORY_NAME_CACHE["names"] or [])
        buckets = (
            ((r.json() or {}).get("aggregations") or {}).get("cats") or {}
        ).get("buckets") or []
        names = [str(b.get("key") or "").strip() for b in buckets if b.get("key")]
        _OS_CATEGORY_NAME_CACHE["ts"] = now
        _OS_CATEGORY_NAME_CACHE["names"] = names
        return names
    except Exception:
        return list(_OS_CATEGORY_NAME_CACHE["names"] or [])


def _os_category_names_for_department(category_id, ctx=None) -> list[str]:
    """
    Fallback name bridge when leaf category ids are unavailable.
    Exact match only between live department-unique labels and indexed category_name
    values — no keyword blocklists; cross-department labels are skipped via the
    live category tree.
    """
    try:
        from services.welfog_api import (
            department_unique_catalog_label_names,
            _normalize_cat_name,
        )
    except ImportError:
        return []

    unique_labels = department_unique_catalog_label_names(category_id, ctx)
    if not unique_labels:
        return []

    by_key: dict[str, str] = {}
    for lab in unique_labels:
        key = _normalize_cat_name(lab)
        if key and key not in by_key:
            by_key[key] = lab.strip()

    matched: list[str] = []
    seen: set[str] = set()
    for os_name in _os_indexed_category_names():
        key = _normalize_cat_name(os_name)
        if not key or key not in by_key:
            continue
        # Prefer the live catalog spelling when present in the index; else OS key.
        pick = by_key[key]
        if pick in (_OS_CATEGORY_NAME_CACHE.get("names") or []):
            out = pick
        else:
            out = str(os_name).strip()
        if out and out not in seen:
            seen.add(out)
            matched.append(out)
    return matched


def _opensearch_category_filter_clause(spec: dict[str, Any]) -> Optional[dict]:
    """
    Scope OpenSearch to a nav department.

    Nav/inner category ids and OpenSearch category_id values are different
    namespaces — never rely on leaf-id-only filters. Prefer exact
    category_name matches from live department-unique labels; also OR the
    requested id (and leaf ids) in case some docs use that space.
    """
    if not spec.get("category_id"):
        return None
    try:
        cid = int(spec["category_id"])
    except (TypeError, ValueError):
        return None

    should: list[dict] = [{"term": {"category_id": cid}}]
    try:
        from services.welfog_api import department_leaf_category_ids

        id_set: list[int] = [cid]
        for lid in department_leaf_category_ids(cid):
            try:
                lid_i = int(lid)
            except (TypeError, ValueError):
                continue
            if lid_i not in id_set:
                id_set.append(lid_i)
        if len(id_set) > 1:
            should[0] = {"terms": {"category_id": id_set}}

        for name in _os_category_names_for_department(cid):
            should.append({"term": {"category_name": name}})
    except Exception:
        pass

    if len(should) == 1:
        return should[0]
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _build_opensearch_body(spec: dict[str, Any], size: int = PAGE_SIZE, offset: int = 0) -> dict:
    must: list[dict] = []
    filters: list[dict] = []

    if spec.get("pro_id"):
        filters.append({"term": {"pro_id": int(spec["pro_id"])}})
    if spec.get("sku"):
        sku_val = spec["sku"]
        if spec.get("strict_sku_match"):
            filters.append({"term": {"sku": {"value": sku_val, "case_insensitive": True}}})
        else:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"sku": {"value": sku_val, "case_insensitive": True}}},
                            {"wildcard": {"sku": f"*{sku_val}*"}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
    if spec.get("category_id"):
        cat_clause = _opensearch_category_filter_clause(spec)
        if cat_clause:
            filters.append(cat_clause)
    if spec.get("color"):
        c = str(spec["color"]).strip()
        color_values: list[str] = []
        for variant in (c, c.title(), c.capitalize(), *_color_spelling_variants(c)):
            v = (variant or "").strip()
            if v and v not in color_values:
                color_values.append(v)
        color_should: list[dict] = []
        for cv in color_values:
            color_should.append(
                {
                    "term": {
                        "color_name": {
                            "value": cv,
                            "case_insensitive": True,
                        }
                    }
                }
            )
            color_should.append({"match_phrase": {"color_name": cv}})
        if c.lower() == "multicolor":
            color_should.append({"match_phrase": {"name": "multicolor"}})
        filters.append({"bool": {"should": color_should, "minimum_should_match": 1}})
    if spec.get("size"):
        filters.append({"match": {"size": spec["size"]}})
    aliases = spec.get("brand_aliases") or []
    brand_tokens = aliases if isinstance(aliases, list) and aliases else []
    if not brand_tokens and spec.get("brand"):
        brand_tokens = [spec["brand"]]
    name_should = _brand_name_should_clauses(brand_tokens)
    if name_should and not spec.get("brand_name_match_only"):
        filters.append({"bool": {"should": name_should, "minimum_should_match": 1}})
    if spec.get("unit_price_min") is not None:
        filters.append({"range": {"unit_price": {"gte": spec["unit_price_min"]}}})
    if spec.get("unit_price_max") is not None:
        filters.append({"range": {"unit_price": {"lte": spec["unit_price_max"]}}})
    if not spec.get("_os_price_filter_disabled"):
        has_landed = bool(spec.get("_landed_price_filter"))
        pmin_os = spec.get("purchase_price_min")
        pmax_os = spec.get("purchase_price_max")
        if has_landed:
            # Index purchase_price excludes shipping — widen max slightly, then post-filter
            # on live landed price (purchase + shipping) for accurate budget matching.
            if pmin_os is not None:
                filters.append({"range": {"purchase_price": {"gte": float(pmin_os)}}})
            if pmax_os is not None:
                try:
                    pmax_f = float(pmax_os)
                except (TypeError, ValueError):
                    pmax_f = None
                if pmax_f is not None:
                    ship_buf = max(120.0, pmax_f * 0.12)
                    filters.append(
                        {"range": {"purchase_price": {"lte": pmax_f + ship_buf}}}
                    )
        else:
            if pmin_os is not None:
                filters.append({"range": {"purchase_price": {"gte": pmin_os}}})
            if pmax_os is not None:
                filters.append({"range": {"purchase_price": {"lte": pmax_os}}})
    if spec.get("rating_min") is not None:
        filters.append({"range": {"rating": {"gte": spec["rating_min"]}}})
    if spec.get("rating_max") is not None:
        filters.append({"range": {"rating": {"lt": float(spec["rating_max"])}}})
    if spec.get("in_stock_only"):
        filters.append({"range": {"stock": {"gt": 0}}})

    # Index often omits published/approved — treat missing as visible, keep explicit 0 out.
    filters.append(
        {
            "bool": {
                "should": [
                    {"term": {"published": 1}},
                    {"bool": {"must_not": {"exists": {"field": "published"}}}},
                ],
                "minimum_should_match": 1,
            }
        }
    )
    filters.append(
        {
            "bool": {
                "should": [
                    {"term": {"approved": 1}},
                    {"bool": {"must_not": {"exists": {"field": "approved"}}}},
                ],
                "minimum_should_match": 1,
            }
        }
    )

    title = (spec.get("title_query") or "").strip()
    if spec.get("sku") and not title:
        pass
    elif title:
        must.append(_build_title_match_clause(title))

    if must or filters:
        query: dict = {"bool": {}}
        if must:
            query["bool"]["must"] = must
        if filters:
            query["bool"]["filter"] = filters
    else:
        query = {"match_all": {}}

    body: dict[str, Any] = {"size": min(size, MAX_PAGE_SIZE), "from": max(0, offset), "query": query}

    sort_key = spec.get("sort")
    if sort_key == "unit_price_asc":
        body["sort"] = [{"unit_price": "asc"}, {"_score": "desc"}]
    elif sort_key == "unit_price_desc":
        body["sort"] = [{"unit_price": "desc"}, {"_score": "desc"}]
    elif sort_key == "rating_desc":
        body["sort"] = [{"rating": {"order": "desc", "missing": "_last"}}, {"_score": "desc"}]
    elif sort_key == "purchase_price_asc":
        body["sort"] = [{"purchase_price": "asc"}, {"_score": "desc"}]
    elif sort_key == "purchase_price_desc":
        body["sort"] = [{"purchase_price": "desc"}, {"_score": "desc"}]

    return body


def opensearch_hits_to_product_cards(hits: list) -> list[dict]:
    from services import kb_service as _kb

    sysmsg = _kb.sysmsg
    cards = []
    seen: set[str] = set()
    for hit in hits or []:
        src = hit.get("_source") if isinstance(hit, dict) else {}
        if not isinstance(src, dict):
            continue
        pro_id = src.get("pro_id")
        slug = (src.get("slug") or "").strip()
        dedupe_key = str(pro_id or "").strip() or slug or ""
        if dedupe_key:
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
        name = src.get("name") or sysmsg("default_product_card_title")
        from services.welfog_api import format_customer_price_display

        price = format_customer_price_display(src, sysmsg("na_price"))
        purchase_price = src.get("purchase_price")
        shipping_cost = src.get("shipping_cost")
        thumb = (src.get("thumbnail_img") or "").lstrip("/")
        link = (
            f"https://welfog.com/products/{slug}"
            if slug
            else "https://welfog.com"
        )
        cards.append(
            {
                "name": name,
                "price": price,
                "purchase_price": purchase_price,
                "shipping_cost": shipping_cost,
                "image": f"{IMAGE_BASE_URL}{thumb}" if thumb else "",
                "link": link,
                "pro_id": pro_id,
                "sku": src.get("sku") or "",
                "size": src.get("size") or "",
                "rating": src.get("rating"),
                "color_name": src.get("color_name") or src.get("color") or "",
                "color": src.get("color_name") or src.get("color") or "",
                "brand": src.get("brand") or "",
                "score": hit.get("_score") or 0,
            }
        )
    return cards


def _opensearch_request(body: dict, *, timeout_sec: float = 8) -> tuple[list, int]:
    cfg = _env_opensearch_config()
    if not cfg:
        return [], 0
    try:
        _OS_REQUEST_TLS.count = int(getattr(_OS_REQUEST_TLS, "count", 0) or 0) + 1
        from utils.reasoning_log import log_reasoning

        log_reasoning(f"OpenSearch HTTP request #{_OS_REQUEST_TLS.count}")
    except Exception:
        pass
    try:
        limit = max(2.5, min(6.0, float(timeout_sec or 5)))
        # Hard wall-clock — socket timeout alone can hang 30–60s on stuck TLS.
        import concurrent.futures

        def _do_post():
            return requests.post(
                cfg["url"], json=body, auth=cfg["auth"], timeout=(2.0, limit)
            )

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_do_post)
            try:
                res = fut.result(timeout=limit + 0.6)
            except concurrent.futures.TimeoutError:
                try:
                    from utils.reasoning_log import log_reasoning

                    log_reasoning(f"OpenSearch hard timeout ({limit}s) — empty hits.")
                except Exception:
                    pass
                return [], 0
        finally:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)
        if res.status_code != 200:
            return [], 0
        data = res.json()
        hits_wrap = data.get("hits") or {}
        hits = hits_wrap.get("hits") or []
        total_obj = hits_wrap.get("total") or {}
        total = int(total_obj.get("value", len(hits)) if isinstance(total_obj, dict) else len(hits))
        return hits, total
    except Exception:
        return [], 0


def _search_opensearch_page(
    spec: dict[str, Any],
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
    post_filter_mode: str = "strict",
    skip_price_sync: bool = False,
) -> tuple[list[dict], int, bool]:
    page = max(1, int(page))
    page_size = min(max(1, int(page_size)), MAX_PAGE_SIZE)
    offset = (page - 1) * page_size
    body = _build_opensearch_body(spec, size=page_size, offset=offset)
    os_timeout = 4.0 if spec.get("_category_only_browse") else 5.0
    hits, os_total = _opensearch_request(body, timeout_sec=os_timeout)
    cards = opensearch_hits_to_product_cards(hits)
    cards = apply_catalog_post_filters(cards, spec, post_filter_mode=post_filter_mode)
    if cards and not skip_price_sync and not spec.get("_ai_single_pass"):
        has_price_filter = (
            spec.get("purchase_price_max") is not None
            or spec.get("purchase_price_min") is not None
        )
        # Ordinary browse must return indexed cards immediately. The live REST
        # price refresh can fan out into several network calls and added 8–30s
        # even when the customer never asked for a budget. Only block on exact
        # landed-price enrichment when a price constraint makes it necessary.
        if has_price_filter:
            from services.welfog_api import sync_product_cards_prices_from_rest_api

            q = (spec.get("title_query") or "").strip()
            if q:
                cards = sync_product_cards_prices_from_rest_api(
                    cards,
                    q,
                    color=spec.get("color"),
                    category_id=spec.get("category_id"),
                )
            cards = filter_products_by_purchase_price(cards, spec)
        if spec.get("rating_min") is not None or spec.get("rating_max") is not None:
            cards = filter_products_by_rating_range(cards, spec)
    raw_has_more = (offset + len(hits)) < max(int(os_total or 0), offset + len(hits))
    has_more = raw_has_more and len(cards) >= page_size
    return cards, len(cards), has_more


def product_search_show_view_more(
    products: list,
    has_more: bool,
    *,
    page_size: int = PAGE_SIZE,
    min_count: int = MIN_PRODUCTS_FOR_VIEW_MORE,
) -> bool:
    """Only offer pagination when the first page is full enough to matter."""
    if not products or not has_more:
        return False
    if len(products) < min_count:
        return False
    if len(products) < page_size:
        return False
    return True


def search_opensearch_products(
    spec: dict[str, Any],
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
    post_filter_mode: str = "strict",
) -> tuple[list[dict], int, bool]:
    """Returns (product_cards, visible_count, has_more)."""
    return _search_opensearch_page(
        spec, page=page, page_size=page_size, post_filter_mode=post_filter_mode
    )


def _is_category_only_browse_spec(spec: dict[str, Any]) -> bool:
    """Category department browse — no product title / SKU / pro_id filter."""
    if not spec or not spec.get("category_id"):
        return False
    if (spec.get("title_query") or "").strip():
        return False
    if spec.get("pro_id") or spec.get("sku"):
        return False
    return True


def search_opensearch_catalog(
    spec: dict[str, Any],
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
) -> tuple[list[dict], dict[str, Any], int, bool]:
    """
    OpenSearch-only catalog search with tiered relax (never returns unrelated REST junk).
    Filters: title_query, color, brand/aliases, size, unit_price, SKU, pro_id, category_id, sort.
    """
    from utils.reasoning_log import log_reasoning

    spec = sanitize_product_search_spec(dict(spec or {}))
    if not is_opensearch_configured():
        log_reasoning("OpenSearch not configured (OPENSEARCH_URL missing).")
        return [], spec, 0, False

    log_reasoning(
        f"OpenSearch catalog: title={spec.get('title_query')!r} color={spec.get('color')!r} "
        f"size={spec.get('size')!r} brand={spec.get('brand')!r} sku={spec.get('sku')!r} pro_id={spec.get('pro_id')} "
        f"category_id={spec.get('category_id')!r} "
        f"price_max={spec.get('purchase_price_max')!r} price_min={spec.get('purchase_price_min')!r} "
        f"rating_min={spec.get('rating_min')!r}"
    )

    if spec.get("pro_id"):
        pro_spec = dict(spec)
        pro_spec["title_query"] = ""
        pro_spec.pop("brand", None)
        pro_spec.pop("brand_aliases", None)
        pro_spec["title_match_strict"] = False
        cards, n, more = _search_opensearch_page(
            pro_spec, page=page, page_size=page_size, post_filter_mode="light"
        )
        if cards:
            log_reasoning(f"OpenSearch pro_id={spec.get('pro_id')}: {len(cards)} product(s).")
            try:
                from services.product_intent_parser import log_opensearch_query

                log_opensearch_query(pro_spec, len(cards))
            except ImportError:
                pass
            return cards, pro_spec, len(cards), more

    if spec.get("category_id") and not (spec.get("title_query") or "").strip():
        # Live department feed first when the REST categories= filter is trustworthy.
        # OpenSearch often lacks top-level nav ids; name-expansion is the fallback.
        try:
            from services.welfog_api import _rest_category_filter_trustworthy, fetch_products_by_category_browse

            rest_rows, eff = fetch_products_by_category_browse(
                spec["category_id"], page=page, color=spec.get("color")
            )
            if rest_rows and _rest_category_filter_trustworthy(eff or spec["category_id"], rest_rows):
                log_reasoning(
                    f"Category browse REST first category_id={eff or spec['category_id']}: "
                    f"{len(rest_rows)} product(s)."
                )
                out_spec = dict(spec)
                out_spec["category_id"] = eff or spec["category_id"]
                out_spec["_category_only_browse"] = True
                pf = "light"
                rest_rows = apply_catalog_post_filters(rest_rows, out_spec, post_filter_mode=pf)
                if rest_rows:
                    has_more = len(rest_rows) >= page_size
                    return rest_rows[:page_size], out_spec, len(rest_rows), has_more
        except Exception:
            pass
        pf = "light"
        cards, n, more = _search_opensearch_page(
            spec, page=page, page_size=page_size, post_filter_mode=pf
        )
        if cards:
            log_reasoning(
                f"OpenSearch category_id={spec.get('category_id')}: {len(cards)} product(s)."
            )
            return cards, spec, len(cards), more
        try:
            from services.welfog_api import _rest_category_filter_trustworthy, fetch_products_from_api

            rest_rows = fetch_products_from_api(
                "", category_id=int(spec["category_id"]), page=page
            )
            if rest_rows and _rest_category_filter_trustworthy(spec["category_id"], rest_rows):
                log_reasoning(
                    f"Category browse catalog fallback category_id={spec['category_id']} "
                    f"({len(rest_rows)} items — OS index empty for this category)."
                )
                has_more = len(rest_rows) >= page_size
                return rest_rows, spec, len(rest_rows), has_more
            if rest_rows:
                log_reasoning(
                    f"Category browse REST ignored category_id={spec['category_id']} "
                    f"(default pool returned — skipped)."
                )
        except Exception:
            pass

    if spec.get("sku"):
        sku_spec = dict(spec)
        sku_spec["title_query"] = ""
        sku_spec.pop("brand", None)
        sku_spec.pop("brand_aliases", None)
        sku_spec.pop("color", None)
        sku_spec.pop("mandatory_match_tokens", None)
        sku_spec["title_match_strict"] = False
        cards, n, more = _search_opensearch_page(
            sku_spec,
            page=page,
            page_size=page_size,
            post_filter_mode="light",
            skip_price_sync=True,
        )
        if cards:
            log_reasoning(f"OpenSearch SKU={spec.get('sku')!r}: {len(cards)} product(s).")
            try:
                from services.product_intent_parser import log_opensearch_query

                log_opensearch_query(sku_spec, len(cards))
            except ImportError:
                pass
            return cards, sku_spec, len(cards), more
        log_reasoning(f"OpenSearch SKU={spec.get('sku')!r}: 0 hits (SKU-only lookup).")
        try:
            from services.product_intent_parser import log_opensearch_query

            log_opensearch_query(sku_spec, 0)
        except ImportError:
            pass
        return [], sku_spec, 0, False

    if spec.get("_ai_single_pass"):
        pf = _post_filter_mode_for_spec(spec)
        # OpenSearch index purchase_price lags the live catalog, so a tight budget
        # (e.g. "baniyan under 400") would drop items that are actually in budget
        # once landed (purchase + shipping) prices are refreshed. Fetch candidates
        # WITHOUT the stale index price range and let filter_products_by_purchase_price
        # enforce the real budget on refreshed prices. Widen the page so enough
        # candidates survive the post-filter.
        _has_price_f = (
            spec.get("purchase_price_max") is not None
            or spec.get("purchase_price_min") is not None
        )
        page_size_eff = page_size
        if _has_price_f and not spec.get("_os_price_filter_disabled"):
            spec = dict(spec)
            spec["_os_price_filter_disabled"] = True
            page_size_eff = min(page_size * 3, MAX_PAGE_SIZE)
        cards, n, more = _search_opensearch_page(
            spec,
            page=page,
            page_size=page_size_eff,
            post_filter_mode=pf,
            skip_price_sync=True,
        )
        if not cards and (spec.get("brand") or spec.get("brand_aliases")):
            relaxed = dict(spec)
            brand = (relaxed.pop("brand", None) or "").strip()
            relaxed.pop("brand_aliases", None)
            relaxed.pop("brand_name_match_only", None)
            relaxed["title_match_strict"] = False
            relaxed.pop("mandatory_match_tokens", None)
            tq = (relaxed.get("title_query") or "").strip()
            if brand and brand.lower() not in tq.lower():
                relaxed["title_query"] = f"{tq} {brand}".strip() if tq else brand
            if spec.get("_requested_brand"):
                relaxed["_requested_brand"] = spec["_requested_brand"]
                relaxed["_requested_brand_aliases"] = spec.get("_requested_brand_aliases")
            cards, n, more = _search_opensearch_page(
                relaxed,
                page=page,
                page_size=page_size,
                post_filter_mode="light",
                skip_price_sync=True,
            )
            if cards:
                spec = relaxed
                log_reasoning(
                    f"OpenSearch AI single-pass brand-relax: {n} product(s) "
                    f"(title search without brand field filter)."
                )
        # Multi-token AI titles / Brain gloss ("chunri" + related "scarf") that
        # catalog names omit — retry each token alone before slow REST fallback.
        if not cards:
            tq = (spec.get("title_query") or "").strip().lower()
            ptype = (spec.get("product_type") or "").strip().lower()
            mandatory = [
                str(m).strip().lower()
                for m in (spec.get("mandatory_match_tokens") or [])
                if m and str(m).strip()
            ]
            toks = [
                t
                for t in re.findall(r"[a-z0-9]{3,}", tq)
                if t not in ("the", "and", "for", "with")
                and t not in _GENERIC_TITLE_MODIFIERS
            ]
            related_blob = " ".join(
                str(x)
                for x in (
                    spec.get("related_search_terms"),
                    spec.get("_related_search_terms"),
                )
                if x
            )
            for t in re.findall(r"[a-z0-9]{3,}", related_blob.lower()):
                if (
                    t not in toks
                    and t not in ("the", "and", "for", "with")
                    and t not in _GENERIC_TITLE_MODIFIERS
                ):
                    toks.append(t)
            if mandatory:
                recovery_pool = mandatory
            elif ptype:
                recovery_pool = [ptype]
            else:
                recovery_pool = toks
            seen_tok: set[str] = set()
            # Prefer longer distinctive tokens first (sidhu/moosewala before frame).
            # Never soft-recover on short generic tails when a longer subject token exists.
            max_tok_len = max((len(t) for t in recovery_pool[:4]), default=0)
            ordered = sorted(recovery_pool[:4], key=lambda t: (-len(t), t))
            for tok in ordered:
                if tok in seen_tok or tok == tq or tok in _GENERIC_TITLE_MODIFIERS:
                    continue
                if len(recovery_pool) >= 2 and max_tok_len >= 6 and len(tok) < max_tok_len - 2:
                    continue
                seen_tok.add(tok)
                soft = dict(spec)
                soft["title_query"] = tok
                soft["title_match_strict"] = False
                if mandatory or ptype:
                    soft["mandatory_match_tokens"] = mandatory or ([ptype] if ptype else [])
                    if ptype:
                        soft["product_type"] = ptype
                else:
                    soft.pop("mandatory_match_tokens", None)
                soft_pf = (
                    "strict"
                    if soft.get("mandatory_match_tokens")
                    else "light"
                )
                soft_cards, soft_n, soft_more = _search_opensearch_page(
                    soft,
                    page=page,
                    page_size=page_size,
                    post_filter_mode=soft_pf,
                    skip_price_sync=True,
                )
                if soft_cards:
                    cards, n, more = soft_cards, soft_n, soft_more
                    spec = soft
                    log_reasoning(
                        f"OpenSearch AI single-pass token-soft {tok!r}: "
                        f"{n} product(s) (multi-token miss recovery)."
                    )
                    break
                if len(seen_tok) >= 2:
                    break
        if cards:
            log_reasoning(
                f"OpenSearch AI single-pass: {n} product(s) "
                f"(no tiered relax — filters from AI only)."
            )
        else:
            log_reasoning(
                f"OpenSearch AI single-pass: {n} product(s) "
                f"(no tiered relax — filters from AI only)."
            )
        if cards:
            req_brand = (spec.get("_requested_brand") or spec.get("brand") or "").strip()
            req_aliases = spec.get("_requested_brand_aliases") or spec.get("brand_aliases")
            if req_brand or req_aliases:
                strict_brand = bool(spec.get("title_match_strict"))
                if not strict_brand and req_brand:
                    try:
                        from services.opensearch_products import _phone_brand_vocab

                        if req_brand.lower() in _phone_brand_vocab():
                            strict_brand = True
                    except ImportError:
                        pass
                cards = filter_products_by_requested_brand(
                    cards,
                    req_brand or None,
                    brand_aliases=req_aliases if isinstance(req_aliases, list) else None,
                    strict_title=strict_brand,
                )
            # _search_opensearch_page already ran apply_catalog_post_filters,
            # including one landed-price refresh and rating enforcement. Repeating
            # them here caused a second bounded REST call on every budget search.
            spec["_post_filters_applied"] = True
            n = len(cards)
        try:
            from services.product_intent_parser import log_opensearch_query

            log_opensearch_query(spec, n)
        except ImportError:
            pass
        return cards, spec, n, more

    pf = _post_filter_mode_for_spec(spec)
    cards, n, more = _search_opensearch_page(
        spec, page=page, page_size=page_size, post_filter_mode=pf
    )
    if cards:
        log_reasoning(f"OpenSearch tier-1: {n} product(s) matched colour/keywords.")
        cards = filter_products_by_purchase_price(cards, spec)
        cards = filter_products_by_rating_range(cards, spec)
        if cards:
            try:
                from services.product_intent_parser import log_opensearch_query

                log_opensearch_query(spec, len(cards))
            except ImportError:
                pass
            return cards, spec, len(cards), more

    if spec.get("strict_no_relax"):
        log_reasoning("OpenSearch strict mode — zero hits, no filter relaxation.")
        try:
            from services.product_intent_parser import log_opensearch_query

            log_opensearch_query(spec, 0)
        except ImportError:
            pass
        return [], spec, 0, False

    has_price_f = (
        spec.get("purchase_price_max") is not None or spec.get("purchase_price_min") is not None
    )
    if not cards and has_price_f and (spec.get("title_query") or "").strip():
        wide_price = dict(spec)
        wide_price["_os_price_filter_disabled"] = True
        wide_price["title_match_strict"] = False
        wide_price.pop("mandatory_match_tokens", None)
        cards, n, more = _search_opensearch_page(
            wide_price,
            page=page,
            page_size=min(page_size * 3, MAX_PAGE_SIZE),
            post_filter_mode="os_filters_only",
        )
        if cards:
            log_reasoning(
                "OpenSearch tier-1p: title search without OS price filter, post-filter budget."
            )
            cards = filter_products_by_purchase_price(cards, spec)
            cards = filter_products_by_rating_range(cards, spec)
            if cards:
                return cards, wide_price, len(cards), more

    if spec.get("sku"):
        sku_only = dict(spec)
        sku_only["title_query"] = ""
        sku_only.pop("mandatory_match_tokens", None)
        sku_only["title_match_strict"] = False
        cards, n, more = _search_opensearch_page(
            sku_only, page=page, page_size=page_size, post_filter_mode="light"
        )
        if cards:
            cards = filter_products_by_requested_sku(cards, str(spec["sku"]))
            if cards:
                log_reasoning(f"OpenSearch tier-sku: {len(cards)} hit(s) for SKU {spec['sku']!r}.")
                return cards, sku_only, len(cards), more

    if spec.get("brand"):
        boosted = dict(spec)
        boosted["title_query"] = _catalog_title_for_search(spec) or spec.get("title_query")
        boosted.pop("mandatory_match_tokens", None)
        boosted["title_match_strict"] = False
        cards, n, more = _search_opensearch_page(
            boosted, page=page, page_size=page_size, post_filter_mode="os_filters_only"
        )
        if cards:
            cards = filter_products_strict_brand_match(
                cards,
                spec.get("brand"),
                spec.get("brand_aliases"),
            )
        if cards:
            log_reasoning(
                f"OpenSearch tier-1c: title={boosted.get('title_query')!r} + brand filter."
            )
            cards = filter_products_by_purchase_price(cards, spec)
            cards = filter_products_by_rating_range(cards, spec)
            if cards:
                return cards, boosted, len(cards), more

    if spec.get("color") and (spec.get("title_query") or "").strip():
        title_pf = dict(spec)
        title_pf.pop("color", None)
        title_pf.pop("mandatory_match_tokens", None)
        title_pf["title_match_strict"] = False
        cards, n, more = _search_opensearch_page(
            title_pf, page=page, page_size=page_size, post_filter_mode="os_filters_only"
        )
        if cards:
            cards = filter_products_by_requested_color(cards, spec.get("color"))
            need = _title_match_tokens(spec.get("title_query") or "")
            if need:
                cards = filter_products_by_ai_mandatory_tokens(
                    cards, need, skip_color_tokens=True
                )
            mats = spec.get("material_tokens") or []
            if mats:
                cards = filter_products_by_material_tokens(cards, mats)
        if cards:
            log_reasoning(
                f"OpenSearch tier-1b: title={spec.get('title_query')!r} + color_name={spec.get('color')!r}."
            )
            return cards, title_pf, len(cards), more

    relaxed = dict(spec)
    relaxed.pop("mandatory_match_tokens", None)
    relaxed["title_match_strict"] = False
    cards, n, more = _search_opensearch_page(
        relaxed, page=page, page_size=page_size, post_filter_mode="os_filters_only"
    )
    if cards:
        log_reasoning("OpenSearch tier-2: OS filters + brand/colour post-filter.")
        return cards, relaxed, n, more

    if spec.get("color") and (spec.get("title_query") or "").strip():
        color_only = dict(spec)
        color_only["title_query"] = ""
        color_only.pop("mandatory_match_tokens", None)
        cards, n, more = _search_opensearch_page(
            color_only, page=page, page_size=page_size, post_filter_mode="os_filters_only"
        )
        if cards:
            need = _title_match_tokens(spec.get("title_query") or "")
            if need:
                cards = [
                    p
                    for p in cards
                    if any(_product_type_token_in_name(t, (p.get("name") or "").lower()) for t in need)
                ]
            if cards:
                log_reasoning(
                    f"OpenSearch tier-2b: colour={spec.get('color')!r} then product-type filter."
                )
                return cards, color_only, len(cards), more

    brand_key = (spec.get("brand") or "").strip().lower()
    if brand_key and brand_key not in _COLOR_NAME_TOKENS:
        wider = dict(relaxed)
        wider.pop("brand", None)
        wider.pop("brand_aliases", None)
        cards, n, more = _search_opensearch_page(
            wider, page=page, page_size=page_size, post_filter_mode="os_filters_only"
        )
        if cards:
            log_reasoning("OpenSearch tier-3: title + colour (phone model relaxed).")
            return cards, wider, n, more

    return [], spec, 0, False


def cheapest_alternative_hint(spec: dict[str, Any]) -> str:
    """When price filter has zero hits, find lowest-priced item in same category keywords."""
    if (spec.get("title_query") or "").strip():
        return ""
    relaxed = dict(spec)
    relaxed.pop("unit_price_max", None)
    relaxed.pop("unit_price_min", None)
    products, _total, _ = search_opensearch_products(relaxed, page=1, page_size=5)
    if not products:
        return ""
    cheapest = min(products, key=lambda p: float(p.get("price") or 999999))
    price = cheapest.get("price")
    name = (cheapest.get("name") or "a product")[:40]
    if price:
        return f" Lowest available match: <b>{name}</b> at Rs {price}."
    return ""


def _category_browse_rest_fallback(
    spec: dict[str, Any],
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
    ctx=None,
) -> tuple[list[dict], dict[str, Any], int, bool]:
    """REST fallback when OpenSearch is empty and browse is category-only (no title filter)."""
    cat_id = spec.get("category_id")
    if not cat_id:
        return [], spec, 0, False
    if (spec.get("title_query") or "").strip():
        return [], spec, 0, False

    from services.welfog_api import fetch_products_by_category_browse
    from utils.reasoning_log import log_reasoning

    log_reasoning(f"Catalog: category-only REST browse categories={cat_id}")
    rest, effective_cid = fetch_products_by_category_browse(
        cat_id,
        ctx=ctx,
        page=page,
        color=spec.get("color"),
    )
    if not rest:
        return [], spec, 0, False

    out_spec = dict(spec)
    out_spec["category_id"] = effective_cid
    try:
        from services.welfog_api import strip_category_browse_conflicts_from_spec

        out_spec = strip_category_browse_conflicts_from_spec(out_spec, ctx=ctx)
    except ImportError:
        pass
    pf = _post_filter_mode_for_spec(out_spec)
    rest = apply_catalog_post_filters(rest, out_spec, post_filter_mode=pf)
    if not rest:
        return [], out_spec, 0, False
    start = (page - 1) * page_size
    slice_rest = rest[start : start + page_size]
    has_more_rest = len(rest) > start + page_size
    return slice_rest, out_spec, len(rest), has_more_rest


def _filter_only_rest_fallback(
    spec: dict[str, Any],
    *,
    page: int = 1,
    page_size: int = PAGE_SIZE,
) -> tuple[list[dict], dict[str, Any], int, bool]:
    """REST pool + post-filter when user asked price/rating only (no product title)."""
    if (spec.get("title_query") or "").strip() or spec.get("pro_id") or spec.get("sku"):
        return [], spec, 0, False
    has_filter = any(
        spec.get(k) is not None
        for k in ("purchase_price_max", "purchase_price_min", "rating_min", "rating_max", "color", "size")
    )
    if not has_filter:
        return [], spec, 0, False
    from services.welfog_api import fetch_catalog_browse_products
    from utils.reasoning_log import log_reasoning

    log_reasoning("Catalog: filter-only REST browse + post-filter.")
    pool = fetch_catalog_browse_products(max_pages=6)
    pf = _post_filter_mode_for_spec(spec)
    rest = apply_catalog_post_filters(pool, spec, post_filter_mode=pf)
    if not rest:
        return [], spec, 0, False
    start = (page - 1) * page_size
    slice_rest = rest[start : start + page_size]
    has_more_rest = len(rest) > start + page_size
    return slice_rest, spec, len(rest), has_more_rest


def search_products_combined(
    text: str,
    *,
    original_msg: str = "",
    msg_en: str = "",
    category_id=None,
    color=None,
    title_hint: str = "",
    pro_id: Optional[int] = None,
    page: int = 1,
    page_size: int = PAGE_SIZE,
    ctx=None,
) -> tuple[list[dict], dict, int, bool]:
    """
    OpenSearch first (filters/sort/catalog), REST API fallback.
    Returns (products, spec, total, has_more).
    """
    if original_msg or msg_en:
        spec = build_product_search_spec(
            original_msg or text,
            msg_en or text,
            category_id=category_id,
            color=color,
            title_hint=title_hint,
            pro_id=pro_id,
        )
    else:
        spec = parse_product_filters_from_text(
            text,
            category_id=category_id,
            color=color,
            title_hint=title_hint,
            pro_id=pro_id,
        )
    products, total, has_more = search_opensearch_products(
        spec, page=page, page_size=page_size, post_filter_mode=_post_filter_mode_for_spec(spec)
    )
    if products:
        return products, spec, total, has_more

    if not (title_hint or spec.get("title_query") or "").strip():
        filt_rest = _filter_only_rest_fallback(spec, page=page, page_size=page_size)
        if filt_rest[0]:
            return filt_rest
        cat_rest = _category_browse_rest_fallback(
            spec, page=page, page_size=page_size, ctx=ctx
        )
        if cat_rest[0]:
            return cat_rest

    q = (title_hint or spec.get("title_query") or "").strip()
    if not q:
        q = _extract_product_keywords((text or "").lower())
    if not q and (original_msg or msg_en):
        q = _extract_product_keywords(f"{original_msg} {msg_en}".lower())
    if not q:
        filt_rest = _filter_only_rest_fallback(spec, page=page, page_size=page_size)
        if filt_rest[0]:
            return filt_rest
        cat_rest = _category_browse_rest_fallback(
            spec, page=page, page_size=page_size, ctx=ctx
        )
        if cat_rest[0]:
            return cat_rest
        return [], spec, 0, False

    from utils.reasoning_log import log_reasoning

    log_reasoning(f"OpenSearch empty — live REST catalog: name={q!r} color={spec.get('color')!r}")
    cat = category_id or spec.get("category_id")
    if cat:
        from services.welfog_api import resolve_search_category_id

        cat = resolve_search_category_id(str(cat), ctx)
    rest = fetch_products_from_api(
        q,
        category_id=cat,
        color=color or spec.get("color"),
        page=page,
    )
    if rest:
        pf = _post_filter_mode_for_spec(spec)
        rest = apply_catalog_post_filters(rest, spec, post_filter_mode=pf)
        if rest:
            start = (page - 1) * page_size
            slice_rest = rest[start : start + page_size]
            has_more_rest = len(rest) > start + page_size
            return slice_rest, spec, len(rest), has_more_rest
    return [], spec, 0, False


def catalog_search_live(
    spec: dict[str, Any],
    *,
    original_msg: str = "",
    msg_en: str = "",
    page: int = 1,
    page_size: int = PAGE_SIZE,
    ctx=None,
) -> tuple[list[dict], dict[str, Any], int, bool]:
    """
    Full Welfog catalog: OpenSearch index first (fast filters), then live REST API for
    fashion/grocery/electronics when the index slice is empty or incomplete.
    """
    from utils.reasoning_log import log_reasoning

    spec = sanitize_product_search_spec(dict(spec or {}))
    import time as _time

    try:
        reset_opensearch_request_count()
    except Exception:
        pass
    _os_t0 = _time.perf_counter()
    products, spec, total, has_more = search_opensearch_catalog(
        spec, page=page, page_size=page_size
    )
    try:
        from services.chat_flow_telemetry import record_api_time, record_phase

        _os_el = _time.perf_counter() - _os_t0
        record_api_time(_os_el)
        record_phase("opensearch", _os_el * 1000.0)
        log_reasoning(
            f"OpenSearch catalog done: {get_opensearch_request_count()} HTTP request(s), "
            f"{len(products)} product(s)."
        )
    except ImportError:
        pass
    if products:
        return products, spec, total, has_more

    single_pass_miss = bool(spec.get("_ai_single_pass"))
    if single_pass_miss:
        # The authoritative agent contract is one entity pass + one OpenSearch
        # request. A zero-hit REST fallback added 12–20s and made misses slower
        # than successful searches. Price enrichment remains bounded separately;
        # search misses return immediately and can use AI recommendations.
        log_reasoning(
            "Catalog AI single-pass: 0 OS hits — return immediately "
            "(skip slow REST search fallback)."
        )
        return [], spec, 0, False

    if spec.get("strict_no_relax") and spec.get("pro_id"):
        return [], spec, 0, False

    if not (spec.get("title_query") or "").strip():
        filt_rest = _filter_only_rest_fallback(spec, page=page, page_size=page_size)
        if filt_rest[0]:
            return filt_rest
        cat_rest = _category_browse_rest_fallback(
            spec, page=page, page_size=page_size, ctx=ctx
        )
        if cat_rest[0]:
            return cat_rest

    text = f"{original_msg} {msg_en}".strip() or (spec.get("title_query") or "")
    q = _catalog_title_for_search(spec) or (spec.get("title_query") or "").strip()
    if not q:
        q = _extract_product_keywords(text.lower())
    if not q:
        filt_rest = _filter_only_rest_fallback(spec, page=page, page_size=page_size)
        if filt_rest[0]:
            return filt_rest
        cat_rest = _category_browse_rest_fallback(
            spec, page=page, page_size=page_size, ctx=ctx
        )
        if cat_rest[0]:
            return cat_rest
        return [], spec, 0, False

    log_reasoning(f"Catalog: OpenSearch 0 hits — live REST API name={q!r}")
    _rest_t0 = _time.perf_counter()
    cat_id = spec.get("category_id")
    if cat_id:
        from services.welfog_api import resolve_search_category_id

        cat_id = resolve_search_category_id(str(cat_id), ctx)
    # Try typo-soft + compound-space name variants (REST is often exact/prefix).
    # Locked AI single-pass: at most 2 short attempts (primary + first token) —
    # never 4×12s sequential hangs that blow chat deadline on Hinglish misses.
    rest_queries: list[str] = []
    if single_pass_miss:
        rest_queries.append(q)
        first_tok = ""
        for tok in re.findall(r"[a-z0-9]{3,}", q.lower()):
            first_tok = tok
            break
        if first_tok and first_tok != q.lower():
            rest_queries.append(first_tok)
    else:
        for cand in _expand_title_query_variants(q) or [q]:
            if cand and cand not in rest_queries:
                rest_queries.append(cand)
        if q and q not in rest_queries:
            rest_queries.insert(0, q)
    rest: list = []
    # Single-pass: OpenSearch already tried title variants. One short REST is enough
    # for index-lag catch-up — a 2nd 4s miss doubled empty-catalog latency.
    rest_cap = 1 if single_pass_miss else 4
    for rq in rest_queries[:rest_cap]:
        if single_pass_miss:
            # Short timeout — empty catalog should fail fast, not stall 60s.
            # Never fall back to the default 12s fetch_products_from_api here
            # (Timeout → except → 12s retry was doubling hangs).
            try:
                from services.welfog_api import _catalog_search_get

                params = {"page": page or 1, "name": rq}
                if cat_id:
                    rows = _catalog_search_get(
                        {**params, "categories": cat_id}, timeout=4
                    ) or _catalog_search_get(params, timeout=4)
                else:
                    rows = _catalog_search_get(params, timeout=4)
                rest = rows if isinstance(rows, list) else []
            except Exception:
                rest = []
        else:
            rest = fetch_products_from_api(
                rq,
                category_id=cat_id,
                color=spec.get("color"),
                page=page,
            )
        if rest:
            if rq != q:
                log_reasoning(f"Catalog: REST hit via typo-soft name={rq!r} (was {q!r})")
            break
    try:
        from services.chat_flow_telemetry import record_api_time, record_phase

        _rest_el = _time.perf_counter() - _rest_t0
        record_api_time(_rest_el)
        record_phase("catalog_rest", _rest_el * 1000.0)
    except ImportError:
        pass
    brand_key = (spec.get("brand") or "").strip()
    # Locked single-pass already tried title REST (2×4s). Brand page crawl via
    # fetch_products_from_api (12s×3) was dominating empty Nike/Adidas turns.
    if not rest and brand_key and not single_pass_miss:
        log_reasoning(f"Catalog: REST brand browse name={brand_key!r}")
        for pg in (1, 2, 3):
            rest.extend(
                fetch_products_from_api(
                    brand_key,
                    category_id=spec.get("category_id"),
                    color=spec.get("color"),
                    page=pg,
                )
            )
            if len(rest) >= 40:
                break
        seen_rest: set = set()
        uniq_rest = []
        for it in rest:
            k = it.get("slug") or it.get("id")
            if k is not None and k in seen_rest:
                continue
            if k is not None:
                seen_rest.add(k)
            uniq_rest.append(it)
        rest = uniq_rest
    elif not rest and brand_key and single_pass_miss:
        log_reasoning(
            f"Catalog: skip REST brand browse on single-pass miss (brand={brand_key!r})."
        )
    if not rest:
        return [], spec, 0, False
    pf = _post_filter_mode_for_spec(spec)
    rest = apply_catalog_post_filters(rest, spec, post_filter_mode=pf)
    if brand_key:
        rest = filter_products_strict_brand_match(
            rest, brand_key, spec.get("brand_aliases")
        )
    if not rest:
        return [], spec, 0, False
    start = (page - 1) * page_size
    slice_rest = rest[start : start + page_size]
    has_more_rest = len(rest) > start + page_size
    return slice_rest, spec, len(rest), has_more_rest


def build_product_cards_html(products: list[dict], sysmsg) -> str:
    html = []
    for p in products:
        html.append("<div class='wf-product-card'>")
        if p.get("image"):
            html.append(
                f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; "
                f"overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; "
                f"border: 1px solid #f0f0f0;'><img src='{p['image']}' alt='{p['name']}' "
                f"style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
            )
        else:
            html.append(
                f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; "
                f"margin-bottom: 10px; display: flex; align-items: center; justify-content: center; "
                f"color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
            )
        name = p.get("name") or ""
        name_short = name[:38] + "..." if len(name) > 38 else name
        html.append(
            f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; "
            f"overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; "
            f"-webkit-box-orient: vertical;'>{name_short}</div>"
        )
        html.append(
            f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; "
            f"margin-top: auto;'>₹{p.get('price', '')}</div>"
        )
        if p.get("link"):
            html.append(
                f"<a href='{p['link']}'>"
                f"{sysmsg('view_product')}</a>"
            )
        html.append("</div>")
    return "".join(html)


def _build_product_view_more_tail(
    sysmsg,
    *,
    has_more: bool,
    products: list,
    next_page: int = 2,
    browse_more_url: str = "",
) -> str:
    if not product_search_show_view_more(products, has_more):
        return ""
    label = sysmsg("products_view_more") or "View more products"
    if browse_more_url:
        safe_url = html_escape(browse_more_url, quote=True)
        return (
            "<div class='wf-product-tail' style='margin-top:12px;text-align:center;'>"
            f"<a href='{safe_url}'  "
            f"class='wf-ph-more wf-product-more wf-product-more-link'>"
            f"<span class='wf-ph-more__label'>{label}</span></a></div>"
        )
    return (
        "<div class='wf-product-tail' style='margin-top:12px;text-align:center;'>"
        f"<button type='button' class='wf-ph-more wf-product-more' data-next-product-page=\"{next_page}\">"
        f"<span class='wf-ph-more__label'>{label}</span></button></div>"
    )


def build_product_rail_with_pagination(
    products: list[dict],
    sysmsg,
    *,
    has_more: bool = False,
    next_page: int = 2,
    browse_more_url: str = "",
) -> str:
    html = "<div class='wf-product-root'><div class='wf-product-rail'>"
    html += build_product_cards_html(products, sysmsg)
    html += "</div>"
    html += _build_product_view_more_tail(
        sysmsg,
        has_more=has_more,
        products=products,
        next_page=next_page,
        browse_more_url=browse_more_url,
    )
    html += "</div>"
    return html


def format_product_search_append_payload(user_ctx_data: dict, page: int) -> dict:
    spec = (user_ctx_data or {}).get("last_os_spec") or {}
    if not spec:
        return {"cards_html": "", "tail_html": ""}
    products, _total, has_more = search_opensearch_products(spec, page=page, page_size=PAGE_SIZE)
    if not products:
        return {"cards_html": "", "tail_html": ""}
    from services import kb_service as _kb
    from services.welfog_api import build_welfog_product_browse_url

    cards = build_product_cards_html(products, _kb.sysmsg)
    browse_url = (user_ctx_data or {}).get("product_browse_url") or build_welfog_product_browse_url(spec)
    tail = _build_product_view_more_tail(
        _kb.sysmsg,
        has_more=has_more,
        products=products,
        next_page=page + 1,
        browse_more_url=browse_url,
    )
    return {"cards_html": cards, "tail_html": tail}

