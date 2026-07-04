"""
LLM-powered product intent extraction for any Indian language / Hinglish / English.
Turns roundabout shopping questions into clean OpenSearch search terms + filters.
"""
import json
import os
import re
from typing import Any, Optional

from services.llm_providers import llm_json_chat_completion
from services.translation_service import language_reply_instruction
from utils.reasoning_log import log_reasoning

_GRAIN_IN_MESSAGE = (
    (re.compile(r"\b(gehu|gehun|gehoon|गेहू|गेहूं|wheat|atta|aata)\b", re.I), "wheat"),
    (re.compile(r"\b(chawal|chaawal|chaal|chaval|rice|chaval)\b", re.I), "rice"),
)


_HINDI_GRAIN_DROP = frozenset(
    {"gehu", "gehun", "chawal", "chaawal", "chaal", "chaval", "dikhana", "dikhao", "dikha"}
)


def align_search_terms_with_message(original_msg: str, search_terms: str) -> str:
    """Ensure extracted terms match grains user said — clean English only (basmati wheat)."""
    if not (original_msg or "").strip():
        return dedupe_search_terms(search_terms)
    low = (original_msg or "").lower()
    for pattern, grain in _GRAIN_IN_MESSAGE:
        if not pattern.search(low):
            continue
        if grain == "wheat":
            return "basmati wheat" if "basmati" in low else "wheat"
        return "basmati rice" if "basmati" in low else "rice"
    return dedupe_search_terms(search_terms)


_CONVERSATIONAL_TAIL_RE = re.compile(
    r"\b(?:so\s+that|because|since|for\s+running|to\s+start|i\s+can|i\s+ca|ca\s+start|"
    r"that\s+i|jisse|jise|taaki|taki|ke\s+liye\s+use|use\s+kr|use\s+kar).*$",
    re.IGNORECASE,
)


def scrub_conversational_tail_from_terms(terms: str) -> str:
    """Drop trailing purpose clauses — 'nike shoes so that i ca start running' → 'nike shoes'."""
    t = (terms or "").strip()
    if not t:
        return t
    t = _CONVERSATIONAL_TAIL_RE.sub("", t).strip()
    return re.sub(r"\s+", " ", t).strip()


def polish_search_terms(terms: str, original_msg: str = "") -> str:
    """User-facing + OpenSearch query: English only, no filler (no 'gehu dikhana wheat')."""
    from services.opensearch_products import _extract_size_from_text, _strip_size_from_title_query

    terms = scrub_conversational_tail_from_terms(terms)
    if original_msg:
        terms = align_search_terms_with_message(original_msg, terms)
    size_val = _extract_size_from_text(f"{terms} {original_msg}")
    if size_val:
        terms = _strip_size_from_title_query(terms or "", size_val)
    words = []
    seen: set[str] = set()
    en_grains = {"wheat", "rice"}
    has_en_grain = any(g in (terms or "").lower().split() for g in en_grains)
    for w in re.findall(r"[a-z0-9]+", (terms or "").lower()):
        if len(w) < 2 or w in _FILLER_WORDS or w in _SHOW_TYPO_DROP:
            continue
        if has_en_grain and w in _HINDI_GRAIN_DROP:
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
    return " ".join(words[:8]).strip()


_SHOW_TYPO_DROP = frozenset({"shoe", "shwo", "sho", "showw", "shw"})

_FILLER_WORDS = frozenset(
    {
        "mereko", "mujhe", "mere", "ko", "me", "mai", "main", "mene", "maine", "bhai", "sir",
        "pls", "please", "dikha", "dikhao", "dikho", "dikhado", "dikhana", "dikhaa", "bata",
        "batao", "btao", "bta", "de", "do", "dena", "dedo", "chahiye", "chiye", "chaahiye",
        "milega", "milegi", "milta", "milti", "mil", "sakta", "skta", "sakti", "hai", "h",
        "he", "ho", "hun", "kya", "ky", "ka", "ke", "ki", "se", "liye", "wala", "wali", "wale",
        "ek", "koi", "kuch", "bas", "sirf", "only", "show", "need", "want", "buy", "lenaa",
        "lena", "pahan", "pahanna", "pehen", "pehenne", "peene", "peena", "khane", "khana",
        "to", "yrr", "yr", "yrrr", "yar", "yaar", "bro", "bhaiy", "pls", "aur", "and", "or", "the", "a", "an",
        "hello", "helo", "hey", "sun", "suno", "na", "ab", "wo", "wahi", "pehle", "pasand",
        "kiya", "tha", "laga", "achha", "accha", "achhe", "bahut", "bhi", "badiya", "badhiya",
        "sa", "saa", "liked", "like", "lagta", "lagti", "jaldi", "please", "plz", "kindly",
        "color", "colour", "rang", "size", "sizes", "product", "products",
        "girlfriend", "boyfriend", "meri", "mera", "mere", "gf", "bf", "wife", "husband",
        "offtopic", "topic", "kese", "kaise", "ese", "galat", "wrong", "theek", "complaint",
        "mast", "si", "jayga", "jayega", "jayega", "kyaaa", "kyaa", "mil", "jayegi", "bahi",
        "keliye", "lene", "lena", "id", "is", "the", "need",
    }
)

_LIFESTYLE_BRANDS = frozenset(
    {
        "nike", "adidas", "puma", "reebok", "milton", "cello", "borosil", "gucci", "prada",
        "woodland", "redtape", "bata", "skechers", "fila", "uspolo", "levis", "hrx",
    }
)

_PHONE_BRANDS = frozenset(
    {
        "samsung", "iphone", "apple", "vivo", "oppo", "realme", "redmi", "mi", "xiaomi",
        "oneplus", "poco", "nothing", "google", "motorola", "nokia", "honor", "infinix",
    }
)

_ALL_KNOWN_BRANDS = _PHONE_BRANDS | _LIFESTYLE_BRANDS

_PRODUCT_NOUNS = (
    "cover", "case", "charger", "cable", "adapter", "protector", "glass", "bumper",
    "phone", "mobile", "earphone", "earbuds", "watch", "watches", "band", "screen",
    "tempered", "back", "skin", "pouch", "suit", "bat", "cricket", "shirt", "tshirt",
    "dress", "jeans", "shoes", "sneaker", "kurta", "ejector", "opener", "remover",
)

# Filler/noise tokens for search-term cleanup only — NOT used for routing intent (AI routes first).
_META_CONVERSATION_WORDS = frozenset(
    {
        "offtopic", "topic", "kese", "kaise", "ese", "galat", "wrong", "theek",
        "complaint", "bol", "bola", "raha", "rahi", "kar", "karo",
    }
)

_CONTEXT_SPLIT_MARKERS = re.compile(
    r"\b(?:to\s+ab|ab\s+na|phir\s+ab|so\s+ab|toh\s+ab|ab\s+to)\b",
    re.I,
)

_KE_LIYE_CLAUSE = re.compile(
    r"([a-z0-9]+(?:\s+[a-z0-9]+){0,3}?)\s+ke\s+liye\s+(.+?)(?=\s+(?:bhi\s+)?(?:bta|batao|btao|dikhao|dikha|de|do|dena|chahiye)\b|$)",
    re.I,
)


def extract_focused_product_query(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    """
    AI-first catalog search terms from any language / long Hinglish message.
    Falls back to lightweight heuristics only when the product LLM is unavailable.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        skip_llm = should_skip_micro_classifier_llm()
    except ImportError:
        skip_llm = False

    if not skip_llm:
        try:
            ai = ai_extract_product_search(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang="",
            )
            if ai and ai.get("is_shopping") is True:
                terms = polish_search_terms(
                    (ai.get("search_terms") or "").strip(),
                    original_msg,
                )
                if terms and len(terms) >= 2:
                    # #region agent log
                    try:
                        from services.debug_session_log import dbg97

                        dbg97(
                            "H16",
                            "product_query_understanding.py:extract_focused_product_query",
                            "ai_product_terms",
                            {
                                "msg_preview": comb[:60],
                                "terms": terms[:60],
                            },
                        )
                    except ImportError:
                        pass
                    # #endregion
                    return terms
        except Exception:
            pass

    return _extract_focused_product_query_heuristic(original_msg, msg_en)


def _extract_focused_product_query_heuristic(original_msg: str, msg_en: str = "") -> str:
    """
    Offline fallback when product LLM is unavailable — not used for routing intent.
    """
    text = re.sub(r"\s+", " ", f"{original_msg or ''} {msg_en or ''}".lower()).strip()
    if not text:
        return ""

    parts = _CONTEXT_SPLIT_MARKERS.split(text)
    focus = (parts[-1] if parts else text).strip() or text

    recipient = re.search(
        r"(?:^|\s)(?:mereko|mujhe|main)?\s*(?:\w+\s+){0,3}(\w+)\s+ke\s+(?:liye|lie)\s+"
        r"(.+?)(?:\s+(?:lene|lena|chahiye|dikha|dikhao|dikhado|bta|batao|badiya|accha)\b|$)",
        text,
        re.I,
    )
    if recipient:
        who = (recipient.group(1) or "").lower()
        if who not in _PHONE_BRANDS and who not in _PRODUCT_NOUNS and len(who) >= 3:
            return clean_product_part_label(recipient.group(2), original_msg)

    clauses = list(_KE_LIYE_CLAUSE.finditer(text))
    if clauses:
        focus = clauses[-1].group(0)

    brand = ""
    for m in re.finditer(r"\b([a-z0-9]+)\s+ke\s+liye\b", focus):
        token = m.group(1).split()[-1]
        if token in _PHONE_BRANDS:
            brand = token

    nouns = [n for n in _PRODUCT_NOUNS if re.search(rf"\b{re.escape(n)}\b", focus)]
    if not nouns:
        nouns = [n for n in _PRODUCT_NOUNS if re.search(rf"\b{re.escape(n)}\b", text)]
        if clauses and nouns:
            focus = clauses[-1].group(0)
            for m in re.finditer(r"\b([a-z0-9]+)\s+ke\s+liye\b", focus):
                token = m.group(1).split()[-1]
                if token in _PHONE_BRANDS:
                    brand = token

    if not brand:
        for b in _PHONE_BRANDS:
            if re.search(rf"\b{re.escape(b)}\b", focus):
                brand = b
                break

    if brand and nouns:
        return clean_product_part_label(f"{brand} {nouns[0]}", original_msg)

    if nouns:
        chunk = focus
        if clauses:
            chunk = clauses[-1].group(2)
        return clean_product_part_label(chunk, original_msg)

    gift = re.search(
        r"(?:meri\s+)?(?:gf|girlfriend|bf|boyfriend)\s+ke\s+liye\s+(.+?)(?:\s+(?:dikha|dikhao|bta|batao|chahiye)\b|$)",
        text,
        re.I,
    )
    if gift:
        return clean_product_part_label(gift.group(1), original_msg)

    for n in _PRODUCT_NOUNS:
        if re.search(rf"\b{re.escape(n)}\b", text):
            m = re.search(
                rf"((?:black|white|red|blue|green|grey|gray|neeli|kali|safed)\s+)?{re.escape(n)}",
                text,
                re.I,
            )
            if m:
                return clean_product_part_label(m.group(0), original_msg)
            return clean_product_part_label(n, original_msg)

    return ""

_PRODUCT_TYPO_MAP = {
    "lofar": "loafer",
    "lofers": "loafers",
    "lofer": "loafer",
    "tshrt": "tshirt",
    "tshert": "tshirt",
    "polo": "polo",
    "inifinix": "infinix",
    "infinx": "infinix",
    "infenix": "infinix",
    "nobile": "mobile",
    "moblie": "mobile",
    "mobil": "mobile",
    "shoe": "shoes",
    "shoese": "shoes",
}


def clean_product_part_label(text: str, original_msg: str = "") -> str:
    """
    Catalog-ready search label: strip Hinglish filler (to yrr, dikha de), fix typos (lofar→loafer).
    """
    if not (text or "").strip():
        return ""
    words = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower()):
        for w in raw.replace("-", " ").split():
            if len(w) < 2 or w in _FILLER_WORDS or w in _SHOW_TYPO_DROP:
                continue
            w = _PRODUCT_TYPO_MAP.get(w, w)
            if w not in seen:
                seen.add(w)
                words.append(w)
    out = " ".join(words[:8]).strip()
    if out:
        return out
    return polish_search_terms(text, original_msg)


def resolve_catalog_search_terms_for_message(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    conversation_context: str = "",
    force_llm: bool = False,
) -> str:
    """
    Catalog search terms for locked product routes — AI extract first (any language),
    heuristics only when LLM is unavailable.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        skip_llm = False if force_llm else should_skip_micro_classifier_llm()
    except ImportError:
        skip_llm = False

    if not skip_llm:
        try:
            ai = ai_extract_product_search(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang="",
            )
            if ai and ai.get("is_shopping") is True:
                terms = polish_search_terms(
                    (ai.get("search_terms") or "").strip(),
                    original_msg,
                )
                if terms and len(terms.strip()) >= 2:
                    return terms.strip()
        except Exception:
            pass

    focused = _extract_focused_product_query_heuristic(original_msg, msg_en)
    if focused and len(focused.strip()) >= 2:
        return focused.strip()

    polished = clean_product_part_label(
        polish_search_terms(comb, original_msg),
        original_msg,
    )
    if polished and len(polished.strip()) >= 2:
        return polished.strip()

    try:
        from utils.helpers import extract_product_search_query

        route = ai_route if isinstance(ai_route, dict) else {
            "intent": "product",
            "data_channel": "catalog",
            "run_catalog_search": True,
            "_product_catalog_locked": True,
        }
        sq = extract_product_search_query(original_msg, msg_en, ai_route=route)
        if sq and len(str(sq).strip()) >= 2:
            return str(sq).strip()
    except ImportError:
        pass
    return ""


def shopping_extract_plausible(
    original_msg: str, msg_en: str, search_terms: str
) -> bool:
    """Reject conversational fragments misread as product search by the extract LLM."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    sq = (search_terms or "").strip()
    if not comb or not sq:
        return False
    if is_noisy_search_query(sq):
        return False
    try:
        from services.opensearch_products import _extract_product_keywords

        if _extract_product_keywords(sq):
            return True
    except ImportError:
        pass
    try:
        from services.turn_intent_gate import (
            is_non_catalog_meta_turn,
            message_has_catalog_search_signal,
        )

        if message_has_catalog_search_signal(comb):
            return True
        if is_non_catalog_meta_turn(comb):
            return False
    except ImportError:
        pass
    sq_words = re.findall(r"[a-z0-9]+", sq.lower())
    if any(w in sq_words for w in ("instagram", "facebook", "youtube", "twitter", "linkedin", "telegram", "social", "link", "handle")):
        return False
    return False


def dedupe_search_terms(text: str) -> str:
    """Remove repeated tokens and filler (shirt shirt -> shirt)."""
    if not text:
        return ""
    try:
        from services.conversation_followup import is_non_product_search_phrase

        if is_non_product_search_phrase(text):
            return ""
    except ImportError:
        pass
    words = []
    seen = set()
    for w in re.findall(r"[a-z0-9\-]+", (text or "").lower()):
        w = w.replace("-", "")
        if len(w) < 2 or w in _FILLER_WORDS:
            continue
        w = _PRODUCT_TYPO_MAP.get(w, w)
        if w not in seen:
            seen.add(w)
            words.append(w)
    return " ".join(words[:8]).strip()


def is_noisy_search_query(text: str, **kwargs) -> bool:
    """
    Legacy alias — prefer catalog_title_unusable(..., ai_route=..., entities=...)
    when brain context is available (entity-aware, no phrase lists).
    """
    if kwargs:
        try:
            from services.catalog_spec_semantics import catalog_title_unusable

            return catalog_title_unusable(text, **kwargs)
        except ImportError:
            pass
    if not text or len(text.strip()) < 2:
        return True
    low = text.lower()
    words = re.findall(r"[a-z0-9]+", low)
    if re.search(r"\b(?:hello|hi|hey)\b", low) and re.search(
        r"\b(?:kesa|kaise|kese|kaisa)\b", low
    ):
        if not any(re.search(rf"\b{re.escape(n)}\b", low) for n in _PRODUCT_NOUNS):
            return True
    if re.search(r"\b(?:tu|tum|aap)\b", low) and "search" in low and not any(
        re.search(rf"\b{re.escape(n)}\b", low) for n in _PRODUCT_NOUNS
    ):
        return True
    if any(w in _META_CONVERSATION_WORDS for w in words) and any(
        re.search(rf"\b{re.escape(n)}\b", low) for n in _PRODUCT_NOUNS
    ):
        return True
    if len(words) >= 7:
        chatter = sum(
            1
            for w in words
            if w in _FILLER_WORDS or w in ("hello", "sun", "suno", "iphone", "pasand", "mene")
        )
        if chatter >= 3:
            return True
    if "sun na" in low or "hello" in low:
        if any(n in low for n in _PRODUCT_NOUNS) and len(words) >= 6:
            return True
    try:
        from utils.helpers import (
            message_is_knowledge_information_request,
            message_is_welfog_about_request,
            _text_is_order_delivery_issue,
        )

        if (
            message_is_welfog_about_request(text)
            or message_is_knowledge_information_request(text)
            or _text_is_order_delivery_issue(text)
        ):
            return True
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(text):
            return True
    except ImportError:
        pass
    low = text.lower()
    if any(
        x in low
        for x in (
            "privacy policy", "privacy", "terms and", "terms of", "refund policy",
            "return policy", "shipping policy", "payment policy", "welfog policy",
        )
    ):
        return True
    if any(x in low for x in ("iska", "iski", "sku iska", "dikana basmati", "paani peene")):
        return len(low.split()) > 4
    words = low.split()
    if len(words) != len(set(words)):
        dup_ratio = len(words) - len(set(words))
        if dup_ratio >= 2 or (dup_ratio >= 1 and len(words) <= 4):
            return True
    return False


def ai_extract_product_search(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> Optional[dict[str, Any]]:
    """
    Use Groq to extract shopping intent from any wording.
    Returns dict with search_terms, color, brand, max_price, sku, reasoning — or None.
    """
    system_prompt = f"""You extract Welfog PRODUCT SEARCH parameters from user messages.
Users write in English, Hinglish, Hindi, Tamil, Telugu, Bengali, Gujarati, Punjabi, Kannada, Malayalam, Marathi, Urdu, or mix.
Messages may be indirect ("paani peene ke liye jug", "pahan ne ke liye safed shirt", "basmati ke chawal").

IMPORTANT — do not confuse products:
- chawal / chaawal → rice ONLY when user said chawal/rice words.
- gehu / gehun / wheat → wheat (NEVER replace with rice).
- "basmati ke gehu" → search_terms "basmati wheat" (NOT basmati rice).
- "basmati ke chawal" → "basmati rice".
- basmati alone does NOT mean rice; use the grain word the user said (gehu, chawal, etc.).
- Use ONLY the CURRENT user message; ignore earlier chat unless they say "same/wahi/pehle wala".

Return ONLY valid JSON:
{{
  "reasoning": "brief step-by-step",
  "search_terms": "2-6 English keywords for catalog search (product type + attributes). NEVER repeat words. No filler.",
  "color": "Single standard English colour (White, Black, Sky Blue, Red...) or empty — no notes in parentheses",
  "brand": "brand if mentioned else empty",
  "brand_aliases": ["title spellings for that brand"],
  "mandatory_match_tokens": ["all tokens required in product name"],
  "product_type": "cover|shirt|rice|... or empty",
  "max_price": number or null,
  "min_price": number or null,
  "rating_min": number or null,
  "rating_max": number or null,
  "pro_id": number or null,
  "sku": "SKU code if user pasted one else empty",
  "match_mode": "strict" | "universal",
  "is_shopping": true/false
}}

RULES:
- Translate to English catalog words: safed→white, kala→black, aasmani/asmani→sky blue, shirt→shirt, jug→water jug.
- Map grains only when that word appears: chawal→rice, gehu→wheat — put result ONLY in search_terms (e.g. "basmati wheat"), never keep gehu/dikhana in search_terms.
- search_terms must be short clean English (2-4 words): "basmati wheat", "black shirt", "water jug" — never "basmati gehu dikhana wheat".
- Do NOT include conversational words: mereko, bhai, dikhao, chahiye, liye, peene, khane, pahan, bta de, sun na, hello, pasand kiya, mene.
- If the user tells a story then asks for a NEW product (e.g. liked iPhone cover, now wants Realme cover), use ONLY the new product in search_terms (e.g. "realme cover") — ignore the old iPhone story.
- Long messages: extract 2-4 English words for what they want NOW, not the full sentence.
- search_terms must NOT duplicate tokens (wrong: "shirt shirt", "basmati chawal basmati chawal").
- Keep product-specific words: basmati rice, basmati wheat, white shirt, water jug, black shirt.
- match_mode strict when brand/model in search_terms (iphone cover, redmi cover); universal for generic charger/rice/shirt only.
- If user only asks about orders/tracking/refund with NO product to buy, is_shopping=false and search_terms="".
- If user vents about relationships, emotions, or personal life (e.g. gf left, setting chhod ke bhag gyi) with NO product to buy, is_shopping=false and search_terms="".
- Typos are OK — infer the product (cabel→cable, moblie→mobile) in search_terms.
- "cabel/cable tumhare pass" / "milega kya" on Welfog = USB/charging cable PRODUCT shopping — NOT internet/WiFi connection.
- If unsure of product, put best guess in search_terms (2-4 words max).
- max_price / min_price → customer purchase price (NOT MRP/unit_price). Use for rs/budget/range only.
- rating_min / rating_max for star filters (e.g. rating > 2 → rating_min=2, under 3 stars → rating_max=3).
- pro_id when user gives numeric product id (2615316 is product id) — set pro_id, search_terms="".
- {language_reply_instruction(reply_lang)}
JSON only."""

    user_payload = f"ORIGINAL MESSAGE:\n{original_msg.strip()}\n"
    if (msg_en or "").strip() and msg_en.strip().lower() != original_msg.strip().lower():
        user_payload += f"\nENGLISH HINT (may be imperfect):\n{msg_en.strip()}\n"
    if (conversation_context or "").strip():
        user_payload = (
            "RECENT CONVERSATION (secondary; CURRENT message overrides if different product):\n"
            f"{conversation_context.strip()[-1200:]}\n\n"
            + user_payload
        )

    out = llm_json_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=220,
        timeout_sec=12,
        max_attempts=2,
    )
    if not out:
        return None
    if out.get("is_shopping") is not True:
        log_reasoning(
            f"Product LLM extract: not shopping — {(out.get('reasoning') or '')[:100]}"
        )
        return out

    terms = polish_search_terms((out.get("search_terms") or "").strip(), original_msg)
    if terms:
        out["search_terms"] = terms
    out["color"] = sanitize_llm_color(
        str(out.get("color") or ""),
        terms,
        original_msg,
    ) or ""
    log_reasoning(
        f"Product LLM extract: terms={out.get('search_terms')!r} color={out.get('color')!r}"
    )
    return out


_CATALOG_COLORS = frozenset(
    {
        "Black", "White", "Red", "Green", "Blue", "Yellow", "Pink", "Purple",
        "Brown", "Grey", "Gray", "Orange", "Multicolor", "Sky Blue", "Light Blue",
        "Navy", "Maroon", "Beige", "Silver", "Gold",
    }
)


def _message_has_multi_product_intent(msg: str) -> bool:
    """True when user asked for 2+ items — do not bleed one item's colour onto another."""
    if not (msg or "").strip():
        return False
    try:
        from services.welfog_api import collect_multi_product_parts, multi_product_parts_are_valid

        parts = collect_multi_product_parts(msg)
        return len(parts) >= 2 and multi_product_parts_are_valid(parts)
    except Exception:
        return False


def sanitize_llm_color(
    color: str,
    search_terms: str,
    original_msg: str,
    *,
    trust_llm: bool = False,
) -> Optional[str]:
    """Reject LLM colour noise; map Hinglish colour words safely."""
    from services.opensearch_products import normalize_color_fuzzy

    c = (color or "").strip()
    if c and (len(c) > 20 or "(" in c or "assuming" in c.lower()):
        c = ""
    blob = f"{search_terms} {c}".lower()
    if "aasmani" in blob or "asmani" in blob or "sky blue" in blob or "sky-blue" in blob:
        return "Sky Blue"
    resolved = normalize_color_fuzzy(c) or normalize_color_fuzzy(search_terms)
    if not resolved and original_msg and not _message_has_multi_product_intent(original_msg):
        resolved = normalize_color_fuzzy(original_msg)
    if resolved and resolved in _CATALOG_COLORS:
        return resolved
    if trust_llm and c and len(c) <= 28 and re.match(r"^[A-Za-z][A-Za-z\s\-]*$", c):
        return c if " " in c else c.title()
    if resolved in ("White", "Black") and ("aasmani" in blob or "asmani" in blob):
        return "Sky Blue"
    return resolved if resolved in _CATALOG_COLORS else None


def merge_llm_into_search_spec(
    spec: dict[str, Any],
    llm: Optional[dict[str, Any]],
    original_msg: str = "",
) -> dict[str, Any]:
    """Apply LLM extraction on top of heuristic OpenSearch spec."""
    if not llm:
        return spec
    terms = polish_search_terms((llm.get("search_terms") or "").strip(), original_msg)
    if terms:
        spec["title_query"] = terms
    llm_color = sanitize_llm_color(
        str(llm.get("color") or ""),
        terms,
        original_msg,
    )
    if llm_color:
        spec["color"] = llm_color
    if llm.get("brand") and not spec.get("brand"):
        spec["brand"] = llm.get("brand")
    try:
        from services.catalog_spec_from_ai import merge_ai_into_catalog_spec

        spec = merge_ai_into_catalog_spec(spec, llm, original_msg=original_msg)
    except ImportError:
        mode = (llm.get("match_mode") or "").strip().lower()
        if mode == "strict":
            spec["title_match_strict"] = True
        elif mode == "universal":
            spec["title_match_strict"] = False
    if llm.get("sku") and not spec.get("sku"):
        spec["sku"] = llm.get("sku")
    try:
        if llm.get("pro_id") is not None and not spec.get("pro_id"):
            spec["pro_id"] = int(llm["pro_id"])
            spec["title_query"] = ""
    except (TypeError, ValueError):
        pass
    try:
        if llm.get("rating_min") is not None and spec.get("rating_min") is None:
            spec["rating_min"] = float(llm["rating_min"])
        if llm.get("rating_max") is not None and spec.get("rating_max") is None:
            spec["rating_max"] = float(llm["rating_max"])
    except (TypeError, ValueError):
        pass
    try:
        if llm.get("max_price") is not None and spec.get("purchase_price_max") is None:
            spec["purchase_price_max"] = float(llm["max_price"])
        if llm.get("min_price") is not None and spec.get("purchase_price_min") is None:
            spec["purchase_price_min"] = float(llm["min_price"])
    except (TypeError, ValueError):
        pass
    return spec


def understand_product_query(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    category_id=None,
    color=None,
    pro_id=None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """
    AI-first OpenSearch spec — one product-search LLM understands filters in any language;
    heuristics only when the model is unavailable.
    """
    from services.opensearch_products import build_catalog_search_spec

    llm: Optional[dict[str, Any]] = None
    ai_payload: Optional[dict[str, Any]] = None

    try:
        from services.product_search_flow import (
            _merge_product_requests,
            _request_dict_to_ai_understanding,
            ai_understand_product_search,
        )

        llm = ai_understand_product_search(
            original_msg,
            conversation_context,
            reply_lang,
        )
        if llm and llm.get("action") != "not_shopping" and llm.get("is_shopping", True):
            ai_payload = dict(llm)
            ai_payload["_ai_first"] = True
            ai_payload["_product_nlu_from_ai"] = True
            ai_payload["is_shopping"] = True

            requests = llm.get("product_requests")
            if isinstance(requests, list) and len(requests) >= 2:
                merged_req = _merge_product_requests(requests)
                for k, v in _request_dict_to_ai_understanding(merged_req).items():
                    if v not in (None, "", []):
                        ai_payload[k] = v
            elif isinstance(requests, list) and len(requests) == 1:
                for k, v in _request_dict_to_ai_understanding(requests[0]).items():
                    if v not in (None, "", []):
                        ai_payload[k] = v
    except ImportError:
        pass

    if not ai_payload:
        try:
            ai_payload = ai_extract_product_search(
                original_msg, msg_en, conversation_context, reply_lang
            )
            if ai_payload:
                ai_payload["_ai_first"] = True
                ai_payload["_product_nlu_from_ai"] = True
        except Exception:
            ai_payload = None

    try:
        from services.product_filter_pipeline import apply_understanding_sku_pro_id_fixes

        if ai_payload:
            ai_payload = apply_understanding_sku_pro_id_fixes(
                ai_payload, original_msg, msg_en
            ) or ai_payload
        else:
            gap = apply_understanding_sku_pro_id_fixes({}, original_msg, msg_en)
            if gap and (
                gap.get("search_terms")
                or gap.get("pro_id")
                or gap.get("sku")
                or gap.get("mandatory_match_tokens")
            ):
                ai_payload = gap
                ai_payload["_ai_first"] = True
    except ImportError:
        pass

    try:
        from services.catalog_spec_semantics import scrub_ai_product_understanding

        if ai_payload:
            ai_payload = scrub_ai_product_understanding(
                ai_payload, original_msg, msg_en
            ) or ai_payload
    except ImportError:
        pass

    spec = build_catalog_search_spec(
        original_msg,
        msg_en,
        ai=ai_payload,
        category_id=category_id,
        color=color or (ai_payload or {}).get("color"),
        pro_id=pro_id,
    )

    if ai_payload and spec.get("title_query"):
        spec["title_query"] = clean_product_part_label(
            spec["title_query"], original_msg
        )

    return spec, llm or ai_payload


def display_label_for_product_search(
    spec: dict,
    llm: Optional[dict],
    original_msg: str = "",
) -> str:
    """Clean label for chat replies — always like 'basmati wheat', never 'gehu dikhana wheat'."""
    if spec.get("pro_id") is not None:
        return f"Product ID {spec['pro_id']}"
    label = ""
    if llm and llm.get("search_terms"):
        st = str(llm["search_terms"]).strip().lower()
        if st not in ("products", "product", ""):
            label = clean_product_part_label(
                polish_search_terms(llm["search_terms"], original_msg), original_msg
            )
    if not label:
        label = clean_product_part_label(spec.get("title_query") or "", original_msg)
    if label and is_noisy_search_query(label, understanding=llm or {}, entities=spec):
        label = ""
    if (not label or label.lower() in ("products", "product")) and spec.get("brand"):
        tq = clean_product_part_label(spec.get("title_query") or "", original_msg)
        label = f"{spec['brand']} {tq}".strip() if tq else str(spec["brand"]).strip()
    if not label:
        return "your search"
    extras = []
    try:
        from services.opensearch_products import _is_valid_sku_token

        if spec.get("sku") and _is_valid_sku_token(str(spec["sku"])):
            extras.append(f"SKU {spec['sku']}")
    except ImportError:
        pass
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
    color = (spec.get("color") or "").strip()
    if color:
        cl = color.lower()
        label_low = label.lower()
        color_in_label = cl in label_low
        if not color_in_label and cl in ("grey", "gray"):
            color_in_label = "grey" in label_low or "gray" in label_low
        if not color_in_label:
            label = f"{color} {label}"
    if spec.get("brand"):
        try:
            from services.product_catalog_resolver import sanitize_catalog_brand

            clean_brand = sanitize_catalog_brand(
                str(spec.get("brand") or ""),
                product_name=str(spec.get("title_query") or ""),
                explicit_from_brain=True,
            )
        except ImportError:
            clean_brand = str(spec.get("brand") or "").strip()
        bl = clean_brand.strip().lower()
        if bl and bl not in label.lower().split():
            extras.append(f"{clean_brand} brand")
    if extras:
        return f"{label} ({', '.join(extras)})"
    return label


def spec_uses_strict_filter_not_found(spec: dict) -> bool:
    """SKU/price/brand/colour/strict match → polite not-found (not unrelated products)."""
    try:
        from services.opensearch_products import _is_valid_sku_token

        has_sku = bool(spec.get("sku") and _is_valid_sku_token(str(spec["sku"])))
    except ImportError:
        has_sku = bool(spec.get("sku"))
    return bool(
        has_sku
        or spec.get("unit_price_max") is not None
        or spec.get("unit_price_min") is not None
        or spec.get("purchase_price_max") is not None
        or spec.get("purchase_price_min") is not None
        or spec.get("rating_min") is not None
        or spec.get("rating_max") is not None
        or spec.get("brand")
        or spec.get("color")
        or spec.get("title_match_strict")
        or spec.get("pro_id")
    )
