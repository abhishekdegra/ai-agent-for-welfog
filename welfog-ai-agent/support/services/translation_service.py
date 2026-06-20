"""
Customer reply language: match how the user writes.

Supported:
- en: English (Latin)
- hinglish: Roman Hinglish (Latin + Hindi particles) — never auto-translate to Devanagari
- hi, mr: Devanagari (Hindi / Marathi — disambiguated when possible)
- pa: Punjabi (Gurmukhi), gu: Gujarati, ta: Tamil, te: Telugu, kn: Kannada, ml: Malayalam, bn: Bengali, ur: Urdu
"""
import re
from langdetect import detect
from deep_translator import GoogleTranslator

from utils.reasoning_log import log_reasoning

# Reply language codes (NOT all are Google Translate targets).
SUPPORTED_LANG_CODES = ["en", "hinglish", "hi", "gu", "pa", "ta", "te", "mr", "ml", "kn", "bn", "ur"]

# Native-script Indian languages — bot templates (English) are translated via to_user().
NATIVE_SCRIPT_LANGS = frozenset({"hi", "gu", "pa", "ta", "te", "mr", "ml", "kn", "bn", "ur"})
LANG_DISPLAY_NAMES = {
    "en": "English",
    "hinglish": "Roman Hinglish",
    "hi": "Hindi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "bn": "Bengali",
    "ur": "Urdu",
}

# Roman-Hinglish markers (Latin script). Used for detection — not for English-only routing.
_HINGLISH_MARKERS = frozenset(
    {
        "kya", "hai", "he", "ho", "hu", "hun", "hain", "kar", "kr", "karu", "kru", "karna", "krna",
        "nahi", "nhi", "nahin", "kyu", "kaise", "kese", "iske", "iska", "uska", "mujhe", "mere", "mera",
        "mereko", "meko", "apna", "apni", "pata", "kahan", "kaha", "kidhar", "batao", "btao", "bta",
        "batade", "bata", "samjho", "samajh", "dekh", "suna", "chahiye", "chiye", "chaahiye",
        "krta", "karti", "karte", "skta", "sakta", "sakti", "milega", "milta", "milti", "dikhao", "dikha",
        "bhai", "bhaiya", "yaar", "yr", "yrr", "dede", "dedo", "bhej", "bhejo",
        "aaya", "aay", "aayi", "aya", "ayi", "abhi", "abi", "abh", "tk", "tak", "fir", "phir",
        "wala", "wale", "wali", "mat", "mt", "bol", "bolo", "sun", "suno", "achha", "accha", "theek", "thik",
        "badiya", "badhiya", "sab", "kuch", "koi", "na", "ji", "haan", "han", "nhi", "nh",
        "yeh", "ye", "woh", "vo", "rhi", "rahi", "rha", "raha", "rahe", "rhe", "liya", "diya",
        "idhar", "udhar", "yahan", "wahan", "please", "plz", "sorry", "maaf",
    }
)

# Common English words in short Latin messages (avoid mis-detecting pure English as Hinglish).
_ENGLISH_CONVERSATIONAL = frozenset(
    {
        "the", "is", "are", "and", "or", "please", "my", "you", "me", "i", "am", "im", "i'm",
        "hey", "hi", "hello", "bro", "brother", "what", "whats", "how", "can", "will", "would",
        "track", "order", "status", "where", "want", "need", "payment", "refund", "return",
        "cancel", "delivery", "saying", "good", "thanks", "thank", "yes", "no", "ok", "okay",
        "whatsapp", "wassup", "sup", "dear", "sir", "maam", "mam", "help", "show", "buy",
        "get", "see", "look", "today", "special", "deals", "product", "products", "search",
        "step", "steps", "website", "app", "email", "sms",
    }
)

HTML_TAG_RE = re.compile(r"(<[^>]+>)")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_DEV_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_DEV_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
_DEV_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_DEV_SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "्": ""}


def _romanize_devanagari_text(text: str) -> str:
    if not text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _DEV_VOWELS:
            out.append(_DEV_VOWELS[ch])
            i += 1
            continue
        if ch in _DEV_CONSONANTS:
            base = _DEV_CONSONANTS[ch]
            if i + 1 < n and text[i + 1] in _DEV_MATRAS:
                out.append(base + _DEV_MATRAS[text[i + 1]])
                i += 2
                continue
            if i + 1 < n and text[i + 1] == "्":
                out.append(base)
                i += 2
                continue
            out.append(base + "a")
            i += 1
            continue
        if ch in _DEV_SIGNS:
            out.append(_DEV_SIGNS[ch])
            i += 1
            continue
        if "\u0900" <= ch <= "\u097F":
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _looks_english_only_latin(msg: str) -> bool:
    if not msg:
        return False
    plain = re.sub(r"<[^>]+>", " ", msg)
    plain = (
        plain.replace("—", "-")
        .replace("–", "-")
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("’", "'")
    )
    if not _latin_script_only(plain):
        return False
    ml = plain.lower()
    hinglish_hits = _hinglish_marker_count(ml)
    english_tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", ml))
    english_hits = len(english_tokens & _ENGLISH_CONVERSATIONAL)
    return hinglish_hits == 0 and english_hits >= 1


def _english_html_to_roman_hinglish(text: str) -> str:
    if not text or not str(text).strip():
        return text
    parts = HTML_TAG_RE.split(text)
    out = []
    for part in parts:
        if HTML_TAG_RE.fullmatch(part):
            out.append(part)
            continue
        seg = (part or "").strip()
        if not seg:
            out.append(part)
            continue
        try:
            hi = GoogleTranslator(source="en", target="hi").translate(seg)
            out.append(_romanize_devanagari_text(hi if isinstance(hi, str) else seg))
        except Exception as e:
            log_reasoning(f"Translation error (en->hinglish): {e}. Using original segment.")
            out.append(part)
    return "".join(out)


def _is_valid_lang_code(lang: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,10}(?:-[a-z]{2,3})?", lang or ""))


def _normalize_language(lang: str) -> str:
    """Map aliases; keep `hinglish` distinct from Hindi (Devanagari)."""
    if not lang or not isinstance(lang, str):
        return "en"
    lang = lang.lower().strip()
    if lang in ("hinglish", "roman-hindi", "roman_hindi", "hing-lish"):
        return "hinglish"
    if lang in SUPPORTED_LANG_CODES:
        return lang
    return "en"


def _latin_script_only(msg: str) -> bool:
    if not msg:
        return False
    for ch in msg:
        if ch.isspace() or ch in ".,!?;:'\"()[]{}-_/\\@#$%&*+=<>":
            continue
        if "\u0900" <= ch <= "\u097F":
            return False
        if "\u0600" <= ch <= "\u06FF":
            return False
        if "\u0A00" <= ch <= "\u0A7F":
            return False
        if ord(ch) > 127:
            return False
    return True


def _hinglish_marker_count(msg_lower: str) -> int:
    padded = f" {msg_lower} "
    count = 0
    for w in _HINGLISH_MARKERS:
        if f" {w} " in padded or msg_lower.startswith(f"{w} ") or msg_lower.endswith(f" {w}"):
            count += 1
    return count


def _detect_script_language(msg: str):
    """Native script → language code by dominant character count (avoids Punjabi→Hindi mis-route)."""
    if not msg:
        return None
    counts = {
        "hi": 0,
        "ur": 0,
        "pa": 0,
        "gu": 0,
        "ta": 0,
        "te": 0,
        "kn": 0,
        "ml": 0,
        "bn": 0,
    }
    for ch in msg:
        if "\u0900" <= ch <= "\u097F":
            counts["hi"] += 1
        elif "\u0600" <= ch <= "\u06FF":
            counts["ur"] += 1
        elif "\u0A00" <= ch <= "\u0A7F":
            counts["pa"] += 1
        elif "\u0A80" <= ch <= "\u0AFF":
            counts["gu"] += 1
        elif "\u0B80" <= ch <= "\u0BFF":
            counts["ta"] += 1
        elif "\u0C00" <= ch <= "\u0C7F":
            counts["te"] += 1
        elif "\u0C80" <= ch <= "\u0CFF":
            counts["kn"] += 1
        elif "\u0D00" <= ch <= "\u0D7F":
            counts["ml"] += 1
        elif "\u0980" <= ch <= "\u09FF":
            counts["bn"] += 1
    best_lang, best_n = max(counts.items(), key=lambda x: x[1])
    return best_lang if best_n > 0 else None


def is_hinglish_message(msg: str) -> bool:
    """
    Roman script + Hindi/Hinglish particles (ChatGPT-style Hinglish).
    Not the same as Hindi in Devanagari script.
    """
    if not msg or not _latin_script_only(msg):
        return False
    ml = msg.lower()
    if _hinglish_marker_count(ml) >= 1:
        return True
    # Short Roman-Hindi patterns without spaces
    if re.search(r"\b(kya|hai|nahi|nhi|kaise|kese|bhai|chahiye|batao|btao|bta)\b", ml):
        return True
    return False


def prefer_roman_hinglish_reply(user_msg: str) -> bool:
    """Backward-compatible alias: True when customer writes Roman Hinglish."""
    return detect_language(user_msg) == "hinglish"


def detect_language(msg):
    """
    Choose how the bot should reply — must match customer style:
    - en: English (Latin, no Hinglish markers)
    - hinglish: Roman Hinglish (Latin + Hindi particles) — never auto-translate to Devanagari
    - hi, ur, pa, ...: native script languages
    """
    if not msg or not msg.strip():
        return "en"

    script_lang = _detect_script_language(msg)
    if script_lang:
        # Devanagari is shared by Hindi and Marathi — use langdetect when possible.
        if script_lang == "hi":
            try:
                ld = _normalize_language(detect(msg))
                if ld in ("mr", "hi"):
                    return ld
            except Exception:
                pass
        return script_lang

    if not _latin_script_only(msg):
        return "en"

    if re.fullmatch(r"\s*[0-9]{4,20}\s*", msg or ""):
        return "en"

    msg_lower = msg.lower()
    hinglish_hits = _hinglish_marker_count(msg_lower)
    english_tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", msg_lower))
    english_hits = len(english_tokens & _ENGLISH_CONVERSATIONAL)

    if is_hinglish_message(msg):
        return "hinglish"

    # Pure English (Latin, no Hinglish markers, common English words present)
    if hinglish_hits == 0 and english_hits >= 1:
        return "en"

    try:
        lang = detect(msg)
        lang = _normalize_language(lang if _is_valid_lang_code(lang) else "en")
        if lang in ("hi", "mr", "bn") and hinglish_hits == 0 and english_hits >= 1:
            return "en"
        if lang != "en" and hinglish_hits >= 1:
            return "hinglish"
        return lang if lang in SUPPORTED_LANG_CODES else "en"
    except Exception as e:
        log_reasoning(f"Language detection error: {e}. Fallback heuristic...")
        if hinglish_hits >= 1:
            return "hinglish"
        return "en"


def language_reply_instruction(lang: str) -> str:
    """Prompt fragment: model must write user-facing text in the customer's language."""
    lang = _normalize_language(lang)
    if lang == "en":
        return (
            "LANGUAGE (mandatory): The customer's latest message is in English. "
            'Write the "response" field in clear, natural English only. '
            "Do NOT use Hindi, Hinglish, or Roman Hindi (no words like aaj, kya, dikhao, chahiye, batao, etc.)."
        )
    if lang == "hinglish":
        return (
            'LANGUAGE (mandatory): The customer writes in Roman Hinglish (Hindi + English in Latin script, like ChatGPT India chats). '
            'Write the "response" field in the SAME Roman Hinglish style — casual, natural, mixed. '
            "Examples: \"bhai\", \"order track kaise karein\", \"Order ID bhej do\", \"main check kar dunga\". "
            "Do NOT use Devanagari script (no क ख ग …). Do NOT reply in formal English-only if they used Hinglish. "
            "Keep brand name Welfog as-is."
        )
    name = LANG_DISPLAY_NAMES.get(lang, lang)
    return (
        f"LANGUAGE (mandatory): The customer's latest message is in {name}. "
        f'Write the "response" field entirely in {name}, using the same script they used. '
        "Do not mix in other languages except brand names (e.g. Welfog)."
    )


def to_en(text):
    """Translate text to English for internal routing. Falls back to original if translation fails."""
    if not text or len(text.strip()) == 0:
        return text
    if detect_language(text) == "hinglish":
        return text
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        if result and isinstance(result, str) and len(result.strip()) > 0:
            return result
    except Exception as e:
        log_reasoning(f"Translation error (to_en): {e}. Using original text for processing.")
    return text


def _translate_html(text: str, lang: str) -> str:
    if not text or lang in ("en", "hinglish"):
        return text
    parts = HTML_TAG_RE.split(text)
    translated_parts = []
    for part in parts:
        if HTML_TAG_RE.fullmatch(part):
            translated_parts.append(part)
        else:
            segment = (part or "").strip()
            if not segment:
                translated_parts.append(part)
                continue
            try:
                translated = GoogleTranslator(source="en", target=lang).translate(segment)
                translated_parts.append(translated if isinstance(translated, str) else part)
            except Exception as e:
                log_reasoning(f"Translation error (HTML segment): {e}. Using original segment.")
                translated_parts.append(part)
    return "".join(translated_parts)


def to_user(text, lang):
    """
    Localize bot output for the customer.
    Hinglish: never Google-translate (keeps Roman script). English: as-is.
    Hindi/Urdu/etc.: translate templates when needed.
    """
    lang = _normalize_language(lang)
    if lang in ("en", "hinglish") or not text or not str(text).strip():
        return text
    try:
        if HTML_TAG_RE.search(text):
            return _translate_html(text, lang)
        # KB/templates are authored in English — explicit source avoids Punjabi→Hindi mistakes.
        return GoogleTranslator(source="en", target=lang).translate(text)
    except Exception as e:
        log_reasoning(f"Translation error (to_user): {e}. Returning original text.")
        return text


def _bot_text_matches_reply_lang(text: str, reply_lang: str) -> bool:
    """True when bot HTML/text already appears to be in the customer's script."""
    if not text or not str(text).strip():
        return False
    reply_lang = _normalize_language(reply_lang)
    if reply_lang == "en":
        return _latin_script_only(text)
    if reply_lang == "hinglish":
        return _latin_script_only(text) and not _detect_script_language(text)
    script = _detect_script_language(text)
    return script == reply_lang


def localize_for_customer(text: str, reply_lang: str) -> str:
    """
    Final user-facing text in the customer's language.
    English/Hinglish pass-through; Indian native scripts via en→target translation.
    """
    return to_user(text, reply_lang)


def customer_reply_language(user_msg: str) -> str:
    """Single entry point: detect how the bot should reply for this message."""
    return detect_language(user_msg)


def resolve_customer_reply_lang(user_msg: str, reply_lang: str = "") -> str:
    """
    Final reply language for this turn — aligns detection with how the user actually wrote.
    Roman Hinglish is never collapsed into English or Devanagari Hindi by mistake.
    """
    rl = _normalize_language(reply_lang or customer_reply_language(user_msg or ""))
    if not (user_msg or "").strip():
        return rl
    if _detect_script_language(user_msg):
        return _detect_script_language(user_msg)
    if rl not in NATIVE_SCRIPT_LANGS and is_hinglish_message(user_msg):
        msg_lower = (user_msg or "").lower()
        english_tokens = set(re.findall(r"\b[a-zA-Z]{2,}\b", msg_lower))
        english_hits = len(english_tokens & _ENGLISH_CONVERSATIONAL)
        hinglish_hits = _hinglish_marker_count(msg_lower)
        if hinglish_hits >= 2 or (hinglish_hits >= 1 and english_hits <= 2):
            return "hinglish"
    return rl


def localized_sysmsg_for_customer(
    key: str, user_msg: str, reply_lang: str = "", **fmt
) -> str:
    """System template in the customer's language (en / hinglish / translated native script)."""
    from services.kb_service import sysmsg

    rl = resolve_customer_reply_lang(user_msg, reply_lang)
    if rl == "hinglish":
        body = sysmsg(f"{key}_hinglish", **fmt) or sysmsg(key, **fmt) or ""
    else:
        body = sysmsg(key, **fmt) or ""
    if not body:
        return ""
    return finalize_customer_reply(body, user_msg, rl)


def customer_facing_template(
    key: str,
    user_msg: str,
    reply_lang: str = "",
    *,
    fallback_en: str = "",
    wrap_html: bool = True,
    **fmt,
) -> str:
    """
    Industry-standard customer copy: KB system-messages + Google Translate for every
    SUPPORTED_LANG_CODES (ta, te, gu, pa, mr, ml, kn, bn, ur, hi, hinglish, en).
    Never hardcode per-language if/elif blocks in Python.
    """
    rl = resolve_customer_reply_lang(user_msg, reply_lang)
    body = localized_sysmsg_for_customer(key, user_msg, reply_lang=rl, **fmt)
    if body.strip():
        return body
    if not (fallback_en or "").strip():
        return ""
    text = fallback_en.strip()
    if wrap_html and "<div" not in text.lower():
        text = f"<div style='color:#333;line-height:1.55;'>{text}</div>"
    return finalize_customer_reply(text, user_msg, rl)


_LIVE_API_HTML_MARKERS = (
    "data-wf-live-api=",
    "wf-od-root",
    "wf-ph-root",
    "wf-wl-root",
    "wf-oid-root",
    "wf-pin-root",
    "wf-product-root",
    "wf-invoice-card",
    "wf-invoice-btn",
    "wf-product-rail",
    "Live order tracking",
    "Your wishlist",
)


def is_live_api_structured_html(text: str) -> bool:
    """Order/wishlist/product API cards — keep factual English for en/hinglish users."""
    if not isinstance(text, str) or not text.strip():
        return False
    low = text.lower()
    return any(m.lower() in low for m in _LIVE_API_HTML_MARKERS)


def finalize_customer_reply(text: str, user_msg: str, reply_lang: str = "") -> str:
    """
    Last-mile localization before sending to the customer.
    Native scripts: translate English/Hinglish templates. en/hinglish: pass through.
    """
    if text is None:
        return text
    if not isinstance(text, str):
        return text
    body = text.strip()
    if not body:
        return text
    rl = resolve_customer_reply_lang(user_msg, reply_lang)
    if rl in ("hinglish", "en"):
        if is_live_api_structured_html(body):
            return text
        if rl == "hinglish":
            if DEVANAGARI_RE.search(body):
                return _romanize_devanagari_text(text)
            # Never block /chat on Google en→hi→roman (30–120s+). Hinglish sysmsg templates
            # and product/API cards are already customer-readable in Latin script.
            return text
        return text
    if rl in NATIVE_SCRIPT_LANGS:
        if _bot_text_matches_reply_lang(body, rl):
            return text
        return localize_for_customer(text, rl)
    return text


def should_translate_reply(user_msg: str, reply_lang: str, bot_text: str = "") -> bool:
    """Translate templated English replies when customer uses a native-script language."""
    reply_lang = resolve_customer_reply_lang(user_msg, reply_lang)
    if reply_lang in ("en", "hinglish"):
        return False
    if bot_text and _bot_text_matches_reply_lang(bot_text, reply_lang):
        return False
    return reply_lang in NATIVE_SCRIPT_LANGS
