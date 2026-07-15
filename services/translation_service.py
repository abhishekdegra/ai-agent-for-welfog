"""
Customer reply language: match how the user writes.

Supported:
- en: English (Latin)
- hinglish: Roman Hinglish (Latin + Hindi particles) — never auto-translate to Devanagari
- hi, mr: Devanagari (Hindi / Marathi — disambiguated when possible)
- pa: Punjabi (Gurmukhi), gu: Gujarati, ta: Tamil, te: Telugu, kn: Kannada, ml: Malayalam, bn: Bengali, ur: Urdu
"""
import os
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


def _english_html_to_roman_hinglish(text: str, user_msg: str = "") -> str:
    """
    Legacy name — rewrites into natural Roman Hinglish via LLM (not en→hi Google + romanize).
    """
    if not text or not str(text).strip():
        return text
    if user_msg:
        return _llm_rewrite_customer_reply(text, user_msg, "hinglish")
    return text


def _infer_reply_text_language(text: str) -> str:
    """Best-effort language of bot output text."""
    plain = re.sub(r"<[^>]+>", " ", text or "").strip()
    if not plain:
        return "en"
    script = _detect_script_language(plain)
    if script:
        return script
    if DEVANAGARI_RE.search(plain):
        return "hi"
    if is_hinglish_message(plain):
        return "hinglish"
    if _looks_english_only_latin(plain):
        return "en"
    if _reply_style_mismatches_hinglish(plain):
        return "hi"
    return "en"


def _reply_style_mismatches_hinglish(text: str) -> bool:
    """Formal romanized Hindi (Google+romanize artifact) — not natural Hinglish."""
    plain = re.sub(r"<[^>]+>", " ", text or "").lower().strip()
    if not plain or not _latin_script_only(plain):
        return False
    if DEVANAGARI_RE.search(text or ""):
        return True
    if is_hinglish_message(plain):
        return False
    if _looks_english_only_latin(plain):
        return False
    hing = _hinglish_marker_count(plain)
    eng = len(set(re.findall(r"\b[a-zA-Z]{2,}\b", plain)) & _ENGLISH_CONVERSATIONAL)
    if hing >= 1:
        return False
    long_words = re.findall(r"\b[a-zA-Z]{5,}\b", plain)
    return eng <= 1 and len(long_words) >= 3


def _reply_needs_style_rewrite(text: str, target_lang: str, user_msg: str = "") -> bool:
    """True when bot text language/style does not match the customer's target."""
    target_lang = resolve_customer_reply_lang(user_msg, target_lang)
    if target_lang == "en":
        plain = re.sub(r"<[^>]+>", " ", text or "").strip()
        return bool(plain) and not _looks_english_only_latin(plain) and not _latin_script_only(plain)
    if target_lang == "hinglish":
        return _reply_style_mismatches_hinglish(text) or _looks_english_only_latin(
            re.sub(r"<[^>]+>", " ", text or "").strip()
        )
    if target_lang in NATIVE_SCRIPT_LANGS:
        return not _bot_text_matches_reply_lang(text, target_lang)
    return False


def _llm_rewrite_customer_reply(
    text: str,
    user_msg: str,
    reply_lang: str,
    *,
    timeout_sec: float | None = None,
    max_tokens: int | None = None,
    max_providers: int | None = None,
) -> str:
    """Rewrite answer text into customer's language/style — facts and numbers preserved."""
    if not (text or "").strip() or not (user_msg or "").strip():
        return text
    rl = resolve_customer_reply_lang(user_msg, reply_lang)
    try:
        from services.ai_service import (
            _llm_json_with_provider_fallback,
            _llm_classifier_provider_chain,
            _trim_text_mid,
        )
    except ImportError:
        return text
    providers = _llm_classifier_provider_chain()
    if not providers:
        return text
    try:
        from services.chat_resilience import chat_turn_abandoned

        if chat_turn_abandoned():
            return text
    except ImportError:
        pass
    to = float(timeout_sec) if timeout_sec is not None else float(
        os.getenv("LLM_REWRITE_TIMEOUT_SEC") or "4"
    )
    toks = int(max_tokens) if max_tokens is not None else 700
    # One provider only — rewrite cascade was stacking second/third LLMs past deadline.
    if max_providers is None:
        try:
            max_providers = int(os.getenv("LLM_REWRITE_MAX_PROVIDERS") or "1")
        except (TypeError, ValueError):
            max_providers = 1
    providers = providers[: max(1, int(max_providers))]
    system_prompt = f"""Rewrite the ANSWER for the customer. Return ONLY JSON: {{"response":"..."}}

RULES:
- Detect the customer's language/script/style ONLY from CUSTOMER WROTE (not from guesses or keyword lists).
- Match that language and conversational style exactly in the rewritten answer.
- {language_reply_instruction(rl)}
- Copy every number, phone, email, URL, day-count, and policy fact EXACTLY from the source.
- Do NOT add facts. Do NOT change meaning. Do NOT answer a different question.
- Keep the answer as short as the source allows — do not expand into extra policy sections.
- Keep HTML tags/structure when the source has HTML.
- Roman Hinglish = casual WhatsApp-style mixed Hindi+English in Latin script — never formal Sanskritized transliteration.
JSON only."""
    user_payload = (
        f"CUSTOMER WROTE:\n{_trim_text_mid(user_msg, 320)}\n\n"
        f"ANSWER TO REWRITE:\n{_trim_text_mid(text, 1800)}"
    )
    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=toks,
        timeout_sec=to,
        max_attempts=1,
    )
    resp = (data.get("response") or "").strip() if data else ""
    return resp if resp else text


def _log_language_preservation(
    user_msg: str,
    reply_lang: str,
    body_in: str,
    body_out: str,
    *,
    translation_applied: bool,
) -> None:
    log_reasoning(
        f"Language preservation: input={resolve_customer_reply_lang(user_msg, reply_lang)} "
        f"output={_infer_reply_text_language(body_out)} translated={translation_applied}"
    )


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


# Process-local cache: Hinglish/native → English for KB dense retrieval (aligns with English chunks).
_RETRIEVAL_EN_CACHE: dict[str, str] = {}
_RETRIEVAL_EN_CACHE_MAX = 128


def text_usable_as_english_retrieval(text: str) -> bool:
    """True when text is clear English — safe to embed against English KB chunks."""
    plain = re.sub(r"<[^>]+>", " ", text or "").strip()
    if len(plain) < 2:
        return False
    if not _latin_script_only(plain):
        return False
    if is_hinglish_message(plain):
        return False
    # Script + existing language detectors only — no word/topic lists.
    return _looks_english_only_latin(plain)


def _retrieval_gloss_looks_corrupted(raw: str, gloss: str) -> bool:
    """
    Reject MT that clearly inverted polarity vs the customer text.

    Structural only: contracted/English negation appeared in gloss but not source.
    """
    src = re.sub(r"<[^>]+>", " ", raw or "").lower()
    dst = re.sub(r"<[^>]+>", " ", gloss or "").lower()
    if not src or not dst or src == dst:
        return False
    src_neg = bool(re.search(r"\bn'?t\b|\bnot\b", src))
    dst_neg = bool(re.search(r"\bn'?t\b|\bnot\b|n't\b", dst))
    return bool(dst_neg and not src_neg)


def to_en_for_retrieval(text: str, lang: str = "") -> str:
    """
    English gloss for Qdrant dense retrieval against English-indexed KB chunks.

    Unlike to_en_for_routing (Brain stays multilingual on raw Hinglish), retrieval
    must embed an English meaning or Hinglish/Hindi queries miss English chunks.
    Uses a short Google translate timeout + process cache — no keyword maps.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    lang = _normalize_language(lang) if lang else resolve_customer_reply_lang(raw)
    if lang == "en" and text_usable_as_english_retrieval(raw):
        return raw.lower()
    if text_usable_as_english_retrieval(raw) and lang not in NATIVE_SCRIPT_LANGS:
        # Already English-like (even if reply_lang mis-detected).
        return raw.lower()

    # Latin script (English / Roman Hinglish / mixed): never block on Google MT.
    # Dense+fulltext hybrid + brain meaning/keys already align retrieval; MT adds
    # latency and polarity flips. Only native scripts need an English gloss.
    if _latin_script_only(raw) and lang not in NATIVE_SCRIPT_LANGS:
        return raw.lower()

    cache_key = f"{lang}|{raw.lower()[:900]}"
    hit = _RETRIEVAL_EN_CACHE.get(cache_key)
    if hit is not None:
        return hit

    gloss = raw.lower()
    # Protect brand name — Google often mistranslates "Welfog" → "welfare".
    brand_token = "WelfogBrandName"
    protected = re.sub(r"\bwelfog\b", brand_token, raw, flags=re.IGNORECASE)
    translated_ok = False
    try:
        import concurrent.futures

        def _do_translate():
            return GoogleTranslator(source="auto", target="en").translate(protected)

        timeout = float(os.getenv("RETRIEVAL_TRANSLATE_TIMEOUT_SEC", "1.2") or "1.2")
        limit = max(0.5, min(timeout, 2.0))
        # wait=False — Google hangs must not block the with-block exit after timeout.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_do_translate)
            result = fut.result(timeout=limit)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if result and isinstance(result, str) and len(result.strip()) > 1:
            gloss = result.strip()
            gloss = re.sub(re.escape(brand_token), "Welfog", gloss, flags=re.IGNORECASE)
            # Residual mistranslation if protector was stripped by the MT engine.
            if re.search(r"\bwelfog\b", raw, flags=re.IGNORECASE):
                gloss = re.sub(r"\bwelfare\b", "Welfog", gloss, flags=re.IGNORECASE)
            gloss = gloss.lower()
            # If MT produced non-English junk, keep raw — dense+fulltext still run on it.
            if not text_usable_as_english_retrieval(gloss) and is_hinglish_message(gloss):
                gloss = raw.lower()
            elif _retrieval_gloss_looks_corrupted(raw, gloss):
                log_reasoning(
                    f"KB retrieval EN gloss rejected (corrupted MT) chars={len(raw)}"
                )
                gloss = raw.lower()
            else:
                translated_ok = True
                log_reasoning(
                    f"KB retrieval EN gloss: lang={lang} chars={len(raw)}→{len(gloss)}"
                )
    except TimeoutError:
        log_reasoning(
            f"KB retrieval translate timeout — embedding original ({len(raw)} chars)."
        )
    except Exception as e:
        log_reasoning(f"KB retrieval translate: {e}")

    # Only cache successful MT — never permanently pin a timeout/raw miss.
    if translated_ok:
        if len(_RETRIEVAL_EN_CACHE) >= _RETRIEVAL_EN_CACHE_MAX:
            try:
                _RETRIEVAL_EN_CACHE.pop(next(iter(_RETRIEVAL_EN_CACHE)))
            except StopIteration:
                _RETRIEVAL_EN_CACHE.clear()
        _RETRIEVAL_EN_CACHE[cache_key] = gloss
    return gloss


def to_en_for_routing(text: str, lang: str = "") -> str:
    """
    English hint for ai_brain_route / OpenSearch — includes Hinglish and native scripts.
    Roman Hinglish skips external translate (brain is multilingual); native scripts use
    a short timeout so /chat never blocks minutes on Google.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    lang = _normalize_language(lang) if lang else resolve_customer_reply_lang(raw)
    if lang in ("en", "hinglish"):
        return raw.lower()
    try:
        import concurrent.futures

        def _do_translate():
            return GoogleTranslator(source="auto", target="en").translate(raw)

        timeout = float(os.getenv("ROUTING_TRANSLATE_TIMEOUT_SEC", "4") or "4")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_do_translate)
            result = fut.result(timeout=max(1.0, timeout))
        if result and isinstance(result, str) and len(result.strip()) > 1:
            return result.strip().lower()
    except TimeoutError:
        log_reasoning(
            f"Routing translate timeout — using original text ({len(raw)} chars)."
        )
    except Exception as e:
        log_reasoning(f"Routing translate (to_en_for_routing): {e}")
    return raw.lower()


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


def finalize_customer_reply(
    text: str,
    user_msg: str,
    reply_lang: str = "",
    *,
    allow_llm_style_rewrite: bool = True,
) -> str:
    """
    Last-mile localization before sending to the customer.
    Match language/script/style of the customer's latest message.

    allow_llm_style_rewrite=False: skip English↔Hinglish LLM rewrite (use when the
    answer LLM already received reply_lang instructions — avoids a second 1–7s call).
    Native-script localization still runs when needed.
    """
    if text is None:
        return text
    if not isinstance(text, str):
        return text
    body = text.strip()
    if not body:
        return text
    rl = resolve_customer_reply_lang(user_msg, reply_lang)
    translation_applied = False
    out = text

    if rl in ("hinglish", "en"):
        if is_live_api_structured_html(body):
            _log_language_preservation(user_msg, rl, body, out, translation_applied=False)
            return text
        if allow_llm_style_rewrite and _reply_needs_style_rewrite(body, rl, user_msg):
            rewritten = _llm_rewrite_customer_reply(text, user_msg, rl)
            if rewritten and rewritten.strip() and rewritten.strip() != body:
                out = rewritten
                translation_applied = True
        if allow_llm_style_rewrite and not translation_applied:
            out_lang = _infer_reply_text_language(out)
            if out_lang != rl:
                rewritten = _llm_rewrite_customer_reply(out, user_msg, rl)
                if rewritten and rewritten.strip():
                    plain_out = re.sub(r"<[^>]+>", " ", out).strip()
                    plain_new = re.sub(r"<[^>]+>", " ", rewritten).strip()
                    if plain_new and plain_new != plain_out:
                        out = rewritten
                        translation_applied = True
        _log_language_preservation(user_msg, rl, body, out, translation_applied=translation_applied)
        return out

    if rl in NATIVE_SCRIPT_LANGS:
        if _bot_text_matches_reply_lang(body, rl):
            _log_language_preservation(user_msg, rl, body, out, translation_applied=False)
            return text
        out = localize_for_customer(text, rl)
        translation_applied = out.strip() != body
        _log_language_preservation(user_msg, rl, body, out, translation_applied=translation_applied)
        return out

    _log_language_preservation(user_msg, rl, body, out, translation_applied=False)
    return text


def should_translate_reply(user_msg: str, reply_lang: str, bot_text: str = "") -> bool:
    """Translate templated English replies when customer uses a native-script language."""
    reply_lang = resolve_customer_reply_lang(user_msg, reply_lang)
    if reply_lang in ("en", "hinglish"):
        return False
    if bot_text and _bot_text_matches_reply_lang(bot_text, reply_lang):
        return False
    return reply_lang in NATIVE_SCRIPT_LANGS
