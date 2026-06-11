"""
Intent phrase matching across Welfog supported languages:
en, hinglish, hi, gu, pa, ta, te, mr, ml, kn, bn, ur.

Use together with English/Hinglish heuristics on:
  combined = f"{original_message} {msg_en}"
where msg_en comes from Google Translate for native-script chats.
"""
from __future__ import annotations

import re

# --- Order history: show MY past orders (not how-to, not single-track) ---

ORDER_HISTORY_NATIVE_PHRASES = (
    # Hindi / Marathi (Devanagari) — shared script
    "ऑर्डर इतिहास",
    "मेरे ऑर्डर",
    "मेरी ऑर्डर",
    "मेरा ऑर्डर",
    "पुराने ऑर्डर",
    "ऑर्डर दिखाओ",
    "ऑर्डर बताओ",
    "ऑर्डर सूची",
    "खरीद इतिहास",
    "माझे ऑर्डर",
    "माझ्या ऑर्डर",
    "ऑर्डर दाखवा",
    # Tamil
    "ஆர்டர்கள்",
    "என் ஆர்டர்",
    "எனது ஆர்டர்",
    "ஆர்டர் வரலாறு",
    "ஆர்டர் பட்டியல்",
    "ஆர்டர் காட்டு",
    "ஆர்டர் காட்டுங்கள்",
    # Telugu
    "నా ఆర్డర్",
    "ఆర్డర్ చరిత్ర",
    "ఆర్డర్ చూపించు",
    "ఆర్డర్ చూపించండి",
    "ఆర్డర్ జాబితా",
    # Kannada
    "ನನ್ನ ಆರ್ಡರ್",
    "ಆರ್ಡರ್ ಇತಿಹಾಸ",
    "ಆರ್ಡರ್ ತೋರಿಸಿ",
    # Malayalam
    "എന്റെ ഓർഡർ",
    "ഓർഡർ ചരിത്രം",
    "ഓർഡർ കാണിക്കുക",
    # Bengali
    "আমার অর্ডার",
    "অর্ডার ইতিহাস",
    "অর্ডার দেখান",
    "অর্ডার তালিকা",
    # Gujarati
    "મારા ઓર્ડર",
    "ઓર્ડર ઇતિહાસ",
    "ઓર્ડર બતાવો",
    # Punjabi (Gurmukhi)
    "ਮੇਰੇ ਆਰਡਰ",
    "ਆਰਡਰ ਇਤਿਹਾਸ",
    "ਆਰਡਰ ਦਿਖਾਓ",
    # Urdu
    "آرڈر",
    "میرے آرڈر",
    "میرا آرڈر",
    "آرڈر کی تاریخ",
    "آرڈر دکھائیں",
)

ORDER_HISTORY_HOWTO_NATIVE = (
    "कैसे देख",
    "कैसे देखू",
    "कैसे चेक",
    "कहाँ देख",
    "कहां देख",
    "எப்படி பார",
    "எப்படி காண",
    "ఎలా చూడ",
    "എങ്ങനെ കാണ",
    "কিভাবে দেখ",
    "કેવી રીતે જોવ",
    "ਕਿਵੇਂ ਦੇਖ",
    "کیسے دیکھ",
)

ORDER_TRACKING_NATIVE_PHRASES = (
    # Hindi / Marathi
    "ऑर्डर कहाँ",
    "ऑर्डर कहां",
    "ऑर्डर स्थिति",
    "ऑर्डर ट्रैक",
    "ट्रैक कर",
    "कब आएगा",
    "कब मिलेगा",
    "डिलीवरी",
    "ऑर्डर नहीं आया",
    "ऑर्डर नहीं मिला",
    # Tamil
    "ஆர்டர் எங்கே",
    "ஆர்டர் நிலை",
    "டிராக்",
    "எப்போது வரும்",
    "டெலிவரி",
    # Telugu
    "ఆర్డర్ ఎక్కడ",
    "ఆర్డర్ స్థితి",
    "ట్రాక్",
    "ఎప్పుడు వస్తుంది",
    # Bengali
    "অর্ডার কোথায়",
    "অর্ডার স্ট্যাটাস",
    "ট্র্যাক",
    # Gujarati
    "ઓર્ડર ક્યાં",
    "ઓર્ડર સ્થિતિ",
    # Punjabi
    "ਆਰਡਰ ਕਿੱਥੇ",
    # Urdu
    "آرڈر کہاں",
    "آرڈر کی حیثیت",
    "ٹریک",
    # Kannada / Malayalam
    "ಆರ್ಡರ್ ಎಲ್ಲಿದೆ",
    "ഓർഡർ എവിടെ",
)

# English + Hinglish (also appear after auto-translate)
ORDER_HISTORY_LATIN_EXTRA = (
    "previous orders",
    "past purchases",
    "purchase list",
    "order records",
    "all my purchases",
    "view my orders",
    "see my orders",
    "list my orders",
    "order history show",
    "show order history",
    "which orders",
    "what orders",
    "orders i placed",
    "orders i made",
    "orders list",
    "my order list",
)

ORDER_TRACKING_LATIN_EXTRA = (
    "where is my package",
    "when will i receive",
    "has not arrived",
    "not received yet",
    "shipping status",
    "courier status",
    "track my order",
    "track order",
    "order tracking",
    "delivery update",
    "parcel status",
    "shipment update",
    "order delayed",
    "order stuck",
    "out for delivery",
    "where is my order",
    "when will order arrive",
    "order not delivered",
    "still waiting for order",
)

# Roman Hindi / Hinglish (no translation needed)
ORDER_TRACKING_HINGLISH_PHRASES = (
    "order track",
    "track kr",
    "track kar",
    "trck kr",
    "trck kar",
    "order trck",
    "order trak",
    "mera order",
    "mere order",
    "order ka status",
    "order status",
    "order kahan",
    "order kaha",
    "order kab",
    "kab aayega",
    "kab aaega",
    "kab milega",
    "kab tak aayega",
    "delivery status",
    "delivery update",
    "parcel kahan",
    "package kahan",
    "order nahi aaya",
    "order nhi aaya",
    "order nahi aya",
    "order nhi aya",
    "order nahi mila",
    "order nhi mila",
    "abhi tak nahi",
    "abhi tak order",
    "order late",
    "order delay",
    "order pending",
    "order stuck",
    "courier kahan",
    "shipment kahan",
    "order id track",
    "id se track",
    "status bata",
    "status btao",
    "status check",
    "live status",
    "tracking bata",
    "track krke bata",
    "track krke batana",
    "nahi pahucha",
    "nhi pahucha",
    "nahi pahuncha",
    "ab bhi nahi",
    "abhi bhi nhi",
    "mere pass nahi",
    "mere paas nahi",
    "delivery nahi",
    "delivery nhi",
)


def intent_combined_text(original: str, msg_en: str = "") -> str:
    """Single string for matching: customer text + English routing translation."""
    parts = [original or "", msg_en or ""]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _contains_any(haystack: str, needles) -> bool:
    if not haystack:
        return False
    return any(n in haystack for n in needles if n)


def native_order_history_request(text: str) -> bool:
    """True when message (any supported script) asks to see own order list."""
    if not text or not text.strip():
        return False
    if _contains_any(text, ORDER_HISTORY_HOWTO_NATIVE):
        return False
    return _contains_any(text, ORDER_HISTORY_NATIVE_PHRASES)


def native_order_tracking_request(text: str) -> bool:
    """True when message asks track/status/where/when for an order."""
    if not text or not text.strip():
        return False
    return _contains_any(text, ORDER_TRACKING_NATIVE_PHRASES)


def multilingual_order_history_match(combined: str, original: str = "") -> bool:
    """Broad multilingual match for purchase-history list intent."""
    src = original or combined
    if native_order_history_request(src):
        return True
    low = f" {combined.lower()} "
    if _contains_any(combined, ORDER_HISTORY_LATIN_EXTRA):
        return True
    # Translated patterns from Indian languages → English
    if "order" in low and "history" in low:
        if not any(x in low for x in (" how ", " how to ", " kaise ", " steps ", " process ")):
            return True
    if re.search(r"\b(my|mine|past|previous|all)\b.*\borders?\b", low):
        if not any(x in low for x in (" how ", " how to ", " kaise ")):
            return True
    return False


def multilingual_order_tracking_match(combined: str, original: str = "") -> bool:
    src = original or combined
    try:
        from utils.helpers import _text_has_delivery_serviceability_intent

        if _text_has_delivery_serviceability_intent(combined, "") or _text_has_delivery_serviceability_intent(
            src, ""
        ):
            return False
    except ImportError:
        pass
    if native_order_tracking_request(src):
        return True
    low = f" {combined.lower()} "
    if _contains_any(combined, ORDER_TRACKING_LATIN_EXTRA):
        return True
    if _contains_any(combined, ORDER_TRACKING_HINGLISH_PHRASES):
        return True
    if re.search(r"\b(track|trck|trak|traking|status|ship|parcel|package|courier)\b", low) and re.search(
        r"\borders?\b|\bordr\w*\b", low
    ):
        return True
    if re.search(r"\bdeliver", low) and re.search(r"\borders?\b", low):
        if any(x in low for x in ("track", "tracking", "status", "kab ", "nahi aaya", "nahi aya")):
            return True
        return False
    if "order" in low and any(
        w in low
        for w in (
            " track",
            "tracking",
            " status",
            " shipment",
            " where ",
            " when ",
            " arrive",
            " received",
            " delayed",
            " pending",
        )
    ):
        return True
    return False


def multilingual_how_to_order_history(combined: str, original: str = "") -> bool:
    """How-to / steps to view orders (help text, not API list)."""
    from utils.helpers import (
        _text_has_order_placement_intent,
        _user_rejects_viewing_wants_placement,
        message_blocks_order_history_routing,
    )

    src = original or combined
    if message_blocks_order_history_routing(src):
        return False
    if _text_has_order_placement_intent(src) or _user_rejects_viewing_wants_placement(src):
        return False
    if _contains_any(src, ORDER_HISTORY_HOWTO_NATIVE):
        return True
    low = f" {combined.lower()} "
    process = (
        " how ",
        " how to ",
        " steps ",
        " process ",
        " kaise ",
        " kese ",
        " guide ",
        " where to find ",
        " kaise dekhu ",
        " kaise dekh ",
        " kaise check ",
        " dekhne ka process ",
        " app pr ",
        " app par ",
        " kaha jake ",
        " bta de ",
        " btao ",
    )
    orderish = ("order", "orders", "history", "ऑर्डर", "ஆர்டர்", "ఆర్డర్", "অর্ডার", "آرڈر")
    return any(p in low for p in process) and any(o in combined for o in orderish)
