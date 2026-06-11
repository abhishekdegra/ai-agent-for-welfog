"""
Deterministic meta-turn classification before catalog / order API / KB routing.

Catches hostile text, bot-behavior complaints, topic denials, and other non-shopping
turns that LLMs often mislabel as product search — especially when providers are down.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from utils.reasoning_log import log_reasoning


@dataclass(frozen=True)
class MetaTurn:
    kind: str
    reply_key: str


_INSULT_TOKENS = frozenset(
    {
        "pagal", "paagal", "bewakoof", "bewakuf", "idiot", "stupid", "dumb", "ullu",
        "gadha", "chutiya", "chutya", "bsdk", "madarchod", "mc", "bc", "bakwas",
        "bekar", "faltu", "nikamma", "nalayak", "kutta", "kutte", "gandu", "gand",
    }
)

_HOSTILE_SUBSTRINGS = (
    "gandm", "gandu", "chutiy", "madarch", "bsdk", "bhosd", "lund", "randi",
)

_BOT_PERF_MARKERS = (
    "itna time", "time lagaya", "time lagta", "time le rha", "time le ra", "time le raha",
    "time laga", "kitna time", "bahut time", "zyada time", "der se", "slow reply",
    "reply dene", "reply dene me", "jawab dene", "jawab dene me", "response time",
    "itne der", "late reply", "late ho", "jaldi nahi", "jldi nahi",
)

_BOT_REF_MARKERS = (
    "tu ", "tum ", "aap ", "you ", "reply", "jawab", "response", "bol rha", "bol raha",
    "de rha", "de raha", "dene me", "lene me", "kyu ", "kyo ", "why ",
)

_TOPIC_DENIAL_MARKERS = (
    "ki baat nhi", "ki baat nahi", "ki baat ni", "baat nhi kr", "baat nahi kr",
    "baat ni kr", "nhi bol rha", "nahi bol rha", "ni bol rha", "nahi bol raha",
    "not talking about", "not about order", "order id ki baat nhi", "order id nahi bol",
    "tracking ki baat nhi", "track ki baat nhi", "ese bol rha", "aise bol rha",
    "me to ese", "main to ese", "me to aise", "main to aise",
)

_ORDER_TOPIC_WORDS = (
    "order id", "orderid", "tracking", "track order", "order track", "order status",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(text))


def message_denies_order_or_tracking_topic(text: str) -> bool:
    tl = f" {_norm(text)} "
    if not any(d in tl for d in _TOPIC_DENIAL_MARKERS):
        return False
    return any(t in tl for t in _ORDER_TOPIC_WORDS) or any(p in tl for p in _BOT_PERF_MARKERS)


def _has_phrase(tl: str, phrase: str) -> bool:
    """Word-boundary aware phrase match — avoids 'der se' inside 'order search'."""
    p = phrase.strip().lower()
    if " " in p:
        return p in tl
    return bool(re.search(rf"\b{re.escape(p)}\b", tl))


def message_is_bot_latency_complaint(text: str) -> bool:
    tl = f" {_norm(text)} "
    if not any(_has_phrase(tl, p) for p in _BOT_PERF_MARKERS):
        return False
    if any(
        x in tl
        for x in (
            "order kab", "kab aayega", "kab aaega", "kab milega", "delivery time",
            "shipping time", "courier", "shipment", "parcel kab",
        )
    ):
        if not any(b in tl for b in ("reply", "jawab", "bol rha", "de rha", "dene me")):
            return False
    return any(b in tl for b in _BOT_REF_MARKERS) or "dene me" in tl or "reply" in tl


def message_is_hostile_or_insult_turn(text: str) -> bool:
    if not (text or "").strip():
        return False
    tl = f" {_norm(text)} "
    try:
        from utils.helpers import message_is_bot_search_complaint

        if message_is_bot_search_complaint(text):
            return False
    except ImportError:
        pass
    words = _tokens(text)
    if any(w in _INSULT_TOKENS for w in words):
        try:
            from utils.helpers import _message_has_catalog_product_signal

            if not _message_has_catalog_product_signal(text):
                return True
        except ImportError:
            return True
    if any(s in tl for s in _HOSTILE_SUBSTRINGS):
        try:
            from utils.helpers import _message_has_catalog_product_signal

            if not _message_has_catalog_product_signal(text):
                return True
        except ImportError:
            return True
    if re.search(r"\b(?:oy|oye|are|arre|sun)\s+(?:tu|tum|aap)\b", tl) and any(
        w in _INSULT_TOKENS for w in words
    ):
        return True
    return False


def is_non_catalog_meta_turn(text: str, conversation_context: str = "") -> bool:
    """True when this turn must never hit OpenSearch / product_ai_flow."""
    if not (text or "").strip():
        return False
    try:
        from utils.helpers import (
            message_is_bot_capability_question,
            message_is_bot_search_complaint,
            message_is_user_confused_or_rephrasing_bot,
        )

        if message_is_bot_search_complaint(text, conversation_context):
            return True
        if message_is_bot_capability_question(text):
            return True
        if message_is_user_confused_or_rephrasing_bot(text, conversation_context):
            if message_is_hostile_or_insult_turn(text) or message_is_bot_latency_complaint(text):
                return True
            if message_denies_order_or_tracking_topic(text):
                return True
    except ImportError:
        pass
    if message_is_hostile_or_insult_turn(text):
        return True
    if message_is_bot_latency_complaint(text):
        return True
    if message_denies_order_or_tracking_topic(text):
        return True
    return False


def message_has_catalog_search_signal(text: str) -> bool:
    """
  Minimum bar for catalog search: real product/brand signal, not insults or meta talk.
    """
    if not (text or "").strip():
        return False
    if is_non_catalog_meta_turn(text):
        return False
    try:
        from utils.helpers import (
            _message_has_catalog_product_signal,
            _message_has_generic_shopping_item_signal,
            _SHOPPING_ACTION_MARKERS,
            extract_product_id,
        )
    except ImportError:
        return False

    if extract_product_id(text):
        return True
    tl = f" {_norm(text)} "
    if _message_has_catalog_product_signal(text) or _message_has_generic_shopping_item_signal(text):
        if any(m in tl for m in _SHOPPING_ACTION_MARKERS):
            return True
        if re.search(r"\b(?:h\s+ky|hai\s+ky|h\s+kya|hai\s+kya|milega|milta|chahiye|chiye)\b", tl):
            return True
        if re.search(r"\b(?:under|below|upto|upto|rs|₹|\d{2,5})\b", tl):
            return True
    return False


def classify_meta_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> Optional[MetaTurn]:
    comb = _norm(f"{original_msg} {msg_en}")
    if not comb:
        return None

    try:
        from services.user_query_semantics import query_is_welfog_company_or_platform

        if query_is_welfog_company_or_platform(original_msg, msg_en, conversation_context):
            return None
        from services.meta_turn_semantics import should_fast_reply_assistant_intro

        if should_fast_reply_assistant_intro(original_msg, msg_en):
            log_reasoning("Meta-turn: assistant identity (semantic failsafe).")
            return MetaTurn("assistant_intro", "assistant_intro")
    except ImportError:
        pass

    try:
        from utils.helpers import message_is_bot_search_complaint

        if message_is_bot_search_complaint(comb, conversation_context):
            log_reasoning("Meta-turn: bot search behavior complaint.")
            return MetaTurn("bot_search_complaint", "bot_search_behavior_help")
    except ImportError:
        pass

    if message_is_bot_latency_complaint(comb):
        log_reasoning("Meta-turn: bot latency complaint — not order tracking.")
        return MetaTurn("bot_latency", "bot_latency_apology")

    if message_denies_order_or_tracking_topic(comb):
        log_reasoning("Meta-turn: user denies order-id/tracking topic — conversational.")
        return MetaTurn("topic_denial", "bot_topic_correction")

    if message_is_hostile_or_insult_turn(comb):
        log_reasoning("Meta-turn: hostile/insult — not catalog search.")
        return MetaTurn("hostile", "bot_insult_calm")

    return None


def format_meta_turn_reply(
    meta: MetaTurn,
    original_msg: str,
    reply_lang: str = "",
) -> str:
    from services.kb_service import sysmsg
    from services.translation_service import (
        customer_reply_language,
        is_hinglish_message,
        localize_for_customer,
    )

    rl = (reply_lang or customer_reply_language(original_msg) or "en").lower()
    key = meta.reply_key
    if rl == "hinglish" or is_hinglish_message(original_msg):
        body = sysmsg(f"{key}_hinglish") or sysmsg(key) or ""
    else:
        body = sysmsg(key) or sysmsg(f"{key}_hinglish") or ""
    if body and rl not in ("en", "hinglish"):
        body = localize_for_customer(body, rl)
    return body or ""


def apply_meta_turn_to_route(route: dict, meta: MetaTurn) -> dict:
    out = dict(route or {})
    out["intent"] = "general"
    out["needs_order_id"] = False
    out["search_query"] = ""
    out["numeric_context"] = "none"
    out["data_channel"] = "kb"
    out["continue_previous_topic"] = False
    out["route_handler"] = meta.reply_key
    out.pop("handler", None)
    if meta.kind == "bot_search_complaint":
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "refund"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        out["route_handler"] = "policy_structured_kb"
    return out
