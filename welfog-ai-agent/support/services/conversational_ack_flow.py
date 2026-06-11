"""
Short acknowledgments (theek h, ok, thanks) after a helpful turn — contextual AI reply
in the user's language, not a generic Order-ID template.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from utils.reasoning_log import log_reasoning


def _last_assistant_turn_lower(conversation_context: str) -> str:
    """Most recent Assistant line only — avoids stale topic bleed from older turns."""
    for line in reversed((conversation_context or "").splitlines()):
        s = line.strip()
        if s.lower().startswith("assistant:"):
            return s.split(":", 1)[-1].lower()
    return (conversation_context or "")[-2000:].lower()


def _topic_from_ctx(ctx: dict | None) -> str:
    if not ctx:
        return ""
    mode = ((ctx.get("data") or {}).get("topic_mode") or "").strip().lower()
    if mode.startswith("wishlist"):
        return "wishlist"
    if mode in ("order_history_list", "order_history_howto"):
        return "order_history"
    if mode == "pincode_check":
        return "pincode_check"
    if "product" in mode or mode == "catalog_search":
        return "product"
    return ""


def infer_recent_assistant_topic(
    conversation_context: str,
    ctx: dict | None = None,
) -> str:
    """
    What we last helped with — drives ack wording (pincode vs order vs product).
    Uses session topic_mode + last Assistant turn (not whole transcript).
    """
    from_ctx = _topic_from_ctx(ctx)
    tail = _last_assistant_turn_lower(conversation_context)
    if not tail.strip():
        return from_ctx or "general"

    pin_markers = (
        "good news",
        "service is available on this pincode",
        "delivery available",
        "delivery not available",
        "yahan delivery",
        "pincode:",
        "pin code",
        "6-digit pin",
        "not available to deliver",
        "serviceability",
        "pincode_check",
    )
    product_markers = (
        "best options for",
        "here are the best",
        "search results",
        "products found",
        "view product",
        "add to cart",
    )
    # Product cards use "View Product" — must win over wishlist (never treat as wishlist).
    if any(m in tail for m in product_markers) or "₹" in tail:
        return "product"
    if any(m in tail for m in pin_markers):
        return "pincode_check"
    if "wishlist" in tail and from_ctx != "product":
        return "wishlist"
    if any(
        m in tail
        for m in ("purchase history", "order history", "your orders", "order list")
    ):
        return "order_history"
    if any(
        m in tail
        for m in (
            "order id",
            "track",
            "shipped",
            "out for delivery",
            "refund",
            "payment status",
            "order status",
        )
    ):
        return "order"
    if from_ctx:
        return from_ctx
    return "general"


def _ack_reply_conflicts_with_topic(reply_html: str, topic: str) -> bool:
    plain = re.sub(r"<[^>]+>", " ", reply_html or "").lower()
    if topic == "product" and "wishlist" in plain:
        return True
    if topic == "pincode_check" and re.search(r"\border\s*id\b", plain):
        return True
    if topic == "wishlist" and re.search(r"\bmy orders\b", plain) and "wishlist" not in plain:
        return True
    return False


def _wrap_ack_html(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if "<" in t and ">" in t:
        return t
    return f"<div style='color:#333;line-height:1.55;'>{t}</div>"


def ai_conversational_ack_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> str:
    if (os.getenv("CONVERSATIONAL_ACK_AI", "1") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import (
        finalize_customer_reply,
        language_reply_instruction,
        resolve_customer_reply_lang,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    topic = infer_recent_assistant_topic(conversation_context, ctx=ctx)
    compact_ctx = _compact_conversation_context(conversation_context or "", 2000)
    user_line = _trim_text_mid(comb, 200)

    topic_hints = {
        "pincode_check": (
            "We just answered a delivery/PIN serviceability check. "
            "Acknowledge briefly. Offer another PIN check or ordering on Welfog if useful. "
            "Do NOT mention Order ID."
        ),
        "order": (
            "We just helped with an order (track/refund/payment). "
            "Acknowledge briefly. Offer further order help if needed. "
            "Only mention Order ID if they were actively tracking one."
        ),
        "order_history": (
            "We showed order history or how-to. Acknowledge briefly; offer list/track/refund help."
        ),
        "wishlist": (
            "We showed wishlist items or how to open wishlist in the app. "
            "Acknowledge briefly. Do NOT say My Orders — wishlist uses the heart/Wishlist icon."
        ),
        "product": (
            "We showed PRODUCT SEARCH results (catalog cards). Acknowledge briefly; invite more "
            "product search or shopping help. NEVER mention wishlist, heart icon, or saved items."
        ),
        "general": (
            "Brief thanks/okay after Welfog support. Warm 1-2 sentences; invite next question."
        ),
    }

    system_prompt = f"""You are Welfog support. The user sent a SHORT acknowledgment or closing (e.g. theek h, ok, thanks, sahi hai) — NOT a new question.
Return ONLY JSON: {{"reasoning":"1 line","response":"reply"}}

LAST_TOPIC: {topic}
TOPIC_HINT: {topic_hints.get(topic, topic_hints["general"])}

RULES:
- "response" in the SAME language/script as the user's LATEST message (Hinglish, Hindi, English, etc.).
- 1-2 short sentences only — natural, friendly, human.
- Match what we JUST did in RECENT CONVERSATION (PIN delivery result, order status, products, etc.).
- NEVER say "Order ID ki zarurat nahi" or push Order ID when LAST_TOPIC is pincode_check or product.
- Do NOT repeat the full previous answer (PIN result, policy dump).
- NEVER copy or echo the user's latest message verbatim (no parroting "sahi to h check kr").
- Do NOT ask them to send PIN again if they only said thanks/okay after a successful check.
- If LAST_TOPIC is product: NEVER say wishlist — they saw search results, not wishlist.
- If LAST_TOPIC is wishlist: only then mention wishlist/heart icon — not for product search.
{language_reply_instruction(rl)}
JSON only."""

    user_payload = f"LAST_TOPIC: {topic}\n"
    if compact_ctx:
        user_payload += f"\nRECENT CONVERSATION:\n{compact_ctx}\n"
    user_payload += f"\nLATEST USER MESSAGE (ack only):\n{user_line}"

    providers = _llm_provider_chain()
    if not providers:
        return ""

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=160,
        timeout_sec=10,
        max_attempts=2,
    )
    if not data:
        return ""
    raw = (data.get("response") or "").strip()
    if not raw:
        return ""
    plain_user = re.sub(r"\s+", " ", comb.lower().strip())
    plain_reply = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw).lower().strip())
    if plain_user and len(plain_user) >= 8 and (
        plain_user == plain_reply or plain_user in plain_reply
    ):
        log_reasoning("Conversational ack rejected — echoed user message verbatim.")
        return ""
    log_reasoning(
        f"Conversational ack AI (topic={topic}): {(data.get('reasoning') or '')[:100]}"
    )
    body = finalize_customer_reply(_wrap_ack_html(raw), original_msg or msg_en, rl)
    if _ack_reply_conflicts_with_topic(body, topic):
        log_reasoning(
            f"Conversational ack rejected — reply mentioned wrong topic (expected {topic})."
        )
        return ""
    return body


def template_ack_for_topic(topic: str, original_msg: str, reply_lang: str) -> str:
    from services.order_history_flow import _localized_sysmsg
    from services.kb_service import sysmsg

    keys = {
        "pincode_check": ("feedback_ack_pincode", "feedback_ack_pincode_hinglish"),
        "order": ("feedback_ack_order", "feedback_ack_order_hinglish"),
        "order_history": ("feedback_ack_order_history", "feedback_ack_order_history_hinglish"),
        "wishlist": ("feedback_ack_wishlist", "feedback_ack_wishlist_hinglish"),
        "product": ("feedback_ack_product", "feedback_ack_product_hinglish"),
        "general": ("feedback_ack_short", "feedback_ack_short_hinglish"),
    }
    en_key, hi_key = keys.get(topic, keys["general"])
    body = (
        _localized_sysmsg(hi_key if reply_lang == "hinglish" else en_key, original_msg, reply_lang=reply_lang)
        or _localized_sysmsg(en_key, original_msg, reply_lang=reply_lang)
        or sysmsg(en_key)
        or sysmsg(hi_key)
        or ""
    )
    return body


def ai_chitchat_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> str:
    """
    Natural ChatGPT-style reply for greetings, thanks, wellbeing, bot availability, bye.
  """
    if (os.getenv("CHITCHAT_AI_REPLY", "1") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import (
        finalize_customer_reply,
        language_reply_instruction,
        resolve_customer_reply_lang,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1800)
    user_line = _trim_text_mid(comb, 240)

    system_prompt = f"""You are Welfog's friendly shopping support assistant in live chat.

The user sent CASUAL CHIT-CHAT (not a shopping/order question): greeting, thanks, praise, bye,
"how are you", "what are you doing", "are you free/busy", light small talk — ANY language/script.

Return ONLY JSON: {{"reasoning":"1 line","response":"reply"}}

RULES:
- "response" in the SAME language/script as the user's LATEST message (Hinglish, Hindi, English, etc.).
- 1-3 short sentences — warm, natural, human (like a helpful friend, not a corporate bot).
- Greeting → greet back briefly; mention you help with Welfog shopping/orders if natural (not a sales pitch).
- Thanks → acknowledge warmly ("welcome", "khushi hui help karke"); do NOT restart with "Hello! How can I help".
- "Are you free/busy?" → you are always here to help on chat; light friendly tone.
- "What are you doing?" → you're here on Welfog support chat, ready to help.
- Bye / no help needed → warm sign-off; invite them back anytime.
- Do NOT search products, do NOT ask for Order ID, do NOT dump policies.
- Do NOT echo the user's message verbatim.
{language_reply_instruction(rl)}
JSON only."""

    user_payload = ""
    if compact_ctx:
        user_payload += f"RECENT CONVERSATION:\n{compact_ctx}\n\n"
    user_payload += f"LATEST USER MESSAGE (chitchat):\n{user_line}"

    providers = _llm_provider_chain()
    if not providers:
        return ""

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=200,
        timeout_sec=12,
        max_attempts=2,
        temperature=0.35,
    )
    if not data:
        return ""
    raw = (data.get("response") or "").strip()
    if not raw:
        return ""
    plain_user = re.sub(r"\s+", " ", comb.lower().strip())
    plain_reply = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw).lower().strip())
    if plain_user and len(plain_user) >= 8 and (
        plain_user == plain_reply or plain_user in plain_reply
    ):
        log_reasoning("Chitchat AI reply rejected — echoed user message.")
        return ""
    log_reasoning(f"Chitchat AI reply: {(data.get('reasoning') or '')[:100]}")
    return finalize_customer_reply(_wrap_ack_html(raw), original_msg or msg_en, rl)


def build_contextual_ack_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
) -> str:
    """AI-first contextual ack; topic-aware template if LLM unavailable."""
    from services.translation_service import resolve_customer_reply_lang

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    ai_body = ai_conversational_ack_reply(
        original_msg, msg_en, conversation_context, reply_lang=rl, ctx=ctx
    )
    if ai_body:
        return ai_body
    topic = infer_recent_assistant_topic(conversation_context, ctx=ctx)
    body = template_ack_for_topic(topic, original_msg or msg_en, rl)
    if body:
        log_reasoning(f"Conversational ack template fallback (topic={topic}).")
    return body
