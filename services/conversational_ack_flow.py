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
        _llm_classifier_provider_chain,
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

    providers = _llm_classifier_provider_chain()
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
        _llm_classifier_provider_chain,
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
    # Tiny context — chitchat must stay snappy.
    compact_ctx = _compact_conversation_context(conversation_context or "", 280)
    user_line = _trim_text_mid(comb, 120)

    system_prompt = f"""You are Welfog's shopping support assistant in live chat (e-commerce help).
User sent CASUAL chitchat (greeting, thanks, bye, how-are-you, free/busy, sun-na) in ANY language/style.

JSON only: {{"reasoning":"1 line","response":"reply"}}

Rules:
- Mirror user's language, script, slang/typos (Hinglish stays Hinglish).
- 1-2 short lines: (1) warm natural ack in their vibe, (2) gently offer Welfog help —
  shopping, orders, delivery, returns — invite what they need. Vary wording.
- You are a Welfog support assistant, not a personal friend inventing private history
  (no fake "we met yesterday" / only hanging out). Stay useful + human.
- Bye/thanks → warm ack + welcome back anytime for Welfog help (do not restart with stiff Hello).
- Never stock "Hello! Good to see you on Welfog…"; never Order ID / product dump / verbatim echo.
{language_reply_instruction(rl)}"""

    user_payload = ""
    if compact_ctx:
        user_payload += f"RECENT:\n{compact_ctx}\n\n"
    user_payload += f"LATEST:\n{user_line}"

    try:
        from services.llm_providers import get_fast_chitchat_provider_chain

        # Prefer Groq Instant when available — admin often pins OpenAI first for Brain.
        # Allow one fallback if Groq stalls (still within short timeout).
        cap = max(1, min(2, int(os.getenv("CHITCHAT_AI_MAX_PROVIDERS", "2") or "2")))
        providers = get_fast_chitchat_provider_chain(max_providers=cap)
    except ImportError:
        providers = _llm_classifier_provider_chain()[:1]
    if not providers:
        return ""

    try:
        timeout_sec = max(
            1.5, min(5.0, float(os.getenv("CHITCHAT_AI_TIMEOUT_SEC", "2.5") or 2.5))
        )
    except (TypeError, ValueError):
        timeout_sec = 2.5

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=140,
        timeout_sec=timeout_sec,
        max_attempts=1,
        temperature=0.45,
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


def ai_ood_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """Out-of-domain reply — AI only, mirrors user language/style. Fast path for production."""
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
    # Tiny context — OOD must finish in ~1s, not re-read a long thread.
    compact_ctx = _compact_conversation_context(conversation_context or "", 600)
    user_line = _trim_text_mid(comb, 200)

    system_prompt = f"""You are Welfog's shopping support assistant in live chat.

The user's message is OUT OF DOMAIN — not about Welfog shopping, orders, delivery, or support.

Return ONLY JSON: {{"reasoning":"1 line","response":"reply"}}

RULES:
- "response" MUST match the user's LATEST message language, script, and tone
  (Hinglish stays Hinglish; English stays English; Hindi script stays Hindi).
- 1–2 short sentences only: briefly nod to their topic in their words, then say you only
  help with Welfog products/orders/delivery/returns, and invite one short Welfog question.
- Do NOT answer the off-topic question with facts, advice, forecasts, or steps.
- Warm and human — never paste the same corporate template every time; vary with their topic.
{language_reply_instruction(rl)}
JSON only."""

    user_payload = ""
    if compact_ctx:
        user_payload += f"RECENT:\n{compact_ctx}\n\n"
    user_payload += f"LATEST (off-topic):\n{user_line}"

    try:
        from services.llm_providers import get_fast_chitchat_provider_chain

        cap = max(1, min(2, int(os.getenv("OOD_AI_MAX_PROVIDERS", "1") or "1")))
        providers = get_fast_chitchat_provider_chain(max_providers=cap)
    except ImportError:
        from services.ai_service import _llm_classifier_provider_chain

        providers = _llm_classifier_provider_chain()[:1]
    if not providers:
        return ""

    try:
        timeout_sec = max(
            1.5, min(5.0, float(os.getenv("OOD_AI_TIMEOUT_SEC", "2.5") or 2.5))
        )
    except (TypeError, ValueError):
        timeout_sec = 2.5

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=120,
        timeout_sec=timeout_sec,
        max_attempts=1,
        temperature=0.45,
    )
    if not data:
        return ""
    raw = (data.get("response") or "").strip()
    if not raw:
        return ""
    log_reasoning(f"OOD AI reply: {(data.get('reasoning') or '')[:100]}")
    return finalize_customer_reply(_wrap_ack_html(raw), original_msg or msg_en, rl)


def ai_harm_sensitive_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """Empathetic safety reply — AI first, harm template if LLM unavailable."""
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
        _llm_classifier_provider_chain,
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

    system_prompt = f"""You are Welfog's shopping support assistant. The user may be in emotional distress or mentioning self-harm.

Return ONLY JSON: {{"reasoning":"1 line","response":"reply"}}

RULES:
- "response" in the SAME language/script as the user's LATEST message (Hinglish = Roman Hindi+English).
- 2-4 short sentences: empathetic, caring, encourage talking to someone they trust NOW.
- Mention India helplines: iCall 9152987821, Vandrevala 1860-2662-345, emergency 112.
- Do NOT mention grievance officer, legal, seller registration, or product search.
- Light note you are Welfog shopping assistant if they need order help later.
{language_reply_instruction(rl)}
JSON only."""

    user_payload = ""
    if compact_ctx:
        user_payload += f"RECENT CONVERSATION:\n{compact_ctx}\n\n"
    user_payload += f"LATEST USER MESSAGE:\n{user_line}"

    providers = _llm_classifier_provider_chain()
    if not providers:
        return ""

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=220,
        timeout_sec=10,
        max_attempts=1,
        temperature=0.25,
    )
    raw = (data.get("response") or "").strip() if data else ""
    if raw:
        log_reasoning(f"Harm AI reply: {(data.get('reasoning') or '')[:100]}")
        return finalize_customer_reply(_wrap_ack_html(raw), original_msg or msg_en, rl)
    return ""


def ai_natural_scope_reply(
    scope: str,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> str:
    """AI-generated chitchat or OOD reply — never static KB templates."""
    scope_l = (scope or "").strip().lower()
    if scope_l == "harm_sensitive":
        return ai_harm_sensitive_reply(
            original_msg, msg_en, conversation_context, reply_lang=reply_lang
        )
    if scope_l == "out_of_domain":
        return ai_ood_reply(
            original_msg, msg_en, conversation_context, reply_lang=reply_lang
        )
    return ai_chitchat_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ctx=ctx,
    )


def resolve_brain_scope_customer_reply(
    brain_route: dict | None,
    scope: str,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> str:
    """
    Use brain scope_reply when present; otherwise one AI reply call (no templates).
    """
    from services.conversation_scope import finalize_scope_reply_html

    try:
        from services.brain_direct_dispatch import _scope_reply_is_placeholder
    except ImportError:

        def _scope_reply_is_placeholder(text: str) -> bool:  # type: ignore
            return not (text or "").strip()

    scope_reply = ""
    if isinstance(brain_route, dict):
        scope_reply = (brain_route.get("scope_reply") or "").strip()
    if scope_reply and not _scope_reply_is_placeholder(scope_reply):
        return finalize_scope_reply_html(
            scope_reply, original_msg, reply_lang=reply_lang
        )

    body = ai_natural_scope_reply(
        scope,
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ctx=ctx,
    )
    if body:
        log_reasoning(f"Brain scope AI reply ({scope}) — no template fallback.")
    return body or ""


def brain_locked_conversational_scope(brain_route: dict | None) -> str:
    """
    Return general_chitchat | out_of_domain | harm_sensitive when Brain already locked
    a non-commerce conversational turn — else empty.

    Uses Brain JSON only (no customer keyword lists). Never conflicts with locked
    product catalog / KB / live order / account-list turns.
    """
    if not isinstance(brain_route, dict):
        return ""
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(brain_route):
            return ""
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import (
            brain_route_indicates_product_catalog,
            brain_route_prefers_kb_answer,
        )

        if brain_route_indicates_product_catalog(brain_route):
            return ""
        if brain_route_prefers_kb_answer(brain_route):
            return ""
    except ImportError:
        pass
    channel = (brain_route.get("data_channel") or "").strip().lower()
    if channel == "catalog" and brain_route.get("run_catalog_search"):
        return ""
    if channel == "kb" and (brain_route.get("kb_keys") or []):
        return ""
    if brain_route.get("needs_order_id"):
        return ""
    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(brain_route):
            return ""
    except ImportError:
        pass

    intent = (brain_route.get("intent") or "").strip().lower()
    scope = (brain_route.get("conversation_scope") or "").strip().lower()
    if scope == "harm_sensitive":
        return "harm_sensitive"
    if intent == "out_of_domain" or scope == "out_of_domain":
        return "out_of_domain"
    if brain_route.get("is_welfog_related") is False and intent not in (
        "product",
        "order",
        "order_history",
        "wishlist",
        "refund",
        "payment",
        "seller",
        "pincode_check",
        "deals",
        "categories",
    ):
        return "out_of_domain"
    if scope == "general_chitchat":
        return "general_chitchat"
    return ""


def try_immediate_brain_scope_reply(
    brain_route: dict | None,
    *,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ctx: dict | None = None,
) -> str:
    """
    Production OOD/chitchat/harm path: Brain-lock → customer reply in one shot.

    Prefer Brain-authored scope_reply (already in routing LLM). Otherwise one short
    AI decline. Never probes OpenSearch / KB / order APIs.
    """
    scope = brain_locked_conversational_scope(brain_route)
    if not scope:
        return ""
    body = resolve_brain_scope_customer_reply(
        brain_route,
        scope,
        original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
        ctx=ctx,
    )
    if body and str(body).strip():
        log_reasoning(f"Immediate Brain scope reply ({scope}) — skip catalog/KB/order.")
        return str(body).strip()
    return ""


def resolve_natural_conversational_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
    ai_route: dict | None = None,
    *,
    check_off_topic: bool = False,
) -> str:
    """
    ChatGPT-style natural reply for greetings, thanks, closings, chitchat, and off-topic.
    One lightweight LLM call; KB templates only when LLM is unavailable.
    """
    from services.conversation_scope import SCOPE_OUT, build_scope_reply
    from services.translation_service import finalize_customer_reply, resolve_customer_reply_lang

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    if isinstance(ai_route, dict):
        scope_reply = (ai_route.get("scope_reply") or "").strip()
        if scope_reply:
            return finalize_customer_reply(
                f"<div style='color:#333;line-height:1.55;'>{scope_reply}</div>"
                if "<" not in scope_reply
                else scope_reply,
                original_msg or msg_en,
                rl,
            )

    if check_off_topic:
        try:
            from services.conversation_scope import ai_classify_scope_and_reply

            scope_dec = ai_classify_scope_and_reply(
                original_msg, msg_en, conversation_context, rl
            )
            if scope_dec and scope_dec.scope == SCOPE_OUT:
                body = build_scope_reply(
                    scope_dec, original_msg, msg_en, rl, prefer_llm=True
                )
                if body:
                    log_reasoning("Natural conversational reply — out-of-domain scope LLM.")
                    return body
        except ImportError:
            pass
        return ""

    body = ai_chitchat_reply(
        original_msg, msg_en, conversation_context, reply_lang=rl, ctx=ctx
    )
    if body:
        log_reasoning("Natural conversational reply — chitchat AI.")
        return body
    return ""


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
