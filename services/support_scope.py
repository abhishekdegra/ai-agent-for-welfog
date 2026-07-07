"""
Detect when the user wants help with another company/app (Amazon, Flipkart, …)
—not Welfog orders, products, or tracking.

Uses heuristics + Groq (no fixed list of all world companies).
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

from utils.reasoning_log import log_reasoning

SupportScope = Literal["welfog", "external", "unclear"]

# Shopping competitors — user wants THAT app's order/wishlist (not Welfog).
_SHOPPING_COMPETITOR_BRANDS = frozenset(
    {
        "amazon", "flipkart", "myntra", "meesho", "ajio", "snapdeal", "shopsy", "glowroad",
        "nykaa", "tatacliq", "jiomart", "bigbasket", "blinkit", "zepto",
        "ebay", "alibaba", "aliexpress", "walmart", "target", "shopify", "etsy",
    }
)

# Other apps (food, ride, social) — separate from e-commerce wishlist on Welfog.
_EXTERNAL_MARKETPLACE_BRANDS = _SHOPPING_COMPETITOR_BRANDS | frozenset(
    {
        "swiggy", "zomato", "uber", "ola", "rapido", "bookmyshow", "makemytrip", "irctc",
        "paytm", "phonepe", "instagram", "facebook", "whatsapp", "youtube", "netflix", "spotify",
        "agoda", "booking", "goibibo", "oyo", "microsoft", "meta",
    }
)

# Product/device brands — user buys ON Welfog; never treat "iphone ka cover" as external order.
_PRODUCT_BRAND_WORDS = frozenset(
    {
        "apple", "samsung", "reliance", "tata", "hp", "dell", "lenovo", "mi",
        "redmi", "iphone", "ipad", "oneplus", "oppo", "vivo", "realme", "nike", "adidas", "puma",
        "google",
    }
)

_KNOWN_EXTERNAL_BRANDS = _EXTERNAL_MARKETPLACE_BRANDS | _PRODUCT_BRAND_WORDS

_SUPPORT_TOPIC_WORDS = (
    "order", "orders", "track", "tracking", "delivery", "shipment", "parcel", "package",
    "refund", "return", "cancel", "payment", "product", "item", "buy", "purchase",
    "status", "kab aayega", "nahi aaya", "nhi aaya", "complaint", "customer care",
    "wishlist", "wish list", "liked", "saved list", "heart", "favourite", "favorite",
    "cart", "basket", "account", "login",
)


def _normalize_scope_text(text: str) -> str:
    from utils.helpers import _normalize_order_chat_text, _normalize_welfog_typos

    return _normalize_welfog_typos(_normalize_order_chat_text(text or ""))


def _has_support_topic(text: str) -> bool:
    tl = f" {_normalize_scope_text(text)} "
    if any(w in tl for w in _SUPPORT_TOPIC_WORDS):
        return True
    if re.search(r"\b(track|trck|deliver|ship|refund|return|cancel|service)\b", tl):
        return True
    return False


def _token_is_welfog(tok: str) -> bool:
    t = (tok or "").lower()
    return t in ("welfog", "wlefog", "welefog", "welfogcom", "welfogapp")


def _names_shopping_competitor(text: str) -> bool:
    """True only when a shopping competitor (Amazon, Flipkart, …) is named in the message."""
    tl = f" {_normalize_scope_text(text)} "
    for brand in _SHOPPING_COMPETITOR_BRANDS:
        if _brand_in_message(tl, brand):
            return True
    return False


def _is_welfog_in_chat_account_request(combined: str) -> bool:
    """
    User wants their own wishlist / order history / purchases in THIS chat — no competitor named.
    e.g. mereko meri wishlist batana, meri wishlist dikhao, show my wishlist.
    """
    if _names_shopping_competitor(combined):
        return False
    try:
        from utils.helpers import (
            _text_asks_wishlist,
            _text_wants_order_history_list_in_chat,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )

        if _text_asks_wishlist(combined) or message_is_wishlist_like_request(combined):
            return True
        if _text_wants_order_history_list_in_chat(combined) or message_is_past_purchase_list_request(
            combined
        ):
            return True
    except ImportError:
        pass
    tl = f" {_normalize_scope_text(combined)} "
    if "wishlist" not in tl and "wish list" not in tl:
        if not any(x in tl for x in ("order history", "my orders", "meri order", "mere order")):
            return False
    possessive = (
        " meri ", " mere ", " mereko ", " mera ", " my ", " apni ", " apna ", " mujhe ",
        " hamari ", " hamare ", " our ", " show me ",
    )
    if any(p in tl for p in possessive):
        return True
    return False


def _brand_in_message(tl: str, brand: str) -> bool:
    b = brand.lower().strip()
    if not b:
        return False
    return re.search(rf"\b{re.escape(b)}\b", tl) is not None


def _welfog_owns_topic_in_message(tl: str, topic: str) -> bool:
    """True when user ties the topic explicitly to Welfog (not another app)."""
    if not re.search(rf"\b(?:welfog|wlefog|welefog)\b", tl):
        return False
    if topic == "wishlist" and re.search(
        r"\b(?:welfog|wlefog|welefog)\b.{0,40}\b(?:wishlist|wish\s*list)\b", tl
    ):
        return True
    if topic == "order" and re.search(
        r"\b(?:welfog|wlefog|welefog)\b.{0,40}\b(?:order|orders)\b", tl
    ):
        return True
    return False


def _external_marketplace_named_with_feature(tl: str) -> bool:
    """
    User named another shop/app and asked for account feature there (wishlist, order, cart…).
    Never map to Welfog — politely decline.
    """
    feature_markers = (
        "wishlist", "wish list", "liked", "saved", "heart", "favourite", "favorite",
        "order", "orders", "track", "tracking", "delivery", "refund", "return",
        "cart", "basket", "account", "login", "checkout",
    )
    if not any(m in tl for m in feature_markers):
        return False

    for brand in _EXTERNAL_MARKETPLACE_BRANDS:
        if not _brand_in_message(tl, brand):
            continue
        if "wishlist" in tl or "wish list" in tl:
            if _welfog_owns_topic_in_message(tl, "wishlist"):
                continue
            return True
        possessive = (
            f" {brand} ki ",
            f" {brand} ka ",
            f" {brand} ke ",
            f" {brand} wali ",
            f" {brand} wala ",
            f" {brand} se ",
            f" {brand} pe ",
            f" {brand} par ",
            f"from {brand}",
            f"on {brand}",
        )
        if any(p in tl for p in possessive):
            return True
        if any(m in tl for m in feature_markers[2:]):  # order, track, etc.
            return True
    return False


def _heuristic_external_company_support(tl: str) -> bool:
    """Another company's order/delivery/track/wishlist — not price comparison on Welfog."""
    try:
        from utils.helpers import _text_has_explicit_how_to_place_order

        if _text_has_explicit_how_to_place_order(tl):
            return False
    except ImportError:
        pass

    if _external_marketplace_named_with_feature(tl):
        return True

    if _token_is_welfog("welfog") and "welfog" in tl:
        # explicit Welfog order → stay in domain even if amazon mentioned for compare
        if re.search(
            r"\b(?:welfog|wlefog|welefog)\s+(?:se|pe|par|ka|ki|me|mein)\s+order",
            tl,
        ) or re.search(r"\border\s+(?:on|from|at)\s+(?:welfog|wlefog)", tl):
            return False
        if re.search(r"\b(?:welfog|wlefog|welefog)\s+(?:pe|par|pr)\b", tl) and "order" in tl:
            return False

    for brand in _EXTERNAL_MARKETPLACE_BRANDS:
        if not _brand_in_message(tl, brand):
            continue
        if any(
            p in tl
            for p in (
                f" {brand} se order",
                f" {brand} pe order",
                f" {brand} par order",
                f" {brand} ka order",
                f" {brand} ki order",
                f" {brand} se order",
                f"order {brand} se",
                f"from {brand}",
                f"on {brand}",
                f" {brand} se kiya",
                f" {brand} pe kiya",
                f" {brand} se liya",
                f" {brand} se buy",
                f" {brand} ki delivery",
                f" {brand} track",
                f"track {brand}",
                f" {brand} se aaya",
                f" {brand} se nahi",
            )
        ):
            return True
        if "order" in tl and any(
            x in tl for x in (f" {brand} se", f" {brand} pe", f" {brand} par", f" {brand} ka order")
        ):
            return True

    # Product brands: only external if they ordered ON that brand's store — not "iphone ka cover".
    for brand in _PRODUCT_BRAND_WORDS:
        if not _brand_in_message(tl, brand):
            continue
        if any(
            p in tl
            for p in (
                f" {brand} se order",
                f" {brand} pe order",
                f" {brand} par order",
                f" {brand} ka order",
                f" {brand} ki order",
                f"order {brand} se",
                f"from {brand}",
                f"on {brand}",
                f" {brand} se kiya",
                f" {brand} pe kiya",
                f" {brand} track",
                f"track {brand}",
            )
        ):
            return True

    # Generic: "<company> se order kiya" (Hinglish) — not Welfog
    for m in re.finditer(
        r"\b([a-z][a-z0-9]{2,18})\s+se\s+(?:order|buy|purchase|liya|kiya)\b",
        tl,
    ):
        name = m.group(1).lower()
        if name not in _KNOWN_EXTERNAL_BRANDS and name not in (
            "maine", "mene", "humne", "tune", "tumne", "aapne", "usne", "iske", "uske",
        ):
            if not _token_is_welfog(name) and name not in ("yahan", "waha", "idhar", "udhar"):
                return True

    for m in re.finditer(
        r"\b(?:order|buy|purchase)\s+(?:from|on|at|through)\s+([a-z][a-z0-9]{2,18})\b",
        tl,
    ):
        name = m.group(1).lower()
        if not _token_is_welfog(name):
            return True

    return False


def _llm_support_scope(user_msg: str, conversation_context: str = "") -> Optional[SupportScope]:
    try:
        from services.llm_providers import llm_json_chat_completion

        conv_block = ""
        if (conversation_context or "").strip():
            conv_block = (
                "\nRECENT CHAT:\n"
                f"{conversation_context.strip()[-3500:]}\n"
            )

        prompt = f"""You classify messages in a Welfog-only e-commerce support chatbot.

{conv_block}
LATEST USER MESSAGE:
\"\"\"{user_msg.strip()}\"\"\"

Question: Should Welfog help with this request, or is the user asking about another company/app's order, product, delivery, or policy?

Rules:
- Welfog ONLY: orders placed on Welfog, Welfog products, Welfog delivery PIN, Welfog refunds, Welfog order ID tracking.
- EXTERNAL (about_welfog=false): help with ANY other company/app/shop/person's order, tracking, refund, delivery, wishlist, cart — when THAT entity is NAMED (Amazon, Meesho, a friend's order, any marketplace you recognize).
- EXTERNAL: "amazon ki wishlist dikhao", "flipkart order nahi aaya", "amazon delivery 302012 pe", "meesho dega kya delivery".
- Do NOT require a fixed brand list — infer from context whether the user wants Welfog or another entity.
- WELFOG (about_welfog=true): "meri wishlist dikhao", "mereko meri wishlist batana", "my wishlist", "meri order history" — user on Welfog chat asking for THEIR saved items; NO competitor named. NEVER external.
- Price compare mentioning another site WITHOUT asking bot to act on that site's order → still Welfog if they want Welfog products.
- Brand names (Samsung, iPhone, Redmi, Nike…) when user wants to BUY on Welfog (cover, shirt, phone accessory) → about_welfog=true — NOT external.
- Typos: wlefog/welefog = Welfog.

Return ONLY JSON: {{"about_welfog": true/false, "reason": "few words"}}"""

        data = llm_json_chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=120,
            timeout_sec=14,
            max_attempts=2,
        )
        if not data:
            return None
        if data.get("about_welfog") is True:
            log_reasoning(f"Support scope LLM → Welfog ({data.get('reason', '')})")
            return "welfog"
        if data.get("about_welfog") is False:
            log_reasoning(f"Support scope LLM → external ({data.get('reason', '')})")
            return "external"
    except Exception as e:
        log_reasoning(f"Support scope LLM skipped: {e}")
    return None


def resolve_support_request_scope(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> SupportScope:
    try:
        from utils.helpers import (
            message_asks_other_company_social_media,
            message_asks_welfog_social_media,
        )

        combined_sm = f"{original_msg} {msg_en}".strip()
        if message_asks_other_company_social_media(combined_sm, conversation_context):
            return "external"
        if message_asks_welfog_social_media(combined_sm, conversation_context):
            return "welfog"
    except ImportError:
        pass
    from utils.helpers import _text_mentions_welfog_brand, message_is_conversation_reset_command

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return "unclear"

    if message_is_conversation_reset_command(original_msg) or message_is_conversation_reset_command(
        msg_en
    ):
        return "welfog"

    try:
        from utils.helpers import (
            _text_has_explicit_how_to_place_order,
            _text_has_order_placement_intent,
            _text_has_product_shopping_intent,
            _text_mentions_welfog_brand,
        )

        if _text_has_explicit_how_to_place_order(combined) or _text_has_order_placement_intent(
            combined
        ):
            return "welfog"
        if _text_mentions_welfog_brand(combined) and _text_has_explicit_how_to_place_order(
            combined
        ):
            return "welfog"
        if _text_has_product_shopping_intent(combined):
            return "welfog"
    except ImportError:
        pass

    tl = f" {_normalize_scope_text(combined)} "

    if _is_welfog_in_chat_account_request(combined):
        return "welfog"

    if _text_mentions_welfog_brand(combined) and re.search(
        r"\b(?:welfog|wlefog|welefog)\b.{0,30}\b(?:order|track|delivery|refund)\b",
        tl,
    ):
        return "welfog"

    if _heuristic_external_company_support(tl):
        return "external"

    if _names_shopping_competitor(combined) and _has_support_topic(combined):
        return "external"

    if not _has_support_topic(combined):
        return "unclear"

    llm = _llm_support_scope(combined, conversation_context)
    if llm == "external" and not _names_shopping_competitor(combined):
        if "wishlist" in tl or "wish list" in tl or _is_welfog_in_chat_account_request(combined):
            log_reasoning("Support scope: ignore LLM external — possessive wishlist/orders on Welfog chat.")
            return "welfog"
    if llm:
        return llm

    return "welfog"


def support_request_is_for_welfog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    scope = resolve_support_request_scope(original_msg, msg_en, conversation_context)
    if scope == "external":
        return False
    return True


def message_mentions_other_company_support(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    route_decision=None,
) -> bool:
    """
    True when user wants another company's order/wishlist — politely decline.
    Never true for meri/mere/my wishlist with no competitor named (trust AI route).
    """
    combined = f"{original_msg} {msg_en}".strip()
    if _is_welfog_in_chat_account_request(combined):
        return False

    if _names_shopping_competitor(combined) and _has_support_topic(combined):
        if not support_request_is_for_welfog(
            original_msg, msg_en, conversation_context
        ):
            log_reasoning(
                "Other-company support topic — decline (overrides pincode route)."
            )
            return True

    if _names_shopping_competitor(combined) and isinstance(ai_route, dict):
        try:
            from utils.helpers import _text_mentions_welfog_brand

            welfog_in_msg = _text_mentions_welfog_brand(combined)
        except ImportError:
            welfog_in_msg = bool(re.search(r"\bwelfog\b", combined, re.I))
        if not welfog_in_msg and (
            (ai_route.get("intent") or "").strip().lower() == "pincode_check"
        ):
            log_reasoning(
                "Competitor named on delivery question — decline Welfog pincode API."
            )
            return True

    if ai_route and ai_route.get("is_welfog_related", True):
        intent = (ai_route.get("intent") or "").strip().lower()
        channel = (ai_route.get("data_channel") or "").strip().lower()
        if intent in (
            "wishlist",
            "order_history",
            "order",
            "refund",
            "payment",
            "seller",
            "pincode_check",
            "deals",
            "categories",
        ) and channel in ("live_api", "catalog", "kb"):
            if not _names_shopping_competitor(combined):
                log_reasoning(f"Trust AI intent={intent} — skip external decline.")
                return False

    if route_decision is not None:
        handler = (getattr(route_decision, "handler", None) or "").strip()
        if getattr(route_decision, "is_welfog_related", True) and handler in (
            "wishlist_api",
            "order_ai_flow",
            "product_ai_flow",
            "order_details_api",
            "order_tracking_api",
            "order_id_ai_flow",
        ):
            if not _names_shopping_competitor(combined):
                log_reasoning(f"Trust route handler={handler} — skip external decline.")
                return False

    if not _names_shopping_competitor(combined) and not _heuristic_external_company_support(
        f" {_normalize_scope_text(combined)} "
    ):
        return False

    return not support_request_is_for_welfog(original_msg, msg_en, conversation_context)


def try_external_scope_fast_decline(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[str]:
    """
    Early polite decline when user asks about another company/person — before order/pincode fast paths.
    Uses heuristics + LLM scope (no fixed list of every company).
    """
    if message_mentions_other_company_support(
        original_msg, msg_en, conversation_context
    ):
        log_reasoning("External scope fast decline — not Welfog support.")
        return build_other_company_support_decline(original_msg, reply_lang=reply_lang)
    return None


def build_other_company_social_decline(original_msg: str, reply_lang: str = "") -> str:
    from services.kb_service import sysmsg
    from services.translation_service import finalize_customer_reply, resolve_customer_reply_lang

    rl = resolve_customer_reply_lang(original_msg, reply_lang)
    quote = _short_quote_for_decline(original_msg, max_len=80)
    if quote:
        if rl == "hinglish":
            body = (
                f"<div style='color:#333;line-height:1.55;'>"
                f"Sorry — <b>\"{quote}\"</b> ke liye main sirf <b>Welfog</b> ke official social media links de sakta hoon. "
                f"Kisi aur company, school/college, ya personal profile ke Instagram/LinkedIn/Facebook ke liye "
                f"unki hi official website ya app check karo.<br><br>"
                f"Agar <b>Welfog</b> ka Insta/LinkedIn/YouTube chahiye ho to seedha puchho — jaise "
                f"<b>Welfog ki Instagram link do</b> ya <b>saare social media links</b>."
                f"</div>"
            )
        else:
            body = (
                f"<div style='color:#333;line-height:1.55;'>"
                f"Sorry — I can only share <b>Welfog's</b> official social media links, not "
                f"<b>\"{quote}\"</b>. For another company, school, college, or someone's personal profile, "
                f"please check their official website or app.<br><br>"
                f"For Welfog, ask directly — e.g. <b>Welfog Instagram link</b> or <b>all social media links</b>."
                f"</div>"
            )
    elif rl == "hinglish":
        body = sysmsg("other_company_social_decline_hinglish") or sysmsg("other_company_social_decline")
    else:
        body = sysmsg("other_company_social_decline") or sysmsg("off_topic_polite")
    return finalize_customer_reply(body or "", original_msg, rl) or ""


def _short_quote_for_decline(text: str, *, max_len: int = 90) -> str:
    from html import escape as html_escape

    q = re.sub(r"\s+", " ", (text or "").strip())
    if not q:
        return ""
    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0] + "…"
    return html_escape(q)


def _decline_body_with_quote(quote: str, *, hinglish: bool) -> str:
    """Customer-facing decline — quoted topic; base text localized per language later."""
    if hinglish:
        return (
            f"<div style='color:#333;line-height:1.55;'>"
            f"Sorry — <b>\"{quote}\"</b> par main madad nahi kar sakta. "
            f"Main sirf <b>Welfog</b> se related queries me help karta hoon "
            f"(products, orders, delivery, refund, policies).<br><br>"
            f"Amazon / Flipkart ya kisi aur app ki wishlist, order ya tracking ke liye "
            f"unki hi website ya app use karo.<br><br>"
            f"Agar <b>Welfog</b> ki wishlist ya order chahiye ho to bina doosri company ke naam ke "
            f"puchho — jaise <b>meri wishlist dikhao</b> ya <b>mera order track karo</b>."
            f"</div>"
        )
    return (
        f"<div style='color:#333;line-height:1.55;'>"
        f"Sorry — I cannot help with <b>\"{quote}\"</b>. "
        f"I only assist with <b>Welfog</b>-related questions "
        f"(products, orders, delivery, refunds, policies).<br><br>"
        f"For another app's or website's wishlist, orders, or tracking, "
        f"please use that platform's support.<br><br>"
        f"If you meant <b>Welfog</b>, ask without naming another store — e.g. "
        f"<b>show my wishlist</b> or paste your <b>Welfog Order ID</b>."
        f"</div>"
    )


def build_other_company_support_decline(original_msg: str, reply_lang: str = "") -> str:
    """
    Polite decline for non-Welfog / other-company requests.
    Reply language matches the customer: en, hinglish, hi, pa, ta, te, mr, gu, bn, ur, kn, ml.
    """
    from services.kb_service import sysmsg
    from services.translation_service import finalize_customer_reply, resolve_customer_reply_lang

    rl = resolve_customer_reply_lang(original_msg, reply_lang)
    quote = _short_quote_for_decline(original_msg)
    if quote:
        body = _decline_body_with_quote(quote, hinglish=(rl == "hinglish"))
    elif rl == "hinglish":
        body = sysmsg("off_topic_other_company_order_hinglish") or sysmsg("off_topic_polite")
    else:
        body = sysmsg("off_topic_other_company_order") or sysmsg("off_topic_polite")
    return finalize_customer_reply(body or "", original_msg, rl) or ""
