"""
Conversation memory: map short replies (ha / yes / explore) to what the assistant just offered.
"""
import re
from typing import Any, Optional

from utils.reasoning_log import log_reasoning

_AFFIRMATIVE_WORDS = frozenset(
    {
        "ha", "haa", "haan", "han", "yes", "yeah", "yep", "yup", "ok", "okay", "sure",
        "ji", "haanji", "theek", "thik", "achha", "accha", "bilkul", "pls", "please",
        "haa", "ho", "done", "go", "ahead",
    }
)

_NON_PRODUCT_PHRASES = (
    "yes i want to explore",
    "yes to explore",
    "want to explore",
    "i want to explore",
    "lets explore",
    "let's explore",
    "explore karna",
    "explore krna",
    "explore krna h",
    "explore karna h",
    "yes explore",
    "ha explore",
)

_DEALS_SHOW_MARKERS = (
    "deals dikhao",
    "deals dikha",
    "deals btao",
    "deals batao",
    "deals bta",
    "deals dikha do",
    "deal dikhao",
    "deal dikha",
    "aaj ke deals",
    "aaj ki deals",
    "aaj ki top",
    "today deals",
    "today's deals",
    "top deals",
    "top deal",
    "deals ki baat",
    "deal ki baat",
    "deals ki hi",
    "sirf deals",
    "bas deals",
    "only deals",
    "ha deals",
    "deals de",
    "deals do",
    "offers dikhao",
    "offer dikhao",
)


def is_deals_request_message(original_msg: str, msg_en: str = "") -> bool:
    """
    User wants today's deals / offers — NOT a product named 'deals'.
    Handles Hinglish: 'deals ki baat kr rha hu dikha do', 'aaj ki top wali deals'.
    """
    combined = re.sub(r"\s+", " ", f"{original_msg} {msg_en}".lower()).strip()
    if not combined:
        return False
    if not re.search(r"\bdeals?\b|\boffers?\b|\bdiscount", combined):
        return False
    if any(m in combined for m in _DEALS_SHOW_MARKERS):
        return True
    if re.search(
        r"\bdeals?\b.*\b(?:dikha|dikhao|dikhado|btao|batao|bta|show|de\s+do|dedo)\b",
        combined,
    ):
        return True
    if re.search(r"\b(?:dikha|dikhao|btao|batao|show)\b.*\bdeals?\b", combined):
        return True
    if re.search(r"\bdeals?\b.*\b(?:baat|bat)\b", combined):
        return True
    if re.search(r"\baaj\s+ki\b.*\bdeals?\b", combined):
        return True
    if re.search(r"\btop\s+wali\b.*\bdeals?\b", combined):
        return True
    if re.search(r"\b(?:today|todays|today's|aaj)\b.*\btop\b.*\bdeals?\b", combined):
        return True
    if re.search(r"\bdeals?\b.*\b(?:today|todays|aaj)\b", combined):
        return True
    if re.search(r"\b(?:ha|haan|han|yes)\b.*\bdeals?\b", combined) and re.search(
        r"\b(?:dikha|dikhao|btao|batao|baat|bat|kr\s+rha|kar\s+raha)\b", combined
    ):
        return True
    tokens = re.findall(r"[a-z]+", combined)
    if "deals" in tokens or "deal" in tokens:
        product_nouns = _has_product_noun_token(tokens)
        if not product_nouns and len(tokens) <= 14:
            if any(t in tokens for t in ("dikha", "dikhao", "dikhado", "btao", "batao", "baat", "bat")):
                return True
    return False


_LINK_ASK_MARKERS = (
    "konsi link",
    "kon si link",
    "link do",
    "link de",
    "link bhej",
    "link bhejo",
    "url do",
    "url de",
    "give link",
    "send link",
    "which link",
    "link kaha",
    "link kahan",
)


def is_non_product_search_phrase(text: str) -> bool:
    """Phrases that mention 'want/explore' but are not catalog product queries."""
    if not text:
        return False
    low = re.sub(r"\s+", " ", text.lower()).strip()
    try:
        from utils.helpers import (
            message_is_bot_capability_question,
            message_is_bot_search_complaint,
            message_needs_policy_answer,
            _text_has_past_order_complaint_context,
        )

        from utils.helpers import (
            message_is_bot_availability_chitchat,
            message_is_casual_farewell_or_closing,
            message_is_user_feedback_or_closing,
            _is_light_smalltalk_fast,
        )

        if message_is_bot_availability_chitchat(text):
            return True
        if message_is_user_feedback_or_closing(text) or message_is_casual_farewell_or_closing(
            text
        ):
            return True
        if _is_light_smalltalk_fast(text):
            return True
        if message_is_bot_capability_question(text) or message_is_bot_search_complaint(text):
            return True
        if message_needs_policy_answer(text) or _text_has_past_order_complaint_context(text):
            return True
    except ImportError:
        pass
    if any(p in low for p in _NON_PRODUCT_PHRASES):
        return True
    if any(
        x in low
        for x in (
            "search krke kyu", "product search krke", "tu search", "theek search",
            "jo bolu wo", "jo chahiye wo", "tere pass h kya", "galat order aaya",
            "search krke bta", "search krke btao", "tu search kr", "puchh rha kesa",
            "ese puchh rha", "bol rha hu ki", "hello bhai kesa", "hello kesa",
            "kaise ho", "kaise h", "kesa h", "kya haal", "how are you",
        )
    ):
        return True
    if re.search(r"\b(?:hello|hi|hey)\b", low) and re.search(
        r"\b(?:kesa|kaise|kese|kaisa|how)\b", low
    ):
        if not _has_product_noun_token(re.findall(r"[a-z]+", low)):
            return True
    tokens = re.findall(r"[a-z]+", low)
    if not tokens or len(tokens) > 9:
        return False
    if "explore" in tokens and not _has_product_noun_token(tokens):
        return True
    if len(tokens) <= 5 and tokens[0] in _AFFIRMATIVE_WORDS and "explore" in tokens:
        return True
    return False


def _has_product_noun_token(tokens: list[str]) -> bool:
    try:
        from services.welfog_api import _token_is_product_noun

        return any(_token_is_product_noun(t) for t in tokens)
    except Exception:
        return False


def _last_assistant_text(conversation_context: str) -> str:
    if not (conversation_context or "").strip():
        return ""
    tail = conversation_context[-4000:]
    parts = re.split(r"(?i)assistant\s*:", tail)
    return (parts[-1] if len(parts) > 1 else tail).lower()


def detect_pending_offer_from_text(assistant_text: str) -> Optional[str]:
    """
    What the bot last offered: explore_menu | deals | deals_or_search | categories
    """
    t = (assistant_text or "").lower()
    if not t:
        return None
    if "deals, categories" in t or "deals, categories, or" in t:
        return "explore_menu"
    if "today's deals" in t or "todays deals" in t or "today deals" in t or "aaj ke deals" in t:
        if "or do you already" in t or "something in mind to search" in t:
            return "deals_or_search"
        if "should i show" in t or "show" in t:
            return "deals"
    if "special deals" in t or "quick deals" in t or "browse deals" in t:
        return "deals"
    if "browse" in t and "categor" in t and "product search" not in t:
        return "categories"
    if "all categories" in t or "category list" in t:
        return "categories"
    if "privacy policy" in t or "privacy policy dikhao" in t or "welfog privacy" in t:
        if any(
            x in t
            for x in (
                "welfog ki",
                "welfog privacy",
                "privacy policy dikhao",
                "policy dikhao",
                "chahiye ho",
                "official policies",
                "kisi aur brand",
                "another brand",
            )
        ):
            return "welfog_privacy_policy"
    if any(
        x in t
        for x in (
            "only help with welfog",
            "sirf welfog",
            "another brand",
            "other company",
            "kisi aur brand",
        )
    ):
        return "welfog_privacy_policy"
    return None


def sync_pending_offer_from_conversation(ctx: dict, conversation_context: str) -> None:
    offer = detect_pending_offer_from_text(_last_assistant_text(conversation_context))
    if offer:
        ctx.setdefault("data", {})["pending_offer"] = offer


def pending_offer_from_greeting_key(reply_key: str) -> Optional[str]:
    key = (reply_key or "").lower()
    if key in ("greeting", "greeting_variant_2", "greeting_variant_3"):
        return "explore_menu"
    if key.startswith("warm_smalltalk"):
        return "explore_menu"
    return None


def _looks_like_welfog_policy_clarification_followup(
    original_msg: str, msg_en: str, conversation_context: str
) -> bool:
    """After bot declined another company's policy, user clarifies they meant Welfog."""
    from services.policy_scope import _user_explicitly_wants_welfog_policy

    if _user_explicitly_wants_welfog_policy(original_msg, conversation_context):
        return True
    tail = (conversation_context or "")[-3500:].lower()
    if not any(
        x in tail
        for x in (
            "only help with",
            "sirf welfog",
            "another brand",
            "other company",
            "kisi aur brand",
            "privacy policy dikhao",
            "welfog privacy",
        )
    ):
        return False
    combined = f"{original_msg} {msg_en}".lower()
    return any(
        x in combined
        for x in (
            "wahi puchh",
            "wahi puch",
            "ha wahi",
            "isi company",
            "yahi company",
            "welfog ki hi",
            "welfog ki",
            "bta de",
            "bata de",
            "btana",
            "dikha",
            "dikhao",
            "batao",
            "btao",
            "privacy",
            "policy",
            "privary",
        )
    )


def _looks_like_short_affirmative(original_msg: str, msg_en: str) -> bool:
    combined = f"{original_msg} {msg_en}".lower()
    tokens = re.findall(r"[a-z]+", combined)
    if not tokens or len(tokens) > 10:
        return False
    if _has_product_noun_token(tokens):
        return False
    if any(t in _AFFIRMATIVE_WORDS for t in tokens):
        return True
    if any(
        x in combined
        for x in (
            "bta do", "btao", "batao", "bata do", "bta de", "dikha", "dikhao", "dikhado",
            "show me", "tell me", "bata de", "bta do bhai",
        )
    ):
        if re.search(r"\b\d{2,}\b", combined):
            return False
        if re.search(r"\b(?:rs|₹|rupee|rating|star|stars|price|color|colour|rang|sku|brand)\b", combined):
            return False
        return True
    return False


def _user_asks_for_link(text: str) -> bool:
    low = f" {text.lower()} "
    return any(m in low for m in _LINK_ASK_MARKERS)


def _assistant_recently_showed_deals(conversation_context: str) -> bool:
    tail = (conversation_context or "")[-5000:].lower()
    return "deals of the day" in tail or "view deal" in tail or "🔥" in tail or "wf-product-rail" in tail


def classify_conversation_followup(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ctx: Optional[dict] = None,
) -> Optional[str]:
    """
    Direct routing for contextual replies.
    Returns: deals | explore_menu | categories | deals_link_clarify | None
    """
    combined = f"{original_msg} {msg_en}".lower()
    ctx = ctx or {}

    try:
        from utils.helpers import (
            _text_has_product_shopping_intent,
            extract_product_id,
            _text_is_product_id_lookup_context,
        )
        from services.opensearch_products import parse_product_filters_from_text

        if (
            _text_has_product_shopping_intent(combined)
            or extract_product_id(combined)
            or _text_is_product_id_lookup_context(combined)
        ):
            return None
        pf = parse_product_filters_from_text(f"{original_msg} {msg_en}")
        if any(
            pf.get(k) is not None
            for k in ("purchase_price_max", "purchase_price_min", "rating_min", "rating_max", "color", "size", "sku")
        ):
            return None
        if re.search(r"\b\d{6,}\b", combined):
            return None
    except ImportError:
        pass

    if is_deals_request_message(original_msg, msg_en):
        log_reasoning("Explicit deals/offers request — route to today's deals.")
        return "deals"

    offer = (ctx.get("data") or {}).get("pending_offer")
    if not offer:
        offer = detect_pending_offer_from_text(_last_assistant_text(conversation_context))

    if _user_asks_for_link(combined):
        if _assistant_recently_showed_deals(conversation_context) or offer in ("deals", "deals_or_search"):
            log_reasoning("Follow-up: user asked for deal link after deals carousel.")
            return "deals_link_clarify"

    if is_non_product_search_phrase(combined):
        log_reasoning("Follow-up: explore/menu reply (not product search).")
        return "explore_menu"

    if not _looks_like_short_affirmative(original_msg, msg_en):
        return None

    if offer == "deals":
        log_reasoning("Follow-up: affirmative → show today's deals.")
        return "deals"
    if offer == "categories":
        log_reasoning("Follow-up: affirmative → show categories.")
        return "categories"
    if offer == "explore_menu":
        log_reasoning("Follow-up: affirmative → explore menu choices.")
        return "explore_menu"
    if offer == "deals_or_search":
        if "deal" in combined or "offer" in combined:
            return "deals"
        if "categor" in combined:
            return "categories"
        if "search" in combined or "product" in combined:
            return None
        last = _last_assistant_text(conversation_context)
        if _looks_like_short_affirmative(original_msg, msg_en) and (
            "today's deals" in last or "today deals" in last or "special deals" in last
        ):
            return "deals"
        return "explore_menu"

    if offer == "welfog_privacy_policy" and _looks_like_short_affirmative(original_msg, msg_en):
        log_reasoning("Follow-up: affirmative → show Welfog privacy/policy from KB.")
        return "welfog_privacy_policy"

    if _looks_like_welfog_policy_clarification_followup(original_msg, msg_en, conversation_context):
        log_reasoning("Follow-up: user clarified they want Welfog policy in chat.")
        return "welfog_privacy_policy"

    last = _last_assistant_text(conversation_context)
    if _looks_like_short_affirmative(original_msg, msg_en):
        if "today's deals" in last or "today deals" in last or "aaj ke deals" in last:
            return "deals"
        if "deals, categories" in last:
            return "explore_menu"
    return None


def apply_conversation_followup_fixes(
    original_msg: str,
    msg_en: str,
    ai_data: dict,
    conversation_context: str,
    ctx: Optional[dict] = None,
) -> None:
    """Correct mis-routes on cached/fresh AI output using chat context."""
    if not ai_data or not ai_data.get("is_welfog_related", True):
        return
    if ai_data.get("intent") == "out_of_domain":
        return

    combined = f"{original_msg} {msg_en}"
    if ai_data.get("_ai_routed"):
        ch = (ai_data.get("data_channel") or "").strip().lower()
        locked = ch in ("live_api", "kb") or (ai_data.get("intent") or "") in (
            "wishlist",
            "order_history",
            "order",
            "refund",
            "payment",
            "seller",
            "pincode_check",
        )
        if locked and not is_deals_request_message(original_msg, msg_en):
            follow_only = classify_conversation_followup(
                original_msg, msg_en, conversation_context, ctx
            )
            if follow_only in ("deals", "categories") and ch == "catalog":
                pass
            elif locked:
                return

    if is_deals_request_message(original_msg, msg_en):
        ai_data["intent"] = "deals"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
        return

    follow = classify_conversation_followup(original_msg, msg_en, conversation_context, ctx)

    if is_non_product_search_phrase(combined):
        ai_data["intent"] = "general"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
        return

    sq = (ai_data.get("search_query") or "").strip().lower()
    if sq and is_non_product_search_phrase(sq):
        ai_data["search_query"] = ""
        if ai_data.get("intent") == "product":
            ai_data["intent"] = "general"
            ai_data["response"] = ""

    if not follow:
        return

    if follow == "deals":
        ai_data["intent"] = "deals"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
    elif follow == "categories":
        ai_data["intent"] = "categories"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
    elif follow in ("explore_menu", "deals_link_clarify"):
        ai_data["intent"] = "general"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
