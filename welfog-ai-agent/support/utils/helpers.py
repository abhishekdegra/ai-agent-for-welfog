import difflib
import random
import re
import threading

from utils.multilingual_intent import (
    multilingual_how_to_order_history,
    multilingual_order_history_match,
    multilingual_order_tracking_match,
    native_order_history_request,
    native_order_tracking_request,
)
from utils.reasoning_log import log_reasoning

_pin_order_guard = threading.local()
_pincode_intent_guard = threading.local()
_pincode_light_guard = threading.local()
_order_tracking_guard = threading.local()
_extract_oid_guard = threading.local()
_refund_lookup_guard = threading.local()


def _strip_html_for_context(text: str, max_len: int = 280) -> str:
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) > max_len:
        plain = plain[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return plain


def _format_conversation_for_llm(msgs: list, max_turns: int = 12) -> str:
    """Compact transcript for routing / answering (pronouns, follow-ups)."""
    if not msgs:
        return ""
    tail = msgs[-max_turns:] if len(msgs) > max_turns else msgs
    lines = []
    for m in tail:
        role = "User" if m.get("sender") == "user" else "Assistant"
        content = (m.get("message") or "").strip()
        if m.get("sender") == "user":
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > 240:
                content = content[:240].rsplit(" ", 1)[0] + "…"
        else:
            content = _strip_html_for_context(content, max_len=280)
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _conversation_cache_suffix(msgs: list) -> str:
    if not msgs:
        return "0"
    blob = "|".join(f"{m.get('sender')}:{(m.get('message') or '')[:160]}" for m in msgs[-8:])
    return str(abs(hash(blob)) % (10**12))


def build_retrieval_query(msg_en: str, conv_block: str, original_msg: str) -> str:
    """Combine recent chat with the latest question so embeddings match follow-ups."""
    base = (msg_en or "").strip() or (original_msg or "").strip()
    if not conv_block.strip():
        return base
    tail = conv_block[-1200:] if len(conv_block) > 1200 else conv_block
    return f"{tail}\n\nCurrent question: {(original_msg or '').strip()}".strip()


user_contexts = {}

def reset_context(ctx):
    ctx["intent"] = None
    ctx["awaiting"] = None
    ctx["data"] = {}
    ctx["last"] = None
    ctx["order_id"] = None


_ORDER_PENDING_ACTIONS = frozenset(
    {
        "track",
        "order_invoice",
        "order_details",
        "refund_status",
        "payment",
    }
)


def ctx_has_order_thread_lock(ctx) -> bool:
    """True while collecting Order ID for a locked refund/track/invoice/details turn."""
    if not isinstance(ctx, dict):
        return False
    if ctx.get("awaiting") == "order_id":
        return True
    pending = ((ctx.get("data") or {}).get("pending_action") or "").strip().lower()
    return pending in _ORDER_PENDING_ACTIONS


def reset_context_unless_order_pending(ctx) -> None:
    """Clear session ctx but keep order-id collection thread (pending_action / ai_route)."""
    if ctx_has_order_thread_lock(ctx):
        return
    reset_context(ctx)


def message_user_switches_order_scope(text: str) -> bool:
    """User wants a different order — do not reuse session order_id or stale intent."""
    tl = f" {(text or '').lower()} "
    if re.search(r"\b(?:dusra|dusre|different|another|other|new)\s+order\b", tl):
        return True
    if "dusra" in tl and ("order" in tl or " id" in tl):
        return True
    if ("yeh nhi" in tl or "ye nhi" in tl) and ("dusra" in tl or "order" in tl):
        return True
    if "not this" in tl and "order" in tl:
        return True
    return False


def clear_order_session_for_new_lookup(ctx: dict | None) -> None:
    """Drop stale order id / locked route when user starts a fresh order question."""
    if not isinstance(ctx, dict):
        return
    ctx["order_id"] = None
    data = ctx.get("data")
    if not isinstance(data, dict):
        return
    for key in ("pending_action", "topic_mode", "ai_route"):
        data.pop(key, None)


def _normalize_repeated_letters(token: str) -> str:
    return re.sub(r"(.)\1{1,}", r"\1", token or "")


def _is_short_pure_greeting(msg: str) -> bool:
    """Hi/hiii/hello — regex only; never call embeddings (chat fast path must not hang)."""
    raw = (msg or "").strip().lower()
    if not raw or len(raw) > 40:
        return False
    letters_only = re.sub(r"[^a-z]+", "", raw)
    if 1 <= len(letters_only) <= 24:
        collapsed = re.sub(r"(.)\1+", r"\1", letters_only)
        if collapsed in ("hi", "hey", "hello", "helo", "hiya", "namaste", "hie", "hiy"):
            return True
        for ref in ("hi", "hey", "hello", "helo", "namaste", "hiya"):
            if difflib.SequenceMatcher(None, collapsed, ref).ratio() >= 0.72:
                return True
    words = re.findall(r"[a-z]+", raw)
    if len(words) == 1 and words[0] in {
        "hi", "hii", "hiii", "hey", "heyy", "hello", "helo", "hiya", "namaste", "namaskar",
    }:
        return True
    return False


_FAST_SHOPPING_WORDS = frozenset(
    {
        "shirt", "shirts", "tshirt", "tshirts", "pant", "pants", "jeans", "dress", "dresses",
        "shoes", "kurta", "kurti", "cover", "case", "iphone", "mobile", "phone", "laptop",
        "watch", "samsung", "white", "black", "red", "blue", "product", "products",
        "bottle", "rice", "wheat", "umbrella", "bat", "ball", "sofa", "bed",
    }
)


def _is_welfog_about_fast_path(msg: str) -> bool:
    """Welfog company/platform questions — regex only (never block on embeddings or welfog_api)."""
    raw = (msg or "").strip().lower()
    if not raw:
        return False
    if not (_text_mentions_welfog_brand(raw) or "welfog" in raw or "wlefog" in raw):
        return False
    tl = f" {raw} "
    if any(
        x in tl
        for x in (
            " about ", " tell me ", " explain ", " batao ", " bata ", " btao ", " samjha ",
            " what is ", " who is ", " kya hai ", " kon hai ", " kaun hai ", " company ",
            " platform ", " story ", " kahani ", " kahaani ", " founder ", " ceo ",
            " kya karta ", " kya karti ", " kya karte ", " kya kaam ",
            " kya h ", " krti kya ", " krta kya ", " krti kya h ", " krta kya h ",
            " ke bare ", " ke baare ", " ke barre ", " bare me ", " baare me ", " barre me ",
            " baar me ", " bttoo ", " bttto ", " btado ", " btana ", " bataoo ",
        )
    ):
        return True
    if "something" in tl and "about" in tl:
        return True
    return False


def _message_looks_like_shopping_query(msg: str) -> bool:
    """Cheap product-turn detector — no welfog_api import, no embeddings (chat must not hang)."""
    raw = (msg or "").strip().lower()
    if not raw:
        return False
    tl = f" {raw} "
    # Account / support turns — never catalog product search (avoids OpenSearch hang on "show my orders").
    if any(
        x in tl
        for x in (
            "wishlist", "order history", "purchase history", "my orders", "meri order", "mere order",
            "track order", "order track", "tracking", "refund", "return order", "pincode",
            "delivery status", "order status",
        )
    ):
        return False
    shop_verbs = (
        "show ", "show me", "show my", "show mw", "looking for", " buy ", " need ",
        " want ", "dikhao", "dikha ", " dikha", " dikhao", "chahiye", "chahie", "chhaie", "chaahiye",
        "milega", "milta", "milti", "milegi", "search ", "dikana", "dikhana",
    )
    if any(v in tl for v in shop_verbs):
        if any(w in tl for w in _FAST_SHOPPING_WORDS):
            return True
        if any(x in tl for x in ("show me", "show my", "show mw", "show the", "looking for")):
            return True
    if re.search(r"\b(?:dikha|dikhao|dikhana|dikhao)\b", raw):
        if any(w in tl for w in _FAST_SHOPPING_WORDS):
            return True
    # Product noun + need/availability (Hinglish typos) — no heavy intent graph.
    if any(w in tl for w in _FAST_SHOPPING_WORDS):
        if any(
            x in tl
            for x in (
                "chahiye", "chahie", "chhaie", "chaahiye", "chiye", "chie",
                "milega", "milta", "milegi", "milti", "available", "availability",
                "kya ", "hai kya", "he kya", "h kya", "mil jayega", "mil jayegi",
            )
        ):
            return True
        if ("welfog" in tl or "wlefog" in tl) and any(
            x in tl for x in ("per", "par", "pe ", "par ", "on welfog", "welfog per", "welfog par")
        ):
            return True
    return False


def _looks_like_greeting_message(msg: str) -> bool:
    raw = (msg or "").strip().lower()
    if not raw:
        return False
    if _is_welfog_about_fast_path(raw):
        return False

    # Collapsed noisy typing: "hheyyyy", "heello", "hhiehhhh" -> compare to hi/hey/hello
    letters_only = re.sub(r"[^a-z]+", "", raw)
    if 1 <= len(letters_only) <= 22:
        collapsed = re.sub(r"(.)\1+", r"\1", letters_only)
        for ref in ("hi", "hey", "hello", "helo", "namaste", "hiya"):
            if difflib.SequenceMatcher(None, collapsed, ref).ratio() >= 0.72:
                return True
        if collapsed in ("hieh", "hie", "hiy", "hii", "hei", "heyy", "heyyy", "helo", "heelo"):
            return True

    words = re.findall(r"[a-z]+", raw)
    if not words:
        return False

    greeting_set = {
        "hi", "hii", "hey", "heyy", "hello", "helo", "hallo", "hiya", "namaste",
        "namaskar", "suno", "sun", "bro", "brother", "ram", "radhe", "radhey",
        "oye", "haan", "han", "hnn", "hn",
        # broader multilingual/romanized greetings
        "vanakkam", "namaskaram", "namaskara", "namaskaaram", "salaam", "assalam",
        "adaab", "sat", "sri", "akal", "sassriakal", "kem", "cho", "pranam",
        "jai", "shree", "shri", "ramram",
    }
    filler_set = {
        "bol", "bolo", "bolna", "bolta", "bolti", "ra", "raha", "rahi", "rahe",
        "hu", "hun", "ho", "hai", "na", "bhai", "bhia", "bhisaa", "bhaiya", "bhaiyya",
        "yaar", "yar", "yrr", "yrrr", "yr", "aap", "kaise", "kya", "bas", "please",
        "haal", "hal", "theek", "thik", "achha", "acha", "sab", "sabhi", "achhi",
        "sun", "suno", "uff", "are", "arre", "abe", "dost", "dosto", "dostt",
    }
    casual_opener_phrases = (
        "sun na", "sunna", "suno na", "haan sun", "oye sun", "bol na", "bolo na",
        "kya scene", "kya haal", "ky haal", "haan bolo", "bol bhai",
    )
    # Short openers without catalog verbs — skip heavy shopping-intent graph (was blocking /chat).
    _short_opener = len(raw) <= 56 and not any(
        m in f" {raw} "
        for m in (
            "show", "buy", "order", "track", "wishlist", "search", "dikhao", "dikha",
            "chahiye", "milega", "product", "refund", "pincode",
        )
    )
    if not _short_opener:
        return False

    for phrase in casual_opener_phrases:
        if phrase in raw and len(words) <= 8:
            return True
    query_blockers = {
        "order", "track", "tracking", "delivery", "payment", "refund", "return",
        "cancel", "buy", "need", "want", "milega", "available", "price", "product",
    }

    def _is_near_greeting_token(token: str) -> bool:
        if token in greeting_set:
            return True
        for g in greeting_set:
            if difflib.SequenceMatcher(None, token, g).ratio() >= 0.72:
                return True
        return False

    normalized = [_normalize_repeated_letters(w) for w in words]
    if len(normalized) == 1 and normalized[0] in filler_set:
        return True
    has_near_greeting = any(_is_near_greeting_token(w) for w in normalized)
    if not has_near_greeting:
        # Handle heavily broken/combined typo forms: "satshriakal", "helloooji", "ramraam", etc.
        compact = "".join(normalized)
        compact_refs = (
            "hello", "helo", "heyo", "hii", "hi", "ramram", "radheradhe",
            "namaste", "namaskar", "vanakkam", "assalam", "salaam", "adaab",
            "satsriakal", "satshriakal", "kemcho", "pranam",
        )
        if not any(difflib.SequenceMatcher(None, compact, ref).ratio() >= 0.64 for ref in compact_refs):
            return False
    # Token-level fuzzy hit for mixed/broken spellings in short greetings.
    fuzzy_refs = (
        "hello", "hi", "hey", "namaste", "namaskar", "vanakkam", "salaam",
        "assalam", "adaab", "sat", "sri", "akal", "ram", "radhe", "pranam",
    )
    fuzzy_hits = 0
    for tok in normalized:
        if any(difflib.SequenceMatcher(None, tok, ref).ratio() >= 0.74 for ref in fuzzy_refs):
            fuzzy_hits += 1
    if (not has_near_greeting) and fuzzy_hits == 0 and len(normalized) <= 5:
        return False
    if len(normalized) > 7:
        return False

    if any(w in query_blockers for w in normalized) and any(_is_near_greeting_token(w) for w in normalized):
        # If the message contains a query keyword, avoid misclassifying it as just a greeting.
        return False

    for w in normalized:
        if _is_near_greeting_token(w):
            continue
        if w in filler_set:
            continue
        if w.startswith("bh") and len(w) <= 7:
            continue
        return False
    return True


_CULTURAL_GREETING_SNIPPETS = (
    "ram ram",
    "radhe radhe",
    "radhey radhey",
    "jai shri ram",
    "jai shree ram",
    "jai siya ram",
    "jai mata di",
    "har har mahadev",
    "om namah",
    "namaskar ji",
    "namaste ji",
    "pranam",
    "pranaam",
    "sat sri akal",
    "sat shri akal",
    "adaab",
    "adab",
    "salaam",
    "assalam",
)

_CASUAL_WELLBEING_SNIPPETS = (
    "sab badiya",
    "sab badhiya",
    "sab badia",
    "sab theek",
    "sab thik",
    "sab achha",
    "sab accha",
    "sab sahi",
    "sab set",
    "sab mast",
    "or to sab",
    "or sab",
    "aur to sab",
    "aur sab",
    "sab bdiya",
    "sab bdia",
    "bdiya",
    "badhiya",
    "badiya",
    "theek hai sab",
    "kaise ho",
    "kaise ho aap",
    "kaisen ho",
    "kese ho",
    "kese ho aap",
    "kesa h",
    "kesa hai",
    "kese h",
    "kaisa h",
    "kaisa hai",
    "bhai kesa",
    "bhai kese",
    "hello bhai",
    "hi bhai",
    "hey bhai",
    "kyse ho",
    "kya haal",
    "kya hal",
    "kya haal hai",
    "kaisa chal",
    "kaise chal",
    "chal raha hai",
    "how are you",
    "how r u",
    "how are u",
    "hows it going",
    "how's it going",
    "what's up",
    "whats up",
    "all good",
    "hope you're",
    "hope you are",
    "miss you",
    "take care",
    "good morning",
    "good evening",
    "good night",
    "good afternoon",
    "thank you",
    "thanku",
    "shukriya",
    "dhanyavaad",
    "dhanyawad",
    "kya kar rha",
    "kya kar rhe",
    "kya kar rahi",
    "kya kr rha",
    "kya kr rhe",
    "kya chal rha",
    "kya chal raha",
    "what are you doing",
    "whatcha doing",
    "what u doing",
    "bye bye",
    "byeee",
    "byyy",
    "alvida",
    "see you",
    "take care bye",
    "dhanyawad",
    "dhanyavad",
    "help nhi chahiye",
    "help ni chahiye",
    "koi help nhi",
    "koi help ni",
    "darling thank",
    "thank you darling",
    "ok thank",
    "okay thank",
    "bol rha hu thank",
    "bol raha hu thank",
)


def message_is_bot_availability_chitchat(text: str) -> bool:
    """
    'Are you free/busy right now?' — casual talk to the bot, NOT a product named 'free'.
    """
    if not (text or "").strip():
        return False
    tl = f" {_normalize_conversational_text(text)} "
    if len(tl) > 100:
        return False
    if re.search(
        r"\b(?:free|fre)\s*(?:h|ho|hai|hain|he)\b",
        tl,
    ) and re.search(r"\b(?:ky|kya|k|tu|tum|aap|abhi|now|busy)\b", tl):
        return True
    if re.search(r"\b(?:busy|available|online|offline)\s*(?:h|ho|hai|hain)\b", tl):
        if re.search(r"\b(?:tu|tum|aap|bot|abhi|now)\b", tl):
            return True
    if re.search(r"\btime\s+h(?:ai|e)\b", tl) and re.search(
        r"\b(?:tu|tum|aap|abhi|free|busy)\b", tl
    ):
        return True
    return False


def message_is_casual_farewell_or_closing(text: str) -> bool:
    """Bye / no more help needed / polite sign-off — not shopping."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_conversational_text(text)} "
    if re.search(r"\b(?:bye|byee|byyy|alvida|see\s+ya|see\s+you)\b", tl):
        return True
    if any(
        x in tl
        for x in (
            "help nhi chahiye", "help ni chahiye", "koi help nhi", "koi help ni",
            "help nahi chahiye", "bas itna hi", "bas ho gaya", "ho gaya bas",
            "chl by", "chl bye", "theek h chl", "thik h chl", "chal bye",
        )
    ):
        return True
    if ("dhanyawad" in tl or "dhanyavad" in tl or "shukriya" in tl) and any(
        x in tl for x in ("help nhi", "help ni", "help nahi", "chahiye nhi", "chahiye ni")
    ):
        return True
    return False


def _is_light_smalltalk_fast(msg: str, msg_en: str = "") -> bool:
    """Wellbeing / thanks / cultural opener — snippet-only (no product embeddings)."""
    raw = (msg or "").strip().lower()
    extra = (msg_en or "").strip().lower()
    sample = re.sub(r"\s+", " ", f"{raw} {extra}").strip()
    if not sample or len(sample) > 160:
        return False
    if _light_smalltalk_blocker_hit(sample):
        return False
    if any(x.lower() in sample for x in _NATIVE_SCRIPT_GREETING_SNIPPETS):
        return True
    if any(x.lower() in sample for x in _NATIVE_SCRIPT_THANKS_SNIPPETS):
        return True
    for c in _CULTURAL_GREETING_SNIPPETS:
        if c in sample:
            return True
    for c in _CASUAL_WELLBEING_SNIPPETS:
        if c in sample:
            return True
    if re.search(r"\b(thanks|thx|thankyou|thanku)\b", sample):
        return True
    if re.search(r"\b(sup|wassup)\b", sample):
        return True
    return False


_NATIVE_SCRIPT_GREETING_SNIPPETS = (
    "नमस्ते", "नमस्कार", "राम राम", "राधे राधे", "प्रणाम",
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਸਤ ਸ਼੍ਰੀ ਅਕਾਲ", "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "ਨਮਸਤੇ",
    "રામ રામ", "નમસ્તે",
    "வணக்கம்", "నమస్తే", "ನಮಸ್ತೆ", "നമസ്കാരം", "নমস্কার",
    "السلام", "السلام علیکم", "السلام علیكم",
)

_NATIVE_SCRIPT_THANKS_SNIPPETS = (
    "धन्यवाद", "शुक्रिया", "बहुत धन्यवाद",
    "ਧੰਨਵਾਦ", "આભાર", "શુક્રિયા",
    "நன்றி", "ధన్యవాదాలు", "ಧನ್ಯವಾದಗಳು", "നന്ദി", "ধন্যবাদ",
    "شکریہ", "مهربانی",
)

_SMALLTALK_BLOCKER_PHRASES = (
    "where is my order",
    "where's my order",
    "mera order",
    "order track",
    "track order",
    "order id",
    "order status",
    "order cancel",
    "cancel order",
    "return order",
    "refund order",
)

_SMALLTALK_BLOCKER_WORDS = (
    "order",
    "refund",
    "track",
    "tracking",
    "payment",
    "delivery",
    "cancel",
    "return",
    "shipment",
    "courier",
    "pincode",
    "warranty",
    "complaint",
    "invoice",
    "seller",
    "charge",
    "charges",
    "fee",
    "fees",
    "login",
    "problem",
    "diqqat",
    "dikkat",
    "service",
    "welfog",
)


def _text_asks_welfog_fees_or_charges(text: str) -> bool:
    """User asks about Welfog service/platform/COD fees — payment KB, not greeting."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(text)} "
    markers = (
        "service charge", "service charges", "service fee", "service fees",
        "handling fee", "cod fee", "cod charge", "platform fee", "delivery charge",
        "charges ka", "charge ka", "fees ka", "fee ka",
    )
    if any(m in tl for m in markers):
        return True
    if ("charge" in tl or "fee" in tl or "fees" in tl) and any(
        x in tl
        for x in ("kya", "ky ", "kitna", "kitne", "scene", "bta", "bata", "tell", "what", "how much", "btana")
    ):
        if _text_mentions_welfog_brand(tl) or "checkout" in tl or "payment" in tl:
            return True
    return False


def _message_switches_topic_out_of_short_video_thread(text: str) -> bool:
    """New subject in chat — do not keep forcing short-video KB (e.g. customer care number)."""
    if _text_asks_customer_care_contact(text):
        return True
    tl = f" {_normalize_welfog_typos(text)} "
    switch_markers = (
        "customer support", "customer care", "cust care", "helpline", "phone number",
        "mobile number", "contact number", "support number", "call center", "call centre",
        "support email", "email ded", "email de", "order track", "track order", "refund",
        "delivery", "pincode", "product dikha", "buy ", "kharid", "payment", "invoice",
    )
    return any(m in tl for m in switch_markers)


def _short_video_thread_still_about_rules(text: str) -> bool:
    """Follow-up inside a short-video thread — only when still asking about video/rules/age."""
    if _message_switches_topic_out_of_short_video_thread(text):
        return False
    tl = f" {_normalize_welfog_typos(text)} "
    return any(
        x in tl
        for x in (
            "short video", "shorts", "reel", "video content", "video rule", "content rule",
            "age rule", "age ", "umar", "18 year", "minor", "guardian", "parent consent",
            "guideline", "policy", "policies", "rule", "bound", "restriction", "asci",
            "misleading", "promotional", "supplier", "hate speech", "copyright",
            "konsi", "konsa", "kaunsi", "bta de", "bata de", "btao", "batana",
            "boundation", "foundation", "allowed", "prohibit", "ha bta", "haan bta",
        )
    )


def _text_asks_short_video_content_rules(text: str, conversation_context: str = "") -> bool:
    """Short video / reels / shorts rules on Welfog — terms + seller KB, not generic AI."""
    turn = (text or "").strip()
    if not turn or _message_switches_topic_out_of_short_video_thread(turn):
        return False
    tl = f" {_normalize_welfog_typos(turn)} "
    topic_markers = (
        "short video", "short videos", "shorts", "short video content",
        "reel", "reels", "video content", "video banane", "video bana",
        "promotional video", "supplier promotional", "video upload",
        "content banana", "content banane",
    )
    ask_markers = (
        "rule", "policy", "guideline", "bound", "restriction", "limit",
        "kya", "ky ", "konsi", "konsa", "kaunsi", "kaunse", "allowed", "prohibit",
        "bta", "bata", "btana", "batana", "tell", "explain", "mana",
        "boundation", "foundation",
    )
    if _conversation_in_short_video_rules_flow(conversation_context):
        return _short_video_thread_still_about_rules(turn)
    if not any(m in tl for m in topic_markers):
        return False
    if any(m in tl for m in ask_markers):
        return True
    if "short video" in tl or "shorts" in tl or "reel" in tl:
        return True
    return False


def _conversation_in_short_video_rules_flow(conversation_context: str) -> bool:
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4000:].lower()
    markers = (
        "short video", "short videos", "shorts", "reel", "reels",
        "video content", "promotional video", "supplier promotional",
        "asci", "content rule", "video rule", "guidelines aur policies",
        "boundation", "boundaries",
    )
    return any(m in tail for m in markers)


def _conversation_in_seller_support_flow(conversation_context: str) -> bool:
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-3500:].lower()
    markers = (
        "seller login", "seller kaise bane", "seller panel", "seller account",
        "seller portal", "sell on welfog", "seller login problems",
        "become a seller", "supplier login", "vendor login",
    )
    return any(m in tail for m in markers)


def _text_has_seller_login_problem_intent(text: str, conversation_context: str = "") -> bool:
    """Seller panel login / OTP issues — not new seller registration."""
    combined = f"{text or ''} {(conversation_context or '')[-1500:]}".strip()
    tl = f" {_normalize_welfog_typos(combined)} "
    has_seller = (
        "seller" in tl
        or "supplier" in tl
        or "vendor" in tl
        or _conversation_in_seller_support_flow(conversation_context)
    )
    has_login = any(
        x in tl
        for x in (
            "seller login", "seller panel", "seller portal", "supplier login",
            "vendor login", "seller account login",
        )
    ) or ("login" in tl and has_seller)
    has_problem = any(
        x in tl
        for x in (
            "problem", "problm", "issue", "error", "fail", "nahi ho", "nhi ho", "ni ho",
            "nahi hora", "nhi hora", "ni hora", "login nahi", "login nhi", "otp",
            "sign in", "log in", "login me", "login mein", "diqqat", "dikkat", "dikat",
            "stuck", "kya karu", "kya kru", "solve", "help",
        )
    )
    if has_seller and has_login:
        return True
    if has_seller and has_login and has_problem:
        return True
    if "seller login" in tl and has_problem:
        return True
    return False


def _user_seller_issue_still_unresolved(text: str, conversation_context: str = "") -> bool:
    if not _conversation_in_seller_support_flow(conversation_context):
        return False
    tl = f" {(text or '').lower()} "
    return any(
        x in tl
        for x in (
            "fir bhi", "phir bhi", "still not", "still no", "nahi hora", "nhi hora",
            "solve nahi", "solve ni", "nahi ho raha", "nhi ho raha", "steps follow",
            "step follow", "sab kar liya", "sare step", "saare step", "kar liye",
            "kuchh nahi", "kuch nahi", "no use", "not working", "nhi hora kuch",
        )
    )


def _user_complains_bot_gave_wrong_topic(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    if not any(
        x in tl
        for x in ("kyu bta", "kyu bol", "why are you", "galat", "wrong topic", "relevant nahi", "kyu bta rha")
    ):
        return False
    return any(
        x in tl
        for x in ("seller", "login", "charge", "service", "payment", "refund", "delivery")
    )


def _text_has_concrete_welfog_support_question(text: str) -> bool:
    """Real Welfog support/policy/fee/seller question — never warm greeting."""
    if not (text or "").strip():
        return False
    if _text_asks_welfog_fees_or_charges(text):
        return True
    if _text_asks_short_video_content_rules(text):
        return True
    if _text_has_seller_login_problem_intent(text):
        return True
    if message_is_seller_on_welfog_request(text):
        return True
    if message_is_knowledge_information_request(text):
        return True
    if message_needs_policy_answer(text):
        return True
    if _text_asks_customer_care_contact(text):
        return True
    if (
        _text_is_pincode_serviceability_question_light(text)
        or _text_is_delivery_serviceability_hypothetical(text)
        or _message_asks_to_recheck_submitted_pincode(text)
    ):
        return True
    if _light_smalltalk_blocker_hit(f" {(text or '').lower()} "):
        return True
    tl = f" {_normalize_welfog_typos(text)} "
    if _text_mentions_welfog_brand(tl) and any(
        x in tl
        for x in (
            "kya", "ky ", "kaise", "how", "bta", "bata", "tell", "explain",
            "charge", "fee", "login", "seller", "policy", "refund", "payment",
        )
    ):
        return True
    return False


def _light_smalltalk_blocker_hit(sample: str) -> bool:
    """True if message is clearly shopping/support, not idle chat."""
    for p in _SMALLTALK_BLOCKER_PHRASES:
        if p in sample:
            return True
    for w in _SMALLTALK_BLOCKER_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", sample):
            return True
    for p in ("dikhao", "dikha", "dikho", "dikhado", "milega", "milta", "chahiye", "dikhaana"):
        if p in sample:
            return True
    return False


def _message_submits_or_corrects_order_id(text: str) -> bool:
    """User pasted or corrected an order id (incl. 'asli id 26051410', 'galat type')."""
    raw = _normalize_order_chat_text(text or "")
    if _bare_order_id_token_from_msg(raw):
        return True
    tl = f" {raw.lower()} "
    if re.search(r"\b[0-9]{4,20}\b", raw) and any(
        x in tl
        for x in (
            "id", "order", "asli", "sahi", "galat", "galt", "wrong", "correct", "sory", "sorry",
            "type ho", "mistake", "ye le", "yeh le", "ye h", "yeh h", "real id", "actual",
        )
    ):
        return True
    return bool(
        re.search(
            r"\b(?:asli|sahi|real|actual|correct)\s+(?:order\s*)?id\b",
            tl,
        )
        or re.search(r"\bgalt(?:i|a)?\s+(?:type|id|order)\b", tl)
    )


def _conversation_in_order_tracking_flow(conversation_context: str) -> bool:
    """Recent chat was about order tracking / order id — never drop to greeting."""
    if not (conversation_context or "").strip():
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False
    if _conversation_bot_offered_order_id_or_tracking(conversation_context):
        return True
    tail = (conversation_context or "")[-4500:].lower()
    markers = (
        "live tracking",
        "order track",
        "track kaise",
        "order id",
        "paste kar do",
        "paste your",
        "live status",
        "welfog_track",
        "aapke order ka",
        "order status",
        "checked id",
        "tracking nahi mil",
        "ask_order_id",
        "order_track_not_found",
    )
    return any(m in tail for m in markers)


def _normalize_conversational_text(text: str) -> str:
    """Loosen typos/spacing for thanks, praise, and casual closings."""
    t = _normalize_welfog_typos(text or "")
    t = re.sub(r"\s+", " ", t.lower()).strip()
    # Broken/spaced thanks: "than k you", "th anks", "thank u"
    t = re.sub(r"\bthan\s*k\s*you\b", "thank you", t)
    t = re.sub(r"\bthank\s*u\b", "thank you", t)
    t = re.sub(r"\bth+\s*ank\s*you\b", "thank you", t)
    t = re.sub(r"\bth+\s*anks\b", "thanks", t)
    for typo, fix in (
        ("dhanyawad", "dhanyavad"),
        ("dhanyawaad", "dhanyavad"),
        ("dhanyvaad", "dhanyavad"),
        ("shukria", "shukriya"),
        ("shukriya", "shukriya"),
        ("khus hua", "khush hua"),
        ("khus hu", "khush hu"),
        ("acha laga", "accha laga"),
        ("achha laga", "accha laga"),
        ("pasand aaya", "pasand aaya"),
        ("pasand aya", "pasand aaya"),
    ):
        t = t.replace(typo, fix)
    return t


def message_is_user_confused_or_rephrasing_bot(text: str, conversation_context: str = "") -> bool:
    """
    User is correcting the bot or re-stating their question — NOT thanks/praise.
    e.g. 'kya bol rha h tu', 'me kya puchh rha hu', 'are yeh kya h me to ese bol rha hu'.
    """
    if not (text or "").strip():
        return False
    if message_clarifies_wishlist_not_order_history(text):
        return True
    if message_denies_wishlist_wants_order_history(text):
        return True
    tl = f" {_normalize_conversational_text(text)} "
    if "wishlist" in tl and any(
        x in tl
        for x in (
            "nhi bol", "nahi bol", "ni bol", "nhi puchh", "nahi puchh", "ni puchh",
            "galat", "wrong", "history nhi", "history nahi", "history ka nhi",
            "me bol rha", "main bol rha", "bol raha hu", "bol rha hu",
        )
    ):
        return True
    confusion = (
        "kya bol rha", "kya bol raha", "kya bol rahi", "kya bol ra",
        "tu kya bol", "tum kya bol", "aap kya bol",
        "me kya puchh", "main kya puchh", "mujhe kya puchh", "me kya puch",
        "kya puchh rha hu", "kya puch rha hu", "kya puchha", "kya pucha",
        "mera sawal", "mera question", "my question", "what i asked",
        "yeh kya h", "ye kya h", "yeh kya hai", "ye kya hai",
        "are yeh kya", "arre yeh kya", "yeh kya bhej", "ye kya bhej",
        "galat jawab", "galat reply", "wrong reply", "wrong answer",
        "samajh nahi", "smjh nahi", "nahi samjha", "nhi samjha", "samjha nahi",
        "ese bol rha", "aise bol rha", "ese bol raha", "aise bol raha",
        "me to ese", "main to ese", "me to aise", "main to aise",
        "kya bol rha h tu", "kya bol raha h tu",
        "search krke kyu", "product search krke", "search kyu de", "theek search nhi",
        "sahi search nhi", "galat order aaya", "wrong product aaya",
    )
    if any(c in tl for c in confusion):
        return True
    if "?" in text and any(x in tl for x in ("kya bol", "kya puchh", "yeh kya", "ye kya", "galat")):
        return True
    return False


def _message_asks_to_recheck_submitted_pincode(text: str) -> bool:
    """User says they already sent PIN — wants live delivery check."""
    if not (text or "").strip():
        return False
    tl = f" {(text or '').lower()} "
    if not any(x in tl for x in ("pincode", "pin code", "pin ", "pin-")):
        return False
    return any(
        x in tl
        for x in (
            "bheja", "bheji", "bheje", "de di", "de diya", "diya", "diye", "di thi",
            "check kr", "check kar", "check kro", "check krke", "check karke",
            "btana", "bata", "btao", "batao", "bta de", "bata de", "wo check",
            "already sent", "already gave", "mene to", "maine to",
        )
    )


def _naive_six_digit_pin_from_text(text: str) -> str:
    """Regex-only PIN — no order-id disambiguation (avoids helper recursion)."""
    if not (text or "").strip():
        return ""
    m = re.search(r"\b([1-9]\d{5})\b", text)
    return m.group(1) if m else ""


def message_has_live_pincode_check_intent(
    text: str, conversation_context: str = "", msg_en: str = ""
) -> bool:
    """User wants a real delivery/PIN check — must not get warm-feedback shortcut."""
    if getattr(_pincode_intent_guard, "active", False):
        return False
    _pincode_intent_guard.active = True
    try:
        return _message_has_live_pincode_check_intent_impl(
            text, conversation_context, msg_en
        )
    finally:
        _pincode_intent_guard.active = False


def _message_has_live_pincode_check_intent_impl(
    text: str, conversation_context: str = "", msg_en: str = ""
) -> bool:
    comb = f"{text or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    tl_early = f" {comb.lower()} "
    if re.search(r"\bsim\b", tl_early) and re.search(r"\bpin\b", tl_early):
        if not re.search(r"\b(?:pincode|pin\s*code)\b", tl_early):
            if any(
                x in tl_early
                for x in (
                    "iphone", "mobile", "samsung", "product", "cover", "case",
                    "chahiye", "dikha", "dikhao", "ejector", "tray", "tool",
                )
            ):
                return False
    if _message_asks_to_recheck_submitted_pincode(comb):
        return True
    if _text_is_delivery_serviceability_hypothetical(comb):
        return True
    pin = _naive_six_digit_pin_from_text(comb)
    if not pin:
        if _text_is_undelivered_order_complaint(comb):
            return False
        if _text_is_pincode_serviceability_question_light(comb):
            return True
        return False
    if _text_has_delivery_or_order_area_intent(comb):
        return True
    if _text_is_pincode_serviceability_question_light(comb):
        return True
    tl = f" {comb.lower()} "
    if re.search(r"\b(?:pincode|pin code|pin)\b", tl) and any(
        x in tl for x in ("check kr", "check kar", "bta", "bata", "service", "delivery")
    ):
        return True
    return any(
        x in tl
        for x in (
            "check kr", "check kar", "check kro", "check krke", "check karke",
            "bta de", "bata de", "btao", "btana", "batao", "milega", "milegi",
            "service", "delivery", " dega", "degi", " ho jayeg", " aa jayeg",
            " pe ", " par ", " pr ", " per ",
        )
    )


def message_is_user_feedback_or_closing(text: str) -> bool:
    """
    User is thanking / complimenting / closing — NOT asking to track or send Order ID.
    Handles negation: 'order id ki ni bool rha, response dekh ke mja aaya'.
    """
    if not (text or "").strip():
        return False
    if (
        _text_has_light_order_tracking_markers(text)
        or _text_is_delivery_serviceability_hypothetical(text)
        or _message_asks_to_recheck_submitted_pincode(text)
        or (
            _naive_six_digit_pin_from_text(text)
            and _text_has_explicit_pincode_subject(text)
        )
    ):
        return False
    if message_is_user_confused_or_rephrasing_bot(text):
        return False
    tl = f" {_normalize_conversational_text(text)} "
    satisfaction = (
        "mja aaya", "maza aaya", "maza aa gaya", "maza aa gya", "maja aaya", "maja aa gaya", "maja aa gya",
        "majaa aaya", "mja aa", "maze aaye", "mja aya",
        "achha laga", "accha laga", "acha laga", "badhiya laga", "bdiya laga", "badiya laga", "sundar laga",
        "theek h ", "thik h ", "theek hai", "thik hai", "sahi h ", "sahi hai",
        "dekh ke mja", "dekh ke maza", "dekh ke maja", "response dekh", "reply dekh",
        "tera response", "teri response", "apka response", "tumhara response",
        "good response", "nice response", "loved it", "bahut achha", "bohot achha",
        "shukriya", "thank you", "thanks ", "thanku", "thankyou", "thx", "dhanyavad", "dhanyawad",
        "khush hua", "khush hu", "khus hua", "khus hu", "pasand aaya", "pasand aya",
        "bol rha hu", "bol raha hu", "bol rahi hu", "welcome bol", "dhanyawad de",
        "thank you bol", "thanks bol", "bol rha thank", "bol raha thank",
        "darling thank", "thank you darling", "ok thank", "okay thank",
        "are thank", "thank you bol rha", "thanks bol rha",
    )
    if any(s in tl for s in satisfaction):
        return True
    if re.search(r"\b(?:thank|thanks|shukriya|dhanyawad|dhanyavad)\b", tl) and re.search(
        r"\b(?:bol|bolo|bol\s+rha|bol\s+raha)\b", tl
    ):
        return True
    if any(x in (text or "") for x in _NATIVE_SCRIPT_THANKS_SNIPPETS):
        return True
    negation = (
        "ni bool", "nahi bol", "nahin bol", "ni bol", "nhi bol", "nahi bol raha", "ni bol raha",
        "bool nahi", "bol nahi", "bol ni", "bol rha ni", "bol raha nahi", "bol rhi nahi",
        "order id ki ni", "order id nahi", "id ki ni", "id nahi bol", "id ni bol",
        "track ki ni", "tracking ki ni", "order ki ni", "order nahi bol",
    )
    if any(n in tl for n in negation):
        return True
    if re.search(r"\b(?:ni|nahi|nahin|nhi)\s+bol", tl) and (
        "order" in tl or " id " in tl or "response" in tl or "reply" in tl
    ):
        return True
    return False


_CONVERSATIONAL_GENERAL_TALK_GUARD = threading.local()


def message_is_conversational_general_talk(
    original_msg: str, msg_en: str = "", conversation_context: str = ""
) -> bool:
    """
    Pure thanks/praise/general talk — no substantive Welfog question in this turn.
    Covers broken spellings and multilingual gratitude without KB lookup.
    """
    if getattr(_CONVERSATIONAL_GENERAL_TALK_GUARD, "active", False):
        return False
    _CONVERSATIONAL_GENERAL_TALK_GUARD.active = True
    try:
        return _message_is_conversational_general_talk_body(
            original_msg, msg_en, conversation_context
        )
    finally:
        _CONVERSATIONAL_GENERAL_TALK_GUARD.active = False


def _message_is_conversational_general_talk_body(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return False
    try:
        from services.chitchat_resolver import turn_is_chitchat_not_shopping

        if turn_is_chitchat_not_shopping(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=False,
        ):
            return True
    except ImportError:
        pass
    if message_is_user_confused_or_rephrasing_bot(combined, conversation_context):
        return False
    # Thanks/praise/farewell before greeting — avoids 'thank you' → hello template.
    if message_is_user_feedback_or_closing(combined):
        return True
    if message_is_casual_farewell_or_closing(combined):
        return True
    if message_is_bot_availability_chitchat(combined):
        return True
    if should_send_warm_greeting_reply(original_msg, msg_en, conversation_context):
        return True
    if message_is_assistant_identity_question(combined):
        return True
    if _should_bypass_warm_greeting_fast_path(combined.lower()):
        return False
    tl = f" {_normalize_conversational_text(combined)} "
    praise_only = (
        "acha laga welfog", "accha laga welfog", "welfog use krke", "welfog use karke",
        "welfog acha", "welfog accha", "welfog badhiya", "welfog badiya",
        "maza aaya welfog", "mja aaya welfog", "pasand aaya welfog",
        "khus hua", "khush hua", "khus hu", "khush hu", "isliye bol raha", "isliye bol ra",
        "bas itna hi bol", "sirf thank", "sirf thanks", "sirf shukriya", "sirf dhanyavad",
        "me to thank", "main to thank", "me to shukriya", "main to shukriya",
        "me to dhanyavad", "main to dhanyavad", "thank you bol", "thanks bol",
        "dhanyawad de", "dhanyavad de", "yeh kya bhej", "yeh kya h bhai",
    )
    if any(p in tl for p in praise_only):
        if _text_has_concrete_welfog_support_question(combined):
            return False
        return True
    if _looks_like_light_smalltalk(original_msg, msg_en) and not _text_has_concrete_welfog_support_question(
        combined
    ):
        return True
    return False


_CONVERSATIONAL_AI_REASONING_MARKERS = (
    "thank", "thanks", "thank you", "gratitude", "greeting", "greets", "greet",
    "compliment", "praise", "satisfied", "appreciation", "feedback", "acknowledg",
    "general talk", "small talk", "smalltalk", "closing", "welcoming", "pleasant",
    "dhanyav", "shukriya", "grateful", "expressing thanks", "user thanks",
    "thanking", "positive feedback", "enjoyed", "happy with",
)


def ai_route_indicates_conversational_talk(route_data: dict | None) -> bool:
    """When Groq reasoning says thanks/praise/general talk but wrongly picks KB."""
    if not route_data or not isinstance(route_data, dict):
        return False
    intent = (route_data.get("intent") or "").strip().lower()
    if intent not in ("general", ""):
        return False
    if route_data.get("needs_order_id"):
        return False
    if (route_data.get("search_query") or "").strip():
        return False
    reasoning = (route_data.get("reasoning") or "").lower()
    return any(m in reasoning for m in _CONVERSATIONAL_AI_REASONING_MARKERS)


def should_use_warm_conversational_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Heuristic + AI-reasoning: thanks/praise/general talk should never hit dynamic_kb."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if message_has_live_pincode_check_intent(
        original_msg, conversation_context, msg_en
    ):
        return False
    if resolve_navigation_help_topic(comb, conversation_context, ai_route=ai_route):
        return False
    if message_is_user_confused_or_rephrasing_bot(comb, conversation_context):
        return False
    if isinstance(ai_route, dict) and (ai_route.get("intent") or "").strip().lower() in (
        "pincode_check",
        "order",
        "order_history",
        "refund",
        "payment",
        "product",
    ):
        return False
    if message_is_conversational_general_talk(original_msg, msg_en, conversation_context):
        return True
    return ai_route_indicates_conversational_talk(ai_route)


def turn_skips_order_micro_classifiers(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """
    Greetings, chitchat, thanks — skip order/refund specialist LLMs after main router.
    """
    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(
            ai_route,
            original_msg,
            msg_en,
            conversation_context,
            allow_llm=True,
        ):
            return True
    except ImportError:
        pass
    if should_use_warm_conversational_reply(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return True
    if not isinstance(ai_route, dict):
        return False
    scope = (ai_route.get("conversation_scope") or "").strip().lower()
    if scope in ("general_chitchat", "harm_sensitive"):
        return True
    mk = (ai_route.get("meta_kind") or "none").strip().lower()
    if mk in ("conversational", "assistant_intro", "hostile", "bot_latency", "topic_denial"):
        return True
    rh = (ai_route.get("route_handler") or "").strip().lower()
    if rh == "warm_feedback":
        return True
    return False


def should_send_warm_feedback_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """After a helpful turn (order list, products, etc.) — acknowledge praise, not Order ID."""
    return should_use_warm_conversational_reply(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )


def build_warm_feedback_reply(
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
    ai_route: dict | None = None,
) -> str:
    from services.conversational_ack_flow import (
        ai_chitchat_reply,
        build_contextual_ack_reply,
    )
    from services.kb_service import sysmsg
    from services.translation_service import (
        finalize_customer_reply,
        is_hinglish_message,
        resolve_customer_reply_lang,
    )

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    scope_reply = ""
    if isinstance(ai_route, dict):
        scope_reply = (ai_route.get("scope_reply") or "").strip()
    if scope_reply:
        low_sr = scope_reply.lower()
        if low_sr in ("warm", "warm reply", "warm natural reply", "natural reply"):
            scope_reply = ""
    if scope_reply:
        return finalize_customer_reply(
            f"<div style='color:#333;line-height:1.55;'>{scope_reply}</div>"
            if "<" not in scope_reply
            else scope_reply,
            original_msg or msg_en,
            rl,
        )

    is_thanks_or_close = (
        message_is_user_feedback_or_closing(f"{original_msg} {msg_en}".strip())
        or message_is_casual_farewell_or_closing(f"{original_msg} {msg_en}".strip())
    )
    if is_thanks_or_close:
        contextual = build_contextual_ack_reply(
            original_msg, msg_en, conversation_context, reply_lang=rl, ctx=ctx
        )
        if contextual:
            return contextual

    ai_body = ai_chitchat_reply(
        original_msg, msg_en, conversation_context, reply_lang=rl, ctx=ctx
    )
    if ai_body:
        return ai_body

    if should_send_warm_greeting_reply(original_msg, msg_en, conversation_context):
        warm = fast_warm_reply_html(original_msg, msg_en, reply_lang=rl)
        if warm:
            return finalize_customer_reply(warm, original_msg or msg_en, rl)
    contextual = build_contextual_ack_reply(
        original_msg, msg_en, conversation_context, reply_lang=rl, ctx=ctx
    )
    if contextual:
        return contextual

    preferred = _preferred_greeting_reply_key(original_msg or msg_en, reply_lang=reply_lang)
    if preferred:
        body = sysmsg(preferred) or ""
    else:
        use_hinglish = rl == "hinglish" or is_hinglish_message(original_msg or msg_en or "")
        if use_hinglish:
            body = sysmsg("feedback_ack_hinglish") or sysmsg("warm_smalltalk_hinglish_3") or ""
        else:
            body = sysmsg("feedback_ack") or sysmsg("warm_smalltalk_3") or ""
    if not body:
        return ""
    return finalize_customer_reply(body, original_msg or msg_en, rl)


def _should_bypass_warm_greeting_fast_path(combined_text: str) -> bool:
    """
    Skip cosine / template greeting shortcuts when the user has a concrete Welfog support
    or shopping question (order delay, tracking, refund, product search, etc.).
    """
    t = (combined_text or "").strip()
    if not t:
        return False
    if _is_short_pure_greeting(t) or _is_light_smalltalk_fast(t):
        return False
    # Casual openers ("sun na", "kya haal") — never run the heavy support/shopping guard chain.
    if _looks_like_greeting_message(t):
        return False
    if _text_asks_wishlist(t) or message_is_wishlist_like_request(t):
        return True
    if _text_asks_order_history(t) or _text_asks_how_to_view_order_history(t):
        return True
    if _message_submits_or_corrects_order_id(t):
        return True
    if _light_smalltalk_blocker_hit(t):
        return True
    if _text_has_light_order_tracking_markers(t) or _text_is_undelivered_order_complaint(t):
        return True
    if _text_has_refund_or_return_intent(t):
        return True
    if _text_asks_order_history(t) or _text_asks_how_to_view_order_history(t):
        return True
    if _text_has_delivery_or_order_area_intent(t):
        return True
    if _text_has_product_shopping_intent_core(t) or _turn_is_catalog_product_request(t):
        return True
    if _text_asks_customer_care_contact(t):
        return True
    if _text_has_concrete_welfog_support_question(t):
        return True
    if _text_is_order_id_help_request(t):
        return True
    return False


def _looks_like_light_smalltalk(msg: str, msg_en: str = "") -> bool:
    """
    Cultural / Indian greetings and casual wellbeing — not product queries.
    Keeps users in-domain with a warm reply instead of out_of_domain or stiff deflection.
    """
    raw = (msg or "").strip().lower()
    extra = (msg_en or "").strip().lower()
    sample = re.sub(r"\s+", " ", f"{raw} {extra}").strip()
    if not sample or len(sample) > 160:
        return False
    if _light_smalltalk_blocker_hit(sample):
        return False
    if any(x.lower() in sample for x in _NATIVE_SCRIPT_GREETING_SNIPPETS):
        return True
    if any(x.lower() in sample for x in _NATIVE_SCRIPT_THANKS_SNIPPETS):
        return True
    for c in _CULTURAL_GREETING_SNIPPETS:
        if c in sample:
            return True
    for c in _CASUAL_WELLBEING_SNIPPETS:
        if c in sample:
            return True
    if re.search(r"\b(thanks|thx|thankyou|thanku)\b", sample):
        return True
    if re.search(r"\b(sup|wassup)\b", sample):
        return True
    return False


def _looks_like_native_script_short_greeting(msg: str) -> bool:
    """
    Script-aware fallback for unseen greeting variants in native scripts.
    Keeps pure greeting messages out of KB/data routes.
    """
    raw = (msg or "").strip()
    if not raw:
        return False
    if _light_smalltalk_blocker_hit(raw):
        return False
    if _text_has_product_shopping_intent_core(raw):
        return False
    try:
        from services.translation_service import _detect_script_language

        script_lang = _detect_script_language(raw)
    except Exception:
        script_lang = None
    if not script_lang:
        return False
    if re.search(r"\d", raw):
        return False
    tokens = [t for t in re.split(r"\s+", raw) if t]
    if len(tokens) > 8:
        return False
    lowered = raw.lower()
    hint_fragments = (
        "नम", "राम", "राधे", "प्रणाम",
        "ਸਤ", "ਅਕਾਲ", "ਨਮਸ",
        "નમસ", "રામ",
        "வண", "నమ", "ನಮ", "നമ", "নম", "سلام", "شکر",
    )
    if any(h in lowered for h in hint_fragments):
        return True
    return len(tokens) <= 3


_GREETING_STYLE_KEYS = ("greeting", "greeting_variant_2", "greeting_variant_3")
_GREETING_HINGLISH_KEYS = ("greeting_hinglish_1", "greeting_hinglish_2", "greeting_hinglish_3")
_SMALLTALK_STYLE_KEYS = (
    "warm_smalltalk_1",
    "warm_smalltalk_2",
    "warm_smalltalk_3",
    "warm_smalltalk_4",
    "warm_smalltalk_5",
)
_SMALLTALK_HINGLISH_KEYS = (
    "warm_smalltalk_hinglish_1",
    "warm_smalltalk_hinglish_2",
    "warm_smalltalk_hinglish_3",
)


def _text_has_thanks_only_intent(text: str) -> bool:
    """Thanks/appreciation only — no active shopping/support ask."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _looks_like_greeting_message(raw) or _is_light_smalltalk_fast(raw):
        return False
    if _should_bypass_warm_greeting_fast_path(raw.lower()):
        return False
    tl = f" {_normalize_welfog_typos(raw).lower()} "
    markers = (
        "thanks", "thank you", "thanku", "thankyou", "thx",
        "shukriya", "dhanyavad", "dhanyawaad", "appreciate",
        "great", "nice", "good job", "good one",
    )
    if any(m in tl for m in markers):
        return True
    return any(x.lower() in tl for x in _NATIVE_SCRIPT_THANKS_SNIPPETS)


def _preferred_greeting_reply_key(original_msg: str, reply_lang: str = "") -> str:
    """
    Match greeting style with user text.
    - ram ram -> ram ram
    - radhe radhe -> radhe radhe
    - hi/hello -> hi/hello
    - thanks only -> welcome-style reply
    """
    from services.translation_service import is_hinglish_message

    low = f" {_normalize_welfog_typos(original_msg or '').lower()} "
    use_hinglish = (reply_lang or "").lower() == "hinglish" or (
        not reply_lang and is_hinglish_message(original_msg or "")
    )

    if _text_has_thanks_only_intent(original_msg):
        return "feedback_welcome_hinglish" if use_hinglish else "feedback_welcome"
    if " ram ram " in low or "राम राम" in (original_msg or ""):
        return "greeting_ram_ram_hinglish" if use_hinglish else "greeting_ram_ram"
    if " radhe radhe " in low or " radhey radhey " in low or "राधे राधे" in (original_msg or ""):
        return "greeting_radhe_hinglish" if use_hinglish else "greeting_radhe"
    if any(x in low for x in (" namaste ", " namaskar ", " pranam ", " pranaam ")) or any(
        x in (original_msg or "") for x in ("नमस्ते", "नमस्कार", "प्रणाम")
    ):
        return "greeting_namaste_hinglish" if use_hinglish else "greeting_namaste"
    if any(x in low for x in (" hi ", " hello ", " hey ", " helo ", " hii ", " heyy ")):
        return "greeting_hinglish_2" if use_hinglish else "greeting_variant_2"
    return ""


def should_send_warm_greeting_reply(
    original_msg: str, msg_en: str = "", conversation_context: str = ""
) -> bool:
    """
    Hi / hello / namaste / light smalltalk — warm reply and reset stale order-id threads.
    """
    combined = f"{original_msg} {msg_en}".strip()
    if _is_short_pure_greeting((original_msg or "").strip()) or _is_light_smalltalk_fast(
        original_msg, msg_en
    ):
        return True
    if _looks_like_native_script_short_greeting(original_msg):
        return True
    if _looks_like_greeting_message(original_msg):
        return True
    if _looks_like_light_smalltalk(original_msg, msg_en):
        return True
    if not combined or _should_bypass_warm_greeting_fast_path(combined.lower()):
        return False
    if _text_has_product_shopping_intent_core(combined):
        return False
    return False


def fast_greeting_reply_html(original_msg: str, reply_lang: str = "") -> str:
    """Instant hello reply — no embedding, no LLM, no random pool (chat must not hang)."""
    return fast_warm_reply_html(original_msg, msg_en="", reply_lang=reply_lang, force_greeting=True)


def fast_warm_reply_html(
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
    *,
    force_greeting: bool = False,
) -> str:
    """Instant greeting / smalltalk — KB templates only, no embeddings or LLM."""
    from services.kb_service import sysmsg
    from services.translation_service import is_hinglish_message

    rl = (reply_lang or "").strip().lower()
    use_h = rl == "hinglish" or (not rl and is_hinglish_message(original_msg or ""))
    smalltalk = (not force_greeting) and _is_light_smalltalk_fast(
        original_msg, msg_en
    ) and not _is_short_pure_greeting(original_msg)
    if smalltalk:
        if use_h:
            body = (
                sysmsg("warm_smalltalk_hinglish_1")
                or sysmsg("warm_smalltalk_hinglish_2")
                or sysmsg("greeting_hinglish")
                or ""
            )
        else:
            body = sysmsg("warm_smalltalk_1") or sysmsg("warm_smalltalk_2") or sysmsg("greeting") or ""
    elif use_h:
        body = (
            sysmsg("greeting_hinglish")
            or sysmsg("greeting_variant_2")
            or sysmsg("greeting")
            or ""
        )
    else:
        body = sysmsg("greeting") or sysmsg("greeting_variant_2") or ""
    if body:
        return body
    if smalltalk:
        return (
            "<div style='color:#333;line-height:1.55;'>"
            "Main theek hoon, shukriya! Welfog pe shopping, orders ya delivery — kya help chahiye?"
            "</div>"
        )
    return (
        "<div style='color:#333;line-height:1.55;'>"
        "Hi! Main Welfog shopping assistant hoon — products, orders, delivery ke liye poochho."
        "</div>"
    )


def message_is_conversation_reset_command(text: str) -> bool:
    """User exits order-id / tracking wait state (cancel, stop)."""
    return (text or "").strip().lower() in ("cancel", "stop", "exit", "no", "band", "ruk")


def should_use_warm_conversation_reply(
    original_msg: str, msg_en: str = "", conversation_context: str = ""
) -> bool:
    """Fresh chat opener only (no prior messages in thread)."""
    if (conversation_context or "").strip():
        return False
    return should_send_warm_greeting_reply(original_msg, msg_en, conversation_context)


def pick_warm_chat_reply_key(smalltalk: bool, original_msg: str = "", reply_lang: str = "") -> str:
    """Random template for greetings vs cultural / casual chat — polite, not pushy links."""
    from services.translation_service import is_hinglish_message

    preferred = _preferred_greeting_reply_key(original_msg, reply_lang=reply_lang)
    if preferred:
        return preferred
    use_hinglish = reply_lang == "hinglish" or (
        not reply_lang and is_hinglish_message(original_msg or "")
    )
    if smalltalk:
        pool = _SMALLTALK_HINGLISH_KEYS if use_hinglish else _SMALLTALK_STYLE_KEYS
    else:
        pool = _GREETING_HINGLISH_KEYS if use_hinglish else _GREETING_STYLE_KEYS
    return random.choice(pool)


def build_warm_conversation_reply(original_msg: str, msg_en: str = "", reply_lang: str = "") -> str:
    """Human-like opener — KB templates only (no heavy guards or embedding path)."""
    return fast_warm_reply_html(original_msg, msg_en, reply_lang=reply_lang)


def _normalize_order_chat_text(text: str) -> str:
    """Fix common Hinglish typos before order-id / tracking detection."""
    if not text:
        return ""
    t = text
    replacements = (
        (r"\bordder\b", "order"),
        (r"\boorder\b", "order"),
        (r"\bordr\b", "order"),
        (r"\btrck\b", "track"),
        (r"\btraking\b", "tracking"),
        (r"\btrak\b", "track"),
        (r"\btrackng\b", "tracking"),
    )
    for pat, repl in replacements:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t


def _text_has_explicit_pincode_subject(t: str) -> bool:
    """User is clearly asking about PIN / delivery area — not order tracking."""
    tl = f" {(t or '').lower()} "
    return any(
        x in tl
        for x in (
            "pincode", "pin code", "pin-code", "zip code", "postal code",
            "pincod", "pin cod", "pin  cod",
            "delivery available", "serviceability", "deliver ho", "milegi delivery",
            "delivery milegi", "service area", "6-digit pin", "6 digit pin",
            "delivery", "delevery", "delivry", "deliver", "pahucha", "pahunch",
            "saman", "samaan", "samān", "phocha", "phocha dega", "pahucha dega",
        )
    )


def _text_is_delivery_serviceability_hypothetical(t: str) -> bool:
    """
    Will Welfog deliver to a place / friend's area — not track an existing order.
    Recursion-safe: no full order-tracking or pincode-serviceability chain.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} "
    if any(
        x in tl
        for x in (
            "nhi aaya", "nahi aaya", "nhi aya", "nahi aya", "nhi mila", "nahi mila",
            "nahi pahucha", "nhi pahucha", "nahi pahuncha", "nhi pahuncha",
            "abhi tak", "ab tak", "order kiya", "order kr diya", "order kar diya",
            "late", "delay", "stuck", "pending", "track", "tracking", "trck",
            "order id", "orderid", "order status", "kab aayega", "kab aaega",
            "mere order", "mera order", "status bata", "status check",
        )
    ):
        return False
    future = (
        "pahucha dega", "pahuchega", "pahunchega", "pahunch jaega", "pahunch jayega",
        "pahucha de", "dega kya", "degi kya", "milega kya", "milegi kya",
        "pahuch jayega", "pahunch sak", "deliver ho sak", "delivery ho sak",
        "aa jayega", "aa jaayega",
    )
    area = (
        "rehta", "rhta", "rehte", "rhti", "dost", "friend", "waha", "wahan",
        "yaha", "yahan", " me rehta", " me rhta",
    )
    if any(f in tl for f in future):
        if any(a in tl for a in area) or "welfog" in tl or "product" in tl:
            return True
    if re.search(r"\bkya\b", tl) and any(
        x in tl for x in ("pahuch", "milega", "dega", "deliver", "aa jayeg", "pahunch")
    ):
        if "welfog" in tl or "product" in tl or any(a in tl for a in area):
            return True
    return False


def _text_is_pincode_serviceability_question(t: str, conversation_context: str = "") -> bool:
    """
  User asks IF Welfog delivers / services an area (any language, Hinglish typos).
  Lightweight path avoids order-id misread on 6-digit PINs.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    if _turn_blocks_pincode_serviceability_routing(raw):
        return False
    if message_mentions_wishlist_topic(raw) or _text_asks_how_to_view_wishlist(
        raw, conversation_context
    ):
        return False
    if _text_is_undelivered_order_complaint(raw):
        return False
    if _text_is_delivery_serviceability_hypothetical(raw):
        return True
    if _text_is_order_tracking_intent_leaf(raw):
        return False
    if _user_denies_pincode_insists_order_id(raw):
        return False
    if _text_has_order_id_context(raw) and not _text_has_explicit_pincode_subject(raw):
        if not re.search(r"\b(?:pincode|pin code)\b", f" {raw.lower()} "):
            return False
    if not re.search(r"\b[1-9]\d{5}\b", raw):
        if _conversation_in_pincode_delivery_flow(conversation_context):
            try:
                from services.location_delivery_resolver import turn_continues_pincode_area_check

                if turn_continues_pincode_area_check(raw, conversation_context):
                    return True
            except ImportError:
                pass
        if _text_has_delivery_or_order_area_intent(raw) and not _user_denies_pincode_insists_order_id(raw):
            if not _text_has_order_id_context(raw) or _text_has_explicit_pincode_subject(raw):
                return True
        if _text_is_pincode_serviceability_question_light(raw):
            return True
        if _conversation_bot_asked_for_pincode(conversation_context):
            pass
        elif not _text_has_delivery_or_order_area_intent(raw):
            return False
    if _text_is_pincode_serviceability_question_light(raw):
        return True
    if _text_has_delivery_serviceability_intent(raw, conversation_context):
        return True
    if _text_has_pincode_delivery_intent(raw, conversation_context):
        return True
    return False


def _pincode_delivery_signal_leaf(t: str) -> bool:
    """
    PIN + delivery/service meaning — leaf only (no wishlist/order-placement chains).
    """
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} "
    if _user_denies_pincode_insists_order_id(raw):
        return False
    if _leaf_non_tracking_order_id_intent(raw):
        return False
    if re.search(r"\b[0-9]{7,20}\b", raw) and re.search(r"\borders?\b", tl):
        return False
    if re.search(r"\b\d{4,20}\b", raw) and re.search(r"\borders?\b", tl):
        if any(
            x in tl
            for x in (
                "kahan", "pahunch", "pahucha", "track", "status", "kab",
                "courier", "shipment", "deliver ho gaya", "delivered",
            )
        ):
            return False
    if re.search(r"\b(?:order\s*id|orderid)\b", tl) and not re.search(
        r"\b(?:pincode|pin code)\s+(?:nhi|nahi|not)\b", tl
    ):
        return False
    has_pin = bool(re.search(r"\b[1-9]\d{5}\b", raw))
    delivery_markers = (
        "delivery", "delevery", "delivry", "deliver", "service", "pahucha", "pahunch",
        "phocha", "milega", "milegi", "mil jaygi", "mil jayega", "mil jayegi", "milta", "milti",
        "available", "serviceable", "pincode", "pincod", "pin code", " per ", " par ", " pe ",
        "yaha", "yahan", "waha", "wahan", "area", "check kr", "check kar", "check kro",
        "bta de", "bata de", "kya service",
    )
    has_area_hint = bool(re.search(r"\b(?:me|mein|par|pe|per)\b", tl))
    if not has_pin and not _text_has_explicit_pincode_subject(raw):
        if any(m in tl for m in delivery_markers) and has_area_hint:
            return True
        if re.search(r"\baur\b", tl) and has_area_hint:
            return True
        return False
    if any(m in tl for m in delivery_markers):
        return True
    if "welfog" in tl and has_pin:
        return True
    return has_pin and _text_has_explicit_pincode_subject(raw)


def _turn_blocks_pincode_serviceability_routing(t: str) -> bool:
    """
    Refund/how-to, company FAQ, customer care — must not hit pincode flow because
    Hinglish uses 'milta/milega' for refunds ('refund kese milta h').
    Recursion-safe: no wishlist how-to / order-placement / pincode-light chains.
    """
    if getattr(_pincode_light_guard, "active", False):
        return False
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} "
    if _text_has_refund_or_return_intent(raw):
        if any(
            x in tl
            for x in (
                "kese", "kaise", "how", "process", "policy", "steps", "kya kar", "kya kr",
                "milta", "milega", "kab milega", "kitne din",
            )
        ):
            if not re.search(
                r"\b(?:pincode|pin\s*code|delivery|deliver|delevery|area|ship to|shipping)\b",
                tl,
            ):
                return True
    if _is_welfog_about_fast_path(raw):
        return True
    if any(
        x in tl
        for x in (
            "customer care", "cust care", "support number", "helpline", "call center",
            "contact number", "phone number", "email id", "grievance officer",
        )
    ):
        return True
    if "wishlist" in tl or "wish list" in tl:
        return True
    if _text_wants_order_history_list_in_chat(raw) or message_is_past_purchase_list_request(raw):
        return True
    if _text_is_order_tracking_intent_leaf(raw) and "delivery" not in tl:
        return True
    if turn_is_obvious_product_shopping_turn(raw):
        return True
    if _text_mentions_welfog_brand(tl) and any(
        x in tl
        for x in (
            " kya h ", " kya hai ", " what is ", " about ", " krti kya ", " krta kya ",
            "tell me", "batao", "bata ", "explain",
        )
    ):
        return True
    if any(
        p in tl
        for p in (
            "return policy", "refund policy", "privacy policy", "terms and",
            "shipping policy", "cancellation policy",
        )
    ):
        return True
    return False


def _text_is_pincode_serviceability_question_light(t: str) -> bool:
    """Recursion-safe: PIN + delivery/service meaning without extract_order_id."""
    if getattr(_pincode_light_guard, "active", False):
        return _pincode_delivery_signal_leaf(t)
    _pincode_light_guard.active = True
    try:
        if not (t or "").strip():
            return False
        if _turn_blocks_pincode_serviceability_routing(t):
            return False
        tl = f" {(t or '').lower()} "
        if "wishlist" in tl or "wish list" in tl:
            return False
        return _pincode_delivery_signal_leaf(t)
    finally:
        _pincode_light_guard.active = False


def _user_denies_pincode_insists_order_id(t: str) -> bool:
    """User corrects bot: 'pincode nhi, yeh order id h'."""
    if not (t or "").strip():
        return False
    tl = f" {t.lower()} "
    denies_pin = bool(
        re.search(r"\b(?:pincode|pin code|pin)\s+(?:nhi|nahi|ni|nahin|not)\b", tl)
        or any(x in tl for x in ("not pincode", "not a pincode", "pincode nhi", "pincode nahi", "pin nhi", "pin nahi"))
    )
    claims_order = bool(
        re.search(r"\b(?:order\s*id|orderid)\b", tl)
        or any(x in tl for x in ("yeh order", "ye order", "order id h", "order id hai", "order id hi", "order id thi"))
    )
    return denies_pin and claims_order


def _message_has_embedded_pincode_for_delivery(
    t: str, conversation_context: str = ""
) -> bool:
    """6-digit PIN in message is for delivery check — not Order ID tracking."""
    raw = (t or "").strip()
    if not re.search(r"\b[1-9]\d{5}\b", raw):
        return False
    # Positive delivery/PIN signals first — never call _text_has_order_id_context here
    # (it chains into order-tracking → feedback → extract_pincode → infinite recursion).
    if (
        _pincode_delivery_signal_leaf(raw)
        or _text_has_explicit_pincode_subject(raw)
        or re.search(r"\b(?:pincode|pin\s*code|pin)\b", raw, re.I)
        or _conversation_bot_asked_for_pincode(conversation_context)
        or _conversation_in_pincode_delivery_flow(conversation_context)
    ):
        return True
    tl = f" {raw.lower()} "
    if re.search(r"\b(?:order\s*id|orderid)\b", tl) and not _text_has_explicit_pincode_subject(raw):
        return False
    return False


def _digits_in_message_are_order_id_not_pincode(
    t: str, conversation_context: str = "", *, ai_route: dict | None = None
) -> bool:
    """7+ digit tokens and explicit order-id wording must not become a truncated PIN."""
    if getattr(_pin_order_guard, "active", False):
        return False
    _pin_order_guard.active = True
    try:
        return _digits_in_message_are_order_id_not_pincode_impl(
            t, conversation_context, ai_route=ai_route
        )
    finally:
        _pin_order_guard.active = False


def _digits_in_message_are_order_id_not_pincode_impl(
    t: str, conversation_context: str = "", *, ai_route: dict | None = None
) -> bool:
    """7+ digit tokens and explicit order-id wording must not become a truncated PIN."""
    raw = (t or "").strip()
    if not raw:
        return False
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        nc = (ai_route.get("numeric_context") or "").strip().lower()
        if intent == "pincode_check" or nc == "pincode":
            return False
        if intent in ("order", "refund", "payment") and nc == "order_id":
            if re.search(r"\b\d{4,20}\b", raw):
                return True
    tl_early = f" {_normalize_order_chat_text(raw).lower()} "
    if _text_has_refund_or_return_intent(raw) and re.search(r"\b[0-9]{4,20}\b", raw):
        return True
    if re.search(r"\b[0-9]{7,20}\b", raw) and any(
        x in tl_early
        for x in (" refund", " return", " id pe", " id ka", " id ke", " is id", " iska", " iski")
    ):
        return True
    if _leaf_non_tracking_order_id_intent(raw):
        return True
    if "order" in tl_early and re.search(r"\b[0-9]{7,20}\b", raw):
        return not _text_has_explicit_pincode_subject(raw)
    if _text_has_order_id_context(raw) and re.search(r"\b[0-9]{4,20}\b", raw):
        return not _text_has_explicit_pincode_subject(raw)
    if _message_has_embedded_pincode_for_delivery(raw, conversation_context):
        return False
    if _text_is_pincode_serviceability_question_light(raw):
        return False
    if _user_denies_pincode_insists_order_id(raw):
        return True
    tl_pid = f" {raw.lower()} "
    if any(
        x in tl_pid
        for x in (
            "product id",
            "pro id",
            "id ka product",
            "yeh product id",
            "product id h",
            "iska product",
        )
    ):
        return True
    if _text_has_light_order_tracking_markers(raw):
        return True
    tl = f" {_normalize_order_chat_text(raw).lower()} "
    if re.search(r"\b(?:order\s*id|orderid|order-id|oid)\b", tl):
        return not _text_has_explicit_pincode_subject(raw)
    if "order" in tl and re.search(r"\b[0-9]{7,20}\b", raw):
        return not _text_has_explicit_pincode_subject(raw)
    if re.search(r"\b[0-9]{7,20}\b", raw) and (
        _text_has_order_id_context(raw)
        or _text_has_light_order_tracking_markers(raw)
    ):
        return not _text_has_explicit_pincode_subject(raw)
    return False


def _text_has_light_order_tracking_markers(t: str) -> bool:
    """Recursion-safe subset of tracking markers (no feedback/pincode extract)."""
    tl = f" {_normalize_order_chat_text(t).lower()} "
    markers = (
        "track", "tracking", "order status", "order id", "orderid", "where is my order",
        "kab aayega", "kab aaega", "kab milega", "order kahan", "shipment", "parcel",
        "delivery status", "order track", "order update", "status bata", "status btao",
    )
    return any(m in tl for m in markers)


def _text_has_order_id_context(t: str) -> bool:
    tl = f" {_normalize_order_chat_text(t).lower()} "
    if re.search(r"\bproduct\s+id\b", tl) or re.search(r"\bpro\s*id\b", tl):
        return False
    if re.search(r"\bsku\b", tl) and any(
        x in tl for x in ("product", "dikha", "bata", "milega", "item")
    ):
        return False
    if re.search(r"\b(?:order\s*id|orderid|order-id|oid)\b", tl):
        return True
    if _text_has_light_order_tracking_markers(t):
        return True
    if re.search(r"\b(?:yeh|ye|hai|h)\b.{0,20}\b(?:order|id)\b", tl):
        return True
    if re.search(r"\b\d{4,20}\b", tl) and "order" in tl:
        return True
    if re.search(r"\b\d{4,20}\b\s*(?:y\s+)?(?:rha|raha|rahi|rhi|hai|h)\b", tl):
        return True
    if any(x in tl for x in (" y rha", "y raha", "y rahi", "ye rha", "yeh rha", "lo id", "le id")):
        return True
    return False


def _conversation_awaiting_order_id(conversation_context: str) -> bool:
    """
    Bot's latest turn asked for Order ID — a bare 4–20 digit line is order id, not pincode.
    """
    if not (conversation_context or "").strip():
        return False
    lines = (conversation_context or "").splitlines()
    last_asst = ""
    for line in reversed(lines):
        low = line.strip().lower()
        if low.startswith("assistant:") or low.startswith("assistant "):
            last_asst = low
            break
    if not last_asst:
        return _conversation_bot_offered_order_id_or_tracking(conversation_context)
    if ("order id" in last_asst or "orderid" in last_asst) and any(
        x in last_asst
        for x in (
            "bhej", "paste", "send", "share", "taaki", "check", "status",
            "optional", "yahan", "yaha", "here", "live", "tracking",
        )
    ):
        return True
    if "order id bhej" in last_asst or "order id paste" in last_asst:
        return True
    return False


def _conversation_bot_asked_for_pincode(conversation_context: str) -> bool:
    """
    Bot's latest turn asked for a 6-digit PIN — bare numeric reply is pincode, not order id.
    """
    if not (conversation_context or "").strip():
        return False
    lines = (conversation_context or "").splitlines()
    last_asst = ""
    for line in reversed(lines):
        low = line.strip().lower()
        if low.startswith("assistant:") or low.startswith("assistant "):
            last_asst = low
            break
    if not last_asst:
        tail = (conversation_context or "")[-3500:].lower()
        if not any(
            x in tail
            for x in (
                "pin code",
                "pincode",
                "6-digit pin",
                "6 digit pin",
                "ask_pincode",
                "check_pincode",
                "delivery check",
            )
        ):
            return False
        last_asst = tail[-1200:]
    pin_ask = any(
        x in last_asst
        for x in (
            "pin code",
            "pincode",
            "pin-code",
            "6-digit pin",
            "6 digit pin",
            "postal code",
            "zip code",
            "ask_pincode",
            "check_pincode",
            "delivery check karne",
            "delivery availability",
            "serviceability",
            "pin code sahi",
            "6 digits",
        )
    )
    if not pin_ask:
        return False
    if "order id" in last_asst and "pin" not in last_asst:
        return False
    return any(
        x in last_asst
        for x in (
            "bhej",
            "send",
            "share",
            "paste",
            "provide",
            "enter",
            "type",
            "dubara",
            "try karo",
            "try again",
            "please",
            "apna",
            "your",
            "sahi",
            "bhejo",
            "share your",
        )
    ) or "?" in last_asst


def _is_plausible_order_id(token: str, context: str = "", *, shallow: bool = False) -> bool:
    """
    Order-id shapes we accept:
    - Numeric-only: 4–20 digits (any length in that range — e.g. 302032, 26051410).
    - Alphanumeric: 4–20 chars with at least one letter and one digit.
    Six-digit numbers are only rejected as PIN when the message is clearly pincode-only.
    shallow=True: used by _bare_order_id_token_from_msg — avoids circular pincode/follow-up calls.
    """
    if not token:
        return False
    t = token.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{4,20}", t):
        return False
    ctx = _normalize_order_chat_text(context or "")
    if re.fullmatch(r"\d+", t):
        if len(t) < 4 or len(t) > 20:
            return False
        if len(t) == 6 and re.fullmatch(r"[1-9]\d{5}", t):
            if _conversation_awaiting_order_id(ctx) or _text_has_order_id_context(ctx):
                return True
            if shallow:
                tl = f" {ctx.lower()} "
                if _text_is_pincode_serviceability_question_light(ctx):
                    return False
                if any(
                    x in tl
                    for x in (
                        "order id", "orderid", "track", "status check", " y rha", "y raha",
                        "y rahi", "bhej do", "paste", "live status",
                    )
                ):
                    return True
                if any(x in tl for x in ("pincode", "pin code", "good news", "delivery available")):
                    if not any(x in tl for x in ("order id", "orderid", "track", "bhej do")):
                        return False
                return False
            if _text_has_pincode_delivery_intent(ctx) and not _text_has_order_id_context(ctx):
                return False
            tl = f" {ctx.lower()} "
            pin_only = any(x in tl for x in ("pincode", "pin code", "zip code", "postal"))
            if pin_only and not _text_has_order_id_context(ctx):
                return False
        return True
    if not re.search(r"[A-Za-z]", t):
        return False
    if not re.search(r"\d", t):
        return False
    ctx_low = f" {_normalize_order_chat_text(context or '').lower()} "
    if re.search(r"\bsku\b", ctx_low) and re.search(r"[-_]", t):
        return False
    if re.search(r"\bproduct\s+id\b", ctx_low) or re.search(r"\bpro\s*id\b", ctx_low):
        return False
    return True


_PRODUCT_ID_PHRASES = (
    r"\bid\s+ke\s+products?",
    r"\bproducts?\s+(?:for|of|with)\s+(?:this\s+)?id\b",
    r"\bis\s+id\s+ke\b",
    r"\bis\s+id\s+ka\b",
    r"\bis\s+product\s*id\b",
    r"\bis\s+product\s*id\s+ka\b",
    r"\bproduct\s+id\b",
    r"\bid\s+ka\s+product",
    r"\bpro[_\s-]?id\b",
    r"\bproduct[_\s-]?id\b",
    r"\bpid\b",
    r"\bsku\b",
)


def _text_is_sku_product_lookup_context(t: str) -> bool:
    """User pasted or named a warehouse SKU to find a product — not Order ID."""
    if not (t or "").strip():
        return False
    tl = f" {_normalize_order_chat_text(t).lower()} "
    if not re.search(r"\bsku\b", tl):
        return False
    if any(re.search(p, tl) for p in _PRODUCT_ID_PHRASES if "sku" not in p):
        return True
    product_markers = (
        "product", "item", "dikha", "dikhao", "dikho", "bata", "btao", "batao",
        "milega", "milegi", "catalog", "stock", "de rha", "de rahi", "de ra",
    )
    return any(re.search(rf"\b{re.escape(m)}\b", tl) for m in product_markers)


def turn_is_catalog_product_lookup(
    original_msg: str,
    msg_en: str = "",
    ai_route: dict | None = None,
    *,
    route_handler: str = "",
) -> bool:
    """
    This user turn is catalog product / pro_id / SKU lookup — never live Order ID tracking.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    if _leaf_non_tracking_order_id_intent(comb):
        return False
    if _text_is_undelivered_order_complaint(comb) or _text_has_past_order_complaint_context(comb):
        return False
    if _text_is_order_tracking_intent_leaf(comb):
        return False
    if message_mentions_wishlist_topic(comb) or _text_asks_how_to_view_wishlist(comb):
        return False
    if message_is_past_purchase_list_request(comb) or _text_wants_order_history_list_in_chat(
        comb
    ):
        return False
    handler = (route_handler or "").strip().lower()
    if handler == "catalog_pro_id":
        return True
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        numeric = (ai_route.get("numeric_context") or "").strip().lower()
        rh = (ai_route.get("route_handler") or "").strip().lower()
        if rh == "catalog_pro_id" or numeric == "product_id":
            return True
        if intent == "product" and numeric != "order_id":
            if (
                extract_product_id(comb)
                or _text_is_product_id_lookup_context(comb)
                or _text_is_sku_product_lookup_context(comb)
            ):
                return True
    if extract_product_id(comb):
        return True
    if _text_is_product_id_lookup_context(comb):
        return True
    if _text_is_sku_product_lookup_context(comb):
        return True
    return False


def turn_is_obvious_product_shopping_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Clear catalog shopping turn — skip delivery/KB micro-classifiers (any language).
    Does not use customer keyword lists for routing; combines structural product signals.
    """
    if turn_is_catalog_product_lookup(original_msg, msg_en):
        return True
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    if _turn_is_catalog_product_request(comb):
        return True
    if _text_has_product_shopping_intent_core(comb):
        return True
    return False


def _text_is_undelivered_order_complaint(t: str) -> bool:
    """
    User says order/product not received yet — track/status, NOT purchase-history list.
    Leaf helper only (no _text_is_order_tracking_intent — avoids recursion).
    """
    raw = _normalize_order_chat_text((t or "").strip())
    if not raw:
        return False
    tl = f" {raw.lower()} "
    not_arrived = (
        "nhi pahucha", "nahi pahucha", "nhi pahuncha", "nahi pahuncha",
        "nhi pahunch", "nahi pahunch", "ab bhi nhi", "abhi bhi nhi", "ab tak nhi",
        "ab tak nahi", "tk bhi nhi", "tak bhi nahi",         "nhi aaya", "nahi aaya",
        "nhi aya", "nahi aya", "nhi aa ", "nahi aa ", "order nhi aa", "order nahi aa",
        "nhi mila", "nahi mila", "aa nahi", "aaya nahi",
        "not received", "not arrived", "hasn't arrived", "has not arrived",
        "still not", "not delivered", "delivery nahi", "delivery nhi",
    )
    if any(x in tl for x in not_arrived):
        return True
    if any(x in tl for x in ("pahucha", "pahuncha", "pahunch", "deliver")) and any(
        x in tl
        for x in ("nhi", "nahi", "ni ", "nahin", "not ", "ab bhi", "abhi bhi", "ab tak nhi", "ab tak nahi")
    ):
        return True
    return False


def _user_rejects_order_history_wants_tracking(t: str, conversation_context: str = "") -> bool:
    """
    User corrects bot: show tracking/status, not order-history list again.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {_normalize_order_chat_text(raw).lower()} "
    rejects_history = any(
        x in tl
        for x in (
            "history kyu", "history kyo", "history kyun", "history kyu dikha",
            "history kyo dikha", "history kyun dikha", "history kyu dikha rha",
            "history kyu dikha raha", "history kyu dikha rhi", "history mat dikha",
            "history mt dikha", "history nahi dikha", "history nhi dikha",
            "list kyu", "list kyun", "list mat dikha", "order history kyu",
            "purchase history kyu", "history dikha rha", "history dikha rahi",
            "history dikha raha", "history dikha rhe",
        )
    )
    wants_track = any(
        x in tl
        for x in (
            "track", "tracking", "status", "kab aayega", "kab aaega", "kb aayega",
            "kahan hai", "kaha hai", "pahucha", "pahuncha", "deliver", "delivery",
            "nhi aaya", "nahi aaya", "nhi pahucha", "nahi pahucha", "atak", "atka",
            "track krke", "track karke", "track kr", "track kar",
        )
    )
    if rejects_history and wants_track:
        return True
    if rejects_history and _text_is_undelivered_order_complaint(raw):
        return True
    if wants_track and any(
        x in tl
        for x in (
            "ki jagah", "ke bajay", "ke badle", "instead of history",
            "not history", "history nahi", "history nhi",
        )
    ):
        return True
    if _conversation_bot_showed_order_history_list(conversation_context) and wants_track:
        if any(x in tl for x in ("kyu", "kyo", "kyun", "kese", "kaise", "ese bol", "bol rha", "bol raha")):
            return True
    return False


def message_is_past_purchase_list_request(t: str) -> bool:
    """
    User wants items/orders they already bought (order history), not catalog search.
    Covers 'purchase kiye … products … list' where 'products' means purchased items.

    Lightweight only — must NOT call _text_asks_order_history / extract_order_id
    (those call back into _text_is_product_id_lookup_context → infinite recursion).
    """
    if not (t or "").strip():
        return False
    if _text_is_undelivered_order_complaint(t):
        return False
    if message_is_wishlist_like_request(t):
        return False
    tl = f" {_normalize_welfog_typos(t).lower()} "
    if _message_has_catalog_product_signal(t) or _message_has_generic_shopping_item_signal(t):
        if any(v in tl for v in ("dikha", "dikhao", "dikhado", "milega", "milegi", "show me", "buy")):
            if not any(
                x in tl
                for x in ("order history", "purchase history", "mangaya", "order kiya", "order kiye")
            ):
                return False
    if "order history" in tl or "purchase history" in tl or "my orders" in tl:
        return True
    if re.search(r"\b(?:meri|mere|my)\s+orders?\b", tl) and any(
        v in tl for v in ("dikhao", "dikha", "list", "batao", "show", "dede", "history")
    ):
        return True
    list_verbs = (
        "list", "dede", "dedo", "dikhao", "dikha", "dikha de", "dikha do", "batao", "bata",
        "btao", "show", "dekho", "dekh", "bhejo", "history",
    )
    past_buy = (
        "purchase", "purchased", "bought", "ordered", "mangaya", "mangaye", "mangwaya",
        "mangwaye", "order kiye", "order kiya", "kharida", "liya",
    )
    if any(p in tl for p in past_buy) and any(v in tl for v in list_verbs):
        if not any(x in tl for x in ("milega", "milegi", "milta", "milti", "kharidna", "buy now")):
            if "order" in tl or "purchase" in tl or "mang" in tl:
                return True
    if "products" in tl and any(p in tl for p in past_buy):
        if any(v in tl for v in list_verbs) or any(
            x in tl for x in ("mene", "maine", "mere", "meri", "jitna", "is wali id")
        ):
            if not any(
                x in tl
                for x in (
                    "nhi pahucha", "nahi pahucha", "nhi aaya", "nahi aaya", "ab bhi nhi",
                    "abhi bhi nhi", "ab tak nhi", "tk bhi nhi", "pahucha", "pahuncha",
                )
            ):
                return True
    return False


def message_is_wishlist_like_request(t: str) -> bool:
    """
    Saved / liked / heart items — NOT purchased order history.
    Lightweight; does not call _text_asks_order_history (recursion safe).
    """
    if not (t or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(t).lower()} "
    if any(
        x in tl
        for x in (
            "mangaya", "mangaye", "mangwaya", "purchase kiye", "purchase kiya", "order kiye",
            "order kiya", "bought", "ordered", "kharida", "mangwaye",
        )
    ):
        return False
    if "wishlist" in tl or "wish list" in tl:
        return True
    if "heart" in tl and any(
        x in tl for x in ("product", "products", "item", "items", "wala", "wale", "list")
    ):
        if not any(
            x in tl
            for x in ("mangaya", "mangaye", "mangwaya", "order kiye", "order kiya", "kharida")
        ):
            if any(
                v in tl
                for v in ("dekh", "dekho", "dikhao", "dikha", "list", "batao", "btao", "show", "chahiye")
            ):
                return True
    like_markers = (
        "like diya", "like kiya", "like ki", "like kiye", "like kri", "like kra",
        "like krta", "like kiya h", "like diye", "liked", "pasand kiya", "pasand ki",
        "favourite", "favorite", "saved items", "saved products", "heart kiya",
        "save kiye", "save kre", "save kiya", "save kr", "save hue", "save kiye hue",
        "saved hue", "save kre the", "save kiye the", "save kiya tha", "save krke",
    )
    if "save" in tl or "saved" in tl:
        if any(x in tl for x in ("product", "products", "saman", "item", "items")):
            if not any(x in tl for x in ("mangaya", "mangaye", "mangwaya", "order kiye", "order kiya")):
                if any(
                    v in tl
                    for v in ("dekh", "dekho", "dikhao", "dikha", "bta", "btao", "show", "list", "dede")
                ):
                    return True
    if not any(m in tl for m in like_markers):
        return False
    if _message_has_catalog_product_signal(t) and re.search(
        r"\b(?:show|dikha|dikhao|dikho|dekh|dekho|chahiye|milega)\b", tl
    ):
        if "wishlist" not in tl and "wish list" not in tl:
            return False
    if any(
        v in tl
        for v in (
            "dekh", "dekho", "dekhna", "dikhao", "dikha", "list", "batao", "btao", "show",
            "dede", "dedo", "bta", "mereko", "mujhe", "mera", "meri",
        )
    ):
        return True
    return "products" in tl or "product" in tl


def _text_is_product_id_lookup_context(t: str) -> bool:
    """Numeric id in message is for catalog lookup, not order tracking."""
    if not t:
        return False
    try:
        if _message_looks_like_shopping_query(t):
            if not re.search(r"\b\d{4,12}\b", t):
                return False
    except Exception:
        pass
    if (
        _text_is_undelivered_order_complaint(t)
        or _text_is_order_tracking_intent_leaf(t)
        or _text_has_past_order_complaint_context(t)
    ):
        return False
    if message_is_past_purchase_list_request(t):
        return False
    if message_mentions_wishlist_topic(t) or _text_asks_how_to_view_wishlist(t):
        return False
    tl = f" {t.lower()} "
    if "wishlist" in tl or "wish list" in tl:
        return False
    if any(re.search(p, tl) for p in _PRODUCT_ID_PHRASES):
        return True
    product_markers = (
        "product", "products", "item", "items", "listing", "catalog",
        "brand", "sku", "size", "color", "colour", "rang", "stock",
        "dikha", "dikhao", "dikho", "dikhado", "dikahana", "dikahna", "batao", "bata", "btao",
        "milega", "milegi", "milta", "milti", "filter", "search",
        "sasta", "price", "rating", "cover", "case", "mobile", "phone",
        "yeh product id", "product id h", "id ka product", "iska product",
    )
    order_track_markers = (
        "track", "tracking", "shipped", "out for delivery", "delivered",
        "order status", "delivery status", "order detail", "order details",
        "order ki detail", "iski order", "us order", "mera order", "order ka status",
        "kab aayega", "kab aaega", "kab pahunchega", "kab pahuchega", "live tracking", "courier",
        "refund status", "return status",
    )
    def _has_word_marker(markers: tuple) -> bool:
        for m in markers:
            if re.search(rf"\b{re.escape(m)}\b", tl):
                return True
        return False

    has_product = _has_word_marker(product_markers)
    has_track = _has_word_marker(order_track_markers)
    if "order" in tl and any(x in tl for x in ("detail", "details", "status", "track", "tracking")):
        return False
    if has_product and not has_track:
        return True
    if re.search(r"\b\d{6,12}\b", tl) and has_product:
        return True
    return False


def extract_product_id(msg: str):
    """Welfog catalog pro_id (numeric), when message is about products not orders."""
    if not msg:
        return None
    if not _text_is_product_id_lookup_context(msg):
        return None
    low = msg.lower()
    patterns = (
        r"\b(?:pro[_\s-]?id|product[_\s-]?id|pid)\s*[:\-#]?\s*(\d{4,12})",
        r"\b(\d{4,12})\s+is\s+(?:product\s*)?id\b",
        r"\b(\d{4,12})\s+is\s+(?:product\s*)?id\s+ka\b",
        r"\b(\d{4,12})\s+is\s+id\s+ke",
        r"\b(\d{4,12})\s+is\s+id\s+ka\b",
        r"\b(\d{4,12})\s+.*\bid\s+ka\s+product\b",
        r"\bproduct\s+id\s+(\d{4,12})",
        r"\bproduct\s+id\s+(?:de\s+rha\s+hu|de\s+raha\s+hu|de\s+rahi\s+hu)\s+(\d{4,12})",
        r"\b(\d{4,12})\s+iska\s+product\b",
        r"\b(\d{4,12})\s+yeh\s+product\s+id\b",
        r"\b(\d{4,12})\b.*\bproduct\s+id\s+h\b",
        r"\byeh\s+product\s+id\s+h\b.*\b(\d{4,12})\b",
        r"\bid\s+(\d{4,12})\s+ka\s+product",
        r"\bid\s+(\d{4,12})\s+ke\s+products?",
        r"\bproducts?\s+(?:for|of)\s+(?:id\s+)?(\d{4,12})",
    )
    for pat in patterns:
        m = re.search(pat, low, re.IGNORECASE)
        if m and m.lastindex and m.lastindex >= 1:
            return int(m.group(1))
    for tk in re.findall(r"\b\d{6,12}\b", msg):
        if re.fullmatch(r"[1-9]\d{5}", tk):
            continue
        return int(tk)
    return None


def _bare_order_id_token_from_msg(msg: str, conversation_context: str = ""):
    """Extract order-id digits without calling follow-up / pincode gates (no recursion)."""
    raw = _normalize_order_chat_text(msg or "")
    if not raw.strip():
        return None
    ctx_comb = f"{raw} {(conversation_context or '')[-1200:]}"
    labeled_rha = re.search(
        r"\b([0-9]{4,20})\b\s*(?:y\s+)?(?:rha|raha|rahi|rhi|hai|h)\b",
        raw,
        re.IGNORECASE,
    )
    if labeled_rha and _is_plausible_order_id(labeled_rha.group(1), context=ctx_comb, shallow=True):
        return labeled_rha.group(1)
    bare = re.search(r"\b([0-9]{4,20})\b", raw)
    if bare and _is_plausible_order_id(bare.group(1), context=ctx_comb, shallow=True):
        return bare.group(1)
    return None


def extract_order_id(msg, conversation_context: str = ""):
    if getattr(_extract_oid_guard, "active", False):
        return _bare_order_id_token_from_msg(msg or "", conversation_context)
    _extract_oid_guard.active = True
    try:
        return _extract_order_id_impl(msg, conversation_context)
    finally:
        _extract_oid_guard.active = False


def _extract_order_id_impl(msg, conversation_context: str = ""):
    raw = _normalize_order_chat_text(msg or "")
    if _leaf_non_tracking_order_id_intent(raw):
        order_id_topic = True
    elif turn_is_catalog_product_lookup(raw):
        return None
    else:
        order_id_topic = False
    ctx_comb = f"{raw} {(conversation_context or '')[-1200:]}"
    if not order_id_topic:
        order_id_topic = (
            _text_has_order_id_context(raw)
            or _text_is_live_order_lookup_intent(raw, conversation_context)
            or _user_denies_pincode_insists_order_id(raw)
        )
    awaiting_oid = _conversation_awaiting_order_id(conversation_context)
    if awaiting_oid or _message_is_order_id_followup_submission(raw, conversation_context):
        early = _bare_order_id_token_from_msg(raw, conversation_context)
        if early:
            return early
    if order_id_topic:
        early = _bare_order_id_token_from_msg(raw, conversation_context)
        if early:
            return early
    if _text_has_delivery_serviceability_intent(raw, conversation_context) and not order_id_topic:
        return None
    if _text_has_pincode_delivery_intent(raw, conversation_context) and not order_id_topic:
        return None
    if _text_is_product_id_lookup_context(raw) and not (
        _text_is_order_tracking_intent_leaf(raw) or _text_has_order_id_context(raw)
    ):
        return None
    labeled = re.search(
        r"\b(?:order\s*id|orderid|order-id|oid)\s*[:\-#]?\s*([A-Za-z0-9]{4,24})\b",
        raw,
        re.IGNORECASE,
    )
    if labeled:
        cand = labeled.group(1).strip()
        if _is_plausible_order_id(cand, context=raw):
            return cand.upper()
    labeled2 = re.search(
        r"\b([0-9]{4,20})\b\s*(?:h\s+|hai\s+)?(?:yeh\s+)?(?:order\s*)?id\b",
        raw,
        re.IGNORECASE,
    )
    if labeled2:
        cand = labeled2.group(1).strip()
        if _is_plausible_order_id(cand, context=raw):
            return cand
    labeled3 = re.search(
        r"\border(?:\s+ki)?\s+id\b.{0,80}\b([0-9]{4,20})\b",
        raw,
        re.IGNORECASE,
    )
    if labeled3:
        cand = labeled3.group(1).strip()
        if _is_plausible_order_id(cand, context=raw):
            return cand
    labeled4 = re.search(
        r"\b(?:asli|sahi|real|actual|correct|ye|yeh)\s+(?:order\s*)?id\s+([0-9]{4,20})\b",
        raw,
        re.IGNORECASE,
    )
    if labeled4:
        cand = labeled4.group(1).strip()
        if _is_plausible_order_id(cand, context=raw):
            return cand
    labeled5 = re.search(
        r"\b([0-9]{4,20})\b\s*(?:h\s+|hai\s+)?(?:asli|sahi|real|actual)\s+(?:order\s*)?id\b",
        raw,
        re.IGNORECASE,
    )
    if labeled5:
        cand = labeled5.group(1).strip()
        if _is_plausible_order_id(cand, context=raw):
            return cand
    for tk in re.findall(r"\b[A-Za-z0-9]{4,24}\b", raw):
        if _is_plausible_order_id(tk, context=raw):
            return tk.upper() if re.search(r"[A-Za-z]", tk) else tk
    return None


def _user_demands_immediate_track_action(text: str) -> bool:
    """Imperative 'track it now' after bot asked — not a hypothetical capability question."""
    raw = _normalize_order_chat_text(text or "")
    if not raw.strip() or re.search(r"\b[0-9]{4,20}\b", raw):
        return False
    tl = f" {raw.lower()} "
    if "track" not in tl and "status" not in tl:
        return False
    if any(x in tl for x in ("dega kya", "dega na kya", "krega kya", "karega kya", "ho sakta", "possible")):
        return False
    return bool(
        re.search(r"\b(?:ab|abhi)\s+(?:kr|kar)\s+de\b", tl)
        or re.search(r"\b(?:kr|kar)\s+de\s+(?:to\s+)?track\b", tl)
        or re.search(r"\btrack\s+(?:kr|kar)\s+de\b", tl)
        or re.search(r"\b(?:krdo|kardo|kr do|kar do)\b", tl)
    )


def _user_asks_hypothetical_tracking_capability(text: str) -> bool:
    """
    User asks IF bot can track another id / ETA / cancel — no id digits in this message.
    e.g. 'agar maan le id laa deu to tu bta dega na kab aayega / cancel to nahi'.
    """
    raw = _normalize_order_chat_text(text or "")
    if not raw.strip() or re.search(r"\b[0-9]{4,20}\b", raw):
        return False
    if _user_demands_immediate_track_action(raw):
        return False
    tl = f" {raw.lower()} "
    if not ("order" in tl or "id" in tl or "track" in tl or "cancel" in tl):
        return False
    hypothetical = (
        "agar maan", "agar man", "agar main", "maan le", "man le", "maan lo", "man lo",
        "suppose", "what if", "if i give", "if i send", "kya tu", "kya tum", "tu bta dega",
        "tu bata dega", "bta dega na", "bata dega na", "btaoge na", "bataoge na",
        "kar sakte", "kr sakte", "kar sakta", "kr sakta", "bta doge", "bata doge",
        "jugaad", "possible hai", "ho sakta", "de paoge", "de paogi", "check kar doge",
        "kr dega kya", "kar dega kya", "krega kya", "karega kya", "dega kya", "dega na kya",
        "kr dega na", "kar dega na", "track kr dega", "track kar dega", "track krega",
        "kyunki kya pata", "kyunki kya", "kya pata",
    )
    capability = hypothetical or (
        ("track" in tl or "cancel" in tl)
        and any(x in tl for x in ("dega kya", "dega na", "krega kya", "karega kya", "kr dega", "kar dega"))
    )
    if not capability:
        return False
    if re.search(r"\b[0-9]{4,20}\b", raw):
        return False
    return True


def _user_announcing_will_provide_order_id(text: str) -> bool:
    """
    User says they WILL send / are sending order id — but no id digits in this message yet.
    e.g. 'mera order id de raha hu track krke bata', 'ek aur order id de ra hu'.
    """
    raw = _normalize_order_chat_text(text or "")
    if not raw.strip():
        return False
    if _bare_order_id_token_from_msg(raw) or _user_asks_hypothetical_tracking_capability(raw):
        return False
    tl = f" {raw.lower()} "
    if not ("order" in tl or "id" in tl or "track" in tl):
        return False
    announce = (
        "de raha", "de ra", "de rhi", "de rahi", "de rha", "de rahi hu", "de raha hu",
        "de ra hu", "de rhi hu", "de rha hu", "deunga", "dunga", "dungi", "dedunga",
        "bhejunga", "bhej raha", "bhejne", "bhej dunga", "id de r", "order id de",
        "ek aur order", "aur order id", "ek aur id", "id laa deu", "id laa du", "laa deu",
        "abhi de", "ab de", "de rha h", "de ra h",
    )
    if any(p in tl for p in announce):
        return True
    if re.search(r"\border\s*id\b", tl) and re.search(r"\bde\s+r", tl) and not re.search(
        r"\b[0-9]{4,20}\b", raw
    ):
        return True
    return False


def _message_is_order_id_followup_submission(current_msg: str, conversation_context: str = "") -> bool:
    """Bare id, 'ye rhi id', 'bhej di', 'asli id' correction — after bot asked."""
    raw = _normalize_order_chat_text((current_msg or "").strip())
    if not raw:
        return False
    if _user_asks_hypothetical_tracking_capability(raw):
        return False
    if _text_is_pincode_serviceability_question(raw, conversation_context):
        return False
    if _text_has_explicit_pincode_subject(raw) and re.search(r"\b[1-9]\d{5}\b", raw):
        return False
    if _text_has_delivery_serviceability_intent(raw, conversation_context) or _text_has_pincode_delivery_intent(
        raw, conversation_context
    ):
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False
    if _conversation_in_pincode_delivery_flow(conversation_context) and re.search(
        r"\b(?:pincode|pin\s*code)\b", raw, re.I
    ):
        return False
    if re.search(r"\b[0-9]{4,20}\b", raw):
        try:
            from services.order_details_flow import _text_has_live_order_lookup_signal

            if _text_has_live_order_lookup_signal(raw):
                return False
        except ImportError:
            pass
    if re.fullmatch(r"[0-9]{4,20}", raw):
        ctx_kind = _resolve_ambiguous_bare_numeric_context(raw, conversation_context)
        if ctx_kind == "pincode":
            return False
        if ctx_kind == "order_id":
            return True
        if re.fullmatch(r"[1-9]\d{5}", raw) and (
            _conversation_in_pincode_delivery_flow(conversation_context)
            or _conversation_bot_asked_for_pincode(conversation_context)
        ):
            return False
        if re.fullmatch(r"[1-9]\d{5}", raw):
            return False
        return True
    if re.fullmatch(r"[A-Za-z0-9]{4,20}", raw) and _is_plausible_order_id(raw, shallow=True):
        return True
    if _bare_order_id_token_from_msg(raw):
        return True
    tl = f" {raw.lower()} "
    if re.search(r"\b[0-9]{4,20}\b", raw):
        try:
            from services.order_details_flow import _text_has_live_order_lookup_signal

            if _text_has_live_order_lookup_signal(raw):
                return False
        except ImportError:
            pass
        if any(
            x in tl
            for x in (
                "order",
                "asli",
                "sahi",
                "galat",
                "galt",
                "wrong",
                "correct",
                "sory",
                "sorry",
                "type ho",
                "mistake",
                "ye le",
                "yeh le",
                "ye h",
                "yeh h",
                "real id",
                "actual",
            )
        ):
            return True
        if re.search(r"\b(?:order\s*)?id\b", tl):
            return True
    if re.search(r"\b(?:asli|sahi|real|actual|correct)\s+(?:order\s*)?id\b", tl):
        return True
    if re.search(r"\bgalt(?:i|a)?\s+(?:type|id|order)\b", tl):
        return True
    markers = (
        "bhej di", "bhej to", "bhej diya", "ye rhi", "yeh rahi", "ye raha", "yeh raha",
        "yahi hai", "yahi h", "id hai", "ye hai", "yeh hai", "lo id", "le id", "ye le",
        "yeh le", "paste kiya", "id paste", "ye rha", "yeh rha", "y rha", "y raha",
        "y rahi", "rha h", "rha hu", "rha hai",
    )
    if re.search(r"\b[0-9]{4,20}\b\s*(?:y\s+)?(?:rha|raha|rahi|rhi)\b", tl):
        return True
    return any(m in tl for m in markers)


def extract_order_id_from_recent_user_lines(
    conversation_context: str, current_msg: str = "", max_lines: int = 4
):
    """Latest order id from recent User: lines only (not assistant cards)."""
    lines = _iter_user_lines_for_context(conversation_context, current_msg)
    found: list[str] = []
    for line in lines[-max_lines:]:
        oid = extract_order_id(line, "")
        if oid:
            found.append(oid)
    return found[-1] if found else None


def resolve_order_id_for_tracking(
    current_msg: str,
    conversation_context: str = "",
    *,
    bot_awaiting_order_id: bool = False,
    ai_extracted: str = "",
):
    """
    Order id for live track API only when the user actually supplied it this turn
    (or clear follow-up after bot asked). Never reuse ids from bot's previous status card.
    """
    msg = _normalize_order_chat_text(current_msg or "")
    if turn_is_catalog_product_lookup(current_msg, msg_en="", ai_route=None):
        return None
    if _user_announcing_will_provide_order_id(msg) or _user_asks_hypothetical_tracking_capability(msg):
        return None

    if _text_has_pincode_delivery_intent(msg, conversation_context):
        return None

    oid = extract_order_id(msg, conversation_context)
    if oid:
        return oid

    ai_id = (ai_extracted or "").strip()
    if ai_id and _is_plausible_order_id(ai_id, context=f"{msg} {conversation_context}"):
        compact = re.sub(r"\s+", "", msg.lower())
        if ai_id.lower() in compact or ai_id in msg:
            return ai_id

    if bot_awaiting_order_id or _message_is_order_id_followup_submission(msg, conversation_context):
        if _conversation_bot_asked_for_pincode(conversation_context):
            return None
        if _conversation_bot_offered_order_id_or_tracking(conversation_context) or bot_awaiting_order_id:
            recent = extract_order_id_from_recent_user_lines(conversation_context, msg)
            if recent:
                return recent

    if _conversation_awaiting_order_id(conversation_context) or bot_awaiting_order_id:
        try:
            from services.conversation_thread_semantics import (
                resolve_explicit_turn_goal_from_message,
            )

            explicit = resolve_explicit_turn_goal_from_message(
                msg, "", conversation_context, None, allow_llm=False
            )
            if explicit in (
                "track",
                "refund_status",
                "payment",
                "order_invoice",
                "order_details",
            ):
                recent = extract_order_id_from_recent_user_lines(conversation_context, msg)
                if recent:
                    return recent
        except ImportError:
            pass
        if _user_demands_immediate_track_action(msg):
            recent = extract_order_id_from_recent_user_lines(conversation_context, msg)
            if recent:
                return recent

    if _user_references_prior_submission(msg):
        latest = extract_latest_order_id_from_user_conversation(conversation_context, msg)
        if latest:
            return latest

    return None


def resolve_live_api_intent_from_conversation(
    conversation_context: str = "",
    ctx_last: str | None = None,
    original_msg: str = "",
    msg_en: str = "",
    ai_route: dict | None = None,
) -> str:
    """
    Pick refund / payment / order for live API after user sends Order ID.
    User message + Groq route win over assistant HTML (e.g. 'Payment: Unpaid' on history cards).
    """
    turn = _normalize_order_chat_text(f"{original_msg} {msg_en}".strip())
    tl = f" {turn.lower()} "

    if _text_is_refund_return_policy_howto(turn):
        return "order"

    route_intent = ""
    route_olk = ""
    if isinstance(ai_route, dict):
        route_intent = (ai_route.get("intent") or "").strip().lower()
        route_olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
        try:
            from services.ai_route_semantics import resolve_order_live_goal_for_turn

            live = resolve_order_live_goal_for_turn(
                ai_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
            if live == "order_invoice":
                return "order"
            if live in ("order_details", "payment"):
                return "order"
            if live == "refund_status":
                return "refund"
            if live == "track":
                return "order"
        except ImportError:
            pass

    if route_olk == "refund_status" or route_intent == "refund":
        return "refund"
    if route_intent == "order":
        return "order"
    if route_intent == "payment":
        return "payment" if _user_explicitly_asks_payment_status(turn) else "order"

    if _text_has_refund_or_return_intent(turn) and (
        _text_needs_order_id_for_refund_or_payment(turn)
        or _text_is_refund_return_status_lookup(turn, conversation_context)
    ):
        return "refund"
    if _text_is_order_tracking_intent(turn):
        return "order"
    if "order" in tl and any(
        x in tl
        for x in (
            "detail", "details", "status", "track", "tracking",
            "kab ", "aayega", "aaega", "bta de", "bata de", "dikha de", "dikhao",
            "bta ", "bata ", "milega", "aaya", "aaya",
        )
    ):
        if not _text_has_refund_or_return_intent(turn):
            return "order"
    if _user_explicitly_asks_payment_status(turn):
        return "payment"

    if _text_is_refund_return_status_lookup(turn, conversation_context):
        user_tail = _conversation_user_text_tail(conversation_context)
        if any(
            x in user_tail
            for x in (
                "refund", "return daal", "return daale", "refund nhi", "refund nahi",
                "paise wapas", "money back",
            )
        ):
            return "refund"
        if ctx_last in ("order", "refund", "payment"):
            return ctx_last

    if ctx_last in ("order", "refund"):
        return ctx_last

    try:
        from services.conversation_thread_semantics import (
            infer_order_thread_goal,
            message_needs_thread_continuation,
        )

        if message_needs_thread_continuation(turn, conversation_context):
            thread = infer_order_thread_goal(
                conversation_context, turn, ctx_last=ctx_last
            )
            if thread == "refund_status":
                return "refund"
            if thread == "payment":
                return "payment"
            if thread in ("track", "order_details", "order_invoice"):
                return "order"
    except ImportError:
        pass
    return "order"


def _conversation_user_text_tail(conversation_context: str, limit: int = 2500) -> str:
    """Recent User: lines only — not assistant order-history HTML."""
    parts: list[str] = []
    for line in (conversation_context or "").splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("user:"):
            parts.append(s[5:].strip())
        elif low.startswith("user "):
            parts.append(s[5:].strip())
    return " ".join(parts)[-limit:].lower()


def _user_explicitly_asks_payment_status(t: str) -> bool:
    """Payment API only when the customer asks about payment — not 'Payment: Unpaid' on a card."""
    tl = f" {_normalize_order_chat_text(t or '').lower()} "
    if not any(x in tl for x in ("payment", "paid", "unpaid", "paisa", "paise", "upi", "cod")):
        return False
    payment_phrases = (
        "payment status", "payment detail", "payment details", "payment method",
        "payment ka", "payment ki", "paid h", "unpaid h", "payment nahi", "payment nhi",
        "paisa cut", "paise cut", "payment ho", "payment hua",
    )
    if any(x in tl for x in payment_phrases):
        return True
    if any(
        x in tl
        for x in ("track", "tracking", "refund", "return", "kab aayega", "kab aaega", "delivery", "shipment")
    ):
        return False
    return any(
        x in tl
        for x in (
            "payment status", "payment detail", "payment details", "payment method",
            "payment ka", "payment ki", "paid h", "unpaid h", "payment nahi", "payment nhi",
            "paisa cut", "paise cut", "payment ho", "payment hua",
        )
    )


def message_needs_live_single_order_lookup(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """User wants live status/refund/track for ONE order — AI-first, heuristics fallback."""
    from services.semantic_intent import resolve_live_lookup_from_ai_or_heuristics

    return resolve_live_lookup_from_ai_or_heuristics(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )


def message_needs_live_single_order_lookup_heuristic(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Keyword/heuristic fallback when LLM routing is unavailable."""
    from services.order_details_flow import message_wants_order_details_or_invoice

    if message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return False
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False
    ctx_kind = _resolve_ambiguous_bare_numeric_context(
        (original_msg or "").strip() or (msg_en or "").strip(),
        conversation_context,
        ai_route=ai_route,
    )
    if ctx_kind == "pincode":
        return False
    if isinstance(ai_route, dict):
        if (ai_route.get("intent") or "").strip().lower() == "pincode_check":
            return False
        if (ai_route.get("numeric_context") or "").strip().lower() == "pincode":
            return False
    if _text_wants_order_history_list_in_chat(comb, conversation_context):
        return False
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if _text_suggests_single_order_status_lookup(comb):
        return True
    if _text_is_live_order_lookup_intent(comb, conversation_context):
        return True
    if extract_order_id(comb, conversation_context) and _text_is_order_tracking_intent(comb):
        return True
    return False


def should_attempt_live_order_api_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Order ID pasted or named — run live track/refund API, not product search."""
    from services.order_details_flow import message_wants_order_details_or_invoice

    if message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return False
    if turn_is_catalog_product_lookup(original_msg, msg_en, ai_route):
        return False
    comb = f"{original_msg} {msg_en}"
    if not user_turn_qualifies_for_live_order_api(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return False
    if message_needs_live_single_order_lookup(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return True
    if _conversation_bot_asked_for_pincode(conversation_context):
        return False
    ctx_kind = _resolve_ambiguous_bare_numeric_context(
        (original_msg or "").strip() or (msg_en or "").strip(),
        conversation_context,
        ai_route=ai_route,
    )
    if ctx_kind == "pincode":
        return False
    if isinstance(ai_route, dict):
        if (ai_route.get("intent") or "").strip().lower() == "pincode_check":
            return False
        if (ai_route.get("numeric_context") or "").strip().lower() == "pincode":
            return False
    if _conversation_bot_offered_order_id_or_tracking(conversation_context) or _conversation_in_order_tracking_flow(
        conversation_context
    ):
        if _message_is_order_id_followup_submission(original_msg, conversation_context):
            return True
        oid = resolve_order_id_for_tracking(
            original_msg.strip() or msg_en.strip(),
            conversation_context,
            bot_awaiting_order_id=True,
        )
        if oid and (
            _text_is_order_tracking_intent(comb)
            or _text_is_refund_return_status_lookup(comb, conversation_context)
            or _user_demands_immediate_track_action(comb)
            or (
                "order" in comb.lower()
                and any(
                    x in comb.lower()
                    for x in (
                        "detail", "details", "status", "track", "bta", "bata",
                        "dikha", "dikhao", "kab ", "aayega", "aaega",
                    )
                )
            )
        ):
            return True
    if _conversation_awaiting_order_id(conversation_context):
        oid = resolve_order_id_for_tracking(
            original_msg.strip() or msg_en.strip(),
            conversation_context,
            bot_awaiting_order_id=True,
        )
        if oid and (
            _message_is_order_id_followup_submission(original_msg, conversation_context)
            or _user_demands_immediate_track_action(comb)
            or _text_is_order_tracking_intent_leaf(comb)
        ):
            return True
    return False


def extract_order_id_from_context(conversation_context: str, current_msg: str = ""):
    """Backward-compatible wrapper — safe user-line scan only."""
    return resolve_order_id_for_tracking(
        current_msg,
        conversation_context,
        bot_awaiting_order_id=_message_is_order_id_followup_submission(current_msg, conversation_context),
    )


def apply_product_id_vs_order_fixes(original_msg: str, msg_en: str, ai_data: dict, ctx=None) -> None:
    """Prevent pro_id / product-id messages from being routed to order tracking."""
    if not ai_data:
        return
    if ai_data.get("_ai_routed"):
        intent = (ai_data.get("intent") or "").strip().lower()
        num_ctx = (ai_data.get("numeric_context") or "").strip().lower()
        if intent != "product" and num_ctx != "product_id":
            return
    comb = f"{original_msg} {msg_en}"
    pid = extract_product_id(comb)
    if not pid and not _text_is_product_id_lookup_context(comb):
        return
    if pid and ctx is not None:
        ctx.setdefault("data", {})
        ctx["data"]["lookup_pro_id"] = pid
        ctx["order_id"] = None
        ctx["awaiting"] = None
    ai_data["intent"] = "product"
    ai_data["is_welfog_related"] = True
    ai_data["needs_order_id"] = False
    if pid:
        ai_data["search_query"] = f"pro_id {pid}"

# --- Roman Hindi / Hinglish product understanding (fills gaps when Groq returns wrong intent) ---
_PRODUCT_QUERY_STOPWORDS = frozenset({
    "h", "hai", "he", "ho", "hun", "hoon", "ky", "kya", "ka", "ke", "ki", "ko", "me", "mai", "main",
    "par", "pe", "se", "tak", "kab", "kaise", "kyu", "kyun", "wala", "wale", "wali", "walon",
    "welfog", "app", "website", "online", "pls", "please", "bhai", "yr", "yrr", "yaar", "dear",
    "mil", "milega", "milegi", "milta", "milti", "milen", "sakta", "skta", "sakti", "skti", "sakte",
        "dikha", "dikhaa", "dikhao", "dikhaan", "dikhado", "dikhaana", "dikhaaoo", "dikhoa", "dikhoah", "de", "do", "dena", "dedo",
    "bata", "batao", "btao", "show", "send", "list", "saare", "sab", "all", "any", "koi", "kuch",
    "chahiye", "chiye", "chaahiye", "lenaa", "lena", "len", "lun", "buy", "need", "want", "looking",
    "for", "the", "a", "an", "is", "are", "there", "have", "has", "stock", "available", "get",
    "categories", "category", "categor", "id", "name", "product", "products", "dikhoa", "dikhoah",
    "hello", "helo", "sun", "suno", "mene", "maine", "pasand", "kiya", "tha", "laga",
    "achha", "accha", "achhe", "bahut", "mujhe", "wo", "wahi", "pehle", "ab", "na",
    "bhi", "badiya", "badhiya", "bta", "btao", "liked", "like", "lagta",
    "lekin", "par", "magar", "abhi", "tak", "tk", "nhi", "nahi", "nahin", "aaya", "aya",
    "aayi", "pahucha", "pahuncha", "mila", "received", "arrived", "delivered", "tracking",
    "track", "status", "delay", "late", "stuck", "pending", "complaint", "galat", "wrong",
    "alag", "aur", "kuchh", "kuch", "order", "ordered", "kiya", "kiye", "mangaya", "mangaye",
})

_POLICY_QUESTION_HINTS = frozenset({
    "refund", "return", "payment", "track", "order id", "orderid", "cancel", "policy", "complaint",
    "support", "grievance", "invoice", "delivery time", "shipping time", "damaged", "wrong item",
})


_PIN_6_RE = re.compile(r"\b([1-9]\d{5})\b")


def _user_denies_pin_in_message_for_different_area(text: str) -> bool:
    """'302012 ka nhi puchh rha, sikar ka puchh rha' — do not use that PIN."""
    if not (text or "").strip():
        return False
    tl = f" {(text or '').lower()} "
    if not re.search(r"\b[1-9]\d{5}\b", text or ""):
        return False
    denies = (
        "ka nhi puchh", "ki nhi puchh", "nahi puchh rha", "nhi puchh rha",
        "nahi puch rha", "nhi puch rha", "not asking about", "wo nahi", "wala nahi",
        "ka nhi bol", "ki nhi bol", "nahi bol rha", "galat pin", "wrong pin",
    )
    return any(d in tl for d in denies)


def message_requests_new_area_without_pin(
    text: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    User asks delivery for a NEW place in THIS message without a PIN here.
    Do not reuse the previous turn's PIN from chat history.
    """
    raw = f"{text or ''} {(msg_en or '').strip()}".strip()
    if not raw:
        return False
    if _user_denies_pin_in_message_for_different_area(raw):
        return True
    if extract_pincode_preferred_from_message(raw, conversation_context):
        return False
    if not (
        _text_has_delivery_or_order_area_intent(raw)
        or _text_is_pincode_serviceability_question_light(raw)
    ):
        return False
    tl = f" {raw.lower()} "
    if re.search(r"\baur\b", tl) and any(
        x in tl
        for x in (
            "pahucha", "pahunch", "deliver", "delivery", "service", "milega",
            "milegi", " me ", " mein ", " par ", " pe ", " ka ", " ki ",
        )
    ):
        return True
    if any(
        x in tl
        for x in (
            "bhi pahucha", "bhi deliver", "bhi delivery", "bhi service",
            "wahan bhi", "yahan bhi", "dusra", "dusre", "alag area", "alag jagah",
            "dusri jagah", "kisi aur", "koi aur",
        )
    ):
        return True
    if re.search(r"\b(?:me|mein|par|pe)\b", tl) and any(
        x in tl for x in ("pahucha", "pahunch", "deliver", "delivery", "service", "dega", "milega")
    ):
        if re.search(r"\baur\s+\w", tl) or not re.search(
            r"\b(?:yaha|yahan|idhar|ih[aā]n)\b", tl
        ):
            return True
    return False


def should_reuse_pincode_from_conversation_history(
    text: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """History PIN only when user clearly refers to a prior submission, not a new city."""
    if message_requests_new_area_without_pin(text, msg_en, conversation_context):
        return False
    raw = f"{text or ''} {(msg_en or '').strip()}".strip()
    if not raw:
        return False
    if re.fullmatch(r"[1-9]\d{5}\s*\??", raw.strip()):
        return True
    if _user_references_prior_submission(raw) or _message_asks_to_recheck_submitted_pincode(raw):
        return True
    if _conversation_bot_asked_for_pincode(conversation_context):
        if extract_pincode_preferred_from_message(raw):
            return True
        if _user_references_prior_submission(raw):
            return True
    return False


def _user_switches_pincode_subject(text: str) -> bool:
    """User negates one PIN and means another (e.g. '302034 ki nhi, 111111 ki')."""
    if not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    pins = _PIN_6_RE.findall(text)
    if len(pins) < 2:
        return False
    switch_markers = (
        "ki nhi", "ki nahi", "nahi kr", "nhi kr", "nahi kar", "nhi kar",
        "nahi baat", "nhi baat", "baat kr rha", "baat kar rha", "bol rha",
        "bol raha", "not talking", "nahi bol", "nhi bol",
    )
    return any(m in tl for m in switch_markers)


def _normalize_pin_candidate(raw: str, *, full_text: str = "") -> str:
    """6-digit Indian PIN; 7-digit typo → first 6 digits only when NOT a product id."""
    digits = re.sub(r"\D", "", raw or "")
    ctx = full_text or raw or ""
    ctx_tl = f" {(ctx or '').lower()} "
    if any(
        x in ctx_tl
        for x in (
            "product id",
            "pro id",
            "id ka product",
            "yeh product id",
            "product id h",
            "iska product",
        )
    ):
        return ""
    if len(digits) == 6 and digits[0] in "123456789":
        return digits
    if len(digits) in (7, 8) and digits[0] in "123456789":
        if re.search(r"\b(?:product\s+id|pro\s*id|id\s+ka\s+product|product\s+dikha)\b", ctx, re.I):
            return ""
        return digits[:6]
    return ""


def extract_all_pincodes_from_text(text: str) -> list[str]:
    if not text:
        return []
    if _digits_in_message_are_order_id_not_pincode(text, ""):
        pins_explicit: list[str] = []
        for m in re.finditer(
            r"\b(?:pincode|pin\s*code|pin)\s*:?\s*([1-9]\d{5})\b",
            text,
            flags=re.IGNORECASE,
        ):
            pins_explicit.append(m.group(1))
        return pins_explicit
    long_numeric_ids = {
        m.group(0) for m in re.finditer(r"\b[0-9]{7,20}\b", text)
    }
    found: list[str] = []
    for m in _PIN_6_RE.finditer(text):
        pin = m.group(1)
        if any(pin in lid and lid != pin for lid in long_numeric_ids):
            continue
        found.append(pin)
    for m in re.finditer(
        r"\bpincode\s+([1-9]\d{6,8})\b", text, flags=re.IGNORECASE
    ):
        p = _normalize_pin_candidate(m.group(1), full_text=text)
        if p:
            found.append(p)
    for m in re.finditer(r"\b([1-9]\d{6,8})\s*(?:per|par|pe)\b", text, flags=re.IGNORECASE):
        p = _normalize_pin_candidate(m.group(1), full_text=text)
        if p:
            found.append(p)
    out: list[str] = []
    for p in found:
        if p not in out:
            out.append(p)
    return out


def message_is_pincode_meta_or_hypothetical(text: str) -> bool:
    """
    User asks about delivery check rules (wrong PIN, without PIN, will you still tell) — not submitting a PIN.
    """
    if not (text or "").strip():
        return False
    tl = f" {(text or '').lower()} "
    markers = (
        "galat pin", "wrong pin", "galat pincode", "wrong pincode",
        "agar me", "agar main", "agar tumhe", "agar aapko", "if i give",
        "kya tum bta", "kya tum bata", "kya bata doge", "kya batoge",
        "bina pincode", "without pin", "pincode ke bina", "pincode na de",
        "pincode nahi de", "pincode nhi de", "pincode de du to bhi",
        "will you still", "kya bhi bata", "sach bataoge", "sahi bataoge",
        "fake pin", "jhuta pin", "galat number", "wrong number de",
    )
    if any(m in tl for m in markers):
        return True
    if "?" in text and any(
        x in tl for x in ("galat", "wrong", "bina", "without", "agar ", "if i")
    ):
        return True
    return False


def extract_malformed_pincode_attempt(text: str, conversation_context: str = "") -> str:
    """
    Digit blob user likely meant as PIN but wrong length (4–5 digits, etc.).
    Skips valid 6-digit and 7–8 digit typos that normalize to 6.
    """
    if not (text or "").strip():
        return ""
    if _digits_in_message_are_order_id_not_pincode(text, conversation_context):
        return ""
    comb = (text or "").strip()
    in_pin_context = (
        _text_has_delivery_or_order_area_intent(comb)
        or _text_has_pincode_delivery_intent(comb, conversation_context)
        or _text_has_delivery_serviceability_intent(comb, conversation_context)
        or _conversation_in_pincode_delivery_flow(conversation_context)
        or _conversation_bot_asked_for_pincode(conversation_context)
        or bool(re.search(r"\b(?:pincode|pin\s*code|pin)\b", comb, re.I))
        or bool(re.search(r"\b(?:per|par|pe|pr)\b", comb, re.I))
    )
    if not in_pin_context:
        return ""
    candidates: list[str] = []
    for m in re.finditer(r"\b([1-9]\d{3,8})\b", comb):
        d = m.group(1)
        norm = _normalize_pin_candidate(d)
        if norm and len(d) in (6, 7, 8):
            continue
        if len(d) in (4, 5) or (len(d) == 6 and d[0] == "0"):
            candidates.append(d)
        elif len(d) in (7, 8) and not norm:
            candidates.append(d)
    return candidates[-1] if candidates else ""


def extract_pincode_preferred_from_message(
    text: str, conversation_context: str = "", *, ai_route: dict | None = None
) -> str:
    """
    PIN user means in THIS message — last pin on correction, not the first digit blob.
    """
    if not text:
        return ""
    if _digits_in_message_are_order_id_not_pincode(
        text, conversation_context, ai_route=ai_route
    ):
        return ""
    pins = extract_all_pincodes_from_text(text)
    if not pins:
        tl = f" {_normalize_order_chat_text(text).lower()} "
        if re.search(r"\b(?:order\s*id|orderid|order-id|oid)\b", tl):
            return ""
        if _text_has_order_id_context(text) and re.search(r"\b[0-9]{7,20}\b", text):
            return ""
        m = re.search(r"\b([1-9]\d{6,8})\b", text)
        if m:
            return _normalize_pin_candidate(m.group(1))
        return ""
    if len(pins) == 1:
        return pins[0]
    if _user_switches_pincode_subject(text):
        return pins[-1]
    m_end = re.search(r"\b([1-9]\d{5})\s*ki\b", text, flags=re.IGNORECASE)
    if m_end:
        return m_end.group(1)
    return pins[-1]


def extract_pincode_from_text(text: str) -> str:
    return extract_pincode_preferred_from_message(text)


def _user_denies_order_history_wants_saved_or_liked(t: str) -> bool:
    """User says NOT order history — they mean wishlist / saved / liked products."""
    if not (t or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(t).lower()} "
    denies_history = any(
        x in tl
        for x in (
            "history nhi", "history nahi", "history ni", "history nahin",
            "not history", "nahi history", "nhi history", "history ka nhi",
            "history ki nhi", "history ki baat nhi", "history ki baat nahi",
        )
    )
    saved_like = any(
        x in tl
        for x in (
            "save", "saved", "like diye", "like kiye", "like kiya", "liked",
            "wishlist", "heart", "pasand", "favourite", "favorite",
        )
    )
    return denies_history and saved_like


def _text_asks_to_view_purchase_or_order_history(t: str) -> bool:
    """User wants to see past purchases / order history — not place a new order."""
    if not (t or "").strip():
        return False
    if message_is_wishlist_like_request(t) or _user_denies_order_history_wants_saved_or_liked(t):
        return False
    tl = f" {_normalize_welfog_typos(t).lower()} "
    history_markers = (
        "order history", "purchase history", "purane order", "past order", "my orders",
        "purchase ki history", "order ki history", "mangaya", "mangaye",
        "order kiye", "order kiya",
    )
    if not any(x in tl for x in history_markers):
        return False
    viewing = (
        "dekh", "dekhu", "dekho", "dekhna", "dekhni", "dikhao", "dikha", "view",
        "kaha", "kahan", "kidhar", "where", "kaise dekhu", "kese dekhu", "milegi",
    )
    return any(v in tl for v in viewing) or _message_has_app_navigation_intent(t)


def _user_rejects_viewing_wants_placement(t: str) -> bool:
    """
    User clarifies they do NOT want to view orders/history — they want to place/buy.
    e.g. 'order dekhna nhi h krna h', 'history nahi order karna hai'.
    Not: casual 'coaching nhi jaa rha' + 'dikha de' product browse.
    """
    if not (t or "").strip():
        return False
    if _text_asks_to_view_purchase_or_order_history(t):
        return False
    if _message_has_catalog_product_signal(t) and any(
        m in f" {(t or '').lower()} "
        for m in ("dikha", "dikhao", "dikhado", "show", "chahiye", "milega", "shopping")
    ):
        return False
    tl = f" {(t or '').lower()} "
    reject_patterns = (
        r"dekhna?\s+(?:nhi|nahi|nahin|not)\b",
        r"(?:nhi|nahi|nahin|not)\s+dekh",
        r"history\s+(?:nhi|nahi|nahin|not)\b",
        r"(?:nhi|nahi|nahin)\s+.*\bhistory\b",
        r"order\s+dekhna?\s+(?:nhi|nahi)",
        r"purane\s+order\s+(?:nhi|nahi)",
        r"order\s+list\s+(?:nhi|nahi)",
    )
    if any(re.search(p, tl) for p in reject_patterns):
        placing = any(
            x in tl
            for x in (
                "krna", "karna", "krte", "karte", "karu", "kru", "kro", "place", "checkout",
                "add to cart", "lena", "leni", "mangna", "mangwana", "buy", "purchase",
                "order kar", "order kr", "order dal",
            )
        )
        return placing
    return False


def _message_overrides_placement_followup(t: str) -> bool:
    """
    User moved on from 'how to place order' — return/refund/damage/complaint on a purchase.
    Must break continue_previous_topic + order_placement_kb from the last turn.
    """
    if _text_has_refund_or_return_intent(t) or _text_has_past_order_complaint_context(t):
        return True
    tl = f" {(t or '').lower()} "
    damage = any(
        x in tl
        for x in (
            "damage", "damaged", "kharab", "kharaab", "defective", "defect", "tuta", "toot",
            "broken", "galat mila", "wrong item", "wrong product", "bekar", "kharab nikla",
        )
    )
    remedy = any(
        x in tl
        for x in (
            "return", "refund", "wapas", "replace", "replacement", "exchange", "claim",
            "return krna", "refund krna", "wapas krna", "badal", "change krna",
        )
    )
    if damage and remedy:
        return True
    if remedy and any(x in tl for x in ("manga liya", "mangaya", "mangwaya", "order kiya", "kharid", "liya tha")):
        return True
    if _text_has_past_order_complaint_context(t) and any(
        x in tl
        for x in (
            "nhi chahiye", "nahi chahiye", "ni chahiye", "chahiye hi nhi", "kya karu", "kya kr",
            "kya kar", "wapas", "refund", "return", "replace",
        )
    ):
        return True
    return False


def _text_has_explicit_how_to_place_order(t: str) -> bool:
    """
    User asks HOW to place/checkout (steps) — even if they mention a product as context.
    e.g. 'iphone cover pasand aaya, order kese kru', 'ise order kaise karu'.
    """
    if _text_asks_to_view_purchase_or_order_history(t):
        return False
    if _user_rejects_viewing_wants_placement(t):
        return True
    if _message_overrides_placement_followup(t):
        return False
    raw = _normalize_order_chat_text((t or "").strip())
    tl = f" {raw.lower()} "
    tracking_hints = (
        "track", "tracking", "status", "kab aayega", "kab aaega", "kab milega",
        "kahan hai order", "order kahan", "nahi aaya", "nahi aya", "nhi aaya",
        "nhi aya", "delay", "stuck", "shipment", "parcel", "courier",
    )
    if any(x in tl for x in tracking_hints):
        return False
    if _pincode_delivery_signal_leaf(t):
        return False
    if _text_is_live_order_lookup_intent(t):
        return False
    if re.search(r"\border\s*id\b", tl) and any(x in tl for x in ("check", "bta", "bata", "refund", "kb tk")):
        return False
    if _message_has_catalog_product_signal(t) or _message_has_generic_shopping_item_signal(t):
        if any(
            m in tl
            for m in ("dikha", "dikhao", "dikhado", "show", "chahiye", "shopping krunga", "shopping karunga")
        ):
            return False
    process_words = (
        "kaise", "kese", "kes", "how to", "how do", "how can", "krte", "karte", "karna", "karu",
        "kru", "kro", "steps", "process", "tarika", "procedure", "bta de", "bata de",
        "btao", "batao", "bta", "samjha", "samjh",
    )

    def _has_process_word(word: str) -> bool:
        if len(word) <= 4:
            return bool(re.search(rf"\b{re.escape(word)}\b", tl))
        return word in tl

    if not any(_has_process_word(x) for x in process_words):
        return False
    if "order" not in tl and "checkout" not in tl and "place order" not in tl:
        return False
    place_phrases = (
        "order kese", "order kese", "order krna", "order karna", "order karu", "order kru",
        "order kaise", "order kes", "order krte", "order karte", "order kaise kar",
        "order kese kr", "ise order", "use order", " place order", "place order", "checkout",
        "place the order", "place my order", "how can i place",
        "add to cart", "cart se", "order kaise karu", "order kese kru", "order kese kru",
        "order krna h", "order karna h", "order krna hai", "order karna hai",
    )
    if any(p in tl for p in place_phrases):
        return True
    if "order" in tl and any(v in tl for v in ("krna", "karna", "karu", "kru", "krte", "karte")):
        return True
    return False


def _text_has_order_placement_intent(t: str) -> bool:
    """
    User asks HOW to place/checkout on Welfog (process steps).
    NOT 'buy me a watch' / 'show suit' — those are product search (AI intent=product).
    """
    if _text_asks_to_view_purchase_or_order_history(t):
        return False
    if _text_has_explicit_how_to_place_order(t):
        return True
    if _user_rejects_viewing_wants_placement(t):
        return True
    if _message_has_catalog_product_signal(t) or _message_has_generic_shopping_item_signal(t):
        return False
    tl = f" {t.lower()} "
    if any(x in tl for x in ("order id", "orderid", "order-id")) and any(
        x in tl for x in ("kya", "kaise", "kahan", "where", "how", "find", "pata")
    ):
        return False
    tracking_hints = (
        "track", "tracking", "status", "kab aayega", "kab aaega", "kab milega",
        "kahan hai order", "order kahan", "nahi aaya", "nahi aya", "nhi aaya",
        "nhi aya", "delay", "stuck", "shipment", "parcel", "courier",
    )
    if any(x in tl for x in tracking_hints):
        return False
    process_words = (
        "kaise", "kese", "kes", "how to", "how do", "how can", "krte", "karte", "karna", "karu",
        "steps", "process", "tarika", "procedure",
    )
    order_words = (
        "order place", "place order", "place an order", "order karne", "order karna",
        "order krna", "order dene", "checkout", "check out", "add to cart", "cart se",
        "place the order", "place my order",
        "order kaise", "order kese", "welfog pe order", "welfog par order",
        "welfog pr order", "welfog pe order", "pr order", "pe order",
    )
    if not any(x in tl for x in process_words):
        return False
    if not any(x in tl for x in order_words) and "order" not in tl:
        return False
    placing_verbs = (
        "krna", "karna", "krte", "karte", "karu", "kru", "kro", "place order", "place an",
        "checkout", "add to cart", "cart se", "buy", "purchase", "lena", "leni",
    )
    viewing_verbs = ("dekh", "dekhu", "dekho", "dekhna", "dikhao", "dikha", "view", "check karu")
    if any(x in tl for x in ("order history", " history", "purane order", "past order", "my orders")):
        if any(v in tl for v in viewing_verbs) and not any(v in tl for v in placing_verbs):
            return False
    if any(x in tl for x in ("dekhna", "dekho", "dikhao", "history", "purane order", "past order")):
        if any(x in tl for x in (" nhi ", " nahi ", " ni ", " not ")):
            if any(v in tl for v in placing_verbs):
                return True
            return False
    return True


def _leaf_non_tracking_order_id_intent(t: str) -> bool:
    """
    Leaf-only: order id present but user wants invoice, refund, address, amount — not shipment track.
    Must not import order_details_flow (breaks recursion with _text_is_order_tracking_intent_leaf).
    """
    raw = _normalize_order_chat_text((t or "").strip())
    if not raw:
        return False
    if not (
        re.search(r"\b\d{4,20}\b", raw) or re.search(r"\borders?\b", raw, re.I)
    ):
        return False
    tl = f" {raw.lower()} "
    if re.search(r"\brefund\b", tl) or "refund status" in tl or "return status" in tl:
        return True
    if re.search(
        r"\b(?:invoice|invoic\w*|bill|receipt|gst|tax\s+invoice)\b", tl, re.I
    ):
        return True
    if re.search(
        r"\baddress\b|konsa\s+laga|lagaya\s+th|pata\b|shipping\s+address|delivery\s+address",
        tl,
        re.I,
    ):
        return True
    if re.search(
        r"\bamount\b|grand\s+total|total\s+amount|kitna\s+tha|kitne\s+ka|kitne\s+rs|"
        r"kitni\s+thi|kitna\s+tha|\bprice\b|\bdaam\b|\bcost\b|"
        r"how\s+much\s+(?:did\s+i\s+)?pay",
        tl,
        re.I,
    ):
        return True
    if re.search(
        r"payment\s+status|payment\s+mode|payment\s+ka|paid\s+or\s+not|unpaid|cod\b",
        tl,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:detail|details|info|summary|vivaran|jankari|जानकारी|विवरण)\b",
        tl,
        re.I,
    ):
        return True
    if re.search(r"\bdetail", tl) and re.search(
        r"\b(?:saari|sari|poori|poore|full|all|complete|chahiye|chahie)\b", tl
    ):
        return True
    return False


def message_is_general_delivery_policy_question(text: str) -> bool:
    """
    Company-wide delivery timeline (no personal order id) → KB, not live tracking API.
    e.g. 'Welfog kitne din me delivery deta hai'
    """
    raw = _normalize_order_chat_text((text or "").strip())
    if not raw:
        return False
    if re.search(r"\b\d{4,20}\b", raw):
        return False
    if _current_turn_has_order_id(raw):
        return False
    tl = f" {raw.lower()} "
    policy_time = (
        "kitne din",
        "kitna time",
        "kitne din me",
        "kitne dino",
        "generally",
        "usually",
        "normally",
        "typically",
        "on average",
    )
    if any(m in tl for m in policy_time) and any(
        x in tl
        for x in (
            "welfog",
            "delivery",
            "shipping",
            "courier",
            "dispatch",
            "deliver",
            "order aata",
            "order aati",
            "order milt",
        )
    ):
        if not re.search(r"\b[1-9]\d{5}\b", raw):
            return True
    try:
        from services.order_details_flow import _text_wants_order_invoice

        if _text_wants_order_invoice(raw, ""):
            return False
    except ImportError:
        pass
    if _text_is_pincode_serviceability_question_light(raw):
        return False
    if not any(m in tl for m in policy_time):
        return False
    if any(
        x in tl
        for x in (
            "welfog",
            "delivery",
            "shipping",
            "courier",
            "dispatch",
            "deliver",
            "order aata",
            "order aati",
            "order milt",
        )
    ):
        return True
    return False


def _text_is_order_tracking_intent_leaf(t: str) -> bool:
    """Order tracking markers only — never calls order_details heuristics (no recursion)."""
    try:
        from services.turn_intent_gate import (
            message_denies_order_or_tracking_topic,
            message_is_bot_latency_complaint,
            message_is_hostile_or_insult_turn,
        )

        if (
            message_is_hostile_or_insult_turn(t)
            or message_is_bot_latency_complaint(t)
            or message_denies_order_or_tracking_topic(t)
        ):
            return False
    except ImportError:
        pass

    raw = _normalize_order_chat_text((t or "").strip())
    if _text_asks_order_history(raw) or _text_wants_order_history_list_in_chat(raw):
        return False
    if _text_is_undelivered_order_complaint(raw):
        return True
    if _text_is_delivery_serviceability_hypothetical(raw):
        return False
    if _text_is_pincode_serviceability_question_light(raw):
        return False
    if _text_has_explicit_pincode_subject(raw) and re.search(r"\b[1-9]\d{5}\b", raw):
        return False
    if _text_has_order_placement_intent(raw):
        return False
    if _leaf_non_tracking_order_id_intent(raw):
        return False
    tl_refund_guard = f" {raw.lower()} "
    if _text_has_refund_or_return_intent(raw) or re.search(
        r"\brefund\b", tl_refund_guard
    ) or re.search(r"\breturn\b", tl_refund_guard):
        return False
    if native_order_tracking_request(raw) or multilingual_order_tracking_match(raw, raw):
        return True
    tl = f" {raw.lower()} "
    markers = (
        "track", "tracking", "trck", "trak", "traking", "teck", "order status", "order id",
        "orderid", "where is my order", "where my order", "delivery status", "shipment",
        "kab aayega", "kab aaega", "kab aaega", "kab tak", "kab tk", "kb tk", "kab milega", "kb aayega",
        "pahuch jayga", "pahuch jayega", "pahunch jayga", "pahunch jayega", "pahunch jaega",
        "mere pass kab", "mere paas kab", "pass kab tk", "paas kab tk",
        "kahan hai order", "order kahan", "order kaha", "parcel", "package kahan",
        "track ese", "track kar", "track kr", "trck kr", "trck kar", "ese track",
        "track kaise", "tracking ese", "track ki", "track krke", "track krke bata",
        "order track", "order ko track", "order trck", "mera order", "mere order",
        "order ka status", "order status", "mujhe order", "order milega", "order aayega",
        "iska status", "is id ka status", "id ka status", "id ki status",
        "status bata", "status btao", "status check", "live status",
        "dekh ke bta", "dekh ke bata", "dekh ke btao", "dekh kr bta", "dekh kar bata",
        "check krke bta", "check karke bata",
        "package kahan hai", "parcel kahan hai", "kab deliver", "deliver kab",
        "courier", "out for delivery", "shipped", "dispatch", "jankari nikal",
        "update bata", "kya hua order", "order update", "delivery update",
        # Order not received / delayed (Roman Hindi, typos)
        "order nhi aaya", "order nahi aaya", "order nhi aya", "order nahi aya",
        "order nhi aayi", "order nahi aayi", "order nhi mila", "order nahi mila",
        "order aa nahi", "order aaya nahi", "order aaya nhi", "order aaya nahi",
        "order abhi tak", "abhi tak order", "ab tak order", "order late", "order delay",
        "order pending", "order stuck", "order nahi aaya", "order nahin aaya",
        "delivery nahi hui", "delivery nhi hui", "parcel nahi aaya", "package nahi aaya",
        "nhi pahucha", "nahi pahucha", "nhi pahuncha", "nahi pahuncha", "pahucha mere",
        "pahuncha mere", "mere pass nahi", "mere paas nahi", "ab bhi nhi pahucha",
        "abhi bhi nahi pahucha", "product order kiya", "order kiya tha",
    )
    if any(m in tl for m in markers):
        return True
    if _text_is_undelivered_order_complaint(raw):
        return True
    if ("product" in tl or "order" in tl) and any(
        x in tl
        for x in (
            "nhi aaya", "nahi aaya", "nhi aya", "nahi aya", "nhi mila", "nahi mila",
            "nhi aayi", "nahi aayi", "abhi tak nahi", "abhi tak nhi", "ab tak nahi", "ab tak nhi",
            "aa nahi", "aaya nahi", "aaya nhi", "late", "delay", "stuck", "pending",
            "nahi pahucha", "nhi pahucha", "nahi pahuncha", "nhi pahuncha",
        )
    ):
        return True
    return False


def _text_is_order_tracking_intent(t: str) -> bool:
    """Track/status of an existing order — not 'can I place an order from X'."""
    if getattr(_order_tracking_guard, "active", False):
        return _text_has_light_order_tracking_markers(t) or _text_is_undelivered_order_complaint(t)
    _order_tracking_guard.active = True
    try:
        if _text_is_order_tracking_intent_leaf(t):
            return True
        from services.order_details_flow import (
            _lightweight_details_or_invoice_signal,
            _user_rejects_invoice_wants_details,
        )

        if _user_rejects_invoice_wants_details(t or "", ""):
            return False
        if _lightweight_details_or_invoice_signal(t or ""):
            return False
        return False
    finally:
        _order_tracking_guard.active = False


def _text_needs_order_id_for_tracking(t: str) -> bool:
    """Return False when the user wants steps / guidance, not an immediate live status lookup."""
    if message_is_user_feedback_or_closing(t):
        return False
    tl = f" {t.lower()} "
    if _text_has_order_placement_intent(t):
        return False
    if _user_announcing_will_provide_order_id(t):
        return True
    # Bare digits only — never call full extract_order_id (wishlist/tracking how-to recursion).
    if _bare_order_id_token_from_msg(t):
        return True

    track_core = any(
        x in tl
        for x in (
            "track",
            "tracking",
            "order status",
            "delivery status",
            "shipment",
            "parcel",
            "package",
            "order track",
            "track order",
            "mera order",
            "mere order",
            "order nhi aaya",
            "order nahi aaya",
            "order nhi aya",
            "order nahi aya",
            "order nhi mila",
            "order nahi mila",
            "order abhi tak",
            "abhi tak order",
        )
    ) or (
        "order" in tl
        and any(
            x in tl
            for x in (
                "nhi aaya", "nahi aaya", "nhi aya", "nahi aya", "nhi mila", "nahi mila",
                "abhi tak", "ab tak", "aa nahi", "aaya nahi", "aaya nhi",
            )
        )
    )
    how_to = any(
        x in tl
        for x in (
            "how to",
            "how can i",
            "how do i",
            "how do we",
            "kaise",
            "kese",
            "kaise kru",
            "kese kru",
            "kaise karu",
            "kese karu",
            "kaise kare",
            "kese kare",
            "kaise krte",
            "kese krte",
            "kaise hot",
            "kese hot",
            " steps",
            "tracking process",
            "track process",
            "order tracking process",
            "tarika",
            "tarike",
            "tareeka",
            "tareeke",
            "guide",
            "batao",
            "btao",
            "bataye",
            "bataiye",
            "tell me",
            "what is the way",
            "kahan se",
            "kaha se",
            "kaha mile",
            "kahan mile",
            "kidhar",
            " tips",
            "tutorial",
        )
    )
    if track_core and how_to:
        return False

    # Imperative: "track karo/krde" = live status request (NOT how-to)
    if re.search(
        r"\b(?:track|trck|trak)\s+(?:karo|kr|krde|krke|krna|karde|kardo|kare|karna)\b",
        tl,
    ):
        return True
    if re.search(r"\b(?:track|trck)\s+krke\b", tl):
        return True
    if re.search(r"\b(?:order|mera\s+order|mere\s+order)\s+.{0,20}(?:track|trck|trak)\b", tl):
        if not how_to:
            return True

    guidance_phrases = (
        "how to track",
        "how can i track",
        "order status please",
        "order status kaise",
        "track order kaise",
        "order track kaise",
        "track karna",
        "tracking kaise",
        "order tracking kaise",
        "tracking process",
        "order tracking process",
        "track karne ka",
        "track karne ka tarika",
        "how to track my order",
        "how to check order status",
        "how to check order",
        "kahan se track",
        "kaha se track",
        "kaha se order track",
        "track karne ka",
        "order status dekhen",
        "order id kaha",
        "order id kahan",
        "order id kaise",
        "order id kese",
        "where to find order id",
        "how to find order id",
        "how to get order id",
        "order id nahi pata",
        "order id nahin pata",
        "track ese",
        "ese track",
        "track kaise",
        "track kese",
        "tracking ese",
        "tracking kese",
        "order track ese",
        "order ko track",
        "track process",
        "track kaise hota",
        "track kese hota",
        "order tracking process",
        "track kaise kru",
        "track kese kru",
        "order track kaise hota hai",
        "order track kese hota hai",
        "how to track",
        "track kaise hota hai",
        "track kese hota hai",
        "order track kaise",
        "order track kese",
    )
    if any(x in tl for x in guidance_phrases):
        return False
    if any(x in tl for x in ("place an order", "how do i place", "how do we place", "how to place")):
        return False
    return True


def _text_is_tracking_howto_request(t: str) -> bool:
    """
    User wants steps to track (kaise/kese), not to paste an Order ID yet.
    Covers 'kese karu track', 'track kaise kare', etc.
    """
    tl = f" {t.lower()} "
    if _text_has_order_placement_intent(t):
        return False
    if not re.search(r"\b(track|tracking)\b", tl):
        return False
    explicit = (
        "how to track",
        "how can i track",
        "how do i track",
        "track kaise",
        "track kese",
        "kaise track",
        "kese track",
        "kese karu track",
        "kaise karu track",
        "kese kru track",
        "kaise kru track",
        "track karu",
        "track karna",
        "track karna hai",
        "tracking kaise",
        "tracking kese",
        "order track kaise",
        "order track kese",
        "track order kaise",
        "track order kese",
        "track kaise kare",
        "track kese kare",
        "track kaise kru",
        "track kese kru",
    )
    if any(p in tl for p in explicit):
        return True
    return not _text_needs_order_id_for_tracking(t)


def _text_is_order_id_help_request(t: str) -> bool:
    try:
        from services.turn_intent_gate import (
            message_denies_order_or_tracking_topic,
            message_is_bot_latency_complaint,
        )

        if message_denies_order_or_tracking_topic(t) or message_is_bot_latency_complaint(t):
            return False
    except ImportError:
        pass
    tl = f" {t.lower()} "
    has_order_id = any(x in tl for x in ("order id", "orderid", "order-id")) or (
        "order" in tl and re.search(r"\bid\b", tl)
    )
    if not has_order_id:
        return False
    help_phrases = (
        "kaha", "kahan", "kaise", "kese", "find", "dhoond", "where", "pata", "nahin pata",
        "kahan se", "kaise nikaal", "kaise nikal", "nikal", "nikale", "nikalte", "nikaal",
        "nahi pata", "puchh", "pooch", "puch rha", "milega", "milta", "kya hota",
    )
    if any(x in tl for x in help_phrases):
        return True
    return False


def _is_conversation_acknowledgment(text: str) -> bool:
    """Short ack after bot answered ('achha theek chal') — not a fresh greeting."""
    tl = f" {(text or '').lower()} "
    ack_tokens = (
        "achha", "accha", "theek", "thik", "ok", "okay", "okee", "okie", "chal", "chalo",
        "ha", "haan", "han", "ji", "sahi", "thik h", "theek h", "ok h",
    )
    words = [w for w in re.findall(r"[a-z]+", tl) if w]
    if not words or len(words) > 6:
        return False
    if any(x in tl for x in ("order", "product", "refund", "delivery", "track", "wishlist")):
        return False
    return all(any(w == a or w.startswith(a) for a in ack_tokens) for w in words)


def _text_needs_order_id_for_refund_or_payment(t: str) -> bool:
    """Return False for refund/payment/cancel queries that are about policy/help, not a specific status lookup.
    Works across English, Hindi, Hinglish.
    """
    tl = f" {t.lower()} "
    status_phrases = (
        "status", "track", "tracking", "kab aayega", "kab aaega", "kab milega", "kaha hai", "kaha hai order",
        "kahan hai", "order kahan", "where is my refund", "refund status", "payment status", "transaction status",
        "order status", "current status", "update", "updated",
    )
    help_phrases = (
        "policy", "policies", "kaise", "kya", "process", "procedure", "how to", "kaise kar", "kaise mil", "order cancel",
        "cancel order", "refund kaise", "return kaise", "exchange kaise", "payment kaise", "refund policy", "return policy",
        "exchange policy", "cancel policy", "cancel karna", "refund karna", "return karna", "exchange karna",
        "refund krna", "return krna", "cancel krna", "order cancel kaise", "cancel order kaise", "refund karna hai", "return karna hai",
        "cancel karna hai", "payment kaise", "payment karna",
        # Additional Hindi/Hinglish patterns for policy/help detection
        "kaise kru", "kaise kar sakte", "kaise ho sakta", "process kya hai", "kya process hai",
        "refund process", "cancel process", "payment process", "paise wapsi process", "refund ka procedure",
        "cancel ka tarika", "refund ka tarika", "kaise socha hai", "kaise dekh sakte",
    )
    if any(x in tl for x in help_phrases) and not any(x in tl for x in status_phrases):
        return False
    return True


def _text_has_delivery_or_order_area_intent(t: str, conversation_context: str = "") -> bool:
    """
  Delivery serviceability / can I order from this place — Roman Hindi + English.
  Must run BEFORE broad 'product' heuristics so city + delivery does not hit product search.
    """
    raw = (t or "").strip()
    if re.search(r"\b[1-9]\d{5}\b", raw):
        tl_oid = f" {raw.lower()} "
        if re.search(r"\b(?:order\s*id|orderid)\b", tl_oid) and not _text_has_explicit_pincode_subject(raw):
            return False
        if _text_is_live_order_lookup_intent(raw, conversation_context):
            return False
    tl = f" {t.lower()} "
    if any(x in tl for x in [" pincode", " pin code", " pin-code", "zip code", "postal code"]):
        return True
    if any(
        x in tl
        for x in [
            "delivery", "delevery", "delivry", "deliver", "shipping", "courier", "dispatch",
            "ship to", "ship kar",
        ]
    ):
        return True
    if any(
        x in tl
        for x in [
            "order kr", "order kar", "order skt", "order sak", "order skte", "order sakte",
            "order kru", "order karu", "place order", "order dal", "order dunga",
            "mangwa", "mangwa sak", "manga sak", "mangwa skt", "mang skt",
            "aa jayegi", "aa jaayegi", "aayegi", "aayega", "pahuch", "pohuch", "pahuchega",
            "de deg", "de dega", "dedega", "dega na", "milegi delivery", "delivery milegi",
            "service area", "deliver ho", "deliver ho sak", "serviceable",
        ]
    ):
        return True
    if "welfog" in tl and "service" in tl and any(
        x in tl
        for x in (
            " me ", " mein ", " par ", " pe ", " per ", " in ", " at ",
            " de dega", "dega", "deta h", "milega", "milegi", "deliver", "provide",
        )
    ):
        return True
    if any(
        x in tl
        for x in (
            " mil jaygi", " mil jayega", " mil jaaygi", " mil jaayega",
            " kya service", "service mil", " service ",
        )
    ) and re.search(r"\b(?:me|mein|par|pe|per)\b", tl):
        return True
    if re.search(r"\baur\b", tl) and re.search(r"\b(?:me|mein|par|pe|per)\b", tl):
        return True
    if "welfog" in tl and any(x in tl for x in ["use kr", "use kar", "use skt", "use sak", "chal sak", "chalega", "chlega"]):
        if any(x in tl for x in [" se ", " me ", " mein ", "wale", "walo", "city", " pin", " pincode"]):
            return True
    return False


def _text_has_delivery_serviceability_intent(t: str, conversation_context: str = "") -> bool:
    """
    User asks IF Welfog delivers / services an area (PIN) — NOT tracking an existing order.
    Covers: friend lives in 302034, order karna chahta, delivery milegi, Tamil/Hindi/English mix.
    """
    raw = (t or "").strip()
    if not raw and not (conversation_context or "").strip():
        return False
    if _conversation_awaiting_order_id(conversation_context):
        return False
    if _digits_in_message_are_order_id_not_pincode(raw, conversation_context):
        return False
    comb = f"{raw} {(conversation_context or '')[-900:]}".strip()
    pin = extract_pincode_preferred_from_message(raw) or extract_pincode_preferred_from_message(comb)
    if not pin:
        return False
    tl = f" {comb.lower()} "
    tracking_only = (
        "track", "tracking", "trck", "order status", "order id", "orderid",
        "kab aayega", "kab aaega", "kab milega", "refund", "kb tk", "kab tak",
        "nahi aaya", "nahi aya", "nahi mila", "where is my order", "live status",
        "checked id", "shipment status", "courier status", "check kr", "check kar",
        "bta de", "bata de", "btao", "batao",
    )
    if any(x in tl for x in tracking_only):
        return False
    if pin and "welfog" in tl:
        return True

    service_markers = (
        "deliver", "delevery", "delivery", "pahucha", "phocha", "pahunchega", "pahunch",
        "milega", "milegi", "service", "available", "ho jayega", "ho jayegi", "serviceable",
        "pincode", "pin code", "area", "yaha", "yahan", "waha", "wahan", "rhta", "rehta",
        "rhte", "rhti", "friend", "dost", "order kr", "order kar", "order karna", "order krna",
        "chahta", "chahte", "chahiye", "welfog se", "se order", "place order", "buy",
        "puchh", "pooch", "puch rha", "available ah", "irukka", "irukan", "vasikk", "vangala",
        "vangalam", "eduthu", "eduka", "area la", "area me", "areala", "ukka area",
        "pannanum", "pannalam", "pannan", "order pan", "order pann", "irunthu", "iruntha",
        "nanaku", "nanu", "ukka", "doubt iruke", "kuduthuten", "kudutten", "koduthuten",
    )
    if any(x in tl for x in service_markers):
        return True
    if "order" in tl and any(
        x in tl for x in ("chahta", "chahte", "chahiye", "karna", "krna", "kar sak", "kr sak", "milega", "dega")
    ):
        if not any(x in tl for x in ("track", "tracking", "status", "kab ")):
            return True
    if re.search(r"\b(per|par|pe)\b", tl) and pin:
        return True
    return False


def _conversation_in_pincode_delivery_flow(conversation_context: str) -> bool:
    """Recent chat was pincode / delivery serviceability — not order tracking."""
    if not (conversation_context or "").strip():
        return False
    if _conversation_bot_asked_for_pincode(conversation_context):
        return True
    if _conversation_awaiting_order_id(conversation_context):
        return False
    tail = (conversation_context or "")[-4500:].lower()
    if _conversation_bot_offered_order_id_or_tracking(tail) and "pincode" not in tail and "pin code" not in tail:
        if "delivery" not in tail and "serviceability" not in tail:
            return False
    markers = (
        "pincode",
        "pin code",
        "check_pincode",
        "good news",
        "not available to deliver",
        "delivery available",
        "serviceability",
        "extracted_pincode",
        "pincode_check",
        "distance:",
        "place your order",
        "6-digit pin",
        "checked delivery",
        "representative pin",
        "wf-pin-root",
        "wf-pin",
    )
    if any(m in tail for m in markers):
        return True
    if "delivery for " in tail and re.search(r"\b[1-9]\d{5}\b", tail):
        return True
    return False


def _conversation_recent_text_window(conversation_context: str, *, window: int = 7) -> str:
    """Last N user+assistant turns as one lowercase blob for topic inference."""
    if not (conversation_context or "").strip():
        return ""
    lines = [ln.strip() for ln in (conversation_context or "").splitlines() if ln.strip()]
    recent = lines[-max(window * 2, 4) :]
    return "\n".join(recent).lower()


def _resolve_ambiguous_bare_numeric_context(
    current_msg: str,
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    window: int = 7,
) -> str:
    """
    When the latest message is mostly digits, infer thread from recent chat (5–7 turns).
    Returns: pincode | order_id | product_id | none
    """
    raw = (current_msg or "").strip()
    if not raw:
        return "none"
    if not re.fullmatch(r"[0-9]{4,20}", raw) and not re.fullmatch(r"[A-Za-z0-9]{4,20}", raw):
        return "none"

    if isinstance(ai_route, dict):
        nc = (ai_route.get("numeric_context") or "").strip().lower()
        if nc in ("pincode", "order_id", "product_id"):
            return nc
        intent = (ai_route.get("intent") or "").strip().lower()
        if intent == "pincode_check":
            return "pincode"
        if intent in ("order", "refund", "payment"):
            return "order_id"
        if intent == "product":
            return "product_id"

    if _conversation_bot_asked_for_pincode(conversation_context):
        return "pincode"
    if _conversation_awaiting_order_id(conversation_context):
        return "order_id"

    tail = _conversation_recent_text_window(conversation_context, window=window)
    if re.fullmatch(r"[1-9]\d{5}", raw):
        pin_markers = (
            "pin code",
            "pincode",
            "6-digit",
            "6 digit",
            "delivery check",
            "serviceability",
            "delivery available",
            "ask_pincode",
            "check_pincode",
            "good news",
            "not available to deliver",
        )
        order_markers = (
            "order id",
            "orderid",
            "track",
            "tracking",
            "live status",
            "live tracking",
            "refund",
            "payment status",
            "ask_order_id",
        )
        pin_score = sum(1 for m in pin_markers if m in tail)
        order_score = sum(1 for m in order_markers if m in tail)
        if _conversation_in_pincode_delivery_flow(conversation_context):
            pin_score += 2
        if _conversation_in_order_tracking_flow(conversation_context):
            order_score += 1
        if pin_score > order_score:
            return "pincode"
        if order_score > pin_score:
            return "order_id"

    if _conversation_in_pincode_delivery_flow(conversation_context) and re.fullmatch(r"[1-9]\d{5}", raw):
        return "pincode"
    if _conversation_bot_offered_order_id_or_tracking(conversation_context) and _conversation_in_order_tracking_flow(
        conversation_context
    ):
        return "order_id"
    return "none"


def message_is_bare_numeric_submission(text: str) -> bool:
    """Latest turn is only digits / short alphanumeric id (pincode, order id, SKU)."""
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(
        re.fullmatch(r"[1-9]\d{5}\s*\??", raw)
        or re.fullmatch(r"[0-9]{4,20}", raw)
        or re.fullmatch(r"[A-Za-z0-9]{4,20}", raw)
    )


def classify_bare_numeric_turn(
    current_msg: str,
    conversation_context: str = "",
    *,
    ctx: dict | None = None,
    ai_route: dict | None = None,
) -> str:
    """
    Disambiguate bare numeric submissions using session lock + recent chat.
    Returns: pincode | order_id | product_id | none
    """
    raw = (current_msg or "").strip()
    if not raw or not message_is_bare_numeric_submission(raw):
        return "none"

    if isinstance(ctx, dict):
        awaiting = (ctx.get("awaiting") or "").strip().lower()
        if awaiting == "pincode":
            return "pincode"
        if awaiting == "order_id":
            return "order_id"
        last = (ctx.get("last") or "").strip().lower()
        data = dict(ctx.get("data") or {})
        if last == "pincode" and re.fullmatch(r"[1-9]\d{5}\s*\??", raw):
            return "pincode"
        topic = (data.get("topic_mode") or "").strip().lower()
        if topic == "pincode_check" and re.fullmatch(r"[1-9]\d{5}\s*\??", raw):
            return "pincode"
        if data.get("lookup_pro_id") or data.get("lookup_sku"):
            return "product_id"
        pending = (data.get("pending_action") or "").strip().lower()
        if pending in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            if re.fullmatch(r"[0-9]{4,20}", raw) or re.fullmatch(r"[0-9]{7,20}", raw):
                return "order_id"

    if re.fullmatch(r"[1-9]\d{5}\s*\??", raw):
        if _conversation_bot_asked_for_pincode(conversation_context):
            return "pincode"
        kind = _resolve_ambiguous_bare_numeric_context(
            raw, conversation_context, ai_route=ai_route
        )
        if kind == "pincode":
            return "pincode"
        if _conversation_in_pincode_delivery_flow(conversation_context):
            return "pincode"
        return kind

    if re.fullmatch(r"[0-9]{7,20}", raw):
        if _conversation_awaiting_order_id(conversation_context):
            return "order_id"
        if _conversation_in_order_tracking_flow(conversation_context):
            return "order_id"
        kind = _resolve_ambiguous_bare_numeric_context(
            raw, conversation_context, ai_route=ai_route
        )
        return kind if kind != "none" else "order_id"

    return _resolve_ambiguous_bare_numeric_context(
        raw, conversation_context, ai_route=ai_route
    )


def set_pincode_await_context(ctx: dict, ai_route: dict | None = None) -> None:
    """Lock session for next bare-PIN reply — do not wipe with full reset_context."""
    if not isinstance(ctx, dict):
        return
    ctx["intent"] = None
    ctx["awaiting"] = "pincode"
    ctx["last"] = "pincode"
    ctx["order_id"] = None
    ctx.setdefault("data", {})
    ctx["data"]["topic_mode"] = "pincode_check"
    if isinstance(ai_route, dict) and ai_route:
        ctx["data"]["ai_route"] = ai_route


def mark_pincode_delivery_completed(
    ctx: dict,
    *,
    pin: str = "",
    ai_route: dict | None = None,
) -> None:
    """Delivery check answered — release pincode thread so wishlist/order/product can run."""
    if not isinstance(ctx, dict):
        return
    ctx["awaiting"] = None
    ctx["last"] = None
    data = ctx.setdefault("data", {})
    data["topic_mode"] = "pincode_check"
    if pin:
        data["last_pincode"] = pin
    if isinstance(ai_route, dict) and ai_route:
        data["ai_route"] = ai_route


def _text_has_pincode_delivery_intent(t: str, conversation_context: str = "") -> bool:
    """
    Delivery / service at a PIN — must win over order-id tracking and product search.
    """
    raw = (t or "").strip()
    if not raw and not (conversation_context or "").strip():
        return False
    if _text_has_delivery_serviceability_intent(raw, conversation_context):
        return True
    if _conversation_awaiting_order_id(conversation_context):
        return False
    comb_oid = f"{raw} {(conversation_context or '')[-600:]}"
    if _text_has_order_id_context(comb_oid):
        return False
    if re.fullmatch(r"[1-9]\d{5}\s*\??", raw.strip()):
        if (
            _conversation_bot_asked_for_pincode(conversation_context)
            or _conversation_in_pincode_delivery_flow(conversation_context)
        ):
            return True
    if re.fullmatch(r"[0-9]{4,20}", raw.strip()):
        if re.fullmatch(r"[1-9]\d{5}\s*\??", raw.strip()):
            return False
        return False
    if re.search(r"\b[0-9]{4,20}\b\s*(?:y\s+)?(?:rha|raha|rahi|rhi)\b", f" {raw.lower()} "):
        return False
    comb = f"{raw} {(conversation_context or '')[-800:]}".strip()

    if _text_is_order_tracking_intent_leaf(raw) and not any(
        x in f" {raw.lower()} "
        for x in (
            "pincode", "pin code", "delivery", "delevery", "delivry", "deliver",
            "service", "pahucha", "phocha", "pahunchega", "serviceability",
        )
    ):
        return False

    if _text_has_delivery_or_order_area_intent(comb):
        return True

    tl = f" {raw.lower()} "
    comb_tl = f" {comb.lower()} "
    pin = extract_pincode_preferred_from_message(raw) or extract_pincode_preferred_from_message(comb)
    if pin:
        if "welfog" in comb_tl:
            return True
        if any(
            m in tl
            for m in (
                "pincode", "pin code", "pin-code", "zip", "postal",
                "delivery", "delevery", "delivry", "deliver", "service",
                "pahucha", "phocha", "pahunchega", "phoch", "milega", "milegi",
                "ho jayegi", "ho jayega", "service dega", "serviceable",
                " per ", " par ", " pe ", " per?", " par?", " pe?",
                " ispe ", " isme ", " is par ", " is per ",
                "wale", "wala", "yaha", "yahan", "idhar", "uspe", "us par",
                "phocha dega", "pahucha dega",
                "pannanum", "pannalam", "areala", "irunthu", "iruntha", "area la",
                "friend", "dost", "nanaku", "doubt",
            )
        ):
            return True
        if "product" in tl and any(
            x in tl for x in ("phocha", "pahucha", "deliver", "delevery", "milega", "dega", "degi", "aayega")
        ):
            return True
        if re.fullmatch(r"[1-9]\d{5}\s*\??", raw.strip()):
            return True
        if re.search(r"\b[1-9]\d{5}\b", raw) and re.search(r"\b(per|par|pe)\b", tl):
            return True

    if any(
        x in tl
        for x in (
            "pincode tha", "ye pincode", "yeh pincode", "yh pincode",
            "galat order id", "order id nahi", "pin code tha", "ye pin tha",
        )
    ):
        return True

    if _conversation_in_pincode_delivery_flow(conversation_context):
        if pin or "pincode" in tl or "delivery" in tl or "delevery" in tl:
            if not _text_is_order_tracking_intent(raw):
                return True
            if "pincode" in tl or "pin code" in tl:
                return True

    return False


def _user_references_prior_submission(text: str) -> bool:
    """
    User says they already sent id/pincode earlier — use latest USER value from chat, not bot cache.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} "
    markers = (
        "de di thi", "de diya", "de di", "de chuka", "de chuki", "de chuke",
        "bhej di", "bheja tha", "bheji thi", "diya tha", "di thi", "dedi thi",
        "abhi di", "abhi de", "abhi bhej", "pehle di", "pehle de", "pehle bhej",
        "pehle bola", "maine de", "mene de", "humne de", "tumhe de", "aapko de",
        "already gave", "already sent", "just sent", "just gave", "likha tha",
        "paste kiya", "paste kr", "id di", "pin di", "pincode di", "de to di",
        "de to diya", "de to diye", "to de diya", "diya to", "diya na",
        "to de di", "galat samjhe", "galat samjha", "upar di", "upar de",
        "kuduthuten", "kudutten", "koduthuten", "kuduthuten pola", "kudutten pola",
        "already gave", "already sent", "just sent", "just gave",
        "wo de di", "wahi id", "wahi pin", "usi id", "usi pin",
        "pincode bheja", "pin code bheja", "pin bheja", "pincode bheji",
        "pin code bheji", "mene pincode", "maine pincode", "mene pin code",
        "pincode to bheja", "pin code to bheja", "pin to bheja",
    )
    if any(m in tl for m in markers):
        return True
    if re.search(r"\b(de|di|diya|diye)\s+(thi|tha|the|chuka|chuki|chuke)\b", tl):
        if any(x in tl for x in ("id", "pin", "pincode", "order", "number", "code")):
            return True
    return False


def _iter_user_lines_for_context(conversation_context: str, current_msg: str = "") -> list[str]:
    lines: list[str] = []
    for line in (conversation_context or "").splitlines():
        low = line.strip().lower()
        if low.startswith("user:") or low.startswith("user "):
            lines.append(line)
    if (current_msg or "").strip():
        lines.append(f"User: {current_msg.strip()}")
    return lines


def extract_latest_pincode_from_user_conversation(
    conversation_context: str, current_msg: str = ""
) -> str:
    """Most recent PIN the user typed — never from Assistant messages."""
    found: list[str] = []
    for line in _iter_user_lines_for_context(conversation_context, current_msg):
        p = extract_pincode_preferred_from_message(line)
        if p:
            found.append(p)
    return found[-1] if found else ""


def extract_pincode_from_recent_user_lines(conversation_context: str, current_msg: str = "") -> str:
    """Alias — latest user PIN only."""
    return extract_latest_pincode_from_user_conversation(conversation_context, current_msg)


def extract_latest_order_id_from_user_conversation(
    conversation_context: str, current_msg: str = ""
):
    """Most recent order id the user typed — never from Assistant status cards."""
    found: list[str] = []
    for line in _iter_user_lines_for_context(conversation_context, current_msg):
        oid = extract_order_id(line, "")
        if oid:
            found.append(oid)
    return found[-1] if found else None


def resolve_pincode_for_check(
    current_msg: str,
    conversation_context: str = "",
    *,
    ai_extracted: str = "",
    msg_en: str = "",
    ai_route: dict | None = None,
) -> str:
    """PIN for live API — current message first; history on pin thread or 'already gave'."""
    raw = (current_msg or "").strip()
    combined = f"{raw} {(msg_en or '').strip()}".strip() or raw
    try:
        from services.location_delivery_resolver import _bare_pin_submission_in_pincode_thread

        bare = _bare_pin_submission_in_pincode_thread(
            raw, conversation_context, ai_route=ai_route
        )
        if bare:
            return bare
    except ImportError:
        pass
    if _digits_in_message_are_order_id_not_pincode(
        combined, conversation_context, ai_route=ai_route
    ):
        return ""
    if _user_denies_pincode_insists_order_id(combined):
        return ""
    pin = extract_pincode_preferred_from_message(
        raw, conversation_context, ai_route=ai_route
    )
    if not pin:
        pin = extract_pincode_preferred_from_message(
            combined, conversation_context, ai_route=ai_route
        )
    if pin and _user_denies_pin_in_message_for_different_area(raw):
        pin = ""
    if pin and (
        _text_has_delivery_serviceability_intent(combined, conversation_context)
        or _text_has_pincode_delivery_intent(combined, conversation_context)
        or message_has_live_pincode_check_intent(combined, conversation_context, msg_en)
    ):
        return pin
    if should_reuse_pincode_from_conversation_history(raw, msg_en, conversation_context):
        if re.fullmatch(r"[1-9]\d{5}\s*\??", raw.strip()):
            return raw.strip()[:6]
        pin_pc = extract_pincode_preferred_from_message(raw)
        if pin_pc:
            return pin_pc
        latest = extract_latest_pincode_from_user_conversation(conversation_context, raw)
        if latest:
            log_reasoning(
                f"Reusing PIN {latest} from history — user referenced prior submission."
            )
            return latest
    if _conversation_awaiting_order_id(conversation_context) or _message_is_order_id_followup_submission(
        raw, conversation_context
    ):
        return ""
    if pin:
        return pin

    if not message_requests_new_area_without_pin(raw, msg_en, conversation_context):
        ai_pin = re.sub(r"\D", "", (ai_extracted or "").strip())
        if len(ai_pin) in (6, 7, 8) and ai_pin[0] in "123456789":
            norm = _normalize_pin_candidate(ai_pin)
            if norm and (norm in combined.replace(" ", "") or norm in raw.replace(" ", "")):
                return norm

    if should_reuse_pincode_from_conversation_history(raw, msg_en, conversation_context):
        latest = extract_latest_pincode_from_user_conversation(conversation_context, raw)
        if latest:
            return latest

    return ""


def extract_embedded_query_identifiers(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> dict:
    """
    Pull pincode, order_id, product_id from the SAME user message when wording allows.
    Avoids asking the customer to re-send a value already present in their query.
    """
    comb = f"{original_msg or ''} {(msg_en or '').strip()}".strip()
    out: dict = {
        "pincode": "",
        "order_id": "",
        "product_id": "",
        "numeric_context": "none",
    }
    if not comb.strip():
        return out

    pid = extract_product_id(comb)
    if pid and _text_is_product_id_lookup_context(comb):
        out["product_id"] = str(pid)
        out["numeric_context"] = "product_id"
        return out

    if _text_is_pincode_serviceability_question(comb, conversation_context) or (
        isinstance(ai_route, dict) and (ai_route.get("intent") or "").strip().lower() == "pincode_check"
    ):
        ai_pin_arg = ""
        if isinstance(ai_route, dict) and not message_requests_new_area_without_pin(
            original_msg, msg_en, conversation_context
        ):
            ai_pin_arg = (ai_route.get("extracted_pincode") or "").strip()
        pin = resolve_pincode_for_check(
            original_msg,
            conversation_context,
            msg_en=msg_en,
            ai_extracted=ai_pin_arg,
            ai_route=ai_route,
        ) or extract_pincode_preferred_from_message(
            original_msg, conversation_context, ai_route=ai_route
        )
        if pin and message_requests_new_area_without_pin(
            original_msg, msg_en, conversation_context
        ):
            pin = ""
        if pin:
            out["pincode"] = pin
            out["numeric_context"] = "pincode"
            return out

    order_topic = (
        _text_has_order_id_context(comb)
        or _text_is_live_order_lookup_intent(comb, conversation_context)
        or _text_is_order_tracking_intent(comb)
        or _text_is_refund_return_status_lookup(comb, conversation_context)
        or (
            _text_has_refund_or_return_intent(comb)
            and _current_turn_has_order_id(original_msg, msg_en)
        )
        or _user_explicitly_asks_payment_status(comb)
        or _user_denies_pincode_insists_order_id(comb)
        or (
            isinstance(ai_route, dict)
            and (ai_route.get("intent") or "") in ("order", "refund", "payment")
            and (ai_route.get("numeric_context") or "") == "order_id"
        )
    )
    oid = (
        extract_order_id(original_msg, "")
        or extract_order_id(msg_en, "")
        or _bare_order_id_token_from_msg(original_msg, "")
        or _bare_order_id_token_from_msg(msg_en, "")
    )
    if not oid and (
        _message_is_order_id_followup_submission(original_msg, conversation_context)
        or _text_is_refund_return_status_lookup(comb, conversation_context)
    ):
        oid = (
            extract_order_id(original_msg, conversation_context)
            or extract_order_id(msg_en, conversation_context)
            or _bare_order_id_token_from_msg(original_msg, conversation_context)
            or _bare_order_id_token_from_msg(msg_en, conversation_context)
        )
    if oid and order_topic and not _conversation_bot_asked_for_pincode(conversation_context):
        if not _text_is_pincode_serviceability_question(comb, conversation_context):
            out["order_id"] = oid
            out["numeric_context"] = "order_id"
            return out

    pin = resolve_pincode_for_check(
        original_msg, conversation_context, msg_en=msg_en
    ) or extract_pincode_preferred_from_message(comb)
    pin_topic = (
        _text_has_pincode_delivery_intent(comb, conversation_context)
        or _text_has_delivery_serviceability_intent(comb, conversation_context)
        or _text_has_delivery_or_order_area_intent(comb)
        or (isinstance(ai_route, dict) and (ai_route.get("intent") or "") == "pincode_check")
    )
    if pin and pin_topic and not _conversation_awaiting_order_id(conversation_context):
        out["pincode"] = pin
        out["numeric_context"] = "pincode"
        return out

    ctx = _resolve_ambiguous_bare_numeric_context(
        (original_msg or "").strip(), conversation_context, ai_route=ai_route
    )
    if ctx == "order_id" and oid:
        out["order_id"] = oid
        out["numeric_context"] = "order_id"
    elif ctx == "pincode" and pin:
        out["pincode"] = pin
        out["numeric_context"] = "pincode"
    elif ctx == "product_id" and pid:
        out["product_id"] = str(pid)
        out["numeric_context"] = "product_id"

    if isinstance(ai_route, dict) and not message_requests_new_area_without_pin(
        original_msg, msg_en, conversation_context
    ):
        ai_pin = (ai_route.get("extracted_pincode") or "").strip()
        if ai_pin and re.fullmatch(r"[1-9]\d{5}", ai_pin) and not out["order_id"]:
            if ai_pin.replace(" ", "") in (original_msg or "").replace(" ", ""):
                out["pincode"] = ai_pin
                if pin_topic and out["numeric_context"] == "none":
                    out["numeric_context"] = "pincode"

    return out


def _text_requests_category_product_browse(t: str, ctx=None) -> bool:
    """
    User wants products FROM a named category (e.g. beauty / electronics ke products bta).
    Not the full 'all categories' department list.
    """
    try:
        from services.welfog_api import (
            message_requests_category_browse,
            resolve_nav_category_id_fast,
            resolve_category_browse_for_catalog,
        )

        if not message_requests_category_browse(t):
            return False
        if resolve_nav_category_id_fast(t, ctx=ctx):
            return True
        route = resolve_category_browse_for_catalog(
            t, ctx=ctx, allow_inner_lookup=False
        )
        return bool(route)
    except ImportError:
        pass
    if _looks_like_browse_all_categories_message(t):
        return False
    try:
        from services.welfog_api import get_category_id_from_text

        named_cat_id = get_category_id_from_text(t, ctx=ctx)
    except Exception:
        named_cat_id = None
    if not named_cat_id:
        return False

    tl = f" {(t or '').lower()} "
    browse_signals = (
        "dikhao", "dikha", "dikhe", "dikhaa", "dikha de", "dikha do", "dekho", "dekh", "show", "list", "display",
        "batao", "btao", "bata", "bta", "btana", "batana", "bta de", "bata de", "bta do",
        "bhi bta", "bhi dikha", "k bhi",
        "chahiye", "chiye", "dena", "de do", "dede", "de de",
        "lao", "la do", "bhejo", "send", "view", "see",
        "puchh", "pooch", "puch", "puch rha", "bol", "bolo",
        "product", "products", "item", "items", "samaan", "cheez",
    )
    return any(s in tl for s in browse_signals)


def _normalize_order_history_typos(t: str) -> str:
    """Roman Hindi / typo-tolerant normalization for order-history detection."""
    s = (t or "").lower()
    for wrong, right in (
        ("hidtory", "history"),
        ("histroy", "history"),
        ("histry", "history"),
        ("histery", "history"),
        ("hostory", "history"),
        ("histoey", "history"),
        ("ordr", "order"),
        ("odrer", "order"),
        ("oder ", "order "),
        ("odr ", "order "),
    ):
        s = s.replace(wrong, right)
    s = re.sub(r"\bhist[oaeiu]{0,3}r[a-z]{0,4}\b", "history", s)
    return s


def _text_asks_order_history(t: str) -> bool:
    """
    User wants a list of past orders / purchase history — DIRECT ACTION intent.
    Works across en, hinglish, and native-script Indian languages (via phrases + msg_en).
    Excludes: process questions (how/steps), other users, and single-order tracking.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    if _text_is_live_order_lookup_intent(raw):
        return False

    if _text_wants_order_history_list_in_chat(raw):
        return True

    if message_asks_my_welfog_purchases(raw):
        return True

    if multilingual_how_to_order_history(raw, raw):
        return False

    if _text_has_order_history_navigation_only(raw):
        return False

    if native_order_history_request(raw) or multilingual_order_history_match(raw, raw):
        if not _text_suggests_single_order_status_lookup(raw):
            if not any(
                x in f" {raw.lower()} "
                for x in ("kahan", "kab ", "track", "status", "delivery", "where ", "when ")
            ):
                return True

    tl = f" {raw.lower()} "
    norm = f" {_normalize_order_history_typos(raw)} "
    has_history_word = "history" in norm or "purchase history" in norm

    # ❌ EXCLUDE: Process/help intent (not asking to see, but asking HOW)
    if any(
        x in tl
        for x in [
            " how ",
            " how to ",
            " steps ",
            " process ",
            " kaise ",
            " step by step ",
            " guide ",
            " way to ",
            " show me steps ",
            " tell me steps ",
            " kya process ",
            " kya tarika ",
            " mujhe batao kaise ",
            " mujhe sikhao ",
            " kaise dekhu ",
            " where to find ",
            " kahan dekhu ",
            " kaise check karu ",
            " kaise dekh sakta ",
            " mujhe pata chale ",
            " i am asking about ",
            " i am not asking ",
            "i'm asking about",
        ]
    ):
        return False

    # ❌ EXCLUDE: Asking about someone else's history
    if any(x in tl for x in [" for user ", " user id ", " other ", " dusre ka ", " kisi aur ka "]):
        if not any(y in tl for y in [" my ", " mera ", " meri ", " mere ", " apna ", " apni "]):
            return False

    # ❌ EXCLUDE: Single-order tracking / ETA / pasted id (not full history list)
    if not has_history_word:
        if _text_suggests_single_order_status_lookup(raw):
            return False
        if extract_order_id(raw) and any(
            x in tl
            for x in [
                "track",
                "where is my order",
                "where my order",
                "kahan",
                "kab aayega",
                "kab aaega",
                "kab tk",
                "kab tak",
                "delivery status",
                " aa ",
                "aayega",
                "aaega",
                "aa jayega",
                "order id",
                "orderid",
            ]
        ):
            return False
        if any(
            x in tl
            for x in [
                "track",
                "where is my order",
                "where my order",
                "kahan hai",
                "kab aayega",
                "kab aaega",
                "kab tk",
                "kab tak",
                "delivery status",
            ]
        ):
            return False

    list_verbs = (
        "dikhao",
        "dikha",
        "dikao",
        "dikaao",
        "dikha do",
        "dikha do",
        "bata",
        "batao",
        "btao",
        "btana",
        "batana",
        "bata de",
        "bta de",
        "batado",
        "btado",
        "show",
        "list",
        "dekho",
        "dekh",
        "see",
        "view",
        "display",
        "send",
        "bhejo",
        "bhej do",
    )
    possessive = any(x in tl for x in (" my ", " mera ", " meri ", " mere ", " apna ", " apni "))
    has_order = "order" in norm or "orders" in norm

    # ✅ Hinglish: "meri order history batao" / "bhau meri order hidtory btana"
    if has_order and has_history_word:
        if possessive or any(v in tl for v in list_verbs):
            return True

    # ✅ "meri order batao" / "mere orders dikhao" (list, not one shipment)
    if has_order and possessive and any(v in tl for v in list_verbs):
        if _text_is_live_order_lookup_intent(raw):
            return False
        if extract_order_id(raw) and ("order id" in tl or "orderid" in tl):
            return False
        if not any(x in tl for x in ("kahan", "kab ", "track", "status", "delivery", "refund")):
            return True

    # ✅ INCLUDE: Direct order history view request
    markers = (
        "order history",
        "puri order",
        "pura order",
        "saari order",
        "sari order",
        "full order history",
        "complete order history",
        "sabhi order",
        "purchase history",
        "order list",
        "orders list",
        "my orders",
        "all orders",
        "all my orders",
        "past orders",
        "previous orders",
        "old orders",
        "booking history",
        "show my orders",
        "show orders",
        "kitne order",
        "kitni order",
        "meri orders",
        "mere orders",
        "meri order",
        "mere order",
        "sab orders",
        "saari orders",
        "saare orders",
        "purane order",
        "purani order",
        "order dikhao",
        "orders dikhao",
        "order list dikhao",
        "order list dikao",
        "order history dikhao",
        "order history batao",
        "order history btao",
        "konse order",
        "kon kon se",
        "kre the mene",
        "kiya tha",
        "kiye the",
        "kiya the",
        "products kre",
        "product kre",
        "order kre",
        "order kiye",
        "order kiya",
        "purane order",
        "pehle ke order",
    )
    if any(m in norm for m in markers) or any(m in tl for m in markers):
        return True

    # ✅ INCLUDE: Pattern match for "show/list/see orders" + action verb
    if "orders" in norm and any(x in tl for x in list_verbs):
        return True

    # ✅ Past orders / which orders placed (Hinglish, no "dikhao")
    if has_order and any(
        x in tl
        for x in (
            "konse", "kon kon", "kre the", "kiya tha", "kiye the", "kiya the",
            "kre the mene", "kiya mene", "kiye mene", "products kre", "product kre",
            "purane", "pehle ke", "saare order", "sab order",
        )
    ):
        if not any(x in tl for x in ("track", "kab aayega", "kab aaega", "status", "kahan")):
            return True

    return multilingual_order_history_match(raw, raw)


def _message_has_app_navigation_intent(t: str) -> bool:
    """Where/how to open something in the app or website (language-agnostic)."""
    if not (t or "").strip():
        return False
    tl = f" {(t or '').lower()} "
    return any(
        x in tl
        for x in (
            " how ", " how to ", " kaise ", " kese ", " kaha ", " kahan ", " kidhar ", " where ",
            " app ", " website ", " manually ", " manual ", " khud ", " dekhu ", " dekh ",
            " dekhna ", " dekhni ", " steps ", " process ", " kholo ", " open ", " jake ",
            " jaske ", " jau ", " milegi ", " dikhe",
        )
    )


def _conversation_bot_showed_wishlist(conversation_context: str) -> bool:
    """Assistant recently showed wishlist in chat or wishlist how-to."""
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4000:].lower()
    markers = (
        "your wishlist",
        "wishlist (",
        "wishlist icon",
        "how to view your wishlist",
        "wishlist app mein",
        "wishlist_help",
        "meri wishlist",
        "wishlist batao",
        "wishlist dikhao",
        "❤",
    )
    if any(m in tail for m in markers):
        if any(
            x in tail
            for x in ("wishlist", "wish list", "saved", "liked", "heart")
        ):
            return True
    return False


def _turn_blocks_wishlist_howto_routing(t: str) -> bool:
    """True when this turn must never be routed to wishlist how-to KB."""
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} "
    if _pincode_delivery_signal_leaf(raw):
        return True
    if _text_has_refund_or_return_intent(raw):
        return True
    if _text_is_order_id_help_request(raw):
        return True
    if _text_has_order_placement_intent(raw):
        return True
    if _text_is_tracking_howto_request(raw):
        return True
    if _text_is_refund_return_policy_howto(raw):
        return True
    if _text_has_refund_or_return_intent(raw) and any(
        x in tl
        for x in (
            "how long",
            "how many days",
            "processing time",
            "timeline",
            "business days",
            "kitne din",
            "when will",
            "take for a refund",
            "refund to be processed",
            "refund processed",
            "refund process",
        )
    ):
        return True
    if ("order id" in tl or "orderid" in tl or re.search(r"\border\b.{0,12}\bid\b", tl)) and any(
        x in tl for x in ("kaha", "kahan", "milegi", "milti", "find", "where", "kaise", "how", "puchh", "puch")
    ):
        return True
    return False


def _text_asks_how_to_view_wishlist(t: str, conversation_context: str = "") -> bool:
    """
    User wants steps to open wishlist in the app/website — not an in-chat product list.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    if _pincode_delivery_signal_leaf(raw) or _text_has_refund_or_return_intent(raw):
        return False
    if _turn_blocks_wishlist_howto_routing(raw):
        return False
    if message_is_wishlist_like_request(raw) and _message_has_app_navigation_intent(raw):
        return True
    if _conversation_bot_showed_wishlist(conversation_context) and _message_has_app_navigation_intent(
        raw
    ):
        if message_mentions_wishlist_topic(raw) or message_is_wishlist_like_request(raw):
            return True
        if message_clarifies_wishlist_not_order_history(raw):
            return True
        return False
    if message_clarifies_wishlist_not_order_history(raw):
        return True
    tl = f" {raw.lower()} ".replace("-", " ")
    if "wishlist" not in tl and "wish list" not in tl:
        return False
    list_verbs = (
        "dikhao", "dikha", "bata", "batao", "btao", "bta", "btado", "bta de", "bata de",
        "show", "list", "dekho", "dekh", "display", "load", "fetch", "bhejo", "bhej do",
    )
    if any(v in tl for v in list_verbs):
        if not any(
            x in tl
            for x in (
                " how ", " how to ", " kaise ", " kahan ", " kidhar ", " kaha ",
                " steps ", " step ", " process ", " guide ", " kholo ", " open ",
                " app me ", " website pe ", " where ",
            )
        ):
            return False
    how_markers = (
        " how ", " how to ", " kaise ", " kese ", " kahan ", " kidhar ", " kaha ", " kaha pe ",
        " steps ", " step by step ", " process ", " guide ", " tarika ", " tarike ",
        " app me ", " website ", " kholo ", " open kru ", " open karu ", " dekhu ",
        " dekh sakta ", " dekh sakte ", " find ", " where to ", " kahan milegi ",
        " wishlist me kaise ", " wishlist kaise ", " add kaise ", " add krna ",
        " manually ", " manual ", " myself ", " khud ", " app ke andar ",
        " kaha se dekh", " kahan se dekh", " where can i see ",
        " jaske dekhu", " jake dekhu", " jaske dekh", " jake dekh", " kaha jaske",
        " kaha jake", " kidhar jau", " kaha jau", " dekhni ho", " dekhna ho",
        " dekhni h ", " dekhna h ", " dekh sakte h", " dekh skte h",
    )
    if any(m in tl for m in how_markers):
        return True
    if message_clarifies_wishlist_not_order_history(raw):
        return True
    if re.search(r"\b(?:kaha|kahan|kidhar)\b.{0,24}\b(?:dekh\w*|jau|jake|jaske)\b", tl):
        return True
    return False


def _wants_wishlist_list_in_chat(t: str) -> bool:
    """Saved/liked products shown in this chat — leaf helper."""
    if not message_is_wishlist_like_request(t):
        return False
    if _text_asks_how_to_view_wishlist(t):
        return False
    tl = f" {(t or '').lower()} "
    show = (
        "dikhao", "dikha", "dikha de", "dikha do", "dikhado", "show", "list",
        "tu hi", "tum hi", "dede", "de do", "bhejo", "bta de", "bata de",
    )
    return any(s in tl for s in show)


def _text_asks_wishlist(t: str) -> bool:
    """
    User wants their saved wishlist shown in chat (not how-to in app).
    """
    if message_denies_wishlist_intent(t):
        return False
    if message_asks_my_welfog_purchases(t) and not message_is_wishlist_like_request(t):
        return False
    if _wants_wishlist_list_in_chat(t):
        return True
    if _text_asks_how_to_view_wishlist(t):
        return False
    raw = (t or "").strip()
    if not raw:
        return False
    tl = f" {raw.lower()} ".replace("-", " ")
    if any(x in tl for x in [" for user ", " user id ", " other user", " dusre ka ", " kisi aur ka "]):
        if not any(y in tl for y in [" my ", " mera ", " meri ", " mere ", " apna ", " apni ", "mereko"]):
            return False
    how_only = any(
        x in tl
        for x in [
            " how ",
            " how to ",
            " kaise ",
            " kese ",
            " steps ",
            " process ",
            " add to wishlist",
            " wishlist me kaise",
            " wishlist kaise ",
        ]
    )
    list_verbs = (
        "dikhao",
        "dikha",
        "bata",
        "batao",
        "btao",
        "bta",
        "btado",
        "bta de",
        "bata de",
        "show",
        "list",
        "dekho",
        "dekh",
        "view",
        "display",
        "load",
        "fetch",
        "bhejo",
        "bhej do",
        "dede",
        "dedo",
        "de do",
        "de de",
        "list dede",
        "list de",
        "list do",
    )
    possessive = any(
        x in tl for x in (" my ", " mera ", " meri ", " mere ", " apna ", " apni ", "mereko", "mujhe")
    )
    if "wishlist" in tl or "wish list" in tl:
        if re.search(r"\blist\b", tl) and any(v in tl for v in list_verbs):
            return True
        if any(
            x in tl
            for x in (
                "wishlist me add", "add kari", "add kiya", "add ki", "add kiye",
                "jo bhi", "ab tk", "ab tak", "jitni bhi", "jitna bhi",
            )
        ) and any(v in tl for v in list_verbs + ("list",)):
            return True
        if how_only and not any(v in tl for v in list_verbs):
            return False
        if possessive or any(v in tl for v in list_verbs):
            return True
        if len(tl.strip()) < 28:
            return True
    markers = (
        "saved items",
        "saved products",
        "saved list",
        "favourite list",
        "favorite list",
        "meri wishlist",
        "mere wishlist",
        "wishlist dikhao",
        "wishlist dikha",
        "wishlist batao",
        "wishlist btao",
        "wishlist show",
        "wishlist dekh",
    )
    if any(m in tl for m in markers):
        return True
    if "विशलिस्ट" in raw or "விஷ்லிஸ்ட்" in raw:
        if any(v in tl for v in list_verbs) or possessive:
            return True
    return False


def _conversation_bot_showed_order_history_steps(conversation_context: str) -> bool:
    """Assistant recently sent My Orders / how-to-view-history steps."""
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-3500:].lower()
    if "assistant:" not in tail and "bot:" not in tail:
        pass
    markers = (
        "my orders",
        "order history kaise",
        "how to view your order history",
        "app mein order history",
        "yahan chat mein list bhejne se alag",
    )
    return any(m in tail for m in markers)


def _text_has_order_history_navigation_only(t: str) -> bool:
    """Where/how in the app — not 'show my history here in chat'. Requires order/history topic."""
    if message_blocks_order_history_routing(t):
        return False
    if _text_is_order_id_help_request(t):
        return False
    tl = f" {(t or '').lower()} "
    if re.search(r"\border\s*id\b", tl) or "orderid" in tl:
        if any(x in tl for x in ("kaha", "kahan", "kaise", "kese", "milega", "milta", "dikhe", "dekhu")):
            return False
    has_order_topic = any(
        x in tl
        for x in (
            "order history", "purchase history", "my orders", "purane order", "past order",
            "order ki history", "purchase ki history", "orders history",
        )
    ) or ("history" in tl and re.search(r"\border\b", tl))
    if not has_order_topic:
        return False
    app_nav = any(
        x in tl
        for x in (
            "kaha jake", "kahan jake", "kidhar jake", "kaha se dekh", "kahan se dekh",
            "kaha dekhu", "kahan dekhu", "kaha jake dekhu", "kahan jake dekhu",
            "app pr", "app par", "app me ", "app mein", "website pr", "website par",
            "dekhne ka process", "dekhna chahe", "dekhna chahta", "dekh skta hu app",
            "dekh sakta hu app", "kaise dekhte", "kese dekhte", "kaise dekhu",
            "how to view", "where to view", "where can i see",
            "manually", "manual", "myself", "khud", "khud dekh",
            "app ke andar", "app me kaha", "app mein kaha",
        )
    )
    process_nav = any(
        x in tl
        for x in ("step by step", " steps ", " process ", " tareeka ", " tarika ")
    )
    return app_nav or process_nav


def _text_requests_purchase_order_list_in_chat(
    t: str, conversation_context: str = ""
) -> bool:
    """
    Purchase/order LIST in chat — leaf helper (no live_order_lookup / tracking recursion).
  e.g. 'jitne bhi order kiye unki list de de', 'order id mat maang list de'.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    if _text_is_undelivered_order_complaint(raw):
        return False
    tl = f" {_normalize_order_chat_text(raw).lower()} "
    if _user_rejects_order_history_wants_tracking(raw, conversation_context):
        return False
    if any(
        x in tl
        for x in (
            "order id maang", "order id mat", "id maang rha", "id maang rahi", "id kyu maang",
            "id kyo maang", "kyu order id", "kyo order id", "id mat maang", "id ni maang",
        )
    ) and any(x in tl for x in ("list", "orders", "order kiye", "order ki", "meri order")):
        return True
    list_phrases = (
        "jitne bhi order", "jitne order", "saare order", "sab order", "saari order",
        "orders ki list", "order ki list", "mere orders", "meri orders", "my orders",
        "unki list", "list de de", "list dede", "list de do", "purchase list", "order list",
        "poori list", "puri list", "saari list",
    )
    if any(p in tl for p in list_phrases):
        if not any(x in tl for x in ("track kaise", "order id kahan", "order id kaha", "id kaha mile")):
            return True
    if ("order kiye" in tl or "order kiya" in tl) and any(
        v in tl for v in ("list", "de de", "dede", "de do", "dikhao", "dikha", "bta de", "bata de")
    ):
        if not any(
            x in tl
            for x in ("track kr", "track kar", "nahi pahucha", "nhi pahucha", "kab aayega", "status")
        ):
            return True
    if any(x in tl for x in ("mene", "maine", "mere", "meri")) and "order" in tl and any(
        v in tl for v in ("list", "de de", "dede", "dikhao", "bta de", "bata de", "de do")
    ):
        return True
    if _conversation_awaiting_order_id(conversation_context) and any(
        x in tl for x in ("list", "orders ki", "order ki list", "history", "saare", "sab ")
    ):
        return True
    return False


def _text_wants_order_history_list_in_chat(
    t: str,
    conversation_context: str = "",
) -> bool:
    """
    User wants their purchase history shown IN THIS CHAT (API), not app navigation steps.
    e.g. 'tu hi bta de meri order history', 'meri order history dikhao'.
    """
    raw = (t or "").strip()
    if not raw:
        return False
    # Do not call detect_account_list_followup_in_chat — it recurses via
    # _followup_blocks_in_chat_list → _text_has_product_shopping_intent → this function.
    if message_is_wishlist_like_request(raw) or _text_asks_wishlist(raw):
        if not message_denies_wishlist_wants_order_history(raw):
            return False
    try:
        from services.account_list_semantics import (
            _conversation_last_wishlist_howto_only,
            _message_asks_show_in_chat,
        )

        if _conversation_last_wishlist_howto_only(conversation_context) and _message_asks_show_in_chat(
            raw
        ):
            return False
    except ImportError:
        pass
    if _text_requests_purchase_order_list_in_chat(raw, conversation_context):
        return True
    # Do not call message_wants_order_history_app_navigation here — it calls this function (recursion).
    if _text_asks_to_view_purchase_or_order_history(raw) and _message_has_app_navigation_intent(raw):
        return False
    if _text_is_live_order_lookup_intent(raw, conversation_context):
        return False
    if _text_is_undelivered_order_complaint(raw):
        return False
    if _user_rejects_order_history_wants_tracking(raw, conversation_context):
        return False
    if _user_clarifies_process_not_order_list(raw, conversation_context):
        return False
    if _text_has_order_history_navigation_only(raw):
        return False

    tl = f" {raw.lower()} "
    possessive = any(
        x in tl
        for x in (" meri ", " mere ", " mera ", " mujhe ", " my ", " apni ", " apna ", " apne ")
    )
    has_hist = any(
        x in tl
        for x in (
            "order history", "order hist", "orders history", "purchase history",
            "purane order", "past order", "jo order kiye", "order kiye",
        )
    ) or ("order" in tl and "history" in tl)

    list_verbs = (
        "dikhao", "dikha", "dikhe", "dikha de", "dikhado", "dikha do", "dikha dena",
        "bta de", "btao", "bata de", "batao", "batade", "bata dena", "bata do",
        "bhej", "bhejo", "de de", "dede", "de do", "dena",
        "tu hi bta", "tu bta", "tu hi bata", "tu bata", "tu hi dikha", "tu dikha",
        "show me", "show my", "tell me my", "give me my", "list dede", "list de",
        "unki list", "saari order", "sab order", "poori history", "puri history",
    )
    wants_data = any(v in tl for v in list_verbs)

    if has_hist and wants_data and possessive:
        return True
    if has_hist and possessive and re.search(r"\b(?:de|dede|de do|dena|bta|btao|bata)\b", tl):
        return True
    if has_hist and any(
        x in tl
        for x in (
            "usme se", "us se", "waha se", "fir usme", "fir waha", "leke de du",
            "de du tere", "de du tujhe", "uthake", "utha ke", "id utha", "order id leke",
            "order id de du", "order id de dunga", "order id nikal",
        )
    ):
        return True
    if has_hist and possessive and any(
        x in tl for x in (" chahiye", " chahie", " mang", " maang", " manga", " lena", " lenaa")
    ):
        if not _text_has_order_history_navigation_only(raw):
            return True

    if _my_welfog_purchases_data_request_signals(raw):
        if not _text_has_order_history_navigation_only(raw):
            return True

    if conversation_context and _conversation_bot_showed_order_history_steps(conversation_context):
        follow_up = (
            "tu hi", "tu bta", "tu bata", "ese bol", "aisa bol", "yahi maang",
            "yahi mang", "history maang", "list maang", "chat me", "yahan",
            "idhar", "yahi bta", "khud bta", "khud dikha", "tum bta", "tum dikha",
            "ek kaam kar", "tu hi bta", "bol rha", "bol raha", "maang rha", "mang rha",
        )
        if any(x in tl for x in follow_up):
            if has_hist or wants_data or not _text_has_order_history_navigation_only(raw):
                return True

    return False


def _user_clarifies_process_not_order_list(
    t: str, conversation_context: str = ""
) -> bool:
    """
    User corrects the bot: they wanted ORDER HISTORY steps in the app, not the list in chat.
    Must NOT fire on wishlist / liked-products / manual wishlist follow-ups.
    """
    if not (t or "").strip():
        return False
    if message_blocks_order_history_routing(t):
        return False
    if _conversation_bot_showed_wishlist(conversation_context) and _message_has_app_navigation_intent(
        t
    ):
        return False
    tl = f" {(t or '').lower()} "
    if any(
        x in tl
        for x in (
            "history nhi", "history nahi", "history ni ", "history nhi maang",
            "history nahi maang", "list nhi", "list nahi", "list ni maang",
            "nhi maang rha", "nahi maang rha", "not asking for history",
            "not history", "process maang", "process mang", "process puchh",
            "tareeka maang", "tarika maang", "steps maang",
            "order id ka nhi", "order id nahi", "order id ni ", "order id nahin",
            "order id ka ni", "order id nhi", "id ka nhi puchh", "id nahi puchh",
        )
    ):
        return True
    if re.search(r"\border\s*id\b", tl) and any(
        x in tl for x in (" nhi ", " nahi ", " ni ", " nahin ", " not ")
    ) and any(o in tl for o in ("history", "purchase", "my orders")):
        return True
    generic_nav = (
        "khud dekhna", "khud dekhunga", "manually dekhna", "manual dekhna",
        "where can i see", "where to see", "kaha se dekhna", "kahan se dekhna",
        "kaha jake", "kahan jake", "kidhar jake", "app pr dekh", "app par dekh",
        "app me dekh", "app mein dekh", "khud jaake", "khud jake", "dekhna chahe",
        "dekhna chahta", "dekh skta", "dekh sakta", "kaha dekh", "kahan dekh",
        "kaha dikhegi", "kahan dikhegi", "kaha dikhega", "kahan dikhega",
        "app pe dikhe", "app par dikhe", "app me dikhe",
    )
    if any(x in tl for x in generic_nav):
        return any(o in tl for o in ("order", "history", "orders", "my orders", "purchase"))
    return False


def _text_asks_how_to_view_order_history(
    t: str,
    conversation_context: str = "",
) -> bool:
    """
    User is asking about the PROCESS/STEPS to view order history in the app.
    Returns True -> route to steps (KB), not purchase-history API.
    """
    raw = (t or "").strip()
    if _text_is_order_id_help_request(raw):
        return False
    if message_blocks_order_history_routing(raw):
        return False
    if message_needs_policy_answer(raw) and not _text_has_order_history_navigation_only(raw):
        return False
    if _text_has_refund_or_return_intent(raw):
        tl_ref = f" {raw.lower()} "
        if any(
            x in tl_ref
            for x in (
                "kitne din", "kab milega", "kab aayega", "kab aaega", "timeline",
                "days", "din me", "mil jata", "mil jayega", "milega", "time lag",
            )
        ) and not _text_has_order_history_navigation_only(raw):
            return False
    if _text_wants_order_history_list_in_chat(raw, conversation_context):
        return False
    if _text_has_order_placement_intent(raw) or _user_rejects_viewing_wants_placement(raw):
        return False
    tl_placement = f" {raw.lower()} "
    if any(
        x in tl_placement
        for x in (
            "place order",
            "place the order",
            "place my order",
            "place an order",
            "checkout",
            "add to cart",
            "how can i place",
            "how do i place",
            "how to place",
        )
    ):
        return False
    if _user_clarifies_process_not_order_list(raw, conversation_context):
        return True
    if _text_has_order_history_navigation_only(raw):
        return True
    if multilingual_how_to_order_history(raw, raw):
        return True
    tl = f" {t.lower()} "

    if re.search(r"\bdekh\w*\s+ka\s+process\b", tl):
        if any(o in tl for o in ("order", "history", "orders")):
            return True
    if re.search(r"\bprocess\s+(?:bta|bat|btao|samjha|dikha|bata)\b", tl):
        if any(o in tl for o in ("order", "history", "orders")):
            return True

    process_words = [
        " how ", " how to ", " steps ", " process ", " kaise ", " kese ", " step by step ",
        " guide ", " where to find ", " kaise check karu ", " kaise dekh sakta ",
        " kaise dekh sakte ", " kaise dekhu ", " kaise dekhte ", " kese dekhte ",
        " tarika ", " tarike ", " tareeka ", " procedure ", " samjha ", " samjh ",
    ]

    order_words = ["order", "history", "orders", "booking"]

    has_process = any(x in tl for x in process_words)
    has_order = any(x in tl for x in order_words)

    return has_process and has_order


def _user_asks_order_history_navigation_help(
    t: str,
    conversation_context: str = "",
) -> bool:
    """View order history in the app (how/where) — not fetch the list in this chat."""
    if message_blocks_order_history_routing(t):
        return False
    if _text_has_order_placement_intent(t) or _user_rejects_viewing_wants_placement(t):
        return False
    if _text_wants_order_history_list_in_chat(t, conversation_context):
        return False
    return _text_asks_how_to_view_order_history(
        t, conversation_context
    ) or _user_clarifies_process_not_order_list(t, conversation_context)


def _current_turn_has_order_id(original_msg: str, msg_en: str = "") -> bool:
    """Order ID digits in this user turn only — no catalog lookup recursion."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    if _message_has_embedded_pincode_for_delivery(comb, ""):
        return False
    if _text_is_pincode_serviceability_question_light(comb):
        return False
    if _text_has_explicit_pincode_subject(comb) and re.search(r"\b[1-9]\d{5}\b", comb):
        return False
    oid = _bare_order_id_token_from_msg(comb, "")
    if not oid:
        return False
    if not _is_plausible_order_id(oid, context=comb, shallow=False):
        return False
    if (
        _text_is_undelivered_order_complaint(comb)
        or _text_has_light_order_tracking_markers(comb)
        or _text_has_refund_or_return_intent(comb)
        or _text_has_past_order_complaint_context(comb)
    ):
        return True
    tl = f" {comb.lower()} "
    if re.search(r"\b(?:product\s*id|pro\s*id|sku)\b", tl) and "order" not in tl and "refund" not in tl:
        return False
    if re.fullmatch(r"[0-9]{4,20}", comb) or re.fullmatch(r"[A-Za-z0-9]{4,20}", comb):
        if _is_plausible_order_id(oid, context=comb, shallow=True):
            return True
    return _text_has_order_id_context(comb) or _text_has_light_order_tracking_markers(comb)


def _text_is_refund_return_status_lookup(t: str, conversation_context: str = "") -> bool:
    """
    LLM-unavailable failsafe only — personal refund/return status on one order.
    When Groq routing is available, use refund_status_semantics + order_lookup_kind.
    """
    if getattr(_refund_lookup_guard, "active", False):
        return False
    _refund_lookup_guard.active = True
    try:
        return _text_is_refund_return_status_lookup_impl(t, conversation_context)
    finally:
        _refund_lookup_guard.active = False


def _text_is_refund_return_status_lookup_impl(t: str, conversation_context: str = "") -> bool:
    raw = _normalize_order_chat_text(t or "")
    if not raw.strip():
        return False
    tl = f" {raw.lower()} "
    status_markers = (
        "check", "status", "kb tk", "kab tk", "kab tak", "kab milega", "kab aayega", "kab aaye",
        "kab hoga", "kab hogi", "kab approve", "kitne din", "aajayega", "aa jayega",
        "atak", "atka", "stuck", "delay", "approve", "approved", "approval",
        "dekh ke bta", "check krke", "check karke", "verify", "record", "bta de",
        "bata de", "btao de", "check kr", "check kar", "return status", "refund status",
        "bata", "btao", "batao", "btana", "bata do", "bta do",
    )
    refund_complaint_markers = (
        "nhi aaya", "nahi aaya", "nhi mila", "nahi mila", "nhi aaye", "nahi aaye",
        "abhi tak", "abhi tk", "not received", "not come", "haven't received",
        "approve hua", "approved hua", "hua kya", "hui kya",
    )
    has_refund_topic = bool(
        _text_has_refund_or_return_intent(raw) or "refund" in tl or "return" in tl
    )
    if has_refund_topic and any(c in tl for c in refund_complaint_markers):
        return True
    if not any(m in tl for m in status_markers):
        return False
    # General timeline without a specific order → policy KB, not live status API.
    if any(x in tl for x in ("kitne din", "kitna time", "kitne din me", "generally", "usually")):
        if not _current_turn_has_order_id(raw) and not re.search(r"\b\d{4,20}\b", raw):
            return False
    payment_status = "payment" in tl and any(
        x in tl
        for x in (
            "payment status", "payment detail", "payment ka", "payment ki",
            "paid h", "unpaid h", "paisa", "paise",
        )
    )
    if not (_text_has_refund_or_return_intent(raw) or "refund" in tl or payment_status):
        return False
    if _current_turn_has_order_id(raw):
        return True
    if re.search(r"\bcheck\s+(?:kr|kar)?ke\b", tl) or "id check" in tl:
        return True
    pronoun = ("iske", "iski", "iska", "is order", "ye wala", "yeh wala", "us order")
    if any(p in tl for p in pronoun):
        tail = _conversation_user_text_tail(conversation_context)
        if any(x in tail for x in ("refund", "return", "order id", "2605270")) or _conversation_in_order_tracking_flow(
            conversation_context
        ):
            return True
    if _conversation_awaiting_order_id(conversation_context) and _message_is_order_id_followup_submission(
        raw, conversation_context
    ):
        return True
    # Personal refund/return status (any language) — live API; order id optional on this turn.
    return True


def _text_is_refund_return_policy_howto(t: str) -> bool:
    """
    Return/refund process or wrong-item help — KB, not repeated live API on a stale Order ID.
    """
    if getattr(_refund_lookup_guard, "active", False):
        return False
    raw = _normalize_order_chat_text(t or "")
    if not raw.strip():
        return False
    tl_pre = f" {raw.lower()} "
    if "refund status" in tl_pre or "return status" in tl_pre:
        return False
    if _text_is_refund_return_status_lookup(raw):
        return False
    tl = f" {raw.lower()} "
    if any(
        x in tl
        for x in (
            "nahi aaya", "nhi aaya", "nahi mila", "nhi mila", "abhi tak", "abhi tk",
            "approve hua", "approved hua", "status bta", "status bata",
        )
    ):
        if not _current_turn_has_order_id(raw) and not re.search(r"\b\d{4,20}\b", raw):
            return False
    howto_markers = (
        "kaise", "kese", "kes ", "krte", "karte", "karna", "krna", "karu", "karun",
        "batana", "batao", "btao", "process", "steps", "step", "kya kar", "kya kr",
        "welfog pe", "welfog par", "app pe", "app par", "website pe", "kaise kr",
        "return kr", "return kar", "return ho", "policy", "eligible", "eligibility",
    )
    general_timeline = (
        "kitne din", "kitna time", "kitne din me", "generally", "usually",
        "normally", "typically", "in general",
        "kab aayega", "kab aaega", "kab milega", "kab tak", "kb tk", "aa jayega",
        "aayega", "mil jata", "mil jati", "kitne dino",
    )
    if _text_has_refund_or_return_intent(raw) and any(x in tl for x in howto_markers):
        return True
    if _text_has_refund_or_return_intent(raw) and any(x in tl for x in general_timeline):
        if not _current_turn_has_order_id(raw) and not re.search(r"\b\d{4,20}\b", raw):
            return True
    if _text_has_past_order_complaint_context(tl) and any(
        x in tl
        for x in (
            "kya kar", "kya kr", "kya karu", "kase kru", "kaise kru", "photo", "dikh rha", "dikh raha",
            "jaisa nahi", "jese nahi", "wese nahi", "waisa nahi", "galat mila",
            "nhi chahiye", "nahi chahiye", "ni chahiye", "chahiye hi nhi", "wapas", "refund", "paise",
        )
    ):
        return True
    if "return" in tl and any(x in tl for x in ("krte", "karte", "kese", "kaise", "karna", "krna")):
        return True
    if re.search(r"\bhow\s+(?:to|do|can)\s+(?:i\s+)?(?:return|refund|replace)\b", tl):
        return True
    if "return on welfog" in tl or "return on the welfog" in tl:
        return True
    return False


def user_turn_qualifies_for_live_order_api(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Gate live track/refund/payment API — never reuse Order ID for unrelated how-to follow-ups."""
    if turn_is_catalog_product_lookup(original_msg, msg_en, ai_route):
        return False
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    try:
        from services.semantic_intent import (
            ai_route_requests_refund_status_lookup,
            llm_semantic_route_available,
        )

        if llm_semantic_route_available(ai_route) and ai_route_requests_refund_status_lookup(
            ai_route, original_msg, msg_en, conversation_context
        ):
            return True
    except ImportError:
        pass
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if _text_has_past_order_complaint_context(comb) and not _text_is_undelivered_order_complaint(comb):
        if not _current_turn_has_order_id(original_msg, msg_en):
            return False
    if _text_is_undelivered_order_complaint(comb) or _user_rejects_order_history_wants_tracking(
        comb, conversation_context
    ):
        return True
    if _text_is_refund_return_policy_howto(comb):
        return False
    if message_needs_policy_answer(comb) and not _text_is_refund_return_status_lookup(
        comb, conversation_context
    ):
        return False
    from services.order_details_flow import message_wants_order_details_or_invoice

    if message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return False
    if _current_turn_has_order_id(original_msg, msg_en):
        return True
    if _text_is_refund_return_status_lookup(comb, conversation_context):
        return bool(
            resolve_order_id_for_tracking(
                original_msg.strip() or msg_en.strip(),
                conversation_context,
                bot_awaiting_order_id=_message_is_order_id_followup_submission(
                    original_msg, conversation_context
                ),
            )
        )
    if _message_is_order_id_followup_submission(original_msg, conversation_context):
        return True
    if _conversation_awaiting_order_id(conversation_context) and (
        _user_demands_immediate_track_action(comb)
        or _text_is_order_tracking_intent_leaf(comb)
    ):
        if resolve_order_id_for_tracking(
            original_msg.strip() or msg_en.strip(),
            conversation_context,
            bot_awaiting_order_id=True,
        ):
            return True
    if isinstance(ai_route, dict) and (ai_route.get("data_channel") or "").lower() == "live_api":
        intent = (ai_route.get("intent") or "").strip().lower()
        if intent in ("order", "refund", "payment") and _text_suggests_single_order_status_lookup(comb):
            return True
    return False


def _text_suggests_single_order_status_lookup(t: str) -> bool:
    """
    User message includes a plausible order id AND language about tracking / ETA / status —
    not a request to browse full purchase history.
    """
    if not extract_order_id(t or ""):
        return False
    tl = f" {(t or '').lower()} "
    markers = (
        "track", "status", "delivery", "shipment", "parcel", "package",
        "kahan", " kab ", "kab tk", "kab tak", "aayega", "aaega", "aa jayega",
        "pahuch", "pohuch", "order id", "orderid",
        "bta ", "btao", "batao", "bataye", "bata de", "btade", "check",
        "live ", "yeh le", "ye le", " le id", "lo id", "hai id",
        "bhej", "paste", "tu hi", "tuhi", "dekh", "pata", "bta de",
        "refund", "paise", "wapas", "money back", "kb tk", "kitne din",
        "atak", "atka", "atak rha", "atka h", "atka hai", "stuck", "delay",
        "kaha ", "kahan ", "kidhar", "kaha atak", "kahan atak",
    )
    return any(m in tl for m in markers)


def _text_is_live_order_lookup_intent(t: str, conversation_context: str = "") -> bool:
    """
    User pasted/named an Order ID and wants live status/refund/tracking — not history list or placement how-to.
    Lightweight checks only (no extract_order_id) to avoid helper recursion.
    """
    from services.order_details_flow import _lightweight_details_or_invoice_signal

    if _lightweight_details_or_invoice_signal(t or ""):
        return False
    turn = _normalize_order_chat_text(t or "")
    if not turn.strip():
        return False
    if _text_is_refund_return_policy_howto(turn):
        return False
    if _message_has_embedded_pincode_for_delivery(turn, conversation_context):
        return False
    tl = f" {turn.lower()} "
    mentions_oid = "order id" in tl or "orderid" in tl
    has_numeric_id = bool(re.search(r"\b\d{4,20}\b", turn))
    has_six_digit = bool(re.search(r"\b[1-9]\d{5}\b", turn))
    if _text_has_refund_or_return_intent(turn) and (mentions_oid or has_numeric_id):
        return True
    status_markers = (
        "check", "status", "kb tk", "kab tak", "kab milega", "kab aayega", "dekh ke bta",
        "check krke", "check karke", "bta de", "bata de",
    )
    if has_numeric_id and any(m in tl for m in status_markers):
        return True
    if not has_numeric_id and any(m in tl for m in status_markers) and any(
        p in tl for p in ("iske", "iski", "iska", "is order", "ye wala", "yeh wala")
    ):
        if _conversation_in_order_tracking_flow(conversation_context):
            return True
    if mentions_oid and any(
        x in tl
        for x in (
            "check", "bta", "bata", "refund", "kb tk", "kab tak", "status", "track",
            "kahan", "aa jayega", "atak", "atka",
        )
    ):
        return True
    if has_numeric_id and mentions_oid and any(
        x in tl for x in ("check", "bta", "bata", "batao", "btao", "de")
    ):
        return True
    status_markers = (
        "check", "bta", "bata", "status", "track", "kahan", "kaha ", "atak", "atka",
        "stuck", "delay", "kab ", "kb tk", "refund", "dekh", "pata",
    )
    pronoun_markers = ("iska", "iske", "iski", "is order", "ye wala", "yeh wala", "us order", "iski")
    if has_numeric_id and any(x in tl for x in status_markers):
        if has_six_digit and not mentions_oid and not re.search(r"\border\b", tl):
            if _conversation_bot_asked_for_pincode(conversation_context):
                return False
            if _text_has_explicit_pincode_subject(turn) or re.search(
                r"\b(?:pincode|pin\s*code|pin)\b", turn, re.I
            ):
                return False
        return True
    if has_numeric_id and any(x in tl for x in pronoun_markers) and any(
        x in tl for x in status_markers + ("bta", "bata", "check", "dekh")
    ):
        return True
    return False


def _should_use_deterministic_core_route(t: str, conversation_context: str = "") -> bool:
    """Skip phrase-based core routes when user needs AI/live order/refund understanding."""
    if _text_is_live_order_lookup_intent(t, conversation_context):
        return False
    return True


def _conversation_bot_offered_order_id_or_tracking(conversation_context: str) -> bool:
    """True if recent assistant text invited the user to send Order ID or offered live tracking."""
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4000:]
    low = tail.lower()
    if ("order id" in low or "orderid" in low) and any(
        x in low
        for x in ("paste", "send", "share", "optional", "here", "live status", "tracking", "check", "yahan", "yaha")
    ):
        return True
    if any(x in tail for x in ("ऑर्डर आईडी", "ऑर्डर id", "भेज", "वैकल्पिक", "ट्रैक", "लाइव", "स्थिति")):
        return True
    if any(
        x in low
        for x in (
            "how to track order",
            "where is my order id",
            "step-by-step",
            "step by step",
            "explain step",
            "tracking_help",
            "how to track your order",
        )
    ):
        return True
    return False


def _should_release_order_id_awaiting_for_routing(combined_text: str) -> bool:
    """Leave order-id lock when the user asks a new in-domain question (not an ID)."""
    if not combined_text:
        return False
    if turn_is_catalog_product_lookup(combined_text, combined_text):
        return True
    if extract_order_id(combined_text):
        return False
    if _text_asks_wishlist(combined_text) or _text_asks_how_to_view_wishlist(combined_text):
        return True
    if message_is_user_feedback_or_closing(combined_text):
        return True
    if _text_is_tracking_howto_request(combined_text):
        return False
    if _text_is_order_id_help_request(combined_text):
        return False
    return _should_bypass_warm_greeting_fast_path(combined_text)


def _text_has_past_order_complaint_context(t: str) -> bool:
    """User describes a bought item problem — not shopping for that SKU."""
    if not (t or "").strip():
        return False
    tl = f" {_normalize_order_chat_text(t).lower()} "
    if any(
        x in tl
        for x in (
            "defective", "defect", "damaged", "damage", "kharab", "kharaab", "galat mila",
            "wrong item", "wrongg product", "wrongg item", "claim", "complaint",
            "mangaya", "managaya", "mangwaya", "mangwayi", "mangayi",
            "order kiya", "nikal gya", "nikal gaya", "nikla", "return",
            "photo me", "photo mein", "dikh rha", "dikh raha", "jaisa nahi", "jese nahi", "wese nahi",
            "replacement", "refund", "nahi mila", "nhi mila", "fir bhi nhi", "fir bhi nahi",
            "claim dala", "claim daala", "claim lagaya",
            "alag aaya", "alag aa", "galat color", "galat colour", "wrong color", "wrong colour",
            "color change", "colour change", "color alag", "colour alag", "rang alag",
            "aaya kuchh aur", "aaya kuch aur", "aya kuchh aur", "aya kuch aur",
            "kuchh aur aaya", "kuch aur aaya", "kuchh aur aya", "kuch aur aya",
            "aaya alag", "aya alag", "galat product", "wrong product", "different item",
            "galat order", "galat item", "galat aaya", "galat aya",
            "nhi aaya", "nahi aaya", "nhi aya", "nahi aya", "abhi tak nhi", "abhi tak nahi",
            "ab tak nhi", "ab tak nahi", "tk bhi nhi", "tak bhi nahi",
            "kiya tha aaya", "kiya tha aya", "order kuchh aur", "order kuch aur",
        )
    ):
        return True
    if re.search(r"\b(?:mene|maine|humne)\s+order\b", tl) and re.search(
        r"\b(?:aaya|aya|aa gya|aa gaya|mila|mili|aayi|ayi)\b", tl
    ):
        return True
    if re.search(r"\bki thi\b", tl) and re.search(
        r"\b(?:aaya|aya|aa gya|aa gaya|aa \w+ gya|aa \w+ gaya)\b", tl
    ):
        return True
    if "ki thi aa" in tl or "ki thi aaya" in tl or "ki thi aya" in tl:
        return True
    if re.search(r"\bkiya tha\b", tl) and re.search(r"\b(?:aaya|aya)\b", tl):
        return True
    if re.search(r"\border\b", tl) and re.search(
        r"\b(?:galat|wrong|alag)\b", tl
    ) and re.search(r"\b(?:aaya|aya|mila|mili)\b", tl):
        return True
    return False


def _text_is_order_tracking_only_issue(t: str) -> bool:
    """Undelivered / track status — live API. Wrong-item remedy goes to refund KB."""
    if not (t or "").strip():
        return False
    if _text_is_refund_return_policy_howto(t) or message_needs_policy_answer(t):
        return False
    if _text_has_past_order_complaint_context(t) and not _text_is_undelivered_order_complaint(t):
        return False
    if _text_is_undelivered_order_complaint(t) or _text_is_order_tracking_intent(t):
        return True
    return False


def message_is_bot_search_complaint(text: str, conversation_context: str = "") -> bool:
    """User is upset that the bot ran product search — not asking to shop."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_conversational_text(text)} "
    search_gripe = any(
        x in tl
        for x in (
            "search krke", "search karke", "search kyu", "search kyo", "product search",
            "order search", "tu search", "tum search", "aap search", "search nhi", "search ni",
            "search nahi", "theek search", "sahi search", "search theek", "kyu de rha",
            "kyu de raha", "kyo de rha", "kyu dikha", "kyu dikha rha", "galat jawab",
            "galat reply", "wrong reply", "wrong answer",
        )
    ) or (
        "search" in tl and "krke" in tl and any(x in tl for x in ("kyu", "kyo", "de rha", "de raha"))
    )
    if not search_gripe:
        return False
    if any(
        x in tl
        for x in (
            "galat order", "galat product", "wrong order", "wrong product", "puchh", "pucha",
            "bol rha", "bol raha", "kya karu", "kya kr", "mere pass", "mere paas", "order aaya",
            "kyu de rha", "kyu de raha", "kyo de rha", "krke kyu", "search kyu",
        )
    ):
        return True
    ctx = f" {(conversation_context or '').lower()} "
    if _text_has_past_order_complaint_context(ctx) or _text_has_past_order_complaint_context(tl):
        return True
    return False


def message_is_assistant_identity_question(text: str, ai_route: dict | None = None) -> bool:
    """
    Bot-identity / what-are-you-doing — semantic (LLM meaning + embeddings), not phrase lists.
    Optional ai_route when post-routing.
    """
    from services.meta_turn_semantics import (
        ai_route_is_assistant_intro_turn,
        should_fast_reply_assistant_intro,
    )

    parts = (text or "").strip()
    if ai_route:
        return ai_route_is_assistant_intro_turn(ai_route, parts, "")
    return should_fast_reply_assistant_intro(parts, "")


def message_is_assistant_identity_question_failsafe(text: str) -> bool:
    """Last resort when LLM and embeddings both unavailable — high-confidence phrases only."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_conversational_text(text)} "
    tokens = re.findall(r"[a-z0-9]+", tl)
    if len(tokens) > 14:
        return False
    if _text_mentions_welfog_brand(tl) and any(
        x in tl for x in ("welfog kya", "welfog kon", "welfog kaun", "founder", "ceo", "company")
    ):
        if not re.search(r"\b(?:tu|tum|aap|you|bot)\s+(?:kon|kaun|who)\b", tl):
            return False
    if _message_has_catalog_product_signal(text) or _text_is_order_tracking_intent(text):
        return False
    if re.search(r"\b(?:tu|tum|aap|you)\b", tl) and re.search(r"\b(?:kon|kaun|who)\b", tl):
        return True
    if re.search(r"\b(?:kya|what)\s+(?:kar|doing)\b", tl) and re.search(
        r"\b(?:tu|tum|aap|you|bot)\b", tl
    ):
        return True
    return False


def _is_assistant_identity_question(text: str) -> bool:
    """Who are you / bot identity — regex only (no embeddings before reply)."""
    tl = f" {(text or '').lower()} "
    if any(
        p in tl
        for p in (
            "who are you",
            "who r you",
            "who is this",
            "what are you",
            "tu kaun",
            "tum kaun",
            "aap kaun",
            "aap kon",
            "kon ho",
            "kaun ho",
            "tum kya ho",
            "aap kya ho",
            "your name",
            "bot ho",
            "ai ho",
            "chatbot ho",
        )
    ):
        if not _message_has_catalog_product_signal(text):
            return True
    return False


def build_assistant_intro_reply(original_msg: str, msg_en: str = "", reply_lang: str = "") -> str:
    from services.kb_service import sysmsg
    from services.translation_service import is_hinglish_message

    use_hinglish = (reply_lang or "").lower() == "hinglish" or (
        not reply_lang and is_hinglish_message(original_msg or msg_en)
    )
    key = "assistant_intro_hinglish" if use_hinglish else "assistant_intro"
    return sysmsg(key) or sysmsg("assistant_intro") or sysmsg("how_can_i_help_welfog") or ""


def message_is_bot_capability_question(text: str) -> bool:
    """Meta questions about what the assistant can show — not a catalog search."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_conversational_text(text)} "
    if _message_has_catalog_product_signal(text) and any(
        x in tl for x in ("dikhao", "dikha", "chahiye", "lena", "kharid", "buy", "show me")
    ):
        return False
    if (
        re.search(r"\bjo\s+(?:bolu|bolun|bolenge)\b", tl)
        and re.search(r"\b(?:dega|dikha|dikhaye|milega|milta)\b", tl)
    ):
        return True
    if (
        re.search(r"\bjo\s+(?:chahiye|chaiye)\b", tl)
        and re.search(r"\b(?:tere\s+pass|tumhare\s+pass|aapke\s+pass|pass\s+h)\b", tl)
    ):
        return True
    if re.search(r"\b(?:dikha|dikhao)\s+dega\s+ky", tl):
        return True
    if any(x in tl for x in ("jo bolu", "jo chahiye", "jo chaiye", "tere pass", "tumhare pass")):
        if any(x in tl for x in ("dega", "milega", "milta", "h kya", "hai kya", "dikha dega")):
            if len(re.findall(r"[a-z]+", tl)) <= 12:
                return True
    return False


def _text_is_order_delivery_issue(t: str) -> bool:
    """Past order not received / wrong item / track status — never catalog product search."""
    if not (t or "").strip():
        return False
    if _text_is_undelivered_order_complaint(t):
        return True
    if _text_is_order_tracking_intent_leaf(t):
        return True
    if _text_has_past_order_complaint_context(t):
        return True
    return False


def _text_asks_customer_care_contact(t: str) -> bool:
    """
    True when the user wants official customer-care phone/email/chat — not shopping for a handset.
    Mentioning 'iphone cover' in a complaint must NOT block this (past purchase context).
    """
    tl = f" {t.lower()} "
    complaint_ctx = _text_has_past_order_complaint_context(tl)
    wants_contact = any(
        x in tl
        for x in (
            "number ded", "number de", "number do", "number dena", "number bhej", "number btao",
            "number batao", "mail dena", "email dena", "mail de", "email de", "mail do", "email do",
            " dena", " dedo", " de do", "dedo", "bhej do", "bhej de",
            "no dedo", "no de do", "no. de", "call kr", "call kar", "phone no", "phone number",
            "contact number", "helpline no", "baat kr", "baat kar", "se baat", "unse baat",
            "customer care ke", "cust care ke", "care ke number", "support ke number",
            "contact kese", "contact kaise", "kese kru", "kaise kru", "kese contact", "kaise contact",
            "se contact kese", "se contact kaise",
        )
    )
    if complaint_ctx and wants_contact:
        return True
    if any(
        p in tl
        for p in (
            "customer care",
            "cust care",
            "customer support",
            "cust support",
            "welfog support",
            "welfog customer",
            "support team",
            "helpline",
            "help line",
            "call center",
            "call centre",
            "support email",
            "support mail",
            "care email",
            "care number",
            "support number",
            "contact welfog",
            "welfog contact",
            "welfog ka number",
            "welfog number",
            "talk to welfog",
            "talk with welfog",
            "speak to welfog",
            "speak with welfog",
            "reach welfog",
            "call welfog",
            "human agent",
            "real person",
            "customer service",
            "baat krna",
            "baat karna",
            "se contact",
            "grievance officer",
            "complaint officer",
        )
    ):
        if complaint_ctx:
            return True
        if any(
            b in tl
            for b in (
                "buy mobile",
                "buy phone",
                "buy iphone",
                "buy samsung",
                "smartphone under",
                "dikhao",
                "dikha",
                "dikhao",
                "chahiye",
                "milega",
                "milta",
            )
        ) and not wants_contact:
            if "mobile cover" in tl or "phone cover" in tl:
                return False
        return True
    if any(p in tl for p in ("mobile number", "phone number", "contact number", "email id", "mail id", "whatsapp number")):
        if any(x in tl for x in ("welfog", "customer", "support", "care", "company", "official")):
            if complaint_ctx:
                return True
            if any(b in tl for b in ("buy mobile", "buy phone")) and "buy" in tl:
                return False
            if "iphone" in tl and complaint_ctx:
                return True
            if any(b in tl for b in ("buy mobile", "buy phone", "iphone", "samsung galaxy")) and not wants_contact:
                return False
            return True
    return False


def message_needs_policy_answer(t: str) -> bool:
    """
    Return/refund/wrong-item/exchange policy — answer from KB, not product catalog.
    Handles long Hinglish messages with multiple questions in one bubble.
    """
    if _text_is_refund_return_policy_howto(t):
        return True
    tl = f" {t.lower()} "
    if _text_has_refund_or_return_intent(t):
        if _text_has_past_order_complaint_context(tl) or any(
            x in tl for x in ("kya ", "ky ", "kitne din", "kab ", "kaise", "kes ", "ho sk", "kar sk", "kr sk", "sakta", "skta")
        ):
            return True
        if any(x in tl for x in ("policy", "process", "eligible", "allowed", "time", "timeline", "days", "din")):
            return True
    if _text_has_past_order_complaint_context(tl) and any(
        x in tl
        for x in (
            "return", "refund", "exchange", "replacement", "wapas", "paise",
            "kitne din", "kab milega", "kab aayega", "kya kar", "kya kr", "ho skta", "ho sakta",
            "kar skta", "kar sakta", "kr skta", "return ho", "return kar",
        )
    ):
        return True
    if _looks_like_policy_faq_message(t) and _text_has_past_order_complaint_context(tl):
        return True
    return False


def _catalog_product_signal_words() -> frozenset:
    """Single source: welfog_api PRODUCT_TYPE_NOUNS + common brand/device tokens."""
    try:
        from services.welfog_api import PRODUCT_TYPE_NOUNS

        base = set(PRODUCT_TYPE_NOUNS)
    except ImportError:
        base = set()
    base.update(
        {
            "iphone", "samsung", "oneplus", "redmi", "vivo", "oppo", "nokia", "realme",
            "laptop", "tv", "kurta", "sneaker", "loafer", "loafers",
            "suit", "bat", "cricket", "watch", "watches", "dress", "shirt", "tshirt",
        }
    )
    return frozenset(base)


_SHOPPING_ACTION_MARKERS = (
    "show", "buy", "need", "want", "looking for", " search ", " find ", "price ", " rate ",
    "cheap", "under rs", "under ₹", "purchase", "shop ",
    "dikha", "dikhao", "dikh", "dikhaa", "dikhaan",
    "milta", "milti", "milega", "milegi",
    "mil sk", "mil sak", "mile sk", "mile sak",
    "chahiye", "chiye", "chaahiye",
    " hai kya", " h kya", " hai ky", " h ky",
    " he kya", " he ky",
    " apke pas", " apke paas", " aapke pas", " aapke paas",
    " tumhare pas", " tumhare paas",
    "bta de", "bta do", "btao de", "bata de", "batao de",
    "dikha de", "dikhao de", "la de", "la do",
)

_KNOWLEDGE_INFO_TOPICS = (
    "privacy policy",
    "privacy",
    "data protection",
    "personal data",
    "terms and conditions",
    "terms of service",
    "terms of use",
    "terms & conditions",
    "shipping policy",
    "delivery policy",
    "payment policy",
    "refund policy",
    "return policy",
    "cancellation policy",
    "exchange policy",
    "seller policy",
    "cookie policy",
    "legal notice",
    "grievance officer",
    "disclaimer",
)

_INFO_READ_VERBS = (
    "dikhao", "dikha", "dikhana", "dikhado", "dikhaa", "dikhaan",
    "batao", "bata", "btao", "bta", "btana", "batana", "bata de", "bta de",
    "show", "tell", "explain", "read", "samjhao", "samjha", "samjh",
    "kya hai", "what is", "what's", "details", "detail", "jaankari", "jankari",
    "kesi", "kaisi", "kaise", "kaisi",
)


def _text_has_kb_topic_hint(t: str) -> bool:
    """Cheap scan — avoid heavy policy/about/social checks on every /chat turn."""
    if not (t or "").strip():
        return False
    tl = f" {_normalize_policy_typos(t)} "
    for topic in _KNOWLEDGE_INFO_TOPICS:
        if topic in tl:
            return True
    if "policy" in tl or "policies" in tl or "privacy" in tl:
        return True
    if any(x in tl for x in (" faq", "faqs", "help & support", "help and support")):
        return True
    if _text_mentions_welfog_brand(tl) and any(
        x in tl
        for x in (
            " kya hai",
            " kya h ",
            " what is ",
            " ke baare",
            " about ",
            " company ",
            " story",
        )
    ):
        return True
    return False


def _text_is_sim_ejector_pin_product_request(t: str) -> bool:
    """SIM tray ejector pin/tool — product buy, NOT delivery pincode."""
    if not (t or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(t).lower()} "
    if not re.search(r"\bsim\b", tl):
        return False
    if not (
        re.search(r"\bpin\b", tl)
        or re.search(r"\bnikal", tl)
        or "ejector" in tl
        or "tray" in tl
    ):
        return False
    if re.search(r"\b(?:pincode|pin\s*code|postal|zip\s*code)\b", tl):
        return False
    if re.search(r"\b[1-9]\d{5}\b", t or ""):
        return False
    return True


def _message_has_catalog_product_signal(t: str) -> bool:
    """True when the user names a buyable SKU type (not just 'dikhao' alone)."""
    if _is_welfog_about_fast_path(t):
        return False
    if _text_is_sim_ejector_pin_product_request(t):
        return True
    tl = f" {_normalize_welfog_typos(t)} "
    words = _catalog_product_signal_words()
    tokens = set(re.findall(r"[a-z0-9]+", tl))
    for w in words:
        if f" {w} " in f" {tl} " or tl.strip().endswith(f" {w}") or w in tokens:
            return True
        if f"{w}s" in tokens or f" {w}s " in f" {tl} ":
            return True
    if not any(m in tl for m in _SHOPPING_ACTION_MARKERS):
        return False
    try:
        from services.product_browse_semantics import token_is_product_noun

        for tok in tokens:
            if len(tok) >= 3 and token_is_product_noun(tok):
                return True
    except ImportError:
        pass
    return False


def _message_has_generic_shopping_item_signal(t: str) -> bool:
    """
    Shopping verb + concrete item words — no fixed product list required.
    e.g. 'pressure cooker dikhao', 'sofa chahiye', 'organic honey milega kya'.
    """
    tl = f" {_normalize_welfog_typos(t)} "
    if not any(m in tl for m in _SHOPPING_ACTION_MARKERS):
        return False
    filler = {
        "welfog", "wlefog", "welkog", "please", "pls", "bhai", "sir", "madam", "na", "hi", "hello",
        "the", "a", "an", "and", "or", "aur", "for", "me", "my", "your", "kya", "ky", "hai", "hain",
        "ho", "ka", "ki", "ke", "ko", "se", "par", "pe", "per", "main", "mai", "mujhe", "mereko",
        "dikhao", "dikha", "dikh", "dikhaa", "dikhaan", "dikhado", "dikhana", "chahiye", "chiye",
        "milega", "milegi", "milta", "milti", "bata", "batao", "btao", "bta", "show", "buy", "need",
        "want", "find", "search", "shop", "some", "kuch", "koi", "ek", "do", "teen",
        "help", "queries", "query", "question", "questions", "actually", "just", "only",
    }
    tokens = re.findall(r"[a-z0-9]+", tl)
    content = [tok for tok in tokens if len(tok) >= 3 and tok not in filler]
    if not content:
        return False
    # Regex-only — extract_pincode_preferred_from_message recurses via order-id helpers.
    pm = re.search(r"\b([1-9]\d{5})\b", t or "")
    pin = pm.group(1) if pm else ""
    if pin and pin in content and re.search(
        r"\b(ispe|isme|ispe|uspe|yahan|yaha|idhar|waha)\b", tl
    ):
        non_pin = [c for c in content if c != pin]
        if not non_pin or all(c in ("ispe", "isme", "uspe") for c in non_pin):
            return False
    return len(content) >= 1


_POLICY_TOPIC_WORDS = frozenset(
    {
        "privacy",
        "policy",
        "policies",
        "terms",
        "conditions",
        "data protection",
        "disclaimer",
        "legal",
    }
)

# Not a company name when it appears before "ki/ka" in a policy question.
_POLICY_KI_STOPWORDS = frozenset(
    {
        "welfog",
        "wlefog",
        "welkog",
        "welfrog",
        "welefog",
        "hamari",
        "hamara",
        "hamare",
        "official",
        "meri",
        "mere",
        "mera",
        "mujhe",
        "mereko",
        "apni",
        "apna",
        "apne",
        "aapki",
        "aapka",
        "koi",
        "kuch",
        "kisi",
        "sab",
        "poori",
        "puri",
        "show",
        "dikhao",
        "dikha",
        "dikhana",
        "batao",
        "bata",
        "btao",
        "bta",
        "please",
        "bhai",
        "na",
        "the",
        "this",
        "that",
        "privacy",
        "policy",
        "terms",
        "conditions",
        "company",
        "platform",
        "website",
        "app",
        "site",
        "information",
        "info",
        "faq",
        "faqs",
        "kesi",
        "kaisi",
        "kaise",
        "kya",
        "ky",
        "hai",
        "hain",
        "ho",
        "btana",
        "btana",
        "batao",
        "btao",
        "bata",
        "bta",
        "tell",
        "about",
        "regarding",
        "related",
        "order",
        "orders",
        "payment",
        "shipping",
        "delivery",
        "return",
        "refund",
        "customer",
        "product",
        "products",
        "seller",
        "data",
        "personal",
        "yeh",
        "ye",
        "woh",
        "wo",
        "is",
        "us",
        "un",
        "in",
        "ke",
        "ki",
        "ka",
    }
)


def _token_is_welfog_variant(tok: str) -> bool:
    """True for Welfog and common misspellings (welefog, wlefog, …)."""
    t = _normalize_welfog_typos((tok or "").lower().strip())
    if not t:
        return False
    if t == "welfog" or t.startswith("welfog"):
        return True
    try:
        import difflib

        if difflib.SequenceMatcher(None, t, "welfog").ratio() >= 0.72:
            return True
    except Exception:
        pass
    return False


def _looks_like_third_party_brand_token(tok: str) -> bool:
    """Candidate company/brand name in a policy question (not Welfog, not Hindi filler)."""
    t = (tok or "").lower().strip()
    if len(t) < 3 or t.isdigit():
        return False
    if t in _POLICY_KI_STOPWORDS:
        return False
    if _token_is_welfog_variant(t):
        return False
    return True


def message_asks_other_company_policy(text: str, conversation_context: str = "") -> bool:
    """User asked for another company's policy — uses policy_scope (LLM + context)."""
    from services.policy_scope import policy_question_is_external_company

    raw = (text or "").strip()
    if not raw:
        return False
    if not any(p in f" {_normalize_policy_typos(raw)} " for p in _POLICY_TOPIC_WORDS):
        return False
    return policy_question_is_external_company(raw, "", conversation_context)


def message_is_knowledge_information_request(
    text: str, conversation_context: str = ""
) -> bool:
    """
    User wants policy / legal / FAQ / company info from KB — NOT catalog product search.
    e.g. 'privacy policy dikhao', 'terms batao', 'welfog ki privacy policy'.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if _looks_like_greeting_message(raw) or _is_short_pure_greeting(raw):
        return False
    if _is_light_smalltalk_fast(raw, ""):
        return False
    if not _text_has_kb_topic_hint(raw) and not _is_welfog_about_fast_path(raw):
        return False

    from services.policy_scope import (
        _user_explicitly_wants_welfog_policy,
        policy_question_is_for_welfog,
    )

    if _text_is_pincode_serviceability_question_light(text or ""):
        return False

    if _turn_is_catalog_product_request(text) or (
        _message_has_catalog_product_signal(text)
        and any(m in f" {(text or '').lower()} " for m in _SHOPPING_ACTION_MARKERS)
    ):
        return False

    if message_asks_other_company_policy(text, conversation_context):
        return False
    if message_asks_welfog_social_media(text):
        return False
    if _user_explicitly_wants_welfog_policy(text, conversation_context):
        return True
    tl_pre = f" {_normalize_policy_typos(text)} "
    if any(p in tl_pre for p in _POLICY_TOPIC_WORDS) and not policy_question_is_for_welfog(
        text, "", conversation_context
    ):
        return False
    if message_is_welfog_about_request(text):
        return True
    tl = f" {_normalize_policy_typos(text)} "
    for topic in _KNOWLEDGE_INFO_TOPICS:
        if topic not in tl:
            continue
        if topic == "privacy" and any(
            x in tl for x in ("screen", "glass", "cover", "case", "protector", "tempered")
        ):
            continue
        return True
    if ("policy" in tl or "policies" in tl) and not _message_has_catalog_product_signal(text):
        if any(v in tl for v in _INFO_READ_VERBS) or "kya" in tl or "what" in tl:
            return True
    if "privacy" in tl and not _message_has_catalog_product_signal(text):
        if any(v in tl for v in _INFO_READ_VERBS) or any(
            x in tl for x in ("kesi", "kaisi", "kaise", "kaisi", "kya", "ky")
        ):
            return True
    if any(x in tl for x in (" faq", "faqs", "help & support", "help and support")):
        if any(v in tl for v in _INFO_READ_VERBS) or "kya" in tl:
            return True
    return False


def message_is_casual_offtopic_not_shopping(text: str) -> bool:
    """Personal / random chat — not Welfog shopping (route general / off_topic, not product)."""
    if message_is_knowledge_information_request(text) or message_is_welfog_about_request(text):
        return False
    tl = f" {text.lower()} "
    if _message_has_catalog_product_signal(text):
        return False
    personal = (
        "do you cook", "can you cook", "where do you live", "where you live",
        "marry me", "date me", "are you human", "are you real", "tell me a joke",
        "cricket score", "weather today", "baarish", "barish", "toofan", "tufan",
        "mausam", "mosam", "kya tum cook", "tum kahan rehte",
        "tumhari umar", "shadi kab", "pyar", "love life",
        "dost nhi mil", "friend nhi mil", "drive kr", "drive kar", "gaadi chala",
        "car drive", "fortuner", "ghumne gya", "ghumne gaya",
    )
    if any(p in tl for p in personal):
        return True
    # Relationship advice only — NOT shopping for gf/bf (suit, watch, gift)
    if re.search(r"\b(gf|bf|girlfriend|boyfriend)\b", tl) and any(
        x in tl for x in ("mil rhi", "mil nahi", "nhi mil", "nahi mil", "kya karu", "kya kru")
    ):
        if not (
            _message_has_catalog_product_signal(text)
            or _message_has_generic_shopping_item_signal(text)
        ):
            return True
    return False


def message_needs_human_support_escalation(t: str) -> bool:
    """
    Complex account/payment/seller issues we may not resolve via API/KB alone —
    offer official customer-care contact after automated help fails.
    """
    if not (t or "").strip():
        return False
    if _text_asks_customer_care_contact(t):
        return False
    if _user_asks_order_history_navigation_help(t):
        return False
    tl = f" {t.lower()} "
    markers = (
        "payment atak", "payment stuck", "payment fail", "payment failed", "payment pending",
        "payment nahi", "paise atak", "paisa atak", "transaction fail", "transaction pending",
        "id block", "account block", "blocked ho", "ban ho gaya", "ban ho gya",
        "supplier panel", "seller panel", "seller account", "vendor panel",
        "upload nahi", "upload fail", "product upload", "id ban nahi", "id nahi ban",
        "register nahi", "signup fail", "sign up fail", "login nahi", "login fail",
        "order nahi ho pa", "checkout fail", "checkout nahi", "place order nahi",
        "bug ki wajah", "technical issue", "system error", "server error",
        "trace nahi", "solve nahi", "fix nahi ho",
    )
    return any(m in tl for m in markers)


def message_is_generic_help_request(t: str) -> bool:
    """
    Broad help opener without a concrete intent yet.
    Should get a warm/menu prompt, not product search or support escalation.
    """
    if not (t or "").strip():
        return False
    raw = (t or "").strip()
    tl = f" {raw.lower()} "
    # Focus on the latest chunk for long prompts.
    tail = tl[-700:] if len(tl) > 700 else tl
    markers = (
        "need your help",
        "need help",
        "help me",
        "help in",
        "i need help",
        "i need your help",
        "can u help",
        "can you assist",
        "please help",
        "some queries",
        "few queries",
        "some query",
        "kuch query",
        "kuchh query",
        "kuch sawal",
        "kuchh sawal",
        "some question",
        "few question",
        "guide me",
        "guide karo",
        "madad karo",
        "madad chahiye",
        "mujhe madad",
        "help karo",
        "meri help",
        "help chahiye",
        "can you help",
        "will you help",
    )
    has_generic_marker = any(m in tail for m in markers)
    # Native-script broad help openers (without concrete shopping/tracking intent).
    native_markers = (
        "मदद", "सहायता", "सवाल", "प्रश्न",
        "ਮਦਦ", "ਸਵਾਲ",
        "મદદ", "પ્રશ્ન",
        "உதவி", "கேள்வி",
        "సహాయం", "ప్రశ్న",
        "ಸಹಾಯ", "ಪ್ರಶ್ನೆ",
        "സഹായം", "ചോദ്യം",
        "সাহায্য", "প্রশ্ন",
        "مدد", "سوال", "سہولت",
    )
    has_native_generic_marker = any(m in raw for m in native_markers)
    if (
        _text_is_order_tracking_intent(t)
        or _text_has_refund_or_return_intent(t)
        or _text_asks_customer_care_contact(t)
        or _text_asks_order_history(t)
        or message_needs_policy_answer(t)
        or message_is_knowledge_information_request(t)
    ):
        return False
    if _text_has_product_shopping_intent(t):
        # "need help" often triggers weak shopping heuristics ("i need ...").
        # Keep generic-help path when there is no concrete catalog signal.
        if not ((has_generic_marker or has_native_generic_marker) and not _message_has_catalog_product_signal(t)):
            return False
    if has_generic_marker or has_native_generic_marker:
        return True
    words = [w for w in re.findall(r"[a-z]+", tail) if w]
    if len(words) <= 6 and any(w in words for w in ("help", "query", "queries", "guide")):
        return True
    return False


def message_needs_support_not_product(t: str) -> bool:
    """Support / customer-care / complaint+contact — never catalog product search."""
    if _message_looks_like_shopping_query(t):
        return False
    if _user_asks_order_history_navigation_help(t):
        return False
    if message_is_bot_search_complaint(t) or message_is_bot_capability_question(t):
        return True
    if message_needs_policy_answer(t):
        return True
    if message_is_knowledge_information_request(t):
        return True
    if message_is_casual_offtopic_not_shopping(t):
        return True
    if _text_asks_customer_care_contact(t):
        return True
    tl = f" {t.lower()} "
    if _text_has_past_order_complaint_context(tl) and any(
        x in tl
        for x in (
            "customer care", "cust care", "support", "helpline", "number", "call", "baat",
            "contact", "email", "mail", "complaint", "claim",
        )
    ):
        return True
    return False


def _normalize_welfog_typos(text: str) -> str:
    t = (text or "").lower()
    for typo, fix in (
        ("welefog", "welfog"),
        ("wlefog", "welfog"),
        ("welfrog", "welfog"),
        ("welkog", "welfog"),
        ("welffog", "welfog"),
        ("welfogg", "welfog"),
        ("wel fog", "welfog"),
        ("wel-fog", "welfog"),
    ):
        t = t.replace(typo, fix)
    return t


def _normalize_policy_typos(text: str) -> str:
    """Welfog + policy word typos for intent detection."""
    t = _normalize_welfog_typos(text)
    for typo, fix in (
        ("priavcy", "privacy"),
        ("priavacy", "privacy"),
        ("privary", "privacy"),
        ("privcy", "privacy"),
        ("polcy", "policy"),
        ("polici", "policy"),
    ):
        t = t.replace(typo, fix)
    return t


def _text_mentions_welfog_brand(text: str) -> bool:
    return "welfog" in _normalize_welfog_typos(text)


_WELFOG_ABOUT_MARKERS = (
    "ke baare",
    "ke barre",
    "ke bare",
    "ke baar",
    "baare bata",
    "baare me bata",
    "baare me",
    "barre me",
    "bare me",
    "baar me",
    "baar eme",
    "about welfog",
    "welfog about",
    "tell me about",
    "batao kuchh",
    "batao kuch",
    "bata kuchh",
    "bata kuch",
    "kuchh welfog",
    "kuch welfog",
    "kya hai welfog",
    "welfog kya hai",
    "welfog kya h",
    "kya h welfog",
    "what is welfog",
    "what's welfog",
    "welfog kya",
    "kon hai welfog",
    "kaun hai welfog",
    "samjhao welfog",
    "samjha welfog",
    "explain welfog",
    "platform kya",
    "company kya",
    "website kya",
    "app kya",
    "kis cheez",
    "kya cheez",
    "kya hota",
    "kya karti",
    "kya karta",
    "kya karte",
    "karti kya",
    "karta kya",
    "welfog karti",
    "welfog karta",
)

_WELFOG_ABOUT_PRODUCT_NOUNS = frozenset(
    {
        "iphone", "mobile", "phone", "laptop", "tv", "tablet", "watch",
        "cover", "case", "charger", "adapter", "cable", "protector", "glass", "bumper",
        "shirt", "tshirt", "pant", "jeans", "dress", "kurta", "shoe", "sneaker", "bag",
        "loafer", "loafers", "rice", "wheat", "jug", "bottle",
    }
)


def message_denies_wishlist_intent(text: str) -> bool:
    """User says they are NOT asking for wishlist (wishlis thodi / wishlist nahi)."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(text).lower()} "
    if message_denies_wishlist_wants_order_history(text):
        return True
    return any(
        x in tl
        for x in (
            "wishlis thodi", "wishlist thodi", "wishlist nahi", "wishlist ni", "wishlist nhi",
            "wishlist nahi maang", "wishlist ni maang", "wishlist mat", "wishlist to nahi",
            "wishlis ni", "wishlis nahi", "not wishlist", "wishlist nahi chahiye",
        )
    )


def message_denies_wishlist_wants_order_history(
    text: str, conversation_context: str = ""
) -> bool:
    """User rejects wishlist and wants purchase/order history (any phrasing)."""
    if not (text or "").strip():
        return False
    tl = f" {_normalize_welfog_typos(text).lower()} "
    if re.search(r"\bwishlist\s+(?:ka|ki|ke)\s+(?:nhi|nahi|ni|nahin)\b", tl):
        if _conversation_bot_showed_order_history_list(conversation_context):
            return True
        if any(
            x in tl
            for x in ("history", "purchase", "order", "mangaya", "mangaye", "bought")
        ):
            return True
        return True
    denies_wl = bool(
        re.search(r"\bwishlist\s+(?:nhi|nahi|ni|nahin|not)\b", tl)
        or re.search(r"\b(?:nhi|nahi|ni|not)\s+wishlist\b", tl)
        or any(x in tl for x in ("wishlist nhi", "wishlist nahi", "wishlist ni", "not wishlist"))
    )
    wants_hist = any(
        x in tl
        for x in (
            "history", "purchase", "purane order", "past order", "my orders",
            "order history", "purchase history", "mangaya", "mangaye", "mangwaya",
            "order kiye", "order kiya", "bought", "ordered",
        )
    )
    if denies_wl and wants_hist:
        return True
    if wants_hist and any(
        x in tl
        for x in (
            "history dekhni", "history dekhna", "history dekhu", "history dekh",
            "purchase ki history", "order ki history",
        )
    ):
        return True
    return False


def message_mentions_wishlist_topic(text: str) -> bool:
    """Wishlist / saved-liked items topic — any script; not purchased order history."""
    if not (text or "").strip() or message_denies_wishlist_intent(text):
        return False
    if message_denies_wishlist_wants_order_history(text):
        return False
    tl = f" {_normalize_welfog_typos(text).lower()} "
    if "wishlist" in tl or "wish list" in tl:
        return True
    if message_is_wishlist_like_request(text):
        return True
    if any(x in tl for x in ("heart icon", "heart wala", "dil wala")) and any(
        x in tl for x in ("saved", "like", "pasand", "favourite", "favorite")
    ):
        return True
    return False


def message_clarifies_wishlist_not_order_history(text: str) -> bool:
    """User corrects bot: wishlist topic, NOT order history (Hinglish/English)."""
    if not (text or "").strip():
        return False
    if message_denies_wishlist_wants_order_history(text):
        return False
    tl = f" {_normalize_welfog_typos(text).lower()} "
    if "wishlist" not in tl and "wish list" not in tl:
        return False
    if re.search(r"\bwishlist\s+(?:ka|ki|ke)\s+(?:nhi|nahi|ni)\b", tl):
        return False
    denies_history = any(
        x in tl
        for x in (
            "history ka nhi", "history ka ni", "history ka nahi", "history nahi",
            "history nhi", "history ni ", "order history nahi", "order history nhi",
            "history nahi puchh", "history nhi puchh", "history nahi maang",
            "history nhi maang", "not history", "not order history",
            "history ki nahi", "history ki ni", "history ka nahi puchh",
        )
    )
    asserts_wishlist = any(
        x in tl
        for x in (
            "wishlist ka", "wishlist ki", "wishlist ke", "wishlist puchh",
            "wishlist puch", "wishlist maang", "wishlist mang", "wishlist dekh",
            "wishlist dekhn", "wishlist dekhna", "wishlist dekhni",
        )
    )
    if denies_history or asserts_wishlist:
        return True
    return any(
        x in tl
        for x in ("puchh rha", "puch raha", "maang rha", "mang raha", "bol rha", "bol raha")
    ) and any(x in tl for x in ("history nhi", "history nahi", "history ni"))


def message_blocks_order_history_routing(text: str) -> bool:
    """Do not route generic app-navigation how-to to order history when wishlist is the topic."""
    if message_denies_wishlist_wants_order_history(text):
        return False
    return message_mentions_wishlist_topic(text) or message_clarifies_wishlist_not_order_history(
        text
    )


def _conversation_bot_showed_order_history_list(conversation_context: str) -> bool:
    """Assistant recently showed purchase history cards in chat."""
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4000:].lower()
    return any(
        m in tail
        for m in (
            "your order history",
            "track order",
            "order placed",
            "payment status",
            "purchase history",
        )
    )


def message_wants_order_history_app_navigation(
    text: str, conversation_context: str = ""
) -> bool:
    """User wants WHERE/HOW to view purchases in the app — not wishlist, not list-in-chat."""
    raw = (text or "").strip()
    if not raw:
        return False
    if message_denies_wishlist_wants_order_history(raw):
        return True
    # Leaf check only — do not call _text_wants_order_history_list_in_chat (recursion via purchases).
    if _order_history_list_in_chat_signals(raw, conversation_context):
        return False
    if not _message_has_app_navigation_intent(raw):
        return False
    tl = f" {_normalize_welfog_typos(raw).lower()} "
    if any(
        x in tl
        for x in (
            "history", "purchase", "purane order", "past order", "my orders",
            "order history", "purchase history", "mangaya", "mangaye", "order kiye",
        )
    ):
        return True
    if _conversation_bot_showed_order_history_list(conversation_context):
        if any(
            x in tl
            for x in (
                "mt dikha", "mat dikha", "mujhe hi bta", "khud", "manual", "manually",
                "app me", "kaise dekhu", "kese dekhu", "kaha jake", "kahan jake",
                "tu mt", "tum mt", "nahi dikha", "ni dikha",
            )
        ):
            return True
    return False


def resolve_navigation_help_topic(
    text: str,
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> str:
    """
    Semantic topic for app-navigation help: wishlist_howto | order_history_howto | "".
    Works across Hinglish/English — corrections and denials beat stale ctx.last.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    if _text_is_order_id_help_request(raw):
        return ""
    if (
        message_clarifies_wishlist_not_order_history(raw)
        or message_is_wishlist_like_request(raw)
        or _user_denies_order_history_wants_saved_or_liked(raw)
    ):
        if _wants_wishlist_list_in_chat(raw) or (
            _text_asks_wishlist(raw) and not _text_asks_how_to_view_wishlist(raw)
        ):
            return ""
        if _text_asks_how_to_view_wishlist(raw, conversation_context) or _message_has_app_navigation_intent(
            raw
        ):
            return "wishlist_howto"
        if _text_asks_wishlist(raw):
            return ""
    if message_denies_wishlist_wants_order_history(raw, conversation_context):
        return "order_history_howto"
    if _text_asks_to_view_purchase_or_order_history(raw):
        return "order_history_howto"
    if message_wants_order_history_app_navigation(raw, conversation_context):
        return "order_history_howto"
    if _user_asks_order_history_navigation_help(raw, conversation_context):
        return "order_history_howto"
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        rh = (ai_route.get("route_handler") or "").strip()
        reasoning = (ai_route.get("reasoning") or "").lower()
        if message_denies_wishlist_wants_order_history(raw):
            return "order_history_howto"
        if intent == "order_history" and _message_has_app_navigation_intent(raw):
            if not _text_wants_order_history_list_in_chat(raw, conversation_context):
                if any(
                    x in raw.lower()
                    for x in ("app me", "kaise dekhu", "kaha jake", "manually", "khud", "mt dikha", "mat dikha")
                ):
                    return "order_history_howto"
        if (intent == "wishlist" or rh == "wishlist_howto_kb") and not message_denies_wishlist_wants_order_history(
            raw
        ):
            if "purchase history" in reasoning or "order history" in reasoning:
                if "not wishlist" in reasoning or "wishlist not" in reasoning:
                    return "order_history_howto"
            elif _text_asks_how_to_view_wishlist(raw, conversation_context):
                return "wishlist_howto"
    return ""


def _order_history_list_in_chat_signals(
    raw: str, conversation_context: str = ""
) -> bool:
    """
    True when user wants purchase list IN CHAT (API), not app how-to.
    Leaf helper — no message_wants / message_asks cross-calls.
    """
    if not (raw or "").strip():
        return False
    if _text_asks_to_view_purchase_or_order_history(raw) and _message_has_app_navigation_intent(raw):
        return False
    tl = f" {raw.lower()} "
    possessive = any(
        x in tl
        for x in (" meri ", " mere ", " mera ", " mujhe ", " my ", " apni ", " apna ", " apne ")
    )
    has_hist = any(
        x in tl
        for x in (
            "order history", "order hist", "orders history", "purchase history",
            "purane order", "past order", "jo order kiye", "order kiye",
        )
    ) or ("order" in tl and "history" in tl)
    list_verbs = (
        "dikhao", "dikha", "dikhe", "dikha de", "dikhado", "dikha do",
        "bta de", "btao", "bata de", "batao", "tu hi bta", "tu bta",
        "show me", "show my", "tell me my", "give me my", "list dede", "list de",
    )
    if has_hist and any(v in tl for v in list_verbs) and possessive:
        return True
    if has_hist and possessive and re.search(r"\b(?:de|dede|de do|dena|bta|btao|bata)\b", tl):
        return True
    if conversation_context and _conversation_bot_showed_order_history_steps(conversation_context):
        follow_up = ("tu hi", "tu bta", "history maang", "list maang", "chat me", "khud dikha")
        if any(x in tl for x in follow_up) and (has_hist or any(v in tl for v in list_verbs)):
            return True
    return False


def _my_welfog_purchases_data_request_signals(text: str) -> bool:
    """
    Past orders / what I bought — leaf helper (no navigation helpers).
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if message_is_wishlist_like_request(raw):
        return False
    if _text_asks_to_view_purchase_or_order_history(raw):
        if _message_has_app_navigation_intent(raw) or "kaha" in raw.lower() or "app" in raw.lower():
            return False
    if _text_is_live_order_lookup_intent(raw):
        return False
    if _text_is_undelivered_order_complaint(raw):
        return False
    if _user_rejects_order_history_wants_tracking(raw):
        return False
    tl_deny = f" {raw.lower()} "
    if re.search(r"\border\s*id\b", tl_deny) and any(
        x in tl_deny for x in (" nhi ", " nahi ", " ni ", " nahin ", " not ")
    ):
        if any(o in tl_deny for o in ("history", "purchase", "my orders")):
            return False
    tl = f" {_normalize_welfog_typos(raw).lower()} "
    if "wishlist" in tl or "wish list" in tl:
        if not message_denies_wishlist_intent(text):
            return False

    show_verbs = (
        "dikha", "dikhao", "dikha de", "dikha do", "data", "list", "bata", "batao",
        "btao", "puchh", "puch", "dekh", "show", "dede", "de do", "bhejo", " de ",
    )
    bought_verbs = (
        "mangaya", "mangaye", "managaye", "mangwaya", "mangwaye", "mangaye h", "mangaya h",
        "mangwaya tha", "mangaye the", "mangaya the", "kiye the", "kiya tha",
        "kre the mene", "kiya mene", "kiye mene", "order kiye", "order kiya", "mangaye h",
    )
    if message_denies_wishlist_intent(text) and any(b in tl for b in bought_verbs):
        if any(v in tl for v in show_verbs):
            return True

    if any(b in tl for b in bought_verbs) and any(v in tl for v in show_verbs):
        if not any(x in tl for x in ("track", "kab aayega", "kab aaega", "status kahan", "wishlist")):
            if any(
                x in tl
                for x in (
                    "mene", "maine", "mere", "mera", "meri", "saman", "samaan", "samaan",
                    "order", "welfog", "wlefog", "welkog", "purane", "ab tk", "ab tak", "jo bhi",
                )
            ):
                return True

    purchase_markers = bought_verbs + (
        "orders dekh", "order dekhne", "mere order", "meri order", "mere orders",
        "meri orders", "dikha deg", "dikha dega", "dikhao deg",
        "ab tak mang", "abhi tk mang", "abhi tak mang", "purane order", "jo ab tak",
    )
    if any(m in tl for m in purchase_markers):
        if _text_mentions_welfog_brand(tl) or "order" in tl or ("saman" in tl and "mang" in tl):
            if not any(x in tl for x in ("track", "kab aayega", "kab aaega", "status kahan")):
                return True
    if "kya kya" in tl and _text_mentions_welfog_brand(tl):
        if any(m in tl for m in ("mangaya", "mangaye", "mangwaya", "mene", "maine", "kiye", "kiya", "mang")):
            if not any(m in tl for m in ("milta", "milega", "milti", "sell", "available", "kharid")):
                return True
    if re.search(r"\bpurchased?\b", tl) and any(v in tl for v in show_verbs):
        if any(x in tl for x in ("mere", "meri", "my", "mujhe", "mene", "maine", "items", "item")):
            return True
    if "items" in tl and any(p in tl for p in ("purchased", "purchase", "bought", "ordered")):
        if any(v in tl for v in ("list", "dede", "de do", "bata", "btao", "dikha", "show", " de ")):
            return True
    return False


def message_asks_my_welfog_purchases(text: str) -> bool:
    """User wants THEIR past orders shown — delegates to leaf signals (no recursion)."""
    return _my_welfog_purchases_data_request_signals(text)


def message_is_seller_on_welfog_request(text: str) -> bool:
    """User wants to sell / become seller on Welfog — not 'what is Welfog' company story."""
    tl = f" {_normalize_welfog_typos(text)} "
    if not _text_mentions_welfog_brand(tl) and "sell" not in tl and "seller" not in tl:
        return False
    seller_markers = (
        "sell on", "sell kr", "sell kar", "seller account", "become seller", "become a seller",
        "create seller", "create a seller", "seller create", "account create", "account bana", "account ban",
        "register seller", "seller register", "seller registration", "new seller",
        "vendor", "supplier", "seller panel", "seller login", "seller ban", "seller bana",
        "listing", "sell krna", "sell karna", "beche", "bechna", "sell on welfog",
    )
    if any(x in tl for x in seller_markers):
        return True
    if "sell" in tl and _text_mentions_welfog_brand(tl):
        return True
    return False


def _text_mentions_social_platform(t: str) -> bool:
    tl = f" {(t or '').lower()} "
    markers = (
        "instagram", " insta ", " insta.", " insta,", "linkedin", "linkdin",
        " facebook", " fb ", "youtube", " you tube", " twitter", " twiter ",
        " twittr ", " social media",
        " social link", " social account", " official link", " official page",
        " handle", " profile", " x.com", " reels",
    )
    if any(m in tl for m in markers):
        return True
    if re.search(r"\bx\.com\b", tl):
        return True
    if re.search(r"\b(?:twiter|twittr|twitt?er|twitter|tweet)\b", tl):
        return True
    if re.search(r"\byout+ube\b", tl):
        return True
    return False


def _is_welfog_social_followup(text: str, conversation_context: str) -> bool:
    """e.g. 'twitter bhi de' right after we showed Welfog social — NOT 'agoda ki bhi linkdin'."""
    if not (text or "").strip():
        return False
    if _external_entity_social_request(text):
        return False
    if not _conversation_recent_welfog_social(conversation_context):
        return False
    tl = f" {_normalize_welfog_typos(text)} "
    # "agoda/raju/microsoft ki bhi ..." — another entity, not Welfog add-on
    subj_bhi = re.search(r"\b([a-z][a-z0-9]{2,20})\s+k[ai]\s+bhi\b", tl)
    if subj_bhi and _social_request_subject_is_external(subj_bhi.group(1)):
        return False
    platform_bhi = re.search(
        r"\b(?:instagram|insta|youtube|yout+ube|twiter|twittr|twitt?er|twitter|"
        r"linkedin|linkdin|facebook|fb)\w*\s+(?:k[e]?\s+)?bhi\b",
        tl,
    )
    if platform_bhi:
        return True
    if re.search(r"\b(?:ki|ke|k)\s+bhi\b", tl) and _text_mentions_social_platform(tl):
        if _text_mentions_welfog_brand(tl) or any(
            p in tl
            for p in (
                " apni ", " apna ", " hamari ", " hamara ", " meri ", " mera ",
                " teri ", " tera ", " tumhari ", " tumhara ", " aapki ", " aapka ",
            )
        ):
            return True
        return False
    if re.search(r"\b(bhi|aur)\s+(?:de|do|dena|dede|dedo|bhej|share|send|bata)\b", tl) and (
        _text_mentions_social_platform(tl)
        or re.search(
            r"\b(?:twiter|twittr|insta|instr?gr?am|youtube|yout+ube|linkdin|twitter|facebook|fb)\b",
            tl,
        )
    ):
        if _text_mentions_welfog_brand(tl):
            return True
        if not subj_bhi:
            return True
        return False
    if re.search(r"\bbhi\b", tl) and _text_mentions_social_platform(tl):
        if _text_mentions_welfog_brand(tl) or not subj_bhi:
            return True
        return False
    return False


def customer_turn_text(original_msg: str, msg_en: str = "") -> str:
    """Current user message only — never merge prior turns for intent/routing."""
    o = (original_msg or "").strip()
    e = (msg_en or "").strip()
    if not e or e == o:
        return o
    if o and (o in e or e in o):
        return o if len(o) >= len(e) else e
    return f"{o} {e}".strip()


def _turn_is_catalog_product_request(t: str) -> bool:
    """Lightweight product-turn check for social routing (no circular imports)."""
    if _text_is_order_delivery_issue(t):
        return False
    if _text_is_phone_product_accessory_context(t):
        return True
    tl = f" {_normalize_welfog_typos(t)} "
    if re.search(
        r"\b(shirt|pant|jeans|dress|shoes|kurta|cover|case|mobile|product|products)\b",
        tl,
    ) and re.search(r"\b(dikha\w*|lena|leni|kharid|buy|show|chahiye)\b", tl):
        return True
    if re.search(r"\b(dono|both)\b", tl) and re.search(
        r"\b(shirt|cover|case|product|mobile)\b", tl
    ):
        return True
    return False


def _text_is_phone_product_accessory_context(t: str) -> bool:
    """e.g. iPhone cover / Samsung case — shopping, not another brand's social media."""
    tl = f" {_normalize_welfog_typos(t)} "
    phone = re.search(
        r"\b(iphone|ipad|samsung|oneplus|oppo|vivo|realme|redmi|mi|apple|mobile|phone)\b",
        tl,
    )
    accessory = re.search(
        r"\b(cover|case|charger|cable|screen guard|tempered|mobile cover|phone cover|back cover)\b",
        tl,
    )
    apparel = re.search(r"\b(shirt|pant|jeans|dress|shoes|kurta|top|hoodie)\b", tl)
    shop = re.search(
        r"\b(buy|shop|lena|leni|dikha|dikhao|show|product|kharid|chahiye|order karna)\b",
        tl,
    )
    if phone and (accessory or apparel or shop):
        return True
    if accessory and re.search(r"\b(mere|mera|meri|ke liye|for my)\b", tl):
        return True
    return False


def _conversation_recent_welfog_social(conversation_context: str) -> bool:
    if not (conversation_context or "").strip():
        return False
    cl = f" {_normalize_welfog_typos(conversation_context)} "
    raw_low = (conversation_context or "").lower()
    if not _text_mentions_social_platform(cl):
        return False
    if _text_mentions_welfog_brand(cl):
        return True
    if any(
        x in raw_low
        for x in (
            "official social",
            "hamari company knowledge",
            "welfog_online",
            "x.com/welfog",
            "official welfog social",
        )
    ):
        return True
    return False


_WELFOG_SOCIAL_POSSESSIVE = frozenset(
    {
        "welfog", "wlefog", "welefog", "welfogcom", "welfogapp",
        "apni", "apna", "apne", "aapki", "aapka", "aapke", "aapki",
        "hamari", "hamara", "hamare", "meri", "mere", "mera",
        "tumhari", "tumhara", "tumhare", "teri", "tera", "tere",
        "your", "our", "official", "company", "brand",
        "customer", "care", "support", "mujhe", "mereko",
    }
)

_SCHOOL_ORG_MARKERS = (
    "school", "skool", "college", "university", "univ", "institute", "institu",
    "academy", "coaching", "vidyalaya", "vidyalay", "mahavidyalaya", "campus",
)

_SOCIAL_PLATFORM_SUBJECT_WORDS = frozenset(
    {
        "instagram", "insta", "instragram", "facebook", "fb", "youtube", "youtbe",
        "linkedin", "linkdin", "twitter", "twiter", "twittr", "telegram", "whatsapp",
        "social", "media", "link", "links", "page", "profile", "handle", "account", "id",
        "aur", "and", "bhi", "saare", "sab", "all",
    }
)


def _social_request_subject_is_external(who: str) -> bool:
    """True when 'who' is not Welfog / self-reference in support chat."""
    phrase = re.sub(r"\s+", " ", (who or "").strip().lower())
    if not phrase or len(phrase) < 2:
        return False
    if _text_mentions_welfog_brand(f" {phrase} "):
        return False
    tokens = [t for t in re.findall(r"[a-z0-9]+", phrase) if t]
    if not tokens:
        return False
    if all(t in _WELFOG_SOCIAL_POSSESSIVE for t in tokens):
        return False
    if any(m in phrase for m in _SCHOOL_ORG_MARKERS):
        return True
    if len(tokens) >= 2:
        if all(
            t in _SOCIAL_PLATFORM_SUBJECT_WORDS or t in ("aur", "and", "or", "ya", "ye")
            for t in tokens
        ):
            return False
        return True
    if tokens[0] in _WELFOG_SOCIAL_POSSESSIVE:
        return False
    if all(t in _SOCIAL_PLATFORM_SUBJECT_WORDS for t in tokens):
        return False
    if any(t in _SOCIAL_PLATFORM_SUBJECT_WORDS for t in tokens):
        return False
    from services.support_scope import _KNOWN_EXTERNAL_BRANDS

    if tokens[0] in _KNOWN_EXTERNAL_BRANDS:
        return True
    if len(tokens[0]) >= 3 and tokens[0] not in _SOCIAL_PLATFORM_SUBJECT_WORDS:
        return True
    return False


def _external_entity_social_request(text: str) -> bool:
    """Another person / company / school social handle — not Welfog official links."""
    tl = f" {_normalize_welfog_typos(text or '')} "
    if not _text_mentions_social_platform(tl):
        return False
    patterns = (
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+k[iae]\s+bhi\s+(?:de\s+de\s+)?(?:dede|do|dena|de\s+)?(?:link\s+)?(?:insta(?:gram)?|instagram|facebook|fb|youtube|yout+ube|linkedin|linkdin|twitter|twiter|twittr|telegram|social)\b",
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+k[iae]\s+bhi\b",
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+k[iae]\s+(?:de\s+de\s+)?(?:link\s+)?(?:insta(?:gram)?|instagram|facebook|fb|youtube|yout+ube|linkedin|linkdin|twitter|twiter|twittr|telegram|social)\b",
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+k[iae]\s+(?:insta|instagram|facebook|fb|youtube|linkedin|linkdin|twitter|twiter|social)\b",
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+(?:ka|ki|ke)\s+(?:insta|instagram|facebook|fb|youtube|linkedin|linkdin|twitter|twiter|social)\s+(?:link|id|handle|page|profile|account)\b",
        r"\b([a-z][a-z0-9\s]{0,48}?)\s+(?:ka|ki|ke)\s+bhi\s+(?:de\s+de\s+)?(?:dede|do|dena|de\s+)?(?:insta|instagram|facebook|fb|youtube|linkedin|linkdin|twitter|twiter|social)\b",
    )
    for pat in patterns:
        m = re.search(pat, tl)
        if m and _social_request_subject_is_external(m.group(1)):
            return True
    if any(m in tl for m in _SCHOOL_ORG_MARKERS) and _text_mentions_social_platform(tl):
        if not _text_mentions_welfog_brand(tl) and not all(
            t in _WELFOG_SOCIAL_POSSESSIVE
            for t in re.findall(r"[a-z]+", tl)
            if t in ("ki", "ka", "ke", "de", "do", "dena", "link", "id")
        ):
            return True
    return False


def message_asks_other_company_social_media(
    text: str, conversation_context: str = ""
) -> bool:
    """Another brand/person/school social handle — not Welfog official links."""
    turn = (text or "").strip()
    if not turn:
        return False
    if _turn_is_catalog_product_request(turn):
        return False
    if not _text_mentions_social_platform(turn):
        return False
    if _external_entity_social_request(turn):
        return True
    tl = f" {_normalize_welfog_typos(turn)} "

    m_bhi = re.search(r"\b([a-z][a-z0-9]{2,20})\s+k[ai]\s+bhi\b", tl)
    if m_bhi and _social_request_subject_is_external(m_bhi.group(1)):
        return True

    from services.support_scope import _KNOWN_EXTERNAL_BRANDS

    social_network_names = frozenset(
        {"instagram", "facebook", "youtube", "whatsapp", "netflix", "spotify", "twitter"}
    )
    for brand in _KNOWN_EXTERNAL_BRANDS:
        if brand in social_network_names:
            continue
        if brand in (
            "google", "apple", "samsung", "mi", "iphone", "ipad", "oneplus", "oppo", "vivo",
            "realme", "nike", "adidas",
        ):
            if _text_mentions_welfog_brand(tl) and re.search(rf"\b{re.escape(brand)}\b", tl):
                continue
            if _text_is_phone_product_accessory_context(turn):
                continue
        if re.search(rf"\b{re.escape(brand)}\b", tl):
            if re.search(
                rf"\b{re.escape(brand)}\b.{{0,40}}(?:insta|instagram|facebook|youtube|twitter|linkedin|social)",
                tl,
            ) or re.search(
                rf"(?:insta|instagram|facebook|youtube|twitter|linkedin|social).{{0,40}}\b{re.escape(brand)}\b",
                tl,
            ):
                return True

    m = re.search(
        r"\b([a-z][a-z0-9]{2,14})\s+k[ai]\s+(?:de\s+de\s+)?(?:link\s+)?(?:insta|instagram|facebook|fb|youtube|linkedin|twitter)\b",
        tl,
    )
    if not m:
        m = re.search(
            r"\b([a-z][a-z0-9]{2,14})\s+k[ai]\s+(?:insta|instagram|facebook|fb|youtube|linkedin|twitter)\b",
            tl,
        )
    if m:
        who = m.group(1).lower()
        if who not in (
            "welfog", "wlefog", "welefog", "meri", "mere", "mujhe", "hamari", "apni", "aapki",
            "teri", "tera", "tumhari", "tumhara", "tumhare",
            "unke", "unka", "unhi", "customer", "care",
        ):
            return True
    return False


def _text_mentions_kb_social_platform(text: str) -> bool:
    """True when text names a platform listed in admin knowledge (Official Links lines)."""
    try:
        from services.kb_service import get_welfog_social_links_from_kb

        tl = f" {(text or '').lower()} "
        for slug, label, _ in get_welfog_social_links_from_kb():
            if slug and slug in tl:
                return True
            label_low = (label or "").lower()
            if label_low and label_low in tl:
                return True
            for word in re.findall(r"[a-z0-9]{4,}", label_low):
                if word not in ("welfog", "official", "link", "links") and word in tl:
                    return True
    except Exception:
        pass
    return False


def message_asks_welfog_social_media(
    text: str, conversation_context: str = ""
) -> bool:
    """Official Welfog social links / handles — from company knowledge."""
    turn = (text or "").strip()
    if not turn:
        return False
    if _turn_is_catalog_product_request(turn):
        return False
    if message_asks_other_company_social_media(turn, conversation_context=conversation_context):
        return False
    if _is_welfog_social_followup(turn, conversation_context):
        return True
    tl = f" {_normalize_welfog_typos(turn)} "
    wants_social = (
        _text_mentions_social_platform(tl)
        or _text_mentions_kb_social_platform(turn)
        or any(
            x in tl for x in ("social media", "social link", "official link", "official page")
        )
    )
    if not wants_social:
        return False

    # Follow-up: "instagram ki bhi dede", "twitter bhi de" (Welfog thread only — not "agoda ki bhi linkdin")
    if re.search(
        r"\b(?:instagram|insta|youtube|yout+ube|twiter|twittr|twitter|linkedin|linkdin|facebook|fb)\w*\s+k[e]?\s+bhi\b",
        tl,
    ) or re.search(r"\b(bhi|aur)\s+(dede|do|dena|bhej|share|send|bata)\b", tl):
        subj = re.search(r"\b([a-z][a-z0-9]{2,20})\s+k[ai]\s+bhi\b", tl)
        if subj and _social_request_subject_is_external(subj.group(1)):
            return False
        if _text_mentions_social_platform(tl) and (
            _text_mentions_welfog_brand(tl)
            or (
                _conversation_recent_welfog_social(conversation_context)
                and not _external_entity_social_request(turn)
            )
        ):
            return True

    if _text_mentions_welfog_brand(tl):
        return True
    if re.search(
        r"\bwelfog\s+k[iae]\s+(?:insta(?:gram)?|instagram|facebook|fb|youtube|linkedin|linkdin|twitter|social)\b",
        tl,
    ):
        return True
    if any(
        x in tl
        for x in (
            "official", "apna", "apni", "aapka", "aapki", "hamara", "hamari",
            "teri", "tera", "tumhari", "tumhara", "tumhare",
            "company", "brand page", "welfog ka", "welfog ki", "welfog ke",
            "customer care", "customer support", "connect hona", "connect karna",
        )
    ):
        return True
    tokens = [t for t in re.findall(r"[a-z0-9]+", tl) if len(t) >= 2]
    # Short ask in Welfog chat: "instagram dena" — not "agoda ki linkdin"
    if len(tokens) <= 10 and (
        _text_mentions_social_platform(tl) or _text_mentions_kb_social_platform(turn)
    ):
        if _external_entity_social_request(turn):
            return False
        return True
    return False


def message_is_welfog_about_request(text: str) -> bool:
    """
    User wants to know what Welfog is / company info — NOT a catalog SKU search.
    Handles typos (wlefog) and Hinglish (ke baare me, batao kuchh welfog).
    """
    if message_asks_welfog_categories_list(text):
        return False
    try:
        from services.conversation_followup import is_deals_request_message

        if is_deals_request_message(text, ""):
            return False
    except ImportError:
        pass
    if message_asks_welfog_social_media(text):
        return False
    if message_is_seller_on_welfog_request(text):
        return False
    if message_asks_my_welfog_purchases(text):
        return False
    if _is_welfog_about_fast_path(text):
        return True
    raw = text or ""
    tl = f" {_normalize_welfog_typos(text)} "
    if _text_mentions_welfog_brand(tl) and any(
        x in tl
        for x in (
            " kya cheez",
            " kya cheeze",
            " cheez h",
            " cheeze h",
            " krta ",
            " krti ",
            " karta ",
            " what is ",
            " kya hai ",
            " kya h ",
            " story",
            " kahani",
        )
    ):
        return True
    if _text_is_pincode_serviceability_question_light(text or ""):
        return False
    browse_markers = (
        "dikha", "dikhao", "dikhado", "milega", "milta", "chahiye", " buy ", " show ",
        "looking for", "under rs", "under ₹", "shopping krunga", "shopping karunga",
    )
    if _turn_is_catalog_product_request(text) or (
        _message_has_catalog_product_signal(text)
        and any(m in tl for m in browse_markers)
    ):
        return False
    if not _text_mentions_welfog_brand(tl):
        return False
    if re.search(
        r"\bwelfog\b.*\b(?:karta|karti|karte|krti|krta)\b|\b(?:kya|ky)\s*(?:karta|karti|karte)\b.*\bwelfog\b",
        tl,
    ):
        return True
    if re.search(r"\bwelfog\b.*\b(?:krta|krti|krta)\b", tl):
        return True
    if any(
        x in tl
        for x in (
            " kya cheez",
            " kya cheeze",
            " ky cheez",
            " cheez h",
            " cheeze h",
            " cheez hai",
            " cheeze hai",
            " kya h ",
            " kya hai ",
        )
    ):
        if _text_mentions_welfog_brand(tl):
            return True
    # Company story / kahani — not order history or product search
    if any(
        x in tl
        for x in (
            " story",
            " kahani",
            " kahaani",
            " our story",
            " ki story",
            " ki kahani",
            " welfog story",
        )
    ):
        return True
    # Punjabi / other scripts (Gurmukhi "about", "tell")
    if any(
        phrase in raw
        for phrase in (
            "\u0a2c\u0a3e\u0a30\u0a47",  # ਬਾਰੇ (baare)
            "\u0a26\u0a71\u0a38",  # ਦੱਸ (dass)
            "\u0a15\u0a40 \u0a39\u0a48",  # ਕੀ ਹੈ
            "\u0a15\u0a3e\u0a02 \u0a15\u0a30\u0a26\u0a3e",  # ਕੰਮ ਕਰਦਾ
        )
    ) and (_text_mentions_welfog_brand(tl) or "welfog" in tl):
        return True
    has_about = any(m in tl for m in _WELFOG_ABOUT_MARKERS)
    has_what = any(
        x in tl
        for x in (
            " kya hai ",
            " kya h ",
            " what is ",
            " who is ",
            " kon hai ",
            " kaun hai ",
            " kya kya ",
        )
    )
    if "kya kya" in tl and any(m in tl for m in ("mangaya", "mangaye", "mangwaya", "mene", "maine", "kiye", "kiya")):
        return False
    if not has_about and not has_what:
        if any(x in tl for x in (" batao ", " bata ", " btao ", " tell me ", " explain ", " samjha ")):
            tokens = [
                t
                for t in re.findall(r"[a-z0-9]+", tl)
                if t not in _PRODUCT_QUERY_STOPWORDS and t != "welfog"
            ]
            if not any(t in _WELFOG_ABOUT_PRODUCT_NOUNS for t in tokens):
                has_about = True
    if not has_about and not has_what:
        return False
    browse_markers = (
        "dikha", "dikhao", "dikhado", "milega", "milta", "chahiye", " buy ", " show ",
        "looking for", "under rs", "under ₹",
    )
    if any(m in tl for m in browse_markers) and any(n in tl for n in _WELFOG_ABOUT_PRODUCT_NOUNS):
        return False
    return True


def _text_has_platform_overview_intent(t: str) -> bool:
    """What is on Welfog / how many product types — not a SKU search."""
    if message_is_welfog_about_request(t):
        return True
    tl = _normalize_welfog_typos(t)
    if any(x in tl for x in ["kitne product", "kitne products", "how many product", "how many products"]):
        return True
    if "kya kya" in tl and any(x in tl for x in ["mil", "milega", "milta", "milegi", "milti", "milt", "available"]):
        return True
    if any(x in tl for x in ["what do you sell", "what can i buy", "what all can i", "what is sold"]):
        return True
    if "welfog" in tl and any(
        x in tl
        for x in ["kya kya", "kya hai", "kya milta", "about welfog", "pe kya", "par kya", "me kya", "mein kya"]
    ):
        if not _text_has_product_shopping_intent_core(tl) and "order track" not in tl:
            return True
    return False


def _merge_extracted_pincode(original_msg: str, msg_en: str, ai_data: dict) -> None:
    _merge_embedded_identifiers_from_message(original_msg, msg_en, ai_data)


def _merge_embedded_identifiers_from_message(
    original_msg: str,
    msg_en: str,
    ai_data: dict,
    conversation_context: str = "",
) -> None:
    """Fill extracted_pincode / order context from the same user message — no re-ask."""
    if not ai_data:
        return
    ids = extract_embedded_query_identifiers(
        original_msg, msg_en, conversation_context, ai_route=ai_data
    )
    pin = (ids.get("pincode") or "").strip()
    if pin and re.fullmatch(r"[1-9]\d{5}", pin):
        ai_data["extracted_pincode"] = pin
        if ids.get("numeric_context") == "pincode":
            ai_data["numeric_context"] = "pincode"
            if (ai_data.get("intent") or "") not in ("product", "wishlist", "order_history"):
                ai_data["intent"] = "pincode_check"
                ai_data["needs_order_id"] = False
                ai_data["search_query"] = ""
    oid = (ids.get("order_id") or "").strip()
    if oid and ids.get("numeric_context") == "order_id":
        ai_data["numeric_context"] = "order_id"
        if (ai_data.get("intent") or "") in ("order", "refund", "payment", "general"):
            ai_data["needs_order_id"] = True
    pid = (ids.get("product_id") or "").strip()
    if pid and ids.get("numeric_context") == "product_id":
        ai_data["intent"] = "product"
        ai_data["search_query"] = f"pro_id {pid}"
        ai_data["needs_order_id"] = False


def _looks_like_conversational_followup(original_msg: str, msg_en: str) -> bool:
    """
    Short follow-ups (uska/uske, 'kya karta hai', 'batao') refer to the last topic — not a product SKU search.
    """
    combined = f" {original_msg} {msg_en} ".lower()
    pronoun = (
        " uska ", " uske ", " uski ", " unka ", " unke ", " unki ", " iska ", " iske ", " iski ",
        " inka ", " inke ", " inki ", " yeh ", " ye ", " woh ", " wo ", " is ", " un ",
    )
    explain = (
        "kya karta", "kya karti", "kya karte", " krta ", " krti ", " krte ", " karte ",
        "kaam kya", "kam kya", "what does", "what do ", "how does", "explain", "details",
        "aur bata", "aur btao", "aur batao", " bta ", " btao ", " batao ", "tell me more",
        "iske bare", "iske baare", "uske bare", "uske baare", "iske bar", "uske bar",
    )
    if any(p in combined for p in pronoun) and any(e in combined for e in explain):
        return True
    if any(p in combined for p in pronoun) and any(x in combined for x in (" bta", "btao", "batao", "bata", "detail")):
        return True
    return False


def _text_has_product_shopping_intent(t: str) -> bool:
    """True if message reads like buy/show/availability — regex/core only (no embedding on hot path)."""
    if _is_assistant_identity_question(t):
        return False
    if _is_light_smalltalk_fast(t) or _is_short_pure_greeting((t or "").strip()):
        return False
    if _turn_is_catalog_product_request(t):
        return True
    if _text_has_product_shopping_intent_core(t):
        return True
    if (
        message_is_welfog_about_request(t)
        or message_is_knowledge_information_request(t)
        or message_is_casual_offtopic_not_shopping(t)
    ):
        return False
    return False


def _text_has_product_shopping_intent_core(t: str) -> bool:
    try:
        from services.conversation_followup import is_non_product_search_phrase

        if is_non_product_search_phrase(t):
            return False
    except ImportError:
        pass
    if (
        message_needs_support_not_product(t)
        or message_needs_policy_answer(t)
        or message_is_knowledge_information_request(t)
    ):
        return False
    if _text_asks_order_history(t) or message_is_past_purchase_list_request(t):
        return False
    if _text_asks_wishlist(t) or message_is_wishlist_like_request(t):
        return False
    if _text_asks_customer_care_contact(t):
        return False
    t = f" {_normalize_welfog_typos(t)} "
    non_product_hints = (
        " about ", " baare ", " barre ", " bare ", " baar me", "department", "departments",
        "team ", "teams ", "staff ", "company ", "policy", "policies", "information", "details",
        " customer care ", " customer support ", " cust care ", " support email ", " helpline ",
    )
    if any(h in t for h in non_product_hints):
        return False
    if any(m in t for m in _SHOPPING_ACTION_MARKERS):
        if _text_has_past_order_complaint_context(t) or _text_has_refund_or_return_intent(t):
            return False
        return _message_has_catalog_product_signal(t) or _message_has_generic_shopping_item_signal(t)

    if _message_has_catalog_product_signal(t):
        if _text_has_past_order_complaint_context(t) or _text_has_refund_or_return_intent(t):
            return False
        return True
    return False


def user_continues_product_browse_from_conversation(
    original_msg: str, conversation_context: str = ""
) -> bool:
    """
    Short follow-up after assistant discussed products (e.g. 'ha to dikhana')
    when Groq routing is unavailable.
    """
    if not (conversation_context or "").strip():
        return False
    conv_low = conversation_context.lower()
    if not any(
        x in conv_low
        for x in (
            "cover", "iphone", "samsung", "mobile case", "phone case", "case ",
            "options milenge", "batata hoon", "product", "infinix", "oppo", "vivo",
            "redmi", "oneplus", "realme", "nokia", "xiaomi",
        )
    ):
        return False
    low_msg = f" {(original_msg or '').lower()} "
    follow_markers = (
        "dikhana", "dikhao", "dikha", "dikha de", "dikhao de", "btao", "batao", "bta",
        "show", "dekh", "ha to", "haan to", "ok to", "theek to", "wahi", "same",
        "ab ", "aur dikha", "phir se",
    )
    if any(m in low_msg for m in follow_markers):
        return True
    if _is_conversation_acknowledgment(original_msg) and any(
        x in conv_low for x in ("cover", "iphone", "samsung", "case", "options")
    ):
        return True
    return False


def _looks_like_browse_all_categories_message(t: str) -> bool:
    t = t.lower()
    return any(
        x in t
        for x in (
            "all categor", "saari categor", "sari categor", "category list", "categories list",
            "catagory list", "catagories list", "list of categor", "konsi categor",
            "kaun si categor", "browse categor", "har categor",
        )
    )


def _text_mentions_category_menu(t: str) -> bool:
    """Category menu mention — typo-tolerant (catagories, catagory, categories)."""
    tl = f" {(t or '').lower()} "
    if "categor" in tl:
        return True
    return bool(re.search(r"\bcat[ae]?g[ao]ri(?:es|e)?s?\b", tl))


def _looks_like_category_list_request(t: str) -> bool:
    """User wants the full Welfog category list/menu — not products inside one category."""
    if not (t or "").strip():
        return False
    if _looks_like_browse_all_categories_message(t):
        return True
    tl = f" {(t or '').lower()} "
    if any(
        x in tl
        for x in (
            "kitni", "kitne", "how many", "total ", "count", "list", "dikhao", "dikha",
            "dikhado", "btao", "batao", "bata de", "bta de", "bta do", "de de", "dede",
            "dena", "show", "dekho", "saari", "sab ", "all ", "poori", "puri",
        )
    ):
        return True
    if re.search(r"\b(?:bta|btao|bata|de|dikha|show|list|batado|batade)\b", tl):
        return True
    if re.search(r"\bwelfog\b", tl) and _text_mentions_category_menu(t):
        return True
    return False


def message_asks_welfog_categories_list(t: str) -> bool:
    """User wants Welfog category names/count/list — not products from one category."""
    if not (t or "").strip():
        return False
    if not _text_mentions_category_menu(t):
        return False
    if _looks_like_category_list_request(t):
        return True
    if _text_requests_category_product_browse(t):
        return False
    from services.welfog_api import _user_explicitly_browses_category

    if _user_explicitly_browses_category(t) and not _looks_like_browse_all_categories_message(t):
        return False
    tl = f" {(t or '').lower()} "
    if re.search(r"\bwelfog\b", tl):
        return True
    return False


def _text_has_refund_or_return_intent(t: str) -> bool:
    """Detect refund, return, cancel, exchange intent. Works across English, Hindi, Hinglish."""
    tl = f" {t.lower()} "
    if "refund" in tl:
        return True
    if "return" in tl and any(
        x in tl
        for x in (
            "order",
            "product",
            "item",
            "exchange",
            "policy",
            "refund",
            "cancel",
            "shipment",
            "parcel",
            "purchase",
            "welfog",
            "on welfog",
        )
    ):
        return True
    if re.search(r"\bhow\s+(?:to|do|can)\s+(?:i\s+)?return\b", tl):
        return True
    
    cancel_phrases = (
        "cancel order", "order cancel", "cancel my order", "cancel purchase", "cancel item",
        "cancel kar", "cancel kr", "cancel karna", "cancel ho", "cancel karu",
        # Additional Hindi/Hinglish patterns for cancel/refund/return
        "refund kar", "refund kr", "refund karna", "refund krna", "refund chahiye", "refund chaiye",
        "paise wapsi", "paise vapsi", "paise return", "money back", "return kar", "return kr",
        "return karna", "return chahiye", "order cancel krna", "order cancel karna",
        "order refund", "order return", "exchange karna", "exchange chahiye", "nahin chahiye",
        "nahi chahiye", "order nahin chahiye", "galat mila", "ghalat mila", "cancel order krna",
        "order cancel kaise", "refund kaise", "return kaise", "exchange kaise", "cancel kaise",
        "refund process", "cancel process", "refund kab", "refund kab milega", "order cancel kab",
        "paise kab milenge", "refund kab milega", "order cancel kar do", "order cancel kr do",
        "refund kar do", "refund kr do", "money back le", "order return karna", "order nahin chahiye",
    )
    if any(phrase in tl for phrase in cancel_phrases):
        return True
    if re.search(r"\breturn\s+(?:kar|kr|ho|ho)\s+sk", tl):
        return True
    if re.search(r"\brefund\s+kitne\s+din", tl) or re.search(r"\bkitne\s+din\s+me\s+refund", tl):
        return True
    if "return" in tl and any(x in tl for x in ("kya", "ky ", "ho sk", "kar sk", "kr sk", "sakta", "skta", "milega")):
        return True
    return False


def _looks_like_policy_faq_message(t: str) -> bool:
    tl = t.lower()
    return any(h in tl for h in _POLICY_QUESTION_HINTS)


def _looks_like_factual_identity_query(text: str) -> bool:
    """
    Only Welfog-related identity/org questions get strict KB grounding.
    Random person names ("abhishek kon h") must NOT hit the admin-panel fallback.
    """
    if message_is_welfog_about_request(text):
        return True
    t = f" {_normalize_welfog_typos(text)} "
    if "welfog" not in t:
        return False
    return any(
        h in t
        for h in (
            "who is",
            "who are",
            "who was",
            "kon h",
            "kon hai",
            "kaun h",
            "kaun hai",
            "kisne banaya",
            "founder",
            "co-founder",
            "cofounder",
            "owner",
            "ceo",
            "cto",
            "cfo",
            "funding partner",
            "partner of welfog",
            "about welfog",
            "welfog about",
            "team",
            "staff",
            "department",
        )
    )


def extract_product_search_query(
    original_msg: str,
    msg_en: str,
    ai_search_query=None,
    *,
    ai_route: dict | None = None,
) -> str:
    """
    Prefer a sensible model-provided search_query; otherwise strip Roman-Hindi filler
    and keep product nouns/brands (e.g. 'cover h ky' -> 'cover').
    """
    ai_sq = (ai_search_query or "")
    if isinstance(ai_sq, str):
        ai_sq = ai_sq.strip()
    else:
        ai_sq = str(ai_sq or "").strip()
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if isinstance(ai_route, dict) and product_catalog_route_is_locked(ai_route):
            route_sq = (ai_route.get("search_query") or "").strip()
            if route_sq:
                return route_sq
            if ai_sq and len(ai_sq) >= 2:
                return ai_sq
    except ImportError:
        pass
    if _message_looks_like_shopping_query(f"{original_msg} {msg_en}".strip()):
        try:
            from services.product_query_understanding import extract_focused_product_query

            focused = extract_focused_product_query(original_msg, msg_en)
            if focused and len(focused.strip()) >= 2:
                return focused.strip()
        except ImportError:
            pass
        if ai_sq and len(ai_sq) >= 2:
            return ai_sq
        try:
            from services.product_query_understanding import clean_product_part_label, polish_search_terms

            quick = clean_product_part_label(
                polish_search_terms(ai_sq or original_msg, original_msg), original_msg
            )
            if quick and len(quick) >= 2:
                return quick
        except ImportError:
            pass
    try:
        from services.ai_route_semantics import ai_route_allows_catalog_search
        from services.semantic_intent import llm_semantic_route_available

        if llm_semantic_route_available(ai_route) and not ai_route_allows_catalog_search(ai_route):
            return ""
    except ImportError:
        pass
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(f"{original_msg} {msg_en}"):
            return ""
    except ImportError:
        pass
    if (
        message_needs_support_not_product(f"{original_msg} {msg_en}")
        or message_needs_policy_answer(f"{original_msg} {msg_en}")
        or message_is_welfog_about_request(f"{original_msg} {msg_en}")
        or message_is_knowledge_information_request(f"{original_msg} {msg_en}")
        or _text_is_order_delivery_issue(f"{original_msg} {msg_en}")
        or message_is_bot_search_complaint(f"{original_msg} {msg_en}")
        or message_is_bot_capability_question(f"{original_msg} {msg_en}")
    ):
        return ""
    pid = extract_product_id(f"{original_msg} {msg_en}")
    if pid:
        return f"pro_id {pid}"
    try:
        from services.conversation_followup import is_deals_request_message

        if is_deals_request_message(original_msg, msg_en):
            return ""
    except ImportError:
        pass
    try:
        from services.product_query_understanding import (
            clean_product_part_label,
            is_noisy_search_query,
            polish_search_terms,
        )
    except ImportError:
        def clean_product_part_label(t, _m=""):
            return (t or "").strip()

        polish_search_terms = lambda t, _m="": (t or "").strip()
        is_noisy_search_query = lambda t: False

    try:
        from services.product_query_understanding import (
            extract_focused_product_query,
            is_noisy_search_query as _is_noisy_q,
        )

        focused = extract_focused_product_query(original_msg, msg_en)
    except ImportError:
        focused = ""
        _is_noisy_q = lambda _t: False

    combined_probe = f"{original_msg} {msg_en}".lower()
    if focused and (
        len(re.findall(r"[a-z0-9]+", combined_probe)) >= 10
        or _is_noisy_q(combined_probe)
    ):
        return focused

    sq = clean_product_part_label(
        polish_search_terms((ai_search_query or "").strip(), original_msg),
        original_msg,
    )
    if sq and len(sq) >= 2 and sq.lower() not in ("product", "item", "thing", "na", "n/a", "none", "null"):
        if not is_noisy_search_query(sq):
            return sq
        if focused:
            return focused
    if focused:
        return focused
    combined = f"{original_msg} {msg_en}".lower()
    if focused:
        return focused
    tokens = re.findall(r"[a-z0-9]+", combined)
    kept: list[str] = []
    seen: set[str] = set()
    for w in tokens:
        if len(w) < 2 or w in _PRODUCT_QUERY_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            kept.append(w)
    return clean_product_part_label(
        polish_search_terms(" ".join(kept).strip(), original_msg),
        original_msg,
    )


def apply_category_product_route_fixes(original_msg: str, msg_en: str, ai_data: dict, ctx=None) -> None:
    """
    If the user names a category and wants products, never show the full categories list.
    Route to product browse with resolved category id.
    """
    if not ai_data or not ai_data.get("is_welfog_related", True):
        return
    if ai_data.get("_ai_routed"):
        ch = (ai_data.get("data_channel") or "").strip().lower()
        if ch not in ("catalog", "") and (ai_data.get("intent") or "") != "product":
            return
    combined = f"{original_msg} {msg_en}"
    if _looks_like_browse_all_categories_message(combined):
        return
    if not _text_requests_category_product_browse(combined, ctx):
        return
    try:
        from services.welfog_api import resolve_category_product_browse_route

        resolved = resolve_category_product_browse_route(combined, ctx=ctx)
        if not resolved:
            return
        cid, sq = resolved
    except Exception:
        return
    ai_data["intent"] = "product"
    ai_data["search_query"] = sq or ""
    if ctx is not None:
        ctx.setdefault("data", {})
        ctx["data"]["selected_category_id"] = cid
        ctx["awaiting"] = None


def apply_hinglish_product_fixes(original_msg: str, msg_en: str, ai_data: dict) -> None:
    """
    Legacy keyword fixes when Groq did not route (_ai_routed absent).
    When Groq already classified the message, only fill missing PIN/search fields.
    """
    if not ai_data:
        return
    if ai_data.get("_ai_routed"):
        intent = (ai_data.get("intent") or "").strip()
        if intent == "pincode_check":
            ai_data["needs_order_id"] = False
            ai_data["search_query"] = ""
            pin = resolve_pincode_for_check(original_msg) or resolve_pincode_for_check(msg_en)
            if pin:
                ai_data["extracted_pincode"] = pin
        elif intent == "product":
            combined_routed = f"{original_msg} {msg_en}".strip()
            pin_routed = extract_pincode_preferred_from_message(combined_routed)
            if pin_routed and (
                _text_has_pincode_delivery_intent(combined_routed)
                or _text_has_delivery_serviceability_intent(combined_routed)
                or re.search(
                    r"\b(ispe|isme|is\s+par|is\s+per|uspe|us\s+par|yahan|yaha|idhar)\b",
                    combined_routed.lower(),
                )
            ):
                ai_data["intent"] = "pincode_check"
                ai_data["needs_order_id"] = False
                ai_data["search_query"] = ""
                ai_data["extracted_pincode"] = pin_routed
                return
            sq = extract_product_search_query(
                original_msg, msg_en, ai_data.get("search_query") or ""
            )
            if sq and not (ai_data.get("search_query") or "").strip():
                ai_data["search_query"] = sq
        return
    combined_early = f"{original_msg} {msg_en}"
    if _text_has_delivery_serviceability_intent(combined_early):
        ai_data["intent"] = "pincode_check"
        ai_data["is_welfog_related"] = True
        ai_data["needs_order_id"] = False
        ai_data["search_query"] = ""
        pin = resolve_pincode_for_check(original_msg) or resolve_pincode_for_check(msg_en)
        if pin:
            ai_data["extracted_pincode"] = pin
        return
    if _message_is_order_id_followup_submission(combined_early):
        ai_data["intent"] = "order"
        ai_data["is_welfog_related"] = True
        ai_data["needs_order_id"] = True
        ai_data["search_query"] = ""
        return
    if _text_has_pincode_delivery_intent(combined_early):
        ai_data["intent"] = "pincode_check"
        ai_data["is_welfog_related"] = True
        ai_data["search_query"] = ""
        ai_data["needs_order_id"] = False
        pin = resolve_pincode_for_check(original_msg) or resolve_pincode_for_check(msg_en)
        if pin:
            ai_data["extracted_pincode"] = pin
        return
    if (ai_data.get("intent") or "").strip() == "pincode_check":
        ai_data["needs_order_id"] = False
        ai_data["search_query"] = ""
        pin = resolve_pincode_for_check(original_msg) or resolve_pincode_for_check(msg_en)
        if pin:
            ai_data["extracted_pincode"] = pin
        return
    if message_is_knowledge_information_request(combined_early):
        ai_data["intent"] = "general"
        ai_data["search_query"] = ""
        ai_data["is_welfog_related"] = True
        return
    if message_is_casual_offtopic_not_shopping(combined_early):
        ai_data["intent"] = "out_of_domain"
        ai_data["search_query"] = ""
        ai_data["is_welfog_related"] = False
        return
    combined = f"{original_msg} {msg_en}"
    if message_needs_policy_answer(combined):
        ai_data["intent"] = "refund" if _text_has_refund_or_return_intent(combined) else "general"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
        ai_data["is_welfog_related"] = True
        ai_data["needs_order_id"] = False
        return
    if message_needs_support_not_product(combined):
        ai_data["intent"] = "general"
        ai_data["search_query"] = ""
        ai_data["response"] = ""
        ai_data["is_welfog_related"] = True
        return
    try:
        from services.conversation_followup import is_deals_request_message

        if is_deals_request_message(original_msg, msg_en):
            ai_data["intent"] = "deals"
            ai_data["search_query"] = ""
            ai_data["response"] = ""
            return
    except ImportError:
        pass
    intent = ai_data.get("intent")
    if intent == "out_of_domain" or not ai_data.get("is_welfog_related", True):
        return

    if _looks_like_conversational_followup(original_msg, msg_en):
        if ai_data.get("intent") == "product":
            ai_data["intent"] = "general"
            ai_data["search_query"] = ""
    intent = ai_data.get("intent")

    combined = f"{original_msg} {msg_en}"
    comb_low = combined.lower()
    if _text_has_delivery_or_order_area_intent(comb_low):
        if intent == "product":
            ai_data["intent"] = "pincode_check"
            ai_data["search_query"] = ""
        return
    if (
        _text_has_platform_overview_intent(comb_low)
        or message_is_welfog_about_request(comb_low)
        or message_is_knowledge_information_request(comb_low)
    ):
        ai_data["intent"] = "general"
        ai_data["search_query"] = ""
        ai_data["is_welfog_related"] = True
        return
    extracted = extract_product_search_query(original_msg, msg_en, ai_data.get("search_query"))

    if intent == "categories" and extracted and not _looks_like_browse_all_categories_message(combined):
        tl = combined.lower()
        short_concrete = len(re.findall(r"[a-z0-9]+", tl)) <= 5 and "categor" not in tl
        if _text_has_product_shopping_intent(combined) or short_concrete:
            ai_data["intent"] = "product"
            ai_data["search_query"] = extracted
            return

    if (
        intent == "general"
        and extracted
        and _text_has_product_shopping_intent(combined)
        and not _looks_like_conversational_followup(original_msg, msg_en)
    ):
        if not _looks_like_policy_faq_message(combined):
            ai_data["intent"] = "product"
            ai_data["search_query"] = extracted
            return

    if intent == "product":
        cur = (ai_data.get("search_query") or "").strip()
        if not cur or len(cur) < 2 or cur.lower() in ("product", "item", "thing"):
            if extracted:
                ai_data["search_query"] = extracted
    if _text_has_product_shopping_intent(combined) and re.search(
        r"\b\w+\s+brand\s+ka\b", comb_low
    ):
        ai_data["intent"] = "product"
    if message_needs_policy_answer(combined):
        return
    try:
        from services.opensearch_products import build_product_search_spec, has_structured_product_filters

        spec_probe = build_product_search_spec(original_msg, msg_en)
        if has_structured_product_filters(spec_probe):
            ai_data["intent"] = "product"
            ai_data["is_welfog_related"] = True
            if spec_probe.get("sku"):
                ai_data["search_query"] = f"sku {spec_probe['sku']}"
    except Exception:
        pass


def apply_order_tracking_fixes(original_msg: str, msg_en: str, ai_data: dict) -> None:
    """
    Order delay / tracking in Hinglish must not fall through to Groq 'general' answers that
    invent fake helplines (e.g. 1800-123-4567). Use order intent + templates / live API instead.
    """
    if not ai_data:
        return
    if ai_data.get("_ai_routed"):
        return
    if not ai_data.get("is_welfog_related", True) or ai_data.get("intent") == "out_of_domain":
        return
    comb = _normalize_order_chat_text(f"{original_msg} {msg_en}")
    if (ai_data.get("intent") or "").strip() == "pincode_check":
        return
    try:
        from services.support_scope import message_mentions_other_company_support

        if message_mentions_other_company_support(original_msg, msg_en):
            ai_data["intent"] = "out_of_domain"
            ai_data["is_welfog_related"] = False
            ai_data["needs_order_id"] = False
            ai_data["response"] = ""
            return
    except ImportError:
        pass
    if _text_is_product_id_lookup_context(comb) or extract_product_id(comb):
        if not _message_submits_or_corrects_order_id(comb):
            return
    if _user_asks_hypothetical_tracking_capability(comb):
        ai_data["intent"] = "order"
        ai_data["needs_order_id"] = True
        ai_data["response"] = ""
        return
    if _message_submits_or_corrects_order_id(comb):
        ai_data["intent"] = "order"
        ai_data["needs_order_id"] = True
        ai_data["response"] = ""
        return
    if not _text_is_order_tracking_intent(comb) or _text_has_order_placement_intent(comb):
        return
    if extract_order_id(comb):
        ai_data["intent"] = "order"
        ai_data["needs_order_id"] = True
        return
    if _text_needs_order_id_for_tracking(comb):
        ai_data["intent"] = "order"
        ai_data["needs_order_id"] = True
        ai_data["response"] = ""
        return
    ai_data["intent"] = "order"
    ai_data["needs_order_id"] = False
    ai_data["response"] = ""


