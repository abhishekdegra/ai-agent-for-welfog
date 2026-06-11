"""
Decide whether a policy/privacy question is about Welfog or another company.
Uses conversation context + LLM for every policy-topic message (no static company list).
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

from utils.reasoning_log import log_reasoning

PolicyScope = Literal["welfog", "external", "unclear"]

_POLICY_WORDS = (
    "privacy",
    "policy",
    "policies",
    "terms",
    "conditions",
    "privary",
    "polcy",
    "data protection",
    "disclaimer",
    "legal",
    "cookie",
)

# Words before "policy/privacy" that are Hindi/English fillers — NOT company names.
_POLICY_TOPIC_FILLERS = frozenset(
    {
        "privacy",
        "policy",
        "policies",
        "terms",
        "conditions",
        "data",
        "personal",
        "kesi",
        "kaisi",
        "kaise",
        "kya",
        "ky",
        "kab",
        "kahan",
        "kitni",
        "kitna",
        "kaisi",
        "h",
        "hai",
        "hain",
        "ho",
        "btana",
        "batao",
        "btao",
        "bata",
        "bta",
        "tell",
        "show",
        "about",
        "regarding",
        "short",
        "full",
        "complete",
        "official",
        "welfog",
        "wlefog",
        "welefog",
        "company",
        "platform",
        "website",
        "app",
        "site",
        "information",
        "info",
        "faq",
        "faqs",
        "order",
        "orders",
        "payment",
        "shipping",
        "delivery",
        "return",
        "refund",
        "customer",
        "product",
        "seller",
        "isi",
        "yahi",
        "yehi",
        "wahi",
        "meri",
        "mere",
        "mera",
        "hamari",
        "hamara",
        "aapki",
        "aapka",
        "the",
        "this",
        "that",
    }
)

_TWO_LETTER_NOT_BRAND = frozenset(
    {"ki", "ka", "ke", "ko", "se", "me", "hi", "hu", "ha", "ho", "na", "to", "re", "le", "de", "or", "in", "on", "at", "is", "it", "of", "if", "so", "no", "ok"}
)


def _has_policy_topic(text: str) -> bool:
    from utils.helpers import _normalize_policy_typos

    tl = f" {_normalize_policy_typos(text)} "
    return any(p in tl for p in _POLICY_WORDS)


def _user_explicitly_wants_welfog_policy(msg: str, conversation_context: str = "") -> bool:
    """Follow-up or explicit Welfog-only policy ask (incl. 'isi company', 'welfog ki hi')."""
    from utils.helpers import _normalize_policy_typos, _text_mentions_welfog_brand

    msg_low = f" {msg.lower()} "
    clarify_only = any(
        x in msg_low
        for x in (
            "welfog ki hi",
            "sirf welfog",
            "only welfog",
            "isi company",
            "yahi company",
            "wahi puchh",
            "wahi puch",
            "ha wahi",
        )
    )
    if not _has_policy_topic(msg) and not _text_mentions_welfog_brand(msg):
        if not clarify_only:
            return False

    tl = f" {_normalize_policy_typos(msg)} "
    if _text_mentions_welfog_brand(msg):
        return True

    welfog_only_phrases = (
        "welfog ki hi",
        "sirf welfog",
        "only welfog",
        "isi company ki",
        "isi company ka",
        "isi app ki",
        "isi website ki",
        "yahi company",
        "yahi app ki",
        "same company",
        "aapki company",
        "tumhari company",
        "meri company ki",
        "hamari company",
        "privacy policy dikhao",
        "policy dikhao",
        "privacy policy batao",
        "privacy policy btao",
        "privacy policy btana",
        "welfog privacy",
        "welfog policy",
        "hamari privacy",
        "hamari policy",
    )
    if any(p in tl for p in welfog_only_phrases):
        return True

    if conversation_context and (_has_policy_topic(msg) or clarify_only):
        tail = conversation_context[-3000:].lower()
        user_low = msg.lower()
        bot_offered_welfog = any(
            x in tail
            for x in (
                "welfog's own privacy",
                "welfog ki privacy",
                "privacy policy dikhao",
                "welfog privacy policy",
                "sirf welfog",
                "only help with welfog",
                "welfog se related",
            )
        )
        user_clarifies = any(
            x in user_low
            for x in (
                "wahi puchh",
                "wahi puch",
                "ha wahi",
                "isi company",
                "yahi company",
                "welfog ki hi",
                "bta de",
                "bata de",
                "btana",
                "dikha",
                "dikhao",
                "batao",
                "btao",
            )
        )
        if bot_offered_welfog and user_clarifies:
            return True

    return False


def _looks_like_third_party_brand_token(tok: str) -> bool:
    from utils.helpers import _token_is_welfog_variant

    t = (tok or "").lower().strip()
    if not t or t in _POLICY_TOPIC_FILLERS:
        return False
    if _token_is_welfog_variant(t):
        return False
    if len(t) == 2:
        if t in _TWO_LETTER_NOT_BRAND:
            return False
        return True
    return len(t) >= 3


def _heuristic_external_brand_named(tl: str) -> bool:
    """Named non-Welfog entity in a policy question (no fixed company list)."""
    if re.search(r"\bprivacy\s+policy\b", tl) or re.search(r"\bpolicy\s+privacy\b", tl):
        pass
    else:
        for m in re.finditer(
            r"\b([a-zA-Z][a-zA-Z0-9]{1,24})\s+(?:privacy|policy|policies|terms|privary|polcy)\b",
            tl,
            re.I,
        ):
            if _looks_like_third_party_brand_token(m.group(1)):
                return True

    for m in re.finditer(
        r"\b([a-zA-Z][a-zA-Z0-9]{1,24})\s+(?:ki|ka|ke)\b",
        tl,
        re.I,
    ):
        if _looks_like_third_party_brand_token(m.group(1)):
            return True

    for m in re.finditer(
        r"\b(?:privacy|policy|policies|terms)\s+(?:of|for|about)\s+([a-zA-Z][a-zA-Z0-9]{1,24})\b",
        tl,
        re.I,
    ):
        if _looks_like_third_party_brand_token(m.group(1)):
            return True

    return False


def _llm_policy_scope(user_msg: str, conversation_context: str = "") -> Optional[PolicyScope]:
    """Groq: Welfog policy vs another company's policy."""
    try:
        import requests

        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None

        conv_block = ""
        if (conversation_context or "").strip():
            conv_block = (
                "\nRECENT CHAT (assistant = bot, user = customer):\n"
                f"{conversation_context.strip()[-3500:]}\n"
            )

        prompt = f"""You classify messages in a Welfog e-commerce support chatbot.

{conv_block}
LATEST USER MESSAGE:
\"\"\"{user_msg.strip()}\"\"\"

Question: Is the user asking about Welfog's OWN policies (privacy, terms, shipping, refund, etc.) OR about another company/brand's policies?

Rules:
- This chat is ONLY for Welfog. Generic "privacy policy batao" / "kesi hai policy" / "isi company ki policy" (after bot talked about Welfog) → Welfog.
- "titan ki privacy", "amazon ki policy", "HP ki privacy", "salesforce policy" → other company.
- Typos: welefog/wlefog = Welfog.
- If user clarifies "welfog ki hi", "isi company ki", "ha wahi puchh rha" after a Welfog policy hint → Welfog.

Return ONLY JSON: {{"about_welfog": true/false, "reason": "few words"}}"""

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=14,
        )
        r.raise_for_status()
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        if data.get("about_welfog") is True:
            log_reasoning(f"Policy scope LLM → Welfog ({data.get('reason', '')})")
            return "welfog"
        if data.get("about_welfog") is False:
            log_reasoning(f"Policy scope LLM → external ({data.get('reason', '')})")
            return "external"
    except Exception as e:
        log_reasoning(f"Policy scope LLM skipped: {e}")
    return None


def resolve_policy_question_scope(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> PolicyScope:
    from utils.helpers import _normalize_policy_typos, _text_mentions_welfog_brand

    combined = f"{original_msg} {msg_en}".strip()
    if not _has_policy_topic(combined):
        return "unclear"

    if _user_explicitly_wants_welfog_policy(original_msg, conversation_context):
        return "welfog"

    tl = f" {_normalize_policy_typos(combined)} "

    if _text_mentions_welfog_brand(combined):
        return "welfog"

    if _heuristic_external_brand_named(tl):
        return "external"

    if re.search(r"\b(privacy|policy|terms|conditions|privary|polcy)\b", tl):
        return "welfog"

    return "unclear"


def policy_question_is_for_welfog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """True → show Welfog KB policy. False → polite decline (other company)."""
    if not _has_policy_topic(f"{original_msg} {msg_en}"):
        return True

    if _user_explicitly_wants_welfog_policy(original_msg, conversation_context):
        return True

    scope = resolve_policy_question_scope(original_msg, msg_en, conversation_context)
    if scope == "welfog":
        return True
    if scope == "external":
        return False

    llm = _llm_policy_scope(
        f"{original_msg} {msg_en}".strip(),
        conversation_context,
    )
    if llm == "welfog":
        return True
    if llm == "external":
        return False

    return True


def policy_question_is_external_company(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    return not policy_question_is_for_welfog(original_msg, msg_en, conversation_context)
