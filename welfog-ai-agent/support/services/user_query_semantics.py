"""
Semantic user-query understanding — any language / phrasing / long prompts.

Decides WHAT the customer wants (company vs bot vs product vs policy) before
keyword lists or wrong fast paths (e.g. Welfog company → assistant intro).
"""
from __future__ import annotations

import re
from typing import Optional

from utils.reasoning_log import log_reasoning


def _comb(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def user_frustration_about_company_not_bot(text: str) -> bool:
    """User says they are NOT asking about the bot — wants Welfog company/platform."""
    tl = f" {(text or '').lower()} "
    if not re.search(
        r"\b(?:teri|tumhari|aapki|your)\s+baat\b|"
        r"\b(?:tujhse|tumse|aap se)\s+(?:nahi|nhi|mat)\b|"
        r"\bnot\s+(?:you|about\s+you|talking\s+to\s+you)\b|"
        r"\b(?:company|platform|business|website|app)\s+(?:ki|ke)?\s+baat\b|"
        r"\bwelfog\s+(?:kya|ky)\s+(?:karta|karti|karte|krti|krta|hai|h)\b|"
        r"\b(?:welfog|company)\s+(?:ke\s+)?baare\b",
        tl,
    ):
        return False
    return bool(re.search(r"\bwelfog\b", tl) or "company" in tl or "platform" in tl)


def query_is_welfog_company_or_platform(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """True when user asks about Welfog the company/platform — NOT the chat bot identity."""
    comb = _comb(original_msg, msg_en)
    if not comb:
        return False
    if user_frustration_about_company_not_bot(comb):
        return True
    try:
        from utils.helpers import (
            _is_welfog_about_fast_path,
            message_is_welfog_about_request,
            _text_has_platform_overview_intent,
        )

        if _is_welfog_about_fast_path(comb):
            return True
        if message_is_welfog_about_request(comb):
            return True
        if _text_has_platform_overview_intent(comb):
            return True
    except ImportError:
        pass
    try:
        from services.meta_turn_semantics import semantic_prefers_company_about

        if semantic_prefers_company_about(comb):
            return True
    except ImportError:
        pass
    return False


def query_is_assistant_identity(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """True only when user asks who THIS chat assistant/bot is — not Welfog company."""
    comb = _comb(original_msg, msg_en)
    if not comb:
        return False
    if query_is_welfog_company_or_platform(original_msg, msg_en, conversation_context):
        return False
    if user_frustration_about_company_not_bot(comb):
        return False
    try:
        from services.meta_turn_semantics import should_fast_reply_assistant_intro

        return should_fast_reply_assistant_intro(original_msg, msg_en)
    except ImportError:
        return False


def try_company_kb_reply_html(
    original_msg: str,
    msg_en: str = "",
    *,
    reply_lang: str = "en",
    conversation_context: str = "",
) -> Optional[str]:
    if not query_is_welfog_company_or_platform(
        original_msg, msg_en, conversation_context
    ):
        return None
    try:
        from services.kb_service import format_welfog_about_reply_from_kb

        body = format_welfog_about_reply_from_kb(
            original_msg, msg_en, reply_lang=reply_lang
        )
        if body:
            log_reasoning(
                "User query semantics: Welfog company/platform → company KB (not bot intro)."
            )
        return body or None
    except ImportError:
        return None


def ai_route_is_assistant_intro_blocked(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """When True, never route to assistant_intro — company/KB path wins."""
    return query_is_welfog_company_or_platform(
        original_msg, msg_en, conversation_context
    )
