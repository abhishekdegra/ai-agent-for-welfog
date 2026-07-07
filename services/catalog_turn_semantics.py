"""
Block catalog search on greetings / meta talk — semantic + lightweight rules.
"""
from __future__ import annotations

import re


def should_skip_catalog_for_conversational_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """True when this turn must NOT run OpenSearch product search."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return True
    try:
        from services.conversation_scope import _has_definite_welfog_shopping_signal

        if _has_definite_welfog_shopping_signal(comb):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _is_light_smalltalk_fast,
            _looks_like_greeting_message,
        )

        if _looks_like_greeting_message(original_msg or comb):
            return True
        if _is_light_smalltalk_fast(original_msg, msg_en):
            return True
    except ImportError:
        pass
    try:
        from services.conversation_followup import is_non_product_search_phrase

        if is_non_product_search_phrase(comb):
            return True
    except ImportError:
        pass
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(comb, conversation_context):
            return True
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _looks_like_greeting_message,
            _looks_like_light_smalltalk,
            _text_has_concrete_welfog_support_question,
            message_is_user_feedback_or_closing,
            message_is_bot_search_complaint,
            message_is_user_confused_or_rephrasing_bot,
        )

        if _looks_like_greeting_message(original_msg or comb):
            return True
        if _looks_like_light_smalltalk(original_msg, msg_en) and not _text_has_concrete_welfog_support_question(
            comb
        ):
            if not _has_explicit_catalog_ask(comb):
                return True
        if message_is_user_feedback_or_closing(comb):
            return True
        if message_is_bot_search_complaint(comb):
            return True
        if message_is_user_confused_or_rephrasing_bot(comb, conversation_context):
            return True
    except ImportError:
        pass
    tl = f" {comb.lower()} "
    if re.search(r"\b(?:hello|hi|hey|namaste)\b", tl) and re.search(
        r"\b(?:kesa|kaise|kese|kaisa|how are you|how r u)\b", tl
    ):
        if not _has_explicit_catalog_ask(comb):
            return True
    if re.search(r"\b(?:tu|tum|aap)\s+search\s+kr", tl) or "search krke bta" in tl:
        return True
    return False


def _has_explicit_catalog_ask(text: str) -> bool:
    try:
        from services.product_browse_semantics import _product_nouns_in_text
        from utils.helpers import _message_has_catalog_product_signal

        if _product_nouns_in_text(text):
            return True
        if _message_has_catalog_product_signal(text):
            return True
        tl = f" {(text or '').lower()} "
        if any(
            m in tl
            for m in (
                "dikhao", "dikha", "dikhana", "chahiye", "milega", "apke pas",
                "do you have", "show me", "under rs", "above rs", "sku",
            )
        ) and _product_nouns_in_text(text):
            return True
    except ImportError:
        pass
    return False
