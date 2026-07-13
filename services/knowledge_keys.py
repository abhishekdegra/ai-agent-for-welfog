"""
Canonical knowledge document identity.

Admin titles, Brain kb_keys, runtime catalog keys, and Qdrant payload titles
must all use the same normalization so soft hints and retrieval stay aligned.
"""
from __future__ import annotations

import re


def canonical_knowledge_key(title: str) -> str:
    """
    Normalize any Admin title / Brain key / payload title to one key form.

    Example: "COD-Policy" / "cod policy" / "cod_policy" → "cod_policy"
    """
    key_base = (title or "").replace("-", "_").replace(" ", "_").lower()
    return re.sub(r"[^a-z0-9_]", "", key_base)


def is_internal_agent_knowledge_key(key_or_title: str) -> bool:
    """
    True only for agent/internal routing docs — never for customer Admin knowledge.

    Matching is by canonical key / exact internal stems, not loose title substrings.
    """
    k = canonical_knowledge_key(key_or_title)
    if not k:
        return False
    if k.startswith("welfog_api"):
        return True
    if k in ("system_messages", "system_messages_2", "system_message"):
        return True
    return False
