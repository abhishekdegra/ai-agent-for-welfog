"""Backward-compatible re-exports — use services.conversation_scope (AI-first)."""
from services.conversation_scope import (  # noqa: F401
    build_off_topic_polite_reply,
    build_scope_reply,
    message_is_obvious_off_topic_outside_welfog,
    resolve_conversation_scope,
    try_conversation_scope_reply,
    SCOPE_CHITCHAT,
    SCOPE_OUT,
    SCOPE_WELFOG,
    ScopeDecision,
)
