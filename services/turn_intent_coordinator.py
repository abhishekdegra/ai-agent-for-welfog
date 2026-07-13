"""
Per-turn intent coordination — ONE micro-LLM classification per topic, cached for the whole turn.

AI infers meaning in any language/style. Keyword helpers are LLM-down failsafe only.
Structural skips (bare ID, pincode flow, catalog SKU) are not phrase lists.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

from utils.reasoning_log import log_reasoning

_ACCOUNT_LIST_CACHE = threading.local()
_AI_CONF_MIN = 0.45


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _turn_cache_key(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    reply_lang: str,
    prefix: str,
) -> str:
    return (
        f"{prefix}|{hash(_combined(original_msg, msg_en))}|"
        f"{hash((conversation_context or '')[-600:])}|{reply_lang or ''}"
    )


def _is_bare_id_token(text: str) -> bool:
    comb = (text or "").strip()
    return bool(
        re.fullmatch(r"[0-9]{4,20}", comb)
        or re.fullmatch(r"[A-Za-z0-9]{4,20}", comb)
    )


def strict_llm_failsafe_enabled() -> bool:
    return (os.getenv("STRICT_LLM_FAILSAFE", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def structural_skip_account_list_classifier(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """
    True when we should NOT spend an account-list LLM call (structural only).
  Not a phrase/intent keyword gate.
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return True
    if _is_bare_id_token(comb):
        return True
    if isinstance(ctx, dict) and ctx.get("awaiting") in ("order_id", "pincode", "category_select"):
        return True
    try:
        from utils.helpers import (
            message_is_conversation_reset_command,
            turn_is_catalog_product_lookup,
            turn_is_obvious_product_shopping_turn,
            _text_is_pincode_serviceability_question,
        )

        if message_is_conversation_reset_command(comb):
            return True
        if turn_is_catalog_product_lookup(original_msg, msg_en):
            return True
        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conversation_context
        ):
            return True
        if _text_is_pincode_serviceability_question(comb, conversation_context):
            return True
    except ImportError:
        pass
    try:
        from services.conversation_followup import is_deals_request_message

        if is_deals_request_message(original_msg, msg_en):
            return True
    except ImportError:
        pass
    try:
        from services.knowledge_query_pipeline import _heuristic_informational_signal

        if _heuristic_informational_signal(
            original_msg, msg_en, conversation_context
        ):
            return True
    except ImportError:
        pass
    brain = _brain_route_for_turn()
    if isinstance(brain, dict):
        bi = (brain.get("intent") or "").strip().lower()
        if bi in ("deals", "categories", "category_feed"):
            return True
    return False


def _brain_route_for_turn(ai_route: dict | None = None) -> dict | None:
    if isinstance(ai_route, dict) and ai_route:
        return ai_route
    try:
        from services.chat_flow_telemetry import get_cached_brain_route

        cached = get_cached_brain_route()
        return cached if isinstance(cached, dict) else None
    except ImportError:
        return None


def get_account_list_ai_classification(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    ai_route: dict | None = None,
) -> Optional[dict]:
    """
    Single micro-LLM per turn — cached for fast-path reply + router + executor.
    Skips LLM when ai_brain_route already set account_list_kind.
    """
    key = _turn_cache_key(original_msg, msg_en, conversation_context, reply_lang, "account_list")
    if getattr(_ACCOUNT_LIST_CACHE, "key", None) == key:
        cached = getattr(_ACCOUNT_LIST_CACHE, "result", None)
        if cached is not None:
            return cached

    brain = _brain_route_for_turn(ai_route)
    if brain:
        try:
            from services.account_list_semantics import (
                KIND_NONE,
                _norm_account_list_kind,
                account_list_action_from_brain_route,
            )

            action = account_list_action_from_brain_route(brain)
            kind = _norm_account_list_kind(action.get("kind") or KIND_NONE)
            if kind != KIND_NONE:
                result = {
                    "account_list_kind": kind,
                    "confidence": float(action.get("confidence") or 0.92),
                    "user_meaning": (action.get("user_meaning") or "").strip(),
                    "source": "ai_brain_route",
                }
                log_reasoning(
                    f"Account-list: reuse brain route kind={kind} — skip micro-LLM."
                )
                _ACCOUNT_LIST_CACHE.key = key
                _ACCOUNT_LIST_CACHE.result = result
                return result
        except ImportError:
            pass

    try:
        from services.account_list_semantics import ai_classify_account_list_turn

        result = ai_classify_account_list_turn(
            original_msg, msg_en, conversation_context, reply_lang
        )
    except ImportError:
        result = None

    _ACCOUNT_LIST_CACHE.key = key
    _ACCOUNT_LIST_CACHE.result = result
    return result


_ACCOUNT_LIST_BLOCK_PRODUCT_MIN_CONF = 0.42


def account_list_ai_blocks_product_path(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    ai_route: dict | None = None,
) -> bool:
    """
    Account-list micro-LLM classified purchase history or wishlist — product catalog
    must not run (any language/script; no customer keyword lists).
    """
    classified = get_account_list_ai_classification(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
    )
    if not classified:
        return False
    try:
        from services.account_list_semantics import KIND_NONE
    except ImportError:
        return False
    kind = (classified.get("account_list_kind") or KIND_NONE).strip().lower()
    conf = float(classified.get("confidence") or 0.0)
    if kind == KIND_NONE or conf < _ACCOUNT_LIST_BLOCK_PRODUCT_MIN_CONF:
        return False
    return kind in (
        "purchase_history_in_chat",
        "purchase_history_howto",
        "wishlist_in_chat",
        "wishlist_howto",
    )


def account_list_ai_is_live_list_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    ai_route: dict | None = None,
    min_conf: float = _ACCOUNT_LIST_BLOCK_PRODUCT_MIN_CONF,
) -> bool:
    """Cached account-list micro-LLM — show MY orders/wishlist in chat (any language)."""
    classified = get_account_list_ai_classification(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
    )
    if not classified:
        return False
    try:
        from services.account_list_semantics import KIND_NONE
    except ImportError:
        return False
    kind = (classified.get("account_list_kind") or KIND_NONE).strip().lower()
    conf = float(classified.get("confidence") or 0.0)
    if kind == KIND_NONE or conf < min_conf:
        return False
    return kind in ("purchase_history_in_chat", "wishlist_in_chat")


def resolve_account_list_action_ai_first(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
    *,
    ai_route: dict | None = None,
) -> dict:
    """
    AI-first account-list action. Keywords only when LLM unavailable (failsafe).
    """
    from services.account_list_semantics import (
        ACTION_NONE,
        KIND_NONE,
        _keyword_resolve_account_list_action,
        _norm_account_list_kind,
        account_list_action_from_brain_route,
    )

    empty = {
        "action": ACTION_NONE,
        "topic": "none",
        "kind": KIND_NONE,
        "confidence": 0.0,
        "source": "",
        "user_meaning": "",
    }

    if structural_skip_account_list_classifier(
        original_msg, msg_en, conversation_context, ctx
    ):
        return empty

    brain = ai_route
    if not brain and isinstance(ctx, dict):
        brain = (ctx.get("data") or {}).get("ai_route")
    brain_action = account_list_action_from_brain_route(brain or _brain_route_for_turn())
    if (brain_action.get("kind") or KIND_NONE) != KIND_NONE:
        log_reasoning(
            f"Account-list dispatch: brain kind={brain_action.get('kind')} "
            f"— no second classifier."
        )
        return brain_action

    classified = get_account_list_ai_classification(
        original_msg, msg_en, conversation_context, reply_lang, ai_route=brain
    )
    if classified:
        kind = _norm_account_list_kind(
            classified.get("account_list_kind") or KIND_NONE
        )
        conf = float(classified.get("confidence") or 0.0)
        um = (classified.get("user_meaning") or "").strip()
        if kind != KIND_NONE and conf >= _AI_CONF_MIN:
            from services.account_list_semantics import _kind_to_action_topic

            action, topic = _kind_to_action_topic(kind)
            log_reasoning(
                f"Account-list (AI-first): kind={kind} action={action} "
                f"conf={conf:.2f} — {um[:90] or 'no meaning'}"
            )
            return {
                "action": action,
                "topic": topic,
                "kind": kind,
                "confidence": conf,
                "source": "account_list_llm",
                "user_meaning": um,
            }

    if strict_llm_failsafe_enabled() and classified is not None:
        kind_llm = _norm_account_list_kind(
            classified.get("account_list_kind") or KIND_NONE
        )
        # LLM picked a live-list kind but below confidence — do not keyword-override.
        if kind_llm != KIND_NONE:
            return empty

    comb = _combined(original_msg, msg_en)
    kw = _keyword_resolve_account_list_action(comb, conversation_context, None)
    if (kw.get("kind") or KIND_NONE) != KIND_NONE:
        log_reasoning(
            f"Account-list (LLM-down keyword failsafe): kind={kw.get('kind')}"
        )
        return kw
    return empty


_KB_TURN_CACHE = threading.local()
_KB_AI_CONF_MIN = 0.52
_KB_CACHE_UNSET = object()


def peek_kb_turn_classification(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
):
    """Return cached KB-turn classify for this message, or _KB_CACHE_UNSET if not cached."""
    key = _turn_cache_key(
        original_msg, msg_en, conversation_context, reply_lang or "", "kb_turn"
    )
    if getattr(_KB_TURN_CACHE, "key", None) != key:
        return _KB_CACHE_UNSET
    return getattr(_KB_TURN_CACHE, "result", _KB_CACHE_UNSET)


def store_kb_turn_classification(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    result: dict | None = None,
) -> None:
    key = _turn_cache_key(
        original_msg, msg_en, conversation_context, reply_lang or "", "kb_turn"
    )
    _KB_TURN_CACHE.key = key
    _KB_TURN_CACHE.result = result


def kb_classified_as_live_api(classified: dict | None, *, min_conf: float = 0.48) -> bool:
    if not isinstance(classified, dict):
        return False
    return bool(classified.get("needs_live_api")) and float(
        classified.get("confidence") or 0.0
    ) >= min_conf


def kb_classified_as_product_catalog(
    classified: dict | None,
    *,
    min_conf: float = 0.48,
) -> bool:
    """KB-turn micro-LLM locked product_search — OpenSearch, not KB/brain stack."""
    if not isinstance(classified, dict):
        return False
    if not bool(classified.get("needs_live_api")):
        return False
    if float(classified.get("confidence") or 0.0) < min_conf:
        return False
    live = (classified.get("live_api_kind") or "none").strip().lower()
    return live == "product_search"


def structural_skip_kb_turn_classifier(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """Structural only — not phrase/intent keyword gates."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return True
    if _is_bare_id_token(comb):
        return True
    if isinstance(ctx, dict) and ctx.get("awaiting") in ("order_id", "pincode", "category_select"):
        return True
    try:
        from utils.helpers import message_is_conversation_reset_command, turn_is_catalog_product_lookup

        if message_is_conversation_reset_command(comb):
            return True
        if turn_is_catalog_product_lookup(original_msg, msg_en):
            return True
    except ImportError:
        pass
    brain = _brain_route_for_turn()
    if brain:
        ch = (brain.get("data_channel") or "").strip().lower()
        if ch in ("live_api", "catalog"):
            return True
        try:
            from services.account_list_semantics import account_list_route_is_locked

            if account_list_route_is_locked(brain):
                return True
        except ImportError:
            pass
        if (brain.get("route_handler") or "").strip().lower() in (
            "refund_status_api",
            "order_tracking_api",
            "order_details_api",
            "wishlist_api",
            "order_ai_flow",
            "product_ai_flow",
            "pincode_delivery_api",
        ):
            return True
    try:
        from services.early_live_dispatch import turn_blocks_kb_pre_scope

        if turn_blocks_kb_pre_scope(
            original_msg, msg_en, conversation_context, ai_route=brain, route_decision=None
        ):
            return True
    except ImportError:
        pass
    return False


def get_kb_turn_ai_classification(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    ai_route: dict | None = None,
    preflight: bool = False,
) -> Optional[dict]:
    """One micro-LLM per turn — cached for knowledge fast path + refund guard."""
    key = _turn_cache_key(
        original_msg, msg_en, conversation_context, reply_lang, "kb_turn"
    )
    if getattr(_KB_TURN_CACHE, "key", None) == key:
        cached = getattr(_KB_TURN_CACHE, "result", None)
        if cached is not None:
            return cached

    if not preflight:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                log_reasoning("KB-turn: defer/skip — universal brain route owns classification.")
                _KB_TURN_CACHE.key = key
                _KB_TURN_CACHE.result = None
                return None
        except ImportError:
            pass

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    try:
        from utils.helpers import turn_is_obvious_product_shopping_turn

        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conversation_context
        ):
            log_reasoning("KB-turn: product shopping — skip micro-LLM.")
            _KB_TURN_CACHE.key = key
            _KB_TURN_CACHE.result = None
            return None
    except ImportError:
        pass

    try:
        from utils.helpers import (
            _is_light_smalltalk_fast,
            _is_short_pure_greeting,
            _looks_like_greeting_message,
        )

        opener = (original_msg or comb).strip()
        if (
            _looks_like_greeting_message(opener)
            or _is_short_pure_greeting(opener)
            or _is_light_smalltalk_fast(original_msg, msg_en)
        ):
            _KB_TURN_CACHE.key = key
            _KB_TURN_CACHE.result = None
            log_reasoning("KB-turn: casual opener — skip micro-LLM.")
            return None
    except ImportError:
        pass

    route = ai_route or _brain_route_for_turn()
    if route:
        ch = (route.get("data_channel") or "").strip().lower()
        scope = (route.get("conversation_scope") or "").strip().lower()
        if scope in (
            "general_chitchat",
            "out_of_domain",
            "harm_sensitive",
        ):
            log_reasoning("KB-turn: brain scope chitchat/OOD — skip micro-LLM.")
            _KB_TURN_CACHE.key = key
            _KB_TURN_CACHE.result = None
            return None
        try:
            from services.ai_route_semantics import (
                brain_route_indicates_informational_kb,
                brain_route_prefers_kb_answer,
            )
            from services.knowledge_query_pipeline import (
                kb_embedding_indicates_informational_turn,
            )

            if brain_route_indicates_informational_kb(
                route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            ):
                keys = list(route.get("kb_keys") or [])
                um = (route.get("user_meaning") or "").strip()
                if not keys and brain_route_prefers_kb_answer(route):
                    keys = list(route.get("kb_keys") or [])
                topic = keys[0] if keys else "general_faq"
                result = {
                    "is_informational_kb": True,
                    "needs_live_api": False,
                    "confidence": 0.88,
                    "kb_topic": topic,
                    "user_meaning_en": um,
                    "source": "ai_brain_route",
                }
                log_reasoning(
                    "KB-turn: brain JSON / embedding informational — skip micro-LLM."
                )
                _KB_TURN_CACHE.key = key
                _KB_TURN_CACHE.result = result
                return result
            if ch == "catalog" and kb_embedding_indicates_informational_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=route,
            ):
                um = (route.get("user_meaning") or "").strip()
                result = {
                    "is_informational_kb": True,
                    "needs_live_api": False,
                    "confidence": 0.85,
                    "kb_topic": "general_faq",
                    "user_meaning_en": um,
                    "source": "kb_embedding",
                }
                log_reasoning(
                    "KB-turn: admin embedding — brain catalog misroute corrected."
                )
                _KB_TURN_CACHE.key = key
                _KB_TURN_CACHE.result = result
                return result
        except ImportError:
            pass
        if ch == "live_api":
            log_reasoning("KB-turn: brain live_api — skip micro-LLM.")
            _KB_TURN_CACHE.key = key
            _KB_TURN_CACHE.result = None
            return None
        if ch == "catalog":
            try:
                from services.product_catalog_resolver import (
                    product_catalog_route_is_locked,
                )
                from utils.helpers import turn_is_obvious_product_shopping_turn

                pe = route.get("_product_entities") or route.get("product_entities") or {}
                has_pe = isinstance(pe, dict) and any(
                    v not in (None, "", [], {}) for v in pe.values()
                )
                if turn_is_obvious_product_shopping_turn(
                    original_msg, msg_en, conversation_context
                ) or (product_catalog_route_is_locked(route) and has_pe):
                    log_reasoning(
                        "KB-turn: brain catalog + shopping — skip micro-LLM."
                    )
                    _KB_TURN_CACHE.key = key
                    _KB_TURN_CACHE.result = None
                    return None
                log_reasoning(
                    "KB-turn: brain catalog but FAQ semantic — run micro-LLM."
                )
            except ImportError:
                log_reasoning("KB-turn: brain catalog — skip micro-LLM.")
                _KB_TURN_CACHE.key = key
                _KB_TURN_CACHE.result = None
                return None
        if ch == "kb" and route.get("kb_keys"):
            keys = list(route.get("kb_keys") or [])
            um = (route.get("user_meaning") or "").strip()
            result = {
                "is_informational_kb": True,
                "needs_live_api": False,
                "confidence": 0.9,
                "kb_topic": keys[0] if keys else "general_faq",
                "user_meaning_en": um,
                "source": "ai_brain_route",
            }
            log_reasoning(f"KB-turn: reuse brain kb_keys={keys[:3]} — skip micro-LLM.")
            _KB_TURN_CACHE.key = key
            _KB_TURN_CACHE.result = result
            return result

    try:
        from services.knowledge_query_pipeline import ai_classify_kb_turn

        result = ai_classify_kb_turn(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            ai_route=ai_route,
            preflight=preflight,
        )
    except ImportError:
        result = None

    _KB_TURN_CACHE.key = key
    _KB_TURN_CACHE.result = result
    return result


def kb_turn_is_informational(
    classified: dict | None,
    *,
    min_conf: float = _KB_AI_CONF_MIN,
) -> bool:
    if not classified:
        return False
    conf = float(classified.get("confidence") or 0.0)
    return (
        bool(classified.get("is_informational_kb"))
        and not bool(classified.get("needs_live_api"))
        and conf >= min_conf
    )


def peek_or_classify_kb_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    preflight: bool = True,
) -> dict | None:
    """One cached KB-turn classification per message (preflight bypasses brain defer)."""
    peeked = peek_kb_turn_classification(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if peeked is not _KB_CACHE_UNSET:
        return peeked
    return get_kb_turn_ai_classification(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        preflight=preflight,
    )


def kb_turn_blocks_product_catalog(
    classified: dict | None,
    *,
    min_conf: float = 0.42,
) -> bool:
    """True when KB micro-LLM locked an informational FAQ/policy turn."""
    return kb_turn_is_informational(classified, min_conf=min_conf)


def resolve_kb_turn_ai_first(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
    *,
    ai_route: dict | None = None,
    preflight: bool = False,
) -> dict:
    """
    AI-first: should this turn be answered from KB (any language)?
    Keywords / embedding-only when LLM unavailable (STRICT_LLM_FAILSAFE).
    """
    empty = {
        "action": "none",
        "kb_topic": "none",
        "user_meaning": "",
        "confidence": 0.0,
        "source": "",
        "kb_keys": [],
    }

    if structural_skip_kb_turn_classifier(
        original_msg, msg_en, conversation_context, ctx
    ):
        return empty

    try:
        if account_list_ai_is_live_list_turn(
            original_msg, msg_en, conversation_context, reply_lang, ai_route=ai_route
        ):
            log_reasoning(
                "KB turn skipped — account-list AI classified live orders/wishlist API."
            )
            return empty
    except ImportError:
        pass

    classified = get_kb_turn_ai_classification(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
        preflight=preflight,
    )
    if kb_turn_is_informational(classified):
        topic = (classified.get("kb_topic") or "general_faq").strip().lower()
        try:
            from services.knowledge_query_pipeline import kb_keys_for_topic

            keys = kb_keys_for_topic(topic)
        except ImportError:
            keys = ["faqs"]
        um = (classified.get("user_meaning_en") or "").strip()
        log_reasoning(
            f"KB turn (AI-first): topic={topic} conf={float(classified.get('confidence') or 0):.2f}"
        )
        return {
            "action": "answer_kb",
            "kb_topic": topic,
            "user_meaning": um,
            "confidence": float(classified.get("confidence") or 0.0),
            "source": "kb_turn_llm",
            "kb_keys": keys,
        }

    if strict_llm_failsafe_enabled() and classified is not None:
        return empty

    try:
        from services.kb_service import resolve_best_faq_chunk_for_question

        faq = resolve_best_faq_chunk_for_question(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        )
        if faq and float(faq.get("score") or 0) >= 0.44:
            log_reasoning(
                f"KB turn (LLM-down embedding failsafe): score={float(faq.get('score') or 0):.2f}"
            )
            return {
                "action": "answer_kb",
                "kb_topic": "general_faq",
                "user_meaning": "",
                "confidence": float(faq.get("score") or 0),
                "source": "embedding_failsafe",
                "kb_keys": ["faqs"],
            }
    except ImportError:
        pass

    return empty
