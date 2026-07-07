"""
Refund intent fast path — ONE micro-LLM classification → policy KB / live API / ask Order ID.

Primary: ai_classify_refund_status_turn (any language, any phrasing).
Fallback: resolve_refund_turn heuristics only when LLM is unavailable.
Structural only (not phrase lists): Order ID digits → live API, never PIN.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Literal, Optional

from utils.reasoning_log import log_reasoning

RefundFastAction = Literal["policy_kb", "personal_api", "ask_order_id"]

_REFUND_FAST_CACHE = threading.local()
_AI_CONF_MIN = 0.5


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _cache_key(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    reply_lang: str,
) -> str:
    return (
        f"{hash(_combined(original_msg, msg_en))}|"
        f"{hash((conversation_context or '')[-500:])}|{reply_lang}"
    )


def _get_refund_ai_classification(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[dict]:
    """Single micro-LLM call per turn — cached for route + reply paths."""
    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if should_skip_micro_classifier_llm():
            log_reasoning(
                "Refund-status: defer/skip — universal brain route owns classification."
            )
            return None
    except ImportError:
        pass

    key = _cache_key(original_msg, msg_en, conversation_context, reply_lang)
    if getattr(_REFUND_FAST_CACHE, "key", None) == key:
        cached = getattr(_REFUND_FAST_CACHE, "result", None)
        if cached is not None:
            return cached

    try:
        from services.refund_status_semantics import ai_classify_refund_status_turn

        result = ai_classify_refund_status_turn(
            original_msg, msg_en, conversation_context, reply_lang
        )
    except ImportError:
        result = None

    _REFUND_FAST_CACHE.key = key
    _REFUND_FAST_CACHE.result = result
    return result


def _kb_turn_blocks_refund_fast_path(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """Cached KB-turn LLM already classified informational FAQ — not refund fast path."""
    try:
        from services.turn_intent_coordinator import (
            get_kb_turn_ai_classification,
            kb_turn_is_informational,
        )

        kb = get_kb_turn_ai_classification(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if kb_turn_is_informational(kb):
            topic = (kb.get("kb_topic") or "").strip().lower()
            if topic != "refund_return_policy":
                return True
    except ImportError:
        pass
    return False


def _refund_llm_misclassified_non_refund(comb: str, classified: dict) -> bool:
    """Refund micro-LLM misroute — use meaning from refund classifier, not keyword lists."""
    um = f" {(classified.get('user_meaning') or '').lower()} "
    rk = (classified.get("refund_turn_kind") or "").strip().lower()
    if rk != "policy_howto":
        return False
    if not um.strip():
        return False
    if any(x in um for x in ("refund", "return", "money back", "wrong item", "damaged")):
        return False
    if any(
        x in um
        for x in (
            "delivery time",
            "shipping time",
            "how long",
            "deliver",
            "delivery",
            "shipping",
            "arrive",
            "pincode",
            "serviceability",
        )
    ):
        return True
    return False


def turn_has_refund_topic(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """
    AI-first gate — one cached micro-LLM per turn. Keywords only when LLM is down.
    """
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import (
            _is_light_smalltalk_fast,
            _is_short_pure_greeting,
            _looks_like_greeting_message,
        )

        if (
            _looks_like_greeting_message(original_msg or comb)
            or _is_short_pure_greeting(comb)
            or _is_light_smalltalk_fast(original_msg, msg_en)
        ):
            return False
    except ImportError:
        pass
    if _kb_turn_blocks_refund_fast_path(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        log_reasoning("Refund topic blocked — KB-turn LLM classified informational FAQ.")
        return False

    if _explicit_non_refund_goal(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return False

    try:
        from services.turn_intent_coordinator import structural_skip_account_list_classifier

        if structural_skip_account_list_classifier(
            original_msg, msg_en, conversation_context, None
        ):
            comb = _combined(original_msg, msg_en)
            if re.fullmatch(r"[0-9]{4,20}", comb.strip()):
                return False
    except ImportError:
        pass

    classified = _get_refund_ai_classification(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if classified:
        if _refund_llm_misclassified_non_refund(comb, classified):
            log_reasoning(
                "Refund topic blocked — delivery/FAQ turn (refund LLM override)."
            )
            return False
        kind = (classified.get("refund_turn_kind") or "none").strip().lower()
        conf = float(classified.get("confidence") or 0.0)
        if kind in ("personal_status", "policy_howto") and conf >= _AI_CONF_MIN:
            return True

    try:
        from services.turn_intent_coordinator import strict_llm_failsafe_enabled
    except ImportError:
        strict_llm_failsafe_enabled = lambda: True  # type: ignore

    if classified is not None and strict_llm_failsafe_enabled():
        return False

    comb = _combined(original_msg, msg_en)
    try:
        from services.refund_status_semantics import _message_has_refund_topic

        if _message_has_refund_topic(comb):
            log_reasoning("Refund topic (LLM-down keyword failsafe).")
            return True
    except ImportError:
        from utils.helpers import _text_has_refund_or_return_intent

        if _text_has_refund_or_return_intent(comb):
            return True
    return False


def refund_blocks_pincode_fast_path(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Refund + order id digits must never become a truncated 6-digit PIN check."""
    if not turn_has_refund_topic(original_msg, msg_en, conversation_context):
        return False
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import (
            _digits_in_message_are_order_id_not_pincode,
            extract_order_id,
        )

        if extract_order_id(comb, conversation_context):
            return True
        if _digits_in_message_are_order_id_not_pincode(
            comb, conversation_context
        ):
            return True
    except ImportError:
        pass
    if re.search(r"\b[0-9]{4,20}\b", comb) and re.search(
        r"\b(?:refund|return|id)\b", comb, re.I
    ):
        return True
    classified = _get_refund_ai_classification(
        original_msg, msg_en, conversation_context
    )
    if classified:
        kind = (classified.get("refund_turn_kind") or "").strip().lower()
        if kind == "personal_status" and re.search(r"\b[0-9]{4,20}\b", comb):
            return True
    return False


def resolve_refund_order_id(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import (
            _bare_order_id_token_from_msg,
            _digits_in_message_are_order_id_not_pincode,
            extract_order_id,
            extract_latest_order_id_from_user_conversation,
            resolve_order_id_for_tracking,
        )
    except ImportError:
        return ""

    labeled = re.search(
        r"\b([0-9]{4,20})\b\s+(?:is\s+)?id\b",
        comb,
        re.IGNORECASE,
    )
    if labeled:
        return labeled.group(1)

    oid = extract_order_id(comb, conversation_context)
    if oid:
        return oid

    if _digits_in_message_are_order_id_not_pincode(comb, conversation_context):
        m7 = re.search(r"\b([0-9]{7,20})\b", comb)
        if m7:
            return m7.group(1)
        bare = _bare_order_id_token_from_msg(comb, conversation_context)
        if bare:
            return bare

    oid = resolve_order_id_for_tracking(
        original_msg.strip() or msg_en.strip(),
        conversation_context,
    )
    if oid:
        return oid

    return extract_latest_order_id_from_user_conversation(
        conversation_context, comb
    ) or ""


def _explicit_non_refund_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> str:
    """Latest turn asks for details / track / invoice — not refund fast path."""
    try:
        from services.conversation_thread_semantics import (
            resolve_explicit_turn_goal_from_message,
        )

        explicit = resolve_explicit_turn_goal_from_message(
            original_msg,
            msg_en,
            conversation_context,
            None,
            allow_llm=True,
            reply_lang=reply_lang,
        )
        if explicit in ("order_details", "order_invoice", "track", "payment"):
            return explicit
    except ImportError:
        pass
    try:
        from services.order_details_flow import message_wants_order_details_or_invoice

        od = message_wants_order_details_or_invoice(
            original_msg, msg_en, conversation_context
        )
        if od in ("order_details", "order_invoice"):
            return od
    except ImportError:
        pass
    try:
        from utils.helpers import _text_is_order_tracking_intent_leaf

        if _text_is_order_tracking_intent_leaf(_combined(original_msg, msg_en)):
            return "track"
    except ImportError:
        pass
    return ""


def _action_from_ai_classification(
    classified: dict,
    *,
    has_order_id: bool,
) -> Optional[RefundFastAction]:
    from services.refund_status_semantics import KIND_PERSONAL_STATUS, KIND_POLICY_HOWTO

    kind = (classified.get("refund_turn_kind") or "none").strip().lower()
    conf = float(classified.get("confidence") or 0.0)
    if kind == "none" or conf < _AI_CONF_MIN:
        return None
    if kind == KIND_POLICY_HOWTO:
        return "policy_kb"
    if kind == KIND_PERSONAL_STATUS:
        return "personal_api" if has_order_id else "ask_order_id"
    return None


def _action_from_heuristic_fallback(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    *,
    has_order_id: bool,
) -> Optional[RefundFastAction]:
    """LLM unavailable only — not the primary classifier."""
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import _text_is_refund_return_policy_howto

        if _text_is_refund_return_policy_howto(comb):
            return "policy_kb"
    except ImportError:
        pass

    try:
        from services.refund_status_semantics import (
            KIND_PERSONAL_STATUS,
            KIND_POLICY_HOWTO,
            resolve_refund_turn,
        )

        resolved = resolve_refund_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=None,
            allow_llm=False,
        )
        if resolved.kind == KIND_POLICY_HOWTO:
            return "policy_kb"
        if resolved.kind == KIND_PERSONAL_STATUS:
            return "personal_api" if has_order_id else "ask_order_id"
    except ImportError:
        pass

    if re.fullmatch(r"refund", comb.strip(), re.I):
        return "ask_order_id"
    return "policy_kb"


def classify_refund_fast_action(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[RefundFastAction]:
    if _explicit_non_refund_goal(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return None

    comb = _combined(original_msg, msg_en)
    if _kb_turn_blocks_refund_fast_path(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return None

    try:
        from utils.helpers import _text_is_refund_return_policy_howto

        if _text_is_refund_return_policy_howto(comb):
            log_reasoning("Refund fast path: policy KB (general timeline/how-to).")
            return "policy_kb"
    except ImportError:
        pass

    if not turn_has_refund_topic(
        original_msg, msg_en, conversation_context, reply_lang
    ):
        return None

    try:
        from services.conversation_thread_semantics import (
            resolve_explicit_turn_goal_from_message,
        )

        explicit = resolve_explicit_turn_goal_from_message(
            original_msg,
            msg_en,
            conversation_context,
            None,
            allow_llm=True,
            reply_lang=reply_lang,
        )
        if explicit == "refund_status":
            from services.refund_status_semantics import (
                KIND_PERSONAL_STATUS,
                KIND_POLICY_HOWTO,
                resolve_refund_turn,
            )

            resolved = resolve_refund_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=None,
                reply_lang=reply_lang,
                allow_llm=True,
            )
            if resolved.kind == KIND_POLICY_HOWTO:
                return "policy_kb"
            if resolved.kind == KIND_PERSONAL_STATUS:
                oid = resolve_refund_order_id(
                    original_msg, msg_en, conversation_context
                )
                return "personal_api" if oid else "ask_order_id"
    except ImportError:
        pass

    oid = resolve_refund_order_id(original_msg, msg_en, conversation_context)
    has_oid = bool(oid)

    classified = _get_refund_ai_classification(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if classified:
        if _refund_llm_misclassified_non_refund(comb, classified):
            log_reasoning("Refund fast path skipped — not a refund/return question.")
            return None
        action = _action_from_ai_classification(classified, has_order_id=has_oid)
        if action:
            log_reasoning(
                f"Refund fast path (AI): {action} — "
                f"{(classified.get('user_meaning') or '')[:100]}"
            )
            return action

    action = _action_from_heuristic_fallback(
        original_msg, msg_en, conversation_context, has_order_id=has_oid
    )
    if action:
        log_reasoning(f"Refund fast path (LLM-down fallback): {action}")
    return action


def try_refund_intent_fast_reply(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    user_id: str,
    reply_lang: str,
    ctx: dict | None,
    *,
    reply_for_live_order_id_lookup,
    reset_context_fn=None,
) -> Optional[str]:
    action = classify_refund_fast_action(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if not action:
        return None

    lang = reply_lang or "en"
    log_reasoning(f"Refund fast path: action={action} (skip routing cascade).")
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        record_route_step("refund_intent_fast")
        record_route(intent="refund", source=f"refund_fast_{action}")
    except ImportError:
        pass

    if isinstance(ctx, dict):
        if reset_context_fn:
            reset_context_fn(ctx)
        ctx["last"] = "refund"
        ctx["awaiting"] = None

    if action == "policy_kb":
        from services.kb_service import format_policy_help_reply_from_kb

        body = format_policy_help_reply_from_kb(original_msg, msg_en, reply_lang=lang)
        if body:
            return body

    if action == "ask_order_id":
        from services.order_history_flow import _localized_sysmsg

        body = _localized_sysmsg(
            "ask_order_id_for_intent", original_msg, reply_lang=lang, intent="refund"
        )
        if body:
            if isinstance(ctx, dict):
                ctx["awaiting"] = "order_id"
            return body

    oid = resolve_refund_order_id(original_msg, msg_en, conversation_context)
    if not oid:
        return None

    if isinstance(ctx, dict):
        ctx["order_id"] = oid
        ctx["awaiting"] = None

    return reply_for_live_order_id_lookup(
        "refund", oid, user_id, original_msg, lang
    )


def try_refund_intent_fast_route(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: dict | None = None,
    reply_lang: str = "en",
) -> Optional[tuple[Any, dict]]:
    action = classify_refund_fast_action(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    if not action:
        return None

    from services.answer_router import AnswerRouteDecision

    oid = resolve_refund_order_id(original_msg, msg_en, conv_for_llm)
    if action == "personal_api" and not oid:
        action = "ask_order_id"

    classified = _get_refund_ai_classification(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    user_meaning = (classified or {}).get("user_meaning") or ""

    if action == "policy_kb":
        from services.refund_status_semantics import _apply_refund_policy_to_route

        route_data = _apply_refund_policy_to_route(
            {
                "user_meaning": user_meaning or "General refund/return policy or timeline",
                "reasoning": "Refund fast path — AI policy KB (no live API).",
            },
            "refund_fast_policy",
        )
        handler = "policy_structured_kb"
        intent = "refund"
    elif action == "ask_order_id":
        route_data = {
            "user_meaning": user_meaning or "Personal refund status — need Order ID",
            "reasoning": "Refund fast path — AI personal status, ask Order ID.",
            "intent": "refund",
            "data_channel": "live_api",
            "needs_order_id": True,
            "numeric_context": "order_id",
            "order_lookup_kind": "refund_status",
            "route_handler": "refund_status_api",
            "_refund_intent_fast": True,
        }
        handler = "refund_status_api"
        intent = "refund"
    else:
        from services.refund_status_semantics import _apply_refund_status_to_route

        route_data = _apply_refund_status_to_route(
            {
                "user_meaning": user_meaning or f"Refund status for order {oid}",
                "reasoning": "Refund fast path — live return-request API.",
                "extracted_order_id": oid,
                "needs_order_id": False,
            },
            "refund_fast_api",
        )
        handler = "refund_status_api"
        intent = "refund"

    route_data["_preflight_api"] = action != "policy_kb"
    route_data["_refund_intent_fast"] = True
    decision = AnswerRouteDecision(
        source="api" if action != "policy_kb" else "kb",
        intent=intent,
        handler=handler,
        is_welfog_related=True,
        reason=f"Refund fast path — {action}",
    )
    return decision, route_data
