"""
AI-first personal refund/return STATUS vs refund POLICY how-to — any language.

Keywords in helpers.py are LLM-unavailable failsafe only (STRICT_AI_INTENT_ROUTING).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from services.ai_route_semantics import coerce_route_str
from utils.reasoning_log import log_reasoning

_REFUND_TURN_CACHE = threading.local()
_RESOLVE_REFUND_CACHE = threading.local()


@dataclass
class ResolvedRefundTurn:
    kind: str  # personal_status | policy_howto | none
    source: str = ""
    confidence: str = ""
    user_meaning: str = ""


def _message_has_refund_topic(comb: str) -> bool:
    if not (comb or "").strip():
        return False
    from utils.helpers import _text_has_refund_or_return_intent

    tl = f" {comb.lower()} "
    return bool(
        _text_has_refund_or_return_intent(comb)
        or "refund" in tl
        or "return" in tl
        or "paise wapas" in tl
        or "money back" in tl
    )

KIND_NONE = "none"
KIND_PERSONAL_STATUS = "personal_status"
KIND_POLICY_HOWTO = "policy_howto"

_PERSONAL_STATUS_MEANING = (
    "refund status",
    "return status",
    "refund record",
    "return record",
    "refund progress",
    "return progress",
    "money back status",
    "refund approved",
    "return approved",
    "refund pending",
    "refund processed",
    "when will refund",
    "when will my refund",
    "when will i get refund",
    "refund for order",
    "return for order",
    "check refund",
    "check return",
    "refund on order",
    "return on order",
    "refund timeline",
    "refund update",
    "paise wapas status",
    "paise kab",
)

_POLICY_HOWTO_MEANING = (
    "refund policy",
    "return policy",
    "how to return",
    "how to refund",
    "how do i return",
    "return process",
    "refund process",
    "return steps",
    "refund steps",
    "return kaise",
    "refund kaise",
    "wrong item",
    "damaged item",
    "defective",
)


def _combined(original_msg: str, msg_en: str) -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _meaning_blob(route: dict | None) -> str:
    r = route or {}
    return f" {(r.get('user_meaning') or '').lower()} {(r.get('reasoning') or '').lower()} "


_ORDER_DETAILS_EXCLUDE = (
    "order detail",
    "order details",
    "order summary",
    "order info",
    "payment status",
    "payment mode",
    "delivery address",
    "shipping address",
    "product in order",
    "what did i order",
    "grand total",
    "order breakdown",
)


def _kind_from_meaning_blob(blob: str) -> str:
    if not blob.strip():
        return KIND_NONE
    try:
        from services.order_tracking_semantics import tracking_blocks_refund_meaning

        if tracking_blocks_refund_meaning(blob):
            return KIND_NONE
    except ImportError:
        pass
    if any(m in blob for m in _ORDER_DETAILS_EXCLUDE):
        return KIND_NONE
    if any(m in blob for m in _POLICY_HOWTO_MEANING):
        if not any(m in blob for m in _PERSONAL_STATUS_MEANING):
            how = any(
                x in blob
                for x in (
                    " how to ",
                    " how do ",
                    " steps ",
                    " process ",
                    " policy ",
                    " kaise ",
                    " kese ",
                )
            )
            if how:
                return KIND_POLICY_HOWTO
    if any(m in blob for m in _PERSONAL_STATUS_MEANING):
        return KIND_PERSONAL_STATUS
    if "refund" in blob or "return" in blob:
        if any(
            x in blob
            for x in (
                "status",
                "approved",
                "pending",
                "processed",
                "when will",
                "check my",
                "my order",
                "order id",
                "this order",
            )
        ):
            return KIND_PERSONAL_STATUS
    return KIND_NONE


def ai_route_requests_refund_policy(route: dict | None) -> bool:
    """General refund/return policy — KB only, not live API."""
    if not route:
        return False
    from services.ai_route_semantics import ai_route_is_kb_read
    from services.semantic_intent import llm_semantic_route_available

    if not llm_semantic_route_available(route):
        return False
    if ai_route_is_kb_read(route):
        intent = (route.get("intent") or "").strip().lower()
        if intent in ("refund", "general"):
            return True
    kind = _kind_from_meaning_blob(_meaning_blob(route))
    return kind == KIND_POLICY_HOWTO


def resolve_refund_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> ResolvedRefundTurn:
    """
    AI-first: personal refund STATUS (live API) vs refund POLICY how-to (KB).
    """
    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import (
            _text_is_refund_return_policy_howto,
            _text_is_refund_return_status_lookup,
        )

        if _text_is_refund_return_status_lookup(comb, conversation_context):
            out = ResolvedRefundTurn(
                kind=KIND_PERSONAL_STATUS,
                source="message_status_early",
                confidence="high",
                user_meaning="Personal refund/return status for one order",
            )
            cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        if _text_is_refund_return_policy_howto(comb):
            out = ResolvedRefundTurn(
                kind=KIND_POLICY_HOWTO,
                source="keyword_policy_early",
                confidence="high",
                user_meaning="General refund/return policy or timeline question",
            )
            cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
    except ImportError:
        pass

    try:
        from services.conversation_thread_semantics import (
            _EXPLICIT_GOAL_GUARD,
            infer_order_thread_goal,
            message_needs_thread_continuation,
        )

        if getattr(_EXPLICIT_GOAL_GUARD, "active", False):
            pass
        elif message_needs_thread_continuation(comb, conversation_context):
            thread = infer_order_thread_goal(
                conversation_context,
                comb,
                ctx_last=(
                    (ai_route or {}).get("_ctx_last")
                    if isinstance(ai_route, dict)
                    else None
                ),
                ai_route=ai_route if isinstance(ai_route, dict) else None,
                reply_lang=reply_lang,
                allow_llm=allow_llm,
            )
            if thread == "refund_status":
                out = ResolvedRefundTurn(
                    kind=KIND_PERSONAL_STATUS,
                    source="thread_continuation",
                    confidence="high",
                    user_meaning="Continue refund status thread with supplied order id",
                )
                _RESOLVE_REFUND_CACHE.key = (
                    f"{hash(comb)}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
                )
                _RESOLVE_REFUND_CACHE.result = out
                return out
    except ImportError:
        pass
    if not _message_has_refund_topic(comb):
        try:
            from utils.helpers import (
                _text_has_refund_or_return_intent,
                _text_is_refund_return_policy_howto,
                extract_latest_order_id_from_user_conversation,
                extract_order_id,
            )

            if _text_has_refund_or_return_intent(comb) and not _text_is_refund_return_policy_howto(
                comb
            ):
                prior_oid = extract_order_id(comb, conversation_context) or (
                    extract_latest_order_id_from_user_conversation(
                        conversation_context, comb
                    )
                )
                if prior_oid:
                    out = ResolvedRefundTurn(
                        kind=KIND_PERSONAL_STATUS,
                        source="refund_correction_with_prior_id",
                        confidence="high",
                        user_meaning="User insists on refund status for order id already shared",
                    )
                    _RESOLVE_REFUND_CACHE.key = (
                        f"{hash(comb)}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
                    )
                    _RESOLVE_REFUND_CACHE.result = out
                    return out
        except ImportError:
            pass
        return ResolvedRefundTurn(kind=KIND_NONE)

    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
    if getattr(_RESOLVE_REFUND_CACHE, "key", None) == cache_key:
        cached = getattr(_RESOLVE_REFUND_CACHE, "result", None)
        if isinstance(cached, ResolvedRefundTurn):
            return cached

    none = ResolvedRefundTurn(kind=KIND_NONE)

    if allow_llm:
        classified = ai_classify_refund_status_turn(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if classified:
            ck = (classified.get("refund_turn_kind") or KIND_NONE).strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            um = str(classified.get("user_meaning") or "").strip()
            if ck == KIND_PERSONAL_STATUS and conf >= 0.5:
                out = ResolvedRefundTurn(
                    kind=KIND_PERSONAL_STATUS,
                    source="refund_turn_llm",
                    confidence="high" if conf >= 0.7 else "medium",
                    user_meaning=um,
                )
                _RESOLVE_REFUND_CACHE.key = cache_key
                _RESOLVE_REFUND_CACHE.result = out
                return out
            if ck == KIND_POLICY_HOWTO and conf >= 0.5:
                out = ResolvedRefundTurn(
                    kind=KIND_POLICY_HOWTO,
                    source="refund_turn_llm",
                    confidence="high" if conf >= 0.7 else "medium",
                    user_meaning=um,
                )
                _RESOLVE_REFUND_CACHE.key = cache_key
                _RESOLVE_REFUND_CACHE.result = out
                return out

    if isinstance(ai_route, dict):
        try:
            from services.ai_route_semantics import ai_route_is_kb_read

            kb_read = ai_route_is_kb_read(ai_route)
        except ImportError:
            kb_read = False
        from utils.helpers import (
            _text_is_refund_return_policy_howto,
            _text_is_refund_return_status_lookup,
        )

        if _text_is_refund_return_status_lookup(comb, conversation_context):
            out = ResolvedRefundTurn(
                kind=KIND_PERSONAL_STATUS, source="message_status_semantics"
            )
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        if _text_is_refund_return_policy_howto(comb):
            out = ResolvedRefundTurn(kind=KIND_POLICY_HOWTO, source="message_policy_semantics")
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        blob_kind = _kind_from_meaning_blob(_meaning_blob(ai_route))
        if blob_kind == KIND_PERSONAL_STATUS:
            out = ResolvedRefundTurn(kind=KIND_PERSONAL_STATUS, source="ai_route_meaning")
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        if blob_kind == KIND_POLICY_HOWTO or (kb_read and (ai_route.get("intent") or "") in ("refund", "general")):
            out = ResolvedRefundTurn(kind=KIND_POLICY_HOWTO, source="ai_route_kb")
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        intent = (ai_route.get("intent") or "").strip().lower()
        olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
        rh = (ai_route.get("route_handler") or "").strip().lower()
        channel = (ai_route.get("data_channel") or "").strip().lower()
        if olk == "refund_status" or rh == "refund_status_api" or intent == "refund_status":
            out = ResolvedRefundTurn(kind=KIND_PERSONAL_STATUS, source="ai_route_field")
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out
        if intent == "refund" and channel == "live_api" and not kb_read:
            out = ResolvedRefundTurn(kind=KIND_PERSONAL_STATUS, source="ai_route_live_api")
            _RESOLVE_REFUND_CACHE.key = cache_key
            _RESOLVE_REFUND_CACHE.result = out
            return out

    from utils.helpers import (
        _text_is_refund_return_policy_howto,
        _text_is_refund_return_status_lookup,
    )

    if _text_is_refund_return_policy_howto(comb):
        out = ResolvedRefundTurn(kind=KIND_POLICY_HOWTO, source="keyword_policy")
        _RESOLVE_REFUND_CACHE.key = cache_key
        _RESOLVE_REFUND_CACHE.result = out
        return out
    if _text_is_refund_return_status_lookup(comb, conversation_context):
        out = ResolvedRefundTurn(kind=KIND_PERSONAL_STATUS, source="keyword_status")
        _RESOLVE_REFUND_CACHE.key = cache_key
        _RESOLVE_REFUND_CACHE.result = out
        return out

    _RESOLVE_REFUND_CACHE.key = cache_key
    _RESOLVE_REFUND_CACHE.result = none
    return none


def current_turn_wants_personal_refund_status(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    """Latest user message wants live refund/return STATUS on one order."""
    cache_key = f"{hash(_combined(original_msg, msg_en))}|{hash((conversation_context or '')[-400:])}|{allow_llm}"
    if getattr(_REFUND_TURN_CACHE, "key", None) == cache_key:
        return bool(getattr(_REFUND_TURN_CACHE, "result", False))

    resolved = resolve_refund_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    result = resolved.kind == KIND_PERSONAL_STATUS
    _REFUND_TURN_CACHE.key = cache_key
    _REFUND_TURN_CACHE.result = result
    return result


def current_turn_wants_refund_policy_kb(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    """Latest user message wants refund/return policy or how-to (KB, not live API)."""
    resolved = resolve_refund_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    return resolved.kind == KIND_POLICY_HOWTO


def ai_route_requests_refund_status_lookup(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Personal refund/return status for one order — Groq JSON + meaning (any language)."""
    from utils.helpers import _text_is_refund_return_policy_howto

    comb = _combined(original_msg, msg_en)
    if _text_is_refund_return_policy_howto(comb):
        return False

    if current_turn_wants_personal_refund_status(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route,
        allow_llm=False,
    ):
        return True

    try:
        from services.order_tracking_semantics import ai_route_requests_order_tracking_lookup

        if ai_route_requests_order_tracking_lookup(
            route, original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass

    from services.semantic_intent import llm_semantic_route_available

    if llm_semantic_route_available(route):
        r = route or {}
        olk = (r.get("order_lookup_kind") or "").strip().lower()
        intent = (r.get("intent") or "").strip().lower()
        rh = (r.get("route_handler") or "").strip().lower()
        if olk == "refund_status" or rh == "refund_status_api" or intent == "refund_status":
            return True
        if _kind_from_meaning_blob(_meaning_blob(r)) == KIND_PERSONAL_STATUS:
            return True
        channel = (r.get("data_channel") or "").strip().lower()
        from services.ai_route_semantics import ai_route_is_kb_read

        if intent == "refund" and channel == "live_api" and not ai_route_is_kb_read(r):
            return True
        if intent == "refund" and r.get("needs_order_id") and not ai_route_is_kb_read(r):
            return True
        return False

    from utils.helpers import _text_is_refund_return_status_lookup

    return _text_is_refund_return_status_lookup(comb, conversation_context)


def _apply_refund_status_to_route(out: dict, source: str) -> dict:
    out["intent"] = "refund"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["order_lookup_kind"] = "refund_status"
    out["route_handler"] = "refund_status_api"
    out["answer_strategy"] = "api_then_ai"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out.pop("kb_keys", None)
    log_reasoning(f"Refund-status ({source}): personal lookup → return-request API.")
    return out


def _apply_refund_policy_to_route(out: dict, source: str) -> dict:
    out["intent"] = "refund"
    out["data_channel"] = "kb"
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out["route_handler"] = "policy_structured_kb"
    out["answer_strategy"] = "kb_then_ai"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    keys = list(out.get("kb_keys") or [])
    for k in ("refund", "faqs", "shipping"):
        if k not in keys:
            keys.append(k)
    out["kb_keys"] = keys[:4]
    log_reasoning(f"Refund-policy ({source}): how-to / rules → KB (not live API).")
    return out


def ai_classify_refund_status_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[dict]:
    """Micro-classifier: personal refund STATUS vs policy HOW-TO (any language)."""
    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if should_skip_micro_classifier_llm():
            log_reasoning(
                "Refund-status LLM: defer/skip — universal brain route owns classification."
            )
            return None
    except ImportError:
        pass

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None
    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1600)
    user_line = _trim_text_mid(comb, 520)

    system_prompt = f"""You classify Welfog refund/return questions for the LATEST user message.

Customers write in ANY language, script, or style (English, Hinglish, Hindi, Tamil, Telugu, Bengali,
Marathi, Gujarati, Kannada, Malayalam, Urdu, typos, voice-to-text). Infer MEANING — never match keywords.

Two different intents:
1) personal_status — customer wants THEIR refund/return status on a specific order they placed
   (approved yet?, money not received, check my refund, when will MY refund arrive for MY order).
   If Order ID is missing, still personal_status — bot will ask for it.
2) policy_howto — general rules/steps/timeline (how to return, refund policy, eligibility,
   how many days refunds usually take IN GENERAL, wrong/damaged item process).
   No live lookup for one order.

Critical distinction:
- General WHEN/HOW LONG questions with no specific-order complaint and no Order ID
  ("refund kitne din me aata hai", "mera refund kab tk aa jayega" as a general timeline question)
  → policy_howto
- Personal complaint or status check on THEIR purchase, even without Order ID yet
  ("refund nahi aaya", "approve hua kya", "paise wapas nahi mile", "check my refund")
  → personal_status (bot will ask Order ID if needed)

Examples (same meaning → same refund_turn_kind):
- "2606020 ka refund kab milega" → personal_status
- "என் ரீபண்ட் எப்போ வரும்" → personal_status (Tamil: when will my refund come)
- "refund এখনও আসেনি" → personal_status (Bengali: refund not come yet)
- "refund kitne din me aata hai" → policy_howto (general timeline)
- "mera refund kab tk aa jayega" (general when, no complaint) → policy_howto
- "return kaise kru welfog pe" → policy_howto
- "refund nhi aaya abhi tk" → personal_status
- "mera refund approve hua kya" (no order id) → personal_status

NOT refund/return (use refund_turn_kind=none): product search, shipment tracking, invoice,
greeting, general DELIVERY TIME / shipping timeline / how long orders take to arrive.
- "welfog pe delivery kitna time lgta hai" → none
- "order kitne din me aata hai" → none
- "refund kitne din me aata hai" → policy_howto

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence",
  "refund_turn_kind": "none" | "personal_status" | "policy_howto",
  "confidence": 0.0 to 1.0
}}

{language_reply_instruction(rl)}"""

    user_payload = f"LATEST USER MESSAGE:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=220,
        timeout_sec=18,
        max_attempts=2,
    )
    if not data:
        return None
    kind = (data.get("refund_turn_kind") or KIND_NONE).strip().lower()
    if kind not in (KIND_PERSONAL_STATUS, KIND_POLICY_HOWTO, KIND_NONE):
        kind = KIND_NONE
    conf = float(data.get("confidence") or 0.0)
    um = (data.get("user_meaning") or "").strip()
    log_reasoning(f"Refund-status LLM: kind={kind} conf={conf:.2f} — {um[:90] or 'no meaning'}")
    return {"refund_turn_kind": kind, "user_meaning": um, "confidence": conf}


def promote_refund_status_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    allow_llm: bool = False,
) -> dict:
    """Align Groq route: personal refund STATUS (API) vs refund POLICY (KB)."""
    out = dict(route or {})
    comb = _combined(original_msg, msg_en)

    try:
        from services.conversation_thread_semantics import (
            infer_order_thread_goal,
            message_needs_thread_continuation,
            resolve_explicit_turn_goal_from_message,
        )

        explicit = resolve_explicit_turn_goal_from_message(
            original_msg, msg_en, conversation_context, out, allow_llm=False
        )
        if explicit and explicit != "refund_status":
            return out
        if message_needs_thread_continuation(comb, conversation_context):
            thread = infer_order_thread_goal(
                conversation_context,
                comb,
                ai_route=out,
                reply_lang=reply_lang,
                allow_llm=False,
            )
            if thread == "refund_status":
                return _apply_refund_status_to_route(out, "thread_continuation")
    except ImportError:
        pass

    if not _message_has_refund_topic(comb):
        return out

    try:
        from utils.helpers import turn_skips_order_micro_classifiers

        if turn_skips_order_micro_classifiers(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            return out
    except ImportError:
        pass

    resolved = resolve_refund_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=out,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    if resolved.user_meaning:
        out["user_meaning"] = resolved.user_meaning

    if resolved.kind == KIND_POLICY_HOWTO:
        return _apply_refund_policy_to_route(out, resolved.source or "refund_resolver")

    if resolved.kind != KIND_PERSONAL_STATUS:
        return out

    try:
        from services.order_tracking_semantics import (
            ai_route_requests_order_tracking_lookup,
            order_tracking_route_is_locked,
        )

        if order_tracking_route_is_locked(out):
            return out
        if ai_route_requests_order_tracking_lookup(
            out, original_msg, msg_en, conversation_context
        ):
            return out
    except ImportError:
        pass

    return _apply_refund_status_to_route(out, resolved.source or "current_turn_refund")


def refund_status_route_is_locked(route: dict | None) -> bool:
    if not route:
        return False
    olk = (route.get("order_lookup_kind") or "").strip().lower()
    if olk == "refund_status":
        return True
    return (route.get("route_handler") or "").strip().lower() == "refund_status_api"
