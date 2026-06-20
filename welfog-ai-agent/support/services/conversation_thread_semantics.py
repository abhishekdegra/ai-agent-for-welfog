"""
Ambiguous bare-ID / numeric handoff: infer active thread from recent chat (1–2 turns).

Industry-standard layering (only for short / id-only follow-ups):
  1. Brain router + specialist LLMs on self-contained messages (any language) — primary.
  2. Structural signals (bot asked for Order ID/PIN + user sent digits) — fast, no keywords.
  3. Heuristic scores on recent User/Assistant lines — LLM-down failsafe.
  4. Micro-LLM thread classifier — when heuristics are weak or tied (any language).

Normal full-sentence queries never enter this module.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Optional

from utils.reasoning_log import log_reasoning

_THREAD_CACHE = threading.local()
_EXPLICIT_GOAL_GUARD = threading.local()

_THREAD_GOALS = frozenset(
    (
        "refund_status",
        "track",
        "order_details",
        "order_invoice",
        "payment",
        "pincode",
        "product_id",
    )
)

# LLM-unavailable failsafe only — not the primary router.
_REFUND_MARKERS = (
    "refund",
    "return status",
    "refund status",
    "money back",
    "paise wapas",
    "paise wapsi",
    "paise nahi",
    "paise nhi",
    "refund nahi",
    "refund nhi",
    "return request",
    "wapas",
)

_TRACK_MARKERS = (
    "track",
    "tracking",
    "trck",
    "trak",
    "live status",
    "live tracking",
    "kab aayega",
    "kab aaega",
    "kab milega",
    "kahan hai",
    "kaha hai",
    "shipment",
    "courier",
    "parcel",
    "delivery status",
)

_INVOICE_MARKERS = (
    "invoice",
    "bill",
    "receipt",
    "gst",
    "tax invoice",
    "चालान",
    "बिल",
)

_DETAILS_MARKERS = (
    "address",
    "amount",
    "payment status",
    "payment mode",
    "grand total",
    "kitna",
    "order detail",
    "order details",
    "summary",
    "konsa laga",
    "lagaya th",
)


def _score_markers(text: str, markers: tuple[str, ...]) -> int:
    tl = f" {(text or '').lower()} "
    return sum(1 for m in markers if m in tl)


def _bot_awaiting_identifier(conversation_context: str) -> bool:
    try:
        from utils.helpers import (
            _conversation_awaiting_order_id,
            _conversation_bot_asked_for_pincode,
        )

        return bool(
            _conversation_awaiting_order_id(conversation_context)
            or _conversation_bot_asked_for_pincode(conversation_context)
        )
    except ImportError:
        return False


def resolve_explicit_turn_goal_from_message(
    current_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    *,
    allow_llm: bool = True,
    reply_lang: str = "",
) -> str:
    """
    What the LATEST user message asks for (any language). Latest turn wins over stale thread.
    Returns '' for bare id handoffs only.
    """
    comb = f"{current_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""
    if re.fullmatch(r"[0-9]{4,20}", comb) or re.fullmatch(r"[A-Za-z0-9]{4,20}", comb):
        return ""

    try:
        from utils.helpers import (
            _text_has_pincode_delivery_intent,
            _text_is_pincode_serviceability_question_light,
            message_has_live_pincode_check_intent,
        )

        if message_has_live_pincode_check_intent(
            current_msg, conversation_context, msg_en
        ) or _text_is_pincode_serviceability_question_light(comb) or _text_has_pincode_delivery_intent(
            comb, conversation_context
        ):
            return ""
    except ImportError:
        pass

    try:
        from utils.helpers import (
            _text_asks_order_history,
            _text_asks_wishlist,
            _text_wants_order_history_list_in_chat,
            message_is_wishlist_like_request,
        )

        if _text_asks_order_history(comb) or _text_wants_order_history_list_in_chat(
            comb, conversation_context
        ):
            return ""
        if _text_asks_wishlist(comb) or message_is_wishlist_like_request(comb):
            return ""
    except ImportError:
        pass

    if getattr(_EXPLICIT_GOAL_GUARD, "active", False):
        return ""

    _EXPLICIT_GOAL_GUARD.active = True
    try:
        route_goal = _thread_goal_from_ai_route_meaning(ai_route)
        if route_goal:
            return route_goal

        try:
            from services.order_details_flow import _lightweight_details_or_invoice_signal

            light = _lightweight_details_or_invoice_signal(comb)
            if light == "order_invoice":
                return "order_invoice"
            if light == "order_details":
                return "order_details"
        except ImportError:
            pass

        try:
            from utils.helpers import _text_is_refund_return_policy_howto

            if _text_is_refund_return_policy_howto(comb):
                return ""
        except ImportError:
            pass

        try:
            from services.refund_status_semantics import (
                KIND_PERSONAL_STATUS,
                resolve_refund_turn,
            )

            resolved = resolve_refund_turn(
                current_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                reply_lang=reply_lang,
                allow_llm=allow_llm,
            )
            if resolved.kind == KIND_PERSONAL_STATUS:
                return "refund_status"
        except ImportError:
            pass

        try:
            from services.order_details_flow import message_wants_order_details_or_invoice

            od = message_wants_order_details_or_invoice(
                current_msg, msg_en, conversation_context
            )
            if od in ("order_details", "order_invoice"):
                return od
        except ImportError:
            pass

        try:
            from utils.helpers import _text_is_order_tracking_intent_leaf

            if _text_is_order_tracking_intent_leaf(comb):
                return "track"
        except ImportError:
            pass
    finally:
        _EXPLICIT_GOAL_GUARD.active = False

    if allow_llm and len(re.findall(r"\S+", (current_msg or msg_en or "").strip())) >= 4:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                return ""
        except ImportError:
            pass
        classified = ai_classify_conversation_thread(
            comb, conversation_context, reply_lang=reply_lang
        )
        if classified:
            ck = (classified.get("thread_goal") or "none").strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if ck in _THREAD_GOALS and conf >= 0.65:
                return ck
    return ""


def message_needs_thread_continuation(
    current_msg: str,
    conversation_context: str = "",
) -> bool:
    """
    True when this turn is mostly handing over an id/pin — not a fresh topic.
    Structural (bot asked + digits / short reply), not phrase lists.
    """
    if resolve_explicit_turn_goal_from_message(
        current_msg, "", conversation_context, None, allow_llm=False
    ):
        return False

    try:
        from utils.helpers import (
            _message_is_order_id_followup_submission,
            _user_announcing_will_provide_order_id,
        )

        if _user_announcing_will_provide_order_id(current_msg):
            return False
        if _message_is_order_id_followup_submission(current_msg, conversation_context):
            if not resolve_explicit_turn_goal_from_message(
                current_msg, "", conversation_context, None, allow_llm=False
            ):
                return True
    except ImportError:
        pass

    raw = (current_msg or "").strip()
    if not raw:
        return False

    if re.fullmatch(r"[0-9]{4,20}", raw) or re.fullmatch(r"[A-Za-z0-9]{4,20}", raw):
        return True

    has_id_digits = bool(re.search(r"\b[0-9]{4,20}\b", raw))
    word_count = len(re.findall(r"\S+", raw))

    if _bot_awaiting_identifier(conversation_context):
        if has_id_digits and word_count <= 4:
            return True

    return False


def _last_assistant_line(conversation_context: str) -> str:
    for line in reversed((conversation_context or "").splitlines()):
        low = line.strip().lower()
        if low.startswith("assistant:") or low.startswith("assistant "):
            return line.split(":", 1)[-1].strip()
    return ""


def _recent_user_lines(conversation_context: str, max_lines: int = 3) -> list[str]:
    lines: list[str] = []
    for line in (conversation_context or "").splitlines():
        low = line.strip().lower()
        if low.startswith("user:") or low.startswith("user "):
            lines.append(line.split(":", 1)[-1].strip())
    return lines[-max_lines:]


def _thread_goal_from_ai_route_meaning(ai_route: dict | None) -> str:
    if not isinstance(ai_route, dict):
        return ""
    blob = f" {(ai_route.get('user_meaning') or '').lower()} {(ai_route.get('reasoning') or '').lower()} "
    olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
    if olk == "refund_status":
        return "refund_status"
    if olk in ("invoice", "order_invoice"):
        return "order_invoice"
    if olk in ("details", "order_details"):
        return "order_details"
    if olk in ("track", "tracking"):
        return "track"
    intent = (ai_route.get("intent") or "").strip().lower()
    if intent == "refund":
        return "refund_status"
    if intent == "pincode_check":
        return "pincode"
    if intent == "product":
        return "product_id"
    if any(x in blob for x in ("refund status", "return status", "money back", "refund for")):
        return "refund_status"
    if any(x in blob for x in ("invoice", "bill", "receipt", "gst")):
        return "order_invoice"
    if any(x in blob for x in ("address", "amount", "payment mode", "order detail")):
        return "order_details"
    if any(x in blob for x in ("track", "shipment", "courier", "where is", "eta")):
        return "track"
    return ""


def _heuristic_thread_scores(
    conversation_context: str,
    current_msg: str = "",
    *,
    ctx_last: str | None = None,
) -> dict[str, int]:
    scores: dict[str, int] = {g: 0 for g in _THREAD_GOALS}

    tl_cur = f" {(current_msg or '').lower()} "
    if "refund" in tl_cur or "paise wapas" in tl_cur or "money back" in tl_cur:
        scores["refund_status"] += 5
    if _score_markers(current_msg, _INVOICE_MARKERS):
        scores["order_invoice"] += 4
    if _score_markers(current_msg, _DETAILS_MARKERS) and not _score_markers(
        current_msg, _TRACK_MARKERS
    ):
        scores["order_details"] += 4
    if _score_markers(current_msg, _TRACK_MARKERS):
        scores["track"] += 4

    last_asst = _last_assistant_line(conversation_context)
    if last_asst:
        la = last_asst.lower()
        if "order id" in la or "orderid" in la:
            scores["refund_status"] += _score_markers(la, _REFUND_MARKERS) * 2 + (
                3 if "refund" in la and "status" in la else 0
            )
            scores["order_invoice"] += _score_markers(la, _INVOICE_MARKERS) * 2
            scores["order_details"] += _score_markers(la, _DETAILS_MARKERS) * 2
            scores["track"] += _score_markers(la, _TRACK_MARKERS) * 2 + (
                2 if "track" in la or "live status" in la else 0
            )
        if "pin" in la and any(x in la for x in ("bhej", "send", "paste", "enter", "6-digit", "6 digit")):
            scores["pincode"] += 4

    for body in reversed(_recent_user_lines(conversation_context, 3)):
        scores["refund_status"] += _score_markers(body, _REFUND_MARKERS) * 2
        scores["track"] += _score_markers(body, _TRACK_MARKERS) * 2
        scores["order_invoice"] += _score_markers(body, _INVOICE_MARKERS) * 2
        scores["order_details"] += _score_markers(body, _DETAILS_MARKERS) * 2

    cl = (ctx_last or "").strip().lower()
    if cl == "refund":
        scores["refund_status"] += 3
    elif cl == "payment":
        scores["payment"] += 3
    elif cl == "order":
        scores["track"] += 1

    return scores


def _pick_goal_from_scores(scores: dict[str, int]) -> tuple[str, int, bool]:
    """Return (goal, best_score, ambiguous) — ambiguous when top-2 within 1 point."""
    ranked = sorted(
        ((g, s) for g, s in scores.items() if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    if not ranked:
        return "", 0, True
    best_goal, best_score = ranked[0]
    ambiguous = len(ranked) > 1 and ranked[1][1] >= best_score - 1
    if scores.get("refund_status", 0) >= scores.get("track", 0) and scores["refund_status"] >= 2:
        if scores["refund_status"] >= best_score - 1:
            return "refund_status", scores["refund_status"], ambiguous
    return best_goal, best_score, ambiguous


def ai_classify_conversation_thread(
    current_msg: str,
    conversation_context: str = "",
    *,
    reply_lang: str = "",
) -> Optional[dict[str, Any]]:
    """
    Micro-LLM: what thread is the user continuing (any language)?
    Only for ambiguous id handoffs — not general routing.
    """
    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if should_skip_micro_classifier_llm():
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

    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    rl = resolve_customer_reply_lang(current_msg, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1800)
    user_line = _trim_text_mid((current_msg or "").strip(), 400)

    system_prompt = f"""You classify Welfog support THREAD CONTINUATION when the customer's LATEST message
is ONLY handing over an ID/PIN or a very short reply — not a new full question.

Customers write in ANY language, script, or style (English, Hinglish, Hindi, Tamil, Telugu, Bengali,
Marathi, Gujarati, Kannada, Malayalam, Urdu, typos, voice-to-text). Infer MEANING — never match keywords.

Read RECENT CONVERSATION: what did the bot ask for, and what was the user trying to do before?

thread_goal (pick ONE):
- refund_status — continue checking THEIR refund/return status on one order
- track — continue live shipment / order timeline for one order
- order_invoice — continue bill/invoice/receipt for one order
- order_details — continue address, amount, payment, product summary for one order
- payment — continue payment status for one order
- pincode — user is answering bot's PIN / delivery-area question (6-digit area code)
- product_id — user is giving a product/SKU id for catalog lookup (rare)
- none — new unrelated topic or message is already self-explanatory

Examples (same meaning → same thread_goal):
- Bot asked Order ID for refund; user sends "2806010" or "yeh lo" or "இதோ ஆர்டர் ஐடி" → refund_status
- Bot asked Order ID to track; user sends bare digits → track
- Bot asked PIN; user sends "302034" → pincode
- User starts a new product search in latest message → none

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence: what they are continuing",
  "thread_goal": "refund_status" | "track" | "order_invoice" | "order_details" | "payment" | "pincode" | "product_id" | "none",
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
        max_tokens=200,
        timeout_sec=16,
        max_attempts=2,
    )
    if not data:
        return None
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), None)
        if not data:
            return None
    if not isinstance(data, dict):
        return None
    goal = (data.get("thread_goal") or "none").strip().lower()
    if goal not in _THREAD_GOALS and goal != "none":
        goal = "none"
    conf = float(data.get("confidence") or 0.0)
    um = (data.get("user_meaning") or "").strip()
    log_reasoning(
        f"Thread-continuation LLM: goal={goal} conf={conf:.2f} — {um[:90] or 'no meaning'}"
    )
    return {"thread_goal": goal, "user_meaning": um, "confidence": conf}


def infer_order_thread_goal(
    conversation_context: str,
    current_msg: str = "",
    *,
    ctx_last: str | None = None,
    ai_route: dict | None = None,
    reply_lang: str = "",
    allow_llm: bool = True,
) -> str:
    """
    Active single-order thread for ambiguous id handoffs.
    Returns '' when not applicable or no confident thread.
    """
    explicit = resolve_explicit_turn_goal_from_message(
        current_msg,
        "",
        conversation_context,
        ai_route,
        allow_llm=allow_llm,
        reply_lang=reply_lang,
    )
    if explicit:
        return explicit

    if not message_needs_thread_continuation(current_msg, conversation_context):
        return ""

    cache_key = (
        f"{hash((current_msg or '').strip())}|{hash((conversation_context or '')[-500:])}|"
        f"{ctx_last}|{allow_llm}"
    )
    if getattr(_THREAD_CACHE, "key", None) == cache_key:
        return str(getattr(_THREAD_CACHE, "result", "") or "")

    try:
        from utils.helpers import _conversation_bot_asked_for_pincode

        if _conversation_bot_asked_for_pincode(conversation_context):
            _THREAD_CACHE.key = cache_key
            _THREAD_CACHE.result = "pincode"
            return "pincode"
    except ImportError:
        pass

    route_goal = _thread_goal_from_ai_route_meaning(ai_route)
    if route_goal:
        _THREAD_CACHE.key = cache_key
        _THREAD_CACHE.result = route_goal
        return route_goal

    scores = _heuristic_thread_scores(
        conversation_context, current_msg, ctx_last=ctx_last
    )
    goal, best_score, ambiguous = _pick_goal_from_scores(scores)

    need_llm = allow_llm and (best_score < 3 or ambiguous or not goal)
    if need_llm:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                need_llm = False
        except ImportError:
            pass
    if need_llm:
        classified = ai_classify_conversation_thread(
            current_msg, conversation_context, reply_lang=reply_lang
        )
        if classified:
            ck = (classified.get("thread_goal") or "none").strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if ck in _THREAD_GOALS and conf >= 0.5:
                _THREAD_CACHE.key = cache_key
                _THREAD_CACHE.result = ck
                return ck

    if best_score < 2 or not goal:
        _THREAD_CACHE.key = cache_key
        _THREAD_CACHE.result = ""
        return ""

    _THREAD_CACHE.key = cache_key
    _THREAD_CACHE.result = goal
    return goal


def apply_thread_goal_to_route(route: dict | None, goal: str, source: str = "thread") -> dict:
    """Lock route to the inferred conversation thread."""
    out = dict(route or {})
    g = (goal or "").strip().lower()
    if g == "refund_status":
        from services.refund_status_semantics import _apply_refund_status_to_route

        return _apply_refund_status_to_route(out, source)
    if g == "order_invoice":
        from services.order_details_flow import _apply_order_details_to_route

        return _apply_order_details_to_route(out, "order_invoice", source)
    if g == "order_details":
        from services.order_details_flow import _apply_order_details_to_route

        return _apply_order_details_to_route(out, "order_details", source)
    if g == "track":
        from services.order_tracking_semantics import _apply_order_tracking_to_route

        return _apply_order_tracking_to_route(out, source)
    if g == "payment":
        out["intent"] = "payment"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["order_lookup_kind"] = "none"
        return out
    return out
