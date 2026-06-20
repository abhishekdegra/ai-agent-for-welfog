"""
Chat resilience — deadlines, rate limits, and polite customer-facing fallbacks.

Customers should never wait indefinitely; when LLMs/APIs are down or rate-limited,
return a short message in their language instead of hanging or a generic 500.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Optional

from utils.reasoning_log import chat_log, log_reasoning

CHAT_MAX_SECONDS = float(os.getenv("CHAT_MAX_SECONDS") or "70")
CHAT_IN_FLIGHT_STALE_SEC = float(
    os.getenv("CHAT_IN_FLIGHT_STALE_SEC")
    or str(max(45.0, CHAT_MAX_SECONDS + 10.0))
)
CHAT_IN_FLIGHT_MAX_SEC = float(
    os.getenv("CHAT_IN_FLIGHT_MAX_SEC")
    or str(max(CHAT_IN_FLIGHT_STALE_SEC, CHAT_MAX_SECONDS + 15.0))
)
_LLM_FAILURE = threading.local()
_CHAT_LOCKS: dict[str, threading.Lock] = {}
_CHAT_LOCKS_GUARD = threading.Lock()
_CHAT_IN_FLIGHT: set[str] = set()
_CHAT_IN_FLIGHT_AT: dict[str, float] = {}


def _chat_turn_lock(chat_id: str) -> threading.Lock:
    key = (chat_id or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key not in _CHAT_LOCKS:
            _CHAT_LOCKS[key] = threading.Lock()
        return _CHAT_LOCKS[key]


_USER_IN_FLIGHT: set[str] = set()
_USER_IN_FLIGHT_AT: dict[str, float] = {}
_TURN_ACQUIRED = threading.local()


def _in_flight_age(key: str, bucket: set[str], times: dict[str, float]) -> float:
    if not key or key not in bucket:
        return 0.0
    return max(0.0, time.monotonic() - float(times.get(key, 0.0) or 0.0))


def _force_clear_in_flight(
    key: str,
    bucket: set[str],
    times: dict[str, float],
    *,
    reason: str,
) -> bool:
    """Hard-unlock chat/user — customer must never stay blocked after deadline."""
    if not key or key not in bucket:
        return False
    age = _in_flight_age(key, bucket, times)
    bucket.discard(key)
    times.pop(key, None)
    log_reasoning(f"In-flight force-cleared ({age:.1f}s) — {reason}")
    return True


def _clear_stale_in_flight(key: str, bucket: set[str], times: dict[str, float]) -> bool:
    """Drop in-flight marker when the prior turn exceeded stale threshold."""
    if not key or key not in bucket:
        return False
    age = _in_flight_age(key, bucket, times)
    if age <= CHAT_IN_FLIGHT_STALE_SEC:
        return False
    return _force_clear_in_flight(
        key, bucket, times, reason="stale threshold — new turn allowed"
    )


def touch_chat_turn_in_flight() -> None:
    """Refresh in-flight timestamp during LLM/API — never extend past hard max."""
    if not getattr(_TURN_ACQUIRED, "acquired", False):
        return
    uid = (getattr(_TURN_ACQUIRED, "user_id", "") or "").strip()
    key = (getattr(_TURN_ACQUIRED, "chat_id", "") or "").strip()
    if not uid and not key:
        return
    now = time.monotonic()
    with _CHAT_LOCKS_GUARD:
        if key and key in _CHAT_IN_FLIGHT:
            age = _in_flight_age(key, _CHAT_IN_FLIGHT, _CHAT_IN_FLIGHT_AT)
            if age >= CHAT_IN_FLIGHT_MAX_SEC:
                return
            _CHAT_IN_FLIGHT_AT[key] = now
        if uid and uid in _USER_IN_FLIGHT:
            age = _in_flight_age(uid, _USER_IN_FLIGHT, _USER_IN_FLIGHT_AT)
            if age >= CHAT_IN_FLIGHT_MAX_SEC:
                return
            _USER_IN_FLIGHT_AT[uid] = now


def force_end_stuck_chat_turn(chat_id: str = "", user_id: str = "") -> None:
    """Release locks after deadline/timeout so customer can send a fresh question."""
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key:
            _force_clear_in_flight(
                key,
                _CHAT_IN_FLIGHT,
                _CHAT_IN_FLIGHT_AT,
                reason="handler ended or deadline",
            )
        if uid:
            _force_clear_in_flight(
                uid,
                _USER_IN_FLIGHT,
                _USER_IN_FLIGHT_AT,
                reason="handler ended or deadline",
            )


def try_begin_chat_turn(chat_id: str, user_id: str = "") -> bool:
    """One in-flight /chat per chat_id and per user_id — avoids parallel LLM pile-up."""
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key:
            if _in_flight_age(key, _CHAT_IN_FLIGHT, _CHAT_IN_FLIGHT_AT) >= CHAT_IN_FLIGHT_MAX_SEC:
                _force_clear_in_flight(
                    key,
                    _CHAT_IN_FLIGHT,
                    _CHAT_IN_FLIGHT_AT,
                    reason="hard cap — customer may retry",
                )
            else:
                _clear_stale_in_flight(key, _CHAT_IN_FLIGHT, _CHAT_IN_FLIGHT_AT)
        if uid:
            if _in_flight_age(uid, _USER_IN_FLIGHT, _USER_IN_FLIGHT_AT) >= CHAT_IN_FLIGHT_MAX_SEC:
                _force_clear_in_flight(
                    uid,
                    _USER_IN_FLIGHT,
                    _USER_IN_FLIGHT_AT,
                    reason="hard cap — customer may retry",
                )
            else:
                _clear_stale_in_flight(uid, _USER_IN_FLIGHT, _USER_IN_FLIGHT_AT)
        if key and key in _CHAT_IN_FLIGHT:
            return False
        # Same user in two tabs — serialize only while a turn is actively within hard cap.
        if uid and uid in _USER_IN_FLIGHT:
            return False
        now = time.monotonic()
        if key:
            _CHAT_IN_FLIGHT.add(key)
            _CHAT_IN_FLIGHT_AT[key] = now
        if uid:
            _USER_IN_FLIGHT.add(uid)
            _USER_IN_FLIGHT_AT[uid] = now
        _TURN_ACQUIRED.acquired = True
        _TURN_ACQUIRED.chat_id = key
        _TURN_ACQUIRED.user_id = uid
        return True


def end_chat_turn(chat_id: str, user_id: str = "") -> None:
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key:
            _CHAT_IN_FLIGHT.discard(key)
            _CHAT_IN_FLIGHT_AT.pop(key, None)
        if uid:
            _USER_IN_FLIGHT.discard(uid)
            _USER_IN_FLIGHT_AT.pop(uid, None)


def end_chat_turn_if_acquired() -> None:
    """Release in-flight lock only for the request that acquired it (never steal another turn)."""
    if not getattr(_TURN_ACQUIRED, "acquired", False):
        return
    end_chat_turn(
        getattr(_TURN_ACQUIRED, "chat_id", "") or "",
        getattr(_TURN_ACQUIRED, "user_id", "") or "",
    )
    _TURN_ACQUIRED.acquired = False


def clear_turn_acquire_state() -> None:
    """Reset per-request acquire flag (call at start of each /chat HTTP handler)."""
    _TURN_ACQUIRED.acquired = False
    _TURN_ACQUIRED.chat_id = ""
    _TURN_ACQUIRED.user_id = ""


class ChatDeadlineExceeded(Exception):
    """Whole /chat handler exceeded CHAT_MAX_SECONDS."""


class LLMProvidersBusy(Exception):
    """All LLM providers failed (rate limit, timeout, or unavailable)."""


def set_last_llm_failure(kind: str) -> None:
    _LLM_FAILURE.kind = (kind or "").strip().lower()


def get_last_llm_failure() -> str:
    return getattr(_LLM_FAILURE, "kind", "") or ""


def clear_last_llm_failure() -> None:
    _LLM_FAILURE.kind = ""


def classify_api_error(status_code: int, body: str) -> str:
    low = (body or "").lower()
    if status_code == 429 or "rate_limit" in low or "rate limit" in low:
        if any(
            x in low
            for x in (
                "tokens per day",
                "tpd",
                "token limit",
                "quota",
                "insufficient",
                "capacity",
                "overloaded",
            )
        ):
            return "rate_limit"
        return "rate_limit"
    if status_code in (503, 502, 529) or "overloaded" in low or "high traffic" in low:
        return "busy"
    if status_code == 413 or "request too large" in low or "context length" in low:
        return "payload_too_large"
    return "error"


def build_in_flight_reply_html(original_msg: str = "", reply_lang: str = "") -> str:
    """Previous message still processing — ask to wait or try again shortly."""
    from services.translation_service import customer_facing_template

    user_msg = (original_msg or "").strip() or "wait"
    body = customer_facing_template(
        "chat_in_flight",
        user_msg,
        reply_lang,
        wrap_html=True,
        fallback_en=(
            "Your last message is still being processed. Please wait a few seconds. "
            "If nothing appears, try again in a moment — you can also send a new question "
            "after ~20 seconds."
        ),
    )
    if body and body.strip():
        return body
    return (
        '<div style="color:#333;line-height:1.55;">'
        "Your last message is still being processed. Please wait a few seconds. "
        "If nothing appears, try again in a moment — you can also send a new question "
        "after ~20 seconds."
        "</div>"
    )


def build_busy_reply_html(original_msg: str = "", reply_lang: str = "") -> str:
    """Polite 'high traffic / try again' in customer's language (all supported langs)."""
    from services.translation_service import customer_facing_template

    user_msg = (original_msg or "").strip() or "try again"
    body = customer_facing_template(
        "server_busy",
        user_msg,
        reply_lang,
        fallback_en=(
            "We're experiencing very high traffic right now, so I couldn't complete your reply. "
            "Please try again in a minute or two — thank you for your patience."
        ),
    )
    if body.strip():
        return body
    return customer_facing_template(
        "server_technical_issue",
        user_msg,
        reply_lang,
        wrap_html=True,
        fallback_en=(
            "Sorry — something went wrong on our side. "
            "Please try again in a moment."
        ),
    )


def build_timeout_reply_html(original_msg: str = "", reply_lang: str = "") -> str:
    return build_busy_reply_html(original_msg, reply_lang)


def json_busy_response(chat_id: str | None = None, original_msg: str = "", reply_lang: str = ""):
    from flask import jsonify

    payload = {
        "type": "text",
        "data": build_busy_reply_html(original_msg, reply_lang),
        "degraded": True,
        "reason": "busy_or_timeout",
    }
    if chat_id:
        payload["chat_id"] = chat_id
    return jsonify(payload), 200


def _handler_reply_is_usable(result: Any) -> bool:
    """True when /chat already built a real customer reply (not degraded busy)."""
    try:
        if result is None or not hasattr(result, "get_json"):
            return False
        payload = result.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("degraded"):
            return False
        body = payload.get("data")
        if isinstance(body, str) and body.strip():
            return True
        if payload.get("type") in ("purchase_history_append", "wishlist_append"):
            return bool(payload.get("cards_html") or payload.get("tail_html"))
    except Exception:
        return False
    return False


def run_with_chat_deadline(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    *,
    app,
    deadline_sec: float = CHAT_MAX_SECONDS,
):
    """
    Hard wall-clock cap for /chat — return busy to client when exceeded.
    Background work may continue on a daemon thread but in-flight lock is released.
    """
    limit = float(deadline_sec or CHAT_MAX_SECONDS)
    result_box: dict[str, Any] = {}
    exc_box: dict[str, BaseException] = {}

    from flask import copy_current_request_context

    @copy_current_request_context
    def _runner() -> None:
        try:
            result_box["result"] = fn(*args, **kwargs)
        except BaseException as exc:
            exc_box["exc"] = exc

    t0 = time.perf_counter()
    worker = threading.Thread(target=_runner, name="chat-turn", daemon=True)
    worker.start()
    worker.join(timeout=limit)

    if worker.is_alive():
        try:
            from services.chat_flow_telemetry import record_timeout_point

            record_timeout_point("chat_deadline_exceeded")
        except ImportError:
            pass
        log_reasoning(
            f"Chat hard deadline ({limit:.0f}s) — client gets busy; in-flight unlocked."
        )
        raise ChatDeadlineExceeded()

    if "exc" in exc_box:
        raise exc_box["exc"]

    result = result_box.get("result")
    elapsed = time.perf_counter() - t0
    if elapsed > limit and _handler_reply_is_usable(result):
        log_reasoning(
            f"Chat over deadline ({elapsed:.1f}s > {limit}s) "
            "— returning computed reply (avoid wasted work)."
        )
    return result


def should_return_busy_fallback(ai_route: dict | None = None) -> bool:
    kind = get_last_llm_failure()
    if kind in ("rate_limit", "busy", "timeout", "all_failed"):
        return True
    if isinstance(ai_route, dict) and (ai_route.get("_llm_failure") or "").strip():
        return True
    return False


def log_busy_fallback(reason: str) -> None:
    chat_log(f"busy fallback: {reason}")
    log_reasoning(f"Customer busy fallback — {reason}")
