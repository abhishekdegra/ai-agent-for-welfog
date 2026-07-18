"""
Chat resilience — deadlines, rate limits, and polite customer-facing fallbacks.

Customers should never wait indefinitely; when LLMs/APIs are down or rate-limited,
return a short message in their language instead of hanging or a generic 500.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Optional

from utils.reasoning_log import chat_log, log_reasoning

CHAT_MAX_SECONDS = float(os.getenv("CHAT_MAX_SECONDS") or "70")
GUARD_PRELOCK_MAX_SEC = float(os.getenv("GUARD_PRELOCK_MAX_SEC", "18") or "18")
CHAT_IN_FLIGHT_STALE_SEC = float(
    os.getenv("CHAT_IN_FLIGHT_STALE_SEC") or "1"
)
CHAT_IN_FLIGHT_MAX_SEC = float(
    os.getenv("CHAT_IN_FLIGHT_MAX_SEC")
    or str(max(CHAT_IN_FLIGHT_STALE_SEC + 5.0, min(CHAT_MAX_SECONDS + 5.0, 45.0)))
)
_LLM_FAILURE = threading.local()
_CHAT_LOCKS_GUARD = threading.Lock()
_CHAT_IN_FLIGHT: set[str] = set()
_CHAT_IN_FLIGHT_AT: dict[str, float] = {}
_CHAT_IN_FLIGHT_TOKEN: dict[str, str] = {}

_USER_IN_FLIGHT: set[str] = set()
_USER_IN_FLIGHT_AT: dict[str, float] = {}
_TURN_ACQUIRED = threading.local()
# Abandoned async /chat worker thread ids — visible across the deadline thread.
_ABANDON_LOCK = threading.Lock()
_ABANDONED_THREAD_IDS: set[int] = set()


def mark_chat_turn_abandoned(
    reason: str = "deadline",
    *,
    thread_id: int | None = None,
) -> None:
    """Signal a /chat worker to stop starting new LLM/embed work."""
    tid = int(thread_id) if thread_id is not None else threading.get_ident()
    with _ABANDON_LOCK:
        _ABANDONED_THREAD_IDS.add(tid)
        if len(_ABANDONED_THREAD_IDS) > 64:
            # Drop oldest-ish by clearing; next marks re-add active zombies.
            _ABANDONED_THREAD_IDS.clear()
            _ABANDONED_THREAD_IDS.add(tid)
    log_reasoning(
        f"Chat turn abandoned — stop further LLM/embed "
        f"(tid={tid} {(reason or 'deadline')[:60]})."
    )


def clear_chat_turn_abandoned(*, thread_id: int | None = None) -> None:
    tid = int(thread_id) if thread_id is not None else threading.get_ident()
    with _ABANDON_LOCK:
        _ABANDONED_THREAD_IDS.discard(tid)


def chat_turn_abandoned() -> bool:
    with _ABANDON_LOCK:
        return threading.get_ident() in _ABANDONED_THREAD_IDS


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
    _CHAT_IN_FLIGHT_TOKEN.pop(key, None)
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
    token = (getattr(_TURN_ACQUIRED, "turn_token", "") or "").strip()
    if not uid and not key:
        return
    now = time.monotonic()
    with _CHAT_LOCKS_GUARD:
        if key and key in _CHAT_IN_FLIGHT:
            if _CHAT_IN_FLIGHT_TOKEN.get(key) != token:
                return
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
    """Release in-flight markers after deadline/timeout so customer can send again."""
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    token = (getattr(_TURN_ACQUIRED, "turn_token", "") or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key:
            if not token or _CHAT_IN_FLIGHT_TOKEN.get(key) == token:
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


def force_end_all_user_chat_turns(user_id: str = "") -> None:
    """New-chat reset — drop every in-flight marker for this user."""
    uid = str(user_id or "").strip()
    with _CHAT_LOCKS_GUARD:
        if uid:
            _force_clear_in_flight(
                uid,
                _USER_IN_FLIGHT,
                _USER_IN_FLIGHT_AT,
                reason="user new-chat reset",
            )


def try_begin_chat_turn(chat_id: str, user_id: str = "") -> bool:
    """
    One active /chat per chat_id — timestamp only (no threading.Lock).

    A slow/zombie handler cannot block the customer forever: after
    CHAT_IN_FLIGHT_STALE_SEC a new message is allowed and gets a fresh token.
    """
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    if not key:
        _TURN_ACQUIRED.acquired = True
        _TURN_ACQUIRED.chat_id = ""
        _TURN_ACQUIRED.user_id = uid
        _TURN_ACQUIRED.turn_token = ""
        return True

    with _CHAT_LOCKS_GUARD:
        if key in _CHAT_IN_FLIGHT:
            age = _in_flight_age(key, _CHAT_IN_FLIGHT, _CHAT_IN_FLIGHT_AT)
            if age > CHAT_IN_FLIGHT_STALE_SEC:
                _force_clear_in_flight(
                    key,
                    _CHAT_IN_FLIGHT,
                    _CHAT_IN_FLIGHT_AT,
                    reason="stale turn replaced by new message",
                )
            else:
                # Latest customer message wins — preempt slow prior handler (no deadlock).
                _force_clear_in_flight(
                    key,
                    _CHAT_IN_FLIGHT,
                    _CHAT_IN_FLIGHT_AT,
                    reason="latest message preempts in-flight turn",
                )
        token = uuid.uuid4().hex
        now = time.monotonic()
        _CHAT_IN_FLIGHT.add(key)
        _CHAT_IN_FLIGHT_AT[key] = now
        _CHAT_IN_FLIGHT_TOKEN[key] = token
        if uid:
            _USER_IN_FLIGHT.add(uid)
            _USER_IN_FLIGHT_AT[uid] = now

    _TURN_ACQUIRED.acquired = True
    _TURN_ACQUIRED.chat_id = key
    _TURN_ACQUIRED.user_id = uid
    _TURN_ACQUIRED.turn_token = token
    return True


def end_chat_turn(chat_id: str, user_id: str = "") -> None:
    key = (chat_id or "").strip()
    uid = str(user_id or "").strip()
    token = (getattr(_TURN_ACQUIRED, "turn_token", "") or "").strip()
    with _CHAT_LOCKS_GUARD:
        if key and _CHAT_IN_FLIGHT_TOKEN.get(key) == token:
            _CHAT_IN_FLIGHT.discard(key)
            _CHAT_IN_FLIGHT_AT.pop(key, None)
            _CHAT_IN_FLIGHT_TOKEN.pop(key, None)
        if uid:
            _USER_IN_FLIGHT.discard(uid)
            _USER_IN_FLIGHT_AT.pop(uid, None)
    _TURN_ACQUIRED.turn_token = ""


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
    _TURN_ACQUIRED.turn_token = ""


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
            "Your last message is still being processed. Please wait a moment — "
            "I will reply shortly. If nothing appears, send your question again."
        ),
    )
    if body and body.strip():
        return body
    return (
        '<div style="color:#333;line-height:1.55;">'
        "Your last message is still being processed. Please wait a moment — "
        "I will reply shortly. If nothing appears, send your question again."
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
            "Sorry — I could not finish that in time. "
            "Please send your message again and I will reply right away."
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
    force_threaded: bool = False,
):
    """
    Wall-clock cap for /chat. Default: synchronous handler (no zombie daemon threads
    contending on encode lock / LLM). Set CHAT_ASYNC_HANDLER=1 for legacy threaded mode.
    force_threaded=True: always use join(timeout) so guard prelocks cannot block Flask.
    """
    limit = float(deadline_sec or CHAT_MAX_SECONDS)
    use_async = force_threaded or (os.getenv("CHAT_ASYNC_HANDLER") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    clear_chat_turn_abandoned()
    if not use_async:
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            raise exc
        elapsed = time.perf_counter() - t0
        if elapsed > limit and not _handler_reply_is_usable(result):
            try:
                from services.chat_flow_telemetry import record_timeout_point

                record_timeout_point("chat_deadline_exceeded")
            except ImportError:
                pass
            mark_chat_turn_abandoned("sync_deadline")
            log_reasoning(
                f"Chat hard deadline ({limit:.0f}s) sync — no usable reply."
            )
            raise ChatDeadlineExceeded()
        if elapsed > limit:
            log_reasoning(
                f"Chat over deadline ({elapsed:.1f}s > {limit}s) "
                "— returning computed reply (sync)."
            )
        return result

    result_box: dict[str, Any] = {}
    exc_box: dict[str, BaseException] = {}

    from flask import copy_current_request_context

    @copy_current_request_context
    def _runner() -> None:
        try:
            clear_chat_turn_abandoned()
            result_box["result"] = fn(*args, **kwargs)
        except BaseException as exc:
            exc_box["exc"] = exc
        finally:
            clear_chat_turn_abandoned()

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
        # Daemon worker may keep running — block further LLM/embed on that thread.
        if worker.ident is not None:
            mark_chat_turn_abandoned("async_deadline", thread_id=int(worker.ident))
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
