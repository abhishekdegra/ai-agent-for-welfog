"""
Chat resilience — deadlines, rate limits, and polite customer-facing fallbacks.

Customers should never wait indefinitely; when LLMs/APIs are down or rate-limited,
return a short message in their language instead of hanging or a generic 500.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Optional

from utils.reasoning_log import chat_log, log_reasoning

CHAT_MAX_SECONDS = float(os.getenv("CHAT_MAX_SECONDS") or "70")
_LLM_FAILURE = threading.local()


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


def run_with_chat_deadline(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    *,
    app,
    deadline_sec: float = CHAT_MAX_SECONDS,
):
    """Run chat handler in a worker thread with a hard wall-clock limit."""
    from flask import copy_current_request_context

    remaining = max(1.0, float(deadline_sec))

    @copy_current_request_context
    def _target():
        with app.app_context():
            return fn(*args, **kwargs)

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_target)
    try:
        return fut.result(timeout=remaining)
    except FuturesTimeout as exc:
        # Do not block the Flask worker until the overrun thread finishes (was causing
        # multi-minute/hour stalls and busy fallbacks after good replies were ready).
        pool.shutdown(wait=False, cancel_futures=True)
        raise ChatDeadlineExceeded() from exc
    else:
        pool.shutdown(wait=True)


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
