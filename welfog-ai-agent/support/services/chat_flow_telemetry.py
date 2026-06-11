"""
Per-chat-turn telemetry — one analysis, count LLM calls, log response time.
Thread-local so concurrent /chat requests stay isolated.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

_TLS = threading.local()


def begin_chat_turn() -> str:
    """Start turn timer; return request_id for debug logs."""
    rid = uuid.uuid4().hex[:10]
    _TLS.request_id = rid
    _TLS.started_at = time.perf_counter()
    _TLS.llm_calls = 0
    _TLS.intent = ""
    _TLS.source = ""
    _TLS.skipped_steps: list[str] = []
    _TLS.route_history: list[str] = []
    _TLS.route = ""
    _TLS.route_reason = ""
    _TLS.ai_route = None
    _TLS.route_decision = None
    _TLS.brain_route_result: dict | None = None
    _TLS.routing_complete = False
    _TLS.brain_route_called = False
    _TLS.expanded_query_cache = None
    return rid


def request_id() -> str:
    return (getattr(_TLS, "request_id", None) or "-").strip() or "-"


def record_route_step(step: str) -> None:
    """Append a routing pipeline step (deduped) for loop detection."""
    name = (step or "").strip()
    if not name:
        return
    hist: list[str] = getattr(_TLS, "route_history", None) or []
    if hist and hist[-1] == name:
        return
    hist.append(name)
    _TLS.route_history = hist


def route_history_str() -> str:
    hist = getattr(_TLS, "route_history", None) or []
    return "→".join(hist) if hist else "-"


def mark_routing_complete() -> None:
    _TLS.routing_complete = True
    record_route_step("routing_complete")


def is_routing_complete() -> bool:
    return bool(getattr(_TLS, "routing_complete", False))


def store_brain_route_result(result: dict | None) -> None:
    if isinstance(result, dict):
        _TLS.brain_route_result = dict(result)
        _TLS.brain_route_called = True


def get_cached_brain_route() -> dict | None:
    r = getattr(_TLS, "brain_route_result", None)
    return dict(r) if isinstance(r, dict) else None


def guard_duplicate_brain_route(caller: str = "ai_brain_route") -> dict | None:
    """
    Return cached ai_brain_route JSON if this turn already ran the main router.
    Prevents timeout from duplicate routing LLM calls.
    """
    if getattr(_TLS, "brain_route_called", False):
        cached = get_cached_brain_route()
        if cached is not None:
            skip_step(caller, "reuse cached brain route")
            return cached
    if is_routing_complete():
        stored = get_stored_ai_route()
        if stored:
            skip_step(caller, "routing already complete")
            return stored
    record_route_step(caller)
    return None


def store_turn_analysis(
    ai_route: dict | None,
    route_decision: Any = None,
) -> None:
    """Persist first AI routing result for reuse in the same HTTP request."""
    if isinstance(ai_route, dict):
        _TLS.ai_route = dict(ai_route)
        store_brain_route_result(ai_route)
    if route_decision is not None:
        _TLS.route_decision = route_decision
    intent = ""
    source = ""
    route_handler = ""
    reason = ""
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        source = (
            (ai_route.get("route_handler") or "")
            or (ai_route.get("data_channel") or "")
        ).strip().lower()
        reason = (ai_route.get("reasoning") or "")[:300]
    if route_decision is not None:
        intent = intent or (getattr(route_decision, "intent", None) or "").strip().lower()
        source = source or (getattr(route_decision, "handler", None) or "").strip().lower()
        route_handler = (getattr(route_decision, "handler", None) or "").strip().lower()
        reason = reason or (getattr(route_decision, "reason", None) or "")[:300]
    _TLS.route = route_handler or source
    _TLS.route_reason = reason
    record_route(intent=intent, source=source)
    mark_routing_complete()


def get_stored_ai_route() -> dict | None:
    r = getattr(_TLS, "ai_route", None)
    if isinstance(r, dict):
        return dict(r)
    return get_cached_brain_route()


def get_stored_route_decision() -> Any:
    return getattr(_TLS, "route_decision", None)


def skip_step(step: str, reason: str = "") -> None:
    """Record a skipped duplicate classifier / promotion for end-of-turn logs."""
    name = (step or "").strip().lower().replace(" ", "_")
    if not name:
        return
    steps: list[str] = getattr(_TLS, "skipped_steps", None) or []
    label = f"{name}:{reason}" if reason else name
    if label not in steps:
        steps.append(label)
    _TLS.skipped_steps = steps


def skipped_steps_str() -> str:
    steps = getattr(_TLS, "skipped_steps", None) or []
    return ",".join(steps) if steps else "-"


def record_route(intent: str = "", source: str = "") -> None:
    if intent:
        _TLS.intent = (intent or "").strip().lower()
    if source:
        _TLS.source = (source or "").strip().lower()


def increment_llm_call(provider: str = "") -> None:
    _TLS.llm_calls = int(getattr(_TLS, "llm_calls", 0) or 0) + 1
    if provider:
        _TLS.last_provider = provider


def llm_calls_count() -> int:
    return int(getattr(_TLS, "llm_calls", 0) or 0)


def response_time_sec() -> float:
    started = getattr(_TLS, "started_at", None)
    if started is None:
        return 0.0
    return max(0.0, time.perf_counter() - float(started))


def log_turn_complete(
    *,
    intent: str = "",
    source: str = "",
    route: str = "",
    reason: str = "",
    extra: str = "",
    confidence: float | None = None,
) -> None:
    intent_f = (intent or getattr(_TLS, "intent", "") or "-").strip().lower() or "-"
    source_f = (source or getattr(_TLS, "source", "") or "-").strip().lower() or "-"
    route_f = (route or getattr(_TLS, "route", "") or source_f or "-").strip().lower() or "-"
    reason_f = (reason or getattr(_TLS, "route_reason", "") or "-").strip()
    if len(reason_f) > 120:
        reason_f = reason_f[:117] + "..."
    calls = llm_calls_count()
    elapsed = response_time_sec()
    conf = confidence
    if conf is None:
        conf = float(getattr(_TLS, "scope_confidence", 0.0) or 0.0)
    conf_s = f"{conf:.2f}" if conf else "-"
    skipped = skipped_steps_str()
    hist = route_history_str()
    rid = request_id()
    msg = (
        f"[chat-flow] request_id={rid} intent={intent_f} route={route_f} "
        f"confidence={conf_s} source={source_f} llm_calls={calls} "
        f"response_time={elapsed:.2f}s reason={reason_f!r} "
        f"route_history={hist} skipped_steps={skipped}"
    )
    if extra:
        msg += f" {extra}"
    log_reasoning(msg)
    chat_log(msg)


def ai_route_already_decided(ai_route: dict | None) -> bool:
    """True when main router already chose intent + channel — skip micro-classifiers."""
    if is_routing_complete():
        return True
    if not isinstance(ai_route, dict):
        stored = get_stored_ai_route()
        if stored:
            ai_route = stored
        else:
            return False
    try:
        from services.early_live_dispatch import ai_route_is_live_api_turn

        if ai_route_is_live_api_turn(ai_route):
            return True
    except ImportError:
        pass
    intent = (ai_route.get("intent") or "").strip().lower()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if intent in ("wishlist", "order_history") and channel == "live_api":
        return True
    if intent in ("order", "refund", "payment", "pincode_check") and channel == "live_api":
        return True
    if intent == "seller" and channel == "kb":
        return True
    if intent == "product" and channel == "catalog":
        return True
    olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
    if olk in ("track", "tracking", "details", "invoice", "refund_status"):
        return True
    if (ai_route.get("account_list_kind") or "").strip().lower() not in ("", "none"):
        return True
    if (ai_route.get("route_handler") or "").strip():
        return True
    if ai_route.get("_turn_promotions_done"):
        return True
    return False
