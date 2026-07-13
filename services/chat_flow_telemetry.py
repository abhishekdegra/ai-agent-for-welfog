"""
Per-chat-turn telemetry — one analysis, count LLM calls, log response time.
Thread-local so concurrent /chat requests stay isolated.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

_TLS = threading.local()

_CFG_CHAT_MAX_LLM_CALLS = int(os.getenv("CHAT_MAX_LLM_CALLS", "5") or "5")
# Allow brain + KB answer (+ one rescue) without starving grounded replies.
# Env CHAT_MAX_LLM_CALLS is honored up to 6 — previous hard min(3) caused KB misses
# after preflight/brain burned the budget.
MAX_LLM_CALLS_PER_TURN = max(1, min(6, _CFG_CHAT_MAX_LLM_CALLS))
MAX_ROUTE_STEPS_PER_TURN = int(os.getenv("CHAT_MAX_ROUTE_STEPS", "16") or "16")


def ensure_chat_turn_started() -> str:
    """One TLS reset per HTTP /chat request (guards + handler share the same turn)."""
    if getattr(_TLS, "started_at", None) is not None:
        return request_id()
    return begin_chat_turn()


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
    _TLS.user_query = ""
    _TLS.detected_language = ""
    _TLS.source_used = ""
    _TLS.phases: list[tuple[str, float]] = []
    _TLS.api_time_ms = 0.0
    _TLS.slowest_ms = 0.0
    _TLS.slow_function = ""
    _TLS.llm_budget_exceeded = False
    _TLS.route_loop_guard = False
    _TLS._micro_defer_logged = False
    _TLS.authoritative_route_lock = None
    _TLS.kb_grounding_hits: list[dict[str, Any]] = []
    _TLS.kb_grounding_corpus = ""
    # KB latency: turn-scoped reuse (same request only — identical semantics).
    _TLS.kb_active_doc_ids = None
    _TLS.kb_active_doc_ids_ready = False
    _TLS.kb_retrieval_base_cache: dict[str, Any] = {}
    _TLS.kb_retrieval_result_cache: dict[str, Any] = {}
    _TLS.kb_retrieval_query_cache: dict[str, str] = {}
    _TLS.kb_openai_embed_key = None
    _TLS.kb_openai_embed_vec = None
    return rid


def get_kb_turn_cache(name: str, default: Any = None) -> Any:
    """Read a turn-local KB cache slot (isolated per /chat request)."""
    return getattr(_TLS, name, default)


def set_kb_turn_cache(name: str, value: Any) -> None:
    setattr(_TLS, name, value)


def set_kb_grounding_context(
    hits: list[dict[str, Any]] | None = None,
    *,
    corpus: str = "",
) -> None:
    """Store authoritative KB chunks/corpus for final fact-contract enforcement."""
    if hits:
        _TLS.kb_grounding_hits = list(hits)
    if corpus:
        _TLS.kb_grounding_corpus = corpus.strip()
    elif hits:
        try:
            from services.knowledge_grounding_validator import _chunk_corpus

            _TLS.kb_grounding_corpus = _chunk_corpus(hits)
        except ImportError:
            pass


def append_kb_grounding_corpus(corpus: str) -> None:
    extra = (corpus or "").strip()
    if not extra:
        return
    prev = (getattr(_TLS, "kb_grounding_corpus", "") or "").strip()
    if prev and extra not in prev:
        _TLS.kb_grounding_corpus = f"{prev}\n\n{extra}"
    elif not prev:
        _TLS.kb_grounding_corpus = extra


def get_kb_grounding_context() -> tuple[list[dict[str, Any]], str]:
    hits = getattr(_TLS, "kb_grounding_hits", None) or []
    corpus = (getattr(_TLS, "kb_grounding_corpus", "") or "").strip()
    return list(hits), corpus


def _route_lock() -> dict[str, Any]:
    lock = getattr(_TLS, "authoritative_route_lock", None)
    if not isinstance(lock, dict):
        lock = {}
        _TLS.authoritative_route_lock = lock
    return lock


def brain_route_authoritative_kb_lock(ai_route: dict | None) -> bool:
    """True when ai_brain_route locked an in-scope KB answer path."""
    if not isinstance(ai_route, dict):
        return False
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(ai_route) or brain_route_indicates_product_catalog(
            ai_route
        ):
            return False
    except ImportError:
        pass
    if ai_route.get("run_catalog_search"):
        return False
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if channel != "kb":
        return False
    if ai_route.get("needs_order_id"):
        return False
    intent = (ai_route.get("intent") or "").strip().lower()
    if intent == "out_of_domain":
        return False
    scope = (ai_route.get("conversation_scope") or "").strip().lower()
    if scope in ("out_of_domain", "harm_sensitive"):
        return False
    if ai_route.get("is_welfog_related") is False:
        return False
    return True


def lock_authoritative_kb_route(
    *,
    source: str,
    handler: str = "kb_grounded",
    chunks: int = 0,
    top_score: float = 0.0,
    ai_route: dict | None = None,
) -> None:
    """Make KB routing immutable for this turn — later OOD/chitchat guards must not override."""
    lock = _route_lock()
    prev = lock.get("channel")
    lock.update(
        {
            "channel": "kb",
            "handler": handler or "kb_grounded",
            "source": source or "brain_route",
            "immutable": True,
            "chunks": int(chunks or 0),
            "top_score": float(top_score or 0.0),
        }
    )
    if isinstance(ai_route, dict):
        lock["intent"] = (ai_route.get("intent") or "general").strip().lower()
        lock["kb_keys"] = list(ai_route.get("kb_keys") or [])
    if prev != "kb":
        log_reasoning(
            f"Authoritative route lock: channel=kb source={lock.get('source')} "
            f"handler={lock.get('handler')} chunks={lock.get('chunks')} "
            f"top_score={lock.get('top_score'):.3f}"
        )


def lock_authoritative_kb_route_from_brain(ai_route: dict | None) -> None:
    if brain_route_authoritative_kb_lock(ai_route):
        lock_authoritative_kb_route(
            source="brain_route",
            handler="kb_brain_locked",
            ai_route=ai_route,
        )


def lock_authoritative_kb_route_from_retrieval(
    *,
    chunks: int,
    top_score: float,
    ai_route: dict | None = None,
) -> None:
    lock_authoritative_kb_route(
        source="kb_retrieval",
        handler="kb_grounded",
        chunks=chunks,
        top_score=top_score,
        ai_route=ai_route,
    )


def is_authoritative_kb_route_locked() -> bool:
    lock = getattr(_TLS, "authoritative_route_lock", None)
    return isinstance(lock, dict) and lock.get("channel") == "kb" and bool(lock.get("immutable"))


def lock_authoritative_pincode_route(
    *,
    source: str = "brain_route",
    ai_route: dict | None = None,
) -> None:
    """
    Make delivery/pincode execution immutable for this turn.

    Mirrors KB authoritative lock: once Brain plans live delivery check,
    product/KB/OOD must not steal the same turn (no keyword routing).
    """
    lock = _route_lock()
    prev = lock.get("channel")
    # Never override an immutable KB lock mid-turn.
    if prev == "kb" and lock.get("immutable"):
        return
    lock.update(
        {
            "channel": "pincode_delivery",
            "handler": "pincode_delivery_api",
            "source": source or "brain_route",
            "immutable": True,
            "chunks": 0,
            "top_score": 0.0,
        }
    )
    if isinstance(ai_route, dict):
        lock["intent"] = (ai_route.get("intent") or "pincode_check").strip().lower()
    if prev != "pincode_delivery":
        log_reasoning(
            f"Authoritative route lock: channel=pincode_delivery "
            f"source={lock.get('source')} handler={lock.get('handler')}"
        )


def lock_authoritative_pincode_route_from_brain(ai_route: dict | None) -> None:
    if not isinstance(ai_route, dict):
        return
    if ai_route.get("_pincode_delivery_locked"):
        lock_authoritative_pincode_route(source="brain_route", ai_route=ai_route)
        return
    intent = (ai_route.get("intent") or "").strip().lower()
    handler = (ai_route.get("route_handler") or "").strip().lower()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    if intent == "pincode_check" or handler == "pincode_delivery_api":
        if channel in ("live_api", "", "none"):
            lock_authoritative_pincode_route(source="brain_route", ai_route=ai_route)


def is_authoritative_pincode_route_locked() -> bool:
    lock = getattr(_TLS, "authoritative_route_lock", None)
    return (
        isinstance(lock, dict)
        and lock.get("channel") == "pincode_delivery"
        and bool(lock.get("immutable"))
    )


def should_skip_post_pincode_route_steal(caller: str = "") -> bool:
    """Product/KB/OOD must not run after authoritative delivery plan is locked."""
    if not is_authoritative_pincode_route_locked():
        return False
    label = f" ({caller})" if caller else ""
    log_reasoning(
        f"Post-delivery steal skipped{label} — authoritative pincode route locked "
        f"({authoritative_route_lock_summary()})."
    )
    return True


def authoritative_route_lock_summary() -> str:
    lock = getattr(_TLS, "authoritative_route_lock", None)
    if not isinstance(lock, dict) or not lock.get("channel"):
        return "-"
    return (
        f"{lock.get('channel')}:{lock.get('source')}:{lock.get('handler')}"
        f":chunks={lock.get('chunks', 0)}:score={float(lock.get('top_score') or 0):.3f}"
    )


def should_skip_post_kb_ood_guard(
    caller: str = "",
    ai_route: dict | None = None,
) -> bool:
    """OOD/chitchat/general guards must not run after authoritative KB lock."""
    if is_authoritative_kb_route_locked():
        label = f" ({caller})" if caller else ""
        log_reasoning(
            f"OOD guard skipped{label} — authoritative KB route locked "
            f"({authoritative_route_lock_summary()})."
        )
        return True
    route = ai_route
    if route is None:
        route = get_cached_brain_route()
    if brain_route_authoritative_kb_lock(route if isinstance(route, dict) else None):
        lock_authoritative_kb_route_from_brain(route if isinstance(route, dict) else None)
        label = f" ({caller})" if caller else ""
        log_reasoning(
            f"OOD guard skipped{label} — brain locked data_channel=kb "
            f"({authoritative_route_lock_summary()})."
        )
        return True
    return False


def record_phase(phase: str, duration_ms: float) -> None:
    """Record wall time for a pipeline segment (ms). Tracks slowest for logs."""
    name = (phase or "").strip()
    if not name:
        return
    ms = max(0.0, float(duration_ms))
    phases: list[tuple[str, float]] = getattr(_TLS, "phases", None) or []
    phases.append((name, ms))
    _TLS.phases = phases
    if ms >= float(getattr(_TLS, "slowest_ms", 0.0) or 0.0):
        _TLS.slowest_ms = ms
        _TLS.slow_function = name


def record_api_time(seconds: float) -> None:
    _TLS.api_time_ms = float(getattr(_TLS, "api_time_ms", 0.0) or 0.0) + max(
        0.0, float(seconds)
    ) * 1000.0


def phases_summary() -> str:
    phases: list[tuple[str, float]] = getattr(_TLS, "phases", None) or []
    if not phases:
        return "-"
    return ",".join(f"{n}:{ms:.0f}ms" for n, ms in phases[-8:])


def record_user_query(original_msg: str, detected_language: str = "") -> None:
    _TLS.user_query = (original_msg or "")[:300]
    if detected_language:
        _TLS.detected_language = detected_language.strip()


def record_routing_confidence(confidence: float) -> None:
    _TLS.scope_confidence = max(0.0, min(1.0, float(confidence or 0.0)))


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
    if len(hist) > MAX_ROUTE_STEPS_PER_TURN:
        _TLS.route_loop_guard = True
        mark_routing_complete()
        log_reasoning(
            f"Route step budget exceeded ({len(hist)}/{MAX_ROUTE_STEPS_PER_TURN}) — lock route."
        )


def route_loop_guard_active() -> bool:
    return bool(getattr(_TLS, "route_loop_guard", False))


def llm_budget_exceeded() -> bool:
    return bool(getattr(_TLS, "llm_budget_exceeded", False))


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
        lock_authoritative_kb_route_from_brain(result)
        if not result.get("llm_unavailable"):
            mark_routing_complete()


def store_early_brain_dispatch(body: str | None, route: dict | None) -> None:
    """Cache first _try_early_ai_brain_reply result — prevent duplicate dispatch per turn."""
    _TLS.early_brain_dispatch_done = True
    _TLS.early_brain_body = body
    _TLS.early_brain_route = dict(route) if isinstance(route, dict) else route


def get_early_brain_dispatch() -> tuple[str | None, dict | None] | None:
    if not getattr(_TLS, "early_brain_dispatch_done", False):
        return None
    return (
        getattr(_TLS, "early_brain_body", None),
        getattr(_TLS, "early_brain_route", None),
    )


def log_routing_decision(
    *,
    query: str = "",
    language: str = "",
    intent: str = "",
    confidence: float | None = None,
    entities: dict[str, str] | None = None,
    selected_route: str = "",
    selected_tool: str = "",
    provider_chain: str = "",
    api_time_ms: float = 0.0,
    rag_score: float = 0.0,
    model_used: str = "",
    extra: str = "",
) -> None:
    """Explicit routing trace for production debugging."""
    q = (query or getattr(_TLS, "user_query", "") or "-")[:120]
    lang = (language or getattr(_TLS, "detected_language", "") or "-").strip() or "-"
    intent_f = (intent or getattr(_TLS, "intent", "") or "-").strip().lower() or "-"
    tool = (
        selected_tool
        or selected_route
        or getattr(_TLS, "route", "")
        or "-"
    ).strip() or "-"
    route_f = (selected_route or tool or "-").strip() or "-"
    prov = (
        provider_chain
        or getattr(_TLS, "provider_chain", "")
        or getattr(_TLS, "last_provider", "")
        or "-"
    )
    api_ms = api_time_ms or float(getattr(_TLS, "api_time_ms", 0.0) or 0.0)
    conf = confidence
    if conf is None:
        conf = float(getattr(_TLS, "scope_confidence", 0.0) or 0.0)
    conf_s = f"{conf:.2f}" if conf else "-"
    ent = entities if entities is not None else getattr(_TLS, "entities", None) or {}
    ent_s = _entities_log_str(ent if isinstance(ent, dict) else {})
    rag_sc = rag_score or float(getattr(_TLS, "chunk_score", 0.0) or 0.0)
    model = model_used or getattr(_TLS, "model_used", "") or "-"
    rid = request_id()
    total_s = response_time_sec()
    msg = (
        f"[routing] request_id={rid} query={q!r} language={lang} intent={intent_f} "
        f"confidence={conf_s} entities={ent_s} selected_route={route_f} "
        f"selected_tool={tool} provider_chain={prov} api_time={api_ms / 1000.0:.2f}s "
        f"rag_score={rag_sc:.3f} model_used={model} total_time={total_s:.2f}s"
    )
    if extra:
        msg += f" {extra}"
    log_reasoning(msg)
    chat_log(msg)


def get_cached_brain_route() -> dict | None:
    r = getattr(_TLS, "brain_route_result", None)
    return dict(r) if isinstance(r, dict) else None


def begin_pre_brain_live_api_preflight() -> None:
    """Allow focused micro-LLMs (account-list, delivery, KB-turn) before ai_brain_route."""
    _TLS.pre_brain_live_api_preflight = True


def end_pre_brain_live_api_preflight() -> None:
    _TLS.pre_brain_live_api_preflight = False


def should_defer_micro_classifiers_to_brain() -> bool:
    """
    True while the universal brain route has not run this turn.
    Micro-classifiers (KB-turn, refund, catalog-menu) must not fire before it.
    """
    if getattr(_TLS, "pre_brain_live_api_preflight", False):
        return False
    if is_routing_complete():
        return False
    if getattr(_TLS, "brain_route_called", False):
        return False
    if get_cached_brain_route():
        return False
    return True


def should_skip_micro_classifier_llm() -> bool:
    """Skip duplicate micro-LLM classifiers — defer until brain runs, then skip after brain classified."""
    if should_defer_micro_classifiers_to_brain():
        if not getattr(_TLS, "_micro_defer_logged", False):
            _TLS._micro_defer_logged = True
            log_reasoning(
                "Micro-classifier deferred — universal ai_brain_route runs first (one LLM)."
            )
        return True
    if is_routing_complete():
        return True
    cached = get_cached_brain_route()
    if isinstance(cached, dict):
        if cached.get("llm_unavailable"):
            return False
        if (cached.get("intent") or "").strip():
            return True
    if getattr(_TLS, "brain_route_called", False):
        if isinstance(cached, dict) and not cached.get("llm_unavailable"):
            return True
    return False


def ensure_brain_route_llm_slot() -> None:
    """
    ai_brain_route is the primary router — always reserve one billable LLM call.
    Micro-classifiers exhausting CHAT_MAX_LLM_CALLS is not an API-key failure.
    """
    if not llm_budget_exceeded():
        return
    log_reasoning(
        f"LLM call budget reset — reserving one slot for ai_brain_route "
        f"(limit={MAX_LLM_CALLS_PER_TURN}; keys are fine)."
    )
    _TLS.llm_budget_exceeded = False
    current = int(getattr(_TLS, "llm_calls", 0) or 0)
    if current >= MAX_LLM_CALLS_PER_TURN:
        _TLS.llm_calls = max(0, MAX_LLM_CALLS_PER_TURN - 1)


def ensure_product_rescue_llm_slot() -> None:
    """Reserve LLM budget for OOD product rescue (classify + extract)."""
    reset_llm_budget_for_recovery()
    _TLS.llm_calls = 1
    log_reasoning(
        f"LLM budget reset for product rescue (limit={MAX_LLM_CALLS_PER_TURN})."
    )


def mark_kb_grounding_operation() -> None:
    """Qdrant retrieval succeeded — grounding LLM is part of the same KB answer operation."""
    _TLS.kb_grounding_reserved = True


def clear_kb_grounding_operation() -> None:
    if hasattr(_TLS, "kb_grounding_reserved"):
        delattr(_TLS, "kb_grounding_reserved")


def kb_grounding_reserved() -> bool:
    return bool(getattr(_TLS, "kb_grounding_reserved", False))


def ensure_kb_grounding_llm_slot() -> None:
    """
    Reserve one LLM call for ai_brain_answer after successful Qdrant retrieval.
    Retrieval + grounded answer are one logical KB operation — not a separate budget surprise.
    """
    if not kb_grounding_reserved():
        return
    if not llm_budget_exceeded() and llm_calls_count() < MAX_LLM_CALLS_PER_TURN:
        return
    log_reasoning(
        f"LLM call budget reset — reserving one slot for KB grounded answer "
        f"(limit={MAX_LLM_CALLS_PER_TURN}; retrieval already succeeded)."
    )
    _TLS.llm_budget_exceeded = False
    current = int(getattr(_TLS, "llm_calls", 0) or 0)
    if current >= MAX_LLM_CALLS_PER_TURN:
        _TLS.llm_calls = max(0, MAX_LLM_CALLS_PER_TURN - 1)


def reset_llm_budget_for_recovery() -> None:
    """One compact classify after brain routing failed — do not return busy on shopping turns."""
    _TLS.llm_budget_exceeded = False
    _TLS.llm_calls = 0


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
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
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
        src_used = (getattr(route_decision, "source", None) or "").strip().lower()
        if src_used:
            _TLS.source_used = src_used
            _TLS.final_source = src_used
    _TLS.route = route_handler or source
    _TLS.route_reason = reason
    _TLS.entities = extract_turn_entities(
        ai_route,
        route_decision,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    record_route(intent=intent, source=source)
    mark_routing_complete()
    log_intent_routing(
        detected_intent=intent,
        selected_existing_tool=route_handler or source,
        entities=_TLS.entities,
        source_used=getattr(_TLS, "source_used", None) or (
            getattr(route_decision, "source", None) if route_decision is not None else source
        ),
        reason=reason,
    )
    try:
        _TLS.tool_called = route_handler or source
    except Exception:
        pass


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


def increment_llm_call(provider: str = "", *, billable: bool = True) -> None:
    try:
        from services.chat_resilience import touch_chat_turn_in_flight

        touch_chat_turn_in_flight()
    except ImportError:
        pass
    if not billable:
        return
    current = int(getattr(_TLS, "llm_calls", 0) or 0)
    if current >= MAX_LLM_CALLS_PER_TURN:
        _TLS.llm_budget_exceeded = True
        log_reasoning(
            f"LLM call budget exceeded ({current}/{MAX_LLM_CALLS_PER_TURN}) — skip further LLM."
        )
        return
    _TLS.llm_calls = current + 1
    if provider:
        _TLS.last_provider = provider
    if _TLS.llm_calls >= MAX_LLM_CALLS_PER_TURN:
        _TLS.llm_budget_exceeded = True
        log_reasoning(
            f"LLM call budget reached ({_TLS.llm_calls}/{MAX_LLM_CALLS_PER_TURN})."
        )


def llm_calls_count() -> int:
    return int(getattr(_TLS, "llm_calls", 0) or 0)


def response_time_sec() -> float:
    started = getattr(_TLS, "started_at", None)
    if started is None:
        return 0.0
    return max(0.0, time.perf_counter() - float(started))


def extract_turn_entities(
    ai_route: dict | None = None,
    route_decision: Any = None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict[str, str]:
    """Pull order_id, pincode, product_id, search_query from route + message."""
    entities: dict[str, str] = {}
    r = ai_route if isinstance(ai_route, dict) else {}
    if r.get("extracted_pincode"):
        entities["pincode"] = str(r.get("extracted_pincode") or "").strip()
    if r.get("extracted_order_id"):
        entities["order_id"] = str(r.get("extracted_order_id") or "").strip()
    sq = (r.get("search_query") or "").strip()
    if sq:
        entities["search_query"] = sq
    if route_decision is not None:
        rsq = (getattr(route_decision, "search_query", None) or "").strip()
        if rsq and "search_query" not in entities:
            entities["search_query"] = rsq
    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    if intent == "product" and channel == "catalog" and entities.get("search_query"):
        return {k: v for k, v in entities.items() if v}
    needs_oid = bool(r.get("needs_order_id"))
    numeric = (r.get("numeric_context") or "").strip().lower()
    if not needs_oid and numeric not in ("order_id", "pincode", "product_id"):
        if entities.get("search_query") or intent in ("general", "out_of_domain"):
            return {k: v for k, v in entities.items() if v}
    if not entities.get("order_id") or not entities.get("pincode"):
        try:
            from utils.helpers import extract_embedded_query_identifiers

            ids = extract_embedded_query_identifiers(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=r,
            )
            if ids.get("order_id") and not entities.get("order_id"):
                entities["order_id"] = str(ids["order_id"]).strip()
            if ids.get("pincode") and not entities.get("pincode"):
                entities["pincode"] = str(ids["pincode"]).strip()
            if ids.get("product_id"):
                entities["product_id"] = str(ids["product_id"]).strip()
        except ImportError:
            pass
    return {k: v for k, v in entities.items() if v}


def _entities_log_str(entities: dict[str, str]) -> str:
    if not entities:
        return "-"
    parts = [f"{k}={v}" for k, v in sorted(entities.items())]
    return ",".join(parts)


def log_intent_routing(
    *,
    detected_intent: str = "",
    selected_existing_tool: str = "",
    entities: dict[str, str] | None = None,
    source_used: str = "",
    response_time: float | None = None,
    reason: str = "",
    extra: str = "",
) -> None:
    """Production routing log: intent → existing tool → entities → source."""
    intent_f = (detected_intent or getattr(_TLS, "intent", "") or "-").strip().lower() or "-"
    tool_f = (
        selected_existing_tool
        or getattr(_TLS, "route", "")
        or getattr(_TLS, "source", "")
        or "-"
    ).strip().lower() or "-"
    source_f = (source_used or getattr(_TLS, "source", "") or tool_f or "-").strip().lower() or "-"
    ent = entities if entities is not None else getattr(_TLS, "entities", None) or {}
    ent_s = _entities_log_str(ent if isinstance(ent, dict) else {})
    elapsed = response_time if response_time is not None else response_time_sec()
    reason_f = (reason or getattr(_TLS, "route_reason", "") or "-").strip()
    if len(reason_f) > 120:
        reason_f = reason_f[:117] + "..."
    rid = request_id()
    msg = (
        f"[intent-routing] request_id={rid} detected_intent={intent_f} "
        f"selected_existing_tool={tool_f} entities={ent_s} "
        f"source_used={source_f} response_time={elapsed:.2f}s "
        f"reason={reason_f!r}"
    )
    if extra:
        msg += f" {extra}"
    log_reasoning(msg)
    chat_log(msg)


def record_timeout_point(point: str) -> None:
    _TLS.timeout_point = (point or "").strip() or "-"


def record_rag_meta(
    *,
    rag_source_file: str = "",
    chunk_score: float = 0.0,
    knowledge_version: str = "",
    model_used: str = "",
) -> None:
    if rag_source_file:
        _TLS.rag_source_file = rag_source_file
    if chunk_score:
        _TLS.chunk_score = float(chunk_score)
    if knowledge_version:
        _TLS.knowledge_version = knowledge_version
    if model_used:
        _TLS.model_used = model_used


def log_conversation_intel(
    *,
    query: str = "",
    language: str = "",
    intent: str = "",
    route: str = "",
    rag_source_file: str = "",
    chunk_score: float = 0.0,
    knowledge_version: str = "",
    source: str = "",
    model_used: str = "",
) -> None:
    """Structured conversation-intelligence log (KB / chitchat / OOD)."""
    rid = request_id()
    elapsed = response_time_sec()
    q = (query or getattr(_TLS, "user_query", "") or "-")[:120]
    lang_f = (language or getattr(_TLS, "detected_language", "") or "-").strip() or "-"
    kv = knowledge_version or getattr(_TLS, "knowledge_version", "") or "-"
    rag = rag_source_file or getattr(_TLS, "rag_source_file", "") or "-"
    score = chunk_score or float(getattr(_TLS, "chunk_score", 0.0) or 0.0)
    model = model_used or getattr(_TLS, "model_used", "") or "-"
    if intent:
        record_route(intent=intent, source=source or route)
    record_rag_meta(
        rag_source_file=rag,
        chunk_score=score,
        knowledge_version=kv,
        model_used=model,
    )
    msg = (
        f"[conversation-intel] request_id={rid} query={q!r} language={lang_f} "
        f"intent={intent or '-'} route={route or '-'} rag_source_file={rag} "
        f"chunk_score={score:.3f} knowledge_version={kv} model_used={model} "
        f"source={source or '-'} response_time={elapsed:.2f}s"
    )
    log_reasoning(msg)
    chat_log(msg)


def log_pipeline_complete(
    *,
    user_query: str = "",
    detected_language: str = "",
    confidence: float | None = None,
    final_source: str = "",
) -> None:
    """Full pipeline summary log for production debugging."""
    rid = request_id()
    intent_f = (getattr(_TLS, "intent", "") or "-").strip().lower() or "-"
    route_f = (getattr(_TLS, "route", "") or "-").strip().lower() or "-"
    tool_f = route_f
    src_f = (
        final_source
        or getattr(_TLS, "source_used", None)
        or getattr(_TLS, "source", "")
        or "-"
    ).strip().lower() or "-"
    ent_s = _entities_log_str(getattr(_TLS, "entities", None) or {})
    elapsed = response_time_sec()
    calls = llm_calls_count()
    conf = confidence
    if conf is None:
        conf = float(getattr(_TLS, "scope_confidence", 0.0) or 0.0)
    conf_s = f"{conf:.2f}" if conf else "-"
    q = (user_query or getattr(_TLS, "user_query", "") or "-")[:120]
    lang_f = (detected_language or getattr(_TLS, "detected_language", "") or "-").strip() or "-"
    timeout_pt = (getattr(_TLS, "timeout_point", "") or "-").strip() or "-"
    rag_f = (getattr(_TLS, "rag_source_file", "") or "-").strip() or "-"
    chunk_sc = float(getattr(_TLS, "chunk_score", 0.0) or 0.0)
    kv_f = (getattr(_TLS, "knowledge_version", "") or "-").strip() or "-"
    model_f = (getattr(_TLS, "model_used", "") or "-").strip() or "-"
    api_ms = float(getattr(_TLS, "api_time_ms", 0.0) or 0.0)
    slow_fn = (getattr(_TLS, "slow_function", "") or "-").strip() or "-"
    phase_s = phases_summary()
    msg = (
        f"[pipeline] request_id={rid} query={q!r} language={lang_f} "
        f"intent={intent_f} confidence={conf_s} selected_route={route_f} "
        f"tool_used={tool_f} entities={ent_s} source={src_f} "
        f"llm_call_count={calls} api_time={api_ms / 1000.0:.2f}s "
        f"total_time={elapsed:.2f}s slow_function={slow_fn} phases={phase_s} "
        f"timeout_point={timeout_pt} rag_source_file={rag_f} "
        f"chunk_score={chunk_sc:.3f} knowledge_version={kv_f} model_used={model_f}"
    )
    log_reasoning(msg)
    chat_log(msg)


def log_order_dispatch(
    *,
    message: str = "",
    detected_intent: str = "",
    detected_language: str = "",
    previous_context: str = "",
    previous_context_used: bool = False,
    pending_action: str = "",
    order_id_found: str = "",
    selected_tool: str = "",
    api_called: bool = False,
    api_time_ms: float = 0.0,
    entities: dict | None = None,
    confidence: float | None = None,
) -> None:
    """Production order-flow log: intent → pending action → tool → API."""
    try:
        record_route_step("order_live_dispatch")
        if detected_intent:
            record_route(intent=detected_intent, source=selected_tool or "order_live")
        mark_routing_complete()
    except Exception:
        pass
    entity_map: dict[str, str] = dict(entities or {})
    if order_id_found:
        entity_map.setdefault("order_id", order_id_found)
        entity_map.setdefault("extracted_order_id", order_id_found)
    if pending_action:
        entity_map.setdefault("pending_action", pending_action)
    prev_snip = (previous_context or "").strip().replace("\n", " ")[:120]
    msg_snip = (message or "").strip().replace("\n", " ")[:120]
    entities_s = ",".join(f"{k}={v}" for k, v in entity_map.items()) or "-"
    ctx_used = previous_context_used or bool(prev_snip)
    conf = confidence
    if conf is None:
        conf = float(getattr(_TLS, "scope_confidence", 0.0) or 0.0)
    conf_s = f"{conf:.2f}" if conf else "-"
    total = response_time_sec()
    extra = (
        f"message={msg_snip or '-'} "
        f"language={detected_language or '-'} "
        f"intent={detected_intent or '-'} "
        f"confidence={conf_s} "
        f"entities={entities_s} "
        f"extracted_order_id={order_id_found or '-'} "
        f"selected_tool={selected_tool or '-'} "
        f"previous_context_used={ctx_used} "
        f"previous_context={prev_snip or '-'} "
        f"pending_action={pending_action or '-'} "
        f"api_called={api_called} api_time={api_time_ms / 1000.0:.2f}s "
        f"total_time={total:.2f}s "
        f"llm_call_count={llm_calls_count()}"
    )
    log_intent_routing(
        detected_intent=detected_intent or "-",
        selected_existing_tool=selected_tool or "-",
        entities=entity_map,
        source_used="brain_direct_order" if api_called else "order_pending",
        response_time=response_time_sec(),
        reason="Order live dispatch",
        extra=extra,
    )
    log_reasoning(
        f"[order-flow] detected_intent={detected_intent or '-'} "
        f"detected_language={detected_language or '-'} "
        f"confidence={conf_s} "
        f"entities={entities_s} "
        f"extracted_order_id={order_id_found or '-'} "
        f"selected_tool={selected_tool or '-'} "
        f"api_time={api_time_ms / 1000.0:.2f}s "
        f"total_time={total:.2f}s "
        f"previous_context_used={ctx_used} "
        f"api_called={api_called} llm_call_count={llm_calls_count()}"
    )


def log_product_dispatch(
    *,
    message: str = "",
    detected_intent: str = "product",
    detected_language: str = "",
    entities: dict | None = None,
    selected_tool: str = "",
    opensearch_query: dict | None = None,
    api_time_ms: float | None = None,
    total_time_sec: float | None = None,
) -> None:
    """Production product-flow log: intent → entities → OpenSearch → timing."""
    try:
        record_route_step("product_catalog_dispatch")
        if detected_intent:
            record_route(intent=detected_intent, source=selected_tool or "product_catalog")
        mark_routing_complete()
    except Exception:
        pass
    entity_map: dict[str, str] = {}
    for k, v in (entities or {}).items():
        if v is not None and v != "" and v != []:
            entity_map[k] = str(v)
    os_q = dict(opensearch_query or {})
    os_s = json.dumps(os_q, default=str, ensure_ascii=False)[:400] if os_q else "-"
    msg_snip = (message or "").strip().replace("\n", " ")[:120]
    entities_s = ",".join(f"{k}={v}" for k, v in entity_map.items()) or "-"
    api_ms = api_time_ms
    if api_ms is None:
        api_ms = float(getattr(_TLS, "api_time_ms", 0.0) or 0.0)
    total = total_time_sec
    if total is None:
        total = response_time_sec()
    lang_f = (detected_language or getattr(_TLS, "detected_language", "") or "-").strip() or "-"
    extra = (
        f"message={msg_snip or '-'} "
        f"detected_language={lang_f} "
        f"intent={detected_intent or 'product'} "
        f"entities={entities_s} "
        f"selected_tool={selected_tool or '-'} "
        f"opensearch_query={os_s} "
        f"api_time={api_ms / 1000.0:.2f}s "
        f"total_time={total:.2f}s "
        f"llm_call_count={llm_calls_count()}"
    )
    log_intent_routing(
        detected_intent=detected_intent or "product",
        selected_existing_tool=selected_tool or "-",
        entities=entity_map,
        source_used="brain_direct_product" if selected_tool else "product_catalog",
        response_time=total,
        reason="Product catalog dispatch",
        extra=extra,
    )
    log_reasoning(f"[product-flow] {extra}")


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
    log_intent_routing(
        detected_intent=intent_f,
        selected_existing_tool=route_f,
        entities=getattr(_TLS, "entities", None),
        source_used=getattr(_TLS, "source_used", None) or source_f,
        response_time=elapsed,
        reason=reason_f,
        extra=f"llm_calls={calls} route_history={hist} skipped_steps={skipped}",
    )


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
