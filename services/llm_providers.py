"""
Multi-provider LLM chain (OpenAI-compatible chat completions).

Default order (LLM_PROVIDER_ORDER): groq → openai → gemini → deepseek

On failure (timeout, rate limit, bad JSON), llm_json_with_provider_fallback tries the
next configured provider automatically until one succeeds or the chain is exhausted.

- groq: GROQ_API_KEY (Groq Cloud)
- openai: OPENAI_API_KEY
- gemini: GEMINI_API_KEY / GOOGLE_API_KEY
- deepseek: DEEPSEEK_API_KEY

Legacy slot "grok" still supported (xAI Grok, or Groq Cloud if no xAI key).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests

from utils.reasoning_log import log_reasoning

_DEFAULT_MODELS = {
    "grok": "grok-3-mini",
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
}

_DEFAULT_ORDER = ("groq", "openai", "gemini", "deepseek")

# After tokens-per-day / hard rate limits, skip that provider briefly so every
# turn does not pay a failing Groq round-trip before OpenAI (~1–3s wasted each call).
_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}
_PROVIDER_TPD_COOLDOWN_SEC = float(os.getenv("LLM_PROVIDER_TPD_COOLDOWN_SEC") or "180")

# Dead / unreachable providers (bad key, no network, stuck TLS) time out on EVERY
# call. Without a cooldown, each turn pays the full per-provider hard timeout for
# every dead provider before reaching the one that works (this was the 34–77s
# product-search "traffic" timeout). Track consecutive transport failures and put
# the provider on a short cooldown so the rest of the session skips it.
_PROVIDER_FAIL_STREAK: dict[str, int] = {}
_PROVIDER_UNREACHABLE_STREAK = int(os.getenv("LLM_PROVIDER_UNREACHABLE_STREAK") or "2")
_PROVIDER_UNREACHABLE_COOLDOWN_SEC = float(
    os.getenv("LLM_PROVIDER_UNREACHABLE_COOLDOWN_SEC") or "120"
)


def _mark_provider_cooldown(provider_name: str, *, seconds: float | None = None) -> None:
    name = (provider_name or "").strip().lower()
    if not name:
        return
    wait = float(seconds) if seconds is not None else _PROVIDER_TPD_COOLDOWN_SEC
    until = time.monotonic() + max(15.0, wait)
    prev = float(_PROVIDER_COOLDOWN_UNTIL.get(name) or 0.0)
    if until > prev:
        _PROVIDER_COOLDOWN_UNTIL[name] = until
        log_reasoning(f"LLM provider {name} cooldown {wait:.0f}s (rate/TPD).")


def _note_provider_transport_failure(provider_name: str, *, hard_timeout: bool = False) -> None:
    """
    Timeout / network / stuck-TLS failure — cooldown so later calls skip a dead provider.

    hard_timeout=True (provider did not respond within the wall-clock budget) is a
    strong "stuck / unreachable" signal → cooldown immediately (one 20s stall is
    already too slow for a shopping assistant). Fast network errors use a short
    streak so a single blip does not disable a healthy provider.
    """
    name = (provider_name or "").strip().lower()
    if not name:
        return
    streak = int(_PROVIDER_FAIL_STREAK.get(name) or 0) + 1
    _PROVIDER_FAIL_STREAK[name] = streak
    if hard_timeout or streak >= _PROVIDER_UNREACHABLE_STREAK:
        _PROVIDER_FAIL_STREAK[name] = 0
        wait = _PROVIDER_UNREACHABLE_COOLDOWN_SEC
        until = time.monotonic() + max(15.0, wait)
        prev = float(_PROVIDER_COOLDOWN_UNTIL.get(name) or 0.0)
        if until > prev:
            _PROVIDER_COOLDOWN_UNTIL[name] = until
        reason = "stuck timeout" if hard_timeout else f"unreachable x{streak}"
        log_reasoning(
            f"LLM provider {name} {reason} — cooldown {wait:.0f}s "
            "(skip on next calls this session)."
        )


def _note_provider_success(provider_name: str) -> None:
    name = (provider_name or "").strip().lower()
    if name:
        _PROVIDER_FAIL_STREAK.pop(name, None)


def _provider_on_cooldown(provider_name: str) -> bool:
    name = (provider_name or "").strip().lower()
    until = float(_PROVIDER_COOLDOWN_UNTIL.get(name) or 0.0)
    if until <= 0:
        return False
    if time.monotonic() >= until:
        _PROVIDER_COOLDOWN_UNTIL.pop(name, None)
        return False
    return True


def filter_providers_not_on_cooldown(providers: list[dict]) -> list[dict]:
    """Drop cooled-down providers; if all cooled, return original chain."""
    if not providers:
        return providers
    alive = [p for p in providers if not _provider_on_cooldown(str(p.get("name") or ""))]
    return alive if alive else list(providers)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _safe_print(msg: str) -> None:
    try:
        print((msg or "").encode("ascii", errors="replace").decode("ascii"))
    except Exception:
        pass


def _model_for(provider: str) -> str:
    return _env(f"{provider.upper()}_MODEL") or _DEFAULT_MODELS.get(provider, "")


def _try_xai_grok() -> Optional[dict[str, Any]]:
    api_key = _env("XAI_API_KEY") or _env("GROK_API_KEY")
    if not api_key:
        return None
    return {
        "name": "grok",
        "url": _env("GROK_API_URL") or "https://api.x.ai/v1/chat/completions",
        "model": _model_for("grok"),
        "api_key": api_key,
    }


def _try_groq_cloud() -> Optional[dict[str, Any]]:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        return None
    return {
        "name": "groq",
        "url": _env("GROQ_API_URL") or "https://api.groq.com/openai/v1/chat/completions",
        "model": _model_for("groq"),
        "api_key": api_key,
    }


def _try_openai() -> Optional[dict[str, Any]]:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return None
    return {
        "name": "openai",
        "url": _env("OPENAI_API_URL") or "https://api.openai.com/v1/chat/completions",
        "model": _model_for("openai"),
        "api_key": api_key,
    }


def _try_gemini() -> Optional[dict[str, Any]]:
    api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    if not api_key:
        return None
    base = _env("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/openai"
    return {
        "name": "gemini",
        "url": base.rstrip("/") + "/chat/completions",
        "model": _model_for("gemini"),
        "api_key": api_key,
    }


def _try_deepseek() -> Optional[dict[str, Any]]:
    api_key = _env("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return {
        "name": "deepseek",
        "url": _env("DEEPSEEK_API_URL") or "https://api.deepseek.com/chat/completions",
        "model": _model_for("deepseek"),
        "api_key": api_key,
    }


_BUILDERS = {
    "grok": lambda: _try_xai_grok() or _try_groq_cloud(),
    "groq": _try_groq_cloud,
    "openai": _try_openai,
    "gemini": _try_gemini,
    "deepseek": _try_deepseek,
}


def provider_available(provider: str) -> bool:
    """Whether API credentials exist for this provider slot."""
    p = (provider or "").strip().lower()
    if p in ("grok",):
        return bool(_try_xai_grok() or _try_groq_cloud())
    builder = _BUILDERS.get(p)
    return bool(builder and builder())


def build_auto_provider_chain() -> list[dict[str, Any]]:
    """Env-based fallback chain (no admin pin)."""
    raw = _env("LLM_PROVIDER_ORDER")
    order = tuple(p.strip().lower() for p in raw.split(",") if p.strip()) if raw else _DEFAULT_ORDER
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        builder = _BUILDERS.get(name)
        if not builder:
            continue
        spec = builder()
        if spec:
            out.append(spec)
    if not out:
        fallback = _try_groq_cloud() or _try_deepseek()
        if fallback:
            out.append(fallback)
    return out


def build_pinned_provider_chain(model_key: str) -> list[dict[str, Any]]:
    """Single provider from admin key like gemini:gemini-2.5-flash."""
    key = (model_key or "").strip()
    if ":" not in key:
        return []
    provider, model = key.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        return []
    builder = _BUILDERS.get(provider)
    if not builder:
        return []
    spec = builder()
    if not spec:
        return []
    pinned = dict(spec)
    pinned["model"] = model
    return [pinned]


def get_configured_provider_chain() -> list[dict[str, Any]]:
    """
    Providers from LLM_PROVIDER_ORDER that have API keys in .env.
    Order is preserved exactly — groq first by default, then openai, gemini, deepseek.
    """
    return build_auto_provider_chain()


def get_standard_fallback_chain(*, max_providers: int | None = None) -> list[dict[str, Any]]:
    """
    Full auto-failover chain for JSON classifiers and routing.
    Uses admin Auto mode (env order) or a single pinned provider when admin pins a model.
    """
    chain = get_llm_provider_chain()
    if not chain:
        return []
    if max_providers is None:
        cap = max(1, min(4, int(os.getenv("LLM_MAX_PROVIDERS", "4") or "4")))
    else:
        cap = max(1, min(4, int(max_providers)))
    if len(chain) <= 1:
        return list(chain)
    return chain[:cap]


def get_fast_chitchat_provider_chain(*, max_providers: int = 1) -> list[dict[str, Any]]:
    """
    Prefer Groq Instant for greeting/chitchat/OOD replies (~sub-2s) when configured,
    then rest of admin/env chain. Does not change product/Brain routing providers.
    """
    chain = get_llm_provider_chain()
    if not chain:
        return []
    prefer = (os.getenv("CHITCHAT_AI_PREFER_PROVIDER") or "groq").strip().lower()
    preferred = [p for p in chain if (p.get("name") or "").strip().lower() == prefer]
    rest = [p for p in chain if (p.get("name") or "").strip().lower() != prefer]
    ordered = preferred + rest if preferred else list(chain)
    cap = max(1, min(2, int(max_providers or 1)))
    return ordered[:cap]


def get_fast_structured_provider_chain(*, max_providers: int = 1) -> list[dict[str, Any]]:
    """
    Low-latency JSON extraction chain for routing/product entities.

    Prefer the configured fast provider (Groq Instant by default) even when the
    Admin UI pins a slower general-answer model. Structured routing is on the
    request critical path and must have a small, deterministic wall-clock budget.
    """
    chain = get_llm_provider_chain()
    if not chain:
        return []
    prefer = (
        os.getenv("FAST_STRUCTURED_AI_PREFER_PROVIDER") or "groq"
    ).strip().lower()
    preferred = [
        p for p in chain if (p.get("name") or "").strip().lower() == prefer
    ]
    rest = [
        p for p in chain if (p.get("name") or "").strip().lower() != prefer
    ]
    ordered = preferred + rest if preferred else list(chain)
    cap = max(1, min(2, int(max_providers or 1)))
    return filter_providers_not_on_cooldown(ordered)[:cap]


def provider_chain_label(chain: list[dict[str, Any]] | None = None) -> str:
    """Human-readable chain for logs, e.g. groq→openai→gemini→deepseek."""
    specs = chain if chain is not None else get_standard_fallback_chain()
    if not specs:
        return "-"
    return "→".join((p.get("name") or "?") for p in specs)


def get_llm_provider_chain() -> list[dict[str, Any]]:
    """Respects admin Auto vs pinned model; falls back to env chain."""
    try:
        from services.agent_llm_settings import resolve_runtime_llm_providers

        chain = resolve_runtime_llm_providers()
        if chain:
            return chain
    except Exception:
        pass
    return get_configured_provider_chain()


def _trim_text_mid(text: str, max_chars: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    keep_head = max(200, int(max_chars * 0.45))
    keep_tail = max(260, max_chars - keep_head - 24)
    return f"{raw[:keep_head].rstrip()}\n...[truncated]...\n{raw[-keep_tail:].lstrip()}"


def _shrink_payload(req: dict, factor: float = 0.72) -> dict:
    out = dict(req or {})
    msgs = list(out.get("messages") or [])
    shrunk = []
    for idx, m in enumerate(msgs):
        content = (m or {}).get("content") or ""
        if isinstance(content, str):
            content = _trim_text_mid(content, 6500 if idx == 0 else 1800)
        shrunk.append({"role": (m or {}).get("role") or "user", "content": content})
    out["messages"] = shrunk
    out["max_tokens"] = max(140, int((out.get("max_tokens") or 320) * factor))
    return out


def _extract_retry_wait_seconds(error_text: str) -> float:
    if not error_text:
        return 1.5
    m = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if not m:
        return 1.5
    try:
        return max(0.5, min(8.0, float(m.group(1)) + 0.2))
    except Exception:
        return 1.5


def llm_json_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    timeout_sec: int = 12,
    max_attempts: int = 3,
    provider_name: str = "provider",
) -> Optional[dict]:
    """Single-provider chat completion with JSON response; retries transient errors."""
    from services.chat_resilience import classify_api_error, set_last_llm_failure

    req = dict(payload or {})
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                from services.chat_resilience import chat_turn_abandoned

                if chat_turn_abandoned():
                    set_last_llm_failure("deadline_abandoned")
                    return None
            except ImportError:
                pass
            try:
                from services.chat_flow_telemetry import (
                    increment_llm_call,
                    llm_budget_exceeded,
                )

                if llm_budget_exceeded():
                    set_last_llm_failure("llm_budget_exceeded")
                    return None
                increment_llm_call(provider_name, billable=(attempt == 1))
                if llm_budget_exceeded():
                    set_last_llm_failure("llm_budget_exceeded")
                    return None
            except ImportError:
                pass
            session = requests.Session()
            session.trust_env = False
            try:
                read_to = float(timeout_sec)
            except (TypeError, ValueError):
                read_to = 12.0
            # Always enforce a hard wall-clock budget. Socket timeouts alone can
            # hang 30–70s on stuck TLS/providers (esp. OpenAI) and poison later turns.
            if read_to <= 5.0:
                read_to = max(1.0, min(5.0, read_to))
            else:
                read_to = max(5.0, min(20.0, read_to))

            def _do_post():
                return session.post(
                    url, headers=headers, json=req, timeout=(3.0, read_to)
                )

            # Wall-clock hard stop for every call — cancel futures without waiting
            # so a stuck HTTP body cannot block Flask for tens of seconds.
            import concurrent.futures

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            timed_out = False
            try:
                fut = pool.submit(_do_post)
                try:
                    res = fut.result(timeout=read_to + 0.75)
                except concurrent.futures.TimeoutError:
                    timed_out = True
                    set_last_llm_failure("timeout")
                    _note_provider_transport_failure(provider_name, hard_timeout=True)
                    log_reasoning(
                        f"{provider_name} hard timeout ({read_to}s) — next provider."
                    )
                    return None
            finally:
                # On success wait briefly so the worker exits cleanly; on timeout
                # abandon immediately so Flask is not blocked by stuck TLS.
                try:
                    pool.shutdown(wait=not timed_out, cancel_futures=True)
                except TypeError:
                    pool.shutdown(wait=False)
            if res.status_code == 200:
                body = res.json()
                content = (
                    (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                    .strip()
                )
                if not content:
                    continue
                try:
                    from services.ai_route_semantics import strip_markdown_json_fence

                    content = strip_markdown_json_fence(content)
                    parsed = json.loads(content)
                    _note_provider_success(provider_name)
                    return parsed
                except Exception:
                    req["max_tokens"] = max(120, int(req.get("max_tokens", 300) * 0.7))
                    time.sleep(0.5)
                    continue

            text = res.text or ""
            low = text.lower()
            is_rate = res.status_code == 429 or "rate_limit_exceeded" in low
            is_json_fail = "json_validate_failed" in low or "failed to generate json" in low
            if "request too large for model" in low and attempt < max_attempts:
                log_reasoning(f"{provider_name} request too large — shrinking prompt and retrying.")
                req = _shrink_payload(req)
                time.sleep(0.5)
                continue
            if is_rate:
                kind = classify_api_error(res.status_code, text)
                set_last_llm_failure(kind)
                if "tokens per day" in low or "tpd" in low or "token limit" in low:
                    log_reasoning(f"{provider_name} daily token limit exceeded — skip retries.")
                    _safe_print(f"{provider_name} API Error: {text[:400]}")
                    _mark_provider_cooldown(provider_name)
                    return None
                if attempt < max_attempts and attempt == 1:
                    wait_sec = min(2.0, _extract_retry_wait_seconds(text))
                    log_reasoning(
                        f"{provider_name} rate-limited; quick retry in {wait_sec:.1f}s"
                    )
                    time.sleep(wait_sec)
                    continue
                # Short RPM cooldown so the next call in the chain / turn skips this provider.
                _mark_provider_cooldown(provider_name, seconds=min(45.0, _extract_retry_wait_seconds(text) + 5.0))
                return None
            if is_json_fail and attempt < max_attempts:
                prev = int(req.get("max_tokens", 300) or 300)
                req["max_tokens"] = min(900, max(prev + 180, 420))
                log_reasoning(
                    f"{provider_name} JSON validate failed — retry with max_tokens={req['max_tokens']}."
                )
                time.sleep(0.35)
                continue
            if is_json_fail:
                log_reasoning(
                    f"{provider_name} JSON validate failed — switching to next provider."
                )
            kind = classify_api_error(res.status_code, text)
            if kind != "error":
                set_last_llm_failure(kind)
            _safe_print(f"{provider_name} API Error: {text[:400]}")
            return None
        except requests.exceptions.Timeout:
            set_last_llm_failure("timeout")
            _note_provider_transport_failure(provider_name, hard_timeout=True)
            log_reasoning(f"{provider_name} request timeout ({timeout_sec}s) — next provider.")
            return None
        except requests.exceptions.RequestException as e:
            set_last_llm_failure("timeout")
            _note_provider_transport_failure(provider_name)
            log_reasoning(f"{provider_name} network error: {e}")
            return None
        except Exception as e:
            set_last_llm_failure("error")
            _safe_print(f"LLM Error ({provider_name}): {e}")
            return None
    return None


def llm_json_with_provider_fallback(
    providers: list[dict],
    messages: list[dict],
    max_tokens: int,
    timeout_sec: int = 12,
    max_attempts: int = 3,
    temperature: float = 0.0,
) -> Optional[dict]:
    from services.chat_resilience import clear_last_llm_failure, set_last_llm_failure

    clear_last_llm_failure()
    if not providers:
        set_last_llm_failure("all_failed")
        _safe_print(
            "ERROR: No LLM provider configured. Set GROQ_API_KEY, OPENAI_API_KEY, "
            "GEMINI_API_KEY, and/or DEEPSEEK_API_KEY."
        )
        return None
    providers = filter_providers_not_on_cooldown(providers)
    if len(providers) > 1:
        log_reasoning(
            f"LLM fallback chain: {' → '.join(p.get('name') or '?' for p in providers)}"
        )
    rate_seen = False
    for idx, p in enumerate(providers):
        headers = {
            "Authorization": f"Bearer {p['api_key']}",
            "Content-Type": "application/json",
        }
        msg_list = list(messages)
        tok = max_tokens
        prov_url = (p.get("url") or "").lower()
        if p["name"] in ("groq",) or (p["name"] == "grok" and "groq.com" in prov_url):
            shrunk = _shrink_payload({"messages": msg_list, "max_tokens": tok}, factor=0.72)
            msg_list = shrunk["messages"]
            # Shrink prompt only — keep output budget so Groq JSON mode can finish.
            tok = max(int(tok), int(shrunk.get("max_tokens") or tok))
        payload = {
            "model": p["model"],
            "messages": msg_list,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": tok,
        }
        attempts = max(1, int(max_attempts)) if idx == 0 else max(1, min(2, int(max_attempts)))
        if rate_seen:
            attempts = 1
        out = llm_json_with_retry(
            p["url"],
            headers,
            payload,
            timeout_sec=timeout_sec,
            max_attempts=attempts,
            provider_name=p["name"],
        )
        if out:
            clear_last_llm_failure()
            if idx > 0:
                log_reasoning(f"LLM fallback success via {p['name']}.")
            else:
                log_reasoning(f"LLM routing success via {p['name']}.")
            return out
        from services.chat_resilience import get_last_llm_failure

        failure = get_last_llm_failure() or ""
        if failure == "llm_budget_exceeded":
            log_reasoning(
                "LLM call budget reached for this turn — not an API key problem. "
                "Raise CHAT_MAX_LLM_CALLS or reduce duplicate classifiers."
            )
            break
        if failure in ("rate_limit", "busy"):
            rate_seen = True
        if idx + 1 < len(providers):
            nxt = providers[idx + 1]["name"]
            log_reasoning(f"LLM provider {p['name']} unavailable; trying {nxt}.")
    set_last_llm_failure("all_failed" if not rate_seen else "rate_limit")
    return None


def llm_json_chat_completion(
    messages: list[dict],
    *,
    max_tokens: int = 380,
    timeout_sec: int = 12,
    max_attempts: int = 3,
    temperature: float = 0.0,
) -> Optional[dict]:
    """Chat completion with JSON response across the env-configured provider chain."""
    return llm_json_with_provider_fallback(
        get_standard_fallback_chain(),
        messages,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        temperature=temperature,
    )
