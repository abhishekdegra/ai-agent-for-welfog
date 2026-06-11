"""
Admin-controlled LLM selection (SQLite). Customers never set this.

- auto: full fallback chain from env (grok/groq → openai → gemini → deepseek)
- provider:model: single pinned model for all JSON chat completions
"""
from __future__ import annotations

from typing import Any, Optional

from utils.reasoning_log import log_reasoning

# Keys allowed in admin UI (provider:model)
PINNED_MODEL_KEYS = (
    "groq:llama-3.1-8b-instant",
    "grok:grok-3-mini",
    "openai:gpt-4.1-mini",
    "gemini:gemini-2.5-flash",
    "deepseek:deepseek-chat",
)

AUTO_KEY = "auto"


def get_selectable_llm_options() -> list[dict[str, Any]]:
    """Options for admin settings UI."""
    from services.llm_providers import provider_available

    auto_chain = describe_auto_chain()
    options: list[dict[str, Any]] = [
        {
            "key": AUTO_KEY,
            "label": "Auto",
            "subtitle": auto_chain,
            "group": None,
            "available": True,
        },
        {
            "key": "groq:llama-3.1-8b-instant",
            "label": "Llama 3.1 8B Instant",
            "subtitle": "Groq Cloud · fast routing & product search",
            "group": "Groq",
            "available": provider_available("groq"),
        },
        {
            "key": "grok:grok-3-mini",
            "label": "Grok 3 Mini",
            "subtitle": "xAI Grok (or Groq if no xAI key)",
            "group": "Grok",
            "available": provider_available("grok"),
        },
        {
            "key": "openai:gpt-4.1-mini",
            "label": "GPT-4.1 Mini",
            "subtitle": "OpenAI ChatGPT",
            "group": "OpenAI",
            "available": provider_available("openai"),
        },
        {
            "key": "gemini:gemini-2.5-flash",
            "label": "Gemini 2.5 Flash",
            "subtitle": "Google Gemini",
            "group": "Gemini",
            "available": provider_available("gemini"),
        },
        {
            "key": "deepseek:deepseek-chat",
            "label": "DeepSeek Chat",
            "subtitle": "DeepSeek",
            "group": "DeepSeek",
            "available": provider_available("deepseek"),
        },
    ]
    return options


def describe_auto_chain() -> str:
    from services.llm_providers import build_auto_provider_chain

    chain = build_auto_provider_chain()
    if not chain:
        return "Configure API keys in .env"
    return " → ".join(f"{p['name']}" for p in chain)


def get_admin_llm_model_key() -> str:
    """Current admin selection; default auto."""
    try:
        from admin_models import AgentSettings
        from extensions import db

        row = AgentSettings.query.get(1)
        if not row:
            row = AgentSettings(id=1, llm_model_key=AUTO_KEY)
            db.session.add(row)
            db.session.commit()
        key = (row.llm_model_key or AUTO_KEY).strip()
        return key if _is_valid_model_key(key) else AUTO_KEY
    except Exception:
        return AUTO_KEY


def set_admin_llm_model_key(model_key: str) -> tuple[bool, str]:
    key = (model_key or "").strip()
    if not _is_valid_model_key(key):
        return False, "Invalid model selection."
    if key != AUTO_KEY:
        provider = key.split(":", 1)[0]
        from services.llm_providers import provider_available

        if not provider_available(provider):
            return False, "Selected model has no API key configured in .env."

    try:
        from admin_models import AgentSettings
        from extensions import db

        row = AgentSettings.query.get(1)
        if not row:
            row = AgentSettings(id=1)
            db.session.add(row)
        row.llm_model_key = key
        db.session.commit()
        return True, "AI model preference saved."
    except Exception as e:
        return False, f"Could not save: {e}"


def _is_valid_model_key(key: str) -> bool:
    if key == AUTO_KEY:
        return True
    return key in PINNED_MODEL_KEYS


def option_for_key(model_key: str) -> Optional[dict[str, Any]]:
    for opt in get_selectable_llm_options():
        if opt["key"] == model_key:
            return opt
    return None


def resolve_runtime_llm_providers() -> list[dict[str, Any]]:
    """Provider list used by all agent LLM JSON calls."""
    from services.llm_providers import build_auto_provider_chain, build_pinned_provider_chain

    key = get_admin_llm_model_key()
    if key == AUTO_KEY:
        chain = build_auto_provider_chain()
        if chain:
            names = " → ".join(f"{p['name']}({p['model']})" for p in chain)
            log_reasoning(f"LLM mode Auto: {names}")
        return chain
    chain = build_pinned_provider_chain(key)
    if chain:
        p = chain[0]
        log_reasoning(f"LLM mode pinned [admin]: {p['name']}({p['model']})")
    return chain
