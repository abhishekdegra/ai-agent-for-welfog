"""
Turn Groq routing JSON into an execution plan: KB, live API, catalog, AI synthesis — any language.
Keyword helpers only fill gaps when LLM routing is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from utils.reasoning_log import log_reasoning


@dataclass
class SemanticAnswerPlan:
    intent: str = "general"
    answer_strategy: str = "kb_then_ai"
    use_kb: bool = True
    use_live_api: bool = False
    use_ai_synthesis: bool = True
    use_catalog: bool = False
    kb_keys: list[str] = field(default_factory=list)
    handler_hint: str = ""
    needs_order_id: bool = False
    search_query: str = ""
    reasoning: str = ""


def _normalize_strategy(raw: str) -> str:
    s = (raw or "").strip().lower().replace(" ", "_").replace("+", "_")
    aliases = {
        "live_api": "live_api_only",
        "api": "live_api_only",
        "api_only": "live_api_only",
        "kb": "kb_only",
        "kb_ai": "kb_then_ai",
        "api_ai": "api_then_ai",
        "api_kb_ai": "api_kb_ai",
        "api_kb": "api_kb_ai",
        "catalog": "catalog_only",
        "structured": "structured_handler",
    }
    return aliases.get(s, s or "kb_then_ai")


def build_semantic_answer_plan(
    route: dict | None,
    *,
    handler: str = "",
) -> SemanticAnswerPlan:
    """Prefer AI answer_strategy + data_channel; infer only when absent."""
    r = route if isinstance(route, dict) else {}
    intent = (r.get("intent") or "general").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    strategy = _normalize_strategy((r.get("answer_strategy") or "").strip())
    kb_keys = [k for k in (r.get("kb_keys") or []) if k]
    h = (handler or r.get("route_handler") or "").strip()

    if not (r.get("answer_strategy") or "").strip():
        if channel == "live_api":
            if intent in ("order", "refund", "payment"):
                strategy = "api_then_ai" if r.get("needs_order_id") else "live_api_only"
            elif intent in ("order_history", "wishlist", "deals", "categories", "category_feed", "pincode_check"):
                strategy = "live_api_only"
            else:
                strategy = "live_api_only"
        elif channel == "catalog":
            strategy = "catalog_only"
        elif channel == "kb":
            strategy = "kb_then_ai"
        elif channel == "none":
            strategy = "structured_handler"

    plan = SemanticAnswerPlan(
        intent=intent,
        answer_strategy=strategy,
        kb_keys=kb_keys,
        handler_hint=h,
        needs_order_id=bool(r.get("needs_order_id")),
        search_query=(r.get("search_query") or "").strip(),
        reasoning=(r.get("reasoning") or "")[:240],
    )

    if strategy in ("live_api_only", "api_then_ai", "api_kb_ai"):
        plan.use_live_api = intent in (
            "order",
            "refund",
            "payment",
            "order_history",
            "wishlist",
            "pincode_check",
            "deals",
            "categories",
            "category_feed",
        )
    if strategy == "live_api_only":
        plan.use_kb = False
        plan.use_ai_synthesis = False
    if strategy in ("kb_only", "kb_then_ai", "api_kb_ai"):
        plan.use_kb = True
    if strategy == "kb_only":
        plan.use_ai_synthesis = False
    if strategy in ("kb_then_ai", "api_then_ai", "api_kb_ai"):
        plan.use_ai_synthesis = True
    if strategy == "api_then_ai":
        plan.use_kb = False
    if strategy == "catalog_only":
        plan.use_catalog = True
        plan.use_kb = False
        plan.use_live_api = False
        plan.use_ai_synthesis = False
    if strategy == "structured_handler":
        plan.use_ai_synthesis = False

    log_reasoning(
        f"Answer plan: intent={plan.intent} strategy={plan.answer_strategy} "
        f"kb={plan.use_kb} api={plan.use_live_api} ai={plan.use_ai_synthesis}"
    )
    return plan


def gather_kb_context_for_plan(
    original_msg: str,
    msg_en: str,
    plan: SemanticAnswerPlan,
    *,
    conversation_context: str = "",
) -> str:
    from services.kb_service import (
        read_concatenated_kb_file_contents,
        resolve_kb_keys_for_question,
        _filter_kb_blob_for_question,
        _strip_kb_blob_for_display,
    )

    keys = plan.kb_keys or resolve_kb_keys_for_question(
        original_msg, msg_en, suggested_keys=plan.kb_keys, max_files=4
    )
    try:
        from services.query_understanding import filter_kb_keys_for_intent

        keys = (
            filter_kb_keys_for_intent(
                keys,
                plan.intent,
                user_meaning=f"{original_msg} {msg_en} {plan.reasoning}".strip(),
            )
            if keys
            else keys
        )
    except ImportError:
        pass
    if not keys:
        return ""
    blob = read_concatenated_kb_file_contents(keys)
    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    filtered = _filter_kb_blob_for_question(blob, combined, max_chars=1100)
    if not filtered.strip() and blob.strip():
        filtered = _strip_kb_blob_for_display(blob)[:1100]
    if not filtered.strip():
        return ""
    return f"KNOWLEDGE BASE ({', '.join(keys)}):\n{filtered}"


def synthesize_grounded_reply(
    original_msg: str,
    msg_en: str,
    *,
    kb_context: str = "",
    live_facts: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """
    One LLM pass: user question + KB excerpts + live API facts → reply in customer's language.
    """
    from services.ai_service import ai_brain_answer
    from services.translation_service import finalize_customer_reply, resolve_customer_reply_lang

    parts: list[str] = []
    if live_facts.strip():
        parts.append(f"LIVE ACCOUNT/API DATA (authoritative — prefer over KB if conflict):\n{live_facts.strip()}")
    if kb_context.strip():
        parts.append(kb_context.strip())
    if not parts:
        return ""

    combined_context = "\n\n---\n\n".join(parts)
    ai = ai_brain_answer(
        original_msg,
        combined_context,
        conversation_context,
        reply_lang=reply_lang,
    ) or {}
    body = (ai.get("response") or "").strip()
    if not body:
        return ""
    rl = resolve_customer_reply_lang(original_msg, reply_lang)
    return finalize_customer_reply(body, original_msg, rl)


def try_semantic_grounded_reply(
    original_msg: str,
    msg_en: str,
    ai_route: dict | None,
    *,
    conversation_context: str = "",
    reply_lang: str = "en",
    handler: str = "",
    live_facts: str = "",
) -> str:
    """
    KB (+ optional live API facts) + AI synthesis in the customer's language.
    Returns empty when plan says deterministic-only or synthesis fails.
    """
    plan = build_semantic_answer_plan(ai_route, handler=handler)
    if plan.answer_strategy in ("kb_only", "live_api_only", "catalog_only", "structured_handler"):
        return ""
    if not plan.use_ai_synthesis:
        return ""
    kb_ctx = ""
    if plan.use_kb:
        kb_ctx = gather_kb_context_for_plan(
            original_msg,
            msg_en,
            plan,
            conversation_context=conversation_context,
        )
    if not kb_ctx.strip() and not (live_facts or "").strip():
        return ""
    return synthesize_grounded_reply(
        original_msg,
        msg_en,
        kb_context=kb_ctx,
        live_facts=live_facts,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
    )
