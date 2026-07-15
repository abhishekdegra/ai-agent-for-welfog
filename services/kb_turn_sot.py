"""
KB Turn Single Source of Truth (SoT).

Industry pattern:
  ONE Brain JSON (channel=kb / order_help_kind=nav_howto)
    → vector retrieve top chunks → extractive answer in customer language
    → polite refuse if no grounded hit (never invent)
  No product-rescue LLM after KB is authoritative.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from utils.reasoning_log import log_reasoning


def route_is_locked_kb_turn(route: dict | None) -> bool:
    if not isinstance(route, dict):
        return False
    if route.get("_kb_route_locked") or route.get("_order_nav_howto_locked"):
        return True
    try:
        from services.ai_route_semantics import brain_route_is_order_nav_howto

        if brain_route_is_order_nav_howto(route):
            return True
    except ImportError:
        pass
    try:
        from services.chat_flow_telemetry import brain_route_authoritative_kb_lock

        if brain_route_authoritative_kb_lock(route):
            return True
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import brain_route_prefers_kb_answer

        return brain_route_prefers_kb_answer(route)
    except ImportError:
        return (route.get("data_channel") or "").strip().lower() == "kb"


def finalize_kb_lock(route: dict) -> dict:
    out = dict(route or {})
    out["data_channel"] = "kb"
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["_kb_route_locked"] = True
    out["_ai_single_pass"] = True
    out.pop("_product_catalog_locked", None)
    out.pop("search_query", None)
    try:
        from services.chat_flow_telemetry import lock_authoritative_kb_route_from_brain

        lock_authoritative_kb_route_from_brain(out)
    except ImportError:
        pass
    return out


def kb_turn_blocks_product_rescue(route: dict | None) -> bool:
    """When True, product rescue / catalog must not run on this turn."""
    if not route_is_locked_kb_turn(route):
        return False
    try:
        from services.ai_route_semantics import (
            _brain_route_has_shopping_entities,
            brain_route_indicates_product_catalog,
        )
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route) or brain_route_indicates_product_catalog(
            route
        ):
            return False
        if _brain_route_has_shopping_entities(route) and route.get("run_catalog_search"):
            return False
    except ImportError:
        pass
    return True


def _dispatch_kb_locked_reply(
    locked: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable[..., str],
) -> Optional[str]:
    try:
        from services.chat_resilience import chat_turn_abandoned

        if chat_turn_abandoned():
            return None
    except ImportError:
        pass
    try:
        from services.brain_direct_dispatch import _try_brain_kb_locked_reply

        return _try_brain_kb_locked_reply(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        )
    except Exception as exc:
        log_reasoning(f"KB SoT dispatch skipped: {exc}")
        return None


def try_run_locked_kb_reply(
    route: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable[..., str],
) -> tuple[Optional[str], Optional[dict]]:
    """
    Authoritative KB from Brain JSON — no product rescue LLM before answer.
    channel=kb → one Qdrant retrieve + extractive reply (no second semantic stack).
    """
    try:
        from services.chat_resilience import chat_turn_abandoned

        if chat_turn_abandoned():
            return None, None
    except ImportError:
        pass

    route = dict(route or {})
    ch0 = (route.get("data_channel") or "").strip().lower()

    # Product catalog already authoritative — never burn KB probes/reconcile here.
    # Real logs: shopping turns spent 2–25s in KB SoT then answered catalog anyway.
    intent0 = (route.get("intent") or "").strip().lower()
    if (
        ch0 == "catalog"
        or route.get("_product_catalog_locked")
        or route.get("run_catalog_search")
        or intent0 in ("product", "product_search")
    ):
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked
            from services.ai_route_semantics import brain_route_indicates_product_catalog

            if (
                product_catalog_route_is_locked(route)
                or brain_route_indicates_product_catalog(route)
                or ch0 == "catalog"
            ):
                log_reasoning(
                    "KB SoT — skip probes; Brain already locked product catalog."
                )
                return None, None
        except ImportError:
            if ch0 == "catalog" or route.get("_product_catalog_locked"):
                log_reasoning(
                    "KB SoT — skip probes; catalog channel / product lock set."
                )
                return None, None

    # Hot path: Brain already locked KB — skip informational embedding probes.
    if ch0 == "kb" and not route.get("needs_order_id"):
        locked = finalize_kb_lock(route)
        body = _dispatch_kb_locked_reply(
            locked,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context_fn,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
        )
        if body:
            log_reasoning(
                "KB SoT — channel=kb lock → extractive answer (no probe/semantic stack)."
            )
            return body, locked
        # Always return locked route so callers skip product rescue even on gap.
        return None, locked

    try:
        from services.ai_route_semantics import (
            brain_route_is_order_nav_howto,
            reconcile_order_nav_howto_from_brain_meaning,
            reconcile_welfog_kb_from_brain_meaning,
            brain_route_indicates_informational_kb,
            brain_route_prefers_kb_answer,
        )

        route = reconcile_order_nav_howto_from_brain_meaning(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        if brain_route_prefers_kb_answer(route) or brain_route_is_order_nav_howto(route):
            route = reconcile_welfog_kb_from_brain_meaning(
                route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
        elif brain_route_indicates_informational_kb(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        ):
            route = reconcile_welfog_kb_from_brain_meaning(
                route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
    except ImportError:
        pass

    if not kb_turn_blocks_product_rescue(route):
        return None, None

    locked = finalize_kb_lock(route)
    body = _dispatch_kb_locked_reply(
        locked,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if body:
        log_reasoning(
            "KB SoT — locked KB → extractive / howto answer (no product rescue)."
        )
        return body, locked
    return None, locked
