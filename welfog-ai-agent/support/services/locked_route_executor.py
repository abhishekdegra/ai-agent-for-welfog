"""
Single-pass execution after AI routing is locked.

Flow: understand once (ai_brain_route) → lock route → execute handler → reply.
No secondary intent classifiers, no legacy cascade, no duplicate KB/LLM passes.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from services.answer_router import AnswerRouteDecision, dispatch_early_answer
from utils.reasoning_log import log_reasoning

_KB_AI_HANDLERS = frozenset(
    {
        "ai_route_and_answer",
        "dynamic_kb",
        "kb_grounded_ai",
        "knowledge_topic_kb",
        "policy_structured_kb",
        "seller_kb",
    }
)

_SCOPE_HANDLERS = frozenset(
    {
        "off_topic",
        "other_company_decline",
        "temporary_load",
        "warm_feedback",
        "warm_greeting",
    }
)


def _record_dispatch(handler: str, reason: str = "") -> None:
    try:
        from services.chat_flow_telemetry import record_route, record_route_step

        record_route_step("locked_dispatch")
        record_route(source=handler)
        if reason:
            from services.chat_flow_telemetry import _TLS

            _TLS.route_reason = reason[:300]
    except ImportError:
        pass


def _try_meta_reply(
    ai_route: dict | None,
    original_msg: str,
    reply_lang: str,
) -> Optional[str]:
    if not ai_route:
        return None
    try:
        from services.ai_route_semantics import meta_turn_from_route
        from services.turn_intent_gate import format_meta_turn_reply

        meta = meta_turn_from_route(ai_route)
        if not meta:
            return None
        if (meta.kind or "").strip().lower() == "assistant_intro":
            try:
                from utils.helpers import message_is_welfog_about_request
                from services.user_query_semantics import query_is_welfog_company_or_platform

                if message_is_welfog_about_request(
                    original_msg
                ) or query_is_welfog_company_or_platform(original_msg, original_msg):
                    log_reasoning(
                        "Locked dispatch: skip assistant_intro meta — company/about KB."
                    )
                    return None
            except ImportError:
                pass
        body = format_meta_turn_reply(meta, original_msg, reply_lang=reply_lang)
        if body:
            log_reasoning(f"Locked dispatch: meta_kind={meta.kind}.")
            _record_dispatch(f"meta_{meta.kind}", "structured meta reply")
            return body
    except ImportError:
        pass
    return None


def _try_scope_reply_from_route(
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    reply_lang: str,
    route_handler: str,
) -> Optional[str]:
    if not ai_route:
        return None
    scope_reply = (ai_route.get("scope_reply") or "").strip()
    scope = (ai_route.get("conversation_scope") or "").strip().lower()
    if scope_reply and scope in (
        "general_chitchat",
        "out_of_domain",
        "harm_sensitive",
    ):
        log_reasoning(f"Locked dispatch: brain scope_reply ({scope}).")
        _record_dispatch(route_handler or scope, "scope_reply from router")
        return scope_reply
    try:
        from services.conversation_scope import try_conversation_scope_reply

        body = try_conversation_scope_reply(
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
            reply_lang=reply_lang,
            ai_route=ai_route,
            route_handler=route_handler,
        )
        if body:
            log_reasoning("Locked dispatch: conversation scope reply.")
            _record_dispatch(route_handler or "scope", "conversation_scope")
            return body
    except ImportError:
        pass
    return None


def _narrow_route_kb_keys(
    ai_route: dict | None,
    route_decision: AnswerRouteDecision,
    original_msg: str,
    msg_en: str,
) -> list[str]:
    """Scope brain kb_keys to the detected topic (payment, refund, etc.) — max 3 files."""
    keys = [
        k
        for k in (
            list(route_decision.kb_keys or [])
            or list((ai_route or {}).get("kb_keys") or [])
        )
        if k
    ]
    if not keys:
        return []
    try:
        from services.query_understanding import (
            filter_kb_keys_for_intent,
            infer_kb_query_category,
            scoped_kb_keys_for_retrieval,
        )

        cat = infer_kb_query_category(
            original_msg,
            msg_en,
            ai_route=ai_route,
        )
        meaning = ((ai_route or {}).get("user_meaning") or "").strip()
        filtered = filter_kb_keys_for_intent(keys, cat, user_meaning=meaning)
        if filtered:
            narrowed = list(dict.fromkeys(filtered))
            if cat == "payment" and "payment" in narrowed:
                return ["payment"]
            if cat == "refund" and "refund" in narrowed:
                return ["refund"]
            return narrowed[:2]
        scoped = scoped_kb_keys_for_retrieval(
            cat, ai_route=ai_route, user_meaning=meaning or f"{original_msg} {msg_en}".strip()
        )
        if scoped:
            return scoped[:3]
    except ImportError:
        pass
    return keys[:3]


def _brain_keys_fast_reply(
    ai_route: dict | None,
    route_decision: AnswerRouteDecision,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    reply_lang: str,
) -> Optional[str]:
    """
    Read only brain-selected KB files (no full-corpus vector scan) then one grounded LLM answer.
    """
    keys = _narrow_route_kb_keys(ai_route, route_decision, original_msg, msg_en)
    if not keys:
        return None
    try:
        from services.kb_service import read_concatenated_kb_file_contents
        from services.semantic_answer_plan import synthesize_grounded_reply

        blob = read_concatenated_kb_file_contents(keys)
        if not blob.strip():
            return None
        if len(blob) > 4500:
            blob = blob[:4500] + "\n...[truncated]..."
        um = ((ai_route or {}).get("user_meaning") or original_msg or "").strip()
        kb_ctx = (
            f"KNOWLEDGE BASE FILES ({', '.join(keys)}):\n{blob}\n\n"
            f"CUSTOMER QUESTION (answer only from text above): {um}"
        )
        body = synthesize_grounded_reply(
            original_msg,
            msg_en,
            kb_context=kb_ctx,
            conversation_context=conv_for_llm,
            reply_lang=reply_lang,
        )
        if body:
            log_reasoning(
                f"Locked dispatch: brain-key KB read ({', '.join(keys[:3])}) + one LLM answer."
            )
            _record_dispatch(
                (route_decision.handler or "ai_route_and_answer"),
                f"kb_keys={','.join(keys[:3])}",
            )
            return body
    except ImportError:
        pass
    return None


def _try_kb_ai_synthesis(
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    retrieval_query: str,
    reply_lang: str,
) -> Optional[str]:
    handler = (route_decision.handler or "").strip()
    if handler not in _KB_AI_HANDLERS and route_decision.source not in (
        "kb",
        "kb_ai",
        "ai",
    ):
        return None

    comb = f"{original_msg} {msg_en}".strip()

    narrow_keys = _narrow_route_kb_keys(
        ai_route, route_decision, original_msg, msg_en
    )
    if narrow_keys:
        try:
            from services.kb_service import format_kb_answer_from_brain_keys

            direct = format_kb_answer_from_brain_keys(
                original_msg,
                msg_en,
                narrow_keys,
                reply_lang=reply_lang,
                conversation_context=conv_for_llm,
            )
            if direct:
                log_reasoning(
                    f"Locked dispatch: direct KB ({', '.join(narrow_keys)}) — no extra LLM."
                )
                _record_dispatch(
                    route_decision.handler or "dynamic_kb",
                    f"kb_keys={','.join(narrow_keys)}",
                )
                return direct
        except ImportError:
            pass

    fast = _brain_keys_fast_reply(
        ai_route, route_decision, original_msg, msg_en, conv_for_llm, reply_lang
    )
    if fast:
        return fast

    try:
        from utils.helpers import (
            _text_asks_customer_care_contact,
            message_needs_human_support_escalation,
        )
        from services.kb_service import format_customer_care_reply_from_kb

        if _text_asks_customer_care_contact(comb):
            cc = format_customer_care_reply_from_kb(original_msg, msg_en)
            if cc:
                log_reasoning("Locked dispatch: customer care KB (deterministic).")
                _record_dispatch("customer_care_kb", "deterministic contact KB")
                return cc
        if message_needs_human_support_escalation(comb):
            from services.kb_service import format_support_escalation_reply_from_kb

            esc = format_support_escalation_reply_from_kb(
                original_msg, msg_en, reply_lang=reply_lang
            )
            if esc:
                log_reasoning("Locked dispatch: support escalation KB.")
                _record_dispatch("support_escalation_kb", "escalation KB")
                return esc
    except ImportError:
        pass

    try:
        from services.kb_service import (
            format_dynamic_kb_answer,
            format_seller_reply_from_kb,
            format_welfog_about_reply_from_kb,
            direct_kb_search,
            get_customer_kb_keys,
        )
        from utils.helpers import message_is_welfog_about_request

        if handler == "welfog_about_kb" or message_is_welfog_about_request(comb):
            about = format_welfog_about_reply_from_kb(
                original_msg, msg_en, reply_lang=reply_lang, conversation_context=conv_for_llm
            )
            if about:
                log_reasoning("Locked dispatch: Welfog company/about KB.")
                _record_dispatch("welfog_about_kb", "company knowledge")
                return about

        if handler == "seller_kb" or (route_decision.intent or "") == "seller":
            seller = format_seller_reply_from_kb(
                original_msg, msg_en, reply_lang=reply_lang, conversation_context=conv_for_llm
            )
            if seller:
                log_reasoning("Locked dispatch: seller KB.")
                _record_dispatch("seller_kb", "seller knowledge")
                return seller

        kb_det = format_dynamic_kb_answer(
            original_msg,
            msg_en,
            reply_lang=reply_lang,
            conversation_context=conv_for_llm,
            ai_route=ai_route,
            suggested_keys=list(route_decision.kb_keys or []),
        )
        if kb_det:
            log_reasoning("Locked dispatch: dynamic KB formatter.")
            _record_dispatch(handler or "dynamic_kb", "deterministic KB match")
            return kb_det

        direct = direct_kb_search(
            retrieval_query or comb,
            keys=route_decision.kb_keys or get_customer_kb_keys(),
            min_score=0.30,
        )
        if direct:
            log_reasoning("Locked dispatch: direct KB search.")
            _record_dispatch(handler or "direct_kb", "vector KB search")
            return direct
    except ImportError:
        pass

    try:
        from services.semantic_answer_plan import (
            build_semantic_answer_plan,
            try_semantic_grounded_reply,
        )

        plan = build_semantic_answer_plan(ai_route or {}, handler=handler)
        if plan.use_ai_synthesis and plan.answer_strategy not in (
            "live_api_only",
            "catalog_only",
            "structured_handler",
        ):
            grounded = try_semantic_grounded_reply(
                original_msg,
                msg_en,
                ai_route,
                conversation_context=conv_for_llm,
                reply_lang=reply_lang,
                handler=handler,
            )
            if grounded:
                log_reasoning(
                    f"Locked dispatch: KB+AI synthesis (strategy={plan.answer_strategy})."
                )
                _record_dispatch(handler or "kb_ai", plan.reasoning or "kb_then_ai")
                return grounded
    except ImportError:
        pass
    return None


def execute_locked_route_turn(
    *,
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    retrieval_query: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable[[dict], None],
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """
    Execute the locked route once. Returns reply HTML/text or None.
  """
    handler = (route_decision.handler or "").strip()
    reason = (route_decision.reason or (ai_route or {}).get("reasoning") or "")[:200]
    _record_dispatch(handler or route_decision.source, reason)

    # --- Welfog social links (deterministic — before scope/LLM can refuse) ---
    try:
        from utils.helpers import (
            message_asks_other_company_social_media,
            message_asks_welfog_social_media,
        )
        from services.kb_service import format_welfog_social_media_reply_from_kb
        from services.support_scope import build_other_company_social_decline

        comb_sm = f"{original_msg} {msg_en}".strip()
        if message_asks_other_company_social_media(comb_sm, conversation_context=conv_for_llm):
            decline = build_other_company_social_decline(original_msg, reply_lang=lang)
            if decline:
                log_reasoning("Locked dispatch: other-company social decline (early).")
                reset_context_fn(ctx)
                return decline
        if message_asks_welfog_social_media(comb_sm, conversation_context=conv_for_llm):
            social = format_welfog_social_media_reply_from_kb(
                original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm
            )
            if social:
                log_reasoning("Locked dispatch: Welfog official social KB (early).")
                reset_context_fn(ctx)
                return social
    except ImportError:
        pass

    # --- LLM unavailable: continue API/KB when route is concrete ---
    try:
        from services.chat_resilience import build_busy_reply_html

        fail_kind = (ai_route or {}).get("_llm_failure") or ""
        if (ai_route or {}).get("llm_unavailable") and fail_kind in (
            "rate_limit",
            "busy",
            "timeout",
            "all_failed",
        ):
            from services.early_live_dispatch import ai_route_is_live_api_turn

            handler_concrete = handler in (
                "wishlist_api",
                "order_ai_flow",
                "order_tracking_api",
                "order_details_api",
                "refund_status_api",
                "pincode_delivery_api",
                "product_ai_flow",
                "deals_api",
                "categories_api",
            )
            can_proceed = (
                handler_concrete
                or ai_route_is_live_api_turn(ai_route, route_decision)
                or handler in _KB_AI_HANDLERS
                or route_decision.source in ("api", "kb", "kb_ai", "ai", "ai_product", "ai_order")
                or (ai_route or {}).get("data_channel") in ("live_api", "kb", "catalog")
            )
            if not can_proceed and handler not in _SCOPE_HANDLERS:
                log_reasoning(f"Locked dispatch: LLM down ({fail_kind}) — busy reply.")
                return build_busy_reply_html(original_msg, lang)
    except ImportError:
        pass

    meta = _try_meta_reply(ai_route, original_msg, lang)
    if meta:
        reset_context_fn(ctx)
        return meta

    if handler in _SCOPE_HANDLERS or (route_decision.source == "reject"):
        scope_body = _try_scope_reply_from_route(
            ai_route, original_msg, msg_en, conv_for_llm, lang, handler
        )
        if scope_body:
            reset_context_fn(ctx)
            return scope_body

    # --- Live API (single path via early_live_dispatch) ---
    try:
        from services.early_live_dispatch import try_early_live_api_reply

        live = try_early_live_api_reply(
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            route_decision=route_decision,
            ai_route=ai_route,
            reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
            reset_context_fn=reset_context_fn,
        )
        if live:
            return live
    except ImportError:
        pass

    # --- Catalog menu APIs (deals / categories list) ---
    intent_low = (route_decision.intent or "").strip().lower()
    if handler == "deals_api" or intent_low == "deals":
        try:
            from services.catalog_menu_replies import build_today_deals_reply_html

            body = build_today_deals_reply_html(original_msg, reply_lang=lang)
            if body:
                log_reasoning("Locked dispatch: today deals API cards.")
                _record_dispatch("deals_api", "today deals")
                reset_context_fn(ctx)
                ctx.setdefault("data", {})["pending_offer"] = "deals"
                return body
        except ImportError:
            pass

    if handler in ("categories_api", "category_feed_api") or intent_low in (
        "categories",
        "category_feed",
    ):
        try:
            from services.catalog_menu_replies import build_categories_list_reply_html
            from services.product_search_flow import run_product_search_ai_flow
            from services.welfog_api import (
                ensure_expanded_categories_map_for_ctx,
                resolve_category_product_browse_route,
            )

            ensure_expanded_categories_map_for_ctx(ctx)
            cat_browse = resolve_category_product_browse_route(
                f"{original_msg} {msg_en}", ctx=ctx
            )
            if cat_browse:
                cid, sq = cat_browse
                ctx.setdefault("data", {})["selected_category_id"] = cid
                ctx["awaiting"] = None
                ps = run_product_search_ai_flow(
                    original_msg,
                    msg_en,
                    user_id,
                    conversation_context=conv_for_llm,
                    reply_lang=lang,
                    ctx=ctx,
                    search_query=sq,
                    ai_route=ai_route,
                )
                if ps.handled and ps.reply_html:
                    log_reasoning(
                        f"Locked dispatch: category browse products (category_id={cid})."
                    )
                    _record_dispatch("categories_api", f"category_browse:{cid}")
                    reset_context_fn(ctx)
                    return ps.reply_html
            body = build_categories_list_reply_html(ctx, original_msg, reply_lang=lang)
            if body:
                log_reasoning("Locked dispatch: Welfog categories list API.")
                _record_dispatch("categories_api", "categories_list")
                reset_context_fn(ctx)
                ctx["awaiting"] = "category_select"
                return body
        except ImportError:
            pass

    # --- Structured handlers (API, KB how-to, product, etc.) ---
    early = dispatch_early_answer(
        route_decision,
        original_msg,
        msg_en,
        reply_lang=lang,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        ai_route=ai_route,
    )
    if early:
        log_reasoning(f"Locked dispatch: dispatch_early_answer handler={handler}.")
        if handler == "order_id_help_kb":
            ctx["awaiting"] = "order_id"
            ctx["last"] = route_decision.intent if route_decision.intent in (
                "refund",
                "payment",
                "order",
            ) else (ctx.get("last") or "order")
            olk = ((ai_route or {}).get("order_lookup_kind") or "").strip().lower()
            try:
                from services.ai_route_semantics import brain_route_to_live_goal

                goal = brain_route_to_live_goal(ai_route) or "order_details"
            except ImportError:
                goal = "order_details"
                if olk in ("invoice",):
                    goal = "order_invoice"
                elif olk in ("details", "order_details"):
                    goal = "order_details"
                elif olk in ("track", "tracking"):
                    goal = "track"
                elif olk == "refund_status" or route_decision.intent == "refund":
                    goal = "refund_status"
                elif route_decision.intent == "payment":
                    goal = "payment"
            ctx.setdefault("data", {})["pending_action"] = goal
            ctx["data"]["topic_mode"] = f"order_{goal}"
        elif handler in ("wishlist_howto_kb", "wishlist_api") or route_decision.intent == "wishlist":
            reset_context_fn(ctx)
            ctx["last"] = "wishlist"
            ctx.setdefault("data", {})["topic_mode"] = (
                "wishlist_howto" if handler == "wishlist_howto_kb" else "wishlist_list"
            )
        elif handler in ("order_history_howto_kb", "order_ai_flow") or route_decision.intent == "order_history":
            reset_context_fn(ctx)
            ctx["last"] = "order_history"
            ctx.setdefault("data", {})["topic_mode"] = (
                "order_history_howto"
                if handler == "order_history_howto_kb"
                else "order_history_list"
            )
        else:
            reset_context_fn(ctx)
        return early

    # --- Product catalog: never fall back to unrelated KB ---
    if handler == "product_ai_flow" or (
        (route_decision.intent or "").strip().lower() == "product"
        and (route_decision.source or "").strip() == "ai_product"
    ):
        try:
            from services.product_search_flow import run_product_search_ai_flow, _localized_sysmsg
            from services.kb_service import sysmsg

            ps = run_product_search_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                search_query=(route_decision.search_query or "").strip(),
                ai_route=ai_route,
            )
            if ps.handled and ps.reply_html:
                reset_context_fn(ctx)
                return ps.reply_html
            sq = (
                (route_decision.search_query or "")
                or ((ai_route or {}).get("search_query") or "")
            ).strip()
            body = _localized_sysmsg(
                "product_not_found", original_msg, reply_lang=lang, query=sq or "that item"
            ) or sysmsg("product_not_found", query=sq or "that item")
            log_reasoning(
                "Locked dispatch: product_ai_flow — catalog empty; KB fallback blocked."
            )
            reset_context_fn(ctx)
            return body
        except ImportError:
            pass

    # --- Pincode/delivery: never fall through to unrelated FAQ KB ---
    try:
        from services.entity_first_handlers import try_pincode_delivery_reply
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        if turn_requests_delivery_serviceability(
            original_msg,
            msg_en,
            conv_for_llm,
            ai_route,
            allow_llm=True,
        ):
            pin = try_pincode_delivery_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                ctx,
                ai_route=ai_route,
            )
            if pin:
                log_reasoning(
                    "Locked dispatch: pincode delivery API (KB synthesis blocked)."
                )
                _record_dispatch("pincode_delivery_api", "delivery serviceability")
                reset_context_fn(ctx)
                return pin
    except ImportError:
        pass

    # --- KB + AI synthesis for general/policy/support turns ---
    kb_ai = _try_kb_ai_synthesis(
        route_decision,
        ai_route,
        original_msg,
        msg_en,
        conv_for_llm,
        retrieval_query,
        lang,
    )
    if kb_ai:
        reset_context_fn(ctx)
        return kb_ai

    # --- Scope / chitchat when no KB match ---
    scope_late = _try_scope_reply_from_route(
        ai_route, original_msg, msg_en, conv_for_llm, lang, handler
    )
    if scope_late:
        reset_context_fn(ctx)
        return scope_late

    return None


def execute_locked_route_or_fallback(
    *,
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    retrieval_query: str,
    user_id: str,
    lang: str,
    ctx: dict,
    reset_context_fn: Callable,
    reply_for_live_order_id_lookup: Callable,
) -> Optional[str]:
    """Single execution pass after routing is locked — no legacy cascade."""
    locked_reply = execute_locked_route_turn(
        route_decision=route_decision,
        ai_route=ai_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        retrieval_query=retrieval_query,
        user_id=user_id,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context_fn,
        reply_for_live_order_id_lookup=reply_for_live_order_id_lookup,
    )
    if locked_reply:
        return locked_reply
    return locked_route_fallback(
        route_decision=route_decision,
        ai_route=ai_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        retrieval_query=retrieval_query,
        lang=lang,
        ctx=ctx,
    )


def _try_assistant_scope_fallback(
    *,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
    route_decision: AnswerRouteDecision,
) -> Optional[str]:
    """Chitchat / out-of-domain — only when route is not a failed live API turn."""
    intent = (route_decision.intent or "").strip().lower()
    handler = (route_decision.handler or "").strip().lower()
    scope_reply = ((ai_route or {}).get("scope_reply") or "").strip()
    scope = ((ai_route or {}).get("conversation_scope") or "").strip().lower()

    if handler in (
        "wishlist_api",
        "order_ai_flow",
        "order_tracking_api",
        "order_details_api",
        "pincode_delivery_api",
        "product_ai_flow",
        "deals_api",
        "categories_api",
    ):
        return None

    if scope_reply and scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        log_reasoning(f"Locked fallback: brain scope_reply ({scope}).")
        return scope_reply

    if intent == "out_of_domain" or not route_decision.is_welfog_related:
        try:
            from services.off_topic_reply import build_off_topic_polite_reply

            off = build_off_topic_polite_reply(
                original_msg,
                msg_en,
                reply_lang=lang,
                ai_route=ai_route,
                conversation_context=conv_for_llm,
            )
            if off:
                log_reasoning("Locked fallback: off-topic polite decline.")
                return off
        except ImportError:
            pass

    try:
        from services.conversation_scope import try_conversation_scope_reply

        scope_html = try_conversation_scope_reply(
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ai_route=ai_route,
            route_handler="",
            prefer_llm=True,
        )
        if scope_html:
            log_reasoning("Locked fallback: conversation scope reply.")
            return scope_html
    except ImportError:
        pass
    return None


def _turn_is_vague_welfog_request(text: str) -> bool:
    """Short unclear asks — guide user instead of weak KB guesses."""
    raw = (text or "").strip()
    if not raw or len(raw) > 90:
        return False
    tl = f" {raw.lower()} "
    vague = (
        "help", "chahiye", "chiye", "batao", "btao", "samajh nahi", "smjh nahi",
        "kya karu", "kya kru", "kya karna", "confused", "pata nahi", "ptm nahi",
        "bata do", "bta do", "guide", "assist",
    )
    specific = (
        "order", "refund", "track", "wishlist", "product", "delivery", "pincode",
        "policy", "payment", "seller", "return", "invoice", "welfog",
    )
    return any(v in tl for v in vague) and not any(s in tl for s in specific)


def _try_assistant_clarify_fallback(
    original_msg: str,
    msg_en: str,
    lang: str,
) -> Optional[str]:
    """Ambiguous Welfog turn — guide user like a standard support assistant."""
    try:
        from services.translation_service import customer_facing_template

        body = customer_facing_template(
            "clarify_request_scope",
            original_msg or msg_en or "help",
            lang,
            fallback_en=(
                "I want to help — could you say in one short line what you need? "
                "For example: product search, order tracking (with Order ID), "
                "delivery PIN check, or refund/policy info on Welfog."
            ),
        )
        if body and body.strip():
            log_reasoning("Locked fallback: clarify_request_scope.")
            return body
    except ImportError:
        pass
    return None


def _skip_kb_fallback_for_misrouted_live_api(
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    ctx: dict | None = None,
) -> bool:
    """Live API route failed — do not dump unrelated FAQ KB (e.g. after list-in-chat)."""
    handler = (route_decision.handler or "").strip().lower()
    intent = (route_decision.intent or "").strip().lower()
    comb = f"{original_msg} {msg_en}".strip()
    try:
        from services.account_list_semantics import (
            detect_account_list_followup_in_chat,
            turn_requests_purchase_history_in_chat,
            turn_requests_wishlist_in_chat,
        )

        if detect_account_list_followup_in_chat(
            original_msg, msg_en, conv_for_llm, ctx=ctx
        ):
            return True
        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route
        ) or turn_requests_wishlist_in_chat(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route
        ):
            return True
    except ImportError:
        pass
    if handler in (
        "order_tracking_api",
        "order_details_api",
        "refund_status_api",
        "wishlist_api",
        "order_ai_flow",
    ) or intent in ("order_history", "wishlist"):
        try:
            from utils.helpers import extract_order_id, resolve_order_id_for_tracking

            oid = extract_order_id(comb, conv_for_llm)
            if not oid:
                oid = resolve_order_id_for_tracking(
                    original_msg.strip() or msg_en.strip(), conv_for_llm
                )
            if not oid and handler == "order_tracking_api":
                return True
        except ImportError:
            if handler == "order_tracking_api":
                return True
    um = ((ai_route or {}).get("user_meaning") or "").lower()
    if um and "order history" in um and "track" not in um and "shipment" not in um:
        return True
    return False


def _pincode_fallback_reply(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
) -> Optional[str]:
    try:
        from services.pincode_delivery_flow import build_pincode_missing_or_invalid_reply

        body = build_pincode_missing_or_invalid_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            reply_lang=lang,
        )
        if body:
            log_reasoning("Locked fallback: pincode ask-PIN (not clarify menu).")
            return body
    except ImportError:
        pass
    return None


def locked_route_fallback(
    *,
    route_decision: AnswerRouteDecision,
    ai_route: dict | None,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    retrieval_query: str,
    lang: str,
    ctx: dict | None = None,
) -> Optional[str]:
    """
    Industry-standard last resort: KB → scope/OOD → clarify → polite decline → busy.
    Never jump straight to a generic server error for normal customer questions.
    """
    comb = f"{original_msg} {msg_en}".strip()
    intent = (route_decision.intent or "").strip().lower()
    handler = (route_decision.handler or "").strip().lower()
    vague_turn = _turn_is_vague_welfog_request(comb)
    block_kb = _skip_kb_fallback_for_misrouted_live_api(
        route_decision, ai_route, original_msg, msg_en, conv_for_llm, ctx=ctx
    )

    if intent == "pincode_check" or handler == "pincode_delivery_api":
        pin_fb = _pincode_fallback_reply(original_msg, msg_en, conv_for_llm, lang)
        if pin_fb:
            return pin_fb

    if vague_turn and route_decision.is_welfog_related:
        clarify = _try_assistant_clarify_fallback(original_msg, msg_en, lang)
        if clarify:
            return clarify

    if block_kb:
        try:
            from services.conversational_ack_flow import build_contextual_ack_reply

            ack = build_contextual_ack_reply(
                original_msg, msg_en, conv_for_llm, reply_lang=lang, ctx=ctx
            )
            if ack:
                log_reasoning("Locked fallback: contextual ack (blocked unrelated KB).")
                return ack
        except ImportError:
            pass
        scope_body = _try_assistant_scope_fallback(
            ai_route=ai_route,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            lang=lang,
            route_decision=route_decision,
        )
        if scope_body:
            log_reasoning("Locked fallback: scope reply (blocked unrelated KB).")
            return scope_body

    # Tier 1: KB (informational only — never after failed live API misroute)
    if not block_kb:
        try:
            from services.kb_service import direct_kb_search, get_customer_kb_keys

            keys = route_decision.kb_keys or get_customer_kb_keys()
            direct = direct_kb_search(retrieval_query or comb, keys=keys, min_score=0.28)
            if direct:
                log_reasoning("Locked fallback: direct KB.")
                return direct
        except ImportError:
            pass

    # Tier 2: KB + AI synthesis when route expected an answer
    if not vague_turn and (
        handler in _KB_AI_HANDLERS
        or intent in ("refund", "payment", "seller", "faqs")
    ):
        kb_ai = _try_kb_ai_synthesis(
            route_decision,
            ai_route,
            original_msg,
            msg_en,
            conv_for_llm,
            retrieval_query,
            lang,
        )
        if kb_ai:
            log_reasoning("Locked fallback: KB+AI synthesis.")
            return kb_ai

    # Tier 3: chitchat / out-of-domain (brain scope or polite decline)
    scope_body = _try_assistant_scope_fallback(
        ai_route=ai_route,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv_for_llm,
        lang=lang,
        route_decision=route_decision,
    )
    if scope_body:
        return scope_body

    # Tier 4: in-domain but unclear — helpful menu (not an error)
    if route_decision.is_welfog_related and intent != "out_of_domain":
        clarify = _try_assistant_clarify_fallback(original_msg, msg_en, lang)
        if clarify:
            return clarify

    # Tier 5: polite off-topic catch-all
    try:
        from services.off_topic_reply import build_off_topic_polite_reply

        off = build_off_topic_polite_reply(
            original_msg,
            msg_en,
            reply_lang=lang,
            ai_route=ai_route,
            conversation_context=conv_for_llm,
            prefer_llm=True,
        )
        if off:
            log_reasoning("Locked fallback: final off-topic polite.")
            return off
    except ImportError:
        pass

    # Tier 6: high traffic / infra only — never mask a routed API/KB turn
    if handler in (
        "order_tracking_api",
        "order_details_api",
        "refund_status_api",
        "wishlist_api",
        "order_ai_flow",
        "pincode_delivery_api",
        "product_ai_flow",
    ) or intent in ("order", "refund", "wishlist", "order_history", "product", "pincode_check"):
        if intent == "pincode_check" or handler == "pincode_delivery_api":
            pin_fb = _pincode_fallback_reply(original_msg, msg_en, conv_for_llm, lang)
            if pin_fb:
                return pin_fb
        clarify = _try_assistant_clarify_fallback(original_msg, msg_en, lang)
        if clarify:
            return clarify

    try:
        from services.chat_resilience import build_busy_reply_html

        log_reasoning("Locked fallback: infra busy reply (last resort).")
        return build_busy_reply_html(original_msg, lang)
    except ImportError:
        return None
