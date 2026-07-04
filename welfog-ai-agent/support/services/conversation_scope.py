"""
AI-first conversation scope — no keyword lists for off-topic / chit-chat.

Classifies each turn (any language, unseen phrasing):
  - welfog_support  → normal KB / API / catalog pipeline
  - general_chitchat → greetings, thanks, who-are-you, light talk (Welfog chat tone)
  - out_of_domain    → unrelated topics; polite decline in customer's language

Primary signal: ai_brain_route JSON (conversation_scope + scope_reply).
Fallback: dedicated lightweight scope LLM. Last resort: generic template (no word lists).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

_SCOPE_ROUTING_CACHE = threading.local()

SCOPE_WELFOG = "welfog_support"
SCOPE_CHITCHAT = "general_chitchat"
SCOPE_OUT = "out_of_domain"

_META_WELFOG_HANDLERS = frozenset(
    {
        "hostile",
        "bot_latency",
        "topic_denial",
        "wrong_search_complaint",
        "bot_search_complaint",
    }
)


@dataclass
class ScopeDecision:
    scope: str
    user_meaning: str = ""
    reply: str = ""
    source: str = ""
    confidence: float = 0.0


def _norm_scope(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    if s in (SCOPE_WELFOG, "welfog", "support", "shopping", "in_domain"):
        return SCOPE_WELFOG
    if s in (SCOPE_CHITCHAT, "chitchat", "chit_chat", "conversational", "greeting", "small_talk"):
        return SCOPE_CHITCHAT
    if s in (SCOPE_OUT, "out_of_domain", "off_topic", "offtopic", "unrelated"):
        return SCOPE_OUT
    if s in ("harm_sensitive", "harm", "crisis", "self_harm", "safety"):
        return "harm_sensitive"
    return ""


def _combined(original_msg: str, msg_en: str = "") -> str:
    return " ".join(p for p in ((original_msg or "").strip(), (msg_en or "").strip()) if p).strip()


_WELFOG_API_HANDLERS = frozenset(
    {
        "wishlist_api",
        "order_ai_flow",
        "order_details_api",
        "order_tracking_api",
        "pincode_delivery_api",
        "deals_api",
        "categories_api",
        "product_ai_flow",
        "wishlist_howto_kb",
        "order_history_howto_kb",
        "order_id_help_kb",
        "order_tracking_howto_kb",
    }
)

_KB_SCOPE_BYPASS_HANDLERS = frozenset(
    {
        "dynamic_kb",
        "knowledge_topic_kb",
        "welfog_about_kb",
        "customer_care_kb",
        "policy_structured_kb",
        "welfog_fees_kb",
        "short_video_rules_kb",
        "seller_kb",
        "support_escalation_kb",
        "welfog_social_kb",
        "kb_grounded_ai",
    }
)


def _substantive_welfog_intent(route: dict) -> bool:
    kind = (route.get("account_list_kind") or "").strip().lower()
    if kind and kind not in ("none", ""):
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent in (
        "product",
        "order",
        "order_history",
        "wishlist",
        "refund",
        "payment",
        "seller",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
    ):
        return True
    if channel in ("live_api", "catalog"):
        return True
    if route.get("run_catalog_search"):
        return True
    if route.get("needs_order_id") and intent in ("order", "refund", "payment"):
        return True
    return False


def scope_from_ai_route(route: dict | None) -> Optional[ScopeDecision]:
    """Trust main routing LLM when it sets conversation_scope or clear intent/meta."""
    if not route or not isinstance(route, dict):
        return None

    mk = (route.get("meta_kind") or "none").strip().lower()
    if mk in _META_WELFOG_HANDLERS:
        return None

    explicit = _norm_scope(route.get("conversation_scope") or "")
    scope_reply = (route.get("scope_reply") or route.get("response") or "").strip()

    if explicit:
        if explicit == SCOPE_WELFOG:
            return ScopeDecision(scope=SCOPE_WELFOG, source="ai_route", confidence=0.9)
        return ScopeDecision(
            scope=explicit,
            user_meaning=(route.get("user_meaning") or "")[:280],
            reply=scope_reply,
            source="ai_route",
            confidence=0.88,
        )

    intent = (route.get("intent") or "").strip().lower()
    is_welfog = bool(route.get("is_welfog_related", True))
    channel = (route.get("data_channel") or "").strip().lower()

    if intent == "out_of_domain" or (not is_welfog and channel == "none" and not _substantive_welfog_intent(route)):
        return ScopeDecision(
            scope=SCOPE_OUT,
            user_meaning=(route.get("user_meaning") or "")[:280],
            reply=scope_reply,
            source="ai_route_intent",
            confidence=0.85,
        )

    if mk in ("conversational", "assistant_intro") and not _substantive_welfog_intent(route):
        return ScopeDecision(
            scope=SCOPE_CHITCHAT,
            user_meaning=(route.get("user_meaning") or "")[:280],
            reply=scope_reply,
            source="ai_route_meta",
            confidence=0.82,
        )

    if _substantive_welfog_intent(route):
        return ScopeDecision(scope=SCOPE_WELFOG, source="ai_route_substantive", confidence=0.9)

    return None


def _has_definite_welfog_shopping_signal(text: str) -> bool:
    """Only hard guard: clear in-domain shopping — never scope-LLM block these."""
    if not (text or "").strip():
        return False
    try:
        from utils.helpers import (
            _is_light_smalltalk_fast,
            _is_short_pure_greeting,
            _looks_like_greeting_message,
            _message_has_catalog_product_signal,
            _text_has_kb_topic_hint,
            _text_has_product_shopping_intent,
            extract_order_id,
            message_is_knowledge_information_request,
            message_is_welfog_about_request,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
            message_asks_welfog_categories_list,
        )
        from services.conversation_followup import is_deals_request_message

        comb = text
        if (
            _looks_like_greeting_message(comb)
            or _is_short_pure_greeting(comb.strip())
            or _is_light_smalltalk_fast(comb, "")
        ):
            return False

        if message_asks_welfog_categories_list(comb) or is_deals_request_message(comb, ""):
            return True
        if extract_order_id(comb, ""):
            return True
        if message_is_past_purchase_list_request(comb) or message_is_wishlist_like_request(comb):
            return True
        if re.search(r"\b[1-9]\d{5}\b", comb) and any(
            x in comb.lower()
            for x in ("delivery", "pincode", "pin code", "deliver", "milega", "pahunch")
        ):
            return True
        if _text_has_product_shopping_intent(comb) or _message_has_catalog_product_signal(comb):
            return True
        if _text_has_kb_topic_hint(comb):
            if message_is_knowledge_information_request(comb) or message_is_welfog_about_request(comb):
                return True
    except ImportError:
        pass
    return False


def ai_classify_scope_and_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    preflight: bool = False,
) -> Optional[ScopeDecision]:
    """
    Dedicated scope LLM — any language, unseen wording. Returns classification + reply
    for chitchat / out_of_domain (empty reply for welfog_support).
    """
    try:
        from services.chat_flow_telemetry import is_routing_complete, skip_step

        if is_routing_complete():
            skip_step("ai_classify_scope_and_reply", "main router already ran")
            return None
    except ImportError:
        pass

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    compact_ctx = _compact_conversation_context(conversation_context or "", 1800)
    user_line = _trim_text_mid(comb, 480)

    system_prompt = f"""You classify the LATEST user message for the Welfog shopping support chatbot.

Welfog IN SCOPE: products to buy on Welfog, order tracking, order history, wishlist, delivery/PIN,
returns/refunds, payments, seller account, Welfog policies/FAQ/company info, customer care on Welfog.

OUT OF DOMAIN: anything NOT tied to Welfog shopping/support (weather, cricket, recipes, other apps'
orders, homework, life advice, random personal stories, jokes, politics, etc.) — any language.

GENERAL CHIT-CHAT: pure greeting, thanks, praise, bye, "who are you", "what can you do" about THIS bot,
casual wellbeing, "what are you doing", "are you free/busy right now", "are you okay" — friendly natural
reply like a human chat assistant (NOT a product search, NOT Order ID, NOT policy dump).

OUT OF DOMAIN examples (any language): weather, cricket, Amazon/other apps, homework, jokes,
relationship requests ("make me a girlfriend"), personal money questions, politics, random facts.

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence — what they want this turn",
  "conversation_scope": "welfog_support" | "general_chitchat" | "out_of_domain",
  "scope_reply": "REQUIRED when scope is general_chitchat or out_of_domain: 1-3 sentences in the CUSTOMER's language/script AND conversational style (casual, formal, slang, poetic — mirror how they wrote). Warm, polite, human like ChatGPT. For out_of_domain: briefly acknowledge their topic in their tone, say you only help with Welfog shopping/support, invite a Welfog question. Do NOT answer off-topic facts. For general_chitchat: natural reply — greeting in their style (not plain Hi), thanks/praise with welcome-back, bye with invite to return. EMPTY string when welfog_support.",
  "confidence": 0.0 to 1.0
}}

{language_reply_instruction(rl)}
Infer meaning semantically — never match fixed keyword lists."""

    user_payload = f"LATEST USER MESSAGE:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    try:
        scope_timeout = 8 if preflight else 11
        scope_attempts = 1 if preflight else 2
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=220,
            timeout_sec=scope_timeout,
            max_attempts=scope_attempts,
            temperature=0.25,
        )
    except Exception as exc:
        log_reasoning(f"Scope LLM error (non-fatal): {exc}")
        return None
    if not data:
        return None

    scope = _norm_scope(data.get("conversation_scope") or "")
    if not scope:
        return None

    reply = (data.get("scope_reply") or "").strip()
    conf = float(data.get("confidence") or 0.75)
    um = (data.get("user_meaning") or "").strip()

    log_reasoning(
        f"Scope LLM: {scope} (conf={conf:.2f}) — {um[:120] or 'no meaning'}"
    )
    return ScopeDecision(
        scope=scope,
        user_meaning=um,
        reply=reply,
        source="scope_llm",
        confidence=conf,
    )


def turn_blocks_product_catalog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """
    Chitchat / out-of-domain — must NOT run product classifier or catalog route lock.
    Semantic signals only (brain scope, embeddings, light conversational helpers).
    """
    if getattr(_SCOPE_ROUTING_CACHE, "in_turn_blocks", False):
        return False
    _SCOPE_ROUTING_CACHE.in_turn_blocks = True
    try:
        return _turn_blocks_product_catalog_body(
            original_msg, msg_en, conversation_context, ai_route
        )
    finally:
        _SCOPE_ROUTING_CACHE.in_turn_blocks = False


def _turn_blocks_product_catalog_body(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    comb = _combined(original_msg, msg_en)
    if not comb:
        return True
    if _has_definite_welfog_shopping_signal(comb):
        return False
    try:
        from services.chitchat_resolver import turn_is_chitchat_not_shopping

        if turn_is_chitchat_not_shopping(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=False,
        ):
            return True
    except ImportError:
        pass
    dec = scope_from_ai_route(ai_route)
    if dec and dec.scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
        if dec.confidence >= 0.52:
            return True
    if isinstance(ai_route, dict):
        scope = _norm_scope(ai_route.get("conversation_scope") or "")
        if scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
            return True
        if (ai_route.get("intent") or "").strip().lower() == "out_of_domain":
            return True
        mk = (ai_route.get("meta_kind") or "none").strip().lower()
        if mk in ("conversational", "assistant_intro") and not ai_route.get("run_catalog_search"):
            return True
    try:
        from services.product_browse_semantics import embedding_browse_scores
        from services.meta_turn_semantics import embedding_meta_scores

        pos, neg = embedding_browse_scores(comb)
        intro_s, company_s = embedding_meta_scores(comb)
        if intro_s >= 0.36 and intro_s > pos + 0.02:
            return True
        if company_s >= 0.36 and company_s > pos + 0.02 and not _has_definite_welfog_shopping_signal(
            comb
        ):
            return True
    except ImportError:
        pass
    return False


def log_scope_routing_telemetry(
    *,
    scope: str,
    route: str,
    confidence: float = 0.0,
    source: str = "",
) -> None:
    line = (
        f"[scope-route] intent={scope} route={route!r} "
        f"confidence={confidence:.2f} source={source or '-'}"
    )
    log_reasoning(line)
    chat_log(line)
    try:
        from services.chat_flow_telemetry import _TLS

        _TLS.scope_confidence = confidence
        _TLS.conversation_scope = scope
    except ImportError:
        pass


def turn_requests_catalog_menu(
    original_msg: str,
    msg_en: str = "",
    ai_route: dict | None = None,
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    """Deals / categories list — brain + AI micro-classifier + semantic fallback."""
    try:
        from services.catalog_menu_resolver import turn_requests_catalog_menu as _resolver_menu

        return _resolver_menu(
            original_msg,
            msg_en,
            ai_route=ai_route,
            conversation_context=conversation_context,
            reply_lang=reply_lang,
            allow_llm=allow_llm,
        )
    except ImportError:
        pass
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        if intent in ("categories", "category_feed", "deals"):
            return True
    return False


_turn_requests_catalog_menu = turn_requests_catalog_menu


def try_scope_routing_decision(
    route_data: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[tuple[Any, dict]]:
    """
    Route chitchat / out-of-domain once after brain — before product/KB enrich loops.
    Returns (AnswerRouteDecision, route_data) or None to continue Welfog pipeline.
    """
    from services.answer_router import AnswerRouteDecision

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None
    if _turn_requests_catalog_menu(original_msg, msg_en, ai_route=route_data):
        return None
    if _has_definite_welfog_shopping_signal(comb):
        return None
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route_data):
            return None
    except ImportError:
        pass
    if route_data and _substantive_welfog_intent(route_data):
        intent = (route_data.get("intent") or "").strip().lower()
        channel = (route_data.get("data_channel") or "").strip().lower()
        if intent == "product" and channel == "catalog":
            return None
        if channel in ("live_api", "catalog") and intent not in ("general", ""):
            return None

    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-200:])}"
    if getattr(_SCOPE_ROUTING_CACHE, "key", None) == cache_key:
        cached = getattr(_SCOPE_ROUTING_CACHE, "result", None)
        if cached is not None:
            return cached

    decision_out: Optional[tuple[Any, dict]] = None
    try:
        from services.chitchat_resolver import resolve_chitchat_scope

        dec = resolve_chitchat_scope(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=route_data,
            reply_lang=reply_lang,
            allow_llm=True,
        )
        if dec.scope == SCOPE_WELFOG:
            dec = scope_from_ai_route(route_data) or dec
    except ImportError:
        dec = scope_from_ai_route(route_data)
        if not dec or dec.scope == SCOPE_WELFOG:
            dec = ai_classify_scope_and_reply(
                original_msg, msg_en, conversation_context, reply_lang
            )

    if dec and dec.scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
        if dec.confidence < 0.52 and dec.source not in (
            "ai_route",
            "ai_route_intent",
            "ai_route_meta",
            "scope_llm",
            "chitchat_ai",
        ):
            _SCOPE_ROUTING_CACHE.key = cache_key
            _SCOPE_ROUTING_CACHE.result = None
            return None

        out = dict(route_data or {})
        out["conversation_scope"] = dec.scope
        out["user_meaning"] = dec.user_meaning or out.get("user_meaning") or ""
        if dec.reply:
            out["scope_reply"] = dec.reply
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out.pop("_product_catalog_locked", None)
        out.pop("route_handler", None)

        if dec.scope == SCOPE_CHITCHAT:
            out["intent"] = "general"
            out["meta_kind"] = "conversational"
            out["is_welfog_related"] = True
            handler = "warm_feedback"
            intent = "general"
            source = "scope"
        elif dec.scope == "harm_sensitive":
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            handler = "off_topic"
            intent = "out_of_domain"
            source = "reject"
        else:
            out["intent"] = "out_of_domain"
            out["is_welfog_related"] = False
            handler = "off_topic"
            intent = "out_of_domain"
            source = "reject"

        log_scope_routing_telemetry(
            scope=dec.scope,
            route=handler,
            confidence=dec.confidence,
            source=dec.source,
        )
        decision_out = (
            AnswerRouteDecision(
                source=source,
                intent=intent,
                handler=handler,
                is_welfog_related=out.get("is_welfog_related", True),
                reason=f"Conversation scope ({dec.scope}) — {dec.user_meaning[:80]}",
            ),
            out,
        )

    _SCOPE_ROUTING_CACHE.key = cache_key
    _SCOPE_ROUTING_CACHE.result = decision_out
    return decision_out


def should_bypass_conversation_scope(
    ai_route: dict | None = None,
    route_handler: str = "",
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    When routing already chose a Welfog API/KB handler, never hijack with chitchat scope.
    Prevents extra scope LLM + wrong greeting after wishlist/order_history was detected.
    """
    rh = (route_handler or "").strip()
    if original_msg or msg_en:
        if turn_blocks_product_catalog(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return False
    if rh in _WELFOG_API_HANDLERS or rh in _KB_SCOPE_BYPASS_HANDLERS:
        return True
    if ai_route and _substantive_welfog_intent(ai_route):
        return True
    if original_msg or msg_en:
        try:
            from services.knowledge_query_pipeline import turn_is_informational_knowledge_only

            if turn_is_informational_knowledge_only(
                original_msg, msg_en, conversation_context
            ):
                return True
        except ImportError:
            pass
    return False


def resolve_conversation_scope(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ai_route: dict | None = None,
    *,
    route_handler: str = "",
) -> ScopeDecision:
    """AI-first scope; minimal shopping guard only."""
    if should_bypass_conversation_scope(
        ai_route, route_handler, original_msg=original_msg, msg_en=msg_en,
        conversation_context=conversation_context,
    ):
        return ScopeDecision(scope=SCOPE_WELFOG, source="api_route_lock", confidence=1.0)

    try:
        from services.knowledge_query_pipeline import turn_is_informational_knowledge_only

        if turn_is_informational_knowledge_only(
            original_msg, msg_en, conversation_context
        ):
            return ScopeDecision(
                scope=SCOPE_WELFOG,
                source="informational_kb",
                confidence=0.95,
            )
    except ImportError:
        pass

    try:
        from services.query_intent_classifier import (
            INTENT_CHITCHAT,
            INTENT_HARM,
            INTENT_OUT,
            INTENT_WELFOG,
            get_request_query_intent,
        )

        cached = get_request_query_intent()
        if cached and cached.detected_intent != INTENT_WELFOG:
            scope_map = {
                INTENT_CHITCHAT: SCOPE_CHITCHAT,
                INTENT_OUT: SCOPE_OUT,
                INTENT_HARM: "harm_sensitive",
            }
            sc = scope_map.get(cached.detected_intent)
            if sc and cached.confidence >= 0.65:
                return ScopeDecision(
                    scope=sc,
                    user_meaning=cached.user_meaning,
                    reply=cached.reply,
                    source=cached.classifier_source or "query_intent",
                    confidence=cached.confidence,
                )
    except ImportError:
        pass

    comb = _combined(original_msg, msg_en)
    if _has_definite_welfog_shopping_signal(comb):
        return ScopeDecision(scope=SCOPE_WELFOG, source="shopping_guard", confidence=1.0)

    try:
        from services.chitchat_resolver import resolve_chitchat_scope

        ai_dec = resolve_chitchat_scope(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            reply_lang=reply_lang,
            allow_llm=True,
        )
        if ai_dec.scope != SCOPE_WELFOG:
            return ai_dec
    except ImportError:
        pass

    from_route = scope_from_ai_route(ai_route)
    if from_route and from_route.scope == SCOPE_WELFOG:
        return from_route

    if from_route and from_route.scope in (SCOPE_OUT, SCOPE_CHITCHAT):
        if from_route.reply:
            return from_route
        try:
            from services.chat_flow_telemetry import is_routing_complete, skip_step

            if is_routing_complete():
                skip_step("scope_reply_fill", "reuse brain scope without extra LLM")
                scope_reply = ((ai_route or {}).get("scope_reply") or "").strip()
                if scope_reply:
                    from_route.reply = scope_reply
                return from_route
        except ImportError:
            pass
        llm_fill = ai_classify_scope_and_reply(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if llm_fill and llm_fill.reply:
            from_route.reply = llm_fill.reply
            from_route.source = f"{from_route.source}+scope_llm_reply"
        return from_route

    try:
        from services.chat_flow_telemetry import is_routing_complete, skip_step

        if is_routing_complete():
            skip_step("resolve_conversation_scope_llm", "reuse stored route scope")
            if from_route:
                return from_route
            return ScopeDecision(scope=SCOPE_WELFOG, source="post_route_default", confidence=0.5)
    except ImportError:
        pass

    llm = ai_classify_scope_and_reply(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if llm:
        return llm

    if from_route:
        return from_route

    return ScopeDecision(scope=SCOPE_WELFOG, source="default", confidence=0.5)


def _generic_scope_fallback(scope: str, use_hinglish: bool) -> str:
    from services.kb_service import sysmsg

    if scope == SCOPE_CHITCHAT:
        return (
            sysmsg("greeting_variant_2")
            or sysmsg("greeting")
            or ""
        )
    if scope == SCOPE_OUT:
        return (
            sysmsg("off_topic_polite_hinglish" if use_hinglish else "off_topic_polite")
            or sysmsg("out_of_domain_hinglish" if use_hinglish else "out_of_domain")
            or sysmsg("off_topic_polite")
            or ""
        )
    return ""


def finalize_scope_reply_html(
    reply: str,
    original_msg: str,
    reply_lang: str = "",
) -> str:
    from services.translation_service import finalize_customer_reply

    text = (reply or "").strip()
    if not text:
        return ""
    if "<" not in text:
        text = f"<div style='color:#333;line-height:1.55;'>{text}</div>"
    return finalize_customer_reply(text, original_msg, reply_lang=reply_lang)


def build_scope_reply(
    decision: ScopeDecision,
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
    *,
    prefer_llm: bool = True,
) -> str:
    """Build customer reply for chitchat / out_of_domain."""
    from services.translation_service import (
        customer_reply_language,
        is_hinglish_message,
        localize_for_customer,
    )

    if decision.scope == SCOPE_WELFOG:
        return ""

    rl = (reply_lang or customer_reply_language(original_msg) or "en").lower()
    body = (decision.reply or "").strip()

    if not body and prefer_llm:
        refill = ai_classify_scope_and_reply(original_msg, msg_en, "", rl)
        if refill and refill.reply:
            body = refill.reply

    if not body:
        try:
            from services.conversational_ack_flow import ai_natural_scope_reply

            body = ai_natural_scope_reply(
                decision.scope,
                original_msg,
                msg_en,
                "",
                reply_lang=rl,
            )
        except ImportError:
            body = ""

    if body and rl not in ("en", "hinglish"):
        body = localize_for_customer(body, rl)

    return finalize_scope_reply_html(body, original_msg, reply_lang=rl)


def try_conversation_scope_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ai_route: dict | None = None,
    *,
    prefer_llm: bool = True,
    route_handler: str = "",
) -> Optional[str]:
    """
    If this turn is chitchat or out-of-domain, return reply HTML and caller should
    skip KB/API/catalog. None → continue Welfog pipeline.
    """
    if should_bypass_conversation_scope(
        ai_route,
        route_handler,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    ):
        return None

    decision = resolve_conversation_scope(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
        route_handler=route_handler,
    )
    if decision.scope == SCOPE_WELFOG:
        return None

    if decision.scope == "harm_sensitive":
        try:
            from services.query_intent_classifier import (
                QueryIntentDecision,
                build_non_welfog_reply,
            )

            harm_dec = QueryIntentDecision(
                detected_intent="harm_sensitive",
                selected_source="safety_response",
                confidence=decision.confidence,
                user_meaning=decision.user_meaning,
                reply=decision.reply,
                classifier_source=decision.source,
            )
            html = build_non_welfog_reply(
                harm_dec, original_msg, msg_en, reply_lang=reply_lang
            )
            if html:
                log_reasoning("Conversation scope — harm_sensitive safety reply.")
                return html
        except ImportError:
            pass

    html = build_scope_reply(
        decision,
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        prefer_llm=prefer_llm,
    )
    if html:
        log_reasoning(
            f"Conversation scope handled: {decision.scope} via {decision.source}"
        )
        return html
    return None


# --- Backward-compatible API (no keyword lists) ---

def message_is_obvious_off_topic_outside_welfog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    d = resolve_conversation_scope(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    return d.scope == SCOPE_OUT


def build_off_topic_polite_reply(
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
    ai_route: dict | None = None,
    conversation_context: str = "",
    *,
    prefer_llm: bool = True,
) -> str:
    decision = resolve_conversation_scope(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
    )
    if decision.scope != SCOPE_OUT:
        decision = ScopeDecision(scope=SCOPE_OUT, source="forced_out")
    return build_scope_reply(
        decision,
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        prefer_llm=prefer_llm,
    ) or ""
