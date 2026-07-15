"""
Production query-intent layer — runs before KB / catalog / live API retrieval.

Classifies by meaning (any language), not phrase lists:
  - welfog_support      → existing KB / API / catalog pipeline
  - general_chitchat    → natural conversational AI reply
  - out_of_domain       → polite decline + Welfog redirect
  - harm_sensitive      → safe supportive reply (no KB / grievance templates)

Logs: [intent-routing] detected_intent=… selected_source=… confidence=…
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from utils.reasoning_log import log_reasoning

INTENT_WELFOG = "welfog_support"
INTENT_CHITCHAT = "general_chitchat"
INTENT_OUT = "out_of_domain"
INTENT_HARM = "harm_sensitive"

SOURCE_WELFOG = "welfog_pipeline"
SOURCE_NATURAL = "natural_ai"
SOURCE_DECLINE = "polite_decline"
SOURCE_SAFETY = "safety_response"

_tls = threading.local()


@dataclass
class QueryIntentDecision:
    detected_intent: str
    selected_source: str
    confidence: float = 0.0
    user_meaning: str = ""
    reply: str = ""
    classifier_source: str = ""


def _norm_intent(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    if s in (INTENT_WELFOG, "welfog", "in_domain", "shopping", "support"):
        return INTENT_WELFOG
    if s in (INTENT_CHITCHAT, "chitchat", "chit_chat", "conversational", "emotional", "casual"):
        return INTENT_CHITCHAT
    if s in (INTENT_OUT, "off_topic", "offtopic", "unrelated", "out_of_domain"):
        return INTENT_OUT
    if s in (INTENT_HARM, "harm", "self_harm", "crisis", "safety", "sensitive"):
        return INTENT_HARM
    return ""


def intent_to_source(intent: str) -> str:
    intent = _norm_intent(intent) or intent
    if intent == INTENT_WELFOG:
        return SOURCE_WELFOG
    if intent == INTENT_CHITCHAT:
        return SOURCE_NATURAL
    if intent == INTENT_OUT:
        return SOURCE_DECLINE
    if intent == INTENT_HARM:
        return SOURCE_SAFETY
    return SOURCE_WELFOG


def log_intent_routing(decision: QueryIntentDecision) -> None:
    log_reasoning(
        "[intent-routing] "
        f"detected_intent={decision.detected_intent} "
        f"selected_source={decision.selected_source} "
        f"confidence={decision.confidence:.2f} "
        f"via={decision.classifier_source or 'unknown'}"
        + (f" meaning={decision.user_meaning[:100]}" if decision.user_meaning else "")
    )


def set_request_query_intent(decision: Optional[QueryIntentDecision]) -> None:
    _tls.query_intent = decision


def get_request_query_intent() -> Optional[QueryIntentDecision]:
    return getattr(_tls, "query_intent", None)


def store_query_intent_ctx(ctx: dict, decision: QueryIntentDecision) -> None:
    set_request_query_intent(decision)
    if isinstance(ctx, dict):
        ctx.setdefault("data", {})["query_intent"] = {
            "detected_intent": decision.detected_intent,
            "selected_source": decision.selected_source,
            "confidence": decision.confidence,
            "user_meaning": (decision.user_meaning or "")[:320],
            "classifier_source": decision.classifier_source,
        }


def query_intent_from_ctx(ctx: dict | None) -> Optional[QueryIntentDecision]:
    cached = get_request_query_intent()
    if cached:
        return cached
    if not isinstance(ctx, dict):
        return None
    raw = (ctx.get("data") or {}).get("query_intent") or {}
    if not raw.get("detected_intent"):
        return None
    return QueryIntentDecision(
        detected_intent=raw.get("detected_intent") or INTENT_WELFOG,
        selected_source=raw.get("selected_source") or intent_to_source(raw.get("detected_intent")),
        confidence=float(raw.get("confidence") or 0),
        user_meaning=raw.get("user_meaning") or "",
        classifier_source=raw.get("classifier_source") or "ctx",
    )


def query_intent_allows_welfog_data(ctx: dict | None = None) -> bool:
    """False → block KB search, dynamic KB, and catalog product flows."""
    d = query_intent_from_ctx(ctx) or get_request_query_intent()
    if not d:
        return True
    if d.detected_intent == INTENT_WELFOG:
        return True
    if d.detected_intent == INTENT_HARM:
        return False
    return d.confidence < 0.62


def query_intent_allows_kb(ctx: dict | None = None) -> bool:
    return query_intent_allows_welfog_data(ctx)


def query_intent_allows_catalog(ctx: dict | None = None) -> bool:
    return query_intent_allows_welfog_data(ctx)


def _combined(original_msg: str, msg_en: str = "") -> str:
    return " ".join(p for p in ((original_msg or "").strip(), (msg_en or "").strip()) if p).strip()


def _structural_welfog_lock(text: str, conversation_context: str = "") -> bool:
    """Entity / topic locks only — not phrase-list classification."""
    if not (text or "").strip():
        return False
    try:
        from services.conversation_scope import _has_definite_welfog_shopping_signal

        if _has_definite_welfog_shopping_signal(text):
            return True
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _text_has_product_shopping_intent_core,
            turn_is_catalog_product_lookup,
        )

        if _text_has_product_shopping_intent_core(text) or turn_is_catalog_product_lookup(
            text, text
        ):
            return True
    except ImportError:
        pass
    try:
        from utils.helpers import (
            extract_embedded_query_identifiers,
            extract_order_id,
            extract_product_id,
        )

        ids = extract_embedded_query_identifiers(text, text, conversation_context)
        if (ids.get("order_id") or "").strip():
            return True
        if extract_order_id(text, conversation_context):
            return True
        if extract_product_id(text):
            return True
        if (ids.get("product_id") or "").strip():
            return True
        if (ids.get("pincode") or "").strip() and (ids.get("numeric_context") or "") == "pincode":
            return True
    except ImportError:
        pass
    try:
        from services.order_details_flow import message_wants_order_details_or_invoice

        if message_wants_order_details_or_invoice(text, text, conversation_context):
            return True
    except ImportError:
        pass
    return False


def _intent_from_ai_route(ai_route: dict | None) -> Optional[QueryIntentDecision]:
    if not ai_route or not isinstance(ai_route, dict):
        return None
    scope_raw = (ai_route.get("conversation_scope") or "").strip().lower()
    intent_raw = (ai_route.get("intent") or "").strip().lower()
    is_welfog = bool(ai_route.get("is_welfog_related", True))
    reply = (ai_route.get("scope_reply") or "").strip()
    meaning = (ai_route.get("user_meaning") or "")[:280]

    if scope_raw in ("harm_sensitive", "harm", "crisis", "self_harm"):
        return QueryIntentDecision(
            detected_intent=INTENT_HARM,
            selected_source=SOURCE_SAFETY,
            confidence=0.9,
            user_meaning=meaning,
            reply=reply,
            classifier_source="ai_route_scope",
        )

    scope = _norm_intent(scope_raw)
    if scope and scope != INTENT_WELFOG:
        return QueryIntentDecision(
            detected_intent=scope,
            selected_source=intent_to_source(scope),
            confidence=0.86,
            user_meaning=meaning,
            reply=reply,
            classifier_source="ai_route_scope",
        )

    if intent_raw == "out_of_domain" or (not is_welfog and not _substantive_route(ai_route)):
        return QueryIntentDecision(
            detected_intent=INTENT_OUT,
            selected_source=SOURCE_DECLINE,
            confidence=0.84,
            user_meaning=meaning,
            reply=reply,
            classifier_source="ai_route_intent",
        )

    mk = (ai_route.get("meta_kind") or "none").strip().lower()
    if mk in ("conversational", "assistant_intro") and not _substantive_route(ai_route):
        return QueryIntentDecision(
            detected_intent=INTENT_CHITCHAT,
            selected_source=SOURCE_NATURAL,
            confidence=0.8,
            user_meaning=meaning,
            reply=reply,
            classifier_source="ai_route_meta",
        )
    return None


def _substantive_route(route: dict) -> bool:
    try:
        from services.conversation_scope import _substantive_welfog_intent

        return _substantive_welfog_intent(route)
    except ImportError:
        intent = (route.get("intent") or "").strip().lower()
        return intent in (
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
        )


def ai_classify_query_intent(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    ignore_routing_complete: bool = False,
) -> Optional[QueryIntentDecision]:
    """Dedicated classifier LLM — semantic only, no keyword examples."""
    try:
        from services.chat_flow_telemetry import is_routing_complete, skip_step

        if not ignore_routing_complete and is_routing_complete():
            skip_step("ai_classify_query_intent", "main router already ran")
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

    compact_ctx = _compact_conversation_context(conversation_context or "", 2000)
    user_line = _trim_text_mid(comb, 520)

    system_prompt = f"""You are the intent classifier for the Welfog shopping support chatbot (India e-commerce).

Read the LATEST user message and RECENT conversation. Infer meaning, language, emotion, and what they want THIS turn.

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence — what they want emotionally/practically this turn",
  "detected_intent": "welfog_support" | "general_chitchat" | "out_of_domain" | "harm_sensitive",
  "natural_reply": "REQUIRED when detected_intent is NOT welfog_support: 2-5 warm sentences in the CUSTOMER's language/script. For harm_sensitive: empathetic, encourage safety and talking to someone they trust; mention India crisis lines iCall 9152987821 or Vandrevala 1860-2662-345 or emergency 112; NEVER mention grievance officer, legal compliance, seller registration, or product search. For general_chitchat: friendly human reply as Welfog assistant (greetings, thanks, jokes, personal feelings, romance) — light Welfog invite optional, no policy dump. For out_of_domain: acknowledge topic briefly, say you only help with Welfog shopping/orders/delivery/support, invite a Welfog question — do NOT answer off-topic facts (weather, cricket score, recipes, etc.). EMPTY string when welfog_support.",
  "confidence": 0.0 to 1.0
}}

INTENT RULES (semantic, any language — do NOT match fixed keyword lists):
- welfog_support: products to buy, orders/tracking/history, wishlist, delivery PIN, refunds, payments, seller on Welfog, Welfog policies/FAQ/company/customer care.
- general_chitchat: pure social talk, compliments, personal life, feelings, humor, bot identity small talk — NOT asking Welfog shopping facts.
- out_of_domain: unrelated factual/info requests (other apps, sports schedules, homework, general life advice unrelated to Welfog).
- harm_sensitive: self-harm, suicide, wanting to die, jumping from height, overdose, serious violence toward self — treat as crisis even if phrased casually or in Hinglish.

If the message clearly contains Welfog order/product/delivery work, choose welfog_support even if emotional words appear.

{language_reply_instruction(rl)}
Infer from meaning only — never from memorized phrase lists."""

    user_payload = f"LATEST USER MESSAGE:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    try:
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=280,
            timeout_sec=12,
            max_attempts=2,
            temperature=0.2,
        )
    except Exception as exc:
        log_reasoning(f"Query intent LLM error (non-fatal): {exc}")
        return None
    if not data:
        return None

    intent = _norm_intent(data.get("detected_intent") or "")
    if not intent:
        return None

    conf = float(data.get("confidence") or 0.78)
    reply = (data.get("natural_reply") or data.get("scope_reply") or "").strip()
    um = (data.get("user_meaning") or "").strip()

    return QueryIntentDecision(
        detected_intent=intent,
        selected_source=intent_to_source(intent),
        confidence=conf,
        user_meaning=um,
        reply=reply,
        classifier_source="query_intent_llm",
    )


def reconcile_query_intent_with_route(
    ai_route: dict | None,
    ctx: dict | None = None,
) -> Optional[QueryIntentDecision]:
    """Apply routing JSON scope without a second classifier LLM call."""
    if isinstance(ai_route, dict):
        route_intent = (ai_route.get("intent") or "").strip().lower()
        if route_intent == "product" and ai_route.get("is_welfog_related", True):
            welfog = QueryIntentDecision(
                detected_intent=INTENT_WELFOG,
                selected_source=SOURCE_WELFOG,
                confidence=0.95,
                user_meaning=(ai_route.get("user_meaning") or "")[:280],
                classifier_source="ai_route_product",
            )
            if ctx is not None:
                store_query_intent_ctx(ctx, welfog)
            else:
                set_request_query_intent(welfog)
            log_intent_routing(welfog)
            return welfog
    from_route = _intent_from_ai_route(ai_route)
    if not from_route:
        from_route = get_request_query_intent()
    if not from_route:
        from_route = QueryIntentDecision(
            detected_intent=INTENT_WELFOG,
            selected_source=SOURCE_WELFOG,
            confidence=0.5,
            classifier_source="default_welfog_fallback",
        )
    if ctx is not None:
        store_query_intent_ctx(ctx, from_route)
    else:
        set_request_query_intent(from_route)
    log_intent_routing(from_route)
    return from_route


def classify_query_intent(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ai_route: dict | None = None,
) -> QueryIntentDecision:
    """
    Primary classification for this turn. Structural Welfog locks win;
    then explicit AI route scope; then dedicated classifier LLM.
    """
    if ai_route:
        from_route = _intent_from_ai_route(ai_route)
        if from_route and from_route.detected_intent != INTENT_WELFOG:
            if from_route.reply:
                log_intent_routing(from_route)
                return from_route

    existing = get_request_query_intent()
    if existing and not ai_route:
        return existing

    comb = _combined(original_msg, msg_en)

    if _structural_welfog_lock(comb, conversation_context):
        d = QueryIntentDecision(
            detected_intent=INTENT_WELFOG,
            selected_source=SOURCE_WELFOG,
            confidence=1.0,
            classifier_source="entity_guard",
        )
        log_intent_routing(d)
        return d

    from_route = _intent_from_ai_route(ai_route)
    if from_route and from_route.detected_intent != INTENT_WELFOG and from_route.reply:
        log_intent_routing(from_route)
        return from_route

    try:
        from services.chat_flow_telemetry import (
            get_stored_ai_route,
            is_routing_complete,
            skip_step,
        )

        if is_routing_complete():
            skip_step("classify_query_intent", "reuse stored brain route")
            stored = ai_route or get_stored_ai_route()
            stored_decision = _intent_from_ai_route(stored)
            if stored_decision:
                log_intent_routing(stored_decision)
                return stored_decision
            if from_route:
                log_intent_routing(from_route)
                return from_route
            d = QueryIntentDecision(
                detected_intent=INTENT_WELFOG,
                selected_source=SOURCE_WELFOG,
                confidence=0.9,
                classifier_source="post_route_welfog",
            )
            log_intent_routing(d)
            return d
    except ImportError:
        pass

    llm = ai_classify_query_intent(
        original_msg, msg_en, conversation_context, reply_lang
    )
    if llm:
        if llm.detected_intent != INTENT_WELFOG and not llm.reply:
            refill = ai_classify_query_intent(original_msg, msg_en, conversation_context, reply_lang)
            if refill and refill.reply:
                llm.reply = refill.reply
        log_intent_routing(llm)
        return llm

    if from_route:
        log_intent_routing(from_route)
        return from_route

    d = QueryIntentDecision(
        detected_intent=INTENT_WELFOG,
        selected_source=SOURCE_WELFOG,
        confidence=0.55,
        classifier_source="default_welfog",
    )
    log_intent_routing(d)
    return d


def finalize_intent_reply_html(
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


def build_non_welfog_reply(
    decision: QueryIntentDecision,
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
) -> str:
    from services.translation_service import (
        customer_reply_language,
        is_hinglish_message,
        localize_for_customer,
    )
    from services.kb_service import sysmsg

    if decision.detected_intent == INTENT_WELFOG:
        return ""

    rl = (reply_lang or customer_reply_language(original_msg) or "en").lower()
    body = (decision.reply or "").strip()

    if not body:
        if decision.detected_intent == INTENT_HARM:
            use_h = rl == "hinglish" or is_hinglish_message(original_msg or msg_en)
            body = (
                sysmsg("harm_safety_hinglish" if use_h else "harm_safety")
                or sysmsg("harm_safety")
                or ""
            )
        elif decision.detected_intent == INTENT_CHITCHAT:
            try:
                from services.conversational_ack_flow import ai_chitchat_reply

                body = ai_chitchat_reply(
                    original_msg, msg_en, "", reply_lang=rl
                ) or ""
            except ImportError:
                body = ""
        elif decision.detected_intent == INTENT_OUT:
            try:
                from services.conversational_ack_flow import ai_ood_reply

                body = ai_ood_reply(original_msg, msg_en, "", reply_lang=rl) or ""
            except ImportError:
                body = ""
            if not body:
                use_h = rl == "hinglish" or is_hinglish_message(original_msg or msg_en)
                body = (
                    sysmsg("off_topic_polite_hinglish" if use_h else "off_topic_polite")
                    or sysmsg("out_of_domain")
                    or ""
                )

    if body and rl not in ("en", "hinglish") and "<" not in body:
        body = localize_for_customer(body, rl)

    return finalize_intent_reply_html(body, original_msg, reply_lang=rl)


def try_query_intent_gate_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    ctx: dict | None = None,
    ai_route: dict | None = None,
) -> Optional[str]:
    """
    Run classifier and return reply HTML when this turn must NOT use KB/API/catalog.
    None → continue Welfog pipeline.
    """
    comb = _combined(original_msg, msg_en)
    if _structural_welfog_lock(comb, conversation_context):
        return None
    try:
        from utils.helpers import (
            _message_has_catalog_product_signal,
            _text_has_product_shopping_intent_core,
        )

        if _text_has_product_shopping_intent_core(comb) or _message_has_catalog_product_signal(
            comb
        ):
            return None
    except ImportError:
        pass
    decision = classify_query_intent(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        ai_route=ai_route,
    )
    if ctx is not None:
        store_query_intent_ctx(ctx, decision)

    if decision.detected_intent == INTENT_WELFOG:
        return None

    if decision.detected_intent != INTENT_HARM and decision.confidence < 0.65:
        return None

    html = build_non_welfog_reply(decision, original_msg, msg_en, reply_lang)
    if html:
        log_reasoning(
            f"Query intent gate — skip KB/API (intent={decision.detected_intent})."
        )
        return html
    return None
