"""
Informational knowledge Q&A — semantic + AI-first, keyword-agnostic.

Answers from admin KB files (auto-refreshed on admin panel save) in ANY language /
phrasing when the knowledge exists — not only hardcoded keyword lists.

Layers (fast → deep):
  1) Block live-API / catalog / pincode action turns
  2) Heuristic fast paths (optional speed boost)
  3) Main AI router signal (data_channel=kb)
  4) Embedding search across all customer KB files
  5) Optional lightweight LLM classifier (borderline / no-embedding cases)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from services.answer_router import AnswerRouteDecision, try_deterministic_kb_reply
from utils.reasoning_log import log_reasoning

_last_kb_vector_score: float = 0.0

_KB_VECTOR_FAST_ACTIVE: bool = False


def kb_vector_fast_lane_active() -> bool:
    return _KB_VECTOR_FAST_ACTIVE


def _set_kb_vector_fast_lane(active: bool) -> None:
    global _KB_VECTOR_FAST_ACTIVE
    _KB_VECTOR_FAST_ACTIVE = bool(active)

_KB_HANDLERS = frozenset(
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

_KB_READ_INTENTS = frozenset({"general", "refund", "payment", "seller"})
_ACTION_INTENTS = frozenset(
    {
        "product",
        "order",
        "order_history",
        "wishlist",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
    }
)

_SEMANTIC_MIN = float(os.getenv("KNOWLEDGE_SEMANTIC_MIN_SCORE", "0.16") or "0.16")
_SEMANTIC_STRONG = float(os.getenv("KNOWLEDGE_SEMANTIC_STRONG_SCORE", "0.22") or "0.22")
_LLM_CLASSIFY_MIN = float(os.getenv("KNOWLEDGE_LLM_CLASSIFY_MIN_CONF", "0.72") or "0.72")


def _skip_kb_borderline_llm(*, embedding_only: bool = False) -> bool:
    """Avoid extra LLM loops when brain owns routing or budget is tight."""
    if embedding_only:
        return True
    try:
        from services.chat_flow_telemetry import (
            llm_budget_exceeded,
            should_defer_micro_classifiers_to_brain,
        )

        if llm_budget_exceeded():
            return True
        if should_defer_micro_classifiers_to_brain():
            return True
    except ImportError:
        pass
    return False


@dataclass
class SemanticKnowledgeDecision:
    """Result of multilingual / semantic KB turn analysis."""

    is_informational: bool = False
    confidence: float = 0.0
    source: str = ""  # heuristic | ai_route | embedding | llm | route_handler
    retrieval_query: str = ""
    user_meaning_en: str = ""
    kb_hit: dict[str, Any] | None = None
    kb_keys: list[str] = field(default_factory=list)
    handler: str = "dynamic_kb"
    reason: str = ""


def build_semantic_retrieval_query(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> str:
    """
    Query for embedding search — merges AI meaning, English gloss, and context expansion.
    Works across scripts/languages without keyword lists.
    """
    try:
        from services.kb_service import build_kb_retrieval_query

        return build_kb_retrieval_query(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
        )
    except ImportError:
        pass

    parts: list[str] = []
    meaning = ((ai_route or {}).get("user_meaning") or "").strip()
    if meaning:
        parts.append(meaning)
    en = (msg_en or "").strip()
    raw = (original_msg or "").strip()
    if en and en.lower() != raw.lower():
        parts.append(en)
    if raw:
        parts.append(raw)
    return " — ".join(parts) if parts else raw


def _turn_has_live_order_id(original_msg: str, msg_en: str = "") -> bool:
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if re.search(r"\b\d{6,}\b", comb):
        return True
    try:
        from utils.helpers import extract_order_id

        return bool(extract_order_id(original_msg, msg_en))
    except ImportError:
        return False


def _try_strong_faq_informational_decision(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    retrieval_query: str = "",
) -> SemanticKnowledgeDecision | None:
    """
    Admin faqs.txt match beats order-invoice / coupon heuristics when no Order ID given.
    Never wins over order-history / wishlist list-in-chat (Groq account_list_kind).
    """
    try:
        from services.account_list_semantics import (
            ai_route_requests_order_history_in_chat,
            ai_route_requests_wishlist_in_chat,
        )

        from services.account_list_semantics import (
            turn_requests_purchase_history_in_chat,
            turn_requests_wishlist_in_chat,
        )

        if (
            turn_requests_purchase_history_in_chat(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
            or turn_requests_wishlist_in_chat(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
            or ai_route_requests_wishlist_in_chat(ai_route)
        ):
            return None
    except ImportError:
        pass

    if _turn_has_live_order_id(original_msg, msg_en):
        return None
    try:
        from utils.helpers import (
            message_is_seller_on_welfog_request,
            message_is_welfog_about_request,
        )
        from services.kb_service import _user_requests_grievance_channel

        if (
            message_is_seller_on_welfog_request(f"{original_msg} {msg_en}")
            or message_is_welfog_about_request(f"{original_msg} {msg_en}")
            or _user_requests_grievance_channel(f"{original_msg} {msg_en}")
        ):
            return None
    except ImportError:
        pass
    try:
        from utils.helpers import turn_is_catalog_product_lookup

        if turn_is_catalog_product_lookup(original_msg, msg_en, ai_route):
            return None
    except ImportError:
        pass

    from services.kb_service import resolve_best_faq_chunk_for_question

    faq = resolve_best_faq_chunk_for_question(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if not faq:
        return None
    score = float(faq.get("score") or 0)
    if score < 0.40:
        return None

    out = SemanticKnowledgeDecision(
        is_informational=True,
        confidence=min(0.93, 0.52 + score * 0.42),
        source="embedding",
        retrieval_query=retrieval_query
        or build_semantic_retrieval_query(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ),
        kb_hit=faq,
        kb_keys=["faqs"],
        handler="kb_grounded_ai" if score >= 0.28 else "dynamic_kb",
        reason=f"Strong FAQ match (faqs.txt score={score:.2f}).",
    )
    return out


def _turn_is_live_action_required(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> bool:
    """True when user needs live API / catalog — NOT read-only KB."""
    try:
        from utils.helpers import turn_is_catalog_product_lookup

        if turn_is_catalog_product_lookup(original_msg, msg_en, ai_route):
            return True
    except ImportError:
        pass

    from utils.helpers import (
        _text_is_pincode_serviceability_question,
        user_turn_qualifies_for_live_order_api,
    )
    from services.order_details_flow import (
        message_wants_order_details_or_invoice,
        text_asks_invoice_howto_navigation,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()

    if text_asks_invoice_howto_navigation(comb, conversation_context):
        return False

    try:
        from services.account_list_semantics import (
            turn_requests_purchase_history_in_chat,
            turn_requests_wishlist_in_chat,
        )

        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return True
        if turn_requests_wishlist_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return True
    except ImportError:
        pass

    try:
        from utils.helpers import (
            _text_has_past_order_complaint_context,
            message_is_past_purchase_list_request,
        )

        if _text_has_past_order_complaint_context(comb) and not message_is_past_purchase_list_request(
            comb
        ):
            return False
    except ImportError:
        pass

    if not _turn_has_live_order_id(original_msg, msg_en):
        try:
            from services.account_list_semantics import ai_route_requests_order_history_howto

            if ai_route_requests_order_history_howto(ai_route):
                pass
            else:
                from services.kb_service import resolve_best_faq_chunk_for_question

                faq = resolve_best_faq_chunk_for_question(
                    original_msg, msg_en, conversation_context, ai_route=ai_route
                )
                if faq and float(faq.get("score") or 0) >= 0.42:
                    return False
        except ImportError:
            from services.kb_service import resolve_best_faq_chunk_for_question

            faq = resolve_best_faq_chunk_for_question(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
            if faq and float(faq.get("score") or 0) >= 0.42:
                return False

    if user_turn_qualifies_for_live_order_api(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return True
    if message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return True
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return True

    r = ai_route or {}
    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    if intent in _ACTION_INTENTS and channel in ("live_api", "catalog"):
        return True
    if intent == "product" and r.get("run_catalog_search"):
        return True
    if intent in ("order", "refund", "payment") and r.get("needs_order_id"):
        if re.search(r"\b\d{4,}\b", comb):
            return True
    if intent in ("order_history", "wishlist") and channel == "live_api":
        return True
    return False


def _heuristic_informational_signal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Fast path — optional; semantic layers still run when this is False."""
    from utils.helpers import (
        _text_asks_customer_care_contact,
        _text_is_refund_return_policy_howto,
        message_is_knowledge_information_request,
        message_is_seller_on_welfog_request,
        message_is_welfog_about_request,
        message_needs_policy_answer,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if _text_asks_customer_care_contact(comb):
        return True
    if message_is_welfog_about_request(comb):
        return True
    if _text_is_refund_return_policy_howto(comb) or message_needs_policy_answer(comb):
        return True
    if message_is_seller_on_welfog_request(comb):
        return True
    if message_is_knowledge_information_request(comb, conversation_context):
        return True
    return False


def _ai_route_suggests_kb_read(ai_route: dict | None) -> tuple[bool, float, str]:
    """Trust main router when it chose KB read (any language via user_meaning)."""
    r = ai_route or {}
    if r.get("llm_unavailable"):
        return False, 0.0, ""

    scope = (r.get("conversation_scope") or "").strip().lower()
    if scope in ("out_of_domain", "general_chitchat", "harm_sensitive"):
        return False, 0.0, ""
    if r.get("is_welfog_related") is False:
        return False, 0.0, ""

    intent = (r.get("intent") or "general").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    strategy = (r.get("answer_strategy") or "").strip().lower()
    mk = (r.get("meta_kind") or "none").strip().lower()

    if mk not in ("none", ""):
        return False, 0.0, ""
    if r.get("run_catalog_search"):
        return False, 0.0, ""

    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if ai_route_is_kb_read(r):
            conf = 0.88
            meaning = (r.get("user_meaning") or "").strip()
            if len(meaning) >= 10:
                conf += 0.05
            if (r.get("kb_keys") or []):
                conf += 0.02
            return True, min(conf, 0.96), meaning
    except ImportError:
        pass

    kb_channel = channel == "kb"
    kb_strategy = strategy in ("kb_only", "kb_then_ai", "api_kb_ai")
    kb_intent = intent in _KB_READ_INTENTS

    if kb_channel and kb_intent and not r.get("needs_order_id"):
        conf = 0.82
        meaning = (r.get("user_meaning") or "").strip()
        if len(meaning) >= 10:
            conf += 0.06
        return True, min(conf, 0.95), meaning

    if kb_strategy and kb_intent and not r.get("needs_order_id"):
        return True, 0.78, (r.get("user_meaning") or "").strip()

    return False, 0.0, ""


def _kb_vector_lane_allowed(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    KB vector fast lane — block only explicit live-data turns (order/PIN/pro id).
    Intent (catalog vs KB) is decided by ai_classify_kb_turn, not keyword lists.
    """
    import re

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    if _turn_has_live_order_id(original_msg, msg_en):
        return False
    try:
        from utils.helpers import extract_order_id, extract_product_id

        comb_low = comb.lower()
        if extract_product_id(comb_low) or extract_order_id(comb_low):
            return False
        if re.search(r"\b[1-9]\d{5}\b", comb):
            return False
    except ImportError:
        pass
    return True


def _semantic_kb_match(
    retrieval_query: str,
    *,
    min_score: float | None = None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    wide_search: bool = False,
    fast_lane: bool = False,
) -> dict[str, Any] | None:
    """Embedding + topic-scoped search across customer KB files (semantic only)."""
    q = (retrieval_query or "").strip()
    if not q:
        return None

    from services.kb_service import (
        get_customer_kb_keys,
        log_kb_retrieval,
        retrieve_best_kb_chunk,
        top_kb_hits,
    )

    floor = min_score if min_score is not None else _SEMANTIC_MIN
    if wide_search:
        category = "general"
        keys = get_customer_kb_keys()
        hit = retrieve_best_kb_chunk(
            q,
            keys=keys,
            ai_route=ai_route,
            min_score=floor,
            conflict_check=True,
        )
        hits = [hit] if hit else []
    else:
        from services.query_understanding import infer_kb_query_category, scoped_kb_keys_for_retrieval

        category = infer_kb_query_category(
            original_msg or q,
            msg_en,
            ai_route=ai_route,
            conversation_context=conversation_context,
        )
        keys = scoped_kb_keys_for_retrieval(category, ai_route=ai_route, user_meaning=q)
        if not keys:
            keys = get_customer_kb_keys()
        hits = top_kb_hits(q, keys=keys, min_score=floor, top_n=8, log_retrieval=False)
        hit = _pick_best_semantic_kb_hit(q, hits, query_category=category)
        if not hit:
            hit = retrieve_best_kb_chunk(q, keys=keys, ai_route=ai_route, min_score=floor)
    meaning = ((ai_route or {}).get("user_meaning") or "").strip()
    if hit or hits:
        log_kb_retrieval(
            query_intent=category,
            query_meaning=meaning,
            retrieval_query=q,
            matched_file=str((hit or hits[0]).get("source") or ""),
            selected_category=category,
            similarity_score=float((hit or hits[0]).get("score") or 0),
            selected_chunks=hits[:4] if hits else ([hit] if hit else []),
        )
    return hit


_POLICY_KB_STEMS = frozenset(
    {"refund", "payment", "faqs", "shipping", "support", "seller", "terms", "privacy"}
)
_OVERVIEW_KB_STEMS = frozenset({"company", "about", "story", "intro", "brand"})


_TOPIC_SOURCE_AFFINITY: dict[str, frozenset[str]] = {
    "refund": frozenset({"refund", "faqs", "terms", "shipping"}),
    "return": frozenset({"refund", "faqs", "terms", "shipping"}),
    "payment": frozenset({"payment", "faqs", "terms"}),
    "shipping": frozenset({"shipping", "faqs", "terms"}),
    "seller": frozenset({"seller", "faqs", "support"}),
    "privacy": frozenset({"privacy", "faqs", "terms", "company"}),
    "terms": frozenset({"terms", "faqs", "privacy"}),
    "order": frozenset({"faqs", "support", "shipping"}),
    "order_history": frozenset({"faqs", "support"}),
}


def _pick_best_semantic_kb_hit(
    query: str,
    hits: list[dict[str, Any]],
    *,
    query_category: str = "general",
) -> dict[str, Any] | None:
    """
    Rerank top embedding hits — prefer policy file over company overview;
    penalize cross-topic chunks (refund vs order history vs shipping).
    """
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]

    try:
        from services.kb_service import _kb_focus_profile
    except ImportError:
        _kb_focus_profile = None

    include_terms, exclude_terms, _ = (
        _kb_focus_profile(query) if _kb_focus_profile else (set(), set(), 3)
    )
    affinity = _TOPIC_SOURCE_AFFINITY.get((query_category or "general").strip().lower(), frozenset())

    def _adjusted(h: dict[str, Any]) -> float:
        s = float(h.get("score") or 0)
        src = (h.get("source") or "").lower()
        chunk_l = re.sub(r"<br\s*/?>", " ", (h.get("chunk") or "")).lower()
        if affinity and not any(a in src for a in affinity):
            s -= 0.055
        if query_category == "shipping" and "shipping" in src:
            s += 0.05
        if query_category == "refund" and "refund" in src:
            s += 0.05
        if query_category == "payment" and "payment" in src:
            s += 0.05
        if query_category == "seller" and "seller" in src:
            s += 0.05
        if exclude_terms and any(t in chunk_l for t in exclude_terms):
            s -= 0.09
        if include_terms:
            hit_n = sum(1 for t in include_terms if t in chunk_l)
            miss_n = sum(1 for t in exclude_terms if t in chunk_l)
            s += 0.04 * hit_n - 0.06 * miss_n
        return s

    ranked = sorted(hits, key=_adjusted, reverse=True)
    top = ranked[0]
    orig_top = sorted(hits, key=lambda h: float(h.get("score") or 0), reverse=True)[0]
    if top.get("source") != orig_top.get("source"):
        log_reasoning(
            f"KB rerank ({query_category}): prefer {top.get('source')} over {orig_top.get('source')} "
            f"(adj { _adjusted(top):.2f} vs { _adjusted(orig_top):.2f})."
        )

    if float(top.get("score") or 0) >= _SEMANTIC_STRONG:
        return top

    top_src = (top.get("source") or "").lower()
    top_is_overview = any(s in top_src for s in _OVERVIEW_KB_STEMS)

    for alt in ranked[1:]:
        gap = float(top.get("score") or 0) - float(alt.get("score") or 0)
        if gap > 0.05:
            break
        alt_src = (alt.get("source") or "").lower()
        alt_is_policy = any(s in alt_src for s in _POLICY_KB_STEMS)
        if top_is_overview and alt_is_policy and gap <= 0.05:
            log_reasoning(
                f"KB rerank: prefer {alt_src} over {top_src} "
                f"(scores {alt.get('score'):.2f} vs {top.get('score'):.2f})."
            )
            return alt

    if top_is_overview and float(top.get("score") or 0) < _SEMANTIC_STRONG:
        try:
            from services.kb_service import _embedding_similarity

            best = top
            best_s = _adjusted(top)
            for cand in ranked[:4]:
                chunk = (cand.get("chunk") or "").strip()
                if not chunk:
                    continue
                sim = _embedding_similarity(query, re.sub(r"<br\s*/?>", " ", chunk))
                combined = _adjusted(cand) * 0.55 + sim * 0.45
                if combined > best_s + 0.02:
                    best = cand
                    best_s = combined
            return best
        except Exception:
            pass

    return top


KB_TOPIC_KEYS: dict[str, list[str]] = {
    "delivery_shipping": ["shipping", "faqs"],
    "refund_return_policy": ["refund", "faqs", "shipping"],
    "payment_fees": ["payment", "faqs"],
    "company_faq": ["company", "faqs"],
    "welfog_social": ["company"],
    "seller": ["seller", "support"],
    "contact_support": ["support", "faqs"],
    "privacy_terms": ["privacy", "terms", "faqs"],
    "order_howto": ["faqs", "welfog_api"],
    "general_faq": ["faqs"],
    "none": ["faqs"],
}


def ai_classify_kb_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
    preflight: bool = False,
) -> dict[str, Any] | None:
    """
    Primary micro-LLM for KB vs live-API — infers meaning in ANY language (no keyword lists).
    Cached per turn via turn_intent_coordinator.get_kb_turn_ai_classification.
    """
    enabled = (os.getenv("ENABLE_KB_TURN_LLM", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return None

    if not preflight:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                log_reasoning(
                    "KB-turn LLM: defer/skip — universal brain route owns classification."
                )
                return None
        except ImportError:
            pass

    try:
        from services.ai_service import (
            _compact_conversation_context,
            _llm_json_with_provider_fallback,
            _llm_classifier_provider_chain,
            _trim_text_mid,
        )
        from services.translation_service import resolve_customer_reply_lang
    except ImportError:
        return None

    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            peek_kb_turn_classification,
            store_kb_turn_classification,
        )

        cached = peek_kb_turn_classification(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if cached is not _KB_CACHE_UNSET:
            if cached is None:
                return None
            log_reasoning("KB-turn LLM: reuse cached classification (same turn).")
            return cached
    except ImportError:
        pass

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1400)
    meaning_hint = ((ai_route or {}).get("user_meaning") or "").strip()
    user_line = _trim_text_mid(comb, 520)

    system = """You classify the LATEST user message in Welfog e-commerce support chat.

Customers write in ANY language, script, or style (English, Hinglish, Hindi, Tamil, Telugu,
Bengali, Marathi, Gujarati, Kannada, Malayalam, Urdu, voice-to-text typos). Infer MEANING —
never match fixed keyword lists or example phrases.

Return ONLY valid JSON:
{
  "user_meaning_en": "one English sentence — what they want THIS turn",
  "is_informational_kb": true/false,
  "is_refund_or_return": true/false,
  "needs_live_api": true/false,
  "kb_topic": "none"|"delivery_shipping"|"refund_return_policy"|"payment_fees"|"company_faq"|"welfog_social"|"seller"|"contact_support"|"privacy_terms"|"order_howto"|"general_faq",
  "live_api_kind": "none"|"pincode"|"order_track"|"order_details"|"wishlist"|"order_history"|"refund_status"|"product_search",
  "confidence": 0.0-1.0
}

is_informational_kb=true — user wants to READ Welfog FAQ/policy/how-it-works (no personal fetch now):
• General delivery/shipping TIME or duration (any phrasing) → delivery_shipping
• Refund/return POLICY or general steps → refund_return_policy, is_refund_or_return=true
• Payment methods, fees, COD → payment_fees
• What is Welfog, company info → company_faq
• Welfog official social media / Instagram / YouTube / Facebook links → welfog_social
• Seller registration/login on Welfog → seller
• Customer care phone/email → contact_support
• Privacy, terms → privacy_terms
• How to place order / track in app (steps, not MY list) → order_howto

needs_live_api=true — personal or live data NOW:
• PIN/serviceability for a named PIN → pincode (NOT delivery_shipping timeline FAQ)
• Track MY order, order details for MY id → order_track / order_details
• MY wishlist / MY order history list → wishlist / order_history
• MY refund status / money not received for MY order → refund_status
• Find/buy/show products → product_search

is_informational_kb=false: greetings, thanks, jokes, other companies' offers/policies/social
handles, off-topic — infer semantically in ANY language (never keyword lists).

Critical: general "how long does delivery take" = informational_kb + delivery_shipping.
Checking if Welfog delivers to pincode 302012 = needs_live_api + pincode."""

    user_payload = f"LATEST USER MESSAGE:\n{user_line}"
    if meaning_hint:
        user_payload += f"\nRouter hint (English): {meaning_hint[:200]}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    try:
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload[:1600]},
            ],
            max_tokens=200,
            timeout_sec=12,
            max_attempts=2,
            temperature=0.1,
        )
    except Exception as exc:
        log_reasoning(f"KB turn LLM classify skipped: {exc}")
        return None

    if not data:
        return None

    topic = (data.get("kb_topic") or "none").strip().lower()
    if topic not in KB_TOPIC_KEYS:
        topic = "general_faq" if data.get("is_informational_kb") else "none"
    live = (data.get("live_api_kind") or "none").strip().lower()
    conf = float(data.get("confidence") or 0.0)
    um = (data.get("user_meaning_en") or data.get("user_meaning") or "").strip()
    log_reasoning(
        f"KB-turn LLM: info={bool(data.get('is_informational_kb'))} "
        f"live={bool(data.get('needs_live_api'))} topic={topic} conf={conf:.2f} — {um[:90]}"
    )
    result = {
        "user_meaning_en": um,
        "is_informational_kb": bool(data.get("is_informational_kb")),
        "is_refund_or_return": bool(data.get("is_refund_or_return")),
        "needs_live_api": bool(data.get("needs_live_api")),
        "kb_topic": topic,
        "live_api_kind": live,
        "confidence": conf,
    }
    try:
        from services.turn_intent_coordinator import store_kb_turn_classification

        store_kb_turn_classification(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang,
            result,
        )
    except ImportError:
        pass
    return result


def ai_classify_informational_kb_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> dict[str, Any] | None:
    """
    Lightweight LLM: is this a read-only Welfog KB question (any language)?
    Prefer ai_classify_kb_turn (cached) — this remains for analyze_informational fallback.
    """
    enabled = (os.getenv("ENABLE_KB_INFORMATION_LLM", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return None

    try:
        from services.chat_flow_telemetry import ai_route_already_decided, skip_step

        if ai_route_already_decided(ai_route):
            channel = ((ai_route or {}).get("data_channel") or "").strip().lower()
            intent = ((ai_route or {}).get("intent") or "").strip().lower()
            if channel == "live_api":
                skip_step("kb_information_llm", "live_api turn decided")
                return None
            if intent in ("seller", "refund", "payment", "general", "privacy", "terms") and channel == "kb":
                skip_step("kb_information_llm", f"router locked {intent}/kb")
                return None
    except ImportError:
        pass

    try:
        from services.ai_service import (
            _llm_json_with_provider_fallback,
            _llm_classifier_provider_chain,
        )
    except ImportError:
        return None

    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    meaning_hint = ((ai_route or {}).get("user_meaning") or "").strip()
    compact_ctx = (conversation_context or "")[-800:].strip()
    user_block = f"Latest message:\n{original_msg.strip()}"
    if msg_en and msg_en.strip().lower() != (original_msg or "").strip().lower():
        user_block += f"\nEnglish gloss:\n{msg_en.strip()}"
    if meaning_hint:
        user_block += f"\nRouter meaning (English): {meaning_hint}"
    if compact_ctx:
        user_block += f"\n\nRecent chat:\n{compact_ctx[-600:]}"

    system = """You classify Welfog e-commerce support chat turns.
Return ONLY JSON:
{
  "is_informational_kb": true/false,
  "needs_live_api": true/false,
  "user_meaning_en": "One English sentence: what the customer wants THIS turn",
  "confidence": 0.0-1.0
}

is_informational_kb=true when the user wants to READ/LEARN Welfog policy, FAQ, company info,
seller rules, fees, contact details, how something works — and does NOT need their personal
order/wishlist/product list fetched right now.

needs_live_api=true when they need live data: track MY order, show MY wishlist, check delivery
to a PIN, browse/buy products, refund status for THEIR order id.

Any Indian language / Hinglish / typos — judge by MEANING not exact words.
Pure greetings/thanks/jokes/off-topic → is_informational_kb=false."""

    try:
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_block[:1400]},
            ],
            max_tokens=120,
            timeout_sec=9,
            max_attempts=2,
            temperature=0.15,
        )
    except Exception as exc:
        log_reasoning(f"KB information LLM classify skipped: {exc}")
        return None

    if not data:
        return None
    return data


def analyze_informational_knowledge_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    route_decision: AnswerRouteDecision | None = None,
    embedding_only: bool = False,
) -> SemanticKnowledgeDecision:
    """
    Multilingual semantic analysis — keywords optional, not required.
    """
    out = SemanticKnowledgeDecision(
        retrieval_query=build_semantic_retrieval_query(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        )
    )

    if embedding_only:
        if _turn_has_live_order_id(original_msg, msg_en):
            out.reason = "Order ID in message — live API lane."
            return out
        hit = _semantic_kb_match(
            out.retrieval_query,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
            ai_route=ai_route,
            wide_search=True,
            fast_lane=True,
        )
        if hit:
            score = float(hit.get("score") or 0)
            if score >= _SEMANTIC_MIN:
                out.is_informational = True
                out.confidence = min(0.93, 0.55 + score * 0.38)
                out.source = "embedding"
                out.kb_hit = hit
                src = (hit.get("source") or "").strip()
                out.kb_keys = [src] if src else []
                out.handler = "kb_grounded_ai" if score >= 0.28 else "dynamic_kb"
                out.reason = f"KB vector lane (score={score:.2f})."
                return out
        out.reason = "No KB embedding match (vector lane)."
        return out

    try:
        from services.turn_intent_coordinator import (
            get_kb_turn_ai_classification,
            kb_turn_is_informational,
        )

        if not embedding_only:
            kb_cls = get_kb_turn_ai_classification(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
            if kb_turn_is_informational(kb_cls):
                topic = (kb_cls.get("kb_topic") or "general_faq").strip().lower()
                keys = list(KB_TOPIC_KEYS.get(topic) or KB_TOPIC_KEYS["general_faq"])
                um = (kb_cls.get("user_meaning_en") or "").strip()
                if um:
                    out.retrieval_query = build_semantic_retrieval_query(
                        original_msg,
                        msg_en,
                        conversation_context,
                        ai_route={"user_meaning": um, **(ai_route or {})},
                    )
                out.is_informational = True
                out.confidence = float(kb_cls.get("confidence") or 0.0)
                out.source = "kb_turn_llm"
                out.user_meaning_en = um
                out.kb_keys = keys
                out.handler = "kb_grounded_ai"
                out.reason = f"KB-turn LLM: {topic} (any language)."
                faq_early = _try_strong_faq_informational_decision(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route={"user_meaning": um} if um else ai_route,
                    retrieval_query=out.retrieval_query,
                )
                if faq_early:
                    return faq_early
                return out
    except ImportError:
        pass

    try:
        from utils.helpers import (
            _text_has_order_placement_intent,
            _text_is_order_id_help_request,
        )

        comb_early = f"{original_msg or ''} {msg_en or ''}".strip()
        if _text_is_order_id_help_request(comb_early):
            out.is_informational = True
            out.source = "heuristic"
            out.confidence = 0.92
            out.handler = "order_id_help_kb"
            out.kb_keys = list((ai_route or {}).get("kb_keys") or ["faqs", "welfog_api_order_id"])
            out.reason = "Where to find Order ID — KB steps."
            return out
        if _text_has_order_placement_intent(comb_early):
            out.is_informational = True
            out.source = "heuristic"
            out.confidence = 0.9
            out.handler = "order_placement_kb"
            out.kb_keys = list((ai_route or {}).get("kb_keys") or ["faqs"])
            out.reason = "How to place order — KB steps."
            return out

        from services.account_list_semantics import (
            ai_route_requests_order_history_in_chat,
            ai_route_requests_wishlist_howto,
        )

        if ai_route_requests_wishlist_howto(
            ai_route, original_msg, msg_en, conversation_context
        ):
            out.is_informational = True
            out.source = "ai_route"
            out.confidence = 0.88
            out.handler = "wishlist_howto_kb"
            out.kb_keys = list((ai_route or {}).get("kb_keys") or ["faqs", "welfog_api_wishlist"])
            out.reason = "Wishlist how-to in app — KB steps (Groq account_list_kind)."
            return out
        from services.account_list_semantics import turn_requests_purchase_history_in_chat

        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            out.is_informational = False
            out.reason = "Order history list in chat — live API (not FAQ how-to)."
            return out
        from services.account_list_semantics import turn_requests_wishlist_in_chat

        if turn_requests_wishlist_in_chat(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            out.is_informational = False
            out.reason = "Wishlist in chat — live API (skip KB retrieval)."
            return out
        try:
            from services.early_live_dispatch import turn_blocks_kb_pre_scope

            live_block = turn_blocks_kb_pre_scope(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
            )
            if live_block in (
                "live_api_route",
                "refund_status",
                "order_details_invoice",
                "live_order_lookup",
            ):
                out.is_informational = False
                out.reason = f"Live API turn ({live_block}) — skip KB retrieval."
                return out
        except ImportError:
            pass
        from services.refund_status_semantics import ai_route_requests_refund_status_lookup

        if ai_route_requests_refund_status_lookup(
            ai_route, original_msg, msg_en, conversation_context
        ):
            out.is_informational = False
            out.reason = "Personal refund status — return-request API (not policy KB)."
            return out
    except ImportError:
        pass

    faq_early = _try_strong_faq_informational_decision(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        retrieval_query=out.retrieval_query,
    )
    if faq_early:
        return faq_early

    from services.semantic_intent import skip_keyword_intent_routes

    if not skip_keyword_intent_routes(ai_route):
        comb_early = f"{original_msg or ''} {msg_en or ''}".strip()
        try:
            from utils.helpers import (
                _text_has_past_order_complaint_context,
                message_is_seller_on_welfog_request,
                message_is_welfog_about_request,
            )
            from services.kb_service import _user_requests_grievance_channel

            if message_is_seller_on_welfog_request(comb_early):
                out.is_informational = True
                out.confidence = 0.9
                out.source = "keyword_failsafe"
                out.handler = "seller_kb"
                out.kb_keys = ["seller", "support"]
                out.reason = "Keyword failsafe: seller account — seller KB."
                return out
            from services.location_delivery_resolver import turn_requests_delivery_serviceability

            if turn_requests_delivery_serviceability(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                allow_llm=True,
            ):
                out.is_informational = False
                out.confidence = 0.92
                out.source = "keyword_failsafe"
                out.handler = "pincode_delivery_api"
                out.kb_keys = ["welfog_api_pincode_delivery", "shipping"]
                out.reason = "Keyword failsafe: delivery/serviceability — pincode API."
                return out
            if _user_requests_grievance_channel(comb_early):
                out.is_informational = True
                out.confidence = 0.9
                out.source = "keyword_failsafe"
                out.handler = "dynamic_kb"
                out.kb_keys = ["company", "privacy"]
                out.reason = "Keyword failsafe: Grievance Officer — company KB."
                return out
            if message_is_welfog_about_request(comb_early):
                out.is_informational = True
                out.confidence = 0.88
                out.source = "keyword_failsafe"
                out.handler = "welfog_about_kb"
                out.kb_keys = ["company", "faqs"]
                out.reason = "Keyword failsafe: What is Welfog — company KB."
                return out
            if _text_has_past_order_complaint_context(comb_early):
                out.is_informational = True
                out.confidence = 0.86
                out.source = "keyword_failsafe"
                out.handler = "dynamic_kb"
                out.kb_keys = ["refund", "faqs", "shipping"]
                out.reason = "Keyword failsafe: wrong/damaged item — refund KB."
                return out
        except ImportError:
            pass

    route_ok, route_conf, route_meaning = _ai_route_suggests_kb_read(ai_route)
    if route_ok:
        out.is_informational = True
        out.confidence = route_conf
        out.source = "ai_route"
        out.user_meaning_en = route_meaning
        out.kb_keys = list((ai_route or {}).get("kb_keys") or [])
        intent_ai = ((ai_route or {}).get("intent") or "").strip().lower()
        if intent_ai == "seller":
            out.handler = "seller_kb"
        elif intent_ai == "refund":
            out.handler = "dynamic_kb"
        else:
            out.handler = "dynamic_kb"
        out.reason = "Groq classified KB read (any language)."
        hit = _semantic_kb_match(
            out.retrieval_query,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
            ai_route=ai_route,
        )
        if hit:
            out.kb_hit = hit
            src = (hit.get("source") or "").strip()
            if src and src not in out.kb_keys:
                out.kb_keys = [src] + out.kb_keys
        return out

    if _turn_is_live_action_required(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        out.is_informational = False
        out.reason = "Live API / catalog action required."
        return out

    rh = (route_decision.handler or "").strip() if route_decision else ""
    if rh in _KB_HANDLERS and (route_decision.source or "") in ("kb", "kb_ai"):
        out.is_informational = True
        out.confidence = 0.88
        out.source = "route_handler"
        out.handler = rh
        out.kb_keys = list(route_decision.kb_keys or [])
        out.reason = route_decision.reason or "Router chose KB handler."
        return out

    if not skip_keyword_intent_routes(ai_route) and _heuristic_informational_signal(
        original_msg, msg_en, conversation_context
    ):
        out.is_informational = True
        out.confidence = 0.85
        out.source = "keyword_failsafe"
        out.reason = "Keyword failsafe informational signal (LLM unavailable)."
        return out

    hit = _semantic_kb_match(
        out.retrieval_query,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
        ai_route=ai_route,
    )
    if hit:
        score = float(hit.get("score") or 0)
        src = (hit.get("source") or "").strip().lower()
        if src == "faqs" and score >= 0.18:
            out.is_informational = True
            out.confidence = min(0.92, 0.6 + score * 0.35)
            out.source = "embedding"
            out.kb_hit = hit
            out.kb_keys = ["faqs"]
            out.handler = "kb_grounded_ai" if score >= 0.28 else "dynamic_kb"
            out.reason = f"FAQ knowledge match (faqs.txt score={score:.2f})."
            return out
        if score >= _SEMANTIC_STRONG or (
            isinstance(score, int) and score >= 3
        ):
            out.is_informational = True
            out.confidence = min(0.92, 0.55 + score * 0.35 if score <= 1 else 0.55 + score * 0.08)
            out.source = "embedding"
            out.kb_hit = hit
            src = (hit.get("source") or "").strip()
            out.kb_keys = [src] if src else []
            out.reason = f"Strong KB embedding match (score={score:.2f})."
            if score >= _SEMANTIC_STRONG:
                out.handler = "kb_grounded_ai"
            return out

        if score >= _SEMANTIC_MIN:
            out.kb_hit = hit
            skip_borderline_llm = False
            try:
                from services.ai_route_semantics import ai_route_is_kb_read
                from services.chat_flow_telemetry import ai_route_already_decided

                if ai_route_already_decided(ai_route) and ai_route_is_kb_read(ai_route):
                    skip_borderline_llm = True
                elif ai_route_is_kb_read(ai_route) and score >= 0.30:
                    skip_borderline_llm = True
            except ImportError:
                pass
            if skip_borderline_llm or _skip_kb_borderline_llm(
                embedding_only=embedding_only
            ):
                out.is_informational = True
                out.confidence = min(0.9, 0.62 + score * 0.25)
                out.source = "embedding"
                src = (hit.get("source") or "").strip()
                out.kb_keys = [src] if src else []
                out.reason = (
                    f"KB embedding (skip borderline LLM, score={score:.2f})."
                )
                if score >= 0.28:
                    out.handler = "kb_grounded_ai"
                return out
            llm = ai_classify_informational_kb_turn(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
            if llm and llm.get("is_informational_kb") and not llm.get("needs_live_api"):
                conf = float(llm.get("confidence") or 0.75)
                meaning_en = (llm.get("user_meaning_en") or "").strip()
                if meaning_en:
                    out.user_meaning_en = meaning_en
                    better = _semantic_kb_match(
                        meaning_en,
                        min_score=_SEMANTIC_MIN,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conversation_context=conversation_context,
                        ai_route=ai_route,
                    )
                    if better:
                        out.kb_hit = better
                        hit = better
                if conf >= _LLM_CLASSIFY_MIN:
                    out.is_informational = True
                    out.confidence = conf
                    out.source = "llm"
                    src = (hit.get("source") or "").strip()
                    out.kb_keys = [src] if src else []
                    if float(hit.get("score") or 0) >= 0.35:
                        out.handler = "kb_grounded_ai"
                    out.reason = "LLM confirmed informational + embedding match."
                    return out

            # Borderline embedding without LLM confirm — still answer if clearly Welfog FAQ
            if score >= (_SEMANTIC_MIN + 0.03):
                out.is_informational = True
                out.confidence = 0.62 + score * 0.2
                out.source = "embedding"
                out.kb_hit = hit
                src = (hit.get("source") or "").strip()
                out.kb_keys = [src] if src else []
                out.reason = f"KB embedding match (score={score:.2f})."
                return out

    # No embedding — try LLM alone for long explanatory questions
    if len((original_msg or "").strip()) >= 12:
        llm = ai_classify_informational_kb_turn(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        )
        if llm and llm.get("is_informational_kb") and not llm.get("needs_live_api"):
            conf = float(llm.get("confidence") or 0.0)
            if conf >= _LLM_CLASSIFY_MIN:
                out.is_informational = True
                out.confidence = conf
                out.source = "llm"
                out.user_meaning_en = (llm.get("user_meaning_en") or "").strip()
                rq = out.user_meaning_en or out.retrieval_query
                hit2 = _semantic_kb_match(
                    rq,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                    ai_route=ai_route,
                )
                if hit2:
                    out.kb_hit = hit2
                    src = (hit2.get("source") or "").strip()
                    out.kb_keys = [src] if src else []
                out.reason = "LLM classified informational KB question."
                return out

    out.is_informational = False
    out.reason = "No informational KB signal."
    return out


def turn_is_informational_knowledge_only(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
    route_decision: AnswerRouteDecision | None = None,
) -> bool:
    """
    User wants to READ from KB — any language/phrasing when knowledge exists.
    """
    decision = analyze_informational_knowledge_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        route_decision=route_decision,
    )
    return decision.is_informational


def resolve_informational_kb_route(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    route_decision: AnswerRouteDecision | None = None,
    *,
    semantic: SemanticKnowledgeDecision | None = None,
    ai_route: dict | None = None,
) -> AnswerRouteDecision:
    """Pick KB handler — semantic analysis overrides weak routes."""
    if semantic is None:
        semantic = analyze_informational_knowledge_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            route_decision=route_decision,
        )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    from services.kb_service import resolve_kb_keys_for_question

    if semantic.kb_hit and semantic.source in ("embedding", "llm", "ai_route"):
        src = (semantic.kb_hit.get("source") or "").strip()
        keys = list(dict.fromkeys((semantic.kb_keys or []) + ([src] if src else [])))
        if not keys:
            keys = resolve_kb_keys_for_question(
                semantic.user_meaning_en or original_msg,
                semantic.user_meaning_en or msg_en,
                suggested_keys=semantic.kb_keys or None,
                conversation_context=conversation_context,
                ai_route=ai_route,
            )
        handler = semantic.handler
        hit_score = float((semantic.kb_hit or {}).get("score") or 0)
        if hit_score >= 0.35:
            handler = "kb_grounded_ai"
        if handler not in _KB_HANDLERS:
            handler = "kb_grounded_ai" if semantic.confidence >= 0.75 else "dynamic_kb"
        source = "kb_ai" if handler == "kb_grounded_ai" else "kb"
        return AnswerRouteDecision(
            source=source,
            intent="general",
            handler=handler,
            kb_keys=keys or ["faqs"],
            kb_hit=semantic.kb_hit,
            kb_min_score=_SEMANTIC_MIN,
            reason=semantic.reason or "Semantic KB match.",
        )

    from services.semantic_intent import skip_keyword_intent_routes

    if not skip_keyword_intent_routes(ai_route):
        from utils.helpers import (
            _text_asks_customer_care_contact,
            _text_is_refund_return_policy_howto,
            message_is_knowledge_information_request,
            message_is_seller_on_welfog_request,
            message_is_welfog_about_request,
            message_needs_policy_answer,
        )
        from services.kb_service import _user_requests_grievance_channel

        if _user_requests_grievance_channel(comb):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="dynamic_kb",
                kb_keys=["company", "privacy"],
                reason="Keyword failsafe: Grievance Officer — company KB.",
            )

        if _text_asks_customer_care_contact(comb) and not _user_requests_grievance_channel(comb):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="customer_care_kb",
                kb_keys=["support"],
                reason="Keyword failsafe: customer-care contact.",
            )

        if message_is_welfog_about_request(comb):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="welfog_about_kb",
                kb_keys=["company", "faqs"],
                reason="Keyword failsafe: What is Welfog.",
            )

        if _text_is_refund_return_policy_howto(comb) or message_needs_policy_answer(comb):
            return AnswerRouteDecision(
                source="kb",
                intent="refund",
                handler="dynamic_kb",
                kb_keys=["refund", "faqs", "shipping"],
                reason="Keyword failsafe: return/refund policy.",
            )

        if message_is_seller_on_welfog_request(comb):
            return AnswerRouteDecision(
                source="kb",
                intent="seller",
                handler="seller_kb",
                kb_keys=["seller", "support"],
                reason="Keyword failsafe: seller account.",
            )

        if message_is_knowledge_information_request(comb, conversation_context):
            keys = resolve_kb_keys_for_question(
                original_msg,
                msg_en,
                conversation_context=conversation_context,
                ai_route=ai_route,
            )
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="dynamic_kb",
                kb_keys=keys or ["faqs"],
                reason="Keyword failsafe: policy / FAQ.",
            )

    if semantic.is_informational and semantic.source == "ai_route":
        keys = list(semantic.kb_keys or []) or resolve_kb_keys_for_question(
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            ai_route=ai_route,
        )
        intent_ai = ((ai_route or {}).get("intent") or "general").strip().lower()
        handler = semantic.handler or "dynamic_kb"
        if intent_ai == "seller":
            handler = "seller_kb"
        return AnswerRouteDecision(
            source="kb",
            intent=intent_ai if intent_ai in _KB_READ_INTENTS else "general",
            handler=handler,
            kb_keys=keys,
            kb_hit=semantic.kb_hit,
            reason=semantic.reason or "Groq KB read + embedding retrieval.",
        )

    rh = (route_decision.handler or "").strip() if route_decision else ""
    if route_decision and rh in (
        "wishlist_howto_kb",
        "order_history_howto_kb",
        "order_tracking_howto_kb",
        "order_placement_kb",
    ):
        return route_decision
    if route_decision and rh in _KB_HANDLERS:
        return route_decision

    keys = resolve_kb_keys_for_question(
        semantic.user_meaning_en or original_msg,
        semantic.user_meaning_en or msg_en,
        conversation_context=conversation_context,
        ai_route=ai_route,
    )
    return AnswerRouteDecision(
        source="kb",
        intent="general",
        handler="dynamic_kb",
        kb_keys=keys or ["faqs"],
        reason=semantic.reason or "Informational KB fallback.",
    )


def try_knowledge_vector_only_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
    embedding_only: bool = True,
    allow_no_answer: bool = False,
) -> Optional[str]:
    """
    Vector retrieval → rerank → KB excerpt (no kb_turn / answer-rewrite LLM).
    Primary fast path for Welfog knowledge Q&A.
    """
    import time

    t0 = time.perf_counter()
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    if not _kb_vector_lane_allowed(original_msg, msg_en, conversation_context):
        return None

    from services.translation_service import customer_reply_language, resolve_customer_reply_lang

    rl = reply_lang or resolve_customer_reply_lang(original_msg)

    body = try_knowledge_reply_before_interference(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=rl,
        ai_route=ai_route,
        embedding_only=embedding_only,
        skip_answer_llm=True,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if body:
        try:
            from services.kb_service import log_kb_pipeline_complete

            log_kb_pipeline_complete(
                query=comb,
                language=rl,
                intent="general",
                route="kb_vector_only",
                retrieved_chunks=1,
                similarity_score=_last_kb_vector_score,
                response_time_ms=elapsed_ms,
                source="embedding",
            )
        except ImportError:
            pass
        return body

    if allow_no_answer:
        try:
            from services.kb_service import format_kb_no_information_reply, log_kb_pipeline_complete

            log_kb_pipeline_complete(
                query=comb,
                language=rl,
                intent="general",
                route="kb_no_match",
                retrieved_chunks=0,
                similarity_score=0.0,
                response_time_ms=elapsed_ms,
                source="none",
            )
            return format_kb_no_information_reply(original_msg, reply_lang=rl)
        except ImportError:
            pass
    return None


def try_ai_first_kb_early_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    ai_route: dict | None = None,
    preflight: bool = False,
) -> Optional[str]:
    """
    Early /chat KB: one AI classifier infers meaning + topic (any language/script),
    then vector KB retrieval. No keyword-list routing.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None

    try:
        from services.conversation_zero_llm_fallback import _zero_llm_kb_turn_blocked

        if not preflight and _zero_llm_kb_turn_blocked(
            original_msg, msg_en, conversation_context
        ):
            return None
    except ImportError:
        pass

    if preflight:
        try:
            from services.turn_intent_coordinator import (
                kb_classified_as_live_api,
                kb_turn_is_informational,
                store_kb_turn_classification,
                structural_skip_kb_turn_classifier,
            )
            from services.kb_service import (
                KB_DIRECT_MIN_SCORE,
                best_kb_hit,
                format_direct_reply_from_kb_hit,
            )
            from services.translation_service import resolve_customer_reply_lang

            if structural_skip_kb_turn_classifier(
                original_msg, msg_en, conversation_context
            ):
                return None

            classified = ai_classify_kb_turn(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang=reply_lang,
                ai_route=ai_route,
                preflight=True,
            )
            store_kb_turn_classification(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang,
                classified,
            )

            if not classified:
                log_reasoning(
                    "KB preflight — AI classify unavailable; defer to brain/catalog."
                )
                return None
            if kb_classified_as_live_api(classified):
                log_reasoning(
                    "KB preflight — AI: live API/catalog turn; skip vector KB."
                )
                return None
            if not kb_turn_is_informational(classified):
                log_reasoning(
                    "KB preflight — AI: not informational KB; skip vector KB."
                )
                return None

            topic = (classified.get("kb_topic") or "general_faq").strip().lower()
            keys = list(KB_TOPIC_KEYS.get(topic) or KB_TOPIC_KEYS["general_faq"])
            um = (classified.get("user_meaning_en") or "").strip()
            rq = um or comb
            rl = reply_lang or resolve_customer_reply_lang(original_msg)

            kb_body = ""
            hit = best_kb_hit(rq, keys=keys, min_score=KB_DIRECT_MIN_SCORE)
            if hit and float(hit.get("score") or 0) >= KB_DIRECT_MIN_SCORE:
                kb_body = format_direct_reply_from_kb_hit(
                    hit,
                    original_msg,
                    reply_lang=rl,
                    retrieval_query=rq,
                    fast_lane=True,
                ) or ""
            if kb_body:
                log_reasoning(
                    f"KB preflight — AI topic={topic} + scoped vector ({keys[:3]})."
                )
                return kb_body

            log_reasoning(
                f"KB preflight — AI topic={topic} but no scoped KB hit; defer."
            )
            return None
        except ImportError:
            return None

    if not preflight:
        try:
            from services.chat_flow_telemetry import get_cached_brain_route

            brain = ai_route or get_cached_brain_route()
            if isinstance(brain, dict):
                ch = (brain.get("data_channel") or "").strip().lower()
                if ch in ("live_api", "catalog"):
                    return None
        except ImportError:
            pass

        try:
            from services.chat_flow_telemetry import get_cached_brain_route

            brain = ai_route if isinstance(ai_route, dict) else get_cached_brain_route()
            if isinstance(brain, dict) and not brain.get("llm_unavailable"):
                ch = (brain.get("data_channel") or "").strip().lower()
                scope = (brain.get("conversation_scope") or "").strip().lower()
                intent = (brain.get("intent") or "").strip().lower()
                if ch in ("live_api", "catalog"):
                    return None
                if ch == "kb" or (brain.get("data_channel") or "").strip().lower() == "kb":
                    meaning = (brain.get("user_meaning") or "").strip()
                    keys = list(brain.get("kb_keys") or [])
                    from services.kb_service import KB_DIRECT_MIN_SCORE, direct_kb_search

                    kb_body = direct_kb_search(
                        meaning or comb,
                        keys=keys or None,
                        min_score=KB_DIRECT_MIN_SCORE,
                        zero_llm_fast=True,
                    )
                    if kb_body:
                        log_reasoning("KB vector from brain channel=kb (zero extra LLM).")
                        return kb_body
                if scope in ("out_of_domain", "general_chitchat", "harm_sensitive"):
                    return None
                if intent == "out_of_domain":
                    return None
                meaning = (brain.get("user_meaning") or "").strip()
                keys = list(brain.get("kb_keys") or [])
                rh = (brain.get("route_handler") or "").strip().lower()
                from services.kb_service import (
                    KB_DIRECT_MIN_SCORE,
                    direct_kb_search,
                    format_welfog_social_media_reply_from_kb,
                )

                if "social" in rh:
                    social = format_welfog_social_media_reply_from_kb(
                        original_msg,
                        msg_en,
                        reply_lang=reply_lang,
                        conversation_context=conversation_context,
                        user_meaning_en=meaning,
                        ai_confirmed=True,
                    )
                    if social:
                        return social
                query = meaning or comb
                body = direct_kb_search(
                    query,
                    keys=keys or None,
                    min_score=KB_DIRECT_MIN_SCORE,
                    zero_llm_fast=True,
                )
                if body:
                    log_reasoning(
                        "KB vector from brain meaning — no duplicate kb_turn LLM."
                    )
                    return body
        except ImportError:
            pass

    vec_body = try_knowledge_vector_only_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ai_route=ai_route,
        embedding_only=True,
    )
    if vec_body:
        log_reasoning("KB early — vector retrieval first (no kb_turn LLM).")
        return vec_body

    cls: dict[str, Any] | None = None
    try:
        from services.chat_flow_telemetry import llm_budget_exceeded

        if not llm_budget_exceeded():
            cls = ai_classify_kb_turn(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang=reply_lang,
                ai_route=ai_route,
                preflight=preflight,
            )
    except ImportError:
        cls = ai_classify_kb_turn(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            ai_route=ai_route,
            preflight=preflight,
        )

    from services.kb_service import KB_DIRECT_MIN_SCORE, direct_kb_search

    if isinstance(cls, dict):
        conf = float(cls.get("confidence") or 0.0)
        meaning = (cls.get("user_meaning_en") or "").strip()
        needs_live = bool(cls.get("needs_live_api"))
        is_info = bool(cls.get("is_informational_kb"))

        if needs_live and conf >= 0.48:
            log_reasoning(
                f"AI-first KB skip — live API turn (conf={conf:.2f}): {meaning[:90]}"
            )
            return None
        if not is_info and conf >= 0.50:
            log_reasoning(
                f"AI-first KB skip — not Welfog KB (conf={conf:.2f}): {meaning[:90]}"
            )
            return None

        if is_info and conf >= 0.42:
            topic = (cls.get("kb_topic") or "general_faq").strip().lower()
            if topic == "none":
                topic = "general_faq"
            retrieval_q = meaning or comb

            if topic == "welfog_social":
                from services.kb_service import format_welfog_social_media_reply_from_kb

                social_body = format_welfog_social_media_reply_from_kb(
                    original_msg,
                    msg_en,
                    reply_lang=reply_lang,
                    conversation_context=conversation_context,
                    user_meaning_en=meaning,
                    ai_confirmed=True,
                )
                if social_body:
                    return social_body

            keys = list(KB_TOPIC_KEYS.get(topic) or KB_TOPIC_KEYS["general_faq"])
            body = direct_kb_search(
                retrieval_q,
                keys=keys,
                min_score=KB_DIRECT_MIN_SCORE,
                zero_llm_fast=True,
            )
            if body:
                return body
            body = direct_kb_search(
                retrieval_q,
                keys=None,
                min_score=KB_DIRECT_MIN_SCORE,
                zero_llm_fast=True,
            )
            if body:
                return body

    return try_knowledge_vector_only_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang=reply_lang,
        ai_route=ai_route,
        embedding_only=True,
    )


def try_knowledge_reply_before_interference(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    route_decision: AnswerRouteDecision | None = None,
    ai_route: dict | None = None,
    embedding_only: bool = False,
    skip_answer_llm: bool = False,
) -> Optional[str]:
    """
    KB reply before catalog / live API interference — semantic layers only.
    """
    if embedding_only:
        _set_kb_vector_fast_lane(True)
    try:
        return _try_knowledge_reply_before_interference_impl(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang=reply_lang,
            route_decision=route_decision,
            ai_route=ai_route,
            embedding_only=embedding_only,
            skip_answer_llm=skip_answer_llm,
        )
    finally:
        if embedding_only:
            _set_kb_vector_fast_lane(False)


def _try_knowledge_reply_before_interference_impl(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "",
    route_decision: AnswerRouteDecision | None = None,
    ai_route: dict | None = None,
    embedding_only: bool = False,
    skip_answer_llm: bool = False,
) -> Optional[str]:
    """
    Answer informational questions from live admin KB before chitchat / pincode hijack.
    Semantic layers handle questions in any language when KB has the answer.
    """
    try:
        from services.early_live_dispatch import turn_blocks_kb_pre_scope

        if not embedding_only and turn_blocks_kb_pre_scope(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            route_decision=route_decision,
        ):
            log_reasoning(f"Skip KB pre-scope — live API / personal data.")
            return None
    except ImportError:
        pass

    semantic = analyze_informational_knowledge_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        route_decision=route_decision,
        embedding_only=embedding_only,
    )

    if not semantic.is_informational:
        return None

    if not embedding_only:
        from services.kb_service import ensure_knowledge_cache_fresh

        ensure_knowledge_cache_fresh()

    decision = resolve_informational_kb_route(
        original_msg,
        msg_en,
        conversation_context,
        route_decision,
        semantic=semantic,
        ai_route=ai_route,
    )

    if decision.kb_hit:
        from services.kb_service import format_direct_reply_from_kb_hit

        direct = format_direct_reply_from_kb_hit(
            decision.kb_hit,
            original_msg,
            reply_lang=reply_lang,
            retrieval_query=semantic.retrieval_query,
            fast_lane=embedding_only,
        )
        if direct:
            log_reasoning(
                f"Pre-scope KB direct chunk (score={float(decision.kb_hit.get('score') or 0):.2f})."
            )
            try:
                global _last_kb_vector_score
                _last_kb_vector_score = float(decision.kb_hit.get("score") or 0)
            except Exception:
                pass
            return direct

    if embedding_only:
        return None

    reply = try_deterministic_kb_reply(
        decision,
        original_msg,
        semantic.user_meaning_en or msg_en,
        reply_lang,
        conversation_context,
        ai_route=ai_route,
    )
    if reply:
        log_reasoning(
            f"Pre-scope KB reply: handler={decision.handler} "
            f"via={semantic.source or 'heuristic'} conf={semantic.confidence:.2f}"
        )
        return reply

    if skip_answer_llm:
        return None

    if decision.handler == "kb_grounded_ai" and decision.kb_hit:
        src = (decision.kb_hit.get("source") or "").strip().lower()
        if src == "faqs":
            from services.kb_service import (
                _customer_needs_kb_localization,
                _faq_answer_text_from_chunk,
                _plain_text_to_html_body,
                polish_faq_reply_for_customer,
            )
            from services.translation_service import (
                customer_reply_language,
                finalize_customer_reply,
            )

            rl = reply_lang or customer_reply_language(original_msg)
            excerpt = _faq_answer_text_from_chunk(decision.kb_hit.get("chunk") or "")
            if excerpt and not _customer_needs_kb_localization(original_msg, rl):
                body = polish_faq_reply_for_customer(
                    _plain_text_to_html_body(excerpt) or excerpt,
                    original_msg,
                )
                if body:
                    log_reasoning(
                        "Pre-scope FAQ direct answer (English — skip LLM question echo)."
                    )
                    return finalize_customer_reply(body, original_msg, rl)

        from services.answer_router import try_kb_ai_reply

        reply = try_kb_ai_reply(
            decision, original_msg, conversation_context, reply_lang
        )
        if reply:
            try:
                from services.kb_service import polish_faq_reply_for_customer

                reply = polish_faq_reply_for_customer(reply, original_msg) or reply
            except ImportError:
                pass
            log_reasoning(
                f"Pre-scope semantic KB+AI: {semantic.source} conf={semantic.confidence:.2f}"
            )
            return reply

    if decision.kb_hit:
        from services.answer_router import try_kb_ai_reply

        reply = try_kb_ai_reply(
            decision, original_msg, conversation_context, reply_lang
        )
        if reply:
            log_reasoning("Pre-scope KB+AI fallback from embedding hit.")
            return reply

    return None
