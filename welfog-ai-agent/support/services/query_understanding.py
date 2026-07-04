"""
Query understanding layer — enriches LLM routing before KB / API / AI answer.

- Language, meaning, intent, entities, required action, confidence
- Short/unclear turns: expand with recent conversation for retrieval + routing
- Intent-filtered KB keys (never ground refund answers in payment-only files, etc.)
- Low confidence → clarification in customer's language (no guessing)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

# KB file stems allowed per intent (subset of customer keys; empty = use AI keys only).
_INTENT_KB_HINTS: dict[str, frozenset[str]] = {
    "refund": frozenset({"refund", "faqs", "terms", "shipping", "support"}),
    "return": frozenset({"refund", "faqs", "terms", "shipping", "support"}),
    "payment": frozenset({"payment", "faqs", "terms", "support"}),
    "shipping": frozenset({"shipping", "faqs", "terms", "support", "refund"}),
    "privacy": frozenset({"privacy", "faqs", "terms", "company"}),
    "terms": frozenset({"terms", "faqs", "privacy", "company"}),
    "seller": frozenset({"seller", "faqs", "support", "terms"}),
    "general": frozenset(
        {"faqs", "company", "support", "terms", "privacy", "payment", "refund", "shipping", "seller"}
    ),
    "pincode_check": frozenset({"faqs", "shipping", "support"}),
    "order": frozenset({"faqs", "support", "refund", "shipping"}),
    "order_history": frozenset({"faqs", "support"}),
    "wishlist": frozenset({"faqs", "support"}),
}

_API_INTENTS = frozenset(
    {
        "order",
        "order_history",
        "wishlist",
        "product",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
        "refund",
        "payment",
    }
)

_CLARIFY_MIN_CONF = float(os.getenv("QUERY_UNDERSTANDING_MIN_CONF") or "0.42")


@dataclass
class QueryUnderstanding:
    reply_lang: str = "en"
    user_meaning: str = ""
    intent: str = "general"
    required_action: str = "kb_then_ai"  # live_api | catalog | kb_only | kb_then_ai | clarify | decline | conversational
    entities: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    is_ambiguous: bool = False
    context_expanded_query: str = ""
    filtered_kb_keys: list[str] = field(default_factory=list)
    continue_previous_topic: bool = False


def _combined(original_msg: str, msg_en: str = "") -> str:
    return " ".join(p for p in ((original_msg or "").strip(), (msg_en or "").strip()) if p).strip()


def _is_short_or_vague_message(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    if re.search(r"\d{4,}", raw):
        return False
    tokens = re.findall(r"[a-z0-9\u0900-\u097f]+", raw.lower())
    if len(tokens) <= 3:
        return True
    vague = (
        "ek kaam",
        "sun bhai",
        "sun na",
        "haan bhai",
        "bolo",
        "bata",
        "dikha",
        "wahi",
        "same",
        "uska",
        "iska",
        "yeh wala",
        "wo wala",
    )
    tl = f" {raw.lower()} "
    return any(v in tl for v in vague) and len(tokens) <= 8


def _last_user_snippets(conversation_context: str, max_snippets: int = 3) -> list[str]:
    if not (conversation_context or "").strip():
        return []
    out: list[str] = []
    for line in (conversation_context or "").splitlines():
        low = line.strip().lower()
        if low.startswith("user:") or low.startswith("customer:"):
            body = line.split(":", 1)[-1].strip()
            if body:
                out.append(body)
    return out[-max_snippets:]


def expand_query_with_conversation(
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> str:
    """Build retrieval/routing query; prepend recent user turns when latest message is thin."""
    try:
        from services.chat_flow_telemetry import _TLS

        cached = getattr(_TLS, "expanded_query_cache", None)
        if isinstance(cached, str) and cached:
            return cached
    except ImportError:
        pass

    base = _combined(original_msg, msg_en)
    if not base:
        return base
    continue_topic = bool((ai_route or {}).get("continue_previous_topic"))
    if not continue_topic and not _is_short_or_vague_message(original_msg or msg_en):
        try:
            from services.chat_flow_telemetry import _TLS

            _TLS.expanded_query_cache = base
        except ImportError:
            pass
        return base
    prior = _last_user_snippets(conversation_context)
    if not prior:
        try:
            from services.chat_flow_telemetry import _TLS

            _TLS.expanded_query_cache = base
        except ImportError:
            pass
        return base
    meaning = ((ai_route or {}).get("user_meaning") or "").strip()
    parts = [f"Earlier user: {p}" for p in prior[-2:]]
    parts.append(f"Latest user: {original_msg.strip()}")
    if meaning:
        parts.append(f"Meaning this turn: {meaning}")
    expanded = " | ".join(parts)
    log_reasoning(f"Query context expanded ({len(prior)} prior user turn(s)).")
    try:
        from services.chat_flow_telemetry import _TLS

        _TLS.expanded_query_cache = expanded
    except ImportError:
        pass
    return expanded


def extract_entities(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    conversation_context: str = "",
) -> dict[str, str]:
    """Pincode, order id, product id, search query — from route JSON + safe extractors."""
    r = ai_route or {}
    out: dict[str, str] = {}
    for key in (
        "extracted_pincode",
        "search_query",
        "account_list_kind",
        "numeric_context",
    ):
        v = (r.get(key) or "").strip()
        if v and v.lower() not in ("none", "null", ""):
            out[key] = v
    try:
        from utils.helpers import extract_order_id, extract_pincode_preferred_from_message, extract_product_id

        oid = extract_order_id(original_msg, msg_en)
        if oid:
            out["order_id"] = oid
        pin = (r.get("extracted_pincode") or "").strip() or extract_pincode_preferred_from_message(
            original_msg, conversation_context, ai_route=r
        )
        if pin:
            out["pincode"] = pin
        pid = extract_product_id(f"{original_msg} {msg_en}")
        if pid:
            out["product_id"] = pid
    except ImportError:
        pass
    return out


def _required_action_from_route(route: dict) -> str:
    intent = (route.get("intent") or "general").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    strategy = (route.get("answer_strategy") or "").strip().lower()
    scope = (route.get("conversation_scope") or "").strip().lower()
    mk = (route.get("meta_kind") or "none").strip().lower()

    if scope in ("out_of_domain",) or intent == "out_of_domain":
        return "decline"
    if scope == "general_chitchat" or mk in ("conversational", "assistant_intro"):
        return "conversational"
    if channel == "live_api" or intent in _API_INTENTS and channel != "kb":
        if intent in ("order", "refund", "payment") and route.get("needs_order_id"):
            return "live_api"
        if intent in ("order_history", "wishlist", "pincode_check", "deals", "categories", "category_feed"):
            return "live_api"
        if intent == "order":
            return "live_api"
    if channel == "catalog" or (intent == "product" and route.get("run_catalog_search")):
        return "catalog"
    if strategy in ("kb_only",):
        return "kb_only"
    if strategy in ("live_api_only", "api_only"):
        return "live_api"
    if strategy in ("catalog_only",):
        return "catalog"
    if channel == "kb":
        return "kb_then_ai"
    return "kb_then_ai"


def score_routing_confidence(
    route: dict | None,
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
) -> float:
    """Heuristic confidence 0–1 from LLM route quality + message clarity."""
    if not route or route.get("llm_unavailable"):
        return 0.25
    conf = 0.55
    meaning = (route.get("user_meaning") or "").strip()
    reasoning = (route.get("reasoning") or "").strip()
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()

    if len(meaning) >= 12:
        conf += 0.15
    if len(reasoning) >= 20:
        conf += 0.05
    if intent and intent != "general":
        conf += 0.08
    if channel in ("live_api", "catalog", "kb", "none"):
        conf += 0.05

    # Channel ↔ intent alignment
    if channel == "live_api" and intent in _API_INTENTS:
        conf += 0.1
    if channel == "catalog" and intent == "product":
        conf += 0.1
    if channel == "kb" and intent in ("general", "refund", "payment", "seller"):
        conf += 0.08
    if channel == "live_api" and intent == "general" and not route.get("needs_order_id"):
        conf -= 0.2
    if channel == "catalog" and intent in ("refund", "payment", "order_history"):
        conf -= 0.25

    comb = _combined(original_msg, msg_en)
    if _is_short_or_vague_message(comb) and not route.get("continue_previous_topic"):
        conf -= 0.22
    if route.get("continue_previous_topic") and _last_user_snippets(conversation_context):
        conf += 0.12
    if (route.get("conversation_scope") or "").strip().lower() == "welfog_support":
        conf += 0.05

    entities = extract_entities(original_msg, msg_en, ai_route=route, conversation_context=conversation_context)
    if intent in ("order", "refund", "payment") and route.get("needs_order_id") and not entities.get("order_id"):
        conf -= 0.15
    if intent == "pincode_check" and not entities.get("pincode") and not entities.get("extracted_pincode"):
        conf -= 0.08

    return max(0.0, min(1.0, conf))


def top_customer_kb_file_match(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    conversation_context: str = "",
    min_score: float = 0.14,
) -> tuple[str, float]:
    """
    Best-matching admin KB file by embedding similarity — any language / phrasing.
    Returns (file_key, score); empty key when no confident match.
    """
    try:
        from services.kb_service import get_customer_kb_keys, rank_customer_kb_files_by_embedding
    except ImportError:
        return "", 0.0

    # Plain semantic query — must NOT call build_kb_retrieval_query (it calls infer_kb_query_category).
    route = ai_route or {}
    meaning = (route.get("user_meaning") or "").strip()
    parts = [p for p in (meaning, (msg_en or "").strip(), (original_msg or "").strip()) if p]
    query = " — ".join(dict.fromkeys(parts))
    if conversation_context:
        try:
            expanded = expand_query_with_conversation(
                original_msg, msg_en, conversation_context, ai_route=route
            )
            if (expanded or "").strip():
                query = expanded.strip()
        except Exception:
            pass
    if not query.strip():
        query = _combined(original_msg, msg_en)
    if not query.strip():
        return "", 0.0

    keys = [
        k
        for k in get_customer_kb_keys()
        if k and not str(k).startswith("welfog_api")
    ]
    if not keys:
        return "", 0.0
    ranked = rank_customer_kb_files_by_embedding(
        query, keys=keys, min_score=min_score, top_n=1
    )
    if not ranked:
        return "", 0.0
    return ranked[0][0], float(ranked[0][1])


def infer_kb_query_category(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    conversation_context: str = "",
) -> str:
    """
    KB retrieval category — brain intent first, else semantic KB file match.
    No per-language keyword lists; embeddings handle Hindi, English, long prompts, etc.
    """
    route = ai_route or {}
    intent = (route.get("intent") or "general").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()

    if intent in ("refund", "payment", "seller", "shipping", "privacy", "terms"):
        return intent
    if intent == "order" and channel == "kb":
        return "order"
    if intent == "order_history":
        return "order_history"

    top_key, top_score = top_customer_kb_file_match(
        original_msg,
        msg_en,
        ai_route=route,
        conversation_context=conversation_context,
    )
    if top_key and top_score >= 0.18:
        stem = top_key.lower().replace("welfog_api_", "")
        if stem in ("refund", "payment", "seller", "shipping", "privacy", "terms"):
            return stem

    if intent in _INTENT_KB_HINTS:
        return intent
    return "general"


def scoped_kb_keys_for_retrieval(
    category: str,
    *,
    ai_route: dict | None = None,
    user_meaning: str = "",
) -> list[str]:
    """Customer KB file keys allowed for this query category."""
    try:
        from services.kb_service import get_customer_kb_keys

        customer = get_customer_kb_keys()
    except ImportError:
        return []

    route = ai_route or {}
    raw_keys = list(route.get("kb_keys") or [])
    cat = (category or "general").strip().lower()
    filtered = filter_kb_keys_for_intent(raw_keys, cat, user_meaning=user_meaning or route.get("user_meaning") or "")
    if filtered:
        scoped = [k for k in filtered if k in customer]
        if scoped:
            return scoped

    if cat in ("", "general"):
        return list(customer)
    allow = _INTENT_KB_HINTS.get(cat) or frozenset()
    scoped = [k for k in customer if k in allow or any(h in k.lower() for h in allow)]
    return scoped if scoped else list(customer)


def filter_kb_keys_for_intent(
    suggested_keys: list[str] | None,
    intent: str,
    *,
    user_meaning: str = "",
) -> list[str]:
    """Drop KB files that do not match detected intent before retrieval/answer."""
    keys = [k for k in (suggested_keys or []) if k]
    if not keys:
        return []
    intent_l = (intent or "general").strip().lower()
    allow = _INTENT_KB_HINTS.get(intent_l) or _INTENT_KB_HINTS.get("general")
    if not allow:
        return keys

    meaning_l = (user_meaning or "").lower()
    filtered: list[str] = []
    for k in keys:
        stem = k.lower().replace("welfog_api_", "").replace("welfog-api-", "")
        if k in allow:
            filtered.append(k)
            continue
        if any(h in stem for h in allow):
            filtered.append(k)
            continue
        # Topic words in meaning → keep matching file
        if "refund" in meaning_l and "refund" in stem:
            filtered.append(k)
        elif "payment" in meaning_l and "payment" in stem:
            filtered.append(k)
        elif "privacy" in meaning_l and "privacy" in stem:
            filtered.append(k)
        elif "seller" in meaning_l and "seller" in stem:
            filtered.append(k)

    if not filtered and keys:
        # Prefer faqs/support over unrelated API routing master
        safe = [k for k in keys if k in ("faqs", "support", "company") or k in allow]
        return safe[:3] if safe else keys[:2]
    return list(dict.fromkeys(filtered))


def build_query_understanding(
    original_msg: str,
    msg_en: str = "",
    *,
    ai_route: dict | None = None,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> QueryUnderstanding:
    from services.translation_service import resolve_customer_reply_lang

    route = dict(ai_route or {})
    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    intent = (route.get("intent") or "general").strip().lower()
    meaning = (route.get("user_meaning") or "").strip()
    entities = extract_entities(
        original_msg, msg_en, ai_route=route, conversation_context=conversation_context
    )
    conf = score_routing_confidence(
        route, original_msg, msg_en, conversation_context=conversation_context
    )
    ambiguous = _is_short_or_vague_message(original_msg or msg_en) and not meaning
    raw_keys = list(route.get("kb_keys") or [])
    filter_intent = intent
    if intent == "general":
        filter_intent = infer_kb_query_category(
            original_msg,
            msg_en,
            ai_route=route,
            conversation_context=conversation_context,
        )
    filtered = filter_kb_keys_for_intent(raw_keys, filter_intent, user_meaning=meaning)
    if filter_intent == "payment" and "payment" in filtered:
        filtered = ["payment"] + [k for k in filtered if k != "payment"]
    elif filter_intent == "refund" and "refund" in filtered:
        filtered = ["refund"]
    if intent in ("product", "order_history", "wishlist", "deals", "categories", "category_feed", "pincode_check"):
        filtered = [k for k in filtered if not k.startswith("welfog_api_routing")]

    return QueryUnderstanding(
        reply_lang=rl,
        user_meaning=meaning,
        intent=intent,
        required_action=_required_action_from_route(route),
        entities=entities,
        confidence=conf,
        is_ambiguous=ambiguous,
        context_expanded_query=expand_query_with_conversation(
            original_msg, msg_en, conversation_context, ai_route=route
        ),
        filtered_kb_keys=filtered,
        continue_previous_topic=bool(route.get("continue_previous_topic")),
    )


def apply_query_understanding(
    route_data: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    reply_lang: str = "en",
) -> dict:
    """Attach understanding fields to ai_route; filter kb_keys in-place."""
    if not route_data or not isinstance(route_data, dict):
        return route_data or {}
    u = build_query_understanding(
        original_msg,
        msg_en,
        ai_route=route_data,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
    )
    out = dict(route_data)
    if u.filtered_kb_keys:
        out["kb_keys"] = u.filtered_kb_keys
    out["_understanding"] = {
        "confidence": round(u.confidence, 3),
        "required_action": u.required_action,
        "entities": u.entities,
        "is_ambiguous": u.is_ambiguous,
        "reply_lang": u.reply_lang,
        "context_expanded_query": u.context_expanded_query[:500],
    }
    chat_log(
        f"understanding intent={u.intent} action={u.required_action} "
        f"conf={u.confidence:.2f} kb={u.filtered_kb_keys[:4]}"
    )
    log_reasoning(
        f"Query understanding: intent={u.intent} conf={u.confidence:.2f} "
        f"action={u.required_action} meaning={(u.user_meaning or '')[:80]}"
    )
    return out


def should_request_clarification(
    understanding: QueryUnderstanding,
    original_msg: str,
    msg_en: str = "",
    *,
    route_decision: Any = None,
) -> bool:
    """True when we should ask user to clarify instead of guessing."""
    if understanding.confidence >= _CLARIFY_MIN_CONF:
        return False
    intent = understanding.intent
    if intent in ("out_of_domain",):
        return False
    if understanding.required_action in ("conversational", "decline"):
        return False
    if understanding.required_action in ("live_api", "catalog") and understanding.confidence >= 0.32:
        return False
    handler = ""
    if route_decision is not None:
        handler = (getattr(route_decision, "handler", None) or "").strip()
    if handler in (
        "pincode_delivery_api",
        "wishlist_api",
        "order_ai_flow",
        "product_ai_flow",
        "warm_feedback",
        "assistant_intro",
        "off_topic",
        "other_company_decline",
    ):
        return False
    if understanding.entities.get("order_id") or understanding.entities.get("pincode"):
        return False
    try:
        from services.kb_service import resolve_best_faq_chunk_for_question

        faq = resolve_best_faq_chunk_for_question(original_msg, msg_en)
        if faq and float(faq.get("score") or 0) >= 0.40:
            return False
    except ImportError:
        pass
    if understanding.is_ambiguous or understanding.continue_previous_topic:
        return True
    if intent in ("order", "refund", "payment", "product") and understanding.confidence < 0.38:
        return True
    return understanding.confidence < 0.32


def build_clarification_reply(
    original_msg: str,
    understanding: QueryUnderstanding,
    *,
    reply_lang: str = "",
) -> str:
    """Short clarification in customer's language — no KB/API guess."""
    from services.translation_service import customer_facing_template

    return customer_facing_template(
        "clarify_request_scope",
        original_msg,
        reply_lang or understanding.reply_lang,
        fallback_en=(
            "I want to help, but I'm not fully sure what you need on this message. "
            "Please reply in one short line: product search, order/refund/tracking "
            "(with Order ID if you have it), delivery PIN check, or a policy question."
        ),
    )


def maybe_clarification_reply(
    ai_route: dict | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    route_decision: Any = None,
) -> Optional[str]:
    """Returns HTML reply when clarification is needed; else None."""
    if not ai_route:
        return None
    u = build_query_understanding(
        original_msg,
        msg_en,
        ai_route=ai_route,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
    )
    if not should_request_clarification(u, original_msg, msg_en, route_decision=route_decision):
        return None
    log_reasoning(f"Low confidence ({u.confidence:.2f}) — asking clarification (no guess).")
    chat_log(f"clarification asked conf={u.confidence:.2f} intent={u.intent}")
    return build_clarification_reply(original_msg, u, reply_lang=reply_lang)


def enhance_retrieval_query(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    *,
    ai_route: dict | None = None,
) -> str:
    """Use context-expanded query for KB embedding search when follow-up is thin."""
    from utils.helpers import build_retrieval_query

    expanded = expand_query_with_conversation(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if expanded != _combined(original_msg, msg_en):
        return build_retrieval_query(msg_en, conversation_context, original_msg) + " " + expanded
    return build_retrieval_query(msg_en, conversation_context, original_msg)
