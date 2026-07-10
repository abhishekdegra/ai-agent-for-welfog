"""
AI-first routing: every customer message is understood by Groq BEFORE picking API / KB / cache.

Deterministic shortcuts ONLY for unambiguous tokens (catalog pro_id in message) — not phrase lists.
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

from services.answer_router import AnswerRouteDecision
from services.ai_service import ai_brain_route
from services.message_understanding import apply_ai_route_corrections
from utils.reasoning_log import log_reasoning

_MIXED_GREETING_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:hi+|hey+|hello+|helo+|namaste+|namaskar+|oye+|sun+o?|bol+o?|"
    r"darling|dear|bhai|bro|yaar|dost|ji|please|ram\s*ram|radhe\s*radhe|vanakkam"
    r")\s*)+",
    re.IGNORECASE,
)


def _transactional_beats_greeting(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Mixed turns: order_history / product search / wishlist MUST defeat greeting filler.
    Structural checks only — no embeddings or extra LLM.
    """
    try:
        from services.conversation_followup import is_deals_request_message
        from utils.helpers import (
            _message_looks_like_shopping_query,
            _text_has_light_order_tracking_markers,
            _text_is_phone_product_accessory_context,
            message_asks_welfog_categories_list,
            message_has_live_pincode_check_intent,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
            turn_is_obvious_product_shopping_turn,
            _text_is_live_order_lookup_intent,
            _text_is_order_tracking_intent,
            _text_wants_order_history_list_in_chat,
        )

        comb = f"{original_msg or ''} {msg_en or ''}".strip()
        if (
            _message_looks_like_shopping_query(comb)
            or _text_is_phone_product_accessory_context(comb)
            or turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conversation_context
            )
        ):
            return True
        if is_deals_request_message(original_msg, msg_en):
            return True
        if message_asks_welfog_categories_list(comb):
            return True
        if (
            message_is_past_purchase_list_request(comb)
            or _text_wants_order_history_list_in_chat(comb, conversation_context)
            or message_is_wishlist_like_request(comb)
        ):
            return True
        if _text_has_light_order_tracking_markers(comb) or message_has_live_pincode_check_intent(
            original_msg, conversation_context, msg_en
        ):
            if (
                _text_is_order_tracking_intent(comb)
                or _text_is_live_order_lookup_intent(comb, conversation_context)
                or message_has_live_pincode_check_intent(
                    original_msg, conversation_context, msg_en
                )
            ):
                return True
    except ImportError:
        pass
    return False


def _strip_greeting_filler_from_mixed_turn(
    original_msg: str,
    msg_en: str = "",
) -> tuple[str, str, bool]:
    """
    'hi darling order history bta' → route on functional tail only (one semantic pass).
    """
    orig = (original_msg or "").strip()
    en = (msg_en or orig.lower()).strip().lower()
    if not orig:
        return orig, en, False
    if _text_is_obvious_off_topic(f"{orig} {en}".strip()):
        return orig, en, False
    if not _transactional_beats_greeting(orig, en, ""):
        return orig, en, False
    stripped_orig = _MIXED_GREETING_PREFIX_RE.sub("", orig).strip()
    stripped_en = (
        _MIXED_GREETING_PREFIX_RE.sub("", en).strip()
        if en != orig.lower()
        else stripped_orig.lower()
    )
    if not stripped_orig or stripped_orig == orig:
        return orig, en, False
    log_reasoning(
        f"Mixed turn — greeting stripped; routing functional tail: {stripped_orig[:72]!r}"
    )
    return stripped_orig, stripped_en or stripped_orig.lower(), True


_OBVIOUS_OOD_SNIPPETS = (
    "meri gf",
    "mera bf",
    "girlfriend banw",
    "boyfriend banw",
    "gf banw",
    "bf banw",
    "baarish",
    "barish",
    "mausam",
    "mosam",
    "weather today",
    "cricket score",
    "going to trip",
    "trip plan",
    "homework",
    "recipe for",
    "marry me",
    "date me",
    "politics",
)


def _text_is_obvious_off_topic(comb: str) -> bool:
    """Fast OOD — only explicit off-topic snippets (no shopping keyword lists)."""
    low = f" {(comb or '').lower()} "
    return any(s in low for s in _OBVIOUS_OOD_SNIPPETS)


def _try_obvious_out_of_domain_route(
    original_msg: str,
    msg_en: str = "",
    *,
    reply_lang: str = "en",
    conversation_context: str = "",
) -> Optional[dict]:
    """
    Obvious off-topic — intent=out_of_domain, data_channel=none, no vector/KB scan.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    try:
        from services.location_delivery_resolver import turn_continues_pincode_area_check
        from utils.helpers import _conversation_in_pincode_delivery_flow

        if (
            conversation_context
            and _conversation_in_pincode_delivery_flow(conversation_context)
            and turn_continues_pincode_area_check(
                comb, conversation_context, ai_route=None
            )
        ):
            return None
    except ImportError:
        pass
    obvious_ood = _text_is_obvious_off_topic(comb)
    if not obvious_ood and _transactional_beats_greeting(original_msg, msg_en, ""):
        return None
    if not obvious_ood:
        try:
            from services.conversation_scope import _has_definite_welfog_shopping_signal
            from utils.helpers import (
                message_is_knowledge_information_request,
                message_is_welfog_about_request,
                _text_mentions_welfog_brand,
            )

            if (
                message_is_knowledge_information_request(comb, "")
                or message_is_welfog_about_request(comb)
                or _has_definite_welfog_shopping_signal(comb)
                or _text_mentions_welfog_brand(comb)
            ):
                return None
            if not _text_is_obvious_off_topic(comb):
                return None
        except ImportError:
            if not _text_is_obvious_off_topic(comb):
                return None

    scope_reply = ""
    try:
        from services.translation_service import resolve_customer_reply_lang

        rl = resolve_customer_reply_lang(original_msg, reply_lang)
    except ImportError:
        rl = reply_lang or "en"

    log_reasoning(
        "Single-pass OOD — obvious off-topic; data_channel=none (no vector/KB)."
    )
    return {
        "user_meaning": f"Off-topic unrelated to Welfog ({comb[:120]})",
        "reasoning": "Obvious out-of-domain — AI scope reply on dispatch (no KB/vector).",
        "intent": "out_of_domain",
        "data_channel": "none",
        "conversation_scope": "out_of_domain",
        "is_welfog_related": False,
        "scope_reply": scope_reply,
        "needs_order_id": False,
        "run_catalog_search": False,
        "kb_keys": [],
        "_universal_brain_route": True,
        "_turn_promotions_done": True,
        "_zero_llm_fast": True,
        "_obvious_ood": True,
    }


def _kb_keys_for_route(
    route_data: dict,
    original_msg: str = "",
    msg_en: str = "",
    *,
    conv_for_llm: str = "",
) -> list[str]:
    """Admin-panel KB keys from brain JSON or embedding — no hardcoded topic file names."""
    keys = [k for k in (route_data or {}).get("kb_keys") or [] if str(k).strip()]
    if keys:
        return keys
    try:
        from services.kb_service import resolve_brain_kb_keys

        resolved = resolve_brain_kb_keys(
            route_data,
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
        )
        if resolved:
            return resolved
    except ImportError:
        pass
    return []


# Existing handlers only — maps route_handler → (source, default_intent). No new APIs.
_EXISTING_HANDLER_SOURCES: dict[str, tuple[str, str]] = {
    "pincode_delivery_api": ("api", "pincode_check"),
    "order_tracking_api": ("api", "order"),
    "order_details_api": ("api", "order"),
    "refund_status_api": ("api", "refund"),
    "order_ai_flow": ("ai_order", "order_history"),
    "order_id_ai_flow": ("ai_order_id", "order_id"),
    "product_ai_flow": ("ai_product", "product"),
    "catalog_pro_id": ("api", "product"),
    "wishlist_api": ("api", "wishlist"),
    "deals_api": ("api", "deals"),
    "categories_api": ("api", "categories"),
    "category_feed_api": ("api", "category_feed"),
    "ai_route_and_answer": ("kb_ai", "general"),
    "dynamic_kb": ("kb", "general"),
    "kb_grounded_ai": ("kb_ai", "general"),
    "knowledge_topic_kb": ("kb", "general"),
    "policy_structured_kb": ("kb", "refund"),
    "seller_kb": ("kb", "seller"),
    "welfog_about_kb": ("kb", "general"),
    "wishlist_howto_kb": ("kb", "general"),
    "order_history_howto_kb": ("kb", "general"),
    "order_tracking_howto_kb": ("kb", "general"),
    "order_id_help_kb": ("kb", "general"),
    "order_placement_kb": ("kb", "general"),
    "warm_feedback": ("kb", "general"),
    "warm_greeting": ("kb", "general"),
    "off_topic": ("reject", "out_of_domain"),
    "temporary_load": ("reject", "general"),
}

_LIVE_API_HANDLERS = frozenset(
    h for h, (src, _) in _EXISTING_HANDLER_SOURCES.items() if src == "api"
)


def _decision_from_existing_handler(
    route_data: dict,
    handler: str,
    reasoning: str,
    *,
    kb_keys: list | None = None,
    search_query: str = "",
) -> AnswerRouteDecision:
    """Map brain route_handler → existing tool with correct source (never default to kb for APIs)."""
    h = (handler or "").strip().lower()
    intent = (route_data.get("intent") or "general").strip().lower()
    src, default_intent = _EXISTING_HANDLER_SOURCES.get(h, ("kb", intent or "general"))
    resolved_intent = intent if intent not in ("", "general") else default_intent
    if h in _LIVE_API_HANDLERS and intent in ("refund", "payment", "order") and h != "pincode_delivery_api":
        resolved_intent = intent
    keys = list(kb_keys or route_data.get("kb_keys") or [])
    if src in ("kb", "kb_ai") and not keys:
        try:
            from services.kb_service import resolve_brain_kb_keys

            keys = resolve_brain_kb_keys(route_data, "", route_data.get("user_meaning") or "")
        except ImportError:
            keys = []
    sq = (search_query or route_data.get("search_query") or "").strip()
    return AnswerRouteDecision(
        source=src,
        intent=resolved_intent,
        handler=h,
        search_query=sq if h == "product_ai_flow" else "",
        kb_keys=keys if src in ("kb", "kb_ai") else None,
        is_welfog_related=bool(route_data.get("is_welfog_related", True)),
        reason=f"Brain handler {h} — {reasoning}",
    )


def _brain_route_is_fast_lockable(route_data: dict) -> bool:
    """True when universal brain already chose a concrete existing handler — skip re-routing."""
    if not route_data.get("_universal_brain_route"):
        return False
    rh = (route_data.get("route_handler") or "").strip().lower()
    olk = (route_data.get("order_lookup_kind") or "").strip().lower()
    ch = (route_data.get("data_channel") or "").strip().lower()
    intent = (route_data.get("intent") or "").strip().lower()
    if rh in _EXISTING_HANDLER_SOURCES:
        return True
    if olk in ("track", "tracking", "details", "invoice", "refund_status") and ch == "live_api":
        return True
    if intent in ("pincode_check", "deals", "categories", "wishlist", "order_history") and ch == "live_api":
        return True
    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(route_data):
            return True
    except ImportError:
        pass
    if route_data.get("_product_catalog_locked") or (intent == "product" and ch == "catalog"):
        return True
    return False


def _quick_decision_from_locked_brain_route(
    route_data: dict,
    original_msg: str = "",
    msg_en: str = "",
) -> Optional[AnswerRouteDecision]:
    """Locked brain JSON → handler decision without extra micro-classifiers."""
    if not isinstance(route_data, dict):
        return None
    from services.account_list_semantics import (
        KIND_PURCHASE_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_WISHLIST_IN_CHAT,
        _kind_from_meaning_blob,
        _norm_account_list_kind,
    )

    reasoning = (route_data.get("reasoning") or "")[:200]
    intent = (route_data.get("intent") or "general").strip().lower()
    rh = (route_data.get("route_handler") or "").strip().lower()
    kb_keys = _kb_keys_for_route(route_data, original_msg, msg_en)
    alk = _norm_account_list_kind(route_data.get("account_list_kind") or "")

    olk = (route_data.get("order_lookup_kind") or "").strip().lower()
    channel = (route_data.get("data_channel") or "").strip().lower()

    if rh == "refund_status_api" or olk == "refund_status":
        return AnswerRouteDecision(
            source="api",
            intent="refund",
            handler="refund_status_api",
            is_welfog_related=True,
            reason=f"Brain refund status — {reasoning}",
        )
    if rh == "order_tracking_api" or olk in ("track", "tracking"):
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            is_welfog_related=True,
            reason=f"Brain order tracking — {reasoning}",
        )
    if rh == "order_details_api" or olk in ("details", "invoice"):
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            is_welfog_related=True,
            reason=f"Brain order details/invoice — {reasoning}",
        )
    if rh == "pincode_delivery_api" or intent == "pincode_check":
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            is_welfog_related=True,
            reason=f"Brain pincode delivery — {reasoning}",
        )
    if rh == "deals_api" or intent == "deals":
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            is_welfog_related=True,
            reason=f"Brain deals — {reasoning}",
        )
    if rh in ("categories_api", "category_feed_api") or intent in ("categories", "category_feed"):
        h_cat = rh if rh in ("categories_api", "category_feed_api") else "categories_api"
        return AnswerRouteDecision(
            source="api",
            intent=intent if intent in ("categories", "category_feed") else "categories",
            handler=h_cat,
            is_welfog_related=True,
            reason=f"Brain categories — {reasoning}",
        )
    meaning_kind = _kind_from_meaning_blob(
        f" {(route_data.get('user_meaning') or '').lower()} "
        f" {(route_data.get('reasoning') or '').lower()} "
    )
    if meaning_kind == KIND_WISHLIST_IN_CHAT:
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Brain wishlist API (meaning) — {reasoning}",
        )
    if meaning_kind == KIND_WISHLIST_HOWTO:
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Brain wishlist how-to (meaning) — {reasoning}",
        )
    if alk == KIND_WISHLIST_IN_CHAT or (intent == "wishlist" and rh != "wishlist_howto_kb"):
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Brain wishlist API — {reasoning}",
        )
    if alk == KIND_WISHLIST_HOWTO:
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Brain wishlist how-to — {reasoning}",
        )
    if alk == KIND_PURCHASE_IN_CHAT or intent == "order_history":
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason=f"Brain order history — {reasoning}",
        )
    if alk == KIND_PURCHASE_HOWTO:
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Brain order history how-to — {reasoning}",
        )
    if route_data.get("_product_catalog_locked") or (
        intent == "product" and route_data.get("run_catalog_search")
    ):
        sq = (route_data.get("search_query") or "").strip()
        return AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason=f"Brain product catalog — {reasoning}",
        )
    scope = (route_data.get("conversation_scope") or "").strip().lower()
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        sr = (route_data.get("scope_reply") or "").strip()
        handler = "warm_feedback" if scope == "general_chitchat" else "off_topic"
        return AnswerRouteDecision(
            source="scope",
            intent="general" if scope == "general_chitchat" else "out_of_domain",
            handler=handler,
            is_welfog_related=scope != "out_of_domain",
            reason=f"Brain scope {scope} — {reasoning}",
        )
    if rh:
        return _decision_from_existing_handler(
            route_data, rh, reasoning, kb_keys=kb_keys
        )
    return None


def _wishlist_answer_route_decision(
    route_data: dict,
    comb_hist: str,
    conv_for_llm: str,
    reasoning: str,
) -> Optional[AnswerRouteDecision]:
    """
    Saved/liked products in chat — must win over purchase-history semantic goal
    when AI JSON or translated text drifts to order_history.
    """
    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        _wants_wishlist_list_in_chat,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
    )

    if message_denies_wishlist_wants_order_history(comb_hist):
        return None

    from services.account_list_semantics import (
        KIND_PURCHASE_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_WISHLIST_IN_CHAT,
        _norm_account_list_kind,
    )

    kind = _norm_account_list_kind(route_data.get("account_list_kind") or "")
    if kind == KIND_WISHLIST_HOWTO:
        log_reasoning("AI route → wishlist how-to (account_list_kind).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=_kb_keys_for_route(route_data, "", comb_hist, conv_for_llm=conv_for_llm),
            is_welfog_related=True,
            reason=f"Wishlist navigation — {reasoning}",
        )
    if kind == KIND_WISHLIST_IN_CHAT:
        log_reasoning("AI route → wishlist API (account_list_kind).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Wishlist API — {reasoning}",
        )
    if kind in (KIND_PURCHASE_IN_CHAT, KIND_PURCHASE_HOWTO):
        return None

    ai_i = (route_data.get("intent") or "").strip().lower()
    if ai_i == "order_history":
        return None
    meaning_blob = (
        f" {(route_data.get('user_meaning') or '').lower()} "
        f" {(route_data.get('reasoning') or '').lower()} "
    )
    if any(
        x in meaning_blob
        for x in (
            "order history",
            "purchase history",
            "past order",
            "orders placed",
            "order placed",
            "my orders",
            "bought",
            "purchase list",
        )
    ):
        return None
    ai_describes_wishlist = any(
        x in meaning_blob
        for x in (
            "wishlist",
            "wish list",
            "saved product",
            "saved products",
            "liked product",
            "liked products",
            "heart ",
            "favourite",
            "favorite",
        )
    ) and not any(
        x in meaning_blob
        for x in ("order history", "purchase history", "past order", "orders placed")
    )

    is_wishlist_turn = (
        ai_i == "wishlist"
        or message_is_wishlist_like_request(comb_hist)
        or _text_asks_wishlist(comb_hist)
        or (ai_describes_wishlist and ai_i != "order_history")
    )
    if not is_wishlist_turn:
        return None

    if _text_asks_how_to_view_wishlist(comb_hist, conv_for_llm):
        log_reasoning("AI route → wishlist how-to (saved/liked topic, not purchase list).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=_kb_keys_for_route(route_data, "", comb_hist, conv_for_llm=conv_for_llm),
            is_welfog_related=True,
            reason=f"Wishlist navigation — {reasoning}",
        )

    if (
        _text_asks_wishlist(comb_hist)
        or _wants_wishlist_list_in_chat(comb_hist)
        or message_is_wishlist_like_request(comb_hist)
        or ai_i == "wishlist"
    ):
        log_reasoning("AI route → wishlist list in chat (trust saved/liked topic over purchase list).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Wishlist API — {reasoning}",
        )
    return None


def _kb_channel_brain_decision(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    comb_hist: str,
    kb_keys: list,
    reasoning: str,
) -> Optional[AnswerRouteDecision]:
    """
    Trust Groq data_channel=kb before semantic-goal overrides.
    Prevents company/payment FAQ turns from being hijacked to refund policy templates.
    """
    channel = (route_data.get("data_channel") or "").strip().lower()
    intent = (route_data.get("intent") or "general").strip().lower()
    if channel != "kb" or intent not in ("general", "refund", "payment", "seller"):
        return None
    try:
        from services.product_catalog_resolver import turn_requests_product_catalog

        if turn_requests_product_catalog(
            original_msg, msg_en, "", ai_route=route_data, allow_llm=False
        ):
            return None
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if not ai_route_is_kb_read(route_data):
            return None
    except ImportError:
        return None

    rh = (route_data.get("route_handler") or "").strip()
    from utils.helpers import (
        message_asks_welfog_categories_list,
        message_is_welfog_about_request,
        should_use_warm_conversational_reply,
    )
    from services.conversation_followup import is_deals_request_message
    from services.semantic_answer_plan import build_semantic_answer_plan

    if is_deals_request_message(original_msg, msg_en):
        log_reasoning("AI route → today's deals API (KB channel blocked — deals menu).")
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            is_welfog_related=True,
            reason=f"Deals / offers — {reasoning}",
        )
    if message_asks_welfog_categories_list(comb_hist):
        log_reasoning("AI route → categories API (KB channel blocked — category list).")
        return AnswerRouteDecision(
            source="api",
            intent="categories",
            handler="categories_api",
            is_welfog_related=True,
            reason=f"Welfog categories — {reasoning}",
        )

    if message_is_welfog_about_request(comb_hist):
        log_reasoning("AI route → Welfog company/about KB (trust brain channel=kb).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="welfog_about_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Welfog company/platform — {reasoning}",
        )

    if intent == "general" and should_use_warm_conversational_reply(
        original_msg, msg_en, comb_hist, route_data
    ):
        log_reasoning("AI route: general talk/thanks — warm feedback (trust brain channel=kb).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="warm_feedback",
            is_welfog_related=True,
            reason=f"Conversational thanks/praise — {reasoning}",
        )

    plan = build_semantic_answer_plan(route_data, handler=rh or "")
    if plan.answer_strategy in ("kb_then_ai", "api_kb_ai"):
        rh_kb = rh or "ai_route_and_answer"
        src = "kb_ai"
    elif plan.answer_strategy == "kb_only":
        rh_kb = rh or "dynamic_kb"
        src = "kb"
    else:
        rh_kb = rh or "dynamic_kb"
        src = "kb"
    log_reasoning(
        f"AI route: trust brain channel=kb strategy={plan.answer_strategy} → {rh_kb}."
    )
    return AnswerRouteDecision(
        source=src,
        intent=intent,
        handler=rh_kb,
        kb_keys=kb_keys or list(route_data.get("kb_keys") or ["faqs"]),
        is_welfog_related=True,
        reason=f"AI KB answer ({plan.answer_strategy}) — {reasoning}",
    )


def _decision_from_ai_route(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    ctx: Optional[dict] = None,
) -> AnswerRouteDecision:
    """Map Groq routing JSON → handler. Trust AI intent; no keyword list overrides here."""
    intent = (route_data.get("intent") or "general").strip()
    is_welfog = bool(route_data.get("is_welfog_related", True))
    kb_keys = _kb_keys_for_route(
        route_data, original_msg, msg_en, conv_for_llm=conv_for_llm
    )
    search_query = (route_data.get("search_query") or "").strip()
    needs_order_id = bool(route_data.get("needs_order_id", False))
    reasoning = (route_data.get("reasoning") or "")[:200]

    from services.product_search_flow import message_eligible_for_product_ai_flow
    from services.ai_route_semantics import ai_route_allows_catalog_search
    from utils.helpers import extract_product_search_query

    comb_hist = f"{original_msg} {msg_en}".strip()

    # Trust locked universal brain handler before keyword semantic overrides.
    if route_data.get("_universal_brain_route"):
        locked_dec = _quick_decision_from_locked_brain_route(
            route_data, original_msg, msg_en
        )
        if locked_dec is not None:
            log_reasoning(
                f"AI route → locked brain tool={locked_dec.handler} "
                f"intent={locked_dec.intent} source={locked_dec.source}."
            )
            return locked_dec

    from services.semantic_intent import ai_route_requests_pincode_delivery
    from utils.helpers import message_has_live_pincode_check_intent

    if ai_route_requests_pincode_delivery(route_data):
        pin = (route_data.get("extracted_pincode") or "").strip()
        log_reasoning(f"AI route → pincode delivery API (semantic){f' PIN {pin}' if pin else ''}.")
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            is_welfog_related=True,
            reason=f"AI semantic delivery/serviceability check — {reasoning}",
        )

    from utils.helpers import should_use_warm_conversational_reply

    if should_use_warm_conversational_reply(
        original_msg, msg_en, comb_hist, route_data
    ) and not _transactional_beats_greeting(
        original_msg, msg_en, conv_for_llm
    ):
        if (
            intent in ("general", "out_of_domain")
            and not needs_order_id
            and not search_query
            and not message_has_live_pincode_check_intent(original_msg, comb_hist, msg_en)
            and intent not in ("pincode_check", "order", "order_history", "refund", "payment", "product")
        ):
            log_reasoning("AI+heuristic: conversational thanks/praise — warm reply (not KB).")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational talk — {reasoning}",
            )

    rh = (route_data.get("route_handler") or "").strip()
    channel = (route_data.get("data_channel") or "").strip().lower()

    strict_semantic_mode = (os.getenv("STRICT_AI_INTENT_ROUTING", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    # Legacy phrase-based structured procedures are disabled in strict semantic mode.
    if not rh and not strict_semantic_mode:
        from services.answer_router import resolve_structured_procedure_route

        structured = resolve_structured_procedure_route(
            original_msg, msg_en, conversation_context=comb_hist
        )
        if structured:
            return structured

    from utils.helpers import _text_wants_order_history_list_in_chat, message_needs_live_single_order_lookup
    from services.semantic_intent import (
        ai_route_requests_live_order_lookup,
        ai_route_requests_order_history_list,
        ai_route_requests_pincode_delivery,
        ai_route_requests_refund_status_lookup,
    )

    ai_intent = (route_data.get("intent") or "").strip().lower()
    channel = (route_data.get("data_channel") or "").strip().lower()

    from utils.helpers import (
        _text_is_order_tracking_intent,
        _text_is_undelivered_order_complaint,
        _user_rejects_order_history_wants_tracking,
    )

    from services.semantic_route_guard import infer_customer_semantic_goal

    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list, _text_requests_category_product_browse

    ai_intent_pre = (route_data.get("intent") or "").strip().lower()
    channel_pre = (route_data.get("data_channel") or "").strip().lower()

    if is_deals_request_message(original_msg, msg_en) or ai_intent_pre == "deals":
        log_reasoning("AI route → today's deals API (semantic deals/offers).")
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            is_welfog_related=True,
            reason=f"Deals / offers — {reasoning}",
        )

    from services.welfog_api import resolve_category_product_browse_route, ensure_expanded_categories_map_for_ctx

    if ctx:
        ensure_expanded_categories_map_for_ctx(ctx)
    cat_browse = resolve_category_product_browse_route(comb_hist, ctx=ctx)
    if cat_browse:
        cid, sq = cat_browse
        if ctx is not None:
            ctx.setdefault("data", {})["selected_category_id"] = cid
            ctx["awaiting"] = None
        log_reasoning(f"AI route → category-filtered products (category_id={cid}).")
        return AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason=f"Products in category {cid} — {reasoning}",
        )

    if (
        message_asks_welfog_categories_list(comb_hist)
        or ai_intent_pre == "category_feed"
        or (
            ai_intent_pre in ("categories", "general")
            and "categor" in comb_hist.lower()
            and channel_pre == "live_api"
            and not _text_requests_category_product_browse(comb_hist, ctx)
        )
    ):
        log_reasoning("AI route → categories API (browse/list/count).")
        return AnswerRouteDecision(
            source="api",
            intent="categories",
            handler="categories_api",
            is_welfog_related=True,
            reason=f"Welfog categories — {reasoning}",
        )

    wishlist_dec = _wishlist_answer_route_decision(route_data, comb_hist, conv_for_llm, reasoning)
    if wishlist_dec is not None:
        return wishlist_dec

    try:
        from services.conversation_scope import (
            SCOPE_CHITCHAT,
            SCOPE_OUT,
            _turn_requests_catalog_menu,
            scope_from_ai_route,
            turn_blocks_product_catalog,
        )

        if _turn_requests_catalog_menu(original_msg, msg_en, ai_route=route_data):
            if is_deals_request_message(original_msg, msg_en):
                log_reasoning("AI route → today's deals API (catalog menu scope guard).")
                return AnswerRouteDecision(
                    source="api",
                    intent="deals",
                    handler="deals_api",
                    is_welfog_related=True,
                    reason=f"Deals / offers — {reasoning}",
                )
            if message_asks_welfog_categories_list(comb_hist):
                log_reasoning("AI route → categories API (catalog menu scope guard).")
                return AnswerRouteDecision(
                    source="api",
                    intent="categories",
                    handler="categories_api",
                    is_welfog_related=True,
                    reason=f"Welfog categories — {reasoning}",
                )
        elif turn_blocks_product_catalog(
            original_msg, msg_en, conv_for_llm, ai_route=route_data
        ):
            scope_dec = scope_from_ai_route(route_data)
            if scope_dec and scope_dec.scope in (SCOPE_CHITCHAT, SCOPE_OUT, "harm_sensitive"):
                if scope_dec.scope == SCOPE_CHITCHAT:
                    log_reasoning("AI route → conversational scope (beats product/KB).")
                    return AnswerRouteDecision(
                        source="scope",
                        intent="general",
                        handler="warm_feedback",
                        is_welfog_related=True,
                        reason=f"Chitchat — {scope_dec.user_meaning[:80]}",
                    )
                log_reasoning("AI route → out_of_domain scope (beats product/KB).")
                return AnswerRouteDecision(
                    source="reject",
                    intent="out_of_domain",
                    handler="off_topic",
                    is_welfog_related=False,
                    reason=f"Out of domain — {scope_dec.user_meaning[:80]}",
                )
    except ImportError:
        pass

    try:
        from services.conversation_scope import turn_blocks_product_catalog
        from services.product_catalog_resolver import product_catalog_route_decision

        if not turn_blocks_product_catalog(
            original_msg, msg_en, conv_for_llm, ai_route=route_data
        ):
            pc_dec, route_data = product_catalog_route_decision(
                route_data, original_msg, msg_en, conv_for_llm, reasoning=reasoning
            )
            if pc_dec is not None:
                return pc_dec
    except ImportError:
        pass

    kb_brain = _kb_channel_brain_decision(
        route_data, original_msg, msg_en, comb_hist, kb_keys, reasoning
    )
    if kb_brain is not None:
        return kb_brain

    sem_goal = ""
    if not route_data.get("_universal_brain_route"):
        sem_goal = infer_customer_semantic_goal(
            original_msg, msg_en, conv_for_llm, ai_route=route_data
        )
    else:
        try:
            from services.chat_flow_telemetry import skip_step

            skip_step("infer_customer_semantic_goal", "universal brain locked")
        except ImportError:
            pass
    # When Groq already chose pincode_check, do not let keyword semantic goals override.
    if ai_route_requests_pincode_delivery(route_data):
        sem_goal = sem_goal if sem_goal == "pincode_delivery" else "pincode_delivery"
    if sem_goal == "refund_policy":
        log_reasoning("AI route → return/refund/wrong-item KB (semantic goal).")
        return AnswerRouteDecision(
            source="kb",
            intent="refund",
            handler="policy_structured_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Return/refund policy — {reasoning}",
        )
    if sem_goal == "refund_status":
        log_reasoning("AI route → personal refund status API (semantic goal).")
        return AnswerRouteDecision(
            source="api",
            intent="refund",
            handler="refund_status_api",
            is_welfog_related=True,
            reason=f"Personal refund/return status — {reasoning}",
        )
    if sem_goal == "track_single_order":
        log_reasoning("AI route → live order tracking (semantic goal, any language).")
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            is_welfog_related=True,
            reason=f"Semantic track order — {reasoning}",
        )
    if sem_goal in ("order_invoice", "order_details"):
        try:
            from services.location_delivery_resolver import pincode_delivery_route_is_locked

            if pincode_delivery_route_is_locked(
                route_data,
                original_msg,
                msg_en,
                comb_hist,
                allow_llm=True,
            ):
                log_reasoning(
                    "Semantic order_details skipped — delivery/serviceability (pincode API)."
                )
                sem_goal = "pincode_delivery"
        except ImportError:
            pass
    if sem_goal in ("order_invoice", "order_details"):
        try:
            from services.order_details_flow import (
                message_is_catalog_product_browse_not_order_details,
            )
            from utils.helpers import (
                _text_has_product_shopping_intent,
                _turn_is_catalog_product_request,
            )

            if (
                message_is_catalog_product_browse_not_order_details(comb_hist)
                or _text_has_product_shopping_intent(comb_hist)
                or _turn_is_catalog_product_request(comb_hist)
            ):
                log_reasoning(
                    "Semantic order_details skipped — product browse/catalog search."
                )
                sem_goal = ""
        except ImportError:
            pass
    if sem_goal in ("order_invoice", "order_details"):
        log_reasoning(f"AI route → order details/invoice API (semantic goal={sem_goal}).")
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            is_welfog_related=True,
            reason=f"Semantic {sem_goal} — {reasoning}",
        )
    if sem_goal == "pincode_delivery":
        log_reasoning("AI route → pincode delivery API (semantic goal, area serviceability).")
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            is_welfog_related=True,
            reason=f"Semantic pincode delivery — {reasoning}",
        )
    if sem_goal == "product_catalog":
        try:
            from services.product_catalog_resolver import (
                apply_product_catalog_to_route,
                log_product_catalog_routing,
                resolve_product_search_turn,
            )

            resolved = resolve_product_search_turn(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=route_data,
                allow_llm=True,
            )
            route_data = apply_product_catalog_to_route(
                route_data,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
            )
            sq = (
                resolved.search_query
                or (route_data.get("search_query") or "").strip()
                or extract_product_search_query(comb_hist, comb_hist, conv_for_llm, ai_route=route_data)
                or ""
            )
            log_product_catalog_routing(
                detected_intent="product_search",
                product_entities=resolved.entities or route_data.get("_product_entities"),
                selected_route="product_ai_flow",
                filters=resolved.entities or {},
                source=resolved.source or "semantic_goal",
            )
        except ImportError:
            sq = (route_data.get("search_query") or "").strip()
        log_reasoning(f"AI route → product catalog API (semantic goal, sq={sq!r}).")
        return AnswerRouteDecision(
            source="ai_product",
            intent="product",
            handler="product_ai_flow",
            search_query=sq,
            is_welfog_related=True,
            reason=f"Semantic product catalog — {reasoning}",
        )
    if sem_goal == "wishlist_howto":
        log_reasoning("AI route → wishlist how-to KB (semantic goal).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=_kb_keys_for_route(route_data, "", comb_hist, conv_for_llm=conv_for_llm),
            is_welfog_related=True,
            reason=f"Wishlist where/how in app — {reasoning}",
        )
    if sem_goal == "wishlist_list":
        log_reasoning("AI route → wishlist list API (semantic goal).")
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            is_welfog_related=True,
            reason=f"Semantic wishlist — {reasoning}",
        )

    if sem_goal == "order_history_list":
        from utils.helpers import (
            _text_has_product_shopping_intent,
            _turn_is_catalog_product_request,
            message_is_wishlist_like_request,
            _text_asks_wishlist,
        )

        if (
            ai_intent_pre == "wishlist"
            or message_is_wishlist_like_request(comb_hist)
            or _text_asks_wishlist(comb_hist)
        ):
            log_reasoning(
                "Semantic order_history_list skipped — saved/liked wishlist topic."
            )
        elif ai_intent_pre in ("product", "deals", "categories", "category_feed") or channel_pre == "catalog":
            log_reasoning(
                "Semantic order_history_list skipped — AI/catalog intent is product browse, not purchase list."
            )
        elif _text_has_product_shopping_intent(comb_hist) or _turn_is_catalog_product_request(comb_hist):
            log_reasoning(
                "Semantic order_history_list skipped — product shopping (show/buy item), not order list."
            )
        else:
            log_reasoning("AI route → order history list (semantic goal).")
            return AnswerRouteDecision(
                source="ai_order",
                intent="order_history",
                handler="order_ai_flow",
                is_welfog_related=True,
                reason=f"Semantic purchase list — {reasoning}",
            )

    try:
        from services.order_tracking_semantics import (
            ai_route_requests_order_tracking_lookup,
        )

        if ai_route_requests_order_tracking_lookup(
            route_data, original_msg, msg_en, comb_hist
        ):
            log_reasoning("AI route → order tracking API (disambiguated from refund/details).")
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_tracking_api",
                is_welfog_related=True,
                reason=f"Live order tracking — {reasoning}",
            )
    except ImportError:
        pass

    try:
        from services.order_details_flow import (
            ai_route_requests_order_details_lookup,
            message_wants_order_details_or_invoice,
        )

        if ai_route_requests_order_details_lookup(
            route_data, original_msg, msg_en, comb_hist
        ):
            od_goal = message_wants_order_details_or_invoice(
                original_msg, msg_en, comb_hist, ai_route=route_data
            ) or "order_details"
            log_reasoning(
                f"AI route → order details API (disambiguated, goal={od_goal})."
            )
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_details_api",
                is_welfog_related=True,
                reason=f"Order details/invoice — {reasoning}",
            )
    except ImportError:
        pass

    if ai_route_requests_refund_status_lookup(route_data, original_msg, msg_en, comb_hist):
        log_reasoning("AI route → refund status API (semantic, any language).")
        return AnswerRouteDecision(
            source="api",
            intent="refund",
            handler="refund_status_api",
            is_welfog_related=True,
            reason=f"Personal refund/return status — {reasoning}",
        )

    if ai_route_requests_live_order_lookup(route_data, original_msg, msg_en, comb_hist):
        live = ai_intent if ai_intent in ("refund", "payment", "order") else "order"
        handler = "order_tracking_api" if live == "order" else "ai_route_and_answer"
        log_reasoning(f"AI route → live {live} lookup (semantic, any language).")
        return AnswerRouteDecision(
            source="api",
            intent=live,
            handler=handler,
            is_welfog_related=True,
            reason=f"AI semantic live {live} — {reasoning}",
        )

    from utils.helpers import (
        message_is_wishlist_like_request,
        resolve_navigation_help_topic,
    )

    from utils.helpers import _text_is_order_id_help_request, _text_is_tracking_howto_request

    ai_intent_early = (route_data.get("intent") or "").strip().lower()
    if _text_is_order_id_help_request(comb_hist):
        log_reasoning("AI route → order_id help (where to find ID, not tracking/history).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_id_help_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Order ID location help — {reasoning}",
        )
    nav_early = resolve_navigation_help_topic(comb_hist, conv_for_llm, ai_route=route_data)
    if ai_intent_early == "wishlist" or message_is_wishlist_like_request(comb_hist):
        from utils.helpers import _text_asks_wishlist, _wants_wishlist_list_in_chat

        if _text_asks_wishlist(comb_hist) or _wants_wishlist_list_in_chat(comb_hist):
            log_reasoning("AI route → wishlist list in chat (trust AI + saved/liked topic).")
            return AnswerRouteDecision(
                source="api",
                intent="wishlist",
                handler="wishlist_api",
                is_welfog_related=True,
                reason=f"Wishlist API — {reasoning}",
            )
    if (
        nav_early == "order_history_howto"
        and ai_intent_early != "wishlist"
        and not message_is_wishlist_like_request(comb_hist)
        and not _text_wants_order_history_list_in_chat(comb_hist, conv_for_llm)
    ):
        log_reasoning("AI route → order_history how-to (app navigation, not list API).")
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"Order history where/how in app — {reasoning}",
        )

    if ai_route_requests_order_history_list(route_data) and nav_early != "order_history_howto":
        log_reasoning("AI route → order_history list (semantic purchase-history API).")
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason=f"AI semantic order list — {reasoning}",
        )

    if not message_needs_live_single_order_lookup(
        original_msg, msg_en, comb_hist, ai_route=route_data
    ) and _text_wants_order_history_list_in_chat(comb_hist, conv_for_llm):
        log_reasoning("AI route → order_history list in chat (purchase-history API).")
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason=f"Show purchase history in chat — {reasoning}",
        )

    if rh == "order_history_howto_kb" and _text_wants_order_history_list_in_chat(comb_hist):
        rh = ""
    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        message_clarifies_wishlist_not_order_history,
        message_mentions_wishlist_topic,
    )

    from utils.helpers import (
        _message_has_app_navigation_intent,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
        message_wants_order_history_app_navigation,
        resolve_navigation_help_topic,
    )

    nav_topic = resolve_navigation_help_topic(comb_hist, conv_for_llm, ai_route=route_data)
    if nav_topic == "order_history_howto" and rh in ("wishlist_howto_kb", ""):
        rh = "order_history_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_history_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: wishlist → order_history_howto (semantic navigation).")
    elif nav_topic == "wishlist_howto" and rh == "order_history_howto_kb":
        rh = "wishlist_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "wishlist_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto → wishlist_howto (semantic navigation).")

    if rh == "order_history_howto_kb" and _text_is_order_id_help_request(comb_hist):
        rh = "order_id_help_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_id_help_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto → order_id_help_kb.")
    elif rh == "order_history_howto_kb" and (
        _text_asks_how_to_view_wishlist(comb_hist, conv_for_llm)
        or message_clarifies_wishlist_not_order_history(comb_hist)
        or message_mentions_wishlist_topic(comb_hist)
        or (
            message_is_wishlist_like_request(comb_hist)
            and _message_has_app_navigation_intent(comb_hist)
        )
    ):
        rh = "wishlist_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "wishlist_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_history_howto -> wishlist_howto (wishlist topic).")
    from utils.helpers import (
        _message_overrides_placement_followup,
        _text_has_explicit_how_to_place_order,
        _text_has_order_placement_intent,
    )

    if rh == "order_placement_kb" and _message_overrides_placement_followup(comb_hist):
        rh = ""
        route_data = dict(route_data)
        route_data["continue_previous_topic"] = False
        route_data.pop("route_handler", None)
        if (route_data.get("intent") or "general") in ("general", "order"):
            route_data["intent"] = "refund"
        log_reasoning("AI route: return/refund overrides stale order_placement_kb — AI+KB answer.")
    elif rh == "order_placement_kb" and (
        nav_topic == "order_history_howto"
        or message_wants_order_history_app_navigation(comb_hist, conv_for_llm)
    ):
        rh = "order_history_howto_kb"
        route_data = dict(route_data)
        route_data["route_handler"] = "order_history_howto_kb"
        route_data["intent"] = "general"
        route_data["continue_previous_topic"] = False
        log_reasoning("AI route corrected: order_placement_kb → order_history_howto (view history).")
    elif rh == "order_placement_kb" and not (
        _text_has_explicit_how_to_place_order(comb_hist)
        or _text_has_order_placement_intent(comb_hist)
    ):
        rh = ""
    from utils.helpers import _text_has_product_shopping_intent, _turn_is_catalog_product_request

    if rh == "order_placement_kb" and (
        _text_has_product_shopping_intent(comb_hist) or _turn_is_catalog_product_request(comb_hist)
    ):
        rh = ""
        route_data = dict(route_data)
        route_data["intent"] = "product"
        route_data["data_channel"] = "catalog"
        route_data.pop("route_handler", None)
        log_reasoning("AI route: product browse overrides mistaken order_placement_kb.")
    if rh in ("dynamic_kb", "knowledge_topic_kb", "welfog_fees_kb", "short_video_rules_kb"):
        ai_intent_now = (route_data.get("intent") or intent or "").strip().lower()
        if should_use_warm_conversational_reply(
            original_msg, msg_en, comb_hist, route_data
        ) and not message_has_live_pincode_check_intent(
            original_msg, comb_hist, msg_en
        ) and ai_intent_now not in (
            "pincode_check",
            "order",
            "order_history",
            "refund",
            "payment",
            "product",
            "wishlist",
        ):
            log_reasoning(f"AI handler={rh} overridden — conversational talk, warm reply.")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational talk — {reasoning}",
            )
    if rh == "order_details_api":
        try:
            from services.order_details_flow import (
                message_is_catalog_product_browse_not_order_details,
            )
            from utils.helpers import (
                _text_has_product_shopping_intent,
                _turn_is_catalog_product_request,
            )

            if (
                message_is_catalog_product_browse_not_order_details(comb_hist)
                or _text_has_product_shopping_intent(comb_hist)
                or _turn_is_catalog_product_request(comb_hist)
            ):
                log_reasoning(
                    "Handler order_details_api skipped — product browse/catalog search."
                )
                rh = ""
                route_data = dict(route_data)
                route_data["intent"] = "product"
                route_data["data_channel"] = "catalog"
                route_data.pop("route_handler", None)
                route_data["order_lookup_kind"] = "none"
                route_data["needs_order_id"] = False
        except ImportError:
            pass
    if rh == "order_details_api":
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_details_api",
            kb_keys=list(route_data.get("kb_keys") or []),
            is_welfog_related=True,
            reason=f"Order details/invoice API — {reasoning}",
        )
    if rh:
        return _decision_from_existing_handler(
            route_data,
            rh,
            reasoning,
            kb_keys=list(route_data.get("kb_keys") or []),
        )

    from services.support_scope import message_mentions_other_company_support

    if not is_welfog or intent == "out_of_domain":
        if message_mentions_other_company_support(
            original_msg, msg_en, conversation_context=comb_hist, ai_route=route_data
        ):
            log_reasoning("AI route: external company — decline (no Welfog remap).")
            return AnswerRouteDecision(
                source="reject",
                intent="out_of_domain",
                handler="other_company_decline",
                is_welfog_related=False,
                reason=f"Not a Welfog request — {reasoning}",
            )
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="off_topic",
            is_welfog_related=False,
            reason=f"AI: off-topic — {reasoning}",
        )

    api_intents = {
        "order_id": ("ai_order_id", "order_id_ai_flow"),
        "order_history": ("ai_order", "order_ai_flow"),
        "wishlist": ("api", "wishlist_api"),
        "product": ("ai_product", "product_ai_flow"),
        "deals": ("api", "deals_api"),
        "categories": ("api", "categories_api"),
        "category_feed": ("api", "category_feed_api"),
        "pincode_check": ("api", "pincode_delivery_api"),
    }

    # Trust AI data_channel when it conflicts with a mis-labelled intent string.
    if channel == "live_api" and intent in api_intents:
        source, handler = api_intents[intent]
        log_reasoning(f"AI route: data_channel=live_api → {handler}.")
        return AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            search_query="",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI live API — {reasoning}",
        )

    if channel == "catalog" and intent == "product":
        sq = extract_product_search_query(original_msg, msg_en, search_query) or search_query
        if ai_route_allows_catalog_search(route_data) and message_eligible_for_product_ai_flow(
            comb_hist, msg_en, original_msg, ai_route=route_data
        ):
            log_reasoning("AI route: data_channel=catalog → product_ai_flow.")
            return AnswerRouteDecision(
                source="ai_product",
                intent="product",
                handler="product_ai_flow",
                search_query=sq,
                is_welfog_related=True,
                reason=f"AI catalog search — {reasoning}",
            )

    if channel == "kb" and intent in ("general", "refund", "payment", "seller"):
        try:
            from services.conversation_scope import turn_blocks_product_catalog
            from services.product_catalog_resolver import product_catalog_route_decision

            if not turn_blocks_product_catalog(
                original_msg, msg_en, conv_for_llm, ai_route=route_data
            ):
                pc_dec, route_data = product_catalog_route_decision(
                    route_data, original_msg, msg_en, conv_for_llm, reasoning=reasoning
                )
                if pc_dec is not None:
                    return pc_dec
        except ImportError:
            pass
        from utils.helpers import should_use_warm_conversational_reply
        from services.semantic_answer_plan import build_semantic_answer_plan

        if intent == "general" and should_use_warm_conversational_reply(
            original_msg, msg_en, comb_hist, route_data
        ):
            log_reasoning("AI route: general talk/thanks — warm feedback (skip dynamic_kb).")
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_feedback",
                is_welfog_related=True,
                reason=f"Conversational thanks/praise — {reasoning}",
            )
        plan = build_semantic_answer_plan(route_data, handler=rh or "")
        if plan.answer_strategy in ("kb_then_ai", "api_kb_ai"):
            rh_kb = rh or "ai_route_and_answer"
            src = "kb_ai"
        elif plan.answer_strategy == "kb_only":
            rh_kb = rh or "dynamic_kb"
            src = "kb"
        else:
            rh_kb = rh or "dynamic_kb"
            src = "kb"
        log_reasoning(f"AI route: data_channel=kb strategy={plan.answer_strategy} → {rh_kb}.")
        return AnswerRouteDecision(
            source=src,
            intent=intent,
            handler=rh_kb,
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI KB answer ({plan.answer_strategy}) — {reasoning}",
        )

    if intent in api_intents:
        if intent == "product":
            sq = extract_product_search_query(original_msg, msg_en, search_query) or search_query
            if not (
                ai_route_allows_catalog_search(route_data)
                and message_eligible_for_product_ai_flow(
                    comb_hist, msg_en, original_msg, ai_route=route_data
                )
            ):
                log_reasoning(
                    "AI route intent=product downgraded — no reliable catalog signal in latest turn."
                )
                return AnswerRouteDecision(
                    source="kb_ai",
                    intent="general",
                    handler="ai_route_and_answer",
                    kb_keys=kb_keys,
                    is_welfog_related=True,
                    reason=f"Ambiguous product intent; ask user to clarify exact need — {reasoning}",
                )
            search_query = sq
        resolved_intent = (route_data.get("intent") or intent).strip()
        if resolved_intent in api_intents and resolved_intent != intent:
            intent = resolved_intent
        source, handler = api_intents[intent]
        return AnswerRouteDecision(
            source=source,
            intent=intent,
            handler=handler,
            search_query=search_query if intent == "product" else "",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI route → {intent} — {reasoning}",
        )

    if intent == "seller":
        return AnswerRouteDecision(
            source="kb",
            intent="seller",
            handler="seller_kb",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI route → seller KB — {reasoning}",
        )

    if intent == "order":
        from utils.helpers import _text_has_order_placement_intent

        comb_route = f"{original_msg} {msg_en}".strip()
        if _text_has_order_placement_intent(comb_route):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="order_placement_kb",
                is_welfog_related=True,
                kb_keys=kb_keys,
                reason=f"How to place order (not tracking) — {reasoning}",
            )
        if needs_order_id:
            from services.ai_route_semantics import infer_semantic_goal_from_ai_route

            sem_goal = infer_semantic_goal_from_ai_route(route_data)
            if sem_goal == "order_details":
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI brain: order details — {reasoning}",
                )
            if sem_goal == "order_invoice":
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI brain: order invoice — {reasoning}",
                )
            if sem_goal == "track_single_order":
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_tracking_api",
                    is_welfog_related=True,
                    reason=f"AI brain: order tracking — {reasoning}",
                )
            if sem_goal == "refund_status":
                return AnswerRouteDecision(
                    source="api",
                    intent="refund",
                    handler="refund_status_api",
                    is_welfog_related=True,
                    reason=f"AI brain: refund status — {reasoning}",
                )
            olk_live = (route_data.get("order_lookup_kind") or "").strip().lower()
            if olk_live in ("details", "invoice"):
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI brain: order_lookup_kind={olk_live} — {reasoning}",
                )
            if olk_live in ("track", "tracking"):
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_tracking_api",
                    is_welfog_related=True,
                    reason=f"AI brain: order_lookup_kind=track — {reasoning}",
                )
            try:
                from services.semantic_intent import strict_ai_semantic_mode

                if strict_ai_semantic_mode():
                    return AnswerRouteDecision(
                        source="api",
                        intent="order",
                        handler="order_details_api",
                        is_welfog_related=True,
                        reason=f"AI order live — details default (brain olk unset) — {reasoning}",
                    )
            except ImportError:
                pass
            od_goal = (route_data.get("_semantic_goal") or "").strip()
            if od_goal not in ("order_invoice", "order_details"):
                from services.order_details_flow import infer_order_details_semantic_goal

                od_goal = infer_order_details_semantic_goal(
                    original_msg, msg_en, comb_hist, ai_route=route_data
                )
            if od_goal in ("order_invoice", "order_details"):
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI: order details/invoice — {reasoning}",
                )
            from services.ai_route_semantics import brain_route_to_live_goal

            live_goal = brain_route_to_live_goal(
                route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=comb_hist,
            )
            if live_goal in ("order_invoice", "order_details"):
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI: order {live_goal} — {reasoning}",
                )
            if live_goal == "track":
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_tracking_api",
                    is_welfog_related=True,
                    reason=f"AI: order tracking — {reasoning}",
                )
            if live_goal == "refund_status":
                return AnswerRouteDecision(
                    source="api",
                    intent="refund",
                    handler="refund_status_api",
                    is_welfog_related=True,
                    reason=f"AI: refund status — {reasoning}",
                )
            try:
                from services.semantic_intent import strict_ai_semantic_mode

                if strict_ai_semantic_mode():
                    return AnswerRouteDecision(
                        source="api",
                        intent="order",
                        handler="order_details_api",
                        is_welfog_related=True,
                        reason=f"AI order live — details default (goal unset) — {reasoning}",
                    )
            except ImportError:
                pass
            return AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_tracking_api",
                is_welfog_related=True,
                reason=f"AI: live order tracking — {reasoning}",
            )
        try:
            from services.order_tracking_semantics import message_user_wants_order_tracking
            from utils.helpers import _text_is_order_tracking_intent_leaf

            if message_user_wants_order_tracking(
                comb_route, comb_hist
            ) or _text_is_order_tracking_intent_leaf(comb_route):
                log_reasoning(
                    "Order intent without needs_order_id — live track API (not how-to KB)."
                )
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_tracking_api",
                    is_welfog_related=True,
                    reason=f"Live order tracking — {reasoning}",
                )
        except ImportError:
            pass
        if _text_is_tracking_howto_request(comb_route):
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="order_tracking_howto_kb",
                is_welfog_related=True,
                kb_keys=kb_keys,
                reason=f"AI: order help / how-to track — {reasoning}",
            )
        try:
            from services.semantic_intent import strict_ai_semantic_mode

            if strict_ai_semantic_mode():
                return AnswerRouteDecision(
                    source="api",
                    intent="order",
                    handler="order_details_api",
                    is_welfog_related=True,
                    reason=f"AI order live — ask order id (details default) — {reasoning}",
                )
        except ImportError:
            pass
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            is_welfog_related=True,
            reason=f"Live order tracking (ask order id) — {reasoning}",
        )

    if intent in ("refund", "payment", "seller"):
        return AnswerRouteDecision(
            source="ai",
            intent=intent,
            handler="ai_route_and_answer",
            kb_keys=kb_keys,
            is_welfog_related=True,
            reason=f"AI: {intent} — {reasoning}",
        )

    return AnswerRouteDecision(
        source="kb_ai",
        intent="general",
        handler="ai_route_and_answer",
        search_query=search_query,
        kb_keys=kb_keys,
        is_welfog_related=True,
        reason=f"AI: general / KB-grounded — {reasoning}",
    )


def _try_conversational_fast_path(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """AI scope preflight — any-language chitchat (no keyword gate)."""
    try:
        from services.chitchat_resolver import try_chitchat_ai_preflight

        preflight = try_chitchat_ai_preflight(
            original_msg, msg_en, conv_for_llm, reply_lang
        )
        if preflight:
            log_reasoning("Conversational AI preflight — scope LLM (any language).")
            return preflight
    except ImportError:
        pass
    return None


def _try_conversational_keyword_fallback(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """LLM unavailable only — deterministic chitchat fallback."""
    from utils.helpers import message_is_conversational_general_talk

    combined = f"{original_msg} {msg_en}".strip()
    if not combined or not message_is_conversational_general_talk(
        original_msg, msg_en, conv_for_llm
    ):
        return None
    log_reasoning("Conversational keyword fallback (LLM down).")
    route_data = {
        "user_meaning": combined[:200],
        "reasoning": "Chitchat — keyword fallback while LLM unavailable.",
        "intent": "general",
        "data_channel": "none",
        "meta_kind": "conversational",
        "conversation_scope": "general_chitchat",
        "is_welfog_related": True,
        "needs_order_id": False,
        "run_catalog_search": False,
        "_preflight_conversational": True,
    }
    return (
        AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="warm_feedback",
            is_welfog_related=True,
            reason="Conversational — LLM-down fallback",
        ),
        route_data,
    )


def _try_account_list_fast_path(
    original_msg: str,
    msg_en: str,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """Wishlist / order history in chat — skip routing LLM when intent is unambiguous."""
    from utils.helpers import (
        _text_asks_how_to_view_order_history,
        _text_asks_how_to_view_wishlist,
        _text_asks_order_history,
        _text_asks_wishlist,
        _wants_wishlist_list_in_chat,
        message_is_wishlist_like_request,
    )

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None

    try:
        from utils.helpers import _text_is_welfog_payment_info_question

        if _text_is_welfog_payment_info_question(combined):
            return None
    except ImportError:
        pass

    try:
        import re

        from services.semantic_intent import should_skip_order_history_list_for_turn

        if re.search(r"\b\d{4,20}\b", combined) or should_skip_order_history_list_for_turn(
            original_msg, msg_en, ""
        ):
            return None
    except ImportError:
        pass

    if (
        _text_asks_wishlist(combined)
        or _wants_wishlist_list_in_chat(combined)
        or message_is_wishlist_like_request(combined)
    ) and not _text_asks_how_to_view_wishlist(combined):
        log_reasoning("Account-list fast-path → wishlist API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show saved/liked products in chat.",
            "intent": "wishlist",
            "data_channel": "live_api",
            "account_list_kind": "wishlist_in_chat",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_api": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="wishlist",
                handler="wishlist_api",
                is_welfog_related=True,
                reason="Wishlist list in chat — fast-path",
            ),
            route_data,
        )

    if _text_asks_order_history(combined) and not _text_asks_how_to_view_order_history(
        combined
    ):
        log_reasoning("Account-list fast-path → order history API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show purchase history in chat.",
            "intent": "order_history",
            "data_channel": "live_api",
            "account_list_kind": "purchase_history_in_chat",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_api": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="order_history",
                handler="order_ai_flow",
                is_welfog_related=True,
                reason="Order history in chat — fast-path",
            ),
            route_data,
        )

    return None


def _try_catalog_menu_fast_path(
    original_msg: str,
    msg_en: str,
    ctx: Optional[dict] = None,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """Categories list / today's deals — skip brain routing LLM (fast, deterministic)."""
    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None

    if is_deals_request_message(original_msg, msg_en):
        log_reasoning("Catalog-menu fast-path → today's deals API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "Show today's deals / offers on Welfog.",
            "intent": "deals",
            "data_channel": "live_api",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_catalog_menu": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="deals",
                handler="deals_api",
                is_welfog_related=True,
                reason="Today's deals — fast-path",
            ),
            route_data,
        )

    if message_asks_welfog_categories_list(combined):
        try:
            from services.welfog_api import (
                ensure_expanded_categories_map_for_ctx,
                resolve_category_product_browse_route,
            )

            if ctx:
                ensure_expanded_categories_map_for_ctx(ctx)
            cat_browse = resolve_category_product_browse_route(combined, ctx=ctx)
            if cat_browse:
                cid, sq = cat_browse
                if ctx is not None:
                    ctx.setdefault("data", {})["selected_category_id"] = cid
                    ctx["awaiting"] = None
                log_reasoning(
                    f"Catalog-menu fast-path → category browse (category_id={cid})."
                )
                route_data = {
                    "user_meaning": combined[:200],
                    "reasoning": f"Products in category {cid}.",
                    "intent": "product",
                    "data_channel": "catalog",
                    "run_catalog_search": True,
                    "search_query": sq,
                    "needs_order_id": False,
                    "numeric_context": "none",
                    "_preflight_catalog_menu": True,
                }
                return (
                    AnswerRouteDecision(
                        source="ai_product",
                        intent="product",
                        handler="product_ai_flow",
                        search_query=sq,
                        is_welfog_related=True,
                        reason=f"Category browse {cid} — fast-path",
                    ),
                    route_data,
                )
        except ImportError:
            pass

        log_reasoning("Catalog-menu fast-path → categories API (skip routing LLM).")
        route_data = {
            "user_meaning": combined[:200],
            "reasoning": "List Welfog shopping categories.",
            "intent": "categories",
            "data_channel": "live_api",
            "needs_order_id": False,
            "run_catalog_search": False,
            "numeric_context": "none",
            "order_lookup_kind": "none",
            "_preflight_catalog_menu": True,
        }
        return (
            AnswerRouteDecision(
                source="api",
                intent="categories",
                handler="categories_api",
                is_welfog_related=True,
                reason="Welfog categories list — fast-path",
            ),
            route_data,
        )

    return None


_DEDICATED_POLICY_KB_STEMS = frozenset(
    {"shipping", "refund", "privacy", "terms", "payment", "seller"}
)


def _refine_kb_preflight_rank(
    combined: str,
    ranked: list[tuple[str, float]],
    customer: list[str],
    *,
    resolve_keys: list[str] | None = None,
) -> tuple[str, float, list[str]]:
    """
  Embedding rank + dedicated admin file preference (shipping.txt over faqs, company over support).
  No keyword routing — only file-level scores from vector search.
    """
    from services.kb_service import rank_customer_kb_files_by_embedding

    if not ranked:
        return "", 0.0, list(resolve_keys or [])[:4]
    score_by_key = {k: float(s) for k, s in ranked}
    top_key, top_score = ranked[0]

    if top_key == "faqs":
        for stem in _DEDICATED_POLICY_KB_STEMS:
            if stem not in customer:
                continue
            stem_score = score_by_key.get(stem, 0.0)
            if stem_score < 0.26:
                continue
            policy_only = rank_customer_kb_files_by_embedding(
                combined, keys=[stem], min_score=0.18, top_n=1
            )
            if not policy_only:
                continue
            pk, ps = policy_only[0]
            if ps >= 0.26 and ps >= top_score - 0.10:
                top_key, top_score = pk, ps
                break

    if top_key == "support":
        co = score_by_key.get("company", 0.0)
        if co >= 0.26 and co >= top_score - 0.08:
            top_key, top_score = "company", co

    keys = list(
        dict.fromkeys(
            [top_key]
            + [k for k, _ in ranked[:4]]
            + list(resolve_keys or [])
        )
    )[:4]
    if top_key in _DEDICATED_POLICY_KB_STEMS:
        keys = [top_key] + [k for k in keys if k not in ("faqs", top_key)][:3]
    elif top_key == "company":
        keys = ["company"] + [k for k in keys if k not in ("support", "company")][:3]
    return top_key, top_score, keys


def _try_policy_kb_preflight(
    original_msg: str,
    msg_en: str,
    reply_lang: str,
) -> Optional[tuple[AnswerRouteDecision, dict]]:
    """
    Informational KB turns skip the routing LLM when embeddings match admin knowledge.
    Any .txt added via admin panel is ranked by vector similarity — no topic keyword map.
    """
    import re

    from services.kb_service import (
        get_customer_kb_keys,
        rank_customer_kb_files_by_embedding,
        resolve_kb_keys_for_question,
    )
    from utils.helpers import (
        _is_plausible_order_id,
        _text_has_concrete_welfog_support_question,
        extract_product_id,
        message_is_conversational_general_talk,
        message_is_welfog_about_request,
        should_send_warm_greeting_reply,
    )

    combined = f"{original_msg} {msg_en}".strip()
    if not combined:
        return None
    try:
        from utils.helpers import _normalize_welfog_typos

        combined = _normalize_welfog_typos(combined)
    except ImportError:
        pass
    comb_low = combined.lower()

    if should_send_warm_greeting_reply(original_msg, msg_en):
        return None
    if message_is_conversational_general_talk(original_msg, msg_en):
        if not (
            _text_has_concrete_welfog_support_question(combined)
            or message_is_welfog_about_request(combined)
        ):
            return None

    try:
        from services.conversation_followup import is_deals_request_message
        from utils.helpers import message_asks_welfog_categories_list

        if is_deals_request_message(original_msg, msg_en):
            return None
        if message_asks_welfog_categories_list(combined):
            return None
    except ImportError:
        pass

    # Structural only: catalog pro_id, order id, PIN → full router / live API
    if extract_product_id(comb_low):
        return None
    if _is_plausible_order_id(comb_low) or re.search(r"\b\d{6,}\b", comb_low):
        return None
    if re.search(r"\b[1-9]\d{5}\b", combined):
        return None

    try:
        from services.ai_first_router import _try_policy_kb_preflight
        from utils.helpers import message_asks_welfog_social_media

        combined_pf = f"{original_msg} {msg_en}".strip()
        if message_asks_welfog_social_media(combined_pf):
            return None
    except ImportError:
        pass

    customer = get_customer_kb_keys()
    if not customer:
        return None

    keys = resolve_kb_keys_for_question(original_msg, msg_en, max_files=4)
    keys = [k for k in (keys or []) if k in customer]
    if not keys:
        return None

    ranked = rank_customer_kb_files_by_embedding(
        combined, keys=keys, min_score=0.14, top_n=4
    )
    if not ranked:
        ranked = rank_customer_kb_files_by_embedding(
            combined, keys=customer, min_score=0.14, top_n=4
        )
    if not ranked:
        return None

    top_key, top_score, keys = _refine_kb_preflight_rank(
        combined, ranked, customer, resolve_keys=keys
    )
    _MIN_PREFLIGHT = 0.26
    if top_score < _MIN_PREFLIGHT:
        return None

    stem = top_key.lower().replace("welfog_api_", "").replace("-", "_")
    intent = (
        stem
        if stem in ("payment", "refund", "shipping", "privacy", "terms", "seller")
        else "general"
    )

    log_reasoning(
        f"KB preflight: embedding {top_key} (score={top_score:.2f}) "
        f"keys={','.join(keys)} — skip routing LLM."
    )
    route_data = {
        "user_meaning": combined[:200],
        "reasoning": f"Admin KB embedding preflight ({top_key}, score={top_score:.2f}).",
        "intent": intent,
        "data_channel": "kb",
        "kb_keys": keys,
        "answer_strategy": "kb_then_ai",
        "conversation_scope": "welfog_support",
        "is_welfog_related": True,
        "meta_kind": "none",
        "needs_order_id": False,
        "run_catalog_search": False,
        "numeric_context": "none",
        "order_lookup_kind": "none",
        "_preflight_kb": True,
        "_preflight_top_score": top_score,
        "_preflight_top_file": top_key,
    }
    return (
        AnswerRouteDecision(
            source="kb_ai",
            intent=intent,
            handler="ai_route_and_answer",
            kb_keys=keys,
            is_welfog_related=True,
            reason=f"Embedding KB preflight ({top_key})",
        ),
        route_data,
    )


def _brain_route_is_usable(route_data: dict | None) -> bool:
    if not isinstance(route_data, dict):
        return False
    if route_data.get("llm_unavailable"):
        return False
    return bool(
        (route_data.get("user_meaning") or "").strip()
        or (route_data.get("intent") or "").strip()
        or (route_data.get("conversation_scope") or "").strip()
    )


def _brain_reasoning_indicates_ood(reasoning: str) -> bool:
    """Trust brain JSON reasoning field only — not customer text."""
    r = (reasoning or "").strip().lower()
    if not r:
        return False
    return any(
        x in r
        for x in (
            "out_of_domain",
            "off-topic",
            "off topic",
            "not welfog",
            "not related to welfog",
            "unrelated to welfog",
            "not a welfog",
            "other company",
            "other app",
            "competitor",
        )
    )


def _brain_user_meaning_indicates_welfog_topic(um: str) -> bool:
    """Brain English user_meaning — Welfog commerce/support topic (not customer keywords)."""
    u = (um or "").strip().lower()
    if not u:
        return False
    if "welfog" in u:
        return True
    return any(
        m in u
        for m in (
            "order",
            "delivery",
            "ship",
            "refund",
            "return",
            "product",
            "catalog",
            "wishlist",
            "pincode",
            "pin code",
            "seller",
            "payment",
            "invoice",
            "track",
            "policy",
            "faq",
            "customer care",
            "support number",
            "checkout",
            "coupon",
            "category",
            "browse",
            "shop",
        )
    )


def _brain_user_meaning_indicates_chitchat(um: str) -> bool:
    """Brain English user_meaning — casual opener / thanks / bye (not customer keywords)."""
    u = (um or "").strip().lower()
    if not u:
        return False
    return any(
        m in u
        for m in (
            "greet",
            "greeting",
            "hello",
            "hi ",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "how are you",
            "small talk",
            "chitchat",
            "casual",
            "wellbeing",
            "what are you doing",
            "free or busy",
            "are you free",
            "are you busy",
            "who are you",
            "what can you do",
            "introduce yourself",
        )
    )


def _brain_um_names_external_marketplace(um: str, reasoning: str = "") -> bool:
    """
    Brain user_meaning / reasoning are English written by the routing LLM — if either
    names another marketplace, treat as OOD (not customer-text keyword matching).
    """
    blob = f"{um or ''} {reasoning or ''}".strip().lower()
    if not blob or "welfog" in blob:
        return False
    return bool(
        re.search(
            r"\b(amazon|flipkart|myntra|zepto|swiggy|instamart|meesho|snapdeal|ajio|blinkit)\b",
            blob,
        )
    )


def _reconcile_off_topic_brain_misroute(
    route: dict,
    original_msg: str,
    *,
    msg_en: str = "",
) -> dict:
    """
    Fix brain misroutes using ai_brain_route JSON only — never customer-text keyword lists.
    Trusts: is_welfog_related, conversation_scope, intent, data_channel, kb_keys, user_meaning.
    """
    out = dict(route or {})
    try:
        from services.chat_flow_telemetry import is_authoritative_kb_route_locked

        if is_authoritative_kb_route_locked():
            log_reasoning(
                "Brain OOD reconcile — skip; authoritative KB route already locked."
            )
            return out
    except ImportError:
        pass
    if out.get("_zero_llm_fast") or out.get("_preflight_api") or out.get(
        "_preflight_catalog_menu"
    ) or out.get("_pincode_delivery_fast"):
        return out
    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    scope = (out.get("conversation_scope") or "").strip().lower()
    related = out.get("is_welfog_related")
    kb_keys = [k for k in (out.get("kb_keys") or []) if str(k).strip()]
    reasoning = (out.get("reasoning") or "").strip().lower()

    if intent == "kb":
        out["intent"] = "general"
        out.setdefault("data_channel", "kb")
        intent = "general"

    en_blob = (msg_en or "").strip().lower()
    if en_blob and "welfog" not in en_blob:
        if re.search(
            r"\b(amazon|flipkart|myntra|zepto|swiggy|instamart|meesho|snapdeal|ajio|blinkit)\b",
            en_blob,
        ):
            out["intent"] = "out_of_domain"
            out["data_channel"] = "none"
            out["conversation_scope"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["kb_keys"] = []
            out["scope_reply"] = ""
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain OOD reconcile — translated English names external marketplace."
            )
            return out

    if _brain_reasoning_indicates_ood(reasoning):
        out["intent"] = "out_of_domain"
        out["data_channel"] = "none"
        out["conversation_scope"] = "out_of_domain"
        out["is_welfog_related"] = False
        out["kb_keys"] = []
        out["scope_reply"] = ""
        out["needs_order_id"] = False
        out["run_catalog_search"] = False
        out["_turn_promotions_done"] = True
        log_reasoning("Brain OOD reconcile — reasoning field indicates off-topic.")
        return out

    if intent == "out_of_domain" or scope == "out_of_domain":
        out["is_welfog_related"] = False
        out["data_channel"] = "none"
        out["conversation_scope"] = "out_of_domain"
        out["_turn_promotions_done"] = True
        return out

    if related is False:
        out["intent"] = "out_of_domain"
        out["data_channel"] = "none"
        out["conversation_scope"] = "out_of_domain"
        out["scope_reply"] = ""
        out["kb_keys"] = []
        out["_turn_promotions_done"] = True
        log_reasoning("Brain OOD reconcile — is_welfog_related=false from ai_brain_route.")
        return out

    um = (out.get("user_meaning") or "").strip()
    if um and _brain_um_names_external_marketplace(um, reasoning):
        out["intent"] = "out_of_domain"
        out["data_channel"] = "none"
        out["conversation_scope"] = "out_of_domain"
        out["is_welfog_related"] = False
        out["kb_keys"] = []
        out["scope_reply"] = ""
        out["_turn_promotions_done"] = True
        log_reasoning(
            "Brain OOD reconcile — user_meaning names external marketplace."
        )
        return out

    if um and not _brain_user_meaning_indicates_welfog_topic(um):
        if _brain_user_meaning_indicates_chitchat(um):
            out["intent"] = "general"
            out["conversation_scope"] = "general_chitchat"
            out["data_channel"] = "none"
            out["is_welfog_related"] = True
            out["kb_keys"] = []
            out["run_catalog_search"] = False
            out["scope_reply"] = out.get("scope_reply") or ""
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — chitchat from user_meaning (non-Welfog topic)."
            )
            return out
        if intent not in (
            "product",
            "order",
            "order_history",
            "wishlist",
            "pincode_check",
            "deals",
            "categories",
        ):
            try:
                from services.ai_route_semantics import _brain_route_has_shopping_entities
                from utils.helpers import turn_is_obvious_product_shopping_turn

                if not _brain_route_has_shopping_entities(
                    out, original_msg=original_msg, msg_en=msg_en
                ) and not turn_is_obvious_product_shopping_turn(
                    original_msg, msg_en, ""
                ):
                    try:
                        from services.kb_service import promote_route_from_semantic_kb_match

                        promoted = promote_route_from_semantic_kb_match(
                            out, original_msg, msg_en=msg_en
                        )
                        if promoted:
                            log_reasoning(
                                "Brain KB reconcile — semantic match overrides OOD user_meaning."
                            )
                            return promoted
                    except ImportError:
                        pass
                    try:
                        from services.chat_flow_telemetry import (
                            brain_route_authoritative_kb_lock,
                            lock_authoritative_kb_route_from_brain,
                        )

                        if brain_route_authoritative_kb_lock(out):
                            lock_authoritative_kb_route_from_brain(out)
                            log_reasoning(
                                "Brain OOD reconcile — skip demotion; brain locked KB route."
                            )
                            return out
                    except ImportError:
                        pass
                    out["intent"] = "out_of_domain"
                    out["data_channel"] = "none"
                    out["conversation_scope"] = "out_of_domain"
                    out["is_welfog_related"] = False
                    out["kb_keys"] = []
                    out["scope_reply"] = ""
                    out["run_catalog_search"] = False
                    out["_turn_promotions_done"] = True
                    log_reasoning(
                        "Brain OOD reconcile — user_meaning is not a Welfog topic."
                    )
                    return out
            except ImportError:
                pass

    if um and "welfog" not in um.lower():
        ood_reasoning = _brain_reasoning_indicates_ood(reasoning)
        platform_kb = set(kb_keys) & {"payment", "refund", "shipping", "seller", "terms", "privacy"}
        if channel == "kb" or intent in ("seller", "refund", "payment") or platform_kb:
            try:
                from services.kb_service import promote_route_from_semantic_kb_match

                promoted = promote_route_from_semantic_kb_match(
                    out, original_msg, msg_en=msg_en
                )
                if promoted:
                    log_reasoning(
                        "Brain KB reconcile — semantic match before non-Welfog user_meaning OOD."
                    )
                    return promoted
            except ImportError:
                pass
        if (
            ood_reasoning
            or (
                related is False
                and channel != "kb"
                and intent not in ("seller", "refund", "payment")
            )
        ):
            out["intent"] = "out_of_domain"
            out["data_channel"] = "none"
            out["conversation_scope"] = "out_of_domain"
            out["is_welfog_related"] = False
            out["kb_keys"] = []
            out["scope_reply"] = ""
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain OOD reconcile — user_meaning names non-Welfog topic."
            )
            return out

    if scope in ("general_chitchat", "harm_sensitive") and channel in (
        "kb",
        "live_api",
        "catalog",
    ):
        if scope == "general_chitchat" and channel == "kb":
            try:
                from services.kb_service import promote_route_from_semantic_kb_match

                promoted = promote_route_from_semantic_kb_match(
                    out, original_msg, msg_en=msg_en
                )
                if promoted:
                    log_reasoning(
                        "Brain scope reconcile — semantic KB over general_chitchat+kb."
                    )
                    return promoted
            except ImportError:
                pass
            try:
                from services.ai_route_semantics import (
                    brain_turn_indicates_welfog_kb,
                    reconcile_welfog_kb_from_brain_meaning,
                )

                if brain_turn_indicates_welfog_kb(out):
                    out = reconcile_welfog_kb_from_brain_meaning(out)
                    out["_turn_promotions_done"] = True
                    log_reasoning(
                        "Brain scope reconcile — Welfog KB promoted from user_meaning."
                    )
                    return out
            except ImportError:
                pass
            out["data_channel"] = "none"
            out["kb_keys"] = []
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — chitchat wins over contradictory kb channel."
            )
            return out
        if channel == "kb":
            out["conversation_scope"] = "welfog_support"
            out["is_welfog_related"] = True
            if not kb_keys:
                try:
                    from services.kb_service import resolve_brain_kb_keys

                    resolved = resolve_brain_kb_keys(
                        out, original_msg, msg_en=msg_en
                    )
                    if resolved:
                        out["kb_keys"] = resolved
                except ImportError:
                    pass
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — kb channel wins over general_chitchat."
            )
            return out
        if channel == "catalog":
            out["run_catalog_search"] = True
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — catalog channel wins over general_chitchat."
            )
            return out
        if channel == "live_api":
            out["conversation_scope"] = "welfog_support"
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — live_api channel wins over general_chitchat."
            )
            return out

    if intent == "general" and channel == "kb" and scope not in ("welfog_support",):
        try:
            from services.kb_service import promote_route_from_semantic_kb_match

            promoted = promote_route_from_semantic_kb_match(
                out, original_msg, msg_en=msg_en
            )
            if promoted:
                log_reasoning(
                    "Brain scope reconcile — semantic KB promotes welfog_support."
                )
                return promoted
        except ImportError:
            pass
        um = (out.get("user_meaning") or "").strip()
        generic_keys = not kb_keys or set(kb_keys) <= {"company", "faqs", "terms"}
        welfog_topic = bool(um and "welfog" in um.lower())
        if welfog_topic or (kb_keys and not generic_keys):
            out["conversation_scope"] = "welfog_support"
            if not kb_keys:
                try:
                    from services.kb_service import resolve_brain_kb_keys

                    resolved = resolve_brain_kb_keys(
                        out, original_msg, msg_en=msg_en
                    )
                    if resolved:
                        out["kb_keys"] = resolved
                except ImportError:
                    pass
            out["_turn_promotions_done"] = True
            return out
        comb = f"{original_msg or ''} {msg_en or ''}".strip()
        echoed = um and comb and um.lower() == comb.lower()
        if echoed or scope in ("general_chitchat", ""):
            out["conversation_scope"] = (
                "out_of_domain" if related is False else "general_chitchat"
            )
            if out["conversation_scope"] == "out_of_domain":
                out["intent"] = "out_of_domain"
                out["is_welfog_related"] = False
            out["data_channel"] = "none"
            out["kb_keys"] = []
            out["scope_reply"] = ""
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Brain scope reconcile — general+kb without resolvable KB match."
            )
    return out


def _reconcile_semantic_kb_route(
    route: dict,
    original_msg: str = "",
    *,
    msg_en: str = "",
) -> dict:
    """
    When admin KB embeddings strongly match the question, lock data_channel=kb
    even if brain JSON said general_chitchat or catalog (no customer keyword lists).
    Never overrides live API (pincode/delivery), catalog, or category browse.
    """
    from services.ai_route_semantics import _brain_route_has_shopping_entities
    from services.kb_service import (
        build_kb_retrieval_query,
        get_support_contact_kb_keys,
        retrieve_best_kb_chunk,
        resolve_brain_kb_keys,
    )

    out = dict(route or {})
    if out.get("_preflight_kb") or out.get("_zero_llm_fast") or out.get(
        "_preflight_catalog_menu"
    ):
        return out

    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    combined = f"{original_msg or ''} {msg_en or ''}".strip()

    if (
        intent == "pincode_check"
        or out.get("_pincode_delivery_fast")
        or out.get("_pincode_delivery_locked")
        or rh in ("pincode_delivery_api", "pincode_delivery_fast")
        or channel == "live_api"
        or channel == "catalog"
        or out.get("run_catalog_search")
        or out.get("category_only_browse")
        or (out.get("category_browse") or "").strip()
        or intent in ("deals", "categories", "order", "order_history", "wishlist")
    ):
        return out

    try:
        from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

        if ai_meaning_describes_delivery_serviceability(out):
            return out
    except ImportError:
        pass

    try:
        from services.location_delivery_resolver import (
            resolve_delivery_turn,
            turn_requests_delivery_serviceability,
            _DELIVERY_AI_CONF_SERVICEABILITY,
        )

        if turn_requests_delivery_serviceability(
            original_msg, msg_en, "", allow_llm=False
        ):
            return out
        understood = resolve_delivery_turn(
            combined,
            (out.get("user_meaning") or "").strip(),
            msg_en,
            ai_route=out,
            allow_llm=False,
        )
        if (
            understood.is_serviceability
            or understood.is_area_followup
            or understood.location_kind in ("city", "pincode", "ask_pin")
        ) and float(understood.confidence or 0) >= 0.35:
            return out
    except ImportError:
        pass

    try:
        from utils.helpers import _text_asks_customer_care_contact

        if _text_asks_customer_care_contact(combined) and not out.get("needs_order_id"):
            sc_keys = get_support_contact_kb_keys()
            if sc_keys:
                out["data_channel"] = "kb"
                out["conversation_scope"] = "welfog_support"
                out["intent"] = "general"
                out["kb_keys"] = sc_keys[:3]
                out["run_catalog_search"] = False
                out["search_query"] = ""
                out["is_welfog_related"] = True
                out.pop("_product_catalog_locked", None)
                log_reasoning("Brain KB reconcile — customer-care contact (support files).")
                return out
    except ImportError:
        pass

    intent_now = (out.get("intent") or "").strip().lower()
    scope_now = (out.get("conversation_scope") or "").strip().lower()
    # OOD / chitchat should never run KB embedding reconcile.
    if (
        intent_now == "out_of_domain"
        or scope_now in ("out_of_domain", "general_chitchat", "harm_sensitive")
        or out.get("is_welfog_related") is False
    ):
        return out

    q = build_kb_retrieval_query(original_msg, msg_en, "", ai_route=out)
    q = q or (out.get("user_meaning") or combined).strip()
    if not q:
        return out

    best = retrieve_best_kb_chunk(q, ai_route=out, min_score=0.22)
    chunk_score = float((best or {}).get("score") or 0)
    if chunk_score < 0.28:
        return out

    has_shopping = _brain_route_has_shopping_entities(
        out, original_msg=original_msg, msg_en=msg_en
    )
    if has_shopping and chunk_score < 0.72:
        return out

    keys = resolve_brain_kb_keys(out, original_msg, msg_en)
    if best and best.get("source"):
        keys = list(dict.fromkeys((keys or []) + [str(best["source"])]))[:4]

    out["data_channel"] = "kb"
    out["conversation_scope"] = "welfog_support"
    out["kb_keys"] = keys
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out.pop("category_only_browse", None)
    out.pop("category_browse", None)
    out.pop("_product_catalog_locked", None)
    if (out.get("intent") or "").strip().lower() in (
        "product",
        "product_search",
        "general_chitchat",
        "chitchat",
    ):
        out["intent"] = "general"
    out["is_welfog_related"] = True
    out["scope_reply"] = ""
    log_reasoning(
        f"Brain semantic KB reconcile — chunk={chunk_score:.2f} "
        f"file={(best or {}).get('source') or '?'}"
    )
    return out


def _reconcile_delivery_policy_from_brain_json(route: dict) -> dict:
    """
    pincode_check without PIN misclassified when brain meant admin KB (policy/timeline).
    Uses brain JSON + admin-KB embeddings only — never customer-text or English phrase lists.
    """
    from services.ai_route_semantics import ai_route_is_kb_read
    from services.kb_service import resolve_brain_kb_keys
    from services.query_understanding import top_customer_kb_file_match

    out = dict(route or {})
    intent = (out.get("intent") or "").strip().lower()
    if intent != "pincode_check":
        return out

    # Live delivery API lock — never downgrade to static FAQ KB.
    if out.get("_pincode_delivery_fast") or out.get("_pincode_delivery_locked"):
        return out
    rh = (out.get("route_handler") or "").strip().lower()
    if rh == "pincode_delivery_api":
        return out
    if (out.get("data_channel") or "").strip().lower() == "live_api":
        return out

    try:
        from services.location_delivery_resolver import (
            resolve_delivery_turn,
            _DELIVERY_AI_CONF_SERVICEABILITY,
        )

        um = (out.get("user_meaning") or "").strip()
        orig = (out.get("_routing_msg") or um).strip()
        understood = resolve_delivery_turn(
            orig or um,
            um,
            "",
            ai_route=out,
            allow_llm=False,
        )
        if (
            understood.is_serviceability
            and understood.confidence >= _DELIVERY_AI_CONF_SERVICEABILITY
        ):
            return out
        if understood.is_area_followup:
            return out
        if understood.location_kind in ("city", "pincode", "ask_pin"):
            return out
    except ImportError:
        pass

    if (out.get("extracted_pincode") or "").strip():
        return out
    nc = (out.get("numeric_context") or "").strip().lower()
    reuse = (out.get("reuse_user_value_from_chat") or "").strip().lower()
    if nc == "pincode" and reuse == "pincode":
        return out

    strategy = (out.get("answer_strategy") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    keys = [k for k in (out.get("kb_keys") or []) if str(k).strip()]
    um = (out.get("user_meaning") or "").strip()

    to_kb = (
        ai_route_is_kb_read(out)
        or channel == "kb"
        or strategy in ("kb_only", "kb_then_ai", "api_kb_ai")
        or bool(keys)
    )
    if not to_kb and um:
        top_key, top_score = top_customer_kb_file_match(um, um, ai_route=out)
        if top_key and top_score >= 0.30:
            to_kb = True
            keys = keys or [top_key]
    if not to_kb:
        return out

    resolved = resolve_brain_kb_keys(out, "", um, max_files=4)
    out["intent"] = "general"
    out["data_channel"] = "kb"
    out["kb_keys"] = resolved or keys
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out["route_handler"] = ""
    out.pop("_pincode_delivery_locked", None)
    out["_turn_promotions_done"] = True
    log_reasoning(
        "Brain delivery-policy reconcile — admin KB from brain JSON + embeddings."
    )
    return out


def _try_zero_llm_universal_brain_route(
    original_msg: str,
    msg_en: str = "",
    *,
    ctx: Optional[dict] = None,
) -> Optional[dict]:
    """
    Obvious wishlist / deals / categories / product browse — skip ai_brain_route (~12s).
    Uses msg_en (auto-translated) so Tamil/Hindi/Hinglish reach OpenSearch without routing LLM.
    """
    api_fast = _try_account_list_fast_path(original_msg, msg_en)
    if api_fast:
        _, route_data = api_fast
        route_data = dict(route_data)
        route_data["_universal_brain_route"] = True
        route_data["_turn_promotions_done"] = True
        route_data["_zero_llm_fast"] = True
        log_reasoning("Universal brain — zero-LLM wishlist/order-list fast path.")
        return route_data

    try:
        from services.ai_route_semantics import (
            LIVE_API_FROM_GOAL,
            _structural_details_or_invoice_goal_from_message,
            _structural_refund_goal_from_message,
            _structural_track_goal_from_message,
        )
        from utils.helpers import extract_order_id

        comb_oid = f"{original_msg or ''} {msg_en or ''}".strip()
        if comb_oid and extract_order_id(comb_oid, ""):
            live_goal = (
                _structural_refund_goal_from_message(original_msg, msg_en)
                or _structural_details_or_invoice_goal_from_message(
                    original_msg, msg_en
                )
                or _structural_track_goal_from_message(original_msg, msg_en)
            )
            if live_goal:
                olk_map = {
                    "track": "track",
                    "order_invoice": "invoice",
                    "order_details": "details",
                    "refund_status": "refund_status",
                    "payment": "details",
                }
                route_data = {
                    "user_meaning": (msg_en or original_msg or "")[:200],
                    "reasoning": f"Order id + {live_goal} — skip routing LLM.",
                    "intent": "refund" if live_goal == "refund_status" else "order",
                    "data_channel": "live_api",
                    "needs_order_id": True,
                    "numeric_context": "order_id",
                    "order_lookup_kind": olk_map.get(live_goal, "details"),
                    "route_handler": LIVE_API_FROM_GOAL.get(
                        live_goal, "order_details_api"
                    ),
                    "run_catalog_search": False,
                    "_preflight_api": True,
                    "_universal_brain_route": True,
                    "_turn_promotions_done": True,
                    "_zero_llm_fast": True,
                }
                log_reasoning(
                    f"Universal brain — zero-LLM order live goal={live_goal}."
                )
                return route_data
    except ImportError:
        pass

    try:
        from services.pincode_delivery_fast_path import (
            try_pincode_delivery_fast_route,
            turn_is_pincode_delivery_fast_path,
        )

        pin_fast = try_pincode_delivery_fast_route(
            original_msg, msg_en, conv_for_llm="", ctx=ctx
        )
        if pin_fast:
            _, route_data = pin_fast
            route_data = dict(route_data)
            route_data["_universal_brain_route"] = True
            route_data["_turn_promotions_done"] = True
            route_data["_zero_llm_fast"] = True
            log_reasoning("Universal brain — zero-LLM pincode fast path (named PIN).")
            return route_data

        try:
            from services.semantic_intent import zero_llm_intent_guess_allowed

            allow_phrase_pin = zero_llm_intent_guess_allowed()
        except ImportError:
            allow_phrase_pin = False

        if allow_phrase_pin and turn_is_pincode_delivery_fast_path(
            original_msg, msg_en, "", ctx
        ):
            meaning = (msg_en or original_msg or "").strip()[:200]
            route_data = {
                "user_meaning": meaning or "Check Welfog delivery for customer's area or PIN",
                "reasoning": "Pincode / delivery area — skip routing LLM.",
                "intent": "pincode_check",
                "data_channel": "live_api",
                "needs_order_id": False,
                "run_catalog_search": False,
                "numeric_context": "pincode",
                "order_lookup_kind": "none",
                "route_handler": "pincode_delivery_api",
                "_preflight_api": True,
                "_pincode_delivery_fast": True,
                "_pincode_delivery_locked": True,
                "_universal_brain_route": True,
                "_turn_promotions_done": True,
                "_zero_llm_fast": True,
                "_routing_msg": (original_msg or "").strip(),
            }
            log_reasoning(
                "Universal brain — zero-LLM pincode area/thread (skip routing LLM)."
            )
            return route_data
    except ImportError:
        pass

    catalog_fast = _try_catalog_menu_fast_path(original_msg, msg_en, ctx=ctx)
    if catalog_fast:
        _, route_data = catalog_fast
        route_data = dict(route_data)
        route_data["_universal_brain_route"] = True
        route_data["_turn_promotions_done"] = True
        route_data["_zero_llm_fast"] = True
        log_reasoning("Universal brain — zero-LLM deals/categories fast path.")
        return route_data

    try:
        from services.semantic_intent import zero_llm_intent_guess_allowed

        if not zero_llm_intent_guess_allowed():
            return None
        from services.conversation_followup import is_deals_request_message
        from services.location_delivery_resolver import turn_requests_delivery_serviceability
        from utils.helpers import (
            _message_looks_like_shopping_query,
            _text_is_phone_product_accessory_context,
            message_asks_welfog_categories_list,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
            turn_is_obvious_product_shopping_turn,
        )
        comb = f"{original_msg or ''} {msg_en or ''}".strip()
        if comb and (
            is_deals_request_message(original_msg, msg_en)
            or message_asks_welfog_categories_list(comb)
        ):
            pass
        elif comb and turn_requests_delivery_serviceability(
            original_msg, msg_en, "", allow_llm=False
        ):
            pass
        elif comb and (
            _message_looks_like_shopping_query(comb)
            or _text_is_phone_product_accessory_context(comb)
            or turn_is_obvious_product_shopping_turn(original_msg, msg_en, "")
        ):
            try:
                from services.pincode_delivery_fast_path import (
                    turn_is_pincode_delivery_fast_path,
                )

                if turn_is_pincode_delivery_fast_path(original_msg, msg_en, "", ctx):
                    return None
            except ImportError:
                pass
            try:
                from utils.helpers import _text_is_delivery_serviceability_hypothetical

                if _text_is_delivery_serviceability_hypothetical(comb):
                    return None
            except ImportError:
                pass
            try:
                from utils.helpers import (
                    _naive_six_digit_pin_from_text,
                    message_has_live_pincode_check_intent,
                )

                if _naive_six_digit_pin_from_text(comb) and message_has_live_pincode_check_intent(
                    original_msg, "", msg_en
                ):
                    return None
            except ImportError:
                pass
            if message_is_wishlist_like_request(comb) or message_is_past_purchase_list_request(
                comb
            ):
                return None
            product_route: dict = {
                "intent": "product",
                "data_channel": "catalog",
                "run_catalog_search": True,
                "_product_catalog_locked": True,
                "_universal_brain_route": True,
                "_turn_promotions_done": True,
                "_zero_llm_fast": True,
                "_needs_product_nlu_llm": False,
                "_ai_single_pass": True,
            }
            log_reasoning(
                "Universal brain — zero-LLM product lock (AI NLU → OpenSearch, no keyword sq)."
            )
            return product_route
    except ImportError:
        pass
    return None


def guard_fast_brain_classify(
    original_msg: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
    *,
    msg_en: str = "",
    ctx: Optional[dict] = None,
) -> Optional[dict]:
    """
    Guard-only: ONE ai_brain_route LLM + account-list reconcile — NO enrich stack.
    Any language/style; completes in seconds so /chat guard does not hit 18s cap.
    """
    try:
        from services.chat_flow_telemetry import (
            ensure_brain_route_llm_slot,
            get_cached_brain_route,
            guard_duplicate_brain_route,
            store_brain_route_result,
        )

        cached = get_cached_brain_route()
        if isinstance(cached, dict) and not cached.get("llm_unavailable"):
            return cached
        dup = guard_duplicate_brain_route("guard_fast_brain_classify")
        if isinstance(dup, dict) and not dup.get("llm_unavailable"):
            return dup
        ensure_brain_route_llm_slot()
    except ImportError:
        pass

    ood_fast = _try_obvious_out_of_domain_route(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conv_for_llm,
    )
    if isinstance(ood_fast, dict):
        try:
            from services.chat_flow_telemetry import store_brain_route_result

            store_brain_route_result(ood_fast)
        except ImportError:
            pass
        return ood_fast

    from services.ai_service import ai_brain_route

    route_data = ai_brain_route(
        original_msg, conv_for_llm, reply_lang=reply_lang, msg_en=msg_en
    )
    if not isinstance(route_data, dict) or route_data.get("llm_unavailable"):
        return route_data if isinstance(route_data, dict) else None

    route_data = dict(route_data)
    try:
        from services.account_list_semantics import reconcile_account_list_from_brain_meaning

        route_data = reconcile_account_list_from_brain_meaning(route_data)
    except ImportError:
        pass
    try:
        from services.catalog_menu_resolver import guard_reconcile_catalog_menu_route

        route_data = guard_reconcile_catalog_menu_route(
            route_data,
            original_msg,
            msg_en,
            conv_for_llm,
            reply_lang,
        )
    except ImportError:
        pass

    comb_guard = f"{original_msg or ''} {msg_en or ''}".strip()
    intent_ub = (route_data.get("intent") or "").strip().lower()
    try:
        from services.chat_flow_telemetry import (
            brain_route_authoritative_kb_lock,
            is_authoritative_kb_route_locked,
        )

        kb_locked = (
            is_authoritative_kb_route_locked()
            or brain_route_authoritative_kb_lock(route_data)
        )
    except ImportError:
        kb_locked = (route_data.get("data_channel") or "").strip().lower() == "kb"

    if comb_guard and _text_is_obvious_off_topic(comb_guard) and not kb_locked:
        welfog_live = intent_ub in (
            "product",
            "order",
            "order_history",
            "wishlist",
            "deals",
            "categories",
            "category_feed",
            "pincode_check",
            "refund",
        )
        if not welfog_live:
            ood_post = _try_obvious_out_of_domain_route(
                original_msg,
                msg_en,
                reply_lang=reply_lang,
                conversation_context=conv_for_llm,
            )
            if isinstance(ood_post, dict):
                try:
                    from services.chat_flow_telemetry import store_brain_route_result

                    store_brain_route_result(ood_post)
                except ImportError:
                    pass
                return ood_post
    elif comb_guard and _text_is_obvious_off_topic(comb_guard) and kb_locked:
        log_reasoning(
            "Guard brain — skip OOD post-check; authoritative KB route from brain."
        )

    intent_ub = (route_data.get("intent") or "").strip().lower()
    if intent_ub == "order_history":
        route_data["data_channel"] = "live_api"
        route_data["needs_order_id"] = False
        route_data.setdefault("account_list_kind", "purchase_history_in_chat")
        route_data["run_catalog_search"] = False
    elif intent_ub == "wishlist":
        route_data["data_channel"] = "live_api"
        route_data.setdefault("account_list_kind", "wishlist_in_chat")
        route_data["needs_order_id"] = False
        route_data["run_catalog_search"] = False
    elif intent_ub in ("deals", "categories", "category_feed"):
        route_data["data_channel"] = "live_api"
        route_data["needs_order_id"] = False
        route_data["run_catalog_search"] = False
    elif intent_ub == "out_of_domain":
        route_data["data_channel"] = "none"
        route_data["conversation_scope"] = "out_of_domain"
        route_data["is_welfog_related"] = False
        route_data["run_catalog_search"] = False

    route_data["_turn_promotions_done"] = True
    route_data["_guard_fast_brain"] = True
    log_reasoning(
        f"Guard fast brain: intent={route_data.get('intent')} "
        f"alk={route_data.get('account_list_kind')} "
        f"channel={route_data.get('data_channel')}"
    )
    try:
        from services.chat_flow_telemetry import store_brain_route_result

        store_brain_route_result(route_data)
    except ImportError:
        pass
    return route_data


def _brain_route_needs_product_rescue(out: dict) -> bool:
    """True only when brain misrouted clear shopping as OOD — not KB/chitchat locks."""
    if not isinstance(out, dict):
        return False
    intent = (out.get("intent") or "").strip().lower()
    scope = (out.get("conversation_scope") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    if channel in ("kb", "live_api"):
        return False
    if scope in ("general_chitchat",) or (out.get("meta_kind") or "").strip().lower() in (
        "conversational",
        "assistant_intro",
    ):
        return False
    try:
        from services.ai_route_semantics import brain_route_prefers_kb_answer

        if brain_route_prefers_kb_answer(out):
            return False
    except ImportError:
        pass
    return intent in ("out_of_domain",) or scope == "out_of_domain"


def _turn_should_skip_product_rescue(
    out: dict,
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
) -> bool:
    """Hard blocks — chitchat, thanks, social links, KB policy; never product rescue."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return True
    try:
        from utils.helpers import (
            message_asks_other_company_social_media,
            message_asks_welfog_social_media,
            message_is_user_feedback_or_closing,
        )

        if message_is_user_feedback_or_closing(comb):
            return True
        if message_asks_welfog_social_media(
            comb, conversation_context=conversation_context
        ):
            return True
        if message_asks_other_company_social_media(
            comb, conversation_context=conversation_context
        ):
            return True
    except ImportError:
        pass
    try:
        from services.chitchat_resolver import (
            _chitchat_lane_skips_transactional_guards,
            turn_is_chitchat_not_shopping,
        )

        if _chitchat_lane_skips_transactional_guards(
            original_msg, msg_en, conversation_context
        ):
            return True
        if turn_is_chitchat_not_shopping(
            original_msg,
            msg_en,
            conversation_context,
            out,
            allow_llm=False,
        ):
            return True
    except ImportError:
        pass
    tl = comb.lower()
    if ("product search" in tl or "product dikh" in tl) and any(
        x in tl
        for x in (
            "baat krni",
            "baat karni",
            "baat krna",
            "just talking",
            "not helping",
            "galat",
            "wrong",
            "bta rha",
            "bata rha",
        )
    ):
        return True
    return False


def _shopping_extract_plausible(original_msg: str, msg_en: str, search_terms: str) -> bool:
    """Reject conversational fragments misread as product search by the extract LLM."""
    try:
        from services.product_query_understanding import shopping_extract_plausible

        return shopping_extract_plausible(original_msg, msg_en, search_terms)
    except ImportError:
        return bool((search_terms or "").strip())


def _try_brain_misroute_product_rescue_via_ai(
    out: dict,
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
) -> dict | None:
    """
    AI product extract rescue — any language/typo, no hardcoded product lists.
    Runs before micro-classifier defer so OOD misroutes still reach catalog search.
    """
    if not _brain_route_needs_product_rescue(out):
        return None
    if _turn_should_skip_product_rescue(
        out,
        original_msg,
        msg_en,
        conversation_context=conversation_context,
    ):
        log_reasoning("Product rescue skipped — chitchat/social/KB/thanks turn.")
        return None
    try:
        from services.chitchat_resolver import turn_is_chitchat_not_shopping

        if turn_is_chitchat_not_shopping(
            original_msg,
            msg_en,
            conversation_context,
            out,
            allow_llm=False,
        ):
            return None
    except ImportError:
        pass
    try:
        from services.chat_flow_telemetry import ensure_product_rescue_llm_slot
        from services.product_catalog_resolver import (
            KIND_PRODUCT_SEARCH,
            ai_classify_product_search_turn,
        )

        ensure_product_rescue_llm_slot()
        clean_route = dict(out)
        clean_route.pop("user_meaning", None)
        clean_route.pop("scope_reply", None)
        clean_route["intent"] = "out_of_domain"
        clean_route["data_channel"] = "none"
        classified = ai_classify_product_search_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=clean_route,
            force_llm=True,
        )
        sq = ""
        conf = 0.0
        if classified:
            kind = (classified.get("turn_kind") or "").strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if kind == KIND_PRODUCT_SEARCH and conf >= 0.62:
                entities = (
                    classified.get("entities")
                    if isinstance(classified.get("entities"), dict)
                    else {}
                )
                sq = (
                    (classified.get("search_query") or "").strip()
                    or str(entities.get("product_name") or "").strip()
                )
        if not sq:
            try:
                from services.product_query_understanding import (
                    resolve_catalog_search_terms_for_message,
                )

                sq = resolve_catalog_search_terms_for_message(
                    original_msg,
                    msg_en,
                    ai_route=clean_route,
                    conversation_context=conversation_context,
                    force_llm=True,
                )
                if sq:
                    conf = 0.68
            except ImportError:
                pass
        if not sq or len(sq) < 2:
            if classified:
                log_reasoning(
                    f"Product rescue skipped — classifier kind={(classified.get('turn_kind') or '')!r} "
                    f"conf={float(classified.get('confidence') or 0):.2f}."
                )
            return None
        if not _shopping_extract_plausible(original_msg, msg_en, sq):
            log_reasoning(
                f"Product rescue skipped — sq={sq!r} not shopping-plausible."
            )
            return None
        promoted = dict(out)
        promoted["intent"] = "product"
        promoted["data_channel"] = "catalog"
        promoted["run_catalog_search"] = True
        promoted["is_welfog_related"] = True
        promoted["needs_order_id"] = False
        promoted["numeric_context"] = "none"
        promoted["meta_kind"] = "none"
        promoted["conversation_scope"] = "welfog_support"
        promoted["search_query"] = sq
        promoted["_product_catalog_locked"] = True
        promoted.setdefault("_product_entities", {})["product_name"] = sq
        promoted.pop("scope_reply", None)
        log_reasoning(
            f"Product rescue (classifier): sq={sq!r} conf={conf:.2f} over brain misroute."
        )
        return promoted
    except ImportError:
        pass
    return None


def _fast_finalize_promote_catalog_menu(
    out: dict,
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
    rescue_misroute: bool = False,
) -> dict | None:
    """Promote deals/categories before product search hijacks menu-style queries."""
    if out.get("_catalog_menu_locked"):
        return None
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    try:
        from services.catalog_menu_resolver import (
            KIND_NONE,
            _brain_catalog_menu_resolution,
            _build_route_data_from_resolution,
            _suspicious_product_search_for_menu,
            _user_meaning_suggests_catalog_menu,
            resolve_catalog_menu_turn,
        )
        from services.ai_route_semantics import (
            brain_turn_indicates_categories,
            brain_turn_indicates_deals,
            reconcile_categories_from_brain_meaning,
            reconcile_deals_from_brain_meaning,
        )

        if brain_turn_indicates_deals(out, original_msg=original_msg, msg_en=msg_en):
            promoted = reconcile_deals_from_brain_meaning(out)
            promoted["_turn_promotions_done"] = True
            promoted["_universal_brain_route"] = True
            return promoted

        if brain_turn_indicates_categories(
            out, original_msg=original_msg, msg_en=msg_en
        ):
            promoted = reconcile_categories_from_brain_meaning(out)
            promoted["_turn_promotions_done"] = True
            promoted["_universal_brain_route"] = True
            return promoted

        intent = (out.get("intent") or "").strip().lower()
        if intent in ("deals", "categories", "category_feed"):
            brain = _brain_catalog_menu_resolution(out)
            if brain:
                promoted = _build_route_data_from_resolution(brain, comb, out)
                promoted["_turn_promotions_done"] = True
                promoted["_universal_brain_route"] = True
                return promoted

        needs_force = (
            _suspicious_product_search_for_menu(out)
            or _user_meaning_suggests_catalog_menu(out)
        )
        if not needs_force and rescue_misroute:
            if out.get("is_welfog_related", True) is not False:
                needs_force = True
        if not needs_force:
            scope = (out.get("conversation_scope") or "").strip().lower()
            channel = (out.get("data_channel") or "").strip().lower()
            if (
                intent in ("general", "out_of_domain", "product")
                and out.get("is_welfog_related", True) is not False
                and (
                    scope in ("general_chitchat",)
                    or channel == "kb"
                    or bool((out.get("scope_reply") or "").strip())
                )
            ):
                needs_force = True

        if not needs_force:
            return None

        resolved = resolve_catalog_menu_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=out,
            allow_llm=True,
            force_llm=True,
        )
        if resolved.kind == KIND_NONE:
            return None

        promoted = _build_route_data_from_resolution(resolved, comb, out)
        promoted["_turn_promotions_done"] = True
        promoted["_universal_brain_route"] = True
        promoted["is_welfog_related"] = True
        promoted["conversation_scope"] = "welfog_support"
        promoted.pop("scope_reply", None)
        return promoted
    except ImportError:
        pass
    return None


def _fast_finalize_promote_product_catalog(
    out: dict,
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
    allow_product_llm: bool = False,
) -> dict | None:
    """Promote catalog search when brain misrouted shopping as OOD/chitchat/KB."""
    try:
        from services.catalog_menu_resolver import (
            _suspicious_product_search_for_menu,
            _user_meaning_suggests_catalog_menu,
        )

        if _suspicious_product_search_for_menu(out) or _user_meaning_suggests_catalog_menu(
            out
        ):
            log_reasoning(
                "Product catalog promote skipped — catalog menu (deals/categories) likely."
            )
            return None
    except ImportError:
        pass
    if _turn_should_skip_product_rescue(
        out,
        original_msg,
        msg_en,
        conversation_context=conversation_context,
    ):
        return None

    rescued = _try_brain_misroute_product_rescue_via_ai(
        out,
        original_msg,
        msg_en,
        conversation_context=conversation_context,
    )
    if isinstance(rescued, dict):
        return rescued

    try:
        from services.ai_route_semantics import (
            brain_route_indicates_product_catalog,
            reconcile_product_catalog_from_brain_meaning,
        )
        from services.product_catalog_resolver import (
            apply_product_catalog_to_route,
            product_catalog_route_is_locked,
            turn_requests_product_catalog,
        )

        if brain_route_indicates_product_catalog(out):
            promoted = reconcile_product_catalog_from_brain_meaning(
                out, original_msg=original_msg, msg_en=msg_en
            )
            sq_brain = (promoted.get("search_query") or "").strip()
            if sq_brain and not _shopping_extract_plausible(
                original_msg, msg_en, sq_brain
            ):
                log_reasoning(
                    f"Product brain lock skipped — sq={sq_brain!r} not shopping-plausible."
                )
            elif product_catalog_route_is_locked(promoted):
                promoted["_turn_promotions_done"] = True
                promoted["_universal_brain_route"] = True
                promoted["is_welfog_related"] = True
                promoted["conversation_scope"] = "welfog_support"
                promoted.pop("scope_reply", None)
                return promoted

        if turn_requests_product_catalog(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=out,
            allow_llm=allow_product_llm,
        ):
            promoted = apply_product_catalog_to_route(
                out,
                original_msg,
                msg_en,
                conversation_context,
            )
            if product_catalog_route_is_locked(promoted):
                promoted["_turn_promotions_done"] = True
                promoted["_universal_brain_route"] = True
                promoted["is_welfog_related"] = True
                promoted["conversation_scope"] = "welfog_support"
                promoted.pop("scope_reply", None)
                return promoted
    except ImportError:
        pass
    return None


def _try_fast_finalize_brain_route_after_ai(
    route_data: dict,
    original_msg: str = "",
    msg_en: str = "",
    *,
    conversation_context: str = "",
    ctx: Optional[dict] = None,
) -> dict | None:
    """
    Trust decisive ai_brain_route JSON — skip KB embedding reconcile and heavy enrich.
    Industry path: one routing LLM → lock handler fields → dispatch (2–4s target).
    """
    out = dict(route_data or {})
    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    scope = (out.get("conversation_scope") or "").strip().lower()
    olk = (out.get("order_lookup_kind") or "").strip().lower()
    meta = (out.get("meta_kind") or "none").strip().lower()
    kb_keys = [str(k).strip() for k in (out.get("kb_keys") or []) if str(k).strip()]
    authoritative_kb_json = (
        channel == "kb"
        and bool(kb_keys)
        and not out.get("needs_order_id")
        and out.get("is_welfog_related", True) is not False
    )

    if (
        not authoritative_kb_json
        and (
            intent == "out_of_domain"
            or scope in ("general_chitchat", "out_of_domain", "harm_sensitive")
            or out.get("is_welfog_related") is False
        )
    ):
        out["_turn_promotions_done"] = True
        out["_universal_brain_route"] = True
        out["run_catalog_search"] = False
        out["data_channel"] = "none"
        if scope in ("out_of_domain",) or out.get("is_welfog_related") is False:
            out["intent"] = "out_of_domain"
            out["conversation_scope"] = "out_of_domain"
            out["is_welfog_related"] = False
        elif scope == "general_chitchat":
            out["intent"] = "general"
            out["is_welfog_related"] = True
        log_reasoning("Universal brain — scope/OOD fast finalize (skip KB/catalog).")
        return out

    catalog_fast = _fast_finalize_promote_catalog_menu(
        out,
        original_msg,
        msg_en,
        conversation_context=conversation_context,
    )
    if isinstance(catalog_fast, dict):
        log_reasoning(
            "Universal brain — catalog menu fast finalize (deals/categories over misroute)."
        )
        return catalog_fast

    try:
        from services.ai_route_semantics import (
            brain_route_indicates_informational_kb,
            reconcile_welfog_kb_from_brain_meaning,
        )

        if brain_route_indicates_informational_kb(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        ):
            out = reconcile_welfog_kb_from_brain_meaning(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
            out["_turn_promotions_done"] = True
            out["_universal_brain_route"] = True
            out["is_welfog_related"] = True
            out["conversation_scope"] = "welfog_support"
            out.pop("scope_reply", None)
            log_reasoning(
                "Universal brain — KB fast finalize (semantic FAQ over catalog misroute)."
            )
            return out
    except ImportError:
        pass

    product_fast = _fast_finalize_promote_product_catalog(
        out,
        original_msg,
        msg_en,
        conversation_context=conversation_context,
        allow_product_llm=False,
    )
    if isinstance(product_fast, dict):
        log_reasoning(
            "Universal brain — product catalog fast finalize (shopping over misroute)."
        )
        return product_fast

    try:
        from services.ai_route_semantics import (
            brain_route_prefers_kb_answer,
            reconcile_welfog_kb_from_brain_meaning,
        )

        if brain_route_prefers_kb_answer(out):
            out = reconcile_welfog_kb_from_brain_meaning(out)
            if (out.get("intent") or "").strip().lower() in (
                "tracking",
                "order_track",
                "order_tracking",
            ):
                out["intent"] = "general"
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Universal brain — KB fast finalize (brain locked kb channel)."
            )
            return out
    except ImportError:
        pass

    try:
        from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path
        from services.location_delivery_resolver import (
            turn_continues_pincode_area_check,
            turn_requests_delivery_serviceability,
        )
        from services.ai_route_semantics import reconcile_pincode_delivery_from_brain_meaning

        comb_del = f"{original_msg or ''} {msg_en or ''}".strip()
        try:
            from services.ai_route_semantics import brain_route_prefers_kb_answer

            kb_locked = brain_route_prefers_kb_answer(out)
        except ImportError:
            kb_locked = (channel == "kb" or bool(out.get("kb_keys")))
        delivery_keep = (
            out.get("_pincode_delivery_locked")
            or out.get("_pincode_delivery_fast")
            or turn_is_pincode_delivery_fast_path(
                original_msg, msg_en, conversation_context
            )
            or turn_continues_pincode_area_check(
                comb_del, conversation_context, out
            )
            or (
                not kb_locked
                and turn_requests_delivery_serviceability(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route=out,
                )
            )
        )
        if delivery_keep:
            out = reconcile_pincode_delivery_from_brain_meaning(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Universal brain — delivery/pincode fast finalize (area follow-up or serviceability)."
            )
            return out
    except ImportError:
        pass

    if scope == "general_chitchat" or (
        intent == "general"
        and channel == "none"
        and meta in ("conversational", "none", "")
        and not (out.get("kb_keys") or [])
    ):
        catalog_over_chitchat = _fast_finalize_promote_catalog_menu(
            out,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            rescue_misroute=True,
        )
        if isinstance(catalog_over_chitchat, dict):
            log_reasoning(
                "Universal brain — catalog menu promoted over chitchat misroute."
            )
            return catalog_over_chitchat
        product_over_chitchat = _fast_finalize_promote_product_catalog(
            out,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            allow_product_llm=True,
        )
        if isinstance(product_over_chitchat, dict):
            log_reasoning(
                "Universal brain — product promoted over chitchat misroute."
            )
            return product_over_chitchat
        out["_turn_promotions_done"] = True
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["is_welfog_related"] = True
        out.setdefault("conversation_scope", "general_chitchat")
        out["scope_reply"] = ""
        log_reasoning("Universal brain — general chitchat fast finalize.")
        return out

    if (
        intent == "out_of_domain"
        or scope == "out_of_domain"
        or out.get("is_welfog_related") is False
    ):
        try:
            from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path
            from services.location_delivery_resolver import (
                turn_continues_pincode_area_check,
                turn_requests_delivery_serviceability,
            )
            from services.ai_route_semantics import reconcile_pincode_delivery_from_brain_meaning

            comb_ood = f"{original_msg or ''} {msg_en or ''}".strip()
            if (
                turn_is_pincode_delivery_fast_path(
                    original_msg, msg_en, conversation_context
                )
                or turn_continues_pincode_area_check(
                    comb_ood, conversation_context, out
                )
                or turn_requests_delivery_serviceability(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route=out,
                )
            ):
                out = reconcile_pincode_delivery_from_brain_meaning(
                    out,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — delivery promoted over OOD/chitchat misroute."
                )
                return out
        except ImportError:
            pass
        try:
            from services.chitchat_resolver import (
                _chitchat_lane_skips_transactional_guards,
                turn_is_chitchat_not_shopping,
            )

            if _chitchat_lane_skips_transactional_guards(
                original_msg, msg_en, conversation_context
            ) or turn_is_chitchat_not_shopping(
                original_msg,
                msg_en,
                conversation_context,
                out,
                allow_llm=False,
            ):
                out["intent"] = "general"
                out["conversation_scope"] = "general_chitchat"
                out["is_welfog_related"] = True
                out["data_channel"] = "none"
                out["run_catalog_search"] = False
                out["scope_reply"] = ""
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — chitchat promoted (was misrouted as OOD)."
                )
                return out
        except ImportError:
            pass
        catalog_over_ood = _fast_finalize_promote_catalog_menu(
            out,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            rescue_misroute=True,
        )
        if isinstance(catalog_over_ood, dict):
            log_reasoning(
                "Universal brain — catalog menu promoted over OOD misroute."
            )
            return catalog_over_ood
        product_over_ood = _fast_finalize_promote_product_catalog(
            out,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            allow_product_llm=True,
        )
        if isinstance(product_over_ood, dict):
            log_reasoning(
                "Universal brain — product catalog promoted over OOD misroute."
            )
            return product_over_ood
        out["_turn_promotions_done"] = True
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["scope_reply"] = ""
        log_reasoning("Universal brain — OOD fast finalize (skip reconcile/enrich).")
        return out

  # Brain misroute: pincode/live_api without delivery signal — only when NOT a delivery turn.
    comb_ff = f"{original_msg or ''} {msg_en or ''}".strip()
    if (
        channel == "live_api"
        and intent == "pincode_check"
        and comb_ff
        and not out.get("_pincode_delivery_locked")
        and not out.get("_pincode_delivery_fast")
    ):
        try:
            from services.pincode_delivery_fast_path import (
                turn_is_pincode_delivery_fast_path,
            )
            from services.location_delivery_resolver import (
                turn_continues_pincode_area_check,
                turn_requests_delivery_serviceability,
            )
            from services.ai_route_semantics import (
                brain_route_prefers_kb_answer,
                brain_turn_indicates_welfog_kb,
                reconcile_welfog_kb_from_brain_meaning,
            )
            from services.kb_service import (
                KB_ANSWER_MIN_CONFIDENCE,
                resolve_best_faq_chunk_for_question,
            )

            faq_probe = resolve_best_faq_chunk_for_question(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=out,
            )
            faq_ok = bool(
                faq_probe
                and float(faq_probe.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE
            )
            if faq_ok or brain_turn_indicates_welfog_kb(out):
                out = reconcile_welfog_kb_from_brain_meaning(out)
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — KB promoted over pincode_check misroute."
                )
                return out

            kb_locked = brain_route_prefers_kb_answer(out)
            if (
                turn_is_pincode_delivery_fast_path(
                    original_msg, msg_en, conversation_context
                )
                or turn_continues_pincode_area_check(
                    comb_ff, conversation_context, out
                )
                or (
                    not kb_locked
                    and turn_requests_delivery_serviceability(
                        original_msg,
                        msg_en,
                        conversation_context,
                        ai_route=out,
                    )
                )
            ):
                from services.ai_route_semantics import (
                    reconcile_pincode_delivery_from_brain_meaning,
                )

                out = reconcile_pincode_delivery_from_brain_meaning(
                    out,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — pincode kept (delivery thread / area follow-up)."
                )
                return out
        except ImportError:
            pass
        has_pin = bool(re.search(r"\b\d{6}\b", comb_ff))
        low_ff = f" {comb_ff.lower()} "
        delivery_hint = any(
            x in low_ff
            for x in (
                "pincode",
                "pin code",
                "zip",
                "deliver",
                "delivery",
                "ship",
                "courier",
                "area",
                "location",
                "address",
            )
        )
        if not has_pin and not delivery_hint:
            try:
                from services.ai_route_semantics import (
                    brain_turn_indicates_welfog_kb,
                    reconcile_welfog_kb_from_brain_meaning,
                )
                from services.kb_service import (
                    KB_ANSWER_MIN_CONFIDENCE,
                    resolve_best_faq_chunk_for_question,
                )

                faq_probe = resolve_best_faq_chunk_for_question(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ai_route=out,
                )
                faq_ok = bool(
                    faq_probe
                    and float(faq_probe.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE
                )
                if brain_turn_indicates_welfog_kb(out) or faq_ok:
                    out = reconcile_welfog_kb_from_brain_meaning(out)
                    out["_turn_promotions_done"] = True
                    log_reasoning(
                        "Universal brain — KB promoted over pincode misroute (FAQ/support)."
                    )
                    return out
            except ImportError:
                pass
            try:
                from services.query_intent_classifier import (
                    INTENT_HARM,
                    INTENT_CHITCHAT,
                    INTENT_OUT,
                    ai_classify_query_intent,
                )

                qd = ai_classify_query_intent(
                    original_msg,
                    msg_en,
                    conversation_context,
                    ignore_routing_complete=True,
                )
                if qd and qd.detected_intent == INTENT_HARM:
                    out["intent"] = "general"
                    out["conversation_scope"] = "harm_sensitive"
                    out["data_channel"] = "none"
                    out["run_catalog_search"] = False
                    out["is_welfog_related"] = False
                    out["scope_reply"] = (qd.reply or "").strip()
                    out["_turn_promotions_done"] = True
                    log_reasoning(
                        "Universal brain — harm_sensitive promoted (pincode misroute)."
                    )
                    return out
                if qd and qd.detected_intent in (INTENT_CHITCHAT, INTENT_OUT):
                    out["intent"] = "general"
                    out["conversation_scope"] = (
                        "general_chitchat"
                        if qd.detected_intent == INTENT_CHITCHAT
                        else "out_of_domain"
                    )
                    out["data_channel"] = "none"
                    out["run_catalog_search"] = False
                    out["is_welfog_related"] = qd.detected_intent == INTENT_CHITCHAT
                    out["scope_reply"] = (qd.reply or "").strip()
                    out["_turn_promotions_done"] = True
                    log_reasoning(
                        "Universal brain — scope repaired from pincode misroute."
                    )
                    return out
                else:
                    out["intent"] = "general"
                    out["conversation_scope"] = "out_of_domain"
                    out["data_channel"] = "none"
                    out["run_catalog_search"] = False
                    out["is_welfog_related"] = False
                    out["scope_reply"] = ""
                    out["_turn_promotions_done"] = True
                    try:
                        from services.conversation_scope import ai_classify_scope_and_reply

                        harm_dec = ai_classify_scope_and_reply(
                            original_msg, msg_en, conversation_context, ""
                        )
                        if harm_dec and harm_dec.scope == "harm_sensitive":
                            out["conversation_scope"] = "harm_sensitive"
                            out["scope_reply"] = (harm_dec.reply or "").strip()
                    except ImportError:
                        pass
                    log_reasoning(
                        "Universal brain — pincode blocked (no delivery signal in message)."
                    )
                    return out
            except ImportError:
                pass

    try:
        from services.ai_route_semantics import (
            ai_meaning_describes_delivery_serviceability,
            brain_route_indicates_account_list_live,
            brain_route_skip_heavy_enrich,
            brain_turn_indicates_categories,
            brain_turn_indicates_deals,
            brain_turn_indicates_welfog_kb,
            reconcile_categories_from_brain_meaning,
            reconcile_deals_from_brain_meaning,
            reconcile_pincode_delivery_from_brain_meaning,
            reconcile_product_catalog_from_brain_meaning,
            reconcile_welfog_kb_from_brain_meaning,
        )
        from services.account_list_semantics import (
            account_list_route_is_locked,
            reconcile_account_list_from_brain_meaning,
        )

        if brain_turn_indicates_deals(
            out, original_msg=original_msg, msg_en=msg_en
        ):
            out = reconcile_deals_from_brain_meaning(out)
            out["_turn_promotions_done"] = True
            log_reasoning("Universal brain — deals fast finalize from brain meaning.")
            return out

        if brain_turn_indicates_categories(
            out, original_msg=original_msg, msg_en=msg_en
        ):
            out = reconcile_categories_from_brain_meaning(out)
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Universal brain — categories fast finalize from brain meaning."
            )
            return out

        if brain_turn_indicates_welfog_kb(out):
            promote_kb = True
            if intent == "out_of_domain" or scope in (
                "out_of_domain",
                "general_chitchat",
                "harm_sensitive",
            ):
                promote_kb = False
            elif out.get("is_welfog_related") is False:
                promote_kb = False
            else:
                try:
                    from services.query_intent_classifier import (
                        INTENT_WELFOG,
                        ai_classify_query_intent,
                    )

                    qd_kb = ai_classify_query_intent(
                        original_msg,
                        msg_en,
                        conversation_context,
                        ignore_routing_complete=True,
                    )
                    if qd_kb and qd_kb.detected_intent != INTENT_WELFOG:
                        promote_kb = False
                        out["intent"] = "general"
                        out["conversation_scope"] = (
                            "harm_sensitive"
                            if qd_kb.detected_intent == "harm_sensitive"
                            else (
                                "general_chitchat"
                                if qd_kb.detected_intent == "general_chitchat"
                                else "out_of_domain"
                            )
                        )
                        out["data_channel"] = "none"
                        out["run_catalog_search"] = False
                        out["is_welfog_related"] = (
                            qd_kb.detected_intent == "general_chitchat"
                        )
                        out["scope_reply"] = (qd_kb.reply or "").strip()
                        out["_turn_promotions_done"] = True
                        log_reasoning(
                            "Universal brain — scope repaired (blocked KB misroute)."
                        )
                        return out
                except ImportError:
                    pass
            if promote_kb:
                out = reconcile_welfog_kb_from_brain_meaning(out)
                out["_turn_promotions_done"] = True
                log_reasoning("Universal brain — Welfog KB fast finalize from brain meaning.")
                return out

        um_low = f" {(out.get('user_meaning') or '').lower()} "
        alk = (out.get("account_list_kind") or "").strip().lower()
        if intent == "product" and (
            "wishlist" in um_low
            or "saved item" in um_low
            or "liked product" in um_low
            or alk.startswith("wishlist")
        ):
            out["intent"] = "wishlist"
            out = reconcile_account_list_from_brain_meaning(out)
            out["data_channel"] = "live_api"
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
            out["_turn_promotions_done"] = True
            out.setdefault("account_list_kind", "wishlist_in_chat")
            log_reasoning("Universal brain — wishlist promoted from brain meaning.")
            return out

        if intent in ("order_history", "wishlist"):
            out = reconcile_account_list_from_brain_meaning(out)
            out["data_channel"] = "live_api"
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
            out["_turn_promotions_done"] = True
            if intent == "wishlist":
                out.setdefault("account_list_kind", "wishlist_in_chat")
            else:
                out.setdefault("account_list_kind", "purchase_history_in_chat")
            log_reasoning(f"Universal brain — {intent} fast finalize.")
            return out

        if intent == "product" and (
            "order history" in um_low
            or "purchase history" in um_low
            or "past order" in um_low
            or alk.startswith("purchase_history")
        ):
            out["intent"] = "order_history"
            out = reconcile_account_list_from_brain_meaning(out)
            out["data_channel"] = "live_api"
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
            out["_turn_promotions_done"] = True
            out.setdefault("account_list_kind", "purchase_history_in_chat")
            log_reasoning("Universal brain — order_history promoted from brain meaning.")
            return out

        if intent in ("deals", "categories", "category_feed"):
            out["data_channel"] = "live_api"
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
            out["_turn_promotions_done"] = True
            log_reasoning(f"Universal brain — {intent} menu fast finalize.")
            return out

        has_order_id = False
        try:
            from utils.helpers import extract_order_id

            comb_oid = f"{original_msg or ''} {msg_en or ''}".strip()
            has_order_id = bool(extract_order_id(comb_oid, conversation_context))
        except ImportError:
            pass

        delivery_meaning = ai_meaning_describes_delivery_serviceability(out)
        if not has_order_id and (intent == "pincode_check" or delivery_meaning):
            out = reconcile_pincode_delivery_from_brain_meaning(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
            if (out.get("intent") or "").strip().lower() == "pincode_check":
                out["data_channel"] = "live_api"
                out["needs_order_id"] = False
                out["run_catalog_search"] = False
                out.setdefault("route_handler", "pincode_delivery_api")
                out["_pincode_delivery_locked"] = True
                out["_turn_promotions_done"] = True
                log_reasoning("Universal brain — pincode_check fast finalize.")
                return out

        if not has_order_id and intent in ("order", "general", "product"):
            pin_promoted = reconcile_pincode_delivery_from_brain_meaning(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
            if (pin_promoted.get("intent") or "").strip().lower() == "pincode_check":
                out = pin_promoted
                out["data_channel"] = "live_api"
                out["needs_order_id"] = False
                out["run_catalog_search"] = False
                out.setdefault("route_handler", "pincode_delivery_api")
                out["_pincode_delivery_locked"] = True
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — delivery area promoted to pincode_check."
                )
                return out

        if intent == "product" and channel == "catalog":
            out = reconcile_product_catalog_from_brain_meaning(
                out, original_msg=original_msg, msg_en=msg_en
            )
            out["_product_catalog_locked"] = True
            out["_needs_product_nlu_llm"] = False
            out["_ai_single_pass"] = True
            out["_turn_promotions_done"] = True
            log_reasoning("Universal brain — product catalog fast finalize.")
            return out

        if account_list_route_is_locked(out) or brain_route_indicates_account_list_live(
            out
        ):
            out["_turn_promotions_done"] = True
            out.setdefault("data_channel", "live_api")
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
            log_reasoning("Universal brain — account-list fast finalize.")
            return out

        if intent in ("order", "refund", "payment") and channel == "live_api":
            if olk not in ("none", "") or out.get("needs_order_id"):
                out["_turn_promotions_done"] = True
                log_reasoning(
                    f"Universal brain — order live fast finalize olk={olk or 'ask_id'}."
                )
                return out

        if scope in ("general_chitchat", "harm_sensitive") and (
            out.get("scope_reply") or ""
        ).strip():
            out["_turn_promotions_done"] = True
            log_reasoning("Universal brain — chitchat scope fast finalize.")
            return out

        if channel == "kb" and brain_route_skip_heavy_enrich(out):
            out["_turn_promotions_done"] = True
            log_reasoning("Universal brain — KB fast finalize.")
            return out
    except ImportError:
        pass

    return None


def early_universal_brain_route(
    original_msg: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
    *,
    msg_en: str = "",
    ctx: Optional[dict] = None,
) -> Optional[dict]:
    """
    ONE LLM at turn start — meaning + intent for greeting, chitchat, API, KB, product, order.
    Cached for the rest of the HTTP request (no duplicate micro-classifiers).
    """
    try:
        from services.semantic_intent import strict_ai_semantic_mode

        _strict_ai = strict_ai_semantic_mode()
    except ImportError:
        _strict_ai = True

    if not _strict_ai:
        ood_fast = _try_obvious_out_of_domain_route(
            original_msg,
            msg_en,
            reply_lang=reply_lang,
            conversation_context=conv_for_llm,
        )
        if ood_fast:
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(ood_fast)
            except ImportError:
                pass
            return ood_fast

    if not _strict_ai:
        route_msg, route_msg_en, _mixed = _strip_greeting_filler_from_mixed_turn(
            original_msg, msg_en
        )
        if _mixed:
            original_msg = route_msg
            msg_en = route_msg_en

    if not _strict_ai:
        ood_fast = _try_obvious_out_of_domain_route(
            original_msg,
            msg_en,
            reply_lang=reply_lang,
            conversation_context=conv_for_llm,
        )
        if ood_fast:
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(ood_fast)
            except ImportError:
                pass
            return ood_fast

    route_data: dict | None = None
    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            guard_duplicate_brain_route,
            store_brain_route_result,
        )

        cached = get_cached_brain_route()
        if _brain_route_is_usable(cached) and cached.get("_turn_promotions_done"):
            log_reasoning(
                f"Universal brain route (cached): intent={cached.get('intent')} "
                f"scope={cached.get('conversation_scope')}"
            )
            return cached
        dup = guard_duplicate_brain_route("early_universal_brain_route")
        if dup is not None:
            if dup.get("llm_unavailable"):
                log_reasoning("Universal brain route (reuse): LLM unavailable this turn.")
                return dup
            if _brain_route_is_usable(dup):
                if dup.get("_turn_promotions_done"):
                    log_reasoning(
                        f"Universal brain route (reuse): intent={dup.get('intent')} "
                        f"scope={dup.get('conversation_scope')}"
                    )
                    return dup
                route_data = dict(dup)
    except ImportError:
        pass

    _t_enrich_gate = time.perf_counter()
    _t_brain_llm = _t_enrich_gate
    if route_data is None and _strict_ai:
        pass
    elif route_data is None:
        route_data = _try_zero_llm_universal_brain_route(
            original_msg, msg_en, ctx=ctx
        )
        if isinstance(route_data, dict):
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(route_data)
            except ImportError:
                pass

    if route_data is None:
        try:
            from services.chat_flow_telemetry import ensure_brain_route_llm_slot

            ensure_brain_route_llm_slot()
        except ImportError:
            pass
        log_reasoning("Universal brain — ai_brain_route starting (one LLM).")
        _t_brain_llm = time.perf_counter()
        route_data = ai_brain_route(
            original_msg, conv_for_llm, reply_lang=reply_lang, msg_en=msg_en
        )
        if isinstance(route_data, dict):
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(route_data)
            except ImportError:
                pass
        if not _brain_route_is_usable(route_data):
            return route_data if isinstance(route_data, dict) else None
        route_data = dict(route_data)

    try:
        from services.ai_route_semantics import repair_brain_json_quality

        route_data = repair_brain_json_quality(route_data, original_msg, msg_en=msg_en)
    except ImportError:
        pass
    route_data["_universal_brain_route"] = True
    route_data = _reconcile_off_topic_brain_misroute(
        route_data, original_msg, msg_en=msg_en
    )

    _t_post_brain = time.perf_counter()
    fast_final = _try_fast_finalize_brain_route_after_ai(
        route_data,
        original_msg,
        msg_en,
        conversation_context=conv_for_llm,
        ctx=ctx,
    )
    if isinstance(fast_final, dict):
        route_data = fast_final
        um = (route_data.get("user_meaning") or "").strip()
        log_reasoning(
            f"Universal brain route: intent={route_data.get('intent')} "
            f"channel={route_data.get('data_channel')} "
            f"olk={route_data.get('order_lookup_kind')} — {um[:90] or '-'}"
        )
        try:
            from services.chat_flow_telemetry import store_brain_route_result

            store_brain_route_result(route_data)
        except ImportError:
            pass
        return route_data

    # Ambiguous turns only — KB embedding reconcile + enrich (never for locked live API/catalog).
    route_data = _reconcile_semantic_kb_route(
        route_data, original_msg, msg_en=msg_en
    )
    route_data = _reconcile_delivery_policy_from_brain_json(route_data)
    intent_ub = (route_data.get("intent") or "").strip().lower()
    try:
        from services.account_list_semantics import reconcile_account_list_from_brain_meaning

        route_data = reconcile_account_list_from_brain_meaning(route_data)
        if (route_data.get("intent") or "").strip().lower() == "wishlist":
            route_data["data_channel"] = "live_api"
            route_data.setdefault("account_list_kind", "wishlist_in_chat")
            route_data["run_catalog_search"] = False
            route_data["_turn_promotions_done"] = True
    except ImportError:
        pass
    try:
        from services.catalog_menu_resolver import guard_reconcile_catalog_menu_route

        route_data = guard_reconcile_catalog_menu_route(
            route_data,
            original_msg,
            msg_en,
            conv_for_llm,
            reply_lang,
        )
    except ImportError:
        pass
    intent_ub = (route_data.get("intent") or "").strip().lower()
    if intent_ub == "order_history":
        route_data["data_channel"] = "live_api"
        route_data["needs_order_id"] = False
        route_data.setdefault("account_list_kind", "purchase_history_in_chat")
        route_data["run_catalog_search"] = False
        route_data["_turn_promotions_done"] = True
    elif intent_ub == "wishlist":
        route_data["data_channel"] = "live_api"
        route_data.setdefault("account_list_kind", "wishlist_in_chat")
        route_data["run_catalog_search"] = False
        route_data["_turn_promotions_done"] = True
    try:
        from services.account_list_semantics import account_list_route_is_locked
        from services.ai_route_semantics import brain_route_indicates_account_list_live

        if intent_ub in ("order_history", "wishlist"):
            route_data["_turn_promotions_done"] = True
            route_data.setdefault("data_channel", "live_api")
            route_data["needs_order_id"] = False
            route_data["run_catalog_search"] = False
            log_reasoning(
                f"Universal brain — {intent_ub} from AI JSON; skip enrich stack."
            )
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(route_data)
            except ImportError:
                pass
            return route_data

        if intent_ub in ("deals", "categories", "category_feed"):
            route_data["data_channel"] = "live_api"
            route_data["needs_order_id"] = False
            route_data["run_catalog_search"] = False
            route_data["_turn_promotions_done"] = True
            log_reasoning(
                f"Universal brain — {intent_ub} catalog menu; skip enrich stack."
            )
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(route_data)
            except ImportError:
                pass
            return route_data

        if account_list_route_is_locked(
            route_data
        ) or brain_route_indicates_account_list_live(route_data):
            route_data["_turn_promotions_done"] = True
            route_data.setdefault("data_channel", "live_api")
            route_data["needs_order_id"] = False
            route_data["run_catalog_search"] = False
            log_reasoning(
                "Universal brain — account-list from AI JSON; skip enrich/product/pincode stack."
            )
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                store_brain_route_result(route_data)
            except ImportError:
                pass
            return route_data
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import (
            brain_route_skip_heavy_enrich,
            enrich_universal_brain_route,
        )

        if brain_route_skip_heavy_enrich(route_data):
            route_data["_turn_promotions_done"] = True
            if isinstance(ctx, dict) and ctx.get("last"):
                route_data["_ctx_last"] = ctx.get("last")
            log_reasoning(
                "Universal brain — raw OOD/KB route; skip enrich before dispatch."
            )
        else:
            route_data = enrich_universal_brain_route(
                route_data,
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=reply_lang,
                ctx=ctx,
            )
    except ImportError:
        if isinstance(ctx, dict) and ctx.get("last"):
            route_data["_ctx_last"] = ctx.get("last")

    um = (route_data.get("user_meaning") or "").strip()
    if not route_data.get("_turn_promotions_done"):
        try:
            from services.query_understanding import score_routing_confidence

            route_conf = score_routing_confidence(
                route_data,
                original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
            from services.chat_flow_telemetry import record_routing_confidence

            record_routing_confidence(route_conf)
        except ImportError:
            route_conf = None
        try:
            from services.chat_flow_telemetry import (
                extract_turn_entities,
                log_routing_decision,
            )

            ent_map = extract_turn_entities(
                route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
            log_routing_decision(
                query=original_msg,
                language=reply_lang or "",
                intent=(route_data.get("intent") or ""),
                confidence=route_conf,
                entities=ent_map,
                selected_route=(route_data.get("data_channel") or ""),
                selected_tool=(
                    route_data.get("route_handler") or route_data.get("data_channel") or ""
                ),
            )
        except ImportError:
            pass
        except Exception as tele_exc:
            log_reasoning(f"Universal brain telemetry skip: {tele_exc}")
    else:
        route_conf = None
    log_reasoning(
        f"Universal brain route: intent={route_data.get('intent')} "
        f"channel={route_data.get('data_channel')} "
        f"olk={route_data.get('order_lookup_kind')} — {um[:90] or '-'}"
    )
    try:
        from services.chat_flow_telemetry import store_brain_route_result

        store_brain_route_result(route_data)
    except ImportError:
        pass
    return route_data


def _finalize_brain_route_decision(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reply_lang: str = "en",
    ctx: Optional[dict] = None,
) -> tuple[AnswerRouteDecision, dict]:
    """Enrich cached brain JSON → locked handler decision (no extra routing LLM)."""
    from services.ai_route_semantics import _normalize_llm_route

    route_data = _normalize_llm_route(dict(route_data))
    route_data.setdefault("_universal_brain_route", True)

    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            promote_account_list_on_route,
        )

        if route_data.get("_universal_brain_route"):
            route_data = promote_account_list_on_route(
                route_data,
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang,
                allow_llm=False,
            )
            comb_ub = f"{original_msg} {msg_en}".strip()
            if (route_data.get("intent") or "").strip().lower() == "refund":
                try:
                    import re

                    from utils.helpers import (
                        _text_is_refund_return_status_lookup,
                        extract_order_id,
                    )

                    if _text_is_refund_return_status_lookup(
                        comb_ub, conv_for_llm
                    ) and (extract_order_id(comb_ub) or re.search(r"\b\d{6,}\b", comb_ub)):
                        route_data["data_channel"] = "live_api"
                        route_data["route_handler"] = "refund_status_api"
                        route_data["order_lookup_kind"] = "refund_status"
                        route_data["needs_order_id"] = True
                        route_data["numeric_context"] = "order_id"
                        route_data["run_catalog_search"] = False
                        log_reasoning(
                            "Universal brain fast path: refund + order id → live API."
                        )
                except ImportError:
                    pass
            fast_locked = account_list_route_is_locked(route_data)
            if not fast_locked:
                intent_ub = (route_data.get("intent") or "").strip().lower()
                ch_ub = (route_data.get("data_channel") or "").strip().lower()
                if intent_ub == "product" and ch_ub == "catalog":
                    route_data["run_catalog_search"] = True
                    route_data["_product_catalog_locked"] = True
                    route_data["needs_order_id"] = False
                    route_data["numeric_context"] = "none"
                    try:
                        from services.ai_route_semantics import (
                            _brain_product_entities_from_route,
                        )

                        ent_ub = _brain_product_entities_from_route(
                            route_data,
                            original_msg=original_msg,
                            msg_en=msg_en,
                        )
                        if ent_ub:
                            route_data["_product_entities"] = ent_ub
                    except ImportError:
                        ent_ub = {}
                    if not (
                        ent_ub.get("pro_id")
                        or ent_ub.get("sku")
                        or ent_ub.get("product_id")
                    ):
                        route_data["_needs_product_nlu_llm"] = True
                    fast_locked = True
                    log_reasoning(
                        "Universal brain fast path: product catalog lock → AI product NLU."
                    )
                else:
                    try:
                        from services.product_catalog_resolver import (
                            apply_product_catalog_to_route,
                            product_catalog_route_is_locked,
                        )

                        route_data = apply_product_catalog_to_route(
                            route_data,
                            original_msg,
                            msg_en,
                            conversation_context=conv_for_llm,
                            reply_lang=reply_lang,
                        )
                        fast_locked = product_catalog_route_is_locked(route_data)
                    except ImportError:
                        pass
            elif (route_data.get("route_handler") or "").strip() == "refund_status_api":
                fast_locked = True
            elif _brain_route_is_fast_lockable(route_data):
                fast_locked = True
                log_reasoning(
                    f"Universal brain fast lock: handler={route_data.get('route_handler')} "
                    f"olk={route_data.get('order_lookup_kind')} channel={route_data.get('data_channel')}."
                )
            if fast_locked:
                product_finalize_locked = False
                try:
                    from services.product_catalog_resolver import (
                        product_catalog_route_is_locked,
                    )

                    product_finalize_locked = product_catalog_route_is_locked(route_data)
                except ImportError:
                    pass
                if not product_finalize_locked:
                    try:
                        from services.ai_route_semantics import enrich_route_from_llm

                        route_data = enrich_route_from_llm(
                            route_data, original_msg, msg_en, conv_for_llm
                        )
                    except ImportError:
                        pass
                try:
                    route_data = apply_ai_route_corrections(
                        route_data, original_msg, msg_en, conv_for_llm
                    )
                except Exception as exc:
                    from services.ai_route_semantics import _normalize_llm_route

                    log_reasoning(
                        f"apply_ai_route_corrections failed (using normalized route): {exc}"
                    )
                    route_data = _normalize_llm_route(route_data)
                decision = (
                    _quick_decision_from_locked_brain_route(
                        route_data, original_msg, msg_en
                    )
                    or _decision_from_ai_route(
                        route_data, original_msg, msg_en, conv_for_llm, ctx=ctx
                    )
                )
                try:
                    from services.chat_flow_telemetry import store_turn_analysis

                    store_turn_analysis(
                        route_data,
                        decision,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conversation_context=conv_for_llm,
                    )
                except ImportError:
                    pass
                return decision, route_data
    except ImportError:
        pass
    except Exception as exc:
        log_reasoning(f"Universal brain fast finalize error (non-fatal): {exc}")

    try:
        from services.catalog_menu_resolver import try_catalog_menu_routing_decision

        menu_dec = try_catalog_menu_routing_decision(
            route_data,
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
            reply_lang=reply_lang,
            ctx=ctx,
        )
        if menu_dec:
            decision, route_data = menu_dec
            try:
                from services.chat_flow_telemetry import (
                    record_route_step,
                    store_turn_analysis,
                )

                record_route_step("catalog_menu_ai_route")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    try:
        from services.product_catalog_resolver import apply_product_catalog_to_route
        from services.account_list_semantics import account_list_route_is_locked

        if not account_list_route_is_locked(route_data):
            route_data = apply_product_catalog_to_route(
                route_data,
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=reply_lang,
            )
    except ImportError:
        pass

    try:
        from services.chitchat_resolver import try_chitchat_routing_decision
        from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path

        if not turn_is_pincode_delivery_fast_path(
            original_msg, msg_en, conv_for_llm, ctx
        ):
            scope_fast = try_chitchat_routing_decision(
                route_data, original_msg, msg_en, conv_for_llm, reply_lang
            )
            if scope_fast:
                decision, route_data = scope_fast
                try:
                    from services.chat_flow_telemetry import (
                        record_route_step,
                        store_turn_analysis,
                    )

                    record_route_step("scope_routing_gate")
                    store_turn_analysis(
                        route_data,
                        decision,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conversation_context=conv_for_llm,
                    )
                except ImportError:
                    pass
                return decision, route_data
    except ImportError:
        pass

    try:
        from services.ai_route_semantics import enrich_route_from_llm

        route_data = enrich_route_from_llm(
            route_data, original_msg, msg_en, conv_for_llm
        )
    except ImportError:
        pass

    try:
        route_data = apply_ai_route_corrections(
            route_data, original_msg, msg_en, conv_for_llm
        )
    except Exception as exc:
        from services.ai_route_semantics import _normalize_llm_route

        log_reasoning(
            f"apply_ai_route_corrections failed (using normalized route): {exc}"
        )
        route_data = _normalize_llm_route(route_data)

    try:
        if not route_data.get("_universal_brain_route"):
            from services.query_understanding import apply_query_understanding

            route_data = apply_query_understanding(
                route_data, original_msg, msg_en, conv_for_llm, reply_lang=reply_lang
            )
    except Exception as exc:
        log_reasoning(f"query_understanding skipped (non-fatal): {exc}")

    log_reasoning(
        f"AI-first route: intent={route_data.get('intent')} "
        f"numeric={route_data.get('numeric_context')} "
        f"continue_topic={route_data.get('continue_previous_topic')} — "
        f"{(route_data.get('reasoning') or '')[:100]}"
    )
    decision = _decision_from_ai_route(
        route_data, original_msg, msg_en, conv_for_llm, ctx=ctx
    )
    try:
        from services.chat_flow_telemetry import store_turn_analysis

        store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
    except ImportError:
        pass
    return decision, route_data


def resolve_answer_route_ai_first(
    original_msg: str,
    msg_en: str,
    retrieval_query: str = "",
    conv_for_llm: str = "",
    reply_lang: str = "en",
    ctx: Optional[dict] = None,
) -> tuple[AnswerRouteDecision, Optional[dict]]:
    """
    1) Optional tiny shortcuts (explicit catalog pro_id)
    2) Groq ai_brain_route with full conversation — primary path for ALL phrasing
    3) Safety corrections (PIN vs order id, wishlist vs orders) — never re-route via keyword lists
    """
    from services.answer_router import resolve_answer_route
    from utils.helpers import (
        _is_conversation_acknowledgment,
        _normalize_order_chat_text,
        extract_product_id,
        _text_is_product_id_lookup_context,
    )

    route_msg, route_msg_en, _mixed = _strip_greeting_filler_from_mixed_turn(
        original_msg, msg_en
    )
    if _mixed:
        original_msg = route_msg
        msg_en = route_msg_en

    ood_obvious = _try_obvious_out_of_domain_route(
        original_msg, msg_en, reply_lang=reply_lang
    )
    if ood_obvious:
        try:
            from services.chat_flow_telemetry import store_brain_route_result

            store_brain_route_result(ood_obvious)
        except ImportError:
            pass
        return _finalize_brain_route_decision(
            ood_obvious,
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            reply_lang=reply_lang,
            ctx=ctx,
        )

    combined = f"{original_msg} {msg_en}".strip()
    comb_norm = _normalize_order_chat_text(combined)
    comb_low = comb_norm.lower()
    strict_llm_failsafe = (os.getenv("STRICT_LLM_FAILSAFE", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    _txn_turn = _transactional_beats_greeting(original_msg, msg_en, conv_for_llm)

    if extract_product_id(comb_low) and _text_is_product_id_lookup_context(comb_low):
        pid = extract_product_id(comb_low)
        return (
            AnswerRouteDecision(
                source="api",
                intent="product",
                handler="catalog_pro_id",
                search_query=f"pro_id {pid}" if pid else "",
                reason="Catalog pro_id lookup (deterministic).",
            ),
            None,
        )

    try:
        from services.chat_flow_telemetry import get_cached_brain_route

        cached_brain = get_cached_brain_route()
        if _brain_route_is_usable(cached_brain):
            log_reasoning(
                "resolve_answer_route: reuse universal brain route — skip pre-route LLM stack."
            )
            return _finalize_brain_route_decision(
                cached_brain,
                original_msg,
                msg_en,
                conv_for_llm=conv_for_llm,
                reply_lang=reply_lang,
                ctx=ctx,
            )
    except ImportError:
        pass

    try:
        from services.account_list_fast_path import try_account_list_fast_route

        account_fast = try_account_list_fast_route(
            original_msg, msg_en, conv_for_llm, reply_lang=reply_lang
        )
        if account_fast:
            decision, route_data = account_fast
            try:
                from services.chat_flow_telemetry import (
                    record_route_step,
                    store_turn_analysis,
                )

                record_route_step("account_list_fast")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    try:
        from services.pincode_delivery_fast_path import try_pincode_delivery_fast_route

        pin_fast = try_pincode_delivery_fast_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx
        )
        if pin_fast:
            decision, route_data = pin_fast
            try:
                from services.chat_flow_telemetry import record_route_step, store_turn_analysis

                record_route_step("pincode_delivery_fast")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    try:
        from services.order_live_intent_fast_path import try_order_live_intent_fast_route

        live_fast = try_order_live_intent_fast_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx, reply_lang=reply_lang
        )
        if live_fast:
            decision, route_data = live_fast
            try:
                from services.chat_flow_telemetry import (
                    record_route_step,
                    store_turn_analysis,
                )

                record_route_step("order_live_intent_fast")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    try:
        from services.order_id_handoff_fast_path import try_order_id_handoff_route

        handoff_route = try_order_id_handoff_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx
        )
        if handoff_route:
            decision, route_data = handoff_route
            try:
                from services.chat_flow_telemetry import (
                    record_route_step,
                    store_turn_analysis,
                )

                record_route_step("order_id_handoff_fast")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    try:
        from services.refund_intent_fast_path import try_refund_intent_fast_route

        refund_fast = try_refund_intent_fast_route(
            original_msg, msg_en, conv_for_llm, ctx=ctx, reply_lang=reply_lang
        )
        if refund_fast:
            decision, route_data = refund_fast
            try:
                from services.chat_flow_telemetry import record_route_step, store_turn_analysis

                record_route_step("refund_intent_fast")
                store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
            except ImportError:
                pass
            return decision, route_data
    except ImportError:
        pass

    conv_fast = None
    if not _txn_turn:
        conv_fast = _try_conversational_fast_path(
            original_msg, msg_en, conv_for_llm, reply_lang
        )
    if conv_fast:
        decision, route_data = conv_fast
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("conversational_fast")
            store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        except ImportError:
            pass
        return decision, route_data

    api_fast = _try_account_list_fast_path(original_msg, msg_en)
    if api_fast:
        decision, route_data = api_fast
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("account_list_fast")
            store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        except ImportError:
            pass
        return decision, route_data

    preflight = None
    if not _txn_turn:
        preflight = _try_policy_kb_preflight(original_msg, msg_en, reply_lang)
    if preflight:
        decision, route_data = preflight
        try:
            from services.query_understanding import apply_query_understanding

            route_data = apply_query_understanding(
                route_data, original_msg, msg_en, conv_for_llm, reply_lang=reply_lang
            )
        except Exception as exc:
            log_reasoning(f"query_understanding skipped on preflight (non-fatal): {exc}")
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("kb_preflight")
            store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        except ImportError:
            pass
        return decision, route_data

    # === ONE semantic evaluation — universal brain (no duplicate ai_brain_route) ===

    route_data = early_universal_brain_route(
        original_msg,
        conv_for_llm,
        reply_lang=reply_lang,
        msg_en=msg_en,
        ctx=ctx,
    )
    if route_data and isinstance(ctx, dict) and ctx.get("last"):
        route_data = dict(route_data)
        route_data["_ctx_last"] = ctx.get("last")

    if route_data:
        return _finalize_brain_route_decision(
            route_data,
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            reply_lang=reply_lang,
            ctx=ctx,
        )

    # === LLM unavailable — deterministic fallbacks only ===
    conv_fallback = _try_conversational_keyword_fallback(
        original_msg, msg_en, conv_for_llm, reply_lang
    )
    if conv_fallback:
        return conv_fallback

    catalog_fallback = _try_catalog_menu_fast_path(original_msg, msg_en, ctx=ctx)
    if catalog_fallback:
        decision, route_data = catalog_fallback
        try:
            from services.chat_flow_telemetry import record_route_step, store_turn_analysis

            record_route_step("catalog_menu_fast_fallback")
            store_turn_analysis(
            route_data,
            decision,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        )
        except ImportError:
            pass
        return decision, route_data

    if _is_conversation_acknowledgment(combined) and conv_for_llm:
        if strict_llm_failsafe:
            from services.conversation_scope import (
                SCOPE_OUT,
                resolve_conversation_scope,
            )

            scope_dec = resolve_conversation_scope(
                original_msg, msg_en, conv_for_llm, reply_lang=reply_lang, ai_route=None
            )
            if scope_dec.scope == SCOPE_OUT:
                log_reasoning(
                    "AI-first: LLM down on ack turn — scope fallback out_of_domain."
                )
                return (
                    AnswerRouteDecision(
                        source="reject",
                        intent="out_of_domain",
                        handler="off_topic",
                        is_welfog_related=False,
                        reason="Off-topic while LLM unavailable.",
                    ),
                    {"llm_unavailable": True, "intent": "out_of_domain", "data_channel": "none"},
                )
            log_reasoning("AI-first: LLM providers unavailable on ack turn — deterministic temporary-load reply.")
            return (
                AnswerRouteDecision(
                    source="reject",
                    intent="general",
                    handler="temporary_load",
                    is_welfog_related=True,
                    reason="LLM providers unavailable; skip heuristic fallback.",
                ),
                {"llm_unavailable": True, "intent": "general", "data_channel": "none"},
            )
        from services.answer_router import try_product_shopping_route_decision

        product_dec = try_product_shopping_route_decision(
            original_msg, msg_en, conv_for_llm=conv_for_llm, ai_route=None
        )
        if product_dec:
            log_reasoning("AI-first: ack + Groq down — heuristic product route.")
            sq = (product_dec.search_query or "").strip()
            return product_dec, {
                "intent": "product",
                "is_welfog_related": True,
                "data_channel": "catalog",
                "search_query": sq,
                "continue_previous_topic": True,
            }

    from utils.helpers import (
        _text_has_delivery_or_order_area_intent,
        _text_has_delivery_serviceability_intent,
        _text_has_pincode_delivery_intent,
        _text_has_refund_or_return_intent,
        _text_is_live_order_lookup_intent,
        _user_denies_pincode_insists_order_id,
        _user_explicitly_asks_payment_status,
        extract_embedded_query_identifiers,
        resolve_live_api_intent_from_conversation,
    )

    from utils.helpers import user_turn_qualifies_for_live_order_api

    ids = extract_embedded_query_identifiers(original_msg, msg_en, conv_for_llm)
    oid_early = (ids.get("order_id") or "").strip()
    if (
        oid_early
        and ids.get("numeric_context") == "order_id"
        and user_turn_qualifies_for_live_order_api(
            original_msg, msg_en, conv_for_llm, ai_route=None
        )
    ):
        live_intent = resolve_live_api_intent_from_conversation(
            conv_for_llm,
            ctx_last=(ctx or {}).get("last") if isinstance(ctx, dict) else None,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if _text_has_refund_or_return_intent(combined):
            live_intent = "refund"
        elif _user_explicitly_asks_payment_status(combined):
            live_intent = "payment"
        log_reasoning(f"AI-first: Order ID {oid_early} in query — live {live_intent} API (not pincode).")
        handler = "order_tracking_api" if live_intent == "order" else "ai_route_and_answer"
        return (
            AnswerRouteDecision(
                source="api",
                intent=live_intent,
                handler=handler,
                is_welfog_related=True,
                reason=f"Live {live_intent} lookup for Order ID {oid_early} already in user message.",
            ),
            {
                "intent": live_intent,
                "needs_order_id": True,
                "numeric_context": "order_id",
                "extracted_pincode": "",
                "data_channel": "live_api",
                "is_welfog_related": True,
                "search_query": "",
            },
        )

    if (ids.get("pincode") or "").strip() and (
        _text_has_pincode_delivery_intent(combined, conv_for_llm)
        or _text_has_delivery_serviceability_intent(combined, conv_for_llm)
        or _text_has_delivery_or_order_area_intent(combined)
    ) and not (
        _text_is_live_order_lookup_intent(combined, conv_for_llm)
        or _user_denies_pincode_insists_order_id(combined)
    ):
        pin = ids["pincode"]
        log_reasoning(f"AI-first: PIN {pin} embedded in query — pincode API.")
        return (
            AnswerRouteDecision(
                source="api",
                intent="pincode_check",
                handler="pincode_delivery_api",
                is_welfog_related=True,
                reason=f"Delivery check for PIN {pin} already in user message.",
            ),
            {
                "intent": "pincode_check",
                "extracted_pincode": pin,
                "numeric_context": "pincode",
                "needs_order_id": False,
                "data_channel": "live_api",
                "is_welfog_related": True,
                "search_query": "",
            },
        )

    from services.answer_router import resolve_structured_procedure_route

    structured_pre = resolve_structured_procedure_route(
        original_msg, msg_en, conversation_context=conv_for_llm
    )
    if structured_pre:
        log_reasoning(f"LLM down — structured procedure fallback: {structured_pre.handler}")
        return structured_pre, None

    try:
        from services.account_list_semantics import turn_requests_purchase_history_in_chat

        if turn_requests_purchase_history_in_chat(
            original_msg, msg_en, conv_for_llm, ai_route=None
        ):
            log_reasoning("AI-first: LLM down — order history list → purchase-history API.")
            return (
                AnswerRouteDecision(
                    source="api",
                    intent="order_history",
                    handler="order_ai_flow",
                    is_welfog_related=True,
                    reason="Purchase history list in chat (heuristic while LLM unavailable).",
                ),
                {
                    "intent": "order_history",
                    "data_channel": "live_api",
                    "needs_order_id": False,
                    "is_welfog_related": True,
                    "search_query": "",
                },
            )
    except ImportError:
        pass

    legacy = resolve_answer_route(
        original_msg, msg_en, retrieval_query=retrieval_query, conv_for_llm=conv_for_llm, ctx=ctx
    )
    if legacy and (legacy.intent or legacy.handler):
        log_reasoning(
            f"LLM down — legacy route fallback: intent={legacy.intent} handler={legacy.handler}"
        )
        return legacy, None

    if strict_llm_failsafe:
        from services.conversation_scope import SCOPE_OUT, resolve_conversation_scope

        scope_dec = resolve_conversation_scope(
            original_msg, msg_en, conv_for_llm, reply_lang=reply_lang, ai_route=None
        )
        if scope_dec.scope == SCOPE_OUT:
            log_reasoning(
                "AI-first: LLM unavailable — scope fallback out_of_domain (no temporary-load)."
            )
            return (
                AnswerRouteDecision(
                    source="reject",
                    intent="out_of_domain",
                    handler="off_topic",
                    is_welfog_related=False,
                    reason="Off-topic while LLM unavailable.",
                ),
                {"llm_unavailable": True, "intent": "out_of_domain", "data_channel": "none"},
            )
        log_reasoning("AI-first: LLM providers unavailable — deterministic temporary-load reply.")
        return (
            AnswerRouteDecision(
                source="reject",
                intent="general",
                handler="temporary_load",
                is_welfog_related=True,
                reason="LLM providers unavailable; skip legacy/random fallback.",
            ),
            {"llm_unavailable": True, "intent": "general", "data_channel": "none"},
        )

    from services.answer_router import try_product_shopping_route_decision

    product_dec = try_product_shopping_route_decision(
        original_msg, msg_en, conv_for_llm=conv_for_llm, ai_route=None
    )
    if product_dec:
        log_reasoning("AI-first: Groq unavailable — heuristic product route.")
        sq = (product_dec.search_query or "").strip()
        return product_dec, {
            "intent": "product",
            "is_welfog_related": True,
            "data_channel": "catalog",
            "search_query": sq,
        }
    log_reasoning("AI-first: all routes exhausted — generic legacy router.")
    return legacy, None
