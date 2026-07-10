"""
AI-first wishlist vs purchase-history — any language, unseen wording.

Semantic action: wants_data → live API | wants_steps → KB how-to.
Primary: account-list micro-classifier (meaning, not phrase lists).
Fallback: keyword helpers when LLM is unavailable.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

_FOLLOWUP_RESOLVE_CACHE = threading.local()
_ACTION_RESOLVE_CACHE = threading.local()

ACTION_WANTS_DATA = "wants_data"
ACTION_WANTS_STEPS = "wants_steps"
ACTION_NONE = "none"

from services.ai_route_semantics import coerce_route_str
from utils.reasoning_log import log_reasoning

KIND_NONE = "none"
KIND_WISHLIST_IN_CHAT = "wishlist_in_chat"
KIND_WISHLIST_HOWTO = "wishlist_howto"
KIND_PURCHASE_IN_CHAT = "purchase_history_in_chat"
KIND_PURCHASE_HOWTO = "purchase_history_howto"

_GOAL_BY_KIND = {
    KIND_WISHLIST_IN_CHAT: "wishlist_list",
    KIND_WISHLIST_HOWTO: "wishlist_howto",
    KIND_PURCHASE_IN_CHAT: "order_history_list",
    KIND_PURCHASE_HOWTO: "order_history_howto",
}

_SAVED_LIKED_MEANING = (
    "wishlist",
    "wish list",
    "saved product",
    "saved products",
    "saved item",
    "liked product",
    "liked products",
    "heart icon",
    "hearted",
    "heart ",
    "favourite",
    "favorite",
    "shortlist",
    "bookmark",
    "products they saved",
    "products they liked",
    "items they saved",
    "items they hearted",
    "customer liked",
    "customer saved",
    "customer's liked",
    "customer's saved",
    "show liked",
    "show saved",
    "list liked",
    "list saved",
    "liked items",
    "liked item",
    "saved items",
    "items they liked",
    "show saved products",
    "show liked products",
    "not purchased",
    "not bought",
    "not ordered yet",
)

_PURCHASE_LIST_MEANING = (
    "purchase history",
    "order history",
    "past orders",
    "orders they placed",
    "orders they bought",
    "what they bought",
    "what they ordered",
    "what they purchased",
    "placed orders",
    "previous orders",
    "all their orders",
    "full order list",
    "mangaya",
    "mangaye",
    "order kiye",
    "order kiya",
)


def _combined(original_msg: str, msg_en: str = "") -> str:
    return " ".join(p for p in ((original_msg or "").strip(), (msg_en or "").strip()) if p).strip()


def _meaning_blob(route: dict | None) -> str:
    if not route:
        return ""
    return f" {(route.get('user_meaning') or '').lower()} {(route.get('reasoning') or '').lower()} "


def _meaning_blob_enriched(route: dict | None, msg_en: str = "") -> str:
    """Brain English fields + auto-translated msg_en (never customer-text keyword routing)."""
    blob = _meaning_blob(route)
    en = (msg_en or "").strip().lower()
    um = (route.get("user_meaning") or "").strip().lower() if route else ""
    if en and len(en) >= 3 and en != um:
        blob = f"{blob} {en} "
    return blob


def _wishlist_kind_from_route(out: dict) -> str:
    return (
        KIND_WISHLIST_HOWTO
        if _route_signals_wishlist_howto(out)
        else KIND_WISHLIST_IN_CHAT
    )


def _brain_hard_product_catalog_evidence(route: dict | None) -> bool:
    """
    True only for structured THIS-turn product browse evidence from Brain JSON:
    category_browse or shopping entities (brand / sku / product_id).

    A bare search_query is NOT evidence — Brain often copies feature paraphrases
    ("the products you liked", "have saved") into search_query while correctly
    setting account_list_kind=wishlist_in_chat. Treating search_query as product
    would convert a Feature Request into OpenSearch.
    """
    if not isinstance(route, dict):
        return False
    if (route.get("category_browse") or "").strip() or route.get("category_only_browse"):
        return True
    try:
        from services.ai_route_semantics import (
            _brain_product_entities_from_route,
            _raw_route_product_entities,
        )

        ent = _raw_route_product_entities(route)
        if not ent:
            ent = _brain_product_entities_from_route(route)
        if not isinstance(ent, dict):
            return False
        for k in ("brand", "sku", "product_id", "pro_id"):
            v = ent.get(k)
            if v not in (None, "", [], {}):
                return True
    except ImportError:
        pass
    return False


def _brain_current_turn_is_product_catalog(route: dict | None) -> bool:
    """
    True when Brain JSON locked a real catalog/product browse for THIS turn.
    Requires hard product evidence — never search_query alone.
    """
    if not isinstance(route, dict):
        return False
    intent = (route.get("intent") or "").strip().lower()
    if intent not in ("product", "catalog") and not route.get("run_catalog_search"):
        if not (route.get("category_browse") or "").strip() and not route.get(
            "category_only_browse"
        ):
            return False
    return _brain_hard_product_catalog_evidence(route)


def _brain_misplaced_account_list_catalog_route(route: dict | None) -> bool:
    """
    Brain labeled catalog/product but has no hard product evidence and search_query
    is empty/generic — account-list feature mislabeled as shopping.
    """
    if not isinstance(route, dict):
        return False
    if _brain_hard_product_catalog_evidence(route):
        return False
    intent = (route.get("intent") or "").strip().lower()
    if intent not in ("product", "catalog", "order_history"):
        return False
    sq = (route.get("search_query") or "").strip()
    try:
        from services.ai_route_semantics import is_generic_catalog_search_phrase

        if not sq:
            return True
        return is_generic_catalog_search_phrase(sq)
    except ImportError:
        return not sq


def _norm_account_list_kind(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    aliases = {
        "wishlist_show": KIND_WISHLIST_IN_CHAT,
        "wishlist_list": KIND_WISHLIST_IN_CHAT,
        "wishlist_in_chat": KIND_WISHLIST_IN_CHAT,
        "show_wishlist": KIND_WISHLIST_IN_CHAT,
        "wishlist_how": KIND_WISHLIST_HOWTO,
        "wishlist_howto": KIND_WISHLIST_HOWTO,
        "wishlist_app": KIND_WISHLIST_HOWTO,
        "order_history": KIND_PURCHASE_IN_CHAT,
        "order_history_list": KIND_PURCHASE_IN_CHAT,
        "purchase_history": KIND_PURCHASE_IN_CHAT,
        "purchase_history_in_chat": KIND_PURCHASE_IN_CHAT,
        "purchase_list": KIND_PURCHASE_IN_CHAT,
        "order_history_howto": KIND_PURCHASE_HOWTO,
        "purchase_history_howto": KIND_PURCHASE_HOWTO,
    }
    if s in aliases:
        return aliases[s]
    if s in (
        KIND_NONE,
        KIND_WISHLIST_IN_CHAT,
        KIND_WISHLIST_HOWTO,
        KIND_PURCHASE_IN_CHAT,
        KIND_PURCHASE_HOWTO,
    ):
        return s
    return KIND_NONE


def _kind_from_meaning_blob(blob: str) -> str:
    if not blob.strip():
        return KIND_NONE
    saved = any(m in blob for m in _SAVED_LIKED_MEANING)
    bought = any(m in blob for m in _PURCHASE_LIST_MEANING)
    how = any(
        x in blob
        for x in (
            " how to ",
            " how do ",
            " where to find",
            " where to view",
            " where can i",
            " steps ",
            " in the app",
            " on the app",
            " in app",
            " kaise dekhe",
            " kaha dekhe",
            " kahan dekhe",
            " navigation",
            " open my orders page",
        )
    )
    show_in_chat = any(
        x in blob
        for x in (
            "show in chat",
            "show here",
            "in this chat",
            "here in chat",
            "display in chat",
            "show me my",
            "show their order",
            "show my order",
            "you show",
            "tell me my order",
            "list in chat",
            "wants to see their order",
            "wants to see my order",
            "display their order",
            "display my order",
            "purchase list here",
            "order list here",
            "show the list",
            "show order list",
        )
    )
    if saved and not bought:
        if any(
            x in blob
            for x in (
                "in chat",
                "in this chat",
                "here in chat",
                "show in chat",
                "display in chat",
                "tell me my wishlist",
                "show my wishlist",
                "show their wishlist",
            )
        ):
            return KIND_WISHLIST_IN_CHAT
        return KIND_WISHLIST_HOWTO if how else KIND_WISHLIST_IN_CHAT
    if bought and not saved:
        if show_in_chat:
            return KIND_PURCHASE_IN_CHAT
        return KIND_PURCHASE_HOWTO if how else KIND_PURCHASE_IN_CHAT
    if saved and bought:
        if how and "wishlist" in blob:
            return KIND_WISHLIST_HOWTO
        if how:
            return KIND_PURCHASE_HOWTO
        if "wishlist" in blob and "history" not in blob and "purchase" not in blob:
            return KIND_WISHLIST_IN_CHAT
        if "history" in blob or "purchase" in blob or "ordered" in blob:
            return KIND_PURCHASE_IN_CHAT
    return KIND_NONE


def semantic_goal_from_account_list_kind(kind: str) -> str:
    return _GOAL_BY_KIND.get(_norm_account_list_kind(kind), "")


def turn_requests_purchase_history_in_chat(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> bool:
    """
    User wants their purchase/order list shown IN THIS CHAT (live API).
    Semantic action first — not keyword-only.
    """
    comb = _combined(original_msg, msg_en)
    if not comb.strip():
        return ai_route_requests_order_history_in_chat(ai_route)
    resolved = resolve_account_list_action(
        original_msg, msg_en, conversation_context, ai_route, reply_lang
    )
    if resolved.get("topic") == "order_history":
        return resolved.get("action") == ACTION_WANTS_DATA
    if resolved.get("action") != ACTION_NONE:
        return False
    return ai_route_requests_order_history_in_chat(ai_route)


def ai_route_requests_order_history_in_chat(route: dict | None) -> bool:
    """User wants purchase list HERE in chat — trust Groq JSON (any customer language)."""
    if not route:
        return False
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind == KIND_PURCHASE_IN_CHAT:
        return True
    if kind == KIND_PURCHASE_HOWTO:
        return False
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent == "order_history" and channel == "live_api":
        return True
    rh = (route.get("route_handler") or "").strip().lower()
    if rh == "order_history_howto_kb":
        return False
    if intent == "order_history" and channel == "kb":
        return False
    blob = _meaning_blob(route)
    if blob.strip():
        inferred = _kind_from_meaning_blob(blob)
        if inferred == KIND_PURCHASE_IN_CHAT:
            return True
        if inferred == KIND_PURCHASE_HOWTO:
            return False
    return False


def ai_route_requests_order_history_howto(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """User wants app navigation steps only — not list in chat."""
    resolved = resolve_account_list_action(
        original_msg,
        msg_en,
        conversation_context,
        route,
        reply_lang,
        allow_llm=not (route or {}).get("llm_unavailable"),
    )
    if resolved.get("topic") == "order_history":
        if resolved.get("action") == ACTION_WANTS_STEPS:
            return True
        if resolved.get("action") == ACTION_WANTS_DATA:
            return False
    return _route_signals_order_history_howto(route)


def turn_requests_wishlist_in_chat(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> bool:
    """User wants saved/liked items shown IN THIS CHAT — live wishlist API."""
    comb = _combined(original_msg, msg_en)
    if not comb.strip():
        return ai_route_requests_wishlist_in_chat(ai_route)
    resolved = resolve_account_list_action(
        original_msg, msg_en, conversation_context, ai_route, reply_lang
    )
    if resolved.get("topic") == "wishlist":
        return resolved.get("action") == ACTION_WANTS_DATA
    if resolved.get("action") != ACTION_NONE:
        return False
    return ai_route_requests_wishlist_in_chat(ai_route)


def ai_route_requests_wishlist_in_chat(route: dict | None) -> bool:
    if not route:
        return False
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind == KIND_WISHLIST_IN_CHAT:
        return True
    if kind == KIND_WISHLIST_HOWTO:
        return False
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    rh = (route.get("route_handler") or "").strip().lower()
    if rh == "wishlist_howto_kb":
        return False
    return intent == "wishlist" and channel == "live_api"


def turn_confirms_wishlist_howto(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Message actually asks wishlist app-navigation — not order ID / refund / place-order."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    try:
        from utils.helpers import (
            _text_asks_how_to_view_wishlist,
            message_clarifies_wishlist_not_order_history,
            message_mentions_wishlist_topic,
            message_is_wishlist_like_request,
            _turn_blocks_wishlist_howto_routing,
        )

        if _turn_blocks_wishlist_howto_routing(comb):
            return False
        if message_clarifies_wishlist_not_order_history(comb):
            return True
        if not (message_mentions_wishlist_topic(comb) or message_is_wishlist_like_request(comb)):
            return False
        return _text_asks_how_to_view_wishlist(comb, conversation_context)
    except ImportError:
        return False


def _route_signals_wishlist_howto(route: dict | None) -> bool:
    if not route:
        return False
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind == KIND_WISHLIST_HOWTO:
        return True
    if kind == KIND_WISHLIST_IN_CHAT:
        return False
    rh = (route.get("route_handler") or "").strip().lower()
    if rh == "wishlist_howto_kb":
        return True
    blob = _meaning_blob(route)
    if _kind_from_meaning_blob(blob) == KIND_WISHLIST_HOWTO:
        return True
    return False


def ai_route_requests_wishlist_howto(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """User wants app navigation steps for wishlist — not list in chat."""
    resolved = resolve_account_list_action(
        original_msg,
        msg_en,
        conversation_context,
        route,
        reply_lang,
        allow_llm=not (route or {}).get("llm_unavailable"),
    )
    if resolved.get("topic") == "wishlist":
        if resolved.get("action") == ACTION_WANTS_STEPS:
            return True
        if resolved.get("action") == ACTION_WANTS_DATA:
            return False
    if not _route_signals_wishlist_howto(route):
        return False
    comb = _combined(original_msg, msg_en)
    if comb and not turn_confirms_wishlist_howto(
        original_msg, msg_en, conversation_context
    ):
        if resolved.get("source") != "account_list_llm":
            return False
    return True


def reconcile_wishlist_from_brain_meaning(
    route: dict | None,
    *,
    msg_en: str = "",
) -> dict:
    """
    Trust brain user_meaning / account_list_kind when catalog drifted to product search.
    Uses English brain fields only — not customer-text keyword routing.

    Feature resolution rule:
      Brain account_list_kind / wishlist meaning → Wishlist API
      Hard product evidence (brand/sku/category_browse) → Product Catalog
      Bare search_query paraphrases must NEVER override a Feature Request
    """
    out = dict(route or {})
    blob = _meaning_blob_enriched(out, msg_en)
    meaning_kind = _kind_from_meaning_blob(blob)
    saved = any(m in blob for m in _SAVED_LIKED_MEANING)
    bought = any(m in blob for m in _PURCHASE_LIST_MEANING)

    kind = _norm_account_list_kind(coerce_route_str(out.get("account_list_kind"), KIND_NONE))

    # Brain already selected wishlist feature — keep it unless THIS turn has hard
    # product evidence (brand/sku/category). Never clear on search_query alone.
    if kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
        if (
            _brain_hard_product_catalog_evidence(out)
            and not saved
            and meaning_kind not in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO)
        ):
            out["account_list_kind"] = KIND_NONE
            out.pop("route_handler", None)
            log_reasoning(
                "Account-list: clear wishlist — Brain hard product evidence this turn."
            )
        else:
            return _apply_kind_to_route(out, kind, "brain_account_list_kind")

    if meaning_kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
        return _apply_kind_to_route(out, meaning_kind, "brain_meaning_wishlist")

    intent = (out.get("intent") or "").strip().lower()
    if intent == "wishlist":
        return _apply_kind_to_route(out, _wishlist_kind_from_route(out), "brain_intent_wishlist")

    rh = (out.get("route_handler") or "").strip().lower()
    if rh == "wishlist_api":
        return _apply_kind_to_route(out, KIND_WISHLIST_IN_CHAT, "brain_route_handler_wishlist")

    if intent == "order_history":
        if saved and not bought:
            return _apply_kind_to_route(
                out, _wishlist_kind_from_route(out), "fix_order_history_wishlist_drift"
            )
        if meaning_kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
            return _apply_kind_to_route(out, meaning_kind, "fix_order_history_wishlist_drift")

    ch = (out.get("data_channel") or "").strip().lower()
    if saved and not bought and ch in ("kb", "none", "") and intent in (
        "general",
        "wishlist",
        "",
    ):
        return _apply_kind_to_route(out, _wishlist_kind_from_route(out), "fix_kb_wishlist_drift")

    misplaced_catalog = _brain_misplaced_account_list_catalog_route(out)
    if misplaced_catalog and saved and not bought:
        return _apply_kind_to_route(
            out, _wishlist_kind_from_route(out), "brain_catalog_meta_wishlist"
        )

    if (
        ch == "catalog"
        or out.get("run_catalog_search")
        or intent in ("product", "catalog")
    ):
        if saved and not bought:
            return _apply_kind_to_route(
                out, _wishlist_kind_from_route(out), "fix_catalog_wishlist_drift"
            )
    return out


def _meaning_blob_indicates_purchase_list(blob: str) -> bool:
    if not blob.strip():
        return False
    if any(m in blob for m in _PURCHASE_LIST_MEANING):
        return True
    return bool(re.search(r"order\s+h\w*stor", blob))


def reconcile_purchase_history_from_brain_meaning(
    route: dict | None,
    *,
    msg_en: str = "",
) -> dict:
    """
    Trust brain user_meaning / account_list_kind when catalog drifted to product search.
    Uses English brain fields only — not customer-text keyword routing.
    """
    out = dict(route or {})
    kind = _norm_account_list_kind(coerce_route_str(out.get("account_list_kind"), KIND_NONE))
    if kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
        return out
    if kind in (KIND_PURCHASE_IN_CHAT, KIND_PURCHASE_HOWTO):
        return _apply_kind_to_route(out, kind, "brain_account_list_kind")

    blob = _meaning_blob_enriched(out, msg_en)
    meaning_kind = _kind_from_meaning_blob(blob)
    if meaning_kind in (KIND_PURCHASE_IN_CHAT, KIND_PURCHASE_HOWTO):
        return _apply_kind_to_route(out, meaning_kind, "brain_meaning_purchase")
    if _meaning_blob_indicates_purchase_list(blob) and meaning_kind == KIND_NONE:
        k = (
            KIND_PURCHASE_HOWTO
            if _route_signals_order_history_howto(out)
            else KIND_PURCHASE_IN_CHAT
        )
        return _apply_kind_to_route(out, k, "brain_meaning_purchase_blob")

    intent = (out.get("intent") or "").strip().lower()
    saved = any(m in blob for m in _SAVED_LIKED_MEANING)
    bought = any(m in blob for m in _PURCHASE_LIST_MEANING)
    _purchase_intents = frozenset(
        {
            "order_history",
            "purchases",
            "purchase_list",
            "orders_list",
            "my_purchases",
            "bought_items",
            "past_purchases",
        }
    )
    if intent in _purchase_intents:
        if saved and not bought:
            return out
        if meaning_kind in (KIND_WISHLIST_IN_CHAT, KIND_WISHLIST_HOWTO):
            return out
        k = (
            KIND_PURCHASE_HOWTO
            if _route_signals_order_history_howto(out)
            else KIND_PURCHASE_IN_CHAT
        )
        return _apply_kind_to_route(out, k, "brain_intent_purchase")

    ch = (out.get("data_channel") or "").strip().lower()
    if (
        ch == "catalog"
        or out.get("run_catalog_search")
        or intent in ("product", "catalog")
    ):
        bought = any(m in blob for m in _PURCHASE_LIST_MEANING)
        saved = any(m in blob for m in _SAVED_LIKED_MEANING)
        if bought and not saved:
            k = (
                KIND_PURCHASE_HOWTO
                if _route_signals_order_history_howto(out)
                else KIND_PURCHASE_IN_CHAT
            )
            return _apply_kind_to_route(out, k, "fix_catalog_purchase_drift")
        if bought and saved and meaning_kind in (
            KIND_PURCHASE_IN_CHAT,
            KIND_PURCHASE_HOWTO,
        ):
            return _apply_kind_to_route(out, meaning_kind, "fix_catalog_purchase_drift")
    return out


def reconcile_account_list_from_brain_meaning(
    route: dict | None,
    *,
    msg_en: str = "",
) -> dict:
    """Wishlist then purchase-history reconcile from brain English fields."""
    out = reconcile_wishlist_from_brain_meaning(route, msg_en=msg_en)
    return reconcile_purchase_history_from_brain_meaning(out, msg_en=msg_en)


def account_list_route_is_locked(route: dict | None) -> bool:
    """Groq account_list_kind / how-to handler — must not be overridden by catalog or pincode."""
    if not route:
        return False
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind != KIND_NONE:
        return True
    rh = (route.get("route_handler") or "").strip().lower()
    return rh in ("wishlist_howto_kb", "order_history_howto_kb")


def account_list_action_from_brain_route(route: dict | None) -> dict:
    """Reuse ai_brain_route account_list_kind — no second micro-LLM."""
    empty = {
        "action": ACTION_NONE,
        "topic": "none",
        "kind": KIND_NONE,
        "confidence": 0.0,
        "source": "",
        "user_meaning": "",
    }
    if not isinstance(route, dict):
        return empty
    kind = _norm_account_list_kind(
        coerce_route_str(route.get("account_list_kind"), KIND_NONE)
    )
    if kind == KIND_NONE:
        intent = (route.get("intent") or "").strip().lower()
        _wishlist_intents = frozenset({
            "wishlist",
            "saved_items",
            "saved_items_list",
            "liked_items",
            "saved_list",
            "favorites",
            "favourites",
        })
        _purchase_intents = frozenset({
            "order_history",
            "purchases",
            "purchase_list",
            "orders_list",
            "my_purchases",
            "bought_items",
            "past_purchases",
        })
        if intent in _wishlist_intents:
            kind = KIND_WISHLIST_IN_CHAT
        elif intent in _purchase_intents:
            kind = KIND_PURCHASE_IN_CHAT
    if kind == KIND_NONE:
        return empty
    action, topic = _kind_to_action_topic(kind)
    um = (route.get("user_meaning") or route.get("reasoning") or "").strip()
    return _action_result(kind, 0.92, "ai_brain_route", um or f"Account list — {kind}")


def infer_account_list_semantic_goal_from_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    """Map route JSON → semantic_route_guard goal (wishlist_list, order_history_list, …)."""
    if not route:
        return ""
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind == KIND_NONE:
        kind = _kind_from_meaning_blob(_meaning_blob(route))
    goal = semantic_goal_from_account_list_kind(kind)
    if goal:
        return goal

    comb = _combined(original_msg, msg_en)
    if comb:
        try:
            from utils.helpers import (
                _text_asks_how_to_view_wishlist,
                _text_asks_wishlist,
                message_denies_wishlist_wants_order_history,
                message_is_wishlist_like_request,
                message_is_past_purchase_list_request,
                _text_wants_order_history_list_in_chat,
            )

            if not message_denies_wishlist_wants_order_history(comb):
                if _text_asks_how_to_view_wishlist(comb, conversation_context):
                    return "wishlist_howto"
                if _text_asks_wishlist(comb) or message_is_wishlist_like_request(comb):
                    return "wishlist_list"
            if (
                message_is_past_purchase_list_request(comb)
                or _text_wants_order_history_list_in_chat(comb, conversation_context)
            ):
                return "order_history_list"
        except ImportError:
            pass

    intent = (route.get("intent") or "").strip().lower()
    blob = _meaning_blob(route)
    if intent == "order_history" and _kind_from_meaning_blob(blob) in (
        KIND_WISHLIST_IN_CHAT,
        KIND_WISHLIST_HOWTO,
    ):
        return semantic_goal_from_account_list_kind(_kind_from_meaning_blob(blob))
    if intent == "wishlist" and _kind_from_meaning_blob(blob) in (
        KIND_PURCHASE_IN_CHAT,
        KIND_PURCHASE_HOWTO,
    ):
        return semantic_goal_from_account_list_kind(_kind_from_meaning_blob(blob))
    if intent == "wishlist":
        rh = (route.get("route_handler") or "").strip().lower()
        if rh == "wishlist_howto_kb":
            return "wishlist_howto"
        return "wishlist_list"
    if intent == "order_history":
        rh = (route.get("route_handler") or "").strip().lower()
        if rh == "order_history_howto_kb":
            return "order_history_howto"
        return "order_history_list"
    return ""


def _turn_might_need_account_list_classifier(route: dict, comb: str) -> bool:
    """True when message likely asks for MY saved items vs MY purchases (any wording)."""
    if not (comb or "").strip():
        return False
    intent = (route.get("intent") or "").strip().lower()
    scope = (route.get("conversation_scope") or "").strip().lower()
    if intent in ("out_of_domain", "product") or scope == "out_of_domain":
        return False
    try:
        from utils.helpers import (
            _text_wants_order_history_list_in_chat,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )

        if message_is_wishlist_like_request(comb):
            return True
        if message_is_past_purchase_list_request(comb) or _text_wants_order_history_list_in_chat(
            comb, ""
        ):
            return True
    except ImportError:
        pass
    intent = (route.get("intent") or "").strip().lower()
    if intent in ("wishlist", "order_history"):
        return True
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind != KIND_NONE:
        return False
    blob = _meaning_blob(route)
    if _kind_from_meaning_blob(blob) != KIND_NONE:
        return False
    if any(
        x in blob
        for x in (
            "wishlist",
            "saved",
            "liked",
            "purchase history",
            "order history",
            "past order",
            "my orders",
            "show their",
            "show my",
            "list in chat",
        )
    ):
        return True
    tl = f" {comb.lower()} "
    possessive = bool(
        re.search(r"\b(?:meri|mere|mera|meroi|meral|mujhe|my|apni|apna|apne)\b", tl)
    )
    list_ask = bool(
        re.search(
            r"\b(?:dikha|dikhao|bata|btao|bta|show|list|dekho|dekh|dede|de do)\w*\b",
            tl,
        )
    )
    account_noun = bool(
        re.search(
            r"\b(?:wishlist|wish\s*list|order|orders|purchase|bought|saved|liked|heart)\w*\b",
            tl,
        )
    )
    return possessive and list_ask and account_noun


def ai_classify_account_list_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> Optional[dict]:
    """
    Micro-classifier: saved/liked vs purchased orders (any language).
    Returns {account_list_kind, user_meaning, confidence} or None.
    """
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

    compact_ctx = _compact_conversation_context(conversation_context or "", 1600)
    user_line = _trim_text_mid(comb, 520)

    system_prompt = f"""You disambiguate Welfog account LIST requests for the LATEST user message.

Two different features (infer from MEANING in ANY language — never match fixed phrase lists):
1) SAVED / LIKED / HEART items — products the customer bookmarked on Welfog but did NOT necessarily buy.
2) PURCHASE / ORDER HISTORY — products they already BOUGHT / placed orders for on Welfog.

Also distinguish:
- SHOW IN CHAT: they want the bot to display the list here in this conversation.
- APP HOW-TO: they only ask WHERE/HOW to find it inside the Welfog app/website (steps), not the list in chat.

If the message is NOT about either list (tracking one order, product search, policy, greeting), use account_list_kind=none.

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence — what they want THIS turn",
  "account_list_kind": "none" | "wishlist_in_chat" | "wishlist_howto" | "purchase_history_in_chat" | "purchase_history_howto",
  "confidence": 0.0 to 1.0
}}

Examples by meaning (not exhaustive):
- "things I hearted, show here" → wishlist_in_chat
- "where do I see stuff I liked in the app" → wishlist_howto
- "everything I ever bought on Welfog" → purchase_history_in_chat
- "how to open my old orders page" → purchase_history_howto
- User wants THEIR past orders shown in this chat (any language/script: Tamil, Telugu, Bengali, Hinglish, English, typos) → purchase_history_in_chat
- User asks for order data/list/history to display here → purchase_history_in_chat (NOT product catalog search)
- Amazon/Flipkart account → none (other company — not Welfog)

{language_reply_instruction(rl)}"""

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
            max_tokens=160,
            timeout_sec=5,
            max_attempts=1,
            temperature=0.2,
        )
    except Exception as exc:
        log_reasoning(f"Account-list LLM error (non-fatal): {exc}")
        return None
    if not data:
        return None

    kind = _norm_account_list_kind(coerce_route_str(data.get("account_list_kind"), KIND_NONE))
    conf = float(data.get("confidence") or 0.72)
    um = (data.get("user_meaning") or "").strip()
    log_reasoning(
        f"Account-list LLM: kind={kind} conf={conf:.2f} — {um[:100] or 'no meaning'}"
    )
    return {
        "account_list_kind": kind,
        "user_meaning": um,
        "confidence": conf,
    }


def _empty_account_list_action() -> dict:
    return {
        "action": ACTION_NONE,
        "topic": "none",
        "kind": KIND_NONE,
        "confidence": 0.0,
        "source": "",
        "user_meaning": "",
    }


def _kind_to_action_topic(kind: str) -> tuple[str, str]:
    kind = _norm_account_list_kind(kind)
    if kind == KIND_WISHLIST_IN_CHAT:
        return ACTION_WANTS_DATA, "wishlist"
    if kind == KIND_WISHLIST_HOWTO:
        return ACTION_WANTS_STEPS, "wishlist"
    if kind == KIND_PURCHASE_IN_CHAT:
        return ACTION_WANTS_DATA, "order_history"
    if kind == KIND_PURCHASE_HOWTO:
        return ACTION_WANTS_STEPS, "order_history"
    return ACTION_NONE, "none"


def _action_result(
    kind: str,
    confidence: float,
    source: str,
    user_meaning: str = "",
) -> dict:
    action, topic = _kind_to_action_topic(kind)
    return {
        "action": action,
        "topic": topic,
        "kind": _norm_account_list_kind(kind),
        "confidence": confidence,
        "source": source,
        "user_meaning": user_meaning,
    }


def _keyword_resolve_account_list_action(
    comb: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> dict:
    """Fallback when LLM unavailable — leaf helpers only."""
    empty = _empty_account_list_action()
    if not comb.strip():
        if ai_route:
            kind = _norm_account_list_kind(
                coerce_route_str(ai_route.get("account_list_kind"), KIND_NONE)
            )
            if kind != KIND_NONE:
                return _action_result(kind, 0.7, "ai_route_field", ai_route.get("user_meaning") or "")
            inferred = _kind_from_meaning_blob(_meaning_blob(ai_route))
            if inferred != KIND_NONE:
                return _action_result(
                    inferred, 0.65, "ai_meaning", ai_route.get("user_meaning") or ""
                )
        return empty
    try:
        from utils.helpers import (
            _message_has_app_navigation_intent,
            _text_asks_how_to_view_order_history,
            _text_asks_how_to_view_wishlist,
            _text_asks_wishlist,
            _text_wants_order_history_list_in_chat,
            _user_asks_order_history_navigation_help,
            message_denies_wishlist_wants_order_history,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )

        tl = f" {comb.lower()} "
        if _message_has_app_navigation_intent(comb):
            if message_is_wishlist_like_request(comb) or "wishlist" in tl:
                if not _text_wants_order_history_list_in_chat(comb, conversation_context):
                    return _action_result(KIND_WISHLIST_HOWTO, 0.74, "keyword_howto")
            if any(
                x in tl
                for x in (
                    "order history",
                    "order hist",
                    "purchase history",
                    "past order",
                    "mere order",
                    "mera order",
                    "apne order",
                    "my orders",
                    "orders kaise",
                    "order kaise",
                )
            ):
                return _action_result(KIND_PURCHASE_HOWTO, 0.74, "keyword_howto")

        if message_denies_wishlist_wants_order_history(comb):
            if _user_asks_order_history_navigation_help(comb, conversation_context):
                return _action_result(KIND_PURCHASE_HOWTO, 0.74, "keyword_howto")
            if _text_wants_order_history_list_in_chat(comb, conversation_context):
                return _action_result(KIND_PURCHASE_IN_CHAT, 0.74, "keyword_data")
        if _text_asks_how_to_view_wishlist(comb, conversation_context):
            return _action_result(KIND_WISHLIST_HOWTO, 0.74, "keyword_howto")
        if _user_asks_order_history_navigation_help(
            comb, conversation_context
        ) or _text_asks_how_to_view_order_history(comb, conversation_context):
            return _action_result(KIND_PURCHASE_HOWTO, 0.74, "keyword_howto")
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            return _action_result(KIND_WISHLIST_IN_CHAT, 0.72, "keyword_data")
        if message_is_past_purchase_list_request(comb) or _text_wants_order_history_list_in_chat(
            comb, conversation_context
        ):
            return _action_result(KIND_PURCHASE_IN_CHAT, 0.72, "keyword_data")
    except ImportError:
        pass
    if ai_route:
        kind = _norm_account_list_kind(
            coerce_route_str(ai_route.get("account_list_kind"), KIND_NONE)
        )
        if kind != KIND_NONE:
            return _action_result(kind, 0.68, "ai_route_field", ai_route.get("user_meaning") or "")
        inferred = _kind_from_meaning_blob(_meaning_blob(ai_route))
        if inferred != KIND_NONE:
            return _action_result(inferred, 0.62, "ai_meaning", ai_route.get("user_meaning") or "")
    inferred = _kind_from_meaning_blob(_meaning_blob({"user_meaning": comb}))
    if inferred != KIND_NONE:
        return _action_result(inferred, 0.6, "text_meaning")
    return empty


def resolve_account_list_action(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> dict:
    """
    Semantic action: wants_data (live API) vs wants_steps (KB how-to).
    Any language — AI micro-classifier first; keywords only when LLM is down.
    """
    comb = _combined(original_msg, msg_en)
    cache = getattr(_ACTION_RESOLVE_CACHE, "result", None)
    cache_key = (
        comb,
        (conversation_context or "")[-600:],
        _norm_account_list_kind(
            coerce_route_str((ai_route or {}).get("account_list_kind"), KIND_NONE)
        ),
        allow_llm,
    )
    if cache and cache.get("key") == cache_key:
        return dict(cache.get("value") or _empty_account_list_action())

    def _finish(result: dict) -> dict:
        _ACTION_RESOLVE_CACHE.result = {"key": cache_key, "value": result}
        return result

    kw = _keyword_resolve_account_list_action(comb, conversation_context, ai_route)
    kw_kind = _norm_account_list_kind(kw.get("kind") or KIND_NONE)
    kw_conf = float(kw.get("confidence") or 0.0)
    if kw_kind != KIND_NONE and kw_conf >= 0.72:
        return _finish(kw)

    if ai_route:
        kind = _norm_account_list_kind(
            coerce_route_str(ai_route.get("account_list_kind"), KIND_NONE)
        )
        if kind != KIND_NONE:
            return _finish(
                _action_result(
                    kind,
                    0.88,
                    "ai_route_field",
                    (ai_route.get("user_meaning") or "").strip(),
                )
            )

    intent = ((ai_route or {}).get("intent") or "").strip().lower()
    universal_brain = bool((ai_route or {}).get("_universal_brain_route"))
    topic_turn = intent in ("wishlist", "order_history") or _turn_might_need_account_list_classifier(
        ai_route or {}, comb
    )
    if not topic_turn and not comb.strip():
        return _finish(_empty_account_list_action())

    if universal_brain and kw_kind != KIND_NONE:
        return _finish(kw)

    if allow_llm and comb.strip() and topic_turn and not universal_brain:
        from services.semantic_intent import llm_semantic_route_available

        llm_ok = llm_semantic_route_available(ai_route or {})
        if llm_ok or intent in ("wishlist", "order_history"):
            try:
                from services.turn_intent_coordinator import get_account_list_ai_classification

                classified = get_account_list_ai_classification(
                    original_msg, msg_en, conversation_context, reply_lang
                )
            except ImportError:
                classified = ai_classify_account_list_turn(
                    original_msg, msg_en, conversation_context, reply_lang
                )
            if classified:
                ck = _norm_account_list_kind(classified.get("account_list_kind") or KIND_NONE)
                conf = float(classified.get("confidence") or 0.0)
                min_conf = 0.62 if llm_ok else 0.72
                if ck != KIND_NONE and conf >= min_conf:
                    return _finish(
                        _action_result(
                            ck,
                            conf,
                            "account_list_llm",
                            (classified.get("user_meaning") or "").strip(),
                        )
                    )

    if ai_route:
        inferred = _kind_from_meaning_blob(_meaning_blob(ai_route))
        if inferred != KIND_NONE:
            return _finish(
                _action_result(
                    inferred,
                    0.7,
                    "ai_meaning",
                    (ai_route.get("user_meaning") or "").strip(),
                )
            )

    return _finish(_keyword_resolve_account_list_action(comb, conversation_context, ai_route))


def _route_signals_order_history_howto(route: dict | None) -> bool:
    if not route:
        return False
    kind = _norm_account_list_kind(coerce_route_str(route.get("account_list_kind"), KIND_NONE))
    if kind == KIND_PURCHASE_HOWTO:
        return True
    if kind == KIND_PURCHASE_IN_CHAT:
        return False
    rh = (route.get("route_handler") or "").strip().lower()
    if rh == "order_history_howto_kb":
        return True
    blob = _meaning_blob(route)
    if _kind_from_meaning_blob(blob) == KIND_PURCHASE_HOWTO:
        return True
    return False


def _apply_kind_to_route(out: dict, kind: str, source: str) -> dict:
    kind = _norm_account_list_kind(kind)
    if kind == KIND_NONE:
        out["account_list_kind"] = KIND_NONE
        return out

    out["account_list_kind"] = kind
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["search_query"] = ""
    out["run_catalog_search"] = False
    out.pop("_product_catalog_locked", None)
    out["continue_previous_topic"] = False
    out.pop("_semantic_override", None)

    if kind == KIND_WISHLIST_IN_CHAT:
        out["intent"] = "wishlist"
        out["data_channel"] = "live_api"
        out["answer_strategy"] = "live_api_only"
        out.pop("route_handler", None)
        log_reasoning(f"Account-list ({source}): saved/liked → wishlist API.")
    elif kind == KIND_WISHLIST_HOWTO:
        out["intent"] = "general"
        out["data_channel"] = "kb"
        out["answer_strategy"] = "structured_handler"
        out["route_handler"] = "wishlist_howto_kb"
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "welfog_api_wishlist"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning(f"Account-list ({source}): wishlist how-to KB.")
    elif kind == KIND_PURCHASE_IN_CHAT:
        out["intent"] = "order_history"
        out["data_channel"] = "live_api"
        out["answer_strategy"] = "live_api_only"
        out.pop("route_handler", None)
        log_reasoning(f"Account-list ({source}): purchase list → order_history API.")
    elif kind == KIND_PURCHASE_HOWTO:
        out["intent"] = "general"
        out["data_channel"] = "kb"
        out["answer_strategy"] = "structured_handler"
        out["route_handler"] = "order_history_howto_kb"
        keys = list(out.get("kb_keys") or [])
        for k in ("faqs", "welfog_api_order_history"):
            if k not in keys:
                keys.append(k)
        out["kb_keys"] = keys
        log_reasoning(f"Account-list ({source}): order history how-to KB.")
    return out


_IN_CHAT_FOLLOWUP_MARKERS = (
    "tu hi",
    "tu bta",
    "tu dikha",
    "tu bata",
    "yahi pe",
    "yahi pr",
    "yaha pe",
    "yaha pr",
    "yahi chat",
    "chat me",
    "idhar hi",
    "yahi par",
    "khud bta",
    "khud dikha",
    "khud bata",
    "khud se",
    "ek kaam kr",
    "ek kaam kar",
    "bhi tu",
    "bol rha",
    "bol raha",
    "maang rha",
    "mang rha",
    "show here",
    "show in chat",
    "tell me here",
)


def _message_asks_show_in_chat(comb: str) -> bool:
    """Keyword fallback for 'show list here in chat' — use only when LLM is unavailable."""
    if not (comb or "").strip():
        return False
    tl = f" {comb.lower()} "
    if any(m in tl for m in _IN_CHAT_FOLLOWUP_MARKERS):
        return True
    if re.search(r"\bkhud\s+se\b", tl) and re.search(
        r"\b(?:dekh|dikha|bta|bata|dekhu|dikhao)\w*\b", tl
    ):
        return True
    if re.search(r"\b(?:dekhni|dekhna|dekhu)\s+h\b", tl):
        return True
    if re.search(r"\b(?:apne\s+aap|myself|self)\b", tl) and re.search(
        r"\b(?:dekh|see|view|show)\w*\b", tl
    ):
        return True
    return False


def _conversation_recently_showed_account_lists(conversation_context: str) -> dict[str, bool]:
    """True when assistant already rendered wishlist / order-history cards in chat."""
    tail = (conversation_context or "")[-5500:].lower()
    if not tail.strip():
        return {"order_history": False, "wishlist": False}
    oh_markers = (
        "your order history",
        "order id:",
        "order placed",
        "track order",
        "payment: unpaid",
        "payment: paid",
    )
    wl_markers = (
        "your wishlist",
        "out of stock",
        "in stock",
        "wishlist (",
    )
    return {
        "order_history": any(m in tail for m in oh_markers),
        "wishlist": any(m in tail for m in wl_markers),
    }


def _followup_context_active(conversation_context: str, ctx: dict | None) -> bool:
    """True when the turn may continue wishlist/order-history list-in-chat (any language)."""
    try:
        from utils.helpers import _conversation_in_pincode_delivery_flow

        if _conversation_in_pincode_delivery_flow(conversation_context):
            tail = (conversation_context or "")[-2800:].lower()
            pin_idx = max(
                tail.rfind("not available to deliver"),
                tail.rfind("delivery available"),
                tail.rfind("service not available"),
                tail.rfind("good news"),
                tail.rfind("check_pincode"),
                tail.rfind("6-digit pin"),
            )
            oh_idx = max(
                tail.rfind("your order history"),
                tail.rfind("order placed"),
            )
            if pin_idx >= 0 and pin_idx > oh_idx:
                return False
    except ImportError:
        pass
    data = (ctx or {}).get("data") or {}
    last_mode = (data.get("topic_mode") or "").strip().lower()
    last_topic = ((ctx or {}).get("last") or "").strip().lower()
    if last_mode == "pincode_check" or last_topic == "pincode":
        return False
    if last_mode in ("wishlist_howto", "order_history_howto"):
        return True
    if last_topic in ("wishlist", "order_history") and (
        _conversation_last_wishlist_howto_only(conversation_context)
        or _conversation_last_order_history_howto_only(conversation_context)
    ):
        return True
    recent = _conversation_recently_showed_account_lists(conversation_context)
    return bool(recent["order_history"] or recent["wishlist"])


def _brain_followup_topic(ai_route: dict | None) -> str:
    """Trust brain account_list_kind / user_meaning when it locks list-in-chat."""
    if not isinstance(ai_route, dict):
        return ""
    kind = _norm_account_list_kind(
        coerce_route_str(ai_route.get("account_list_kind"), KIND_NONE)
    )
    if kind == KIND_WISHLIST_IN_CHAT:
        return "wishlist"
    if kind == KIND_PURCHASE_IN_CHAT:
        return "order_history"
    return ""


def ai_classify_account_list_followup_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
    reply_lang: str = "",
) -> Optional[dict]:
    """
    Micro-classifier for FOLLOW-UP turns: user wants wishlist or order history
    shown IN THIS CHAT after how-to steps or after lists were displayed.
    Any language — infer meaning, not phrase lists.
    Returns {list_topic, user_meaning, confidence} or None.
    """
    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    comb = _combined(original_msg, msg_en)
    if not comb or _followup_blocks_in_chat_list(comb, conversation_context):
        return None
    if not _followup_context_active(conversation_context, ctx):
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    data = (ctx or {}).get("data") or {}
    last_mode = (data.get("topic_mode") or "").strip().lower()
    last_topic = ((ctx or {}).get("last") or "").strip().lower()
    recent = _conversation_recently_showed_account_lists(conversation_context)
    ctx_hint = []
    if last_mode == "wishlist_howto":
        ctx_hint.append("Assistant just explained how to view WISHLIST in the app.")
    elif last_mode == "order_history_howto":
        ctx_hint.append("Assistant just explained how to view ORDER HISTORY in the app.")
    if recent["wishlist"]:
        ctx_hint.append("Wishlist cards were already shown in this chat.")
    if recent["order_history"]:
        ctx_hint.append("Order-history cards were already shown in this chat.")
    if last_topic == "wishlist":
        ctx_hint.append("Last topic was wishlist.")
    elif last_topic == "order_history":
        ctx_hint.append("Last topic was order history.")

    compact_ctx = _compact_conversation_context(conversation_context or "", 2000)
    user_line = _trim_text_mid(comb, 520)
    hint_block = "\n".join(ctx_hint) if ctx_hint else "Recent chat suggests a list follow-up."

    system_prompt = f"""You classify FOLLOW-UP messages on Welfog support chat.

The customer already got how-to steps and/or list cards. Their LATEST message asks to
see their SAVED/LIKED items (wishlist) OR their PURCHASE/ORDER HISTORY list HERE in this
chat — in ANY language or casual phrasing (Hinglish, Tamil, English, etc.).

Do NOT match fixed keyword lists. Infer intent from meaning and conversation context.

NOT a follow-up (list_topic=none):
- Tracking one order by ID, delivery status, refund for one order
- Delivery serviceability / pincode check / another city or area to deliver to
- Product search / catalog / categories / deals
- Policy, company info, greetings, unrelated new topic
- They only repeat how-to questions without wanting the list in chat

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence — what they want THIS turn",
  "list_topic": "none" | "wishlist" | "order_history",
  "confidence": 0.0 to 1.0
}}

wishlist = saved/liked/hearted products on Welfog (not necessarily purchased).
order_history = products they already bought / past orders on Welfog.

{language_reply_instruction(rl)}"""

    user_payload = f"CONTEXT HINT:\n{hint_block}\n\nLATEST USER MESSAGE:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    try:
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=140,
            timeout_sec=10,
            max_attempts=2,
            temperature=0.15,
        )
    except Exception as exc:
        log_reasoning(f"Account-list follow-up LLM error (non-fatal): {exc}")
        return None
    if not data:
        return None

    topic = (data.get("list_topic") or "none").strip().lower()
    if topic not in ("none", "wishlist", "order_history"):
        topic = "none"
    conf = float(data.get("confidence") or 0.72)
    um = (data.get("user_meaning") or "").strip()
    log_reasoning(
        f"Account-list follow-up LLM: topic={topic} conf={conf:.2f} — {um[:100] or 'no meaning'}"
    )
    return {"list_topic": topic, "user_meaning": um, "confidence": conf}


def _followup_topic_from_recent_lists(
    comb: str,
    conversation_context: str,
    ctx: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> str:
    """Pick wishlist vs order_history when user wants self-view after lists were shown."""
    recent = _conversation_recently_showed_account_lists(conversation_context)
    if not (recent["order_history"] or recent["wishlist"]):
        return ""
    if _followup_blocks_in_chat_list(comb, conversation_context):
        return ""

    if allow_llm:
        classified = ai_classify_account_list_followup_turn(
            original_msg or comb,
            msg_en,
            conversation_context,
            ctx=ctx,
            reply_lang=reply_lang,
        )
        if classified:
            topic = (classified.get("list_topic") or "none").strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if topic in ("wishlist", "order_history") and conf >= 0.62:
                return topic

    if not _message_asks_show_in_chat(comb):
        return ""
    tl = f" {comb.lower()} "
    if "wishlist" in tl:
        return "wishlist"
    if any(x in tl for x in ("order history", "order hist", "purchase", "orders")):
        return "order_history"
    last = ((ctx or {}).get("last") or "").strip().lower()
    mode = (((ctx or {}).get("data") or {}).get("topic_mode") or "").strip().lower()
    if last == "wishlist" or mode.startswith("wishlist"):
        return "wishlist"
    if last == "order_history" or mode.startswith("order_history"):
        return "order_history"
    if recent["wishlist"] and not recent["order_history"]:
        return "wishlist"
    if recent["order_history"] and not recent["wishlist"]:
        return "order_history"
    return "order_history"


def _conversation_last_wishlist_howto_only(conversation_context: str) -> bool:
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4500:].lower()
    howto_markers = (
        "wishlist app mein kaise",
        "wishlist app me kaise",
        "how to view your wishlist",
        "wishlist_help",
        "wishlist icon",
        "heart) icon",
    )
    list_markers = ("your wishlist", "wishlist (", "out of stock", "view product")
    how_idx = max(tail.rfind(m) for m in howto_markers)
    list_idx = max(tail.rfind(m) for m in list_markers)
    if how_idx < 0:
        return False
    return list_idx < how_idx


def _conversation_last_order_history_howto_only(conversation_context: str) -> bool:
    if not (conversation_context or "").strip():
        return False
    tail = (conversation_context or "")[-4500:].lower()
    try:
        from utils.helpers import _conversation_bot_showed_order_history_steps

        if not _conversation_bot_showed_order_history_steps(conversation_context):
            return False
    except ImportError:
        pass
    list_markers = ("your order history", "order id:", "order placed", "track order")
    how_idx = max(
        tail.rfind("order history"),
        tail.rfind("app mein order history"),
        tail.rfind("order_history_help"),
    )
    list_idx = max(tail.rfind(m) for m in list_markers)
    if how_idx < 0:
        return False
    return list_idx < how_idx


def _followup_blocks_in_chat_list(comb: str, conversation_context: str = "") -> bool:
    """True when message clearly switched to another Welfog task (not list-in-chat follow-up)."""
    if not (comb or "").strip():
        return True
    try:
        from utils.helpers import (
            _SHOPPING_ACTION_MARKERS,
            _conversation_in_pincode_delivery_flow,
            _message_has_catalog_product_signal,
            _text_has_pincode_delivery_intent,
            _text_is_live_order_lookup_intent,
            _text_is_order_tracking_intent,
            _text_is_pincode_serviceability_question,
            _turn_is_catalog_product_request,
            extract_order_id,
            message_is_conversation_reset_command,
        )
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        if message_is_conversation_reset_command(comb):
            return True
        if turn_requests_delivery_serviceability(
            comb, "", conversation_context, allow_llm=True
        ):
            return True
        if _text_is_pincode_serviceability_question(comb, conversation_context):
            return True
        if _text_has_pincode_delivery_intent(comb, conversation_context):
            return True
        if _conversation_in_pincode_delivery_flow(conversation_context):
            from services.location_delivery_resolver import turn_continues_pincode_area_check

            if turn_continues_pincode_area_check(comb, conversation_context):
                return True
        if extract_order_id(comb, ""):
            return True
        if _text_is_order_tracking_intent(comb) or _text_is_live_order_lookup_intent(comb):
            return True
        # Light shopping probe only — _text_has_product_shopping_intent recurses back here
        # via order-history helpers → detect_account_list_followup_in_chat.
        if _turn_is_catalog_product_request(comb):
            return True
        tl = f" {comb.lower()} "
        if any(m in tl for m in _SHOPPING_ACTION_MARKERS) and _message_has_catalog_product_signal(
            comb
        ):
            return True
    except ImportError:
        pass
    return False


def _llm_followup_list_topic(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    expected: str,
    ctx: dict | None = None,
    reply_lang: str = "",
) -> str:
    """AI-first: wishlist vs purchase list follow-up in chat (any language)."""
    classified = ai_classify_account_list_followup_turn(
        original_msg,
        msg_en,
        conversation_context,
        ctx=ctx,
        reply_lang=reply_lang,
    )
    if classified:
        topic = (classified.get("list_topic") or "none").strip().lower()
        conf = float(classified.get("confidence") or 0.0)
        if topic in ("wishlist", "order_history") and conf >= 0.62:
            if not expected or topic == expected:
                return topic
            if conf >= 0.78:
                return topic

    hint = (
        "The assistant JUST explained how to view WISHLIST (saved/liked) in the app. "
        "The user now wants that list shown HERE in chat — any wording/language."
        if expected == "wishlist"
        else "The assistant JUST explained how to view ORDER HISTORY in the app. "
        "The user now wants their purchase list shown HERE in chat — any wording/language."
    )
    ctx_block = f"{hint}\n\n{(conversation_context or '')[-2000:]}"
    classified = ai_classify_account_list_turn(
        original_msg, msg_en, ctx_block, reply_lang=reply_lang
    )
    if not classified:
        return ""
    kind = _norm_account_list_kind(classified.get("account_list_kind") or KIND_NONE)
    conf = float(classified.get("confidence") or 0.0)
    if expected == "wishlist" and kind == KIND_WISHLIST_IN_CHAT and conf >= 0.62:
        return "wishlist"
    if expected == "order_history" and kind == KIND_PURCHASE_IN_CHAT and conf >= 0.62:
        return "order_history"
    return ""


def _howto_to_in_chat_followup(
    comb: str,
    conversation_context: str,
    last_mode: str,
    original_msg: str,
    msg_en: str,
    ctx: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> str:
    """
    After how-to KB, user continues in any language.
    AI infers meaning first; keywords only when allow_llm=False or LLM unavailable.
    """
    if _followup_blocks_in_chat_list(comb, conversation_context):
        return ""
    if last_mode == "wishlist_howto":
        try:
            from utils.helpers import (
                _text_asks_how_to_view_wishlist,
                message_denies_wishlist_wants_order_history,
            )

            if message_denies_wishlist_wants_order_history(comb):
                return "order_history"
            if _text_asks_how_to_view_wishlist(comb, conversation_context):
                return ""
        except ImportError:
            pass
        if allow_llm:
            llm_topic = _llm_followup_list_topic(
                original_msg,
                msg_en,
                conversation_context,
                "wishlist",
                ctx=ctx,
                reply_lang=reply_lang,
            )
            if llm_topic:
                return llm_topic
        if _message_asks_show_in_chat(comb):
            return "wishlist"
        return ""

    if last_mode == "order_history_howto":
        try:
            from utils.helpers import (
                _text_asks_how_to_view_wishlist,
                message_clarifies_wishlist_not_order_history,
            )

            if message_clarifies_wishlist_not_order_history(comb):
                return "wishlist"
            if _text_asks_how_to_view_wishlist(comb, conversation_context):
                return ""
        except ImportError:
            pass
        if allow_llm:
            llm_topic = _llm_followup_list_topic(
                original_msg,
                msg_en,
                conversation_context,
                "order_history",
                ctx=ctx,
                reply_lang=reply_lang,
            )
            if llm_topic:
                return llm_topic
        if _message_asks_show_in_chat(comb):
            return "order_history"
        return ""
    return ""


def detect_account_list_followup_in_chat(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> str:
    """
    After how-to steps or list display, user wants list IN CHAT — any language.
    Priority: brain route → AI follow-up classifier → keyword fallback (LLM down only).
    """
    comb = _combined(original_msg, msg_en)
    if _followup_blocks_in_chat_list(comb, conversation_context):
        return ""

    cache = getattr(_FOLLOWUP_RESOLVE_CACHE, "topic", None)
    cache_key = (comb, (conversation_context or "")[-800:], str((ctx or {}).get("last")))
    if cache and cache.get("key") == cache_key:
        return cache.get("topic") or ""

    def _finish(topic: str) -> str:
        _FOLLOWUP_RESOLVE_CACHE.topic = {"key": cache_key, "topic": topic}
        return topic

    brain_topic = _brain_followup_topic(ai_route)
    if brain_topic:
        log_reasoning(f"Account-list follow-up from brain: {brain_topic}")
        return _finish(brain_topic)

    data = (ctx or {}).get("data") or {}
    last_mode = (data.get("topic_mode") or "").strip().lower()
    last_topic = ((ctx or {}).get("last") or "").strip().lower()

    locked = _howto_to_in_chat_followup(
        comb,
        conversation_context,
        last_mode,
        original_msg,
        msg_en,
        ctx=ctx,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    if locked:
        return _finish(locked)

    recent_topic = _followup_topic_from_recent_lists(
        comb,
        conversation_context,
        ctx,
        original_msg=original_msg,
        msg_en=msg_en,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    if recent_topic:
        return _finish(recent_topic)

    if allow_llm and _followup_context_active(conversation_context, ctx):
        classified = ai_classify_account_list_followup_turn(
            original_msg,
            msg_en,
            conversation_context,
            ctx=ctx,
            reply_lang=reply_lang,
        )
        if classified:
            topic = (classified.get("list_topic") or "none").strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if topic in ("wishlist", "order_history") and conf >= 0.62:
                return _finish(topic)

    if not _message_asks_show_in_chat(comb):
        return _finish("")

    if last_mode == "wishlist_howto" or (
        last_topic == "wishlist" and _conversation_last_wishlist_howto_only(conversation_context)
    ):
        return _finish("wishlist")
    if last_mode == "order_history_howto" or (
        last_topic == "order_history"
        and _conversation_last_order_history_howto_only(conversation_context)
    ):
        return _finish("order_history")

    if _conversation_last_wishlist_howto_only(conversation_context):
        if not _conversation_last_order_history_howto_only(conversation_context):
            return _finish("wishlist")
    if _conversation_last_order_history_howto_only(conversation_context):
        return _finish("order_history")

    try:
        from utils.helpers import (
            message_clarifies_wishlist_not_order_history,
            message_denies_wishlist_wants_order_history,
            message_is_wishlist_like_request,
            message_mentions_wishlist_topic,
        )

        if message_clarifies_wishlist_not_order_history(comb) or (
            message_mentions_wishlist_topic(comb) and not message_denies_wishlist_wants_order_history(comb)
        ):
            return _finish("wishlist")
        if message_denies_wishlist_wants_order_history(comb):
            return _finish("order_history")
        if message_is_wishlist_like_request(comb):
            return _finish("wishlist")
    except ImportError:
        pass
    return _finish("")


def message_requests_account_list_data(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> bool:
    """True when user wants wishlist or purchase list DATA in chat (not how-to steps)."""
    resolved = resolve_account_list_action(
        original_msg, msg_en, conversation_context, ai_route, reply_lang
    )
    return resolved.get("action") == ACTION_WANTS_DATA


def _route_blocks_account_list_promotion(
    route: dict | None,
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
) -> bool:
    """Delivery/PIN serviceability route must not be promoted to wishlist/order-history."""
    if not isinstance(route, dict):
        return False
    handler = (route.get("route_handler") or "").strip().lower()
    intent = (route.get("intent") or "").strip().lower()
    if handler == "pincode_delivery_api" or intent == "pincode_check":
        return True
    try:
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        if turn_requests_delivery_serviceability(
            original_msg,
            msg_en,
            conversation_context,
            route,
            allow_llm=True,
        ):
            return True
    except ImportError:
        pass
    return False


def promote_account_list_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> dict:
    """
    Align intent with account_list_kind / user_meaning / micro-classifier.
    Call right after main ai_brain_route normalize.
    """
    out = dict(route or {})
    comb = _combined(original_msg, msg_en)
    intent_pre = (out.get("intent") or "").strip().lower()

    out = reconcile_account_list_from_brain_meaning(out)
    if account_list_route_is_locked(out):
        kind = _norm_account_list_kind(coerce_route_str(out.get("account_list_kind"), KIND_NONE))
        if kind != KIND_NONE:
            return _apply_kind_to_route(out, kind, "brain_account_list_reconcile")

    if out.get("_universal_brain_route"):
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            ch_pre = (out.get("data_channel") or "").strip().lower()
            rh_pre = (out.get("route_handler") or "").strip().lower()
            olk = (out.get("order_lookup_kind") or "").strip().lower()
            if account_list_route_is_locked(out):
                kind = _norm_account_list_kind(
                    coerce_route_str(out.get("account_list_kind"), KIND_NONE)
                )
                if kind != KIND_NONE:
                    return _apply_kind_to_route(out, kind, "account_list_locked")
            if product_catalog_route_is_locked(out) or (
                intent_pre == "product" and ch_pre == "catalog"
            ):
                return out
            if rh_pre == "refund_status_api" or olk == "refund_status":
                return out
            if olk in ("invoice", "details", "track", "tracking", "refund_status"):
                return out
            if rh_pre in (
                "order_details_api",
                "order_tracking_api",
                "refund_status_api",
            ):
                return out
            if account_list_route_is_locked(out):
                return out
        except ImportError:
            pass

    if _route_blocks_account_list_promotion(out, original_msg, msg_en, conversation_context):
        log_reasoning("Account-list promotion skipped — pincode/delivery serviceability route locked.")
        return out

    follow_topic = detect_account_list_followup_in_chat(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=out,
        reply_lang=reply_lang,
        allow_llm=allow_llm and not out.get("_universal_brain_route"),
    )
    if follow_topic == "wishlist":
        return _apply_kind_to_route(out, KIND_WISHLIST_IN_CHAT, "howto_followup")
    if follow_topic == "order_history":
        return _apply_kind_to_route(out, KIND_PURCHASE_IN_CHAT, "howto_followup")

    existing = _norm_account_list_kind(coerce_route_str(out.get("account_list_kind"), KIND_NONE))
    try:
        from utils.helpers import (
            _text_wants_order_history_list_in_chat,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )

        keyword_account_turn = (
            message_is_wishlist_like_request(comb)
            or message_is_past_purchase_list_request(comb)
            or _text_wants_order_history_list_in_chat(comb, conversation_context)
        )
    except ImportError:
        keyword_account_turn = False
    account_list_turn = bool(
        comb
        and (
            intent_pre in ("wishlist", "order_history", "general")
            or existing != KIND_NONE
            or keyword_account_turn
            or _turn_might_need_account_list_classifier(out, comb)
        )
    )
    if account_list_turn:
        route_for_resolve = dict(out)
        route_for_resolve["account_list_kind"] = KIND_NONE
        llm_ok = allow_llm and not out.get("llm_unavailable") and not out.get("_universal_brain_route")
        resolved = resolve_account_list_action(
            original_msg,
            msg_en,
            conversation_context,
            route_for_resolve,
            reply_lang,
            allow_llm=llm_ok,
        )
        kind = _norm_account_list_kind(resolved.get("kind") or KIND_NONE)
        if kind != KIND_NONE:
            if resolved.get("user_meaning"):
                out["user_meaning"] = resolved["user_meaning"]
            if resolved.get("source") == "account_list_llm":
                out["_account_list_llm"] = True
            source = resolved.get("source") or "account_list_action"
            log_reasoning(
                f"Account-list action: {resolved.get('action')} "
                f"topic={resolved.get('topic')} kind={kind} ({source})"
            )
            return _apply_kind_to_route(out, kind, source)

    if existing != KIND_NONE:
        source = "ai_route_field"
        if out.get("_account_list_llm"):
            source = "account_list_llm"
        return _apply_kind_to_route(out, existing, source)

    return out
