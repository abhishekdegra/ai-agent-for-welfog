"""
AI route semantics — trust Groq/DeepSeek meaning fields before keyword heuristics.

When llm_unavailable is set, callers may fall back to turn_intent_gate keyword failsafe.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from services.semantic_intent import llm_semantic_route_available
from utils.reasoning_log import log_reasoning


def coerce_route_str(val: Any, default: str = "") -> str:
    """LLMs sometimes return bool/number for string fields — never crash on .strip()."""
    if val is None or val is False:
        return default
    if val is True:
        return default
    if isinstance(val, (int, float)):
        return str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
    if not isinstance(val, str):
        s = str(val).strip()
    else:
        s = val.strip()
    if s.lower() in ("", "none", "null", "n/a", "na"):
        return default
    return s


def coerce_route_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    if isinstance(val, (int, float)):
        return bool(val)
    return default


_OFF_DOMAIN_MEANING_TOPICS = (
    "weather",
    "storm",
    "rain",
    "toofan",
    "tufan",
    "hurricane",
    "cyclone",
    "cricket",
    "football",
    "ipl",
    "recipe",
    "cooking tip",
    "homework",
    "exam",
    "girlfriend",
    "boyfriend",
    "breakup",
    "politics",
    "astrology",
    "kundli",
    "unrelated to welfog",
    "not related to welfog",
    "not about welfog",
    "off-topic",
    "off topic",
    "outside welfog",
    "personal story",
    "life advice",
    "other company",
    "amazon order",
    "flipkart order",
)

_WELFOG_MEANING_MARKERS = (
    "welfog",
    "order",
    "product",
    "delivery",
    "refund",
    "return",
    "track",
    "pincode",
    "wishlist",
    "seller",
    "checkout",
    "deals",
    "category",
)


def _promote_off_domain_from_llm_meaning(out: dict) -> dict:
    """When user_meaning/reasoning describe off-domain topic, fix mis-set intent=general/kb."""
    blob = f"{out.get('user_meaning') or ''} {out.get('reasoning') or ''}".lower()
    if not blob.strip():
        return out
    if any(t in blob for t in _OFF_DOMAIN_MEANING_TOPICS):
        if not any(w in blob for w in _WELFOG_MEANING_MARKERS):
            out["intent"] = "out_of_domain"
            out["data_channel"] = "none"
            out["run_catalog_search"] = False
            out["search_query"] = ""
            out["is_welfog_related"] = False
            out["conversation_scope"] = "out_of_domain"
            if out.get("meta_kind") == "none":
                out["meta_kind"] = "out_of_domain"
            log_reasoning("Route fix: LLM meaning is off-domain — block KB/catalog.")
    return out


_META_KIND_TO_REPLY = {
    "hostile": "bot_insult_calm",
    "insult": "bot_insult_calm",
    "bot_latency": "bot_latency_apology",
    "latency": "bot_latency_apology",
    "topic_denial": "bot_topic_correction",
    "wrong_search_complaint": "bot_search_behavior_help",
    "bot_search_complaint": "bot_search_behavior_help",
    "conversational": "",
    "assistant_intro": "assistant_intro",
    "out_of_domain": "",
}


def strip_markdown_json_fence(text: str) -> str:
    """Remove ```json fences some providers wrap around routing JSON."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _normalize_llm_route(route: dict | None) -> dict:
    """Normalize AI routing JSON fields only (no semantic promotion)."""
    out = dict(route or {})
    if not out.get("extracted_pincode"):
        pin_alias = out.get("pincode") or out.get("pin_code") or out.get("postal_code")
        if pin_alias:
            out["extracted_pincode"] = coerce_route_str(pin_alias, "")
    if not (out.get("search_query") or "").strip():
        sq_alias = out.get("search_terms") or out.get("search_term") or out.get("query")
        if sq_alias:
            out["search_query"] = coerce_route_str(sq_alias, "")
    if not (out.get("extracted_order_id") or "").strip():
        oid_alias = out.get("order_id") or out.get("orderId")
        if oid_alias:
            out["extracted_order_id"] = coerce_route_str(oid_alias, "")
    try:
        from utils.helpers import coerce_valid_order_id

        oid_clean = coerce_valid_order_id(
            out.get("extracted_order_id"),
            context=f"{out.get('user_meaning') or ''} {out.get('reasoning') or ''}",
        )
        out["extracted_order_id"] = oid_clean
        if not oid_clean and coerce_route_bool(out.get("needs_order_id"), False):
            out["numeric_context"] = "order_id"
    except ImportError:
        pass
    out["meta_kind"] = coerce_route_str(out.get("meta_kind"), "none")
    out["intent"] = coerce_route_str(out.get("intent"), "general")
    out["data_channel"] = coerce_route_str(out.get("data_channel"), "")
    out["search_query"] = coerce_route_str(out.get("search_query"), "")
    out["category_browse"] = coerce_route_str(out.get("category_browse"), "")
    out["category_id"] = coerce_route_str(out.get("category_id"), "")
    if "category_only_browse" in out:
        out["category_only_browse"] = coerce_route_bool(
            out.get("category_only_browse"), False
        )
    out["numeric_context"] = coerce_route_str(out.get("numeric_context"), "none")
    out["extracted_pincode"] = coerce_route_str(out.get("extracted_pincode"), "")
    out["reuse_user_value_from_chat"] = coerce_route_str(out.get("reuse_user_value_from_chat"), "")
    out["user_meaning"] = coerce_route_str(out.get("user_meaning"), "")
    out["reasoning"] = coerce_route_str(out.get("reasoning"), "")
    out["scope_reply"] = coerce_route_str(out.get("scope_reply"), "")
    out["conversation_scope"] = coerce_route_str(out.get("conversation_scope"), "")
    out["account_list_kind"] = coerce_route_str(out.get("account_list_kind"), "none").lower()
    out["order_lookup_kind"] = coerce_route_str(out.get("order_lookup_kind"), "none").lower()
    out["field_focus"] = coerce_route_str(out.get("field_focus"), "").lower()
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk not in ("none", "track", "details", "invoice", "refund_status", ""):
        out["order_lookup_kind"] = "none"
    elif olk == "":
        out["order_lookup_kind"] = "none"

    intent_raw = (out.get("intent") or "").strip().lower().replace("-", "_")
    if intent_raw in (
        "invoice",
        "order_invoice",
        "order_bill",
        "order_bill_request",
        "bill_request",
        "bill",
        "receipt",
        "order_receipt",
        "gst_invoice",
    ):
        out["intent"] = "order"
        out["order_lookup_kind"] = "invoice"
        out["data_channel"] = out.get("data_channel") or "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
    elif intent_raw in ("order_track", "order_tracking", "tracking_request"):
        out["intent"] = "order"
        out["order_lookup_kind"] = "track"
        out["data_channel"] = out.get("data_channel") or "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
    elif intent_raw in ("order_details", "order_detail", "order_info"):
        out["intent"] = "order"
        out["order_lookup_kind"] = "details"
        out["data_channel"] = out.get("data_channel") or "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
    if intent_raw == "refund_status":
        out["intent"] = "refund"
        out["order_lookup_kind"] = "refund_status"
        out["data_channel"] = out.get("data_channel") or "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
    out["is_welfog_related"] = coerce_route_bool(out.get("is_welfog_related"), True)
    if "run_catalog_search" in out:
        out["run_catalog_search"] = coerce_route_bool(out.get("run_catalog_search"), False)
    if "needs_order_id" in out:
        out["needs_order_id"] = coerce_route_bool(out.get("needs_order_id"), False)
    if "continue_previous_topic" in out:
        out["continue_previous_topic"] = coerce_route_bool(
            out.get("continue_previous_topic"), False
        )

    mk = (out.get("meta_kind") or "none").strip().lower()
    if mk in ("", "null", "none", "n/a"):
        out["meta_kind"] = "none"
    else:
        out["meta_kind"] = mk

    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    sq = (out.get("search_query") or "").strip()

    alk_now = (out.get("account_list_kind") or "none").strip().lower()
    if intent in (
        "saved_items",
        "saved_items_list",
        "liked_items",
        "saved_list",
        "favorites",
        "favourites",
        "favorite",
        "favourite",
    ):
        out["intent"] = "wishlist"
        if alk_now in ("", "none"):
            out["account_list_kind"] = "wishlist_in_chat"
        out["data_channel"] = channel or "live_api"
        out["run_catalog_search"] = False
        out["search_query"] = ""
        intent = "wishlist"
        channel = (out.get("data_channel") or "").strip().lower()
        sq = ""
    elif intent in (
        "purchases",
        "purchase_list",
        "orders_list",
        "my_purchases",
        "bought_items",
        "past_purchases",
    ):
        out["intent"] = "order_history"
        if alk_now in ("", "none"):
            out["account_list_kind"] = "purchase_history_in_chat"
        out["data_channel"] = channel or "live_api"
        out["run_catalog_search"] = False
        out["search_query"] = ""
        intent = "order_history"
        channel = (out.get("data_channel") or "").strip().lower()
        sq = ""

    if "run_catalog_search" not in out:
        out["run_catalog_search"] = bool(
            intent == "product"
            and channel in ("catalog", "")
            and out["meta_kind"] == "none"
            and sq
            or (
                intent == "product"
                and channel == "catalog"
                and out["meta_kind"] == "none"
            )
        )
    else:
        out["run_catalog_search"] = bool(out.get("run_catalog_search"))

    if out["meta_kind"] != "none":
        out["run_catalog_search"] = False
        if intent == "product":
            out["intent"] = "general"
            out["data_channel"] = "kb"
            out["search_query"] = ""

    # Contradiction guard: if LLM reasoning itself says the query is not related to Welfog,
    # never allow catalog/product execution even when intent text was mis-set to "product".
    meaning_blob = f"{out.get('user_meaning') or ''} {out.get('reasoning') or ''}".lower()
    off_domain_markers = (
        "not related to welfog",
        "not related to welfog's services",
        "not related to welfog services",
        "out_of_domain",
        "out of domain",
        "off-topic",
        "off topic",
    )
    if any(m in meaning_blob for m in off_domain_markers):
        out["intent"] = "out_of_domain"
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["search_query"] = ""
        out["is_welfog_related"] = False
        if out.get("meta_kind") == "none":
            out["meta_kind"] = "out_of_domain"

    if not (out.get("user_meaning") or "").strip():
        out["user_meaning"] = (out.get("reasoning") or "")[:280]

    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()
    sq = (out.get("search_query") or "").strip()
    if intent == "product" and sq and (out.get("meta_kind") or "none") == "none":
        if not channel:
            out["data_channel"] = "catalog"
        if not out.get("run_catalog_search"):
            out["run_catalog_search"] = True

    intent_raw = (out.get("intent") or "").strip().lower().replace("-", "_")
    if intent_raw in ("general_chitchat", "chitchat", "conversational"):
        out["intent"] = "general"
        out["conversation_scope"] = "general_chitchat"
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["needs_order_id"] = False
        if (out.get("meta_kind") or "none") == "none":
            out["meta_kind"] = "conversational"

    if (out.get("meta_kind") or "") in ("conversational", "assistant_intro"):
        out["conversation_scope"] = "general_chitchat"
        out["data_channel"] = "none"
        out["run_catalog_search"] = False
        out["needs_order_id"] = False

    cs = (out.get("conversation_scope") or "").strip().lower().replace("-", "_")
    if cs in ("welfog_support", "general_chitchat", "out_of_domain", "harm_sensitive"):
        out["conversation_scope"] = cs
    elif out.get("intent") == "out_of_domain" or out.get("is_welfog_related") is False:
        out["conversation_scope"] = "out_of_domain"
        out["intent"] = "out_of_domain"
    else:
        out.setdefault("conversation_scope", "welfog_support")

    sr = (out.get("scope_reply") or "").strip()
    if sr:
        out["scope_reply"] = sr

    out = _promote_off_domain_from_llm_meaning(out)
    out = finalize_order_lookup_from_brain_json(out)
    out = apply_order_live_route_handler(out)
    ent = _brain_product_entities_from_route(out)
    if ent:
        out["_product_entities"] = ent
        pn = (ent.get("product_name") or "").strip()
        if (
            pn
            and out.get("run_catalog_search")
            and (out.get("meta_kind") or "none") == "none"
        ):
            out["search_query"] = pn
    return out


_VALID_BRAIN_INTENTS = frozenset({
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
    "general",
    "out_of_domain",
})

# Brain JSON cross-field maps — never customer-message keywords.
_ACCOUNT_LIST_KIND_TO_INTENT: dict[str, str] = {
    "wishlist_in_chat": "wishlist",
    "wishlist_howto": "wishlist",
    "purchase_history_in_chat": "order_history",
    "purchase_history_howto": "order_history",
}

_ROUTE_HANDLER_TO_INTENT: dict[str, str] = {
    "wishlist_api": "wishlist",
    "order_history_api": "order_history",
    "order_tracking_api": "order",
    "order_details_api": "order",
    "refund_status_api": "refund",
    "pincode_delivery_api": "pincode_check",
    "deals_api": "deals",
    "categories_api": "categories",
    "category_feed_api": "category_feed",
}

_INTENT_ALIAS_TO_SCHEMA: dict[str, str] = {
    "catalog_deals": "deals",
    "deals_today": "deals",
    "top_deals": "deals",
    "today_deals": "deals",
    "catalog_categories": "categories",
    "category_list": "categories",
    "category_count": "categories",
    "categories_list": "categories",
}


def _coerce_brain_intent_to_schema(out: dict) -> dict:
    """
    Align brain JSON to routing schema using ONLY fields the LLM returned
    (account_list_kind, route_handler, intent token) — never customer text.
    """
    intent = (out.get("intent") or "").strip().lower()
    if intent in _VALID_BRAIN_INTENTS:
        return out

    alias = _INTENT_ALIAS_TO_SCHEMA.get(intent)
    if alias:
        log_reasoning(
            f"Brain schema reconcile — intent alias {intent!r} → {alias!r}."
        )
        out["intent"] = alias
        if alias in ("deals", "categories", "category_feed"):
            out.setdefault("data_channel", "live_api")
            out["run_catalog_search"] = False
            out["needs_order_id"] = False
            out.setdefault(
                "route_handler",
                "deals_api" if alias == "deals" else "categories_api",
            )
        return out

    if "deal" in intent:
        log_reasoning(f"Brain schema reconcile — compound intent {intent!r} → deals.")
        out["intent"] = "deals"
        out.setdefault("data_channel", "live_api")
        out["run_catalog_search"] = False
        out["needs_order_id"] = False
        out.setdefault("route_handler", "deals_api")
        return out

    if "categor" in intent:
        log_reasoning(
            f"Brain schema reconcile — compound intent {intent!r} → categories."
        )
        out["intent"] = "categories"
        out.setdefault("data_channel", "live_api")
        out["run_catalog_search"] = False
        out["needs_order_id"] = False
        out.setdefault("route_handler", "categories_api")
        return out

    alk = (out.get("account_list_kind") or "none").strip().lower()
    mapped = _ACCOUNT_LIST_KIND_TO_INTENT.get(alk)
    if mapped:
        log_reasoning(
            f"Brain schema reconcile — account_list_kind={alk!r} → intent={mapped!r}."
        )
        out["intent"] = mapped
        if alk in ("wishlist_in_chat", "purchase_history_in_chat"):
            out.setdefault("data_channel", "live_api")
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
        elif alk.endswith("_howto"):
            out.setdefault("data_channel", "kb")
        return out

  # Brain JSON intent typos (wishlisst, whishlist) — token from LLM only, not customer text.
    if intent not in _VALID_BRAIN_INTENTS and intent and intent.startswith("wish"):
        log_reasoning(
            f"Brain schema reconcile — intent typo token {intent!r} → wishlist."
        )
        out["intent"] = "wishlist"
        out.setdefault("data_channel", "live_api")
        out["needs_order_id"] = False
        out["run_catalog_search"] = False
        out.setdefault("account_list_kind", "wishlist_in_chat")
        return out

    rh = (out.get("route_handler") or "").strip().lower()
    mapped = _ROUTE_HANDLER_TO_INTENT.get(rh)
    if mapped:
        log_reasoning(
            f"Brain schema reconcile — route_handler={rh!r} → intent={mapped!r}."
        )
        out["intent"] = mapped
        if rh in ("wishlist_api", "order_history_api"):
            out.setdefault("data_channel", "live_api")
            out["needs_order_id"] = False
            out["run_catalog_search"] = False
        return out

    # LLM compound token (e.g. wishlist_view) — coerce root if it is a valid enum.
    if "_" in intent:
        root = intent.split("_", 1)[0].strip().lower()
        if root in _VALID_BRAIN_INTENTS:
            log_reasoning(
                f"Brain schema reconcile — intent token {intent!r} → {root!r}."
            )
            out["intent"] = root
            if root in ("wishlist", "order_history"):
                out.setdefault("data_channel", "live_api")
                out["needs_order_id"] = False
                out["run_catalog_search"] = False
                if root == "wishlist":
                    out.setdefault("account_list_kind", "wishlist_in_chat")
                else:
                    out.setdefault("account_list_kind", "purchase_history_in_chat")
            return out

    if intent in ("shipping", "kb", "general_chitchat", "seller_login_help"):
        if intent == "shipping":
            out["intent"] = "general"
            out.setdefault("data_channel", "kb")
            if not out.get("kb_keys"):
                out["kb_keys"] = ["shipping"]
        elif intent == "kb":
            out["intent"] = "general"
            out.setdefault("data_channel", "kb")
        elif intent == "general_chitchat":
            out["intent"] = "general"
            out.setdefault("conversation_scope", "general_chitchat")
            out.setdefault("data_channel", "none")
            out["kb_keys"] = []
        elif intent == "seller_login_help":
            out["intent"] = "seller"
            out.setdefault("data_channel", "kb")
            out.setdefault("kb_keys", ["seller", "faqs"])
        log_reasoning(f"Brain schema reconcile — intent token {intent!r} normalized.")
    return out


def repair_brain_json_quality(route: dict | None, user_msg: str = "", msg_en: str = "") -> dict:
    """
    Normalize brain output so routing uses structured JSON only (any customer language).
    Does not re-classify intent from customer text — only fixes invalid/missing enum fields.
    """
    out = dict(route or {})
    out = _coerce_brain_intent_to_schema(out)
    um = (out.get("user_meaning") or "").strip()
    raw = (user_msg or "").strip()
    en = (msg_en or "").strip()
    if um and raw and um.lower() == raw.lower():
        if en and en.lower() != raw.lower() and len(en) >= 3:
            out["user_meaning"] = en[:300]
            log_reasoning(
                "Brain user_meaning echoed customer text — substituted English translation."
            )
        else:
            log_reasoning(
                "Brain user_meaning echoed customer text — backend will use "
                "order_lookup_kind/route_handler/field_focus only."
            )
    elif en and len(en) >= 3 and not um:
        out["user_meaning"] = en[:300]

    nc = coerce_route_str(out.get("numeric_context"), "none")
    if re.fullmatch(r"\d{4,20}", nc):
        if len(nc) >= 7:
            out["numeric_context"] = "order_id"
            out["needs_order_id"] = True
        elif (out.get("intent") or "").strip().lower() == "pincode_check":
            out["numeric_context"] = "pincode"
            out["needs_order_id"] = False

    intent = (out.get("intent") or "").strip().lower()
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if intent in ("order", "refund", "payment") and olk in (
        "track",
        "details",
        "invoice",
        "refund_status",
    ):
        out["data_channel"] = out.get("data_channel") or "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False

    if intent == "pincode_check":
        out["needs_order_id"] = False
        if (out.get("numeric_context") or "none").strip().lower() not in (
            "pincode",
            "order_id",
        ):
            out["numeric_context"] = "pincode"

    ch = (out.get("data_channel") or "").strip().lower()
    if ch in ("order", "order_details", "order_tracking", "tracking"):
        out["data_channel"] = "live_api"

    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            reconcile_account_list_from_brain_meaning,
        )

        out = reconcile_account_list_from_brain_meaning(out, msg_en=msg_en)
        if account_list_route_is_locked(out):
            return out
    except ImportError:
        pass

    intent = (out.get("intent") or "").strip().lower()
    if intent == "product" and (out.get("meta_kind") or "none") == "none":
        try:
            from services.account_list_semantics import account_list_route_is_locked

            if account_list_route_is_locked(out):
                return out
        except ImportError:
            pass
        out["data_channel"] = "catalog"
        out["run_catalog_search"] = True
        ent = _brain_product_entities_from_route(
            out, original_msg=user_msg, msg_en=msg_en
        )
        if ent:
            out["_product_entities"] = ent
        sq = resolve_catalog_search_phrase(
            out,
            original_msg=user_msg,
            msg_en=msg_en,
        )
        if sq:
            out["search_query"] = sq

    return out


# Brain sometimes misplaces sub-intent in conversation_scope — trust AI token, not customer text.
_BRAIN_SCOPE_TO_ORDER_LOOKUP: dict[str, str] = {
    "order_details": "details",
    "order_detail": "details",
    "order_info": "details",
    "order_track": "track",
    "order_tracking": "track",
    "tracking": "track",
    "order_invoice": "invoice",
    "invoice": "invoice",
    "refund_status": "refund_status",
}


def reconcile_order_sub_intent_from_brain_json(route: dict | None) -> dict:
    """
    Map AI routing JSON sub-intent tokens → order_lookup_kind + live_api.
    Uses brain fields only (conversation_scope, scope_reply) — never customer message text.
    """
    out = dict(route or {})
    intent = (out.get("intent") or "").strip().lower()
    if intent not in ("order", "payment", "refund"):
        return out

    ch = (out.get("data_channel") or "").strip().lower()
    if ch in ("order", "order_details", "order_tracking", "tracking"):
        out["data_channel"] = "live_api"
        out["run_catalog_search"] = False

    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk not in ("none", ""):
        return out

    for field in ("conversation_scope", "scope_reply"):
        token = (out.get(field) or "").strip().lower()
        mapped = _BRAIN_SCOPE_TO_ORDER_LOOKUP.get(token)
        if mapped:
            out["order_lookup_kind"] = mapped
            out["data_channel"] = "live_api"
            out["needs_order_id"] = True
            out["numeric_context"] = "order_id"
            out["run_catalog_search"] = False
            out["kb_keys"] = []
            ff = (out.get("field_focus") or "").strip().lower()
            if mapped == "track" and ff not in ("timeline",):
                out["field_focus"] = "timeline"
            elif mapped == "details" and ff not in (
                "payment",
                "product",
                "delivery",
                "summary",
                "status",
            ):
                out["field_focus"] = "summary"
            elif mapped == "invoice":
                out["field_focus"] = "invoice"
            if field == "conversation_scope":
                out["conversation_scope"] = "welfog_support"
            if field == "scope_reply":
                out["scope_reply"] = ""
            log_reasoning(
                f"Brain JSON scope → order_lookup_kind={mapped} (not customer keywords)."
            )
            break

    return out


def infer_order_lookup_from_brain_english_fields(route: dict | None) -> dict:
    """
    When order_lookup_kind is still empty — read AI English user_meaning/reasoning only.
    Never scans the customer's original message.
    """
    out = dict(route or {})
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk not in ("none", ""):
        return out
    intent = (out.get("intent") or "").strip().lower()
    if intent not in ("order", "payment", "refund") or not out.get("needs_order_id"):
        return out

    if intent == "refund" and not ai_route_is_kb_read(out):
        out["order_lookup_kind"] = "refund_status"
        out["data_channel"] = "live_api"
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
        out["kb_keys"] = []
        log_reasoning("Brain intent=refund live → order_lookup_kind=refund_status.")
        return out

    blob = _ai_meaning_blob(out).strip()
    if not blob:
        return out

    def _score(markers: tuple[str, ...]) -> int:
        return sum(1 for m in markers if m in blob)

    inv_score = _score(
        ("invoice", "bill", "receipt", "gst invoice", "tax invoice", "download invoice")
    )
    track_score = _score(
        (
            "track order",
            "order tracking",
            "track the order",
            "track my order",
            "track this order",
            "shipment",
            "courier",
            "eta",
            "when will",
            "when it arrive",
            "when will it arrive",
            "delivery timeline",
            "order timeline",
            "live status",
        )
    )
    if re.search(r"\btrack(?:ing)?\b", blob):
        track_score += 3
    if re.search(r"\bwhen\b.{0,24}\barriv", blob):
        track_score += 2
    detail_score = _score(
        (
            "order details",
            "order detail",
            "complete details",
            "full details",
            "details of",
            "order info",
            "order summary",
            "shipping address",
            "delivery address",
            "payment",
            "payment status",
            "amount",
            "price",
            "how much",
            "address",
            "what did i order",
        )
    )
    refund_score = _score(
        (
            "refund status",
            "return status",
            "refund record",
            "return record",
            "refund progress",
            "money back",
            "refund for order",
            "when will refund",
            "check refund",
            "refund timeline",
        )
    )
    if re.search(r"\brefund\b", blob):
        refund_score += 3
    if re.search(r"\breturn\b", blob) and "return policy" not in blob:
        refund_score += 2

    if refund_score >= max(inv_score, track_score, detail_score) and refund_score > 0:
        out["order_lookup_kind"] = "refund_status"
        out["intent"] = "refund"
        out["data_channel"] = "live_api"
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
        out["kb_keys"] = []
        log_reasoning(
            f"Brain English fields → order_lookup_kind=refund_status "
            f"(scores refund={refund_score})."
        )
        return out
    if inv_score >= max(track_score, detail_score) and inv_score > 0:
        out["order_lookup_kind"] = "invoice"
        out["field_focus"] = "invoice"
    elif track_score >= 3 or (track_score > 0 and track_score >= detail_score):
        out["order_lookup_kind"] = "track"
        out["field_focus"] = "timeline"
    elif detail_score > 0:
        out["order_lookup_kind"] = "details"
        ff = (out.get("field_focus") or "").strip().lower()
        if ff not in ("payment", "product", "delivery", "summary", "status"):
            if "address" in blob:
                out["field_focus"] = "delivery"
            elif "payment" in blob or "amount" in blob or "price" in blob:
                out["field_focus"] = "payment"
            else:
                out["field_focus"] = "summary"
    else:
        ff = (out.get("field_focus") or "").strip().lower()
        if ff == "timeline":
            out["order_lookup_kind"] = "track"
            out["field_focus"] = "timeline"
        elif ff == "invoice":
            out["order_lookup_kind"] = "invoice"
            out["field_focus"] = "invoice"
        elif ff in ("payment", "product", "delivery", "summary", "status"):
            out["order_lookup_kind"] = "details"
        else:
            out["order_lookup_kind"] = "details"
            out["field_focus"] = ff or "summary"

    out["data_channel"] = "live_api"
    out["numeric_context"] = "order_id"
    out["run_catalog_search"] = False
    out["kb_keys"] = []
    log_reasoning(
        f"Brain English fields → order_lookup_kind={out.get('order_lookup_kind')} "
        f"(scores detail={detail_score} track={track_score})."
    )
    return out


def _ai_meaning_blob(route: dict | None) -> str:
    r = route or {}
    return f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".lower()


def _ai_meaning_denies_order_tracking(blob: str) -> bool:
    """LLM explicitly says this is NOT tracking an existing order."""
    return bool(
        re.search(
            r"\b(?:not|no|n't|never|without)\s+"
            r"(?:track(?:ing)?|an?\s+existing\s+order|order\s+track(?:ing)?|"
            r"shipment\s+status|order\s+status)",
            blob,
        )
        or "not order tracking" in blob
        or "not tracking" in blob
    )


def reconcile_category_browse_from_brain_meaning(route: dict | None) -> dict:
    """
    Trust brain user_meaning when customer wants category/department products
    but intent drifted to KB/chitchat/delivery — no customer-text keyword lists.
    """
    out = dict(route or {})
    um = (out.get("user_meaning") or "").strip().lower()
    if not um:
        return out
    browse_markers = (
        "electronics",
        "beauty",
        "fashion",
        "grocery",
        "home kitchen",
        "products in",
        "products from",
        "show products",
        "see products",
        "browse",
        "department",
        "category",
    )
    if not any(m in um for m in browse_markers):
        return out
    if (out.get("data_channel") or "").strip().lower() == "catalog" and (
        out.get("category_browse") or out.get("category_only_browse")
    ):
        return out
    import re

    if re.search(
        r"\b(track|tracking|refund|invoice|pincode|delivery service|deliver to)\b", um
    ):
        return out
    out["intent"] = "product"
    out["data_channel"] = "catalog"
    out["run_catalog_search"] = True
    out["category_only_browse"] = True
    out["needs_order_id"] = False
    out["order_lookup_kind"] = "none"
    out["conversation_scope"] = "welfog_support"
    out["scope_reply"] = ""
    if not (out.get("category_browse") or "").strip():
        for name in (
            "electronics",
            "beauty",
            "men fashion",
            "women fashion",
            "home kitchen",
            "grocery",
        ):
            if name in um:
                out["category_browse"] = name
                break
    if not (out.get("search_query") or "").strip():
        out["search_query"] = ""
    log_reasoning(
        "Brain user_meaning category browse — lock catalog (fix KB/delivery drift)."
    )
    return out


_CATALOG_INTENT_IN_BRAIN_BLOB = (
    "product search",
    "catalog search",
    "browse catalog",
    "browse products",
    "browse product",
    "shop for",
    "looking for",
    "wants to buy",
    "wants to find",
    "wants to see",
    "show products",
    "find products",
    "search for",
    "customer wants",
    "user wants",
    "wants a",
    "needs a",
    "filter by",
    "run_catalog_search",
)

_CATALOG_BLOCK_IN_BRAIN_BLOB = (
    "order id",
    "track order",
    "tracking order",
    "refund",
    "invoice",
    "wishlist",
    "order history",
    "pincode",
    "pin code",
    "delivery status",
    "delivery service",
    "delivery in",
    "deliver to",
    "deliver in",
    "serviceability",
    "does welfog deliver",
    "can welfog deliver",
    "ship to",
    "not related to welfog",
    "out of domain",
    "off-topic",
    "off topic",
    "today's deal",
    "todays deal",
    "top deal",
    "flash sale",
    "deal of the day",
    "how many categor",
    "number of categor",
    "categories on welfog",
    "category list",
    "what does welfog",
    "customer care",
    "contact support",
    "contact welfog",
)

_DEALS_MEANING_MARKERS = (
    "today's deal",
    "todays deal",
    "top deal",
    "today deal",
    "flash sale",
    "deal of the day",
    "show deals",
    "best deals",
    "daily deal",
    "offers and discount",
)

_CATEGORIES_MEANING_MARKERS = (
    "how many categor",
    "number of categor",
    "categories on welfog",
    "welfog categor",
    "list of categor",
    "category list",
    "shop categor",
    "departments on welfog",
    "browse categor",
    "count categor",
)

def brain_turn_indicates_welfog_kb(route: dict | None) -> bool:
    """Brain JSON already chose KB (channel/kb_keys/route_handler) — no keyword lists."""
    return brain_route_prefers_kb_answer(route)


def brain_route_indicates_informational_kb(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Brain JSON or admin KB embeddings lock FAQ — not product catalog.
    Never uses customer-text or user_meaning keyword lists.
    """
    if not isinstance(route, dict):
        return False
    scope = coerce_route_str(route.get("conversation_scope")).lower()
    if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
        return False
    if (route.get("intent") or "").strip().lower() == "out_of_domain":
        return False
    if route.get("is_welfog_related") is False:
        return False
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent == "product" and channel in ("catalog", ""):
        return False
    if route.get("run_catalog_search") is True:
        return False
    if _brain_route_has_shopping_entities(route):
        return False
    entities = route.get("_product_entities") or route.get("product_entities") or {}
    if isinstance(entities, dict) and any(
        v not in (None, "", [], {}) for v in entities.values()
    ):
        return False
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import turn_is_obvious_product_shopping_turn

        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conversation_context
        ):
            return False
    except ImportError:
        pass
    if brain_route_prefers_kb_answer(route):
        return True
    if original_msg or msg_en:
        try:
            from services.knowledge_query_pipeline import (
                kb_embedding_indicates_informational_turn,
            )

            if kb_embedding_indicates_informational_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=route,
            ):
                return True
        except ImportError:
            pass
    return False


def _brain_meaning_blob(route: dict | None) -> str:
    r = route or {}
    return f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".strip().lower()


def brain_turn_indicates_deals(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    if not isinstance(route, dict):
        return False
    intent = (route.get("intent") or "").strip().lower()
    if intent in ("deals",):
        return True
    if "deal" in intent:
        return True
    blob = _brain_meaning_blob(route)
    if any(m in blob for m in _DEALS_MEANING_MARKERS):
        return True
    sq = (route.get("search_query") or "").strip().lower()
    if sq and re.search(r"\bdeals?\b", sq) and not re.search(
        r"\b(?:cover|shirt|shoes|mobile|phone|jeans)\b", sq
    ):
        return True
    if original_msg or msg_en:
        try:
            from services.conversation_followup import is_deals_request_message

            return is_deals_request_message(original_msg, msg_en)
        except ImportError:
            pass
    return False


def brain_turn_indicates_categories(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    if not isinstance(route, dict):
        return False
    intent = (route.get("intent") or "").strip().lower()
    if intent in ("categories", "category_feed"):
        return True
    blob = _brain_meaning_blob(route)
    if any(m in blob for m in _CATEGORIES_MEANING_MARKERS):
        return True
    if original_msg or msg_en:
        try:
            from utils.helpers import message_asks_welfog_categories_list

            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            return message_asks_welfog_categories_list(comb)
        except ImportError:
            pass
    return False


def brain_route_prefers_kb_answer(route: dict | None) -> bool:
    """Brain already chose knowledge-base answer — do not re-run delivery micro-classifiers."""
    if not isinstance(route, dict):
        return False
    ch = coerce_route_str(route.get("data_channel")).lower()
    if ch == "kb":
        return True
    if route.get("kb_keys"):
        return True
    rh = coerce_route_str(route.get("route_handler")).lower()
    if rh in ("kb", "knowledge", "faq", "faqs", "kb_search"):
        return True
    if route.get("_preflight_kb"):
        return True
    return False


def reconcile_deals_from_brain_meaning(route: dict | None) -> dict:
    out = dict(route or {})
    out["intent"] = "deals"
    out["data_channel"] = "live_api"
    out["route_handler"] = "deals_api"
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["search_query"] = ""
    out["conversation_scope"] = "welfog_support"
    out["is_welfog_related"] = True
    out["scope_reply"] = ""
    out["meta_kind"] = "none"
    out.pop("_product_catalog_locked", None)
    out.pop("_product_entities", None)
    return out


def reconcile_categories_from_brain_meaning(route: dict | None) -> dict:
    out = dict(route or {})
    out["intent"] = "categories"
    out["data_channel"] = "live_api"
    out["route_handler"] = "categories_api"
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["search_query"] = ""
    out["conversation_scope"] = "welfog_support"
    out["is_welfog_related"] = True
    out["scope_reply"] = ""
    out["meta_kind"] = "none"
    out.pop("_product_catalog_locked", None)
    return out


def reconcile_welfog_kb_from_brain_meaning(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    out = dict(route or {})
    out["intent"] = "general"
    out["data_channel"] = "kb"
    out["conversation_scope"] = "welfog_support"
    out["is_welfog_related"] = True
    out["run_catalog_search"] = False
    out["needs_order_id"] = False
    out["scope_reply"] = ""
    out["meta_kind"] = "none"
    keys = [k for k in (out.get("kb_keys") or []) if k]
    if not keys:
        try:
            from services.kb_service import resolve_brain_kb_keys

            keys = resolve_brain_kb_keys(
                out,
                original_msg,
                msg_en,
                conversation_context=conversation_context,
                ai_route=out,
            )
        except ImportError:
            keys = []
    out["kb_keys"] = keys
    return out


def _brain_product_name_is_noisy(name: str, route: dict | None = None) -> bool:
    pn = (name or "").strip()
    if not pn:
        return True
    try:
        from services.catalog_spec_semantics import (
            catalog_title_unusable,
            coerce_catalog_entity_map,
        )

        r = route or {}
        pe = coerce_catalog_entity_map(r.get("_product_entities"))
        if not pe:
            pe = coerce_catalog_entity_map(r.get("product_entities"))
        return catalog_title_unusable(
            pn,
            entities={**pe, "product_name": pn},
            ai_route=r,
        )
    except Exception:
        pass
    if len(pn.split()) > 6:
        return True
    try:
        from services.product_filter_pipeline import brain_search_query_is_noisy

        return brain_search_query_is_noisy(pn)
    except ImportError:
        return False


def _clean_brain_product_name(name: str) -> str:
    s = (name or "").strip()
    low = s.lower()
    for prefix in (
        "i want a ",
        "i want ",
        "i need a ",
        "i need ",
        "show me ",
        "find ",
        "search for ",
        "search ",
        "customer wants ",
        "user wants ",
    ):
        if low.startswith(prefix):
            s = s[len(prefix) :].strip()
            low = s.lower()
    s = re.sub(r"\s+for\s+my\s+.+$", "", s, flags=re.I).strip()
    s = re.sub(r"\s+ke\s+liye\s*$", "", s, flags=re.I).strip()
    return s


def _product_noun_from_brain_english(blob: str) -> str:
    """Extract product type from brain English fields only (not customer Hinglish)."""
    low = (blob or "").lower()
    patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"mobile\s+cover|phone\s+cover|back\s+cover", re.I), "mobile cover"),
        (re.compile(r"\bcovers?\b", re.I), "mobile cover"),
        (re.compile(r"\bjeans?\b", re.I), "jeans"),
        (re.compile(r"\b(pajama|pajami|nightwear|night\s*wear)\b", re.I), "pajama"),
        (re.compile(r"water\s+bottle", re.I), "water bottle"),
        (re.compile(r"track\s+pants?|\blower\b", re.I), "track pants"),
        (re.compile(r"\biphones?\b", re.I), "iphone"),
    )
    for pattern, canonical in patterns:
        if pattern.search(low):
            return canonical
    return ""


def _coerce_brain_scalar_field(value) -> str:
    """LLM JSON sometimes returns strings as lists — safe for .strip() callers."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value).strip()
    if isinstance(value, list):
        for item in value:
            s = _coerce_brain_scalar_field(item)
            if s:
                return s
        return ""
    if isinstance(value, dict):
        return ""
    s = str(value).strip()
    if s.lower() in ("null", "none", "n/a"):
        return ""
    return s


def _infer_brand_from_brain_english(
    user_meaning: str,
    reasoning: str = "",
    product_entities: dict | None = None,
) -> str:
    """
    Brand from brain JSON English only — any capitalized name the LLM wrote.
    No fixed brand keyword list; never scans customer Hinglish text.
    """
    pe = product_entities or {}
    brand_raw = _coerce_brain_scalar_field(pe.get("brand"))
    if brand_raw:
        return brand_raw
    blob = f"{user_meaning or ''} {reasoning or ''}".strip()
    if not blob:
        return ""
    for pattern in (
        r"\b(?:for|wants?|needs?|buy|show|find|see)\s+(?:an?\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)\s+(?:mobile|phone)?\s*covers?\b",
        r"\b([A-Z][A-Za-z0-9]+)\s+(?:mobile|phone)\s+cover",
        r"\b([A-Z][A-Za-z0-9]+)\s+covers?\b",
        r"\b(?:brand|from)\s+([A-Z][A-Za-z0-9]+)\b",
    ):
        m = re.search(pattern, blob)
        if m:
            brand = m.group(1).strip()
            if brand.lower() not in (
                "user",
                "customer",
                "friend",
                "sister",
                "brother",
                "son",
                "daughter",
                "mobile",
                "phone",
                "cover",
                "covers",
                "product",
                "products",
                "wants",
                "show",
                "best",
            ):
                try:
                    from services.product_catalog_resolver import sanitize_catalog_brand

                    return sanitize_catalog_brand(
                        brand,
                        product_name="",
                        explicit_from_brain=False,
                    ) or ""
                except ImportError:
                    return brand
    return ""


def _repair_brain_product_entities(
    ent: dict,
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """Normalize brain product_entities — trust LLM JSON; no customer-text brand lists."""
    out = dict(ent or {})
    r = route or {}
    pn = _coerce_brain_scalar_field(out.get("product_name"))
    sq = _coerce_brain_scalar_field(r.get("search_query"))
    um = _coerce_brain_scalar_field(r.get("user_meaning"))
    reasoning = _coerce_brain_scalar_field(r.get("reasoning"))
    raw_pe = r.get("product_entities") if isinstance(r.get("product_entities"), dict) else {}
    brain_blob = f"{um} {reasoning} {sq} {pn}".strip()

    _scalar_entity_keys = (
        "brand",
        "color",
        "size",
        "sku",
        "product_id",
        "price_min",
        "price_max",
        "rating_min",
        "product_intent",
        "model",
        "category",
    )
    for k in (
        "brand",
        "color",
        "size",
        "sku",
        "product_id",
        "price_min",
        "price_max",
        "rating_min",
        "product_intent",
        "related_search_terms",
        "allow_related_fallback",
        "exclude_title_tokens",
        "mandatory_match_tokens",
        "model",
        "category",
    ):
        if not out.get(k) and isinstance(raw_pe, dict) and raw_pe.get(k) not in (None, "", "null", []):
            out[k] = raw_pe[k]

    for k in _scalar_entity_keys:
        if k in out and k not in (
            "price_min",
            "price_max",
            "rating_min",
            "product_id",
        ):
            coerced = _coerce_brain_scalar_field(out.get(k))
            if coerced:
                out[k] = coerced
            else:
                out.pop(k, None)

    if pn and _brain_product_name_is_noisy(pn, r):
        cleaned = _clean_brain_product_name(pn)
        if cleaned and not _brain_product_name_is_noisy(cleaned, r):
            pn = cleaned
        elif sq and not _brain_product_name_is_noisy(sq, r):
            pn = sq
        elif um:
            pn_um = _clean_brain_product_name(um)
            if pn_um and not _brain_product_name_is_noisy(pn_um, r):
                pn = pn_um

    inferred_from_um = _product_noun_from_brain_english(brain_blob)
    if pn and inferred_from_um:
        pn_tokens = set(re.findall(r"[\w]+", pn.lower()))
        inf_tokens = {t for t in re.findall(r"[\w]+", inferred_from_um.lower()) if len(t) >= 3}
        if inf_tokens and not (pn_tokens & inf_tokens):
            pn = inferred_from_um

    color_words = (
        "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
        "grey", "gray", "orange", "brown", "silver", "gold", "navy", "maroon", "beige",
    )
    words = pn.split()
    if len(words) >= 2 and words[0].lower() in color_words and not out.get("color"):
        out["color"] = words[0].title()
        pn = " ".join(words[1:]).strip()

    if not pn or pn.lower() in ("products", "product") or _brain_product_name_is_noisy(pn, r):
        inferred = _product_noun_from_brain_english(brain_blob)
        if inferred:
            pn = inferred

    if pn.lower() in ("iphones", "iphone phones"):
        pn = "iphone"

    if um and pn:
        noun = _product_noun_from_brain_english(f"{um} {reasoning}".strip())
        um_low = um.lower()
        if noun and noun.lower() not in pn.lower():
            if "cover" in um_low or "case" in um_low:
                type_word = "cover" if "cover" in um_low else "case"
                if type_word not in pn.lower():
                    pn = f"{pn} {type_word}".strip()
            elif len(noun.split()) >= 2 and noun.lower() not in pn.lower():
                pn = f"{pn} {noun}".strip()

    if not out.get("brand"):
        brand = _infer_brand_from_brain_english(um, reasoning, raw_pe)
        if brand:
            out["brand"] = brand

    if not out.get("brand") and (original_msg or msg_en):
        try:
            from services.opensearch_products import (
                _extract_brand_literal_from_text,
                _infer_brand_from_message,
            )

            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            inferred = _infer_brand_from_message(comb) or _extract_brand_literal_from_text(
                comb
            )
            if inferred:
                out["brand"] = inferred
        except ImportError:
            pass

    if not out.get("color") and (original_msg or msg_en):
        try:
            from services.opensearch_products import (
                extract_color_and_product_title,
                normalize_color_fuzzy,
            )

            comb = f"{original_msg or ''} {msg_en or ''}".strip()
            col, _ = extract_color_and_product_title(comb)
            if not col:
                col = normalize_color_fuzzy(comb)
            if col:
                out["color"] = col
        except ImportError:
            pass

    if not out.get("product_name") and (original_msg or msg_en):
        comb_pn = f"{original_msg or ''} {msg_en or ''}".strip()
        if re.search(r"\bshorts?\b", comb_pn, re.I):
            out["product_name"] = "shorts"

    try:
        from services.product_catalog_resolver import sanitize_catalog_brand

        clean = sanitize_catalog_brand(
            out.get("brand"),
            product_name=pn or out.get("product_name") or "",
            explicit_from_brain=bool(
                _coerce_brain_scalar_field(
                    raw_pe.get("brand") if isinstance(raw_pe, dict) else ""
                )
            ),
            user_meaning=um,
        )
        if clean:
            out["brand"] = clean
        else:
            out.pop("brand", None)
    except ImportError:
        pass

    if out.get("brand"):
        try:
            from services.opensearch_products import _phone_brand_vocab

            bl = str(out["brand"]).strip().lower()
            if bl in _phone_brand_vocab():
                mmt = list(out.get("mandatory_match_tokens") or [])
                if bl not in [str(t).lower() for t in mmt]:
                    mmt.insert(0, bl)
                out["mandatory_match_tokens"] = mmt[:4]
        except ImportError:
            pass

    if out.get("color"):
        pn = re.sub(r"\bcolor\b", "", pn, flags=re.I).strip()
        pn = " ".join(
            w for w in pn.split() if w.lower() != str(out["color"]).lower()
        ).strip()

    if out.get("price_max") is not None or out.get("price_min") is not None:
        pn = re.sub(r"\b(?:under|below|upto|above|over)\s*\d+.*$", "", pn, flags=re.I).strip()
        pn = re.sub(r"\b\d+\s*(?:rs|rupees?|inr)\b", "", pn, flags=re.I).strip()
        pn = re.sub(r"\s+", " ", pn).strip()

    if out.get("price_max") is None and out.get("price_min") is None and original_msg:
        try:
            from services.opensearch_products import _extract_price_bounds

            pmax, pmin = _extract_price_bounds(original_msg, original_msg.lower())
            if pmax is not None:
                out["price_max"] = pmax
            if pmin is not None:
                out["price_min"] = pmin
        except ImportError:
            pass

    if not out.get("size") and re.search(
        r"\b(kids?|children|son|daughter|little)\b", brain_blob, re.I
    ):
        out["size"] = "kids"

    if not out.get("mandatory_match_tokens"):
        mandatory: list[str] = []
        model = _coerce_brain_scalar_field(
            out.get("model")
            or (raw_pe.get("model") if isinstance(raw_pe, dict) else None)
        )
        if model:
            try:
                from services.product_catalog_resolver import _ai_entity_token_list

                mandatory.extend(_ai_entity_token_list(model)[:3])
            except ImportError:
                mandatory.extend(
                    t
                    for t in re.findall(r"[a-z0-9]{2,}", model.lower())
                    if t
                )[:3]
        if pn:
            try:
                from services.opensearch_products import (
                    _PRODUCT_NOUNS,
                    _title_match_tokens,
                )

                for tok in _title_match_tokens(pn):
                    base = (
                        tok.rstrip("s")
                        if tok.endswith("s") and tok[:-1] in _PRODUCT_NOUNS
                        else tok
                    )
                    if base in _PRODUCT_NOUNS and base not in mandatory:
                        mandatory.append(base)
            except ImportError:
                pass
        brand_m = _coerce_brain_scalar_field(out.get("brand"))
        if brand_m and brand_m.lower() not in [str(m).lower() for m in mandatory]:
            mandatory.insert(0, brand_m.lower())
        if mandatory:
            out["mandatory_match_tokens"] = list(dict.fromkeys(mandatory))[:4]

    if pn:
        out["product_name"] = pn
    return out


def _sanitize_brand_in_entities(ent: dict) -> dict:
    try:
        from services.product_catalog_resolver import sanitize_catalog_brand

        out = dict(ent or {})
        pn = _coerce_brain_scalar_field(out.get("product_name"))
        raw_brand = _coerce_brain_scalar_field(out.get("brand"))
        clean = sanitize_catalog_brand(
            raw_brand,
            product_name=pn,
            explicit_from_brain=bool(raw_brand),
        )
        if clean:
            out["brand"] = clean
        else:
            out.pop("brand", None)
        return out
    except ImportError:
        return dict(ent or {})


def _brain_product_entities_from_route(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """Normalize ai_brain_route product_entities → catalog filter dict."""
    r = route or {}
    raw = r.get("product_entities")
    if isinstance(raw, list):
        raw = next((x for x in raw if isinstance(x, dict)), {})
    elif not isinstance(raw, dict):
        raw = {}
    ent: dict = {}
    _skip_keys = frozenset(
        {
            "related_search_terms",
            "exclude_title_tokens",
            "mandatory_match_tokens",
            "allow_related_fallback",
        }
    )
    _num_keys = frozenset({"price_min", "price_max", "rating_min", "rating_max"})
    for k in (
        "product_name",
        "brand",
        "color",
        "size",
        "sku",
        "product_id",
        "pro_id",
        "price_min",
        "price_max",
        "rating_min",
        "rating_max",
        "category",
        "model",
        "product_intent",
        "related_search_terms",
    ):
        v = raw.get(k)
        if v is None or k in _skip_keys:
            continue
        if k in _num_keys:
            try:
                ent[k] = float(v)
            except (TypeError, ValueError):
                s = _coerce_brain_scalar_field(v)
                if s:
                    try:
                        ent[k] = float(s)
                    except (TypeError, ValueError):
                        pass
            continue
        s = _coerce_brain_scalar_field(v)
        if not s:
            continue
        ent[k] = s

    sq = _coerce_brain_scalar_field(r.get("search_query"))
    if not ent.get("product_name") and sq:
        ent["product_name"] = sq

    pid = ent.get("product_id") or ent.get("pro_id")
    if pid is not None:
        ps = str(pid).strip()
        if ps.isdigit():
            ent["product_id"] = int(ps)
        else:
            ent.pop("product_id", None)
            ent.pop("pro_id", None)

    for pk in ("price_min", "price_max", "rating_min", "rating_max"):
        if ent.get(pk) is not None:
            try:
                ent[pk] = float(ent[pk])
            except (TypeError, ValueError):
                ent.pop(pk, None)

    sku = ent.get("sku")
    if sku is not None:
        ent["sku"] = _coerce_brain_scalar_field(sku)

    if "allow_related_fallback" in raw:
        ent["allow_related_fallback"] = coerce_route_bool(
            raw.get("allow_related_fallback"), True
        )
    for list_key in ("exclude_title_tokens", "mandatory_match_tokens"):
        if list_key in raw and raw.get(list_key) not in (None, "", []):
            try:
                from services.product_catalog_resolver import _ai_entity_token_list

                toks = _ai_entity_token_list(raw.get(list_key))
                if toks:
                    ent[list_key] = toks
            except ImportError:
                vals = raw.get(list_key)
                if isinstance(vals, list):
                    ent[list_key] = [str(x).strip() for x in vals if str(x).strip()]

    if r.get("numeric_context") == "product_id" and not ent.get("product_id"):
        pass

    ent = _repair_brain_product_entities(
        ent, r, original_msg=original_msg, msg_en=msg_en
    )
    ent = _sanitize_brand_in_entities(ent)
    return ent


_GENERIC_CATALOG_PHRASES = frozenset(
    {
        "product",
        "products",
        "item",
        "items",
        "thing",
        "things",
        "show products",
        "browse products",
        "find products",
        "search products",
        "show product",
        "browse product",
        "list products",
        "all products",
        "shopping",
        "shopping options",
        "catalog",
        "options",
    }
)


def is_generic_catalog_search_phrase(phrase: str) -> bool:
    """Reject meta browse phrases that must not become OpenSearch title_query."""
    t = (phrase or "").strip().lower()
    if not t or len(t) < 2:
        return True
    if t in _GENERIC_CATALOG_PHRASES:
        return True
    if t in ("show", "find", "search", "browse", "buy", "shop"):
        return True
    if re.search(r"\bbrowse\s+products?\b", t):
        return True
    if re.search(r"\bto\s+browse\s+products?\b", t):
        return True
    for prefix in (
        "show products",
        "browse products",
        "find products",
        "search products",
        "customer wants ",
        "user wants ",
        "user is asking ",
        "customer is asking ",
        "customer is browsing ",
        "user is browsing ",
        "the customer is ",
        "the user is ",
        "customer wants to browse",
        "user wants to browse",
        "browsing the product catalog",
        "browsing the catalog",
    ):
        if t.startswith(prefix):
            return True
    if "product catalog" in t and len(t.split()) >= 5:
        return True
    if len(t.split()) > 10:
        return True
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(t):
            return True
    except ImportError:
        pass
    return False


def _usable_brain_search_query(sq: str, route: dict | None = None) -> str:
    sq = (sq or "").strip()
    if not sq or is_generic_catalog_search_phrase(sq):
        return ""
    try:
        from services.product_filter_pipeline import brain_search_query_is_noisy

        if brain_search_query_is_noisy(sq):
            return ""
    except ImportError:
        pass
    return sq[:120]


def _usable_brain_category_browse(route: dict | None) -> str:
    """Brain category_browse / category_only_browse — normalized department name."""
    r = route or {}
    cb = (r.get("category_browse") or "").strip()
    if not cb or is_generic_catalog_search_phrase(cb):
        return ""
    return cb[:120]


def _usable_brain_entity_category(ent: dict | None) -> str:
    cat = _coerce_brain_scalar_field((ent or {}).get("category"))
    if not cat or is_generic_catalog_search_phrase(cat):
        return ""
    return cat[:120]


def _brain_provided_catalog_phrase(route: dict | None) -> bool:
    """True when brain JSON already named a catalog target (never fall back to raw user text)."""
    r = route or {}
    if _usable_brain_search_query(r.get("search_query") or "", r):
        return True
    if _usable_brain_category_browse(r):
        return True
    if _usable_brain_entity_category(_raw_route_product_entities(r)):
        return True
    if _usable_catalog_entity_phrase(r.get("user_meaning") or "", r):
        return True
    return False


def _usable_catalog_entity_phrase(phrase: str, route: dict | None = None) -> str:
    cleaned = _clean_brain_product_name((phrase or "").strip())
    if not cleaned or is_generic_catalog_search_phrase(cleaned):
        return ""
    if route and _brain_product_name_is_noisy(cleaned, route):
        return ""
    return cleaned[:120]


def _original_user_product_phrase(
    original_msg: str,
    msg_en: str = "",
    *,
    route: dict | None = None,
) -> str:
    for raw in (original_msg, msg_en):
        raw = (raw or "").strip()
        if not raw or is_generic_catalog_search_phrase(raw):
            continue
        try:
            from services.product_query_understanding import (
                clean_product_part_label,
                polish_search_terms,
            )

            cleaned = clean_product_part_label(
                polish_search_terms(raw, original_msg or raw),
                original_msg or raw,
            )
            cleaned = (cleaned or "").strip()
            if cleaned and not is_generic_catalog_search_phrase(cleaned):
                if not route or not _brain_product_name_is_noisy(cleaned, route):
                    return cleaned[:120]
        except ImportError:
            if not is_generic_catalog_search_phrase(raw):
                return raw[:120]
    return ""


def _raw_route_product_entities(route: dict | None) -> dict:
    r = route or {}
    for key in ("_product_entities", "product_entities"):
        pe = r.get(key)
        if isinstance(pe, dict):
            return dict(pe)
    return {}


def resolve_catalog_search_phrase(
    ai_route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    ai_search_query: str = "",
) -> str:
    """
    Highest-quality catalog product phrase for OpenSearch title_query.

    Priority (first usable wins):
      brain.search_query → category_browse → product_entities.category →
      product_entities.product_name → user_meaning → original user query (last resort only)
    """
    r = ai_route or {}

    sq = _usable_brain_search_query(r.get("search_query") or ai_search_query, r)
    if sq:
        return sq

    cb = _usable_brain_category_browse(r)
    if cb:
        return cb

    for ent in (
        _raw_route_product_entities(r),
        _brain_product_entities_from_route(r, original_msg=original_msg, msg_en=msg_en),
    ):
        cat = _usable_brain_entity_category(ent)
        if cat:
            return cat

    for ent in (
        _raw_route_product_entities(r),
        _brain_product_entities_from_route(r, original_msg=original_msg, msg_en=msg_en),
    ):
        pn = _usable_catalog_entity_phrase(ent.get("product_name") or "", r)
        if pn:
            return pn

    um = _usable_catalog_entity_phrase(r.get("user_meaning") or "", r)
    if um:
        return um

    if _brain_provided_catalog_phrase(r):
        return ""

    return _original_user_product_phrase(original_msg, msg_en, route=r)


def _catalog_search_query_from_brain_route(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """English product-type search terms from ai_brain_route JSON only."""
    return resolve_catalog_search_phrase(
        route,
        original_msg=original_msg,
        msg_en=msg_en,
    )


def _brain_route_has_shopping_entities(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """True when ai_brain_route JSON already extracted product filters (any language in)."""
    ent = _brain_product_entities_from_route(
        route, original_msg=original_msg, msg_en=msg_en
    )
    if not ent:
        return False
    for k in (
        "product_name",
        "brand",
        "model",
        "color",
        "sku",
        "product_id",
        "pro_id",
    ):
        v = ent.get(k)
        if v not in (None, "", [], {}):
            return True
    if ent.get("price_max") is not None or ent.get("price_min") is not None:
        return True
    return False


def brain_route_indicates_product_catalog(route: dict | None) -> bool:
    """
    True when universal brain JSON already understood a catalog turn (any language in → English out).
    Uses intent/channel/search_query/_product_entities/user_meaning — never customer keyword lists.
    """
    if not isinstance(route, dict):
        return False
    if brain_route_prefers_kb_answer(route):
        return False
    if (route.get("data_channel") or "").strip().lower() == "kb":
        return False
    if ai_meaning_describes_delivery_serviceability(route):
        return False
    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(route, allow_llm=True):
            return False
    except ImportError:
        pass
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route):
            return True
    except ImportError:
        pass
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent == "category_browse" and not (route.get("category_browse") or "").strip():
        if not (route.get("search_query") or "").strip() and not route.get("run_catalog_search"):
            return False
    if intent in ("categories", "category_feed"):
        return False
    if intent in ("order", "refund", "order_history", "wishlist", "pincode_check", "delivery"):
        return False
    if channel == "live_api" or route.get("needs_order_id"):
        return False
    if intent == "product" and channel in ("catalog", ""):
        return True
    if route.get("run_catalog_search"):
        return True
    if route.get("category_only_browse") or (route.get("category_browse") or "").strip():
        return True
    if _brain_route_has_shopping_entities(route):
        return True
    entities = route.get("_product_entities") or {}
    if isinstance(entities, dict) and any(v not in (None, "", [], {}) for v in entities.values()):
        return True
    blob = _brain_meaning_blob(route)
    if any(b in blob for b in _CATALOG_BLOCK_IN_BRAIN_BLOB):
        return False
    sq = (route.get("search_query") or "").strip()
    if sq:
        if intent in ("order", "refund", "wishlist", "order_history", "pincode_check"):
            return False
        if intent == "product" or route.get("run_catalog_search") is True:
            return True
        if intent in ("general", "refund", "payment", "seller") and not route.get(
            "run_catalog_search"
        ):
            return False
        if channel == "catalog":
            return True
        return False
    if any(p in blob for p in _CATALOG_INTENT_IN_BRAIN_BLOB):
        return True
    if "product" in blob and any(
        x in blob for x in ("want", "buy", "browse", "shop", "find", "show", "search", "looking")
    ):
        return True
    return False


def reconcile_pincode_delivery_from_brain_meaning(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """
    Lock delivery / pincode serviceability — brain JSON enums first, then delivery AI classifier.
    Never customer-text keyword lists.
    """
    out = dict(route or {})
    out = correct_delivery_vs_tracking_from_ai_meaning(out)

    if ai_meaning_describes_delivery_serviceability(out):
        out["intent"] = "pincode_check"
        out["data_channel"] = "live_api"
        out["route_handler"] = "pincode_delivery_api"
        out["needs_order_id"] = False
        out["numeric_context"] = "pincode"
        out["run_catalog_search"] = False
        out["search_query"] = ""
        out["order_lookup_kind"] = "none"
        out["scope_reply"] = ""
        out["conversation_scope"] = "welfog_support"
        out["meta_kind"] = "none"
        out["is_welfog_related"] = True
        out.pop("_product_catalog_locked", None)
        out["_pincode_delivery_locked"] = True
        log_reasoning(
            "Brain delivery reconcile — pincode_check from ai_brain_route JSON."
        )
        return out

    if brain_route_indicates_product_catalog(out):
        return out

    if original_msg or msg_en:
        try:
            from utils.helpers import turn_is_obvious_product_shopping_turn

            if turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conversation_context
            ):
                return out
        except ImportError:
            pass

    if original_msg or msg_en or conversation_context:
        try:
            from services.location_delivery_resolver import promote_pincode_delivery_on_route

            promoted = promote_pincode_delivery_on_route(
                out,
                original_msg,
                msg_en,
                conversation_context,
            )
            if (promoted.get("intent") or "").strip().lower() == "pincode_check":
                promoted["_pincode_delivery_locked"] = True
                promoted.pop("_product_catalog_locked", None)
                log_reasoning(
                    "Brain delivery reconcile — delivery AI classifier locked pincode_check."
                )
                return promoted
        except ImportError:
            pass

    return out


def reconcile_product_catalog_from_brain_meaning(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """
    Brain sometimes labels shopping as out_of_domain / sets meta_kind that blocks catalog.
    Trust ai_brain_route JSON (user_meaning, search_query, entities) — not customer keywords.
    """
    out = dict(route or {})
    if brain_route_indicates_informational_kb(out):
        return out
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(out):
            out["_product_catalog_locked"] = True
            return out
    except ImportError:
        pass
    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(out):
            return out
    except ImportError:
        pass
    if brain_turn_indicates_deals(out, original_msg=original_msg, msg_en=msg_en):
        return reconcile_deals_from_brain_meaning(out)
    if brain_turn_indicates_categories(
        out, original_msg=original_msg, msg_en=msg_en
    ):
        return reconcile_categories_from_brain_meaning(out)
    if ai_meaning_describes_delivery_serviceability(out):
        return out
    if original_msg or msg_en:
        try:
            from services.location_delivery_resolver import (
                pincode_delivery_route_is_locked,
            )

            if pincode_delivery_route_is_locked(
                out,
                original_msg,
                msg_en,
                "",
                allow_llm=False,
            ):
                return out
        except ImportError:
            pass
    if not brain_route_indicates_product_catalog(out):
        try:
            from utils.helpers import turn_is_obvious_product_shopping_turn

            if not turn_is_obvious_product_shopping_turn(original_msg, msg_en, ""):
                return out
            log_reasoning(
                "Brain shopping reconcile — structural product turn (msg_en), lock catalog."
            )
        except ImportError:
            return out
    ent = _brain_product_entities_from_route(out, original_msg=original_msg, msg_en=msg_en)
    if ent:
        out["_product_entities"] = ent
    sq = _catalog_search_query_from_brain_route(
        out, original_msg=original_msg, msg_en=msg_en
    )
    out["intent"] = "product"
    out["data_channel"] = "catalog"
    out["run_catalog_search"] = True
    out["_product_catalog_locked"] = True
    out["meta_kind"] = "none"
    out["conversation_scope"] = "welfog_support"
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out["scope_reply"] = ""
    out["is_welfog_related"] = True
    if sq:
        out["search_query"] = sq
    log_reasoning(
        f"Brain shopping reconcile — lock product catalog sq={sq!r} (AI JSON, no keyword gate)."
    )
    return out


def reconcile_invoice_from_brain_meaning(route: dict | None) -> dict:
    """
    When ai_brain_route user_meaning (English) says invoice/bill/receipt/challan but
    intent/channel drifted to KB/chitchat — trust user_meaning, not customer keywords.
    """
    out = dict(route or {})
    olk = (out.get("order_lookup_kind") or "").strip().lower()
    focus = (out.get("field_focus") or "").strip().lower()
    intent = (out.get("intent") or "").strip().lower()
    if olk == "invoice" and (out.get("data_channel") or "").strip().lower() == "live_api":
        return out
    if focus == "invoice" or intent in (
        "invoice",
        "order_invoice",
        "order_bill",
        "bill",
        "receipt",
        "gst_invoice",
    ):
        out["intent"] = "order"
        out["data_channel"] = "live_api"
        out["order_lookup_kind"] = "invoice"
        out["route_handler"] = "order_details_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
        out["kb_keys"] = []
        out["field_focus"] = "invoice"
        out["conversation_scope"] = "welfog_support"
        out["scope_reply"] = ""
        out["meta_kind"] = "none"
        log_reasoning(
            "Brain JSON field_focus/intent invoice — lock live invoice API."
        )
        return out
    um = (out.get("user_meaning") or "").strip().lower()
    if not um:
        return out
    markers = (
        "invoice",
        "bill",
        "receipt",
        "gst",
        "chalan",
        "challan",
        "tax invoice",
        "download invoice",
        "involve",
        "invoic",
    )
    if not any(m in um for m in markers):
        return out
    import re

    if not re.search(r"\b\d{4,20}\b", um) and not re.search(
        r"\b\d{4,20}\b", (out.get("reasoning") or "")
    ):
        if "challan" not in um and "chalan" not in um:
            return out
    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["order_lookup_kind"] = "invoice"
    out["route_handler"] = "order_details_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["run_catalog_search"] = False
    out["kb_keys"] = []
    out["conversation_scope"] = "welfog_support"
    out["scope_reply"] = ""
    out["meta_kind"] = "none"
    log_reasoning(
        "Brain user_meaning invoice/challan — lock live invoice API (fix KB/chitchat drift)."
    )
    return out


def finalize_order_lookup_from_brain_json(route: dict | None) -> dict:
    """
    Trust ai_brain_route JSON only — sync structured fields, zero keyword re-classification.
    The routing LLM must set order_lookup_kind / route_handler from customer meaning (any language).
    """
    out = dict(route or {})
    if not llm_semantic_route_available(out):
        return out
    if (out.get("meta_kind") or "none").strip().lower() != "none":
        return out
    if ai_route_is_kb_read(out):
        return out

    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    intent = (out.get("intent") or "").strip().lower()

    handler_to_olk = {
        "order_tracking_api": "track",
        "order_details_api": "details",
        "refund_status_api": "refund_status",
    }
    olk_to_handler = {
        "track": "order_tracking_api",
        "tracking": "order_tracking_api",
        "details": "order_details_api",
        "invoice": "order_details_api",
        "refund_status": "refund_status_api",
    }

    if olk in ("none", "") and rh in handler_to_olk:
        out["order_lookup_kind"] = handler_to_olk[rh]
        olk = out["order_lookup_kind"]
    elif olk in olk_to_handler and rh not in olk_to_handler.values():
        out["route_handler"] = olk_to_handler[olk]
    elif olk in olk_to_handler and rh in olk_to_handler.values():
        expected = olk_to_handler[olk]
        if rh != expected:
            out["route_handler"] = expected

    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk in olk_to_handler and intent in ("order", "refund", "payment"):
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
    return out


def apply_order_live_route_handler(route: dict | None) -> dict:
    """Map order_lookup_kind → existing live API handler (no extra LLM)."""
    out = dict(route or {})
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    handlers = {
        "track": "order_tracking_api",
        "tracking": "order_tracking_api",
        "details": "order_details_api",
        "invoice": "order_details_api",
        "refund_status": "refund_status_api",
    }
    if olk not in handlers:
        return out
    out["route_handler"] = handlers[olk]
    out["data_channel"] = "live_api"
    out["answer_strategy"] = "live_api_only"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    if olk == "refund_status":
        out["intent"] = "refund"
    elif olk in ("track", "tracking", "details", "invoice"):
        if (out.get("intent") or "").strip().lower() not in ("refund", "payment"):
            out["intent"] = "order"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    return out


def lock_order_live_api_from_brain(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """After brain + meaning alignment: lock live API fields — no customer-text promotions."""
    out = dict(route or {})
    out = correct_pincode_vs_order_id_numeric_context(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    out = correct_order_details_vs_tracking_from_ai_meaning(out)
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk in ("none", ""):
        return out
    return apply_order_live_route_handler(out)


_LIVE_GOAL_FROM_SEMANTIC = {
    "order_invoice": "order_invoice",
    "order_details": "order_details",
    "track_single_order": "track",
    "refund_status": "refund_status",
}

LIVE_API_FROM_GOAL = {
    "order_invoice": "order_details_api",
    "order_details": "order_details_api",
    "track": "order_tracking_api",
    "payment": "order_details_api",
    "refund_status": "refund_status_api",
}


def ensure_brain_order_route_locked(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """Idempotent order sub-intent enrichment before goal/tool mapping."""
    out = dict(route or {})
    if out.get("_order_live_route_locked"):
        return out
    if not llm_semantic_route_available(out):
        return out
    out = repair_brain_json_quality(out, original_msg)
    out = infer_order_lookup_from_brain_english_fields(out)
    out = reconcile_order_sub_intent_from_brain_json(out)
    out = reconcile_invoice_from_brain_meaning(out)
    out = finalize_order_lookup_from_brain_json(out)
    out = lock_order_live_api_from_brain(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    out = reconcile_structural_order_sub_intent_from_message(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    out = reconcile_structural_order_sub_intent_from_tracking_message(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    out = lock_order_live_api_from_brain(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    out["_order_live_route_locked"] = True
    return out


def brain_route_to_live_goal(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    """Single source: enriched brain JSON → live dispatch goal string."""
    locked = ensure_brain_order_route_locked(
        route,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    semantic = infer_semantic_goal_from_ai_route(locked)
    if semantic in _LIVE_GOAL_FROM_SEMANTIC:
        return _LIVE_GOAL_FROM_SEMANTIC[semantic]
    olk = (locked.get("order_lookup_kind") or "").strip().lower()
    rh = (locked.get("route_handler") or "").strip().lower()
    focus = (locked.get("field_focus") or "").strip().lower()
    intent = (locked.get("intent") or "").strip().lower()
    if focus == "invoice" or olk == "invoice":
        return "order_invoice"
    if olk in ("details", "order_details") or rh == "order_details_api":
        return "order_details"
    if olk in ("track", "tracking") or rh == "order_tracking_api":
        return "track"
    if olk == "refund_status" or rh == "refund_status_api":
        return "refund_status"
    if intent == "refund" and (locked.get("data_channel") or "").strip().lower() == "live_api":
        return "refund_status"
    if intent in ("invoice", "order_invoice", "order_bill", "bill", "receipt"):
        return "order_invoice"
    if focus in ("payment", "product", "delivery", "summary", "status"):
        return "order_details"
    if focus == "timeline":
        return "track"
    return ""


def brain_route_to_api_tool(
    route: dict | None,
    *,
    live_goal: str = "",
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    goal = (live_goal or "").strip().lower()
    if not goal:
        goal = brain_route_to_live_goal(
            route,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
    return LIVE_API_FROM_GOAL.get(goal, "")


def _structural_refund_goal_from_message(
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """Guardrail — order id + personal refund/return status (not policy KB)."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb or not re.search(r"\b\d{4,20}\b", comb):
        return ""
    try:
        from services.refund_status_semantics import _message_has_refund_topic
        from utils.helpers import _text_is_refund_return_status_lookup

        if _message_has_refund_topic(comb) or _text_is_refund_return_status_lookup(
            comb, ""
        ):
            return "refund_status"
    except ImportError:
        pass
    return ""


def _structural_details_or_invoice_goal_from_message(
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """
    Guardrail only — order id + non-tracking markers (invoice/address/price/details).
    Overrides brain track misroutes; not the primary router when brain is clear.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb or not re.search(r"\b\d{4,20}\b", comb):
        return ""
    if _structural_refund_goal_from_message(original_msg, msg_en):
        return ""
    try:
        from utils.helpers import _leaf_non_tracking_order_id_intent

        if not _leaf_non_tracking_order_id_intent(comb):
            return ""
    except ImportError:
        pass
    try:
        from services.order_details_flow import _lightweight_details_or_invoice_signal

        light = (_lightweight_details_or_invoice_signal(comb) or "").strip()
        if light in ("order_invoice", "order_details"):
            return light
    except ImportError:
        pass
    return ""


def _structural_track_goal_from_message(
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """
    Guardrail: personal order tracking (ETA/status/not received) with order id in message.
    Does not fire for general delivery-policy KB questions.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""
    try:
        from utils.helpers import (
            message_is_general_delivery_policy_question,
            _text_is_order_tracking_intent_leaf,
            _text_is_undelivered_order_complaint,
        )

        if message_is_general_delivery_policy_question(comb):
            return ""
        if _structural_details_or_invoice_goal_from_message(original_msg, msg_en):
            return ""
        if not re.search(r"\b\d{4,20}\b", comb):
            return ""
        if _text_is_order_tracking_intent_leaf(comb) or _text_is_undelivered_order_complaint(
            comb
        ):
            return "track"
    except ImportError:
        pass
    return ""


def _message_is_personal_order_tracking_without_id(
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """Personal track/ETA/not-arrived — no order id in text (brain enrich → ask id)."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb or re.search(r"\b\d{4,20}\b", comb):
        return False
    try:
        from utils.helpers import (
            _text_is_order_tracking_intent_leaf,
            _text_is_undelivered_order_complaint,
            message_is_general_delivery_policy_question,
        )

        if message_is_general_delivery_policy_question(comb):
            return False
        return bool(
            _text_is_order_tracking_intent_leaf(comb)
            or _text_is_undelivered_order_complaint(comb)
        )
    except ImportError:
        return False


def reconcile_structural_order_sub_intent_from_tracking_message(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """When brain locked details but message+order_id clearly wants tracking — fix JSON."""
    out = dict(route or {})
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    if olk in ("invoice", "refund_status") or rh in (
        "order_details_api",
        "refund_status_api",
    ):
        if olk in ("invoice", "refund_status", "details", "order_details"):
            return out
    structural = _structural_track_goal_from_message(original_msg, msg_en)
    if structural != "track" and _message_is_personal_order_tracking_without_id(
        original_msg, msg_en
    ):
        structural = "track"
    if structural != "track":
        return out
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    if olk in ("track", "tracking") and rh == "order_tracking_api":
        return out
    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["run_catalog_search"] = False
    out["kb_keys"] = []
    out["order_lookup_kind"] = "track"
    out["field_focus"] = "timeline"
    out["route_handler"] = "order_tracking_api"
    out.pop("_order_tracking_locked", None)
    log_reasoning(
        "Structural guard: message+order_id → track (fix brain details drift)."
    )
    return out


def reconcile_general_delivery_policy_from_message(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """General delivery timeline — admin KB via brain JSON or embeddings (not keyword lists)."""
    from services.kb_service import resolve_brain_kb_keys
    from services.query_understanding import top_customer_kb_file_match

    out = dict(route or {})
    if llm_semantic_route_available(out):
        if ai_route_is_kb_read(out):
            out["kb_keys"] = resolve_brain_kb_keys(out, original_msg, msg_en)
            return out
        um = (out.get("user_meaning") or "").strip()
        if um:
            top_key, top_score = top_customer_kb_file_match(
                um, um, ai_route=out
            )
            ch = (out.get("data_channel") or "").strip().lower()
            olk = (out.get("order_lookup_kind") or "").strip().lower()
            intent = (out.get("intent") or "").strip().lower()
            misrouted_live = ch == "live_api" and olk in (
                "track",
                "tracking",
                "",
                "none",
            )
            if (
                top_key
                and top_score >= 0.32
                and misrouted_live
                and intent in ("order", "pincode_check", "general")
                and not out.get("needs_order_id")
            ):
                out["intent"] = "general"
                out["data_channel"] = "kb"
                out["kb_keys"] = resolve_brain_kb_keys(out, original_msg, msg_en)
                out["needs_order_id"] = False
                out["numeric_context"] = "none"
                out["order_lookup_kind"] = "none"
                out["route_handler"] = ""
                out["run_catalog_search"] = False
                out["conversation_scope"] = "welfog_support"
                log_reasoning(
                    "Brain+embedding delivery policy — admin KB (not live track API)."
                )
        return out

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    top_key, top_score = top_customer_kb_file_match(
        original_msg, msg_en, ai_route=out
    )
    if not top_key or top_score < 0.30:
        return out
    out["intent"] = "general"
    out["data_channel"] = "kb"
    out["kb_keys"] = resolve_brain_kb_keys(out, original_msg, msg_en)
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out["route_handler"] = ""
    out["run_catalog_search"] = False
    out["conversation_scope"] = "welfog_support"
    log_reasoning(
        "Embedding delivery-policy match — admin KB (LLM unavailable fallback)."
    )
    return out


def reconcile_structural_order_sub_intent_from_message(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """When brain locked track but message+order_id clearly wants details/invoice — fix JSON."""
    out = dict(route or {})
    refund_struct = _structural_refund_goal_from_message(original_msg, msg_en)
    if refund_struct == "refund_status":
        olk_rf = (out.get("order_lookup_kind") or "none").strip().lower()
        rh_rf = (out.get("route_handler") or "").strip().lower()
        if olk_rf == "refund_status" and rh_rf == "refund_status_api":
            return out
        out["intent"] = "refund"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
        out["kb_keys"] = []
        out["order_lookup_kind"] = "refund_status"
        out["field_focus"] = "status"
        out.pop("_order_tracking_locked", None)
        log_reasoning(
            "Structural guard: message+order_id → refund_status (fix brain details drift)."
        )
        return apply_order_live_route_handler(out)
    structural = _structural_details_or_invoice_goal_from_message(original_msg, msg_en)
    if not structural:
        return out
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    if structural == "order_invoice":
        if olk == "invoice" and rh == "order_details_api":
            return out
    elif structural == "order_details":
        if olk in ("details", "invoice") and rh == "order_details_api":
            return out
    if olk not in ("none", "", "track", "tracking") and rh not in (
        "",
        "order_tracking_api",
    ):
        if structural != "order_invoice" or olk == "invoice":
            return out
    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["run_catalog_search"] = False
    out["kb_keys"] = []
    if structural == "order_invoice":
        out["order_lookup_kind"] = "invoice"
        out["field_focus"] = "invoice"
    else:
        out["order_lookup_kind"] = "details"
        focus = (out.get("field_focus") or "").strip().lower()
        if focus not in ("payment", "product", "delivery", "summary", "status"):
            out["field_focus"] = "summary"
    out.pop("_order_tracking_locked", None)
    log_reasoning(
        f"Structural guard: message+order_id → {structural} (fix brain track drift)."
    )
    return apply_order_live_route_handler(out)


def resolve_order_live_goal_for_turn(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> str:
    """Brain enrich + structural guard — single live goal for dispatch."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    try:
        from utils.helpers import message_is_general_delivery_policy_question

        if message_is_general_delivery_policy_question(comb):
            return ""
    except ImportError:
        pass

    refund_struct = _structural_refund_goal_from_message(original_msg, msg_en)
    if refund_struct == "refund_status":
        return "refund_status"

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if comb and not re.search(r"\b\d{4,20}\b", comb):
        try:
            from services.semantic_intent import llm_semantic_route_available

            if route and llm_semantic_route_available(route):
                locked_pre = ensure_brain_order_route_locked(
                    route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                brain_goal = brain_route_to_live_goal(
                    locked_pre,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conversation_context,
                )
                if brain_goal in (
                    "refund_status",
                    "order_invoice",
                    "order_details",
                    "track",
                    "payment",
                ):
                    return brain_goal
        except ImportError:
            pass
        try:
            from services.order_details_flow import _lightweight_details_or_invoice_signal

            light = (_lightweight_details_or_invoice_signal(comb) or "").strip()
            if light == "order_invoice":
                return "order_invoice"
        except ImportError:
            pass
        try:
            from utils.helpers import (
                _text_is_order_tracking_intent_leaf,
                _text_is_undelivered_order_complaint,
            )

            if _text_is_order_tracking_intent_leaf(
                comb
            ) or _text_is_undelivered_order_complaint(comb):
                return "track"
        except ImportError:
            pass

    details_struct = _structural_details_or_invoice_goal_from_message(
        original_msg, msg_en
    )
    track_struct = _structural_track_goal_from_message(original_msg, msg_en)

    if details_struct == "order_invoice":
        return "order_invoice"
    if details_struct == "order_details" and track_struct != "track":
        return "order_details"
    if track_struct == "track":
        return "track"

    locked = ensure_brain_order_route_locked(
        route,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    locked = reconcile_structural_order_sub_intent_from_message(
        locked,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    locked = reconcile_structural_order_sub_intent_from_tracking_message(
        locked,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    goal = brain_route_to_live_goal(
        locked,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    if goal in ("order_details", "payment") and track_struct == "track":
        if details_struct in ("order_invoice", "order_details"):
            return details_struct if details_struct == "order_invoice" else "order_details"
        return "track"
    try:
        from utils.helpers import _text_is_order_tracking_intent_leaf

        if (
            goal in ("order_details", "payment", "")
            and re.search(r"\b\d{4,20}\b", comb)
            and _text_is_order_tracking_intent_leaf(comb)
            and not details_struct
        ):
            return "track"
    except ImportError:
        pass
    return goal


def ai_route_is_kb_read(route: dict | None) -> bool:
    """Trust Groq JSON — any language; no phrase lists on customer text."""
    if not llm_semantic_route_available(route):
        return False
    r = route or {}
    if r.get("run_catalog_search"):
        return False
    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    strategy = (r.get("answer_strategy") or "").strip().lower()
    olk = (r.get("order_lookup_kind") or "none").strip().lower()
    kb_keys = list(r.get("kb_keys") or [])
    if intent in ("general", "refund", "payment", "seller") and channel == "kb":
        return True
    if strategy in ("kb_only", "kb_then_ai") and intent in (
        "general",
        "refund",
        "payment",
        "seller",
    ):
        return True
    if (
        intent in ("general", "refund", "payment", "seller")
        and kb_keys
        and not r.get("needs_order_id")
        and olk in ("none", "")
        and channel in ("kb", "", "none")
    ):
        return True
    return False


def ai_meaning_describes_delivery_serviceability(route: dict | None) -> bool:
    """
    PIN / area delivery — structured ai_brain_route JSON only.
    The universal brain LLM classifies any customer language → intent/channel/handler;
    never scan customer text or English phrase lists here.
    """
    if not llm_semantic_route_available(route):
        return False
    r = route or {}
    if r.get("_pincode_delivery_locked"):
        return True
    handler = (r.get("route_handler") or "").strip().lower()
    if handler == "pincode_delivery_api":
        return True
    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    nc = (r.get("numeric_context") or "").strip().lower()
    if r.get("needs_order_id") or nc == "order_id":
        return False
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    if olk in ("track", "tracking", "details", "invoice", "refund_status"):
        return False
    if ai_meaning_describes_order_details(r):
        return False
    focus = (r.get("field_focus") or "").strip().lower()
    if focus in ("payment", "product", "summary", "status", "timeline", "invoice"):
        return False
    if intent == "pincode_check":
        return True
    if nc == "pincode" and channel == "live_api":
        return True
    return False


def ai_meaning_describes_order_details(route: dict | None) -> bool:
    """One-order info — brain order_lookup_kind / route_handler / field_focus only."""
    if not llm_semantic_route_available(route):
        return False
    r = route or {}
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    rh = (r.get("route_handler") or "").strip().lower()
    if olk == "invoice":
        return False
    if olk in ("details", "order_details") or rh == "order_details_api":
        return True
    focus = (r.get("field_focus") or "").strip().lower()
    if focus in ("payment", "product", "delivery", "summary", "status"):
        return True
    intent = (r.get("intent") or "").strip().lower()
    if intent in ("order_details", "order_detail", "order_info"):
        return bool(r.get("needs_order_id"))
    if (
        intent == "payment"
        and r.get("needs_order_id")
        and (r.get("data_channel") or "").strip().lower() == "live_api"
    ):
        return True
    return False


def correct_pincode_vs_order_id_numeric_context(
    route: dict | None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """
    Groq sometimes echoes order id in numeric_context with intent=pincode_check.
    Entity shape + AI user_meaning — not customer-text keyword routing.
    """
    out = dict(route or {})
    if not llm_semantic_route_available(out):
        return out
    nc_val = coerce_route_str(out.get("numeric_context"), "none")
    order_digits = ""
    if re.fullmatch(r"\d{4,20}", nc_val):
        order_digits = nc_val
    if not order_digits:
        comb = f"{original_msg or ''} {msg_en or ''}".strip()
        if comb and (out.get("intent") or "").strip().lower() == "pincode_check":
            try:
                from utils.helpers import (
                    _digits_in_message_are_order_id_not_pincode_impl,
                    extract_order_id,
                )

                if _digits_in_message_are_order_id_not_pincode_impl(comb):
                    oid = (extract_order_id(comb, conversation_context) or "").strip()
                    if oid:
                        order_digits = oid
            except ImportError:
                pass
    if not order_digits:
        return out

    intent = (out.get("intent") or "").strip().lower()
    is_order_id_entity = len(order_digits) >= 7
    if not is_order_id_entity and intent == "pincode_check":
        if ai_meaning_describes_order_details(out) or out.get("needs_order_id"):
            is_order_id_entity = True

    if not is_order_id_entity:
        return out

    out["numeric_context"] = "order_id"
    out["needs_order_id"] = True
    out["extracted_pincode"] = ""
    out["run_catalog_search"] = False
    out["data_channel"] = "live_api"
    if intent == "pincode_check":
        out["intent"] = "order"
        out["conversation_scope"] = "welfog_support"
        out["scope_reply"] = ""
        log_reasoning(
            "Route fix: order id entity in brain JSON — order live API (not pincode_check)."
        )
    out = correct_order_details_vs_tracking_from_ai_meaning(out)
    olk_after = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk_after in ("none", ""):
        out["order_lookup_kind"] = "details"
        ff = (out.get("field_focus") or "").strip().lower()
        if ff not in ("payment", "product", "delivery", "summary", "status", "invoice"):
            out["field_focus"] = "summary"
        out["route_handler"] = "order_details_api"
    return out


def correct_order_details_vs_tracking_from_ai_meaning(route: dict | None) -> dict:
    """
    Groq sometimes sets order_lookup_kind=track when field_focus/olk say details.
    Trust brain JSON fields only — not phrase lists on user_meaning or customer text.
    """
    out = dict(route or {})
    if not llm_semantic_route_available(out):
        return out
    if (out.get("meta_kind") or "none").strip().lower() != "none":
        return out
    if ai_route_is_kb_read(out):
        return out

    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    focus = (out.get("field_focus") or "").strip().lower()

    if olk in ("details", "invoice") and rh == "order_details_api":
        return out
    if olk == "track" and rh == "order_tracking_api" and focus == "timeline":
        return out

    if not ai_meaning_describes_order_details(out):
        return out
    if olk == "track" and focus == "timeline":
        return out

    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    if olk == "invoice" or focus == "invoice":
        out["order_lookup_kind"] = "invoice"
        out["field_focus"] = "invoice"
    else:
        out["order_lookup_kind"] = "details"
        if focus not in ("payment", "product", "delivery", "summary", "status"):
            out["field_focus"] = "summary"
    out["route_handler"] = "order_details_api"
    out.pop("_order_tracking_locked", None)
    log_reasoning(
        "Route fix: brain JSON fields → order_details_api (not track/pincode)."
    )
    return out


def ai_meaning_describes_existing_order_track(route: dict | None) -> bool:
    """Live track/status of one order — brain order_lookup_kind / route_handler / field_focus."""
    if not llm_semantic_route_available(route):
        return False
    r = route or {}
    if ai_meaning_describes_order_details(r):
        focus = (r.get("field_focus") or "").strip().lower()
        if focus and focus != "timeline":
            return False
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    rh = (r.get("route_handler") or "").strip().lower()
    if olk in ("details", "invoice"):
        return False
    if olk == "track" or rh == "order_tracking_api":
        return True
    focus = (r.get("field_focus") or "").strip().lower()
    if focus == "timeline" and r.get("needs_order_id"):
        return True
    if olk in ("none", ""):
        return False
    intent = (r.get("intent") or "").strip().lower()
    if intent in ("order_track", "order_tracking", "tracking_request"):
        return True
    return False


def infer_semantic_goal_from_ai_route(ai_route: dict | None) -> str:
    """
    Customer goal from Groq JSON only — used when LLM routing succeeded.
    Empty when unclear; keyword layers must not invent a competing goal.
    """
    if not llm_semantic_route_available(ai_route):
        return ""
    route = dict(ai_route or {})
    try:
        route = correct_order_details_vs_tracking_from_ai_meaning(route)
    except Exception:
        pass
    intent = (route.get("intent") or "").strip().lower()
    olk = (route.get("order_lookup_kind") or "").strip().lower()

    if intent in (
        "invoice",
        "order_invoice",
        "order_bill",
        "order_bill_request",
        "bill_request",
        "bill",
        "receipt",
        "order_receipt",
        "gst_invoice",
    ):
        return "order_invoice"
    if intent == "refund" and (route.get("data_channel") or "").strip().lower() == "live_api":
        if not ai_route_is_kb_read(route):
            return "refund_status"

    if intent in ("product", "deals", "categories", "category_feed") or route.get(
        "run_catalog_search"
    ):
        return ""
    if (route.get("data_channel") or "").strip().lower() == "catalog":
        return ""

    if ai_meaning_describes_delivery_serviceability(route) or intent == "pincode_check":
        return "pincode_delivery"
    focus = (route.get("field_focus") or "").strip().lower()
    if focus == "invoice" or olk == "invoice":
        return "order_invoice"
    if olk == "details" or ai_meaning_describes_order_details(route):
        return "order_details"
    rh = (route.get("route_handler") or "").strip().lower()
    if rh == "order_details_api":
        return "order_invoice" if olk == "invoice" else "order_details"
    if olk in ("track", "tracking") or rh == "order_tracking_api":
        return "track_single_order"
    if olk == "track" or (
        intent in ("order", "payment")
        and ai_meaning_describes_existing_order_track(route)
    ):
        return "track_single_order"
    if intent == "refund_status" or olk == "refund_status":
        return "refund_status"
    if intent == "refund" and (route.get("data_channel") or "").strip().lower() == "live_api":
        if route.get("needs_order_id") and not ai_route_is_kb_read(route):
            return "refund_status"
    if ai_route_is_kb_read(route):
        if intent == "refund":
            return "refund_policy"
        keys = {
            k.lower().replace("welfog_api_", "")
            for k in (route.get("kb_keys") or [])
        }
        if "refund" in keys:
            return "refund_policy"
        if keys and keys <= frozenset({"refund", "faqs", "shipping", "terms"}):
            return "refund_policy"
        return ""
    if intent == "seller" and ai_route_is_kb_read(route):
        return ""
    if intent == "order_history" and (route.get("data_channel") or "").strip().lower() == "live_api":
        return "order_history_list"
    if intent == "wishlist":
        return "wishlist_list"
    return ""


def _default_kb_keys_for_intent(
    intent: str,
    existing: list[str],
    route: dict | None = None,
) -> list[str]:
    """Fill empty kb_keys — semantic search across admin files, not hardcoded topic names."""
    keys = list(existing or [])
    if keys:
        return keys
    try:
        from services.kb_service import resolve_brain_kb_keys

        r = dict(route or {})
        if intent and not r.get("intent"):
            r["intent"] = intent
        resolved = resolve_brain_kb_keys(r, "", r.get("user_meaning") or "")
        if resolved:
            return resolved
    except ImportError:
        pass
    try:
        from services.kb_service import get_customer_kb_keys

        return get_customer_kb_keys()[:3]
    except ImportError:
        return []


def promote_informational_kb_from_ai_meaning(route: dict | None) -> dict:
    """
    Align route with Groq KB-read classification (intent + data_channel + kb_keys).
    No English phrase lists — customer may write in any language.
    """
    out = dict(route or {})
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(out):
            return out
    except ImportError:
        pass
    if not llm_semantic_route_available(out):
        return out
    if (out.get("meta_kind") or "none").strip().lower() != "none":
        return out
    if out.get("run_catalog_search"):
        return out

    intent = (out.get("intent") or "").strip().lower()
    channel = (out.get("data_channel") or "").strip().lower()

    if not ai_route_is_kb_read(out):
        if intent in ("order", "order_history", "wishlist", "product", "pincode_check"):
            return out
        return out

    handler_pre = (out.get("route_handler") or "").strip().lower()
    if handler_pre == "pincode_delivery_api" or intent == "pincode_check":
        return out
    if ai_meaning_describes_delivery_serviceability(out):
        return out

    try:
        from services.account_list_semantics import (
            ai_route_requests_order_history_in_chat,
            ai_route_requests_wishlist_in_chat,
            ai_route_requests_wishlist_howto,
        )

        if (
            ai_route_requests_order_history_in_chat(out)
            or ai_route_requests_wishlist_in_chat(out)
            or ai_route_requests_wishlist_howto(out)
        ):
            return out
        from services.refund_status_semantics import refund_status_route_is_locked
        from services.order_details_flow import order_details_route_is_locked

        if refund_status_route_is_locked(out) or order_details_route_is_locked(out):
            return out
    except ImportError:
        pass

    out["data_channel"] = "kb"
    out["needs_order_id"] = False
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out["numeric_context"] = "none"
    out["order_lookup_kind"] = "none"
    out["kb_keys"] = _default_kb_keys_for_intent(
        intent, list(out.get("kb_keys") or []), route=out
    )
    try:
        from services.query_understanding import infer_kb_query_category, scoped_kb_keys_for_retrieval

        meaning = (out.get("user_meaning") or "").strip()
        if len(out.get("kb_keys") or []) > 4:
            cat = infer_kb_query_category(
                meaning,
                "",
                ai_route=out,
                conversation_context="",
            )
            scoped = scoped_kb_keys_for_retrieval(
                cat, ai_route=out, user_meaning=meaning
            )
            if scoped:
                out["kb_keys"] = scoped[:4]
    except ImportError:
        pass
    out.pop("route_handler", None)

    if intent == "order" and channel != "live_api":
        out["intent"] = "refund"
        out["kb_keys"] = _default_kb_keys_for_intent("refund", out["kb_keys"])
        log_reasoning(
            "Route fix: Groq KB read with intent=order → refund/policy KB (not live order API)."
        )
    else:
        log_reasoning(
            f"Route fix: trust Groq KB read — intent={intent} "
            f"keys={','.join(out.get('kb_keys') or [])[:80]}."
        )
    return out


def correct_delivery_vs_tracking_from_ai_meaning(route: dict | None) -> dict:
    """
    Fix clear LLM contradictions (intent=order but meaning=delivery area) using English meaning fields.
    No customer-message keyword lists.
    """
    out = dict(route or {})
    if not llm_semantic_route_available(out):
        return out
    out = correct_pincode_vs_order_id_numeric_context(out)
    out = correct_order_details_vs_tracking_from_ai_meaning(out)
    if (out.get("meta_kind") or "none").strip().lower() != "none":
        return out

    is_delivery = ai_meaning_describes_delivery_serviceability(out)
    is_track = ai_meaning_describes_existing_order_track(out)
    intent = (out.get("intent") or "").strip().lower()

    if is_delivery and not is_track:
        if intent != "pincode_check":
            out["intent"] = "pincode_check"
            out["data_channel"] = "live_api"
            out["needs_order_id"] = False
            out["numeric_context"] = "pincode"
            out["run_catalog_search"] = False
            out["search_query"] = ""
            out["order_lookup_kind"] = "none"
            out.pop("route_handler", None)
            log_reasoning(
                "Route fix: LLM meaning is delivery/serviceability — pincode_check (not order tracking)."
            )
        return out

    if is_track and intent == "pincode_check":
        out["intent"] = "order"
        out["data_channel"] = "live_api"
        out["needs_order_id"] = True
        out["numeric_context"] = "order_id"
        out["run_catalog_search"] = False
        if (out.get("order_lookup_kind") or "none").strip().lower() == "none":
            out["order_lookup_kind"] = "track"
        out.pop("route_handler", None)
        log_reasoning(
            "Route fix: LLM meaning is existing-order track — order (not pincode_check)."
        )
    out = finalize_order_lookup_from_brain_json(out)
    out = apply_order_live_route_handler(out)
    return out


def correct_api_vs_kb_from_embedding(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """
    When embeddings strongly match an admin FAQ file, prefer KB over mistaken live API route.
    Language-agnostic — fixes e.g. delivery-policy questions routed to order tracking.
    """
    import re

    out = dict(route or {})
    if out.get("_universal_brain_route"):
        return out
    ch = (out.get("data_channel") or "").strip().lower()
    if ch == "kb":
        return out
    if ch == "live_api":
        return out
    intent = (out.get("intent") or "").strip().lower()
    if intent in ("wishlist", "order_history", "product", "order", "refund", "categories", "deals"):
        return out
    try:
        from services.account_list_semantics import (
            KIND_PURCHASE_IN_CHAT,
            KIND_WISHLIST_IN_CHAT,
            _norm_account_list_kind,
        )

        alk = _norm_account_list_kind(out.get("account_list_kind") or "")
        if alk in (KIND_WISHLIST_IN_CHAT, KIND_PURCHASE_IN_CHAT):
            return out
    except ImportError:
        pass
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return out
    try:
        from services.location_delivery_resolver import turn_requests_delivery_serviceability

        if turn_requests_delivery_serviceability(
            original_msg,
            msg_en,
            conversation_context,
            out,
            allow_llm=True,
        ):
            return out
        if (out.get("route_handler") or "").strip().lower() == "pincode_delivery_api":
            return out
        if (out.get("intent") or "").strip().lower() == "pincode_check":
            return out
        if ai_meaning_describes_delivery_serviceability(out):
            return out
    except ImportError:
        pass
    try:
        from utils.helpers import _is_plausible_order_id

        if _is_plausible_order_id(comb) or re.search(r"\b\d{6,}\b", comb):
            return out
        from services.order_tracking_semantics import message_user_wants_order_tracking

        if message_user_wants_order_tracking(comb, conversation_context):
            return out
        if out.get("needs_order_id") and (out.get("intent") or "") == "order":
            return out
        from services.query_understanding import top_customer_kb_file_match

        top_key, score = top_customer_kb_file_match(
            original_msg, msg_en, conversation_context=conversation_context
        )
        if score >= 0.55 and top_key in (
            "faqs",
            "shipping",
            "payment",
            "refund",
            "company",
            "support",
        ):
            out["intent"] = {
                "shipping": "shipping",
                "payment": "payment",
                "refund": "refund",
            }.get(top_key, "general")
            out["data_channel"] = "kb"
            out["kb_keys"] = [top_key]
            out["needs_order_id"] = False
            out["numeric_context"] = "none"
            out["order_lookup_kind"] = "none"
            out["run_catalog_search"] = False
            out.pop("route_handler", None)
            log_reasoning(
                f"Route fix: KB match {top_key} ({score:.2f}) overrides API misroute."
            )
    except ImportError:
        pass
    return out


def brain_route_indicates_account_list_live(route: dict | None) -> bool:
    """True when ai_brain_route JSON locks wishlist or purchase-history live API."""
    if not isinstance(route, dict):
        return False
    alk = (route.get("account_list_kind") or "").strip().lower()
    if alk in ("wishlist_in_chat", "purchase_history_in_chat"):
        return True
    rh = (route.get("route_handler") or "").strip().lower()
    if rh in ("wishlist_api", "order_history_api"):
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    return intent in ("wishlist", "order_history") and channel == "live_api"


def brain_route_skip_heavy_enrich(route: dict | None) -> bool:
    """OOD / chitchat / pure KB / locked account-list — skip product/order enrich (10–90s)."""
    if not isinstance(route, dict):
        return False
    if route.get("_zero_llm_fast") or route.get("_product_catalog_locked"):
        return True
    if route.get("_preflight_catalog_menu") or route.get("_preflight_api"):
        return True
    if route.get("_pincode_delivery_fast") or route.get("_pincode_delivery_locked"):
        return True
    rh = (route.get("route_handler") or "").strip().lower()
    if rh in ("wishlist_api", "order_history_api"):
        return True
    alk = (route.get("account_list_kind") or "").strip().lower()
    if alk in (
        "wishlist_in_chat",
        "purchase_history_in_chat",
        "wishlist_howto",
        "purchase_history_howto",
    ):
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent in ("wishlist", "order_history") and channel == "live_api":
        return True
    if intent in ("deals", "categories", "category_feed") and channel == "live_api":
        return True
    if intent == "pincode_check" and channel == "live_api":
        return True
    # General refund/policy KB — not personal refund_status API (was running 100s+ enrich).
    if intent in ("refund", "payment", "general") and channel == "kb" and not route.get(
        "needs_order_id"
    ):
        return True
    if intent in (
        "order",
        "refund",
    ):
        return False
    if intent == "product" and channel == "catalog" and route.get("run_catalog_search"):
        return True
    scope = (route.get("conversation_scope") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    if intent == "out_of_domain":
        return True
    if scope in ("out_of_domain", "general_chitchat", "harm_sensitive"):
        return True
    if channel == "kb" and not route.get("needs_order_id"):
        if intent not in (
            "order",
            "order_history",
            "wishlist",
            "product",
            "pincode_check",
            "deals",
            "categories",
            "category_feed",
        ):
            return True
    return False


def enrich_universal_brain_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    ctx: dict | None = None,
) -> dict:
    """
    Lock refund / invoice / order live API on universal brain JSON — no second routing LLM.
    Fixes brain misroutes (refund status → return-policy KB, invoice → track).
    """
    out = _normalize_llm_route(dict(route or {}))
    out["_universal_brain_route"] = True
    out = repair_brain_json_quality(out, original_msg, msg_en=msg_en)
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog
        from services.semantic_intent import ai_route_is_product_catalog

        if ai_route_is_product_catalog(out) or brain_route_indicates_product_catalog(out):
            out = reconcile_product_catalog_from_brain_meaning(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            out["_product_catalog_locked"] = True
            out["_needs_product_nlu_llm"] = False
            out["_ai_single_pass"] = True
            out["_turn_promotions_done"] = True
            if isinstance(ctx, dict) and ctx.get("last"):
                out["_ctx_last"] = ctx.get("last")
            log_reasoning(
                "Universal brain — AI product catalog locked; skip enrich/embed stack."
            )
            return out
    except ImportError:
        pass
    if brain_route_skip_heavy_enrich(out):
        out["_turn_promotions_done"] = True
        if isinstance(ctx, dict) and ctx.get("last"):
            out["_ctx_last"] = ctx.get("last")
        log_reasoning(
            "Universal brain — OOD/chitchat/KB fast path; skip product/order enrich."
        )
        return out
    intent_ub = (out.get("intent") or "").strip().lower()
    olk_ub = (out.get("order_lookup_kind") or "").strip().lower()
    rh_ub = (out.get("route_handler") or "").strip().lower()
    if intent_ub in ("order", "refund", "payment") and olk_ub not in ("none", ""):
        out = infer_order_lookup_from_brain_english_fields(out)
        out = reconcile_order_sub_intent_from_brain_json(out)
        out = reconcile_invoice_from_brain_meaning(out)
        out = finalize_order_lookup_from_brain_json(out)
        out = lock_order_live_api_from_brain(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
        out = reconcile_structural_order_sub_intent_from_message(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        out = reconcile_structural_order_sub_intent_from_tracking_message(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        out = lock_order_live_api_from_brain(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conversation_context,
        )
        out["_turn_promotions_done"] = True
        if isinstance(ctx, dict) and ctx.get("last"):
            out["_ctx_last"] = ctx.get("last")
        log_reasoning(
            f"Universal brain — locked order sub-intent olk={olk_ub} handler={rh_ub or '-'}; "
            "skip product/catalog enrich."
        )
        return out
    out = reconcile_pincode_delivery_from_brain_meaning(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    if out.get("_pincode_delivery_locked") or (
        (out.get("intent") or "").strip().lower() == "pincode_check"
        and (out.get("data_channel") or "").strip().lower() == "live_api"
    ):
        out["_turn_promotions_done"] = True
        if isinstance(ctx, dict) and ctx.get("last"):
            out["_ctx_last"] = ctx.get("last")
        log_reasoning(
            "Universal brain — pincode delivery locked; skip product/order enrich."
        )
        return out
    out = reconcile_product_catalog_from_brain_meaning(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(out):
            out["_turn_promotions_done"] = True
            if isinstance(ctx, dict) and ctx.get("last"):
                out["_ctx_last"] = ctx.get("last")
            log_reasoning(
                "Universal brain — product catalog locked; skip order/refund enrich stack."
            )
            return out
    except ImportError:
        pass
    if _brain_route_has_shopping_entities(out, original_msg=original_msg, msg_en=msg_en):
        out = reconcile_product_catalog_from_brain_meaning(
            out,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            if product_catalog_route_is_locked(out):
                out["_turn_promotions_done"] = True
                log_reasoning(
                    "Universal brain — product_entities JSON locked; skip order enrich + classifier."
                )
                return out
        except ImportError:
            pass
    try:
        from services.product_browse_semantics import promote_product_browse_on_route
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if not _brain_route_has_shopping_entities(
            out, original_msg=original_msg, msg_en=msg_en
        ):
            out = promote_product_browse_on_route(
                out,
                original_msg,
                msg_en,
                conversation_context,
            )
        if product_catalog_route_is_locked(out):
            out["_product_catalog_locked"] = True
            out["_turn_promotions_done"] = True
            log_reasoning(
                "Universal brain — semantic product browse promoted; skip order enrich stack."
            )
            return out
    except ImportError:
        pass
    out = reconcile_general_delivery_policy_from_message(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    if (out.get("data_channel") or "").strip().lower() == "kb":
        out["_turn_promotions_done"] = True
        return out
    if isinstance(ctx, dict) and ctx.get("last"):
        out["_ctx_last"] = ctx.get("last")

    try:
        from services.account_list_semantics import reconcile_account_list_from_brain_meaning

        out = reconcile_account_list_from_brain_meaning(out, msg_en=msg_en)
    except ImportError:
        pass

    out = infer_order_lookup_from_brain_english_fields(out)
    out = reconcile_order_sub_intent_from_brain_json(out)
    out = reconcile_invoice_from_brain_meaning(out)
    out = reconcile_category_browse_from_brain_meaning(out)
    out = finalize_order_lookup_from_brain_json(out)
    out = lock_order_live_api_from_brain(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )
    out = reconcile_structural_order_sub_intent_from_message(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    out = reconcile_structural_order_sub_intent_from_tracking_message(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
    )
    out = lock_order_live_api_from_brain(
        out,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb and not conversation_context:
        out["_turn_promotions_done"] = True
        return out

    try:
        from services.account_list_semantics import promote_account_list_on_route

        out = promote_account_list_on_route(
            out,
            original_msg,
            msg_en,
            conversation_context,
            reply_lang,
            allow_llm=False,
        )
    except ImportError:
        pass

    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk in ("none", ""):
        try:
            from services.refund_status_semantics import promote_refund_status_on_route

            out = promote_refund_status_on_route(
                out,
                original_msg,
                msg_en,
                conversation_context,
                reply_lang,
                allow_llm=False,
            )
            out = finalize_order_lookup_from_brain_json(out)
            out = lock_order_live_api_from_brain(
                out,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conversation_context,
            )
        except ImportError:
            pass

    out["_turn_promotions_done"] = True
    return out


def enrich_route_from_llm(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """Normalize + semantic promotion (product browse before assistant intro)."""
    out = _normalize_llm_route(route)
    try:
        from services.chat_flow_telemetry import is_routing_complete, skip_step

        if is_routing_complete():
            skip_step("enrich_route_from_llm", "routing already locked")
            return out
    except ImportError:
        pass
    alk = (out.get("account_list_kind") or "").strip().lower()
    if alk in (
        "wishlist_in_chat",
        "purchase_history_in_chat",
        "wishlist_howto",
        "purchase_history_howto",
    ):
        out["_turn_promotions_done"] = True
        try:
            from services.chat_flow_telemetry import skip_step

            skip_step("enrich_route_from_llm", f"account_list_kind={alk}")
        except ImportError:
            pass
        return out
    if out.get("_turn_promotions_done"):
        try:
            from services.chat_flow_telemetry import record_route_step, skip_step

            skip_step("enrich_route_from_llm", "already done this turn")
            record_route_step("enrich_skip")
        except ImportError:
            pass
        return out
    if out.get("_universal_brain_route"):
        try:
            from services.chat_flow_telemetry import record_route_step, skip_step

            record_route_step("enrich_route_from_llm_light")
            skip_step(
                "enrich_route_from_llm",
                "universal brain — order/refund/invoice promotions",
            )
        except ImportError:
            pass
        return enrich_universal_brain_route(
            out,
            original_msg,
            msg_en,
            conversation_context,
        )
    try:
        from services.chat_flow_telemetry import record_route_step

        record_route_step("enrich_route_from_llm")
    except ImportError:
        pass
    if original_msg or msg_en:
        from services.account_list_semantics import (
            KIND_NONE,
            _norm_account_list_kind,
            promote_account_list_on_route,
        )

        intent_pre = (out.get("intent") or "").strip().lower()
        mk_pre = (out.get("meta_kind") or "none").strip().lower()
        alk_pre = _norm_account_list_kind(
            coerce_route_str(out.get("account_list_kind"), KIND_NONE)
        )
        skip_account_promote = (
            alk_pre == KIND_NONE
            and intent_pre == "general"
            and mk_pre in (
                "assistant_intro",
                "conversational",
                "hostile",
                "bot_latency",
                "topic_denial",
            )
        )
        if not skip_account_promote:
            out = promote_account_list_on_route(
                out, original_msg, msg_en, conversation_context
            )
        intent = (out.get("intent") or "").strip().lower()
        if intent == "product" or out.get("run_catalog_search"):
            from services.product_browse_semantics import promote_product_browse_on_route

            out = promote_product_browse_on_route(out, original_msg, msg_en)
        if intent == "general" and not out.get("run_catalog_search"):
            from services.meta_turn_semantics import promote_assistant_intro_on_route

            out = promote_assistant_intro_on_route(out, original_msg, msg_en)
    product_catalog_locked = False
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        product_catalog_locked = product_catalog_route_is_locked(out)
    except ImportError:
        pass
    if (original_msg or msg_en) and not product_catalog_locked:
        try:
            from services.location_delivery_resolver import promote_pincode_delivery_on_route

            out = promote_pincode_delivery_on_route(
                out, original_msg, msg_en, conversation_context
            )
        except ImportError:
            pass
    out = promote_informational_kb_from_ai_meaning(out)
    if original_msg or msg_en:
        out = correct_api_vs_kb_from_embedding(
            out, original_msg, msg_en, conversation_context
        )
    skip_order_promotions = product_catalog_locked
    if (original_msg or msg_en) and not skip_order_promotions:
        try:
            from utils.helpers import turn_skips_order_micro_classifiers

            skip_order_promotions = turn_skips_order_micro_classifiers(
                original_msg, msg_en, conversation_context, ai_route=out
            )
        except ImportError:
            pass
    if (original_msg or msg_en) and not skip_order_promotions:
        try:
            from services.refund_status_semantics import promote_refund_status_on_route

            out = promote_refund_status_on_route(
                out, original_msg, msg_en, conversation_context
            )
        except ImportError:
            pass
        try:
            from services.order_details_flow import promote_order_details_on_route

            out = promote_order_details_on_route(
                out, original_msg, msg_en, conversation_context
            )
        except ImportError:
            pass
        try:
            from services.order_tracking_semantics import promote_order_tracking_on_route

            out = promote_order_tracking_on_route(
                out, original_msg, msg_en, conversation_context
            )
        except ImportError:
            pass
        out = correct_delivery_vs_tracking_from_ai_meaning(out)
    elif skip_order_promotions:
        try:
            from services.chat_flow_telemetry import skip_step

            skip_step("order_refund_promotions", "conversational turn")
        except ImportError:
            pass
    if original_msg or msg_en:
        try:
            from services.conversation_scope import (
                turn_blocks_product_catalog,
                turn_requests_catalog_menu,
            )
            from services.product_catalog_resolver import (
                apply_product_catalog_to_route,
                product_catalog_route_is_locked,
            )

            if turn_requests_catalog_menu(
                original_msg,
                msg_en,
                ai_route=out,
                conversation_context=conversation_context,
                allow_llm=False,
            ):
                out.pop("_product_catalog_locked", None)
                out.pop("run_catalog_search", None)
            elif not turn_blocks_product_catalog(
                original_msg, msg_en, conversation_context, ai_route=out
            ):
                refreshed = apply_product_catalog_to_route(
                    out, original_msg, msg_en, conversation_context=conversation_context
                )
                if product_catalog_route_is_locked(refreshed):
                    out = refreshed
        except ImportError:
            pass
    out["_turn_promotions_done"] = True
    return out


def meta_reply_key_from_route(route: dict | None) -> str:
    mk = (route or {}).get("meta_kind") or "none"
    mk = str(mk).strip().lower()
    return _META_KIND_TO_REPLY.get(mk, "")


def meta_turn_from_route(route: dict | None):
    """Build MetaTurn from AI meta_kind when LLM classified a non-shopping turn."""
    from services.turn_intent_gate import MetaTurn

    if not llm_semantic_route_available(route):
        return None
    intent = (route.get("intent") or "").strip().lower()
    mk_pre = (route.get("meta_kind") or "none").strip().lower()
    if intent == "out_of_domain" or mk_pre == "out_of_domain":
        return None
    key = meta_reply_key_from_route(route)
    if not key:
        return None
    mk = (route.get("meta_kind") or "none").strip().lower()
    log_reasoning(f"AI meta_kind={mk} — structured reply (no catalog).")
    return MetaTurn(mk, key)


def ai_route_allows_catalog_search(route: dict | None) -> bool:
    """
    Catalog OpenSearch only when the routing LLM explicitly allows it.
    Never use keyword shopping heuristics when a valid AI route exists.
    """
    if not llm_semantic_route_available(route):
        return False
    r = _normalize_llm_route(dict(route))
    if r.get("meta_kind") != "none":
        return False
    if not r.get("run_catalog_search"):
        return False
    if (r.get("intent") or "").strip().lower() != "product":
        return False
    if (r.get("data_channel") or "").strip().lower() not in ("catalog", ""):
        return False
    return True


def apply_ai_meta_to_route(route: dict, original_msg: str, msg_en: str) -> dict | None:
    """
    If AI set meta_kind, patch route for KB structured reply. Returns route if handled.
    """
    from services.turn_intent_gate import apply_meta_turn_to_route, format_meta_turn_reply

    meta = meta_turn_from_route(route)
    if not meta:
        return None
    out = apply_meta_turn_to_route(dict(route), meta)
    out["_meta_reply_key"] = meta.reply_key
    return out


def classify_meta_turn_ai_first(
    route: dict | None,
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
):
    """
    Prefer AI meta_kind + semantic promotion; keyword gate only when LLM routing failed.
    """
    if llm_semantic_route_available(route):
        try:
            from services.chat_flow_telemetry import skip_step

            skip_step("classify_meta_enrich", "reuse locked route meta_kind")
        except ImportError:
            pass
        return meta_turn_from_route(dict(route or {}))
    from services.turn_intent_gate import classify_meta_turn

    return classify_meta_turn(original_msg, msg_en, conversation_context)
