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


def _normalize_llm_route(route: dict | None) -> dict:
    """Normalize AI routing JSON fields only (no semantic promotion)."""
    out = dict(route or {})
    out["meta_kind"] = coerce_route_str(out.get("meta_kind"), "none")
    out["intent"] = coerce_route_str(out.get("intent"), "general")
    out["data_channel"] = coerce_route_str(out.get("data_channel"), "")
    out["search_query"] = coerce_route_str(out.get("search_query"), "")
    out["numeric_context"] = coerce_route_str(out.get("numeric_context"), "none")
    out["extracted_pincode"] = coerce_route_str(out.get("extracted_pincode"), "")
    out["reuse_user_value_from_chat"] = coerce_route_str(out.get("reuse_user_value_from_chat"), "")
    out["user_meaning"] = coerce_route_str(out.get("user_meaning"), "")
    out["reasoning"] = coerce_route_str(out.get("reasoning"), "")
    out["scope_reply"] = coerce_route_str(out.get("scope_reply"), "")
    out["conversation_scope"] = coerce_route_str(out.get("conversation_scope"), "")
    out["account_list_kind"] = coerce_route_str(out.get("account_list_kind"), "none").lower()
    out["order_lookup_kind"] = coerce_route_str(out.get("order_lookup_kind"), "none").lower()
    olk = (out.get("order_lookup_kind") or "none").strip().lower()
    if olk not in ("none", "track", "details", "invoice", "refund_status", ""):
        out["order_lookup_kind"] = "none"
    elif olk == "":
        out["order_lookup_kind"] = "none"

    intent_raw = (out.get("intent") or "").strip().lower()
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

    cs = (out.get("conversation_scope") or "").strip().lower().replace("-", "_")
    if cs in ("welfog_support", "general_chitchat", "out_of_domain", "harm_sensitive"):
        out["conversation_scope"] = cs
    elif out.get("intent") == "out_of_domain" or out.get("is_welfog_related") is False:
        out["conversation_scope"] = "out_of_domain"
    else:
        out.setdefault("conversation_scope", "welfog_support")

    sr = (out.get("scope_reply") or "").strip()
    if sr:
        out["scope_reply"] = sr

    out = _promote_off_domain_from_llm_meaning(out)
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
    """PIN / area delivery — from Groq intent fields (works for any customer language)."""
    r = route or {}
    handler = (r.get("route_handler") or "").strip().lower()
    if handler == "pincode_delivery_api":
        return True
    if not llm_semantic_route_available(route):
        return False
    intent = (r.get("intent") or "").strip().lower()
    if intent == "pincode_check":
        return True
    if (r.get("numeric_context") or "").strip().lower() == "pincode":
        return True
    blob = _ai_meaning_blob(r)
    _delivery_meaning = (
        "delivery",
        "deliver to",
        "deliver in",
        "service in",
        "service at",
        "serviceable",
        "ship to",
        "pin code",
        "pincode",
        "postal code",
        "area service",
        "can you deliver",
        "will welfog deliver",
        "provide service",
        "serviceability",
        "service availability",
        "availability in",
        "check service",
        "delivery check",
    )
    if any(m in blob for m in _delivery_meaning):
        try:
            from services.order_tracking_semantics import tracking_blocks_refund_meaning

            if tracking_blocks_refund_meaning(blob):
                return False
        except ImportError:
            pass
        if "refund" in blob and "delivery" not in blob and "service" not in blob:
            return False
        return True
    if intent in ("order", "general") and _ai_meaning_denies_order_tracking(blob):
        if not r.get("needs_order_id"):
            return True
    return False


def ai_meaning_describes_existing_order_track(route: dict | None) -> bool:
    """Live track/status of one order — Groq order_lookup_kind + intent, not keyword lists."""
    if not llm_semantic_route_available(route):
        return False
    r = route or {}
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    if olk == "track":
        return True
    if olk in ("details", "invoice", "none", ""):
        if olk in ("details", "invoice"):
            return False
    intent = (r.get("intent") or "").strip().lower()
    if intent in ("order", "refund", "payment") and r.get("needs_order_id"):
        if olk == "track":
            return True
        if ai_route_is_kb_read(r):
            return False
        channel = (r.get("data_channel") or "").strip().lower()
        return channel == "live_api"
    return False


def infer_semantic_goal_from_ai_route(ai_route: dict | None) -> str:
    """
    Customer goal from Groq JSON only — used when LLM routing succeeded.
    Empty when unclear; keyword layers must not invent a competing goal.
    """
    if not llm_semantic_route_available(ai_route):
        return ""
    route = dict(ai_route or {})
    intent = (route.get("intent") or "").strip().lower()
    olk = (route.get("order_lookup_kind") or "").strip().lower()

    if intent == "invoice":
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
    if olk == "invoice":
        return "order_invoice"
    if olk == "details":
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


def _default_kb_keys_for_intent(intent: str, existing: list[str]) -> list[str]:
    """Fill empty kb_keys only — Groq should set these; defaults are safety net."""
    keys = list(existing or [])
    intent = (intent or "").strip().lower()
    defaults: tuple[str, ...] = ()
    if intent == "seller":
        defaults = ("seller", "support")
    elif intent == "refund":
        defaults = ("refund", "faqs", "shipping")
    elif intent == "payment":
        defaults = ("payment", "faqs")
    elif intent == "general":
        defaults = ("faqs", "company")
    for k in defaults:
        if k not in keys:
            keys.append(k)
    return keys


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
    out["kb_keys"] = _default_kb_keys_for_intent(intent, list(out.get("kb_keys") or []))
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
    if (out.get("data_channel") or "").strip().lower() == "kb":
        return out
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
        if score >= 0.40 and top_key in (
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
