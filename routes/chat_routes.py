"""Public chat UI and JSON APIs."""
import json
import os
import re
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, current_app, g, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from services.answer_router import dispatch_early_answer, resolve_answer_route
from services.brain_direct_dispatch import (
    try_structural_order_live_reply,
)
from services.ai_service import ai_brain_answer, ai_brain_route
from services.embedding import GREETINGS, encode_texts, get_greetings_vecs
from services.kb_service import (
    INTERNAL_KB_KEYS,
    best_kb_hit,
    keyword_kb_hit,
    direct_kb_search,
    ensure_knowledge_cache_fresh,
    format_customer_care_reply_from_kb,
    format_knowledge_information_reply_from_kb,
    format_policy_help_reply_from_kb,
    format_welfog_about_reply_from_kb,
    get_customer_kb_keys,
    get_knowledge_context,
    get_support_contact_kb_keys,
    read_concatenated_kb_file_contents,
    smart_instant_router,
    sysmsg,
    _KB_SNAPSHOT,
)
from services.mysql_service import (
    db_get_recent_messages,
    db_store_message,
    generate_chat_token,
    get_mysql_connection,
    sql_collate,
)
from services.translation_service import (
    customer_reply_language,
    detect_language,
    finalize_customer_reply,
    is_hinglish_message,
    is_live_api_structured_html,
    localize_for_customer,
    localized_sysmsg_for_customer,
    resolve_customer_reply_lang,
    should_translate_reply,
    to_en,
)
from services.opensearch_products import (
    build_product_rail_with_pagination,
    cheapest_alternative_hint,
    apply_catalog_post_filters,
    sanitize_product_search_spec,
    format_filter_display_label,
    format_product_search_append_payload,
    has_structured_product_filters,
    normalize_color_fuzzy,
    search_opensearch_products,
    search_products_combined,
)
from services.conversation_followup import (
    apply_conversation_followup_fixes,
    classify_conversation_followup,
    pending_offer_from_greeting_key,
    sync_pending_offer_from_conversation,
)
from services.product_query_understanding import (
    clean_product_part_label,
    display_label_for_product_search,
    is_noisy_search_query,
    spec_uses_strict_filter_not_found,
    understand_product_query,
)
from services.welfog_api import (
    _normalize_color,
    collect_multi_product_parts,
    multi_product_parts_are_valid,
    repair_multi_product_joiners,
    search_multi_product_parts,
    check_pincode_delivery,
    fetch_api,
    fetch_welfog_order_tracking_for_user,
    format_order_tracking_reply,
    fetch_category_wise_feed,
    fetch_nav_categories,
    fetch_products_from_api,
    fetch_today_deals,
    format_purchase_history_append_payload,
    format_purchase_history_reply,
    format_wishlist_append_payload,
    format_wishlist_page_html,
    format_wishlist_reply,
    get_category_id_from_text,
    category_browse_search_name,
    category_name_for_id,
    build_welfog_product_browse_url,
    ensure_expanded_categories_map_for_ctx,
    format_inner_categories_reply,
    is_top_level_main_category,
    main_category_has_inner_children,
)
from utils.helpers import (
    _conversation_cache_suffix,
    _conversation_bot_offered_order_id_or_tracking,
    _conversation_bot_asked_for_pincode,
    _resolve_ambiguous_bare_numeric_context,
    _format_conversation_for_llm,
    _looks_like_browse_all_categories_message,
    _looks_like_conversational_followup,
    _looks_like_factual_identity_query,
    _looks_like_greeting_message,
    _looks_like_light_smalltalk,
    _is_light_smalltalk_fast,
    _is_short_pure_greeting,
    _is_welfog_about_fast_path,
    fast_warm_reply_html,
    _message_looks_like_shopping_query,
    _merge_extracted_pincode,
    _merge_embedded_identifiers_from_message,
    _text_has_delivery_or_order_area_intent,
    _text_has_platform_overview_intent,
    _text_has_order_placement_intent,
    _text_has_explicit_how_to_place_order,
    _text_has_product_shopping_intent,
    _text_has_product_shopping_intent_core,
    _turn_is_catalog_product_request,
    turn_is_catalog_product_lookup,
    _text_has_refund_or_return_intent,
    _text_is_order_id_help_request,
    _text_is_tracking_howto_request,
    _should_release_order_id_awaiting_for_routing,
    _text_is_order_tracking_intent,
    _text_needs_order_id_for_refund_or_payment,
    _text_needs_order_id_for_tracking,
    _text_asks_customer_care_contact,
    _text_asks_order_history,
    _text_asks_how_to_view_order_history,
    _user_asks_order_history_navigation_help,
    _user_clarifies_process_not_order_list,
    message_needs_human_support_escalation,
    _text_asks_how_to_view_wishlist,
    _text_asks_wishlist,
    message_is_wishlist_like_request,
    _text_requests_category_product_browse,
    _text_suggests_single_order_status_lookup,
    _text_is_live_order_lookup_intent,
    apply_hinglish_product_fixes,
    apply_category_product_route_fixes,
    apply_order_tracking_fixes,
    apply_product_id_vs_order_fixes,
    build_retrieval_query,
    message_is_generic_help_request,
    should_send_warm_feedback_reply,
    should_use_warm_conversational_reply,
    build_warm_feedback_reply,
    message_asks_my_welfog_purchases,
    message_is_casual_offtopic_not_shopping,
    message_is_knowledge_information_request,
    message_is_welfog_about_request,
    message_needs_policy_answer,
    message_needs_support_not_product,
    extract_order_id,
    extract_latest_order_id_from_user_conversation,
    extract_product_id,
    extract_product_search_query,
    _message_is_order_id_followup_submission,
    _message_submits_or_corrects_order_id,
    _conversation_in_order_tracking_flow,
    resolve_live_api_intent_from_conversation,
    resolve_order_id_for_tracking,
    should_attempt_live_order_api_reply,
    message_needs_live_single_order_lookup,
    _text_is_product_id_lookup_context,
    build_warm_conversation_reply,
    pick_warm_chat_reply_key,
    should_send_warm_greeting_reply,
    build_assistant_intro_reply,
    _is_assistant_identity_question,
    message_clarifies_wishlist_not_order_history,
    message_mentions_wishlist_topic,
    message_is_past_purchase_list_request,
    message_is_seller_on_welfog_request,
    message_is_user_confused_or_rephrasing_bot,
    message_wants_order_history_app_navigation,
    customer_turn_text,
    resolve_navigation_help_topic,
    reset_context,
    reset_context_unless_order_pending,
    user_contexts,
    _should_bypass_warm_greeting_fast_path,
)
from utils.reasoning_log import chat_log, log_reasoning
from utils.cache import _cache_get, _cache_set

DEFAULT_USER_ID = "STATIC_USER_001"


def _user_ctx_key(user_id: str, chat_id: str | None) -> str:
    """Isolate session state per chat — concurrent chats for one user must not share ctx."""
    uid = str(user_id or "").strip() or DEFAULT_USER_ID
    cid = (chat_id or "").strip()
    return f"{uid}:{cid}" if cid else uid


def _get_or_create_user_ctx(user_id: str, chat_id: str | None) -> dict:
    key = _user_ctx_key(user_id, chat_id)
    if key not in user_contexts:
        user_contexts[key] = {
            "intent": None,
            "awaiting": None,
            "data": {},
            "last": None,
            "order_id": None,
        }
    ctx = user_contexts[key]
    return ctx


def _bind_ctx_to_chat_id(user_id: str, chat_id: str, ctx: dict) -> dict:
    """First turn often has chat_id=None — move pending order state to chat-scoped key."""
    cid = (chat_id or "").strip()
    if not cid or not isinstance(ctx, dict):
        return ctx
    cid_key = _user_ctx_key(user_id, cid)
    uid_key = _user_ctx_key(user_id, None)
    if cid_key == uid_key:
        return ctx
    user_contexts[cid_key] = ctx
    if uid_key in user_contexts and user_contexts[uid_key] is ctx:
        user_contexts.pop(uid_key, None)
    return user_contexts[cid_key]


def _localized_sysmsg(key: str, user_msg: str, reply_lang: str = "en", **fmt) -> str:
    return localized_sysmsg_for_customer(key, user_msg, reply_lang=reply_lang, **fmt)


def _clarify_request_scope_reply(user_msg: str, reply_lang: str = "en") -> str:
    from services.translation_service import customer_facing_template

    return customer_facing_template(
        "clarify_request_scope",
        user_msg,
        reply_lang,
        fallback_en=(
            "Got it. Please tell me in one short line what you want right now: "
            "product search, order/refund/tracking (with Order ID), or general Welfog help."
        ),
    )


def _register_new_chat_session_mysql(user_id: str, user_msg: str, chat_token: str) -> None:
    """Persist new sidebar chat row — never block greeting; MySQL uses connect timeout."""
    title = (user_msg[:30] + "...") if len(user_msg) > 30 else (user_msg or "New chat")
    conn = get_mysql_connection()
    if not conn:
        log_reasoning("MySQL unavailable; ephemeral chat session (no sidebar row).")
        return
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, title, chat_token, customer_id) VALUES (%s, %s, %s, %s)",
                    (str(user_id), title, chat_token, str(user_id)),
                )
            except Exception as e:
                if "customer_id" in str(e).lower():
                    cur.execute(
                        "INSERT INTO chat_sessions (user_id, title, chat_token) VALUES (%s, %s, %s)",
                        (str(user_id), title, chat_token),
                    )
                else:
                    raise
        conn.commit()
    except Exception as e:
        print(f"New chat session insert error: {e}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _is_low_information_turn_for_task_routing(
    original_msg: str, msg_en: str = "", ctx: dict | None = None
) -> bool:
    text = (original_msg or msg_en or "").strip().lower()
    if not text:
        return False
    if isinstance(ctx, dict):
        last = (ctx.get("last") or "").strip().lower()
        if last in ("product", "order", "refund", "invoice", "order_history", "wishlist"):
            return False
        if ctx.get("awaiting") in ("order_id", "pincode"):
            return False
    if re.search(r"\d{4,}", text):
        return False
    # If a concrete support/shopping signal exists, this is not low-information.
    if re.search(
        r"\b(order|refund|return|track|tracking|delivery|pincode|wishlist|history|product|cover|shirt|umbrella|deal|category|payment|seller|policy|privacy|terms|support|buy|show|search|price)\b",
        text,
    ):
        return False
    tokens = re.findall(r"[a-z0-9]+", text)
    if len(tokens) > 10:
        return False
    vague_phrases = (
        "ek kaam tha",
        "ek baat thi",
        "sun bhai",
        "meri baat sun",
        "dhyan se",
        "sun na",
        "haan bhai",
        "haan bolo",
        "hmm",
        "ok bhai",
        "acha",
        "accha",
        "theek",
        "thik",
    )
    return len(tokens) <= 3 or any(p in text for p in vague_phrases)


def _reply_for_live_order_id_lookup(
    live_intent: str,
    order_id: str,
    user_id: str,
    original_msg: str,
    reply_lang: str,
) -> str:
    """Live refund / payment / tracking API after user pasted Order ID."""
    lang = reply_lang or "en"
    if live_intent == "refund":
        from services.refund_status_flow import _fetch_and_format_refund_status

        return _fetch_and_format_refund_status(
            order_id,
            user_id,
            original_msg,
            lang,
            source="live_order_id_lookup",
        )
    if live_intent == "payment":
        res = fetch_api("payment", order_id, user_id=user_id)
        if res and res.get("status"):
            return f"Payment status: {res['status']} via {res['method']}."
        return "Payment details not found for this order in your account."
    from services.order_details_flow import log_order_data_pipeline

    log_reasoning(f"Fetching live order track for id={order_id} (user_id={user_id} ownership check)")
    log_order_data_pipeline(
        action="track_single_order",
        source="live_api",
        api="welfog_track",
        focus="timeline",
        order_id=str(order_id),
        fields=["order_id", "status", "timeline", "payment", "product", "eta"],
    )
    track_data, track_err = fetch_welfog_order_tracking_for_user(order_id, user_id)
    if track_data:
        return format_order_tracking_reply(track_data, order_id, lang=lang)
    if track_err == "login_required":
        return _localized_sysmsg("order_track_login_required", original_msg, reply_lang=lang) or (
            "Sorry — please log in to Welfog and open this chat from your account "
            "so we can show your order status safely."
        )
    if track_err == "not_owned":
        return _localized_sysmsg("order_track_not_owned", original_msg, reply_lang=lang) or (
            "Sorry — this Order ID doesn't seem linked to your account. "
            "Please share the ID from your own order SMS/email or My Orders."
        )
    if track_err == "unverified":
        return _localized_sysmsg("order_track_unverified", original_msg, reply_lang=lang) or (
            "Sorry — I couldn't verify this Order ID with your logged-in account from here. "
            "Please check My Orders in the Welfog app, or try again with the account that placed the order."
        )
    return _localized_sysmsg("order_track_not_found", original_msg, reply_lang=lang) or (
        "Sorry — we couldn't find that Order ID. "
        "Please check your confirmation SMS/email or My Orders and try again."
    )


def _ctx_has_order_pending(ctx: dict) -> bool:
    if not isinstance(ctx, dict):
        return False
    if ctx.get("awaiting") == "order_id":
        return True
    pending = ((ctx.get("data") or {}).get("pending_action") or "").strip().lower()
    return pending in (
        "track",
        "order_invoice",
        "order_details",
        "refund_status",
        "payment",
    )


def _conversation_snapshot_for_chat(chat_id: str | None, *, limit: int = 8) -> str:
    """Load recent chat from DB for bare-numeric disambiguation (survives in-memory ctx loss)."""
    if not chat_id:
        return ""
    try:
        msgs = db_get_recent_messages(chat_id, limit)
        return _format_conversation_for_llm(msgs, max_turns=limit)
    except Exception:
        return ""


def _message_is_complete_standalone_question(comb: str) -> bool:
    """Complete turn — no MySQL history needed (structural, not keyword routing)."""
    t = (comb or "").strip()
    if not t:
        return False
    words = re.findall(r"\S+", t)
    if len(words) >= 6:
        return True
    if len(words) >= 4 and ("?" in t or "？" in t or "।" in t):
        return True
    return False


def _conversation_context_for_routing(
    chat_id: str | None,
    original_msg: str,
    msg_en: str,
    ctx: dict | None,
    *,
    limit: int = 4,
) -> str:
    """
    Load recent chat only when the current turn is incomplete or session-bound.
    Never blanket-load history — fresh complete requests route without stale bleed.
    """
    if not chat_id:
        return ""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if _message_is_complete_standalone_question(comb):
        return ""
    try:
        from services.ai_first_router import _text_is_obvious_off_topic

        if comb and _text_is_obvious_off_topic(comb):
            return ""
    except ImportError:
        pass
    needs_ctx = False
    if isinstance(ctx, dict):
        if ctx.get("awaiting") in ("order_id", "pincode"):
            needs_ctx = True
        if ctx.get("last"):
            needs_ctx = True
        data = ctx.get("data") if isinstance(ctx.get("data"), dict) else {}
        if (data.get("pending_action") or "").strip():
            needs_ctx = True
    if not needs_ctx and comb:
        try:
            from services.query_understanding import _is_short_or_vague_message

            if _is_short_or_vague_message(comb):
                needs_ctx = True
        except ImportError:
            pass
        if not needs_ctx and re.search(
            r"\b(?:is|us|uska|uske|iska|iski|yeh|ye|wo|same|wahi|pehle|previous|above)\b",
            comb,
            re.I,
        ):
            needs_ctx = True
        if not needs_ctx:
            try:
                from utils.helpers import extract_embedded_query_identifiers

                ids = extract_embedded_query_identifiers(
                    original_msg, msg_en, "", ai_route=None
                )
                if not ids.get("order_id") and re.search(r"\b\d{4,20}\b", comb):
                    needs_ctx = False
                elif not ids.get("order_id") and not ids.get("pincode"):
                    if len(comb) <= 48:
                        needs_ctx = True
            except ImportError:
                pass
    if needs_ctx and _message_is_complete_standalone_question(comb):
        needs_ctx = False
    if not needs_ctx:
        if _turn_has_structural_fast_lane_token(original_msg, msg_en, ""):
            return ""
        if _message_is_complete_standalone_question(comb):
            return ""
        if len(comb) > 72:
            return ""
    snap = _conversation_snapshot_for_chat(chat_id, limit=limit)
    if snap:
        log_reasoning(
            f"Routing context loaded ({limit} turns) — incomplete turn or session follow-up."
        )
    return snap


def _transactional_turn_skip_kb_scope_preflight(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """
    Structural tokens only — skip KB scope micro-LLMs for bare PIN/order/pro_id turns.
    Intent for natural language is always ai_brain_route (any language, no keyword lists).
    """
    return _turn_has_structural_fast_lane_token(
        original_msg, msg_en, conversation_context
    )


_WEFOG_ORDER_ID_ATTACHED_RE = re.compile(
    r"(?<!\d)(?P<id>260\d{4,7})(?!\d)",
    re.IGNORECASE,
)
_GLUED_ORDER_ID_RE = re.compile(
    r"(?<!\d)(?P<id>\d{4,20})(?=[a-zA-Z]{1,12})",
)
_ALNUM_BOUNDARY_RE = re.compile(
    r"(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)"
)


def _normalize_alphanumeric_id_boundaries(text: str) -> str:
    """
    Split glued digit/letter runs: '2606252is id' → '2606252 is id'.
    Word-boundary \\b fails when IDs touch letters — normalize before routing.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    normalized = _ALNUM_BOUNDARY_RE.sub(" ", raw)
    if normalized != raw:
        log_reasoning(
            f"Alphanumeric guard — separated glued id/text: {raw[:48]!r} → {normalized[:56]!r}"
        )
    return normalized


def _turn_has_structural_fast_lane_token(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """
    Unambiguous tokens only — bare PIN, order id, catalog pro_id in message.
    No phrase/keyword intent routing; ai_brain_route handles all natural language.
    """
    raw = (original_msg or "").strip()
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not raw:
        return False
    if re.fullmatch(r"[1-9]\d{5}", raw):
        return True
    if isinstance(ctx, dict) and ctx.get("awaiting") in ("order_id", "pincode"):
        if re.fullmatch(r"\d{4,20}", raw):
            return True
    for pat in (_WEFOG_ORDER_ID_ATTACHED_RE, _GLUED_ORDER_ID_RE):
        if pat.search(raw):
            return True
    if re.search(
        r"\b(?:pro[_\s-]?id|product[_\s-]?id|pid)\s*[:\-#]?\s*\d{4,12}\b",
        comb,
        re.I,
    ):
        return True
    bare = raw.strip()
    if bare and " " not in bare:
        try:
            from services.catalog_spec_semantics import user_mentions_sku_this_turn

            if user_mentions_sku_this_turn(bare):
                return True
        except ImportError:
            pass
    return False


def _turn_should_prioritize_brain_route(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """
    Default AI-first for natural-language turns (any language/style).
    Only bypass brain-first for truly structured turns like bare ids/pins
    or explicit single-order live-id lookups.
    """
    raw = (original_msg or "").strip()
    if not raw:
        return True
    # Pure numeric follow-ups (order-id/pincode) should stay deterministic.
    if re.fullmatch(r"[1-9]\d{5,19}", raw):
        return False
    # Keep ultra-fast live path only when this turn clearly includes order-id + live goal.
    if _guard_quick_single_order_live_turn(raw):
        return False
    # Everything else uses one universal brain route first.
    return True


def _instant_lane_blocked_by_transactional(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Block instant greeting only when same turn has structural PIN/order/pro_id token."""
    return _turn_has_structural_fast_lane_token(
        original_msg, msg_en, conversation_context
    )


def _extract_attached_welfog_order_id(
    text: str, conversation_context: str = ""
) -> str:
    """
    Pull Welfog order ids from glued Hinglish: '2606252is id ki invoice'.
    Word-boundary \\b fails when digits touch letters — scan raw numerics instead.
    """
    try:
        from utils.helpers import (
            _is_plausible_order_id,
            _normalize_order_chat_text,
            extract_order_id,
        )

        raw = _normalize_order_chat_text(text or "")
        if not raw:
            return ""
        ctx = f"{raw} {(conversation_context or '')[-800:]}"
        for pat in (_WEFOG_ORDER_ID_ATTACHED_RE, _GLUED_ORDER_ID_RE):
            for m in pat.finditer(raw):
                cand = (m.group("id") or "").strip()
                if cand and _is_plausible_order_id(cand, context=ctx, shallow=True):
                    return cand
        oid = extract_order_id(raw, conversation_context)
        return (oid or "").strip()
    except ImportError:
        m = _WEFOG_ORDER_ID_ATTACHED_RE.search(text or "")
        return (m.group("id") or "").strip() if m else ""


def _turn_bypasses_embedding_vector(
    original_msg: str,
    msg_en: str = "",
    ctx: dict | None = None,
    conversation_context: str = "",
) -> bool:
    """
    Transactional / chitchat / order-id turns must never wait on SentenceTransformer.encode.
    """
    if _transactional_turn_skip_kb_scope_preflight(
        original_msg, msg_en, conversation_context
    ):
        return True
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return False
    try:
        from services.conversation_followup import is_deals_request_message
        from utils.helpers import message_asks_welfog_categories_list

        if is_deals_request_message(original_msg, msg_en):
            return True
        if message_asks_welfog_categories_list(comb):
            return True
        from utils.helpers import _text_asks_customer_care_contact

        if _text_asks_customer_care_contact(comb):
            return True
        from utils.helpers import message_asks_welfog_social_media

        if message_asks_welfog_social_media(comb, conversation_context=conversation_context):
            return True
    except ImportError:
        pass
    try:
        from services.chitchat_resolver import _is_instant_greeting_thanks_lane

        if _is_instant_greeting_thanks_lane(original_msg, msg_en, conversation_context):
            return True
    except ImportError:
        pass
    try:
        from services.chat_flow_telemetry import get_cached_brain_route

        brain = get_cached_brain_route()
        if isinstance(brain, dict) and brain.get("_universal_brain_route"):
            return True
    except ImportError:
        pass
    if _extract_attached_welfog_order_id(comb, conversation_context):
        return True
    try:
        from utils.helpers import (
            _text_is_live_order_lookup_intent,
            _text_is_order_tracking_intent,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )

        if (
            message_is_past_purchase_list_request(comb)
            or message_is_wishlist_like_request(comb)
            or _text_is_order_tracking_intent(comb)
            or _text_is_live_order_lookup_intent(comb, conversation_context)
        ):
            return True
    except ImportError:
        pass
    return False


def _sanitize_stale_ctx_for_fresh_intent(
    original_msg: str,
    msg_en: str,
    ctx: dict | None,
    *,
    conversation_context: str = "",
) -> None:
    """Drop stale order/pincode session tags when user starts a new intent lane."""
    if not isinstance(ctx, dict):
        return
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return
    stale = False
    try:
        from utils.helpers import (
            _text_is_delivery_serviceability_hypothetical,
            _text_is_pincode_serviceability_question,
            clear_order_session_for_new_lookup,
            message_asks_welfog_categories_list,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
            turn_is_obvious_product_shopping_turn,
        )
        from services.conversation_followup import is_deals_request_message

        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conversation_context
        ):
            stale = True
        elif _text_is_pincode_serviceability_question(
            comb, conversation_context
        ) or _text_is_delivery_serviceability_hypothetical(comb):
            stale = True
        elif is_deals_request_message(original_msg, msg_en):
            stale = True
        elif message_asks_welfog_categories_list(comb):
            stale = True
        elif message_is_past_purchase_list_request(comb) or message_is_wishlist_like_request(
            comb
        ):
            stale = True
        if stale:
            pending = (
                (ctx.get("data") or {}).get("pending_action") or ""
            ).strip().lower()
            if pending or ctx.get("awaiting") in ("order_id", "pincode"):
                log_reasoning(
                    f"Stale ctx sanitized — fresh intent clears pending={pending or ctx.get('awaiting')}."
                )
                clear_order_session_for_new_lookup(ctx)
                ctx["awaiting"] = None
                ctx["last"] = None
                data = ctx.get("data")
                if isinstance(data, dict):
                    data.pop("topic_mode", None)
                    data.pop("pending_offer", None)
    except ImportError:
        pass


def _try_pre_brain_live_api_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    conv_for_llm: str,
) -> tuple[str, dict] | None:
    """
    AI micro-classifiers + text KB — before slow universal brain.
    Order: text KB → delivery → knowledge KB → account list (any language).
    """
    try:
        from services.chat_flow_telemetry import (
            begin_pre_brain_live_api_preflight,
            end_pre_brain_live_api_preflight,
        )

        begin_pre_brain_live_api_preflight()
        try:
            text_kb = _try_instant_text_kb_reply(
                original_msg, msg_en, lang, conversation_context=conv_for_llm
            )
            if text_kb:
                log_reasoning("Pre-brain — text KB (support/social/policy, no embedding).")
                return text_kb, {
                    "intent": "general",
                    "data_channel": "kb",
                    "route_handler": "kb_text_preflight",
                }

            from services.account_list_fast_path import (
                try_account_list_fast_reply,
                try_account_list_fast_route,
            )
            from services.kb_service import sysmsg

            acct_body = try_account_list_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                user_id,
                lang,
                ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                localized_sysmsg=_localized_sysmsg,
                sysmsg=sysmsg,
                reset_context_fn=reset_context,
            )
            if acct_body:
                fast_pair = try_account_list_fast_route(
                    original_msg, msg_en, conv_for_llm, lang, ctx
                )
                route = fast_pair[1] if fast_pair else {
                    "intent": "order_history",
                    "data_channel": "live_api",
                    "route_handler": "account_list_ai_preflight",
                }
                log_reasoning(
                    "Pre-brain — account-list AI → live API (before delivery/product)."
                )
                return acct_body, route

            from services.knowledge_fast_path import try_knowledge_fast_reply

            kb_body = try_knowledge_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=lang,
            )
            if kb_body:
                log_reasoning(
                    "Pre-brain — knowledge AI → vector KB (refund/policy/FAQ)."
                )
                return kb_body, {
                    "intent": "general",
                    "data_channel": "kb",
                    "route_handler": "knowledge_ai_preflight",
                }

            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            pin_body = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                ctx,
                reset_context_fn=reset_context,
                skip_turn_check=False,
                allow_llm=True,
            )
            if pin_body:
                log_reasoning(
                    "Pre-brain — delivery AI → geocode/PIN live API (any language)."
                )
                return pin_body, (ctx.get("data") or {}).get("ai_route") or {
                    "intent": "pincode_check",
                    "data_channel": "live_api",
                    "route_handler": "pincode_delivery_ai_preflight",
                }

            from services.product_catalog_resolver import try_product_ai_first_catalog

            prod_hit = try_product_ai_first_catalog(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
                user_id=str(user_id),
            )
            if prod_hit:
                prod_body, prod_route = prod_hit[0], prod_hit[1] if len(prod_hit) > 1 else {}
                if (prod_body or "").strip():
                    log_reasoning(
                        "Pre-brain — product AI → catalog (after account/delivery)."
                    )
                    return prod_body, prod_route or {
                        "intent": "product",
                        "data_channel": "catalog",
                        "route_handler": "product_ai_preflight",
                    }

        finally:
            end_pre_brain_live_api_preflight()
    except ImportError:
        pass
    except Exception as exc:
        log_reasoning(f"Pre-brain AI preflight skip (non-fatal): {exc}")
    return None


def _try_brain_first_chat_turn(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    conv_for_llm: str,
    chat_id: str | None,
) -> tuple[str | None, dict | None]:
    """
    One ai_brain_route LLM — semantic intent in any language/style, then dispatch.
    No keyword pre-routing; Groq/OpenAI/Gemini/DeepSeek classify meaning first.
    """
    t0 = time.perf_counter()
    log_reasoning(
        "Chat — AI brain first (one LLM intent → API/KB/catalog/chitchat)."
    )
    t_early = time.perf_counter()
    body, route = _try_early_ai_brain_reply(
        original_msg,
        msg_en,
        user_id,
        lang,
        ctx,
        conv_for_llm=conv_for_llm,
        chat_id=chat_id,
    )
    if body:
        try:
            from services.chat_flow_telemetry import record_phase

            record_phase("ai_brain_universal", (time.perf_counter() - t0) * 1000.0)
        except ImportError:
            pass
        return body, route
    if isinstance(route, dict) and not route.get("llm_unavailable"):
        try:
            from services.brain_direct_dispatch import try_finish_brain_classified_turn

            finish_body = try_finish_brain_classified_turn(
                route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if finish_body:
                try:
                    from services.chat_flow_telemetry import record_phase

                    record_phase(
                        "ai_brain_universal", (time.perf_counter() - t0) * 1000.0
                    )
                except ImportError:
                    pass
                return finish_body, route
        except ImportError:
            pass
    return None, route


def _try_scope_ood_fast_lane_reply(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
) -> tuple[str, dict] | None:
    """Scope LLM fallback when brain LLM unavailable — not primary routing."""
    try:
        from services.chitchat_resolver import try_scope_ai_early_reply

        return try_scope_ai_early_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            reply_lang=lang,
            preflight=True,
        )
    except ImportError:
        return None


def _try_instant_text_kb_reply(
    original_msg: str,
    msg_en: str,
    lang: str,
    conversation_context: str = "",
) -> str | None:
    """Keyword/file KB only — zero SentenceTransformer (no encode lock)."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    try:
        from services.conversation_followup import is_deals_request_message
        from utils.helpers import (
            message_asks_welfog_social_media,
            message_is_knowledge_information_request,
            message_is_welfog_about_request,
        )
        from services.kb_service import (
            format_knowledge_information_reply_from_kb,
            format_welfog_about_reply_from_kb,
            format_welfog_social_media_reply_from_kb,
        )

        if is_deals_request_message(original_msg, msg_en):
            return None

        if message_asks_welfog_social_media(comb, conversation_context=conversation_context):
            body = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
            )
            if body:
                log_reasoning(
                    "Instant lane — Welfog social links text KB (no embedding)."
                )
                return body

        if message_is_welfog_about_request(comb):
            body = format_welfog_about_reply_from_kb(
                original_msg, msg_en, reply_lang=lang
            )
            if body:
                log_reasoning(
                    "Instant lane — text KB about Welfog (no embedding)."
                )
                return body
        if message_is_knowledge_information_request(comb, conversation_context):
            body = format_knowledge_information_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
            )
            if body:
                log_reasoning(
                    "Instant lane — text KB policy/info (no embedding)."
                )
                return body
        from utils.helpers import _text_asks_customer_care_contact

        if _text_asks_customer_care_contact(comb):
            from services.kb_service import (
                format_direct_reply_from_kb_hit,
                read_concatenated_kb_file_contents,
            )

            raw = read_concatenated_kb_file_contents(["support"])
            if raw:
                hit = {"source": "support", "chunk": raw[:1200], "score": 0.9}
                body = format_direct_reply_from_kb_hit(
                    hit, original_msg, reply_lang=lang, fast_lane=True
                )
                if body:
                    log_reasoning(
                        "Instant lane — support contact text KB (no embedding)."
                    )
                    return body
    except ImportError:
        pass
    except Exception as exc:
        log_reasoning(f"Instant text KB skip (non-fatal): {exc}")
    return None


def _should_skip_kb_ood_preflight_for_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    lang: str = "",
) -> bool:
    """Skip KB/OOD preflight for catalog/API turns — one brain route, no duplicate LLMs."""
    if _transactional_turn_skip_kb_scope_preflight(
        original_msg, msg_en, conversation_context
    ):
        return True
    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            kb_classified_as_live_api,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(
            original_msg, msg_en, conversation_context, lang
        )
        if peeked is not _KB_CACHE_UNSET and kb_classified_as_live_api(peeked):
            log_reasoning(
                "KB/OOD preflight skip — AI classified live API/catalog (cached)."
            )
            return True
        try:
            from services.turn_intent_coordinator import (
                kb_classified_as_product_catalog,
            )

            if peeked is not _KB_CACHE_UNSET and kb_classified_as_product_catalog(
                peeked
            ):
                log_reasoning(
                    "KB/OOD preflight skip — AI product_search (OpenSearch lane)."
                )
                return True
        except ImportError:
            pass
    except ImportError:
        pass
    return False


def _try_non_transactional_kb_ood_preflight(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    lang: str,
) -> tuple[str | None, dict | None]:
    """
    Non-shopping turns: AI infers meaning + topic, then vector KB; scope AI for OOD.
    Runs before ai_brain_route + enrich (no keyword routing).
    """
    try:
        from utils.helpers import message_asks_welfog_social_media
        from services.kb_service import format_welfog_social_media_reply_from_kb

        comb = f"{original_msg or ''} {msg_en or ''}".strip()
        if message_asks_welfog_social_media(comb, conversation_context=conversation_context):
            social = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
            )
            if social:
                log_reasoning("KB preflight — Welfog social links from company KB.")
                return social, {
                    "intent": "general",
                    "data_channel": "kb",
                    "route_handler": "welfog_social_kb",
                }
    except ImportError:
        pass

    try:
        from services.knowledge_query_pipeline import try_ai_first_kb_early_reply

        conv = (conversation_context or "").strip()
        kb_body = try_ai_first_kb_early_reply(
            original_msg,
            msg_en,
            conv,
            reply_lang=lang,
            preflight=True,
        )
        if kb_body and kb_body.strip():
            log_reasoning("KB/OOD preflight — AI classifier + vector KB (admin files).")
            return kb_body, {
                "intent": "general",
                "data_channel": "kb",
                "route_handler": "kb_ai_first_preflight",
            }
    except ImportError:
        pass

    try:
        from services.knowledge_fast_path import try_knowledge_fast_reply

        conv = (conversation_context or "").strip()
        kb_fast = try_knowledge_fast_reply(
            original_msg,
            msg_en,
            conv,
            reply_lang=lang,
        )
        if kb_fast and kb_fast.strip():
            log_reasoning("KB/OOD preflight — knowledge fast path (AI + vector KB).")
            return kb_fast, {
                "intent": "general",
                "data_channel": "kb",
                "route_handler": "knowledge_fast_preflight",
            }
    except ImportError:
        pass

    try:
        from services.ai_first_router import _try_obvious_out_of_domain_route

        ood_route = _try_obvious_out_of_domain_route(
            original_msg, msg_en, reply_lang=lang
        )
        if ood_route and (ood_route.get("scope_reply") or "").strip():
            log_reasoning(
                "KB/OOD preflight — obvious OOD template (no vector/KB scan)."
            )
            return ood_route["scope_reply"], ood_route
    except ImportError:
        pass

    # kb_turn / scope micro-LLMs removed — one ai_brain_route per turn handles all intent.

    try:
        from services.chitchat_resolver import try_scope_ai_early_reply

        conv = (conversation_context or "").strip()
        scope_hit = try_scope_ai_early_reply(
            original_msg,
            msg_en,
            conv,
            reply_lang=lang,
            preflight=True,
        )
        if scope_hit:
            scope_body, scope_route = scope_hit
            if scope_body and scope_body.strip():
                log_reasoning("KB/OOD preflight — scope AI (chitchat/OOD, user language).")
                return scope_body, scope_route
    except ImportError:
        pass

    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            kb_classified_as_live_api,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(
            original_msg, msg_en, conversation_context, lang
        )
        if peeked is not _KB_CACHE_UNSET:
            if kb_classified_as_live_api(peeked):
                log_reasoning(
                    "KB/OOD preflight — live API turn; skip scope, brain/catalog handles."
                )
                return None, None
            conf = float((peeked or {}).get("confidence") or 0.0)
            if (
                isinstance(peeked, dict)
                and not peeked.get("is_informational_kb")
                and conf >= 0.48
            ):
                log_reasoning(
                    "KB/OOD preflight — AI ruled non-KB; skip scope LLM."
                )
                return None, None
    except ImportError:
        pass

    return None, None


def _guard_skip_kb_ood_preflight(
    original_msg: str,
    msg_en: str,
    chat_id: str | None,
    user_id: str,
) -> bool:
    """
    Skip guard KB/OOD for catalog/API/shopping — brain + live API handle those turns.
    """
    try:
        from utils.helpers import message_is_bare_numeric_submission

        if message_is_bare_numeric_submission(original_msg):
            return True
    except ImportError:
        pass
    return _transactional_turn_skip_kb_scope_preflight(original_msg, msg_en, "")


def _guard_scope_ai_preflight_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    lang_hint: str = "",
):
    """
    Scope LLM before heavy chat() — chitchat / OOD in the customer's language.
    Semantic detection only (no fixed keyword lists or static greeting templates).
    """
    original_msg = _normalize_alphanumeric_id_boundaries((user_msg or "").strip())
    if not original_msg or len(original_msg) > 160:
        return None
    if _guard_quick_purchase_list_turn(original_msg) or _guard_quick_wishlist_turn(
        original_msg
    ):
        return None

    try:
        from services.translation_service import (
            resolve_customer_reply_lang,
            to_en_for_routing,
        )

        glang = (lang_hint or "").strip().lower() or resolve_customer_reply_lang(
            original_msg
        )
        msg_en = to_en_for_routing(original_msg, glang)
    except ImportError:
        glang = (lang_hint or "").strip().lower() or customer_reply_language(
            original_msg
        )
        if glang in ("en", "hinglish"):
            msg_en = original_msg.lower()
        else:
            try:
                msg_en = to_en(original_msg).lower().strip()
            except Exception:
                msg_en = original_msg.lower()

    if _turn_has_structural_fast_lane_token(original_msg, msg_en, ""):
        return None

    comb = f"{original_msg} {msg_en}".strip()
    try:
        from services.conversation_scope import _has_definite_welfog_shopping_signal

        if _has_definite_welfog_shopping_signal(comb):
            return None
    except ImportError:
        pass

    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            kb_turn_is_informational,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(original_msg, msg_en, "", glang)
        if peeked is not _KB_CACHE_UNSET and kb_turn_is_informational(peeked, min_conf=0.42):
            return None
    except ImportError:
        pass

    conv = _conversation_snapshot_for_chat(chat_id, limit=6)
    lang = glang or customer_reply_language(original_msg)
    hit = _try_scope_ood_fast_lane_reply(original_msg, msg_en, conv, lang)
    if not hit:
        return None
    body, _route = hit
    if not (body or "").strip():
        return None

    log_reasoning("Guard — scope AI preflight (chitchat/OOD, user language).")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_prelock_instant_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    lang_hint: str,
):
    """Disabled — LLM preflight moved into chat() single path (faster guard)."""
    return None


def _guard_order_history_typos_norm(comb: str) -> str:
    """Inline typo normalize — avoid importing utils.helpers on guard hot path."""
    s = (comb or "").lower()
    for wrong, right in (
        ("hstory", "history"),
        ("histroy", "history"),
        ("histry", "history"),
        ("histery", "history"),
        ("hostory", "history"),
        ("histoey", "history"),
        ("hidtory", "history"),
    ):
        s = s.replace(wrong, right)
    s = re.sub(r"\bh\w*stor\w*\b", "history", s)
    s = re.sub(r"\bhist[oaeiu]{0,3}r[a-z]{0,4}\b", "history", s)
    return s


def _guard_quick_single_order_live_turn(text: str) -> bool:
    """Order ID + invoice/details/track — NOT purchase-history list."""
    raw = (text or "").strip()
    if not raw:
        return False
    if not re.search(r"\b[0-9]{4,20}\b", raw):
        return False
    tl = f" {raw.lower()} "
    if re.search(r"\b(?:invoice|invoce|bill|receipt|chalan|challan)\b", tl):
        return True
    if re.search(r"\b(?:detail|details)\b", tl) and "order" in tl:
        return True
    if re.search(r"\b(?:track|tracking|status|stutus|refund)\b", tl):
        return True
    if re.search(r"\border\s+id\b", tl):
        return True
    return False


def _guard_quick_policy_refund_howto(tl: str) -> bool:
    """General refund/return policy — KB, not personal status lookup."""
    if any(
        x in tl
        for x in ("policy", "kaise kare", "kaise karu", "how to", "process", "procedure")
    ):
        return True
    if "return policy" in tl or "refund policy" in tl:
        return True
    if re.search(r"\breturn\b", tl) and not any(
        m in tl
        for m in (
            "refund",
            "nhi aaya",
            "nahi aaya",
            "nhi aa",
            "kab",
            "status",
            "milega",
            "aaya",
            "received",
        )
    ):
        return True
    return False


def _guard_quick_is_purchase_history_list_only(tl: str) -> bool:
    """List-all-orders phrasing — not single-order refund/track/invoice."""
    if any(
        x in tl
        for x in (
            "order history",
            "purchase history",
            "my orders",
            "meri order",
            "mere order",
            "orders history",
            "order hist",
        )
    ):
        if not any(
            m in tl
            for m in ("refund", "invoice", "track", "detail", "kab", "nhi", "nahi")
        ):
            return True
    if "purchased" in tl and "order" in tl:
        return True
    if re.search(r"\b(?:meri|mere|my)\s+orders?\b", tl) and any(
        v in tl for v in ("dikha", "data", "history", "list", "bta", "show", "dega")
    ):
        if not any(
            m in tl
            for m in ("refund", "invoice", "track", "detail", "kab", "nhi", "nahi")
        ):
            return True
    return False


def _guard_quick_live_goal_no_order_id(text: str) -> str:
    """
    Personal single-order live API goal without Order ID digits in message.
    Returns: refund_status | order_invoice | order_details | track | ''.
    """
    raw = (text or "").strip()
    if not raw or re.search(r"\b[0-9]{4,20}\b", raw):
        return ""
    tl = f" {raw.lower()} "
    if _guard_quick_is_purchase_history_list_only(tl):
        return ""
    if "refund" in tl or "refnd" in tl:
        if not _guard_quick_policy_refund_howto(tl):
            return "refund_status"
    if re.search(r"\b(?:invoice|invoce|bill|receipt)\b", tl):
        if "order" in tl or "id" in tl or "bhej" in tl or "dega" in tl or "de de" in tl:
            return "order_invoice"
    if re.search(r"\b(?:detail|details)\b", tl) and "order" in tl:
        return "order_details"
    if re.search(r"\b(?:track|tracking|status|stutus)\b", tl) and "order" in tl:
        return "track"
    if re.search(r"\border\b", tl) and any(
        m in tl
        for m in (
            "nhi aaya",
            "nahi aaya",
            "nhi aa",
            "nahi aa",
            "kab aa",
            "kab tak",
            "kb tk",
            "not received",
            "not arrived",
            "pahunch",
            "pahucha",
            "abhi tk",
            "ab tak",
        )
    ):
        return "track"
    if re.search(r"\b(?:mera|mere|meri|my)\s+order", tl) and "refund" in tl:
        return "refund_status"
    return ""


def _guard_quick_purchase_list_turn(text: str) -> bool:
    """
    Lightweight purchase-history list detector for guard fast paths only.
    Avoids heavy helpers that can recurse or block on catalog/KB signals.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if _guard_quick_single_order_live_turn(raw):
        return False
    if _guard_quick_live_goal_no_order_id(raw):
        return False
    tl = f" {raw.lower()} "
    if "wishlist" in tl or "wish list" in tl:
        return False
    if any(
        x in tl
        for x in (
            "order history",
            "purchase history",
            "my orders",
            "meri order",
            "mere order",
            "orders history",
            "order hist",
        )
    ):
        return True
    if "purchased" in tl and "order" in tl:
        return True
    if re.search(r"\b(?:meri|mere|my)\s+orders?\b", tl) and any(
        v in tl for v in ("dikha", "data", "history", "list", "bta", "show", "dega")
    ):
        return True
    return False


def _guard_quick_wishlist_turn(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    return "wishlist" in tl or "wish list" in tl or (
        "heart" in tl and any(x in tl for x in ("list", "saved", "like", "liked"))
    )


def _fast_guard_ood_template_reply(
    original_msg: str,
    lang: str = "en",
    msg_en: str = "",
    conv: str = "",
    *,
    scope: str = "out_of_domain",
) -> str:
    """AI-generated scope reply in customer's language; static template only if LLM unavailable."""
    scope_l = (scope or "out_of_domain").strip().lower()
    try:
        from services.conversational_ack_flow import ai_natural_scope_reply

        body = ai_natural_scope_reply(
            scope_l,
            original_msg,
            msg_en,
            conv,
            reply_lang=lang,
        )
        if body and str(body).strip():
            log_reasoning(f"Scope AI reply ({scope_l}) — no static template.")
            return str(body)
    except ImportError:
        pass
    try:
        from services.kb_service import sysmsg
        from services.translation_service import (
            finalize_customer_reply,
            is_hinglish_message,
            resolve_customer_reply_lang,
        )

        rl = resolve_customer_reply_lang(original_msg, lang)
        if scope_l == "general_chitchat":
            key = "greeting_variant_2" if is_hinglish_message(original_msg) or rl == "hinglish" else "greeting"
            tpl = sysmsg(key) or sysmsg("greeting") or ""
        elif scope_l == "harm_sensitive":
            key = (
                "harm_safety_hinglish"
                if is_hinglish_message(original_msg) or rl == "hinglish"
                else "harm_safety"
            )
            tpl = sysmsg(key) or sysmsg("harm_safety") or ""
        else:
            key = (
                "out_of_domain_hinglish"
                if is_hinglish_message(original_msg) or rl == "hinglish"
                else "out_of_domain"
            )
            tpl = sysmsg(key) or sysmsg("out_of_domain") or ""
        out = finalize_customer_reply(tpl, original_msg, rl) if tpl else ""
        return out
    except ImportError:
        return ""


def _try_guard_misclassified_general_kb_ood(
    brain: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv: str,
    lang: str,
) -> str | None:
    """
    Brain misrouted obvious off-topic (loan, coding, etc.) to general+KB.
    Structural Welfog signals only — no customer keyword lists.
    """
    if not isinstance(brain, dict):
        return None
    try:
        from services.chat_flow_telemetry import should_skip_post_kb_ood_guard

        if should_skip_post_kb_ood_guard("_try_guard_misclassified_general_kb_ood"):
            return None
    except ImportError:
        pass
    intent = (brain.get("intent") or "").strip().lower()
    channel = (brain.get("data_channel") or "").strip().lower()
    if intent not in ("general", "payment") or channel != "kb":
        return None
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None
    try:
        from services.conversation_scope import _has_definite_welfog_shopping_signal
        from utils.helpers import (
            message_is_knowledge_information_request,
            message_is_welfog_about_request,
            message_needs_policy_answer,
        )

        if (
            message_is_knowledge_information_request(comb, conv)
            or message_is_welfog_about_request(comb)
            or message_needs_policy_answer(comb)
            or _has_definite_welfog_shopping_signal(comb)
        ):
            return None
    except ImportError:
        return None
    try:
        from services.conversational_ack_flow import ai_ood_reply

        body = ai_ood_reply(original_msg, msg_en, conv, lang)
        if body and str(body).strip():
            log_reasoning(
                "Guard — general+KB with no Welfog signal → AI OOD reply."
            )
            return str(body)
    except ImportError:
        pass
    return None


def _try_guard_account_list_fallback_dispatch(
    brain: dict,
    *,
    original_msg: str,
    msg_en: str,
    conv: str,
    user_id: str,
    lang: str,
    ctx: dict,
) -> str | None:
    """Brain dispatch missed — dedicated account-list classifier + live API (any language)."""
    try:
        from services.account_list_fast_path import try_account_list_fast_reply
        from services.kb_service import sysmsg

        if isinstance(brain, dict):
            ctx.setdefault("data", {})["ai_route"] = brain
        body = try_account_list_fast_reply(
            original_msg,
            msg_en or original_msg.lower(),
            conv,
            str(user_id),
            lang,
            ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            localized_sysmsg=_localized_sysmsg,
            sysmsg=sysmsg,
            reset_context_fn=reset_context,
        )
        if body and str(body).strip():
            log_reasoning(
                "Guard — account-list fallback after brain dispatch miss."
            )
            return str(body)
    except ImportError:
        pass
    return None


def _guard_brain_classify_with_deadline(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    conv: str,
) -> tuple[str | None, dict | None]:
    """
    Run guard brain classify inline.
    Background timeout threads were continuing after deadline and causing
    lock contention on subsequent turns.
    """
    pair = _guard_brain_ai_classify_and_dispatch(
        original_msg,
        msg_en,
        user_id,
        lang,
        ctx,
        conv,
    )
    if isinstance(pair, tuple) and len(pair) == 2:
        return pair
    return None, None


def _guard_brain_ai_classify_and_dispatch(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    conv: str,
) -> tuple[str | None, dict | None]:
    """
    ONE ai_brain_route LLM → fast live-API dispatch (orders, deals, categories, catalog, pincode).
    No heavy enrich stack — chat() finishes remaining intents under CHAT_MAX_SECONDS.
    """
    from services.ai_first_router import guard_fast_brain_classify
    from services.brain_direct_dispatch import (
        _try_brain_account_list_live_api_reply,
        _try_brain_catalog_menu_direct_reply,
        _try_brain_pincode_direct_reply,
        _try_brain_product_catalog_fallback_dispatch,
    )

    brain = guard_fast_brain_classify(
        original_msg, conv, lang, msg_en=msg_en, ctx=ctx
    )
    if not isinstance(brain, dict):
        return None, None
    if brain.get("llm_unavailable"):
        return None, brain
    try:
        from services.chat_flow_telemetry import should_skip_post_kb_ood_guard

        kb_locked = should_skip_post_kb_ood_guard("guard_brain_dispatch")
    except ImportError:
        kb_locked = (brain.get("data_channel") or "").strip().lower() == "kb"

    ctx.setdefault("data", {})["ai_route"] = brain
    intent = (brain.get("intent") or "").strip().lower()
    log_reasoning(
        f"Guard brain classified: intent={intent} "
        f"channel={brain.get('data_channel')} "
        f"alk={brain.get('account_list_kind')} — "
        f"{(brain.get('user_meaning') or '')[:80]}"
    )

    scope = (brain.get("conversation_scope") or "").strip().lower()
    if not kb_locked and (
        intent == "out_of_domain"
        or scope == "out_of_domain"
        or scope == "general_chitchat"
        or brain.get("is_welfog_related") is False
    ):
        try:
            from services.brain_direct_dispatch import _try_brain_immediate_scope_or_kb_reply

            ood_body = _try_brain_immediate_scope_or_kb_reply(
                brain,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if not ood_body and intent == "out_of_domain":
                ood_body = _fast_guard_ood_template_reply(
                    original_msg, lang, msg_en=msg_en, conv=conv, scope="out_of_domain"
                )
            if not ood_body and scope == "general_chitchat":
                ood_body = _fast_guard_ood_template_reply(
                    original_msg,
                    lang,
                    msg_en=msg_en,
                    conv=conv,
                    scope="general_chitchat",
                )
            if ood_body and str(ood_body).strip():
                return str(ood_body), brain
        except ImportError:
            pass

    kb_misroute_ood = None
    if not kb_locked:
        kb_misroute_ood = _try_guard_misclassified_general_kb_ood(
            brain,
            original_msg=original_msg,
            msg_en=msg_en,
            conv=conv,
            lang=lang,
        )
    if kb_misroute_ood and str(kb_misroute_ood).strip():
        return str(kb_misroute_ood), brain

    acct = _try_brain_account_list_live_api_reply(
        brain,
        user_id=str(user_id),
        format_purchase_history_reply=format_purchase_history_reply,
        format_wishlist_reply=format_wishlist_reply,
    )
    if acct and str(acct).strip():
        return str(acct), brain

    menu = _try_brain_catalog_menu_direct_reply(
        brain,
        original_msg=original_msg,
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context,
    )
    if menu and str(menu).strip():
        return str(menu), brain

    if intent == "category_browse" and not (brain.get("category_browse") or "").strip():
        try:
            from services.catalog_menu_replies import build_categories_list_reply_html

            cat_list = build_categories_list_reply_html(
                ctx, original_msg, reply_lang=lang
            )
            if cat_list and str(cat_list).strip():
                log_reasoning(
                    "Guard — category_browse without target → categories list API."
                )
                return str(cat_list), brain
        except ImportError:
            pass

    prod = _try_brain_product_catalog_fallback_dispatch(
        brain,
        original_msg=original_msg,
        msg_en=msg_en,
        conv_for_llm=conv,
        user_id=str(user_id),
        lang=lang,
        ctx=ctx,
        reset_context_fn=reset_context,
    )
    if prod and str(prod).strip():
        return str(prod), brain

    if intent == "pincode_check":
        pin = _try_brain_pincode_direct_reply(
            brain,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if pin and str(pin).strip():
            return str(pin), brain

    acct_fb = _try_guard_account_list_fallback_dispatch(
        brain,
        original_msg=original_msg,
        msg_en=msg_en,
        conv=conv,
        user_id=str(user_id),
        lang=lang,
        ctx=ctx,
    )
    if acct_fb and str(acct_fb).strip():
        return str(acct_fb), brain

    return None, brain


def _guard_ai_brain_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """
    ONE ai_brain_route LLM — any language/style → intent → live API / catalog.
    Sync fast path (no enrich stack, no threaded deadline that drops the reply).
    """
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    conv = ""
    if chat_id:
        try:
            conv = _conversation_snapshot_for_chat(chat_id)
        except Exception:
            conv = ""
    lang = lang_hint or "en"
    chat_log("guard AI brain start — one LLM intent (any language)")
    try:
        body, _route = _guard_brain_classify_with_deadline(
            original_msg,
            msg_en or "",
            str(user_id),
            lang,
            ctx,
            conv,
        )
        if body and str(body).strip():
            log_reasoning(
                "Guard pre-lock — AI brain classified intent → direct dispatch."
            )
            try:
                import threading

                from services.mysql_service import db_store_turn_pair

                _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
                threading.Thread(
                    target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
                    daemon=True,
                ).start()
            except Exception:
                pass
            return jsonify({"chat_id": chat_id, "type": "text", "data": body})
    except Exception as exc:
        chat_log(f"guard AI brain skip: {exc}")
        log_reasoning(f"Guard AI brain skip (non-fatal): {exc}")
    return None


def _guard_knowledge_ai_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """Refund/policy/FAQ — KB-turn AI + vector RAG before brain queue."""
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    if _guard_quick_live_goal_no_order_id(original_msg):
        return None
    try:
        from services.turn_intent_coordinator import account_list_ai_is_live_list_turn

        if account_list_ai_is_live_list_turn(
            original_msg,
            msg_en or original_msg.lower(),
            "",
            lang_hint or "",
        ):
            return None
    except ImportError:
        pass
    try:
        from utils.helpers import _text_is_pincode_serviceability_question

        if _text_is_pincode_serviceability_question(
            original_msg, msg_en or original_msg.lower()
        ):
            return None
    except ImportError:
        pass
    try:
        from services.chat_flow_telemetry import (
            begin_pre_brain_live_api_preflight,
            end_pre_brain_live_api_preflight,
        )
        from services.knowledge_fast_path import try_knowledge_fast_reply

        text_kb = _try_instant_text_kb_reply(
            original_msg, msg_en or original_msg.lower(), lang_hint or "en"
        )
        if text_kb:
            body = text_kb
        else:
            begin_pre_brain_live_api_preflight()
            try:
                body = try_knowledge_fast_reply(
                    original_msg,
                    msg_en or original_msg.lower(),
                    "",
                    reply_lang=lang_hint or "en",
                )
            finally:
                end_pre_brain_live_api_preflight()
    except ImportError:
        return None
    except Exception as exc:
        log_reasoning(f"Guard knowledge AI skip: {exc}")
        return None
    if not (body or "").strip():
        return None
    log_reasoning("Guard pre-lock — knowledge AI → KB (skip brain queue).")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_order_ask_id_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """Refund/invoice/details/track without Order ID — ask ID, not history/KB."""
    original_msg = _normalize_alphanumeric_id_boundaries((user_msg or "").strip())
    if not original_msg:
        return None
    goal = _guard_quick_live_goal_no_order_id(original_msg)
    if not goal:
        return None
    chat_log(f"guard order-ask-id start goal={goal} msg={original_msg[:60]!r}")
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    lang = lang_hint or "en"
    conv = ""
    if chat_id:
        try:
            conv = _conversation_snapshot_for_chat(chat_id)
        except Exception:
            conv = ""
    try:
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

        body = try_order_live_intent_fast_reply(
            original_msg,
            msg_en or original_msg.lower(),
            conv,
            str(user_id),
            lang,
            ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
            preset_goal=goal,
        )
    except Exception as exc:
        log_reasoning(f"Guard order-ask-id skip: {exc}")
        return None
    if not body or not str(body).strip():
        return None
    log_reasoning(f"Guard pre-lock — ask Order ID for goal={goal}.")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_order_live_ai_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """Order ID + invoice/details/track — live API before account-list / brain."""
    original_msg = _normalize_alphanumeric_id_boundaries((user_msg or "").strip())
    if not original_msg or not _guard_quick_single_order_live_turn(original_msg):
        return None
    chat_log(f"guard order-live start msg={original_msg[:60]!r}")
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    lang = lang_hint or "en"
    conv = ""
    if chat_id:
        try:
            conv = _conversation_snapshot_for_chat(chat_id)
        except Exception:
            conv = ""
    try:
        body, _route = _try_instant_order_id_structural_reply(
            original_msg,
            msg_en or original_msg.lower(),
            str(user_id),
            lang,
            ctx,
            conv_for_llm=conv,
        )
    except Exception as exc:
        log_reasoning(f"Guard order-live skip: {exc}")
        return None
    if not body or not str(body).strip():
        return None
    log_reasoning("Guard pre-lock — order ID live API (invoice/details/track).")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_account_list_ai_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """Order history / wishlist — account-list AI before brain queue."""
    original_msg = _normalize_alphanumeric_id_boundaries((user_msg or "").strip())
    if not original_msg:
        return None
    if _guard_quick_single_order_live_turn(original_msg):
        return None
    if _guard_quick_live_goal_no_order_id(original_msg):
        return None
    chat_log(f"guard account-list AI start msg={original_msg[:60]!r}")
    lang = lang_hint or "en"
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    # Zero-LLM: guard-local structural purchase-history / wishlist (no heavy helper imports).
    if _guard_quick_wishlist_turn(original_msg):
        body = format_wishlist_reply(user_id, page=1, append_only=False)
        if body and str(body).strip():
            log_reasoning(
                "Guard account-list — wishlist structural fast path (zero LLM)."
            )
            return jsonify({"chat_id": chat_id, "type": "text", "data": body})
    if _guard_quick_purchase_list_turn(original_msg):
        body = format_purchase_history_reply(user_id, page=1, append_only=False)
        if body and str(body).strip():
            log_reasoning(
                "Guard account-list — purchase-history structural fast path (zero LLM)."
            )
            return jsonify({"chat_id": chat_id, "type": "text", "data": body})
    try:
        from services.account_list_fast_path import try_account_list_fast_reply
        from services.chat_flow_telemetry import (
            begin_pre_brain_live_api_preflight,
            end_pre_brain_live_api_preflight,
        )
        from services.kb_service import sysmsg

        t_acct = time.perf_counter()
        begin_pre_brain_live_api_preflight()
        try:
            body = try_account_list_fast_reply(
                original_msg,
                msg_en or original_msg.lower(),
                "",
                user_id,
                lang,
                ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                localized_sysmsg=_localized_sysmsg,
                sysmsg=sysmsg,
                reset_context_fn=reset_context,
            )
        finally:
            end_pre_brain_live_api_preflight()
        if body:
            chat_log(
                f"guard account-list AI done in {time.perf_counter() - t_acct:.2f}s"
            )
    except ImportError:
        return None
    except Exception as exc:
        log_reasoning(f"Guard account-list AI skip: {exc}")
        return None
    if not (body or "").strip():
        return None
    log_reasoning("Guard pre-lock — account-list AI → live API (skip brain queue).")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_product_ai_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """
    Product catalog fallback — AI micro-classifier when universal brain LLM unavailable.
    No keyword/phrase gates; classifier understands any language/style.
    """
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    comb = f"{original_msg} {msg_en or ''}".strip()
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    conv = ""
    if chat_id:
        try:
            conv = _conversation_snapshot_for_chat(chat_id)
        except Exception:
            conv = ""
    try:
        from services.turn_intent_coordinator import account_list_ai_blocks_product_path

        if account_list_ai_blocks_product_path(
            original_msg,
            msg_en or "",
            conv,
            lang_hint or "",
        ):
            return None
    except ImportError:
        pass
    try:
        from services.product_catalog_resolver import _clearly_live_account_support

        if _clearly_live_account_support(
            comb,
            None,
            original_msg=original_msg,
            msg_en=msg_en or "",
            conversation_context=conv,
            reply_lang=lang_hint or "",
        ):
            return None
    except ImportError:
        pass

    try:
        from services.chat_flow_telemetry import (
            begin_pre_brain_live_api_preflight,
            end_pre_brain_live_api_preflight,
        )
        from services.product_catalog_resolver import try_product_ai_first_catalog

        lang = lang_hint or "en"
        begin_pre_brain_live_api_preflight()
        try:
            hit = try_product_ai_first_catalog(
                original_msg,
                msg_en or original_msg.lower(),
                conv,
                reply_lang=lang,
                ctx=ctx,
                user_id=str(user_id),
                guard_fast=True,
            )
        finally:
            end_pre_brain_live_api_preflight()
    except ImportError:
        return None
    except Exception as exc:
        log_reasoning(f"Guard product AI skip: {exc}")
        return None
    if not hit:
        return None
    body, route = hit[0], hit[1] if len(hit) > 1 else {}
    if not (body or "").strip():
        return None
    log_reasoning("Guard pre-lock — product AI classifier → catalog.")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_delivery_ai_prelock_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    msg_en: str = "",
    lang_hint: str = "",
):
    """City/PIN delivery serviceability — delivery AI micro-LLM before brain queue."""
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    try:
        from services.chat_flow_telemetry import (
            begin_pre_brain_live_api_preflight,
            end_pre_brain_live_api_preflight,
        )
        from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

        ctx = _get_or_create_user_ctx(user_id, chat_id)
        lang = lang_hint or "en"
        begin_pre_brain_live_api_preflight()
        try:
            body = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en or original_msg.lower(),
                "",
                lang,
                ctx,
                reset_context_fn=reset_context,
                skip_turn_check=False,
                allow_llm=True,
            )
        finally:
            end_pre_brain_live_api_preflight()
    except ImportError:
        return None
    except Exception as exc:
        log_reasoning(f"Guard delivery AI skip: {exc}")
        return None
    if not (body or "").strip():
        return None
    log_reasoning("Guard pre-lock — delivery AI → live API (skip brain queue).")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_session_fast_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    lang_hint: str,
):
    """
    Zero-LLM session continuation on main thread (PIN / order id after bot asked).
    Runs before KB guard so bare 302023 never waits on brain/KB stack.
    """
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    if lang_hint in ("en", "hinglish"):
        msg_en = original_msg.lower()
    else:
        try:
            msg_en = to_en(original_msg).lower().strip()
        except Exception:
            msg_en = original_msg.lower()
    ctx = _get_or_create_user_ctx(user_id, chat_id)
    try:
        from utils.helpers import message_is_bare_numeric_submission

        bare_numeric = message_is_bare_numeric_submission(original_msg)
    except ImportError:
        bare_numeric = bool(re.fullmatch(r"[1-9]\d{5}", original_msg.strip()))
    pending = _ctx_pending_continuation_applies(
        ctx, original_msg, msg_en, conversation_context=""
    )
    if not bare_numeric and not pending:
        return None
    lang = lang_hint or customer_reply_language(original_msg)

    # Bare 6-digit PIN only → live API, zero LLM (even if ctx.awaiting was lost).
    if re.fullmatch(r"[1-9]\d{5}", original_msg.strip()):
        pin = original_msg.strip()
        try:
            from services.pincode_delivery_flow import (
                format_pincode_check_reply,
                validate_pincode_before_api,
                _pin_localized,
            )
            from services.welfog_api import check_pincode_delivery

            ok, err_key, fmt = validate_pincode_before_api(pin, original_msg)
            if not ok:
                body = _pin_localized(err_key, original_msg, lang, **fmt)
            else:
                log_reasoning(
                    f"Guard bare PIN fast — live API {pin} (zero LLM)."
                )
                api_res = check_pincode_delivery(pin)
                body = format_pincode_check_reply(
                    pin, api_res, original_msg, lang
                )
            try:
                from utils.helpers import mark_pincode_delivery_completed

                mark_pincode_delivery_completed(ctx, pin=pin)
            except ImportError:
                pass
            try:
                import threading

                from services.mysql_service import db_store_turn_pair

                _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
                threading.Thread(
                    target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
                    daemon=True,
                ).start()
            except Exception:
                pass
            return jsonify(
                {"chat_id": chat_id, "type": "text", "data": body}
            )
        except ImportError:
            pass

    body, route = _try_ctx_continuation_reply(
        original_msg, msg_en, user_id, lang, ctx, chat_id=chat_id
    )
    if not body:
        return None
    log_reasoning(f"Guard session fast — {route}, zero LLM.")
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception:
        pass
    return jsonify({"chat_id": chat_id, "type": "text", "data": body})


def _guard_kb_ood_preflight_reply(
    user_msg: str,
    chat_id: str | None,
    user_id: str,
    lang_hint: str,
):
    """
    Main-thread KB/OOD before run_with_chat_deadline — AI intent + vector KB or scope AI.
    Returns Flask jsonify response or None.
    """
    original_msg = (user_msg or "").strip()
    if not original_msg:
        return None
    try:
        from services.translation_service import (
            resolve_customer_reply_lang,
            to_en_for_routing,
        )

        glang = (lang_hint or "").strip().lower() or resolve_customer_reply_lang(
            original_msg
        )
        msg_en = to_en_for_routing(original_msg, glang)
    except ImportError:
        glang = (lang_hint or "").strip().lower() or customer_reply_language(
            original_msg
        )
        if glang in ("en", "hinglish"):
            msg_en = original_msg.lower()
        else:
            try:
                msg_en = to_en(original_msg).lower().strip()
            except Exception:
                msg_en = original_msg.lower()
    lang = glang
    if _guard_skip_kb_ood_preflight(original_msg, msg_en, chat_id, user_id):
        return None
    try:
        from services.chat_flow_telemetry import ensure_chat_turn_started, record_user_query

        ensure_chat_turn_started()
        record_user_query(original_msg, lang)
        log_reasoning("[chat-flow] guard KB/OOD preflight — main thread")
    except ImportError:
        pass
    pre_body, pre_route = _try_non_transactional_kb_ood_preflight(
        original_msg,
        msg_en,
        "",
        lang,
    )
    if not pre_body:
        return None
    try:
        import threading

        from services.mysql_service import db_store_turn_pair

        _cid, _uid, _um, _out = chat_id, str(user_id), user_msg, pre_body
        threading.Thread(
            target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
            daemon=True,
        ).start()
    except Exception as db_exc:
        chat_log(f"guard preflight DB skip: {db_exc}")
    try:
        from services.chat_flow_telemetry import (
            log_turn_complete,
            mark_routing_complete,
            record_route,
        )

        ar = pre_route or {}
        record_route(
            intent=ar.get("intent") or "",
            source=ar.get("route_handler") or ar.get("data_channel") or "",
        )
        mark_routing_complete()
        log_turn_complete(
            intent=ar.get("intent") or "",
            route=ar.get("route_handler") or "",
            source=ar.get("data_channel") or "",
            reason=ar.get("reasoning") or "",
        )
    except Exception:
        pass
    preview = pre_body[:120].replace("\n", " ")
    chat_log(f"guard preflight reply ({len(pre_body)} chars): {preview!r}")
    return jsonify(
        {
            "chat_id": chat_id,
            "type": "text",
            "data": pre_body,
        }
    )


def _try_bare_numeric_live_fast_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    chat_id: str | None = None,
    conv_for_llm: str = "",
):
    """
    Zero-LLM: bare PIN / order id / product id using session lock + recent chat.
    Pincode, order, and product id are never mixed on the same path.
    """
    from utils.helpers import (
        classify_bare_numeric_turn,
        message_is_bare_numeric_submission,
    )

    if not message_is_bare_numeric_submission(original_msg):
        return None, ""

    raw = original_msg.strip()
    # Pure 6-digit PIN — live API immediately; never block on MySQL conv snapshot.
    if re.fullmatch(r"[1-9]\d{5}", raw):
        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            body = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                "",
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if body:
                return body, "bare_pincode_live_fast"
        except ImportError:
            pass

    conv = (conv_for_llm or "").strip() or _conversation_snapshot_for_chat(chat_id)
    kind = classify_bare_numeric_turn(original_msg, conv, ctx=ctx)

    if kind == "pincode":
        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            body = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv,
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if body:
                return body, "bare_pincode_live_fast"
        except ImportError:
            pass

    if kind == "order_id":
        try:
            from services.order_id_handoff_fast_path import try_order_id_handoff_reply

            body = try_order_id_handoff_reply(
                original_msg,
                msg_en,
                conv,
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if body:
                return body, "bare_order_id_live_fast"
        except ImportError:
            pass

    return None, ""


def _ctx_pending_continuation_applies(
    ctx: dict,
    original_msg: str,
    msg_en: str,
    *,
    conversation_context: str = "",
) -> bool:
    """Session lock: bot asked for Order ID / PIN — resolve from ctx without brain LLM."""
    if not isinstance(ctx, dict):
        return False
    comb = f"{original_msg} {msg_en}".strip()
    awaiting = ctx.get("awaiting")
    if awaiting == "order_id":
        return bool(re.search(r"\b[0-9]{4,20}\b", comb))
    if _ctx_has_order_pending(ctx) and re.search(r"\b[0-9]{4,20}\b", comb):
        return True
    last = (ctx.get("last") or "").strip().lower()
    if awaiting == "pincode" and re.search(r"\b[1-9]\d{5}\b", comb):
        return True
    if last == "pincode" and re.search(r"\b[1-9]\d{5}\b", comb):
        return True
    # Fresh catalog/KB turns — skip delivery/thread micro-classifiers (~10s+).
    if not awaiting and not _ctx_has_order_pending(ctx):
        return False
    if awaiting == "pincode":
        if re.search(r"\b[1-9]\d{5}\b", comb):
            return True
        if not re.search(
            r"\b(pin\s*code|pincode|delivery|deliver|area|city|zip|postal|pincod)\b",
            comb,
            re.I,
        ):
            return False
        if len(comb) > 36:
            return False
        try:
            from services.location_delivery_resolver import (
                _short_area_followup_in_pincode_thread,
                turn_continues_pincode_area_check,
            )

            if turn_continues_pincode_area_check(
                comb, conversation_context
            ) or _short_area_followup_in_pincode_thread(comb):
                return True
        except ImportError:
            pass
        return False
    try:
        from utils.helpers import _message_is_order_id_followup_submission

        if _message_is_order_id_followup_submission(original_msg, ""):
            return True
    except ImportError:
        pass
    return False


def _try_ctx_continuation_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    chat_id: str | None = None,
):
    """
    Zero-LLM follow-up when ctx says we are waiting for Order ID or pincode.
    Runs before ai_brain_route so numeric replies are not misread as pincode.
    """
    from utils.helpers import message_is_bare_numeric_submission

    conv = ""
    pending = _ctx_pending_continuation_applies(
        ctx, original_msg, msg_en, conversation_context=""
    )
    bare_numeric = message_is_bare_numeric_submission(original_msg)
    if not pending and not bare_numeric:
        return None, ""

    try:
        from services.ai_route_semantics import (
            _structural_details_or_invoice_goal_from_message,
            _structural_track_goal_from_message,
        )
        from utils.helpers import (
            _normalize_order_chat_text,
            clear_order_session_for_new_lookup,
            extract_order_id,
        )

        norm_o = _normalize_order_chat_text(original_msg)
        norm_e = _normalize_order_chat_text(msg_en) if msg_en else msg_en
        comb_norm = f"{norm_o or ''} {norm_e or ''}".strip()
        if _extract_attached_welfog_order_id(
            comb_norm, ""
        ) or extract_order_id(comb_norm, ""):
            msg_goal = (
                _structural_track_goal_from_message(norm_o, norm_e)
                or _structural_details_or_invoice_goal_from_message(norm_o, norm_e)
            )
            pending_act = (
                (ctx.get("data") or {}).get("pending_action") or ""
            ).strip().lower()
            if msg_goal and pending_act:
                act_goal = {
                    "track": "track",
                    "order_invoice": "order_invoice",
                    "invoice": "order_invoice",
                    "order_details": "order_details",
                    "details": "order_details",
                }.get(pending_act, pending_act)
                if act_goal != msg_goal:
                    log_reasoning(
                        f"Order session reset — new goal={msg_goal} overrides pending={pending_act}."
                    )
                    clear_order_session_for_new_lookup(ctx)
                    return None, ""
    except ImportError:
        pass

    # Pure PIN — zero LLM, skip MySQL snapshot (can hang minutes under load).
    if re.fullmatch(r"[1-9]\d{5}", original_msg.strip()):
        bare_body, bare_route = _try_bare_numeric_live_fast_reply(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            chat_id=chat_id,
            conv_for_llm="",
        )
        if bare_body:
            return bare_body, bare_route

    if bare_numeric and isinstance(ctx, dict) and ctx.get("awaiting") == "pincode":
        if re.search(r"\b[1-9]\d{5}\b", original_msg):
            bare_body, bare_route = _try_bare_numeric_live_fast_reply(
                original_msg,
                msg_en,
                user_id,
                lang,
                ctx,
                chat_id=chat_id,
                conv_for_llm="",
            )
            if bare_body:
                return bare_body, bare_route

    if chat_id:
        conv = _conversation_snapshot_for_chat(chat_id)
        pending = _ctx_pending_continuation_applies(
            ctx, original_msg, msg_en, conversation_context=conv
        )
        if not pending and not bare_numeric:
            return None, ""

    if not conv and (pending or bare_numeric):
        conv = _conversation_snapshot_for_chat(chat_id)

    bare_body, bare_route = _try_bare_numeric_live_fast_reply(
        original_msg,
        msg_en,
        user_id,
        lang,
        ctx,
        chat_id=chat_id,
        conv_for_llm=conv,
    )
    if bare_body:
        return bare_body, bare_route

    if not pending:
        return None, ""

    awaiting = ctx.get("awaiting") if isinstance(ctx, dict) else None
    last = (ctx.get("last") or "").strip().lower() if isinstance(ctx, dict) else ""

    if awaiting == "pincode":
        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            pin_body = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv,
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if pin_body:
                return pin_body, "pincode_ctx_continuation"
        except ImportError:
            pass

    if _ctx_has_order_pending(ctx) or awaiting == "order_id":
        try:
            from services.order_id_handoff_fast_path import try_order_id_handoff_reply

            handoff = try_order_id_handoff_reply(
                original_msg,
                msg_en,
                conv,
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if handoff:
                return handoff, "order_id_ctx_continuation"
        except ImportError:
            pass
    else:
        try:
            from utils.helpers import _message_is_order_id_followup_submission
            from services.order_id_handoff_fast_path import try_order_id_handoff_reply

            if _message_is_order_id_followup_submission(original_msg, conv):
                handoff = try_order_id_handoff_reply(
                    original_msg,
                    msg_en,
                    conv,
                    user_id,
                    lang,
                    ctx,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context,
                )
                if handoff:
                    return handoff, "order_id_followup_continuation"
        except ImportError:
            pass

    return None, ""


def _can_resolve_without_conversation(
    original_msg: str,
    msg_en: str,
    lang: str,
    ctx: dict,
) -> bool:
    """
    Cheap structural-only gate (no phrase/intent graphs).
    True when order-id / SKU paths can run without MySQL conversation load.
    Semantic order turns (id + natural language) need ai_brain_route first.
    """
    comb = f"{original_msg} {msg_en}".strip()
    if not comb:
        return True
    if _ctx_pending_continuation_applies(ctx, original_msg, msg_en):
        return True
    if isinstance(ctx, dict) and ctx.get("awaiting") == "order_id":
        if re.search(r"\b[0-9]{4,20}\b", comb):
            return True
    if re.fullmatch(r"[0-9]{4,20}", (original_msg or "").strip()):
        return True
    if re.fullmatch(r"[0-9]{4,20}", comb.strip()):
        return True
    if re.search(r"\b[0-9]{4,20}\b", comb) and re.search(r"\S+\s+\S+", comb):
        try:
            from services.semantic_intent import strict_ai_semantic_mode

            if strict_ai_semantic_mode():
                return False
        except ImportError:
            pass
    if re.search(
        r"\b(?:pro[_\s-]?id|product[_\s-]?id|pid)\s*[:\-#]?\s*\d{4,12}\b", comb, re.I
    ):
        return True
    if re.search(r"\bsku\b", comb, re.I) and re.search(
        r"\b[A-Za-z0-9][A-Za-z0-9_\-]{3,80}\b", comb
    ):
        return True
    try:
        from services.order_live_intent_fast_path import resolve_structural_locked_order_goal

        structural_goal = resolve_structural_locked_order_goal(
            original_msg, msg_en, "", ctx
        )
        if structural_goal in (
            "track",
            "order_invoice",
            "order_details",
            "refund_status",
            "payment",
        ):
            return True
    except ImportError:
        pass
    return False


def _try_early_structural_live_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    chat_id: str | None = None,
):
    """Order live API / handoff / catalog — zero LLM, no MySQL conversation."""
    conv = _conversation_snapshot_for_chat(chat_id)
    try:
        structural_order = try_structural_order_live_reply(
            original_msg,
            msg_en,
            conv_for_llm=conv,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
        )
        if structural_order:
            return structural_order, "structural_order_early"
    except ImportError:
        pass

    try:
        from services.order_id_handoff_fast_path import try_order_id_handoff_reply

        handoff = try_order_id_handoff_reply(
            original_msg,
            msg_en,
            conv,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
        )
        if handoff:
            return handoff, "order_id_handoff_early"
    except ImportError:
        pass

    return None, ""


def _try_instant_order_id_structural_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
) -> tuple[str | None, dict | None]:
    """Order id + track/invoice/details — live API before product OpenSearch or brain KB."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None, None
    if not re.search(r"\b[0-9]{4,20}\b", comb):
        return None, None
    try:
        from utils.helpers import _normalize_order_chat_text

        norm_orig = _normalize_order_chat_text(original_msg)
        norm_en = _normalize_order_chat_text(msg_en) if msg_en else msg_en
    except ImportError:
        norm_orig, norm_en = original_msg, msg_en
    try:
        from services.ai_route_semantics import (
            _structural_details_or_invoice_goal_from_message,
            _structural_refund_goal_from_message,
            _structural_track_goal_from_message,
        )
        from services.order_live_intent_fast_path import try_order_live_intent_fast_reply
        from utils.helpers import extract_order_id, turn_is_catalog_product_lookup

        if turn_is_catalog_product_lookup(original_msg, msg_en):
            return None, None
        comb_norm = f"{norm_orig or ''} {norm_en or ''}".strip()
        oid = _extract_attached_welfog_order_id(
            comb_norm, conv_for_llm
        ) or extract_order_id(comb_norm, conv_for_llm)
        if not oid:
            return None, None
        live_goal = (
            _structural_refund_goal_from_message(norm_orig, norm_en)
            or _structural_details_or_invoice_goal_from_message(norm_orig, norm_en)
            or _structural_track_goal_from_message(norm_orig, norm_en)
        )
        if not live_goal:
            tl = comb_norm.lower()
            if re.search(r"\b(?:invoice|bill|receipt)\b", tl):
                live_goal = "order_invoice"
            elif re.search(
                r"\b(?:status|stutus|track|update|kab|aa\s+jayega|pahunch)\b", tl
            ):
                live_goal = "track"
            elif re.search(r"\b(?:detail|details|saari|sari)\b", tl):
                live_goal = "order_details"
        if not live_goal:
            return None, None
        body = try_order_live_intent_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
            preset_goal=live_goal,
        )
        if not body:
            return None, None
        route = (ctx.get("data") or {}).get("ai_route") or {
            "_zero_llm_fast": True,
            "_universal_brain_route": True,
            "intent": "refund" if live_goal == "refund_status" else "order",
            "data_channel": "live_api",
            "order_lookup_kind": live_goal,
        }
        ctx.setdefault("data", {})["ai_route"] = route
        log_reasoning(f"Instant lane — order live API goal={live_goal} (zero brain).")
        return body, route
    except ImportError:
        return None, None


def _try_instant_pincode_delivery_reply(
    original_msg: str,
    msg_en: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
    allow_llm: bool = True,
) -> tuple[str | None, dict | None]:
    """Delivery/PIN serviceability — AI detects intent, then live API (before product)."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None, None
    try:
        from services.location_delivery_resolver import turn_requests_delivery_serviceability
        from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

        if not turn_requests_delivery_serviceability(
            original_msg, msg_en, conv_for_llm, allow_llm=allow_llm
        ):
            return None, None

        pin_body = try_pincode_delivery_fast_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            lang,
            ctx,
            reset_context_fn=reset_context,
        )
        if pin_body:
            log_reasoning("Instant lane — pincode/city delivery (AI + live API).")
            pr = {
                "_zero_llm_fast": True,
                "_universal_brain_route": True,
                "intent": "pincode_check",
                "data_channel": "live_api",
                "_pincode_delivery_fast": True,
            }
            ctx.setdefault("data", {})["ai_route"] = pr
            return pin_body, pr
    except ImportError:
        pass
    return None, None


def _try_ai_classified_product_catalog_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
) -> tuple[str | None, dict | None]:
    """
    KB-turn AI already said product_search — OpenSearch immediately (no brain stack).
    Reuses cached ai_classify_kb_turn from guard preflight on the same message.
    """
    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            kb_classified_as_product_catalog,
            kb_turn_blocks_product_catalog,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(
            original_msg, msg_en, conv_for_llm, lang
        )
        if peeked is not _KB_CACHE_UNSET and kb_turn_blocks_product_catalog(peeked):
            log_reasoning(
                "Product catalog skip — KB classifier locked informational."
            )
            return None, None
        if peeked is _KB_CACHE_UNSET or not kb_classified_as_product_catalog(peeked):
            return None, None
    except ImportError:
        return None, None

    um = (peeked.get("user_meaning_en") or "").strip()
    try:
        from services.brain_direct_dispatch import (
            _prepare_brain_product_route,
            _run_product_catalog_flow,
        )
        from services.product_query_understanding import (
            resolve_catalog_search_terms_for_message,
        )

        product_route = {
            "intent": "product",
            "data_channel": "catalog",
            "run_catalog_search": True,
            "_product_catalog_locked": True,
            "_universal_brain_route": True,
            "_zero_llm_fast": True,
            "_ai_single_pass": True,
            "_needs_product_nlu_llm": False,
            "user_meaning": um,
            "search_query": um,
        }
        sq = resolve_catalog_search_terms_for_message(
            original_msg, msg_en, ai_route=product_route
        )
        if not sq:
            sq = um
        if not sq or len(str(sq).strip()) < 2:
            return None, None
        product_route, sq = _prepare_brain_product_route(
            product_route, original_msg, msg_en
        )
        log_reasoning(
            f"KB AI product_search sq={sq!r} → OpenSearch (skip brain wait)."
        )
        body = _run_product_catalog_flow(
            product_route,
            sq,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if body:
            return body, product_route
    except ImportError:
        pass
    return None, None


def _try_instant_product_catalog_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
) -> tuple[str | None, dict | None]:
    """
    Product find/search — structural query extract + OpenSearch only (~2–5s).
    No routing LLM, no KB embedding. _run_product_catalog_flow already caps search time.
    """
    try:
        from services.brain_direct_dispatch import try_structural_product_catalog_reply

        body = try_structural_product_catalog_reply(
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if not body:
            return None, None
        ar = (ctx.get("data") or {}).get("ai_route") or {
            "intent": "product",
            "data_channel": "catalog",
            "run_catalog_search": True,
            "_product_catalog_locked": True,
            "_zero_llm_fast": True,
            "_universal_brain_route": True,
        }
        return body, ar
    except ImportError:
        return None, None


def _try_instant_zero_llm_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
) -> tuple[str | None, dict | None]:
    """
    Wishlist-speed lane — live API / OpenSearch / KB vector only (no brain stack, no routing LLM).
    Runs before pincode-before-brain and ai_brain_route so deals/categories/product match wishlist latency.
    """
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None, None

    route_stub: dict = {"_zero_llm_fast": True, "_universal_brain_route": True}

    try:
        from services.chitchat_resolver import _is_instant_greeting_thanks_lane
        from utils.helpers import fast_greeting_reply_html

        if (
            _is_instant_greeting_thanks_lane(
                original_msg, msg_en, conv_for_llm
            )
            and not _instant_lane_blocked_by_transactional(
                original_msg, msg_en, conv_for_llm
            )
        ):
            greet_body = fast_greeting_reply_html(original_msg, reply_lang=lang)
            if greet_body:
                log_reasoning("Instant lane — greeting/chitchat (zero LLM).")
                gr = {
                    **route_stub,
                    "intent": "general",
                    "conversation_scope": "general_chitchat",
                    "data_channel": "none",
                    "meta_kind": "conversational",
                }
                ctx.setdefault("data", {})["ai_route"] = gr
                return greet_body, gr
    except ImportError:
        pass

    order_body, order_route = _try_instant_order_id_structural_reply(
        original_msg, msg_en, user_id, lang, ctx, conv_for_llm=conv_for_llm
    )
    if order_body:
        return order_body, order_route

    try:
        from services.ai_first_router import _try_account_list_fast_path

        acct = _try_account_list_fast_path(original_msg, msg_en)
        if acct:
            _, route = acct
            route = dict(route)
            alk = (route.get("account_list_kind") or "").strip().lower()
            intent = (route.get("intent") or "").strip().lower()
            if alk == "wishlist_in_chat" or intent == "wishlist":
                body = format_wishlist_reply(user_id, page=1, append_only=False)
                if body:
                    log_reasoning("Instant lane — wishlist API (zero brain).")
                    ctx.setdefault("data", {})["ai_route"] = route
                    return body, route
            if alk == "purchase_history_in_chat" or intent == "order_history":
                body = format_purchase_history_reply(user_id, page=1, append_only=False)
                if body:
                    log_reasoning("Instant lane — order history API (zero brain).")
                    ctx.setdefault("data", {})["ai_route"] = route
                    return body, route
    except ImportError:
        pass

    try:
        from services.brain_direct_dispatch import try_explicit_sku_catalog_reply

        sku_body = try_explicit_sku_catalog_reply(
            original_msg,
            msg_en,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if sku_body:
            log_reasoning("Instant lane — standalone SKU OpenSearch (zero brain).")
            sku_route = (ctx.get("data") or {}).get("ai_route") or {
                "intent": "product",
                "data_channel": "catalog",
                "route_handler": "sku_structural_fast",
                "_product_catalog_locked": True,
                "_zero_llm_fast": True,
            }
            ctx.setdefault("data", {})["ai_route"] = sku_route
            return sku_body, sku_route
    except ImportError:
        pass

    pin_body, pin_route = _try_instant_pincode_delivery_reply(
        original_msg, msg_en, lang, ctx, conv_for_llm=conv_for_llm
    )
    if pin_body:
        return pin_body, pin_route

    prod_body, prod_route = _try_instant_product_catalog_reply(
        original_msg, msg_en, user_id, lang, ctx, conv_for_llm=conv_for_llm
    )
    if prod_body:
        log_reasoning(
            "Instant lane — product OpenSearch (structural terms, zero brain/KB embed)."
        )
        ctx.setdefault("data", {})["ai_route"] = prod_route
        return prod_body, prod_route

    skip_structural_order = _turn_has_structural_fast_lane_token(
        original_msg, msg_en, conv_for_llm, ctx
    )

    if not skip_structural_order:
        try:
            structural_order = try_structural_order_live_reply(
                original_msg,
                msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if structural_order:
                log_reasoning("Instant lane — structural order session API (zero brain).")
                ar = (ctx.get("data") or {}).get("ai_route") or {
                    **route_stub,
                    "intent": "order",
                    "data_channel": "live_api",
                }
                ctx.setdefault("data", {})["ai_route"] = ar
                return structural_order, ar
        except ImportError:
            pass

    try:
        from services.ai_first_router import _try_catalog_menu_fast_path
        from services.catalog_menu_replies import (
            build_categories_list_reply_html,
            build_today_deals_reply_html,
        )

        menu = _try_catalog_menu_fast_path(original_msg, msg_en, ctx=ctx)
        if menu:
            _, route = menu
            route = dict(route)
            intent = (route.get("intent") or "").strip().lower()
            if intent == "deals":
                body = build_today_deals_reply_html(original_msg, reply_lang=lang)
                if body:
                    log_reasoning("Instant lane — today's deals API (zero brain).")
                    ctx.setdefault("data", {})["ai_route"] = route
                    return body, route
            if intent == "categories":
                body = build_categories_list_reply_html(ctx, original_msg, reply_lang=lang)
                if body:
                    log_reasoning("Instant lane — categories API (zero brain).")
                    ctx.setdefault("data", {})["ai_route"] = route
                    return body, route
    except ImportError:
        pass

    if not _turn_bypasses_embedding_vector(original_msg, msg_en, ctx):
        text_kb = _try_instant_text_kb_reply(
            original_msg, msg_en, lang, conversation_context=conv_for_llm
        )
        if text_kb:
            kr = {**route_stub, "intent": "general", "data_channel": "kb"}
            ctx.setdefault("data", {})["ai_route"] = kr
            return text_kb, kr

    if _turn_bypasses_embedding_vector(original_msg, msg_en, ctx):
        log_reasoning(
            "Instant lane — skip embedding/vector (transactional or order-id turn)."
        )
    else:
        try:
            from services.conversation_zero_llm_fallback import try_embedding_kb_only_reply
            from utils.helpers import turn_is_obvious_product_shopping_turn

            if not turn_is_obvious_product_shopping_turn(
                original_msg, msg_en, conv_for_llm
            ):
                import concurrent.futures

                kb_body = None
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        try_embedding_kb_only_reply,
                        original_msg,
                        msg_en,
                        conv_for_llm,
                        reply_lang=lang,
                    )
                    try:
                        kb_body = fut.result(timeout=4.0)
                    except concurrent.futures.TimeoutError:
                        log_reasoning(
                            "Instant lane — KB vector skipped (4s cap); defer to brain."
                        )
                if kb_body:
                    log_reasoning("Instant lane — KB vector (zero brain).")
                    kr = {**route_stub, "intent": "general", "data_channel": "kb"}
                    ctx.setdefault("data", {})["ai_route"] = kr
                    return kb_body, kr
        except ImportError:
            pass

    return None, None


def _try_early_ai_brain_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
    chat_id: str | None = None,
) -> tuple[str | None, dict | None]:
    """
    ONE ai_brain_route per turn — any language/style → locked API / KB / chitchat.
    Cached for the rest of the request (no duplicate routing LLM).
    """
    try:
        from services.chat_flow_telemetry import get_early_brain_dispatch, store_early_brain_dispatch

        cached_dispatch = get_early_brain_dispatch()
        if cached_dispatch is not None:
            return cached_dispatch
    except ImportError:
        pass

    def _finish_early_brain(body, route):
        try:
            from services.chat_flow_telemetry import store_early_brain_dispatch

            store_early_brain_dispatch(body, route)
        except ImportError:
            pass
        return body, route

    conv = (conv_for_llm or "").strip()
    if not conv and chat_id:
        comb_early = f"{original_msg or ''} {msg_en or ''}".strip()
        load_conv = False
        if isinstance(ctx, dict) and ctx.get("awaiting") in ("order_id", "pincode"):
            load_conv = True
        elif comb_early and not _message_is_complete_standalone_question(comb_early):
            word_n = len(re.findall(r"\S+", comb_early))
            if word_n <= 3:
                load_conv = True
        if load_conv:
            conv = _conversation_context_for_routing(
                chat_id, original_msg, msg_en, ctx, limit=4
            )
    if isinstance(ctx, dict) and ctx.get("awaiting") == "order_id":
        try:
            from services.order_id_handoff_fast_path import try_order_id_handoff_reply

            handoff = try_order_id_handoff_reply(
                original_msg,
                msg_en,
                conv,
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if handoff:
                log_reasoning(
                    "Early brain skipped — awaiting Order ID handoff (zero extra LLM)."
                )
                return _finish_early_brain(
                    handoff, (ctx.get("data") or {}).get("ai_route")
                )
        except ImportError:
            pass
        return _finish_early_brain(None, (ctx.get("data") or {}).get("ai_route"))
    if isinstance(ctx, dict) and ctx.get("awaiting") == "pincode":
        comb_pin = f"{original_msg} {msg_en}".strip()
        try:
            from utils.helpers import (
                _turn_blocks_pincode_serviceability_routing,
                message_is_past_purchase_list_request,
            )

            if (
                _turn_blocks_pincode_serviceability_routing(comb_pin)
                or _turn_has_structural_fast_lane_token(
                    original_msg, msg_en, conv, ctx
                )
                or message_is_past_purchase_list_request(comb_pin)
            ):
                ctx["awaiting"] = None
                ctx["last"] = None
            else:
                return None, None
        except ImportError:
            return None, None
    try:
        from services.ai_first_router import early_universal_brain_route

        t_route = time.perf_counter()
        brain_route_data = early_universal_brain_route(
            original_msg, conv, lang, msg_en=msg_en, ctx=ctx
        )
        from services.brain_direct_dispatch import (
            _try_brain_account_list_live_api_reply,
            _try_brain_immediate_scope_or_kb_reply,
            _try_brain_order_live_fallback_dispatch,
            _try_brain_pincode_direct_reply,
            try_brain_direct_dispatch,
        )
        if not isinstance(brain_route_data, dict):
            return None, None
        if brain_route_data.get("llm_unavailable"):
            return _finish_early_brain(None, brain_route_data)

        ctx.setdefault("data", {})["ai_route"] = brain_route_data

        acct_live = _try_brain_account_list_live_api_reply(
            brain_route_data,
            user_id=user_id,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
        )
        if acct_live:
            return _finish_early_brain(acct_live, brain_route_data)

        order_struct_body, order_struct_route = _try_instant_order_id_structural_reply(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm=conv,
        )
        if order_struct_body:
            log_reasoning(
                "Brain early — structural order id live API (skip KB/product stack)."
            )
            return _finish_early_brain(order_struct_body, order_struct_route)

        _ch_early = (brain_route_data.get("data_channel") or "").strip().lower()
        _intent_early = (brain_route_data.get("intent") or "").strip().lower()
        _scope_ch_early = (brain_route_data.get("conversation_scope") or "").strip().lower()
        _kb_keys_early = [
            str(k).strip() for k in (brain_route_data.get("kb_keys") or []) if str(k).strip()
        ]
        _kb_authoritative_json = (
            _ch_early == "kb"
            and bool(_kb_keys_early)
            and not brain_route_data.get("needs_order_id")
            and _intent_early not in ("out_of_domain",)
            and brain_route_data.get("is_welfog_related", True) is not False
        )

        # Authoritative KB route — must run before any OOD/chitchat override.
        if (
            _kb_authoritative_json
        ):
            try:
                from services.brain_direct_dispatch import _try_brain_kb_locked_reply

                kb_brain = _try_brain_kb_locked_reply(
                    brain_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if kb_brain:
                    log_reasoning("Brain early — KB from AI JSON (authoritative, before OOD).")
                    return _finish_early_brain(kb_brain, brain_route_data)
            except ImportError:
                pass
            try:
                from services.knowledge_query_pipeline import try_brain_semantic_kb_reply

                kb_semantic_early = try_brain_semantic_kb_reply(
                    original_msg,
                    msg_en,
                    conv,
                    reply_lang=lang,
                    brain_route=brain_route_data,
                )
                if kb_semantic_early and kb_semantic_early.strip():
                    log_reasoning(
                        "Brain early — semantic KB grounded answer (authoritative, before OOD)."
                    )
                    return _finish_early_brain(
                        kb_semantic_early,
                        {
                            "intent": "general",
                            "data_channel": "kb",
                            "route_handler": "kb_brain_semantic",
                        },
                    )
            except ImportError:
                pass

        try:
            from services.chat_flow_telemetry import should_skip_post_kb_ood_guard

            _skip_ood = should_skip_post_kb_ood_guard("brain_early")
        except ImportError:
            _skip_ood = _ch_early == "kb"
        _skip_ood = _skip_ood or _kb_authoritative_json

        if not _skip_ood:
            mis_ood = _try_guard_misclassified_general_kb_ood(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv=conv,
                lang=lang,
            )
            if mis_ood and mis_ood.strip():
                log_reasoning("Brain early — misrouted off-topic → AI OOD reply.")
                return _finish_early_brain(
                    mis_ood,
                    {
                        "intent": "out_of_domain",
                        "data_channel": "none",
                        "conversation_scope": "out_of_domain",
                        "is_welfog_related": False,
                    },
                )

        # OOD / chitchat — only when authoritative KB route is not locked.
        if not _skip_ood and (
            _intent_early == "out_of_domain"
            or _scope_ch_early == "out_of_domain"
            or _scope_ch_early == "harm_sensitive"
            or brain_route_data.get("is_welfog_related") is False
            or _scope_ch_early == "general_chitchat"
        ):
            try:
                from services.brain_direct_dispatch import _try_brain_immediate_scope_or_kb_reply

                scope_early = _try_brain_immediate_scope_or_kb_reply(
                    brain_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if scope_early:
                    log_reasoning("Brain early — AI scope/chitchat/OOD reply.")
                    return _finish_early_brain(scope_early, brain_route_data)
            except ImportError:
                pass
            ood_scope = (
                "general_chitchat"
                if _scope_ch_early == "general_chitchat"
                else (
                    "harm_sensitive"
                    if _scope_ch_early == "harm_sensitive"
                    else "out_of_domain"
                )
            )
            ood_fast = _fast_guard_ood_template_reply(
                original_msg,
                lang,
                msg_en=msg_en,
                conv=conv,
                scope=ood_scope,
            )
            if ood_fast:
                log_reasoning(f"Brain early — scope AI reply ({ood_scope}).")
                return _finish_early_brain(ood_fast, brain_route_data)

        if (
            _kb_authoritative_json
        ):
            try:
                from services.brain_direct_dispatch import _try_brain_kb_locked_reply

                kb_brain = _try_brain_kb_locked_reply(
                    brain_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if kb_brain:
                    log_reasoning("Brain early — KB from AI JSON (skip pincode/product).")
                    return _finish_early_brain(kb_brain, brain_route_data)
            except ImportError:
                pass

        if _intent_early == "pincode_check":
            pin_early = _try_brain_pincode_direct_reply(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if pin_early:
                return _finish_early_brain(pin_early, brain_route_data)

        _scope_early = (brain_route_data.get("conversation_scope") or "").strip().lower()
        _skip_immediate_for_live_order = False
        try:
            from services.brain_direct_dispatch import _brain_is_order_live_turn

            if _brain_is_order_live_turn(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
            ):
                _skip_immediate_for_live_order = True
        except ImportError:
            pass
        if not _skip_immediate_for_live_order and (
            _scope_early in ("general_chitchat", "out_of_domain", "harm_sensitive")
            or _intent_early == "out_of_domain"
            or not brain_route_data.get("is_welfog_related", True)
            or (
                _ch_early == "kb"
                and not brain_route_data.get("needs_order_id")
                and _intent_early in ("general", "refund", "payment", "seller", "")
            )
        ):
            immediate_early = _try_brain_immediate_scope_or_kb_reply(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if immediate_early:
                log_reasoning(
                    "Brain early — immediate scope/KB/chitchat (skip heavy stack)."
                )
                return _finish_early_brain(immediate_early, brain_route_data)

        _ch_kb = _ch_early
        _int_kb = _intent_early
        _skip_kb_for_order_id = False
        try:
            from utils.helpers import extract_order_id

            if extract_order_id(f"{original_msg} {msg_en}".strip(), conv):
                _skip_kb_for_order_id = True
        except ImportError:
            pass
        if (
            _ch_kb == "kb"
            and not brain_route_data.get("needs_order_id")
            and not _skip_kb_for_order_id
        ):
            if _int_kb in ("refund", "payment", "general", "seller", ""):
                try:
                    from services.brain_direct_dispatch import _try_brain_kb_locked_reply

                    kb_fast = _try_brain_kb_locked_reply(
                        brain_route_data,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conv_for_llm=conv,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                        reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                    )
                    if kb_fast:
                        log_reasoning(
                            "Brain early — KB policy instant (skip product/order stack)."
                        )
                        return _finish_early_brain(kb_fast, brain_route_data)
                except ImportError:
                    pass

        comb_wl = f"{original_msg or ''} {msg_en or ''}".strip()
        try:
            from services.account_list_semantics import (
                account_list_route_is_locked,
                reconcile_account_list_from_brain_meaning,
            )

            brain_route_data = reconcile_account_list_from_brain_meaning(brain_route_data)
            ctx["data"]["ai_route"] = brain_route_data
            if (
                account_list_route_is_locked(brain_route_data)
                or (brain_route_data.get("intent") or "").strip().lower() == "wishlist"
            ):
                alk = (brain_route_data.get("account_list_kind") or "").strip().lower()
                if alk == "wishlist_in_chat" or (
                    brain_route_data.get("intent") or ""
                ).strip().lower() == "wishlist":
                    wl_body = format_wishlist_reply(
                        user_id, page=1, append_only=False
                    )
                    if wl_body:
                        log_reasoning(
                            "Brain wishlist — account_list_kind from ai_brain_route."
                        )
                        return _finish_early_brain(wl_body, brain_route_data)
        except ImportError:
            pass

        # Live APIs first — same order as instant lane; never wait on KB/scope stack.
        try:
            from services.account_list_semantics import (
                account_list_route_is_locked,
                reconcile_account_list_from_brain_meaning,
            )

            brain_route_data = reconcile_account_list_from_brain_meaning(brain_route_data)
            ctx["data"]["ai_route"] = brain_route_data
            if account_list_route_is_locked(brain_route_data):
                alk = (brain_route_data.get("account_list_kind") or "").strip().lower()
                if alk == "purchase_history_in_chat" or (
                    brain_route_data.get("intent") or ""
                ).strip().lower() == "order_history":
                    ph_body = format_purchase_history_reply(
                        user_id, page=1, append_only=False
                    )
                    if ph_body:
                        log_reasoning("Brain early — order history API (skip KB stack).")
                        return _finish_early_brain(ph_body, brain_route_data)
        except ImportError:
            pass

        _intent_live = (brain_route_data.get("intent") or "").strip().lower()
        if _intent_live in ("deals", "categories", "category_feed"):
            try:
                from services.brain_direct_dispatch import _try_brain_catalog_menu_direct_reply

                menu_early = _try_brain_catalog_menu_direct_reply(
                    brain_route_data,
                    original_msg=original_msg,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context,
                )
                if menu_early:
                    log_reasoning(
                        f"Brain early — {_intent_live} API (skip KB/scope stack)."
                    )
                    return _finish_early_brain(menu_early, brain_route_data)
            except ImportError:
                pass

        if (brain_route_data.get("intent") or "").strip().lower() == "pincode_check":
            pin_early = _try_brain_pincode_direct_reply(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if pin_early:
                return _finish_early_brain(pin_early, brain_route_data)

        invoice_fast = _try_brain_locked_invoice_fast_reply(
            brain_route_data,
            original_msg=original_msg,
            msg_en=msg_en,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
        )
        if invoice_fast:
            return _finish_early_brain(invoice_fast, brain_route_data)

        order_early = _try_brain_order_live_fallback_dispatch(
            brain_route_data,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
        )
        if order_early:
            log_reasoning(
                "Brain early — order live API (before product/KB stack)."
            )
            return _finish_early_brain(order_early, brain_route_data)

        # Product catalog — before KB vector can steal shopping turns.
        try:
            from services.ai_route_semantics import (
                brain_route_indicates_informational_kb,
                brain_route_indicates_product_catalog,
            )
            from services.product_catalog_resolver import product_catalog_route_is_locked
            from services.brain_direct_dispatch import (
                _prepare_brain_product_route,
                _run_product_catalog_flow,
            )
            from utils.helpers import extract_order_id

            _info_kb = brain_route_indicates_informational_kb(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv,
            )
            _prod_cat = (
                product_catalog_route_is_locked(brain_route_data)
                or brain_route_indicates_product_catalog(brain_route_data)
            )
            if not extract_order_id(f"{original_msg} {msg_en}".strip(), conv):
                if _prod_cat and not _info_kb:
                    try:
                        from services.brain_direct_dispatch import (
                            try_category_browse_catalog_reply,
                        )
                        from services.welfog_api import (
                            message_requests_category_browse,
                        )

                        if message_requests_category_browse(
                            original_msg
                        ) and not (
                            brain_route_data.get("category_only_browse")
                            or (brain_route_data.get("category_browse") or "").strip()
                        ):
                            cat_browse_body = try_category_browse_catalog_reply(
                                original_msg,
                                msg_en,
                                user_id=user_id,
                                lang=lang,
                                ctx=ctx,
                                reset_context_fn=reset_context,
                            )
                            if cat_browse_body:
                                log_reasoning(
                                    "Brain early — catalog-map category browse "
                                    "(overrides product title search)."
                                )
                                return _finish_early_brain(
                                    cat_browse_body, brain_route_data
                                )
                    except ImportError:
                        pass
                    product_route, sq = _prepare_brain_product_route(
                        brain_route_data, original_msg, msg_en
                    )
                    log_reasoning(
                        f"Brain early product path sq={sq!r} → OpenSearch."
                    )
                    product_body = _run_product_catalog_flow(
                        product_route,
                        sq,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conv_for_llm=conv,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                    )
                    if product_body:
                        return _finish_early_brain(product_body, brain_route_data)
        except ImportError:
            pass

        # KB — after product catalog miss (FAQ/policy/company only).
        try:
            from services.knowledge_query_pipeline import (
                try_brain_semantic_kb_reply,
                try_kb_informational_locked_reply,
            )

            kb_semantic = try_brain_semantic_kb_reply(
                original_msg,
                msg_en,
                conv,
                reply_lang=lang,
                brain_route=brain_route_data,
            )
            if kb_semantic and kb_semantic.strip():
                log_reasoning(
                    "Brain early — KB semantic lock from brain JSON (skip product catalog)."
                )
                return _finish_early_brain(
                    kb_semantic,
                    {
                        "intent": "general",
                        "data_channel": "kb",
                        "route_handler": "kb_brain_semantic",
                    },
                )

            kb_locked_body = try_kb_informational_locked_reply(
                original_msg,
                msg_en,
                conv,
                reply_lang=lang,
                brain_route=brain_route_data,
            )
            if kb_locked_body and kb_locked_body.strip():
                log_reasoning(
                    "Brain early — KB informational lock (skip product catalog)."
                )
                return _finish_early_brain(
                    kb_locked_body,
                    {
                        "intent": "general",
                        "data_channel": "kb",
                        "route_handler": "kb_informational_lock",
                    },
                )
        except ImportError:
            pass

        # Product catalog fallback — reconcile may promote shopping after KB miss.
        try:
            from services.ai_route_semantics import (
                brain_route_indicates_informational_kb,
                brain_route_indicates_product_catalog,
            )
            from services.product_catalog_resolver import product_catalog_route_is_locked
            from services.brain_direct_dispatch import (
                _prepare_brain_product_route,
                _run_product_catalog_flow,
            )
            from utils.helpers import extract_order_id

            if not extract_order_id(f"{original_msg} {msg_en}".strip(), conv):
                if brain_route_indicates_informational_kb(
                    brain_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conv,
                ):
                    pass
                elif product_catalog_route_is_locked(
                    brain_route_data
                ) or brain_route_indicates_product_catalog(brain_route_data):
                    try:
                        from services.brain_direct_dispatch import (
                            try_category_browse_catalog_reply,
                        )
                        from services.welfog_api import (
                            message_requests_category_browse,
                        )

                        if message_requests_category_browse(
                            original_msg
                        ) and not (
                            brain_route_data.get("category_only_browse")
                            or (brain_route_data.get("category_browse") or "").strip()
                        ):
                            cat_browse_body = try_category_browse_catalog_reply(
                                original_msg,
                                msg_en,
                                user_id=user_id,
                                lang=lang,
                                ctx=ctx,
                                reset_context_fn=reset_context,
                            )
                            if cat_browse_body:
                                log_reasoning(
                                    "Brain early — catalog-map category browse "
                                    "(overrides product title search)."
                                )
                                return _finish_early_brain(
                                    cat_browse_body, brain_route_data
                                )
                    except ImportError:
                        pass
                    product_route, sq = _prepare_brain_product_route(
                        brain_route_data, original_msg, msg_en
                    )
                    log_reasoning(
                        f"Brain early product path sq={sq!r} → OpenSearch."
                    )
                    product_body = _run_product_catalog_flow(
                        product_route,
                        sq,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conv_for_llm=conv,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                    )
                    if product_body:
                        return _finish_early_brain(product_body, brain_route_data)
        except ImportError:
            pass

        if not _skip_kb_for_order_id:
            immediate = _try_brain_immediate_scope_or_kb_reply(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if immediate:
                log_reasoning(
                    "Brain early — KB/scope after live APIs (one LLM max)."
                )
                return _finish_early_brain(immediate, brain_route_data)

        # Product fallback (brain JSON locked shopping but direct dispatch missed).
        try:
            from services.brain_direct_dispatch import (
                _try_brain_product_catalog_fallback_dispatch,
            )

            product_fb = _try_brain_product_catalog_fallback_dispatch(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if product_fb:
                return _finish_early_brain(product_fb, brain_route_data)
        except ImportError:
            pass

        brain_direct = try_brain_direct_dispatch(
            brain_route_data,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            format_purchase_history_reply=format_purchase_history_reply,
            format_wishlist_reply=format_wishlist_reply,
            reset_context_fn=reset_context,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
        )
        if brain_direct:
            return _finish_early_brain(brain_direct, brain_route_data)
        order_fb = _try_brain_order_live_fallback_dispatch(
            brain_route_data,
            original_msg=original_msg,
            msg_en=msg_en,
            conv_for_llm=conv,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
        )
        if order_fb:
            return _finish_early_brain(order_fb, brain_route_data)
        try:
            from services.brain_direct_dispatch import (
                _try_brain_product_catalog_fallback_dispatch,
            )

            product_fb = _try_brain_product_catalog_fallback_dispatch(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if product_fb:
                return _finish_early_brain(product_fb, brain_route_data)
        except ImportError:
            pass
        return _finish_early_brain(None, brain_route_data)
    except ImportError:
        return None, None


def _try_brain_locked_invoice_fast_reply(
    brain_route: dict,
    *,
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
) -> str | None:
    """When ai_brain_route locked invoice for one order — direct API, no dispatch detours."""
    if not isinstance(brain_route, dict):
        return None
    live_goal = ""
    try:
        from services.ai_route_semantics import resolve_order_live_goal_for_turn

        live_goal = resolve_order_live_goal_for_turn(
            brain_route,
            original_msg=original_msg,
            msg_en=msg_en,
        )
    except ImportError:
        olk = (brain_route.get("order_lookup_kind") or "").strip().lower()
        rh = (brain_route.get("route_handler") or "").strip().lower()
        intent = (brain_route.get("intent") or "").strip().lower()
        if olk == "invoice" or intent in (
            "invoice",
            "order_bill_request",
            "order_bill",
            "bill",
            "receipt",
            "order_receipt",
        ):
            if not rh or rh in ("order_details_api", ""):
                live_goal = "order_invoice"
    if live_goal != "order_invoice":
        return None
    oid = (brain_route.get("extracted_order_id") or "").strip()
    if not oid:
        try:
            from utils.helpers import resolve_order_id_for_tracking

            oid = (
                resolve_order_id_for_tracking(
                    f"{original_msg} {msg_en}".strip(),
                    "",
                    bot_awaiting_order_id=ctx.get("awaiting") == "order_id",
                    ai_extracted=brain_route.get("extracted_order_id"),
                )
                or ""
            ).strip()
        except ImportError:
            oid = ""
    if not oid:
        return None
    try:
        from services.order_id_handoff_fast_path import _fetch_details_handoff_reply
        from services.chat_flow_telemetry import log_order_dispatch, store_turn_analysis
        from services.answer_router import AnswerRouteDecision
        import time as _time

        _t_api = _time.perf_counter()
        body = _fetch_details_handoff_reply(
            "order_invoice", oid, user_id, original_msg, lang
        )
        api_ms = (_time.perf_counter() - _t_api) * 1000.0
        if not body:
            return None
        route_for_api = dict(brain_route)
        route_for_api["order_lookup_kind"] = "invoice"
        route_for_api["route_handler"] = "order_details_api"
        route_for_api["extracted_order_id"] = oid
        route_for_api["needs_order_id"] = False
        log_order_dispatch(
            detected_intent="order_invoice",
            message=f"{original_msg} {msg_en}".strip(),
            previous_context="",
            pending_action="order_invoice",
            order_id_found=oid,
            selected_tool="order_details_api",
            api_called=True,
            api_time_ms=api_ms,
        )
        store_turn_analysis(
            route_for_api,
            AnswerRouteDecision(
                source="api",
                intent="order",
                handler="order_details_api",
                is_welfog_related=True,
                reason="Brain invoice fast path — order_details_api",
            ),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context="",
        )
        ctx["order_id"] = oid
        ctx["awaiting"] = None
        ctx["last"] = "order"
        ctx.setdefault("data", {})["ai_route"] = route_for_api
        ctx["data"].pop("pending_action", None)
        log_reasoning(
            f"Brain invoice fast path: order_details_api id={oid} ({api_ms:.0f}ms API)."
        )
        return body
    except ImportError:
        return None


def _try_live_order_id_reply_early(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    user_id: str,
    lang: str,
    ctx: dict,
    ai_route: dict | None = None,
) -> str | None:
    """Run live track/refund API before ctx.last pinning or order-history shortcuts."""
    from services.order_details_flow import message_wants_order_details_or_invoice
    from utils.helpers import user_turn_qualifies_for_live_order_api

    live_goal = ""
    try:
        from services.ai_route_semantics import resolve_order_live_goal_for_turn

        if isinstance(ai_route, dict):
            live_goal = resolve_order_live_goal_for_turn(
                ai_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
    except ImportError:
        pass

    od_goal = message_wants_order_details_or_invoice(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    )
    if not od_goal:
        if live_goal == "order_invoice":
            od_goal = "order_invoice"
        elif live_goal in ("order_details", "payment"):
            od_goal = "order_details"
    if od_goal:
        try:
            from utils.helpers import resolve_order_id_for_tracking

            oid = (
                resolve_order_id_for_tracking(
                    original_msg.strip() or msg_en.strip(),
                    conv_for_llm,
                    bot_awaiting_order_id=ctx.get("awaiting") == "order_id"
                    or _message_is_order_id_followup_submission(original_msg, conv_for_llm),
                    ai_extracted=(ai_route or {}).get("extracted_order_id")
                    if isinstance(ai_route, dict)
                    else None,
                )
                or ""
            ).strip()
            if oid:
                from services.order_id_handoff_fast_path import _fetch_details_handoff_reply
                import time as _time

                log_reasoning(
                    f"Details/invoice guard dispatch: goal={od_goal} id={oid} "
                    f"(skip tracking API)."
                )
                _t_api = _time.perf_counter()
                ai_focus = ""
                if isinstance(ai_route, dict):
                    ai_focus = (ai_route.get("field_focus") or "").strip()
                details_goal = "order_details" if od_goal == "payment" else od_goal
                body = _fetch_details_handoff_reply(
                    details_goal,
                    oid,
                    user_id,
                    original_msg,
                    lang,
                    ai_focus=ai_focus or ("payment" if od_goal == "payment" else ""),
                )
                api_ms = (_time.perf_counter() - _t_api) * 1000.0
                try:
                    from services.chat_flow_telemetry import log_order_dispatch

                    log_order_dispatch(
                        detected_intent=od_goal,
                        message=f"{original_msg} {msg_en}".strip(),
                        previous_context=(conv_for_llm or "")[:200],
                        pending_action=od_goal,
                        order_id_found=oid,
                        selected_tool="order_details_api",
                        api_called=bool(body),
                        api_time_ms=api_ms,
                    )
                except ImportError:
                    pass
                if body:
                    ctx["order_id"] = oid
                    ctx["awaiting"] = None
                    ctx["last"] = "invoice" if od_goal == "order_invoice" else "order"
                    return body
        except ImportError:
            pass
        log_reasoning(
            f"Skip live track early path — customer wants {od_goal} (details/invoice API)."
        )
        return None

    track_turn = live_goal == "track"
    if not track_turn:
        try:
            from utils.helpers import (
                _text_is_order_tracking_intent_leaf,
                message_is_general_delivery_policy_question,
            )

            comb_track = f"{original_msg} {msg_en}".strip()
            track_turn = bool(
                not message_is_general_delivery_policy_question(comb_track)
                and _text_is_order_tracking_intent_leaf(comb_track)
            )
        except ImportError:
            track_turn = False

    if track_turn:
        try:
            from utils.helpers import resolve_order_id_for_tracking
            import time as _time

            oid = (
                resolve_order_id_for_tracking(
                    original_msg.strip() or msg_en.strip(),
                    conv_for_llm,
                    bot_awaiting_order_id=ctx.get("awaiting") == "order_id"
                    or _message_is_order_id_followup_submission(original_msg, conv_for_llm),
                    ai_extracted=(ai_route or {}).get("extracted_order_id")
                    if isinstance(ai_route, dict)
                    else None,
                )
                or ""
            ).strip()
            if oid:
                _t_api = _time.perf_counter()
                body = _reply_for_live_order_id_lookup(
                    "order", oid, user_id, original_msg, lang
                )
                api_ms = (_time.perf_counter() - _t_api) * 1000.0
                try:
                    from services.chat_flow_telemetry import log_order_dispatch

                    log_order_dispatch(
                        detected_intent="track",
                        message=f"{original_msg} {msg_en}".strip(),
                        previous_context=(conv_for_llm or "")[:200],
                        pending_action="track",
                        order_id_found=oid,
                        selected_tool="order_tracking_api",
                        api_called=bool(body),
                        api_time_ms=api_ms,
                    )
                except ImportError:
                    pass
                if body:
                    log_reasoning(
                        f"Tracking guard dispatch: goal=track id={oid} "
                        f"({api_ms:.0f}ms API)."
                    )
                    ctx["order_id"] = oid
                    ctx["awaiting"] = None
                    ctx["last"] = "order"
                    return body
        except ImportError:
            pass

    if not user_turn_qualifies_for_live_order_api(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    ):
        return None
    if not should_attempt_live_order_api_reply(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    ):
        return None
    pending_oid = resolve_order_id_for_tracking(
        original_msg.strip() or msg_en.strip(),
        conv_for_llm,
        bot_awaiting_order_id=ctx.get("awaiting") == "order_id"
        or _message_is_order_id_followup_submission(original_msg, conv_for_llm),
    )
    if not pending_oid:
        return None
    live_intent = resolve_live_api_intent_from_conversation(
        conv_for_llm,
        ctx.get("last"),
        original_msg,
        msg_en,
        ai_route=ai_route if isinstance(ai_route, dict) else None,
    )
    log_reasoning(
        f"Semantic live {live_intent} for Order ID {pending_oid} "
        "(before ctx.last / keyword shortcuts)."
    )
    ctx["last"] = live_intent if live_intent in ("refund", "payment", "order") else "order"
    ctx["order_id"] = pending_oid
    ctx["awaiting"] = None
    return _reply_for_live_order_id_lookup(
        live_intent, pending_oid, user_id, original_msg, lang
    )


def _try_pincode_delivery_reply_early(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str,
    lang: str,
    ctx: dict,
    ai_route: dict | None = None,
) -> str | None:
    """Live pincode serviceability API before KB dumps or order-id misread."""
    from services.entity_first_handlers import try_pincode_delivery_reply

    return try_pincode_delivery_reply(
        original_msg, msg_en, conv_for_llm, lang, ctx, ai_route=ai_route
    )


def _normalize_reply_lang(reply_lang: str, user_msg: str) -> str:
    lang = (reply_lang or "en").lower().strip()
    if lang == "hinglish" or is_hinglish_message(user_msg):
        return "hinglish"
    return lang


def _tracking_help_reply(user_msg: str, reply_lang: str = "en") -> str:
    return (
        _localized_sysmsg("tracking_help", user_msg, reply_lang=reply_lang)
        or sysmsg("tracking_help")
        or sysmsg("how_can_i_help")
    )


def _resolve_user_id():
    if request.method == "POST":
        payload = request.json or {}
        return request.args.get("user_id") or payload.get("user_id") or DEFAULT_USER_ID
    return request.args.get("user_id") or DEFAULT_USER_ID

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def home():
    return render_template("index.html")


@chat_bp.route("/wishlist")
def wishlist_page():
    user_id = _resolve_user_id()
    return render_template(
        "wishlist.html",
        user_id=user_id,
        wishlist_html=format_wishlist_page_html(user_id, page=1),
    )


@chat_bp.route("/api/chat/new", methods=["POST"])
def new_chat_reset():
    user_id = str(_resolve_user_id() or "").strip()
    if not user_id:
        return jsonify({"status": "cleared"})
    prefix = f"{user_id}:"
    for key in list(user_contexts.keys()):
        if key == user_id or key.startswith(prefix):
            reset_context(user_contexts[key])
    try:
        from services.chat_resilience import force_end_all_user_chat_turns, force_end_stuck_chat_turn

        data = request.get_json(silent=True) or {}
        force_end_stuck_chat_turn(chat_id=str(data.get("chat_id") or ""), user_id=user_id)
        force_end_all_user_chat_turns(user_id)
    except ImportError:
        pass
    return jsonify({"status": "cleared"})


@chat_bp.route("/api/chat/unlock", methods=["POST"])
def unlock_stuck_chat():
    """Release per-chat mutex after client timeout so the user can send again."""
    user_id = str(_resolve_user_id() or "").strip()
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get("chat_id") or "").strip()
    try:
        from services.chat_resilience import force_end_stuck_chat_turn

        force_end_stuck_chat_turn(chat_id=chat_id, user_id=user_id)
    except ImportError:
        pass
    return jsonify({"ok": True, "chat_id": chat_id or None})

@chat_bp.route("/api/chats", methods=["GET"])
def get_history():
    user_id = _resolve_user_id()
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    conn = get_mysql_connection()
    if not conn:
        return jsonify([])
    try:
        with conn.cursor() as cur:
            # List only sessions that still have rows in `chats` with real JSON messages.
            # Otherwise clearing `chats` in phpMyAdmin leaves "ghost" titles from `chat_sessions`.
            cur.execute(
                f"""
                SELECT DISTINCT
                    COALESCE(cs.chat_token, c.chat_token, CAST(c.chat_id AS CHAR)) AS chat_id,
                    COALESCE(
                        NULLIF(TRIM(cs.title), ''),
                        'New chat'
                    ) AS title,
                    COALESCE(cs.created_at, c.created_at, c.updated_at) AS created_at
                FROM chats c
                LEFT JOIN chat_sessions cs ON (
                    {sql_collate('cs.chat_token')} = {sql_collate('c.chat_token')}
                    OR {sql_collate('cs.chat_token')} = {sql_collate('CAST(c.chat_id AS CHAR)')}
                    OR {sql_collate('CAST(cs.id AS CHAR)')} = {sql_collate('CAST(c.chat_id AS CHAR)')}
                )
                WHERE ({sql_collate('cs.user_id')} = %s OR {sql_collate('c.user_id')} = %s)
                  AND COALESCE(cs.created_at, c.created_at, c.updated_at) >= %s
                  AND c.chat_data IS NOT NULL
                  AND CHAR_LENGTH(TRIM(c.chat_data)) > 5
                ORDER BY created_at DESC
                """,
                (user_id, user_id, seven_days_ago),
            )
            rows = cur.fetchall()
            chats = []
            for r in rows:
                ts = r["created_at"]
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts).split(" ")[0]
                title = (r.get("title") or "").strip()
                if not title or title.lower() == "new chat":
                    try:
                        cur.execute(
                            "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                            (r["chat_id"], r["chat_id"]),
                        )
                        crow = cur.fetchone()
                        if crow and crow.get("chat_data"):
                            data = json.loads(crow["chat_data"])
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and item.get("sender") == "user":
                                        t = (item.get("text") or "").strip()
                                        if t:
                                            title = (t[:30] + "...") if len(t) > 30 else t
                                            break
                    except Exception:
                        pass
                title = title or "New chat"
                chats.append({"chat_id": r["chat_id"], "title": title, "date_str": date_str})
        return jsonify(chats)
    except Exception as e:
        print(f"❌ get_history MySQL error: {e}")
        return jsonify([])
    finally:
        conn.close()

@chat_bp.route("/api/chat/delete/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    user_id = _resolve_user_id()
    conn = get_mysql_connection()
    if not conn:
        return jsonify({"error": "database_unreachable", "message": "Database unavailable."}), 503
    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, chat_token FROM chat_sessions "
                "WHERE (chat_token = %s OR id = %s) AND user_id = %s LIMIT 1",
                (chat_id, numeric_chat_id if numeric_chat_id is not None else -1, user_id),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "not_found", "message": "Chat not found or not owned by you."}), 404

            session_id = row["id"]
            token = row["chat_token"]
            cur.execute(
                "DELETE FROM chats WHERE chat_token = %s OR chat_id = %s OR chat_id = %s OR chat_id = %s",
                (token, token, str(session_id), session_id),
            )
            cur.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
        conn.commit()
        return jsonify({"status": "deleted"})
    except Exception as e:
        conn.rollback()
        print(f"❌ delete_chat MySQL error: {e}")
        return jsonify({"error": "database_error", "message": "Unable to delete chat."}), 500
    finally:
        conn.close()

@chat_bp.route("/api/chat/messages/<chat_id>", methods=["GET"])
def get_messages(chat_id):
    conn = get_mysql_connection()
    if not conn:
        return jsonify([])  # Agar DB connect na ho toh khali list bhej do
        
    try:
        cursor = conn.cursor()
        numeric_chat_id = int(chat_id) if chat_id.isdigit() and len(chat_id) < 12 else None
        if numeric_chat_id is not None:
            cursor.execute(
                "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                (chat_id, numeric_chat_id),
            )
        else:
            cursor.execute("SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1", (chat_id,))
        row = cursor.fetchone()

        if not row or not row.get("chat_data"):
            return jsonify({"error": "not_found"}), 404

        try:
            chat_data = json.loads(row["chat_data"])
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return jsonify({"error": "invalid_data"}), 500

        if isinstance(chat_data, dict):
            chat_data = [chat_data]
        elif not isinstance(chat_data, list):
            return jsonify({"error": "not_found"}), 404

        if len(chat_data) == 0:
            return jsonify({"error": "not_found"}), 404

        msgs = []
        for item in chat_data:
            if not isinstance(item, dict):
                continue
            msgs.append({
                "sender": item.get("sender"),
                "message": item.get("text") if item.get("text") is not None else item.get("message"),
            })

        return jsonify(msgs)
        
    except Exception as e:
        print(f"❌ MySQL Fetch Error: {e}")
        return jsonify([])
    finally:
        if conn:
            conn.close()


def _pagination_append_json_response(
    data: dict,
    user_id: str,
    chat_id: str | None,
    ctx: dict | None = None,
):
    """
    View-more pagination — return card fragments for in-place append (same chat bubble).
    Must run before brain/KB stack so empty message + page param never spawns a new bubble.
    """
    if not isinstance(data, dict):
        return None
    current_chat_id = (chat_id or "").strip() or None
    ctx = ctx if isinstance(ctx, dict) else {}

    ps_raw = data.get("product_search_page")
    if ps_raw is not None:
        if not current_chat_id:
            return jsonify({"error": "chat_id_required", "message": "Missing chat session."}), 400
        try:
            ps_page = max(2, int(ps_raw))
        except (TypeError, ValueError):
            ps_page = 2
        log_reasoning(f"Product search pagination page={ps_page}")
        append_parts = format_product_search_append_payload(ctx.get("data") or {}, ps_page)
        return jsonify(
            {
                "chat_id": current_chat_id,
                "type": "product_search_append",
                "cards_html": append_parts.get("cards_html", ""),
                "tail_html": append_parts.get("tail_html", ""),
            }
        )

    ph_raw = data.get("purchase_history_page")
    if ph_raw is not None:
        if not current_chat_id:
            return jsonify({"error": "chat_id_required", "message": "Missing chat session."}), 400
        try:
            ph_page = max(1, int(ph_raw))
        except (TypeError, ValueError):
            ph_page = 1
        log_reasoning(f"Purchase history pagination page={ph_page}")
        append_parts = format_purchase_history_append_payload(user_id, ph_page)
        return jsonify(
            {
                "chat_id": current_chat_id,
                "type": "purchase_history_append",
                "cards_html": append_parts.get("cards_html", ""),
                "tail_html": append_parts.get(
                    "tail_html", append_parts.get("footer_html", "")
                ),
            }
        )

    wl_raw = data.get("wishlist_page")
    if wl_raw is not None:
        try:
            wl_page = max(1, int(wl_raw))
        except (TypeError, ValueError):
            wl_page = 1
        log_reasoning(f"Wishlist pagination page={wl_page}")
        append_parts = format_wishlist_append_payload(user_id, wl_page)
        out = {
            "type": "wishlist_append",
            "cards_html": append_parts.get("cards_html", ""),
            "tail_html": append_parts.get("tail_html", ""),
        }
        if current_chat_id:
            out["chat_id"] = current_chat_id
        return jsonify(out)

    return None


def _chat_request_guard(fn):
    """Wall-clock limit + polite busy reply (rate limit / timeout) instead of hanging."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        from services.chat_resilience import (
            ChatDeadlineExceeded,
            build_busy_reply_html,
            log_busy_fallback,
            run_with_chat_deadline,
        )

        t0 = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        user_msg = _normalize_alphanumeric_id_boundaries(
            (payload.get("message") or "").strip()
        )
        chat_id = payload.get("chat_id")
        if not chat_id and user_msg:
            from services.mysql_service import generate_chat_token

            chat_id = generate_chat_token()
            g.provisional_chat_id = chat_id
        app = current_app._get_current_object()
        guard_user_id = str(_resolve_user_id())
        chat_log(f"INCOMING user_id={guard_user_id} chat_id={chat_id!r} msg={user_msg[:80]!r}")
        try:
            from services.chat_flow_telemetry import (
                begin_chat_turn,
                record_user_query,
            )

            begin_chat_turn()
            if user_msg:
                record_user_query(user_msg, "")
        except ImportError:
            pass
        if any(
            payload.get(k) is not None
            for k in ("purchase_history_page", "wishlist_page", "product_search_page")
        ):
            try:
                ctx_pg = _get_or_create_user_ctx(guard_user_id, chat_id)
                pag_resp = _pagination_append_json_response(
                    payload, guard_user_id, chat_id, ctx_pg
                )
                if pag_resp is not None:
                    chat_log(
                        f"guard pagination append done in {time.perf_counter() - t0:.2f}s"
                    )
                    return pag_resp
            except Exception as pag_exc:
                chat_log(f"guard pagination skip: {pag_exc}")
        turn_acquired = False
        try:
            from services.translation_service import resolve_customer_reply_lang

            lang_hint = resolve_customer_reply_lang(user_msg) if user_msg else "en"
        except ImportError:
            lang_hint = customer_reply_language(user_msg) if user_msg else "en"
        try:
            session_hit = _guard_session_fast_reply(
                user_msg, chat_id, guard_user_id, lang_hint
            )
            if session_hit is not None:
                chat_log(
                    f"guard session fast (early) done in {time.perf_counter() - t0:.2f}s"
                )
                return session_hit
        except Exception as sess_exc:
            chat_log(f"guard session fast (early) skip: {sess_exc}")
        try:
            from services.translation_service import to_en_for_routing

            if lang_hint in ("en", "hinglish", ""):
                _pre_en = (user_msg or "").lower()
            else:
                _pre_en = to_en_for_routing(user_msg, lang_hint) if user_msg else ""
        except ImportError:
            _pre_en = (user_msg or "").lower()
        order_prelock = _guard_order_live_ai_prelock_reply(
            user_msg,
            chat_id,
            guard_user_id,
            _pre_en,
            lang_hint,
        )
        if order_prelock is not None:
            chat_log(
                f"guard order-live prelock done in {time.perf_counter() - t0:.2f}s"
            )
            return order_prelock
        # AI-first: order-ask-id handled by ai_brain_route + brain dispatch (no keyword prelock).
        # AI-first: keep only deterministic order/session fast paths in guard.
        # All other intent understanding and routing runs once inside chat().
        # One ai_brain_route per turn inside chat() — reuses cached brain route.
        try:
            from services.chat_resilience import clear_turn_acquire_state

            clear_turn_acquire_state()
        except ImportError:
            pass
        try:
            from services.chat_resilience import (
                end_chat_turn,
                try_begin_chat_turn,
            )

            try_begin_chat_turn(chat_id or "", guard_user_id)
            turn_acquired = True
        except ImportError:
            turn_acquired = True
        try:
            resp = run_with_chat_deadline(fn, args, kwargs, app=app)
            chat_log(f"done in {time.perf_counter() - t0:.2f}s")
            return resp
        except ChatDeadlineExceeded:
            try:
                from services.chat_flow_telemetry import record_timeout_point

                record_timeout_point("chat_deadline_exceeded")
            except ImportError:
                pass
            try:
                from services.chat_resilience import force_end_stuck_chat_turn

                force_end_stuck_chat_turn(chat_id or "", guard_user_id)
                turn_acquired = False
            except ImportError:
                pass
            log_busy_fallback(f"deadline>{time.perf_counter() - t0:.1f}s")
            try:
                from services.conversation_zero_llm_fallback import (
                    try_zero_llm_customer_reply,
                )

                recovered = try_zero_llm_customer_reply(
                    user_msg,
                    user_msg.lower() if lang_hint in ("en", "hinglish") else user_msg,
                    _conversation_snapshot_for_chat(chat_id, limit=6),
                    reply_lang=lang_hint,
                    ctx=_get_or_create_user_ctx(guard_user_id, chat_id),
                )
                if recovered:
                    chat_log(
                        f"deadline recovery — zero-LLM KB/chitchat ({time.perf_counter() - t0:.1f}s)"
                    )
                    return jsonify(
                        {
                            "type": "text",
                            "data": recovered,
                            "degraded": False,
                            "reason": "deadline_recovered",
                            "chat_id": chat_id,
                        }
                    )
            except ImportError:
                pass
            return (
                jsonify(
                    {
                        "type": "text",
                        "data": build_busy_reply_html(user_msg, lang_hint),
                        "degraded": True,
                        "reason": "timeout",
                        "chat_id": chat_id,
                    }
                ),
                200,
            )
        except Exception as exc:
            from services.chat_resilience import LLMProvidersBusy, should_return_busy_fallback

            chat_log(f"ERROR after {time.perf_counter() - t0:.2f}s: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if isinstance(exc, LLMProvidersBusy) or should_return_busy_fallback():
                reason = "busy"
                body = build_busy_reply_html(user_msg, lang_hint)
            else:
                reason = "error"
                from services.translation_service import customer_facing_template

                body = customer_facing_template(
                    "server_technical_issue",
                    user_msg or "help",
                    lang_hint,
                    wrap_html=True,
                    fallback_en=(
                        "Sorry — something went wrong on our side. "
                        "Please try again in a moment."
                    ),
                ) or build_busy_reply_html(user_msg, lang_hint)
            return (
                jsonify(
                    {
                        "type": "text",
                        "data": body,
                        "degraded": True,
                        "reason": reason,
                        "chat_id": chat_id,
                    }
                ),
                200,
            )
        finally:
            try:
                from services.chat_resilience import (
                    clear_turn_acquire_state,
                    end_chat_turn_if_acquired,
                    force_end_stuck_chat_turn,
                )

                end_chat_turn_if_acquired()
                force_end_stuck_chat_turn(chat_id or "", guard_user_id)
                clear_turn_acquire_state()
            except ImportError:
                pass
            try:
                from services.chat_flow_telemetry import log_pipeline_complete

                log_pipeline_complete(user_query=user_msg)
            except ImportError:
                pass

    return wrapper


def _try_llm_down_transactional_recovery(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
    *,
    conv_for_llm: str = "",
) -> tuple[str | None, dict | None]:
    """
    Brain LLM budget exhausted — structural live API only.
    Never dump full order history for single-order track/invoice/refund turns.
    """
    order_body, order_route = _try_instant_order_id_structural_reply(
        original_msg,
        msg_en,
        user_id,
        lang,
        ctx,
        conv_for_llm=conv_for_llm,
    )
    if order_body:
        log_reasoning("LLM-down recovery — order id + live API (structural).")
        return order_body, order_route

    try:
        from services.brain_direct_dispatch import try_structural_order_live_reply

        struct_body = try_structural_order_live_reply(
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
        )
        if struct_body:
            route = (ctx.get("data") or {}).get("ai_route") or {
                "intent": "order",
                "data_channel": "live_api",
                "route_handler": "structural_order_llm_down",
            }
            log_reasoning("LLM-down recovery — structural order ask-id/live API.")
            return struct_body, route
    except ImportError:
        pass

    pin_body, pin_route = _try_instant_pincode_delivery_reply(
        original_msg,
        msg_en,
        lang,
        ctx,
        conv_for_llm=conv_for_llm,
        allow_llm=False,
    )
    if pin_body:
        log_reasoning("LLM-down recovery — pincode delivery API (no micro-LLM).")
        return pin_body, pin_route

    return None, None


def _try_order_history_list_fast_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    *,
    ctx: dict | None = None,
    conversation_context: str = "",
) -> tuple[str | None, dict | None]:
    """Past orders / purchase history in chat — live API list, never ask Order ID."""
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return None, None
    try:
        from utils.helpers import (
            _text_wants_order_history_list_in_chat,
            message_asks_my_welfog_purchases,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
        )
        from services.semantic_intent import should_skip_order_history_list_for_turn
        from services.welfog_api import format_purchase_history_reply

        if should_skip_order_history_list_for_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=None,
            ctx=ctx,
        ):
            return None, None
        if message_is_wishlist_like_request(comb):
            return None, None
        if not (
            _text_wants_order_history_list_in_chat(comb, "")
            or message_is_past_purchase_list_request(comb)
            or message_asks_my_welfog_purchases(comb)
        ):
            return None, None
        body = format_purchase_history_reply(user_id, page=1, append_only=False)
        if not body:
            return None, None
        log_reasoning(
            "Order history list — purchase-history API (not single-order track)."
        )
        return body, {
            "intent": "order_history",
            "data_channel": "live_api",
            "route_handler": "purchase_history_api",
            "account_list_kind": "purchase_history_in_chat",
            "needs_order_id": False,
        }
    except ImportError:
        pass
    return None, None


def _try_zero_llm_order_nl_fast_reply(
    original_msg: str,
    msg_en: str,
    user_id: str,
    lang: str,
    ctx: dict,
) -> tuple[str | None, dict | None]:
    """
    Zero-LLM order ask-id — LLM-unavailable failsafe only.
    Under STRICT_AI_INTENT_ROUTING, ai_brain_route classifies intent in any language,
    then brain_direct_dispatch asks for Order ID and stores pending_action / ai_route.
    """
    try:
        from services.semantic_intent import strict_ai_semantic_mode

        if strict_ai_semantic_mode():
            return None, None
    except ImportError:
        pass
    comb_nl = f"{original_msg} {msg_en}".strip()
    nl_goal = ""
    comb_low = comb_nl.lower()
    try:
        from utils.helpers import (
            _text_wants_order_history_list_in_chat,
            message_is_past_purchase_list_request,
            message_asks_my_welfog_purchases,
        )

        if (
            _text_wants_order_history_list_in_chat(comb_nl, "")
            or message_is_past_purchase_list_request(comb_nl)
            or message_asks_my_welfog_purchases(comb_nl)
        ):
            return None, None
    except ImportError:
        pass
    if re.search(r"\b(?:invoice|bill|receipt)\b", comb_nl, re.I) and not any(
        m in comb_low
        for m in (
            "track",
            "tracking",
            "kab aa",
            "kab tak",
            "nhi aa",
            "nahi aa",
            "nhi aaya",
            "nahi aaya",
        )
    ):
        nl_goal = "order_invoice"
    elif any(
        m in comb_low
        for m in (
            "track",
            "tracking",
            "kab aa",
            "kab tak",
            "nhi aa",
            "nahi aa",
            "nhi aaya",
            "nahi aaya",
            "order nhi aa",
        )
    ) and re.search(r"\border", comb_low):
        nl_goal = "track"
    if nl_goal and not re.search(r"\b[0-9]{4,20}\b", comb_nl):
        try:
            from services.order_history_flow import (
                _localized_sysmsg as order_history_localized_sysmsg,
            )

            ask_intent = (
                "refund"
                if nl_goal == "refund_status"
                else "invoice"
                if nl_goal == "order_invoice"
                else "order"
            )
            nl_body = order_history_localized_sysmsg(
                "ask_order_id_for_intent",
                original_msg,
                reply_lang=lang or "en",
                intent=ask_intent,
            )
            if nl_body and isinstance(ctx, dict):
                ctx["order_id"] = None
                ctx["awaiting"] = "order_id"
                ctx["last"] = ask_intent
                ctx.setdefault("data", {})["pending_action"] = nl_goal
                ctx["data"]["topic_mode"] = f"order_{nl_goal}"
                route_snapshot = {
                    "intent": "refund" if nl_goal == "refund_status" else "order",
                    "data_channel": "live_api",
                    "needs_order_id": True,
                    "numeric_context": "order_id",
                    "order_lookup_kind": nl_goal,
                    "route_handler": "order_nl_structural_fast",
                }
                try:
                    from services.ai_route_semantics import LIVE_API_FROM_GOAL

                    olk_map = {
                        "track": "track",
                        "order_invoice": "invoice",
                        "order_details": "details",
                        "payment": "details",
                        "refund_status": "refund_status",
                    }
                    route_snapshot["order_lookup_kind"] = olk_map.get(nl_goal, nl_goal)
                    route_snapshot["route_handler"] = LIVE_API_FROM_GOAL.get(
                        nl_goal, route_snapshot["route_handler"]
                    )
                except ImportError:
                    pass
                ctx["data"]["ai_route"] = route_snapshot
                log_reasoning(
                    f"Natural-language order ask-id fast path: goal={nl_goal} (zero LLM)."
                )
                return nl_body, route_snapshot
        except ImportError:
            pass
    try:
        from services.order_live_intent_fast_path import (
            _structural_message_live_goal_no_id,
            try_order_live_intent_fast_reply,
        )

        if not nl_goal:
            nl_goal = _structural_message_live_goal_no_id(original_msg, msg_en, "")
        if nl_goal:
            nl_body = try_order_live_intent_fast_reply(
                original_msg,
                msg_en,
                "",
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
                preset_goal=nl_goal,
            )
            if nl_body:
                log_reasoning(
                    f"Natural-language order structural fast path: goal={nl_goal} (zero LLM)."
                )
                route_snapshot = (ctx.get("data") or {}).get("ai_route") or {
                    "intent": "order",
                    "data_channel": "live_api",
                    "route_handler": "order_nl_structural_fast",
                    "order_lookup_kind": nl_goal,
                    "needs_order_id": True,
                }
                return nl_body, route_snapshot
    except ImportError:
        pass
    return None, None


@chat_bp.route("/chat", methods=["POST"])
@_chat_request_guard
def chat():
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    user_id = _resolve_user_id()
    chat_log(f"POST user_id={user_id} msg={user_msg[:100]!r}")

    # 🔥 FIX 1: Frontend se chat_id fetch karo
    current_chat_id = data.get("chat_id") or getattr(g, "provisional_chat_id", None)

    ctx = _get_or_create_user_ctx(user_id, current_chat_id)

    pag_early = _pagination_append_json_response(
        data, user_id, current_chat_id, ctx
    )
    if pag_early is not None:
        return pag_early

    defer_chat_session_insert = False
    if not current_chat_id:
        current_chat_id = generate_chat_token()
        defer_chat_session_insert = True
        ctx = _bind_ctx_to_chat_id(user_id, current_chat_id, ctx)
    elif getattr(g, "provisional_chat_id", None) == current_chat_id:
        defer_chat_session_insert = True

    original_msg = user_msg.strip()
    original_msg = _normalize_alphanumeric_id_boundaries(original_msg)
    try:
        from services.translation_service import resolve_customer_reply_lang, to_en_for_routing

        lang = resolve_customer_reply_lang(original_msg)
        msg_en = to_en_for_routing(original_msg, lang)
    except ImportError:
        lang = customer_reply_language(original_msg)
        if lang in ("en", "hinglish"):
            msg_en = original_msg.lower().strip()
        else:
            msg_en = to_en(original_msg).lower().strip()
    _brain_route_first = _turn_should_prioritize_brain_route(
        original_msg, msg_en, "", ctx
    )
    if _brain_route_first and isinstance(ctx, dict):
        try:
            from utils.helpers import (
                clear_order_session_for_new_lookup,
                message_is_bare_numeric_submission,
            )

            comb_bf = f"{original_msg or ''} {msg_en or ''}".strip()
            has_oid = bool(re.search(r"\b[0-9]{4,20}\b", comb_bf))
            bare_num = message_is_bare_numeric_submission(original_msg)
            if ctx.get("awaiting") in ("order_id", "pincode") and not bare_num and not has_oid:
                log_reasoning(
                    "Brain-first — clear stale order/pincode session for fresh NL turn."
                )
                clear_order_session_for_new_lookup(ctx)
                ctx["awaiting"] = None
                ctx["last"] = None
                data = ctx.get("data")
                if isinstance(data, dict):
                    data.pop("pending_action", None)
                    data.pop("topic_mode", None)
        except ImportError:
            pass
    if not _brain_route_first:
        conv_for_llm = _conversation_context_for_routing(
            current_chat_id, original_msg, msg_en, ctx
        )
        _sanitize_stale_ctx_for_fresh_intent(
            original_msg, msg_en, ctx, conversation_context=conv_for_llm
        )
    else:
        conv_for_llm = ""

    # Pre-extract glued order ids for downstream live API handlers (skip before brain — brain JSON owns entities).
    comb_pre = f"{original_msg or ''} {msg_en or ''}".strip()
    glued_oid = ""
    if not _brain_route_first and not _message_is_complete_standalone_question(comb_pre):
        try:
            glued_oid = _extract_attached_welfog_order_id(
                original_msg, conv_for_llm
            ) or ""
        except ImportError:
            glued_oid = _extract_attached_welfog_order_id(
                original_msg, conv_for_llm
            ) or ""
    if glued_oid:
        ctx.setdefault("data", {})["extracted_order_id"] = glued_oid
        log_reasoning(f"Alphanumeric guard — attached order id {glued_oid} for routing.")

    chat_request_id = "-"
    try:
        from services.chat_flow_telemetry import ensure_chat_turn_started, record_user_query

        chat_request_id = ensure_chat_turn_started()
        record_user_query(original_msg, lang)
        log_reasoning(f"[chat-flow] request_id={chat_request_id} turn started")
    except ImportError:
        pass

    def _fast_path_json_reply(
        body: str,
        *,
        ai_route_snapshot: dict | None = None,
    ):
        """Fast paths — align KB/chitchat copy to customer language; keep live API HTML as-is."""
        if not isinstance(body, str) or not body.strip():
            body = sysmsg("cancelled") or "OK"
        try:
            from services.translation_service import (
                finalize_customer_reply,
                is_live_api_structured_html,
                resolve_customer_reply_lang,
            )

            rl = resolve_customer_reply_lang(original_msg, lang)
            if not is_live_api_structured_html(body):
                body = finalize_customer_reply(body, original_msg, rl)
        except ImportError:
            pass
        try:
            import threading

            from services.mysql_service import db_store_turn_pair

            _cid, _uid, _um, _out = current_chat_id, str(user_id), user_msg, body

            threading.Thread(
                target=lambda: db_store_turn_pair(_cid, _um, _out, _uid),
                daemon=True,
            ).start()
        except Exception as db_exc:
            chat_log(f"fast path DB skip: {db_exc}")
        preview = body[:120].replace("\n", " ")
        chat_log(f"reply sent ({len(body)} chars): {preview!r}")
        try:
            from services.chat_flow_telemetry import (
                log_routing_decision,
                log_turn_complete,
                mark_routing_complete,
                record_route,
                response_time_sec,
            )

            ar = ai_route_snapshot or (ctx.get("data") or {}).get("ai_route") or {}
            record_route(
                intent=ar.get("intent") or "",
                source=ar.get("route_handler") or ar.get("data_channel") or "",
            )
            mark_routing_complete()
            log_routing_decision(
                query=original_msg,
                language=lang,
                intent=ar.get("intent") or "",
                selected_tool=ar.get("route_handler") or ar.get("data_channel") or "",
                api_time_ms=response_time_sec() * 1000.0,
            )
            log_turn_complete(
                intent=ar.get("intent") or "",
                route=ar.get("route_handler") or "",
                source=ar.get("data_channel") or "",
                reason=ar.get("reasoning") or "",
            )
        except Exception:
            pass
        return jsonify({"chat_id": current_chat_id, "type": "text", "data": body})

    # === INSTANT GREETING (<1s — never queue behind OpenSearch/KB) ===
    try:
        from services.chitchat_resolver import _is_instant_greeting_thanks_lane
        from utils.helpers import fast_greeting_reply_html

        if (
            _is_instant_greeting_thanks_lane(
                original_msg, msg_en, conv_for_llm
            )
            and not _instant_lane_blocked_by_transactional(
                original_msg, msg_en, conv_for_llm
            )
        ):
            greet_html = fast_greeting_reply_html(original_msg, reply_lang=lang)
            if greet_html:
                log_reasoning("Chat — instant greeting (zero LLM/API).")
                return _fast_path_json_reply(
                    greet_html,
                    ai_route_snapshot={
                        "intent": "general",
                        "conversation_scope": "general_chitchat",
                        "data_channel": "none",
                        "meta_kind": "conversational",
                    },
                )
    except ImportError:
        pass

    # === ORDER SESSION FIRST (PIN / order id — zero LLM, live API) ===
    _t_order_sess = time.perf_counter()
    comb_standalone = f"{original_msg or ''} {msg_en or ''}".strip()
    ctx_sess_body = None
    ctx_sess_route = ""
    skip_sess_cont = bool(
        _brain_route_first
        and _message_is_complete_standalone_question(comb_standalone)
        and not (
            isinstance(ctx, dict)
            and ctx.get("awaiting") in ("order_id", "pincode")
        )
    )
    if skip_sess_cont:
        log_reasoning(
            "Brain-first standalone — skip order-session continuation (no MySQL)."
        )
    else:
        ctx_sess_body, ctx_sess_route = _try_ctx_continuation_reply(
            original_msg, msg_en, user_id, lang, ctx, chat_id=current_chat_id
        )
    if ctx_sess_body:
        try:
            from services.chat_flow_telemetry import record_phase, record_route

            record_phase(
                "order_session_first",
                (time.perf_counter() - _t_order_sess) * 1000.0,
            )
            ar = (ctx.get("data") or {}).get("ai_route") or {}
            record_route(
                intent=ar.get("intent") or "order",
                source=ar.get("route_handler") or ctx_sess_route,
            )
        except ImportError:
            pass
        log_reasoning(
            f"Order session fast path ({ctx_sess_route}) — zero LLM, live API."
        )
        return _fast_path_json_reply(
            ctx_sess_body,
            ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
            or {
                "intent": "order",
                "data_channel": "live_api",
                "route_handler": ctx_sess_route,
            },
        )

    order_body, order_route = _try_instant_order_id_structural_reply(
        original_msg, msg_en, user_id, lang, ctx, conv_for_llm=conv_for_llm
    )
    if order_body:
        log_reasoning(
            "Pre-brain — order id + live API (invoice/track/refund, zero brain wait)."
        )
        return _fast_path_json_reply(
            order_body,
            ai_route_snapshot=order_route,
        )

    brain_uni_body = None
    brain_uni_route = None
    _t_brain_universal = time.perf_counter()

    conv_for_llm = conv_for_llm or _conversation_context_for_routing(
        current_chat_id, original_msg, msg_en, ctx
    )

    if _brain_route_first and isinstance(ctx, dict):
        data = ctx.get("data")
        if isinstance(data, dict) and (data.get("topic_mode") or "").strip().lower() == "pincode_check":
            data.pop("topic_mode", None)
            log_reasoning(
                "AI-first turn — clear stale pincode_check ctx; brain re-classifies from conv."
            )

    # === AI-FIRST STRICT MODE ===
    # Skip micro-LLM preflight classifiers; route once via universal brain below.
    # === AI BRAIN FIRST (one LLM — intent in any language, then dispatch) ===
    if _brain_route_first:
        bf_body, bf_route = _try_brain_first_chat_turn(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm,
            current_chat_id,
        )
        if bf_body:
            log_reasoning(
                "Universal brain — AI intent + dispatch (keyword-free routing)."
            )
            return _fast_path_json_reply(
                bf_body,
                ai_route_snapshot=bf_route,
            )

    # === Structural fast paths (PIN / pro_id / SKU — no NL routing LLM) ===
    if not _brain_route_first:
        try:
            from services.ai_first_router import _try_account_list_fast_path

            acct_pre = _try_account_list_fast_path(original_msg, msg_en)
            if acct_pre:
                _, acct_route = acct_pre
                acct_route = dict(acct_route)
                alk = (acct_route.get("account_list_kind") or "").strip().lower()
                intent_acct = (acct_route.get("intent") or "").strip().lower()
                if alk == "wishlist_in_chat" or intent_acct == "wishlist":
                    wl_body = format_wishlist_reply(user_id, page=1, append_only=False)
                    if wl_body:
                        log_reasoning("Structural lane — wishlist API.")
                        return _fast_path_json_reply(
                            wl_body, ai_route_snapshot=acct_route
                        )
                if alk == "purchase_history_in_chat" or intent_acct == "order_history":
                    oh_body = format_purchase_history_reply(
                        user_id, page=1, append_only=False
                    )
                    if oh_body:
                        log_reasoning("Structural lane — order history API.")
                        return _fast_path_json_reply(
                            oh_body, ai_route_snapshot=acct_route
                        )
        except ImportError:
            pass

        try:
            from services.ai_first_router import _try_catalog_menu_fast_path
            from services.catalog_menu_replies import (
                build_categories_list_reply_html,
                build_today_deals_reply_html,
            )

            menu_pre = _try_catalog_menu_fast_path(original_msg, msg_en, ctx=ctx)
            if menu_pre:
                _, menu_route = menu_pre
                menu_route = dict(menu_route)
                menu_intent = (menu_route.get("intent") or "").strip().lower()
                menu_body = None
                if menu_intent == "deals":
                    menu_body = build_today_deals_reply_html(
                        original_msg, reply_lang=lang
                    )
                elif menu_intent == "categories":
                    menu_body = build_categories_list_reply_html(
                        ctx, original_msg, reply_lang=lang
                    )
                if menu_body:
                    log_reasoning(f"Structural lane — {menu_intent} API.")
                    return _fast_path_json_reply(
                        menu_body, ai_route_snapshot=menu_route
                    )
        except ImportError:
            pass

        text_kb_pre = _try_instant_text_kb_reply(
            original_msg, msg_en, lang, conversation_context=conv_for_llm
        )
        if text_kb_pre:
            log_reasoning("Structural lane — text KB (no embedding).")
            return _fast_path_json_reply(
                text_kb_pre,
                ai_route_snapshot={
                    "intent": "general",
                    "data_channel": "kb",
                    "route_handler": "kb_text_fast",
                },
            )

        try:
            from services.ai_first_router import _try_zero_llm_universal_brain_route
            from services.brain_direct_dispatch import try_brain_direct_dispatch

            zero_route = _try_zero_llm_universal_brain_route(
                original_msg, msg_en, ctx=ctx
            )
            if isinstance(zero_route, dict):
                if isinstance(ctx.get("data"), dict):
                    ctx["data"]["ai_route"] = zero_route
                zero_body = try_brain_direct_dispatch(
                    zero_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if zero_body:
                    log_reasoning("Structural lane — zero-LLM dispatch.")
                    return _fast_path_json_reply(
                        zero_body, ai_route_snapshot=zero_route
                    )
        except ImportError:
            pass

        try:
            from services.brain_direct_dispatch import try_category_browse_catalog_reply

            cat_pre_brain = try_category_browse_catalog_reply(
                original_msg,
                msg_en,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if cat_pre_brain:
                log_reasoning("Structural lane — catalog-map category browse.")
                return _fast_path_json_reply(
                    cat_pre_brain,
                    ai_route_snapshot={
                        "intent": "product",
                        "data_channel": "catalog",
                        "route_handler": "category_browse_structural_fast",
                    },
                )
        except ImportError:
            pass

        try:
            from services.location_delivery_resolver import turn_requests_delivery_serviceability
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply
            from utils.helpers import (
                _naive_six_digit_pin_from_text,
                _turn_blocks_pincode_serviceability_routing,
            )

            comb_del = f"{original_msg or ''} {msg_en or ''}".strip()
            del_fast = False
            del_blocked = False
            if comb_del:
                if _turn_blocks_pincode_serviceability_routing(comb_del):
                    del_blocked = True
                elif _naive_six_digit_pin_from_text(comb_del):
                    del_fast = True
                else:
                    try:
                        from utils.helpers import (
                            _text_is_delivery_serviceability_hypothetical,
                            message_has_live_pincode_check_intent,
                        )

                        if (
                            _text_is_delivery_serviceability_hypothetical(comb_del)
                            or message_has_live_pincode_check_intent(
                                comb_del, conv_for_llm, msg_en
                            )
                        ):
                            del_fast = True
                    except ImportError:
                        pass
                    if not del_fast:
                        try:
                            from services.location_delivery_resolver import (
                                turn_continues_pincode_area_check,
                            )
                            from utils.helpers import _conversation_in_pincode_delivery_flow

                            if (
                                conv_for_llm
                                and _conversation_in_pincode_delivery_flow(conv_for_llm)
                                and turn_continues_pincode_area_check(
                                    comb_del, conv_for_llm, ai_route=None
                                )
                            ):
                                del_fast = True
                        except ImportError:
                            pass
                if (
                    not del_fast
                    and not del_blocked
                    and turn_requests_delivery_serviceability(
                        original_msg,
                        msg_en,
                        conv_for_llm,
                        allow_llm=True,
                    )
                ):
                    del_fast = True
            if del_fast:
                pin_pre_brain = try_pincode_delivery_fast_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm,
                    lang,
                    ctx,
                    reset_context_fn=reset_context,
                    skip_turn_check=True,
                    allow_llm=True,
                )
                if pin_pre_brain:
                    log_reasoning(
                        "Structural lane — delivery micro-classifier + live API."
                    )
                    return _fast_path_json_reply(
                        pin_pre_brain,
                        ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
                        or {
                            "intent": "pincode_check",
                            "data_channel": "live_api",
                            "route_handler": "pincode_delivery_fast",
                        },
                    )
        except ImportError:
            pass

    # === KB fallback when universal brain did not run or returned no body ===
    _brain_already_classified = False
    if _brain_route_first:
        try:
            from services.chat_flow_telemetry import get_cached_brain_route

            _brain_already_classified = bool(get_cached_brain_route())
        except ImportError:
            pass

    if (
        not _brain_already_classified
        and not _turn_bypasses_embedding_vector(
            original_msg, msg_en, ctx, conv_for_llm
        )
    ):
        try:
            from services.knowledge_query_pipeline import try_ai_first_kb_early_reply

            kb_ai_body = try_ai_first_kb_early_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=lang,
                preflight=True,
            )
            if kb_ai_body and kb_ai_body.strip():
                log_reasoning("KB AI preflight — classifier + vector KB (admin files).")
                return _fast_path_json_reply(
                    kb_ai_body,
                    ai_route_snapshot={
                        "intent": "general",
                        "data_channel": "kb",
                        "route_handler": "kb_ai_first_preflight",
                    },
                )
        except ImportError:
            pass
        try:
            from services.ai_first_router import _try_policy_kb_preflight
            from services.brain_direct_dispatch import try_brain_direct_dispatch

            kb_pf = _try_policy_kb_preflight(original_msg, msg_en, lang)
            if kb_pf:
                _kb_decision, kb_route_data = kb_pf
                if isinstance(ctx.get("data"), dict):
                    ctx["data"]["ai_route"] = kb_route_data
                kb_pf_body = try_brain_direct_dispatch(
                    kb_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if kb_pf_body:
                    log_reasoning(
                        "KB embedding preflight (admin files — vector rank + RAG)."
                    )
                    return _fast_path_json_reply(
                        kb_pf_body,
                        ai_route_snapshot=kb_route_data,
                    )
        except ImportError:
            pass

    # === Scope LLM — only when brain LLM unavailable (no duplicate preflight) ===
    if _brain_route_first:
        try:
            from services.chat_flow_telemetry import get_cached_brain_route

            _brain_cached = get_cached_brain_route()
        except ImportError:
            _brain_cached = None
        if isinstance(_brain_cached, dict) and _brain_cached.get("llm_unavailable"):
            scope_fast = _try_scope_ood_fast_lane_reply(
                original_msg, msg_en, conv_for_llm, lang
            )
            if scope_fast:
                scope_body, scope_route = scope_fast
                log_reasoning("Scope AI fallback — brain LLM unavailable.")
                return _fast_path_json_reply(
                    scope_body,
                    ai_route_snapshot=scope_route,
                )

    # === PIN / DELIVERY LIVE API (structural only — bare PIN / explicit delivery id) ===
    if not _brain_route_first or _turn_has_structural_fast_lane_token(
        original_msg, msg_en, conv_for_llm, ctx
    ):
        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            pin_pre = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if pin_pre:
                log_reasoning("Pre-brain — pincode/delivery live API (zero routing LLM).")
                return _fast_path_json_reply(
                    pin_pre,
                    ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
                    or {
                        "intent": "pincode_check",
                        "data_channel": "live_api",
                        "route_handler": "pincode_delivery_fast",
                    },
                )
        except ImportError:
            pass
        try:
            pin_bare_body, pin_bare_route = _try_bare_numeric_live_fast_reply(
                original_msg,
                msg_en,
                user_id,
                lang,
                ctx,
                chat_id=current_chat_id,
                conv_for_llm=conv_for_llm,
            )
            if pin_bare_body:
                log_reasoning(f"Pre-brain — bare PIN/order fast ({pin_bare_route}).")
                return _fast_path_json_reply(
                    pin_bare_body,
                    ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
                    or {"intent": "pincode_check", "data_channel": "live_api"},
                )
        except Exception:
            pass

    # === PRODUCT OPENSEARCH (structural SKU/pro_id — not natural-language browse) ===
    if not _brain_route_first:
        _t_prod_pre_brain = time.perf_counter()
        prod_pre_body, prod_pre_route = _try_instant_product_catalog_reply(
            original_msg, msg_en, user_id, lang, ctx, conv_for_llm=conv_for_llm
        )
        if prod_pre_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "product_opensearch_fast",
                    (time.perf_counter() - _t_prod_pre_brain) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning(
                "Product fast path — structural OpenSearch before ai_brain_route."
            )
            return _fast_path_json_reply(
                prod_pre_body,
                ai_route_snapshot=prod_pre_route,
            )

        # === INSTANT ZERO-LLM (structural catalog/order/deals only) ===
        _t_instant = time.perf_counter()
        instant_body, instant_route = _try_instant_zero_llm_reply(
            original_msg, msg_en, user_id, lang, ctx, conv_for_llm=conv_for_llm
        )
        if instant_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "instant_zero_llm",
                    (time.perf_counter() - _t_instant) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning("Instant zero-LLM lane — reply without ai_brain_route stack.")
            return _fast_path_json_reply(
                instant_body,
                ai_route_snapshot=instant_route,
            )

        # === AI BRAIN (structural turns that still need LLM routing) ===
        _t_brain_universal = time.perf_counter()
        log_reasoning("Chat — one ai_brain_route (intent + dispatch).")
        brain_uni_body, brain_uni_route = _try_early_ai_brain_reply(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm=conv_for_llm,
            chat_id=current_chat_id,
        )
        if brain_uni_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "ai_brain_universal",
                    (time.perf_counter() - _t_brain_universal) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning(
                "Universal brain — AI intent + dispatch (no keyword routing)."
            )
            return _fast_path_json_reply(
                brain_uni_body,
                ai_route_snapshot=brain_uni_route,
            )

        if isinstance(brain_uni_route, dict) and not brain_uni_route.get(
            "llm_unavailable"
        ):
            try:
                from services.brain_direct_dispatch import try_finish_brain_classified_turn

                finish_body = try_finish_brain_classified_turn(
                    brain_uni_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if finish_body:
                    log_reasoning(
                        "Brain classified turn — finish dispatch (no legacy LLM stack)."
                    )
                    return _fast_path_json_reply(
                        finish_body,
                        ai_route_snapshot=brain_uni_route,
                    )
            except ImportError:
                pass

    # Legacy fallthrough only when AI-first lane did not return above.
    if brain_uni_route is None and isinstance(ctx, dict):
        brain_uni_route = (ctx.get("data") or {}).get("ai_route")

    _brain_preflight_done = bool(
        isinstance(brain_uni_route, dict) and not brain_uni_route.get("llm_unavailable")
    )
    if not _brain_preflight_done:
        try:
            from services.chat_flow_telemetry import (
                get_cached_brain_route,
                get_early_brain_dispatch,
            )

            if get_cached_brain_route() or get_early_brain_dispatch():
                _brain_preflight_done = True
        except ImportError:
            pass

    early_top_route = (
        brain_uni_route if isinstance(brain_uni_route, dict) else None
    )

    if _brain_preflight_done:
        conv_finish = conv_for_llm
        if not conv_finish:
            try:
                comb_finish = f"{original_msg or ''} {msg_en or ''}".strip()
                if len(comb_finish) <= 72 or ctx.get("awaiting") or ctx.get("last"):
                    conv_finish = _conversation_context_for_routing(
                        current_chat_id, original_msg, msg_en, ctx
                    )
            except Exception:
                pass
        try:
            from services.brain_direct_dispatch import try_finish_brain_classified_turn

            finish_retry = try_finish_brain_classified_turn(
                brain_uni_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_finish,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if finish_retry:
                try:
                    from services.chat_flow_telemetry import record_phase

                    record_phase(
                        "brain_finish_retry",
                        (time.perf_counter() - _t_brain_universal) * 1000.0,
                    )
                except ImportError:
                    pass
                log_reasoning(
                    "Brain classified — finish dispatch retry (skip legacy stack)."
                )
                return _fast_path_json_reply(
                    finish_retry,
                    ai_route_snapshot=brain_uni_route,
                )
        except ImportError:
            pass
        # Last resort before busy: structural product / catalog menu from msg_en (no extra LLM).
        _busy_channel = (
            (brain_uni_route.get("data_channel") or "").strip().lower()
            if isinstance(brain_uni_route, dict)
            else ""
        )
        _busy_intent = (
            (brain_uni_route.get("intent") or "").strip().lower()
            if isinstance(brain_uni_route, dict)
            else ""
        )
        if _busy_channel == "kb":
            try:
                from services.brain_direct_dispatch import try_finish_brain_classified_turn

                kb_busy = try_finish_brain_classified_turn(
                    brain_uni_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_finish,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if kb_busy:
                    return _fast_path_json_reply(
                        kb_busy,
                        ai_route_snapshot=brain_uni_route,
                    )
            except ImportError:
                pass
        elif _busy_channel not in ("live_api",) and _busy_intent not in (
            "order_history",
            "wishlist",
            "refund",
            "payment",
        ):
            try:
                from services.brain_direct_dispatch import (
                    _try_brain_catalog_menu_direct_reply,
                    try_structural_product_catalog_reply,
                )
                from services.conversation_followup import is_deals_request_message
                from utils.helpers import message_asks_welfog_categories_list

                comb_busy = f"{original_msg or ''} {msg_en or ''}".strip()
                if is_deals_request_message(original_msg, msg_en) or message_asks_welfog_categories_list(
                    comb_busy
                ):
                    menu_body = _try_brain_catalog_menu_direct_reply(
                        brain_uni_route,
                        original_msg=original_msg,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                    )
                    if menu_body:
                        log_reasoning(
                            "Brain busy recovery — deals/categories API from msg_en (zero LLM)."
                        )
                        return _fast_path_json_reply(
                            menu_body,
                            ai_route_snapshot=brain_uni_route,
                        )
                structural_busy = try_structural_product_catalog_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm=conv_finish,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reset_context_fn=reset_context,
                )
                if structural_busy:
                    log_reasoning(
                        "Brain busy recovery — structural product OpenSearch from msg_en."
                    )
                    return _fast_path_json_reply(
                        structural_busy,
                        ai_route_snapshot=brain_uni_route,
                    )
                try:
                    from services.brain_direct_dispatch import (
                        try_llm_down_product_catalog_recovery,
                    )

                    llm_down_busy = try_llm_down_product_catalog_recovery(
                        original_msg,
                        msg_en,
                        conv_for_llm=conv_finish,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                    )
                    if llm_down_busy:
                        log_reasoning(
                            "Brain busy recovery — compact product classify + OpenSearch."
                        )
                        return _fast_path_json_reply(
                            llm_down_busy,
                            ai_route_snapshot=brain_uni_route,
                        )
                except ImportError:
                    pass
                try:
                    from services.pincode_delivery_fast_path import (
                        try_pincode_delivery_fast_reply,
                    )
                    from utils.helpers import turn_is_obvious_product_shopping_turn

                    if not turn_is_obvious_product_shopping_turn(
                        original_msg, msg_en, conv_finish
                    ):
                        pin_busy = try_pincode_delivery_fast_reply(
                            original_msg,
                            msg_en,
                            conv_finish,
                            lang,
                            ctx,
                            reset_context_fn=reset_context,
                        )
                        if pin_busy:
                            log_reasoning(
                                "Brain busy recovery — pincode live API from msg_en."
                            )
                            return _fast_path_json_reply(
                                pin_busy,
                                ai_route_snapshot=brain_uni_route,
                            )
                except ImportError:
                    pass
            except ImportError:
                pass
        # Order track/invoice — brain classified but product/KB busy recovery missed.
        try:
            from services.brain_direct_dispatch import (
                _try_brain_order_live_direct_reply,
                _try_brain_order_live_fallback_dispatch,
            )

            order_busy = _try_brain_order_live_direct_reply(
                brain_uni_route,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_finish,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if not order_busy:
                order_busy = _try_brain_order_live_fallback_dispatch(
                    brain_uni_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_finish,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context,
                )
            if order_busy:
                log_reasoning(
                    "Brain busy recovery — order track/invoice ask-ID or live API."
                )
                return _fast_path_json_reply(
                    order_busy,
                    ai_route_snapshot=brain_uni_route,
                )
        except ImportError:
            pass
        log_reasoning(
            "Brain classified — skip legacy LLM stack (avoid duplicate routing / timeout)."
        )
        try:
            from services.sysmsg import sysmsg

            busy_body = sysmsg("server_busy")
        except ImportError:
            busy_body = (
                "Thoda load hai abhi — ek baar phir try karein."
            )
        return _fast_path_json_reply(
            busy_body,
            ai_route_snapshot=brain_uni_route,
        )

    # === ORDER HISTORY LIST (structural fallback — only when brain LLM unavailable) ===
    if isinstance(brain_uni_route, dict) and brain_uni_route.get("llm_unavailable"):
        try:
            from services.brain_direct_dispatch import try_llm_down_product_catalog_recovery

            llm_down_body = try_llm_down_product_catalog_recovery(
                original_msg,
                msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if llm_down_body:
                log_reasoning(
                    "Brain LLM down — compact product classify + OpenSearch (Hinglish recovery)."
                )
                return _fast_path_json_reply(
                    llm_down_body,
                    ai_route_snapshot={
                        "intent": "product",
                        "data_channel": "catalog",
                        "_product_catalog_locked": True,
                    },
                )
        except ImportError:
            pass
        rec_body, rec_route = _try_llm_down_transactional_recovery(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm=conv_for_llm,
        )
        if rec_body:
            log_reasoning(
                "Brain LLM down — structural order/pincode recovery (skip order history)."
            )
            return _fast_path_json_reply(rec_body, ai_route_snapshot=rec_route)
        _t_ph_list = time.perf_counter()
        ph_body, ph_route = _try_order_history_list_fast_reply(
            original_msg, msg_en, user_id, ctx=ctx, conversation_context=conv_for_llm
        )
        if ph_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "order_history_list",
                    (time.perf_counter() - _t_ph_list) * 1000.0,
                )
            except ImportError:
                pass
            return _fast_path_json_reply(ph_body, ai_route_snapshot=ph_route)

        try:
            from services.chat_resilience import build_busy_reply_html

            busy_body = build_busy_reply_html(original_msg, lang)
        except ImportError:
            busy_body = "Thoda load hai abhi — ek baar phir try karein."
        log_reasoning(
            "Brain LLM unavailable — skip legacy stack (no duplicate LLM/embed)."
        )
        return _fast_path_json_reply(
            busy_body,
            ai_route_snapshot=brain_uni_route if isinstance(brain_uni_route, dict) else None,
        )

    # === Brain already classified — never run legacy mega-stack (90s embed/duplicate LLM) ===
    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            is_routing_complete,
        )
        from services.brain_direct_dispatch import (
            try_brain_direct_dispatch,
            try_finish_brain_classified_turn,
        )

        if is_routing_complete():
            _cr = get_cached_brain_route() or (
                brain_uni_route if isinstance(brain_uni_route, dict) else None
            )
            if isinstance(_cr, dict) and not _cr.get("llm_unavailable"):
                _conv_exit = conv_for_llm or _conversation_context_for_routing(
                    current_chat_id, original_msg, msg_en, ctx
                )
                finish_exit = try_finish_brain_classified_turn(
                    _cr,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=_conv_exit,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if finish_exit:
                    return _fast_path_json_reply(
                        finish_exit, ai_route_snapshot=_cr
                    )
                direct_exit = try_brain_direct_dispatch(
                    _cr,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=_conv_exit,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                if direct_exit:
                    return _fast_path_json_reply(
                        direct_exit, ai_route_snapshot=_cr
                    )
            log_reasoning(
                "Routing complete — skip legacy stack (prevent timeout)."
            )
            try:
                from services.chat_resilience import build_busy_reply_html

                busy_exit = build_busy_reply_html(original_msg, lang)
            except ImportError:
                busy_exit = "Thoda load hai — ek baar phir try karein."
            return _fast_path_json_reply(
                busy_exit,
                ai_route_snapshot=_cr if isinstance(_cr, dict) else None,
            )
    except ImportError:
        pass

    # === ORDER TRACK/INVOICE ask-id (fallback — only when brain did not run) ===
    if not brain_uni_body:
        _t_order_nl = time.perf_counter()
        order_nl_body, order_nl_route = _try_zero_llm_order_nl_fast_reply(
            original_msg, msg_en, user_id, lang, ctx
        )
        if order_nl_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "order_nl_zero_llm",
                    (time.perf_counter() - _t_order_nl) * 1000.0,
                )
            except ImportError:
                pass
            return _fast_path_json_reply(
                order_nl_body,
                ai_route_snapshot=order_nl_route,
            )

    _is_transactional_turn = _transactional_turn_skip_kb_scope_preflight(
        original_msg, msg_en, ""
    )

    # === CATALOG / API FAST LANE (brain cache — universal route already ran above) ===
    if _is_transactional_turn and not _brain_preflight_done:
        _t_brain_txn = time.perf_counter()
        txn_body, txn_route = _try_early_ai_brain_reply(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm="",
            chat_id=current_chat_id,
        )
        if txn_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "ai_brain_transactional",
                    (time.perf_counter() - _t_brain_txn) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning(
                "Transactional fast lane — brain + live API/catalog (no KB stack)."
            )
            return _fast_path_json_reply(
                txn_body,
                ai_route_snapshot=txn_route,
            )

    conv_snap_early = ""
    if not _is_transactional_turn:
        conv_snap_early = _conversation_snapshot_for_chat(current_chat_id, limit=4)
    _kb_ood_preflight_done = _brain_preflight_done

    # === AI BRAIN EARLY (only when universal brain did not run) ===
    if not _brain_preflight_done:
        _t_brain_top = time.perf_counter()
        early_top_body, early_top_route = _try_early_ai_brain_reply(
            original_msg,
            msg_en,
            user_id,
            lang,
            ctx,
            conv_for_llm=conv_snap_early,
            chat_id=current_chat_id,
        )
        if early_top_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "ai_brain_early",
                    (time.perf_counter() - _t_brain_top) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning(
                "AI brain early (top) — semantic route reply (one ai_brain_route)."
            )
            return _fast_path_json_reply(
                early_top_body,
                ai_route_snapshot=early_top_route,
            )
    else:
        early_top_route = brain_uni_route if isinstance(brain_uni_route, dict) else None

    _skip_kb_scope_preflight = bool(_brain_preflight_done)
    if isinstance(early_top_route, dict):
        ch = (early_top_route.get("data_channel") or "").strip().lower()
        intent = (early_top_route.get("intent") or "").strip().lower()
        if ch in ("live_api", "catalog"):
            _skip_kb_scope_preflight = True
        elif intent in (
            "wishlist",
            "order_history",
            "order",
            "pincode_check",
            "deals",
            "categories",
            "category_feed",
            "product",
        ) and ch != "kb":
            _skip_kb_scope_preflight = True

    if not _skip_kb_scope_preflight:
        _brain_unavailable = (
            isinstance(early_top_route, dict)
            and early_top_route.get("llm_unavailable")
        )
        if _brain_unavailable and _is_transactional_turn and not _kb_ood_preflight_done:
            rec_body, rec_route = _try_llm_down_transactional_recovery(
                original_msg,
                msg_en,
                user_id,
                lang,
                ctx,
                conv_for_llm=conv_snap_early or conv_for_llm,
            )
            if rec_body:
                log_reasoning(
                    "Transactional LLM-down — structural live API (skip KB embed stack)."
                )
                return _fast_path_json_reply(rec_body, ai_route_snapshot=rec_route)
            _t_scope_ai = time.perf_counter()
            try:
                from services.chitchat_resolver import try_scope_ai_early_reply

                scope_hit = try_scope_ai_early_reply(
                    original_msg,
                    msg_en,
                    conv_snap_early,
                    reply_lang=lang,
                )
                if scope_hit:
                    scope_html, scope_route = scope_hit
                    try:
                        from services.chat_flow_telemetry import record_phase

                        record_phase(
                            "scope_ai_early",
                            (time.perf_counter() - _t_scope_ai) * 1000.0,
                        )
                    except ImportError:
                        pass
                    log_reasoning(
                        "Scope AI fallback — brain LLM unavailable."
                    )
                    return _fast_path_json_reply(
                        scope_html, ai_route_snapshot=scope_route
                    )
            except ImportError:
                pass

        _t_kb_sem = time.perf_counter()
        if not _turn_bypasses_embedding_vector(
            original_msg, msg_en, ctx, conv_snap_early
        ):
            text_kb_fb = _try_instant_text_kb_reply(
                original_msg, msg_en, lang, conversation_context=conv_snap_early
            )
            if text_kb_fb:
                log_reasoning(
                    "KB text fallback — no embedding (brain unavailable path)."
                )
                return _fast_path_json_reply(
                    text_kb_fb,
                    ai_route_snapshot={
                        "intent": "general",
                        "data_channel": "kb",
                        "route_handler": "kb_text_fallback",
                    },
                )
        try:
            if _turn_bypasses_embedding_vector(
                original_msg, msg_en, ctx, conv_snap_early
            ):
                log_reasoning(
                    "Skip vector KB fallback — transactional turn (no encode lock)."
                )
            else:
                from services.conversation_zero_llm_fallback import (
                    try_vector_kb_only_reply,
                )

                kb_body = try_vector_kb_only_reply(
                    original_msg,
                    msg_en,
                    conv_snap_early,
                    reply_lang=lang,
                    ai_route=early_top_route if isinstance(early_top_route, dict) else None,
                )
                if kb_body:
                    try:
                        from services.chat_flow_telemetry import record_phase

                        record_phase(
                            "kb_vector_fallback",
                            (time.perf_counter() - _t_kb_sem) * 1000.0,
                        )
                    except ImportError:
                        pass
                    log_reasoning(
                        "Vector KB fallback — brain meaning + embeddings (no extra classifier)."
                    )
                    return _fast_path_json_reply(
                        kb_body,
                        ai_route_snapshot={
                            "intent": "general",
                            "data_channel": "kb",
                            "route_handler": "kb_brain_vector_fallback",
                        },
                    )
        except ImportError:
            pass
    else:
        log_reasoning(
            "Skip scope/KB preflight — brain locked live API/catalog route."
        )

    # === PINCODE / DELIVERY (city/PIN — after brain, before deep fallbacks) ===
    _t_pin_pre = time.perf_counter()
    _skip_pin_pre_brain = False
    try:
        from utils.helpers import turn_is_obvious_product_shopping_turn

        _skip_pin_pre_brain = turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conv_snap_early
        )
    except ImportError:
        pass
    if not _skip_pin_pre_brain:
        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            pin_pre_brain = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv_snap_early,
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if pin_pre_brain:
                try:
                    from services.chat_flow_telemetry import record_phase

                    record_phase(
                        "pincode_pre_brain",
                        (time.perf_counter() - _t_pin_pre) * 1000.0,
                    )
                except ImportError:
                    pass
                log_reasoning("Pincode delivery pre-brain — live API / area follow-up.")
                return _fast_path_json_reply(
                    pin_pre_brain,
                    ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
                    or {
                        "intent": "pincode_check",
                        "data_channel": "live_api",
                        "route_handler": "pincode_delivery_fast",
                    },
                )
            try:
                from services.pincode_delivery_flow import run_delivery_location_check
                from services.pincode_delivery_fast_path import (
                    turn_is_pincode_delivery_fast_path,
                )
                from utils.helpers import _text_is_pincode_serviceability_question

                pin_route_stub = {
                    "intent": "pincode_check",
                    "data_channel": "live_api",
                    "_pincode_delivery_locked": True,
                }
                if turn_is_pincode_delivery_fast_path(
                    original_msg, msg_en, conv_snap_early, ctx
                ) or _text_is_pincode_serviceability_question(
                    f"{original_msg} {msg_en}".strip(), conv_snap_early
                ):
                    loc_res = run_delivery_location_check(
                        original_msg,
                        msg_en,
                        conv_snap_early,
                        reply_lang=lang,
                        ai_route=pin_route_stub,
                        allow_llm=True,
                    )
                    if loc_res.handled and loc_res.reply_html:
                        try:
                            from services.chat_flow_telemetry import record_phase

                            record_phase(
                                "pincode_pre_brain_geocode",
                                (time.perf_counter() - _t_pin_pre) * 1000.0,
                            )
                        except ImportError:
                            pass
                        if isinstance(ctx, dict):
                            try:
                                from utils.helpers import mark_pincode_delivery_completed

                                mark_pincode_delivery_completed(
                                    ctx, ai_route=pin_route_stub
                                )
                            except ImportError:
                                ctx["last"] = None
                                ctx["awaiting"] = None
                                ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
                                ctx.setdefault("data", {})["ai_route"] = pin_route_stub
                        log_reasoning(
                            "Pincode pre-brain geocode — city/PIN live API (zero brain LLM)."
                        )
                        return _fast_path_json_reply(
                            loc_res.reply_html,
                            ai_route_snapshot=pin_route_stub,
                        )
            except ImportError:
                pass
        except ImportError:
            pass

    # === ZERO-LLM EXPLICIT SKU (first — before ctx/brain/product AI) ===
    _t_sku_fast = time.perf_counter()
    try:
        from services.brain_direct_dispatch import try_explicit_sku_catalog_reply

        sku_body = try_explicit_sku_catalog_reply(
            original_msg,
            msg_en,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if sku_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "sku_structural",
                    (time.perf_counter() - _t_sku_fast) * 1000.0,
                )
            except ImportError:
                pass
            log_reasoning("Explicit SKU structural fast path — zero LLM, OpenSearch only.")
            return _fast_path_json_reply(
                sku_body,
                ai_route_snapshot={
                    "intent": "product",
                    "data_channel": "catalog",
                    "route_handler": "sku_structural_fast",
                },
            )
    except ImportError:
        pass

    # === ZERO-LLM CATEGORY BROWSE (Beauty/Electronics/any dept — before brain) ===
    _t_cat_fast = time.perf_counter()
    try:
        from services.brain_direct_dispatch import try_category_browse_catalog_reply

        cat_body = try_category_browse_catalog_reply(
            original_msg,
            msg_en,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
        )
        if cat_body:
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "category_browse_structural",
                    (time.perf_counter() - _t_cat_fast) * 1000.0,
                )
            except ImportError:
                pass
            return _fast_path_json_reply(
                cat_body,
                ai_route_snapshot={
                    "intent": "product",
                    "data_channel": "catalog",
                    "route_handler": "category_browse_structural_fast",
                },
            )
    except ImportError:
        pass

    # === ZERO-LLM STRUCTURAL API/CATALOG (order id / SKU — before AI brain) ===
    if _can_resolve_without_conversation(original_msg, msg_en, lang, ctx):
        _t_early_live = time.perf_counter()
        early_body, early_route = _try_early_structural_live_reply(
            original_msg, msg_en, user_id, lang, ctx, chat_id=current_chat_id
        )
        if early_body:
            try:
                from services.chat_flow_telemetry import record_phase, record_route

                record_phase(
                    "structural_live_early",
                    (time.perf_counter() - _t_early_live) * 1000.0,
                )
                ar = (ctx.get("data") or {}).get("ai_route") or {}
                record_route(
                    intent=ar.get("intent") or "order",
                    source=ar.get("route_handler") or early_route,
                )
            except ImportError:
                pass
            log_reasoning(
                f"Structural live fast path ({early_route}) — before guards (zero LLM)."
            )
            return _fast_path_json_reply(
                early_body,
                ai_route_snapshot=(ctx.get("data") or {}).get("ai_route")
                or {
                    "intent": "order",
                    "data_channel": "live_api",
                    "route_handler": early_route,
                },
            )

    skip_conv_load = _can_resolve_without_conversation(
        original_msg, msg_en, lang, ctx
    )
    recent_msgs: list = []
    try:
        import threading

        _cid_store, _uid_store, _um_store = current_chat_id, str(user_id), user_msg

        threading.Thread(
            target=lambda: db_store_message(_cid_store, "user", _um_store, _uid_store),
            daemon=True,
        ).start()
    except Exception as db_exc:
        print(f"[chat] user msg DB skip: {db_exc}", flush=True)

    if skip_conv_load:
        conv_for_llm = ""
        log_reasoning("Skip MySQL conversation load — structural/brain path can proceed.")
    else:
        try:
            _t_mysql = time.perf_counter()
            recent_msgs = db_get_recent_messages(current_chat_id, 10)
            conv_for_llm = _format_conversation_for_llm(recent_msgs, max_turns=8)
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "mysql_conversation_load",
                    (time.perf_counter() - _t_mysql) * 1000.0,
                )
            except ImportError:
                pass
        except Exception as db_exc:
            print(f"[chat] conv load skip: {db_exc}", flush=True)
            conv_for_llm = ""

    from utils.helpers import message_is_conversation_reset_command

    if message_is_conversation_reset_command(original_msg):
        log_reasoning("User reset command — clear order-id state.")
        reset_context(ctx)
        body = sysmsg("cancelled")
        try:
            db_store_message(current_chat_id, "user", user_msg, str(user_id))
            db_store_message(current_chat_id, "bot", body, str(user_id))
        except Exception as db_exc:
            chat_log(f"reset DB skip: {db_exc}")
        return jsonify({"chat_id": current_chat_id, "type": "text", "data": body})

    # KB vectors refresh lazily when a KB handler runs — not on every /chat (saves seconds).

    if defer_chat_session_insert:
        import threading

        _uid_reg, _msg_reg, _tok_reg = str(user_id), user_msg, current_chat_id

        def _register_session_bg():
            try:
                _register_new_chat_session_mysql(_uid_reg, _msg_reg, _tok_reg)
            except Exception as exc:
                chat_log(f"session register skip: {exc}")

        threading.Thread(target=_register_session_bg, daemon=True).start()

    # 🔥 FIX 4: Smart helper jo har reply ko (zarurat par) translate, save aur return karega.
    def _is_live_api_structured_reply(text_data: str) -> bool:
        """Order/product API cards — never replace with support escalation templates."""
        if not isinstance(text_data, str) or not text_data.strip():
            return False
        markers = (
            "data-wf-live-api=",
            "wf-od-root",
            "wf-ph-root",
            "wf-wl-root",
            "wf-oid-root",
            "wf-product-root",
            "wf-invoice-card",
            "wf-invoice-btn",
            "wf-product-rail",
            "Live order tracking",
            "Your wishlist",
            "Aapke order ki details",
            "Your invoice",
            "Aapka invoice",
        )
        return any(m in text_data for m in markers)

    def _enforce_official_support_contacts(
        text_data: str, lang_code: str, ai_route_snapshot: dict | None = None
    ) -> str:
        """
        Guardrail: never send hallucinated phone/email.
        If support/escalation query is active and response contains non-KB contacts,
        replace with deterministic KB contact response.
        """
        if not isinstance(text_data, str) or not text_data.strip():
            return text_data
        if _is_live_api_structured_reply(text_data):
            return text_data
        try:
            from services.knowledge_grounding_validator import apply_final_kb_fact_contract

            text_data = apply_final_kb_fact_contract(
                text_data,
                original_msg=original_msg,
                msg_en=msg_en,
            )
        except Exception:
            pass
        try:
            from services.catalog_turn_semantics import should_skip_catalog_for_conversational_turn

            if should_skip_catalog_for_conversational_turn(
                original_msg, msg_en, conv_for_llm
            ):
                return text_data
        except ImportError:
            pass
        ar = ai_route_snapshot if isinstance(ai_route_snapshot, dict) else (ctx.get("data") or {}).get("ai_route") or {}
        scope = (ar.get("conversation_scope") or "").strip().lower()
        intent = (ar.get("intent") or "").strip().lower()
        meta = (ar.get("meta_kind") or "").strip().lower()
        channel = (ar.get("data_channel") or "").strip().lower()
        if scope in ("general_chitchat", "out_of_domain", "harm_sensitive"):
            return text_data
        if channel in ("catalog", "live_api"):
            return text_data
        if intent in ("general", "out_of_domain", "product", "order_history", "wishlist", "refund") or meta in (
            "conversational",
            "assistant_intro",
        ):
            return text_data
        try:
            from services.order_details_flow import message_wants_order_details_or_invoice

            if message_wants_order_details_or_invoice(
                original_msg, msg_en, conv_for_llm,
                ai_route=ar,
            ):
                return text_data
        except ImportError:
            pass
        try:
            from services.kb_service import (
                extract_contacts_from_plain_text,
                format_customer_care_reply_from_kb,
                format_support_escalation_reply_from_kb,
                get_support_contact_kb_keys,
                read_concatenated_kb_file_contents,
            )
            from utils.helpers import (
                _text_asks_customer_care_contact,
                message_needs_human_support_escalation,
                message_is_knowledge_information_request,
                message_asks_welfog_social_media,
            )

            comb = f"{(original_msg or '').lower()} {(msg_en or '').lower()}".strip()
            # Do not rewrite policy/privacy/social replies into customer-care templates.
            if (
                message_is_knowledge_information_request(comb)
                or message_asks_welfog_social_media(comb)
                or any(x in comb for x in ("fraud", "phishing", "scam", "grievance", "children privacy", "privacy"))
            ):
                return text_data
            keys = get_support_contact_kb_keys()
            blob = read_concatenated_kb_file_contents(keys) if keys else ""
            if not blob.strip():
                return text_data
            official_phones, routine_emails, grievance_emails = extract_contacts_from_plain_text(
                blob, include_grievance=True
            )
            official_phone_set = {p.strip() for p in official_phones}
            official_email_set = {e.lower().strip() for e in (routine_emails + grievance_emails)}

            reply_plain = re.sub(r"<[^>]+>", " ", text_data)
            contact_artifacts_in_reply = bool(
                re.search(r"\b(?:\+?91[\s-]*)?[6-9]\d{9}\b", reply_plain)
                or re.search(r"\b(?:1800|1860|1888)\s*[- ]?\s*\d{3,4}\s*[- ]?\s*\d{3,4}\b", reply_plain)
                or re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", reply_plain)
            )
            support_cue_in_reply = contact_artifacts_in_reply and any(
                re.search(rf"\b{re.escape(x)}\b", reply_plain.lower())
                for x in (
                    "customer care",
                    "customer support",
                    "welfog support",
                    "helpline",
                    "help line",
                )
            )
            support_turn = (
                _text_asks_customer_care_contact(comb)
                or message_needs_human_support_escalation(comb)
                or support_cue_in_reply
            )
            if not support_turn:
                return text_data

            # 10-digit mobile + toll-free patterns like 1800-123-4567
            phones_in_reply = set(re.findall(r"\b(?:\+?91[\s-]*)?([6-9]\d{9})\b", reply_plain))
            tollfree_in_reply = set(
                re.findall(r"\b(?:1800|1860|1888)\s*[- ]?\s*\d{3,4}\s*[- ]?\s*\d{3,4}\b", reply_plain)
            )
            emails_in_reply = {
                e.lower().strip()
                for e in re.findall(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", reply_plain)
            }
            bad_phone = bool(phones_in_reply and not phones_in_reply.issubset(official_phone_set))
            bad_tollfree = bool(tollfree_in_reply)  # not allowed unless explicitly present in KB contacts
            bad_email = bool(emails_in_reply and not emails_in_reply.issubset(official_email_set))
            if bad_phone or bad_tollfree or bad_email:
                log_reasoning("Support contact guardrail: replaced non-KB phone/email with official KB contacts.")
                if _text_asks_customer_care_contact(comb):
                    fixed = format_customer_care_reply_from_kb(original_msg, msg_en)
                else:
                    fixed = format_support_escalation_reply_from_kb(
                        original_msg, msg_en, reply_lang=lang_code
                    )
                if fixed:
                    return fixed
        except Exception:
            return text_data
        return text_data

    def send_reply(text_data, lang_code, ai_route_snapshot: dict | None = None):
        """
        Reply in the same language/style as the customer (en / hinglish / Indian scripts).
        """
        if isinstance(text_data, str):
            text_data = _enforce_official_support_contacts(
                text_data, lang_code, ai_route_snapshot=ai_route_snapshot
            )
        rl = resolve_customer_reply_lang(original_msg, lang_code or "")
        skip_finalize = isinstance(text_data, str) and is_live_api_structured_html(text_data)
        if isinstance(text_data, str):
            final_output = text_data if skip_finalize else finalize_customer_reply(text_data, original_msg, rl)
        else:
            final_output = text_data
        try:
            import threading

            _cid, _uid, _out = current_chat_id, str(user_id), final_output

            def _persist_bot_msg():
                try:
                    db_store_message(_cid, "bot", _out, _uid)
                except Exception as db_exc:
                    chat_log(f"bot msg DB skip: {db_exc}")

            threading.Thread(target=_persist_bot_msg, daemon=True).start()
        except Exception as db_exc:
            try:
                db_store_message(current_chat_id, "bot", final_output, str(user_id))
            except Exception as db_exc2:
                chat_log(f"bot msg DB skip: {db_exc2}")
        preview = (
            final_output[:120].replace("\n", " ")
            if isinstance(final_output, str)
            else f"<{type(final_output).__name__}>"
        )
        chat_log(f"reply sent ({len(final_output) if isinstance(final_output, str) else 'obj'} chars): {preview!r}")
        try:
            from services.chat_flow_telemetry import log_routing_decision, log_turn_complete, response_time_sec

            ar = ai_route_snapshot or (ctx.get("data") or {}).get("ai_route") or {}
            adr = (ctx.get("data") or {}).get("answer_route") or {}
            log_routing_decision(
                query=original_msg,
                language=rl,
                intent=ar.get("intent") or adr.get("intent") or "",
                selected_tool=adr.get("handler") or ar.get("route_handler") or "",
                api_time_ms=response_time_sec() * 1000.0,
            )
            log_turn_complete(
                intent=ar.get("intent") or adr.get("intent") or "",
                route=adr.get("handler") or ar.get("route_handler") or "",
                source=adr.get("handler") or ar.get("route_handler") or adr.get("source") or "",
                reason=adr.get("reason") or ar.get("reasoning") or "",
            )
        except Exception:
            pass
        return jsonify({"chat_id": current_chat_id, "type": "text", "data": final_output})

    # Pagination handled at top of chat() and in guard — see _pagination_append_json_response.

    # User message already stored above; conv_for_llm loaded for routing.
    comb_ph_early = f"{original_msg} {msg_en}".lower()

    # === ORDER ID HANDOFF — bare ID / awaiting thread (zero LLM before brain) ===
    try:
        from services.order_id_handoff_fast_path import try_order_id_handoff_reply

        handoff_early = try_order_id_handoff_reply(
            original_msg,
            msg_en,
            conv_for_llm,
            user_id,
            lang,
            ctx,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            reset_context_fn=reset_context,
        )
        if handoff_early:
            log_reasoning("Order-ID handoff — before brain (pending intent + submitted id).")
            return send_reply(handoff_early, lang)
    except ImportError:
        pass

    # === STRUCTURAL ORDER: live API without brain LLM (invoice / track / refund / details) ===
    try:
        from services.brain_direct_dispatch import try_structural_order_live_reply

        structural_order = try_structural_order_live_reply(
            original_msg,
            msg_en,
            conv_for_llm=conv_for_llm,
            user_id=user_id,
            lang=lang,
            ctx=ctx,
            reset_context_fn=reset_context,
            reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
        )
        if structural_order:
            log_reasoning("Structural order fast path — before brain.")
            return send_reply(structural_order, lang)
    except ImportError:
        pass

    # === AI BRAIN: one LLM — detect meaning in any language/style, then route ===
    brain_route_data = None
    brain_route_ok = False
    _skip_repeat_brain = False
    try:
        from services.chat_flow_telemetry import get_cached_brain_route, is_routing_complete

        _cached_br = get_cached_brain_route()
        _skip_repeat_brain = is_routing_complete() or (
            isinstance(_cached_br, dict) and not _cached_br.get("llm_unavailable")
        )
        if _skip_repeat_brain and isinstance(_cached_br, dict):
            brain_route_data = _cached_br
            brain_route_ok = not _cached_br.get("llm_unavailable")
    except ImportError:
        pass
    if _skip_repeat_brain:
        log_reasoning(
            "Skip duplicate ai_brain_route — reuse cached brain classification."
        )
    if not _skip_repeat_brain:
        # Awaiting Order ID + numeric reply — never run brain (avoids timeout on bare id).
        if isinstance(ctx, dict) and ctx.get("awaiting") == "order_id":
            from utils.helpers import message_is_bare_numeric_submission

            if message_is_bare_numeric_submission(original_msg):
                try:
                    from services.order_id_handoff_fast_path import try_order_id_handoff_reply

                    handoff_locked = try_order_id_handoff_reply(
                        original_msg,
                        msg_en,
                        conv_for_llm,
                        user_id,
                        lang,
                        ctx,
                        reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                        reset_context_fn=reset_context,
                    )
                    if handoff_locked:
                        log_reasoning(
                            "Order-ID locked handoff — skip ai_brain_route (zero LLM)."
                        )
                        return send_reply(handoff_locked, lang)
                except ImportError:
                    pass
        try:
            from services.ai_first_router import early_universal_brain_route
            from services.brain_direct_dispatch import try_brain_direct_dispatch

            _t_brain = time.perf_counter()
            brain_route_data = early_universal_brain_route(
                original_msg, conv_for_llm, lang, msg_en=msg_en, ctx=ctx
            )
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "ai_brain_route",
                    (time.perf_counter() - _t_brain) * 1000.0,
                )
            except ImportError:
                pass
            brain_route_ok = bool(
                isinstance(brain_route_data, dict)
                and not brain_route_data.get("llm_unavailable")
            )
            if brain_route_ok:
                ctx.setdefault("data", {})["ai_route"] = brain_route_data
                _t_dispatch = time.perf_counter()
                brain_direct = try_brain_direct_dispatch(
                    brain_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm,
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    format_purchase_history_reply=format_purchase_history_reply,
                    format_wishlist_reply=format_wishlist_reply,
                    reset_context_fn=reset_context,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                )
                try:
                    from services.chat_flow_telemetry import record_phase

                    record_phase(
                        "brain_direct_dispatch",
                        (time.perf_counter() - _t_dispatch) * 1000.0,
                    )
                except ImportError:
                    pass
                if brain_direct:
                    log_reasoning("AI brain first — reply from brain analysis.")
                    return send_reply(
                        brain_direct, lang, ai_route_snapshot=brain_route_data
                    )
                try:
                    from services.brain_direct_dispatch import (
                        _try_brain_product_catalog_fallback_dispatch,
                    )

                    product_fb = _try_brain_product_catalog_fallback_dispatch(
                        brain_route_data,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conv_for_llm=conv_for_llm,
                        user_id=user_id,
                        lang=lang,
                        ctx=ctx,
                        reset_context_fn=reset_context,
                    )
                    if product_fb:
                        log_reasoning(
                            "Brain direct miss — product catalog fallback (zero extra LLM)."
                        )
                        return send_reply(
                            product_fb, lang, ai_route_snapshot=brain_route_data
                        )
                except ImportError:
                    pass
        except ImportError:
            pass

    elif brain_route_ok and isinstance(brain_route_data, dict):
        try:
            from services.brain_direct_dispatch import (
                try_brain_direct_dispatch,
                try_finish_brain_classified_turn,
            )

            ctx.setdefault("data", {})["ai_route"] = brain_route_data
            finish_cached = try_finish_brain_classified_turn(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if finish_cached:
                log_reasoning("Legacy path — finish dispatch from cached brain route.")
                return send_reply(
                    finish_cached, lang, ai_route_snapshot=brain_route_data
                )
            brain_direct = try_brain_direct_dispatch(
                brain_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            if brain_direct:
                log_reasoning("Legacy path — brain direct from cached route.")
                return send_reply(
                    brain_direct, lang, ai_route_snapshot=brain_route_data
                )
        except ImportError:
            pass

    try:
        from utils.helpers import (
            _text_is_product_id_lookup_context,
            extract_product_id,
            turn_is_catalog_product_lookup,
        )
        from services.opensearch_products import _extract_sku_from_text
        from services.catalog_spec_semantics import user_mentions_sku_this_turn
        from services.product_search_flow import run_product_search_ai_flow

        pid_struct = None
        if _text_is_product_id_lookup_context(original_msg) or turn_is_catalog_product_lookup(
            original_msg
        ):
            pid_struct = extract_product_id(original_msg)
        sku_struct = (
            _extract_sku_from_text(original_msg)
            if user_mentions_sku_this_turn(original_msg)
            else None
        )
        if pid_struct or sku_struct:
            ctx.setdefault("data", {})
            if sku_struct:
                ctx["data"].pop("lookup_pro_id", None)
                ctx["data"]["lookup_sku"] = sku_struct
                ctx["data"].pop("selected_category_id", None)
            elif pid_struct:
                ctx["data"]["lookup_pro_id"] = pid_struct
                ctx["data"].pop("lookup_sku", None)
                ctx["order_id"] = None
            ctx["awaiting"] = None
            ps = run_product_search_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
            )
            if ps.handled and ps.reply_html:
                log_reasoning(
                    f"Structural catalog fast exit (pro_id={pid_struct}, sku={sku_struct!r})."
                )
                reset_context(ctx)
                return send_reply(ps.reply_html, lang)
    except ImportError:
        pass

    retrieval_query = build_retrieval_query(msg_en, conv_for_llm, original_msg)

    def _refresh_retrieval_query(ai_route: dict | None = None) -> str:
        try:
            from services.query_understanding import enhance_retrieval_query

            return enhance_retrieval_query(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route
            )
        except ImportError:
            return build_retrieval_query(msg_en, conv_for_llm, original_msg)

    def _should_offer_human_escalation(user_text: str) -> bool:
        """
        Offer human escalation only for repeated serious unresolved cases.
        This avoids unnecessary customer-care dumps on normal policy/info turns.
        """
        low = f" {user_text.lower()} "
        aggressive = any(
            x in low
            for x in (
                "frustrat",
                "angry",
                "useless",
                "idiot",
                "bekar",
                "bevkuf",
                "faltu",
                "same reply",
                "not helping",
                "nahi ho raha",
                "bar bar",
                "again",
                "still",
                "not satisfied",
                "unsatisfied",
                "not resolved",
                "resolve nahi",
            )
        )
        normalized_now = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", low)).strip()
        prev_user_msgs = [
            (m.get("message") or "").lower().strip()
            for m in (recent_msgs or [])
            if (m.get("sender") or "").lower() == "user"
        ]
        similar_count = 0
        for p in prev_user_msgs[-5:]:
            pn = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", p)).strip()
            if not pn:
                continue
            overlap = set(normalized_now.split()) & set(pn.split())
            if len(overlap) >= 3:
                similar_count += 1
        repeated_unresolved = similar_count >= 2
        base_support_case = message_needs_human_support_escalation(user_text)
        if base_support_case and (aggressive or repeated_unresolved):
            return True
        serious_topic = any(
            x in low for x in ("fraud", "phishing", "scam", "complaint", "grievance", "legal")
        )
        return serious_topic and aggressive and repeated_unresolved

    comb_pre_route = f"{original_msg} {msg_en}".lower()

    # === UNIVERSAL BRAIN (cached from AI-first pass above) ===
    if not brain_route_ok:
        # === LLM down — legacy fast paths (extra micro-LLMs) ===
        try:
            from services.support_scope import try_external_scope_fast_decline

            external_decline = try_external_scope_fast_decline(
                original_msg, msg_en, conv_for_llm, reply_lang=lang
            )
            if external_decline:
                log_reasoning("External scope — decline before fast paths.")
                reset_context(ctx)
                return send_reply(external_decline, lang)
        except ImportError:
            pass

        try:
            from services.account_list_fast_path import try_account_list_fast_reply

            account_list_reply = try_account_list_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                user_id,
                lang,
                ctx,
                format_purchase_history_reply=format_purchase_history_reply,
                format_wishlist_reply=format_wishlist_reply,
                localized_sysmsg=_localized_sysmsg,
                sysmsg=sysmsg,
                reset_context_fn=reset_context,
            )
            if account_list_reply:
                log_reasoning("Account-list fast path — order history or wishlist.")
                return send_reply(account_list_reply, lang)
        except ImportError:
            pass

        try:
            from services.pincode_delivery_fast_path import try_pincode_delivery_fast_reply

            pin_fast = try_pincode_delivery_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                ctx,
                reset_context_fn=reset_context,
            )
            if pin_fast:
                log_reasoning("Pincode delivery fast path — reply before AI routing.")
                return send_reply(pin_fast, lang)
        except ImportError:
            pass

        try:
            from services.order_live_intent_fast_path import try_order_live_intent_fast_reply

            live_goal_reply = try_order_live_intent_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if live_goal_reply:
                log_reasoning("Order live intent fast path — latest message goal.")
                return send_reply(live_goal_reply, lang)
        except ImportError:
            pass

        try:
            from services.knowledge_fast_path import try_knowledge_fast_reply

            kb_fast = try_knowledge_fast_reply(
                original_msg, msg_en, conv_for_llm, reply_lang=lang
            )
            if kb_fast:
                log_reasoning("Knowledge fast path — FAQ/policy before refund routing.")
                reset_context(ctx)
                return send_reply(kb_fast, lang)
        except ImportError:
            pass

        try:
            from services.refund_intent_fast_path import try_refund_intent_fast_reply

            refund_fast = try_refund_intent_fast_reply(
                original_msg,
                msg_en,
                conv_for_llm,
                user_id,
                lang,
                ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if refund_fast:
                log_reasoning("Refund fast path — reply before AI routing.")
                return send_reply(refund_fast, lang)
        except ImportError:
            pass
    else:
        log_reasoning(
            "Universal brain route active — locked executor (no second routing LLM)."
        )
        _t_finalize = time.perf_counter()
        try:
            from services.ai_first_router import _finalize_brain_route_decision
            from services.locked_route_executor import execute_locked_route_or_fallback

            route_decision, ai_route_data = _finalize_brain_route_decision(
                brain_route_data,
                original_msg,
                msg_en,
                conv_for_llm=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
            )
            ai_route_data = ai_route_data or brain_route_data
            ctx.setdefault("data", {})["ai_route"] = ai_route_data
            ctx["data"]["answer_route"] = {
                "source": route_decision.source,
                "intent": route_decision.intent,
                "handler": route_decision.handler,
                "reason": route_decision.reason,
            }
            try:
                from services.chat_flow_telemetry import record_phase, record_route

                record_phase(
                    "brain_finalize",
                    (time.perf_counter() - _t_finalize) * 1000.0,
                )
                record_route(
                    intent=(ai_route_data or {}).get("intent") or route_decision.intent,
                    source=route_decision.handler
                    or (ai_route_data or {}).get("route_handler")
                    or route_decision.source,
                )
            except ImportError:
                pass
            retrieval_query = _refresh_retrieval_query(ai_route_data)
            _t_locked = time.perf_counter()
            final_body = execute_locked_route_or_fallback(
                route_decision=route_decision,
                ai_route=ai_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                retrieval_query=retrieval_query,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            try:
                from services.chat_flow_telemetry import record_phase

                record_phase(
                    "locked_route_executor",
                    (time.perf_counter() - _t_locked) * 1000.0,
                )
            except ImportError:
                pass
            reset_context_unless_order_pending(ctx)
            return send_reply(final_body or sysmsg("how_can_i_help_welfog"), lang)
        except ImportError:
            pass
        except Exception as brain_exec_exc:
            import traceback

            traceback.print_exc()
            log_reasoning(f"Brain locked executor failed (fallback): {brain_exec_exc}")
            reset_context_unless_order_pending(ctx)
            try:
                from services.brain_direct_dispatch import (
                    _try_brain_order_live_fallback_dispatch,
                )

                order_exc_fb = _try_brain_order_live_fallback_dispatch(
                    brain_route_data or {},
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conv_for_llm=conv_for_llm
                    or _conversation_snapshot_for_chat(current_chat_id),
                    user_id=user_id,
                    lang=lang,
                    ctx=ctx,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context,
                )
                if order_exc_fb:
                    return send_reply(
                        order_exc_fb, lang, ai_route_snapshot=brain_route_data
                    )
            except ImportError:
                pass
            scope_reply = ""
            if isinstance(brain_route_data, dict):
                scope_reply = (brain_route_data.get("scope_reply") or "").strip()
            try:
                from services.conversation_zero_llm_fallback import (
                    try_zero_llm_customer_reply,
                )

                recovered = try_zero_llm_customer_reply(
                    original_msg,
                    msg_en,
                    conv_for_llm
                    or _conversation_snapshot_for_chat(current_chat_id),
                    reply_lang=lang,
                    ai_route=brain_route_data,
                )
                if recovered:
                    log_reasoning(
                        "Brain executor failed — zero-LLM KB/chitchat recovery."
                    )
                    return send_reply(
                        recovered, lang, ai_route_snapshot=brain_route_data
                    )
            except ImportError:
                pass
            return send_reply(
                scope_reply or sysmsg("server_busy"),
                lang,
                ai_route_snapshot=brain_route_data,
            )

    if brain_route_ok:
        try:
            from services.brain_direct_dispatch import _try_brain_order_live_fallback_dispatch

            busy_fallback = _try_brain_order_live_fallback_dispatch(
                brain_route_data or {},
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                reset_context_fn=reset_context,
            )
            if busy_fallback:
                log_reasoning(
                    "Brain route OK — order live fallback before busy reply (zero extra LLM)."
                )
                return send_reply(busy_fallback, lang, ai_route_snapshot=brain_route_data)
            from services.brain_direct_dispatch import (
                _try_brain_product_catalog_fallback_dispatch,
            )

            product_busy_fb = _try_brain_product_catalog_fallback_dispatch(
                brain_route_data or {},
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
            )
            if product_busy_fb:
                log_reasoning(
                    "Brain route OK — product catalog fallback before busy reply (zero extra LLM)."
                )
                return send_reply(
                    product_busy_fb, lang, ai_route_snapshot=brain_route_data
                )
            try:
                from services.order_id_handoff_fast_path import try_order_id_handoff_reply

                conv_busy = conv_for_llm or _conversation_snapshot_for_chat(
                    current_chat_id
                )
                handoff_busy = try_order_id_handoff_reply(
                    original_msg,
                    msg_en,
                    conv_busy,
                    user_id,
                    lang,
                    ctx,
                    reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
                    reset_context_fn=reset_context,
                )
                if handoff_busy:
                    log_reasoning(
                        "Brain route OK — order ID handoff before busy reply (zero extra LLM)."
                    )
                    return send_reply(
                        handoff_busy, lang, ai_route_snapshot=brain_route_data
                    )
            except ImportError:
                pass
        except ImportError:
            pass
        try:
            from services.conversation_zero_llm_fallback import (
                try_zero_llm_customer_reply,
            )

            recovered = try_zero_llm_customer_reply(
                original_msg,
                msg_en,
                conv_for_llm or _conversation_snapshot_for_chat(current_chat_id),
                reply_lang=lang,
                ai_route=brain_route_data,
            )
            if recovered:
                log_reasoning(
                    "Brain route OK but no dispatch — zero-LLM KB/chitchat recovery."
                )
                return send_reply(
                    recovered, lang, ai_route_snapshot=brain_route_data
                )
        except ImportError:
            pass
        reset_context_unless_order_pending(ctx)
        return send_reply(
            sysmsg("server_busy"),
            lang,
            ai_route_snapshot=brain_route_data,
        )

    # === AI-FIRST fallback — only when universal brain route unavailable ===
    log_reasoning(f"AI routing fallback: {original_msg[:120]!r}")
    from services.ai_first_router import resolve_answer_route_ai_first
    from services.semantic_intent import skip_keyword_intent_routes, should_skip_ctx_last_pinning

    try:
        route_decision, ai_route_data = resolve_answer_route_ai_first(
            original_msg,
            msg_en,
            retrieval_query=retrieval_query,
            conv_for_llm=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
        )
    except Exception as route_exc:
        import traceback

        traceback.print_exc()
        log_reasoning(f"AI routing failed (safe fallback): {route_exc}")
        try:
            from services.conversation_scope import try_conversation_scope_reply

            scope_on_fail = try_conversation_scope_reply(
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ai_route=None,
                route_handler="",
                prefer_llm=False,
            )
            if scope_on_fail:
                log_reasoning("Routing exception — conversation scope reply.")
                reset_context(ctx)
                return send_reply(scope_on_fail, lang)
        except ImportError:
            pass
        from services.answer_router import resolve_answer_route

        route_decision = resolve_answer_route(
            original_msg, msg_en, retrieval_query=retrieval_query, conv_for_llm=conv_for_llm, ctx=ctx
        )
        ai_route_data = None
    ctx.setdefault("data", {})["answer_route"] = {
        "source": route_decision.source,
        "intent": route_decision.intent,
        "handler": route_decision.handler,
        "reason": route_decision.reason,
    }
    if ai_route_data:
        ctx["data"]["ai_route"] = ai_route_data

    try:
        from services.chat_flow_telemetry import record_route

        record_route(
            intent=(ai_route_data or {}).get("intent") or route_decision.intent,
            source=route_decision.handler or (ai_route_data or {}).get("route_handler") or route_decision.source,
        )
    except ImportError:
        pass

    retrieval_query = _refresh_retrieval_query(ai_route_data)

    # === LOCKED ROUTE: one analysis → one handler → one reply ===
    try:
        from services.chat_flow_telemetry import is_routing_complete
        from services.locked_route_executor import execute_locked_route_or_fallback

        if is_routing_complete():
            final_body = execute_locked_route_or_fallback(
                route_decision=route_decision,
                ai_route=ai_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                retrieval_query=retrieval_query,
                user_id=user_id,
                lang=lang,
                ctx=ctx,
                reset_context_fn=reset_context,
                reply_for_live_order_id_lookup=_reply_for_live_order_id_lookup,
            )
            reset_context(ctx)
            return send_reply(final_body or sysmsg("how_can_i_help_welfog"), lang)
    except ImportError:
        pass

    try:
        from services.chat_flow_telemetry import is_routing_complete
        from services.query_intent_classifier import try_query_intent_gate_reply

        if not is_routing_complete():
            intent_gate = try_query_intent_gate_reply(
                original_msg,
                msg_en,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
                ai_route=ai_route_data,
            )
            if intent_gate:
                log_reasoning("Query intent gate (post-route) — skip KB/API/catalog.")
                reset_context(ctx)
                return send_reply(intent_gate, lang)
    except ImportError:
        pass

    _routing_locked = False
    try:
        from services.chat_flow_telemetry import is_routing_complete

        _routing_locked = is_routing_complete()
    except ImportError:
        pass

    if not _routing_locked and (route_decision.handler or "").strip() == "warm_feedback":
        try:
            from services.product_catalog_resolver import turn_requests_product_catalog

            if turn_requests_product_catalog(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=ai_route_data,
                allow_llm=False,
            ):
                log_reasoning(
                    "Warm feedback skipped — product catalog turn (shopping signal)."
                )
            else:
                log_reasoning("Warm conversational reply — early exit (no KB/API).")
                reset_context(ctx)
                return send_reply(
                    build_warm_feedback_reply(
                        original_msg,
                        msg_en,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        ctx=ctx,
                    ),
                    lang,
                )
        except ImportError:
            log_reasoning("Warm conversational reply — early exit (no KB/API).")
            reset_context(ctx)
            return send_reply(
                build_warm_feedback_reply(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                    ctx=ctx,
                ),
                lang,
            )

    if not _routing_locked:
        try:
            from services.query_understanding import maybe_clarification_reply

            clarify_html = maybe_clarification_reply(
                ai_route_data,
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                route_decision=route_decision,
            )
            if clarify_html:
                log_reasoning("Query understanding — clarification (low confidence, skip guess).")
                reset_context(ctx)
                return send_reply(clarify_html, lang)
        except ImportError:
            pass

        try:
            from services.account_list_semantics import detect_account_list_followup_in_chat

            follow_list = detect_account_list_followup_in_chat(
                original_msg,
                msg_en,
                conv_for_llm,
                ctx=ctx,
                ai_route=ai_route_data,
                reply_lang=lang,
            )
            if follow_list == "wishlist":
                log_reasoning("Follow-up after wishlist how-to → show list in chat.")
                reset_context(ctx)
                ctx["last"] = "wishlist"
                ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
                return send_reply(
                    format_wishlist_reply(user_id, page=1, append_only=False), lang
                )
            if follow_list == "order_history":
                log_reasoning("Follow-up after order-history how-to → show list in chat.")
                reset_context(ctx)
                ctx["last"] = "order_history"
                ctx.setdefault("data", {})["topic_mode"] = "order_history_list"
                return send_reply(
                    format_purchase_history_reply(user_id, page=1, append_only=False), lang
                )
        except ImportError:
            pass

    # === LEGACY CASCADE (only when main router did not complete — LLM unavailable) ===
    from services.query_intent_classifier import (
        INTENT_HARM,
        INTENT_WELFOG,
        build_non_welfog_reply,
        get_request_query_intent,
        reconcile_query_intent_with_route,
    )
    from services.conversation_scope import try_conversation_scope_reply

    post_route_qi = reconcile_query_intent_with_route(ai_route_data, ctx) or get_request_query_intent()
    if post_route_qi and post_route_qi.detected_intent != INTENT_WELFOG:
        from services.conversation_scope import should_bypass_conversation_scope

        if not should_bypass_conversation_scope(
            ai_route_data,
            route_handler=(route_decision.handler or ""),
            original_msg=original_msg,
            msg_en=msg_en,
            conversation_context=conv_for_llm,
        ):
            post_html = build_non_welfog_reply(
                post_route_qi, original_msg, msg_en, reply_lang=lang
            )
            if post_html and (
                post_route_qi.detected_intent == INTENT_HARM
                or post_route_qi.confidence >= 0.68
            ):
                log_reasoning(
                    "Query intent gate (post-route) — block KB/API/catalog."
                )
                reset_context(ctx)
                return send_reply(post_html, lang)

    scope_early = try_conversation_scope_reply(
        original_msg,
        msg_en,
        conversation_context=conv_for_llm,
        reply_lang=lang,
        ai_route=ai_route_data,
        route_handler=(route_decision.handler or ""),
    )
    if scope_early:
        try:
            from services.pincode_delivery_fast_path import turn_is_pincode_delivery_fast_path

            if turn_is_pincode_delivery_fast_path(
                original_msg, msg_en, conv_for_llm, ctx
            ):
                log_reasoning(
                    "Skip chitchat scope — pincode delivery turn (live API wins)."
                )
                scope_early = None
        except ImportError:
            pass
    if scope_early:
        log_reasoning("AI conversation scope — chitchat/out_of_domain (skip KB/API/catalog).")
        reset_context(ctx)
        return send_reply(scope_early, lang)

    from utils.helpers import _text_is_pincode_serviceability_question
    from services.pincode_delivery_flow import run_pincode_delivery_ai_flow

    try:
        from services.account_list_semantics import (
            account_list_route_is_locked,
            ai_route_requests_wishlist_howto,
        )

        skip_pin_early = account_list_route_is_locked(ai_route_data) or ai_route_requests_wishlist_howto(
            ai_route_data, original_msg, msg_en, conv_for_llm
        )
    except ImportError:
        skip_pin_early = False

    if not skip_pin_early and _text_is_pincode_serviceability_question(
        f"{original_msg} {msg_en}", conv_for_llm
    ):
        pin_pre = run_pincode_delivery_ai_flow(
            original_msg, msg_en, conv_for_llm, reply_lang=lang
        )
        if pin_pre.handled and pin_pre.reply_html:
            log_reasoning(
                "Delivery serviceability — pincode flow before intent executor."
            )
            reset_context(ctx)
            try:
                from utils.helpers import mark_pincode_delivery_completed

                mark_pincode_delivery_completed(ctx)
            except ImportError:
                ctx["last"] = None
                ctx["awaiting"] = None
                ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
            return send_reply(pin_pre.reply_html, lang)

    from services.intent_executor import (
        ai_route_blocks_generic_shortcuts,
        execute_detected_intent_reply,
    )
    from services.semantic_intent import ai_route_is_pincode_intent, ai_route_requests_pincode_delivery

    ai_pin_semantic = (
        route_decision.intent == "pincode_check"
        or route_decision.handler == "pincode_delivery_api"
        or ai_route_is_pincode_intent(ai_route_data)
        or ai_route_requests_pincode_delivery(ai_route_data)
    )

    def _live_order_cb():
        return _try_live_order_id_reply_early(
            original_msg, msg_en, conv_for_llm, user_id, lang, ctx, ai_route=ai_route_data
        )

    intent_reply = execute_detected_intent_reply(
        route_decision,
        ai_route_data,
        original_msg,
        msg_en,
        conv_for_llm,
        user_id,
        lang,
        ctx,
        try_live_order_callback=_live_order_cb,
    )
    if intent_reply:
        return send_reply(intent_reply, lang)

    from utils.helpers import (
        message_is_user_confused_or_rephrasing_bot,
        _conversation_in_pincode_delivery_flow,
        _text_is_pincode_serviceability_question,
    )

    if message_is_user_confused_or_rephrasing_bot(
        f"{original_msg} {msg_en}".strip(), conv_for_llm
    ):
        comb_conf = f"{original_msg} {msg_en}".strip()
        try:
            from utils.helpers import (
                _turn_blocks_pincode_serviceability_routing,
                message_is_past_purchase_list_request,
                turn_is_obvious_product_shopping_turn,
            )

            topic_switch = (
                _turn_blocks_pincode_serviceability_routing(comb_conf)
                or turn_is_obvious_product_shopping_turn(
                    original_msg, msg_en, conv_for_llm
                )
                or message_is_past_purchase_list_request(comb_conf)
            )
        except ImportError:
            topic_switch = False
        if not topic_switch and (
            ai_pin_semantic
            or _conversation_in_pincode_delivery_flow(conv_for_llm)
            or _text_is_pincode_serviceability_question(
                f"{original_msg} {msg_en}".strip(), conv_for_llm
            )
        ):
            from services.pincode_delivery_flow import build_pincode_missing_or_invalid_reply

            clarify_pin = build_pincode_missing_or_invalid_reply(
                original_msg, msg_en, conv_for_llm, reply_lang=lang
            )
            if clarify_pin:
                log_reasoning("User confused on delivery thread — re-ask PIN (not warm/KB).")
                try:
                    from utils.helpers import set_pincode_await_context

                    set_pincode_await_context(ctx)
                except ImportError:
                    ctx["last"] = "pincode"
                    ctx.setdefault("data", {})["topic_mode"] = "pincode_check"
                return send_reply(clarify_pin, lang)

    keyword_routes_ok = not skip_keyword_intent_routes(ai_route_data)

    # Core deterministic intents — only when LLM unavailable (keyword fallback).
    comb_core = f"{original_msg} {msg_en}".strip()
    from utils.helpers import _should_use_deterministic_core_route

    use_det_core = _should_use_deterministic_core_route(comb_core, conv_for_llm)
    comb_core_low = f" {comb_core.lower()} "
    explicit_order_history_howto = (
        any(x in comb_core_low for x in ("order history", "my orders", "purchase history"))
        and any(x in comb_core_low for x in (" where ", " how ", " steps ", " process ", " app ", " website ", " view "))
        and not any(x in comb_core_low for x in ("show my", "show me", "in chat", "list in chat", "dikhao"))
    )
    if keyword_routes_ok and use_det_core and (
        (_text_has_explicit_how_to_place_order(comb_core) or _text_has_order_placement_intent(comb_core))
        and not _text_has_refund_or_return_intent(comb_core)
    ):
        log_reasoning("Deterministic core route: order placement how-to.")
        reset_context(ctx)
        ctx["last"] = "general"
        return send_reply(
            _localized_sysmsg("order_placement_help", original_msg, reply_lang=lang)
            or sysmsg("order_placement_help")
            or "",
            lang,
        )
    if keyword_routes_ok and use_det_core and (
        explicit_order_history_howto or _user_asks_order_history_navigation_help(comb_core, conv_for_llm)
    ):
        log_reasoning("Deterministic core route: order history how-to.")
        reset_context(ctx)
        ctx["last"] = "order_history"
        ctx.setdefault("data", {})["topic_mode"] = "order_history_howto"
        return send_reply(
            _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
            or sysmsg("order_history_help")
            or "",
            lang,
        )
    if keyword_routes_ok and use_det_core and _text_asks_how_to_view_wishlist(
        comb_core, conv_for_llm
    ):
        log_reasoning("Deterministic core route: wishlist how-to.")
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
        return send_reply(
            _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
            or sysmsg("wishlist_help")
            or "",
            lang,
        )
    if keyword_routes_ok and use_det_core and (_text_asks_wishlist(comb_core) or message_is_wishlist_like_request(comb_core)):
        log_reasoning("Deterministic core route: wishlist list.")
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
        return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)
    if keyword_routes_ok and use_det_core:
        try:
            from services.semantic_intent import should_skip_order_history_list_for_turn

            skip_ph_list = should_skip_order_history_list_for_turn(
                original_msg,
                msg_en,
                conv_for_llm,
                ai_route=(ctx.get("data") or {}).get("ai_route")
                if isinstance(ctx, dict)
                else None,
                ctx=ctx,
            )
        except ImportError:
            skip_ph_list = False
    else:
        skip_ph_list = True
    if (
        keyword_routes_ok
        and use_det_core
        and not skip_ph_list
        and (
            _text_asks_order_history(comb_core) or message_asks_my_welfog_purchases(comb_core)
        )
    ):
        log_reasoning("Deterministic core route: order history list.")
        reset_context(ctx)
        ctx["last"] = "order_history"
        ctx.setdefault("data", {})["topic_mode"] = "order_history_list"
        return send_reply(format_purchase_history_reply(user_id, page=1, append_only=False), lang)

    # Deterministic social routing before LLM to save tokens.
    from utils.helpers import message_asks_other_company_social_media, message_asks_welfog_social_media
    from services.support_scope import build_other_company_social_decline
    from services.kb_service import format_welfog_social_media_reply_from_kb

    social_turn = f"{original_msg} {msg_en}".strip()
    if message_asks_other_company_social_media(social_turn, conversation_context=conv_for_llm):
        log_reasoning("Other person/company social — deterministic decline (pre-AI).")
        reset_context(ctx)
        return send_reply(build_other_company_social_decline(original_msg, reply_lang=lang), lang)
    if message_asks_welfog_social_media(social_turn, conversation_context=conv_for_llm):
        social_reply = format_welfog_social_media_reply_from_kb(
            original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm
        )
        if social_reply:
            log_reasoning("Welfog social links — deterministic KB (pre-AI).")
            reset_context(ctx)
            return send_reply(social_reply, lang)

    # Direct KB policy/info queries: answer without LLM routing to avoid token burn.
    comb_direct = f"{original_msg} {msg_en}".strip()
    from utils.helpers import message_is_conversational_general_talk, _text_is_pincode_serviceability_question

    early_pin = _try_pincode_delivery_reply_early(
        original_msg, msg_en, conv_for_llm, lang, ctx, ai_route=ai_route_data
    )
    if early_pin:
        return send_reply(early_pin, lang)

    policy_or_support_turn = message_needs_policy_answer(comb_direct)
    if (
        (
            any(x in comb_direct.lower() for x in ("fraud", "phishing", "scam", "fake", "complaint", "grievance"))
            or policy_or_support_turn
            or message_is_knowledge_information_request(comb_direct, conversation_context=conv_for_llm)
            or message_is_welfog_about_request(comb_direct)
            or _text_asks_customer_care_contact(comb_direct)
        )
        and not ai_pin_semantic
        and not ai_route_blocks_generic_shortcuts(
            route_decision, ai_route_data, original_msg, msg_en, conv_for_llm
        )
        and not message_is_conversational_general_talk(original_msg, msg_en, conv_for_llm)
        and (policy_or_support_turn or not _text_has_product_shopping_intent(comb_direct))
        and not _text_is_order_tracking_intent(comb_direct)
        and not _text_asks_wishlist(comb_direct)
        and not _text_asks_order_history(comb_direct)
        and not _text_is_pincode_serviceability_question(comb_direct, conv_for_llm)
    ):
        if any(x in comb_direct.lower() for x in ("fraud", "phishing", "scam", "fake", "complaint", "grievance")):
            from services.kb_service import format_fraud_complaint_reply_from_kb

            kb_quick = format_fraud_complaint_reply_from_kb(original_msg, msg_en, reply_lang=lang)
        elif _text_asks_customer_care_contact(comb_direct):
            kb_quick = format_customer_care_reply_from_kb(original_msg, msg_en)
        elif message_is_welfog_about_request(comb_direct):
            kb_quick = format_welfog_about_reply_from_kb(original_msg, msg_en, reply_lang=lang)
        elif message_needs_policy_answer(comb_direct):
            kb_quick = format_policy_help_reply_from_kb(original_msg, msg_en, reply_lang=lang)
        else:
            kb_quick = format_knowledge_information_reply_from_kb(original_msg, msg_en, reply_lang=lang)
        if kb_quick:
            log_reasoning("Direct KB fast path — skip AI routing for policy/info query.")
            reset_context(ctx)
            return send_reply(kb_quick, lang)

    early_live = _try_live_order_id_reply_early(
        original_msg, msg_en, conv_for_llm, user_id, lang, ctx, ai_route=ai_route_data
    )
    if early_live:
        return send_reply(early_live, lang)

    # Ambiguous follow-up should continue the latest active topic (wishlist/history),
    # not jump to older context from the same chat.
    comb_topic = f"{original_msg} {msg_en}".strip()

    nav_topic_early = resolve_navigation_help_topic(
        comb_topic, conv_for_llm, ai_route=ai_route_data
    )
    explicit_wishlist_now = nav_topic_early == "wishlist_howto"
    explicit_history_now = nav_topic_early == "order_history_howto" or (
        not nav_topic_early
        and (
            _text_asks_order_history(comb_topic)
            or _user_asks_order_history_navigation_help(comb_topic, conv_for_llm)
            or message_wants_order_history_app_navigation(comb_topic, conv_for_llm)
        )
    )
    msg_low = f" {original_msg.lower()} "
    followup_ambiguous = _looks_like_conversational_followup(original_msg, msg_en) or any(
        x in msg_low for x in (" he bta", " ye bta", " yeh bta", "tum ni", "tu ni", "nhi btaya", "nahi btaya")
    )
    skip_topic_pin = (
        message_needs_live_single_order_lookup(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
        )
        or should_skip_ctx_last_pinning(
            ai_route_data, original_msg, msg_en, conv_for_llm
        )
    )
    if followup_ambiguous and not explicit_wishlist_now and not explicit_history_now and not skip_topic_pin:
        last_topic = (ctx.get("last") or "").strip().lower()
        last_mode = ((ctx.get("data") or {}).get("topic_mode") or "").strip().lower()
        if last_topic == "wishlist":
            log_reasoning("Ambiguous follow-up pinned by ctx.last=wishlist.")
            reset_context(ctx)
            ctx["last"] = "wishlist"
            if last_mode == "wishlist_howto":
                return send_reply(
                    _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
                    or sysmsg("wishlist_help")
                    or "",
                    lang,
                )
            return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)
        if last_topic == "order_history":
            log_reasoning("Ambiguous follow-up pinned by ctx.last=order_history.")
            reset_context(ctx)
            ctx["last"] = "order_history"
            if last_mode == "order_history_howto":
                return send_reply(
                    _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                    or sysmsg("order_history_help")
                    or "",
                    lang,
                )
            return send_reply(format_purchase_history_reply(user_id, page=1, append_only=False), lang)
        conv_low = (conv_for_llm or "").lower()
        wishlist_idx = max(
            conv_low.rfind("wishlist app mein kaise"),
            conv_low.rfind("wishlist app me kaise"),
            conv_low.rfind("wishlist"),
            conv_low.rfind("your wishlist"),
        )
        history_idx = max(
            conv_low.rfind("order history"),
            conv_low.rfind("your order history"),
            conv_low.rfind("my orders"),
            conv_low.rfind("purchase history"),
        )
        if wishlist_idx > history_idx and wishlist_idx >= 0:
            if any(x in conv_low[wishlist_idx:] for x in ("app mein kaise", "app me kaise", "heart", "icon")):
                log_reasoning("Ambiguous follow-up pinned to latest topic=wishlist how-to.")
                reset_context(ctx)
                ctx["last"] = "wishlist"
                ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
                return send_reply(
                    _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
                    or sysmsg("wishlist_help")
                    or "",
                    lang,
                )
            log_reasoning("Ambiguous follow-up pinned to latest topic=wishlist list.")
            reset_context(ctx)
            ctx["last"] = "wishlist"
            ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
            return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)

    from services.support_scope import (
        build_other_company_support_decline,
        message_mentions_other_company_support,
    )

    if message_mentions_other_company_support(
        original_msg,
        msg_en,
        conv_for_llm,
        ai_route=ai_route_data,
        route_decision=route_decision,
    ):
        log_reasoning("Non-Welfog query — polite decline with user's topic (post AI route).")
        reset_context(ctx)
        return send_reply(
            build_other_company_support_decline(original_msg, reply_lang=lang), lang
        )

    ai_ch = ((ai_route_data or {}).get("data_channel") or "").strip().lower()
    ai_intent = ((ai_route_data or {}).get("intent") or "").strip().lower()
    ai_continue = bool((ai_route_data or {}).get("continue_previous_topic"))
    ai_handler = (route_decision.handler or "").strip()

    # Industry guard: do not carry stale task intent on vague short follow-ups.
    # Example: "sun bhai", "ek kaam tha", "dhyan se sun".
    # Ask clarification in the same user language instead of forcing refund/order/product.
    if (
        ctx.get("awaiting") is None
        and _is_low_information_turn_for_task_routing(original_msg, msg_en, ctx)
        and not message_needs_live_single_order_lookup(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
        )
        and ai_intent in ("order", "refund", "payment", "product", "pincode_check")
        and ai_continue
    ):
        log_reasoning(
            f"Low-information follow-up detected; blocking stale intent carryover ({ai_intent})."
        )
        reset_context(ctx)
        return send_reply(_clarify_request_scope_reply(original_msg, lang), lang)

    # Last-topic app-navigation lock (language-agnostic fallback):
    # If user is in wishlist/order-history thread and asks "where/how in app",
    # keep same topic how-to instead of drifting to tracking/pincode/off-topic.
    last_topic = (ctx.get("last") or "").strip().lower()
    low_now = f" {original_msg.lower()} "
    app_anchor = any(
        x in low_now
        for x in (
            " app ",
            " website ",
            " where ",
            " how ",
            " kaha ",
            " kahan ",
            " kidhar ",
            " dekh ",
            " dekhun ",
            " app me ",
            " app mein ",
            " app la ",
            " app lo ",
            "ऐप",
            "ਅੈਪ",
            "ஆப்",
            "అప్",
        )
    )
    if (
        explicit_wishlist_now
        and app_anchor
        and not extract_order_id(original_msg)
        and not re.search(r"\b[1-9]\d{5}\b", original_msg)
        and not _text_has_product_shopping_intent(original_msg)
    ):
        log_reasoning("App-navigation lock -> wishlist how-to (explicit wishlist topic).")
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
        return send_reply(
            _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
            or sysmsg("wishlist_help")
            or "",
            lang,
        )
    if (
        last_topic in ("wishlist", "order_history")
        and app_anchor
        and not extract_order_id(original_msg)
        and not re.search(r"\b[1-9]\d{5}\b", original_msg)
        and not _text_has_product_shopping_intent(original_msg)
        and not explicit_wishlist_now
        and ai_intent in ("general", "out_of_domain", "order", "payment", "pincode_check", "")
    ):
        log_reasoning(f"App-navigation lock -> keep last_topic={last_topic} how-to.")
        reset_context(ctx)
        ctx["last"] = last_topic
        if last_topic == "wishlist":
            ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
            return send_reply(
                _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
                or sysmsg("wishlist_help")
                or "",
                lang,
            )
        if explicit_history_now or not message_mentions_wishlist_topic(comb_topic):
            ctx.setdefault("data", {})["topic_mode"] = "order_history_howto"
            return send_reply(
                _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                or sysmsg("order_history_help")
                or "",
                lang,
            )

    # Language-agnostic follow-up lock: trust AI continue_previous_topic across all scripts/languages.
    # If user sends an ambiguous continuation, keep the last topic (wishlist/order_history).
    if (
        ai_continue
        and ai_intent in ("general", "")
        and not explicit_wishlist_now
        and not explicit_history_now
        and not message_needs_live_single_order_lookup(
            original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
        )
        and not should_skip_ctx_last_pinning(
            ai_route_data, original_msg, msg_en, conv_for_llm
        )
        and (ctx.get("last") or "").strip().lower() in ("wishlist", "order_history")
    ):
        last_topic = (ctx.get("last") or "").strip().lower()
        last_mode = ((ctx.get("data") or {}).get("topic_mode") or "").strip().lower()
        log_reasoning(f"AI continue_previous_topic lock -> last_topic={last_topic}, mode={last_mode or 'auto'}.")
        reset_context(ctx)
        ctx["last"] = last_topic
        if last_topic == "wishlist":
            if last_mode == "wishlist_howto":
                ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
                return send_reply(
                    _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
                    or sysmsg("wishlist_help")
                    or "",
                    lang,
                )
            ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
            return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)
        if last_mode == "order_history_howto":
            ctx.setdefault("data", {})["topic_mode"] = "order_history_howto"
            return send_reply(
                _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                or sysmsg("order_history_help")
                or "",
                lang,
            )
        ctx.setdefault("data", {})["topic_mode"] = "order_history_list"
        return send_reply(format_purchase_history_reply(user_id, page=1, append_only=False), lang)

    from services.kb_service import format_dynamic_kb_answer, should_try_admin_kb_answer

    from services.product_search_flow import (
        message_eligible_for_product_ai_flow,
        run_product_search_ai_flow,
    )

    from services.turn_intent_gate import format_meta_turn_reply
    from services.ai_route_semantics import (
        ai_route_allows_catalog_search,
        classify_meta_turn_ai_first,
    )

    meta_early = classify_meta_turn_ai_first(
        ai_route_data, original_msg, msg_en, conv_for_llm
    )
    if meta_early:
        meta_body = format_meta_turn_reply(meta_early, original_msg, reply_lang=lang)
        if meta_body:
            log_reasoning(f"Meta-turn reply: {meta_early.kind} (AI-first).")
            reset_context(ctx)
            return send_reply(meta_body, lang)

    from services.query_intent_classifier import query_intent_allows_catalog

    comb_product_early = f"{original_msg} {msg_en}".strip()
    if (
        query_intent_allows_catalog(ctx)
        and ai_route_allows_catalog_search(ai_route_data)
        and route_decision.intent == "product"
        and message_eligible_for_product_ai_flow(
            comb_product_early, msg_en, original_msg, ai_route=ai_route_data, ctx=ctx
        )
        and ai_handler not in ("wishlist_api", "order_ai_flow", "order_tracking_api", "order_details_api")
    ):
        ps_early = run_product_search_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
            search_query=extract_product_search_query(
                original_msg,
                msg_en,
                (ai_route_data or {}).get("search_query") or "",
                ai_route=ai_route_data,
            ),
            ai_route=ai_route_data,
        )
        if ps_early.handled and ps_early.reply_html:
            if ps_early.os_spec:
                ctx.setdefault("data", {})["last_os_spec"] = ps_early.os_spec
            log_reasoning("Product shopping — catalog API (before dynamic KB).")
            reset_context(ctx)
            return send_reply(ps_early.reply_html, lang)

    comb_nav = f"{original_msg} {msg_en}".strip()
    nav_help = resolve_navigation_help_topic(
        comb_nav, conv_for_llm, ai_route=ai_route_data
    )
    if nav_help == "order_history_howto":
        log_reasoning("Navigation topic resolved → order history how-to (semantic).")
        reset_context(ctx)
        ctx["last"] = "order_history"
        ctx.setdefault("data", {})["topic_mode"] = "order_history_howto"
        return send_reply(
            _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
            or sysmsg("order_history_help")
            or "",
            lang,
        )
    if nav_help == "wishlist_howto":
        log_reasoning("Navigation topic resolved → wishlist how-to (semantic).")
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
        return send_reply(
            _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
            or sysmsg("wishlist_help")
            or "",
            lang,
        )

    # Keep final response aligned with detected "how to place order" intent.
    # Avoid broad dynamic KB paragraphs when user asked only placement steps.
    comb_place_guard = customer_turn_text(original_msg, msg_en)
    place_guard_low = f" {comb_place_guard.lower()} "
    explicit_not_refund = (
        "not about return" in place_guard_low
        or "not about refund" in place_guard_low
        or "not return" in place_guard_low
        or "not refund" in place_guard_low
        or "return nahi" in place_guard_low
        or "refund nahi" in place_guard_low
        or "return ni" in place_guard_low
        or "refund ni" in place_guard_low
    )
    explicit_place_steps_query = bool(
        re.search(r"\b(how\s+(to|do|can)\s+i?\s*place)\b", place_guard_low)
        or re.search(r"\b(place|checkout)\s+(the\s+)?order\b", place_guard_low)
        or re.search(r"\border\s+(kaise|kese|kes)\b", place_guard_low)
    )
    if (
        nav_help != "order_history_howto"
        and (_text_has_order_placement_intent(comb_place_guard) or explicit_place_steps_query)
        and (explicit_not_refund or not _text_has_refund_or_return_intent(comb_place_guard))
        and ai_handler not in ("order_tracking_api", "order_details_api", "order_ai_flow", "wishlist_api", "product_ai_flow")
        and ai_intent not in ("product", "order_history", "wishlist", "refund", "payment")
    ):
        log_reasoning("Intent alignment guard: order placement question → placement steps only.")
        reset_context(ctx)
        return send_reply(
            _localized_sysmsg("order_placement_help", original_msg, reply_lang=lang)
            or sysmsg("order_placement_help")
            or "",
            lang,
        )

    kb_handler_whitelist = {
        "order_placement_kb",
        "order_tracking_howto_kb",
        "order_history_howto_kb",
        "seller_kb",
        "other_company_decline",
    }
    kb_question_gate = (
        message_is_knowledge_information_request(original_msg, conversation_context=conv_for_llm)
        or ai_handler in kb_handler_whitelist
    )

    from services.meta_turn_semantics import ai_route_is_assistant_intro_turn
    from services.user_query_semantics import try_company_kb_reply_html

    company_post = try_company_kb_reply_html(
        original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm
    )
    if company_post:
        log_reasoning("Post-AI: Welfog company/platform — KB (not assistant intro).")
        reset_context(ctx)
        return send_reply(company_post, lang)

    if ai_route_is_assistant_intro_turn(ai_route_data, original_msg, msg_en):
        from services.conversation_scope import (
            ScopeDecision,
            SCOPE_CHITCHAT,
            build_scope_reply,
            scope_from_ai_route,
        )

        intro_dec = scope_from_ai_route(ai_route_data) or ScopeDecision(
            scope=SCOPE_CHITCHAT, source="assistant_intro"
        )
        intro_body = build_scope_reply(intro_dec, original_msg, msg_en, reply_lang=lang) or build_assistant_intro_reply(
            original_msg, msg_en, reply_lang=lang
        )
        log_reasoning("Post-AI: assistant identity — AI scope reply or intro fallback.")
        reset_context(ctx)
        return send_reply(intro_body, lang)

    if should_use_warm_conversational_reply(
        original_msg, msg_en, conv_for_llm, ai_route_data
    ) and ai_intent in ("general", ""):
        log_reasoning("Post-AI: conversational thanks/praise — warm reply (skip dynamic_kb).")
        reset_context(ctx)
        return send_reply(build_warm_feedback_reply(
            original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm, ctx=ctx
        ), lang)

    from services.query_intent_classifier import query_intent_allows_kb

    if (
        query_intent_allows_kb(ctx)
        and kb_question_gate
        and should_try_admin_kb_answer(original_msg, msg_en, conv_for_llm)
        and not should_send_warm_feedback_reply(
            original_msg, msg_en, conversation_context=conv_for_llm, ai_route=ai_route_data
        )
        and ai_ch in ("kb", "")
        and ai_intent not in ("product", "wishlist", "order_history", "order", "pincode_check", "deals", "categories")
        and ai_handler not in ("order_history_howto_kb", "wishlist_howto_kb", "order_tracking_howto_kb")
        and ai_handler not in ("product_ai_flow", "wishlist_api", "order_ai_flow", "order_tracking_api", "order_details_api")
    ):
        kb_reply = format_dynamic_kb_answer(
            original_msg,
            msg_en,
            reply_lang=lang,
            conversation_context=conv_for_llm,
            ai_route=ai_route_data,
        )
        if kb_reply:
            log_reasoning("Admin panel KB — dynamic file match (AI channel=kb).")
            reset_context(ctx)
            return send_reply(kb_reply, lang)

    if (
        should_send_warm_greeting_reply(original_msg, msg_en, conversation_context=conv_for_llm)
        and ai_intent in ("general", "")
        and ai_ch in ("none", "kb", "")
        and ai_handler not in ("product_ai_flow", "wishlist_api", "order_ai_flow")
    ):
        log_reasoning("Greeting/smalltalk — warm reply (AI agrees general/none).")
        warm_html = build_warm_conversation_reply(original_msg, msg_en, reply_lang=lang)
        reset_context(ctx)
        pending = pending_offer_from_greeting_key(
            pick_warm_chat_reply_key(False, original_msg=original_msg, reply_lang=lang)
        )
        if pending:
            ctx.setdefault("data", {})["pending_offer"] = pending
        return send_reply(warm_html, lang)

    comb_scope_early = f"{original_msg} {msg_en}".strip()
    if not (
        _text_has_order_placement_intent(comb_scope_early)
        or _text_has_explicit_how_to_place_order(comb_scope_early)
    ) and message_mentions_other_company_support(
        original_msg,
        msg_en,
        conv_for_llm,
        ai_route=ai_route_data,
        route_decision=route_decision,
    ):
        log_reasoning("Other-company order/support — polite decline before routing.")
        return send_reply(build_other_company_support_decline(original_msg, reply_lang=lang), lang)

    if (
        _message_submits_or_corrects_order_id(original_msg)
        or _conversation_in_order_tracking_flow(conv_for_llm)
    ):
        log_reasoning("Order-tracking thread — continue with AI-first routing.")

    conv_sig = _conversation_cache_suffix(recent_msgs)
    sync_pending_offer_from_conversation(ctx, conv_for_llm)
    follow_skip_ai = None
    comb_support_early = f"{original_msg} {msg_en}"

    skip_order_id_for_pin_thread = (
        ai_intent == "pincode_check"
        or ((ai_route_data or {}).get("numeric_context") or "").strip().lower() == "pincode"
        or _conversation_bot_asked_for_pincode(conv_for_llm)
        or _resolve_ambiguous_bare_numeric_context(
            original_msg.strip(), conv_for_llm, ai_route=ai_route_data
        )
        == "pincode"
    )

    from utils.helpers import turn_is_catalog_product_lookup

    if not skip_order_id_for_pin_thread and not turn_is_catalog_product_lookup(
        original_msg, msg_en, ai_route_data, route_handler=route_decision.handler
    ) and should_attempt_live_order_api_reply(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
    ):
        pending_oid = (
            resolve_order_id_for_tracking(
                original_msg.strip() or msg_en.strip(),
                conv_for_llm,
                bot_awaiting_order_id=ctx.get("awaiting") == "order_id",
            )
            or extract_latest_order_id_from_user_conversation(conv_for_llm, original_msg)
        )
        if pending_oid:
            ai_route_early = (ctx.get("data") or {}).get("ai_route")
            live_intent = resolve_live_api_intent_from_conversation(
                conv_for_llm,
                ctx.get("last"),
                original_msg,
                msg_en,
                ai_route=ai_route_early if isinstance(ai_route_early, dict) else None,
            )
            log_reasoning(
                f"Order ID thread — live {live_intent} lookup for {pending_oid} (skip payment/product misroute)."
            )
            reply_html = _reply_for_live_order_id_lookup(
                live_intent, pending_oid, user_id, original_msg, lang
            )
            ctx["last"] = live_intent if live_intent in ("refund", "payment", "order") else "order"
            ctx["order_id"] = pending_oid
            ctx["awaiting"] = None
            return send_reply(reply_html, lang)

    early_reply = dispatch_early_answer(
        route_decision,
        original_msg,
        msg_en,
        reply_lang=lang,
        conv_for_llm=conv_for_llm,
        user_id=user_id,
        ai_route=ai_route_data,
    )
    if early_reply:
        if route_decision.handler == "order_id_help_kb":
            ctx["awaiting"] = "order_id"
            ctx["last"] = route_decision.intent if route_decision.intent in ("refund", "payment", "order") else (
                ctx.get("last") or "order"
            )
            olk = ((ai_route_data or {}).get("order_lookup_kind") or "").strip().lower()
            try:
                from services.ai_route_semantics import brain_route_to_live_goal

                goal = brain_route_to_live_goal(
                    ai_route_data,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    conversation_context=conv_for_llm,
                ) or "order_details"
            except ImportError:
                goal = "order_details"
                if olk == "invoice":
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
            return send_reply(early_reply, lang)
        reset_context(ctx)
        if route_decision.handler in ("wishlist_howto_kb", "wishlist_api") or route_decision.intent == "wishlist":
            ctx["last"] = "wishlist"
            ctx.setdefault("data", {})["topic_mode"] = (
                "wishlist_howto" if route_decision.handler == "wishlist_howto_kb" else "wishlist_list"
            )
        elif route_decision.handler in ("order_history_howto_kb", "order_ai_flow") or route_decision.intent == "order_history":
            ctx["last"] = "order_history"
            ctx.setdefault("data", {})["topic_mode"] = (
                "order_history_howto" if route_decision.handler == "order_history_howto_kb" else "order_history_list"
            )
        return send_reply(early_reply, lang)

    # Product search: AI reads product-search KB → catalog API (strict brand/model match)
    if route_decision.handler == "deals_api" or route_decision.intent == "deals":
        from services.catalog_menu_replies import build_today_deals_reply_html

        log_reasoning("Router → today deals API cards.")
        reset_context(ctx)
        ctx["data"] = {"pending_offer": "deals"}
        return send_reply(build_today_deals_reply_html(original_msg, reply_lang=lang), lang)

    if route_decision.handler == "categories_api" or route_decision.intent == "categories":
        from services.welfog_api import resolve_category_product_browse_route
        from services.product_search_flow import run_product_search_ai_flow

        ensure_expanded_categories_map_for_ctx(ctx)
        cat_browse = resolve_category_product_browse_route(f"{original_msg} {msg_en}", ctx=ctx)
        if cat_browse:
            cid, sq = cat_browse
            ctx.setdefault("data", {})["selected_category_id"] = cid
            ctx["awaiting"] = None
            log_reasoning(f"Category name browse → filtered products (category_id={cid}).")
            ps_result = run_product_search_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ctx=ctx,
                search_query=sq,
                ai_route=ai_route_data,
            )
            if ps_result.handled and ps_result.reply_html:
                if ps_result.os_spec:
                    ctx.setdefault("data", {})["last_os_spec"] = ps_result.os_spec
                reset_context(ctx)
                return send_reply(ps_result.reply_html, lang)

        from services.catalog_menu_replies import build_categories_list_reply_html

        log_reasoning("Router → Welfog categories list API.")
        reset_context(ctx)
        ctx["awaiting"] = "category_select"
        return send_reply(build_categories_list_reply_html(ctx, original_msg, reply_lang=lang), lang)

    if route_decision.handler == "product_ai_flow":
        from services.product_search_flow import run_product_search_ai_flow

        ps_result = run_product_search_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
            search_query=(route_decision.search_query or "").strip(),
            ai_route=ai_route_data,
        )
        if ps_result.handled and ps_result.reply_html:
            if ps_result.os_spec:
                ctx.setdefault("data", {})["last_os_spec"] = ps_result.os_spec
            reset_context(ctx)
            return send_reply(ps_result.reply_html, lang)
        comb_wl = f"{original_msg} {msg_en}"
        if (
            (_text_asks_wishlist(comb_wl) or message_is_wishlist_like_request(comb_wl))
            and not _text_asks_how_to_view_wishlist(comb_wl)
        ):
            log_reasoning("Product search skipped — user wants wishlist API.")
            reset_context(ctx)
            return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)

    # Order details / invoice: purchase-history-details + invoice download button
    if route_decision.handler == "order_details_api":
        from services.order_details_flow import run_order_details_ai_flow

        od_result = run_order_details_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ai_route=ai_route_data,
        )
        if od_result.handled and od_result.reply_html:
            if od_result.needs_order_id:
                ctx["last"] = "order"
                ctx["awaiting"] = "order_id"
                ctx["order_id"] = None
            else:
                reset_context(ctx)
            return send_reply(od_result.reply_html, lang)

    # Order tracking: AI + tracking KB → live welfog_track API
    if route_decision.handler == "order_tracking_api":
        from services.order_details_flow import (
            message_wants_order_details_or_invoice,
            run_order_details_ai_flow,
        )
        from services.ai_route_semantics import resolve_order_live_goal_for_turn

        live_goal = ""
        if isinstance(ai_route_data, dict):
            live_goal = resolve_order_live_goal_for_turn(
                ai_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conversation_context=conv_for_llm,
            )
        od_redirect = (
            message_wants_order_details_or_invoice(
                original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
            )
            if live_goal != "track"
            else ""
        )
        if od_redirect:
            log_reasoning(
                f"Tracking handler overridden → order_details_api ({od_redirect})."
            )
            od_result = run_order_details_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conv_for_llm,
                reply_lang=lang,
                ai_route=ai_route_data,
            )
            if od_result.handled and od_result.reply_html:
                if od_result.needs_order_id:
                    ctx["last"] = "order"
                    ctx["awaiting"] = "order_id"
                    ctx["order_id"] = None
                else:
                    reset_context(ctx)
                return send_reply(od_result.reply_html, lang)

        from services.order_tracking_flow import run_order_tracking_ai_flow

        ot_result = run_order_tracking_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
        )
        if ot_result.handled and ot_result.reply_html:
            if ot_result.needs_order_id:
                ctx["last"] = "order"
                ctx["awaiting"] = "order_id"
                ctx["order_id"] = None
            else:
                reset_context(ctx)
            return send_reply(ot_result.reply_html, lang)

    # Order ID: AI reads order-id API KB → ID list or help
    if route_decision.handler == "order_id_ai_flow":
        from services.order_id_flow import run_order_id_ai_flow

        oid_result = run_order_id_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
        )
        if oid_result.handled and oid_result.reply_html:
            reset_context(ctx)
            return send_reply(oid_result.reply_html, lang)
        if oid_result.intent == "order":
            ctx["order_id"] = oid_result.order_id or ctx.get("order_id")
            ctx["awaiting"] = "order_id" if oid_result.needs_order_id else None
            ctx["last"] = "order"
            if oid_result.reply_html:
                reset_context(ctx)
                return send_reply(oid_result.reply_html, lang)

    # Order list: AI reads API KB → purchase-history API (any phrasing)
    order_flow_delegate = None
    order_ai_intent_override = None
    if route_decision.handler == "wishlist_api" or route_decision.intent == "wishlist":
        log_reasoning("AI route wishlist — wishlists API.")
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
        return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)

    if route_decision.handler == "order_ai_flow":
        from services.order_history_flow import run_order_ai_flow

        of_result = run_order_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ai_route=ai_route_data,
        )
        if of_result.handled and of_result.reply_html:
            reset_context(ctx)
            if of_result.intent in ("order_history", "general"):
                ctx["last"] = "order_history"
                ctx.setdefault("data", {})["topic_mode"] = (
                    "order_history_howto" if of_result.intent == "general" else "order_history_list"
                )
            return send_reply(of_result.reply_html, lang)
        if of_result.intent == "order":
            order_flow_delegate = of_result
            log_reasoning("Order AI flow delegated to single-order tracking pipeline.")
        elif not of_result.handled:
            order_ai_intent_override = of_result.intent
            comb_of = f"{original_msg} {msg_en}"
            if _text_has_order_placement_intent(comb_of):
                log_reasoning("Order flow incomplete — user wants how to place order (KB).")
                reset_context(ctx)
                return send_reply(
                    _localized_sysmsg("order_placement_help", original_msg, reply_lang=lang)
                    or sysmsg("order_placement_help")
                    or "",
                    lang,
                )
            if _text_asks_how_to_view_order_history(comb_of) or _user_clarifies_process_not_order_list(
                comb_of
            ):
                log_reasoning("Order flow incomplete — user wants history how-to in app (KB).")
                reset_context(ctx)
                return send_reply(
                    _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                    or sysmsg("order_history_help")
                    or "",
                    lang,
                )
            if route_decision.intent == "order_history":
                log_reasoning(
                    "Order flow incomplete but router=order_history — direct purchase-history API."
                )
                reset_context(ctx)
                return send_reply(
                    format_purchase_history_reply(user_id, page=1, append_only=False), lang
                )

    intent = route_decision.intent if route_decision.intent else "general"
    if order_ai_intent_override:
        intent = order_ai_intent_override
    search_query = route_decision.search_query or ""
    is_welfog = route_decision.is_welfog_related
    ai_response_text = ""
    ai_data = {}
    extracted_id = None
    category_browse_id = None

    if (
        not _looks_like_browse_all_categories_message(comb_ph_early)
        and _text_requests_category_product_browse(comb_ph_early)
    ):
        category_browse_id = get_category_id_from_text(comb_ph_early, ctx=ctx)
        if category_browse_id:
            log_reasoning(f"Category product browse pre-route (category_id={category_browse_id}).")
            intent = "product"
            search_query = ""
            ai_data = {"intent": "product", "is_welfog_related": True, "search_query": ""}
            ctx.setdefault("data", {})
            ctx["data"]["selected_category_id"] = category_browse_id
            ctx["data"]["selected_color"] = _normalize_color(msg_en)
            ctx["awaiting"] = None

    _router_locked_order_history = route_decision.intent == "order_history" or (
        route_decision.handler == "order_ai_flow"
        and (ai_route_data or {}).get("intent") == "order_history"
    )
    skip_product_id_early = False
    if not _router_locked_order_history:
        comb_track_guard = f"{original_msg} {msg_en}"
        oid_track_guard = extract_order_id(comb_track_guard, conv_for_llm) or extract_latest_order_id_from_user_conversation(
            conv_for_llm, original_msg
        )
        skip_product_id_early = bool(
            oid_track_guard
            and (
                _conversation_in_order_tracking_flow(conv_for_llm)
                or _conversation_bot_offered_order_id_or_tracking(conv_for_llm)
                or _text_is_order_tracking_intent(comb_track_guard)
                or should_attempt_live_order_api_reply(
                    original_msg, msg_en, conv_for_llm, ai_route=ai_route_data
                )
            )
        )
        if not skip_product_id_early and (
            _text_is_product_id_lookup_context(comb_ph_early) or extract_product_id(comb_ph_early)
        ):
            pid_early = extract_product_id(comb_ph_early)
            log_reasoning(f"Direct product-id lookup (pro_id={pid_early}).")
            intent = "product"
            search_query = f"pro_id {pid_early}" if pid_early else ""
            ai_data = {"intent": "product", "is_welfog_related": True, "search_query": search_query}
            ctx.setdefault("data", {})
            if pid_early:
                ctx["data"]["lookup_pro_id"] = pid_early
            ctx["order_id"] = None
            ctx["awaiting"] = None

    # 1) STRICT STATE LOCKS (awaiting loops)
    run_normal_flow = _router_locked_order_history or (
        not skip_product_id_early
        and not (_text_is_product_id_lookup_context(comb_ph_early) or extract_product_id(comb_ph_early))
    )
    if ctx.get("awaiting") == "order_id":
        run_normal_flow = False
        comb_await = f"{original_msg} {msg_en}".lower()

        if should_send_warm_greeting_reply(original_msg, msg_en, conversation_context=conv_for_llm):
            log_reasoning("Greeting while awaiting order-id — reset state.")
            reset_context(ctx)
            warm_html = build_warm_conversation_reply(original_msg, msg_en, reply_lang=lang)
            return send_reply(warm_html, lang)

        if message_asks_my_welfog_purchases(f"{original_msg} {msg_en}") or _text_asks_order_history(
            f"{original_msg} {msg_en}"
        ):
            log_reasoning("Awaiting order-id released — user asked for order history.")
            ctx["awaiting"] = None
            ctx["last"] = "order"
            return send_reply(format_purchase_history_reply(user_id, page=1, append_only=False), lang)
        if _text_asks_wishlist(f"{original_msg} {msg_en}"):
            log_reasoning("Awaiting order-id released — user asked for wishlist.")
            ctx["awaiting"] = None
            reset_context(ctx)
            return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)
        if original_msg.lower() in ["cancel", "stop", "exit", "no"]:
            log_reasoning("User cancelled order-id collection state.")
            reset_context(ctx)
            return send_reply(sysmsg("cancelled"), lang)

        if _text_is_order_id_help_request(original_msg) or _text_is_order_id_help_request(msg_en):
            guidance = sysmsg("order_id_help") or (
                "Your Order ID appears in the confirmation email, SMS, or on your Welfog account orders page. "
                "Please share it when you're ready, or type 'cancel' to ask something else."
            )
            return send_reply(guidance, lang)

        if _text_is_tracking_howto_request(comb_await) or (
            _conversation_bot_offered_order_id_or_tracking(conv_for_llm)
            and re.search(r"\b(track|tracking)\b", comb_await)
            and not _text_needs_order_id_for_tracking(comb_await)
        ):
            log_reasoning("Awaiting order-id: user asked how to track; serving tracking steps.")
            ctx["awaiting"] = None
            return send_reply(_tracking_help_reply(original_msg, lang), lang)

        extracted_id = resolve_order_id_for_tracking(
            original_msg.strip() or msg_en.strip(),
            conv_for_llm,
            bot_awaiting_order_id=True,
        )
        if extracted_id:
            log_reasoning("Awaiting order-id state resolved with a valid ID.")
            try:
                from services.order_id_handoff_fast_path import (
                    _fetch_details_handoff_reply,
                    resolve_bare_order_id_handoff_goal,
                )
                from services.chat_flow_telemetry import log_order_dispatch

                goal = resolve_bare_order_id_handoff_goal(
                    ctx,
                    conv_for_llm,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    reply_lang=lang,
                )
                if not goal:
                    log_reasoning(
                        "Awaiting order-id: no locked goal — ask customer to restate intent."
                    )
                    return send_reply(
                        _localized_sysmsg("ask_order_id_generic", original_msg, reply_lang=lang),
                        lang,
                    )
                ctx["order_id"] = extracted_id
                ctx["awaiting"] = None
                if goal in ("order_invoice", "order_details", "payment"):
                    reply_html = _fetch_details_handoff_reply(
                        "order_details" if goal == "payment" else goal,
                        extracted_id,
                        user_id,
                        original_msg,
                        lang,
                    )
                elif goal == "refund_status":
                    from services.refund_status_flow import _fetch_and_format_refund_status

                    reply_html = _fetch_and_format_refund_status(
                        extracted_id,
                        user_id,
                        original_msg,
                        lang,
                        source="awaiting_order_id_handoff",
                    )
                else:
                    reply_html = _reply_for_live_order_id_lookup(
                        "order", extracted_id, user_id, original_msg, lang
                    )
                ctx["last"] = (
                    "refund"
                    if goal == "refund_status"
                    else "invoice"
                    if goal == "order_invoice"
                    else "order"
                )
                ctx.setdefault("data", {})
                ctx["data"].pop("pending_action", None)
                ctx["data"]["topic_mode"] = f"order_{goal}"
                log_order_dispatch(
                    detected_intent=goal,
                    pending_action=goal,
                    order_id_found=extracted_id,
                    selected_tool=(
                        "order_details_api"
                        if goal in ("order_invoice", "order_details")
                        else "refund_status_api"
                        if goal == "refund_status"
                        else "order_tracking_api"
                    ),
                    api_called=True,
                )
                return send_reply(reply_html, lang)
            except ImportError:
                intent = ctx.get("last") or "order"
                ctx["order_id"] = extracted_id
                ctx["awaiting"] = None
                ai_data = {
                    "intent": intent,
                    "is_welfog_related": True,
                    "needs_order_id": False,
                }
        elif _should_release_order_id_awaiting_for_routing(comb_await):
            log_reasoning("Awaiting order-id: new topic — releasing lock for full agent routing.")
            ctx["awaiting"] = None
            run_normal_flow = True
        elif turn_is_catalog_product_lookup(original_msg, msg_en):
            log_reasoning("Awaiting order-id: catalog product browse — releasing lock.")
            ctx["awaiting"] = None
            run_normal_flow = True
        elif _is_low_information_turn_for_task_routing(original_msg, msg_en, ctx):
            log_reasoning("Awaiting order-id released on low-information turn; asking clear intent.")
            ctx["awaiting"] = None
            reset_context(ctx)
            return send_reply(_clarify_request_scope_reply(original_msg, lang), lang)
        else:
            log_reasoning("Awaiting order-id state: user message not a valid order ID.")
            return send_reply(_localized_sysmsg("ask_order_id_generic", original_msg, reply_lang=lang), lang)

    if ctx.get("awaiting") == "category_select":
        run_normal_flow = False
        if original_msg.lower() in ["cancel", "stop", "exit", "no"]:
            reset_context(ctx)
            return send_reply(sysmsg("cancelled"), lang)

        # If user asks a fresh/general question while awaiting category id,
        # unlock this state and continue with normal routing.
        comb_cat_await = f"{original_msg} {msg_en}".lower()
        if (
            _looks_like_browse_all_categories_message(comb_cat_await)
            or "department" in msg_en
            or "staff" in msg_en
        ):
            ctx["awaiting"] = None

        # Try parse category id (e.g., "16", "id 16", "category 16", "electronics", "shoes")
        cat_id = get_category_id_from_text(comb_cat_await, ctx=ctx)
        if not cat_id:
            m = re.search(r"\b(\d{1,5})\b", msg_en)
            if m:
                cat_id = m.group(1)

        color = _normalize_color(msg_en)
        if ctx.get("awaiting") == "category_select" and not cat_id:
            return send_reply(sysmsg("ask_category_select"), lang)

        # Main department (e.g. Men Fashion id 10) → show inner subcategories unless user asked for products
        if (
            cat_id
            and is_top_level_main_category(cat_id, ctx)
            and main_category_has_inner_children(cat_id, ctx)
            and not _text_requests_category_product_browse(comb_cat_await)
        ):
            inner_html = format_inner_categories_reply(cat_id, ctx)
            if inner_html:
                ctx["awaiting"] = "category_select"
                return send_reply(inner_html, lang)

        intent = "product"
        search_query = ""  # category browse
        ai_data = {"intent": "product", "is_welfog_related": True, "search_query": ""}
        ctx["awaiting"] = None
        ctx["data"]["selected_category_id"] = cat_id
        ctx["data"]["selected_color"] = color

    follow_early = classify_conversation_followup(original_msg, msg_en, conv_for_llm, ctx)
    if follow_early == "explore_menu":
        ctx.setdefault("data", {})["pending_offer"] = "explore_menu"
        return send_reply(
            _localized_sysmsg("explore_menu_reply", original_msg, reply_lang=lang)
            or sysmsg("explore_menu_reply"),
            lang,
        )
    if follow_early == "deals_link_clarify":
        return send_reply(
            _localized_sysmsg("deals_link_clarify", original_msg, reply_lang=lang)
            or sysmsg("deals_link_clarify"),
            lang,
        )
    if follow_early == "welfog_privacy_policy":
        ctx.setdefault("data", {})["pending_offer"] = None
        policy_html = format_knowledge_information_reply_from_kb(
            original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm
        )
        if policy_html:
            reset_context(ctx)
            return send_reply(policy_html, lang)
    if follow_early in ("deals", "categories"):
        follow_skip_ai = follow_early
        intent = follow_early
        is_welfog = True
        search_query = ""
        ai_data = {"intent": follow_early, "is_welfog_related": True, "search_query": ""}

    # 2. NORMAL FLOW
    if run_normal_flow and not follow_skip_ai:
        if ctx.get("data", {}).get("ai_route") and not ai_data:
            rd = ctx["data"]["ai_route"]
            intent = rd.get("intent", intent)
            is_welfog = rd.get("is_welfog_related", is_welfog)
            search_query = rd.get("search_query") or search_query
            ai_data = {
                "intent": intent,
                "is_welfog_related": is_welfog,
                "search_query": search_query,
                "needs_order_id": rd.get("needs_order_id", False),
            }
            from services.message_understanding import merge_ai_route_into_ai_data

            merge_ai_route_into_ai_data(rd, ai_data)
            log_reasoning(f"Normal flow seeded from AI-first route: intent={intent}")

        # If user directly says a category name ("electronics ke products dikhao"),
        # auto resolve category id and show products without asking for id.
        comb_browse = f"{original_msg} {msg_en}".lower()
        auto_cat_id = get_category_id_from_text(comb_browse, ctx=ctx)
        if auto_cat_id and _text_requests_category_product_browse(comb_browse):
            ctx.setdefault("data", {})
            ctx["data"]["selected_category_id"] = auto_cat_id
            ctx["data"]["selected_color"] = _normalize_color(msg_en)
            intent = "product"
            search_query = ""  # category browse
            ai_data = {"intent": "product", "is_welfog_related": True, "search_query": ""}

        # FAST GREETING CHECK — never override order thread or id correction
        comb_early = f"{original_msg} {msg_en}".lower()
        skip_greeting_fast = (
            _should_bypass_warm_greeting_fast_path(comb_early)
            or _conversation_in_order_tracking_flow(conv_for_llm)
            or _message_submits_or_corrects_order_id(original_msg)
        )
        is_greet = not skip_greeting_fast and (
            _looks_like_greeting_message(original_msg)
            or any(original_msg.lower() == g for g in GREETINGS)
        )
        is_small = not skip_greeting_fast and _looks_like_light_smalltalk(original_msg, msg_en)
        # Do not replay warm greeting when AI already classified a substantive Welfog request.
        skip_greeting_normal = bool(ctx.get("data", {}).get("ai_route")) and not (is_greet or is_small)
        if (is_greet or is_small) and not skip_greeting_normal:
            log_reasoning("Greeting or light smalltalk — AI chitchat reply (no template).")
            use_smalltalk_pool = is_small and not _looks_like_greeting_message(original_msg)
            reply_key = pick_warm_chat_reply_key(
                use_smalltalk_pool, original_msg=original_msg, reply_lang=lang
            )
            reset_context(ctx)
            pending = pending_offer_from_greeting_key(reply_key)
            if pending:
                ctx.setdefault("data", {})["pending_offer"] = pending
            return send_reply(
                build_warm_conversation_reply(original_msg, msg_en, reply_lang=lang)
                or "",
                lang,
            )

        # ================= ⚡ 1. SMART INSTANT ROUTER (skip when AI-first already decided) ⚡ =================
        fast_result = None
        if not ctx.get("data", {}).get("ai_route"):
            fast_result = smart_instant_router(original_msg, msg_en)
        
        if fast_result:
            if fast_result["action"] == "reject" or fast_result["action"] == "text":
                log_reasoning(f"Fast router action='{fast_result['action']}'")
                reset_context(ctx)
                return send_reply(fast_result["data"], lang)
                
            elif fast_result["action"] == "product":
                reset_context(ctx)
                intent = "product"
                search_query = fast_result["query"]
                is_welfog = True
                
            elif fast_result["action"] == "ask_order_id":
                log_reasoning("Fast router requires order id for order/refund/payment flow.")
                try:
                    from services.order_id_handoff_fast_path import (
                        lock_order_id_ask_from_intent_label,
                    )

                    lock_order_id_ask_from_intent_label(
                        ctx, fast_result["intent"]
                    )
                except ImportError:
                    ctx["intent"] = fast_result["intent"]
                    ctx["last"] = fast_result["intent"]
                    ctx["awaiting"] = "order_id"
                return send_reply(
                    _localized_sysmsg(
                        "ask_order_id_for_intent", original_msg, reply_lang=lang, intent=fast_result["intent"]
                    ),
                    lang,
                )
                
            elif fast_result["action"] == "direct_order_id":
                log_reasoning("Single-token message matched strict order-id pattern.")
                extracted_id = fast_result["order_id"]
                intent = ctx.get("last") or "order"
                ctx["order_id"] = extracted_id
                ctx["awaiting"] = None
                ai_data = {"intent": intent, "is_welfog_related": True, "needs_order_id": True}

            elif fast_result["action"] == "wishlist":
                comb_fast = f"{original_msg} {msg_en}"
                if message_asks_my_welfog_purchases(comb_fast) or _text_asks_order_history(comb_fast):
                    log_reasoning("Fast router wishlist skipped — user wants order history.")
                    reset_context(ctx)
                    return send_reply(
                        format_purchase_history_reply(user_id, page=1, append_only=False), lang
                    )
                log_reasoning("Fast router: wishlist list.")
                reset_context(ctx)
                return send_reply(format_wishlist_reply(user_id, page=1, append_only=False), lang)

        else:
            # ================= 🐢 2. AI BRAIN (Only for complex questions) 🐢 =================
            # kb_match = direct_kb_search(msg_en)
            # if kb_match:
            #     reset_context(ctx)
            #     return send_reply(kb_match, lang)
            
            if ctx.get("awaiting") == "order_id" and not extracted_id:
                reset_context(ctx)

            # ================= 📚 KB-FIRST PASS (no hardcoding) =================
            # If answer exists in ANY admin-added knowledge file, prefer that first.
            # Skip this for shopping/product/deals/category flows where API results are expected.
            comb_low = f"{original_msg} {msg_en}".lower()
            route_intent = (route_decision.intent or "").strip().lower()
            ai_intent = ""
            if isinstance(ai_route_data, dict):
                ai_intent = (ai_route_data.get("intent") or "").strip().lower()

            if route_intent == "seller" or ai_intent == "seller" or message_is_seller_on_welfog_request(
                comb_low
            ):
                from services.kb_service import format_seller_reply_from_kb

                seller_reply = format_seller_reply_from_kb(
                    original_msg, msg_en, reply_lang=lang, conversation_context=conv_for_llm
                )
                if seller_reply:
                    log_reasoning("Seller intent — answered from seller KB (not company about).")
                    reset_context(ctx)
                    return send_reply(seller_reply, lang)

            from utils.helpers import (
                message_asks_other_company_social_media,
                message_asks_welfog_social_media,
            )
            from services.kb_service import format_welfog_social_media_reply_from_kb
            from services.support_scope import build_other_company_social_decline

            if message_asks_other_company_social_media(
                comb_low, conversation_context=conv_for_llm
            ):
                log_reasoning(
                    "Other person/company social (early path) — decline; Welfog links only."
                )
                reset_context(ctx)
                return send_reply(
                    build_other_company_social_decline(original_msg, reply_lang=lang), lang
                )

            if (
                not _text_has_product_shopping_intent(comb_low)
                and message_asks_welfog_social_media(
                    comb_low, conversation_context=conv_for_llm
                )
            ):
                social_reply = format_welfog_social_media_reply_from_kb(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                )
                if social_reply:
                    log_reasoning("Welfog social links — deterministic KB buttons (no LLM).")
                    reset_context(ctx)
                    return send_reply(social_reply, lang)

            if (
                message_is_welfog_about_request(comb_low)
                and not _text_asks_order_history(comb_low)
                and not message_is_seller_on_welfog_request(comb_low)
                and route_intent != "seller"
                and ai_intent != "seller"
            ):
                about_reply = format_welfog_about_reply_from_kb(
                    original_msg, msg_en, reply_lang=lang
                )
                if about_reply:
                    log_reasoning(
                        f"Welfog about/company question answered from KB (company.txt), reply_lang={lang}."
                    )
                    reset_context(ctx)
                    return send_reply(about_reply, lang)

            ai_route_ch = ai_ch or (
                (ai_route_data or {}).get("data_channel") or ""
            ).strip().lower()
            kb_first_allowed = ai_route_ch == "kb" and ai_intent not in (
                "product",
                "wishlist",
                "order_history",
                "order",
                "pincode_check",
                "deals",
                "categories",
            )
            if not kb_first_allowed and not ctx.get("data", {}).get("ai_route"):
                kb_first_allowed = route_decision.source in ("kb", "kb_ai", "ai") or (
                    message_is_welfog_about_request(comb_low)
                    or message_is_knowledge_information_request(comb_low)
                    or message_needs_support_not_product(comb_low)
                    or message_is_casual_offtopic_not_shopping(comb_low)
                    or not (
                        _text_has_product_shopping_intent(comb_low)
                        or _text_has_delivery_or_order_area_intent(comb_low)
                        or any(
                            w in f" {msg_en} "
                            for w in ["deal", "deals", "offer", "offers", "discount", "today deal"]
                        )
                        or _looks_like_browse_all_categories_message(msg_en)
                        or _text_is_order_tracking_intent(comb_low)
                    )
                )
            if ai_route_ch in ("catalog", "live_api"):
                kb_first_allowed = False
            # Live API flows (order list via order_ai_flow, wishlist) skip generic KB-first
            _STRUCTURED_PROCEDURE_HANDLERS = frozenset(
                {
                    "order_history_howto_kb",
                    "welfog_social_kb",
                    "other_company_social_decline",
                    "wishlist_howto_kb",
                    "order_tracking_howto_kb",
                    "order_placement_kb",
                    "order_id_help_kb",
                    "customer_care_kb",
                    "support_escalation_kb",
                }
            )
            if (
                route_decision.handler in ("order_ai_flow", "product_ai_flow")
                or route_decision.handler in _STRUCTURED_PROCEDURE_HANDLERS
                or message_asks_my_welfog_purchases(comb_low)
                or _text_asks_order_history(comb_low)
                or _text_asks_wishlist(comb_low)
                or _text_asks_how_to_view_wishlist(comb_low)
                or _user_asks_order_history_navigation_help(comb_low)
            ):
                kb_first_allowed = False

            if should_use_warm_conversational_reply(
                original_msg, msg_en, conv_for_llm, ai_route_data
            ):
                kb_first_allowed = False
                warm_fb = build_warm_feedback_reply(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                )
                if warm_fb and route_decision.handler in (
                    "dynamic_kb",
                    "knowledge_topic_kb",
                    "ai_route_and_answer",
                    "warm_feedback",
                ):
                    log_reasoning("KB-first blocked — conversational thanks/praise warm reply.")
                    reset_context(ctx)
                    return send_reply(warm_fb, lang)
            # Phone/email straight from admin knowledge files (no LLM paraphrase = no wrong grievance inbox).
            if kb_first_allowed and _text_asks_customer_care_contact(comb_low):
                cc_reply = format_customer_care_reply_from_kb(original_msg, msg_en)
                if cc_reply:
                    log_reasoning("Customer-care reply built from KB file contents (deterministic).")
                    reset_context(ctx)
                    return send_reply(cc_reply, lang)

            hit = None
            if kb_first_allowed:
                support_keys = get_support_contact_kb_keys()
                if _text_asks_customer_care_contact(comb_low) and support_keys:
                    hit = best_kb_hit(retrieval_query, keys=support_keys, min_score=0.18)
                    if not hit:
                        hit = keyword_kb_hit(retrieval_query, keys=support_keys, min_hits=1)
                if not hit:
                    kb_keys_route = route_decision.kb_keys or get_customer_kb_keys()
                    min_sc = route_decision.kb_min_score if route_decision.source == "kb_ai" else 0.22
                    hit = best_kb_hit(retrieval_query, keys=kb_keys_route, min_score=min_sc)
                if not hit:
                    hit = keyword_kb_hit(
                        retrieval_query, keys=route_decision.kb_keys or get_customer_kb_keys(), min_hits=2
                    )
                if not hit and route_decision.kb_hit:
                    hit = route_decision.kb_hit
                if hit:
                    score_str = f"{hit['score']:.2f}" if isinstance(hit.get("score"), (int, float)) else str(hit.get("score"))
                    log_reasoning(f"KB-first matched source={hit['source']} score={score_str}")
                    ai_route_plan_src = ai_route_data or ctx.get("data", {}).get("ai_route") or {}
                    from services.semantic_answer_plan import (
                        build_semantic_answer_plan,
                        try_semantic_grounded_reply,
                    )

                    plan = build_semantic_answer_plan(
                        ai_route_plan_src, handler=route_decision.handler or ""
                    )
                    if plan.use_ai_synthesis and plan.answer_strategy not in (
                        "kb_only",
                        "live_api_only",
                        "catalog_only",
                        "structured_handler",
                    ):
                        grounded = try_semantic_grounded_reply(
                            original_msg,
                            msg_en,
                            ai_route_plan_src,
                            conversation_context=conv_for_llm,
                            reply_lang=lang,
                            handler=route_decision.handler or "",
                        )
                        if grounded:
                            log_reasoning(
                                f"KB+AI grounded reply (strategy={plan.answer_strategy}, user's language)."
                            )
                            reset_context(ctx)
                            return send_reply(grounded, lang)
                    # Prefer deterministic KB formatter when strategy is kb_only / structured.
                    if plan.answer_strategy in ("kb_only", "structured_handler"):
                        kb_det = format_dynamic_kb_answer(
                            original_msg,
                            msg_en,
                            reply_lang=lang,
                            conversation_context=conv_for_llm,
                            suggested_keys=list(route_decision.kb_keys or [hit.get("source") or ""]),
                        )
                        if kb_det:
                            log_reasoning("KB-first deterministic reply (skip AI paraphrase).")
                            reset_context(ctx)
                            return send_reply(kb_det, lang)
                    # Legacy deterministic path when no AI synthesis requested.
                    kb_det = format_dynamic_kb_answer(
                        original_msg,
                        msg_en,
                        reply_lang=lang,
                        conversation_context=conv_for_llm,
                        suggested_keys=list(route_decision.kb_keys or [hit.get("source") or ""]),
                    )
                    if kb_det:
                        log_reasoning("KB-first deterministic reply (skip AI paraphrase).")
                        reset_context(ctx)
                        return send_reply(kb_det, lang)
                    # Use Groq once to turn the chunk into a proper answer (and handle follow-ups).
                    if _text_asks_customer_care_contact(comb_low) and support_keys:
                        support_blob = read_concatenated_kb_file_contents(support_keys)
                        if support_blob.strip():
                            kb_context = (
                                "AUTHORITATIVE SUPPORT/CONTACT KNOWLEDGE (full files; use ONLY these for phone/email):\n"
                                f"{support_blob}\n\n---\nRetrieval excerpt:\n"
                                f"[source={hit['source']} score={score_str}] {hit['chunk']}"
                            )
                        else:
                            kb_context = f"[source={hit['source']} score={score_str}] {hit['chunk']}"
                    else:
                        kb_context = f"[source={hit['source']} score={score_str}] {hit['chunk']}"
                    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm, reply_lang=lang) or {}
                    ai_data.setdefault("intent", "general")
                    ai_data.setdefault("is_welfog_related", True)
                    intent = ai_data.get("intent", "general")
                    is_welfog = ai_data.get("is_welfog_related", True)
                    search_query = ai_data.get("search_query") or msg_en
                    ai_response_text = ai_data.get("response", "")
                    try:
                        from services.knowledge_grounding_validator import ground_kb_llm_response

                        ai_response_text = ground_kb_llm_response(
                            ai_response_text,
                            kb_context=kb_context,
                            original_msg=original_msg,
                            msg_en=msg_en,
                        )
                    except ImportError:
                        pass
                    # Final merge happens after cache branch; KB-first answers must not be overwritten by a second Groq call.

            # AI-first: do not replay cached routing/answers for different user questions
            computed_fresh = False
            if ctx.get("data", {}).get("ai_route") and not (kb_first_allowed and hit):
                route_data = ctx["data"]["ai_route"]
                intent = route_data.get("intent", intent)
                is_welfog = route_data.get("is_welfog_related", is_welfog)
                search_query = route_data.get("search_query") or search_query
                ai_data = {
                    "intent": intent,
                    "is_welfog_related": is_welfog,
                    "search_query": search_query,
                    "needs_order_id": route_data.get("needs_order_id", False),
                }
                computed_fresh = True
                log_reasoning("Using AI-first route data (no response cache).")
            elif kb_first_allowed and hit:
                computed_fresh = True
            elif route_decision.source in ("api", "ai_order", "ai_product") and route_decision.intent:
                computed_fresh = True
                intent = route_decision.intent
                is_welfog = route_decision.is_welfog_related
                search_query = route_decision.search_query or search_query
                ai_data = {
                    "intent": intent,
                    "is_welfog_related": is_welfog,
                    "search_query": search_query,
                }
                if ctx.get("data", {}).get("ai_route"):
                    ai_data["needs_order_id"] = ctx["data"]["ai_route"].get("needs_order_id", False)
                log_reasoning(f"Answer router set API intent={intent} (AI-first, no cache).")
            else:
                computed_fresh = True
                route_data = ctx.get("data", {}).get("ai_route")
                if not route_data:
                    try:
                        from services.chat_flow_telemetry import get_stored_ai_route, skip_step

                        route_data = get_stored_ai_route()
                        if route_data:
                            skip_step("ai_brain_route_fallback", "reuse stored route")
                    except ImportError:
                        pass
                if not route_data:
                    log_reasoning("Answer router: AI understanding (reuse cached brain route).")
                    try:
                        from services.chat_flow_telemetry import guard_duplicate_brain_route

                        route_data = guard_duplicate_brain_route("ai_brain_route_fallback")
                    except ImportError:
                        route_data = None
                    if route_data is None:
                        route_data = ai_brain_route(original_msg, conv_for_llm, reply_lang=lang)
                if not route_data:
                    fallback_text = sysmsg("server_busy")
                    return send_reply(fallback_text, lang)

                intent_route = route_data.get("intent", "general")
                is_welfog = route_data.get("is_welfog_related", True)
                search_query = route_data.get("search_query", "")
                extracted_id = route_data.get("extracted_pincode", "")

                log_reasoning(f"AI routing decision => intent={intent_route}, is_welfog_related={is_welfog}")

                # Hard guard: off-topic only when authoritative KB route is not locked.
                comb_turn = f"{original_msg} {msg_en}".lower()
                try:
                    from services.chat_flow_telemetry import should_skip_post_kb_ood_guard

                    kb_locked = should_skip_post_kb_ood_guard("legacy_answer_router")
                except ImportError:
                    kb_locked = (route_data.get("data_channel") or "").strip().lower() == "kb"
                if not kb_locked and (not is_welfog or intent_route == "out_of_domain"):
                    from services.off_topic_reply import build_off_topic_polite_reply

                    reset_context(ctx)
                    return send_reply(
                        build_off_topic_polite_reply(
                            original_msg,
                            msg_en,
                            reply_lang=lang,
                            ai_route=route_data,
                            conversation_context=conv_for_llm,
                        ),
                        lang,
                    )
                from services.support_scope import resolve_support_request_scope

                scope_now = resolve_support_request_scope(original_msg, msg_en, conversation_context=conv_for_llm)
                if not kb_locked and scope_now == "external":
                    from services.off_topic_reply import build_off_topic_polite_reply

                    reset_context(ctx)
                    return send_reply(
                        build_off_topic_polite_reply(
                            original_msg,
                            msg_en,
                            reply_lang=lang,
                            ai_route=route_data,
                            conversation_context=conv_for_llm,
                        ),
                        lang,
                    )

                # Handle special intents that don't need a second AI call
                if intent_route == "order_history":
                    ai_data = {"intent": "order_history", "is_welfog_related": is_welfog}
                elif intent_route == "wishlist":
                    ai_data = {"intent": "wishlist", "is_welfog_related": is_welfog}
                elif (route_data.get("scope_reply") or "").strip() and (
                    (route_data.get("conversation_scope") or "").strip().lower()
                    in ("general_chitchat", "out_of_domain", "harm_sensitive")
                ):
                    from services.conversation_scope import finalize_scope_reply_html

                    ai_data = {
                        "intent": intent_route,
                        "is_welfog_related": is_welfog,
                        "response": finalize_scope_reply_html(
                            route_data.get("scope_reply") or "",
                            original_msg,
                            reply_lang=lang,
                        ),
                    }
                    log_reasoning(
                        "Legacy stack — reuse brain scope_reply (skip ai_brain_answer)."
                    )
                elif (route_data.get("route_handler") or "").strip() and (
                    route_data.get("data_channel") or ""
                ).strip().lower() in ("live_api", "catalog"):
                    ai_data = {
                        "intent": intent_route,
                        "is_welfog_related": is_welfog,
                        "search_query": search_query,
                        "needs_order_id": route_data.get("needs_order_id", False),
                        "_ai_routed": True,
                    }
                    log_reasoning(
                        "Legacy stack — reuse brain route_handler (skip ai_brain_answer)."
                    )
                else:
                    # Get KB context for grounding
                    kb_keys = route_data.get("kb_keys") or []
                    # Always include API playbook for shopping/deals/category flows
                    if intent_route in ["product", "deals", "categories", "category_feed"] and "welfog_api" not in kb_keys:
                        kb_keys = list(kb_keys) + ["welfog_api"]
                    # For non-shopping informational intents, don't ground on internal playbook files.
                    if intent_route in ["general", "seller", "refund", "payment"] and kb_keys:
                        kb_keys = [k for k in kb_keys if k not in INTERNAL_KB_KEYS] or get_customer_kb_keys()
                    # If model didn't pick any KB keys for general info, search ALL customer KB
                    if intent_route == "general":
                        kb_keys = get_customer_kb_keys()

                    kb_context = get_knowledge_context(
                        retrieval_query, keys=kb_keys, top_k=6, min_score=0.10
                    )
                    
                    if _text_asks_customer_care_contact(comb_low) or (
                        message_needs_human_support_escalation(comb_low)
                        and intent_route == "general"
                    ):
                        sb = read_concatenated_kb_file_contents(get_support_contact_kb_keys())
                        if sb.strip():
                            kb_context = (
                                "AUTHORITATIVE SUPPORT/CONTACT KNOWLEDGE FILES (full text):\n"
                                f"{sb}\n\n---\nOTHER GROUNDING:\n" + (kb_context or "")
                            )
                    
                    # Step 2: Get actual response from AI using the routing decision + KB
                    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm, reply_lang=lang) or {}
                    ai_data.setdefault("intent", intent_route)
                    ai_data.setdefault("is_welfog_related", is_welfog)
                    
                    # Copy routing fields if response didn't include them
                    if "search_query" not in ai_data:
                        ai_data["search_query"] = search_query
                    
                    intent = ai_data.get("intent", intent_route)
                    is_welfog = ai_data.get("is_welfog_related", is_welfog)
                    ai_response_text = ai_data.get("response", "")
                    try:
                        from services.knowledge_grounding_validator import ground_kb_llm_response

                        ai_response_text = ground_kb_llm_response(
                            ai_response_text,
                            kb_context=kb_context,
                            original_msg=original_msg,
                            msg_en=msg_en,
                        )
                    except ImportError:
                        pass

            # Note: ai_data already normalized above
            
            if not ai_data:
                if _user_asks_order_history_navigation_help(f"{original_msg} {msg_en}"):
                    reset_context(ctx)
                    return send_reply(
                        _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                        or sysmsg("order_history_help")
                        or "",
                        lang,
                    )
                grounded_fallback = direct_kb_search(retrieval_query, keys=get_customer_kb_keys(), min_score=0.34)
                if grounded_fallback:
                    log_reasoning("Groq unavailable; serving direct KB fallback.")
                    return send_reply(grounded_fallback, lang)
                fallback_text = sysmsg("server_busy")
                return send_reply(fallback_text, lang)

            if ctx.get("data", {}).get("ai_route"):
                from services.message_understanding import merge_ai_route_into_ai_data

                merge_ai_route_into_ai_data(ctx["data"]["ai_route"], ai_data)

            apply_hinglish_product_fixes(original_msg, msg_en, ai_data)
            apply_conversation_followup_fixes(
                original_msg, msg_en, ai_data, conv_for_llm, ctx
            )
            apply_category_product_route_fixes(original_msg, msg_en, ai_data, ctx=ctx)
            apply_product_id_vs_order_fixes(original_msg, msg_en, ai_data, ctx=ctx)
            apply_order_tracking_fixes(original_msg, msg_en, ai_data)
            _merge_embedded_identifiers_from_message(original_msg, msg_en, ai_data, conv_for_llm)
            # Response cache disabled — each message gets fresh AI understanding
            
            if not ai_data.get("_ai_routed"):
                if ai_data.get("intent") == "order" and _text_has_order_placement_intent(comb_low):
                    ai_data["intent"] = "general"

            if not ai_data.get("_ai_routed") and ai_data.get("intent") == "order" and not _text_needs_order_id_for_tracking(comb_low):
                ai_data["needs_order_id"] = False
            if ai_data.get("intent") in ["refund", "payment"] and not _text_needs_order_id_for_refund_or_payment(comb_low):
                oid_turn = resolve_order_id_for_tracking(
                    f"{original_msg} {msg_en}",
                    conv_for_llm,
                    bot_awaiting_order_id=ctx.get("awaiting") == "order_id",
                )
                if not oid_turn and not _conversation_bot_offered_order_id_or_tracking(conv_for_llm):
                    ai_data["needs_order_id"] = False
            intent = ai_data.get("intent", "general")
            is_welfog = ai_data.get("is_welfog_related", True)
            search_query = ai_data.get("search_query") or msg_en 
            ai_response_text = ai_data.get("response", "")
            try:
                from services.knowledge_grounding_validator import apply_final_kb_fact_contract

                ai_response_text = apply_final_kb_fact_contract(
                    ai_response_text,
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
            except ImportError:
                pass

            # Strict anti-stale guard:
            # If the user asks a factual "who/owner/founder/partner" style query,
            # answer only when CURRENT KB has evidence. This prevents old memory answers
            # after file update/delete from admin panel.
            factual_query = _looks_like_factual_identity_query(f"{original_msg} {msg_en}")
            if intent in ["general", "seller", "refund", "payment"] and factual_query:
                kb_now_hit = best_kb_hit(retrieval_query, keys=get_customer_kb_keys(), min_score=0.31)
                if not kb_now_hit:
                    ai_response_text = (
                        "I don't have confirmed official details for that Welfog question right now. "
                        "Please ask again with a bit more detail — for example which team, policy, or feature you mean."
                    )
                    log_reasoning("Welfog factual query: no supporting evidence in current KB snapshot.")

            # If message or search text names a category, switch to category browse (empty name filter).
            if intent == "product":
                typed_cat_id = get_category_id_from_text(
                    f"{original_msg} {msg_en} {search_query}", ctx=ctx
                )
                if typed_cat_id:
                    ctx.setdefault("data", {})
                    ctx["data"]["selected_category_id"] = typed_cat_id
                    search_query = ""
                    ai_data["search_query"] = ""

            # Domain Control for AI responses (never trust model paraphrase for off-topic — avoids loopholes)
            if not is_welfog or intent == "out_of_domain":
                from services.product_search_flow import (
                    message_eligible_for_product_ai_flow,
                    run_product_search_ai_flow,
                )

                comb_off = f"{original_msg} {msg_en}".lower()
                from services.query_intent_classifier import query_intent_allows_catalog

                if (
                    query_intent_allows_catalog(ctx)
                    and message_eligible_for_product_ai_flow(
                        comb_off,
                        msg_en,
                        original_msg,
                        ai_route=ai_data
                        if isinstance(ai_data, dict)
                        else ctx.get("data", {}).get("ai_route"),
                        ctx=ctx,
                    )
                ):
                    ps_fix = run_product_search_ai_flow(
                        original_msg,
                        msg_en,
                        user_id,
                        conversation_context=conv_for_llm,
                        reply_lang=lang,
                        ctx=ctx,
                    )
                    if ps_fix.handled and ps_fix.reply_html:
                        reset_context(ctx)
                        return send_reply(ps_fix.reply_html, lang)
                if should_send_warm_greeting_reply(
                    original_msg, msg_en, conversation_context=conv_for_llm
                ):
                    reset_context(ctx)
                    return send_reply(
                        build_warm_conversation_reply(original_msg, msg_en, reply_lang=lang)
                        or "",
                        lang,
                    )
                from services.off_topic_reply import build_off_topic_polite_reply

                reset_context(ctx)
                return send_reply(
                    build_off_topic_polite_reply(
                        original_msg,
                        msg_en,
                        reply_lang=lang,
                        ai_route=ai_data if isinstance(ai_data, dict) else None,
                        conversation_context=conv_for_llm,
                    ),
                    lang,
                )

    comb_exec = f"{original_msg} {msg_en}".lower()
    if order_flow_delegate and order_flow_delegate.intent == "order":
        intent = "order"
        is_welfog = True
        ai_data = dict(ai_data or {})
        ai_data["intent"] = "order"
        ai_data["is_welfog_related"] = True
        ai_data["needs_order_id"] = order_flow_delegate.needs_order_id
    elif intent == "order_history":
        if _text_is_product_id_lookup_context(comb_exec) or extract_product_id(comb_exec):
            log_reasoning("order_history misroute corrected to product (product-id context).")
            intent = "product"
            ai_data = dict(ai_data or {})
            ai_data["intent"] = "product"
        oid = extract_order_id(original_msg, conv_for_llm) or extract_order_id(msg_en, conv_for_llm)
        if intent != "product" and oid and (
            _text_suggests_single_order_status_lookup(comb_exec)
            or _text_is_live_order_lookup_intent(comb_exec, conv_for_llm)
            or _conversation_bot_offered_order_id_or_tracking(conv_for_llm)
        ):
            log_reasoning("order_history misroute corrected to order (inline id + tracking).")
            intent = "order"
            is_welfog = True
            ai_data = dict(ai_data or {})
            ai_data["intent"] = "order"
            ai_data["needs_order_id"] = True
            ctx["order_id"] = oid
        elif _text_is_order_tracking_intent(comb_exec):
            log_reasoning("order_history misroute corrected to order tracking.")
            intent = "order"
            is_welfog = True
            ai_data = dict(ai_data or {})
            ai_data["intent"] = "order"
            ai_data["needs_order_id"] = _text_needs_order_id_for_tracking(comb_exec)

    comb_low = f"{original_msg} {msg_en}".lower()

    if intent == "categories" and not _looks_like_browse_all_categories_message(comb_low):
        if _text_requests_category_product_browse(comb_low):
            named_cat = get_category_id_from_text(comb_low, ctx=ctx)
            if named_cat:
                log_reasoning("categories misroute -> product (named category in message).")
                intent = "product"
                search_query = ""
                ctx.setdefault("data", {})
                ctx["data"]["selected_category_id"] = named_cat
                ctx["awaiting"] = None
                if isinstance(ai_data, dict):
                    ai_data["intent"] = "product"
                    ai_data["search_query"] = ""

    if intent == "wishlist" and (
        message_asks_my_welfog_purchases(comb_exec) or _text_asks_order_history(comb_exec)
    ) and not message_is_wishlist_like_request(comb_exec):
        log_reasoning("wishlist misroute corrected to order_history (mangaya/mangaye).")
        intent = "order_history"
        if isinstance(ai_data, dict):
            ai_data["intent"] = "order_history"
    elif intent == "wishlist" and _text_asks_how_to_view_wishlist(comb_exec):
        log_reasoning("wishlist misroute corrected to how-to (app steps).")
        intent = "general"
        if isinstance(ai_data, dict):
            ai_data["intent"] = "general"
    # ================= INTENT EXECUTION =================
    if intent == "order_id":
        from services.welfog_api import format_order_ids_reply

        response_text = format_order_ids_reply(user_id, page=1)
        reset_context(ctx)
    elif intent == "order_history":
        if _text_is_order_id_help_request(comb_low) or _text_asks_how_to_view_order_history(comb_low):
            log_reasoning("order_history intent blocked — user asked how-to, not list.")
            response_text = (
                _localized_sysmsg("order_id_help", original_msg, reply_lang=lang)
                if _text_is_order_id_help_request(comb_low)
                else (
                    _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                    or sysmsg("order_history_help")
                )
            )
        else:
            response_text = format_purchase_history_reply(user_id, page=1, append_only=False)
        reset_context(ctx)
        ctx["last"] = "order_history"
        ctx.setdefault("data", {})["topic_mode"] = (
            "order_history_howto"
            if (_text_is_order_id_help_request(comb_low) or _text_asks_how_to_view_order_history(comb_low))
            else "order_history_list"
        )
    elif intent == "wishlist":
        if not _text_asks_wishlist(comb_exec) and not message_is_wishlist_like_request(comb_exec):
            if route_decision.intent == "wishlist" or (ai_route_data or {}).get("intent") == "wishlist":
                log_reasoning("Wishlist: trust AI route despite keyword gate.")
            else:
                log_reasoning("wishlist intent blocked — serving clarify.")
                reset_context(ctx)
                return send_reply(
                    _localized_sysmsg("wishlist_clarify", original_msg, reply_lang=lang)
                    or "Tell me if you want your saved wishlist — e.g. <b>meri wishlist dikhao</b>.",
                    lang,
                )
        response_text = format_wishlist_reply(user_id, page=1, append_only=False)
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_list"
    elif intent == "general" and _text_asks_how_to_view_wishlist(comb_low):
        response_text = (
            _localized_sysmsg("wishlist_help", original_msg, reply_lang=lang)
            or sysmsg("wishlist_help")
        )
        reset_context(ctx)
        ctx["last"] = "wishlist"
        ctx.setdefault("data", {})["topic_mode"] = "wishlist_howto"
    elif intent == "product":
        if not (isinstance(ai_data, dict) and ai_data.get("_ai_routed")) and (
            message_is_welfog_about_request(comb_low) or message_is_knowledge_information_request(comb_low)
        ):
            kb_reply = (
                format_welfog_about_reply_from_kb(original_msg, msg_en, reply_lang=lang)
                if message_is_welfog_about_request(comb_low)
                else format_knowledge_information_reply_from_kb(
                    original_msg, msg_en, reply_lang=lang
                )
            )
            if kb_reply:
                log_reasoning(
                    f"Product intent overridden -> KB info (privacy/terms/about), reply_lang={lang}."
                )
                reset_context(ctx)
                return send_reply(kb_reply, lang)
            intent = "general"
            search_query = ""
            if isinstance(ai_data, dict):
                ai_data["intent"] = "general"
                ai_data["search_query"] = ""
        elif message_is_casual_offtopic_not_shopping(comb_low):
            from services.off_topic_reply import build_off_topic_polite_reply

            log_reasoning("Product intent overridden: casual/off-topic (not shopping).")
            reset_context(ctx)
            return send_reply(
                build_off_topic_polite_reply(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                    prefer_llm=False,
                ),
                lang,
            )
        pq_llm = None
        # If user selected a category previously, use it automatically
        selected_cat = ctx.get("data", {}).get("selected_category_id")
        if not selected_cat:
            selected_cat = get_category_id_from_text(f"{original_msg} {msg_en}", ctx=ctx)
            if selected_cat:
                ctx.setdefault("data", {})["selected_category_id"] = selected_cat
        selected_color = ctx.get("data", {}).get("selected_color")
        # Also detect color from current query (overrides previous)
        detected_color = normalize_color_fuzzy(f"{original_msg} {msg_en}") or _normalize_color(msg_en)
        if detected_color:
            selected_color = detected_color

        normalized_query = extract_product_search_query(original_msg, msg_en, search_query)
        compound_query = repair_multi_product_joiners(
            (msg_en or original_msg or search_query or "").strip()
        )
        primary_query = search_query.strip() or normalized_query or compound_query
        if selected_cat:
            primary_query = category_browse_search_name(selected_cat, primary_query, ctx)
            compound_query = category_browse_search_name(selected_cat, compound_query, ctx) or ""
        missing_parts = []
        products = []
        os_spec = {}
        os_has_more = False
        product_page = 1
        strict_title = False

        multi_parts = []
        if not message_needs_support_not_product(f"{original_msg} {msg_en}") and not message_needs_policy_answer(
            f"{original_msg} {msg_en}"
        ):
            # One pass on user text — avoid duplicate splits from msg_en + compound_query.
            multi_parts = collect_multi_product_parts(msg_en or original_msg or "", original_msg or "")
        if len(multi_parts) >= 2 and multi_product_parts_are_valid(multi_parts):
            products, missing_parts = search_multi_product_parts(
                multi_parts,
                category_id=selected_cat,
                message_color=selected_color,
                page=1,
            )
        elif len(multi_parts) >= 2:
            multi_parts = []

        lookup_pro_id = ctx.get("data", {}).get("lookup_pro_id") or extract_product_id(
            f"{original_msg} {msg_en} {search_query}"
        )
        if search_query and is_noisy_search_query(search_query):
            search_query = extract_product_search_query(original_msg, msg_en, "")

        pq_llm = None
        os_total = 0
        if not products:
            os_spec, pq_llm = understand_product_query(
                original_msg,
                msg_en,
                conv_for_llm,
                lang,
                category_id=selected_cat,
                color=selected_color,
                pro_id=lookup_pro_id,
            )
            if pq_llm and pq_llm.get("search_terms"):
                search_query = pq_llm["search_terms"]
            os_spec = sanitize_product_search_spec(os_spec or {})
            from services.opensearch_products import search_opensearch_catalog

            products, os_spec, os_total, os_has_more = search_opensearch_catalog(
                os_spec, page=product_page
            )
            title_match = (os_spec or {}).get("title_query") or search_query or ""
            strict_title = bool((os_spec or {}).get("title_match_strict"))
            must_match = strict_title or os_spec.get("brand") or os_spec.get("color")
            if not products and not must_match and not has_structured_product_filters(os_spec):
                products, os_spec, os_total, os_has_more = search_products_combined(
                    f"{original_msg} {msg_en}",
                    original_msg=original_msg,
                    msg_en=msg_en,
                    category_id=selected_cat,
                    color=selected_color,
                    title_hint=os_spec.get("title_query") or search_query,
                    pro_id=lookup_pro_id,
                    page=product_page,
                    ctx=ctx,
                )
                os_spec = sanitize_product_search_spec(os_spec or {})
                products = apply_catalog_post_filters(products, os_spec)
                os_total = len(products)
                os_has_more = False
        if not products and len(multi_parts) < 2:
            rest_q = (os_spec.get("title_query") if os_spec else "") or search_query or primary_query
            rest_q = rest_q if rest_q and not is_noisy_search_query(rest_q) else ""
            if rest_q:
                products = fetch_products_from_api(
                    rest_q, category_id=selected_cat, color=selected_color, page=1
                )
                tm = (os_spec or {}).get("title_query") or rest_q
                if products and os_spec:
                    products = apply_catalog_post_filters(products, os_spec)
            if not products and len(multi_parts) == 1:
                products, missing_parts = search_multi_product_parts(
                    multi_parts,
                    category_id=selected_cat,
                    message_color=selected_color,
                    page=1,
                )
            os_has_more = False
        else:
            _os_f = {
                k: os_spec.get(k)
                for k in ("color", "size", "brand", "sku", "pro_id", "sort", "unit_price_max")
                if os_spec.get(k) is not None
            }
            log_reasoning(f"OpenSearch/catalog hits={len(products)} total={os_total} filters={_os_f}")
            ctx.setdefault("data", {})
            ctx["data"]["last_os_spec"] = {
                k: os_spec.get(k)
                for k in (
                    "title_query", "color", "size", "brand", "sku", "pro_id", "category_id",
                    "unit_price_min", "unit_price_max", "purchase_price_min", "purchase_price_max",
                    "rating_min", "in_stock_only", "sort",
                )
                if os_spec.get(k) is not None
            }
            ctx["data"]["product_page"] = product_page
            ctx["data"]["product_browse_url"] = build_welfog_product_browse_url(os_spec or {}, ctx=ctx)

        if products:
            filter_label = display_label_for_product_search(
                os_spec or {}, pq_llm, original_msg
            )
            title_q = search_query.strip()
            if len(multi_parts) >= 2 and multi_product_parts_are_valid(multi_parts):
                display_query = ", ".join(
                    clean_product_part_label(p, original_msg) or p for p in multi_parts
                )
            else:
                display_query = filter_label or title_q or normalized_query
            if is_noisy_search_query(display_query):
                display_query = filter_label or title_q or "products"
            if not filter_label:
                display_query = re.sub(
                    r"\b(color|colour|ki|ka|ke|liye|de|dikha|dikhao|dikho|dikhaa|dikhaan)\b",
                    "",
                    display_query,
                    flags=re.IGNORECASE,
                )
                display_query = re.sub(r"\s+", " ", display_query).strip()
            if selected_cat and not title_q and not filter_label:
                cat_label = category_name_for_id(selected_cat, ctx) or "this category"
                response_text = sysmsg("products_title_category_named", category=cat_label) or sysmsg(
                    "products_title_category"
                )
            else:
                if (
                    selected_color
                    and not filter_label
                    and selected_color.lower() not in display_query.lower()
                ):
                    display_query = f"{selected_color} {display_query}".strip()
                response_text = sysmsg("products_title_query", query=display_query)
            
            from services.opensearch_products import product_search_show_view_more

            response_text += build_product_rail_with_pagination(
                products,
                sysmsg,
                has_more=product_search_show_view_more(products, os_has_more),
                next_page=product_page + 1,
                browse_more_url=build_welfog_product_browse_url(os_spec or {}, ctx=ctx),
            )
            if missing_parts:
                missing_text = ", ".join(missing_parts)
                found_parts = [
                    clean_product_part_label(p, original_msg) or p
                    for p in multi_parts
                    if (clean_product_part_label(p, original_msg) or p) not in missing_parts
                ]
                if found_parts:
                    partial_msg = _localized_sysmsg(
                        "products_partial_missing_named",
                        original_msg,
                        reply_lang=lang,
                        missing=missing_text,
                        found=", ".join(found_parts),
                    ) or sysmsg(
                        "products_partial_missing_named",
                        missing=missing_text,
                        found=", ".join(found_parts),
                    )
                else:
                    unavailable_text = ", ".join([f"'{part}'" for part in missing_parts])
                    partial_msg = _localized_sysmsg(
                        "products_partial_missing",
                        original_msg,
                        reply_lang=lang,
                        items=unavailable_text,
                    )
                response_text += partial_msg or (
                    f"<div style='margin-top: 12px; color:#555; font-size:13px;'>"
                    f"Sorry, we don't have exact matches for {unavailable_text} on Welfog right now. "
                    f"Available options are shown above.</div>"
                )
        else:
            if selected_cat and not (search_query or "").strip():
                cat_label = category_name_for_id(selected_cat, ctx) or "this category"
                response_text = _localized_sysmsg(
                    "category_products_not_found",
                    original_msg,
                    reply_lang=lang,
                    category=cat_label,
                ) or sysmsg("category_products_not_found")
            else:
                if len(multi_parts) >= 2 and multi_product_parts_are_valid(multi_parts):
                    fallback_query = ", ".join(
                        clean_product_part_label(p, original_msg) or p for p in multi_parts
                    )
                else:
                    fallback_query = display_label_for_product_search(
                        os_spec or {}, pq_llm, original_msg
                    ) or search_query.strip() or normalized_query
                if is_noisy_search_query(fallback_query):
                    fallback_query = (
                        ", ".join(
                            clean_product_part_label(p, original_msg) or p for p in multi_parts
                        )
                        if len(multi_parts) >= 2
                        else display_label_for_product_search(os_spec or {}, pq_llm, original_msg)
                    ) or "products"
                if spec_uses_strict_filter_not_found(os_spec or {}):
                    alt = cheapest_alternative_hint(os_spec)
                    response_text = _localized_sysmsg(
                        "products_filtered_not_found",
                        original_msg,
                        reply_lang=lang,
                        query=fallback_query,
                    ) or sysmsg("products_filtered_not_found", query=fallback_query)
                    if alt:
                        response_text += alt
                else:
                    response_text = (
                        _localized_sysmsg("product_not_found", original_msg, reply_lang=lang, query=fallback_query)
                        or sysmsg("product_not_found", query=fallback_query)
                    )
        reset_context(ctx)
    elif intent == "categories":
        if False and (
            not _looks_like_browse_all_categories_message(comb_low)
            and _text_requests_category_product_browse(comb_low)
        ):
            named_cat = get_category_id_from_text(comb_low, ctx=ctx)
            if named_cat:
                log_reasoning("categories intent overridden -> product (named category in message).")
                intent = "product"
                ctx.setdefault("data", {})
                ctx["data"]["selected_category_id"] = named_cat
                search_query = ""
                selected_cat = named_cat
                selected_color = ctx.get("data", {}).get("selected_color")
                detected_color = _normalize_color(msg_en)
                if detected_color:
                    selected_color = detected_color
                normalized_query = extract_product_search_query(original_msg, msg_en, search_query)
                compound_query = repair_multi_product_joiners((msg_en or original_msg or "").strip())
                primary_query = category_browse_search_name(
                    selected_cat,
                    search_query.strip() or normalized_query or compound_query,
                    ctx,
                )
                products = fetch_products_from_api(primary_query, category_id=selected_cat, color=selected_color, page=1)
                if products:
                    cat_label = category_name_for_id(selected_cat, ctx) or "this category"
                    response_text = sysmsg("products_title_category_named", category=cat_label) or sysmsg(
                        "products_title_category"
                    )
                    response_text += "<motion class='wf-product-rail'>"
                    for p in products:
                        response_text += "<div class='wf-product-card'>"
                        if p["image"]:
                            response_text += f"<motion style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{p['image']}' alt='{p['name']}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                        else:
                            response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
                        name_short = p["name"][:38] + "..." if len(p["name"]) > 38 else p["name"]
                        response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                        response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{p['price']}</div>"
                        if p["link"]:
                            response_text += f"<a href='{p['link']}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_product')}</a>"
                        response_text += "</div>"
                    response_text += "</div>"
                else:
                    cat_label = category_name_for_id(selected_cat, ctx) or "this category"
                    response_text = _localized_sysmsg(
                        "category_products_not_found", original_msg, reply_lang=lang, category=cat_label
                    ) or sysmsg("category_products_not_found")
                reset_context(ctx)
                return send_reply(response_text, lang)

        cats = fetch_nav_categories()
        if not cats:
            response_text = sysmsg("categories_unavailable")
        else:
            # Try to extract a reasonable list from various possible shapes
            items = []
            if isinstance(cats, dict):
                for key in ["data", "categories", "result"]:
                    if isinstance(cats.get(key), list):
                        items = cats.get(key)
                        break
            elif isinstance(cats, list):
                items = cats

            # Flatten first-level items only
            shown = []
            for it in items[:20]:
                if not isinstance(it, dict):
                    continue
                cid = it.get("id") or it.get("category_id") or it.get("cat_id")
                name = it.get("name") or it.get("title") or it.get("category_name")
                if cid and name:
                    shown.append((cid, name))

            if not shown:
                response_text = sysmsg("categories_parse_failed")
            else:
                # Store full nav + inner subcategory map for next message selection
                ctx.setdefault("data", {})
                ensure_expanded_categories_map_for_ctx(ctx)
                response_text = sysmsg("categories_title")
                response_text += sysmsg("categories_list_wrap_start")
                for cid, name in shown:
                    response_text += f"• <b>{name}</b> (id: {cid})<br>"
                response_text += sysmsg("categories_list_wrap_end") + sysmsg("categories_footer")
        # Keep context so next user message can select a category
        ctx["awaiting"] = "category_select"
    elif intent == "deals":
        deals = fetch_today_deals()
        items = []
        if isinstance(deals, dict):
            for key in ["data", "products", "result", "today_deal"]:
                if isinstance(deals.get(key), list):
                    items = deals.get(key)
                    break
        elif isinstance(deals, list):
            items = deals

        if not items:
            response_text = sysmsg("deals_unavailable")
        else:
            IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
            response_text = sysmsg("deals_title", title=(deals.get("title") if isinstance(deals, dict) else None) or sysmsg("default_deals_name"))
            response_text += "<div class='wf-product-rail'>"

            shown = 0
            for p in items:
                if shown >= 5:
                    break
                if not isinstance(p, dict):
                    continue
                name = p.get("name") or p.get("product_name") or sysmsg("default_deal_card_title")
                from services.welfog_api import customer_sale_price, format_customer_price_display

                new_price = format_customer_price_display(p, sysmsg("na_price"))
                old_price = p.get("stroked_price") or p.get("old_price") or p.get("unit_price")
                if old_price and customer_sale_price(p) and float(old_price) <= float(customer_sale_price(p)):
                    old_price = None
                slug = p.get("slug") or ""
                thumb = p.get("thumbnail_img") or p.get("thumbnail_image") or p.get("image") or ""
                image = (IMAGE_BASE_URL + str(thumb).lstrip("/")) if thumb else ""
                link = f"https://welfog.com/product_details/{slug}" if slug else "https://welfog.com"

                response_text += "<div class='wf-product-card'>"
                if image:
                    response_text += f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{image}' alt='{name}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                else:
                    response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"

                name_short = name[:38] + "..." if len(name) > 38 else name
                response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                if old_price and str(old_price).strip() and str(old_price) != str(new_price):
                    response_text += "<div style='margin-bottom: 10px; margin-top: auto;'>"
                    response_text += f"<span style='font-size: 15px; font-weight: bold; color: #ff7a00;'>₹{new_price}</span> "
                    response_text += f"<span style='font-size: 12px; color: #888; text-decoration: line-through;'>₹{old_price}</span>"
                    response_text += "</div>"
                else:
                    response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{new_price}</div>"
                response_text += f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_deal')}</a>"
                response_text += "</div>"
                shown += 1

            response_text += "</div>"
            response_text += (
                _localized_sysmsg("deals_carousel_footer", original_msg, reply_lang=lang)
                or sysmsg("deals_carousel_footer")
                or ""
            )
        reset_context(ctx)
        ctx["data"] = {"pending_offer": "deals"}
    elif intent == "category_feed":
        feed = fetch_category_wise_feed(page=1)
        groups = []
        if isinstance(feed, dict) and isinstance(feed.get("data"), list):
            groups = feed.get("data")

        if not groups:
            response_text = sysmsg("cat_feed_unavailable")
        else:
            IMAGE_BASE_URL = "https://d1f02fefkbso7w.cloudfront.net/"
            response_text = sysmsg("category_feed_title")
            shown_groups = 0
            for grp in groups:
                if shown_groups >= 2:
                    break
                if not isinstance(grp, dict):
                    continue
                cat = grp.get("category") or {}
                cat_name = cat.get("name") or sysmsg("default_category_title")
                prods = grp.get("products") if isinstance(grp.get("products"), list) else []
                if not prods:
                    continue

                response_text += f"<div style='margin: 10px 0 6px 0; color:#333; font-weight:700;'>{cat_name}</div>"
                response_text += "<div class='wf-product-rail'>"
                shown = 0
                for p in prods:
                    if shown >= 5:
                        break
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name") or sysmsg("default_product_card_title")
                    price = p.get("price") or sysmsg("na_price")
                    link_slug = p.get("link") or p.get("slug") or ""
                    thumb = p.get("image") or p.get("thumbnail_img") or p.get("thumbnail_image") or ""
                    image = (IMAGE_BASE_URL + str(thumb).lstrip("/")) if thumb else ""
                    link = f"https://welfog.com/product_details/{link_slug}" if link_slug else "https://welfog.com"

                    response_text += "<div class='wf-product-card'>"
                    if image:
                        response_text += f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; border: 1px solid #f0f0f0;'><img src='{image}' alt='{name}' style='max-width: 100%; max-height: 100%; object-fit: contain; display: block;'></div>"
                    else:
                        response_text += f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
                    name_short = name[:38] + "..." if len(name) > 38 else name
                    response_text += f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
                    response_text += f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; margin-bottom: 12px; margin-top: auto;'>₹{price}</div>"
                    response_text += f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_product')}</a>"
                    response_text += "</div>"
                    shown += 1

                response_text += "</div>"
                shown_groups += 1
        reset_context(ctx)
    elif intent == "pincode_check":
        from services.pincode_delivery_flow import run_pincode_delivery_ai_flow

        pin_result = run_pincode_delivery_ai_flow(
            original_msg, msg_en, conversation_context=conv_for_llm, reply_lang=lang
        )
        if pin_result.handled and pin_result.reply_html:
            response_text = pin_result.reply_html
        else:
            from services.pincode_delivery_flow import format_pincode_check_reply

            from utils.helpers import resolve_pincode_for_check

            pincode = (
                ai_data.get("extracted_pincode", "")
                or resolve_pincode_for_check(
                    original_msg, conv_for_llm, msg_en=msg_en
                )
            )
            if not pincode:
                response_text = _localized_sysmsg("ask_pincode", original_msg, reply_lang=lang)
            else:
                api_res = check_pincode_delivery(pincode)
                response_text = format_pincode_check_reply(
                    pincode, api_res, original_msg, lang
                )
        reset_context(ctx)

    elif intent in ["order", "refund", "payment"]:
        comb_order = f"{original_msg} {msg_en}".lower()
        if _text_is_product_id_lookup_context(comb_order) or extract_product_id(comb_order):
            log_reasoning("order intent overridden -> product (product-id / catalog context).")
            intent = "product"
            pid = extract_product_id(comb_order)
            ctx.setdefault("data", {})
            if pid:
                ctx["data"]["lookup_pro_id"] = pid
            ctx["order_id"] = None
            ctx["awaiting"] = None
            selected_cat = ctx.get("data", {}).get("selected_category_id")
            selected_color = _normalize_color(msg_en) or ctx.get("data", {}).get("selected_color")
            normalized_query = extract_product_search_query(original_msg, msg_en, search_query)
            os_text = f"{original_msg} {msg_en}".strip()
            products, os_spec, os_total, os_has_more = search_products_combined(
                os_text,
                category_id=selected_cat,
                color=selected_color,
                title_hint=normalized_query,
                pro_id=pid,
                page=1,
                ctx=ctx,
            )
            if products:
                ctx["data"]["last_os_spec"] = {
                    k: os_spec.get(k)
                    for k in (
                        "title_query", "color", "size", "brand", "sku", "pro_id", "category_id",
                        "unit_price_min", "unit_price_max", "purchase_price_min", "purchase_price_max",
                        "rating_min", "in_stock_only", "sort",
                    )
                    if os_spec.get(k) is not None
                }
                label = f"Product ID {pid}" if pid else (normalized_query or "products")
                response_text = sysmsg("products_title_query", query=label)
                browse_url = build_welfog_product_browse_url(os_spec or {}, ctx=ctx)
                response_text += build_product_rail_with_pagination(
                    products,
                    sysmsg,
                    has_more=os_has_more,
                    next_page=2,
                    browse_more_url=browse_url,
                )
                reset_context(ctx)
                return send_reply(response_text, lang)
            fallback_q = normalized_query or str(pid or "")
            response_text = sysmsg("product_not_found", query=fallback_q)
            reset_context(ctx)
            return send_reply(response_text, lang)

        ctx["last"] = intent
        needs_id = ai_data.get("needs_order_id", True)

        from utils.helpers import (
            message_is_user_feedback_or_closing,
            _user_announcing_will_provide_order_id,
        )

        if should_send_warm_feedback_reply(original_msg, msg_en, conv_for_llm):
            reset_context(ctx)
            return send_reply(
                build_warm_feedback_reply(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                ),
                lang,
            )

        user_line_order = original_msg.strip() or msg_en.strip()
        if message_is_user_feedback_or_closing(user_line_order):
            needs_id = False
        if _user_announcing_will_provide_order_id(user_line_order):
            ctx["order_id"] = None
            needs_id = True
            log_reasoning("User will send order id — cleared stale ctx order_id.")
        else:
            extracted_inline_id = resolve_order_id_for_tracking(
                user_line_order,
                conv_for_llm,
                bot_awaiting_order_id=ctx.get("awaiting") == "order_id",
            )
            if extracted_inline_id:
                ctx["order_id"] = extracted_inline_id
                log_reasoning(f"Order id for this turn: {extracted_inline_id}")
            elif intent == "order":
                ctx["order_id"] = None

        current_order_id = ctx.get("order_id")

        if current_order_id:
            locked_goal = ""
            try:
                from services.order_id_handoff_fast_path import resolve_bare_order_id_handoff_goal

                locked_goal = resolve_bare_order_id_handoff_goal(
                    ctx,
                    conv_for_llm,
                    original_msg=original_msg,
                    msg_en=msg_en,
                    reply_lang=lang,
                )
            except ImportError:
                pass
            if not locked_goal and isinstance(ctx, dict):
                ar = (ctx.get("data") or {}).get("ai_route") or {}
                try:
                    from services.ai_route_semantics import resolve_order_live_goal_for_turn

                    locked_goal = resolve_order_live_goal_for_turn(
                        ar,
                        original_msg=original_msg,
                        msg_en=msg_en,
                        conversation_context=conv_for_llm,
                    )
                except ImportError:
                    pass
            if locked_goal == "refund_status" or intent == "refund":
                from services.refund_status_flow import _fetch_and_format_refund_status

                response_text = _fetch_and_format_refund_status(
                    current_order_id,
                    user_id,
                    original_msg,
                    lang,
                    source="legacy_ctx_refund",
                )
            elif locked_goal == "order_invoice":
                try:
                    from services.order_id_handoff_fast_path import _fetch_details_handoff_reply

                    response_text = _fetch_details_handoff_reply(
                        "order_invoice",
                        current_order_id,
                        user_id,
                        original_msg,
                        lang,
                    )
                except ImportError:
                    response_text = ""
            elif locked_goal in ("order_details", "payment") or intent == "payment":
                try:
                    from services.order_id_handoff_fast_path import _fetch_details_handoff_reply

                    response_text = _fetch_details_handoff_reply(
                        "order_details",
                        current_order_id,
                        user_id,
                        original_msg,
                        lang,
                        ai_focus="payment" if intent == "payment" else "",
                    )
                except ImportError:
                    response_text = ""
            elif intent == "order" or locked_goal == "track":
                log_reasoning(
                    f"Fetching live order track for id={current_order_id} (user_id={user_id} ownership check)"
                )
                track_data, track_err = fetch_welfog_order_tracking_for_user(current_order_id, user_id)
                if track_data:
                    response_text = format_order_tracking_reply(track_data, current_order_id, lang=lang)
                elif track_err == "login_required":
                    response_text = _localized_sysmsg("order_track_login_required", original_msg, reply_lang=lang) or (
                        "Sorry — please log in to Welfog and open this chat from your account "
                        "so we can show your order status safely."
                    )
                elif track_err == "not_owned":
                    response_text = _localized_sysmsg("order_track_not_owned", original_msg, reply_lang=lang) or (
                        "Sorry — this Order ID doesn't seem linked to your account. "
                        "Please share the ID from your own order SMS/email or My Orders."
                    )
                elif track_err == "unverified":
                    response_text = _localized_sysmsg("order_track_unverified", original_msg, reply_lang=lang) or (
                        "Sorry — I couldn't verify this Order ID with your logged-in account from here. "
                        "Please check My Orders in the Welfog app, or try again with the account that placed the order."
                    )
                else:
                    response_text = _localized_sysmsg("order_track_not_found", original_msg, reply_lang=lang) or (
                        "Sorry — we couldn't find that Order ID. "
                        "Please check your confirmation SMS/email or My Orders and try again."
                    )
            reset_context(ctx)
        elif needs_id:
            try:
                from services.order_id_handoff_fast_path import lock_order_id_ask_from_intent_label

                lock_order_id_ask_from_intent_label(
                    ctx,
                    intent,
                    ai_route=ai_data if isinstance(ai_data, dict) else None,
                )
            except ImportError:
                ctx["awaiting"] = "order_id"
                ctx["last"] = intent
            response_text = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=lang, intent=intent)
        else:
            # FAQ-style steps only — never trust LLM-invented phone/email for tracking
            if intent == "order" and _text_is_order_tracking_intent(comb_low):
                response_text = _tracking_help_reply(original_msg, lang) or sysmsg("how_can_i_help")
            else:
                response_text = ai_response_text if ai_response_text else sysmsg("how_can_i_help")
                if intent == "order" and _text_is_order_tracking_intent(comb_low):
                    foot = sysmsg("order_tracking_optional_id_footer")
                    if foot and response_text:
                        response_text = f"{response_text.rstrip()}<br><br>{foot}"
            reset_context(ctx)

    else:
        comb_low = f"{original_msg} {msg_en}".lower()
        from utils.helpers import (
            message_is_user_feedback_or_closing,
        )

        if should_send_warm_feedback_reply(original_msg, msg_en, conv_for_llm):
            reset_context(ctx)
            return send_reply(
                build_warm_feedback_reply(
                    original_msg,
                    msg_en,
                    reply_lang=lang,
                    conversation_context=conv_for_llm,
                ),
                lang,
            )
        if _text_is_order_tracking_intent(comb_low) and not message_is_user_feedback_or_closing(
            comb_low
        ):
            if _text_needs_order_id_for_tracking(comb_low):
                try:
                    from services.order_id_handoff_fast_path import lock_order_id_ask_from_intent_label

                    lock_order_id_ask_from_intent_label(ctx, "order")
                except ImportError:
                    ctx["last"] = "order"
                    ctx["awaiting"] = "order_id"
                return send_reply(
                    _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=lang, intent="order"), lang
                )
            return send_reply(_tracking_help_reply(original_msg, lang) or sysmsg("how_can_i_help_welfog"), lang)
        if _user_asks_order_history_navigation_help(comb_low):
            response_text = (
                _localized_sysmsg("order_history_help", original_msg, reply_lang=lang)
                or sysmsg("order_history_help")
                or ""
            )
        elif ai_response_text:
            response_text = ai_response_text
        elif _should_offer_human_escalation(comb_low):
            from services.kb_service import format_support_escalation_reply_from_kb

            response_text = (
                format_support_escalation_reply_from_kb(original_msg, msg_en, reply_lang=lang)
                or sysmsg("how_can_i_help_welfog")
            )
        elif _text_asks_customer_care_contact(comb_low):
            response_text = format_customer_care_reply_from_kb(original_msg, msg_en) or sysmsg(
                "how_can_i_help_welfog"
            )
        else:
            from services.query_intent_classifier import (
                build_non_welfog_reply,
                query_intent_allows_kb,
                query_intent_from_ctx,
            )

            if not query_intent_allows_kb(ctx):
                qd = query_intent_from_ctx(ctx)
                response_text = (
                    build_non_welfog_reply(qd, original_msg, msg_en, reply_lang=lang)
                    if qd
                    else sysmsg("how_can_i_help_welfog")
                )
            else:
                grounded = direct_kb_search(
                    retrieval_query, keys=get_customer_kb_keys(), min_score=0.38
                )
                response_text = grounded if grounded else sysmsg("how_can_i_help_welfog")
        reset_context(ctx)

    # ================= FINAL RESPONSE =================
    return send_reply(response_text, lang)

@chat_bp.route("/api/voice/transcribe", methods=["POST"])
def voice_transcribe():
    """Fallback STT when browser Web Speech API cannot reach Google (network error)."""
    from services.voice_transcribe import transcribe_audio_blob

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"ok": False, "error": "no_audio", "text": ""}), 400
    raw = audio.read()
    if len(raw) > 12 * 1024 * 1024:
        return jsonify({"ok": False, "error": "file_too_large", "text": ""}), 413
    out = transcribe_audio_blob(raw, filename=audio.filename or "audio.webm", mime=audio.mimetype or "audio/webm")
    status = 200 if out.get("ok") else 502
    return jsonify(out), status


def register_chat_routes(app):
    app.register_blueprint(chat_bp)
