"""Public chat UI and JSON APIs."""
import json
import re
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, current_app, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from services.answer_router import dispatch_early_answer, resolve_answer_route
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
)
from services.translation_service import (
    customer_reply_language,
    detect_language,
    finalize_customer_reply,
    is_hinglish_message,
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
    should_use_warm_conversation_reply,
    build_assistant_intro_reply,
    fast_greeting_reply_html,
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
    user_contexts,
    _should_bypass_warm_greeting_fast_path,
)
from utils.reasoning_log import chat_log, log_reasoning
from utils.cache import _cache_get, _cache_set

DEFAULT_USER_ID = "STATIC_USER_001"


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


def _is_low_information_turn_for_task_routing(original_msg: str, msg_en: str = "") -> bool:
    text = (original_msg or msg_en or "").strip().lower()
    if not text:
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

    od_goal = message_wants_order_details_or_invoice(
        original_msg, msg_en, conv_for_llm, ai_route=ai_route
    )
    if od_goal:
        log_reasoning(
            f"Skip live track early path — customer wants {od_goal} (details/invoice API)."
        )
        return None

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
    user_id = _resolve_user_id()
    if user_id in user_contexts:
        reset_context(user_contexts[user_id])
    return jsonify({"status": "cleared"})

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
                """
                SELECT DISTINCT
                    COALESCE(cs.chat_token, CAST(cs.id AS CHAR)) AS chat_id,
                    cs.title,
                    cs.created_at
                FROM chat_sessions cs
                INNER JOIN chats c ON (
                    c.chat_token = cs.chat_token
                    OR c.chat_id = cs.chat_token
                    OR CAST(c.chat_id AS CHAR) = CAST(cs.id AS CHAR)
                )
                WHERE cs.user_id = %s
                  AND cs.created_at >= %s
                  AND c.chat_data IS NOT NULL
                  AND CHAR_LENGTH(TRIM(c.chat_data)) > 5
                ORDER BY cs.created_at DESC
                """,
                (user_id, seven_days_ago),
            )
            rows = cur.fetchall()
        chats = []
        for r in rows:
            ts = r["created_at"]
            date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts).split(" ")[0]
            chats.append({"chat_id": r["chat_id"], "title": r["title"], "date_str": date_str})
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
        user_msg = (payload.get("message") or "").strip()
        chat_id = payload.get("chat_id")
        lang_hint = customer_reply_language(user_msg) if user_msg else "en"
        app = current_app._get_current_object()
        try:
            resp = run_with_chat_deadline(fn, args, kwargs, app=app)
            chat_log(f"done in {time.perf_counter() - t0:.2f}s")
            return resp
        except ChatDeadlineExceeded:
            log_busy_fallback(f"deadline>{time.perf_counter() - t0:.1f}s")
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

    return wrapper


@chat_bp.route("/chat", methods=["POST"])
@_chat_request_guard
def chat():
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    user_id = _resolve_user_id()
    chat_log(f"POST user_id={user_id} msg={user_msg[:100]!r}")

    # 🔥 FIX 1: Frontend se chat_id fetch karo
    current_chat_id = data.get("chat_id")

    if user_id not in user_contexts:
        user_contexts[user_id] = {"intent": None, "awaiting": None, "data": {}, "last": None, "order_id": None}

    ctx = user_contexts[user_id]

    defer_chat_session_insert = False
    if not current_chat_id:
        current_chat_id = generate_chat_token()
        defer_chat_session_insert = True

    # KB cache (silent — no static routing stages before AI).
    ensure_knowledge_cache_fresh()

    if defer_chat_session_insert:
        _register_new_chat_session_mysql(str(user_id), user_msg, current_chat_id)

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

    def _enforce_official_support_contacts(text_data: str, lang_code: str) -> str:
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
            from services.order_details_flow import message_wants_order_details_or_invoice

            if message_wants_order_details_or_invoice(
                original_msg, msg_en, conv_for_llm
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

    def send_reply(text_data, lang_code):
        """
        Reply in the same language/style as the customer (en / hinglish / Indian scripts).
        """
        try:
            from utils.debug_session_log import dbg

            dbg(
                "H3",
                "chat_routes.py:send_reply_enter",
                "send_reply called",
                {"len": len(text_data) if isinstance(text_data, str) else -1},
            )
        except Exception:
            pass
        if isinstance(text_data, str):
            text_data = _enforce_official_support_contacts(text_data, lang_code)
        rl = resolve_customer_reply_lang(original_msg, lang_code or "")
        if isinstance(text_data, str):
            final_output = finalize_customer_reply(text_data, original_msg, rl)
        else:
            final_output = text_data
        try:
            db_store_message(current_chat_id, "bot", final_output, str(user_id))
        except Exception as db_exc:
            chat_log(f"bot msg DB skip: {db_exc}")
        preview = (
            final_output[:120].replace("\n", " ")
            if isinstance(final_output, str)
            else f"<{type(final_output).__name__}>"
        )
        chat_log(f"reply sent ({len(final_output) if isinstance(final_output, str) else 'obj'} chars): {preview!r}")
        try:
            from services.chat_flow_telemetry import log_turn_complete

            ar = (ctx.get("data") or {}).get("ai_route") or {}
            adr = (ctx.get("data") or {}).get("answer_route") or {}
            log_turn_complete(
                intent=ar.get("intent") or adr.get("intent") or "",
                route=adr.get("handler") or ar.get("route_handler") or "",
                source=adr.get("handler") or ar.get("route_handler") or adr.get("source") or "",
                reason=adr.get("reason") or ar.get("reasoning") or "",
            )
        except Exception:
            pass
        return jsonify({"chat_id": current_chat_id, "type": "text", "data": final_output})

    # Pagination: product search "View more"
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

    # Pagination: "View more" — return fragments only; UI appends into the same .wf-ph-root (no new bubble).
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
                "tail_html": append_parts.get("tail_html", append_parts.get("footer_html", "")),
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

    chat_request_id = "-"
    try:
        from services.chat_flow_telemetry import begin_chat_turn

        chat_request_id = begin_chat_turn()
        log_reasoning(f"[chat-flow] request_id={chat_request_id} turn started")
    except ImportError:
        pass

    original_msg = user_msg.strip()
    lang = customer_reply_language(original_msg)

    if lang in ("en", "hinglish"):
        msg_en = original_msg.lower().strip()
    else:
        msg_en = to_en(original_msg).lower().strip()

    comb_ph_early = f"{original_msg} {msg_en}".lower()

    # User message — never block reply if MySQL is slow/down.
    try:
        db_store_message(current_chat_id, "user", user_msg, str(user_id))
    except Exception as db_exc:
        print(f"[chat] user msg DB skip: {db_exc}", flush=True)

    # Keep only recent compact chat context to control LLM token usage.
    recent_msgs = db_get_recent_messages(current_chat_id, 10)
    conv_for_llm = _format_conversation_for_llm(recent_msgs, max_turns=8)

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

    from utils.helpers import message_is_conversation_reset_command

    if message_is_conversation_reset_command(original_msg):
        log_reasoning("User reset command — clear order-id state.")
        reset_context(ctx)
        return send_reply(sysmsg("cancelled"), lang)

    comb_pre_route = f"{original_msg} {msg_en}".lower()
    _pure_hello = _is_short_pure_greeting(original_msg) or _is_light_smalltalk_fast(
        original_msg, msg_en
    )
    if (
        ctx.get("awaiting") is None
        and not _message_submits_or_corrects_order_id(original_msg)
        and (
            _pure_hello
            or (
                not _should_bypass_warm_greeting_fast_path(comb_pre_route)
                and should_send_warm_greeting_reply(original_msg, msg_en, conv_for_llm)
                and not _conversation_in_order_tracking_flow(conv_for_llm)
            )
        )
    ):
        log_reasoning("Instant greeting — template reply (skip LLM routing).")
        reset_context(ctx)
        return send_reply(fast_greeting_reply_html(original_msg, reply_lang=lang), lang)

    try:
        from services.product_catalog_resolver import try_product_ai_first_catalog

        ai_product = try_product_ai_first_catalog(
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
            user_id=user_id,
        )
        if ai_product:
            reply_html, _ai_route_product = ai_product
            reset_context(ctx)
            return send_reply(reply_html, lang)
    except ImportError:
        pass

    # === AI-FIRST: Groq/LLM classifies intent → then KB / live API / AI answer ===
    log_reasoning(f"AI routing: {original_msg[:120]!r}")
    from services.ai_first_router import resolve_answer_route_ai_first
    from services.semantic_intent import skip_keyword_intent_routes, should_skip_ctx_last_pinning
    from utils.debug_session_log import dbg

    try:
        route_decision, ai_route_data = resolve_answer_route_ai_first(
            original_msg,
            msg_en,
            retrieval_query=retrieval_query,
            conv_for_llm=conv_for_llm,
            reply_lang=lang,
            ctx=ctx,
        )
        dbg(
            "H2",
            "chat_routes.py:post_route",
            "resolve_answer_route_ai_first done",
            {
                "intent": route_decision.intent,
                "handler": route_decision.handler,
                "ai_intent": (ai_route_data or {}).get("intent"),
            },
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
        from services.query_intent_classifier import try_query_intent_gate_reply

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

    if (route_decision.handler or "").strip() == "warm_feedback":
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

    retrieval_query = _refresh_retrieval_query(ai_route_data)

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

    # === LOCKED ROUTE: one analysis → one handler → one reply (skip legacy cascade) ===
    try:
        from services.chat_flow_telemetry import is_routing_complete
        from services.locked_route_executor import (
            execute_locked_route_turn,
            locked_route_fallback,
        )

        if is_routing_complete():
            locked_reply = execute_locked_route_turn(
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
            if locked_reply:
                return send_reply(locked_reply, lang)
            fallback_reply = locked_route_fallback(
                route_decision=route_decision,
                ai_route=ai_route_data,
                original_msg=original_msg,
                msg_en=msg_en,
                conv_for_llm=conv_for_llm,
                retrieval_query=retrieval_query,
                lang=lang,
                ctx=ctx,
            )
            reset_context(ctx)
            return send_reply(fallback_reply or sysmsg("server_busy"), lang)
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
    dbg("H2", "chat_routes.py:post_scope", "conversation_scope checked", {"has_reply": bool(scope_early)})
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
            ctx["last"] = "pincode"
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
    dbg("H2", "chat_routes.py:post_intent", "intent_executor done", {"has_reply": bool(intent_reply)})
    if intent_reply:
        dbg("H3", "chat_routes.py:send_reply", "returning intent_reply", {"len": len(intent_reply or "")})
        return send_reply(intent_reply, lang)

    from utils.helpers import (
        message_is_user_confused_or_rephrasing_bot,
        _conversation_in_pincode_delivery_flow,
        _text_is_pincode_serviceability_question,
    )

    if message_is_user_confused_or_rephrasing_bot(
        f"{original_msg} {msg_en}".strip(), conv_for_llm
    ) and (
        ai_pin_semantic
        or _conversation_in_pincode_delivery_flow(conv_for_llm)
        or _text_is_pincode_serviceability_question(f"{original_msg} {msg_en}".strip(), conv_for_llm)
    ):
        from services.pincode_delivery_flow import build_pincode_missing_or_invalid_reply

        clarify_pin = build_pincode_missing_or_invalid_reply(
            original_msg, msg_en, conv_for_llm, reply_lang=lang
        )
        if clarify_pin:
            log_reasoning("User confused on delivery thread — re-ask PIN (not warm/KB).")
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
    if keyword_routes_ok and use_det_core and (
        _text_asks_order_history(comb_core) or message_asks_my_welfog_purchases(comb_core)
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
        and _is_low_information_turn_for_task_routing(original_msg, msg_en)
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
            intent = ctx.get("last") or "order"
            ctx["order_id"] = extracted_id
            ctx["awaiting"] = None
            ai_data = {"intent": intent, "is_welfog_related": True, "needs_order_id": False}
        elif _should_release_order_id_awaiting_for_routing(comb_await):
            log_reasoning("Awaiting order-id: new topic — releasing lock for full agent routing.")
            ctx["awaiting"] = None
            run_normal_flow = True
        elif turn_is_catalog_product_lookup(original_msg, msg_en):
            log_reasoning("Awaiting order-id: catalog product browse — releasing lock.")
            ctx["awaiting"] = None
            run_normal_flow = True
        elif _is_low_information_turn_for_task_routing(original_msg, msg_en):
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
            or cosine_similarity(encode_texts([msg_en]), get_greetings_vecs()).max() > 0.58
        )
        is_small = not skip_greeting_fast and _looks_like_light_smalltalk(original_msg, msg_en)
        # Do not replay warm greeting when AI already classified a substantive Welfog request.
        skip_greeting_normal = bool(ctx.get("data", {}).get("ai_route")) and not (is_greet or is_small)
        if (is_greet or is_small) and not skip_greeting_normal:
            log_reasoning("Greeting or light smalltalk; warm Welfog template reply.")
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
                or sysmsg(reply_key)
                or sysmsg("greeting"),
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
                    log_reasoning("Answer router: AI understanding (fallback Groq route).")
                    route_data = ai_brain_route(original_msg, conv_for_llm, reply_lang=lang)
                if not route_data:
                    fallback_text = sysmsg("server_busy")
                    return send_reply(fallback_text, lang)

                intent_route = route_data.get("intent", "general")
                is_welfog = route_data.get("is_welfog_related", True)
                search_query = route_data.get("search_query", "")
                extracted_id = route_data.get("extracted_pincode", "")

                log_reasoning(f"AI routing decision => intent={intent_route}, is_welfog_related={is_welfog}")

                # Hard guard: if this turn is clearly off-topic / non-Welfog, force polite decline path.
                comb_turn = f"{original_msg} {msg_en}".lower()
                if not is_welfog or intent_route == "out_of_domain":
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
                if scope_now == "external":
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
                        or sysmsg("greeting"),
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
                "",
                lang,
                category_id=selected_cat,
                color=selected_color,
                pro_id=lookup_pro_id,
            )
            if pq_llm and pq_llm.get("search_terms"):
                search_query = pq_llm["search_terms"]
            os_spec = sanitize_product_search_spec(os_spec or {})
            products, os_total, os_has_more = search_opensearch_products(
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
            for g in groups:
                if shown_groups >= 2:
                    break
                if not isinstance(g, dict):
                    continue
                cat = g.get("category") or {}
                cat_name = cat.get("name") or sysmsg("default_category_title")
                prods = g.get("products") if isinstance(g.get("products"), list) else []
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
            if intent == "order":
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
            elif intent == "refund":
                from services.refund_status_flow import _fetch_and_format_refund_status

                response_text = _fetch_and_format_refund_status(
                    current_order_id,
                    user_id,
                    original_msg,
                    lang,
                    source="legacy_ctx_refund",
                )
            elif intent == "payment":
                res = fetch_api("payment", current_order_id, user_id=user_id)
                if res and res.get("status"):
                    response_text = f"Payment status: {res['status']} via {res['method']}."
                else:
                    response_text = "Payment details not found for this order in your account."
            reset_context(ctx)
        elif needs_id:
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
