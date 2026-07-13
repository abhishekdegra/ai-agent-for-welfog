"""
Decide HOW to answer a customer message: Knowledge Base, Welfog API, or grounded AI.

Priority:
  1) Deterministic KB (privacy, terms, about, policy, contact)
  2) Live API (order history, tracking, catalog, deals, categories, pincode)
  3) KB + AI (strong KB match — paraphrase in customer's language)
  4) AI route + answer (understanding when KB/API are not a fit)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from utils.reasoning_log import log_reasoning


@dataclass
class AnswerRouteDecision:
    """Where the bot should get the reply from."""

    source: str  # kb | api | kb_ai | ai | reject
    intent: str
    handler: str
    reason: str
    search_query: str = ""
    kb_keys: list[str] = field(default_factory=list)
    is_welfog_related: bool = True
    kb_hit: Optional[dict[str, Any]] = None
    kb_min_score: float = 0.22

    def to_log_line(self) -> str:
        extra = f" sq={self.search_query!r}" if self.search_query else ""
        return f"source={self.source} intent={self.intent} handler={self.handler}{extra} — {self.reason}"


def _kb_retrieval_hit(retrieval_query: str, keys: list[str] | None = None) -> Optional[dict]:
    from services.kb_service import best_kb_hit, get_customer_kb_keys, keyword_kb_hit

    if not (retrieval_query or "").strip():
        return None
    key_list = keys or get_customer_kb_keys()
    hit = best_kb_hit(retrieval_query, keys=key_list, min_score=0.20)
    if hit:
        return hit
    return keyword_kb_hit(retrieval_query, keys=key_list, min_hits=2)


def _kb_keys_for_message(combined: str) -> list[str]:
    from utils.helpers import _normalize_welfog_typos

    tl = f" {_normalize_welfog_typos(combined)} "
    keys: list[str] = []
    if "privacy" in tl:
        keys.append("privacy")
    if "terms" in tl or "condition" in tl:
        keys.append("terms")
    if "shipping" in tl or "delivery" in tl:
        keys.extend(["shipping", "faqs"])
    if "payment" in tl:
        keys.extend(["payment", "faqs"])
    if "refund" in tl or "return" in tl:
        keys.extend(["refund", "faqs"])
    if "seller" in tl:
        keys.append("seller")
    if not keys:
        keys = ["faqs", "company", "support"]
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def try_product_shopping_route_decision(
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    *,
    ai_route: Optional[dict] = None,
) -> Optional[AnswerRouteDecision]:
    """
    Fallback catalog route only when Groq routing JSON is missing — never override AI intent.
    """
    if ai_route and (ai_route.get("intent") or "").strip().lower() not in ("product", ""):
        return None
    from services.product_search_flow import product_flow_hard_exclusions
    from utils.helpers import (
        _normalize_order_chat_text,
        _text_has_order_placement_intent,
        _text_has_product_shopping_intent,
        extract_product_search_query,
        message_is_casual_offtopic_not_shopping,
        user_continues_product_browse_from_conversation,
    )

    combined = f"{original_msg} {msg_en}".strip()
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(combined):
            return None
    except ImportError:
        pass
    comb_low = _normalize_order_chat_text(combined).lower()
    if message_is_casual_offtopic_not_shopping(combined):
        return None
    if _text_has_order_placement_intent(combined):
        return None
    if product_flow_hard_exclusions(combined, msg_en, original_msg):
        return None
    try:
        from services.turn_intent_gate import message_has_catalog_search_signal

        eligible = message_has_catalog_search_signal(combined) or user_continues_product_browse_from_conversation(
            original_msg, conv_for_llm
        )
    except ImportError:
        eligible = _text_has_product_shopping_intent(combined) or user_continues_product_browse_from_conversation(
            original_msg, conv_for_llm
        )
    if not eligible:
        return None
    sq = extract_product_search_query(original_msg, msg_en, "") or ""
    return AnswerRouteDecision(
        source="ai_product",
        intent="product",
        handler="product_ai_flow",
        search_query=sq,
        is_welfog_related=True,
        reason="Product shopping — catalog flow (Groq-independent heuristic).",
    )


def resolve_structured_procedure_route(
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
) -> Optional[AnswerRouteDecision]:
    """
    Fixed step-by-step replies (system messages) — must run before Groq general+KB retrieval.
  """
    from utils.helpers import (
        _text_asks_customer_care_contact,
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        _text_has_order_placement_intent,
        _text_is_order_id_help_request,
        _text_is_tracking_howto_request,
        _text_wants_order_history_list_in_chat,
        _user_asks_order_history_navigation_help,
        message_asks_other_company_social_media,
        message_asks_welfog_social_media,
    )

    from utils.helpers import (
        _turn_is_catalog_product_request,
        customer_turn_text,
    )

    from utils.helpers import (
        _message_is_order_id_followup_submission,
        _message_submits_or_corrects_order_id,
    )

    turn = customer_turn_text(original_msg, msg_en)

    if _message_is_order_id_followup_submission(turn, conversation_context) or _message_submits_or_corrects_order_id(turn):
        return None

    if message_asks_other_company_social_media(turn, conversation_context=conversation_context):
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="other_company_social_decline",
            is_welfog_related=False,
            reason="User asked for another company's social media — Welfog links only.",
        )
    if message_asks_welfog_social_media(turn, conversation_context=conversation_context):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="welfog_social_kb",
            kb_keys=["company"],
            reason="Official Welfog social media links from company knowledge.",
        )

    from services.order_details_flow import text_asks_invoice_howto_navigation

    if text_asks_invoice_howto_navigation(turn, conversation_context):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="dynamic_kb",
            kb_keys=["faqs"],
            reason="Invoice download how-to from FAQ — not live order list.",
        )

    from utils.helpers import _text_is_order_id_help_request

    if _text_is_order_id_help_request(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_id_help_kb",
            kb_keys=["faqs", "shipping", "welfog_api_order_id"],
            reason="Where/how to find Order ID (before wishlist/history).",
        )

    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        _text_asks_to_view_purchase_or_order_history,
        _text_asks_wishlist,
        _text_wants_order_history_list_in_chat,
        message_clarifies_wishlist_not_order_history,
        message_is_wishlist_like_request,
        message_wants_order_history_app_navigation,
    )

    from utils.helpers import _wants_wishlist_list_in_chat

    if message_is_wishlist_like_request(turn) or message_clarifies_wishlist_not_order_history(turn):
        if _wants_wishlist_list_in_chat(turn) or (
            _text_asks_wishlist(turn) and not _text_asks_how_to_view_wishlist(turn, conversation_context)
        ):
            return None
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs"],
            reason="Saved/liked products — wishlist how-to (before order history).",
        )

    if message_wants_order_history_app_navigation(turn, conversation_context) or (
        _text_asks_to_view_purchase_or_order_history(turn)
        and not _text_wants_order_history_list_in_chat(turn, conversation_context)
    ):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=["welfog_api_order_history", "faqs"],
            reason="Purchase/order history app navigation (before policy guard).",
        )

    from utils.helpers import (
        _text_has_refund_or_return_intent,
        _text_is_refund_return_policy_howto,
        _text_is_refund_return_status_lookup,
        message_needs_policy_answer,
    )

    if _text_is_refund_return_policy_howto(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="refund",
            handler="policy_structured_kb",
            kb_keys=["refund", "faqs"],
            reason="Return/refund how-to from policy KB.",
        )
    if message_needs_policy_answer(turn) or _text_has_refund_or_return_intent(turn):
        if _text_is_refund_return_status_lookup(turn, conversation_context):
            return None
        return AnswerRouteDecision(
            source="kb",
            intent="refund",
            handler="dynamic_kb",
            kb_keys=["refund", "faqs"],
            reason="Refund/return policy from KB.",
        )

    from utils.helpers import _text_has_product_shopping_intent

    if _turn_is_catalog_product_request(turn) or _text_has_product_shopping_intent(turn):
        return None
    if _text_wants_order_history_list_in_chat(turn, conversation_context):
        return None
    from utils.helpers import (
        _conversation_in_seller_support_flow,
        _text_has_seller_login_problem_intent,
        _user_complains_bot_gave_wrong_topic,
        _user_seller_issue_still_unresolved,
    )

    if (
        _conversation_in_seller_support_flow(conversation_context)
        or _text_has_seller_login_problem_intent(turn, conversation_context)
        or _user_seller_issue_still_unresolved(turn, conversation_context)
        or _user_complains_bot_gave_wrong_topic(turn)
    ):
        return None
    from utils.helpers import (
        _message_has_app_navigation_intent,
        message_clarifies_wishlist_not_order_history,
        message_denies_wishlist_wants_order_history,
        message_is_wishlist_like_request,
        message_mentions_wishlist_topic,
        message_wants_order_history_app_navigation,
    )

    if message_wants_order_history_app_navigation(turn, conversation_context):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=["welfog_api_order_history", "faqs"],
            reason="Purchase/order history app navigation (semantic).",
        )
    if (
        not message_denies_wishlist_wants_order_history(turn)
        and (
            _text_asks_how_to_view_wishlist(turn, conversation_context)
            or message_clarifies_wishlist_not_order_history(turn)
            or (
                message_is_wishlist_like_request(turn)
                and _message_has_app_navigation_intent(turn)
            )
        )
    ):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs"],
            reason="Steps to view wishlist in the Welfog app (any language).",
        )
    if message_mentions_wishlist_topic(turn) and not _text_asks_wishlist(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            kb_keys=["faqs"],
            reason="Wishlist topic with app-navigation wording.",
        )
    if _user_asks_order_history_navigation_help(turn, conversation_context):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_history_howto_kb",
            kb_keys=["welfog_api_order_history"],
            reason="Steps to view order history in the Welfog app (not purchase-history API).",
        )
    if _text_has_order_placement_intent(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_placement_kb",
            kb_keys=["faqs", "shipping"],
            reason="How to place a new order on Welfog (after history how-to).",
        )
    if _text_is_order_id_help_request(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_id_help_kb",
            kb_keys=["faqs", "shipping"],
            reason="How to find Order ID.",
        )
    if _text_is_tracking_howto_request(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_tracking_howto_kb",
            kb_keys=["shipping", "faqs"],
            reason="How to track an order on Welfog.",
        )
    if _text_asks_customer_care_contact(turn):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="customer_care_kb",
            kb_keys=["support"],
            reason="User asked for customer-care phone/email.",
        )
    return None


def resolve_answer_route(
    original_msg: str,
    msg_en: str,
    retrieval_query: str = "",
    conv_for_llm: str = "",
    ctx: Optional[dict] = None,
) -> AnswerRouteDecision:
    """
    Analyze the user message and pick the best answer channel (KB / API / AI).
    Does not call external APIs — only classifies.
    """
    from utils.helpers import (
        _looks_like_browse_all_categories_message,
        _text_asks_customer_care_contact,
        _text_asks_how_to_view_order_history,
        _text_asks_order_history,
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        _text_has_delivery_or_order_area_intent,
        _text_has_product_shopping_intent,
        _text_is_order_tracking_intent,
        extract_product_id,
        extract_product_search_query,
        message_is_casual_offtopic_not_shopping,
        message_asks_other_company_policy,
        message_is_knowledge_information_request,
        message_is_welfog_about_request,
        message_needs_policy_answer,
        message_needs_support_not_product,
        _text_has_past_order_complaint_context,
        _text_has_refund_or_return_intent,
    )

    combined = f"{original_msg} {msg_en}".strip()
    comb_low = combined.lower()

    from utils.helpers import should_use_warm_conversation_reply, build_warm_conversation_reply
    from services.translation_service import customer_reply_language

    if should_use_warm_conversation_reply(original_msg, msg_en, conversation_context=conv_for_llm or ""):
        rl = customer_reply_language(original_msg)
        warm = build_warm_conversation_reply(original_msg, msg_en, reply_lang=rl)
        if warm:
            return AnswerRouteDecision(
                source="kb",
                intent="general",
                handler="warm_greeting",
                reason="Casual hi/hello or Hinglish opener — polite template, no AI paraphrase.",
            )

    from utils.helpers import message_is_conversational_general_talk, should_use_warm_conversational_reply

    if message_is_conversational_general_talk(original_msg, msg_en, conv_for_llm or ""):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="warm_feedback",
            reason="Thanks/praise/general talk — warm reply, not KB dump.",
        )

    try:
        from services.conversation_followup import is_deals_request_message
    except ImportError:

        def is_deals_request_message(_o, _e):
            return False

    from utils.helpers import _normalize_order_chat_text

    comb_norm = _normalize_order_chat_text(comb_low)

    structured = resolve_structured_procedure_route(
        original_msg, msg_en, conversation_context=conv_for_llm
    )
    if structured:
        return structured

    from services.support_scope import message_mentions_other_company_support

    if message_mentions_other_company_support(original_msg, msg_en, conv_for_llm or ""):
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="off_topic",
            is_welfog_related=False,
            reason="User asked about another company's order/tracking — Welfog-only support.",
        )

    from utils.helpers import _text_has_order_placement_intent

    if _text_has_order_placement_intent(comb_norm):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="order_placement_kb",
            kb_keys=["faqs", "shipping"],
            reason="How to place a new order on Welfog — checkout steps from KB.",
        )

    # Keyword pincode routing — LLM-failsafe only (primary path is resolve_answer_route_ai_first).
    from utils.helpers import (
        _text_has_pincode_delivery_intent,
        _text_is_delivery_serviceability_hypothetical,
        _text_is_pincode_serviceability_question,
        _text_is_undelivered_order_complaint,
    )

    if not _text_is_undelivered_order_complaint(comb_norm) and (
        _text_is_delivery_serviceability_hypothetical(comb_norm)
        or _text_is_pincode_serviceability_question(comb_norm, conv_for_llm or "")
        or _text_has_pincode_delivery_intent(comb_low, conv_for_llm or "")
        or _text_has_delivery_or_order_area_intent(comb_low)
    ):
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            reason="Keyword failsafe: delivery / pincode serviceability (no LLM route).",
        )

    if _text_is_order_tracking_intent(comb_norm):
        return AnswerRouteDecision(
            source="api",
            intent="order",
            handler="order_tracking_api",
            reason="Order status / tracking — live track API when ID available.",
        )

    # --- Order ID: AI + order-id API KB → list IDs or where-to-find help ---
    from services.order_id_flow import message_eligible_for_order_id_ai_flow

    if message_eligible_for_order_id_ai_flow(comb_norm, msg_en, original_msg):
        return AnswerRouteDecision(
            source="ai_order_id",
            intent="order_id",
            handler="order_id_ai_flow",
            reason="Order ID topic — AI reads order-id API KB, then API or help text.",
        )

    # --- Order list / history: AI reads order API KB first, then purchase-history API ---
    from services.order_history_flow import message_eligible_for_order_ai_flow

    if message_eligible_for_order_ai_flow(comb_low, msg_en, original_msg):
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            reason="Order-related message — AI understands via order API knowledge, then API/KB.",
        )

    from utils.helpers import message_asks_my_welfog_purchases

    if message_asks_my_welfog_purchases(comb_low) or _text_asks_order_history(comb_low):
        if not _text_asks_how_to_view_order_history(comb_low):
            return AnswerRouteDecision(
                source="ai_order",
                intent="order_history",
                handler="order_ai_flow",
                reason="Past purchases / order list — before wishlist.",
            )

    if _text_asks_how_to_view_wishlist(comb_low):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="wishlist_howto_kb",
            reason="User wants steps to view wishlist in the Welfog app.",
        )

    if _text_asks_wishlist(comb_low):
        return AnswerRouteDecision(
            source="api",
            intent="wishlist",
            handler="wishlist_api",
            reason="User wants their saved wishlist from wishlists API.",
        )

    if is_deals_request_message(original_msg, msg_en):
        return AnswerRouteDecision(
            source="api",
            intent="deals",
            handler="deals_api",
            reason="Deals / offers request — catalog deals API.",
        )

    if _looks_like_browse_all_categories_message(msg_en) or _looks_like_browse_all_categories_message(
        original_msg
    ):
        return AnswerRouteDecision(
            source="api",
            intent="categories",
            handler="categories_api",
            reason="Browse all categories — nav API.",
        )

    # --- KB before pincode (avoid "order" in refund questions → pincode mis-route) ---
    if message_needs_policy_answer(comb_low):
        policy_keys = ["faqs", "refund", "shipping"]
        hit = _kb_retrieval_hit(retrieval_query or combined, keys=policy_keys)
        return AnswerRouteDecision(
            source="kb_ai" if hit else "ai",
            intent="refund",
            handler="kb_grounded_ai" if hit else "ai_route_and_answer",
            kb_keys=policy_keys,
            kb_hit=hit,
            kb_min_score=0.20,
            reason="Return/refund policy — AI summarizes KB (not canned template).",
        )

    if message_needs_support_not_product(comb_low) and _text_asks_customer_care_contact(comb_low):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="customer_care_kb",
            kb_keys=["support"],
            reason="Customer-care phone/email — support KB only.",
        )

    if message_is_welfog_about_request(comb_low):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="welfog_about_kb",
            kb_keys=["company", "faqs"],
            reason="What is Welfog — company knowledge.",
        )

    from services.policy_scope import _has_policy_topic, policy_question_is_for_welfog

    if _has_policy_topic(comb_low) and not policy_question_is_for_welfog(
        original_msg, msg_en, conv_for_llm
    ):
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="off_topic",
            is_welfog_related=False,
            reason="User asked another company's policy — not Welfog.",
        )

    if message_is_knowledge_information_request(comb_low, conv_for_llm):
        from services.kb_service import resolve_kb_keys_for_question

        keys = resolve_kb_keys_for_question(
            original_msg, msg_en, suggested_keys=_kb_keys_for_message(comb_low)
        )
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="dynamic_kb",
            kb_keys=keys,
            reason="Policy / FAQ — admin KB files (auto-discovered, filtered).",
        )

    from utils.helpers import _text_asks_welfog_fees_or_charges

    if _text_asks_welfog_fees_or_charges(comb_low):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="dynamic_kb",
            kb_keys=["payment", "faqs"],
            reason="Service/platform fees — payment KB (concise).",
        )

    from utils.helpers import _text_asks_short_video_content_rules

    if _text_asks_short_video_content_rules(comb_low, conv_for_llm):
        return AnswerRouteDecision(
            source="kb",
            intent="general",
            handler="dynamic_kb",
            kb_keys=[],
            reason="Short video / shorts content rules — unscoped Admin semantic KB.",
        )


    from utils.helpers import _text_has_pincode_delivery_intent

    from utils.helpers import (
        _conversation_bot_offered_order_id_or_tracking,
        _message_is_order_id_followup_submission,
        resolve_live_api_intent_from_conversation,
        resolve_order_id_for_tracking,
    )

    if _message_is_order_id_followup_submission(original_msg, conv_for_llm or "") and _conversation_bot_offered_order_id_or_tracking(
        conv_for_llm or ""
    ):
        oid = resolve_order_id_for_tracking(
            original_msg.strip() or msg_en.strip(),
            conv_for_llm or "",
            bot_awaiting_order_id=True,
        )
        if oid:
            live_intent = resolve_live_api_intent_from_conversation(
                conv_for_llm or "", ctx_last=None, original_msg=original_msg, msg_en=msg_en
            )
            handler = "order_tracking_api"
            if live_intent == "refund":
                handler = "refund_status_api"
            elif live_intent != "order":
                handler = "ai_route_and_answer"
            return AnswerRouteDecision(
                source="api",
                intent=live_intent,
                handler=handler,
                is_welfog_related=True,
                reason=f"Order ID follow-up — live {live_intent} lookup (not pincode).",
            )

    from utils.helpers import _digits_in_message_are_order_id_not_pincode

    if _digits_in_message_are_order_id_not_pincode(comb_low, conv_for_llm or ""):
        pass
    elif _text_has_pincode_delivery_intent(comb_low, conv_for_llm or ""):
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            reason="Delivery / pincode / serviceability — AI + live API.",
        )

    if not _digits_in_message_are_order_id_not_pincode(comb_low, conv_for_llm or "") and _text_has_delivery_or_order_area_intent(comb_low):
        return AnswerRouteDecision(
            source="api",
            intent="pincode_check",
            handler="pincode_delivery_api",
            reason="Delivery / pincode / serviceability — shipping API.",
        )

    if extract_product_id(comb_low):
        pid = extract_product_id(comb_low)
        return AnswerRouteDecision(
            source="api",
            intent="product",
            handler="catalog_pro_id",
            reason=f"Catalog lookup by product id {pid}.",
            search_query=f"pro_id {pid}" if pid else "",
        )

    # --- Reject off-topic personal chat ---
    if message_is_casual_offtopic_not_shopping(comb_low):
        return AnswerRouteDecision(
            source="reject",
            intent="out_of_domain",
            handler="off_topic",
            is_welfog_related=False,
            reason="Personal/off-topic — not Welfog shopping or support.",
        )

    # --- Order list: Groq order specialist (legacy path when main router unavailable) ---
    from services.order_history_flow import message_eligible_for_order_ai_flow

    if message_eligible_for_order_ai_flow(comb_low, msg_en, original_msg):
        return AnswerRouteDecision(
            source="ai_order",
            intent="order_history",
            handler="order_ai_flow",
            is_welfog_related=True,
            reason="Order AI: purchase history / order list.",
        )

    product_dec = try_product_shopping_route_decision(
        original_msg, msg_en, conv_for_llm=conv_for_llm
    )
    if product_dec:
        return product_dec

    # --- KB + AI: informational question with knowledge match ---
    keys = _kb_keys_for_message(comb_low)
    hit = _kb_retrieval_hit(retrieval_query or combined, keys=keys)
    if hit:
        score = hit.get("score")
        score_f = float(score) if isinstance(score, (int, float)) else 0.35
        if score_f >= 0.24 or (isinstance(score, int) and score >= 2):
            return AnswerRouteDecision(
                source="kb_ai",
                intent="general",
                handler="kb_grounded_ai",
                kb_keys=keys,
                kb_hit=hit,
                kb_min_score=0.20,
                reason=f"Knowledge base match (source={hit.get('source')}, score={score}) — AI summarizes KB.",
            )

    # --- AI: full understanding + routing (fallback) ---
    return AnswerRouteDecision(
        source="ai",
        intent="general",
        handler="ai_route_and_answer",
        kb_keys=keys,
        reason="No deterministic route; use AI to understand then KB-grounded or API intent.",
    )


def try_deterministic_kb_reply(
    decision: AnswerRouteDecision,
    original_msg: str,
    msg_en: str,
    reply_lang: str,
    conv_for_llm: str = "",
    *,
    ai_route: dict | None = None,
) -> Optional[str]:
    """Return HTML reply when handler is a fixed KB formatter; else None."""
    from services.kb_service import (
        format_customer_care_reply_from_kb,
        format_knowledge_information_reply_from_kb,
        format_policy_help_reply_from_kb,
        format_welfog_about_reply_from_kb,
    )

    from services.translation_service import (
        customer_reply_language,
        localized_sysmsg_for_customer,
    )

    rl = reply_lang or customer_reply_language(original_msg)

    def _sysmsg_body(key: str) -> str:
        return localized_sysmsg_for_customer(key, original_msg, reply_lang=rl) or ""

    h = decision.handler
    if h == "order_history_howto_kb":
        return _sysmsg_body("order_history_help") or None
    if h == "welfog_social_kb":
        from services.kb_service import format_welfog_social_media_reply_from_kb

        return (
            format_welfog_social_media_reply_from_kb(
                original_msg, msg_en, reply_lang=rl, conversation_context=conv_for_llm
            )
            or None
        )
    if h == "other_company_social_decline":
        from services.support_scope import build_other_company_social_decline

        return build_other_company_social_decline(original_msg, reply_lang=rl) or None
    if h == "wishlist_howto_kb":
        return _sysmsg_body("wishlist_help") or None
    if h == "order_tracking_howto_kb":
        return _sysmsg_body("tracking_help") or None
    if h in (
        "dynamic_kb",
        "knowledge_topic_kb",
        "welfog_fees_kb",
        "short_video_rules_kb",
        "seller_kb",
    ):
        from services.kb_service import format_dynamic_kb_answer

        return (
            format_dynamic_kb_answer(
                original_msg,
                msg_en,
                reply_lang=rl,
                conversation_context=conv_for_llm,
                suggested_keys=list(decision.kb_keys or []),
                ai_route=ai_route,
            )
            or None
        )
    if h == "order_placement_kb":
        return _sysmsg_body("order_placement_help") or None
    if h == "assistant_capability_kb":
        return _sysmsg_body("assistant_capability") or None
    if h == "assistant_intro":
        from utils.helpers import build_assistant_intro_reply

        return build_assistant_intro_reply(original_msg, msg_en, reply_lang=rl) or None
    if h in (
        "bot_latency_apology",
        "bot_topic_correction",
        "bot_insult_calm",
        "bot_search_behavior_help",
    ):
        return _sysmsg_body(h) or None
    if h == "order_id_help_kb":
        return _sysmsg_body("order_id_help") or None
    if h == "support_escalation_kb":
        from services.kb_service import format_support_escalation_reply_from_kb

        return format_support_escalation_reply_from_kb(original_msg, msg_en, reply_lang=rl) or None
    if h == "policy_structured_kb":
        return format_policy_help_reply_from_kb(original_msg, msg_en, reply_lang=reply_lang) or None
    if h == "customer_care_kb":
        return format_customer_care_reply_from_kb(original_msg, msg_en) or None
    if h == "welfog_about_kb":
        return format_welfog_about_reply_from_kb(original_msg, msg_en, reply_lang=reply_lang) or None
    return None


def try_kb_ai_reply(
    decision: AnswerRouteDecision,
    original_msg: str,
    conv_for_llm: str,
    reply_lang: str,
) -> Optional[str]:
    """KB excerpt + Groq answer in customer language."""
    from services.ai_service import ai_brain_answer
    from services.kb_service import (
        direct_kb_search,
        get_support_contact_kb_keys,
        read_concatenated_kb_file_contents,
    )
    from utils.helpers import _text_asks_customer_care_contact

    hit = decision.kb_hit or {}
    chunk = (hit.get("chunk") or "").strip()
    if not chunk:
        return None

    try:
        from services.kb_service import _faq_answer_text_from_chunk

        chunk = _faq_answer_text_from_chunk(chunk) or chunk
    except ImportError:
        pass

    score = hit.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
    from utils.helpers import message_needs_human_support_escalation

    comb = f"{original_msg}".lower()
    kb_context = (
        "RULE: Do NOT repeat or restate the user's question. Start directly with the answer "
        "in 1-4 sentences, using ONLY the knowledge below.\n"
        f"[source={hit.get('source')} score={score_str}] {chunk}"
    )

    if _text_asks_customer_care_contact(comb) or message_needs_human_support_escalation(comb):
        support_keys = get_support_contact_kb_keys()
        blob = read_concatenated_kb_file_contents(support_keys)
        if blob.strip():
            kb_context = (
                "AUTHORITATIVE SUPPORT/CONTACT KNOWLEDGE (use ONLY for phone/email):\n"
                f"{blob}\n\n---\n{kb_context}"
            )

    ai_data = ai_brain_answer(original_msg, kb_context, conv_for_llm, reply_lang=reply_lang) or {}
    text = (ai_data.get("response") or "").strip()
    if text:
        try:
            from services.kb_service import polish_faq_reply_for_customer

            text = polish_faq_reply_for_customer(text, original_msg)
        except ImportError:
            pass
        return text
    return direct_kb_search(original_msg, keys=decision.kb_keys or None, min_score=0.30)


def dispatch_early_answer(
    decision: AnswerRouteDecision,
    original_msg: str,
    msg_en: str,
    reply_lang: str,
    conv_for_llm: str = "",
    user_id: str = "",
    ai_route: Optional[dict] = None,
) -> Optional[str]:
    """
    If this turn can be fully answered without catalog search, return reply HTML/text.
    Otherwise None (caller continues normal flow with decision.intent).
    """
    log_reasoning(f"Answer router: {decision.to_log_line()}")

    if decision.source == "reject":
        from services.kb_service import sysmsg
        from services.translation_service import customer_reply_language, localize_for_customer
        from services.policy_scope import policy_question_is_external_company
        from services.support_scope import (
            build_other_company_social_decline,
            build_other_company_support_decline,
            message_mentions_other_company_support,
        )

        comb = f"{original_msg} {msg_en}".strip()
        rl = reply_lang or customer_reply_language(original_msg)
        if decision.handler == "temporary_load":
            from services.conversation_scope import (
                SCOPE_OUT,
                build_off_topic_polite_reply,
                resolve_conversation_scope,
            )

            if resolve_conversation_scope(
                original_msg,
                msg_en,
                conv_for_llm,
                reply_lang=rl,
                ai_route=ai_route,
            ).scope == SCOPE_OUT:
                body = build_off_topic_polite_reply(
                    original_msg,
                    msg_en,
                    reply_lang=rl,
                    ai_route=ai_route,
                    conversation_context=conv_for_llm,
                    prefer_llm=False,
                )
                if body:
                    log_reasoning(
                        "LLM unavailable — deterministic off-topic decline (not temporary-load)."
                    )
                    return body
            body = (
                "Sorry — we're facing temporary high load right now. "
                "Please retry in a few seconds; I'll continue from the same topic."
            )
            if rl == "hinglish":
                body = (
                    "Sorry — abhi thoda temporary high load hai. "
                    "Kuch seconds baad retry karo, main isi topic se continue karunga."
                )
            elif rl not in ("en", "hinglish"):
                body = localize_for_customer(body, rl)
            return body
        if decision.handler == "other_company_social_decline":
            body = build_other_company_social_decline(original_msg, reply_lang=rl)
            if body:
                log_reasoning("Other company social media — polite decline.")
                return body
        if policy_question_is_external_company(
            original_msg, msg_en, conv_for_llm
        ):
            if rl == "hinglish":
                body = sysmsg("off_topic_other_company_policy_hinglish")
            else:
                body = sysmsg("off_topic_other_company_policy") or sysmsg("off_topic_polite")
                if rl not in ("en", "hinglish") and body:
                    body = localize_for_customer(body, rl)
            if body:
                return body
        if message_mentions_other_company_support(original_msg, msg_en, conv_for_llm):
            body = build_other_company_support_decline(original_msg, reply_lang=rl)
            if body:
                log_reasoning("Other-company query — decline with quoted topic.")
                return body
        if decision.handler in ("other_company_decline",) or (
            decision.intent == "out_of_domain"
            and not decision.is_welfog_related
            and message_mentions_other_company_support(original_msg, msg_en, conv_for_llm)
        ):
            body = build_other_company_support_decline(original_msg, reply_lang=rl)
            if body:
                return body
        from utils.helpers import (
            should_send_warm_greeting_reply,
            build_warm_conversation_reply,
            _text_has_product_shopping_intent,
        )
        from services.product_search_flow import (
            message_eligible_for_product_ai_flow,
            run_product_search_ai_flow,
        )

        if decision.handler == "off_topic" and decision.intent == "out_of_domain":
            from services.off_topic_reply import build_off_topic_polite_reply

            body = build_off_topic_polite_reply(
                original_msg,
                msg_en,
                reply_lang=rl,
                ai_route=ai_route,
                conversation_context=conv_for_llm,
            )
            if body:
                log_reasoning("Off-topic reject — polite decline (no product override).")
                return body

        comb_rej = f"{original_msg} {msg_en}".strip()
        try:
            from services.query_intent_classifier import query_intent_allows_catalog

            catalog_ok = query_intent_allows_catalog()
        except ImportError:
            catalog_ok = True
        if (
            catalog_ok
            and message_eligible_for_product_ai_flow(
                comb_rej, msg_en, original_msg, conversation_context=conv_for_llm
            )
        ):
            ps = run_product_search_ai_flow(
                original_msg, msg_en, user_id, conversation_context=conv_for_llm, reply_lang=rl
            )
            if ps.handled and ps.reply_html:
                log_reasoning("Reject route overridden — product search.")
                return ps.reply_html

        if should_send_warm_greeting_reply(original_msg, msg_en, conv_for_llm):
            warm = build_warm_conversation_reply(original_msg, msg_en, reply_lang=rl)
            if warm:
                log_reasoning("Reject route overridden — greeting/smalltalk warm reply.")
                return warm
        from services.off_topic_reply import build_off_topic_polite_reply

        return build_off_topic_polite_reply(
            original_msg,
            msg_en,
            reply_lang=rl,
            ai_route=ai_route,
            conversation_context=conv_for_llm,
        )

    if decision.handler == "warm_feedback":
        from utils.helpers import build_warm_feedback_reply

        warm = build_warm_feedback_reply(
            original_msg,
            msg_en,
            reply_lang=reply_lang,
            conversation_context=conv_for_llm or "",
            ai_route=ai_route,
        )
        if warm:
            log_reasoning("Early warm feedback / thanks reply.")
            return warm

    if decision.handler == "warm_greeting":
        from utils.helpers import build_warm_conversation_reply

        warm = build_warm_conversation_reply(original_msg, msg_en, reply_lang=reply_lang)
        if warm:
            log_reasoning("Early warm greeting template.")
            return warm

    # API / substantive handlers BEFORE warm thanks — detected intent must get real answers.
    if decision.handler == "pincode_delivery_api":
        from services.pincode_delivery_flow import run_pincode_delivery_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_pincode_delivery_ai_flow(
            original_msg,
            msg_en,
            conversation_context=conv_for_llm,
            reply_lang=rl,
            ai_route=ai_route,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        return None

    if decision.handler == "order_details_api":
        from services.order_details_flow import run_order_details_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_order_details_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
            ai_route=ai_route,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        return None

    if decision.handler == "order_tracking_api":
        from services.order_tracking_flow import run_order_tracking_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_order_tracking_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        return None

    if decision.handler == "refund_status_api":
        from services.refund_status_flow import run_refund_status_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_refund_status_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
            ai_route=ai_route,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        return None

    from services.intent_executor import ai_route_blocks_generic_shortcuts

    blocks_generic = ai_route_blocks_generic_shortcuts(
        decision, ai_route, original_msg, msg_en, conv_for_llm or ""
    )

    if decision.source == "kb" and not blocks_generic:
        reply = try_deterministic_kb_reply(
            decision, original_msg, msg_en, reply_lang, conv_for_llm, ai_route=ai_route
        )
        if reply:
            log_reasoning(f"Early KB reply via handler={decision.handler}")
            return reply

    if decision.source == "kb_ai" and not blocks_generic:
        reply = try_kb_ai_reply(decision, original_msg, conv_for_llm, reply_lang)
        if reply:
            log_reasoning("Early KB+AI grounded reply.")
            return reply

    from utils.helpers import should_use_warm_conversational_reply

    if (
        not blocks_generic
        and should_use_warm_conversational_reply(original_msg, msg_en, conv_for_llm or "", ai_route)
    ):
        warm = build_warm_feedback_reply(
            original_msg,
            msg_en,
            reply_lang=reply_lang,
            conversation_context=conv_for_llm or "",
            ai_route=ai_route,
        )
        if warm:
            log_reasoning("Conversational thanks/praise — warm reply (skip KB).")
            return warm

    if decision.source == "kb":
        reply = try_deterministic_kb_reply(
            decision, original_msg, msg_en, reply_lang, conv_for_llm, ai_route=ai_route
        )
        if reply:
            log_reasoning(f"Early KB reply via handler={decision.handler}")
            return reply

    if decision.source == "kb_ai":
        reply = try_kb_ai_reply(decision, original_msg, conv_for_llm, reply_lang)
        if reply:
            log_reasoning("Early KB+AI grounded reply.")
            return reply

    if decision.handler == "product_ai_flow":
        from services.product_search_flow import run_product_search_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        route_ctx = ai_route if isinstance(ai_route, dict) else None
        if not route_ctx and decision.intent == "product":
            route_ctx = {
                "intent": "product",
                "is_welfog_related": True,
                "data_channel": "catalog",
                "search_query": (decision.search_query or "").strip(),
            }
        result = run_product_search_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
            search_query=(decision.search_query or "").strip(),
            ai_route=route_ctx,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        sq = (decision.search_query or (route_ctx or {}).get("search_query") or "").strip()
        if sq:
            from services.kb_service import sysmsg
            from services.product_search_flow import _localized_sysmsg

            body = _localized_sysmsg(
                "product_not_found", original_msg, reply_lang=rl, query=sq
            ) or sysmsg("product_not_found", query=sq)
            log_reasoning(
                f"Product route locked — catalog empty for {sq!r}; not falling back to KB."
            )
            return body
        return None

    if decision.handler == "order_id_ai_flow":
        from services.order_id_flow import run_order_id_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_order_id_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        return None

    if decision.handler == "seller_kb" or decision.intent == "seller":
        from services.kb_service import format_seller_reply_from_kb
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        body = format_seller_reply_from_kb(
            original_msg, msg_en, reply_lang=rl, conversation_context=conv_for_llm
        )
        if body:
            log_reasoning("Early seller KB reply.")
            return body

    if decision.handler == "order_placement_kb":
        from services.translation_service import localized_sysmsg_for_customer

        return (
            localized_sysmsg_for_customer(
                "order_placement_help", original_msg, reply_lang=reply_lang
            )
            or None
        )

    if decision.handler == "order_tracking_howto_kb":
        from services.translation_service import localized_sysmsg_for_customer

        return (
            localized_sysmsg_for_customer("tracking_help", original_msg, reply_lang=reply_lang)
            or localized_sysmsg_for_customer("how_can_i_help", original_msg, reply_lang=reply_lang)
            or None
        )

    if decision.handler == "order_id_help_kb":
        from services.translation_service import localized_sysmsg_for_customer

        return (
            localized_sysmsg_for_customer("order_id_help", original_msg, reply_lang=reply_lang)
            or None
        )

    if decision.handler == "support_escalation_kb":
        from services.kb_service import format_support_escalation_reply_from_kb
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        return format_support_escalation_reply_from_kb(original_msg, msg_en, reply_lang=rl) or None

    if decision.handler == "order_ai_flow":
        from services.order_history_flow import run_order_ai_flow
        from services.translation_service import customer_reply_language

        rl = reply_lang or customer_reply_language(original_msg)
        result = run_order_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conv_for_llm,
            reply_lang=rl,
            ai_route=ai_route if isinstance(ai_route, dict) else None,
        )
        if result.handled and result.reply_html:
            return result.reply_html
        from utils.helpers import _user_asks_order_history_navigation_help

        if _user_asks_order_history_navigation_help(f"{original_msg} {msg_en}"):
            from services.translation_service import localized_sysmsg_for_customer

            return (
                localized_sysmsg_for_customer(
                    "order_history_help", original_msg, reply_lang=rl
                )
                or None
            )
        if decision.intent == "order_history":
            from services.welfog_api import format_purchase_history_reply

            log_reasoning("Early order_ai_flow: router order_history — purchase-history API.")
            return format_purchase_history_reply(user_id, page=1, append_only=False)
        return None

    if decision.handler == "wishlist_api":
        from services.welfog_api import format_wishlist_reply

        return format_wishlist_reply(user_id, page=1, append_only=False)

    if decision.handler == "wishlist_howto_kb":
        from services.translation_service import localized_sysmsg_for_customer

        return (
            localized_sysmsg_for_customer("wishlist_help", original_msg, reply_lang=reply_lang)
            or None
        )

    return None
