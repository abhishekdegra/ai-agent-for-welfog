"""
Order questions: AI understands intent first, reads order API knowledge base, then calls APIs.

Scope (phase 1): purchase-history / order list. Tracking delegates to main chat flow.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

from services.kb_service import read_concatenated_kb_file_contents, sysmsg
from services.translation_service import (
    is_hinglish_message,
    language_reply_instruction,
    localize_for_customer,
)
from utils.reasoning_log import log_reasoning


def _localized_sysmsg(key: str, user_msg: str, reply_lang: str = "en", **fmt) -> str:
    from services.translation_service import localized_sysmsg_for_customer

    return localized_sysmsg_for_customer(key, user_msg, reply_lang=reply_lang, **fmt)

ORDER_API_KB_KEYS = ("welfog_api_order_history", "welfog_api")

# Minimal pre-gate — typos OK; AI + API KB makes the real decision.
_ORDER_WORD_RE = re.compile(
    r"(?:\borders?\b|\bordr\w*\b|ऑर्डर|ఆర్డర్|ஆர்டர்|booking|hist[o0]ry|hsitory|histroy|purane\s+order)",
    re.IGNORECASE,
)


@dataclass
class OrderFlowResult:
    handled: bool
    reply_html: str = ""
    intent: str = "general"
    needs_order_id: bool = False
    order_id: str = ""


def get_order_api_knowledge_context() -> str:
    """Full text of order API playbook for Groq grounding."""
    blob = read_concatenated_kb_file_contents(list(ORDER_API_KB_KEYS))
    if not blob.strip():
        return "(Order API knowledge files missing — refresh knowledge cache.)"
    return blob.strip()


def _message_is_order_help_not_list(text: str) -> bool:
    """How/where to find order ID or view history — not 'show my orders' in chat."""
    from utils.helpers import (
        _text_asks_how_to_view_order_history,
        _text_is_order_id_help_request,
        _text_is_order_tracking_intent,
        _text_is_tracking_howto_request,
    )

    if _text_is_order_tracking_intent(text):
        return False
    tl = f" {text.lower()} "
    if any(
        x in tl
        for x in (
            "nahi aaya", "nhi aaya", "nhi aya", "nahi aya", "abhi tak", "late", "delay",
            "kab aayega", "kab aaega", "kab milega", "nahi mila", "nhi mila", "stuck",
            "kya kru", "kya karu", "what should i do",
        )
    ):
        return False

    if _text_is_order_id_help_request(text):
        return True
    if _text_asks_how_to_view_order_history(text):
        return True
    if _text_is_tracking_howto_request(text):
        return True
    tl = f" {text.lower()} "
    if re.search(r"\borders?\b", tl) and re.search(r"\bid\b", tl):
        if any(
            x in tl
            for x in (
                "kaise", "kese", "kahan", "kaha", "where", "how to", "how do",
                "nikal", "nikaal", "nikale", "nikalte", "find", "pata", "milega id",
                "id milega", "id kaha", "id kahan",
            )
        ):
            if not any(
                x in tl
                for x in (
                    "dikhao", "dikha", "dikhe", "list", "history", "meri order",
                    "mere order", "my orders", "saari order", "sab order", "show my",
                )
            ):
                return True
    return False


def message_eligible_for_order_ai_flow(combined: str, msg_en: str, original_msg: str) -> bool:
    """
    Minimal gate: message might be about the customer's orders.
    Disambiguation is done by AI with API KB, not by hundreds of phrase patterns.
    """
    from utils.helpers import extract_product_id, _text_is_product_id_lookup_context

    try:
        from services.order_id_flow import message_eligible_for_order_id_ai_flow

        if message_eligible_for_order_id_ai_flow(combined, msg_en, original_msg):
            return False
    except ImportError:
        pass

    from utils.helpers import _text_asks_order_history, message_asks_my_welfog_purchases

    text = f"{original_msg} {msg_en} {combined}".lower()
    from utils.helpers import _user_asks_order_history_navigation_help

    try:
        from services.order_details_flow import text_asks_invoice_howto_navigation

        if text_asks_invoice_howto_navigation(f"{original_msg} {msg_en}", ""):
            return False
    except ImportError:
        pass
    if _user_asks_order_history_navigation_help(text):
        return False
    if _text_asks_order_history(text) or message_asks_my_welfog_purchases(text):
        if not _message_is_order_help_not_list(text):
            return True
    if _message_is_order_help_not_list(text):
        return False

    from utils.helpers import message_is_past_purchase_list_request

    if message_is_past_purchase_list_request(text):
        return True

    order_ai = ai_understand_order_request(
        (original_msg or "").strip() or (msg_en or "").strip(),
        conversation_context="",
        reply_lang="en",
    )
    if order_ai:
        action = (order_ai.get("action") or "").strip()
        if action == "show_order_list":
            return True
        if action in ("order_id_help", "order_history_howto", "track_single_order", "not_order_topic"):
            return False
    if extract_product_id(text):
        return False
    if _text_is_product_id_lookup_context(text) and not re.search(
        r"\borders?\b|hist[o0]?ry|hsitory|meri\s+order|mere\s+order|my\s+orders?",
        text,
        re.I,
    ):
        return False
    return bool(_ORDER_WORD_RE.search(text))


def _groq_order_understand_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 220,
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=12)
        if res.status_code != 200:
            print(f"Order AI understand error: {res.text[:200]}")
            return None
        raw = res.json()["choices"][0]["message"]["content"]
        return json.loads(raw)
    except Exception as e:
        print(f"Order AI understand exception: {e}")
        return None


def ai_understand_order_request(
    user_msg: str,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> Optional[dict]:
    """
    Step 1: AI reads order API knowledge and classifies what the user wants.
    """
    api_kb = get_order_api_knowledge_context()
    system_prompt = f"""You are Welfog's order-support router. Read the ORDER API KNOWLEDGE BASE below.
Decide what the customer wants. Return ONLY valid JSON.

ORDER API KNOWLEDGE BASE:
\"\"\"
{api_kb}
\"\"\"

JSON SCHEMA:
{{
  "reasoning": "1-3 lines: what they asked in plain language",
  "action": "show_order_list" | "order_id_help" | "order_history_howto" | "track_single_order" | "order_details" | "order_invoice" | "not_order_topic" | "unclear",
  "confidence": "high" | "medium" | "low",
  "is_welfog_related": true/false
}}

RULES:
- show_order_list: user wants THEIR order list/purchases shown IN THIS CHAT (meri orders, order history dikhao, list, purchase kiye, bought from this id, products I ordered, unki list dede, jo maine mangaya/kharida).
- show_order_list: if user says "products" but means items they already bought (not catalog search), action=show_order_list.
- show_order_list: meta messages complaining bot wrongly searched catalog ("product kyu man rha", "order list maang rha") → show_order_list.
- order_id_help: user asks HOW/WHERE to find or get Order ID (kaise/kahan nikale, order id kaha milega) — NOT show_order_list.
- order_history_howto: steps to open order history in the app only (process/tareeka/kaha jake dekhun/app par) — NOT show_order_list.
- order_history_howto: "order history dekhne ka process bta de", "history nhi process maang rha", "kaha jake dekh skta hu app pr".
- track_single_order: ONE shipment status/ETA/timeline (often has or needs one order id).
- order_details: ONE order payment/product/address/summary (NOT courier timeline) — purchase-history-details API.
- order_invoice: ONE order bill/invoice/receipt download — verify ownership then invoice link.
- not_order_topic: products, deals, pincode, unrelated chat.
- unclear: ask to clarify — NEVER default to show_order_list if they only asked how to find order id.
- Read RECENT CONVERSATION: if user clarifies they asked about order ID steps (not the list), use order_id_help.
{language_reply_instruction(reply_lang)}
Return JSON only."""

    user_payload = user_msg
    if (conversation_context or "").strip():
        user_payload = (
            "RECENT CONVERSATION:\n"
            f"{conversation_context.strip()}\n\n"
            f"LATEST USER MESSAGE:\n{user_msg}"
        )

    data = _groq_order_understand_json(system_prompt, user_payload)
    if data:
        log_reasoning(
            f"Order AI understand: action={data.get('action')} "
            f"confidence={data.get('confidence')} — {data.get('reasoning', '')[:120]}"
        )
    return data


def run_order_ai_flow(
    original_msg: str,
    msg_en: str,
    user_id: str,
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    ai_route: Optional[dict] = None,
) -> OrderFlowResult:
    """
    Step 1: AI + API KB understanding.
    Step 2: Execute purchase-history API or KB help; delegate tracking to main flow.
    """
    from services.welfog_api import format_purchase_history_reply
    from services.translation_service import customer_reply_language

    comb = f"{original_msg} {msg_en}".strip()
    rl = reply_lang or customer_reply_language(original_msg)

    from utils.helpers import (
        _text_asks_how_to_view_order_history,
        _text_asks_order_history,
        _text_has_order_placement_intent,
        _user_asks_order_history_navigation_help,
        _user_rejects_viewing_wants_placement,
        message_asks_my_welfog_purchases,
        message_is_past_purchase_list_request,
    )

    from utils.helpers import _message_overrides_placement_followup, _text_has_refund_or_return_intent

    if _message_overrides_placement_followup(comb) or (
        _text_has_refund_or_return_intent(comb) and not _text_has_order_placement_intent(comb)
    ):
        log_reasoning("Order flow: return/refund — skip placement; defer to policy KB.")
        return OrderFlowResult(handled=False, intent="refund")

    if _text_has_order_placement_intent(comb) or _user_rejects_viewing_wants_placement(comb):
        log_reasoning("Order flow: user wants how to PLACE order — placement KB, not history.")
        body = (
            _localized_sysmsg("order_placement_help", original_msg, reply_lang=rl)
            or sysmsg("order_placement_help")
            or ""
        )
        return OrderFlowResult(handled=True, reply_html=body, intent="general")

    routed_intent = ((ai_route or {}).get("intent") or "").strip().lower()
    routed_channel = ((ai_route or {}).get("data_channel") or "").strip().lower()

    def _serve_order_history_howto(reason: str) -> OrderFlowResult:
        log_reasoning(reason)
        body = (
            _localized_sysmsg("order_history_help", original_msg, reply_lang=rl)
            or sysmsg("order_history_help")
            or ""
        )
        return OrderFlowResult(handled=True, reply_html=body, intent="general")

    def _serve_purchase_history(reason: str) -> OrderFlowResult:
        log_reasoning(reason)
        html = format_purchase_history_reply(user_id, page=1, append_only=False)
        return OrderFlowResult(handled=True, reply_html=html, intent="order_history")

    from utils.helpers import _text_wants_order_history_list_in_chat

    if _text_wants_order_history_list_in_chat(comb, conversation_context):
        return _serve_purchase_history(
            "User wants order list in chat — purchase-history API (not app steps)."
        )

    if _user_asks_order_history_navigation_help(comb, conversation_context):
        return _serve_order_history_howto(
            "Order history navigation/how-to — app steps (KB), not purchase-history API."
        )

    if routed_intent == "order_history" and not _text_asks_how_to_view_order_history(comb):
        if not _message_is_order_help_not_list(comb):
            return _serve_purchase_history(
                "Main router intent=order_history — purchase-history API (no 2nd Groq)."
            )

    wants_list = (
        _text_asks_order_history(comb)
        or message_asks_my_welfog_purchases(comb)
        or message_is_past_purchase_list_request(comb)
    )
    if wants_list and not _text_asks_how_to_view_order_history(comb):
        if not _message_is_order_help_not_list(comb):
            return _serve_purchase_history("Order history list (semantic) — purchase-history API.")

    user_line = original_msg.strip() or msg_en.strip()
    understanding = ai_understand_order_request(
        user_line, conversation_context=conversation_context, reply_lang=reply_lang
    )

    if not understanding:
        if _user_asks_order_history_navigation_help(comb):
            return _serve_order_history_howto(
                "Order AI unavailable — heuristic how-to for viewing history in app."
            )
        if routed_intent == "order_history" or routed_channel == "live_api":
            if not _message_is_order_help_not_list(comb):
                return _serve_purchase_history(
                    "Order specialist Groq unavailable — trust main router; purchase-history API."
                )
            return _serve_order_history_howto(
                "Order AI unavailable — how-to detected, not list."
            )
        if message_is_past_purchase_list_request(comb) and not _message_is_order_help_not_list(comb):
            return _serve_purchase_history(
                "Past-purchase list phrasing — purchase-history API (Groq fallback)."
            )
        log_reasoning("Order AI understand failed — not a confirmed order-list request.")
        return OrderFlowResult(handled=False)

    action = (understanding.get("action") or "unclear").strip().lower()
    conf = (understanding.get("confidence") or "medium").strip().lower()
    if not understanding.get("is_welfog_related", True):
        return OrderFlowResult(handled=False, intent="out_of_domain")

    if action == "order_id_help":
        return OrderFlowResult(handled=False)

    if action == "order_history_howto":
        return _serve_order_history_howto("Order AI: order_history_howto — app navigation steps.")

    if action == "show_order_list":
        html = format_purchase_history_reply(user_id, page=1, append_only=False)
        intro = ""
        if conf != "high":
            intro = (
                _localized_sysmsg("order_history_fetch_intro", original_msg, reply_lang=rl)
                or ""
            )
        if intro:
            html = f"<div style='margin-bottom:8px;color:#444;font-size:14px;'>{intro}</div>{html}"
        return OrderFlowResult(handled=True, reply_html=html, intent="order_history")

    if action == "track_single_order":
        log_reasoning("Order AI: delegate single-order tracking to main flow.")
        return OrderFlowResult(
            handled=False,
            intent="order",
            needs_order_id=True,
        )

    if action in ("order_details", "order_invoice"):
        from services.order_details_flow import run_order_details_ai_flow

        log_reasoning(f"Order AI: delegate to details/invoice flow (action={action}).")
        od = run_order_details_ai_flow(
            original_msg,
            msg_en,
            user_id,
            conversation_context=conversation_context,
            reply_lang=rl,
            ai_route=ai_route,
        )
        if od.handled:
            return od
        return OrderFlowResult(handled=False, intent="order", needs_order_id=True)

    if action == "not_order_topic":
        log_reasoning("Order AI: not an order-list/track topic — continue main pipeline.")
        return OrderFlowResult(handled=False, intent="general")

    if action == "unclear" and conf in ("low", "medium"):
        clarify = (
            _localized_sysmsg("order_history_clarify", original_msg, reply_lang=rl)
            or sysmsg("order_history_clarify")
            or ""
        )
        if clarify:
            return OrderFlowResult(handled=True, reply_html=clarify, intent="general")

    return OrderFlowResult(handled=False)
