"""
Order ID questions: AI understands first, reads order-id API KB, then KB help or purchase-history API.

Separate from order_history_flow (full order cards / history list).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

from services.kb_service import read_concatenated_kb_file_contents, sysmsg
from services.order_history_flow import OrderFlowResult, _localized_sysmsg
from services.translation_service import language_reply_instruction
from utils.reasoning_log import log_reasoning

ORDER_ID_KB_KEYS = ("welfog_api_order_id", "welfog_api_order_history", "welfog_api")

_ORDER_ID_TOPIC_RE = re.compile(
    r"(?:\border\s*id\b|\borderid\b|order-id|order\s+id\b|"
    r"\bid\b.{0,30}\border\b|\border\b.{0,30}\bid\b|"
    r"oid\b|order\s+code)",
    re.IGNORECASE,
)


def get_order_id_api_knowledge_context() -> str:
    blob = read_concatenated_kb_file_contents(list(ORDER_ID_KB_KEYS))
    return blob.strip() if blob.strip() else "(Order ID API knowledge missing.)"


def message_eligible_for_order_id_ai_flow(combined: str, msg_en: str, original_msg: str) -> bool:
    """Minimal gate — AI + API KB decides the exact action."""
    from utils.helpers import (
        extract_order_id,
        extract_product_id,
        _normalize_order_chat_text,
        _text_is_product_id_lookup_context,
        _text_is_order_tracking_intent,
    )

    text = _normalize_order_chat_text(f"{original_msg} {msg_en} {combined}").lower()
    if extract_product_id(text) or _text_is_product_id_lookup_context(text):
        return False
    if not _ORDER_ID_TOPIC_RE.search(text):
        return False
    # Delay/tracking without id focus → main order tracking pipeline
    if _text_is_order_tracking_intent(text) and not re.search(
        r"\b(id|orderid|order-id|oid)\b", text, re.I
    ):
        return False
    return True


def _groq_order_id_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
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
            return None
        return json.loads(res.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Order ID AI error: {e}")
        return None


def ai_understand_order_id_request(
    user_msg: str,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> Optional[dict]:
    api_kb = get_order_id_api_knowledge_context()
    system_prompt = f"""You are Welfog's order-ID specialist. Read the ORDER ID API KNOWLEDGE BASE.
Return ONLY valid JSON.

ORDER ID API KNOWLEDGE BASE:
\"\"\"
{api_kb}
\"\"\"

JSON SCHEMA:
{{
  "reasoning": "1-3 lines in plain language",
  "action": "show_my_order_ids" | "order_id_where_how" | "track_with_order_id" | "ask_order_id_for_tracking" | "not_order_id_topic" | "unclear",
  "confidence": "high" | "medium" | "low",
  "extracted_order_id": "digits/alphanumeric order id if user provided one, else empty string",
  "is_welfog_related": true/false
}}

RULES:
- show_my_order_ids: user wants their order IDs listed/shown in chat (dikhao, list, batao mere id).
- order_id_where_how: only HOW/WHERE to find id (nikalu, milega, kahan) — NOT full list in chat.
- track_with_order_id: specific id + status/track/kab aayega.
- ask_order_id_for_tracking: wants tracking but no id in message.
- NEVER show_my_order_ids for pure "kaise nikalu" without wanting list.
- NEVER order_id_where_how when user clearly says "dikhao/list/saare id".
- Read RECENT CONVERSATION for follow-ups.
{language_reply_instruction(reply_lang)}
Return JSON only."""

    user_payload = user_msg
    if (conversation_context or "").strip():
        user_payload = (
            f"RECENT CONVERSATION:\n{conversation_context.strip()}\n\n"
            f"LATEST USER MESSAGE:\n{user_msg}"
        )
    data = _groq_order_id_json(system_prompt, user_payload)
    if data:
        log_reasoning(
            f"Order ID AI: action={data.get('action')} "
            f"id={data.get('extracted_order_id', '')} — {(data.get('reasoning') or '')[:100]}"
        )
    return data


def _message_wants_id_list(combined: str) -> bool:
    tl = f" {combined.lower()} "
    if any(
        x in tl
        for x in (
            "dikhao", "dikha", "dikhe", "batao", "btao", "bata", "list", "show",
            "dekh", "dekho", "saare", "sab ", "all my", "mere order id", "meri order id",
        )
    ):
        return True
    if re.search(r"\border\s*id\b", tl) and any(x in tl for x in ("dikhao", "dikha", "show", "list", "bata")):
        return True
    return False


def _message_wants_id_where_how(combined: str) -> bool:
    tl = f" {combined.lower()} "
    if not _ORDER_ID_TOPIC_RE.search(tl):
        return False
    if _message_wants_id_list(tl):
        return False
    return any(
        x in tl
        for x in (
            "kaise", "kese", "kahan", "kaha", "where", "how", "nikal", "nikaal",
            "nikale", "nikalte", "milega", "milta", "pata", "find", "api se",
        )
    )


def run_order_id_ai_flow(
    original_msg: str,
    msg_en: str,
    user_id: str,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> OrderFlowResult:
    from services.welfog_api import format_order_ids_reply
    from services.translation_service import customer_reply_language
    from utils.helpers import _normalize_order_chat_text, extract_order_id

    comb = _normalize_order_chat_text(f"{original_msg} {msg_en}".strip())
    rl = reply_lang or customer_reply_language(original_msg)

    user_line = original_msg.strip() or msg_en.strip()
    from utils.helpers import (
        _conversation_bot_offered_order_id_or_tracking,
        _message_is_order_id_followup_submission,
    )

    if _message_is_order_id_followup_submission(user_line, conversation_context) and _conversation_bot_offered_order_id_or_tracking(
        conversation_context
    ):
        log_reasoning("Order ID follow-up — delegate to main live API flow.")
        return OrderFlowResult(handled=False)

    understanding = ai_understand_order_id_request(
        user_line, conversation_context=conversation_context, reply_lang=reply_lang
    )

    action = ""
    conf = "medium"
    extracted = ""

    if understanding:
        action = (understanding.get("action") or "").strip().lower()
        conf = (understanding.get("confidence") or "medium").strip().lower()
        if not understanding.get("is_welfog_related", True):
            return OrderFlowResult(handled=False, intent="out_of_domain")
        extracted = (understanding.get("extracted_order_id") or "").strip()
    else:
        log_reasoning("Order ID AI failed — heuristic fallback.")
        if _message_wants_id_list(comb):
            action = "show_my_order_ids"
        elif _message_wants_id_where_how(comb):
            action = "order_id_where_how"
        else:
            action = "unclear"

    from utils.helpers import resolve_order_id_for_tracking, _user_announcing_will_provide_order_id

    inline_id = resolve_order_id_for_tracking(
        original_msg.strip() or msg_en.strip(),
        conversation_context,
        ai_extracted=extracted,
    )
    if _user_announcing_will_provide_order_id(comb):
        action = "ask_order_id_for_tracking"
    elif inline_id and any(
        x in comb.lower()
        for x in ("track", "status", "kab", "delivery", "kahan hai", "aaega", "aayega")
    ):
        action = "track_with_order_id"

    if action == "show_my_order_ids" and _message_wants_id_where_how(comb) and not _message_wants_id_list(comb):
        log_reasoning("Override: show_my_order_ids → order_id_where_how (how-to, not list).")
        action = "order_id_where_how"

    if action == "order_id_where_how":
        body = _localized_sysmsg("order_id_help", original_msg, reply_lang=rl) or sysmsg("order_id_help") or ""
        extra = (
            "<div style='margin-top:10px;font-size:13px;color:#555;'>"
            "You can also ask: <b>show my order ids</b> — I will list them from your account here."
            "</div>"
        )
        return OrderFlowResult(handled=True, reply_html=body + extra, intent="general")

    if action == "show_my_order_ids":
        html = format_order_ids_reply(user_id, page=1)
        return OrderFlowResult(handled=True, reply_html=html, intent="order_id")

    if action == "track_with_order_id" and inline_id:
        from services.order_tracking_flow import _fetch_and_format_tracking

        body = _fetch_and_format_tracking(str(inline_id), user_id, original_msg, rl)
        return OrderFlowResult(
            handled=True,
            reply_html=body,
            intent="order",
            order_id=str(inline_id),
        )

    if action == "ask_order_id_for_tracking":
        body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order") or ""
        return OrderFlowResult(handled=True, reply_html=body, intent="order", needs_order_id=True)

    if action == "not_order_id_topic":
        return OrderFlowResult(handled=False)

    if action == "unclear":
        body = (
            _localized_sysmsg("order_id_clarify", original_msg, reply_lang=rl)
            or (
                "Do you want to <b>see your Order IDs</b> here, or learn <b>where to find</b> "
                "an Order ID (app / SMS / email)? Say what you need."
            )
        )
        return OrderFlowResult(handled=True, reply_html=body, intent="general")

    return OrderFlowResult(handled=False)
