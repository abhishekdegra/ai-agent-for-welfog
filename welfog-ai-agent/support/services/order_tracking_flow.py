"""

Order tracking: AI understands → reads tracking API KB → live welfog_track API.



Handles thousands of phrasings (Hinglish, typos, regional languages) via:

  1) API knowledge base + example playbook

  2) Groq JSON specialist (primary)

  3) Deterministic heuristic merge (always runs — safe when LLM wrong/offline)

"""

from __future__ import annotations



import json

import os

import re

from typing import Any, Optional



import requests



from services.kb_service import sysmsg

from services.order_history_flow import OrderFlowResult, _localized_sysmsg

from services.translation_service import language_reply_instruction

from utils.reasoning_log import log_reasoning



ORDER_TRACKING_KB_KEYS = (

    "welfog_api_order_tracking",

    "welfog_api_order_tracking_examples",

    "welfog_api_order_id",

    "welfog_api_order_history",

    "welfog_api",

)



# Loose gate: might be tracking — AI + heuristics decide action

_TRACKING_LOOSE_RE = re.compile(

    r"(?:\borders?\b|\bordr\w*\b|\boorder\b).{0,48}(?:track|trck|trak|traking|teck|status|deliver|ship|"

    r"kab|kahan|kaha|aaya|aayega|aayega|milega|pending|delay|parcel|package|courier|stuck|update|jankari)|"

    r"(?:track|trck|trak|status|deliver|ship|parcel|package|courier).{0,48}(?:\borders?\b|\bordr\w*\b)|"

    r"\b(?:track|trck)\s+(?:kr|kar|karo|krke|krna)\b",

    re.IGNORECASE,

)





def get_order_tracking_api_knowledge_context() -> str:

    from services.kb_service import read_concatenated_kb_file_contents



    blob = read_concatenated_kb_file_contents(list(ORDER_TRACKING_KB_KEYS))

    return blob.strip() if blob.strip() else "(Order tracking API knowledge missing.)"





def message_eligible_for_order_tracking_ai_flow(

    combined: str, msg_en: str, original_msg: str

) -> bool:

    """Minimal gate — real routing is AI + heuristics + API KB."""

    from utils.helpers import (

        _normalize_order_chat_text,

        _text_has_pincode_delivery_intent,

        _text_has_order_placement_intent,

        _text_is_order_tracking_intent,

        extract_product_id,

        _text_is_product_id_lookup_context,

    )



    text = _normalize_order_chat_text(f"{original_msg} {msg_en} {combined}")
    if _text_has_pincode_delivery_intent(f"{original_msg} {msg_en} {combined}"):
        return False

    if _text_is_product_id_lookup_context(text) or extract_product_id(text):

        if not _text_is_order_tracking_intent(text):

            return False

    if _text_has_order_placement_intent(text):

        return False

    if _text_is_order_tracking_intent(text):

        return True

    return bool(_TRACKING_LOOSE_RE.search(text))





def heuristic_understand_order_tracking(

    user_msg: str,

    conversation_context: str = "",

) -> dict[str, Any]:

    """

    Deterministic understanding — always available (Groq down / wrong JSON).

    Same schema as ai_understand_order_tracking.

    """

    from utils.helpers import (

        _normalize_order_chat_text,

        _text_is_order_tracking_intent,

        _text_is_tracking_howto_request,

        _text_needs_order_id_for_tracking,

        _user_announcing_will_provide_order_id,

        _user_asks_hypothetical_tracking_capability,

        resolve_order_id_for_tracking,

    )



    msg = _normalize_order_chat_text(user_msg or "")

    base = {

        "reasoning": "heuristic",

        "action": "not_tracking",

        "extracted_order_id": "",

        "is_welfog_related": True,

        "confidence": "high",

    }



    if not msg.strip():

        return base



    trackingish = _text_is_order_tracking_intent(msg) or bool(_TRACKING_LOOSE_RE.search(msg))

    oid = resolve_order_id_for_tracking(msg, conversation_context)



    if not trackingish and not oid:

        return base



    if _user_announcing_will_provide_order_id(msg):

        base["action"] = "ask_order_id"

        base["reasoning"] = "User will send order id — ask first, no API."

        return base

    if _user_asks_hypothetical_tracking_capability(msg):

        base["action"] = "capability_confirm"

        base["reasoning"] = "User asks if bot can track another id — confirm, no API yet."

        return base



    if oid:

        base["action"] = "track_live"

        base["extracted_order_id"] = str(oid)

        base["reasoning"] = f"Order id in latest user message: {oid}"

        return base



    if trackingish and (
        _text_needs_order_id_for_tracking(msg)
        or re.search(r"\b(?:track|trck)\s+(?:karo|kr|krde)\b", f" {msg.lower()} ")
    ):
        base["action"] = "ask_order_id"
        base["reasoning"] = "Live tracking request — need order id from user."
        return base

    if _text_is_tracking_howto_request(msg):
        base["action"] = "tracking_howto"
        base["reasoning"] = "How-to track / find order id — steps only."
        return base

    if trackingish:
        base["action"] = "ask_order_id"
        base["reasoning"] = "Tracking intent without order id in latest message."
        return base



    return base





def _merge_tracking_understanding(

    ai_data: Optional[dict],

    heuristic: dict[str, Any],

    user_msg: str,

    conversation_context: str = "",

) -> dict[str, Any]:

    """

    Merge Groq + heuristics — heuristics win on safety (no id → never track_live).

    """

    from utils.helpers import (

        _user_announcing_will_provide_order_id,

        resolve_order_id_for_tracking,

    )



    h = dict(heuristic or {})

    a = dict(ai_data or {}) if ai_data else {}



    oid = resolve_order_id_for_tracking(

        user_msg, conversation_context, ai_extracted=(a.get("extracted_order_id") or "").strip()

    )



    if _user_announcing_will_provide_order_id(user_msg):

        return {

            "reasoning": "merge: user announcing id — ask_order_id",

            "action": "ask_order_id",

            "extracted_order_id": "",

            "is_welfog_related": True,

            "confidence": "high",

        }

    from utils.helpers import _user_asks_hypothetical_tracking_capability

    if _user_asks_hypothetical_tracking_capability(user_msg):

        return {

            "reasoning": "merge: hypothetical tracking question — no stale id API",

            "action": "capability_confirm",

            "extracted_order_id": "",

            "is_welfog_related": True,

            "confidence": "high",

        }



    if oid:

        return {

            "reasoning": (a.get("reasoning") or h.get("reasoning") or "")[:200],

            "action": "track_live",

            "extracted_order_id": str(oid),

            "is_welfog_related": True,

            "confidence": "high",

        }



    h_action = (h.get("action") or "").strip().lower()

    a_action = (a.get("action") or "").strip().lower()



    if a_action == "track_live":

        a_action = "ask_order_id"



    if not a:

        return h



    if h_action in ("ask_order_id", "tracking_howto") and a_action == "track_live":

        return h



    if a_action in ("ask_order_id", "tracking_howto", "not_tracking") and h_action in (

        "ask_order_id",

        "tracking_howto",

    ):

        if h_action == "tracking_howto" or a_action == "tracking_howto":

            pick = "tracking_howto"

        else:

            pick = "ask_order_id"

        return {

            "reasoning": f"merge: {pick} ({(a.get('reasoning') or '')[:80]})",

            "action": pick,

            "extracted_order_id": "",

            "is_welfog_related": bool(a.get("is_welfog_related", True)),

            "confidence": a.get("confidence") or h.get("confidence") or "medium",

        }



    return {

        "reasoning": (a.get("reasoning") or h.get("reasoning") or "")[:200],

        "action": a_action or h_action or "ask_order_id",

        "extracted_order_id": "",

        "is_welfog_related": bool(a.get("is_welfog_related", h.get("is_welfog_related", True))),

        "confidence": a.get("confidence") or h.get("confidence") or "medium",

    }





def _groq_order_tracking_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    from services.llm_providers import llm_json_chat_completion

    try:
        return llm_json_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=380,
            timeout_sec=14,
            max_attempts=3,
        )
    except Exception as e:

        print(f"Order tracking AI error: {e}")

        return None





def ai_understand_order_tracking(

    user_msg: str,

    conversation_context: str = "",

    reply_lang: str = "en",

) -> Optional[dict]:

    api_kb = get_order_tracking_api_knowledge_context()

    system_prompt = f"""You are Welfog's order-tracking specialist. Read the ORDER TRACKING API KNOWLEDGE BASE (includes phrasing examples).

Return ONLY valid JSON.



ORDER TRACKING API KNOWLEDGE BASE:

\"\"\"

{api_kb[:12000]}

\"\"\"



JSON SCHEMA:

{{

  "reasoning": "1-3 lines: what user wants + whether id is in LATEST message",

  "action": "track_live" | "ask_order_id" | "tracking_howto" | "capability_confirm" | "not_tracking",

  "extracted_order_id": "only if digits/id appear in LATEST user message, else empty",

  "is_welfog_related": true/false,

  "confidence": "high" | "medium" | "low"

}}



DECISION RULES (thousands of phrasings — apply to LATEST user message only):

1) track_live — user pasted order id (2605150, 302032, etc.) AND wants status/track/ETA/delivery/delay update.

2) ask_order_id — wants track/status/delay ("order nahi aaya", "kab aayega", "track karo") but NO id in latest message.

3) ask_order_id — user says they WILL send id ("de raha hu", "de ra hu", "id batata hu") without digits yet.

4) tracking_howto — only how/where to track or find order id ("kaise track karu", "order id kahan") — no live lookup.

5) not_tracking — product shopping, order history list, place-new-order help.



NEVER track_live without id in latest user message. NEVER copy id from assistant's previous order status card.

Typos: ordder=order, trck/trak/teck=track.



EXAMPLES:

- "mera order track karo" → ask_order_id

- "order id de raha hu track krke bata" → ask_order_id (no digits yet)

- "ye rhi id 2605150 status bata" → track_live, extracted_order_id=2605150

- "112233" after bot asked for id → track_live

- "track kaise karu" → tracking_howto

{language_reply_instruction(reply_lang)}

JSON only."""



    user_payload = user_msg

    if (conversation_context or "").strip():

        user_payload = (

            f"RECENT CONVERSATION:\n{conversation_context.strip()[-2000:]}\n\n"

            f"LATEST USER MESSAGE:\n{user_msg}"

        )

    data = _groq_order_tracking_json(system_prompt, user_payload)

    if data:

        log_reasoning(

            f"Order tracking AI: action={data.get('action')} "

            f"id={data.get('extracted_order_id', '')!r} conf={data.get('confidence')} "

            f"— {(data.get('reasoning') or '')[:90]}"

        )

    return data





def _tracking_help_html(original_msg: str, reply_lang: str) -> str:

    body = _localized_sysmsg("tracking_help", original_msg, reply_lang=reply_lang) or sysmsg("tracking_help")

    return body or sysmsg("how_can_i_help") or ""





def _fetch_and_format_tracking(

    order_id: str,

    user_id: str,

    original_msg: str,

    reply_lang: str,

) -> str:

    from services.welfog_api import fetch_welfog_order_tracking_for_user, format_order_tracking_reply



    track_data, track_err = fetch_welfog_order_tracking_for_user(order_id, user_id)

    if track_data:

        return format_order_tracking_reply(track_data, order_id, lang=reply_lang)

    if track_err == "login_required":

        return _localized_sysmsg("order_track_login_required", original_msg, reply_lang=reply_lang) or (

            "Please log in to Welfog and open chat from your account to see order status."

        )

    if track_err == "not_owned":

        return _localized_sysmsg("order_track_not_owned", original_msg, reply_lang=reply_lang) or (

            "This Order ID does not match your account. Use the ID from your SMS/email or My Orders."

        )

    if track_err == "unverified":

        return _localized_sysmsg("order_track_unverified", original_msg, reply_lang=reply_lang) or (

            "Could not verify this Order ID with your account. Try the account that placed the order."

        )

    hint = (

        _localized_sysmsg(

            "order_track_not_found", original_msg, reply_lang=reply_lang, order_id=order_id

        )

        or _localized_sysmsg("order_track_not_found", original_msg, reply_lang=reply_lang)

    )

    if hint and order_id and str(order_id) not in hint:

        hint = hint.replace("</div>", f"<br><br>Checked ID: <b>{order_id}</b></div>", 1)

    return hint or (

        f"We could not find Order ID <b>{order_id}</b>. Check confirmation SMS/email or My Orders."

    )





def run_order_tracking_ai_flow(

    original_msg: str,

    msg_en: str,

    user_id: str,

    conversation_context: str = "",

    reply_lang: str = "en",

) -> OrderFlowResult:

    from utils.helpers import (

        _normalize_order_chat_text,

        _text_is_order_tracking_intent,

        _text_needs_order_id_for_tracking,

        _user_announcing_will_provide_order_id,

        resolve_order_id_for_tracking,

    )



    comb = _normalize_order_chat_text(f"{original_msg} {msg_en}".strip())

    user_line = original_msg.strip() or msg_en.strip()

    rl = reply_lang or "en"

    from services.order_details_flow import understand_single_order_request

    sub = understand_single_order_request(
        original_msg, msg_en, conversation_context, reply_lang=rl
    )
    if (sub.get("goal") or "").strip() in ("order_invoice", "order_details"):
        log_reasoning(
            f"Tracking flow skipped — classified as {sub.get('goal')} "
            f"(source={sub.get('source')})."
        )
        return OrderFlowResult(handled=False)

    if not message_eligible_for_order_tracking_ai_flow(comb, msg_en, original_msg):

        return OrderFlowResult(handled=False)

    from services.support_scope import (
        build_other_company_support_decline,
        message_mentions_other_company_support,
    )

    if message_mentions_other_company_support(original_msg, msg_en, conversation_context):
        log_reasoning("Order tracking blocked — other company order/support request.")
        return OrderFlowResult(
            handled=True,
            reply_html=build_other_company_support_decline(original_msg, reply_lang=rl),
            intent="out_of_domain",
        )

    if _user_announcing_will_provide_order_id(user_line):

        log_reasoning("User will provide order id later — ask, do not call track API.")

        body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order") or ""

        return OrderFlowResult(

            handled=True,

            reply_html=body,

            intent="order",

            needs_order_id=True,

        )

    from utils.helpers import (
        _conversation_awaiting_order_id,
        _message_is_order_id_followup_submission,
        _user_asks_hypothetical_tracking_capability,
        _user_demands_immediate_track_action,
        resolve_order_id_for_tracking,
    )

    if _conversation_awaiting_order_id(conversation_context):
        proceed_id = resolve_order_id_for_tracking(
            user_line,
            conversation_context,
            bot_awaiting_order_id=True,
        )
        if proceed_id and (
            _user_demands_immediate_track_action(user_line)
            or _message_is_order_id_followup_submission(user_line, conversation_context)
        ):
            log_reasoning(
                f"Order-ID handoff in tracking flow — live track id={proceed_id}."
            )
            body = _fetch_and_format_tracking(
                proceed_id, user_id, original_msg, rl
            )
            return OrderFlowResult(
                handled=True,
                reply_html=body,
                intent="order",
                order_id=proceed_id,
            )

    if _user_asks_hypothetical_tracking_capability(user_line):

        log_reasoning("Hypothetical tracking question — confirm capability, no stale API replay.")

        body = (
            _localized_sysmsg("order_tracking_capability_yes", original_msg, reply_lang=rl)
            or sysmsg("order_tracking_capability_yes")
            or _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order")
        )

        return OrderFlowResult(

            handled=True,

            reply_html=body or "",

            intent="order",

            needs_order_id=True,

        )



    heuristic = heuristic_understand_order_tracking(user_line, conversation_context)

    ai_raw = ai_understand_order_tracking(

        user_line,

        conversation_context=conversation_context,

        reply_lang=reply_lang,

    )

    understanding = _merge_tracking_understanding(ai_raw, heuristic, user_line, conversation_context)



    if not understanding.get("is_welfog_related", True):

        return OrderFlowResult(handled=False)



    action = (understanding.get("action") or "").strip().lower()

    extracted = (understanding.get("extracted_order_id") or "").strip()



    inline_id = resolve_order_id_for_tracking(

        user_line, conversation_context, ai_extracted=extracted

    )

    if inline_id and action != "tracking_howto":

        action = "track_live"

    elif not inline_id and action == "track_live":

        action = "ask_order_id"



    log_reasoning(

        f"Order tracking merged action={action} id={inline_id or ''!r} "

        f"(heuristic={heuristic.get('action')}, ai={ai_raw.get('action') if ai_raw else 'none'})"

    )



    if action == "tracking_howto" or (

        action == "ask_order_id" and not _text_needs_order_id_for_tracking(comb)

    ):

        body = _tracking_help_html(original_msg, rl)

        foot = sysmsg("order_tracking_optional_id_footer")

        if foot and body:

            body = f"{body.rstrip()}<br><br>{foot}"

        return OrderFlowResult(handled=True, reply_html=body, intent="order")



    if action == "capability_confirm":
        from services.semantic_answer_plan import try_semantic_grounded_reply

        grounded = try_semantic_grounded_reply(
            original_msg,
            msg_en,
            {
                "intent": "order",
                "data_channel": "live_api",
                "needs_order_id": True,
                "answer_strategy": "kb_then_ai",
                "kb_keys": ["welfog_api_order_tracking", "shipping", "faqs"],
            },
            conversation_context=conversation_context,
            reply_lang=rl,
            handler="order_tracking_api",
        )
        body = grounded or (
            _localized_sysmsg("order_tracking_capability_yes", original_msg, reply_lang=rl)
            or sysmsg("order_tracking_capability_yes")
            or ""
        )

        return OrderFlowResult(

            handled=True,

            reply_html=body,

            intent="order",

            needs_order_id=True,

        )

    if action == "ask_order_id" or (not inline_id and _text_is_order_tracking_intent(comb)):

        from services.semantic_answer_plan import try_semantic_grounded_reply

        grounded = try_semantic_grounded_reply(
            original_msg,
            msg_en,
            {
                "intent": "order",
                "data_channel": "live_api",
                "needs_order_id": True,
                "answer_strategy": "kb_then_ai",
                "kb_keys": ["welfog_api_order_tracking", "shipping", "faqs"],
            },
            conversation_context=conversation_context,
            reply_lang=rl,
            handler="order_tracking_api",
        )
        body = grounded or (
            _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order") or ""
        )

        return OrderFlowResult(

            handled=True,

            reply_html=body,

            intent="order",

            needs_order_id=True,

        )



    if inline_id and action in ("track_live", "ask_order_id", ""):

        log_reasoning(f"Live order track API for id={inline_id}")

        body = _fetch_and_format_tracking(inline_id, user_id, original_msg, rl)

        return OrderFlowResult(

            handled=True,

            reply_html=body,

            intent="order",

            order_id=str(inline_id),

        )



    if action == "not_tracking":

        return OrderFlowResult(handled=False)



    body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order") or ""

    return OrderFlowResult(handled=True, reply_html=body, intent="order", needs_order_id=True)


