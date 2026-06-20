"""
Pincode / delivery serviceability: AI understands → KB → live check_pincode API.
"""
from __future__ import annotations

import json
import os
import re
from html import escape as html_escape
from typing import Any, Optional

import requests

from services.kb_service import sysmsg
from services.order_history_flow import OrderFlowResult, _localized_sysmsg
from services.translation_service import language_reply_instruction
from utils.reasoning_log import log_reasoning

PINCODE_KB_KEYS = ("welfog_api_pincode_delivery", "welfog_api", "shipping", "faqs")

_PIN_LOOSE_RE = re.compile(
    r"(?:pincode|pin\s*code|delivery|delevery|delivry|deliver|service|pahucha|phocha|pahunchega|"
    r"serviceability|ship\s*to|milega\s+delivery|delivery\s+milegi|ho\s+jayegi|ho\s+jayega)",
    re.I,
)


def get_pincode_api_knowledge_context() -> str:
    from services.kb_service import read_concatenated_kb_file_contents

    blob = read_concatenated_kb_file_contents(list(PINCODE_KB_KEYS))
    return blob.strip() if blob.strip() else "(Pincode delivery API knowledge missing.)"


def message_eligible_for_pincode_ai_flow(
    combined: str, msg_en: str, original_msg: str, conversation_context: str = ""
) -> bool:
    from utils.helpers import (
        _conversation_awaiting_order_id,
        _text_has_delivery_serviceability_intent,
        _text_has_pincode_delivery_intent,
        extract_pincode_preferred_from_message,
    )

    if _conversation_awaiting_order_id(conversation_context):
        return False
    text = f"{original_msg} {msg_en} {combined}".strip()
    if _text_has_pincode_delivery_intent(text, conversation_context):
        return True
    if _text_has_delivery_serviceability_intent(text, conversation_context):
        return True
    pin = extract_pincode_preferred_from_message(text, conversation_context)
    return bool(pin and "welfog" in text.lower())


def _groq_pincode_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    from services.llm_providers import llm_json_chat_completion

    return llm_json_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=200,
        timeout_sec=12,
        max_attempts=2,
    )


def heuristic_understand_pincode(user_msg: str, conversation_context: str = "") -> dict[str, Any]:
    from utils.helpers import (
        _conversation_in_pincode_delivery_flow,
        _text_has_pincode_delivery_intent,
        _text_is_pincode_serviceability_question,
        extract_pincode_from_text,
        resolve_pincode_for_check,
    )

    msg = (user_msg or "").strip()
    base = {
        "reasoning": "heuristic",
        "action": "not_pincode_topic",
        "extracted_pincode": "",
        "is_welfog_related": True,
    }
    if not (
        _text_has_pincode_delivery_intent(msg, conversation_context)
        or _text_is_pincode_serviceability_question(msg, conversation_context)
    ):
        return base

    pin = resolve_pincode_for_check(msg, conversation_context)
    if pin:
        base["action"] = "check_pin_live"
        base["extracted_pincode"] = pin
        base["reasoning"] = f"PIN {pin} for delivery check"
        return base

    try:
        from services.location_delivery_resolver import (
            extract_place_query_from_delivery_message,
            geocode_city_to_pincode,
        )

        place = extract_place_query_from_delivery_message(msg)
        if place:
            geo = geocode_city_to_pincode(place)
            if geo.get("pincode"):
                base["action"] = "check_pin_live"
                base["extracted_pincode"] = str(geo["pincode"])
                base["reasoning"] = f"City/area {place} → PIN {geo['pincode']} (geocode)"
                return base
    except ImportError:
        pass

    if _conversation_in_pincode_delivery_flow(conversation_context) or _PIN_LOOSE_RE.search(msg):
        base["action"] = "ask_pin"
        base["reasoning"] = "Delivery/pincode topic but no PIN or resolvable place"
    return base


def ai_understand_pincode(user_msg: str, conversation_context: str = "", reply_lang: str = "en") -> Optional[dict]:
    api_kb = get_pincode_api_knowledge_context()
    system_prompt = f"""You are Welfog's pincode/delivery serviceability specialist.
Return ONLY JSON.

KNOWLEDGE:
\"\"\"
{api_kb[:8000]}
\"\"\"

JSON: {{"reasoning":"...","action":"check_pin_live"|"ask_pin"|"not_pincode_topic","extracted_pincode":"","is_welfog_related":true}}

RULES:
- check_pin_live: user wants delivery/service at a PIN — digits may be ANYWHERE in a long Tamil/Hindi/English sentence (e.g. friend in 302034 area, Welfog order pannanum). Extract that PIN.
- ask_pin: delivery question but place is too vague / unknown — ask for 6-digit PIN.
- For clear city/town names (Jaipur, Kota, Udaipur, Delhi) without PIN — still pincode topic; upstream geocoder resolves city → PIN (not ask_pin).
- City or area name alone is NOT an order id — never ask_order_id or track_live.
- NEVER treat 6-digit PIN as Order ID for tracking.
- If user says they ALREADY sent pincode ("de di thi", "abhi diya", "upar wala pin") but this message has no digits, read RECENT CONVERSATION User lines and use the LATEST 6-digit PIN they typed — then check_pin_live.
- If user CORRECTS PIN ("302034 ki nahi, 111111 ki") use the NEW pin from the latest message, not the old one.
- "pincode 1111111" (7 digits) → use 111111 for API check.
- NEVER answer with long shipping policy text only — action must be check_pin_live when a PIN is known.
- "302034 per?" = check_pin_live with 302034.
- "pincode tha / yh pincode" = still pincode, not product search.
{language_reply_instruction(reply_lang)}
JSON only."""

    payload = user_msg
    if conversation_context.strip():
        payload = f"RECENT CONVERSATION:\n{conversation_context.strip()[-2000:]}\n\nLATEST:\n{user_msg}"
    data = _groq_pincode_json(system_prompt, payload)
    if data:
        log_reasoning(
            f"Pincode AI: action={data.get('action')} pin={data.get('extracted_pincode', '')!r}"
        )
    return data


def _pin_localized(key: str, user_msg: str, reply_lang: str = "en", **fmt) -> str:
    from services.order_history_flow import _localized_sysmsg

    return _localized_sysmsg(key, user_msg, reply_lang=reply_lang, **fmt) or ""


def _pincode_reply_scenario(
    comb: str,
    conversation_context: str = "",
) -> tuple[str, str]:
    """(scenario, malformed_pin) — scenario: need_pin | invalid_pin | meta_question."""
    from utils.helpers import (
        extract_malformed_pincode_attempt,
        message_is_pincode_meta_or_hypothetical,
    )

    bad = extract_malformed_pincode_attempt(comb, conversation_context)
    if bad:
        return "invalid_pin", bad
    if message_is_pincode_meta_or_hypothetical(comb):
        return "meta_question", ""
    return "need_pin", ""


def _wrap_pincode_ai_html(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if "<" in t and ">" in t:
        return t
    return f"<div style='color:#333;line-height:1.55;'>{t}</div>"


def ai_pincode_conversational_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    scenario: str = "need_pin",
    malformed_pin: str = "",
) -> str:
    """
    Contextual delivery/PIN reply in the user's language — not a fixed template.
  Falls back to empty string so caller can use deterministic sysmsg.
    """
    if (os.getenv("PINCODE_AI_CONVERSATIONAL", "1") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import (
        finalize_customer_reply,
        language_reply_instruction,
        resolve_customer_reply_lang,
    )

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not comb:
        return ""

    from utils.helpers import resolve_pincode_for_check

    pin_known = resolve_pincode_for_check(
        original_msg, conversation_context, msg_en=msg_en
    )
    if pin_known:
        log_reasoning(
            f"Pincode AI conversational skipped — valid PIN {pin_known} present for API."
        )
        return ""

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    api_kb = get_pincode_api_knowledge_context()[:4000]
    compact_ctx = _compact_conversation_context(conversation_context or "", 1800)
    user_line = _trim_text_mid(original_msg or msg_en or comb, 700)

    scenario = (scenario or "need_pin").strip().lower()
    facts = (
        "Welfog checks delivery via live API with a valid 6-digit PIN. "
        "Clear city names (Jaipur, Kota) are auto-mapped to a representative PIN — user does not need to know PIN. "
        "Vague/unknown place names need an explicit 6-digit PIN. "
        "Wrong or incomplete PIN cannot give a real yes/no."
    )
    if scenario == "invalid_pin" and malformed_pin:
        facts += f" User typed: {malformed_pin} — not a valid 6-digit PIN."
    elif scenario == "meta_question":
        facts += " Answer their hypothetical/rule question honestly in plain language, then ask for PIN if they want a real check."

    system_prompt = f"""You are Welfog customer support for delivery / PIN serviceability.
Return ONLY JSON: {{"reasoning":"1 line","response":"customer reply"}}

KNOWLEDGE (facts):
\"\"\"
{api_kb}
\"\"\"

CORE FACTS: {facts}

RULES:
- Write "response" in the SAME language, tone, and script as the user's LATEST message (Hinglish if they use Hinglish, Hindi, English, etc.).
- 2-4 short sentences max. Sound human and polite — like a helpful agent, not a robot.
- Acknowledge what THEY actually said (friend's place, wrong PIN question, area name, etc.) — do NOT copy a generic script.
- NEVER repeat or echo the user's latest message verbatim (no parroting their words back).
- Do NOT start every reply with "Samajh gaya — aap puchh rahe ho ki Welfog wahan deliver..." — vary naturally.
- Do NOT mention "city name" unless the user mentioned a city or area name.
- Do NOT mention "dost" / friend unless the user did.
- For need_pin: politely ask for the 6-digit PIN of that delivery address so you can check live.
- For invalid_pin: say their number is not a complete valid 6-digit PIN; ask for the correct one.
- For meta_question: answer their what-if (e.g. wrong PIN → cannot trust result; without PIN → cannot check) then invite a valid 6-digit PIN if they want a real answer.
- No Order ID, no product search, no long policy dump.
- Simple HTML allowed in response: <b> for PIN example only; keep light.
{language_reply_instruction(rl)}
JSON only."""

    user_payload = f"SCENARIO: {scenario}\n"
    if malformed_pin:
        user_payload += f"INVALID_PIN_TYPED: {malformed_pin}\n"
    if compact_ctx:
        user_payload += f"\nRECENT CONVERSATION:\n{compact_ctx}\n"
    user_payload += f"\nLATEST USER MESSAGE:\n{user_line}"

    providers = _llm_classifier_provider_chain()
    if not providers:
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]
    data = _llm_json_with_provider_fallback(
        providers, messages, max_tokens=220, timeout_sec=12, max_attempts=2
    )
    if not data:
        return ""
    raw = (data.get("response") or "").strip()
    if not raw:
        return ""
    plain_user = re.sub(r"\s+", " ", (original_msg or msg_en or comb).lower().strip())
    plain_reply = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw).lower().strip())
    if plain_user and len(plain_user) >= 8 and (
        plain_user == plain_reply or plain_user in plain_reply
    ):
        log_reasoning("Pincode AI reply rejected — echoed user message verbatim.")
        return ""
    log_reasoning(
        f"Pincode AI conversational reply (scenario={scenario}): {(data.get('reasoning') or '')[:120]}"
    )
    body = _wrap_pincode_ai_html(raw)
    return finalize_customer_reply(body, original_msg or msg_en, rl)


def build_pincode_missing_or_invalid_reply(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
) -> str:
    """
    Prefer AI contextual reply; deterministic sysmsg only when LLM unavailable.
    """
    from services.kb_service import sysmsg
    from utils.helpers import resolve_pincode_for_check

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    pin_ready = resolve_pincode_for_check(
        original_msg, conversation_context, msg_en=msg_en
    )
    if pin_ready:
        log_reasoning(
            f"build_pincode_missing skipped — valid PIN {pin_ready} should use live API."
        )
        return ""

    try:
        from services.location_delivery_resolver import _try_geocode_place_from_message

        geo = _try_geocode_place_from_message(comb)
        if geo.kind == "city_geocoded" and geo.pincode:
            log_reasoning(
                f"build_pincode_missing skipped — place geocoded to PIN {geo.pincode}."
            )
            return ""
    except ImportError:
        pass

    scenario, bad = _pincode_reply_scenario(comb, conversation_context)

    ai_body = ai_pincode_conversational_reply(
        original_msg,
        msg_en,
        conversation_context,
        reply_lang,
        scenario=scenario,
        malformed_pin=bad,
    )
    if ai_body:
        return ai_body

    if bad:
        ok, err_key, fmt = validate_pincode_before_api(bad, comb)
        if not ok:
            body = _pin_localized(err_key, original_msg, reply_lang, **fmt)
            if body:
                log_reasoning("Pincode reply — template fallback (invalid PIN, LLM off).")
                return body

    body = (
        _pin_localized("ask_pincode", original_msg, reply_lang)
        or sysmsg("ask_pincode")
        or ""
    )
    if body:
        log_reasoning("Pincode reply — template fallback (ask PIN, LLM off).")
    return body


def validate_pincode_before_api(pin: str, original_msg: str = "") -> tuple[bool, str, dict]:
    """
    (ok, error_sysmsg_key, format_kwargs) — invalid PIN → no API call.
    """
    digits = re.sub(r"\D", "", (pin or "").strip())
    fmt: dict = {"pincode": digits or pin or ""}
    if len(digits) != 6 or digits[0] == "0":
        fmt["typed"] = digits or (pin or "").strip()
        return False, "pincode_invalid_format", fmt
    typed_long = ""
    for m in re.finditer(r"\b([1-9]\d{5,8})\b", original_msg or ""):
        typed_long = m.group(1)
    if typed_long and len(typed_long) > 6 and typed_long[:6] == digits:
        fmt["typed"] = typed_long
        fmt["pincode"] = digits
    return True, "", fmt


def format_pincode_check_reply(
    pincode: str,
    api_res: dict | None,
    original_msg: str,
    reply_lang: str = "en",
    *,
    typo_note: str = "",
    city_label: str = "",
    geocode_note: str = "",
) -> str:
    """Unified pincode API reply — available, not available, invalid, or server error."""
    rl = reply_lang or "en"
    pin = re.sub(r"\D", "", (pincode or "")) or pincode

    ok, err_key, _fmt = validate_pincode_before_api(pin, original_msg)
    if not ok:
        return _pin_localized(err_key, original_msg, rl, **_fmt)

    prefix = ""
    if typo_note:
        note = _pin_localized(
            "pincode_typo_extra_digit", original_msg, rl, typed=typo_note, pincode=pin
        )
        if note:
            prefix = note + "<br><br>"
    if geocode_note:
        prefix = (prefix or "") + f"<span style='color:#555;font-size:13px;'>{geocode_note}</span><br><br>"

    api_data = (api_res or {}).get("data") if isinstance(api_res, dict) else {}
    if not isinstance(api_data, dict):
        api_data = {}
    lat = api_data.get("user_latitude")
    lng = api_data.get("user_longitude")
    map_line = ""
    if lat is not None and lng is not None:
        map_line = (
            f"🗺️ Location: <b>{float(lat):.4f}, {float(lng):.4f}</b><br>"
        )
    city_line = ""
    if (city_label or "").strip():
        city_line = f"🏙️ Area: <b>{html_escape(city_label.strip())}</b><br>"

    if api_res and api_res.get("result") is True:
        message = (api_res.get("message") or "Service is available on this pincode!").strip()
        distance = api_res.get("distance", "nearby")
        body = (
            f"<div class='wf-pin-root' data-wf-live-api='pincode_check'>"
            f"✅ <b>Good News!</b><br>"
            f"{message}<br><br>"
            f"{city_line}"
            f"📍 Pincode: <b>{pin}</b><br>"
            f"{map_line}"
            f"🚚 Distance: <b>{distance}</b><br><br>"
            f"You can place your order on Welfog!"
            f"</div>"
        )
        return prefix + body if prefix else body

    if api_res and api_res.get("result") is False:
        api_msg = (api_res.get("message") or "").strip()
        base = _pin_localized("pincode_not_available", original_msg, rl, pincode=pin)
        extra = ""
        if city_line or map_line:
            extra = f"<br>{city_line}{map_line}"
        if api_msg and api_msg.lower() not in base.lower():
            return (
                f"{prefix}<div class='wf-pin-root' data-wf-live-api='pincode_check'>"
                f"{base}{extra}<br><br><span style='color:#555;font-size:13px;'>{api_msg}</span>"
                f"</div>"
            )
        return (
            f"{prefix}<div class='wf-pin-root' data-wf-live-api='pincode_check'>"
            f"{base}{extra}</div>"
        )

    err_kind = (api_res or {}).get("error") if isinstance(api_res, dict) else ""
    if err_kind == "timeout":
        err = (
            _pin_localized("pincode_api_busy", original_msg, rl)
            or "Delivery check is taking longer than usual — please try your PIN again in a moment."
        )
    elif err_kind in ("http_error", "exception", "bad_json"):
        err = _pin_localized("server_technical_issue", original_msg, rl) or sysmsg(
            "server_technical_issue"
        )
    else:
        err = _pin_localized("server_technical_issue", original_msg, rl) or sysmsg(
            "server_technical_issue"
        )
    return prefix + err if prefix else err


def _format_pincode_api_reply(pincode: str, api_res: dict | None, original_msg: str, reply_lang: str) -> str:
    return format_pincode_check_reply(pincode, api_res, original_msg, reply_lang)


def run_delivery_location_check(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ai_route: dict | None = None,
    *,
    allow_llm: bool = True,
) -> OrderFlowResult:
    """
    AI-first: city name → geocode → PIN → live API, or direct PIN, or ask_pin.
    When brain already locked pincode intent, pass allow_llm=False to skip micro-classifiers.
    """
    from services.location_delivery_resolver import (
        ResolvedDeliveryLocation,
        resolve_delivery_check_target,
        turn_requests_delivery_serviceability,
    )
    from services.welfog_api import check_pincode_delivery

    rl = reply_lang or "en"
    if not turn_requests_delivery_serviceability(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        allow_llm=allow_llm,
    ):
        return OrderFlowResult(handled=False)

    from services.location_delivery_resolver import (
        resolve_delivery_turn,
        should_ask_user_for_pincode,
    )

    understood = resolve_delivery_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=rl,
        allow_llm=allow_llm,
    )
    loc: ResolvedDeliveryLocation = resolve_delivery_check_target(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=rl,
        allow_llm=allow_llm,
    )
    if loc.kind == "ask_pin":
        if not should_ask_user_for_pincode(
            understood,
            original_msg,
            msg_en,
            conversation_context,
            loc=loc,
        ):
            log_reasoning(
                "Delivery: skip PIN prompt — not incomplete/vague/unresolvable."
            )
            return OrderFlowResult(handled=False)
        body = build_pincode_missing_or_invalid_reply(
            original_msg, msg_en, conversation_context, reply_lang=rl
        )
        if body:
            return OrderFlowResult(handled=True, reply_html=body, intent="pincode_check")
        return OrderFlowResult(handled=False)

    if loc.kind not in ("pincode", "city_geocoded") or not loc.pincode:
        return OrderFlowResult(handled=False)

    ok, err_key, fmt = validate_pincode_before_api(loc.pincode, original_msg)
    if not ok:
        body = _pin_localized(err_key, original_msg, rl, **fmt)
        return OrderFlowResult(handled=True, reply_html=body, intent="pincode_check")

    log_reasoning(
        f"Live pincode API for PIN {loc.pincode} "
        f"(source={loc.source}, city={loc.city_label or '-'})"
    )
    api_res = check_pincode_delivery(loc.pincode)
    geo_note = ""
    if loc.kind == "city_geocoded" and loc.city_label:
        geo_note = (
            f"Checked delivery for <b>{html_escape(loc.city_label)}</b> "
            f"using representative PIN <b>{loc.pincode}</b>."
        )
    body = format_pincode_check_reply(
        loc.pincode,
        api_res,
        original_msg,
        rl,
        city_label=loc.city_label,
        geocode_note=geo_note,
    )
    return OrderFlowResult(handled=True, reply_html=body, intent="pincode_check")


def run_pincode_delivery_ai_flow(
    original_msg: str,
    msg_en: str,
    conversation_context: str = "",
    reply_lang: str = "en",
    ai_route: dict | None = None,
) -> OrderFlowResult:
    """
    Unified pincode flow — city/area geocode → live API, ask_pin only when place unknown.
    Delegates to run_delivery_location_check (same path as locked-route executor).
    """
    comb = f"{original_msg} {msg_en}".strip()
    if not message_eligible_for_pincode_ai_flow(comb, msg_en, original_msg, conversation_context):
        return OrderFlowResult(handled=False)

    log_reasoning("Pincode flow: city/PIN resolver + live API (unified path).")
    return run_delivery_location_check(
        original_msg,
        msg_en,
        conversation_context=conversation_context,
        reply_lang=reply_lang or "en",
        ai_route=ai_route,
    )
