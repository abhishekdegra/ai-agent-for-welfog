"""
Order details & invoice: AI-first intent (any language) → purchase-history-details / invoice / track APIs.
Hardcoded keywords are fallback only when LLM routing is unavailable.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Optional

_LIGHTWEIGHT_SIGNAL_GUARD = threading.local()

from services.order_history_flow import OrderFlowResult, _localized_sysmsg
from services.translation_service import language_reply_instruction
from services.welfog_api import (
    fetch_purchase_history_details_for_user,
    format_order_details_reply,
    format_order_invoice_reply_html,
)
from utils.reasoning_log import chat_log, log_reasoning

_INVOICE_RE = re.compile(
    r"(?:\binvoice\b|\binvoic\w*\b|\bbill\b|\breceipt\b|\bgst\b|tax\s+invoice|"
    r"download\s+bill|bill\s+download|invoice\s+download|download\s+invoice|"
    r"चालान|बिल|இன்வாய்ஸ்|பில்)",
    re.IGNORECASE,
)

_INVOICE_GIVE_MARKERS = (
    " de ",
    " dede ",
    " dena ",
    " do ",
    " dijiye ",
    " chahiye ",
    " bhej ",
    " send ",
    " download ",
    " tu hi de",
    " tum hi de",
    " aap hi de",
    " you give",
    " give me",
    "maang",
    "mang rha",
    "mang rahi",
)

_INVOICE_PRONOUN_MARKERS = (
    "isi id",
    "usi id",
    "same id",
    "this id",
    "iska",
    "iski",
    "uska",
    "uski",
    "yeh id",
    "ye id",
    "wo id",
    "id ki",
    "id ka",
    "id ke",
)

_DETAILS_MARKERS = (
    "order detail",
    "order details",
    "order info",
    "order summary",
    "details of order",
    "order data",
    "order ki detail",
    "order ke detail",
    "dikhana",
    "dikhao",
    "dikha do",
    "dikha de",
    "batao",
    "bata do",
    "bata de",
    "बताओ",
    "दिखाओ",
    "विवरण",
    "जानकारी",
    "ஆர்டர் விவர",
    "order_details",
)

_ORDER_DETAILS_EXPLICIT_PHRASES = (
    "order detail",
    "order details",
    "order info",
    "order summary",
    "details of order",
    "order data",
    "order ki detail",
    "order ke detail",
    "order_details",
    "ஆர்டர் விவர",
)

_LOOSE_DETAIL_ACTION_MARKERS = (
    "dikhana",
    "dikhao",
    "dikha do",
    "dikha de",
    "batao",
    "bata do",
    "bata de",
    "बताओ",
    "दिखाओ",
    "विवरण",
    "जानकारी",
)

_PAYMENT_FOCUS_RE = re.compile(
    r"(?:payment\s+status|paid\s+or\s+not|payment\s+done|paisa|paid|unpaid|cod|"
    r"upi|razorpay|payment\s+mode|kitna\s+pay|payment\s+ka)",
    re.IGNORECASE,
)

_PRODUCT_FOCUS_RE = re.compile(
    r"(?:product\s+name|kya\s+order\s+kiya|kya\s+mangaya|which\s+product|"
    r"item\s+name|product\s+info|product\s+detail)",
    re.IGNORECASE,
)

_DELIVERY_FOCUS_RE = re.compile(
    r"(?:delivery\s+status|delivery\s+address|shipping\s+address|ship\s+status|"
    r"delivered\s+or\s+not|dispatch|shipped\s+or\s+not|"
    r"\baddress\b|pata\b|konsa\s+laga|kahan\s+bhej|ship\s+kahan|"
    r"delivery\s+kahan|address\s+konsa|konsa\s+address)",
    re.IGNORECASE,
)

_DETAILS_LOOSE_RE = re.compile(
    r"(?:\borders?\b).{0,40}(?:detail|info|summary|breakdown|data|status\s+of)|"
    r"(?:order\s+detail|order\s+info|order\s+summary|payment\s+for\s+order|"
    r"details\s+of\s+order|order\s+ka\s+status)",
    re.IGNORECASE,
)

_TRACK_ETA_MARKERS = (
    "kab aayega",
    "kab aaega",
    "kab milega",
    "kab tak",
    "kb tk",
    "kahan hai",
    "kaha hai",
    "kidhar",
    "track",
    "tracking",
    "live status",
    "courier",
    "parcel",
    "delay",
    "stuck",
    "atak",
    "atka",
)


def _combined(original_msg: str, msg_en: str = "") -> str:
    return " ".join(p for p in ((original_msg or "").strip(), (msg_en or "").strip()) if p).strip()


_SINGLE_ORDER_CACHE = threading.local()
_UNDERSTAND_GUARD = threading.local()

ORDER_LOOKUP_KB_KEYS = (
    "welfog_api_order_history",
    "welfog_api_order_tracking",
    "welfog_api",
)


def get_single_order_lookup_kb_context() -> str:
    from services.kb_service import read_concatenated_kb_file_contents

    blob = read_concatenated_kb_file_contents(list(ORDER_LOOKUP_KB_KEYS))
    return blob.strip()[:14000] if blob.strip() else "(Order API knowledge missing.)"


def _current_turn_wants_invoice(text: str, conversation_context: str = "") -> bool:
    """Latest user message asks bill/invoice — beats stale locked order_details route."""
    if not (text or "").strip():
        return False
    if _user_rejects_invoice_wants_details(text, conversation_context):
        return False
    if _text_wants_order_invoice(text, conversation_context):
        return True
    return bool(_conversation_hints_invoice_followup(text, conversation_context))


def _user_rejects_invoice_wants_details(text: str, conversation_context: str = "") -> bool:
    """User explicitly wants full order details/summary — NOT invoice download."""
    if not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    denies_invoice = any(
        x in tl
        for x in (
            "invoice nhi",
            "invoice nahi",
            "invoice ni",
            "invoice nahin",
            "invoice not",
            "not invoice",
            "no invoice",
            "without invoice",
            "invoice ke alawa",
            "invoice nhi chahiye",
            "invoice nhi chaiye",
            "invoice nahi chahiye",
            "invoive nhi",
            "invoive nahi",
            "bill nhi",
            "bill nahi",
            "not bill",
            "no bill",
        )
    )
    wants_details = any(
        x in tl
        for x in (
            "pure details",
            "poori details",
            "porri details",
            "puri details",
            "full details",
            "complete details",
            "details chahiye",
            "detail chahiye",
            "details chaiye",
            "detail chaiye",
            "sari details",
            "saari details",
            "pura detail",
            "only details",
            "sirf details",
            "details hi",
            "details do",
            "details de",
            "details bata",
            "details dikha",
        )
    )
    if denies_invoice and (wants_details or re.search(r"\bdetail", tl)):
        return True
    if wants_details and not _INVOICE_RE.search(text):
        return True
    if denies_invoice and re.search(r"\borders?\b", tl) and re.search(r"\b\d{4,20}\b", text):
        return True
    return False


def _current_turn_wants_tracking(text: str, conversation_context: str = "") -> bool:
    """Shipment/status/ETA for one order — beats details/invoice on same turn."""
    if not (text or "").strip():
        return False
    if _user_rejects_invoice_wants_details(text, conversation_context):
        return False
    try:
        from utils.helpers import _text_is_order_tracking_intent_leaf

        if _text_is_order_tracking_intent_leaf(text):
            return True
    except ImportError:
        pass
    try:
        from services.order_tracking_semantics import message_user_wants_order_tracking

        if message_user_wants_order_tracking(text, conversation_context):
            return True
    except ImportError:
        pass
    return False


def turn_blocks_delivery_for_single_order_lookup(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """
    Live track/details/invoice for ONE order — never pincode/serviceability API.
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return False
    try:
        from utils.helpers import (
            _current_turn_has_order_id,
            _text_has_order_id_context,
            _text_is_order_tracking_intent,
            extract_order_id,
        )
    except ImportError:
        return False

    if _current_turn_wants_tracking(comb, conversation_context):
        return True
    if _user_rejects_invoice_wants_details(comb, conversation_context):
        return True
    goal = _fast_order_lookup_goal(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if goal in ("track_single_order", "order_details", "order_invoice"):
        return True
    if _text_has_live_order_lookup_signal(comb):
        if _text_is_order_tracking_intent(comb) or _text_wants_order_details_not_tracking(
            comb, conversation_context
        ):
            return True
        if _text_wants_order_invoice(comb, conversation_context):
            return True
    if _current_turn_has_order_id(comb, msg_en) or extract_order_id(comb, conversation_context):
        if _text_has_order_id_context(comb) or re.search(r"\borders?\b", comb, re.I):
            return True
    if isinstance(ai_route, dict):
        olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
        rh = (ai_route.get("route_handler") or "").strip().lower()
        if rh in ("order_tracking_api", "order_details_api") or olk in (
            "track",
            "tracking",
            "details",
            "invoice",
        ):
            return True
    return False


def _resolve_brain_goal_with_message_override(
    brain_goal: str,
    comb: str,
    conversation_context: str = "",
) -> str:
    """Latest user message overrides stale router goal (track / details / invoice)."""
    if _user_rejects_invoice_wants_details(comb, conversation_context):
        if brain_goal == "order_invoice":
            return "order_details"
    if _current_turn_wants_tracking(comb, conversation_context):
        if brain_goal in ("order_invoice", "order_details"):
            return "track_single_order"
    if brain_goal == "order_details" and _current_turn_wants_invoice(comb, conversation_context):
        return "order_invoice"
    return brain_goal


def _goal_from_brain_route(ai_route: dict | None) -> str:
    if not isinstance(ai_route, dict):
        return ""
    intent = str(ai_route.get("intent") or "").strip().lower()
    if intent == "invoice":
        return "order_invoice"
    olk = str(ai_route.get("order_lookup_kind") or "").strip().lower()
    if olk in ("invoice", "order_invoice"):
        return "order_invoice"
    if olk in ("details", "order_details"):
        return "order_details"
    if olk in ("track", "tracking", "track_single_order"):
        return "track_single_order"
    um = str(ai_route.get("user_meaning") or "").lower()
    if any(x in um for x in ("invoice", "bill", "receipt", "gst", "tax invoice")):
        return "order_invoice"
    if any(
        x in um
        for x in (
            "order detail",
            "order details",
            "payment status",
            "product in order",
            "what did i order",
            "delivery address",
        )
    ):
        return "order_details"
    if any(
        x in um
        for x in ("track", "tracking", "eta", "where is", "kab aayega", "shipment", "courier")
    ):
        return "track_single_order"
    return ""


def _single_order_from_locked_ai_route(
    ai_route: dict | None,
    comb: str,
    conversation_context: str = "",
    *,
    reply_lang: str = "en",
) -> dict[str, Any] | None:
    """Reuse main router order_lookup_kind — skip duplicate single-order LLM."""
    if not isinstance(ai_route, dict):
        return None
    rh = (ai_route.get("route_handler") or "").strip().lower()
    olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
    intent = (ai_route.get("intent") or "").strip().lower()

    try:
        from services.refund_status_semantics import current_turn_wants_personal_refund_status

        if current_turn_wants_personal_refund_status(
            comb, "", conversation_context, ai_route=ai_route, allow_llm=False
        ):
            return None
    except ImportError:
        pass

    goal = _goal_from_brain_route(ai_route)
    goal = _resolve_brain_goal_with_message_override(goal, comb, conversation_context)
    if not goal:
        if rh == "order_details_api":
            goal = "order_invoice" if olk == "invoice" else "order_details"
            goal = _resolve_brain_goal_with_message_override(
                goal, comb, conversation_context
            )
        elif rh == "order_tracking_api" or olk in ("track", "tracking"):
            goal = "track_single_order"
        elif rh == "refund_status_api" or olk == "refund_status":
            return None
        elif intent == "refund":
            return None

    if not goal:
        return None

    action_map = {
        "track_single_order": "track_live",
        "order_details": "order_details",
        "order_invoice": "order_invoice",
    }
    action = action_map.get(goal, "not_order_topic")
    oid = _resolve_order_id(comb, "", conversation_context) or ""
    if not oid:
        from utils.helpers import resolve_order_id_for_tracking

        oid = str(
            resolve_order_id_for_tracking(comb, conversation_context) or ""
        ).strip()

    focus = "invoice" if goal == "order_invoice" else (
        "timeline" if goal == "track_single_order" else "summary"
    )
    return _pack_understanding_result(
        pick_action=action if oid or action == "ask_order_id" else "ask_order_id",
        oid=str(oid or ""),
        field_focus=focus,
        confidence="high",
        source="ai_brain_route",
        reasoning="Reuse main router order_lookup_kind (no duplicate LLM).",
        is_welfog_related=True,
    )


def _conversation_has_order_thread(conversation_context: str) -> bool:
    """Recent chat mentions a personal order — run AI even without English keywords."""
    ctx = (conversation_context or "").strip()
    if not ctx:
        return False
    if re.search(r"\b\d{4,20}\b", ctx):
        return True
    low = ctx.lower()
    return any(
        x in low
        for x in (
            "order id",
            "orderid",
            "invoice",
            "bill",
            "receipt",
            "track",
            "refund",
            "ऑर्डर",
            "चालान",
            "बिल",
            "ट्रैक",
            "விலைப்பட்டியல்",
            "பில்",
        )
    )


def _text_has_live_order_lookup_signal(text: str) -> bool:
    """Light signal for track/details/invoice — avoids helpers false-positives (not keyword routing)."""
    if not (text or "").strip():
        return False
    tl = f" {(text or '').lower()} "
    if re.search(r"\b(?:track|tracking|trck|trak)\b", tl):
        return True
    if _INVOICE_RE.search(text):
        return True
    if _DETAILS_LOOSE_RE.search(text) or any(m in tl for m in _DETAILS_MARKERS):
        return True
    return bool(re.search(r"\b\d{4,20}\b", text) and re.search(r"\borders?\b", tl))


def _message_could_be_personal_order_lookup(
    comb: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Broad gate: if this might be track/details/invoice, call the order specialist LLM."""
    if not (comb or "").strip():
        return False
    try:
        from utils.helpers import (
            _text_wants_order_history_list_in_chat,
            message_is_past_purchase_list_request,
        )

        if _text_wants_order_history_list_in_chat(comb, conversation_context):
            return False
        if message_is_past_purchase_list_request(comb):
            return False
        from utils.helpers import (
            _text_asks_wishlist,
            message_is_seller_on_welfog_request,
            message_is_wishlist_like_request,
        )
        from services.account_list_semantics import turn_requests_wishlist_in_chat

        if turn_requests_wishlist_in_chat(comb, comb, conversation_context, ai_route):
            return False
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            return False
        if message_is_seller_on_welfog_request(comb):
            return False
    except ImportError:
        pass
    if isinstance(ai_route, dict):
        route_intent = (ai_route.get("intent") or "").strip().lower()
        if route_intent in ("wishlist", "seller", "order_history", "pincode_check"):
            return False
        try:
            from services.ai_route_semantics import ai_route_is_kb_read

            if ai_route_is_kb_read(ai_route):
                return False
            ch = (ai_route.get("data_channel") or "").strip().lower()
            if route_intent in ("general", "refund", "payment", "seller") and ch == "kb":
                if not ai_route.get("needs_order_id"):
                    return False
        except ImportError:
            pass
        try:
            from services.early_live_dispatch import ai_route_is_live_api_turn

            if ai_route_is_live_api_turn(ai_route):
                olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
                rh = (ai_route.get("route_handler") or "").strip().lower()
                if rh in ("order_details_api",) or olk in ("details", "invoice"):
                    pass
                elif olk in ("track", "tracking") or rh == "order_tracking_api":
                    pass
                else:
                    return False
        except ImportError:
            pass
    from utils.helpers import (
        _current_turn_has_order_id,
        _text_is_pincode_serviceability_question,
        extract_order_id,
    )

    if _text_has_live_order_lookup_signal(comb):
        return True
    if _text_is_pincode_serviceability_question(comb, conversation_context):
        return False
    if message_is_catalog_product_browse_not_order_details(comb):
        return False
    if _current_turn_has_order_id(comb, comb) or extract_order_id(comb, conversation_context):
        return True
    if _conversation_has_order_thread(conversation_context):
        return True
    try:
        from utils.helpers import turn_skips_order_micro_classifiers

        if turn_skips_order_micro_classifiers(comb, comb, conversation_context, ai_route):
            return False
    except ImportError:
        pass
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        if intent in ("order", "refund", "payment") and (
            ai_route.get("needs_order_id")
            or (ai_route.get("data_channel") or "").strip().lower() == "live_api"
        ):
            return True
        olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
        if olk in ("track", "tracking", "details", "invoice", "refund_status"):
            return True
    if len(comb.strip()) <= 96:
        if _lightweight_details_or_invoice_signal(comb):
            return True
        if re.search(r"\b\d{4,20}\b", comb):
            return True
        return False
    if re.search(r"\b\d{4,20}\b", comb):
        return True
    if _lightweight_details_or_invoice_signal(comb):
        return True
    return bool(re.search(r"\borders?\b", f" {comb.lower()} "))


def _should_run_single_order_ai(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    comb = _combined(original_msg, msg_en)
    if isinstance(ai_route, dict):
        try:
            from services.ai_route_semantics import ai_route_is_kb_read

            if ai_route_is_kb_read(ai_route):
                return False
            intent = (ai_route.get("intent") or "").strip().lower()
            ch = (ai_route.get("data_channel") or "").strip().lower()
            if intent in ("general", "refund", "payment", "seller") and ch == "kb":
                if not ai_route.get("needs_order_id"):
                    return False
        except ImportError:
            pass
    if _fast_order_lookup_goal(comb, "", conversation_context, ai_route):
        return False
    if _single_order_from_locked_ai_route(ai_route, comb, conversation_context):
        return False
    return _message_could_be_personal_order_lookup(comb, conversation_context, ai_route)


def _groq_single_order_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    from services.llm_providers import llm_json_chat_completion

    try:
        return llm_json_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=320,
            timeout_sec=14,
            max_attempts=2,
        )
    except Exception as e:
        log_reasoning(f"Single-order AI understand error: {e}")
        return None


def ai_understand_single_order_request(
    user_msg: str,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> Optional[dict]:
    """
    LLM classifies ONE order request in ANY language/script:
    track vs purchase-history-details vs invoice download.
    """
    api_kb = get_single_order_lookup_kb_context()
    system_prompt = f"""You are Welfog's single-order MEANING router. Customers write in ANY language,
script, or style (English, Hinglish, Hindi, Tamil, Telugu, Bengali, Punjabi, Marathi, Gujarati, Kannada,
Malayalam, Odia, Urdu, chat shorthand, voice typos). Understand WHAT THEY WANT NOW — never match keywords.

ORDER API KNOWLEDGE:
\"\"\"
{api_kb}
\"\"\"

Return ONLY valid JSON:
{{
  "reasoning": "1-3 lines English: what they want in THIS latest message",
  "action": "track_live" | "order_details" | "order_invoice" | "ask_order_id" | "not_order_topic",
  "field_focus": "timeline" | "summary" | "payment" | "product" | "delivery" | "status" | "invoice",
  "extracted_order_id": "digits only if in LATEST user message; else empty",
  "confidence": "high" | "medium" | "low",
  "is_welfog_related": true/false
}}

MEANING RULES (latest message wins; use RECENT CONVERSATION only for pronouns / bare-id follow-ups):
- order_invoice: download bill/invoice/receipt/GST for ONE order they already placed.
- order_details: payment mode, address, product name, totals, summary — NOT shipment ETA.
  If user says invoice/bill NOT wanted but wants full/pure/complete details → order_details NOT order_invoice.
- track_live: shipment status, where is package, ETA, delay, courier, "status check", "track krke" — live timeline.
  If user says track/status/kahan hai/kab aayega → track_live even when order id is present.
- ask_order_id: they want one-order help but no order id in latest message and none inferable from their prior user lines.
- not_order_topic: product shopping, full order list, pincode delivery check, unrelated chat.

THREAD CONTINUATION (critical):
- Bare order id only (e.g. "2805147") → continue the intent from the user's PREVIOUS messages in conversation
  (track vs invoice vs details), NOT a random default.
- User switches topic (invoice then track) → latest message intent overrides older topic.
- Pronouns (isko/usko/this one/same id) → resolve from recent USER lines only, not assistant cards.

Do NOT use keyword lists. Same meaning in different words MUST get the same action.
{language_reply_instruction(reply_lang)}
JSON only."""

    payload = user_msg
    if (conversation_context or "").strip():
        payload = (
            f"RECENT CONVERSATION:\n{conversation_context.strip()[-2500:]}\n\n"
            f"LATEST USER MESSAGE:\n{user_msg}"
        )
    return _groq_single_order_json(system_prompt, payload)


def heuristic_understand_single_order_request(
    user_msg: str,
    conversation_context: str = "",
) -> dict[str, Any]:
    """Fallback when LLM unavailable — lightweight markers only."""
    from utils.helpers import resolve_order_id_for_tracking

    msg = (user_msg or "").strip()
    base: dict[str, Any] = {
        "reasoning": "heuristic",
        "action": "not_order_topic",
        "field_focus": "summary",
        "extracted_order_id": "",
        "confidence": "medium",
        "is_welfog_related": True,
    }
    if not msg:
        return base

    quick_goal = _lightweight_details_or_invoice_signal(msg)
    oid = resolve_order_id_for_tracking(msg, conversation_context)
    tl = f" {msg.lower()} "

    if quick_goal == "order_invoice":
        base.update(
            action="order_invoice",
            field_focus="invoice",
            extracted_order_id=str(oid or ""),
            confidence="high",
            reasoning="Heuristic: invoice/bill intent",
        )
        return base
    if quick_goal == "order_details":
        focus = detect_order_details_focus(msg, action="order_details")
        base.update(
            action="order_details",
            field_focus=focus,
            extracted_order_id=str(oid or ""),
            confidence="high",
            reasoning="Heuristic: order details (not tracking)",
        )
        return base

    if any(m in tl for m in _TRACK_ETA_MARKERS) or (
        oid and re.search(r"\b(?:track|status|kahan|kab)\b", tl)
    ):
        base.update(
            action="track_live" if oid else "ask_order_id",
            field_focus="timeline",
            extracted_order_id=str(oid or ""),
            confidence="high" if oid else "medium",
            reasoning="Heuristic: tracking / ETA",
        )
        return base

    if oid:
        if _text_wants_order_invoice(msg, conversation_context) or _conversation_hints_invoice_followup(
            msg, conversation_context
        ):
            base.update(
                action="order_invoice",
                field_focus="invoice",
                extracted_order_id=str(oid),
                confidence="high",
                reasoning="Heuristic: invoice/bill with order id",
            )
            return base
        base.update(
            action="track_live",
            field_focus="timeline",
            extracted_order_id=str(oid),
            confidence="low",
            reasoning="Heuristic: order id only — default track unless AI overrides",
        )
    return base


def log_invoice_flow(
    *,
    intent: str,
    order_id: str = "",
    selected_flow: str = "",
    invoice_status: str = "",
) -> None:
    msg = (
        f"[invoice-flow] intent={intent or '-'} order_id={order_id or '-'} "
        f"selected_flow={selected_flow or '-'} invoice_status={invoice_status or '-'}"
    )
    log_reasoning(msg)
    chat_log(msg)


def _recent_user_lines_from_context(conversation_context: str, max_lines: int = 5) -> list[str]:
    lines: list[str] = []
    for line in (conversation_context or "").splitlines():
        low = line.strip().lower()
        if low.startswith("user:"):
            lines.append(line.split(":", 1)[-1].strip())
    return lines[-max_lines:]


def _action_to_goal(action: str) -> str:
    return {
        "track_live": "track_single_order",
        "order_details": "order_details",
        "order_invoice": "order_invoice",
        "ask_order_id": "",
        "not_order_topic": "",
    }.get((action or "").strip().lower(), "")


def _pack_understanding_result(
    *,
    pick_action: str,
    oid: str,
    field_focus: str = "",
    confidence: str = "medium",
    source: str = "heuristic",
    reasoning: str = "",
    is_welfog_related: bool = True,
) -> dict[str, Any]:
    if pick_action == "order_invoice":
        log_invoice_flow(
            intent="order_invoice",
            order_id=str(oid or ""),
            selected_flow=source,
            invoice_status="detected" if oid else "need_order_id",
        )
    focus = (field_focus or "summary").strip().lower()
    if pick_action == "order_invoice":
        focus = "invoice"
    elif pick_action == "track_live":
        focus = "timeline"
    goal = _action_to_goal(pick_action)
    return {
        "goal": goal if oid or pick_action == "ask_order_id" else "",
        "action": pick_action if oid or pick_action == "ask_order_id" else "ask_order_id",
        "field_focus": focus,
        "order_id": str(oid or ""),
        "confidence": confidence,
        "source": source,
        "reasoning": (reasoning or "")[:240],
        "is_welfog_related": is_welfog_related,
    }


def _fast_order_lookup_goal(
    text: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """
    Resolve track/details/invoice without specialist LLM when meaning is already clear.
    AI router JSON + message semantics first; LLM only when still ambiguous.
    Priority: track > details (incl. invoice rejection) > invoice.
    """
    comb = _combined(text, msg_en)
    if not comb:
        return ""
    if _current_turn_wants_tracking(comb, conversation_context):
        return "track_single_order"
    if _user_rejects_invoice_wants_details(comb, conversation_context):
        return "order_details"
    brain = _goal_from_brain_route(ai_route)
    if brain in ("order_invoice", "order_details", "track_single_order"):
        return _resolve_brain_goal_with_message_override(
            brain, comb, conversation_context
        )
    if order_details_route_is_locked(ai_route):
        olk = ((ai_route or {}).get("order_lookup_kind") or "").strip().lower()
        if olk == "invoice":
            return "order_invoice"
        if olk == "details":
            locked = "order_details"
            return _resolve_brain_goal_with_message_override(
                locked, comb, conversation_context
            )
    fast = _lightweight_details_or_invoice_signal(comb)
    if fast in ("order_invoice", "order_details"):
        return fast
    if _text_wants_order_invoice(comb, conversation_context):
        return "order_invoice"
    if _text_wants_order_details_not_tracking(comb, conversation_context):
        return "order_details"
    return ""


def _order_lookup_goal_for_message(
    text: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """Cached AI meaning → track_single_order | order_invoice | order_details | ''."""
    fast = _fast_order_lookup_goal(text, msg_en, conversation_context, ai_route)
    if fast:
        return fast
    sub = understand_single_order_request(
        text, msg_en, conversation_context, ai_route=ai_route
    )
    return (sub.get("goal") or "").strip()


def _message_explicitly_wants_tracking(text: str, conversation_context: str = "") -> bool:
    return _order_lookup_goal_for_message(text, "", conversation_context) == "track_single_order"


def _message_explicitly_wants_order_details_not_invoice(
    text: str, conversation_context: str = ""
) -> bool:
    return _order_lookup_goal_for_message(text, "", conversation_context) == "order_details"


def _infer_followup_goal_keyword_fallback(
    text: str,
    conversation_context: str = "",
) -> str:
    """Bare id thread — read latest user turns (track/details/invoice signals)."""
    from utils.helpers import (
        _conversation_bot_offered_order_id_or_tracking,
        _conversation_in_order_tracking_flow,
        _message_is_order_id_followup_submission,
    )

    if not _message_is_order_id_followup_submission(text, conversation_context):
        return ""
    try:
        from utils.helpers import _text_is_refund_return_status_lookup

        if _text_is_refund_return_status_lookup(text, conversation_context):
            return ""
    except ImportError:
        pass
    if _text_has_live_order_lookup_signal(text):
        if _INVOICE_RE.search(text):
            return "order_invoice"
        if re.search(r"\b(?:track|tracking|trck|trak)\b", f" {text.lower()} "):
            return "track_single_order"
        if _DETAILS_LOOSE_RE.search(text) or any(
            m in f" {text.lower()} " for m in _DETAILS_MARKERS
        ):
            return "order_details"
    for body in reversed(_recent_user_lines_from_context(conversation_context, 4)):
        tl = f" {body.lower()} "
        if re.search(r"\b(?:track|tracking|trck|trak)\b", tl) or any(
            x in tl for x in ("kab aayega", "kahan hai", "live status", "status check")
        ):
            return "track_single_order"
        if _INVOICE_RE.search(body):
            return "order_invoice"
        if any(x in tl for x in ("detail", "details", "summary", "info")) and not _INVOICE_RE.search(
            body
        ):
            return "order_details"
    if (
        _conversation_bot_offered_order_id_or_tracking(conversation_context)
        or _conversation_in_order_tracking_flow(conversation_context)
    ):
        return "track_single_order"
    return ""


def _infer_followup_goal_from_conversation(
    text: str,
    conversation_context: str = "",
) -> str:
    """
    Bare order-id follow-up — AI meaning first, keyword fallback when LLM down.
    Returns goal: '' | track_single_order | order_details | order_invoice
    """
    from utils.helpers import (
        _message_is_order_id_followup_submission,
        _text_is_refund_return_status_lookup,
    )

    if _text_is_refund_return_status_lookup(text, conversation_context):
        return ""
    if not _message_is_order_id_followup_submission(text, conversation_context):
        return ""
    # Thread continuation from recent user turns (before LLM on bare id alone).
    kw = _infer_followup_goal_keyword_fallback(text, conversation_context)
    if kw:
        return kw
    ai = ai_understand_single_order_request(text, conversation_context)
    if ai:
        conf = (ai.get("confidence") or "").strip().lower()
        action = (ai.get("action") or "").strip().lower()
        if conf in ("high", "medium") and action in (
            "track_live",
            "order_details",
            "order_invoice",
        ):
            return _action_to_goal(action)
    return ""


def _conversation_hints_invoice_followup(text: str, conversation_context: str = "") -> bool:
    """LLM-down fallback for pronoun invoice follow-ups."""
    if not (text or "").strip():
        return False
    tl = f" {(text or '').lower()} "
    if not any(m in tl for m in _INVOICE_GIVE_MARKERS):
        return False
    ctx_l = f" {(conversation_context or '').lower()} "
    if not any(x in ctx_l for x in ("invoice", "bill", "receipt", "gst", "चालान", "बिल")):
        return False
    return bool(_INVOICE_RE.search(text) or any(m in tl for m in _INVOICE_PRONOUN_MARKERS))


def message_user_wants_order_invoice(
    text: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Bill/invoice/receipt — router + message semantics first; specialist LLM only if ambiguous."""
    if not (text or "").strip():
        return False
    if _goal_from_brain_route(ai_route) == "order_invoice":
        return True
    if order_details_route_is_locked(ai_route):
        olk = ((ai_route or {}).get("order_lookup_kind") or "").strip().lower()
        if olk == "invoice":
            return True
    fast = _fast_order_lookup_goal(text, "", conversation_context, ai_route)
    if fast:
        return fast == "order_invoice"
    if _text_wants_order_invoice(text, conversation_context):
        return True
    if _conversation_hints_invoice_followup(text, conversation_context):
        return True
    return False


def _merge_single_order_understanding(
    ai_data: Optional[dict],
    heuristic: dict[str, Any],
    user_msg: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> dict[str, Any]:
    from utils.helpers import resolve_order_id_for_tracking

    h = dict(heuristic or {})
    a = dict(ai_data or {}) if ai_data else {}
    brain_goal = _goal_from_brain_route(ai_route)

    oid = _resolve_order_id(user_msg, "", conversation_context) or resolve_order_id_for_tracking(
        user_msg,
        conversation_context,
        ai_extracted=(a.get("extracted_order_id") or h.get("extracted_order_id") or "").strip(),
        bot_awaiting_order_id=bool(_infer_followup_goal_keyword_fallback(user_msg, conversation_context)),
    )

    try:
        from services.refund_status_semantics import current_turn_wants_personal_refund_status

        if current_turn_wants_personal_refund_status(
            user_msg, "", conversation_context, ai_route=ai_route, allow_llm=False
        ):
            return _pack_understanding_result(
                pick_action="not_order_topic",
                oid=str(oid or ""),
                confidence="high",
                source="refund_status_guard",
                reasoning="Current turn is personal refund status — not details/invoice.",
            )
    except ImportError:
        pass

    if _user_rejects_invoice_wants_details(user_msg, conversation_context):
        return _pack_understanding_result(
            pick_action="order_details" if oid else "ask_order_id",
            oid=str(oid or ""),
            field_focus="summary",
            confidence="high",
            source="invoice_rejection",
            reasoning="User wants full order details — not invoice.",
        )

    if _current_turn_wants_tracking(user_msg, conversation_context):
        return _pack_understanding_result(
            pick_action="track_live" if oid else "ask_order_id",
            oid=str(oid or ""),
            field_focus="timeline",
            confidence="high",
            source="tracking_priority",
            reasoning="User wants shipment/status timeline — not details/invoice.",
        )

    followup_goal = _infer_followup_goal_from_conversation(user_msg, conversation_context)
    if followup_goal:
        if followup_goal == "order_invoice" and _user_rejects_invoice_wants_details(
            user_msg, conversation_context
        ):
            followup_goal = "order_details"
        action_map = {
            "track_single_order": "track_live",
            "order_details": "order_details",
            "order_invoice": "order_invoice",
        }
        return _pack_understanding_result(
            pick_action=action_map.get(followup_goal, "ask_order_id") if oid else "ask_order_id",
            oid=str(oid or ""),
            confidence="high",
            source="conversation_followup",
            reasoning=f"Continue active thread: {followup_goal}",
        )

    a_action = (a.get("action") or "").strip().lower()
    h_action = (h.get("action") or "").strip().lower()
    a_conf = (a.get("confidence") or "").strip().lower()
    _valid_actions = frozenset(
        ("track_live", "order_details", "order_invoice", "ask_order_id", "not_order_topic")
    )
    if a_action in ("search_products", "products_search", "product_search", "catalog_search"):
        a_action = "not_order_topic"
    if a_action and a_action not in _valid_actions:
        a_action = "not_order_topic"

    # === AI-FIRST: specialist LLM meaning (any language) ===
    if a and a_conf in ("high", "medium") and a_action in _valid_actions - {"not_order_topic"}:
        pick_action = a_action
        if pick_action == "track_live" and not oid:
            pick_action = "ask_order_id"
        return _pack_understanding_result(
            pick_action=pick_action,
            oid=str(oid or ""),
            field_focus=str(a.get("field_focus") or ""),
            confidence=a_conf,
            source="ai_specialist",
            reasoning=str(a.get("reasoning") or ""),
            is_welfog_related=bool(a.get("is_welfog_related", True)),
        )

    try:
        from services.order_tracking_semantics import message_user_rejects_refund_wants_tracking

        if message_user_rejects_refund_wants_tracking(user_msg):
            return _pack_understanding_result(
                pick_action="track_live" if oid else "ask_order_id",
                oid=str(oid or ""),
                confidence="high",
                source="user_correction",
                reasoning="User correcting bot — wants tracking",
            )
    except ImportError:
        pass

    pick_action = h_action or "not_order_topic"
    brain_goal = _resolve_brain_goal_with_message_override(
        brain_goal, user_msg, conversation_context
    )
    if brain_goal == "track_single_order":
        pick_action = "track_live"
    elif brain_goal == "order_details":
        pick_action = "order_details"
    elif brain_goal == "order_invoice":
        pick_action = "order_invoice"
    elif a_action and a_conf == "low":
        pick_action = a_action

    if pick_action == "track_live" and not oid:
        pick_action = "ask_order_id"

    source = "heuristic"
    if brain_goal and pick_action in ("order_invoice", "order_details", "track_live"):
        source = "ai_brain_route"
    elif a:
        source = "ai_specialist+heuristic"

    return _pack_understanding_result(
        pick_action=pick_action,
        oid=str(oid or ""),
        field_focus=str(a.get("field_focus") or h.get("field_focus") or ""),
        confidence=a_conf or str(h.get("confidence") or "medium"),
        source=source,
        reasoning=str(a.get("reasoning") or h.get("reasoning") or ""),
        is_welfog_related=bool(a.get("is_welfog_related", h.get("is_welfog_related", True))),
    )


def understand_single_order_request(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    ai_route: dict | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Primary classifier for one-order flows. Cached per request thread to avoid duplicate LLM calls.
    Returns goal: '' | order_invoice | order_details | track_single_order
    """
    comb = _combined(original_msg, msg_en)
    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-500:])}"
    if (
        not force_refresh
        and getattr(_SINGLE_ORDER_CACHE, "key", None) == cache_key
        and isinstance(getattr(_SINGLE_ORDER_CACHE, "result", None), dict)
    ):
        return _SINGLE_ORDER_CACHE.result

    if getattr(_UNDERSTAND_GUARD, "active", False):
        return {
            "goal": "",
            "action": "not_order_topic",
            "field_focus": "summary",
            "order_id": "",
            "confidence": "low",
            "source": "reentrant_guard",
            "reasoning": "",
            "is_welfog_related": True,
        }

    _UNDERSTAND_GUARD.active = True
    try:
        try:
            from utils.helpers import _text_is_pincode_serviceability_question

            if _text_is_pincode_serviceability_question(comb, conversation_context):
                not_order = {
                    "goal": "",
                    "action": "not_order_topic",
                    "field_focus": "summary",
                    "order_id": "",
                    "confidence": "high",
                    "source": "delivery_serviceability_guard",
                    "reasoning": "Delivery/serviceability at PIN or area — not single-order lookup.",
                    "is_welfog_related": True,
                }
                _SINGLE_ORDER_CACHE.key = cache_key
                _SINGLE_ORDER_CACHE.result = not_order
                log_reasoning(
                    "Single-order AI skipped — delivery serviceability (pincode pipeline)."
                )
                return not_order
        except ImportError:
            pass

        empty = {
            "goal": "",
            "action": "not_order_topic",
            "field_focus": "summary",
            "order_id": "",
            "confidence": "low",
            "source": "none",
            "reasoning": "",
            "is_welfog_related": True,
        }
        if not comb:
            _SINGLE_ORDER_CACHE.key = cache_key
            _SINGLE_ORDER_CACHE.result = empty
            return empty

        try:
            from services.refund_status_semantics import current_turn_wants_personal_refund_status

            if current_turn_wants_personal_refund_status(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                allow_llm=False,
            ):
                refund_skip = {
                    "goal": "",
                    "action": "not_order_topic",
                    "field_focus": "summary",
                    "order_id": "",
                    "confidence": "high",
                    "source": "refund_status_guard",
                    "reasoning": "Personal refund status — refund API path, not order details.",
                    "is_welfog_related": True,
                }
                _SINGLE_ORDER_CACHE.key = cache_key
                _SINGLE_ORDER_CACHE.result = refund_skip
                log_reasoning(
                    "Single-order AI skipped — current turn is personal refund status."
                )
                return refund_skip
        except ImportError:
            pass

        brain_goal = _goal_from_brain_route(ai_route)
        route_locked = _single_order_from_locked_ai_route(
            ai_route, comb, conversation_context, reply_lang=reply_lang
        )
        if isinstance(ai_route, dict):
            try:
                from services.ai_route_semantics import ai_route_is_kb_read

                if ai_route_is_kb_read(ai_route):
                    kb_skip = {
                        "goal": "",
                        "action": "not_order_topic",
                        "field_focus": "summary",
                        "order_id": "",
                        "confidence": "high",
                        "source": "kb_read_route",
                        "reasoning": "Router chose KB policy/read-only answer — not live order lookup.",
                        "is_welfog_related": True,
                    }
                    _SINGLE_ORDER_CACHE.key = cache_key
                    _SINGLE_ORDER_CACHE.result = kb_skip
                    try:
                        from services.chat_flow_telemetry import skip_step

                        skip_step("single_order_llm", "kb read route")
                    except ImportError:
                        pass
                    return kb_skip
            except ImportError:
                pass
        if route_locked:
            log_reasoning(
                f"Single-order AI skipped — reuse router "
                f"goal={route_locked.get('goal')} source={route_locked.get('source')}."
            )
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("single_order_llm", "router order_lookup_kind locked")
            except ImportError:
                pass
            _SINGLE_ORDER_CACHE.key = cache_key
            _SINGLE_ORDER_CACHE.result = route_locked
            log_reasoning(
                f"[order-intent] goal={route_locked.get('goal') or 'none'} "
                f"action={route_locked.get('action')} source={route_locked.get('source')} "
                f"oid={route_locked.get('order_id') or '-'}"
            )
            chat_log(
                f"[order-intent] goal={route_locked.get('goal') or 'none'} "
                f"source={route_locked.get('source')}"
            )
            return route_locked

        heuristic = heuristic_understand_single_order_request(comb, conversation_context)
        ai_data = None
        llm_down = isinstance(ai_route, dict) and bool(ai_route.get("llm_unavailable"))
        if _should_run_single_order_ai(original_msg, msg_en, conversation_context, ai_route) and not llm_down:
            ai_data = ai_understand_single_order_request(
                comb, conversation_context, reply_lang=reply_lang
            )
            if ai_data:
                log_reasoning(
                    f"Single-order AI: action={ai_data.get('action')} "
                    f"focus={ai_data.get('field_focus')} conf={ai_data.get('confidence')} — "
                    f"{(ai_data.get('reasoning') or '')[:100]}"
                )
        elif not llm_down:
            try:
                from services.chat_flow_telemetry import skip_step

                skip_step("single_order_llm", "goal already resolved")
            except ImportError:
                pass

        merged = _merge_single_order_understanding(
            ai_data, heuristic, comb, conversation_context, ai_route=ai_route
        )
        if (
            not merged.get("goal")
            and brain_goal
            and (merged.get("action") or "").strip().lower() != "not_order_topic"
        ):
            merged["goal"] = brain_goal
            merged["source"] = "ai_brain_route"

        log_reasoning(
            f"[order-intent] goal={merged.get('goal') or 'none'} action={merged.get('action')} "
            f"focus={merged.get('field_focus')} source={merged.get('source')} "
            f"conf={merged.get('confidence')} oid={merged.get('order_id') or '-'}"
        )
        chat_log(
            f"[order-intent] goal={merged.get('goal') or 'none'} source={merged.get('source')} "
            f"focus={merged.get('field_focus')}"
        )

        _SINGLE_ORDER_CACHE.key = cache_key
        _SINGLE_ORDER_CACHE.result = merged
        return merged
    finally:
        _UNDERSTAND_GUARD.active = False


def _lightweight_details_or_invoice_signal(text: str) -> str:
    """
    Fast marker check only — must NOT call helpers._text_is_order_tracking_intent
    (avoids recursion with user_turn_qualifies_for_live_order_api).
    """
    if getattr(_LIGHTWEIGHT_SIGNAL_GUARD, "active", False):
        return ""
    _LIGHTWEIGHT_SIGNAL_GUARD.active = True
    try:
        return _lightweight_details_or_invoice_signal_impl(text)
    finally:
        _LIGHTWEIGHT_SIGNAL_GUARD.active = False


def _lightweight_details_or_invoice_signal_impl(text: str) -> str:
    if not (text or "").strip():
        return ""
    if _user_rejects_invoice_wants_details(text, ""):
        if re.search(r"\borders?\b", f" {text.lower()} ") or re.search(
            r"\b\d{4,20}\b", text
        ):
            return "order_details"
    try:
        from utils.helpers import _text_is_order_tracking_intent_leaf

        if _text_is_order_tracking_intent_leaf(text):
            return ""
    except ImportError:
        pass
    tl_guard = f" {(text or '').lower()} "
    if not re.search(r"\borders?\b", tl_guard) and "order id" not in tl_guard:
        if re.search(
            r"\b(?:dikha|dikhao|dikho|cover|shirt|shoes|product|products|sku|mobile|"
            r"price|rating|color|colour|rang|brand|size)\b",
            tl_guard,
        ) and not re.search(
            r"\b(?:invoice|bill|receipt|gst|track|tracking|payment\s+status|courier)\b",
            tl_guard,
        ):
            return ""
    if _text_wants_order_invoice(text, ""):
        return "order_invoice"
    tl = f" {text.lower()} "
    if any(m in tl for m in _TRACK_ETA_MARKERS):
        return ""
    if (
        _PAYMENT_FOCUS_RE.search(text)
        or _PRODUCT_FOCUS_RE.search(text)
        or _DELIVERY_FOCUS_RE.search(text)
        or _DETAILS_LOOSE_RE.search(text)
    ):
        return "order_details"
    if any(m in tl for m in _DETAILS_MARKERS):
        if re.search(r"\borders?\b", tl):
            return "order_details"
        return ""
    if "order" in tl and any(
        x in tl for x in ("detail", "details", "info", "summary", "invoice", "bill", "receipt")
    ):
        if _INVOICE_RE.search(text):
            return "order_invoice"
        if any(x in tl for x in ("detail", "details", "info", "summary")):
            return "order_details"
    return ""


def message_wants_order_details_or_invoice(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """'' | 'order_invoice' | 'order_details' — blocks live tracking shortcuts."""
    sub = understand_single_order_request(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    goal = (sub.get("goal") or "").strip()
    if goal in ("order_invoice", "order_details"):
        return goal
    return ""


def log_order_data_pipeline(
    *,
    action: str,
    source: str,
    api: str,
    focus: str,
    order_id: str,
    fields: list[str],
) -> None:
    fields_s = ",".join(fields) if fields else "none"
    msg = (
        f"[order-data] intent={action} source={source} api={api} "
        f"focus={focus} order_id={order_id} fields={fields_s}"
    )
    log_reasoning(msg)
    chat_log(msg)


def fields_included_for_focus(focus: str) -> list[str]:
    focus = (focus or "summary").lower().strip()
    base = ["order_id"]
    if focus == "order_invoice":
        return base + ["invoice_button"]
    mapping = {
        "payment": base + ["payment", "total"],
        "product": base + ["product", "product_image", "product_link"],
        "delivery": base + ["shipping_address", "name", "phone", "address"],
        "status": base + ["status", "delivery"],
        "invoice": base + ["invoice_button"],
        "summary": base + ["product", "payment", "status", "date", "total"],
    }
    return mapping.get(focus, mapping["summary"])


def message_is_catalog_product_browse_not_order_details(text: str) -> bool:
    """Product show/buy/search — not an existing-order details lookup."""
    if not (text or "").strip():
        return False
    if _text_wants_order_invoice(text) or _INVOICE_RE.search(text or ""):
        return False
    if _text_has_live_order_lookup_signal(text):
        return False
    from utils.helpers import (
        _message_has_catalog_product_signal,
        _text_has_product_shopping_intent,
        _text_is_phone_product_accessory_context,
        _turn_is_catalog_product_request,
    )

    if _text_is_phone_product_accessory_context(text):
        return True
    if _text_has_product_shopping_intent(text) or _turn_is_catalog_product_request(text):
        return True

    tl = f" {(text or '').lower()} "
    has_order_ctx = bool(re.search(r"\borders?\b", tl)) or bool(
        re.search(r"\b\d{4,20}\b", text)
    )
    if has_order_ctx:
        return False
    if _message_has_catalog_product_signal(text) and re.search(
        r"\b(?:dikha\w*|show|buy|kharid|lena|leni|chahiye|browse|search|filter)\b",
        tl,
    ):
        return True
    if re.search(
        r"\b(?:dikha\w*|dikho|show|buy|kharid|lena|leni|chahiye)\b",
        tl,
    ) and re.search(
        r"\b(?:cover|case|shirt|shoes|mobile|phone|product|products|sku|dress|kurta|"
        r"jeans|charger|cable|watch|bag|price|rating|color|colour|rang|brand|size)\b",
        tl,
    ):
        return True
    return False


_INVOICE_HOWTO_NAV_RE = re.compile(
    r"(?:\bwhere\s+(?:can|do)\s+i\s+find\b|\bhow\s+(?:can|do)\s+i\s+(?:find|download|get)\b|"
    r"\bhow\s+to\s+(?:find|download|get)\b|\bwhere\s+(?:is|to\s+find)\b).{0,50}\b(?:invoice|bill|receipt)\b|"
    r"\b(?:invoice|bill|receipt)\b.{0,40}\b(?:kaise|kahan|kaha|where|how)\b",
    re.IGNORECASE,
)


def text_asks_invoice_howto_navigation(text: str, conversation_context: str = "") -> bool:
    """
    Where/how to download invoice (FAQ steps) — not live order-history API.
    """
    if not (text or "").strip():
        return False
    if not _INVOICE_RE.search(text):
        return False
    if re.search(r"\b\d{4,20}\b", text):
        return False
    if any(m in f" {text.lower()} " for m in _INVOICE_GIVE_MARKERS):
        return False
    if _conversation_hints_invoice_followup(text, conversation_context):
        return False
    if _INVOICE_HOWTO_NAV_RE.search(text):
        return True
    tl = f" {text.lower()} "
    if any(
        x in tl
        for x in (
            "where can i find",
            "where do i find",
            "how can i find",
            "how to find",
            "how can i download",
            "how to download",
            "how do i download",
            "find my invoice",
            "get my invoice",
            "invoice kahan",
            "bill kahan",
            "invoice kaise",
            "bill kaise",
        )
    ):
        return True
    return False


def _text_wants_order_invoice(text: str, conversation_context: str = "") -> bool:
    if not (text or "").strip():
        return False
    if _user_rejects_invoice_wants_details(text, conversation_context):
        return False
    if not _INVOICE_RE.search(text):
        return False
    if text_asks_invoice_howto_navigation(text, conversation_context):
        return False
    tl = f" {text.lower()} "
    if re.search(r"\b\d{4,20}\b", text):
        return True
    if "order" in tl or " id" in tl or "orderid" in tl:
        return True
    if any(m in tl for m in _INVOICE_GIVE_MARKERS) or "maang" in tl:
        return True
    if _conversation_hints_invoice_followup(text, conversation_context):
        return True
    return False


def _text_wants_order_details_not_tracking(text: str, conversation_context: str = "") -> bool:
    from utils.helpers import _text_is_order_tracking_intent

    if not (text or "").strip():
        return False
    if message_is_catalog_product_browse_not_order_details(text):
        return False
    if _text_wants_order_invoice(text):
        return False
    if (
        _PAYMENT_FOCUS_RE.search(text)
        or _PRODUCT_FOCUS_RE.search(text)
        or _DELIVERY_FOCUS_RE.search(text)
    ):
        return True
    if _text_is_order_tracking_intent(text):
        return False
    tl = f" {text.lower()} "
    if any(m in tl for m in _TRACK_ETA_MARKERS):
        return False
    if (
        _PAYMENT_FOCUS_RE.search(text)
        or _PRODUCT_FOCUS_RE.search(text)
        or _DELIVERY_FOCUS_RE.search(text)
        or _DETAILS_LOOSE_RE.search(text)
    ):
        return True
    if any(m in tl for m in _ORDER_DETAILS_EXPLICIT_PHRASES):
        return True
    if any(m in tl for m in _LOOSE_DETAIL_ACTION_MARKERS):
        if re.search(r"\borders?\b", tl) or re.search(r"\b\d{4,20}\b", text):
            return True
        return False
    if "order" in tl and any(
        x in tl
        for x in (
            "detail",
            "details",
            "info",
            "summary",
            "payment status",
            "product",
            "grand total",
            "total amount",
        )
    ):
        return True
    return False


def infer_order_details_semantic_goal(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """order_invoice | order_details | '' (track/list handled elsewhere)."""
    if isinstance(ai_route, dict):
        try:
            from services.ai_route_semantics import ai_route_is_kb_read

            if ai_route_is_kb_read(ai_route):
                return ""
        except ImportError:
            pass
    sub = understand_single_order_request(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    goal = (sub.get("goal") or "").strip()
    if goal in ("order_invoice", "order_details"):
        return goal
    return ""


def detect_order_details_focus(
    text: str,
    action: str = "order_details",
    *,
    ai_focus: str = "",
) -> str:
    """payment | product | delivery | status | summary — AI field_focus overrides when set."""
    if action == "order_invoice":
        return "invoice"
    focus = (ai_focus or "").strip().lower()
    if focus in ("payment", "product", "delivery", "status", "summary", "invoice"):
        return "invoice" if focus == "invoice" else focus
    if _PAYMENT_FOCUS_RE.search(text):
        return "payment"
    if _PRODUCT_FOCUS_RE.search(text):
        return "product"
    if _DELIVERY_FOCUS_RE.search(text):
        return "delivery"
    tl = f" {text.lower()} "
    if any(x in tl for x in ("status", "delivered", "shipped", "cancel")):
        return "status"
    return "summary"


def message_eligible_for_order_details_flow(
    combined: str, msg_en: str, original_msg: str, conversation_context: str = ""
) -> bool:
    return bool(
        infer_order_details_semantic_goal(
            original_msg, msg_en, conversation_context, ai_route=None
        )
        or _text_wants_order_invoice(combined)
        or _text_wants_order_details_not_tracking(combined, conversation_context)
    )


def _resolve_order_id(original_msg: str, msg_en: str, conversation_context: str) -> str:
    from utils.helpers import extract_order_id, resolve_order_id_for_tracking

    comb = _combined(original_msg, msg_en)
    oid = extract_order_id(original_msg, conversation_context) or extract_order_id(
        msg_en, conversation_context
    )
    if oid:
        return str(oid).strip()
    resolved = resolve_order_id_for_tracking(comb, conversation_context)
    if resolved:
        return str(resolved).strip()
    m = re.search(
        r"(?:^|\s)(\d{4,20})\s+(?:is\s+)?(?:id|order|oid)\b",
        comb,
        re.IGNORECASE,
    )
    if m and (
        _text_wants_order_invoice(comb, conversation_context)
        or message_user_wants_order_invoice(comb, conversation_context)
        or _message_explicitly_wants_order_details_not_invoice(comb)
        or _message_explicitly_wants_tracking(comb, conversation_context)
    ):
        return str(m.group(1)).strip()
    m_ka = re.search(r"\b(\d{4,20})\s+ka\b", comb, re.IGNORECASE)
    if m_ka and (
        _INVOICE_RE.search(comb)
        or _text_wants_order_invoice(comb, conversation_context)
        or any(m in f" {comb.lower()} " for m in _INVOICE_GIVE_MARKERS)
    ):
        return str(m_ka.group(1)).strip()
    tl = f" {comb.lower()} "
    if any(x in tl for x in ("isko", "iske", "iski", "usko", "uske", "uski", "this order", "same order")):
        from utils.helpers import extract_order_id_from_recent_user_lines

        recent = extract_order_id_from_recent_user_lines(conversation_context, comb)
        if recent:
            return str(recent).strip()
    return ""


def _details_error_reply(
    err: str, order_id: str, original_msg: str, reply_lang: str
) -> str:
    if err == "login_required":
        return (
            "<div style='color:#333;line-height:1.55;'>Please open this chat from the Welfog app "
            "while logged in, or add <b>?user_id=YOUR_ID</b> to this page URL.</div>"
        )
    if err == "not_owned":
        return _localized_sysmsg("order_track_not_owned", original_msg, reply_lang=reply_lang) or (
            "This Order ID is not linked to your account."
        )
    hint = _localized_sysmsg("order_track_not_found", original_msg, reply_lang=reply_lang)
    if hint and order_id and str(order_id) not in hint:
        hint = hint.replace("</div>", f"<br><br>Checked ID: <b>{order_id}</b></div>", 1)
    return hint or (
        f"We could not find Order ID <b>{order_id}</b>. Check confirmation SMS/email or My Orders."
    )


def run_order_details_ai_flow(
    original_msg: str,
    msg_en: str,
    user_id: str,
    conversation_context: str = "",
    reply_lang: str = "en",
    ai_route: dict | None = None,
) -> OrderFlowResult:
    comb = _combined(original_msg, msg_en)
    rl = reply_lang or "en"
    sub = understand_single_order_request(
        original_msg, msg_en, conversation_context, reply_lang=rl, ai_route=ai_route
    )
    goal = (sub.get("goal") or "").strip()
    if goal not in ("order_invoice", "order_details") and not message_eligible_for_order_details_flow(
        comb, msg_en, original_msg, conversation_context
    ):
        return OrderFlowResult(handled=False)

    action = goal or (
        "order_invoice" if _text_wants_order_invoice(comb) else "order_details"
    )
    inline_id = (sub.get("order_id") or "").strip() or _resolve_order_id(
        original_msg, msg_en, conversation_context
    )

    if not inline_id:
        log_reasoning(f"Order {action} — need order id.")
        if action == "order_invoice":
            log_invoice_flow(
                intent="order_invoice",
                order_id="",
                selected_flow="order_details_api",
                invoice_status="need_order_id",
            )
        body = _localized_sysmsg("ask_order_id_for_intent", original_msg, reply_lang=rl, intent="order") or ""
        return OrderFlowResult(
            handled=True,
            reply_html=body,
            intent="order",
            needs_order_id=True,
        )

    api_name = "purchase-history-details"
    log_reasoning(f"Order {action} API for id={inline_id}")
    if action == "order_invoice":
        log_invoice_flow(
            intent="order_invoice",
            order_id=inline_id,
            selected_flow="order_details_api",
            invoice_status="ownership_check",
        )
    row, err = fetch_purchase_history_details_for_user(inline_id, user_id)
    if err or not row:
        log_order_data_pipeline(
            action=action,
            source="api_error",
            api=api_name,
            focus="none",
            order_id=inline_id,
            fields=[],
        )
        if action == "order_invoice":
            log_invoice_flow(
                intent="order_invoice",
                order_id=inline_id,
                selected_flow="order_details_api",
                invoice_status=err or "not_found",
            )
        body = _details_error_reply(err or "not_found", inline_id, original_msg, rl)
        return OrderFlowResult(
            handled=True,
            reply_html=body,
            intent="order",
            order_id=inline_id,
        )

    if action == "order_invoice":
        focus = "invoice"
        fields = fields_included_for_focus("order_invoice")
        body = format_order_invoice_reply_html(inline_id, lang=rl)
        log_invoice_flow(
            intent="order_invoice",
            order_id=inline_id,
            selected_flow="order_details_api",
            invoice_status="ready",
        )
        log_order_data_pipeline(
            action=action,
            source="live_api",
            api=f"{api_name}+invoice_url",
            focus=focus,
            order_id=inline_id,
            fields=fields,
        )
    else:
        focus = detect_order_details_focus(
            comb, action=action, ai_focus=str(sub.get("field_focus") or "")
        )
        fields = fields_included_for_focus(focus)
        body = format_order_details_reply(row, inline_id, focus=focus, lang=rl)
        log_order_data_pipeline(
            action=action,
            source="live_api",
            api=api_name,
            focus=focus,
            order_id=inline_id,
            fields=fields,
        )

    return OrderFlowResult(
        handled=True,
        reply_html=body,
        intent="order",
        order_id=inline_id,
    )


_ORDER_DETAILS_MEANING = (
    "order detail",
    "order details",
    "order summary",
    "order info",
    "payment status",
    "payment mode",
    "delivery address",
    "shipping address",
    "product in order",
    "what did i order",
    "grand total",
    "order breakdown",
    "address btao",
    "address dikhao",
)


def message_user_rejects_refund_wants_order_details(text: str) -> bool:
    """User correcting bot: not refund/return — they want order details."""
    if not (text or "").strip():
        return False
    tl = f" {text.lower()} "
    wants_details = any(
        x in tl
        for x in (
            "order detail",
            "order details",
            "payment status",
            "delivery address",
            "shipping address",
            "address btao",
            "address dikhao",
        )
    )
    if not wants_details:
        return False
    rejects_refund = any(
        x in tl
        for x in (
            " nhi ",
            " nahi ",
            " not ",
            " no ",
            " galat ",
            " wrong ",
            "are refund",
            "refund status nhi",
            "refund nhi",
            "return nhi",
            "return status nhi",
        )
    )
    return rejects_refund or ("order detail" in tl and "refund" in tl)


def _text_clearly_wants_refund_status_not_order_details(
    text: str, conversation_context: str = ""
) -> bool:
    """Refund/return status on one order — not payment/address/order-details."""
    from utils.helpers import _text_is_refund_return_status_lookup

    if not _text_is_refund_return_status_lookup(text, conversation_context):
        return False
    if message_user_rejects_refund_wants_order_details(text):
        return False
    if _text_wants_order_invoice(text) or _text_wants_order_details_not_tracking(
        text, conversation_context
    ):
        return False
    return True


def order_details_route_is_locked(route: dict | None) -> bool:
    if not route:
        return False
    olk = (route.get("order_lookup_kind") or "").strip().lower()
    if olk in ("details", "invoice"):
        return True
    return (route.get("route_handler") or "").strip().lower() == "order_details_api"


def _apply_order_details_to_route(out: dict, goal: str, source: str) -> dict:
    out["intent"] = "order"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = True
    out["numeric_context"] = "order_id"
    out["order_lookup_kind"] = "invoice" if goal == "order_invoice" else "details"
    out["route_handler"] = "order_details_api"
    out["answer_strategy"] = "live_api_only"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    log_reasoning(
        f"Order-details ({source}): personal lookup → purchase-history-details API."
    )
    return out


def ai_route_requests_order_details_lookup(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """Personal order details/invoice for one order — message + Groq JSON (any language)."""
    comb = _combined(original_msg, msg_en)
    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(
            route,
            original_msg,
            msg_en,
            conversation_context,
            allow_llm=True,
        ):
            return False
    except ImportError:
        pass
    try:
        from services.refund_status_semantics import current_turn_wants_personal_refund_status

        if current_turn_wants_personal_refund_status(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=route,
            allow_llm=False,
        ):
            return False
    except ImportError:
        if _text_clearly_wants_refund_status_not_order_details(comb, conversation_context):
            return False

    if order_details_route_is_locked(route):
        fast_locked = _fast_order_lookup_goal(
            original_msg, msg_en, conversation_context, ai_route=route
        )
        if fast_locked in ("order_invoice", "order_details"):
            return True
        if message_user_wants_order_invoice(comb, conversation_context, ai_route=route):
            return True
        if _text_wants_order_details_not_tracking(comb, conversation_context):
            return True
        return False
    fast = _fast_order_lookup_goal(
        original_msg, msg_en, conversation_context, ai_route=route
    )
    if fast in ("order_invoice", "order_details"):
        return True
    if message_user_wants_order_invoice(comb, conversation_context, ai_route=route):
        return True
    try:
        from services.order_tracking_semantics import (
            message_user_rejects_refund_wants_tracking,
            message_user_wants_order_tracking,
            order_tracking_route_is_locked,
        )

        if order_tracking_route_is_locked(route):
            return False
        if message_user_rejects_refund_wants_tracking(comb):
            return False
        if message_user_wants_order_tracking(comb, conversation_context):
            return False
    except ImportError:
        pass
    if message_is_catalog_product_browse_not_order_details(comb):
        return False
    r_pre = route or {}
    intent_pre = (r_pre.get("intent") or "").strip().lower()
    if intent_pre in ("product", "deals", "categories", "category_feed") or r_pre.get(
        "run_catalog_search"
    ):
        return False
    if message_user_rejects_refund_wants_order_details(comb):
        return True
    if order_details_route_is_locked(route):
        return True
    if _text_wants_order_invoice(comb):
        return True
    if _text_wants_order_details_not_tracking(comb, conversation_context):
        return True
    if _text_clearly_wants_refund_status_not_order_details(comb, conversation_context):
        return False
    goal = message_wants_order_details_or_invoice(
        original_msg, msg_en, conversation_context, ai_route=route
    )
    if goal:
        return True
    r = route or {}
    olk = (r.get("order_lookup_kind") or "").strip().lower()
    if olk in ("details", "invoice"):
        return True
    if (r.get("route_handler") or "").strip().lower() == "order_details_api":
        return True
    um = f" {(r.get('user_meaning') or '').lower()} "
    if any(x in um for x in _ORDER_DETAILS_MEANING):
        return True
    return False


def promote_order_details_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> dict:
    """Align Groq route with personal order details/invoice (before refund-status promotion)."""
    out = dict(route or {})
    comb = _combined(original_msg, msg_en)

    try:
        from services.location_delivery_resolver import pincode_delivery_route_is_locked

        if pincode_delivery_route_is_locked(
            out,
            original_msg,
            msg_en,
            conversation_context,
            allow_llm=True,
        ):
            return out
    except ImportError:
        pass

    if _current_turn_wants_tracking(comb, conversation_context):
        out.pop("order_lookup_kind", None)
        out.pop("route_handler", None)
        out["run_catalog_search"] = False
        return out

    if _user_rejects_invoice_wants_details(comb, conversation_context):
        return _apply_order_details_to_route(out, "order_details", "invoice_rejection")

    try:
        from services.ai_route_semantics import ai_route_is_kb_read

        if ai_route_is_kb_read(out):
            return out
    except ImportError:
        pass

    try:
        from utils.helpers import turn_skips_order_micro_classifiers

        if turn_skips_order_micro_classifiers(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            return out
    except ImportError:
        pass

    try:
        from services.refund_status_semantics import (
            KIND_POLICY_HOWTO,
            current_turn_wants_personal_refund_status,
            resolve_refund_turn,
        )

        if current_turn_wants_personal_refund_status(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=out,
            allow_llm=False,
        ):
            return out
        rf = resolve_refund_turn(
            original_msg, msg_en, conversation_context, ai_route=out, allow_llm=False
        )
        if rf.kind == KIND_POLICY_HOWTO:
            return out
    except ImportError:
        pass

    if order_details_route_is_locked(out):
        olk_locked = (out.get("order_lookup_kind") or "").strip().lower()
        if olk_locked != "invoice" and _current_turn_wants_invoice(comb, conversation_context):
            log_invoice_flow(
                intent="order_invoice",
                order_id=_resolve_order_id(original_msg, msg_en, conversation_context),
                selected_flow="promote_order_details",
                invoice_status="routed",
            )
            return _apply_order_details_to_route(out, "order_invoice", "invoice_override_locked")
        return out

    brain = _goal_from_brain_route(out)
    olk_pre = (out.get("order_lookup_kind") or "").strip().lower()
    rh_pre = (out.get("route_handler") or "").strip().lower()
    if brain in ("order_invoice", "order_details") or olk_pre in ("details", "invoice") or rh_pre == "order_details_api":
        goal = (
            "order_invoice"
            if brain == "order_invoice" or olk_pre == "invoice"
            else "order_details"
        )
        if goal == "order_invoice":
            log_invoice_flow(
                intent="order_invoice",
                order_id=_resolve_order_id(original_msg, msg_en, conversation_context),
                selected_flow="ai_brain_route",
                invoice_status="routed",
            )
        return _apply_order_details_to_route(out, goal, "ai_brain_route")

    fast_goal = _fast_order_lookup_goal(
        original_msg, msg_en, conversation_context, ai_route=out
    )
    if fast_goal in ("order_invoice", "order_details"):
        if fast_goal == "order_invoice":
            log_invoice_flow(
                intent="order_invoice",
                order_id=_resolve_order_id(original_msg, msg_en, conversation_context),
                selected_flow="message_semantics",
                invoice_status="routed",
            )
        return _apply_order_details_to_route(out, fast_goal, "message_semantics")

    followup = _infer_followup_goal_from_conversation(comb, conversation_context)
    if followup == "track_single_order":
        out.pop("order_lookup_kind", None)
        out.pop("route_handler", None)
        out["run_catalog_search"] = False
        return out
    if followup == "order_details":
        return _apply_order_details_to_route(out, "order_details", "conversation_followup")
    if followup == "order_invoice":
        log_invoice_flow(
            intent="order_invoice",
            order_id=_resolve_order_id(original_msg, msg_en, conversation_context),
            selected_flow="promote_order_details",
            invoice_status="routed",
        )
        return _apply_order_details_to_route(out, "order_invoice", "conversation_followup")

    if _message_explicitly_wants_tracking(comb, conversation_context):
        out.pop("order_lookup_kind", None)
        out.pop("route_handler", None)
        out["run_catalog_search"] = False
        return out
    if _message_explicitly_wants_order_details_not_invoice(comb, conversation_context):
        return _apply_order_details_to_route(out, "order_details", "details_semantic")
    if message_user_wants_order_invoice(comb, conversation_context, ai_route=out):
        log_invoice_flow(
            intent="order_invoice",
            order_id=_resolve_order_id(original_msg, msg_en, conversation_context),
            selected_flow="promote_order_details",
            invoice_status="routed",
        )
        return _apply_order_details_to_route(out, "order_invoice", "invoice_semantic")

    intent = (out.get("intent") or "").strip().lower()
    if intent in ("product", "deals", "categories", "category_feed") or out.get(
        "run_catalog_search"
    ):
        return out
    if message_is_catalog_product_browse_not_order_details(comb):
        return out

    try:
        from services.order_tracking_semantics import (
            message_user_rejects_refund_wants_tracking,
            message_user_wants_order_tracking,
            order_tracking_route_is_locked,
        )

        if order_tracking_route_is_locked(out):
            return out
        if message_user_rejects_refund_wants_tracking(comb):
            return out
        if message_user_wants_order_tracking(comb, conversation_context):
            return out
    except ImportError:
        pass

    if message_user_rejects_refund_wants_order_details(comb):
        return _apply_order_details_to_route(out, "order_details", "user_correction")

    olk = (out.get("order_lookup_kind") or "").strip().lower()
    rh = (out.get("route_handler") or "").strip().lower()
    if olk in ("details", "invoice") or rh == "order_details_api":
        if olk == "invoice" and not message_user_wants_order_invoice(
            comb, conversation_context, ai_route=out
        ):
            return out
        goal = "order_invoice" if olk == "invoice" else "order_details"
        return _apply_order_details_to_route(out, goal, "ai_route_field")

    if _text_wants_order_invoice(comb):
        return _apply_order_details_to_route(out, "order_invoice", "text_signal")
    if _text_wants_order_details_not_tracking(comb, conversation_context):
        return _apply_order_details_to_route(out, "order_details", "text_signal")

    if _text_clearly_wants_refund_status_not_order_details(comb, conversation_context):
        return out

    goal = infer_order_details_semantic_goal(
        original_msg, msg_en, conversation_context, ai_route=out
    )
    if goal:
        return _apply_order_details_to_route(out, goal, "order_intent")

    um = f" {(out.get('user_meaning') or '').lower()} "
    if any(x in um for x in _ORDER_DETAILS_MEANING):
        inv = any(x in um for x in ("invoice", "bill", "receipt", "gst"))
        return _apply_order_details_to_route(
            out, "order_invoice" if inv else "order_details", "ai_meaning"
        )

    return out
