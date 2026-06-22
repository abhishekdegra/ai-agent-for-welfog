"""
Product search: AI understands first, reads product-search API KB, then catalog API / OpenSearch.

Strict brand/model matching avoids wrong cross-brand results (e.g. Redmi cover when user asked iPhone cover).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from services.kb_service import read_concatenated_kb_file_contents, sysmsg
from services.translation_service import (
    customer_reply_language,
    is_hinglish_message,
    language_reply_instruction,
    localize_for_customer,
)
from utils.reasoning_log import log_reasoning

PRODUCT_SEARCH_KB_KEYS = ("welfog_api_product_search", "welfog_api")

def _localized_sysmsg(key: str, user_msg: str, reply_lang: str = "en", **fmt) -> str:
    lang = (reply_lang or "en").lower().strip()
    if lang == "hinglish" or is_hinglish_message(user_msg):
        localized = sysmsg(f"{key}_hinglish", **fmt)
        if localized:
            return localized
    en_text = sysmsg(key, **fmt) or ""
    if lang in ("en", "hinglish"):
        return en_text
    return localize_for_customer(en_text, lang) if en_text else ""


@dataclass
class ProductFlowResult:
    handled: bool
    reply_html: str = ""
    intent: str = "product"
    os_spec: dict = field(default_factory=dict)


def get_product_search_api_knowledge_context() -> str:
    blob = read_concatenated_kb_file_contents(list(PRODUCT_SEARCH_KB_KEYS))
    return blob.strip() if blob.strip() else "(Product search API knowledge missing.)"


def product_flow_hard_exclusions(combined: str, msg_en: str, original_msg: str) -> bool:
    """True when message must NOT run catalog product search (orders, PIN, policies, etc.)."""
    try:
        from services.catalog_turn_semantics import should_skip_catalog_for_conversational_turn

        if should_skip_catalog_for_conversational_turn(original_msg, msg_en):
            return True
    except ImportError:
        pass
    from utils.helpers import (
        _text_has_pincode_delivery_intent,
        extract_product_id,
        message_is_casual_offtopic_not_shopping,
        message_is_knowledge_information_request,
        message_is_welfog_about_request,
        message_needs_policy_answer,
        message_needs_support_not_product,
        message_is_bot_search_complaint,
        message_is_bot_capability_question,
        _message_has_catalog_product_signal,
        _message_has_generic_shopping_item_signal,
        _text_asks_order_history,
        _text_asks_wishlist,
        _text_is_order_id_help_request,
        _text_is_order_tracking_intent,
        _text_is_order_delivery_issue,
        _text_is_undelivered_order_complaint,
        _normalize_order_chat_text,
        message_is_wishlist_like_request,
    )

    text = _normalize_order_chat_text(f"{original_msg} {msg_en} {combined}").lower()
    comb = f"{original_msg} {msg_en} {combined}"
    if _text_has_pincode_delivery_intent(comb):
        return True
    if _text_is_order_delivery_issue(comb) or _text_is_undelivered_order_complaint(comb):
        return True
    if message_is_bot_search_complaint(comb) or message_is_bot_capability_question(comb):
        return True
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn

        if is_non_catalog_meta_turn(comb):
            return True
    except ImportError:
        pass
    if _text_is_order_tracking_intent(text):
        return True
    if (
        message_is_welfog_about_request(text)
        or message_is_knowledge_information_request(text)
        or message_is_casual_offtopic_not_shopping(text)
        or message_needs_support_not_product(text)
        or message_needs_policy_answer(text)
        or _text_asks_order_history(text)
        or _text_asks_wishlist(text)
        or message_is_wishlist_like_request(text)
        or _text_is_order_tracking_intent(text)
        or _text_is_order_id_help_request(text)
    ):
        return True
    if extract_product_id(text):
        try:
            from utils.helpers import _text_is_product_id_lookup_context

            if _text_is_product_id_lookup_context(text):
                return False
        except ImportError:
            pass
    try:
        from services.catalog_spec_semantics import user_mentions_sku_this_turn
        from services.opensearch_products import _extract_sku_from_text

        if user_mentions_sku_this_turn(original_msg or text) and _extract_sku_from_text(
            original_msg or text
        ):
            return False
    except ImportError:
        pass
    if re.search(r"\b(?:deals?|offers?|discount)\b", text) and not (
        _message_has_catalog_product_signal(text)
        or _message_has_generic_shopping_item_signal(text)
    ):
        from services.conversation_followup import is_deals_request_message

        if is_deals_request_message(original_msg or text, msg_en or ""):
            return True
        return True
    return False


def message_eligible_for_product_ai_flow(
    combined: str,
    msg_en: str,
    original_msg: str,
    *,
    ai_route: Optional[dict] = None,
    conversation_context: str = "",
    ctx: Optional[dict] = None,
) -> bool:
    """
    Gate before product_ai_flow.
    Prefer Groq intent=product; when Groq is down, allow shopping heuristics.
    """
    try:
        from services.product_catalog_resolver import turn_requests_product_catalog

        if turn_requests_product_catalog(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=True,
        ):
            return True
    except ImportError:
        pass
    try:
        from services.query_intent_classifier import query_intent_allows_catalog

        if not query_intent_allows_catalog(ctx):
            return False
    except ImportError:
        pass
    from utils.helpers import (
        _normalize_order_chat_text,
        _text_has_product_shopping_intent,
        message_is_casual_offtopic_not_shopping,
        user_continues_product_browse_from_conversation,
    )

    text = _normalize_order_chat_text(f"{original_msg} {msg_en} {combined}")
    from services.conversation_followup import is_deals_request_message
    from utils.helpers import message_asks_welfog_categories_list

    if is_deals_request_message(original_msg, msg_en) or message_asks_welfog_categories_list(text):
        return False
    if product_flow_hard_exclusions(combined, msg_en, original_msg):
        return False
    route = ai_route or {}
    intent = (route.get("intent") or "").strip().lower()
    try:
        from services.turn_intent_gate import is_non_catalog_meta_turn, message_has_catalog_search_signal

        if is_non_catalog_meta_turn(text, conversation_context):
            return False
        if intent == "product" and not message_has_catalog_search_signal(text):
            return False
    except ImportError:
        pass
    if message_is_casual_offtopic_not_shopping(text):
        return False
    if intent in ("deals", "categories", "category_feed"):
        return False
    try:
        from services.product_browse_semantics import message_is_product_availability_browse

        if message_is_product_availability_browse(text, ai_route=route):
            return True
    except ImportError:
        pass

    try:
        from services.ai_route_semantics import ai_route_allows_catalog_search
        from services.semantic_intent import llm_semantic_route_available

        if llm_semantic_route_available(route):
            if message_is_product_availability_browse(text, ai_route=route):
                return True
            return ai_route_allows_catalog_search(route)
    except ImportError:
        pass

    if intent == "product" and route.get("is_welfog_related", True):
        channel = (route.get("data_channel") or "").strip().lower()
        if channel and channel not in ("catalog", "none", ""):
            return False
        if product_flow_hard_exclusions(text, msg_en, original_msg):
            return False
        return True
    if _text_has_product_shopping_intent(text):
        try:
            from services.turn_intent_gate import message_has_catalog_search_signal

            return message_has_catalog_search_signal(text)
        except ImportError:
            return True
    if user_continues_product_browse_from_conversation(original_msg, conversation_context):
        return True
    return False


def _groq_product_search_json(system_prompt: str, user_payload: str) -> Optional[dict]:
    from services.llm_providers import llm_json_chat_completion

    return llm_json_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=420,
        timeout_sec=14,
        max_attempts=3,
    )


def ai_understand_product_search(
    user_msg: str,
    conversation_context: str = "",
    reply_lang: str = "en",
) -> Optional[dict]:
    api_kb = get_product_search_api_knowledge_context()
    system_prompt = f"""You are Welfog's product-search specialist. Read the PRODUCT SEARCH API KNOWLEDGE BASE.
Return ONLY valid JSON.

PRODUCT SEARCH API KNOWLEDGE BASE:
\"\"\"
{api_kb}
\"\"\"

JSON SCHEMA:
{{
  "reasoning": "1-3 lines",
  "action": "search_products" | "not_shopping" | "clarify",
  "product_requests": [
    {{
      "label": "short display name e.g. iPhone cover",
      "search_terms": "2-6 English words for THIS item only",
      "color": "colour for THIS item only or empty",
      "brand": "brand for THIS item or empty",
      "brand_aliases": [],
      "mandatory_match_tokens": [],
      "product_type": "cover, shirt, etc.",
      "size": "catalog size ONLY — 10, Free Size, M, XL — NEVER in search_terms",
      "exclude_title_tokens": [],
      "match_mode": "strict" | "universal",
      "max_price": number or null,
      "min_price": number or null,
      "rating_min": number or null,
      "rating_max": number or null,
      "sku": "for this item only or empty",
      "pro_id": "for this item only or null"
    }}
  ],
  "search_terms": "only when product_requests has ONE item — else empty",
  "color": "catalog colour (Black, Green, Multicolor, Sky Blue...) or empty — NEVER put in sku",
  "brand": "company/brand user asked for (any spelling you infer) or empty",
  "brand_aliases": ["spellings that may appear in titles for that brand — at least ONE must match, e.g. oneplus, one plus"],
  "mandatory_match_tokens": ["product type / feature tokens required in title — e.g. cover, velvet"],
  "product_type": "main item noun (cover, shirt, rice, charger) or empty",
  "size": "size filter only — e.g. 10, Free Size, M, XL — empty if not asked. NEVER put size in search_terms.",
  "exclude_title_tokens": ["phrases to EXCLUDE from titles, e.g. lg velvet when user wants velvet material cover"],
  "pro_id": "numeric catalog product id if user gave one, else null",
  "category_browse": "English category name when user browses a whole category (electronics, fashion, grocery, home, beauty) — empty if not category browse",
  "category_only_browse": true when user wants category products without naming a specific product type,
  "match_mode": "strict" | "universal",
  "max_price": number or null,
  "min_price": number or null,
  "rating_min": number or null (e.g. 4 for 4+ stars; 0.01 when user wants any rated product above zero),
  "rating_max": number or null (e.g. 3 for under 3 stars / low rating browse),
  "sku": "warehouse SKU ONLY if user explicitly said SKU/product code or pasted a real code (e.g. SSKKOO, TP-SamsungS25) — NEVER chahiye/dikhao/batana or any Hindi/English verb",
  "is_shopping": true/false
}}

RULES (dynamic — full Welfog ecommerce: fashion, grocery, electronics, home, beauty):
- MEANING FIRST (industry): Understand the LATEST message + RECENT CONVERSATION semantically — ANY language/script, typos, slang, long paragraphs, indirect phrasing. User may ask in 1000 different ways; infer intent from meaning, NOT from matching Hindi/English keyword lists. Do NOT echo the full user sentence into search_terms. Your JSON is the ONLY source for color, price, rating, brand, size, SKU, pro_id.
- NEVER put conversational words in sku (chahiye, dikhao, batana, please, milega, shaadi, mast, …) — sku is ONLY a warehouse code with digits or explicit "SKU ABC123" from the user.
- ALL FILTERS: Put every constraint the user meant into the correct JSON field (top-level or per product_requests[]). Long message with price + colour + product + rating → fill all fields. Multiple products with different filters → separate product_requests[] rows each with their own color/max_price/rating_min/etc.
- NEW TURN WINS: If the user changes product type or colour, do NOT carry old budget/price from chat unless they say same/wahi/pehle wala/usi range. Colour-only follow-up → update color, keep search_terms from context.
- PRO_ID: numeric product id in message → pro_id field set, search_terms="" brand="" sku="" — never search words like dikhao/dikhoa/id.
- CATEGORY BROWSE: "electronics ke products", "fashion dikhao" → category_browse=electronics/fashion, category_only_browse=true, search_terms="" unless user also named a product type.
- COLOUR (any language/script): Map semantically to catalog color — Black, White, Green, Grey, Sky Blue, etc. CSS names (DarkSlateGray), Hindi (kala, neeli), Tamil/Telugu, typos — YOU decide the catalog colour; never hardcode phrase lists in search_terms.
- COLOUR + PRODUCT: "green color ke covers", "black cover dikhao", "neeli shirt chahiye" → search_terms=cover/shirt (English product type), color=Green/Black/Blue. Never search_terms="color cover dikhao" or whole sentence.
- COLOUR FOLLOW-UP: If assistant just showed a product type and user only changes colour ("ab black", "dusra rang green", "same cover blue") → reuse product_type/search_terms from conversation; update color only.
- SIZE (critical — separate from product name): "size=10", "size 10", "10 size", "free size" → size field ONLY (e.g. "10", "Free Size", "M"). search_terms = product type ONLY: cover, shirt, case — NEVER "size 10 product cover dikhao" as one title. Example: "size=10 ka product cover dikhao" → size="10", search_terms="cover", product_type="cover". Example: "free size wali shirt" → size="Free Size", search_terms="shirt". NEVER put size number or word "size" in search_terms, sku, or mandatory_match_tokens.
- SIZE FOLLOW-UP: After showing covers, "size 10 wale" → keep search_terms=cover, set size=10 only.
- ONE ASK = ONE SEARCH (critical): "transparent cover", "green cover", "black mobile case" → exactly ONE product_requests entry (or top-level search_terms only). NEVER split into "transparent cover" AND "cover" — that shows extra wrong products. transparent/crystal/clear are MATERIAL in search_terms or mandatory_match_tokens, NOT a second product and NOT catalog color.
- MULTIPLE PRODUCTS (critical): ANY ecommerce combo in one message — phone cover + shirt, rice + oil, samsung cover + infinix cover, jeans + shoes, toy + book, etc. Return product_requests[] with ONE object per distinct item (each with search_terms, brand, color, product_type as needed). Example: "ek samsung ka aur ek iphone ka cover" → samsung cover (brand samsung, strict) + iphone cover (brand iphone, strict). Example: "ek shirt aur ek jeans" → two entries. NEVER collapse to search_terms="cover" or "product" only when user named 2+ different things.
- Also: iphone cover AND red tshirt; cover and water bottle; ek to X aur ek na Y — each with its own search_terms/brand/color.
- "show me iphone cover for my mobile and a red tshirt" → two entries: {{"search_terms": "iphone cover", "brand": "iphone"}} and {{"search_terms": "tshirt", "color": "Red"}}.
- Single product only → product_requests with exactly one object OR use top-level search_terms only (product_requests empty or one entry).
- Works for ANY Welfog catalog item (grocery, fashion, electronics, home, beauty, toys, furniture, etc.) in any allowed language — translate to English search_terms; do not limit to phones/covers/shirts.
- brand_aliases: spellings for the brand the user named (any company worldwide).
- mandatory_match_tokens: ONLY for match_mode strict + brand/model (e.g. iphone + cover). NEVER colour words.
- match_mode universal: generic shirt, jeans, charger, rice — search_terms = product type only.
- match_mode strict: user named brand/model (samsung cover, nike shoes).
- colour → color field only on the item that mentions it; NEVER copy red/blue from another product in the same sentence (iphone cover + red tshirt → red only on tshirt entry).
- colour → search_terms without colour word (black shirt → search_terms shirt, color Black).
- Exclude wrong brands: set brand_aliases to the requested brand only — do not leave empty if user said OnePlus.
- velvet cover (material) → mandatory [velvet, cover], exclude_title_tokens [lg velvet, velvet phone].
- OpenSearch uses: title_query, color, brand, brand_aliases, sku, pro_id, purchase_price min/max (customer pays purchase_price + shipping_cost on cards — NOT unit_price/MRP), size, category.
- max_price / min_price in JSON → purchase_price_max / purchase_price_min (landed price = purchase + shipping) — ONLY when user asked for rupees/budget, NEVER for star/rating thresholds.
- rating_min / rating_max in JSON for star filters — NEVER put star numbers in min_price/max_price.
- PRICE-ONLY browse (critical): "under 500 rs", "500 se kam wale item", "under 190" → max_price/min_price ONLY; search_terms="" brand="" — do NOT add Samsung/mobile/cover from old chat unless user said wahi/same/pehle wala product.
- BUDGET + PRODUCT (critical): "I have 150 rs show mobile covers", "190 rs budget mobile cover in this range" → search_terms=mobile cover (English product type), max_price=150 or 190, brand="" unless user named brand THIS message.
- Never copy brand from RECENT CONVERSATION unless user repeats that brand in the LATEST message.
- RATING filter: set rating_min / rating_max ONLY when the LATEST message explicitly mentions stars/rating — otherwise null. NEVER default rating_min=0.01 or rating_max=5. "4 star wale" → rating_min=4; "under 3 rating" → rating_max=3. Colour/price-only messages → rating fields MUST be null.
- PRICE filter: set max_price/min_price ONLY when the LATEST message mentions rs/rupees/budget/under/over price — otherwise null. NEVER copy 150/200 from old chat unless user repeats budget this message.
- PRO_ID: numeric product id in message → pro_id field set, search_terms="" brand="" sku="" — never search words like dikhao/dikhoa/id.
- CATEGORY BROWSE: whole-category asks → category_browse=English category name, category_only_browse=true, search_terms="" unless user also named a product type.
- not_shopping: orders, tracking, policies, purchase history / past orders list / "products I bought" / "is id se purchase" — is_shopping=false, action=not_shopping.
- {language_reply_instruction(reply_lang)}
JSON only."""

    user_payload = user_msg
    if (conversation_context or "").strip():
        user_payload = (
            f"RECENT CONVERSATION (secondary; LATEST message wins if different product):\n"
            f"{conversation_context.strip()[-1200:]}\n\n"
            f"LATEST USER MESSAGE:\n{user_msg}"
        )
    data = _groq_product_search_json(system_prompt, user_payload)
    if data:
        log_reasoning(
            f"Product search AI: action={data.get('action')} terms={data.get('search_terms')!r} "
            f"mode={data.get('match_mode')} — {(data.get('reasoning') or '')[:90]}"
        )
    return data


def _product_requests_redundant(requests: list[dict]) -> bool:
    """
    True only when parts are the same ask split twice (transparent cover + cover).
    NOT redundant: samsung cover + iphone cover (same product type, different brands).
    """
    if len(requests) < 2:
        return False
    terms: list[str] = []
    brands: list[str] = []
    for r in requests:
        t = (r.get("search_terms") or r.get("label") or "").strip().lower()
        if t:
            terms.append(t)
        b = (r.get("brand") or "").strip().lower()
        if b:
            brands.append(b)
    if len(terms) < 2:
        return True
    if len(set(brands)) >= 2:
        return False
    terms.sort(key=len, reverse=True)
    primary = terms[0]
    for other in terms[1:]:
        if other == primary:
            continue
        if other in primary or primary in other:
            continue
        return False
    return True


def _merge_product_requests(requests: list[dict]) -> dict:
    """Keep the most specific search_terms; merge colour/material from siblings."""
    from services.opensearch_products import extract_material_tokens, normalize_color_fuzzy

    best = max(
        requests,
        key=lambda r: len((r.get("search_terms") or r.get("label") or "").strip()),
    )
    merged = dict(best)
    for r in requests:
        c = (r.get("color") or "").strip()
        if c and not merged.get("color"):
            nc = normalize_color_fuzzy(c)
            if nc:
                merged["color"] = nc
        for tok in r.get("mandatory_match_tokens") or []:
            mlist = list(merged.get("mandatory_match_tokens") or [])
            if tok and tok not in mlist:
                mlist.append(tok)
            merged["mandatory_match_tokens"] = mlist
    mats = extract_material_tokens(
        " ".join(
            (x.get("search_terms") or x.get("label") or "")
            for x in requests
        )
    )
    if mats:
        mlist = list(merged.get("mandatory_match_tokens") or [])
        for m in mats:
            if m not in mlist:
                mlist.append(m)
        merged["mandatory_match_tokens"] = mlist
    return merged


def _heuristic_product_understanding(
    original_msg: str,
    msg_en: str = "",
    *,
    route_search_query: str = "",
) -> Optional[dict]:
    """When Groq product AI fails (rate limit), parse multi or single product from user text."""
    from services.opensearch_products import extract_color_and_product_title, normalize_color_fuzzy
    from services.product_query_understanding import (
        extract_focused_product_query,
        is_noisy_search_query,
        polish_search_terms,
    )
    from services.welfog_api import collect_multi_product_parts, multi_product_parts_are_valid

    blob = f"{original_msg} {msg_en}".strip()
    if not blob:
        return None

    sq_route = (route_search_query or "").strip()
    if sq_route:
        comma_reqs = _multi_requests_from_comma_text(sq_route, original_msg)
        if len(comma_reqs) >= 2:
            return {
                "action": "search_products",
                "is_shopping": True,
                "search_terms": "",
                "color": "",
                "product_requests": comma_reqs,
            }
    if sq_route and not is_noisy_search_query(sq_route):
        color, _ = extract_color_and_product_title(blob)
        if not color:
            color = normalize_color_fuzzy(blob)
        return {
            "action": "search_products",
            "is_shopping": True,
            "search_terms": polish_search_terms(sq_route, original_msg),
            "color": color or "",
            "product_requests": [],
        }

    parts = collect_multi_product_parts(msg_en, original_msg)
    if len(parts) >= 2 and multi_product_parts_are_valid(parts):
        reqs = [polish_multi_product_request(p, original_msg, msg_en) for p in parts]
        if len(reqs) >= 2 and not _product_requests_redundant(reqs):
            return {
                "action": "search_products",
                "is_shopping": True,
                "search_terms": "",
                "color": "",
                "product_requests": reqs,
            }

    focused = extract_focused_product_query(original_msg, msg_en)
    color, title = extract_color_and_product_title(blob)
    from services.opensearch_products import _extract_size_from_text, _strip_size_from_title_query

    size_val = _extract_size_from_text(blob)
    if not color:
        color = normalize_color_fuzzy(blob)
    if focused:
        title = focused
    elif not title:
        from services.opensearch_products import _extract_product_keywords

        title = _extract_product_keywords(blob.lower())
    if title and size_val:
        title = _strip_size_from_title_query(title, size_val)
    if title and is_noisy_search_query(title):
        refocus = extract_focused_product_query(original_msg, msg_en)
        if refocus:
            title = refocus
    if not color and not title:
        return None
    return {
        "action": "search_products",
        "is_shopping": True,
        "search_terms": title,
        "color": color or "",
        "size": size_val or "",
        "product_requests": [],
    }


def _should_use_heuristic_multi_over_ai(
    understanding: Optional[dict],
    original_msg: str,
    msg_en: str,
) -> bool:
    """Groq sometimes returns one generic search_terms='cover' — prefer text split."""
    from services.welfog_api import collect_multi_product_parts, multi_product_parts_are_valid

    parts = collect_multi_product_parts(msg_en, original_msg)
    if len(parts) < 2 or not multi_product_parts_are_valid(parts):
        return False
    if not understanding:
        return True
    raw = understanding.get("product_requests")
    if not isinstance(raw, list) or len(raw) < 2:
        st = (understanding.get("search_terms") or "").strip().lower()
        if st in ("cover", "covers", "case", "cases", "product", "products", ""):
            return True
        return True
    terms = {
        (str(x.get("search_terms") or "")).strip().lower()
        for x in raw
        if isinstance(x, dict)
    }
    terms.discard("")
    if len(terms) <= 1:
        return True
    return False


def coalesce_product_understanding(understanding: Optional[dict]) -> Optional[dict]:
    """Flatten redundant product_requests[] into one top-level search spec."""
    if not understanding:
        return understanding
    u = dict(understanding)
    raw = u.get("product_requests")
    if not isinstance(raw, list) or not raw:
        return u
    items = [x for x in raw if isinstance(x, dict)]
    if len(items) >= 2 and _product_requests_redundant(items):
        log_reasoning("Product AI: redundant product_requests — single combined search.")
        merged = _merge_product_requests(items)
        u["product_requests"] = []
        for key in (
            "search_terms", "color", "brand", "brand_aliases", "mandatory_match_tokens",
            "product_type", "exclude_title_tokens", "match_mode", "label",
            "max_price", "min_price", "rating_min", "rating_max", "sku", "pro_id", "size",
        ):
            if merged.get(key) is not None and merged.get(key) != "" and not u.get(key):
                u[key] = merged[key]
        if merged.get("search_terms"):
            u["search_terms"] = merged["search_terms"]
        if merged.get("color"):
            u["color"] = merged["color"]
        return u
    if len(items) == 1:
        one = items[0]
        if not u.get("search_terms") and one.get("search_terms"):
            u["search_terms"] = one["search_terms"]
        if not u.get("color") and one.get("color"):
            u["color"] = one["color"]
        if not u.get("brand") and one.get("brand"):
            u["brand"] = one["brand"]
        for key in (
            "max_price", "min_price", "rating_min", "rating_max", "sku", "pro_id", "size",
        ):
            if one.get(key) is not None and one.get(key) != "" and u.get(key) in (None, ""):
                u[key] = one[key]
        u["product_requests"] = []
    return u


def _scope_color_to_part_request(req: dict, *, part_raw: str = "") -> dict:
    """Keep colour only when this part's text mentions it (not from another item in the sentence)."""
    from services.opensearch_products import (
        _strip_color_from_title_query,
        resolve_color_for_part_text,
    )

    req = dict(req or {})
    terms = (req.get("search_terms") or req.get("label") or "").strip()
    raw_blob = (part_raw or terms or req.get("label") or "").strip()
    verified = resolve_color_for_part_text(raw_blob, (req.get("color") or "").strip())
    req["color"] = verified
    if not terms:
        return req
    if verified:
        stripped = _strip_color_from_title_query(terms, verified)
        if stripped:
            req["search_terms"] = stripped
        label = (req.get("label") or "").strip()
        if label:
            req["label"] = _strip_color_from_title_query(label, verified) or stripped or label
    return req


def _products_match_part_spec(
    products: list,
    spec: dict,
    ai_part: dict,
    *,
    relaxed_fallback: bool = True,
) -> list:
    """Strict match per part — block tier-2 relax showing wrong category (cover for charger)."""
    from services.opensearch_products import (
        apply_catalog_post_filters,
        conflict_exclude_tokens_for_product_type,
        filter_products_by_ai_mandatory_tokens,
        filter_products_by_exclude_tokens,
        filter_products_strict_brand_match,
    )

    if not products:
        return []
    strict = dict(spec or {})
    brand = (strict.get("brand") or ai_part.get("brand") or "").strip()
    if brand:
        strict["brand"] = brand
        strict["brand_name_match_only"] = True
        strict["title_match_strict"] = True
        aliases = strict.get("brand_aliases") or ai_part.get("brand_aliases") or [brand]
        strict["brand_aliases"] = aliases
    ptype = (strict.get("product_type") or ai_part.get("product_type") or "").strip().lower()
    mandatory = list(strict.get("mandatory_match_tokens") or [])
    if brand and brand.lower() not in [m.lower() for m in mandatory]:
        mandatory.insert(0, brand.lower())
    if ptype and ptype not in mandatory:
        mandatory.insert(0, ptype)
    if mandatory:
        strict["mandatory_match_tokens"] = mandatory[:4]
    ex = list(strict.get("exclude_title_tokens") or [])
    ex.extend(conflict_exclude_tokens_for_product_type(ptype))
    if ex:
        strict["exclude_title_tokens"] = ex
    out = apply_catalog_post_filters(products, strict, post_filter_mode="strict")
    if brand:
        out = filter_products_strict_brand_match(
            out,
            brand,
            strict.get("brand_aliases"),
        )
    if ptype and out:
        out = filter_products_by_ai_mandatory_tokens(
            out,
            [ptype],
            product_type=ptype,
            skip_color_tokens=True,
        )
    if ex:
        out = filter_products_by_exclude_tokens(out, ex)
    if not out and products and relaxed_fallback:
        return products[:12]
    return out


def _message_is_single_product_for_recipient(original_msg: str, msg_en: str = "") -> bool:
    """
    'meenakshi ke liye sunglasses' — one catalog item; recipient name is not a second product.
    """
    import re

    text = re.sub(r"\s+", " ", f"{original_msg or ''} {msg_en or ''}".lower()).strip()
    if not text or re.search(r"\b(?:aur|and|plus)\b", text):
        return False
    if not re.search(r"\bke\s+(?:liye|lie)\b", text):
        return False
    from services.welfog_api import collect_multi_product_parts, multi_product_parts_are_valid

    parts = collect_multi_product_parts(original_msg, msg_en)
    if len(parts) >= 2 and multi_product_parts_are_valid(parts):
        return False
    from services.opensearch_products import _PRODUCT_NOUNS

    nouns = [n for n in _PRODUCT_NOUNS if re.search(rf"\b{re.escape(n)}\b", text)]
    uniq = []
    for n in nouns:
        if n not in uniq:
            uniq.append(n)
    return len(uniq) <= 1


def _message_has_explicit_multi_intent(original_msg: str, msg_en: str = "") -> bool:
    """True when user clearly asked for 2+ different products (aur / comma / ek to …)."""
    import re

    comb = re.sub(r"\s+", " ", f"{original_msg or ''} {msg_en or ''}".strip())
    if not comb:
        return False
    if _message_is_single_product_for_recipient(original_msg, msg_en):
        return False
    if "," in comb:
        from services.welfog_api import multi_product_parts_are_valid

        parts = [p.strip() for p in comb.split(",") if p.strip()]
        if len(parts) >= 2 and multi_product_parts_are_valid(parts):
            return True
    if re.search(r"\b(?:aur|and|plus)\b", comb, re.I):
        from services.welfog_api import collect_multi_product_parts, multi_product_parts_are_valid

        parts = collect_multi_product_parts(msg_en, original_msg)
        return len(parts) >= 2 and multi_product_parts_are_valid(parts)
    return False


def _resolve_early_multi_product(
    original_msg: str,
    msg_en: str,
    route_sq: str,
) -> list[dict]:
    """Pre-AI multi: trust router comma-list; else split user text only when clearly multi."""
    route_sq = (route_sq or "").strip()
    try:
        from services.catalog_turn_semantics import should_skip_catalog_for_conversational_turn

        if should_skip_catalog_for_conversational_turn(original_msg, msg_en):
            return []
    except ImportError:
        pass
    if _message_is_single_product_for_recipient(original_msg, msg_en):
        return []
    if "," in route_sq:
        multi = _multi_requests_from_comma_text(route_sq, original_msg)
        if len(multi) >= 2:
            return multi
    if route_sq and "," not in route_sq and not _message_has_explicit_multi_intent(
        original_msg, msg_en
    ):
        return []

    from services.translation_service import NATIVE_SCRIPT_LANGS, customer_reply_language

    rl = customer_reply_language(original_msg)
    # Devanagari / Tamil / … → Groq product_requests (semantic); avoid Roman-only regex split.
    if rl in NATIVE_SCRIPT_LANGS:
        if "," in (route_sq or ""):
            multi = _multi_requests_from_comma_text(route_sq, original_msg)
            if len(multi) >= 2:
                return multi
        return []

    return _try_text_first_multi_parts(original_msg, msg_en)


def _multi_requests_from_comma_text(text: str, original_msg: str) -> list[dict]:
    """Groq/router often joins items: 'samsung mobile cover, infinix mobile cover'."""
    if not text or "," not in text:
        return []
    from services.welfog_api import _normalize_query_part, multi_product_parts_are_valid

    parts = [_normalize_query_part(p.strip()) for p in text.split(",") if p.strip()]
    if len(parts) < 2 or not multi_product_parts_are_valid(parts):
        return []
    reqs = [polish_multi_product_request(p, original_msg) for p in parts]
    if len(reqs) >= 2 and not _product_requests_redundant(reqs):
        return reqs[:6]
    return []


def _try_text_first_multi_parts(original_msg: str, msg_en: str) -> list[dict]:
    """Split user message before AI merge — shirt + cover, samsung + infinix, etc."""
    if _message_is_single_product_for_recipient(original_msg, msg_en):
        return []
    if not _message_has_explicit_multi_intent(original_msg, msg_en):
        return []
    from services.welfog_api import (
        collect_multi_product_parts,
        multi_product_parts_are_valid,
        repair_multi_product_joiners,
    )

    parts = collect_multi_product_parts(msg_en, original_msg)
    if len(parts) < 2 or not multi_product_parts_are_valid(parts):
        comb = repair_multi_product_joiners(f"{msg_en} {original_msg}".strip())
        parts = collect_multi_product_parts(comb, original_msg)
    if len(parts) < 2 or not multi_product_parts_are_valid(parts):
        return []
    reqs = [polish_multi_product_request(p, original_msg, msg_en) for p in parts]
    if len(reqs) >= 2 and not _product_requests_redundant(reqs):
        return reqs[:6]
    return []


def _part_has_product_category_noun(text: str) -> bool:
    """True when this split already names what to shop (jean, cover, shirt, …)."""
    import re

    from services.opensearch_products import _PRODUCT_NOUNS

    tl = f" {(text or '').lower()} "
    apparel_extra = (
        "jean", "jeans", "denim", "trouser", "trousers", "pant", "pants", "shirt", "tshirt",
        "tee", "kurta", "dress", "hoodie", "jacket", "sweater", "top", "skirt", "shorts",
        "blazer", "coat", "saree", "lehenga", "sandal", "sneaker", "boot", "belt",
    )
    for w in apparel_extra:
        if re.search(rf"\b{re.escape(w)}\b", tl):
            return True
    for n in _PRODUCT_NOUNS:
        if re.search(rf"\b{re.escape(n)}\b", tl):
            return True
    return False


def _clause_containing_part(part: str, original_msg: str, msg_en: str = "") -> str:
    """Same clause as this part — never pull nouns from the sibling item."""
    import re

    p = (part or "").strip().lower()
    msg = (msg_en or original_msg or "").strip()
    if not p or not msg:
        return original_msg or ""
    chunks = re.split(r"\s+(?:aur|and|or)\s+", msg, flags=re.IGNORECASE)
    if len(chunks) < 2:
        return msg
    best, best_score = msg, 0
    for ch in chunks:
        cl = ch.lower()
        score = sum(1 for w in p.split() if len(w) > 2 and w in cl)
        if score > best_score:
            best_score = score
            best = ch
    return best if best_score > 0 else msg


def _inherit_product_noun_from_message(
    part: str, original_msg: str, msg_en: str = ""
) -> str:
    """Bare brand fragment in one clause → add noun from that clause only (not sibling item)."""
    import re

    from services.opensearch_products import _PRODUCT_NOUNS, _extract_product_keywords

    scope = (part or "").strip()
    if not scope or not (original_msg or "").strip():
        return part
    if _part_has_product_category_noun(scope):
        return scope

    blob = _clause_containing_part(scope, original_msg, msg_en).lower()
    scope_low = scope.lower()
    for noun in sorted(_PRODUCT_NOUNS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(noun)}\b", scope_low):
            return scope
        if re.search(rf"\b{re.escape(noun)}\b", blob):
            return f"{scope} {noun}".strip()
    kw = _extract_product_keywords(blob)
    if kw and kw not in scope_low:
        return f"{scope} {kw}".strip()
    return scope


def polish_multi_product_request(
    part: str, original_msg: str = "", msg_en: str = ""
) -> dict:
    """One split segment → clean search_terms, colour, brand for OpenSearch."""
    import re

    from services.opensearch_products import normalize_color_fuzzy, _strip_color_from_title_query
    from services.product_query_understanding import clean_product_part_label, _PHONE_BRANDS, _PRODUCT_NOUNS

    part = _inherit_product_noun_from_message(part, original_msg, msg_en)
    scope_msg = (part or "").strip()
    if len(scope_msg.split()) >= 5:
        try:
            from services.product_query_understanding import extract_focused_product_query

            focused = extract_focused_product_query(scope_msg, "")
            if focused and len(focused.split()) <= 10:
                scope_msg = focused
        except Exception:
            pass
    brand_ctx = scope_msg
    from services.opensearch_products import extract_color_and_product_title

    from services.opensearch_products import resolve_color_for_part_text

    msg_color, msg_title = extract_color_and_product_title(scope_msg)
    raw = scope_msg
    raw = re.sub(
        r"\s+for\s+(?:me|my(?:\s+mobile)?|mobile|phone)\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s+for\s*$", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(
        r"\b(?:uske|unke|iske|inki|inki|inke|mere|mera|meri|liye|ke|ki|ka|ko)\b",
        " ",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s+", " ", raw).strip()
    label = clean_product_part_label(raw, scope_msg) or raw
    color = resolve_color_for_part_text(scope_msg, msg_color or "")
    if msg_title:
        terms = msg_title
        label = f"{color} {msg_title}".strip() if color else msg_title
    else:
        terms = label
    if color:
        stripped = _strip_color_from_title_query(terms, color)
        if stripped:
            terms = stripped
        label = f"{color} {terms}".strip() if color else terms
    from services.opensearch_products import _scrub_color_meta_from_title

    terms = _scrub_color_meta_from_title(terms)
    label = _scrub_color_meta_from_title(label) or terms
    brand = ""
    tl = f" {terms.lower()} "
    bl = f" {brand_ctx.lower()} "
    for b in sorted(_PHONE_BRANDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(b)}\b", tl) or re.search(rf"\b{re.escape(b)}\b", bl):
            brand = b
            if b not in terms.lower().split():
                terms = f"{b} {terms}".strip()
                label = f"{b} {label}".strip() if label else terms
            break
    ptype = ""
    for n in sorted(_PRODUCT_NOUNS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(n)}\b", tl):
            ptype = n
            break
    if ptype in ("mobile", "phone", "tablet") and re.search(
        r"\b(?:cover|case|bumper|protector)\b", tl
    ):
        ptype = "cover"
    mode = "strict" if brand else "universal"
    mandatory = []
    if ptype:
        mandatory = [ptype]
        if brand and mode == "strict" and brand not in mandatory:
            mandatory.insert(0, brand)
    from services.opensearch_products import conflict_exclude_tokens_for_product_type

    exclude = conflict_exclude_tokens_for_product_type(ptype)
    display = f"{color} {terms}".strip() if color else terms
    req = {
        "label": display,
        "search_terms": terms,
        "color": color,
        "brand": brand,
        "brand_aliases": [brand] if brand else [],
        "mandatory_match_tokens": mandatory,
        "product_type": ptype,
        "exclude_title_tokens": exclude,
        "match_mode": mode,
    }
    return _scope_color_to_part_request(req, part_raw=scope_msg)


def _request_dict_to_ai_understanding(req: dict) -> dict:
    """One product_requests[] entry → spec for build_catalog_search_spec."""
    terms = (req.get("search_terms") or req.get("label") or "").strip()
    out = {
        "search_terms": terms,
        "color": req.get("color") or "",
        "size": req.get("size") or "",
        "brand": req.get("brand") or "",
        "brand_aliases": req.get("brand_aliases") or [],
        "mandatory_match_tokens": req.get("mandatory_match_tokens") or [],
        "product_type": req.get("product_type") or "",
        "exclude_title_tokens": req.get("exclude_title_tokens") or [],
        "match_mode": req.get("match_mode") or ("strict" if req.get("brand") else "universal"),
        "is_shopping": True,
    }
    for src, dst in (
        ("max_price", "max_price"),
        ("min_price", "min_price"),
        ("rating_min", "rating_min"),
        ("rating_max", "rating_max"),
    ):
        if req.get(src) is not None:
            out[src] = req[src]
    if req.get("sku"):
        out["sku"] = req["sku"]
    if req.get("pro_id") is not None:
        out["pro_id"] = req["pro_id"]
    return out


def resolve_multi_product_requests(
    understanding: Optional[dict],
    original_msg: str,
    msg_en: str,
) -> list[dict]:
    """
    User-text split first; then AI product_requests[]; then comma-joined search_terms.
    """
    from services.opensearch_products import is_single_color_product_query

    text_multi = _try_text_first_multi_parts(original_msg, msg_en)
    if len(text_multi) >= 2:
        log_reasoning(f"Multi-product from user text: {len(text_multi)} parts.")
        return text_multi

    comb = f"{original_msg} {msg_en}".strip()
    if is_single_color_product_query(comb):
        log_reasoning("Colour+product query — single search (no multi split).")
        return []

    out: list[dict] = []
    from services.welfog_api import (
        _part_contains_multiple_product_nouns,
        collect_multi_product_parts,
        multi_product_parts_are_valid,
        repair_multi_product_joiners,
    )

    if understanding:
        raw = understanding.get("product_requests")
        if isinstance(raw, list) and len(raw) >= 2:
            for item in raw:
                if not isinstance(item, dict):
                    continue
                terms = (item.get("search_terms") or item.get("label") or "").strip()
                if not terms or _part_contains_multiple_product_nouns(terms):
                    continue
                polished = polish_multi_product_request(terms, terms)
                for k, v in item.items():
                    if not v:
                        continue
                    if k == "color":
                        continue
                    polished[k] = v
                polished = _scope_color_to_part_request(polished)
                if item.get("color"):
                    from services.opensearch_products import normalize_color_fuzzy

                    c = normalize_color_fuzzy(str(item.get("color")))
                    if c:
                        polished["color"] = c
                out.append(polished)
    if len(out) >= 2:
        if _product_requests_redundant(out):
            log_reasoning("Multi-product: redundant parts — use single catalog search.")
            return []
        return out[:6]

    sq = (understanding or {}).get("search_terms") or ""
    comma_reqs = _multi_requests_from_comma_text(str(sq).strip(), original_msg)
    if len(comma_reqs) >= 2:
        log_reasoning(f"Multi-product from comma search_terms: {len(comma_reqs)} parts.")
        return comma_reqs

    return []


def _run_multi_product_catalog_search(
    original_msg: str,
    msg_en: str,
    user_id: str,
    product_requests: list[dict],
    *,
    ctx: Optional[dict] = None,
    reply_lang: str = "en",
    conversation_context: str = "",
) -> ProductFlowResult:
    """Separate OpenSearch/API call per product; polite note for missing parts."""
    from services.opensearch_products import (
        build_catalog_search_spec,
        build_product_rail_with_pagination,
        catalog_search_live,
        is_opensearch_configured,
        sanitize_product_search_spec,
    )
    from services.product_query_understanding import (
        clean_product_part_label,
        display_label_for_product_search,
        spec_uses_strict_filter_not_found,
    )
    from services.welfog_api import (
        resolve_category_id_for_product_search,
        category_name_for_id,
        build_welfog_product_browse_url,
    )
    from utils.reasoning_log import log_reasoning

    ctx = ctx if ctx is not None else {}
    lang = reply_lang or customer_reply_language(original_msg)
    selected_cat = (ctx.get("data") or {}).get("selected_category_id")
    if not selected_cat:
        selected_cat = resolve_category_id_for_product_search(f"{original_msg} {msg_en}", ctx=ctx)

    all_products: list = []
    seen_keys: set = set()
    missing_labels: list[str] = []
    found_labels: list[str] = []
    section_blocks: list[str] = []
    per_part_limit = 6

    for req in product_requests:
        part_text = (req.get("search_terms") or req.get("label") or "").strip()
        req = _scope_color_to_part_request(req, part_raw=part_text)
        label = clean_product_part_label(part_text, part_text) or part_text or "product"
        ai_part = _request_dict_to_ai_understanding(req)
        from services.opensearch_products import resolve_color_for_part_text

        part_scope = part_text or label
        part_color = resolve_color_for_part_text(
            part_scope, (ai_part.get("color") or "")
        )
        os_spec = build_catalog_search_spec(
            part_scope,
            "",
            ai=ai_part,
            category_id=selected_cat,
            color=part_color,
        )
        if req.get("brand"):
            os_spec["brand"] = req["brand"]
            os_spec["brand_aliases"] = req.get("brand_aliases") or [req["brand"]]
        elif os_spec.get("brand"):
            b = str(os_spec.get("brand") or "").lower()
            if b and b not in part_scope.lower():
                os_spec["brand"] = ""
                os_spec["brand_aliases"] = []
        from services.opensearch_products import finalize_catalog_search_spec

        os_spec = finalize_catalog_search_spec(os_spec, part_scope, "")
        try:
            sanitize_product_search_spec(os_spec)
        except Exception:
            pass

        part_products: list = []
        if is_opensearch_configured():
            part_products, os_spec, _total, _more = catalog_search_live(
                os_spec,
                original_msg=part_scope,
                msg_en="",
                page=1,
                ctx=ctx,
            )
        else:
            from services.opensearch_products import search_products_combined

            part_products, os_spec, _total, _more = search_products_combined(
                label,
                original_msg=part_text or label,
                msg_en="",
                category_id=selected_cat,
                color=os_spec.get("color"),
                title_hint=os_spec.get("title_query"),
                page=1,
                ctx=ctx,
            )

        part_products = _products_match_part_spec(
            part_products, os_spec, ai_part, relaxed_fallback=False
        )
        display_l = (
            clean_product_part_label(part_text or label, original_msg)
            or display_label_for_product_search(os_spec, ai_part, part_text)
            or label
        )
        added = 0
        if part_products:
            found_labels.append(display_l)
            section_blocks.append(
                sysmsg("products_multi_section_title", label=display_l)
                or f"<div style='margin-top:14px;font-weight:600;'>For <b>{display_l}</b>:</div>"
            )
            section_blocks.append(
                build_product_rail_with_pagination(
                    part_products[:per_part_limit],
                    sysmsg,
                    has_more=False,
                    next_page=2,
                    browse_more_url=build_welfog_product_browse_url(os_spec, ctx=ctx),
                )
            )
            for item in part_products[:per_part_limit]:
                key = item.get("slug") or item.get("id") or item.get("pro_id")
                if key is not None and key in seen_keys:
                    continue
                if key is not None:
                    seen_keys.add(key)
                all_products.append(item)
                added += 1
        if not added:
            miss = display_l.strip()
            if req.get("brand") and req["brand"].lower() not in miss.lower():
                miss = f"{req['brand'].title()} {miss}".strip()
            if miss and miss not in missing_labels:
                missing_labels.append(miss)
            section_blocks.append(
                sysmsg("products_multi_section_title", label=display_l)
                or f"<div style='margin-top:14px;font-weight:600;'>For <b>{display_l}</b>:</div>"
            )
            section_blocks.append(
                _localized_sysmsg(
                    "products_multi_part_not_found",
                    original_msg,
                    reply_lang=lang,
                    label=display_l,
                )
                or sysmsg("products_multi_part_not_found", label=display_l)
                or (
                    f"<div style='margin-top:6px;color:#777;font-size:13px;'>"
                    f"No products found for <b>{display_l}</b> on Welfog right now.</div>"
                )
            )
            log_reasoning(f"Multi-product: no hits for {miss!r}")

    labels_joined = ", ".join(
        clean_product_part_label(
            (r.get("label") or r.get("search_terms") or ""), original_msg
        )
        or r.get("search_terms", "")
        for r in product_requests
    )
    if not all_products:
        if missing_labels and len(missing_labels) < len(product_requests):
            intro = (
                _localized_sysmsg("products_multi_intro", original_msg, reply_lang=lang, items=labels_joined)
                or sysmsg("products_multi_intro", items=labels_joined)
                or ""
            )
            unavailable = ", ".join(f"'{x}'" for x in missing_labels)
            body = (intro or "") + (
                _localized_sysmsg(
                    "products_partial_missing", original_msg, reply_lang=lang, items=unavailable
                )
                or sysmsg("products_partial_missing", items=unavailable)
                or ""
            )
            return ProductFlowResult(handled=True, reply_html=body, intent="product")
        if spec_uses_strict_filter_not_found({}):
            body = _localized_sysmsg(
                "products_filtered_not_found", original_msg, reply_lang=lang, query=labels_joined
            )
        else:
            body = _localized_sysmsg(
                "product_not_found", original_msg, reply_lang=lang, query=labels_joined
            ) or sysmsg("product_not_found", query=labels_joined)
        return ProductFlowResult(handled=True, reply_html=body or "", intent="product")

    if found_labels:
        found_joined = ", ".join(found_labels)
        if missing_labels:
            intro = (
                _localized_sysmsg(
                    "products_multi_partial_intro", original_msg, reply_lang=lang, found=found_joined
                )
                or sysmsg("products_multi_partial_intro", found=found_joined)
                or sysmsg("products_multi_intro", items=found_joined)
            )
        else:
            intro = (
                _localized_sysmsg("products_multi_intro", original_msg, reply_lang=lang, items=found_joined)
                or sysmsg("products_multi_intro", items=found_joined)
                or sysmsg("products_title_query", query=found_joined)
            )
    elif missing_labels:
        intro = ""
    else:
        intro = (
            _localized_sysmsg("products_multi_intro", original_msg, reply_lang=lang, items=labels_joined)
            or sysmsg("products_multi_intro", items=labels_joined)
            or sysmsg("products_title_query", query=labels_joined)
        )
    response_text = (intro or "") + "".join(section_blocks)
    if missing_labels:
        missing_only = ", ".join(missing_labels)
        if found_labels:
            found_joined = ", ".join(found_labels)
            response_text += (
                _localized_sysmsg(
                    "products_partial_missing_named",
                    original_msg,
                    reply_lang=lang,
                    missing=missing_only,
                    found=found_joined,
                )
                or sysmsg(
                    "products_partial_missing_named",
                    missing=missing_only,
                    found=found_joined,
                )
                or ""
            )
        else:
            unavailable = ", ".join(f"'{x}'" for x in missing_labels)
            response_text += (
                _localized_sysmsg(
                    "products_partial_missing", original_msg, reply_lang=lang, items=unavailable
                )
                or sysmsg("products_partial_missing", items=unavailable)
                or ""
            )

    log_reasoning(
        f"Multi-product search: {len(product_requests)} parts, "
        f"{len(all_products)} cards, missing={missing_labels}"
    )
    return ProductFlowResult(handled=True, reply_html=response_text, intent="product")


def _run_locked_catalog_search_fast(
    original_msg: str,
    msg_en: str,
    user_id: str,
    *,
    ctx: Optional[dict] = None,
    reply_lang: str = "en",
    search_query: str = "",
    conversation_context: str = "",
    ai_understanding: Optional[dict] = None,
    ai_route: Optional[dict] = None,
) -> ProductFlowResult:
    """
    Brain/structural locked product browse — proper filter spec, then OpenSearch.
    Never treat brain search_query (with price/rating text) as raw title_query.
    """
    from services.kb_service import sysmsg
    from services.opensearch_products import (
        build_product_rail_with_pagination,
        catalog_search_live,
        product_search_show_view_more,
    )
    from services.product_filter_pipeline import (
        _build_catalog_spec_from_locked_entities,
        _build_related_accessory_understanding,
        _enrich_brain_entities_structural,
        _structural_gap_ai_from_message,
        apply_catalog_result_filters,
        brain_search_query_is_noisy,
        log_product_search_pipeline,
    )
    from services.product_query_understanding import (
        display_label_for_product_search,
        spec_uses_strict_filter_not_found,
    )
    from services.welfog_api import build_welfog_product_browse_url

    sq = (
        (search_query or "").strip()
        or ((ai_route or {}).get("search_query") or "").strip()
        or ((ai_understanding or {}).get("search_terms") or "").strip()
    )
    try:
        from services.ai_route_semantics import _catalog_search_query_from_brain_route

        sq_brain = _catalog_search_query_from_brain_route(ai_route)
        if sq_brain:
            sq = sq_brain
    except ImportError:
        pass
    locked_u = dict(ai_understanding or {})
    entities = dict((ai_route or {}).get("_product_entities") or {})
    entities = _enrich_brain_entities_structural(
        entities, original_msg, msg_en, brain_route=ai_route
    )
    if entities and ai_route is not None:
        ai_route = dict(ai_route)
        ai_route["_product_entities"] = entities

    try:
        from services.product_catalog_resolver import entities_to_understanding

        eu = entities_to_understanding(
            entities, search_query=sq, original_msg=original_msg
        )
        if eu:
            locked_u = eu
    except ImportError:
        pass

    if sq and not brain_search_query_is_noisy(sq) and not locked_u.get("search_terms"):
        locked_u["search_terms"] = sq
    if locked_u:
        locked_u["_ai_first"] = True

    pre_fallback_u = dict(locked_u)

    gap = _structural_gap_ai_from_message(
        original_msg,
        msg_en,
        locked_understanding=locked_u,
        brain_search_query=sq,
    )
    os_spec, pq_llm = _build_catalog_spec_from_locked_entities(
        original_msg,
        msg_en,
        gap=gap,
        ctx=ctx,
        ai_route=ai_route,
        locked_understanding=gap,
        brain_search_query=sq,
    )
    os_spec["_ai_single_pass"] = True
    log_reasoning(
        "Catalog brain-locked entity spec — skip duplicate product NLU LLM."
    )
    log_reasoning(
        f"Catalog locked path: title={os_spec.get('title_query')!r} "
        f"brand={os_spec.get('brand')!r} color={os_spec.get('color')!r} "
        f"price_max={os_spec.get('purchase_price_max')!r} "
        f"rating_min={os_spec.get('rating_min')!r} pro_id={os_spec.get('pro_id')!r} "
        f"sku={os_spec.get('sku')!r} category_id={os_spec.get('category_id')!r}"
    )

    ctx = ctx if ctx is not None else {}
    products, os_spec, os_total, os_has_more = catalog_search_live(
        os_spec,
        original_msg=original_msg,
        msg_en=msg_en,
        page=1,
        ctx=ctx,
    )
    raw_count = len(products)
    products = apply_catalog_result_filters(products, os_spec)
    removed_reason = ""
    if not products and raw_count == 0:
        req_brand = (os_spec.get("_requested_brand") or os_spec.get("brand") or "").strip()
        mandatory_device = list(os_spec.get("mandatory_match_tokens") or [])
        if req_brand and not mandatory_device:
            retry_spec = dict(os_spec)
            retry_spec.pop("brand", None)
            retry_spec.pop("brand_aliases", None)
            retry_spec.pop("brand_name_match_only", None)
            retry_spec["title_match_strict"] = False
            tq = (retry_spec.get("title_query") or "").strip()
            if req_brand.lower() not in tq.lower():
                retry_spec["title_query"] = f"{tq} {req_brand}".strip() if tq else req_brand
            retry_products, retry_spec, retry_total, retry_more = catalog_search_live(
                retry_spec,
                original_msg=original_msg,
                msg_en=msg_en,
                page=1,
                ctx=ctx,
            )
            retry_products = apply_catalog_result_filters(retry_products, retry_spec)
            if retry_products:
                log_reasoning(
                    f"Catalog post-filter brand-relax: {len(retry_products)} hit(s) "
                    f"with title brand match (field filter dropped)."
                )
                products = retry_products
                os_spec = retry_spec
                os_total = retry_total
                os_has_more = retry_more
    elif not products and raw_count > 0:
        removed_reason = "post_filter_mismatch"

    if not products and locked_u.get("device_browse") and not locked_u.get("specific_accessory"):
        fb_u = _build_related_accessory_understanding(locked_u, entities)
        if fb_u:
            fb_gap = _structural_gap_ai_from_message(
                original_msg,
                msg_en,
                locked_understanding=fb_u,
                brain_search_query=fb_u.get("search_terms") or "",
            )
            fb_spec, _fb_pq = _build_catalog_spec_from_locked_entities(
                original_msg,
                msg_en,
                gap=fb_gap,
                ctx=ctx,
                ai_route=ai_route,
                locked_understanding=fb_gap,
                brain_search_query=fb_u.get("search_terms") or "",
            )
            fb_spec["_related_fallback"] = True
            fb_spec["_ai_single_pass"] = True
            log_reasoning(
                "Catalog device browse: 0 phones — retry related accessories "
                f"(title={fb_spec.get('title_query')!r})."
            )
            fb_products, fb_spec, fb_total, fb_more = catalog_search_live(
                fb_spec,
                original_msg=original_msg,
                msg_en=msg_en,
                page=1,
                ctx=ctx,
            )
            fb_products = apply_catalog_result_filters(fb_products, fb_spec)
            if fb_products:
                products = fb_products
                os_spec = fb_spec
                os_total = fb_total
                os_has_more = fb_more
                locked_u = fb_u
                removed_reason = ""

    route_entities = None
    if ai_route and isinstance(ai_route.get("_product_entities"), dict):
        route_entities = ai_route.get("_product_entities")
    log_product_search_pipeline(
        original_msg=original_msg,
        msg_en=msg_en,
        ai_understanding=pq_llm or locked_u,
        spec=os_spec,
        products=products,
        reply_lang=reply_lang or "en",
        product_entities=route_entities,
        total_results=os_total,
        removed_reason=removed_reason,
    )

    display_query = display_label_for_product_search(os_spec, pq_llm or locked_u, original_msg)

    if not products:
        if spec_uses_strict_filter_not_found(os_spec):
            body = _localized_sysmsg(
                "products_filtered_not_found",
                original_msg,
                reply_lang=reply_lang,
                query=display_query,
            ) or sysmsg("products_filtered_not_found", query=display_query)
        else:
            body = _localized_sysmsg(
                "product_not_found", original_msg, reply_lang=reply_lang, query=display_query
            ) or sysmsg("product_not_found", query=display_query)
        return ProductFlowResult(handled=True, reply_html=body, intent="product", os_spec=os_spec)

    if os_spec.get("_related_fallback"):
        device_label = display_label_for_product_search(
            os_spec, pq_llm or pre_fallback_u, original_msg
        )
        if not device_label or device_label.lower() in ("products", "product", "your search"):
            device_label = (
                (pre_fallback_u.get("search_terms") or sq or "this item").strip()
            )
        response_text = _localized_sysmsg(
            "products_related_fallback",
            original_msg,
            reply_lang=reply_lang,
            query=device_label,
            related=display_query,
        ) or sysmsg(
            "products_related_fallback",
            query=device_label,
            related=display_query,
        )
    else:
        response_text = sysmsg("products_title_query", query=display_query)
    browse_url = build_welfog_product_browse_url(os_spec, ctx=ctx)
    response_text += build_product_rail_with_pagination(
        products,
        sysmsg,
        has_more=product_search_show_view_more(products, os_has_more),
        next_page=2,
        browse_more_url=browse_url,
    )
    return ProductFlowResult(
        handled=True,
        reply_html=response_text,
        intent="product",
        os_spec=os_spec,
    )


def _run_catalog_search(
    original_msg: str,
    msg_en: str,
    user_id: str,
    *,
    ctx: Optional[dict] = None,
    reply_lang: str = "en",
    search_query: str = "",
    conversation_context: str = "",
    ai_understanding: Optional[dict] = None,
    ai_route: Optional[dict] = None,
) -> ProductFlowResult:
    from services.opensearch_products import (
        build_catalog_search_spec,
        build_product_rail_with_pagination,
        catalog_search_live,
        cheapest_alternative_hint,
        is_opensearch_configured,
    )
    from services.product_query_understanding import (
        clean_product_part_label,
        display_label_for_product_search,
        is_noisy_search_query,
        spec_uses_strict_filter_not_found,
        understand_product_query,
    )
    from utils.reasoning_log import log_reasoning
    from services.welfog_api import (
        category_name_for_id,
        resolve_category_id_for_product_search,
    )
    from utils.helpers import (
        extract_product_id,
        extract_product_search_query,
        message_needs_policy_answer,
        message_needs_support_not_product,
    )

    ctx = ctx if ctx is not None else {}
    lang = reply_lang or customer_reply_language(original_msg)
    comb = f"{original_msg} {msg_en}".strip()

    try:
        from services.product_catalog_resolver import turn_requests_product_catalog

        product_turn = turn_requests_product_catalog(
            original_msg, msg_en, conversation_context, ai_route=ai_route, allow_llm=False
        )
    except ImportError:
        product_turn = False
    if not product_turn and (
        message_needs_support_not_product(comb) or message_needs_policy_answer(comb)
    ):
        return ProductFlowResult(handled=False)

    sku_lookup = (ai_understanding or {}).get("sku") if ai_understanding else None
    if not sku_lookup:
        try:
            from services.opensearch_products import _extract_sku_from_text

            sku_lookup = _extract_sku_from_text(f"{original_msg} {msg_en}")
        except ImportError:
            sku_lookup = None

    catalog_single_pass = bool(
        (ai_route or {}).get("_ai_single_pass")
        or (ai_understanding or {}).get("_ai_first")
    )
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        catalog_single_pass = catalog_single_pass or product_catalog_route_is_locked(ai_route)
    except ImportError:
        pass

    locked_browse_sq = (
        catalog_single_pass
        and (
            (search_query or "").strip()
            or ((ai_route or {}).get("search_query") or "").strip()
            or ((ai_understanding or {}).get("search_terms") or "").strip()
        )
    )

    selected_cat = (ctx.get("data") or {}).get("selected_category_id")
    if sku_lookup:
        selected_cat = None
        ctx.setdefault("data", {})["lookup_sku"] = str(sku_lookup).strip()
    elif not selected_cat and not catalog_single_pass:
        selected_cat = resolve_category_id_for_product_search(f"{original_msg} {msg_en}", ctx=ctx)
        if selected_cat:
            ctx.setdefault("data", {})["selected_category_id"] = selected_cat

    from services.opensearch_products import (
        color_hue_mentioned_in_text,
        extract_color_and_product_title,
        normalize_color_fuzzy,
    )
    from services.welfog_api import _normalize_color

    selected_color = (ctx.get("data") or {}).get("selected_color")
    full_blob = f"{original_msg} {msg_en}".strip()
    detected_color, _ = extract_color_and_product_title(full_blob)
    if not detected_color:
        fuzzy = normalize_color_fuzzy(full_blob) or _normalize_color(msg_en)
        if fuzzy and color_hue_mentioned_in_text(fuzzy, full_blob):
            detected_color = fuzzy
    if detected_color:
        selected_color = detected_color

    products: list = []
    os_spec: dict = {}
    pq_llm = None
    os_has_more = False
    os_total = 0
    title_match = ""

    if ai_understanding:
        terms_ai = (ai_understanding.get("search_terms") or "").strip()
        if terms_ai:
            title_match = terms_ai
        from services.product_query_understanding import sanitize_llm_color

        llm_color = sanitize_llm_color(
            str(ai_understanding.get("color") or ""),
            terms_ai,
            original_msg,
        )
        if llm_color:
            selected_color = llm_color
        if ai_understanding.get("brand"):
            ctx.setdefault("data", {})["_pending_brand"] = ai_understanding.get("brand")

    lookup_pro_id = (ctx.get("data") or {}).get("lookup_pro_id")
    if not lookup_pro_id and not locked_browse_sq:
        lookup_pro_id = extract_product_id(f"{original_msg} {msg_en}")
    if not lookup_pro_id and ai_understanding:
        ai_pid = ai_understanding.get("pro_id")
        if ai_pid is not None and str(ai_pid).strip().isdigit():
            lookup_pro_id = int(str(ai_pid).strip())

    if lookup_pro_id:
        from services.welfog_api import fetch_product_by_pro_id

        log_reasoning(f"Catalog pro_id lookup (pro_id={lookup_pro_id}).")
        card = fetch_product_by_pro_id(int(lookup_pro_id))
        if not card and is_opensearch_configured():
            os_pro_spec = {"pro_id": int(lookup_pro_id), "title_query": ""}
            os_products, _, _, _ = catalog_search_live(
                os_pro_spec, original_msg=original_msg, page=1
            )
            if os_products:
                card = os_products[0]
        if card:
            filter_label = f"Product ID {lookup_pro_id}"
            response_text = sysmsg("product_id_lookup_title", pro_id=lookup_pro_id) or sysmsg(
                "products_title_query", query=filter_label
            )
            from services.opensearch_products import product_search_show_view_more

            response_text += build_product_rail_with_pagination(
                [card],
                sysmsg,
                has_more=False,
                next_page=2,
                browse_more_url=card.get("link") or "",
            )
            return ProductFlowResult(
                handled=True,
                reply_html=response_text,
                intent="product",
                os_spec={"pro_id": int(lookup_pro_id)},
            )
        body = (
            _localized_sysmsg("product_not_found", original_msg, reply_lang=lang, query=str(lookup_pro_id))
            or sysmsg("product_not_found", query=str(lookup_pro_id))
        )
        return ProductFlowResult(
            handled=True,
            reply_html=body,
            intent="product",
            os_spec={"pro_id": int(lookup_pro_id)},
        )

    if catalog_single_pass and ai_understanding:
        from services.product_filter_pipeline import build_catalog_spec_for_product_turn

        sq_fast = (search_query or (ai_route or {}).get("search_query") or "").strip()
        os_spec, pq_llm = build_catalog_spec_for_product_turn(
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            reply_lang=lang,
            ctx=ctx,
            ai_route=ai_route,
            locked_understanding=ai_understanding,
            brain_search_query=sq_fast,
            allow_product_nlu_llm=False,
        )
        title_match = (os_spec.get("title_query") or "").strip()
        log_reasoning(f"Catalog ultra-fast path: OpenSearch sq={title_match!r}")
    else:
        os_spec = build_catalog_search_spec(
            original_msg,
            msg_en,
            ai=ai_understanding,
            category_id=selected_cat,
            color=selected_color,
            pro_id=int(lookup_pro_id) if lookup_pro_id else None,
            ctx=ctx,
            ai_route=ai_route,
        )
        try:
            from services.opensearch_products import reconcile_catalog_spec_with_user_turn

            os_spec = reconcile_catalog_spec_with_user_turn(
                os_spec,
                original_msg,
                msg_en,
                ctx=ctx,
                ai_route=ai_route,
                ai_understanding=ai_understanding,
            )
        except ImportError:
            pass
        try:
            from services.opensearch_products import _normalize_generic_title_query, sanitize_product_search_spec

            _normalize_generic_title_query(os_spec, original_msg)
            sanitize_product_search_spec(os_spec)
        except ImportError:
            pass
        if not ai_understanding:
            from services.product_query_understanding import extract_focused_product_query

            focused = extract_focused_product_query(original_msg, msg_en)
            os_spec, pq_llm = understand_product_query(
                original_msg,
                msg_en,
                conversation_context,
                lang,
                category_id=selected_cat,
                color=selected_color,
                pro_id=int(lookup_pro_id) if lookup_pro_id else None,
            )
            os_spec = build_catalog_search_spec(
                original_msg,
                msg_en,
                ai=pq_llm,
                category_id=selected_cat,
                color=os_spec.get("color") or selected_color,
                pro_id=int(lookup_pro_id) if lookup_pro_id else None,
                ctx=ctx,
                ai_route=ai_route,
            )
            if focused and not (os_spec.get("title_query") or "").strip():
                os_spec["title_query"] = focused
            try:
                from services.opensearch_products import reconcile_catalog_spec_with_user_turn

                os_spec = reconcile_catalog_spec_with_user_turn(
                    os_spec, original_msg, msg_en, ctx=ctx, ai_route=ai_route
                )
            except ImportError:
                pass

        title_match = os_spec.get("title_query") or title_match or ""

        if selected_cat:
            from services.welfog_api import category_browse_search_name, query_should_use_category_id_only, strip_category_browse_conflicts_from_spec

            combined_q = title_match or comb
            if query_should_use_category_id_only(selected_cat, combined_q, ctx):
                os_spec["title_query"] = ""
                title_match = ""
                os_spec = strip_category_browse_conflicts_from_spec(os_spec, ctx=ctx)
            else:
                stripped_name = category_browse_search_name(selected_cat, combined_q, ctx)
                if stripped_name != combined_q:
                    os_spec["title_query"] = stripped_name
                    title_match = stripped_name

    if is_opensearch_configured():
        products, os_spec, os_total, os_has_more = catalog_search_live(
            os_spec,
            original_msg=original_msg,
            msg_en=msg_en,
            page=1,
            ctx=ctx,
        )
    else:
        from services.opensearch_products import search_products_combined

        log_reasoning("OpenSearch URL not set — live REST product API only.")
        products, os_spec, os_total, os_has_more = search_products_combined(
            original_msg,
            original_msg=original_msg,
            msg_en=msg_en,
            category_id=selected_cat,
            color=os_spec.get("color"),
            title_hint=os_spec.get("title_query"),
            pro_id=int(lookup_pro_id) if lookup_pro_id else None,
            page=1,
            ctx=ctx,
        )

    try:
        from services.product_filter_pipeline import log_product_search_pipeline

        route_entities = None
        if ai_route and isinstance(ai_route.get("_product_entities"), dict):
            route_entities = ai_route.get("_product_entities")
        log_product_search_pipeline(
            original_msg=original_msg,
            msg_en=msg_en,
            ai_understanding=ai_understanding,
            spec=os_spec,
            products=products,
            reply_lang=lang,
            product_entities=route_entities,
        )
    except ImportError:
        pass

    if selected_cat:
        ctx.setdefault("data", {})
        ctx["data"]["last_os_spec"] = {
            k: os_spec.get(k)
            for k in (
                "title_query", "color", "size", "brand", "brand_aliases", "mandatory_match_tokens",
                "product_type", "sku", "pro_id", "category_id",
                "unit_price_min", "unit_price_max", "purchase_price_min", "purchase_price_max",
                "rating_min", "rating_max", "title_match_strict",
            )
            if os_spec.get(k) is not None
        }
        ctx["data"]["product_page"] = 1
        from services.welfog_api import build_welfog_product_browse_url

        ctx["data"]["product_browse_url"] = build_welfog_product_browse_url(os_spec, ctx=ctx)

    if not os_spec.get("_ai_single_pass"):
        if os_spec.get("color") and products:
            from services.opensearch_products import filter_products_by_requested_color

            color_filtered = filter_products_by_requested_color(products, os_spec["color"])
            if color_filtered:
                products = color_filtered
            elif spec_uses_strict_filter_not_found(os_spec or {}):
                products = []

        if products and os_spec:
            from services.opensearch_products import apply_catalog_post_filters, _post_filter_mode_for_spec

            products = apply_catalog_post_filters(
                products,
                os_spec,
                post_filter_mode=_post_filter_mode_for_spec(os_spec),
            )
    else:
        from services.product_filter_pipeline import apply_catalog_result_filters

        products = apply_catalog_result_filters(products, os_spec)

    ctx.setdefault("data", {})
    ctx["data"]["last_os_spec"] = {
        k: os_spec.get(k)
        for k in (
            "title_query", "color", "size", "brand", "brand_aliases", "mandatory_match_tokens",
            "product_type", "sku", "pro_id", "category_id",
            "unit_price_min", "unit_price_max", "purchase_price_min", "purchase_price_max",
            "rating_min", "rating_max", "title_match_strict",
        )
        if os_spec.get(k) is not None
    }

    if not products:
        fallback_query = display_label_for_product_search(os_spec or {}, pq_llm, original_msg)
        if is_noisy_search_query(fallback_query):
            fallback_query = title_match or "products"
        try:
            from services.catalog_spec_semantics import user_mentions_sku_this_turn

            sku_turn = user_mentions_sku_this_turn(f"{original_msg} {msg_en}")
        except ImportError:
            sku_turn = bool((os_spec or {}).get("sku"))
        if sku_turn and (os_spec or {}).get("sku"):
            body = (
                _localized_sysmsg("product_not_found", original_msg, reply_lang=lang, query=str(os_spec["sku"]))
                or sysmsg("product_not_found", query=str(os_spec["sku"]))
            )
            return ProductFlowResult(handled=True, reply_html=body, intent="product", os_spec=os_spec or {})
        if spec_uses_strict_filter_not_found(os_spec or {}):
            body = _localized_sysmsg(
                "products_filtered_not_found",
                original_msg,
                reply_lang=lang,
                query=fallback_query,
            ) or sysmsg("products_filtered_not_found", query=fallback_query)
            alt = cheapest_alternative_hint(os_spec)
            if alt:
                body += alt
        else:
            body = (
                _localized_sysmsg("product_not_found", original_msg, reply_lang=lang, query=fallback_query)
                or sysmsg("product_not_found", query=fallback_query)
            )
        return ProductFlowResult(handled=True, reply_html=body, intent="product", os_spec=os_spec or {})

    filter_label = display_label_for_product_search(os_spec or {}, pq_llm or ai_understanding, original_msg)
    title_from_spec = (title_match or (os_spec.get("title_query") or "").strip())
    if catalog_single_pass and title_from_spec:
        display_query = filter_label or title_from_spec
    else:
        display_query = filter_label or title_from_spec or clean_product_part_label(
            extract_product_search_query(
                original_msg, msg_en, title_from_spec, ai_route=ai_route
            ),
            original_msg,
        )
    if is_noisy_search_query(display_query):
        display_query = filter_label or title_match or "products"

    if selected_cat and not (os_spec.get("title_query") or "").strip() and not filter_label:
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
    from services.welfog_api import build_welfog_product_browse_url

    browse_url = build_welfog_product_browse_url(os_spec or {}, ctx=ctx)
    response_text += build_product_rail_with_pagination(
        products,
        sysmsg,
        has_more=product_search_show_view_more(products, os_has_more),
        next_page=2,
        browse_more_url=browse_url,
    )
    return ProductFlowResult(
        handled=True,
        reply_html=response_text,
        intent="product",
        os_spec=os_spec or {},
    )


def run_product_search_ai_flow(
    original_msg: str,
    msg_en: str,
    user_id: str,
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    ctx: Optional[dict] = None,
    search_query: str = "",
    ai_route: Optional[dict] = None,
) -> ProductFlowResult:
    comb = f"{original_msg} {msg_en}"
    from utils.helpers import extract_product_id

    # Brain/direct locked catalog — skip gates, classifiers, and duplicate order-intent LLMs.
    try:
        from services.product_catalog_resolver import (
            product_catalog_route_is_locked,
            understanding_from_locked_product_route,
        )

        if product_catalog_route_is_locked(ai_route):
            work_route = dict(ai_route) if isinstance(ai_route, dict) else {}
            try:
                from services.ai_route_semantics import _brain_product_entities_from_route
                from services.product_filter_pipeline import _enrich_brain_entities_structural

                ent = _brain_product_entities_from_route(
                    work_route, original_msg=original_msg, msg_en=msg_en
                )
                ent = _enrich_brain_entities_structural(
                    ent, original_msg, msg_en, brain_route=work_route
                )
                if ent:
                    work_route["_product_entities"] = ent
                    pn = (ent.get("product_name") or "").strip()
                    if pn and pn.lower() not in ("products", "product"):
                        work_route["search_query"] = pn
                ai_route = work_route
            except ImportError:
                pass

            route_sq_brain = (
                (search_query or "").strip()
                or ((ai_route or {}).get("search_query") or "").strip()
            )
            locked_u = understanding_from_locked_product_route(
                ai_route,
                route_sq_brain,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            if not locked_u and route_sq_brain and route_sq_brain.lower() not in (
                "products",
                "product",
            ):
                locked_u = {
                    "action": "search_products",
                    "is_shopping": True,
                    "search_terms": route_sq_brain,
                    "_ai_first": True,
                }
            if not locked_u:
                locked_u = {
                    "action": "search_products",
                    "is_shopping": True,
                    "_ai_first": True,
                }
            log_reasoning(
                "Product brain-locked fast path — skip gates/classifiers → OpenSearch."
            )
            if isinstance(ai_route, dict):
                ai_route.setdefault("_ai_single_pass", True)
            return _run_locked_catalog_search_fast(
                original_msg,
                msg_en,
                user_id,
                ctx=ctx,
                reply_lang=reply_lang,
                search_query=route_sq_brain,
                conversation_context=conversation_context,
                ai_understanding=locked_u,
                ai_route=ai_route,
            )
    except ImportError:
        pass

    pid_early = (ctx.get("data") or {}).get("lookup_pro_id") if ctx else None
    if not pid_early:
        pid_early = extract_product_id(comb)
    if not pid_early and isinstance(ai_route, dict):
        route_ent = ai_route.get("_product_entities") or {}
        route_pid = route_ent.get("product_id") or route_ent.get("pro_id")
        if route_pid is not None and str(route_pid).strip().isdigit():
            pid_early = int(str(route_pid).strip())
    try:
        from services.opensearch_products import _extract_sku_from_text
        from services.catalog_spec_semantics import user_mentions_sku_this_turn

        sku_turn = (
            user_mentions_sku_this_turn(original_msg)
            and _extract_sku_from_text(original_msg)
        )
    except ImportError:
        sku_turn = False
    if sku_turn and ctx is not None:
        ctx.setdefault("data", {}).pop("lookup_pro_id", None)
        pid_early = None

    if pid_early and ctx is not None:
        ctx.setdefault("data", {})["lookup_pro_id"] = pid_early
    if pid_early:
        return _run_catalog_search(
            original_msg,
            msg_en,
            user_id,
            ctx=ctx,
            reply_lang=reply_lang,
            search_query=f"pro_id {pid_early}",
            conversation_context=conversation_context,
            ai_understanding={"pro_id": pid_early, "action": "search_products", "is_shopping": True},
            ai_route=ai_route,
        )

    try:
        from services.catalog_turn_semantics import should_skip_catalog_for_conversational_turn

        if should_skip_catalog_for_conversational_turn(
            original_msg, msg_en, conversation_context
        ):
            log_reasoning("Product search skipped — conversational/greeting turn (not catalog).")
            return ProductFlowResult(handled=False)
    except ImportError:
        pass
    if product_flow_hard_exclusions(comb, msg_en, original_msg):
        return ProductFlowResult(handled=False)

    route_sq_early = (search_query or (ai_route or {}).get("search_query") or "").strip()
    try:
        from utils.helpers import _message_looks_like_shopping_query

        skip_meta = bool(route_sq_early) or _message_looks_like_shopping_query(comb)
    except ImportError:
        skip_meta = bool(route_sq_early)

    if not skip_meta:
        from services.turn_intent_gate import classify_meta_turn, format_meta_turn_reply

        meta = classify_meta_turn(original_msg, msg_en, conversation_context)
        if meta:
            body = format_meta_turn_reply(meta, original_msg, reply_lang=reply_lang)
            if body:
                return ProductFlowResult(handled=True, reply_html=body, intent="general")
            return ProductFlowResult(handled=False)

    try:
        from services.opensearch_products import _extract_sku_from_text

        sku_early = _extract_sku_from_text(original_msg)
        if not sku_early and msg_en.strip().lower() != (original_msg or "").strip().lower():
            sku_early = _extract_sku_from_text(msg_en)
        if sku_early:
            log_reasoning(f"Product search fast path — explicit SKU ({sku_early!r}).")
            if ctx is not None:
                ctx.setdefault("data", {})["lookup_sku"] = sku_early
                ctx["data"].pop("selected_category_id", None)
            return _run_catalog_search(
                original_msg,
                msg_en,
                user_id,
                ctx=ctx,
                reply_lang=reply_lang,
                search_query="",
                conversation_context=conversation_context,
                ai_understanding={
                    "sku": sku_early,
                    "search_terms": "",
                    "action": "search_products",
                    "is_shopping": True,
                },
                ai_route=ai_route,
            )
    except ImportError:
        pass
    from utils.helpers import message_is_casual_offtopic_not_shopping

    if message_is_casual_offtopic_not_shopping(comb):
        from services.kb_service import sysmsg
        from services.translation_service import customer_reply_language, localize_for_customer

        rl = reply_lang or customer_reply_language(original_msg)
        from services.off_topic_reply import build_off_topic_polite_reply

        body = build_off_topic_polite_reply(original_msg, msg_en, prefer_llm=False) or ""
        if body and rl not in ("en", "hinglish"):
            body = localize_for_customer(body, rl)
        if body:
            log_reasoning("Product flow blocked — personal/off-topic message.")
            return ProductFlowResult(handled=True, reply_html=body, intent="out_of_domain")
        return ProductFlowResult(handled=False)
    if not message_eligible_for_product_ai_flow(
        comb,
        msg_en,
        original_msg,
        ai_route=ai_route,
        conversation_context=conversation_context,
    ):
        return ProductFlowResult(handled=False)

    route_sq = (search_query or (ai_route or {}).get("search_query") or "").strip()
    understanding = None
    catalog_locked = False
    try:
        from services.product_catalog_resolver import (
            entities_to_understanding,
            log_product_catalog_routing,
            product_catalog_route_is_locked,
            understanding_from_locked_product_route,
        )

        catalog_locked = product_catalog_route_is_locked(ai_route)
        locked_u = understanding_from_locked_product_route(
            ai_route,
            route_sq,
            original_msg=original_msg,
            msg_en=msg_en,
        )
        if locked_u:
            log_reasoning(
                "Product search fast path — AI entities locked (one OpenSearch pass)."
            )
            return _run_locked_catalog_search_fast(
                original_msg,
                msg_en,
                user_id,
                ctx=ctx,
                reply_lang=reply_lang,
                search_query=route_sq,
                conversation_context=conversation_context,
                ai_understanding=locked_u,
                ai_route=ai_route,
            )
        if catalog_locked and isinstance(ai_route, dict):
            route_entities = ai_route.get("_product_entities") or {}
            if route_entities:
                understanding = entities_to_understanding(
                    route_entities,
                    search_query=route_sq,
                    original_msg=original_msg,
                )
                if understanding:
                    log_reasoning(
                        "Product search — locked route entities (skip duplicate classifier LLM)."
                    )
                    log_product_catalog_routing(
                        detected_intent="product_search",
                        product_entities=route_entities,
                        selected_route="product_ai_flow",
                        filters=route_entities,
                        source="ai_route_locked",
                    )
                    return _run_locked_catalog_search_fast(
                        original_msg,
                        msg_en,
                        user_id,
                        ctx=ctx,
                        reply_lang=reply_lang,
                        search_query=route_sq,
                        conversation_context=conversation_context,
                        ai_understanding=understanding,
                        ai_route=ai_route,
                    )
            if not understanding:
                log_reasoning(
                    "Product catalog locked — force fast OpenSearch (no Product-catalog LLM)."
                )
                return _run_locked_catalog_search_fast(
                    original_msg,
                    msg_en,
                    user_id,
                    ctx=ctx,
                    reply_lang=reply_lang,
                    search_query=route_sq,
                    conversation_context=conversation_context,
                    ai_route=ai_route,
                )
    except ImportError:
        pass

    if not understanding and not catalog_locked:
        try:
            from services.product_catalog_resolver import (
                entities_to_understanding,
                log_product_catalog_routing,
                resolve_product_search_turn,
            )

            resolved = resolve_product_search_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                allow_llm=True,
            )
            if resolved.kind == "product_search":
                route_entities = (ai_route or {}).get("_product_entities") or resolved.entities
                if resolved.search_query and not route_sq:
                    route_sq = resolved.search_query
                if route_entities:
                    understanding = entities_to_understanding(
                        route_entities,
                        search_query=resolved.search_query or route_sq,
                        original_msg=original_msg,
                    )
                    if understanding and resolved.search_query and not understanding.get(
                        "search_terms"
                    ):
                        understanding["search_terms"] = resolved.search_query
                log_product_catalog_routing(
                    detected_intent="product_search",
                    product_entities=route_entities,
                    selected_route="product_ai_flow",
                    filters=route_entities or {},
                    source=resolved.source or "product_search_flow",
                )
        except ImportError:
            pass

    skip_parser = bool(route_sq) or catalog_locked or bool(understanding and understanding.get("_ai_first"))
    if not skip_parser:
        try:
            from services.product_intent_parser import ProductIntentParser

            parsed = ProductIntentParser.parse(
                original_msg, msg_en, conversation_context, ai_route=ai_route
            )
        except ImportError:
            parsed = None
    else:
        parsed = None
    if parsed is not None:
        if parsed.needs_clarification and parsed.confidence < 0.7:
            from services.kb_service import sysmsg

            body = (
                parsed.clarification_prompt
                or sysmsg("product_search_clarify")
                or "Which product are you looking for — brand, type, or colour?"
            )
            log_reasoning("Product NLU low confidence — ask clarification.")
            return ProductFlowResult(handled=True, reply_html=body, intent="product")
        if parsed.filters.product_id:
            return _run_catalog_search(
                original_msg,
                msg_en,
                user_id,
                ctx=ctx,
                reply_lang=reply_lang,
                search_query=f"pro_id {parsed.filters.product_id}",
                conversation_context=conversation_context,
                ai_understanding=parsed.to_understanding_dict(),
                ai_route=ai_route,
            )
        understanding = parsed.to_understanding_dict()
        understanding["_nlu_source"] = parsed.source
        log_reasoning(
            f"Product NLU ({parsed.source}): intent={parsed.intent} "
            f"query={parsed.product_query!r} brand={parsed.filters.brand!r} "
            f"color={parsed.filters.color!r} conf={parsed.confidence:.2f}"
        )

    skip_extra_nlu = catalog_locked or bool(
        understanding and understanding.get("_ai_first")
    )
    if not skip_extra_nlu:
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            skip_extra_nlu = product_catalog_route_is_locked(ai_route) or bool(route_sq)
        except ImportError:
            skip_extra_nlu = bool(route_sq)

    if route_sq and not understanding:
        understanding = _heuristic_product_understanding(
            original_msg, msg_en, route_search_query=route_sq
        )
        if understanding:
            log_reasoning(
                f"Product search (heuristic-first): terms={understanding.get('search_terms')!r}"
            )

    if not understanding and not skip_extra_nlu:
        understanding = ai_understand_product_search(
            original_msg.strip() or msg_en.strip(),
            conversation_context=conversation_context,
            reply_lang=reply_lang,
        )
    elif understanding and understanding.get("_ai_first"):
        log_reasoning(
            "Product search — skip second product NLU LLM (AI-first filters already set)."
        )
    if not understanding:
        understanding = _heuristic_product_understanding(
            original_msg, msg_en, route_search_query=route_sq
        )
        if understanding:
            log_reasoning(
                f"Heuristic product parse: terms={understanding.get('search_terms')!r} "
                f"color={understanding.get('color')!r}"
            )
    elif _should_use_heuristic_multi_over_ai(understanding, original_msg, msg_en):
        h_multi = _heuristic_product_understanding(
            original_msg, msg_en, route_search_query=route_sq
        )
        if h_multi and h_multi.get("product_requests"):
            log_reasoning("AI merged multi-item query — using heuristic product_requests[].")
            understanding = h_multi
    try:
        from services.catalog_spec_semantics import scrub_ai_product_understanding

        understanding = scrub_ai_product_understanding(
            understanding, original_msg, msg_en
        )
    except ImportError:
        pass
    understanding = coalesce_product_understanding(understanding)
    if understanding:
        try:
            from services.product_query_understanding import (
                is_noisy_search_query,
                polish_search_terms,
                scrub_conversational_tail_from_terms,
            )

            st = scrub_conversational_tail_from_terms(
                (understanding.get("search_terms") or "").strip()
            )
            if st:
                st = polish_search_terms(st, original_msg)
                if st and not is_noisy_search_query(st):
                    understanding["search_terms"] = st
                else:
                    understanding["search_terms"] = ""
        except ImportError:
            pass
    try:
        from services.product_filter_pipeline import apply_understanding_sku_pro_id_fixes

        understanding = apply_understanding_sku_pro_id_fixes(
            understanding, original_msg, msg_en
        )
    except ImportError:
        pass

    multi_reqs = resolve_multi_product_requests(understanding, original_msg, msg_en)

    if len(multi_reqs) >= 2:
        log_reasoning(f"Multi-product flow: {len(multi_reqs)} separate catalog searches.")
        return _run_multi_product_catalog_search(
            original_msg,
            msg_en,
            user_id,
            multi_reqs,
            ctx=ctx,
            reply_lang=reply_lang,
            conversation_context=conversation_context,
        )

    if understanding:
        action = (understanding.get("action") or "search_products").strip().lower()
        catalog_locked = False
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            catalog_locked = product_catalog_route_is_locked(ai_route) or bool(
                route_sq or search_query
            )
        except ImportError:
            catalog_locked = bool(route_sq or search_query)
        if (
            not catalog_locked
            and (not understanding.get("is_shopping", True) or action == "not_shopping")
        ):
            log_reasoning("Product search AI: not_shopping — try order-history flow.")
            from services.order_history_flow import run_order_ai_flow

            of = run_order_ai_flow(
                original_msg,
                msg_en,
                user_id,
                conversation_context=conversation_context,
                reply_lang=reply_lang,
                ai_route=ai_route,
            )
            if of.handled and of.reply_html:
                return ProductFlowResult(
                    handled=True,
                    reply_html=of.reply_html,
                    intent=of.intent or "order_history",
                )
            return ProductFlowResult(handled=False)
        if catalog_locked and (
            not understanding.get("is_shopping", True) or action == "not_shopping"
        ):
            log_reasoning(
                "Product route locked — ignore not_shopping NLU; run catalog search."
            )
            if route_sq and not understanding.get("search_terms"):
                understanding["search_terms"] = route_sq
            understanding["is_shopping"] = True
            understanding["action"] = "search_products"
        if action == "clarify":
            body = (
                _localized_sysmsg("product_search_clarify", original_msg, reply_lang=reply_lang)
                or (
                    "Which product are you looking for? Tell me the <b>type</b> "
                    "(cover, shirt, rice…) and <b>brand/model</b> if you have one — "
                    "and colour if it matters."
                )
            )
            return ProductFlowResult(handled=True, reply_html=body, intent="general")

    return _run_catalog_search(
        original_msg,
        msg_en,
        user_id,
        ctx=ctx,
        reply_lang=reply_lang,
        search_query=search_query,
        conversation_context=conversation_context,
        ai_understanding=understanding,
        ai_route=ai_route,
    )
