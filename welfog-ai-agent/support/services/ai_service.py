import json
import os
import re
import time

import requests

from services.kb_service import get_runtime_knowledge_files, read_concatenated_kb_file_contents
from services.translation_service import language_reply_instruction, resolve_customer_reply_lang
from utils.reasoning_log import log_reasoning

_ROUTING_MASTER_MAX_CHARS = 1500
_ROUTING_SYSTEM_MAX_CHARS = 4800
_ROUTING_CONTEXT_MAX_CHARS = 1200
_ROUTING_USER_MAX_CHARS = 1200
_ANSWER_CONTEXT_MAX_CHARS = 2200
_ANSWER_SYSTEM_MAX_CHARS = 3800
_LLM_TIMEOUT_SEC = max(8, min(45, int(os.getenv("AI_TIMEOUT", "20") or 20)))


def _safe_print(msg: str) -> None:
    try:
        print((msg or "").encode("ascii", errors="replace").decode("ascii"))
    except Exception:
        pass


def _routing_master_for_prompt() -> str:
    files_map = get_runtime_knowledge_files()
    path = files_map.get("welfog_api_routing_master")
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                body = (f.read() or "").strip()
            if body:
                return body
        except OSError:
            pass
    return "(See intent list in schema.)"


def _trim_text_mid(text: str, max_chars: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    keep_head = max(200, int(max_chars * 0.45))
    keep_tail = max(260, max_chars - keep_head - 24)
    return f"{raw[:keep_head].rstrip()}\n...[truncated]...\n{raw[-keep_tail:].lstrip()}"


def _compact_conversation_context(conversation_context: str, max_chars: int) -> str:
    raw = (conversation_context or "").strip()
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 60:
        lines = lines[-60:]
    preferred = [
        ln
        for ln in lines
        if ln.lower().startswith("user:")
        or ln.lower().startswith("assistant:")
        or ln.lower().startswith("customer:")
        or ln.lower().startswith("bot:")
    ]
    compact = "\n".join(preferred[-24:] if preferred else lines[-24:])
    return _trim_text_mid(compact, max_chars)


def _shrink_groq_payload(req: dict, factor: float = 0.72) -> dict:
    out = dict(req or {})
    msgs = list(out.get("messages") or [])
    shrunk = []
    for idx, m in enumerate(msgs):
        content = (m or {}).get("content") or ""
        if isinstance(content, str):
            content = _trim_text_mid(content, 6500 if idx == 0 else 1800)
        shrunk.append({"role": (m or {}).get("role") or "user", "content": content})
    out["messages"] = shrunk
    out["max_tokens"] = max(140, int((out.get("max_tokens") or 320) * factor))
    return out


def _llm_provider_chain() -> list[dict]:
    from services.llm_providers import get_llm_provider_chain

    return get_llm_provider_chain()


def _llm_classifier_provider_chain() -> list[dict]:
    """
    JSON classifiers / routing: env chain (LLM_PROVIDER_ORDER), capped for latency.
    Set LLM_CLASSIFIER_MAX_PROVIDERS=4 in .env to try all keys; default 2 (groq→openai).
    """
    from services.llm_providers import get_configured_provider_chain

    chain = get_configured_provider_chain()
    cap_default = (
        "4"
        if (os.getenv("STRICT_LLM_FAILSAFE", "") or "").strip().lower()
        in ("1", "true", "yes", "on")
        else "2"
    )
    cap = max(1, min(4, int(os.getenv("LLM_CLASSIFIER_MAX_PROVIDERS", cap_default) or cap_default)))
    return chain[:cap] if chain else []


def _llm_routing_provider_chain() -> list[dict]:
    """
    ai_brain_route — try full env chain (groq→openai→gemini→deepseek by default).
    LLM_ROUTING_MAX_PROVIDERS caps how many keys to try (default 4 = all configured).
    """
    from services.llm_providers import get_configured_provider_chain

    chain = get_configured_provider_chain()
    cap = max(1, min(4, int(os.getenv("LLM_ROUTING_MAX_PROVIDERS", "4") or "4")))
    prefer_groq = (os.getenv("LLM_ROUTING_PREFER_GROQ", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if prefer_groq and chain:
        groq = [p for p in chain if (p.get("name") or "").strip().lower() == "groq"]
        rest = [p for p in chain if (p.get("name") or "").strip().lower() != "groq"]
        chain = groq + rest if groq else list(chain)
    return chain[:cap] if chain else []


def _llm_json_with_provider_fallback(
    providers: list[dict],
    messages: list[dict],
    max_tokens: int,
    timeout_sec: int = _LLM_TIMEOUT_SEC,
    max_attempts: int = 3,
    temperature: float = 0.0,
):
    from services.llm_providers import llm_json_with_provider_fallback

    return llm_json_with_provider_fallback(
        providers,
        messages,
        max_tokens,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        temperature=temperature,
    )


def ai_brain_route(user_msg, conversation_context: str = "", reply_lang: str = "en"):
    """
    Step 1: Use Groq to understand the user message (any language),
    decide intent + which knowledge files should be used for grounding.
    """
    try:
        from services.chat_flow_telemetry import (
            guard_duplicate_brain_route,
            store_brain_route_result,
        )

        cached = guard_duplicate_brain_route("ai_brain_route")
        if cached is not None:
            return cached
    except ImportError:
        pass

    reply_lang = resolve_customer_reply_lang(user_msg, reply_lang)
    try:
        providers = _llm_routing_provider_chain()
        if not providers:
            _safe_print(
                "ERROR: No AI provider key found — set GROQ/OPENAI/GEMINI/DEEPSEEK API keys."
            )
            return None

        kb_keys_list = ", ".join(
            [
                f'"{k}"'
                for k in get_runtime_knowledge_files().keys()
                if k != "welfog_api_routing_master"
            ][:28]
        )
        routing_master = _trim_text_mid(_routing_master_for_prompt(), _ROUTING_MASTER_MAX_CHARS)
        system_prompt = f"""You are 'Welfog AI' routing brain. Classify the LATEST user message using MEANING and RECENT CONVERSATION — never keyword lists.

ROUTING PLAYBOOK (topics → intent → API vs KB):
\"\"\"
{routing_master}
\"\"\"

Customer-facing knowledge keys (kb_keys when intent needs KB):
[{kb_keys_list}]

JSON SCHEMA (LATEST USER MESSAGE ONLY):
{{
  "user_meaning": "One clear English sentence: what the customer wants THIS turn (never copy their message verbatim).",
  "reasoning": "Why you chose intent/channel — short English paragraph.",
  "intent": "product" | "order" | "order_history" | "wishlist" | "refund" | "payment" | "seller" | "pincode_check" | "deals" | "categories" | "category_feed" | "general" | "out_of_domain",
  "data_channel": "live_api" | "catalog" | "kb" | "none",
  "run_catalog_search": true/false,
  "meta_kind": "none" | "hostile" | "bot_latency" | "topic_denial" | "wrong_search_complaint" | "conversational" | "assistant_intro",
  "kb_keys": ["keys for KB answers only — empty if live_api/catalog handles it"],
  "search_query": "English PRODUCT TYPE only (2-6 words) when run_catalog_search=true — jeans, mobile cover, iphone, pajama. NEVER brand/color/price/time words. Must match product_entities.product_name. Empty when not shopping.",
  "product_entities": {{
    "product_name": "English product type to search (mobile cover, jeans, iphone, water bottle) — no brand/color/price",
    "brand": "Brand if mentioned (Redmi, Nike, Apple) else empty",
    "color": "Color if mentioned (black, blue) else empty",
    "size": "Size if mentioned (S, M, L, kids) else empty",
    "price_min": null,
    "price_max": null,
    "rating_min": null,
    "sku": "Exact SKU preserving hyphens (Xiaomi-SK) or empty",
    "product_id": "Numeric Welfog product id when user gave pro_id/product id, else empty",
    "model": "Device/model if mentioned (iPhone 15, Redmi Note 10, Galaxy S22) else empty",
    "product_intent": "device or accessory or general",
    "allow_related_fallback": true,
    "related_search_terms": "English fallback product type if device absent, else empty string",
    "exclude_title_tokens": ["optional English title words to exclude for this query"],
    "mandatory_match_tokens": ["optional English title tokens required for this query"]
  }},
  "category_browse": "English Welfog department name when user browses a whole category (electronics, men fashion, women fashion, beauty, home kitchen) — empty if not category browse",
  "category_id": "numeric Welfog top-level category id when known, else empty",
  "category_only_browse": true when user wants products from a department without naming a specific product/model/SKU,
  "extracted_pincode": "6-digit PIN if pincode_check, else empty",
  "needs_order_id": true/false,
  "is_welfog_related": true/false,
  "continue_previous_topic": true/false,
  "numeric_context": "pincode" | "order_id" | "product_id" | "none",
  "reuse_user_value_from_chat": "pincode" | "order_id" | "",
  "answer_strategy": "live_api_only" | "kb_only" | "kb_then_ai" | "api_then_ai" | "api_kb_ai" | "catalog_only" | "structured_handler",
  "conversation_scope": "welfog_support" | "general_chitchat" | "out_of_domain" | "harm_sensitive",
  "scope_reply": "For general_chitchat/out_of_domain/harm_sensitive: 2-5 sentences in customer language (harm=empathetic safety, no KB/legal). Empty when welfog_support.",
  "account_list_kind": "none" | "wishlist_in_chat" | "wishlist_howto" | "purchase_history_in_chat" | "purchase_history_howto",
  "order_lookup_kind": "none" | "track" | "details" | "invoice" | "refund_status",
  "field_focus": "timeline" | "summary" | "payment" | "product" | "delivery" | "status" | "invoice" | "",
  "route_handler": "order_tracking_api" | "order_details_api" | "refund_status_api" | "order_history_api" | "wishlist_api" | "pincode_delivery_api" | "" 
}}

CORE RULES (latest message only; follow ROUTING PLAYBOOK for details):
- Any Indian language / Hinglish / typos / slang — YOU understand meaning. Write user_meaning as one clear English sentence. NEVER echo the customer message word-for-word.
- Backend routes ONLY from your JSON (intent, data_channel, order_lookup_kind, route_handler, field_focus, kb_keys) — there is NO keyword fallback on customer text. If a field is wrong or missing, the wrong API runs.
- ORDER SUB-INTENT (critical — customer ANY language/style; YOU decide; backend trusts your JSON only):
  * details → order_lookup_kind=details, route_handler=order_details_api
  * track → order_lookup_kind=track, route_handler=order_tracking_api
  * invoice → order_lookup_kind=invoice, route_handler=order_details_api
  * refund_status → order_lookup_kind=refund_status, route_handler=refund_status_api
  ALWAYS set BOTH order_lookup_kind AND route_handler when needs_order_id=true for one order.
  MANDATORY: when intent=order and needs_order_id=true you MUST set order_lookup_kind (never leave none). data_channel MUST be live_api (never "order"). Put sub-intent in order_lookup_kind — NOT in conversation_scope (never conversation_scope=order_details).
  details = full order info (items/payment/address/amount). track = shipment/ETA/courier only. Never confuse them.
  field_focus when order_lookup_kind=details: delivery (shipping address on order), payment, product, summary, status. field_focus=timeline only for track.
  Typos / Hinglish / voice-to-text: involve/invoce/invois/bill/chalan/maang rha + order id → invoice (NOT track).
- Full purchase list IN CHAT ("show me my order history", "you show orders", any language) → intent=order_history, account_list_kind=purchase_history_in_chat, data_channel=live_api, needs_order_id=false. NOT faqs.txt how-to steps.
- HOW/WHERE to open order history in the app only (steps, navigation) → account_list_kind=purchase_history_howto, data_channel=kb, route order_history_howto — NOT purchase-history API.
- ONE order: set order_lookup_kind — track (ETA/shipment/courier), details (payment/product/summary), invoice (bill/receipt download), refund_status (MY refund/return status for one order). Match intent=order/refund/payment, needs_order_id=true, numeric_context=order_id.
- Personal refund/return status for ONE order (any language, any wording) → intent=refund, order_lookup_kind=refund_status, data_channel=live_api, needs_order_id=true. NOT general refund policy KB.
- General refund policy / how to return / refund time / process (no personal order status) → intent=refund or general, data_channel=kb, kb_keys include refund — NOT return-request API.
- Infer personal refund status vs policy from user_meaning — do NOT match fixed Hindi/English keyword lists in the customer message.
- ONE order track/shipment timeline → order_lookup_kind=track, field_focus=timeline. Invoice/bill/receipt/GST → invoice (NOT track). Payment/amount/total/product on ONE order → order_lookup_kind=details with matching field_focus (NOT order_history list, NOT track).
- Shipping/delivery ADDRESS already saved on an existing order (any language: address kya laga / konsa address / pata kya tha / which address on order) → order_lookup_kind=details, field_focus=delivery, route_handler=order_details_api, needs_order_id=true. NOT order_lookup_kind=track, NOT pincode_check (pincode_check is only for whether Welfog delivers to an area before ordering).
- "Order ID: 12345 invoice/refund/address/amount" → set order_lookup_kind from what they asked (invoice/refund_status/details), NOT track, NOT purchase_history_in_chat.
- Saved/liked products IN CHAT ("meri wishlist dikhao", "meri pasand ki cheezein", "saved items", any language) → intent=wishlist, account_list_kind=wishlist_in_chat, data_channel=live_api, run_catalog_search=false. NOT product catalog search.
- HOW/WHERE to view wishlist in app (steps, navigation, "wishlist kaise dekhu") → account_list_kind=wishlist_howto, data_channel=kb — NOT pincode_check, NOT product catalog.
- Saved/liked products → intent=wishlist (NOT order_history). Amazon/Flipkart etc. → out_of_domain.
- PIN / delivery area / "can you deliver to X" / city name (Jaipur, Kota) / friend lives in an area (ANY language) → pincode_check, data_channel=live_api, needs_order_id=false, order_lookup_kind=none, run_catalog_search=false. This is NOT order tracking — never set intent=order or needs_order_id=true for hypothetical delivery to a place. Clear city → live geocode+API; vague place → ask for 6-digit PIN. Extract PIN when present.
- Existing order timeline (shipment/courier/status/not received) → order + order_lookup_kind=track, needs_order_id=true. Do not confuse with pincode_check.
- Product browse/buy (ANY product, brand, color, price, SKU, product id — ANY language/style) → intent=product, data_channel=catalog, run_catalog_search=true. Fill product_entities with ALL filters the user asked for; search_query = product_entities.product_name (product TYPE only in English).
- PRODUCT ENTITIES (critical — backend uses these for OpenSearch filters, not raw customer text):
  * product_name = what to find (jeans, mobile cover, iphone, pajama, water bottle, track pants) — NEVER put brand/color/price/time words (night, black, redmi) in product_name.
  * brand = manufacturer/company ONLY (Redmi, Infinix, Nike, BoAt) — NEVER product type.
  * NEVER set brand to product_name or product type (mobile cover, iphone cover, shirt are NOT brands).
  * If user did not name a brand, leave brand="" — search by product_name/title only.
  * "iphone cover" without "apple" → product_name=iphone cover, brand="" (NOT Apple).
  * "under 150 rs" / "150 se kam" / "150 rs ke andar" / "200rs ke andar andar" → price_max=number. ALWAYS set price_max when user gives a rupee budget — keep brand/color filters too.
  * "above 500" → price_min=500.
  * product_intent + allow_related_fallback + related_search_terms: REQUIRED on every product_entities — backend has no fixed product keyword lists.
  * device (phone/handset/laptop device): product_intent=device, allow_related_fallback=true, related_search_terms=closest related Welfog catalog type in English, exclude_title_tokens=accessory words that would pollute THIS device search (you choose per product).
  * exact accessory/part (cover, charger, cable, jeans, shirt, bottle…): product_intent=accessory or general, allow_related_fallback=false, related_search_terms="".
  * "redmi mobile cover" → product_name=mobile cover, brand=Redmi.
  * "black mobile covers" → product_name=mobile cover, color=black.
  * product id / pro_id + cover → numeric_context=product_id, product_id=that number, product_name=mobile cover (accessory for that product).
  * SKU "Xiaomi-SK" → sku=Xiaomi-SK (keep hyphen exactly). numeric_context=none for SKU-only turns.
  * pajama / night wear → product_name=pajama or nightwear — NEVER put "night" in brand.
  * kids / bachche → size=kids when user shops for children.
  * Hinglish patterns (MUST fill product_entities — never leave empty on product turns):
    - "infinix mobile ka cover chahiye" / "dost ko X cover" → product_name=mobile cover, brand=Infinix (or named brand).
    - "redmi ke covers dikha" / "X ke cover dikhao" → product_name=mobile cover, brand=Redmi (brand from X).
    - "behan ke liye iphone cover" → product_name=mobile cover, brand=Apple (iphone accessory).
    - ANY brand user names (BoAt, Noise, Itel, Lava, etc.) → put in product_entities.brand — do not limit to famous brands.
  * mandatory_match_tokens: when user names ANY specific model/device/brand product (any language/style), set 1-4 English title tokens that MUST appear in catalog results — you infer from meaning, not fixed lists. Examples: iphone cover → ["iphone"]; philips trimmer → ["philips","trimmer"]; redmi note 10 case → ["redmi","note"]. Do NOT use generic type words alone (cover, shirt, bottle). REQUIRED when a named product/model/brand is the focus.
  * product_entities is REQUIRED when run_catalog_search=true — never empty product_name AND empty brand on a named-product browse.
- Today's deals / offers / discounts / flash sale (ANY language) → intent=deals, data_channel=live_api, run_catalog_search=false, needs_order_id=false. NOT a product named "deals".
- Full Welfog category/department list (ANY language — what sections can I shop, show all categories) → intent=categories, data_channel=live_api, run_catalog_search=false. NOT company/about KB.
- Products inside ONE named category (beauty, electronics, grocery, home kitchen, men/women fashion) → intent=product, data_channel=catalog, category_browse=English department name, category_only_browse=true, search_query="" unless user also named a product type.
- NEVER set run_catalog_search=true with search_query categories/deals/offers — those are catalog MENU APIs, not product SKUs.
- Read-only questions (policy, FAQ, company, seller, fees, contact, how-to) in ANY language → set intent + data_channel=kb + kb_keys from ROUTING PLAYBOOK. Use user_meaning in English to pick files — do NOT rely on matching Hindi/English keywords in the customer message. Seller → intent=seller. Grievance → company+privacy KB. Wrong/damaged item policy → refund+kb (NOT order_history). Off-topic → out_of_domain. Self-harm → harm_sensitive, no KB.
- Who is THIS bot → meta_kind=assistant_intro. What is Welfog company → kb company (NOT assistant_intro).
- Greeting / thanks / bye / "how are you" / "what are you doing" / "are you free busy" (ANY language) → conversation_scope=general_chitchat, scope_reply=warm natural reply in customer language, run_catalog_search=false. NEVER product search for words like free/busy when user talks to the bot.
- run_catalog_search=false for meta_kind other than none.
- {language_reply_instruction(reply_lang)}
Return JSON only."""
        system_prompt = _trim_text_mid(system_prompt, _ROUTING_SYSTEM_MAX_CHARS)

        msg_for_route = _trim_text_mid(user_msg, _ROUTING_USER_MAX_CHARS)
        compact_ctx = _compact_conversation_context(conversation_context, _ROUTING_CONTEXT_MAX_CHARS)
        user_payload = msg_for_route
        if compact_ctx:
            user_payload = (
                "RECENT CONVERSATION (use to resolve pronouns and follow-ups):\n"
                f"{compact_ctx}\n\n"
                f"LATEST USER MESSAGE:\n{msg_for_route}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        log_reasoning(f"Calling LLM routing ({len(system_prompt)} char system prompt)...")
        route_timeout = max(8, min(14, int(os.getenv("AI_ROUTE_TIMEOUT", "12") or 12)))
        import time as _time

        _t_route = _time.perf_counter()
        out = _llm_json_with_provider_fallback(
            providers,
            messages,
            max_tokens=max(420, int(os.getenv("AI_ROUTE_MAX_TOKENS", "520") or 520)),
            timeout_sec=route_timeout,
            max_attempts=max(1, min(3, int(os.getenv("AI_ROUTE_MAX_ATTEMPTS", "2") or 2))),
        )
        _route_ms = (_time.perf_counter() - _t_route) * 1000.0
        prov_names = "→".join((p.get("name") or "?") for p in providers)
        log_reasoning(f"LLM routing done in {_route_ms:.0f}ms (chain: [{prov_names}])")
        if not out:
            from services.chat_resilience import get_last_llm_failure

            kind = get_last_llm_failure() or "all_failed"
            log_reasoning(f"LLM routing returned no JSON — failure={kind}")
            return {
                "llm_unavailable": True,
                "_llm_failure": kind,
                "intent": "general",
                "conversation_scope": "welfog_support",
                "is_welfog_related": True,
            }
        if out:
            try:
                from services.chat_flow_telemetry import store_brain_route_result

                if isinstance(out, dict):
                    store_brain_route_result(out)
            except ImportError:
                pass
            try:
                from services.ai_route_semantics import (
                    _normalize_llm_route,
                    repair_brain_json_quality,
                )

                out = _normalize_llm_route(out)
                out = repair_brain_json_quality(out, user_msg)
                um = (out.get("user_meaning") or "").strip()
                log_reasoning(
                    um
                    or out.get("reasoning")
                    or f"Routing: intent={out.get('intent')} meta={out.get('meta_kind')}"
                )
            except ImportError:
                log_reasoning(out.get("reasoning") or "Routing completed.")
        try:
            from services.chat_flow_telemetry import store_brain_route_result

            if isinstance(out, dict):
                store_brain_route_result(out)
        except ImportError:
            pass
        return out
            
    except Exception as e:
        _safe_print(f"AI Brain Error: {e}")
        return None


def _groq_json_with_retry(url, headers, payload, timeout_sec=12, max_attempts=3, provider_name="provider"):
    """Backward-compatible alias; uses shared multi-provider retry logic."""
    from services.llm_providers import llm_json_with_retry

    return llm_json_with_retry(
        url, headers, payload, timeout_sec, max_attempts, provider_name
    )


def ai_brain_answer(user_msg, kb_context, conversation_context: str = "", reply_lang: str = "en"):
    """
    Step 2: Use Groq with selected KB context to generate final answer JSON.
    """
    reply_lang = resolve_customer_reply_lang(user_msg, reply_lang)
    try:
        providers = _llm_provider_chain()
        if not providers:
            _safe_print(
                "ERROR: No AI provider key found — set GROQ/OPENAI/GEMINI/DEEPSEEK API keys."
            )
            return None

        system_prompt = f"""You are 'Welfog AI', an intelligent e-commerce assistant for Welfog.
Analyze the user message semantically. Use KNOWLEDGE BASE and any LIVE ACCOUNT/API DATA as sources of truth, and return ONLY a valid JSON object.

CONTEXT (may include KNOWLEDGE BASE excerpts and/or LIVE ACCOUNT/API DATA):
\"\"\"
{kb_context}
\"\"\"

GROUNDING RULE: Answer ONLY from the context above. Ignore any KB sentence that does not relate to the user's latest question. If context lacks the answer, say it is not in the available knowledge — do not invent.

JSON SCHEMA:
{{
  "reasoning": "Short reasoning (1-3 lines).",
  "intent": "product" | "order" | "order_history" | "wishlist" | "refund" | "payment" | "seller" | "pincode_check" | "deals" | "categories" | "category_feed" | "general" | "out_of_domain",
  "search_query": "Clean product name if product intent, else empty",
  "extracted_pincode": "Extract the 6-digit PIN code if the user provides one, else empty string",
  "needs_order_id": true/false,
  "is_welfog_related": true/false,
  "response": "Final answer: complete enough to satisfy the question. If intent=categories or deals, guide user what you are showing."
}}

RULES:
- SCOPE: You ONLY help with Welfog — products, deals, categories, orders (tracking/history steps), delivery/PIN, refunds/payments, seller topics, and facts present in the knowledge base. If the message is unrelated (weather, jokes, recipes, cricket, personal stories, other apps/companies, etc.): set intent="out_of_domain", is_welfog_related=false, and set "response" to 1-3 short sentences in the user's language: briefly acknowledge their topic in one phrase, say you are the Welfog shopping assistant and cannot help with that, invite Welfog product/order/delivery help — do NOT give facts/advice/forecast for the off-topic question. EXCEPTION: cultural / Indian greetings (Ram Ram, Radhe Radhe, Namaste, Adaab), casual wellbeing ("sab badhiya", "kaise ho"), Hinglish openers ("sun na", "hi hello") — keep is_welfog_related=true, intent="general", and reply warmly in 1-2 human sentences; on a pure greeting do NOT paste deals URLs or long "I can only help with..." disclaimers.
- Answer ONLY what the user asked when it is in-scope; keep the "response" concise (no extra sections, no unrelated marketing). Match answer length to question — one-line question → 1-3 sentences, not a full policy essay.
- SELLER LOGIN: If user reports seller login/OTP/panel error, give login troubleshooting bullets from seller knowledge — NOT "how to become seller" registration steps. After failed attempts, add customer care contact from KB.
- SERVICE CHARGES: If user asks about fees/charges on Welfog, answer only from payment knowledge (checkout display, COD/premium return fees) in 2-4 sentences — no order-tracking or greeting text.
- SHORT VIDEO / SHORTS / REELS: Rules come from terms (Short Video Content Rules), seller (Supplier Promotional Videos, ASCI), privacy (age/consent). intent=general, data_channel=kb, kb_keys terms+seller+privacy. NEVER say "no restrictions" if KB lists prohibited content. NEVER invent quality/timing/brand-image rules not in KB.
- KB-ONLY FACTS: Every policy/fee/rule sentence must come from KNOWLEDGE BASE text provided. If a detail is not in the context, say it is not in the available knowledge — do NOT guess or use generic filler ("follow guidelines", "maintain brand image").
- LIVE API DATA: If LIVE ACCOUNT/API DATA is present, use it for order/refund/payment status facts; use KB for policy/how-to context. If they conflict on status, trust LIVE data.
- NEVER write bracket placeholders like [insert phone number from KB] or [insert ...] — copy exact phone/email/dates from the knowledge text, or omit that line entirely.
- NARROW ANSWERS: If the user asks only about age rules, only about fees, or only about one policy topic, answer ONLY that slice — do not repeat the entire policy document.
- CONTACT / CUSTOMER CARE: This chat is Welfog support — do not imply the user must open a separate in-app "Help & Support" area to reach humans. If the KB excerpt gives a customer-care phone or support email, copy those values exactly (same digits and email spelling). NEVER invent or guess phone numbers, toll-free lines, or emails (e.g. do not use placeholder numbers like 1800-123-4567). If no phone/email appears in the KB context, omit contact lines entirely. Use grievance-style emails only when the user asks for complaints/grievance escalation or when the KB excerpt is clearly about the Grievance Officer — never as the default support inbox. Do not tell the user to "only use Help & Support in the app" when they asked for contact details; you may mention the app as optional, but you must still give the phone/email from the KB if present.
- Follow the API PLAYBOOK instructions if present in KB (source=welfog_api).
- Do NOT invent products, prices, categories, or deals. If intent is "product", give a SHORT line that results are being shown — do NOT tell the user to only visit the website/app instead of searching.
- KNOWLEDGE ANSWERS: Read the excerpts fully. If the user asks for a list (e.g. department names, team list), output the actual names/details from the text — do NOT reply with only a document title, tag line, or \"According to...\" meta phrase. Use bullets or short sentences when listing multiple items.
- If the requested detail is not present in KB context, clearly say it is not available right now instead of guessing.
- If the user asks about refund, return, cancel, payment, or order policy, answer directly and keep it concise. Do not add unrelated paragraphs or marketing text.
- REFUND/RETURN NARROW QUESTIONS: If they ask ONLY about timeline/duration (e.g. "kitne din me milega", "when will refund come", "kab aayega"), reply with ONLY the refund processing time from the knowledge base (e.g. 5-7 business days after approval/pickup). Do NOT repeat full return steps or app navigation unless they explicitly asked how to return or start a refund.
- MULTI-QUESTION MESSAGES: If one message has several questions (e.g. wrong colour + can I return + refund in how many days), answer EACH part in order (1, 2, 3) using only the knowledge base. Never answer with product search or "no product found".
- If they already bought an item (mangaya/order) and ask return/refund timeline, intent is refund/general policy — NOT product shopping, even if they say "realme cover" or "iphone".
- ORDER TRACKING: If they ask HOW/WHERE to track (steps, process, "kaise/kese", order id location) and did not give an order id, set needs_order_id=false and give short numbered steps (app/website → My Orders → SMS/email for Order ID). If they want live status for their order and have or will share an id, needs_order_id=true.
- Roman Hindi / Hinglish: specific item availability ("milega", "hai kya") -> intent "product" + search_query. Delivery-to-a-place / can-I-order-from-here / PIN-only messages -> intent "pincode_check" (ask for 6-digit PIN if missing). General "what is on Welfog" -> intent "general", not product.
- FOLLOW-UPS: If RECENT CONVERSATION is provided, resolve pronouns (uska/uske/iska/yeh) from the last turns; keep intent "general" when continuing an explanatory topic (e.g. what a department does) unless the user clearly switches to buying a product or a colour/size variant of a product they were browsing (then intent=product).
- ORDER-ID FOLLOW-UP: If RECENT CONVERSATION shows you already offered to check live status when the user sends their Order ID, and the latest message contains that id (or says "yeh le id", "kab aaega", etc.), keep intent "order" with needs_order_id=true — do NOT pivot to order history lists or unrelated how-to blocks.
- {language_reply_instruction(reply_lang)}
- LANGUAGE MATCH: Write the response in the SAME language/script as the customer's latest message (all Indian languages listed in routing + English)."""

        compact_ctx = _compact_conversation_context(conversation_context, _ANSWER_CONTEXT_MAX_CHARS)
        msg_for_answer = _trim_text_mid(user_msg, _ROUTING_USER_MAX_CHARS)
        user_payload = msg_for_answer
        if compact_ctx:
            user_payload = (
                "RECENT CONVERSATION:\n"
                f"{compact_ctx}\n\n"
                f"LATEST USER MESSAGE:\n{msg_for_answer}"
            )

        messages = [
            {"role": "system", "content": _trim_text_mid(system_prompt, _ANSWER_SYSTEM_MAX_CHARS)},
            {"role": "user", "content": user_payload},
        ]
        out = _llm_json_with_provider_fallback(
            providers, messages, max_tokens=240, timeout_sec=10, max_attempts=2
        )
        if out:
            log_reasoning(out.get("reasoning") or "Answer generation completed.")
        return out
    except Exception as e:
        _safe_print(f"AI Brain Error: {e}")
        return None
