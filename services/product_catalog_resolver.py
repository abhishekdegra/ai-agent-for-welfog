"""
AI-first product browse / buy / availability — any language, long prompts.

Resolves: product search API vs KB/support (NOT keyword-only routing).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.reasoning_log import chat_log, log_reasoning

_RESOLVE_CACHE = threading.local()

KIND_NONE = "none"
KIND_PRODUCT_SEARCH = "product_search"
KIND_POLICY_KB = "policy_kb"
KIND_ORDER_SUPPORT = "order_support"

_BOGUS_CATALOG_BRANDS = frozenset(
    {
        "mobile cover",
        "mobile covers",
        "phone cover",
        "iphone cover",
        "back cover",
        "mobile",
        "cover",
        "covers",
        "case",
        "cases",
        "phone",
        "iphone",
        "iphones",
        "shirt",
        "shirts",
        "jeans",
        "pajama",
        "pajami",
        "product",
        "products",
        "water bottle",
        "track pants",
        "lower",
        "lowers",
    }
)


def sanitize_catalog_brand(
    brand: str | None,
    *,
    product_name: str = "",
    explicit_from_brain: bool = False,
    user_meaning: str = "",
) -> str:
    """
    Drop bogus brands (product types mistaken as brand by LLM).
    Apple only when user/brain English explicitly mentions Apple.
    """
    b = (brand or "").strip()
    if not b:
        return ""
    bl = b.lower()
    pn = (product_name or "").strip().lower()
    um_low = (user_meaning or "").lower()
    if bl in _BOGUS_CATALOG_BRANDS:
        return ""
    if pn and bl == pn:
        return ""
    if pn and not explicit_from_brain:
        pn_tokens = set(re.findall(r"[a-z0-9]+", pn))
        b_tokens = set(re.findall(r"[a-z0-9]+", bl))
        if b_tokens and b_tokens <= pn_tokens:
            return ""
    if bl in ("apple", "iphone"):
        if pn in ("iphone", "iphones") or (
            "iphone" in pn and "cover" not in pn and "case" not in pn
        ):
            return ""
        if bl == "apple" and "apple" not in um_low:
            return ""
    if bl in ("mobile", "cover", "phone", "case", "shirt", "jeans"):
        return ""
    return b


@dataclass
class ResolvedProductSearchTurn:
    kind: str = KIND_NONE
    search_query: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    confidence: str = ""
    user_meaning: str = ""


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _is_likely_chitchat_not_product(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """Brain scope + keyword fallback — no full conversational recursion."""
    try:
        from services.chitchat_resolver import turn_is_chitchat_not_shopping

        return turn_is_chitchat_not_shopping(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=False,
        )
    except ImportError:
        pass
    return False


def _catalog_menu_turn(
    original_msg: str,
    msg_en: str = "",
    ai_route: dict | None = None,
    conversation_context: str = "",
) -> bool:
    """Deals / categories list — never product catalog search."""
    try:
        from services.catalog_menu_resolver import turn_requests_catalog_menu

        return turn_requests_catalog_menu(
            original_msg,
            msg_en,
            ai_route=ai_route,
            conversation_context=conversation_context,
            allow_llm=False,
        )
    except ImportError:
        pass
    try:
        from services.conversation_scope import turn_requests_catalog_menu

        return turn_requests_catalog_menu(
            original_msg, msg_en, ai_route=ai_route, allow_llm=False
        )
    except ImportError:
        return False


def _embedding_suggests_product_browse(comb: str) -> bool:
    """Semantic similarity — works across languages without phrase lists."""
    try:
        from services.product_browse_semantics import embedding_browse_scores

        pos, neg = embedding_browse_scores(comb)
        return pos >= 0.30 and neg < pos - 0.03
    except Exception:
        return False


def _clearly_live_account_support(
    comb: str,
    ai_route: dict | None = None,
    *,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> bool:
    """Skip product classifier when account-list AI or structural order-id/PIN applies."""
    om = (original_msg or comb or "").strip()
    me = (msg_en or "").strip()
    try:
        from services.turn_intent_coordinator import account_list_ai_blocks_product_path

        if account_list_ai_blocks_product_path(
            om,
            me,
            conversation_context,
            reply_lang,
            ai_route=ai_route,
        ):
            return True
    except ImportError:
        pass
    try:
        from utils.helpers import (
            extract_order_id,
            extract_pincode_preferred_from_message,
            _text_has_pincode_delivery_intent,
        )

        oid = extract_order_id(comb)
        if oid and isinstance(ai_route, dict):
            intent = (ai_route.get("intent") or "").strip().lower()
            olk = (ai_route.get("order_lookup_kind") or "").strip().lower()
            if intent in ("order", "refund", "payment") or olk in (
                "track",
                "details",
                "invoice",
                "refund_status",
            ):
                return True
        pin = extract_pincode_preferred_from_message(comb)
        if pin and _text_has_pincode_delivery_intent(comb) and not _embedding_suggests_product_browse(
            comb
        ):
            return True
    except ImportError:
        pass
    return False


def _should_invoke_product_classifier(
    comb: str,
    ai_route: dict | None = None,
    conversation_context: str = "",
    *,
    ctx: dict | None = None,
) -> bool:
    """
    Run product micro-classifier only when shopping is plausible — not on chitchat/off-topic.
    """
    if not comb or len(comb.strip()) < 2:
        return False
    try:
        from services.welfog_api import message_requests_category_browse

        if message_requests_category_browse(comb):
            return False
    except ImportError:
        pass
    if _catalog_menu_turn(comb, "", ai_route, conversation_context=conversation_context):
        return False
    if _is_likely_chitchat_not_product(
        comb, "", conversation_context, ai_route=ai_route
    ):
        return False
    try:
        from services.conversation_scope import turn_blocks_product_catalog

        if turn_blocks_product_catalog(
            comb, "", conversation_context, ai_route=ai_route
        ):
            return False
    except ImportError:
        pass
    try:
        from services.product_browse_semantics import _hard_exclude_browse

        if _hard_exclude_browse(comb):
            return False
    except ImportError:
        pass
    if _clearly_live_account_support(
        comb,
        ai_route,
        original_msg=comb,
        msg_en="",
        conversation_context=conversation_context,
    ):
        return False
    if product_catalog_route_is_locked(ai_route):
        return False
    try:
        import re
        from services.opensearch_products import _extract_sku_from_text, _sku_token_acceptable

        if re.search(r"\bsku\b", comb, re.I):
            sku = _extract_sku_from_text(comb)
            if sku and _sku_token_acceptable(sku, explicit_sku_mention=True):
                return False
    except ImportError:
        pass
    if isinstance(ai_route, dict):
        intent = (ai_route.get("intent") or "").strip().lower()
        channel = (ai_route.get("data_channel") or "").strip().lower()
        if intent == "product" and channel == "catalog" and ai_route.get("run_catalog_search"):
            return False
    if _embedding_suggests_product_browse(comb):
        return True
    try:
        from utils.helpers import (
            _text_has_product_shopping_intent_core,
            _turn_is_catalog_product_request,
            _message_has_generic_shopping_item_signal,
        )

        if (
            _turn_is_catalog_product_request(comb)
            or _text_has_product_shopping_intent_core(comb)
            or _message_has_generic_shopping_item_signal(comb)
        ):
            return True
    except ImportError:
        pass
    if isinstance(ai_route, dict):
        try:
            from services.product_browse_semantics import llm_meaning_is_product_browse

            if llm_meaning_is_product_browse(ai_route):
                return True
        except ImportError:
            pass
        intent = (ai_route.get("intent") or "").strip().lower()
        channel = (ai_route.get("data_channel") or "").strip().lower()
        if intent == "product" and channel == "catalog" and ai_route.get("run_catalog_search"):
            return True
    return False


def product_turn_should_beat_delivery(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    """
    Product catalog beats stale pincode/delivery thread — AI micro-classifier first,
    structural product signals as fast path. No customer keyword routing lists.
    """
    try:
        from utils.helpers import turn_is_obvious_product_shopping_turn

        if turn_is_obvious_product_shopping_turn(
            original_msg, msg_en, conversation_context
        ):
            return True
    except ImportError:
        pass
    if not allow_llm:
        return False
    comb = _combined(original_msg, msg_en)
    if not _should_invoke_product_classifier(comb, None, conversation_context):
        return False
    classified = ai_classify_product_search_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=None,
    )
    if not classified:
        return False
    kind = (classified.get("turn_kind") or KIND_NONE).strip().lower()
    conf = float(classified.get("confidence") or 0.0)
    return kind == KIND_PRODUCT_SEARCH and conf >= 0.42


def ai_classify_product_search_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    force_llm: bool = False,
) -> Optional[dict[str, Any]]:
    """Micro-classifier: product_search vs policy_kb vs order_support vs none."""
    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            should_skip_micro_classifier_llm,
        )

        if not force_llm and should_skip_micro_classifier_llm():
            route = ai_route if isinstance(ai_route, dict) else get_cached_brain_route()
            allow_ood_rescue = False
            if isinstance(route, dict):
                intent_b = (route.get("intent") or "").strip().lower()
                scope_b = (route.get("conversation_scope") or "").strip().lower()
                if intent_b in ("out_of_domain",) or scope_b == "out_of_domain":
                    allow_ood_rescue = True
            if not allow_ood_rescue:
                locked = _locked_product_turn_from_route(route)
                if locked and locked.kind == KIND_PRODUCT_SEARCH:
                    log_reasoning(
                        f"Product-catalog LLM skipped — brain locked sq={locked.search_query!r}."
                    )
                    return {
                        "turn_kind": KIND_PRODUCT_SEARCH,
                        "search_query": locked.search_query,
                        "entities": locked.entities,
                        "confidence": 1.0,
                        "user_meaning": locked.user_meaning or (route or {}).get("user_meaning") or "",
                    }
                log_reasoning(
                    "Product-catalog LLM skipped — universal brain already classified turn."
                )
                return None
    except ImportError:
        pass

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    comb = _combined(original_msg, msg_en)
    if not comb:
        return None

    try:
        from services.turn_intent_coordinator import account_list_ai_blocks_product_path

        if account_list_ai_blocks_product_path(
            original_msg,
            msg_en,
            conversation_context,
            reply_lang,
            ai_route=ai_route,
        ):
            log_reasoning(
                "Product-catalog LLM skipped — account-list AI classified live API turn."
            )
            return {
                "turn_kind": KIND_ORDER_SUPPORT,
                "confidence": 0.9,
                "user_meaning": "",
                "search_query": "",
                "entities": {},
            }
    except ImportError:
        pass

    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 1400)
    user_line = _trim_text_mid(comb, 600)
    route_hint = ""
    if isinstance(ai_route, dict):
        route_hint = (
            f"intent={ai_route.get('intent') or '-'} "
            f"meaning={(ai_route.get('user_meaning') or '')[:100]}"
        )

    system_prompt = f"""You classify Welfog SHOPPING vs non-shopping for the LATEST user message.

The user may write in ANY language, script, dialect, slang, typos, Hinglish, or long casual prompts.
Do NOT rely on specific English/Hindi keywords — understand MEANING.

Return ONLY JSON:
{{
  "user_meaning": "one English sentence",
  "turn_kind": "none" | "product_search" | "policy_kb" | "order_support",
  "search_query": "2-6 English product keywords for catalog API (empty if not product_search)",
  "entities": {{
    "product_name": "2-6 English catalog keywords ONLY (mobile cover, rice, lip gloss) — empty when user only filters by price/rating with no product type",
    "category": "",
    "brand": "ANY brand/company user meant (worldwide, any spelling) — empty if not mentioned",
    "model": "",
    "color": "catalog colour or empty",
    "size": "",
    "price_min": null,
    "price_max": null,
    "rating_min": null,
    "rating_max": null,
    "sku": "warehouse SKU only if user pasted/gave a code",
    "product_id": "numeric Welfog catalog product id or null"
  }},
  "confidence": 0.0 to 1.0
}}

Rules:
- product_search: user wants to find/buy/see/check if Welfog sells ANY product or item
  (groceries, electronics, fashion, beauty, home, food ingredients, accessories, etc.)
- policy_kb: refund/return/shipping/payment policy how-to — NOT shopping
- order_support: track MY order, refund on order id, invoice, pincode/city delivery check
- none: greeting, casual chat, feelings, jokes, personal/off-topic talk (NOT shopping)
- none: thanks/praise/love you/closing — "thank you", "love you bhai", "tu achha h"
- none: social media link/id/handle requests — Instagram, Facebook, YouTube, "welfog ki instagram id", "kisi ka facebook link"
- none: user wants to chat / vent / ask non-shopping favours — "free ho?", "baat karni hai", "koi kaam kr dega?"
- policy_kb: Welfog refund/return/shipping AND Welfog official social/contact from KB — NOT catalog product_search
- search_query: same as entities.product_name (English catalog keywords). EMPTY for filter-only browse (rating/price only, no product type).
- entities.brand: infer ANY brand from meaning — NOT from a fixed list. redmi→Redmi, samung→Samsung, himalaya→Himalaya.
- entities.product_name + brand + color + price + rating + sku + product_id → fill ALL that user meant.
- rating_min: "more than 2 stars", "rating above 2", "4+ rating" → number. rating_max for "under 3 stars".
- price_min/price_max: rupees budget (under 200 → price_max=200; "200 rs ki range" → price_max=200). NEVER put star numbers in price fields.
- Filter-only: "products with rating more than 2" → search_query="", product_name="", rating_min=2.
- Brand + product: "redmi mobile cover" → brand=Redmi, product_name=mobile cover, search_query=mobile cover.
- SKU lookup: "Xiaomi-SKU is sku ka item dikha" → sku="Xiaomi-SKU", search_query="", product_name="".
- Product id: "product id de rha hu 2815318 iska product bta" → product_id=2815318, search_query="".
- Brand typos: "jioo ke covers" → brand=Jio, product_name=mobile cover.
- Do NOT put rating/price/filter words in search_query or product_name.
- Do NOT answer availability from memory — only classify.

Examples (many languages/styles — same intent = product_search):
- "mujhe lal mirch chahiye" → search_query="red chilli"
- "flour mil jayga kya welfog pe" → search_query="flour"
- "neenga inga mobile cover vangalaama" (Tamil) → search_query="mobile cover"
- "ee shop lo black fan unda" (Telugu) → search_query="fan", color="black"
- "আমাকে লিপগ্লস চাই" (Bengali) → search_query="lip gloss"
- "mala silver ring pahije" (Marathi) → search_query="silver ring"
- "koi sasta laptop dikha do under 20k" → search_query="laptop", price_max=20000
- "i need itel charger" → search_query="itel charger"
- "refund kitne din me aata hai" → policy_kb (NOT product_search)
- "welfog ki instagram id dena" → policy_kb or none (NOT product_search — no catalog search)
- "toing sonu ki facebook link chahiye" → none (NOT product_search)
- "ok bro thank you love you" → none
- "bhai tu bahut achha h love you" → none
- "categories list", "catagories bta", "aaj ki deals" → none (catalog menu API — NOT product_search)

{language_reply_instruction(rl)}
JSON only."""

    user_payload = f"ROUTER: {route_hint}\nLATEST:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CHAT:\n{compact_ctx}\n\n{user_payload}"

    data = _llm_json_with_provider_fallback(
        providers,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_tokens=320,
        timeout_sec=12,
        max_attempts=1,
    )
    if not data:
        return None
    kind = (data.get("turn_kind") or KIND_NONE).strip().lower()
    if kind not in (KIND_PRODUCT_SEARCH, KIND_POLICY_KB, KIND_ORDER_SUPPORT, KIND_NONE):
        kind = KIND_NONE
    sq = (data.get("search_query") or "").strip()
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    conf = float(data.get("confidence") or 0.0)
    um = (data.get("user_meaning") or "").strip()
    log_reasoning(
        f"Product-catalog LLM: kind={kind} sq={sq!r} conf={conf:.2f} — {um[:80] or '-'}"
    )
    return {
        "turn_kind": kind,
        "search_query": sq,
        "entities": entities,
        "confidence": conf,
        "user_meaning": um,
    }


def _locked_product_turn_from_route(
    ai_route: dict | None,
    *,
    search_query: str = "",
) -> ResolvedProductSearchTurn | None:
    """Reuse main router product JSON — no duplicate product micro-classifier LLM."""
    if not isinstance(ai_route, dict):
        return None
    intent = (ai_route.get("intent") or "").strip().lower()
    channel = (ai_route.get("data_channel") or "").strip().lower()
    sq = (search_query or ai_route.get("search_query") or "").strip()
    entities = dict(ai_route.get("_product_entities") or {})
    locked = product_catalog_route_is_locked(ai_route)
    brain_product = (
        intent == "product"
        and (
            channel == "catalog"
            or bool(ai_route.get("run_catalog_search"))
            or bool(ai_route.get("_universal_brain_route"))
            or bool(sq)
        )
        and (
            bool(sq)
            or bool(entities)
            or bool(ai_route.get("_product_catalog_locked"))
            or bool(ai_route.get("run_catalog_search"))
        )
    )
    if not locked and not brain_product:
        return None
    return ResolvedProductSearchTurn(
        kind=KIND_PRODUCT_SEARCH,
        search_query=sq,
        entities=entities,
        source="ai_route_locked" if locked else "ai_brain_route",
        confidence="high",
        user_meaning=str(ai_route.get("user_meaning") or "")[:200],
    )


def understanding_from_locked_product_route(
    ai_route: dict | None,
    search_query: str = "",
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> dict[str, Any] | None:
    """Map locked product route → catalog understanding from AI entities only."""
    locked = _locked_product_turn_from_route(ai_route, search_query=search_query)
    if not locked or locked.kind != KIND_PRODUCT_SEARCH:
        return None

    entities = dict(locked.entities or {})
    if isinstance(ai_route, dict):
        route_ent = ai_route.get("_product_entities")
        if isinstance(route_ent, dict):
            for key, val in route_ent.items():
                if val not in (None, ""):
                    entities[key] = val

    sq = (search_query or locked.search_query or "").strip()
    u = entities_to_understanding(
        entities,
        search_query=sq,
        original_msg=original_msg or msg_en,
    )
    if not u:
        return None
    return u


def _turn_blocks_ai_product_first(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ctx: dict | None = None,
) -> bool:
    """Skip product AI-first only for clear non-catalog turns (no brand/product keyword lists)."""
    comb = _combined(original_msg, msg_en)
    if not comb:
        return True
    if ctx and ctx.get("awaiting") in ("order_id", "pincode", "category_select"):
        if (ctx.get("awaiting") or "").strip().lower() == "pincode":
            try:
                from utils.helpers import (
                    clear_order_session_for_new_lookup,
                    turn_is_obvious_product_shopping_turn,
                )

                if turn_is_obvious_product_shopping_turn(
                    original_msg, msg_en, conversation_context
                ):
                    clear_order_session_for_new_lookup(ctx)
                    ctx["awaiting"] = None
                    ctx["last"] = None
                    data = ctx.get("data")
                    if isinstance(data, dict):
                        data.pop("topic_mode", None)
                    return False
                from services.product_catalog_resolver import (
                    product_turn_should_beat_delivery,
                )

                if product_turn_should_beat_delivery(
                    original_msg,
                    msg_en,
                    conversation_context,
                    allow_llm=True,
                ):
                    clear_order_session_for_new_lookup(ctx)
                    ctx["awaiting"] = None
                    ctx["last"] = None
                    data = ctx.get("data")
                    if isinstance(data, dict):
                        data.pop("topic_mode", None)
                    return False
            except ImportError:
                pass
        if (ctx.get("awaiting") or "").strip().lower() != "pincode":
            return True
    if _catalog_menu_turn(
        original_msg, msg_en, None, conversation_context=conversation_context
    ):
        return True
    try:
        from utils.helpers import (
            _text_asks_order_history,
            _text_asks_wishlist,
            _text_has_pincode_delivery_intent,
            _text_is_order_tracking_intent,
            extract_order_id,
            message_asks_my_welfog_purchases,
            message_is_past_purchase_list_request,
            message_is_wishlist_like_request,
            turn_is_catalog_product_lookup,
            _text_is_product_id_lookup_context,
        )

        if (
            message_is_past_purchase_list_request(comb)
            or message_is_wishlist_like_request(comb)
            or _text_asks_order_history(comb)
            or message_asks_my_welfog_purchases(comb)
            or _text_asks_wishlist(comb)
        ):
            return True
        if _text_has_pincode_delivery_intent(comb, conversation_context):
            return True
        oid = extract_order_id(comb, conversation_context)
        if oid and _text_is_order_tracking_intent(comb) and not (
            _text_is_product_id_lookup_context(original_msg)
            or turn_is_catalog_product_lookup(original_msg)
        ):
            return True
    except ImportError:
        pass
    return False


def try_product_ai_first_catalog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "en",
    *,
    ctx: dict | None = None,
    user_id: str = "",
    guard_fast: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """
    Industry path: ONE product AI call extracts intent + filters → OpenSearch.
    Skips brain route, delivery AI, pincode, catalog-menu, KB retrieval.
    guard_fast: main-thread guard — higher confidence bar + 12s catalog cap.
    """
    comb = _combined(original_msg, msg_en)
    if not comb or _turn_blocks_ai_product_first(
        original_msg, msg_en, conversation_context, ctx
    ):
        return None

    try:
        from services.turn_intent_coordinator import (
            _KB_CACHE_UNSET,
            get_kb_turn_ai_classification,
            kb_turn_blocks_product_catalog,
            peek_kb_turn_classification,
        )

        peeked = peek_kb_turn_classification(
            original_msg, msg_en, conversation_context, reply_lang
        )
        if peeked is _KB_CACHE_UNSET:
            peeked = get_kb_turn_ai_classification(
                original_msg,
                msg_en,
                conversation_context,
                reply_lang,
                preflight=True,
            )
        if kb_turn_blocks_product_catalog(peeked):
            log_reasoning(
                "Product AI skipped — KB classifier locked informational turn."
            )
            return None
    except ImportError:
        pass

    if _clearly_live_account_support(
        comb,
        None,
        original_msg=original_msg,
        msg_en=msg_en,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
    ):
        return None

    try:
        import re
        from services.opensearch_products import (
            _extract_sku_from_text,
            _sku_token_acceptable,
        )

        if re.search(r"\bsku\b", comb, re.I):
            sku_early = _extract_sku_from_text(original_msg) or _extract_sku_from_text(
                msg_en
            )
            if sku_early and _sku_token_acceptable(sku_early, explicit_sku_mention=True):
                from services.brain_direct_dispatch import try_explicit_sku_catalog_reply

                body = try_explicit_sku_catalog_reply(
                    original_msg,
                    msg_en,
                    user_id=user_id,
                    lang=reply_lang or "en",
                    ctx=ctx or {},
                    reset_context_fn=lambda _c: None,
                )
                if body:
                    return body, {
                        "intent": "product",
                        "data_channel": "catalog",
                        "route_handler": "sku_structural_fast",
                        "_product_catalog_locked": True,
                        "_product_entities": {"sku": sku_early},
                    }
    except ImportError:
        pass

    if guard_fast:
        pass  # guard fallback — always run AI product classifier (any language)
    elif not _should_invoke_product_classifier(comb, None, conversation_context, ctx=ctx):
        return None

    classified = ai_classify_product_search_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=None,
        reply_lang=reply_lang,
    )
    if not classified:
        return None
    kind = (classified.get("turn_kind") or KIND_NONE).strip().lower()
    conf = float(classified.get("confidence") or 0.0)
    min_conf = 0.55 if guard_fast else 0.42
    if kind != KIND_PRODUCT_SEARCH or conf < min_conf:
        return None

    entities = dict(classified.get("entities") or {})
    sq = (classified.get("search_query") or "").strip()
    if sq and not (entities.get("product_name") or "").strip():
        entities["product_name"] = sq

    ai_route: dict[str, Any] = {
        "intent": "product",
        "data_channel": "catalog",
        "run_catalog_search": True,
        "_product_catalog_locked": True,
        "search_query": sq,
        "_product_entities": entities,
        "user_meaning": classified.get("user_meaning") or "",
        "needs_order_id": False,
        "numeric_context": "none",
        "meta_kind": "none",
        "is_welfog_related": True,
    }

    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|True"
    _RESOLVE_CACHE.key = cache_key
    _RESOLVE_CACHE.result = ResolvedProductSearchTurn(
        kind=KIND_PRODUCT_SEARCH,
        search_query=sq,
        entities=entities,
        source="product_ai_first",
        confidence="high" if conf >= 0.7 else "medium",
        user_meaning=str(classified.get("user_meaning") or ""),
    )

    log_product_catalog_routing(
        detected_intent="product_search",
        product_entities=entities,
        selected_route="product_ai_flow",
        filters=entities,
        source="product_ai_first",
    )
    log_reasoning(
        "Product AI-first gate — AI extracted filters; skip brain/delivery/pincode/KB routing."
    )

    try:
        from services.chat_flow_telemetry import mark_routing_complete

        mark_routing_complete()
    except ImportError:
        pass

    from services.product_search_flow import run_product_search_ai_flow

    ps = run_product_search_ai_flow(
        original_msg,
        msg_en,
        user_id,
        conversation_context=conversation_context,
        reply_lang=reply_lang,
        ctx=ctx,
        search_query=sq,
        ai_route=ai_route,
    )
    if ps and ps.handled and ps.reply_html:
        return ps.reply_html, ai_route
    return None


def resolve_product_search_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> ResolvedProductSearchTurn:
    """AI-first product search vs KB/support."""
    try:
        from services.knowledge_query_pipeline import kb_vector_fast_lane_active

        if kb_vector_fast_lane_active():
            return ResolvedProductSearchTurn(kind=KIND_NONE)
    except Exception:
        pass
    comb = _combined(original_msg, msg_en)
    if not comb:
        return ResolvedProductSearchTurn(kind=KIND_NONE)

    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            should_defer_micro_classifiers_to_brain,
        )
        from services.ai_route_semantics import brain_route_indicates_product_catalog

        brain = ai_route if isinstance(ai_route, dict) else get_cached_brain_route()
        if isinstance(brain, dict):
            ch = (brain.get("data_channel") or "").strip().lower()
            intent = (brain.get("intent") or "").strip().lower()
            if ch in ("kb", "live_api") and intent not in (
                "product",
                "deals",
                "categories",
                "category_feed",
            ):
                if not brain_route_indicates_product_catalog(brain):
                    return ResolvedProductSearchTurn(kind=KIND_NONE)
        if should_defer_micro_classifiers_to_brain():
            if isinstance(brain, dict) and brain_route_indicates_product_catalog(brain):
                pass
            else:
                structural_product = False
                try:
                    from utils.helpers import (
                        _text_has_product_shopping_intent_core,
                        _turn_is_catalog_product_request,
                    )

                    structural_product = bool(
                        _turn_is_catalog_product_request(comb)
                        or _text_has_product_shopping_intent_core(comb)
                    )
                except ImportError:
                    structural_product = False
                if not structural_product and not _embedding_suggests_product_browse(comb):
                    return ResolvedProductSearchTurn(kind=KIND_NONE)
    except ImportError:
        pass

    locked_turn = _locked_product_turn_from_route(ai_route)
    if locked_turn:
        log_reasoning(
            f"Product resolver fast path ({locked_turn.source}): sq={locked_turn.search_query!r}"
        )
        cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
        _RESOLVE_CACHE.key = cache_key
        _RESOLVE_CACHE.result = locked_turn
        return locked_turn

    if isinstance(ai_route, dict) and ai_route.get("_universal_brain_route"):
        try:
            from services.ai_route_semantics import (
                brain_route_indicates_product_catalog,
                _catalog_search_query_from_brain_route,
            )

            if brain_route_indicates_product_catalog(ai_route):
                sq_brain = _catalog_search_query_from_brain_route(
                    ai_route,
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
                entities_brain = dict(ai_route.get("_product_entities") or {})
                locked_brain = ResolvedProductSearchTurn(
                    kind=KIND_PRODUCT_SEARCH,
                    search_query=sq_brain,
                    entities=entities_brain,
                    source="ai_brain_route",
                    confidence="high",
                    user_meaning=str(ai_route.get("user_meaning") or "")[:200],
                )
                log_reasoning(
                    f"Product resolver — brain JSON only sq={sq_brain!r} (skip micro-classifier LLM)."
                )
                cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = locked_brain
                return locked_brain
        except ImportError:
            pass

    try:
        from services.chat_flow_telemetry import (
            get_cached_brain_route,
            should_skip_micro_classifier_llm,
        )

        if allow_llm and should_skip_micro_classifier_llm():
            brain_route = ai_route if isinstance(ai_route, dict) else get_cached_brain_route()
            brain_ood_misroute = False
            if isinstance(brain_route, dict):
                intent_b = (brain_route.get("intent") or "").strip().lower()
                scope_b = (brain_route.get("conversation_scope") or "").strip().lower()
                if intent_b in ("out_of_domain",) or scope_b == "out_of_domain":
                    # Brain often mislabels catalog asks (brand names, Hinglish) as OOD — allow rescue LLM.
                    brain_ood_misroute = True
                elif intent_b not in ("product",) and scope_b != "out_of_domain":
                    brain_ood_misroute = _embedding_suggests_product_browse(comb)
                    if not brain_ood_misroute:
                        try:
                            from utils.helpers import (
                                _text_has_product_shopping_intent_core,
                                _turn_is_catalog_product_request,
                            )

                            brain_ood_misroute = bool(
                                _turn_is_catalog_product_request(comb)
                                or _text_has_product_shopping_intent_core(comb)
                            )
                        except ImportError:
                            pass
            if not brain_ood_misroute:
                locked_brain = _locked_product_turn_from_route(brain_route)
                if locked_brain and locked_brain.kind == KIND_PRODUCT_SEARCH:
                    log_reasoning(
                        f"Product-catalog: brain locked sq={locked_brain.search_query!r} "
                        "(skip micro-classifier LLM)."
                    )
                    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
                    _RESOLVE_CACHE.key = cache_key
                    _RESOLVE_CACHE.result = locked_brain
                    return locked_brain
                log_reasoning(
                    "Product-catalog: defer — universal brain route owns classification."
                )
                none = ResolvedProductSearchTurn(kind=KIND_NONE)
                cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = none
                return none
    except ImportError:
        pass

    if _catalog_menu_turn(
        original_msg, msg_en, ai_route, conversation_context=conversation_context
    ):
        return ResolvedProductSearchTurn(kind=KIND_NONE)
    if _is_likely_chitchat_not_product(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    ):
        return ResolvedProductSearchTurn(kind=KIND_NONE)

    try:
        from services.product_query_understanding import resolve_catalog_search_terms_for_message

        run_ai_terms = _embedding_suggests_product_browse(comb)
        if isinstance(ai_route, dict):
            intent_b = (ai_route.get("intent") or "").strip().lower()
            scope_b = (ai_route.get("conversation_scope") or "").strip().lower()
            ch_b = (ai_route.get("data_channel") or "").strip().lower()
            if intent_b in ("out_of_domain",) or scope_b == "out_of_domain":
                run_ai_terms = True
            if ch_b == "catalog" or ai_route.get("_product_catalog_locked"):
                run_ai_terms = True
            try:
                from services.ai_route_semantics import brain_route_indicates_product_catalog

                if brain_route_indicates_product_catalog(ai_route):
                    run_ai_terms = True
            except ImportError:
                pass
        if run_ai_terms:
            ood_misroute = bool(
                isinstance(ai_route, dict)
                and (
                    (ai_route.get("intent") or "").strip().lower() in ("out_of_domain",)
                    or (ai_route.get("conversation_scope") or "").strip().lower()
                    == "out_of_domain"
                )
            )
            classified = ai_classify_product_search_turn(
                original_msg,
                msg_en,
                conversation_context,
                ai_route=ai_route,
                force_llm=ood_misroute,
            )
            if classified:
                kind_c = (classified.get("turn_kind") or "").strip().lower()
                conf_c = float(classified.get("confidence") or 0.0)
                entities_c = (
                    classified.get("entities")
                    if isinstance(classified.get("entities"), dict)
                    else {}
                )
                sq_ai = (
                    (classified.get("search_query") or "").strip()
                    or str(entities_c.get("product_name") or "").strip()
                )
                if (
                    kind_c == KIND_PRODUCT_SEARCH
                    and conf_c >= 0.62
                    and sq_ai
                    and len(sq_ai.strip()) >= 2
                ):
                    plausible = True
                    try:
                        from services.product_query_understanding import shopping_extract_plausible

                        plausible = shopping_extract_plausible(
                            original_msg, msg_en, sq_ai.strip()
                        )
                    except ImportError:
                        pass
                    if plausible:
                        cache_key = (
                            f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
                        )
                        out = ResolvedProductSearchTurn(
                            kind=KIND_PRODUCT_SEARCH,
                            search_query=sq_ai.strip()[:160],
                            entities=entities_c or {"product_name": sq_ai.strip()[:160]},
                            source="ai_product_classifier",
                            confidence="high",
                        )
                        _RESOLVE_CACHE.key = cache_key
                        _RESOLVE_CACHE.result = out
                        log_reasoning(
                            f"Product resolver — classifier sq={sq_ai.strip()!r} conf={conf_c:.2f}."
                        )
                        return out
    except ImportError:
        pass

    try:
        import re
        from services.opensearch_products import _extract_sku_from_text, _sku_token_acceptable

        if re.search(r"\bsku\b", comb, re.I):
            sku = _extract_sku_from_text(original_msg) or _extract_sku_from_text(msg_en)
            if sku and _sku_token_acceptable(sku, explicit_sku_mention=True):
                out = ResolvedProductSearchTurn(
                    kind=KIND_PRODUCT_SEARCH,
                    search_query="",
                    entities={"sku": sku},
                    source="explicit_sku_extract",
                    confidence="high",
                    user_meaning=f"Product for SKU {sku}",
                )
                cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = out
                log_reasoning(
                    f"Product resolver explicit SKU ({sku!r}) — skip micro-classifier."
                )
                return out
    except ImportError:
        pass

    cache_key = f"{hash(comb)}|{hash((conversation_context or '')[-300:])}|{allow_llm}"
    if getattr(_RESOLVE_CACHE, "key", None) == cache_key:
        cached = getattr(_RESOLVE_CACHE, "result", None)
        if isinstance(cached, ResolvedProductSearchTurn):
            return cached

    none = ResolvedProductSearchTurn(kind=KIND_NONE)

    invoke_llm = allow_llm and _should_invoke_product_classifier(
        comb, ai_route, conversation_context
    )
    if allow_llm and isinstance(ai_route, dict):
        intent_b = (ai_route.get("intent") or "").strip().lower()
        scope_b = (ai_route.get("conversation_scope") or "").strip().lower()
        if intent_b in ("out_of_domain",) or scope_b == "out_of_domain":
            invoke_llm = True
    if not invoke_llm and not _embedding_suggests_product_browse(comb):
        if isinstance(ai_route, dict):
            intent = (ai_route.get("intent") or "").strip().lower()
            channel = (ai_route.get("data_channel") or "").strip().lower()
            if not (intent == "product" and channel == "catalog"):
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = none
                return none
        else:
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = none
            return none

    if invoke_llm:
        classified = ai_classify_product_search_turn(
            original_msg, msg_en, conversation_context, ai_route=ai_route, reply_lang=reply_lang
        )
        if classified:
            ck = (classified.get("turn_kind") or KIND_NONE).strip().lower()
            conf = float(classified.get("confidence") or 0.0)
            if ck == KIND_PRODUCT_SEARCH and conf >= 0.48:
                out = ResolvedProductSearchTurn(
                    kind=KIND_PRODUCT_SEARCH,
                    search_query=(classified.get("search_query") or "").strip(),
                    entities=dict(classified.get("entities") or {}),
                    source="product_catalog_llm",
                    confidence="high" if conf >= 0.7 else "medium",
                    user_meaning=str(classified.get("user_meaning") or ""),
                )
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = out
                return out
            if ck in (KIND_POLICY_KB, KIND_ORDER_SUPPORT) and conf >= 0.5:
                out = ResolvedProductSearchTurn(
                    kind=ck,
                    source="product_catalog_llm",
                    confidence="high" if conf >= 0.7 else "medium",
                    user_meaning=str(classified.get("user_meaning") or ""),
                )
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = out
                return out

    if isinstance(ai_route, dict):
        try:
            from services.product_browse_semantics import llm_meaning_is_product_browse

            if llm_meaning_is_product_browse(ai_route):
                sq = (ai_route.get("search_query") or "").strip()
                out = ResolvedProductSearchTurn(
                    kind=KIND_PRODUCT_SEARCH,
                    search_query=sq,
                    source="ai_route_meaning",
                    confidence="medium",
                )
                _RESOLVE_CACHE.key = cache_key
                _RESOLVE_CACHE.result = out
                return out
        except ImportError:
            pass
        intent = (ai_route.get("intent") or "").strip().lower()
        channel = (ai_route.get("data_channel") or "").strip().lower()
        if intent == "product" and channel == "catalog" and ai_route.get("run_catalog_search"):
            out = ResolvedProductSearchTurn(
                kind=KIND_PRODUCT_SEARCH,
                search_query=(ai_route.get("search_query") or "").strip(),
                source="ai_route_field",
                confidence="high",
            )
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = out
            return out

    if getattr(_RESOLVE_CACHE, "in_resolve", False):
        _RESOLVE_CACHE.key = cache_key
        _RESOLVE_CACHE.result = none
        return none

    try:
        from services.product_browse_semantics import ai_route_is_product_browse_turn

        _RESOLVE_CACHE.in_resolve = True
        if ai_route_is_product_browse_turn(
            ai_route, original_msg, msg_en, _from_resolver=True
        ):
            from services.product_browse_semantics import extract_browse_search_terms

            sq = extract_browse_search_terms(comb, ai_route)
            out = ResolvedProductSearchTurn(
                kind=KIND_PRODUCT_SEARCH,
                search_query=sq,
                source="browse_semantics",
                confidence="medium",
            )
            _RESOLVE_CACHE.key = cache_key
            _RESOLVE_CACHE.result = out
            return out
    except ImportError:
        pass
    finally:
        _RESOLVE_CACHE.in_resolve = False

    _RESOLVE_CACHE.key = cache_key
    _RESOLVE_CACHE.result = none
    return none


def turn_requests_product_catalog(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    resolved = resolve_product_search_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    return resolved.kind == KIND_PRODUCT_SEARCH


def product_catalog_route_is_locked(route: dict | None) -> bool:
    """True when this turn is locked to catalog product search — KB must not override."""
    if not isinstance(route, dict):
        return False
    if route.get("_product_catalog_locked"):
        return True
    intent = (route.get("intent") or "").strip().lower()
    channel = (route.get("data_channel") or "").strip().lower()
    return bool(
        intent == "product"
        and channel == "catalog"
        and route.get("run_catalog_search")
    )


def apply_product_catalog_to_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    reply_lang: str = "",
) -> dict:
    """Force product catalog route when AI resolver says product_search."""
    out = dict(route or {})
    try:
        from services.chat_flow_telemetry import get_cached_brain_route

        if not out.get("intent") and isinstance(get_cached_brain_route(), dict):
            cached = get_cached_brain_route()
            if cached.get("intent"):
                out = {**cached, **out}
    except ImportError:
        pass
    if not out.get("_universal_brain_route"):
        try:
            from services.ai_route_semantics import brain_route_indicates_product_catalog

            if brain_route_indicates_product_catalog(out):
                out["_universal_brain_route"] = True
        except ImportError:
            pass
    if out.get("_universal_brain_route"):
        try:
            from services.ai_route_semantics import (
                brain_route_indicates_product_catalog,
                reconcile_product_catalog_from_brain_meaning,
            )

            if brain_route_indicates_product_catalog(out):
                return reconcile_product_catalog_from_brain_meaning(
                    out,
                    original_msg=original_msg,
                    msg_en=msg_en,
                )
        except ImportError:
            pass
    if product_catalog_route_is_locked(out):
        if not out.get("_product_catalog_locked"):
            out["_product_catalog_locked"] = True
        return out
    intent_pre = (out.get("intent") or "").strip().lower()
    channel_pre = (out.get("data_channel") or "").strip().lower()
    sq_pre = (out.get("search_query") or "").strip()
    if out.get("_universal_brain_route") and intent_pre == "product" and channel_pre == "catalog":
        comb_pc = _combined(original_msg, msg_en)
        if not sq_pre:
            sq_pre = (out.get("user_meaning") or "").strip()
        if not sq_pre and comb_pc:
            try:
                from utils.helpers import extract_product_search_query

                sq_pre = extract_product_search_query(
                    original_msg, msg_en, out.get("search_query"), ai_route=out
                )
            except ImportError:
                sq_pre = ""
            if not sq_pre:
                sq_pre = comb_pc
        if sq_pre:
            out["search_query"] = sq_pre
        out["run_catalog_search"] = True
        out["_product_catalog_locked"] = True
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["meta_kind"] = "none"
        log_reasoning(
            f"Product-catalog route (universal brain): sq={sq_pre!r} — skip micro-classifier."
        )
        return out
    if (
        intent_pre == "product"
        and channel_pre == "catalog"
        and out.get("run_catalog_search")
        and sq_pre
    ):
        out["_product_catalog_locked"] = True
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["meta_kind"] = "none"
        log_reasoning(f"Product-catalog route (brain locked): sq={sq_pre!r}")
        return out
    if _catalog_menu_turn(
        original_msg, msg_en, out, conversation_context=conversation_context
    ):
        out.pop("_product_catalog_locked", None)
        out.pop("search_query", None)
        out.pop("run_catalog_search", None)
        return out
    try:
        from services.conversation_scope import turn_blocks_product_catalog

        if turn_blocks_product_catalog(
            original_msg, msg_en, conversation_context, ai_route=out
        ):
            out.pop("_product_catalog_locked", None)
            return out
    except ImportError:
        pass
    resolved_fast = resolve_product_search_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=out,
        reply_lang=reply_lang,
        allow_llm=False,
    )
    if resolved_fast.kind == KIND_PRODUCT_SEARCH:
        sq = (
            resolved_fast.search_query
            or (out.get("search_query") or "").strip()
        )
        if not sq:
            try:
                from utils.helpers import extract_product_search_query

                sq = extract_product_search_query(
                    original_msg, msg_en, out.get("search_query"), ai_route=out
                )
            except ImportError:
                sq = ""
        out["intent"] = "product"
        out["data_channel"] = "catalog"
        out["run_catalog_search"] = True
        out["is_welfog_related"] = True
        out["needs_order_id"] = False
        out["numeric_context"] = "none"
        out["meta_kind"] = "none"
        out["search_query"] = sq
        out["_product_catalog_locked"] = True
        if resolved_fast.entities:
            out["_product_entities"] = resolved_fast.entities
        log_reasoning(
            f"Product-catalog route ({resolved_fast.source} fast): sq={sq!r}"
        )
        return out
    if not _should_invoke_product_classifier(
        _combined(original_msg, msg_en), out, conversation_context
    ):
        return out
    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if should_skip_micro_classifier_llm():
            log_reasoning(
                "Product-catalog route apply: defer micro-classifier — brain owns turn."
            )
            return out
    except ImportError:
        pass
    resolved = resolve_product_search_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=out,
        reply_lang=reply_lang,
        allow_llm=True,
    )
    if resolved.kind != KIND_PRODUCT_SEARCH:
        return out
    if resolved.user_meaning:
        out["user_meaning"] = resolved.user_meaning
    sq = (
        resolved.search_query
        or (out.get("search_query") or "").strip()
    )
    if not sq:
        try:
            from services.product_browse_semantics import extract_browse_search_terms

            sq = extract_browse_search_terms(_combined(original_msg, msg_en), out)
        except ImportError:
            sq = ""
    log_reasoning(
        f"Product-catalog route ({resolved.source}): intent=product sq={sq!r}"
    )
    out["intent"] = "product"
    out["data_channel"] = "catalog"
    out["run_catalog_search"] = True
    out["is_welfog_related"] = True
    out["needs_order_id"] = False
    out["numeric_context"] = "none"
    out["meta_kind"] = "none"
    out["search_query"] = sq
    out.pop("route_handler", None)
    out["_product_catalog_locked"] = True
    if resolved.entities:
        out["_product_entities"] = resolved.entities
    return out


def product_catalog_route_decision(
    route_data: dict,
    original_msg: str,
    msg_en: str,
    conv_for_llm: str = "",
    reasoning: str = "",
):
    """
    Build AnswerRouteDecision for locked product catalog, or None if not a product turn.
    """
    from services.answer_router import AnswerRouteDecision
    from utils.helpers import extract_product_search_query

    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(route_data):
            sq_locked = (route_data.get("search_query") or "").strip()
            if not sq_locked:
                try:
                    from services.ai_route_semantics import _catalog_search_query_from_brain_route

                    sq_locked = _catalog_search_query_from_brain_route(route_data)
                except ImportError:
                    sq_locked = ""
            log_reasoning(
                f"AI route → product catalog API (brain locked sq={sq_locked!r}) — beats KB."
            )
            return (
                AnswerRouteDecision(
                    source="ai_product",
                    intent="product",
                    handler="product_ai_flow",
                    search_query=sq_locked,
                    is_welfog_related=True,
                    reason=reasoning or "Brain locked product catalog.",
                ),
                route_data,
            )
    except ImportError:
        pass

    if not turn_requests_product_catalog(
        original_msg,
        msg_en,
        conv_for_llm,
        ai_route=route_data,
        allow_llm=True,
    ):
        return None, route_data

    resolved = resolve_product_search_turn(
        original_msg,
        msg_en,
        conv_for_llm,
        ai_route=route_data,
        allow_llm=True,
    )
    route_data = apply_product_catalog_to_route(
        route_data,
        original_msg,
        msg_en,
        conversation_context=conv_for_llm,
    )
    comb = _combined(original_msg, msg_en)
    sq = (
        resolved.search_query
        or (route_data.get("search_query") or "").strip()
        or extract_product_search_query(comb, comb, conv_for_llm, ai_route=route_data)
        or ""
    )
    log_product_catalog_routing(
        detected_intent="product_search",
        product_entities=resolved.entities or route_data.get("_product_entities"),
        selected_route="product_ai_flow",
        filters=resolved.entities or {},
        source=resolved.source or "product_catalog_resolver",
    )
    log_reasoning(
        f"AI route → product catalog API (sq={sq!r}) — locked, beats KB."
    )
    decision = AnswerRouteDecision(
        source="ai_product",
        intent="product",
        handler="product_ai_flow",
        search_query=sq,
        is_welfog_related=True,
        reason=f"Product browse/search — {reasoning}",
    )
    return decision, route_data


def _ai_entity_token_list(value: Any) -> list[str]:
    """Normalize brain token list fields (string or JSON array)."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [t.strip() for t in re.split(r"[,;|]", value) if t.strip()]
    return []


def _ai_entity_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return default


def entities_to_understanding(
    entities: dict | None,
    *,
    search_query: str = "",
    original_msg: str = "",
) -> dict[str, Any] | None:
    """Map AI product entities → OpenSearch/catalog understanding (no hardcoded brands)."""
    e = dict(entities or {})
    out: dict[str, Any] = {"action": "search_products", "is_shopping": True}

    product_name = (e.get("product_name") or "").strip()
    if not product_name and search_query:
        try:
            from services.product_filter_pipeline import brain_search_query_is_noisy

            if not brain_search_query_is_noisy(search_query):
                product_name = search_query.strip()
        except ImportError:
            product_name = search_query.strip()
    if product_name:
        out["search_terms"] = product_name

    brand = sanitize_catalog_brand(
        (e.get("brand") or "").strip(),
        product_name=product_name,
        explicit_from_brain=bool((e.get("brand") or "").strip()),
        user_meaning="",
    )
    if not brand and original_msg:
        try:
            from services.opensearch_products import (
                _extract_brand_literal_from_text,
                _infer_brand_from_message,
            )

            inferred = _infer_brand_from_message(original_msg) or _extract_brand_literal_from_text(
                original_msg
            )
            if inferred:
                brand = sanitize_catalog_brand(
                    inferred,
                    product_name=product_name,
                    explicit_from_brain=True,
                    user_meaning="",
                )
        except ImportError:
            pass
    if brand:
        out["brand"] = brand
        aliases: list[str] = []
        bl = brand.lower()
        aliases.append(bl)
        if " " in bl:
            aliases.append(bl.replace(" ", ""))
        compact = re.sub(r"[^a-z0-9]", "", bl)
        if compact and compact not in aliases:
            aliases.append(compact)
        out["brand_aliases"] = aliases

    intent = (e.get("product_intent") or "").strip().lower()
    allow_related = _ai_entity_bool(e.get("allow_related_fallback"), default=True)
    related_terms = (e.get("related_search_terms") or "").strip()
    exclude_toks = _ai_entity_token_list(e.get("exclude_title_tokens"))
    mandatory_toks = _ai_entity_token_list(e.get("mandatory_match_tokens"))

    if intent == "device":
        out["device_browse"] = True
    if not allow_related:
        out["specific_accessory"] = True
    if related_terms:
        out["related_search_terms"] = related_terms
    if exclude_toks:
        out["exclude_title_tokens"] = exclude_toks
    if mandatory_toks:
        out["mandatory_match_tokens"] = mandatory_toks

    for src, dst in (
        ("color", "color"),
        ("size", "size"),
        ("sku", "sku"),
        ("category", "product_type"),
        ("model", "model"),
    ):
        v = e.get(src)
        if v is not None and str(v).strip():
            out[dst] = str(v).strip()

    if not out.get("color") and original_msg:
        try:
            from services.opensearch_products import (
                extract_color_and_product_title,
                normalize_color_fuzzy,
            )

            col, _ = extract_color_and_product_title(original_msg)
            if not col:
                col = normalize_color_fuzzy(original_msg)
            if col:
                out["color"] = col
        except ImportError:
            pass

    pid = e.get("product_id") or e.get("pro_id")
    if pid is not None and str(pid).strip().isdigit():
        out["pro_id"] = int(str(pid).strip())
        out["search_terms"] = ""
        out["strict_no_relax"] = True

    if out.get("sku"):
        out["strict_sku_match"] = False
        out["search_terms"] = ""
        out.pop("brand", None)
        out.pop("brand_aliases", None)
        out.pop("color", None)
        out.pop("model", None)
        out.pop("product_type", None)
        out.pop("mandatory_match_tokens", None)

    for src, dst in (
        ("price_min", "min_price"),
        ("price_max", "max_price"),
        ("rating_min", "rating_min"),
        ("rating_max", "rating_max"),
    ):
        v = e.get(src)
        if v is not None and v != "":
            try:
                out[dst] = float(v)
            except (TypeError, ValueError):
                pass

    if out.get("max_price") is None and out.get("min_price") is None and original_msg:
        try:
            from services.opensearch_products import _extract_price_bounds

            pmax, pmin = _extract_price_bounds(original_msg, original_msg.lower())
            if pmax is not None:
                out["max_price"] = pmax
            if pmin is not None:
                out["min_price"] = pmin
        except ImportError:
            pass

    if out.get("brand"):
        try:
            from services.opensearch_products import _phone_brand_vocab

            bl = str(out["brand"]).strip().lower()
            if bl in _phone_brand_vocab():
                mmt = list(out.get("mandatory_match_tokens") or [])
                if bl not in [str(t).lower() for t in mmt]:
                    mmt.insert(0, bl)
                out["mandatory_match_tokens"] = mmt[:4]
        except ImportError:
            pass

    has_filter = any(
        out.get(k) is not None
        for k in ("rating_min", "rating_max", "min_price", "max_price")
    )
    has_product = bool(out.get("search_terms") or out.get("brand") or out.get("sku") or out.get("pro_id"))
    if has_filter and not out.get("search_terms"):
        out["search_terms"] = ""
    if not has_product and not has_filter:
        return None
    out["_ai_first"] = True
    out["is_shopping"] = True
    out["action"] = "search_products"
    if out.get("pro_id"):
        out["strict_no_relax"] = True
    try:
        from services.catalog_spec_semantics import align_catalog_terms_to_user_message

        if original_msg:
            out = align_catalog_terms_to_user_message(out, original_msg, "")
    except ImportError:
        pass
    return out


def log_product_catalog_routing(
    *,
    detected_intent: str,
    product_entities: dict | None = None,
    selected_route: str = "",
    filters: dict | None = None,
    result_count: int | None = None,
    source: str = "",
) -> None:
    line = (
        f"[product-route] detected_intent={detected_intent} "
        f"product_entities={product_entities or {}} "
        f"selected_route={selected_route!r} "
        f"filters={filters or {}} "
        f"result_count={result_count if result_count is not None else '-'} "
        f"source={source or '-'}"
    )
    log_reasoning(line)
    chat_log(line)
