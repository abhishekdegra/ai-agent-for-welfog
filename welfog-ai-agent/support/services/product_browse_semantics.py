"""
Product browse / availability — semantic-first (any language, typos, long prompts).

Priority: Groq user_meaning + intent → embedding vs contrast exemplars → light structure.
Fixed phrase lists are fallback only, not the primary detector.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from utils.reasoning_log import log_reasoning

# Fallback only — semantic path must work without these exact strings.
_AVAILABILITY_MARKERS_FALLBACK = (
    "apke pas", "apke paas", "aapke pas", "tumhare pas", "paas hai", "paas he",
    "hai kya", "h kya", "he kya", "milega kya", "milta kya", "do you have",
    "in stock", "sell karte", "available hai",
)

_PRODUCT_BROWSE_EXEMPLARS = (
    "do you have kurtis on welfog",
    "shirts available in your store can I buy",
    "can I buy bottles here on welfog",
    "kurtiyaa hai kya apke pas",
    "bottals he kya tumhare paas",
    "show me if you sell rice and wheat",
    "mujhe iphone cover chahiye kya milta hai",
    "I am looking for red shoes under 500 rupees",
    "aaj shopping karni hai white shirt dikhao",
    "neenga inga dress vangalaama",
    "ee shop la mobile cover irukka",
    "kya aapke paas ye product available hai",
    "long prompt but basically want to see kurta options on welfog",
    "mereko lal mirch chahiye khana banane ke liye",
    "flour mil jayga kya welfog pe",
    "merko black fan chahiye ghar ke liye",
    "i need a lipgloss of himalaya",
    "silver ki ring leni h mujhe",
    "naku red saree kavali budget lo",
    "enaku idhu phone charger venum",
    "mujhe atta chahiye welfog se",
    "koi accha shampoo suggest karo",
    "show cheapest earbuds",
    "wo wireless mouse hai kya",
    "mala kitchen mixer pahije",
    "আমাকে লিপগ্লস চাই",
    "मुझे काली मिर्च चाहिए",
)

_NOT_PRODUCT_BROWSE_EXEMPLARS = (
    "who are you chatbot",
    "what is welfog company founder",
    "track my order 887889",
    "refund not received",
    "delivery to pincode 302012",
    "show my wishlist saved items",
    "privacy policy terms and conditions",
)

_FUZZY_NOUN_MAP = {
    "kurtiyaa": "kurta",
    "kurtiya": "kurta",
    "kurtis": "kurta",
    "kurti": "kurta",
    "kurtas": "kurta",
    "bottals": "bottle",
    "bottal": "bottle",
    "bottles": "bottle",
    "shirts": "shirt",
    "tshirts": "tshirt",
    "mobiles": "mobile",
    "covers": "cover",
    "cases": "case",
}

_BROWSE_SCORE_HIGH = 0.44
_BROWSE_SCORE_LLM_ASSIST = 0.36
_CONTRAST_MARGIN = 0.04
_GENERIC_QUERY_TOKENS = frozenset(
    {
        "help",
        "please",
        "plz",
        "kaam",
        "baat",
        "sun",
        "dhyan",
        "school",
        "college",
        "rain",
        "barish",
        "coffee",
        "drive",
        "speed",
        "love",
        "you",
        "chal",
        "chalega",
        "jana",
        "jaana",
        "kaise",
        "kese",
    }
)
_SHOPPING_REQUEST_MARKERS = (
    "dikha",
    "show",
    "buy",
    "chahiye",
    "chahie",
    "search",
    "find",
    "available",
    "milega",
    "milta",
    "stock",
    "order",
)


def _norm(text: str) -> str:
    from utils.helpers import _normalize_welfog_typos

    return f" {_normalize_welfog_typos(text or '')} "


def _llm_blob(route: dict | None) -> str:
    r = route or {}
    return f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".lower()


def llm_meaning_is_product_browse(route: dict | None) -> bool:
    """Trust routing LLM English summary — any customer language/script."""
    blob = _llm_blob(route)
    if not blob.strip():
        return False
    if llm_meaning_is_not_product_browse(route):
        return False
    browse_patterns = (
        r"\b(?:find|search|browse|show|buy|shop(?:ping)?)\s+(?:for\s+)?(?:a\s+)?\w+",
        r"\b(?:do you|does welfog)\s+(?:have|sell|stock|carry)\b",
        r"\b(?:product|item|catalog)\s+(?:search|availability|browse)\b",
        r"\b(?:available|availability|in stock)\b.*\b(?:product|shirt|kurta|cover|bottle)\b",
        r"\b(?:wants?|asking)\s+(?:to\s+)?(?:see|find|buy|browse)\b",
        r"\b(?:customer\s+)?wants?\s+.*\b(?:product|items?|kurti|shirt|cover|phone)\b",
        r"\bcheck\s+if\b.*\b(?:sell|have|stock)\b",
        r"\blooking\s+for\b.*\b(?:on welfog|welfog)\b",
        r"\bhai\s+kya\b.*\b(?:pas|stock|available)\b",
        r"\b(?:milega|milta|milti)\b.*\b(?:product|cover|shirt)\b",
    )
    return any(re.search(p, blob) for p in browse_patterns)


def llm_meaning_is_not_product_browse(route: dict | None) -> bool:
    blob = _llm_blob(route)
    if not blob.strip():
        return False
    block_patterns = (
        r"\b(?:track|tracking)\s+(?:order|shipment)\b",
        r"\b(?:refund|return)\s+(?:status|policy)\b",
        r"\b(?:order\s+)?history\b",
        r"\b(?:pincode|pin\s+code|delivery)\s+(?:to|at|check)\b",
        r"\b(?:who|what)\s+(?:you|the\s+bot)\s+(?:are|is)\b",
        r"\b(?:wishlist|saved\s+items)\b",
        r"\b(?:privacy|terms|policy)\s+(?:read|show)\b",
        r"\b(?:founder|ceo|company\s+story)\b",
    )
    return any(re.search(p, blob) for p in block_patterns)


def token_is_product_noun(token: str) -> bool:
    if not token or len(token) < 3:
        return False
    t = token.lower()
    try:
        from services.welfog_api import PRODUCT_TYPE_NOUNS, _token_is_product_noun

        if _token_is_product_noun(t):
            return True
        mapped = _FUZZY_NOUN_MAP.get(t)
        if mapped and mapped in PRODUCT_TYPE_NOUNS:
            return True
        if t in _FUZZY_NOUN_MAP:
            return True
        if len(t) > 4 and t.endswith("aa"):
            stem = t[:-1]
            if _token_is_product_noun(stem) or stem in _FUZZY_NOUN_MAP:
                return True
    except ImportError:
        pass
    return t in _FUZZY_NOUN_MAP


def _product_nouns_in_text(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", _norm(text)):
        if token_is_product_noun(tok):
            mapped = _FUZZY_NOUN_MAP.get(tok, tok)
            if mapped not in out:
                out.append(mapped)
    return out


def _has_concrete_catalog_anchor(text: str, route: dict | None = None) -> bool:
    """
    Product route must have a concrete item anchor in THIS turn.
    Prevents stale context from turning generic chat into catalog search.
    """
    try:
        from services.product_catalog_resolver import KIND_PRODUCT_SEARCH, resolve_product_search_turn

        resolved = resolve_product_search_turn(text, "", allow_llm=False)
        if resolved.kind == KIND_PRODUCT_SEARCH and (
            resolved.search_query or resolved.entities
        ):
            return True
    except ImportError:
        pass
    if _product_nouns_in_text(text):
        return True
    r = route or {}
    sq = (r.get("search_query") or "").strip().lower()
    if not sq:
        return False
    toks = [t for t in re.findall(r"[a-z0-9]+", sq) if t]
    if not toks:
        return False
    if any(token_is_product_noun(t) for t in toks):
        return True
    # Allow short queries only when the user explicitly asks to shop/search.
    text_low = (text or "").lower()
    if (
        len(toks) <= 3
        and not all(t in _GENERIC_QUERY_TOKENS for t in toks)
        and any(m in text_low for m in _SHOPPING_REQUEST_MARKERS)
    ):
        return True
    return False


@lru_cache(maxsize=1)
def _browse_exemplar_vectors():
    from services.embedding import encode_texts

    pos = encode_texts(list(_PRODUCT_BROWSE_EXEMPLARS))
    neg = encode_texts(list(_NOT_PRODUCT_BROWSE_EXEMPLARS))
    return pos, neg


def _max_cosine_to_matrix(text: str, matrix) -> float:
    from sklearn.metrics.pairwise import cosine_similarity

    from services.embedding import encode_texts

    q = (text or "").strip()
    if not q:
        return 0.0
    try:
        qv = encode_texts([q[:480]])
        if qv is None:
            return 0.0
        sims = cosine_similarity(qv, matrix)[0]
        return float(max(sims)) if len(sims) else 0.0
    except Exception:
        return 0.0


def embedding_browse_scores(text: str) -> tuple[float, float]:
    """(positive browse score, negative/conflict score)."""
    pos_m, neg_m = _browse_exemplar_vectors()
    return _max_cosine_to_matrix(text, pos_m), _max_cosine_to_matrix(text, neg_m)


def semantic_prefers_product_browse(text: str, min_score: float = _BROWSE_SCORE_HIGH) -> bool:
    if _hard_exclude_browse(text):
        return False
    pos, neg = embedding_browse_scores(text)
    if pos < min_score:
        return False
    if neg > pos - _CONTRAST_MARGIN:
        return False
    try:
        from services.meta_turn_semantics import embedding_meta_scores

        intro_s, company_s = embedding_meta_scores(text)
        if intro_s > pos - _CONTRAST_MARGIN or company_s > pos - _CONTRAST_MARGIN:
            return False
    except ImportError:
        pass
    return True


def _product_availability_not_policy_kb(text: str) -> bool:
    """Stock/availability on Welfog — semantic browse, not policy KB read."""
    try:
        from utils.helpers import (
            _text_has_refund_or_return_intent,
            message_is_welfog_about_request,
            message_needs_policy_answer,
        )

        if message_needs_policy_answer(text) or _text_has_refund_or_return_intent(text):
            return False
        if message_is_welfog_about_request(text):
            return False
    except ImportError:
        pass
    try:
        pos, neg = embedding_browse_scores(text)
        if pos >= 0.28 and neg < pos - 0.02:
            return True
    except Exception:
        pass
    tl = _norm(text)
    if "welfog" in tl or "wlefog" in tl or "welkog" in tl:
        return True
    return False


def _hard_exclude_browse(text: str) -> bool:
    """Lightweight excludes only — must not call should_skip_catalog (recursion with intro path)."""
    tl = _norm(text)
    if _product_availability_not_policy_kb(text):
        return False
    try:
        from utils.helpers import _is_light_smalltalk_fast, _is_short_pure_greeting

        if _is_short_pure_greeting(text) or _is_light_smalltalk_fast(text):
            return True
    except ImportError:
        pass
    try:
        from services.conversation_followup import is_non_product_search_phrase

        if is_non_product_search_phrase(text):
            return True
    except ImportError:
        pass
    if re.search(r"\b(?:kese|kaise|kaisa|kesa|how are you|how r u)\b", tl) and not _product_nouns_in_text(
        text
    ):
        if len(re.findall(r"[a-z]+", tl)) <= 10:
            return True
    if re.search(r"\b(?:hello|hi|hey|namaste)\b", tl) and re.search(
        r"\b(?:kesa|kaise|kese|kaisa|how are you|how r u)\b", tl
    ):
        if not _product_nouns_in_text(text):
            return True
    if re.search(r"\b(?:tu|tum|aap)\s+search\s+kr", tl) or "search krke bta" in tl:
        return True
    if re.search(r"\b(?:tu|tum|aap|you|bot)\b", tl) and re.search(
        r"\b(?:kon|kaun|who)\b", tl
    ):
        if not any(m in tl for m in _AVAILABILITY_MARKERS_FALLBACK):
            return True
    try:
        from utils.helpers import (
            message_is_welfog_about_request,
            message_is_knowledge_information_request,
            message_needs_policy_answer,
            _text_is_order_tracking_intent,
            _text_has_refund_or_return_intent,
            _text_has_pincode_delivery_intent,
        )

        comb = text
        if _product_availability_not_policy_kb(comb):
            return False
        if message_is_welfog_about_request(comb):
            return True
        if message_is_knowledge_information_request(comb):
            try:
                pos, neg = embedding_browse_scores(comb)
                if pos >= 0.28 and neg < pos - 0.02:
                    return False
            except Exception:
                pass
            return True
        if message_needs_policy_answer(comb):
            return True
        if _text_is_order_tracking_intent(comb) or _text_has_refund_or_return_intent(comb):
            return True
        if _text_has_pincode_delivery_intent(comb):
            from utils.helpers import extract_pincode_preferred_from_message

            if extract_pincode_preferred_from_message(comb):
                return True
            if re.search(r"\b(?:pincode|pin\s*code|serviceability)\b", tl):
                return True
            # "home delivery" in a product-stock question — not a PIN check turn.
    except ImportError:
        pass
    return False


def _keyword_failsafe_browse(text: str) -> bool:
    """Last resort when LLM + embeddings uncertain — minimal markers + product noun."""
    if _hard_exclude_browse(text):
        return False
    tl = _norm(text)
    nouns = _product_nouns_in_text(text)
    if nouns and any(m in tl for m in _AVAILABILITY_MARKERS_FALLBACK):
        return True
    return bool(nouns and any(m in tl for m in _SHOPPING_REQUEST_MARKERS))


def ai_route_is_product_browse_turn(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    *,
    _from_resolver: bool = False,
) -> bool:
    """Unified product-browse detection — not customer keyword lists."""
    from services.ai_route_semantics import _normalize_llm_route
    from services.semantic_intent import llm_semantic_route_available

    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    r = _normalize_llm_route(dict(route or {}))

    intent = (r.get("intent") or "").strip().lower()
    if intent in (
        "order_history",
        "wishlist",
        "order",
        "refund",
        "payment",
        "pincode_check",
        "seller",
        "out_of_domain",
    ):
        return False

    if _hard_exclude_browse(comb):
        return False

    if not _from_resolver:
        try:
            from services.product_catalog_resolver import turn_requests_product_catalog

            if turn_requests_product_catalog(
                original_msg, msg_en, "", ai_route=r, allow_llm=True
            ):
                return True
        except ImportError:
            pass

    try:
        from utils.helpers import _text_has_product_shopping_intent_core, _turn_is_catalog_product_request

        if _turn_is_catalog_product_request(comb) or _text_has_product_shopping_intent_core(comb):
            return True
    except ImportError:
        pass
    if _keyword_failsafe_browse(comb):
        return True

    intent = (r.get("intent") or "").strip().lower()
    channel = (r.get("data_channel") or "").strip().lower()
    if (
        intent == "product"
        and channel in ("catalog", "")
        and r.get("run_catalog_search")
        and _has_concrete_catalog_anchor(comb, r)
    ):
        return True

    if llm_semantic_route_available(r):
        if llm_meaning_is_product_browse(r):
            return True
        if llm_meaning_is_not_product_browse(r):
            return False
        if intent == "product" and channel == "catalog" and _has_concrete_catalog_anchor(comb, r):
            return True

    try:
        pos, neg = embedding_browse_scores(comb)
        if pos >= _BROWSE_SCORE_HIGH and neg < pos - _CONTRAST_MARGIN:
            if _has_concrete_catalog_anchor(comb, r):
                return True
        if semantic_prefers_product_browse(comb, min_score=_BROWSE_SCORE_LLM_ASSIST) and _has_concrete_catalog_anchor(
            comb, r
        ):
            return True
        if _product_nouns_in_text(comb) and pos >= _BROWSE_SCORE_LLM_ASSIST and neg < pos - _CONTRAST_MARGIN:
            return True
    except Exception:
        pass
    return _keyword_failsafe_browse(comb)


def message_is_product_availability_browse(
    text: str,
    ai_route: dict | None = None,
) -> bool:
    if ai_route:
        return ai_route_is_product_browse_turn(ai_route, text, "")
    return ai_route_is_product_browse_turn(None, text, "")


def blocks_assistant_intro_fast_path(text: str) -> bool:
    """Product-stock question — do not answer with assistant intro (no full browse graph)."""
    if _hard_exclude_browse(text):
        return False
    tl = _norm(text)
    if _product_nouns_in_text(text) and any(
        m in tl for m in _AVAILABILITY_MARKERS_FALLBACK
    ):
        return True
    return False


def extract_browse_search_terms(text: str, ai_route: dict | None = None) -> str:
    r = ai_route or {}
    sq = (r.get("search_query") or "").strip()
    if sq and sq.lower() not in ("product", "item", "none", "null"):
        return sq
    nouns = _product_nouns_in_text(text)
    if nouns:
        return " ".join(nouns[:4])
    try:
        from utils.helpers import extract_product_search_query

        return extract_product_search_query(text, text, "", ai_route=ai_route) or ""
    except ImportError:
        return ""


def promote_product_browse_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    from services.ai_route_semantics import _normalize_llm_route

    out = _normalize_llm_route(dict(route or {}))
    try:
        from services.account_list_semantics import account_list_route_is_locked

        if account_list_route_is_locked(out):
            return out
    except ImportError:
        pass
    try:
        from services.product_catalog_resolver import apply_product_catalog_to_route

        out = apply_product_catalog_to_route(
            out, original_msg, msg_en, conversation_context=conversation_context
        )
        if (out.get("intent") or "").strip().lower() == "product" and (
            out.get("data_channel") or ""
        ).strip().lower() == "catalog":
            return out
    except ImportError:
        pass
    mk = (out.get("meta_kind") or "none").strip().lower()
    if mk not in ("none", "", "conversational"):
        return out
    comb = f"{original_msg or ''} {msg_en or ''}".strip()
    if not ai_route_is_product_browse_turn(out, original_msg, msg_en):
        return out
    if (out.get("intent") or "").strip().lower() == "product" and (
        out.get("data_channel") or ""
    ).strip().lower() == "catalog":
        return out

    sq = extract_browse_search_terms(comb, out) or (out.get("search_query") or "").strip()
    log_reasoning(f"Promoted route → product catalog (semantic browse, sq={sq!r}).")
    out["intent"] = "product"
    out["data_channel"] = "catalog"
    out["run_catalog_search"] = True
    out["is_welfog_related"] = True
    out["needs_order_id"] = False
    out["meta_kind"] = "none"
    out["search_query"] = sq
    out.pop("route_handler", None)
    return out
