"""
Semantic meta-turn detection (assistant identity, company-about) — any language/phrasing.

Priority: LLM meta_kind + user_meaning → embedding exemplars → minimal keyword failsafe.
Not a fixed phrase list on the customer message.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from utils.reasoning_log import log_reasoning

# Embedding similarity (multilingual-ish via mixed exemplars + user text).
_ASSISTANT_INTRO_EXEMPLARS = (
    "who are you in this chat",
    "what are you doing right now",
    "introduce yourself as the support assistant",
    "are you a bot or a human agent",
    "what is your name and role here",
    "tell me about yourself as the shopping chatbot",
    "what can you do for me on welfog support",
    "explain what kind of assistant you are",
    "neenga yaar intha chatbot",
    "nee yaaru ithu bot aa",
    "tum kaun ho yahan",
    "aap kaun hain assistant",
    "tu kya kar raha hai abhi",
    "kya kaam karte ho tum",
    "please explain your purpose in this conversation",
    "what is your role in helping me shop",
    "mujhe samjhao tum yahan kya help karte ho",
)

_COMPANY_ABOUT_EXEMPLARS = (
    "what is welfog as a company",
    "tell me about welfog business and founder",
    "welfog company story ceo founder",
    "what does welfog platform do for sellers",
    "about welfog brand history",
    "welfog ke baare me batao kuchh",
    "welfog kya hai company platform",
    "welfog kya karti hai kya kaam karti hai",
    "what is welfog marketplace not the chatbot",
    "explain welfog ecommerce platform",
    "welfog kya cheez hai",
    "tell me about welfog the company not you the bot",
    "welfog website app kya hai",
    "founder of welfog company",
)

_INTRO_SCORE_HIGH = 0.46
_INTRO_SCORE_LLM_ASSIST = 0.38
_INTRO_MARGIN_OVER_COMPANY = 0.035


def _combined_user_text(original_msg: str, msg_en: str = "") -> str:
    parts = [(original_msg or "").strip(), (msg_en or "").strip()]
    return " ".join(p for p in parts if p).strip()


def _turn_blocks_assistant_intro(text: str) -> bool:
    """Substantive shopping/support — never treat as bot small-talk."""
    if not (text or "").strip():
        return True
    try:
        from services.product_browse_semantics import message_is_product_availability_browse

        if message_is_product_availability_browse(text):
            return True
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _message_has_catalog_product_signal,
            _text_asks_wishlist,
            _text_has_product_shopping_intent,
            _text_has_refund_or_return_intent,
            _text_is_order_tracking_intent,
            _text_wants_order_history_list_in_chat,
            extract_order_id,
            message_is_wishlist_like_request,
        )

        comb = text
        if _text_has_product_shopping_intent(comb):
            return True
        if _message_has_catalog_product_signal(comb) and _text_has_product_shopping_intent(comb):
            return True
        if _text_is_order_tracking_intent(comb) or _text_has_refund_or_return_intent(comb):
            return True
        if extract_order_id(comb, ""):
            return True
        if message_is_wishlist_like_request(comb) or _text_asks_wishlist(comb):
            return True
        if _text_wants_order_history_list_in_chat(comb):
            return True
    except ImportError:
        pass
    try:
        from services.account_list_semantics import message_requests_account_list_data

        if message_requests_account_list_data(text):
            return True
    except ImportError:
        pass
    return False


@lru_cache(maxsize=1)
def _exemplar_vectors():
    from services.embedding import encode_texts

    intro = encode_texts(list(_ASSISTANT_INTRO_EXEMPLARS))
    company = encode_texts(list(_COMPANY_ABOUT_EXEMPLARS))
    return intro, company


def _max_cosine_to_exemplars(text: str, exemplar_matrix) -> float:
    from sklearn.metrics.pairwise import cosine_similarity

    from services.embedding import encode_texts

    q = (text or "").strip()
    if not q or len(q) < 2:
        return 0.0
    try:
        qv = encode_texts([q[:420]])
        sims = cosine_similarity(qv, exemplar_matrix)[0]
        return float(max(sims)) if len(sims) else 0.0
    except Exception:
        return 0.0


def embedding_meta_scores(text: str) -> tuple[float, float]:
    """Returns (assistant_intro_score, company_about_score)."""
    intro_vecs, company_vecs = _exemplar_vectors()
    return (
        _max_cosine_to_exemplars(text, intro_vecs),
        _max_cosine_to_exemplars(text, company_vecs),
    )


def _llm_blob(route: dict | None) -> str:
    r = route or {}
    return f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".lower()


def llm_meaning_is_assistant_intro(route: dict | None) -> bool:
    """Trust routing LLM English summary — works for any customer language."""
    blob = _llm_blob(route)
    if not blob.strip():
        return False
    if llm_meaning_is_company_about(route):
        return False
    intro_patterns = (
        r"\b(?:who|what)\s+(?:you|the\s+(?:bot|assistant|chatbot|ai))\s+(?:are|is)\b",
        r"\b(?:who|what)\s+the\s+(?:support\s+)?(?:chat\s?)?bot\s+is\b",
        r"\basking\s+(?:who|what)\s+(?:you|the\s+assistant)\b",
        r"\b(?:bot|assistant|chatbot)\s+identity\b",
        r"\bintroduce\s+(?:yourself|the\s+assistant)\b",
        r"\b(?:human|person|real)\s+or\s+(?:bot|ai|robot|automated)\b",
        r"\bwhat\s+(?:you|the\s+bot)\s+(?:can\s+do|helps?\s+with)\b",
        r"\bwhat\s+are\s+you\s+doing\b",
        r"\b(?:explain|describe)\s+(?:your\s+)?purpose\b",
        r"\brole\s+of\s+(?:this\s+)?(?:chat\s+)?bot\b",
        r"\babout\s+(?:you|the\s+assistant)\s+not\s+welfog\b",
        r"\bwho\s+is\s+this\s+(?:chat\s+)?bot\b",
        r"\bself[- ]?introduction\b",
        r"\bcustomer\s+wants\s+to\s+know\b.*\b(?:bot|assistant|you)\b",
        r"\bnot\s+(?:a\s+)?company\s+(?:info|faq)\b",
        r"\bwhat\s+kind\s+of\s+(?:agent|assistant)\b",
    )
    return any(re.search(p, blob) for p in intro_patterns)


def llm_meaning_is_company_about(route: dict | None) -> bool:
    blob = _llm_blob(route)
    if not blob.strip():
        return False
    company_patterns = (
        r"\bwhat\s+(?:is\s+)?welfog\b",
        r"\bwelfog\s+(?:company|business|platform|brand|story|marketplace|ecommerce|website|app)\b",
        r"\b(?:founder|ceo|about\s+us)\b",
        r"\bcompany\s+(?:info|information|profile)\b",
        r"\babout\s+welfog\b",
        r"\bcustomer\s+wants\s+(?:to\s+)?know\s+(?:about\s+)?welfog\b",
        r"\bwelfog\s+(?:does|karta|karti|karte)\b",
        r"\bwhat\s+welfog\s+(?:is|does)\b",
    )
    if not any(re.search(p, blob) for p in company_patterns):
        return False
    if re.search(r"\b(?:this\s+)?(?:chat\s+)?bot\b", blob) and re.search(
        r"\b(?:who|what)\s+you\s+are\b", blob
    ):
        if not re.search(r"\bwelfog\s+(?:company|business|founder)\b", blob):
            return False
    return True


def semantic_prefers_company_about(text: str, min_score: float = 0.38) -> bool:
    if not (text or "").strip():
        return False
    try:
        from services.user_query_semantics import user_frustration_about_company_not_bot

        if user_frustration_about_company_not_bot(text):
            return True
        from utils.helpers import (
            message_is_welfog_about_request,
            _text_has_platform_overview_intent,
        )

        if message_is_welfog_about_request(text) or _text_has_platform_overview_intent(text):
            return True
    except ImportError:
        pass
    intro_s, company_s = embedding_meta_scores(text)
    if company_s < min_score:
        return False
    return company_s > intro_s + _INTRO_MARGIN_OVER_COMPANY


def semantic_prefers_assistant_intro(text: str, min_score: float = _INTRO_SCORE_HIGH) -> bool:
    if _turn_blocks_assistant_intro(text):
        return False
    if semantic_prefers_company_about(text, min_score=0.34):
        return False
    intro_s, company_s = embedding_meta_scores(text)
    if intro_s < min_score:
        return False
    if company_s > intro_s - _INTRO_MARGIN_OVER_COMPANY:
        return False
    return True


def ai_route_is_assistant_intro_turn(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """Unified: LLM meta/meaning + embedding; not customer keyword lists."""
    from services.ai_route_semantics import _normalize_llm_route
    from services.semantic_intent import llm_semantic_route_available

    comb = _combined_user_text(original_msg, msg_en)
    r = _normalize_llm_route(dict(route or {}))

    try:
        from services.user_query_semantics import query_is_welfog_company_or_platform

        if query_is_welfog_company_or_platform(original_msg, msg_en):
            return False
    except ImportError:
        pass

    if (r.get("meta_kind") or "").strip().lower() == "assistant_intro":
        return True
    if (r.get("route_handler") or "").strip() == "assistant_intro":
        return True

    if llm_semantic_route_available(r):
        if llm_meaning_is_assistant_intro(r):
            return True
        if llm_meaning_is_company_about(r):
            return False
        kb_keys = [str(k).lower() for k in (r.get("kb_keys") or [])]
        if "company" in kb_keys and (r.get("intent") or "").lower() == "general":
            if semantic_prefers_assistant_intro(comb, min_score=_INTRO_SCORE_LLM_ASSIST):
                log_reasoning(
                    "Semantic override: LLM picked company KB but meaning/embed = assistant intro."
                )
                return True
            return False

    return semantic_prefers_assistant_intro(comb, min_score=_INTRO_SCORE_LLM_ASSIST)


def should_fast_reply_assistant_intro(original_msg: str, msg_en: str = "") -> bool:
    """
    Pre-AI fast path: embedding-only (any language) — skips LLM + KB for clear bot-identity turns.
    """
    comb = _combined_user_text(original_msg, msg_en)
    if not comb:
        return False
    try:
        from services.user_query_semantics import query_is_welfog_company_or_platform

        if query_is_welfog_company_or_platform(original_msg, msg_en):
            return False
    except ImportError:
        pass
    try:
        from utils.helpers import message_is_welfog_about_request

        if message_is_welfog_about_request(comb):
            return False
    except ImportError:
        pass
    try:
        from services.product_browse_semantics import blocks_assistant_intro_fast_path

        if blocks_assistant_intro_fast_path(comb):
            return False
    except ImportError:
        pass
    if semantic_prefers_assistant_intro(comb, min_score=_INTRO_SCORE_HIGH):
        log_reasoning(
            f"Semantic assistant-intro fast path (embed intro≥{_INTRO_SCORE_HIGH})."
        )
        return True
    try:
        from utils.helpers import message_is_assistant_identity_question_failsafe

        if message_is_assistant_identity_question_failsafe(comb):
            log_reasoning("Assistant-intro keyword failsafe (LLM/embed miss).")
            return True
    except ImportError:
        pass
    return False


def patch_route_for_assistant_intro(route: dict | None) -> dict:
    out = dict(route or {})
    out["meta_kind"] = "assistant_intro"
    out["intent"] = "general"
    out["needs_order_id"] = False
    out["is_welfog_related"] = True
    out["search_query"] = ""
    out["run_catalog_search"] = False
    out["data_channel"] = "none"
    out["route_handler"] = "assistant_intro"
    out["kb_keys"] = []
    out["continue_previous_topic"] = False
    out.pop("handler", None)
    return out


def promote_assistant_intro_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
) -> dict:
    """After LLM route: fix missed meta_kind using user_meaning + embeddings."""
    from services.ai_route_semantics import _normalize_llm_route

    out = _normalize_llm_route(dict(route or {}))
    kind = (out.get("account_list_kind") or "").strip().lower()
    if kind and kind not in ("none", ""):
        return out
    if (out.get("intent") or "").strip().lower() in ("wishlist", "order_history"):
        if (out.get("meta_kind") or "").strip().lower() == "assistant_intro":
            out["meta_kind"] = "none"
            out.pop("route_handler", None)
        return out
    try:
        from services.account_list_semantics import message_requests_account_list_data

        if message_requests_account_list_data(original_msg, msg_en):
            return out
    except ImportError:
        pass
    mk = (out.get("meta_kind") or "none").strip().lower()
    comb = _combined_user_text(original_msg, msg_en)
    try:
        from services.user_query_semantics import query_is_welfog_company_or_platform
        from utils.helpers import message_is_welfog_about_request

        if query_is_welfog_company_or_platform(original_msg, msg_en) or message_is_welfog_about_request(
            comb
        ):
            if (out.get("meta_kind") or "").strip().lower() == "assistant_intro":
                out["meta_kind"] = "none"
                out.pop("route_handler", None)
                out["data_channel"] = "kb"
                keys = list(out.get("kb_keys") or [])
                for k in ("company", "faqs"):
                    if k not in keys:
                        keys.append(k)
                out["kb_keys"] = keys
                log_reasoning(
                    "Route fix: Welfog company/platform question — KB (not assistant_intro)."
                )
            return out
    except ImportError:
        pass

    if mk in (
        "hostile",
        "bot_latency",
        "topic_denial",
        "out_of_domain",
        "wrong_search_complaint",
        "assistant_intro",
    ):
        if mk == "assistant_intro":
            return patch_route_for_assistant_intro(out)
        return out

    try:
        from services.user_query_semantics import query_is_welfog_company_or_platform

        if query_is_welfog_company_or_platform(original_msg, msg_en):
            return out
    except ImportError:
        pass
    if not ai_route_is_assistant_intro_turn(out, original_msg, msg_en):
        return out

    log_reasoning("Promoted route → assistant_intro (semantic, any language).")
    return patch_route_for_assistant_intro(out)
