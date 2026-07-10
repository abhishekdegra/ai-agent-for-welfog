"""
Immutable KB fact grounding contract.

Retrieved chunk text is the sole authority for factual fields in KB answers.
The final response must never invent, swap, or omit values that already exist
in the grounding corpus (phones, emails, URLs, dates, identifiers).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroundedFacts:
    """Factual values extracted from authoritative KB corpus / retrieved chunks."""

    phones: frozenset[str] = field(default_factory=frozenset)
    toll_free: frozenset[str] = field(default_factory=frozenset)
    emails: frozenset[str] = field(default_factory=frozenset)
    urls: frozenset[str] = field(default_factory=frozenset)
    dates: frozenset[str] = field(default_factory=frozenset)
    identifiers: frozenset[str] = field(default_factory=frozenset)

    def has_contact_facts(self) -> bool:
        return bool(self.phones or self.emails)


def _chunk_corpus(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for h in hits or []:
        c = (h.get("chunk") or "").strip()
        if c:
            parts.append(c)
    return "\n".join(parts)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _normalize_url(url: str) -> str:
    u = (url or "").strip().lower().rstrip(".,;)")
    if u.startswith("www."):
        u = "https://" + u
    return u


def _emails_in_text(text: str) -> set[str]:
    return {
        e.lower()
        for e in re.findall(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", (text or "").lower())
    }


def _phones_in_text(text: str) -> set[str]:
    try:
        from services.kb_service import extract_contacts_from_plain_text

        phones, _, _ = extract_contacts_from_plain_text(text or "", include_grievance=False)
        return set(phones)
    except ImportError:
        out: set[str] = set()
        for m in re.finditer(r"\b(?:\+?91[\s-]*)?([6-9]\d{9})\b", text or ""):
            out.add(m.group(1))
        return out


def _tollfree_numbers_in_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(
        r"\b(1[\s\-]?800(?:[\s\-]?\d{3,4}){1,2})\b",
        (text or "").lower(),
    ):
        out.add(re.sub(r"[\s\-]+", "", m.group(1)))
    return out


def _social_handles_in_text(text: str) -> set[str]:
    return {h.lower() for h in re.findall(r"@[\w]+", text or "", re.I)}


def _urls_in_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text or "", re.I):
        out.add(_normalize_url(m.group(0)))
    return out


def _dates_in_text(text: str) -> set[str]:
    out: set[str] = set()
    patterns = (
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    )
    low = (text or "").lower()
    for pat in patterns:
        for m in re.finditer(pat, low, re.I):
            out.add(m.group(0).strip())
    return out


def _identifiers_in_text(text: str) -> set[str]:
    """GSTIN, pin codes in address context, and other official alphanumeric ids."""
    out: set[str] = set()
    for m in re.finditer(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", text or "", re.I):
        out.add(m.group(0).upper())
    for m in re.finditer(r"\b(?:pin(?:code)?|zip)\s*[:\-]?\s*(\d{6})\b", text or "", re.I):
        out.add(m.group(1))
    return out


def extract_grounded_facts(corpus: str) -> GroundedFacts:
    """Extract all grounded factual values from authoritative KB text."""
    blob = corpus or ""
    if not blob.strip():
        return GroundedFacts()
    return GroundedFacts(
        phones=frozenset(_phones_in_text(blob)),
        toll_free=frozenset(_tollfree_numbers_in_text(blob)),
        emails=frozenset(_emails_in_text(blob)),
        urls=frozenset(_urls_in_text(blob)),
        dates=frozenset(_dates_in_text(blob)),
        identifiers=frozenset(_identifiers_in_text(blob)),
    )


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _value_in_corpus(value: str, corpus: str) -> bool:
    if not value or not corpus:
        return False
    norm = (value or "").strip().lower()
    corp_low = corpus.lower()
    if norm in corp_low:
        return True
    if _digits_only(norm) and _digits_only(norm) in _digits_only(corpus):
        return True
    return False


def answer_has_ungrounded_facts(answer: str, facts: GroundedFacts, corpus: str) -> bool:
    """True when the answer cites factual values absent from the grounding corpus."""
    if not corpus.strip() or not (answer or "").strip():
        return False
    plain = _strip_html(answer)

    ans_phones = _phones_in_text(plain)
    if ans_phones and not ans_phones.issubset(facts.phones):
        return True

    ans_toll = _tollfree_numbers_in_text(plain)
    if ans_toll and not ans_toll.issubset(facts.toll_free):
        return True

    ans_emails = _emails_in_text(plain)
    if ans_emails and not ans_emails.issubset(facts.emails):
        return True

    ans_urls = _urls_in_text(plain)
    if ans_urls and not ans_urls.issubset(facts.urls):
        return True

    ans_dates = _dates_in_text(plain)
    if ans_dates and not ans_dates.issubset(facts.dates):
        return True

    ans_ids = _identifiers_in_text(plain)
    if ans_ids and not ans_ids.issubset(facts.identifiers):
        return True

    corp_handles = _social_handles_in_text(corpus)
    ans_handles = _social_handles_in_text(plain)
    if ans_handles and facts.urls:
        if ans_handles - corp_handles:
            return True

    for seq in re.findall(r"(?<!\d)(\d{10})(?!\d)", _strip_html(answer)):
        if seq not in facts.phones and seq not in _digits_only(corpus):
            return True

    if _tollfree_numbers_in_text(plain) and not _tollfree_numbers_in_text(corpus):
        return True

    return False


def _answer_evades_grounded_social(answer: str, facts: GroundedFacts) -> bool:
    """True when answer deflects despite grounded social URLs in corpus."""
    if not facts.urls:
        return False
    plain = _strip_html(answer).lower()
    if not plain.strip():
        return True
    if any(u.rstrip("/") in plain.replace(" ", "") for u in facts.urls):
        return False
    invented_handles = _social_handles_in_text(plain) - _social_handles_in_text(
        " ".join(facts.urls)
    )
    if invented_handles:
        return True
    evasion = (
        "not available",
        "not publicly available",
        "couldn't find",
        "could not find",
        "don't have",
        "do not have",
        "shopping assistant",
        "social media manager",
        "not a social",
        "outside what i can help",
        "search for @",
        "searching for @",
        "browser mein",
        "on welfog.com",
        "check our website",
        "visit our website",
        "nahi hai",
    )
    return any(p in plain for p in evasion)


def _answer_evades_grounded_contacts(answer: str, facts: GroundedFacts) -> bool:
    """True when answer deflects to website/app despite grounded contact facts."""
    if not facts.has_contact_facts():
        return False
    plain = _strip_html(answer).lower()
    if not plain.strip():
        return True
    has_phone = any(p in _digits_only(plain) for p in facts.phones)
    has_email = any(e in plain for e in facts.emails)
    if has_phone or has_email:
        return False
    evasion = (
        "not available",
        "not publicly available",
        "couldn't find",
        "could not find",
        "don't have that",
        "do not have that",
        "order confirmation email",
        "on the welfog website",
        "on our website",
        "help & support section",
        "help and support section",
        "check our website",
        "visit our website",
    )
    return any(p in plain for p in evasion)


def _sanitize_ungrounded_facts(answer: str, facts: GroundedFacts, corpus: str) -> str:
    """Replace hallucinated factual values with grounded corpus values."""
    if not (answer or "").strip() or not corpus.strip():
        return (answer or "").strip()

    result = answer
    plain = _strip_html(result)

    # Toll-free numbers not in corpus → remove or substitute mobile from corpus.
    for toll in _tollfree_numbers_in_text(plain):
        if toll not in facts.toll_free:
            result = re.sub(
                re.escape(toll),
                next(iter(sorted(facts.phones)), ""),
                result,
                flags=re.I,
            )
            result = re.sub(
                r"\b1[\s\-]?800[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b",
                next(iter(sorted(facts.phones)), ""),
                result,
                flags=re.I,
            )

    # Wrong phones → grounded phone.
    wrong_phones = _phones_in_text(_strip_html(result)) - set(facts.phones)
    if wrong_phones and facts.phones:
        replacement = sorted(facts.phones)[0]
        for wrong in wrong_phones:
            result = re.sub(rf"\b{re.escape(wrong)}\b", replacement, result)
            result = re.sub(
                rf"\+91[\s\-]*{re.escape(wrong)}\b",
                replacement,
                result,
                flags=re.I,
            )

    # Wrong emails → grounded email (preserve display casing from corpus when possible).
    wrong_emails = _emails_in_text(_strip_html(result)) - set(facts.emails)
    if wrong_emails and facts.emails:
        replacement = sorted(facts.emails)[0]
        for wrong in sorted(wrong_emails, key=len, reverse=True):
            result = re.sub(re.escape(wrong), replacement, result, flags=re.I)

    # Wrong URLs → first grounded URL of same domain class, else drop hallucinated URL.
    wrong_urls = _urls_in_text(_strip_html(result)) - set(facts.urls)
    if wrong_urls:
        if facts.urls:
            replacement = sorted(facts.urls)[0]
            for wrong in wrong_urls:
                result = re.sub(re.escape(wrong), replacement, result, flags=re.I)
        else:
            for wrong in wrong_urls:
                result = re.sub(re.escape(wrong), "", result, flags=re.I)

    # Wrong dates / identifiers — remove tokens not in corpus.
    wrong_dates = _dates_in_text(_strip_html(result)) - set(facts.dates)
    for wrong in wrong_dates:
        if not _value_in_corpus(wrong, corpus):
            result = re.sub(re.escape(wrong), "", result, flags=re.I)

    wrong_ids = _identifiers_in_text(_strip_html(result)) - set(facts.identifiers)
    for wrong in wrong_ids:
        if not _value_in_corpus(wrong, corpus):
            result = re.sub(re.escape(wrong), "", result, flags=re.I)

    # Any standalone 10-digit run not grounded (fake numbers, toll-free tails, etc.).
    plain_for_digits = _strip_html(result)
    for m in re.finditer(r"(?<!\d)(\d{10})(?!\d)", plain_for_digits):
        seq = m.group(1)
        if seq in facts.phones or seq in _digits_only(corpus):
            continue
        if facts.phones:
            replacement = sorted(facts.phones)[0]
            result = re.sub(rf"(?<!\d){re.escape(seq)}(?!\d)", replacement, result)
            result = re.sub(
                rf"\+91[\s\-]*{re.escape(seq)}\b",
                replacement,
                result,
                flags=re.I,
            )
        else:
            result = re.sub(
                rf"\+?91[\s\-]*{re.escape(seq)}\b",
                "",
                result,
                flags=re.I,
            )
            result = re.sub(rf"(?<!\d){re.escape(seq)}(?!\d)", "", result)

    return re.sub(r"\s{2,}", " ", result).strip()


def rebuild_contact_answer_from_corpus(
    corpus: str,
    *,
    original_msg: str,
    msg_en: str = "",
) -> str:
    """Deterministic contact reply built only from corpus text."""
    from services.translation_service import customer_reply_language, finalize_customer_reply
    from utils.helpers import _text_asks_customer_care_contact

    if not _text_asks_customer_care_contact(f"{original_msg} {msg_en}"):
        return ""
    if not (corpus or "").strip():
        return ""
    from services.kb_service import extract_contacts_from_plain_text

    phones, emails, _g = extract_contacts_from_plain_text(corpus, include_grievance=False)
    if not phones and not emails:
        return ""
    lines: list[str] = []
    if phones:
        lines.append(
            "Official customer-care number: " + ", ".join(f"<b>{p}</b>" for p in phones)
        )
    if emails:
        lines.append("Support email: " + ", ".join(f"<b>{e}</b>" for e in emails))
    body = "<br><br>".join(lines)
    return finalize_customer_reply(body, original_msg, customer_reply_language(original_msg)) or ""


def rebuild_contact_answer_from_chunks(
    hits: list[dict[str, Any]],
    *,
    original_msg: str,
    msg_en: str = "",
) -> str:
    """Deterministic contact reply built only from retrieved chunk text."""
    return rebuild_contact_answer_from_corpus(
        _chunk_corpus(hits),
        original_msg=original_msg,
        msg_en=msg_en,
    )


def enforce_kb_fact_grounding(
    answer: str,
    hits: list[dict[str, Any]] | None = None,
    *,
    corpus: str = "",
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """
    Immutable fact contract: final answer must only use factual values from corpus.
    Replaces hallucinations, injects grounded contacts when LLM omitted/evaded them.
    """
    grounding_corpus = (corpus or _chunk_corpus(hits or [])).strip()
    if not grounding_corpus:
        return (answer or "").strip()

    facts = extract_grounded_facts(grounding_corpus)
    text = (answer or "").strip()

    try:
        from utils.helpers import _text_asks_customer_care_contact

        contact_turn = _text_asks_customer_care_contact(f"{original_msg} {msg_en}")
    except ImportError:
        contact_turn = False

    if text and answer_has_ungrounded_facts(text, facts, grounding_corpus):
        text = _sanitize_ungrounded_facts(text, facts, grounding_corpus)

    if contact_turn and facts.has_contact_facts():
        if not text or _answer_evades_grounded_contacts(text, facts):
            rebuilt = rebuild_contact_answer_from_corpus(
                grounding_corpus,
                original_msg=original_msg,
                msg_en=msg_en,
            )
            if rebuilt:
                return rebuilt

    try:
        from utils.helpers import message_asks_welfog_social_media

        social_turn = message_asks_welfog_social_media(f"{original_msg} {msg_en}")
    except ImportError:
        social_turn = bool(facts.urls)

    if social_turn and facts.urls:
        if (
            not text
            or _answer_evades_grounded_social(text, facts)
            or answer_has_ungrounded_facts(text, facts, grounding_corpus)
        ):
            try:
                from services.kb_service import format_welfog_social_media_reply_from_kb

                social = format_welfog_social_media_reply_from_kb(
                    original_msg,
                    msg_en,
                    ai_confirmed=True,
                )
                if social:
                    return social
            except ImportError:
                pass

    return text


def ground_kb_llm_response(
    answer: str,
    *,
    kb_context: str = "",
    hits: list[dict[str, Any]] | None = None,
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """Apply fact contract after any KB LLM synthesis path."""
    corpus = (kb_context or _chunk_corpus(hits or [])).strip()
    try:
        from services.chat_flow_telemetry import set_kb_grounding_context

        set_kb_grounding_context(hits or [], corpus=corpus)
    except ImportError:
        pass
    return enforce_kb_fact_grounding(
        answer,
        hits,
        corpus=corpus,
        original_msg=original_msg,
        msg_en=msg_en,
    )


def _answer_missing_required_grounded_facts(
    answer: str,
    facts: GroundedFacts,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> bool:
    """True when a factual KB turn lacks the grounded values present in corpus."""
    if not (answer or "").strip():
        return True
    try:
        from utils.helpers import (
            _text_asks_customer_care_contact,
            message_asks_welfog_social_media,
        )

        comb = f"{original_msg} {msg_en}"
        if _text_asks_customer_care_contact(comb) and facts.has_contact_facts():
            plain = _strip_html(answer).lower()
            has_phone = any(p in _digits_only(plain) for p in facts.phones)
            has_email = any(e in plain for e in facts.emails)
            return not (has_phone or has_email)
        if message_asks_welfog_social_media(comb) and facts.urls:
            return _answer_evades_grounded_social(answer, facts)
    except ImportError:
        pass
    return False


def apply_final_kb_fact_contract(
    answer: str,
    *,
    original_msg: str = "",
    msg_en: str = "",
) -> str:
    """
    Last-line enforcement before customer sees the reply.
    Ensures grounding corpus exists, then prefers structured KB facts over
    any LLM/chitchat answer that conflicts with retrieved knowledge.
    """
    hits: list[dict[str, Any]] = []
    corpus = ""
    try:
        from services.knowledge_answer_service import (
            ensure_kb_grounding_corpus_for_turn,
            try_structured_kb_answer_from_corpus,
        )

        hits, corpus = ensure_kb_grounding_corpus_for_turn(
            original_msg,
            msg_en,
            ai_route=None,
        )
    except ImportError:
        try:
            from services.chat_flow_telemetry import get_kb_grounding_context

            hits, corpus = get_kb_grounding_context()
        except ImportError:
            pass

    if not corpus.strip() and not hits:
        return (answer or "").strip()

    facts = extract_grounded_facts(corpus or _chunk_corpus(hits))
    structured = ""
    try:
        from services.knowledge_answer_service import try_structured_kb_answer_from_corpus

        structured = try_structured_kb_answer_from_corpus(
            corpus or _chunk_corpus(hits),
            hits,
            original_msg,
            msg_en,
        )
    except ImportError:
        pass

    grounded = enforce_kb_fact_grounding(
        answer,
        hits,
        corpus=corpus or _chunk_corpus(hits),
        original_msg=original_msg,
        msg_en=msg_en,
    )

    candidate = (grounded or "").strip() or (answer or "").strip()
    if structured and (
        not candidate
        or answer_has_ungrounded_facts(candidate, facts, corpus or _chunk_corpus(hits))
        or _answer_evades_grounded_contacts(candidate, facts)
        or _answer_evades_grounded_social(candidate, facts)
        or _answer_missing_required_grounded_facts(
            candidate,
            facts,
            original_msg=original_msg,
            msg_en=msg_en,
        )
    ):
        return structured

    try:
        from services.chat_flow_telemetry import is_authoritative_kb_route_locked

        if is_authoritative_kb_route_locked() and structured:
            if not candidate or answer_has_ungrounded_facts(
                candidate, facts, corpus or _chunk_corpus(hits)
            ):
                return structured
    except ImportError:
        pass

    return candidate


# Backward-compatible contact-only helpers -------------------------------------

def answer_contains_ungrounded_contacts(answer: str, hits: list[dict[str, Any]]) -> bool:
    corpus = _chunk_corpus(hits)
    if not corpus.strip():
        return False
    facts = extract_grounded_facts(corpus)
    return answer_has_ungrounded_facts(answer, facts, corpus)


def enforce_grounded_kb_answer(
    answer: str,
    hits: list[dict[str, Any]],
    *,
    original_msg: str = "",
    msg_en: str = "",
    reply_lang: str = "",
) -> str:
    """Legacy entry — delegates to immutable fact contract."""
    return enforce_kb_fact_grounding(
        answer,
        hits,
        original_msg=original_msg,
        msg_en=msg_en,
    )
