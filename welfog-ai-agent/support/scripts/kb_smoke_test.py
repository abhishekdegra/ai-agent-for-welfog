"""Smoke-test KB retrieval + answers (no full chat router)."""
from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

from services.kb_service import (
    format_dynamic_kb_answer,
    refresh_knowledge_cache,
    resolve_best_faq_chunk_for_question,
    resolve_kb_keys_for_question,
)

CASES = [
    {
        "q": "refund kitne din mein aata hai?",
        "en": "how long does refund take",
        "lang": "hinglish",
        "expect_files": ("refund", "faqs"),
        "expect_in_answer": ("5", "7", "refund", "business", "din"),
    },
    {
        "q": "What is Welfog's return policy?",
        "en": "",
        "lang": "en",
        "expect_files": ("faqs", "refund"),
        "expect_in_answer": ("5", "return", "day"),
    },
    {
        "q": "privacy policy kya hai data safe hai?",
        "en": "is my data safe privacy policy",
        "lang": "hinglish",
        "expect_files": ("privacy",),
        "expect_in_answer": ("privacy", "data", "personal"),
    },
    {
        "q": "seller account kaise banaye?",
        "en": "how to create seller account",
        "lang": "hinglish",
        "expect_files": ("seller",),
        "expect_in_answer": ("seller", "create", "register", "portal", "ban"),
    },
    {
        "q": "customer care number kya hai?",
        "en": "customer care contact number",
        "lang": "hinglish",
        "expect_files": ("support", "company", "faqs"),
        "expect_in_answer": ("9828", "support", "contact", "care", "email", "@"),
    },
    {
        "q": "shipping kitne din mein deliver hoti hai?",
        "en": "how long does shipping delivery take",
        "lang": "hinglish",
        "expect_files": ("shipping", "faqs"),
        "expect_in_answer": ("deliver", "ship", "day", "location"),
    },
    {
        "q": "What payment methods does Welfog accept?",
        "en": "",
        "lang": "en",
        "expect_files": ("payment", "faqs"),
        "expect_in_answer": ("payment", "upi", "card", "cod", "cash"),
    },
]


def _plain(html: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", html or "")
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).lower().strip()


def main() -> int:
    print("Loading KB index...")
    refresh_knowledge_cache(build_vectors=True)
    failed = 0

    for i, case in enumerate(CASES, 1):
        q, en, lang = case["q"], case["en"], case["lang"]
        print(f"\n{'='*60}\n[{i}] Q: {q}")
        if en:
            print(f"    EN: {en}")

        keys = resolve_kb_keys_for_question(q, en)
        faq = resolve_best_faq_chunk_for_question(q, en)
        answer = format_dynamic_kb_answer(q, en, reply_lang=lang)

        print(f"    files: {keys[:3]}")
        if faq:
            print(f"    faq: {_plain(faq.get('chunk', ''))[:100]}...")

        ok_files = any(any(exp in k for exp in case["expect_files"]) for k in keys[:3])
        plain_ans = _plain(answer)
        ok_ans = any(w in plain_ans for w in case["expect_in_answer"]) if answer else False

        if not ok_files:
            print(f"    FAIL file routing (expected one of {case['expect_files']})")
            failed += 1
        if not answer or len(plain_ans) < 35:
            print("    FAIL empty or too short answer")
            failed += 1
        elif not ok_ans:
            print(f"    FAIL answer off-topic (need one of {case['expect_in_answer']})")
            print(f"    got: {plain_ans[:220]}...")
            failed += 1
        else:
            print(f"    OK: {plain_ans[:200]}...")

    print(f"\n{'='*60}\nDone. Failures: {failed}/{len(CASES)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
