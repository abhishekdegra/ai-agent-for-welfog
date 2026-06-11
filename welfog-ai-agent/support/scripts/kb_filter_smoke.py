"""Quick checks: KB answers must not echo questions or leak metadata labels."""
from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")

from services.kb_service import (
    format_dynamic_kb_answer,
    format_kb_answer_from_brain_keys,
    refresh_knowledge_cache,
)

CASES = [
    ("What is Welfog's return policy?", "", "en", ["faqs", "refund"]),
    ("What payment methods does Welfog accept?", "", "en", ["payment", "faqs"]),
    ("privacy policy kya hai data safe hai?", "is my data safe", "hinglish", ["privacy", "faqs"]),
    ("customer care number kya hai?", "customer care number", "hinglish", ["support"]),
    ("refund kitne din mein aata hai?", "how long refund", "hinglish", ["refund", "faqs"]),
    ("Does Welfog have a mobile app?", "", "en", ["faqs"]),
]


def plain(html: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", html or "")
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_question_echo(q: str, ans: str) -> bool:
    ql = q.lower().rstrip("?").strip()
    al = plain(ans).lower()
    if al.startswith(ql):
        return True
    if f"{ql}:" in al[: len(ql) + 20]:
        return True
    if re.search(r"[\u0900-\u097F]", al):
        return True
    if "welfog customer support, customer care" in al:
        return True
    if re.search(r"\?:\s*$", al):
        return True
    return False


def main() -> int:
    refresh_knowledge_cache(build_vectors=True)
    failed = 0
    for q, en, lang, keys in CASES:
        a1 = format_dynamic_kb_answer(q, en, reply_lang=lang)
        a2 = format_kb_answer_from_brain_keys(q, en, keys, reply_lang=lang)
        for label, ans in (("dynamic", a1), ("brain_keys", a2)):
            p = plain(ans)
            bad = not ans or has_question_echo(q, ans)
            status = "FAIL" if bad else "OK"
            if bad:
                failed += 1
            print(f"\n[{status}] {label} | {q[:50]}")
            print(f"  {p[:180]}...")
    print(f"\nFailures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
