"""Probe KB vector-only path — multilingual, latency, no full /chat stack."""
from __future__ import annotations

import re
import sys
import time

sys.path.insert(0, ".")

CASES = [
    {
        "q": "refund kitne din mein aata hai?",
        "lang_hint": "hinglish",
        "expect_any": ("refund", "5", "7", "din", "business"),
    },
    {
        "q": "What is Welfog's return policy?",
        "lang_hint": "en",
        "expect_any": ("return", "5", "7", "day", "refund"),
    },
    {
        "q": "वेलफोग पर सेलर अकाउंट कैसे बनाएं?",
        "lang_hint": "hi",
        "expect_any": ("seller", "सेलर", "register", "account", "बन"),
    },
    {
        "q": "customer care number kya hai?",
        "lang_hint": "hinglish",
        "expect_any": ("9828", "support", "contact", "care", "email", "@"),
    },
    {
        "q": "Welfog accepts which payment methods?",
        "lang_hint": "en",
        "expect_any": ("payment", "upi", "card", "cod", "cash"),
    },
    {
        "q": "shipping kitne din mein deliver hoti hai?",
        "lang_hint": "hinglish",
        "expect_any": ("deliver", "ship", "day", "din", "location"),
    },
]


def _plain(html: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", html or "")
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    from services.kb_service import refresh_knowledge_cache

    print("Warming KB index + embeddings...")
    t0 = time.perf_counter()
    from services.kb_service import ensure_kb_vectors, ensure_knowledge_cache_fresh

    ensure_knowledge_cache_fresh()
    ensure_kb_vectors()
    print(f"Index ready in {(time.perf_counter() - t0):.1f}s\n")

    from services.knowledge_query_pipeline import try_knowledge_vector_only_reply

    failed = 0
    slow = 0
    for i, case in enumerate(CASES, 1):
        q = case["q"]
        try:
            from services.translation_service import to_en_for_routing

            msg_en = to_en_for_routing(q, case.get("lang_hint") or "")
        except ImportError:
            msg_en = q
        t1 = time.perf_counter()
        body = try_knowledge_vector_only_reply(q, msg_en, "", reply_lang=case["lang_hint"])
        ms = (time.perf_counter() - t1) * 1000.0
        plain = _plain(body or "").lower()
        ok = bool(body and body.strip())
        if ok:
            hit = any(x.lower() in plain for x in case["expect_any"])
        else:
            hit = False
        status = "OK" if ok and hit else ("SLOW/EMPTY" if not ok else "WEAK")
        if not ok or not hit:
            failed += 1
        if ms > 3000:
            slow += 1
            status += f" SLOW({ms:.0f}ms)"
        else:
            status += f" ({ms:.0f}ms)"
        print(f"[{i}] {status}")
        print(f"    Q: {q[:70]}")
        print(f"    A: {plain[:180] or '(no answer)'}\n")

    print(f"Summary: {len(CASES) - failed}/{len(CASES)} good, {slow} slow (>3s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
