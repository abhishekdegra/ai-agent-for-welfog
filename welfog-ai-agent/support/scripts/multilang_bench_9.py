"""9-turn multilingual/style bench for production /chat."""
from __future__ import annotations

import json
import re
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
USER_ID = "1167"
TIMEOUT = 120

CASES = [
    {
        "id": "hinglish_product",
        "msg": "bhai mujhe trimmer dikha de sasta wala",
        "lang": "hinglish",
        "expect_any": ("trimmer", "product", "₹", "rs", "search", "pincode", "delivery"),
        "reject_any": ("grievance", "complaint", "taking longer"),
    },
    {
        "id": "en_kb_refund",
        "msg": "How long does Welfog take to process refunds?",
        "lang": "en",
        "expect_any": ("refund", "5", "7", "day", "business"),
        "reject_any": ("taking longer", "trimmer", "product card"),
    },
    {
        "id": "hi_about",
        "msg": "वेलफोग क्या है?",
        "lang": "hi",
        "expect_any": ("welfog", "वेलफोग", "marketplace", "e-commerce", "shop", "platform", "company"),
        "reject_any": ("taking longer", "how can i help you today", "looking for something to shop"),
    },
    {
        "id": "hinglish_pincode",
        "msg": "302041 pe delivery hoti hai kya?",
        "lang": "hinglish",
        "expect_any": ("302041", "deliver", "delivery", "service", "pincode", "available"),
        "reject_any": ("order id", "taking longer"),
    },
    {
        "id": "marathi_support",
        "msg": "customer care cha number sang",
        "lang": "mr",
        "expect_any": ("9828", "support", "contact", "care", "email", "@", "help"),
        "reject_any": ("taking longer",),
    },
    {
        "id": "en_product_budget",
        "msg": "show me mobile covers under 200",
        "lang": "en",
        "expect_any": ("cover", "mobile", "₹", "rs", "product", "200", "pincode"),
        "reject_any": ("grievance", "taking longer"),
    },
    {
        "id": "hinglish_cod",
        "msg": "cash on delivery accept karta hai kya welfog?",
        "lang": "hinglish",
        "expect_any": ("cod", "cash", "delivery", "payment", "upi", "accept"),
        "reject_any": ("taking longer", "trimmer"),
    },
    {
        "id": "hinglish_orders",
        "msg": "meri purani orders dikhao na",
        "lang": "hinglish",
        "expect_any": ("order", "history", "past", "recent", "login", "sign"),
        "reject_any": ("taking longer",),
    },
    {
        "id": "informal_greeting",
        "msg": "heeeellllo bhai kya haal",
        "lang": "hinglish",
        "expect_any": ("hello", "hi", "welfog", "help", "namaste", "haal"),
        "reject_any": ("taking longer",),
    },
]


def strip_html(s: str) -> str:
    t = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(t.split())


def score_case(case: dict, preview: str, degraded: bool) -> tuple[bool, str]:
    low = preview.lower()
    if degraded or "taking longer than usual" in low:
        return False, "timeout/degraded"
    if not preview.strip():
        return False, "empty"
    for bad in case.get("reject_any") or ():
        if bad.lower() in low:
            return False, f"reject:{bad}"
    hits = [x for x in case.get("expect_any") or () if x.lower() in low]
    if hits:
        return True, f"hit:{hits[0]}"
    return False, "no_expect_match"


def main() -> None:
    rows = []
    print("=== 9-turn multilingual bench ===\n", flush=True)
    for case in CASES:
        chat_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        err = ""
        preview = ""
        ok = False
        reason = ""
        degraded = False
        try:
            r = requests.post(
                f"{BASE}/chat",
                params={"user_id": USER_ID},
                json={"message": case["msg"], "chat_id": chat_id},
                timeout=TIMEOUT,
            )
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
            else:
                data = r.json()
                degraded = bool(data.get("degraded"))
                body = data.get("data") or data.get("cards_html") or ""
                preview = strip_html(body if isinstance(body, str) else str(body))
                ok, reason = score_case(case, preview, degraded)
                if not ok and not err:
                    err = reason
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err = type(exc).__name__ + ": " + str(exc)[:100]
        rows.append((case["id"], case["lang"], elapsed, ok, err, preview[:180]))
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {case['id']:20} {elapsed:5.1f}s ({case['lang']}) | {preview[:100]}"
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        if err and not ok:
            print(f"         reason: {err}", flush=True)
        time.sleep(0.5)

    times = [r[2] for r in rows if r[3]]
    slow = [r for r in rows if r[2] > 15]
    bad = [r for r in rows if not r[3]]
    print("\n=== SUMMARY ===")
    print(f"pass={len(rows)-len(bad)}/{len(rows)}  slow(>15s)={len(slow)}")
    if times:
        print(f"avg_pass_latency={sum(times)/len(times):.1f}s  max={max(times):.1f}s")
    for r in bad:
        print(f"  FAIL {r[0]} ({r[1]}): {r[4]}")


if __name__ == "__main__":
    main()
