"""Sequential API smoke — one request at a time (Flask dev server is single-threaded)."""
from __future__ import annotations

import json
import re
import sys
import time
import uuid

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5000"
UID = "1167"
TIMEOUT = 90.0
GAP = 3.0

TESTS = [
    ("deals", "bhai aaj ki top deals bata de", ["deal", "offer", "₹", "view product", "today"]),
    ("categories", "welfog ki kitni categories hain list dikha", ["categor", "fashion", "electronic", "beauty"]),
    ("wishlist", "mere like kiye hue products dikha", ["wishlist", "saved", "product", "view", "item"]),
    ("product_en", "show me mobile cover under rs 300", ["mobile", "cover", "₹", "view product"]),
    ("product_hi", "sun redmi ka mobile cover dikha", ["mobile", "cover", "redmi", "₹"]),
    ("product_ta", "வணக்கம் Infinix மொபைலுக்கு cover வேண்டும்", ["infinix", "cover", "mobile", "₹", "product"]),
    ("pincode", "302034 pe delivery hoti hai kya", ["302034", "deliver", "service", "pin"]),
    ("pincode_city", "kya welfog jaipur me deliver karta hai", ["jaipur", "deliver", "pin", "service"]),
    ("order_history", "meri saari orders chat me dikha", ["order", "purchase", "history", "id"]),
    ("kb_refund", "refund kitne din me milta hai", ["refund", "day", "return"]),
    ("kb_shipping", "delivery kitne din lagti hai", ["deliver", "day", "ship"]),
    ("kb_company", "welfog company kya karti hai", ["welfog", "marketplace", "shop", "platform"]),
    ("chitchat", "hello bhai kaise ho", ["welfog", "help", "hello", "hi", "assist"]),
]


def main() -> None:
    results = []
    print(f"Sequential smoke -> {BASE}  (timeout={TIMEOUT}s, gap={GAP}s)\n")
    for api, msg, hints in TESTS:
        t0 = time.perf_counter()
        err = ""
        body = ""
        st = 0
        try:
            r = requests.post(
                f"{BASE}/chat?user_id={UID}",
                json={"message": msg, "chat_id": uuid.uuid4().hex},
                timeout=TIMEOUT,
            )
            st = r.status_code
            body = (r.json().get("data") or "").strip()
        except Exception as exc:
            err = str(exc)[:100]
        sec = round(time.perf_counter() - t0, 1)
        low = body.lower()
        hits = [h for h in hints if h.lower() in low]
        bad = ""
        if "taking longer" in low or "load hai" in low:
            bad = "busy/timeout"
        elif "supplier legal" in low or "legal metrology" in low:
            bad = "wrong_kb"
        elif not body.strip():
            bad = "empty"
        elif not hits:
            bad = "irrelevant"
        ok = st == 200 and not bad and sec <= 25.0
        prev = re.sub(r"\s+", " ", body)[:160]
        row = {"api": api, "sec": sec, "ok": ok, "bad": bad, "hits": hits, "preview": prev, "err": err}
        results.append(row)
        flag = "OK" if ok else "FAIL"
        print(f"[{flag}] {api:14} {sec:5.1f}s  hits={len(hits)}/{len(hints)}  {bad or err or ''}")
        print(f"       Q: {msg[:65]}")
        print(f"       A: {prev}")
        print()
        time.sleep(GAP)

    ok_n = sum(1 for r in results if r["ok"])
    print("=" * 55)
    print(f"PASS {ok_n}/{len(results)}")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['api']}: {r['bad'] or r['err'] or 'slow'} ({r['sec']}s)")
    with open("scripts/api_smoke_sequential_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
