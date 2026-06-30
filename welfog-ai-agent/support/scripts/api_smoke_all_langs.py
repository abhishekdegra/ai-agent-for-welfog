"""One question per API route — mixed languages/styles."""
from __future__ import annotations

import json
import re
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
UID = "1167"
MAX_SEC = 50.0
GAP_SEC = 2.0

# (api_label, message, expect_hints — substrings that should appear in response)
TESTS = [
    (
        "deals_api",
        "bhai aaj ki top deals bata de",
        ["deal", "offer", "₹", "product", "view"],
    ),
    (
        "categories_api",
        "welfog ki kitni categories hain list dikha",
        ["categor", "fashion", "electronic", "beauty", "home"],
    ),
    (
        "wishlist_api",
        "mere like kiye hue products ki list dikha bhai",
        ["wishlist", "saved", "like", "product", "empty", "view", "कोई", "नहीं"],
    ),
    (
        "product_en",
        "show me mobile cover under rs 300",
        ["mobile", "cover", "₹", "view product", "product"],
    ),
    (
        "product_hinglish",
        "sun redmi ka mobile cover dikha",
        ["mobile", "cover", "redmi", "₹", "view"],
    ),
    (
        "product_tamil",
        "வணக்கம், எனது Infinix மொபைலுக்கு ஒரு கவர் வேண்டும்",
        ["infinix", "cover", "mobile", "₹", "view", "product"],
    ),
    (
        "pincode_pin",
        "302034 pe delivery hoti hai kya",
        ["302034", "deliver", "service", "yes", "no", "pin", "हाँ", "नहीं"],
    ),
    (
        "pincode_city",
        "does welfog deliver to jaipur",
        ["jaipur", "deliver", "pin", "302", "service", "yes", "no"],
    ),
    (
        "order_history",
        "meri saari orders dikha do chat me",
        ["order", "purchase", "history", "id", "empty", "कोई"],
    ),
    (
        "kb_refund",
        "refund policy kya hai welfog ki",
        ["refund", "return", "day", "policy", "order"],
    ),
    (
        "kb_shipping",
        "how many days for delivery on welfog",
        ["deliver", "day", "ship", "order"],
    ),
    (
        "kb_company",
        "welfog kya hai company ke baare me bata",
        ["welfog", "marketplace", "shop", "india", "platform"],
    ),
    (
        "chitchat",
        "hello bhai kaise ho",
        ["welfog", "help", "hello", "hi", "namaste", "assist"],
    ),
]


def score_response(hints: list[str], body: str) -> tuple[int, list[str]]:
    low = body.lower()
    hits = [h for h in hints if h.lower() in low]
    return len(hits), hits


def classify_fail(body: str, err: str) -> str:
    if err:
        return "error"
    if not body.strip():
        return "empty"
    low = body.lower()
    if "taking longer" in low or "server_busy" in low or "load hai" in low:
        return "timeout/busy"
    if "supplier legal" in low or "legal metrology" in low:
        return "wrong_kb"
    if "pin code" in low and "6-digit" in low and "delivery check" in low:
        return "wrong_pincode_prompt"
    return ""


def main() -> None:
    results = []
    print(f"API smoke test -> {BASE}/chat user_id={UID}\n")
    for label, msg, hints in TESTS:
        chat_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        err = ""
        body = ""
        status = 0
        try:
            r = requests.post(
                f"{BASE}/chat?user_id={UID}",
                json={"message": msg, "chat_id": chat_id},
                timeout=MAX_SEC,
            )
            status = r.status_code
            data = r.json()
            body = (data.get("data") or "").strip()
        except Exception as exc:
            err = str(exc)[:120]
        dt = time.perf_counter() - t0
        hit_n, hits = score_response(hints, body)
        fail_kind = classify_fail(body, err)
        ok = (
            not fail_kind
            and status == 200
            and hit_n >= 1
            and dt <= 20.0
        )
        preview = re.sub(r"\s+", " ", body)[:200]
        row = {
            "api": label,
            "msg": msg,
            "sec": round(dt, 2),
            "status": status,
            "ok": ok,
            "fail_kind": fail_kind,
            "hint_hits": hits,
            "preview": preview,
            "err": err,
        }
        results.append(row)
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {label:18} {dt:5.1f}s  hits={hit_n}/{len(hints)}")
        print(f"       Q: {msg[:70]}")
        if err:
            print(f"       ERR: {err}")
        elif fail_kind:
            print(f"       ISSUE: {fail_kind}")
        print(f"       -> {preview[:160]}")
        print()
        time.sleep(GAP_SEC)

    passed = sum(1 for r in results if r["ok"])
    slow = [r for r in results if r["sec"] > 20]
    fails = [r for r in results if not r["ok"]]
    print("=" * 60)
    print(f"PASS {passed}/{len(results)}  |  slow(>20s)={len(slow)}  |  fail={len(fails)}")
    if fails:
        print("\nFailed:")
        for r in fails:
            print(f"  - {r['api']}: {r['fail_kind'] or 'low relevance'} ({r['sec']}s)")
    out = "scripts/api_smoke_all_langs_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
