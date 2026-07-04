"""Quick 7-query route + latency check (sequential, unique chat_id per turn)."""
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
TIMEOUT = 45.0
GAP = 2.0
MAX_OK_SEC = 20.0

TESTS = [
    (
        "account_list",
        "mere purchased orders ka data dikha de",
        ["wf-ph", "order", "history", "purchase"],
        ["pincode", "6-digit", "delivery check", "mobile cover", "refund policy"],
    ),
    (
        "product_hinglish",
        "bhai redmi ka mobile cover dikha",
        ["cover", "mobile", "₹", "product", "view"],
        ["order history", "wf-ph-title", "pincode"],
    ),
    (
        "delivery_pin",
        "302012 pe welfog delivery karta hai kya",
        ["302012", "deliver", "pin", "service"],
        ["order history", "wf-ph"],
    ),
    (
        "kb_refund",
        "refund kitne din me milta hai welfog pe",
        ["refund", "day", "return", "policy"],
        ["wf-ph", "302012"],
    ),
    (
        "order_english",
        "show my past orders here in chat please",
        ["wf-ph", "order", "history"],
        ["pincode", "6-digit pin"],
    ),
    (
        "product_english",
        "find nike shoes on welfog",
        ["shoe", "nike", "₹", "product", "view"],
        ["order history", "pincode"],
    ),
    (
        "deals",
        "aaj ki best deals dikhao welfog pe",
        ["deal", "offer", "₹", "view", "today"],
        ["pincode", "order history"],
    ),
]


def score(body: str, must: list[str], reject: list[str]) -> tuple[bool, str]:
    low = (body or "").lower()
    if not body.strip():
        return False, "empty"
    if "taking longer" in low or "still being processed" in low:
        return False, "timeout_msg"
    if "12s cap" in low:
        return False, "stale_cap"
    for r in reject:
        if r.lower() in low:
            return False, f"wrong_route:{r}"
    hits = sum(1 for h in must if h.lower() in low)
    if hits < max(1, len(must) // 2):
        return False, f"weak_hits:{hits}/{len(must)}"
    return True, f"hits:{hits}/{len(must)}"


def main() -> None:
    try:
        requests.get(f"{BASE}/?user_id={UID}", timeout=5)
    except Exception as exc:
        print(f"SERVER DOWN: {exc}")
        sys.exit(1)

    print(f"Quick route bench -> {BASE} (max_ok={MAX_OK_SEC}s)\n")
    rows = []
    for api, msg, must, reject in TESTS:
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
            err = str(exc)[:120]
        sec = round(time.perf_counter() - t0, 2)
        ok, note = score(body, must, reject) if not err else (False, err)
        if ok and sec > MAX_OK_SEC:
            ok = False
            note = f"slow:{sec}s"
        prev = re.sub(r"\s+", " ", body)[:140]
        rows.append({"api": api, "sec": sec, "ok": ok, "note": note, "preview": prev})
        flag = "OK" if ok else "FAIL"
        print(f"[{flag}] {api:18} {sec:6.2f}s  {note}")
        print(f"       Q: {msg}")
        print(f"       A: {prev}")
        print()
        time.sleep(GAP)

    ok_n = sum(1 for r in rows if r["ok"])
    print("=" * 60)
    print(f"PASS {ok_n}/{len(rows)}")
    slow = [r for r in rows if r["sec"] > MAX_OK_SEC]
    if slow:
        print(f"SLOW (>{MAX_OK_SEC}s): {[r['api'] for r in slow]}")
    out = "scripts/quick_route_bench_7_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
