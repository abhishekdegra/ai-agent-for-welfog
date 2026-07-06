"""Quick mixed-turn bench — API / KB / OOD / chitchat via live /chat."""
from __future__ import annotations

import json
import sys
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
UID = "1167"
TIMEOUT = 95.0

TESTS: list[dict] = [
    # --- one per major API / route ---
    {"cat": "API:product", "msg": "bhai redmi ka mobile cover dikha de", "expect": ["wf-product", "product"]},
    {"cat": "API:pincode", "msg": "302012 pincode pe welfog delivery karta hai kya", "expect": ["wf-pin", "pincode", "delivery"]},
    {"cat": "API:order_history", "msg": "mere purchased orders ka data dikha de", "expect": ["wf-ph", "order", "purchase"]},
    {"cat": "API:wishlist", "msg": "meri wishlist dikhao welfog pe", "expect": ["wishlist", "wf-wl", "saved"]},
    {"cat": "API:deals", "msg": "aaj ki best deals dikhao welfog pe", "expect": ["deal", "wf-deal", "offer"]},
    {"cat": "API:categories", "msg": "welfog pe categories kitni h", "expect": ["categor", "id:"]},
    {"cat": "API:category_feed", "msg": "home page pe category wise products dikhao", "expect": ["categor", "product", "wf-"]},
    # --- KB (5) mixed language/style ---
    {"cat": "KB:refund", "msg": "refund kitne din me milta hai bhai", "expect": ["refund", "5", "7", "business"]},
    {"cat": "KB:payment", "msg": "What payment methods does Welfog accept?", "expect": ["upi", "card", "payment", "cod"]},
    {"cat": "KB:seller", "msg": "welfog pe seller account banane ki steps bta na", "expect": ["seller", "otp", "register", "portal"]},
    {"cat": "KB:company", "msg": "welfog kya hai company ke baare me btao", "expect": ["welfog", "marketplace", "commerce", "platform"]},
    {"cat": "KB:support", "msg": "customer care number kya hai welfog ka", "expect": ["9828", "support", "info@", "care"]},
    # --- OOD (5) ---
    {"cat": "OOD", "msg": "bhai meri girlfriend bnwa de yrrr", "expect": ["welfog", "shop", "order", "help", "sorry", "can't", "cannot", "nahi"]},
    {"cat": "OOD", "msg": "bhia baarish kb aayegi delhi me", "expect": ["welfog", "shop", "order", "weather", "sorry", "can't", "cannot"]},
    {"cat": "OOD", "msg": "mereko bhukh lag rhi zepto ya blinkit se order kru", "expect": ["welfog", "sorry", "can't", "cannot", "shop"]},
    {"cat": "OOD", "msg": "can you build a building for me", "expect": ["welfog", "sorry", "can't", "cannot", "shop", "help"]},
    {"cat": "OOD", "msg": "bhai mujhe bike chalana sikha de", "expect": ["welfog", "sorry", "can't", "cannot", "shop"]},
    # --- Chitchat (5) ---
    {"cat": "CHITCHAT", "msg": "hi", "expect": ["hello", "hi", "help", "welfog", "namaste"]},
    {"cat": "CHITCHAT", "msg": "hewillooo darling", "expect": ["hello", "hi", "hey", "welfog", "help"]},
    {"cat": "CHITCHAT", "msg": "thank you bhai bahut help ho gayi", "expect": ["welcome", "thank", "glad", "happy", "anytime", "dhany"]},
    {"cat": "CHITCHAT", "msg": "tu kaise hai yaar", "expect": ["help", "welfog", "here", "good", "fine", "ready"]},
    {"cat": "CHITCHAT", "msg": "bye bye see you", "expect": ["bye", "soon", "back", "welcome", "take care"]},
]


def run_one(case: dict) -> dict:
    t0 = time.perf_counter()
    err = ""
    status = 0
    body = ""
    try:
        r = requests.post(
            f"{BASE}/chat?user_id={UID}",
            json={"message": case["msg"], "chat_id": uuid.uuid4().hex},
            timeout=TIMEOUT,
        )
        status = r.status_code
        data = r.json()
        body = (data.get("data") or "").strip()
    except requests.exceptions.Timeout:
        err = "TIMEOUT"
    except Exception as exc:
        err = str(exc)[:80]
    sec = round(time.perf_counter() - t0, 2)
    low = body.lower()
    if err:
        ok = False
        note = err
    elif status != 200 or not body:
        ok = False
        note = f"HTTP {status}" if status != 200 else "empty"
    elif "taking longer" in low or "something went wrong" in low:
        ok = False
        note = "timeout_msg"
    else:
        hits = [e for e in case.get("expect", []) if e.lower() in low]
        ok = bool(hits)
        note = f"match:{','.join(hits[:3])}" if hits else "weak_match"
        if case["cat"].startswith("OOD") and any(
            x in low for x in ("wf-product", "wf-ph-card", "refund policy", "return policy")
        ):
            ok = False
            note = "wrong_kb_or_api"
    return {
        "cat": case["cat"],
        "msg": case["msg"][:55],
        "sec": sec,
        "ok": ok,
        "note": note,
        "len": len(body),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        requests.get(f"{BASE}/health", timeout=5)
    except Exception:
        print("ERROR: server not reachable at", BASE)
        sys.exit(1)

    results = []
    for case in TESTS:
        row = run_one(case)
        results.append(row)
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"[{mark}] {row['sec']:>5}s | {row['cat']:<18} | {row['note']:<22} | {row['msg']}")

    by_cat: dict[str, list] = {}
    for r in results:
        key = r["cat"].split(":")[0]
        by_cat.setdefault(key, []).append(r)

    print("\n=== SUMMARY ===")
    for key, rows in by_cat.items():
        passed = sum(1 for r in rows if r["ok"])
        avg = round(sum(r["sec"] for r in rows) / len(rows), 2)
        mx = max(r["sec"] for r in rows)
        print(f"{key}: {passed}/{len(rows)} pass | avg {avg}s | max {mx}s")

    out = BASE.replace("http://127.0.0.1:5000", "tests") + "/bench_mixed_results.json"
    # save next to script
    from pathlib import Path

    out_path = Path(__file__).with_name("bench_mixed_results.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
