"""Quick /chat bench — latency + reply preview for mixed intents/languages."""
from __future__ import annotations

import json
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
USER_ID = "1167"
TIMEOUT = 95

CASES = [
    ("greeting", "heeeellllo bhai", "hinglish"),
    ("product", "iphone dikha de mujhe", "hinglish"),
    ("kb_refund", "welfog refund policy kya hai", "hinglish"),
    ("pincode", "302012 pe delivery milegi kya", "hinglish"),
    ("wishlist", "meri wishlist dikhao", "hinglish"),
    ("order_history", "mere past orders dikhao", "hinglish"),
    ("deals", "aaj ke best deals batao", "hinglish"),
    ("about", "welfog company ke baare me batao", "hinglish"),
    ("categories", "categories list dikhao", "en"),
    ("ood", "meri gf banwade", "hinglish"),
    ("product_en", "show me white shirts under 500", "en"),
]


def strip_html(s: str) -> str:
    import re

    t = re.sub(r"<[^>]+>", " ", s or "")
    return " ".join(t.split())[:220]


def main() -> None:
    rows = []
    for label, msg, _lang in CASES:
        chat_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        err = ""
        preview = ""
        ok = False
        try:
            r = requests.post(
                f"{BASE}/chat",
                params={"user_id": USER_ID},
                json={"message": msg, "chat_id": chat_id},
                timeout=TIMEOUT,
            )
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
            else:
                data = r.json()
                body = data.get("data") or data.get("cards_html") or ""
                preview = strip_html(body if isinstance(body, str) else str(body))
                ok = bool(preview) and "taking longer than usual" not in preview.lower()
                if data.get("degraded"):
                    err = "degraded"
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            err = type(exc).__name__ + ": " + str(exc)[:80]
        rows.append((label, msg, elapsed, ok, err, preview))
        print(
            f"[{label}] {elapsed:5.1f}s ok={ok} err={err or '-'} | {preview[:120]}",
            flush=True,
        )

    slow = [r for r in rows if r[2] > 25]
    bad = [r for r in rows if not r[3]]
    print("\n=== SUMMARY ===")
    print(f"total={len(rows)} ok={len(rows)-len(bad)} slow(>25s)={len(slow)} fail={len(bad)}")
    if bad:
        for r in bad:
            print(f"  FAIL {r[0]}: {r[4] or 'empty'}")
    if slow:
        for r in slow:
            print(f"  SLOW {r[0]}: {r[2]:.1f}s")


if __name__ == "__main__":
    main()
