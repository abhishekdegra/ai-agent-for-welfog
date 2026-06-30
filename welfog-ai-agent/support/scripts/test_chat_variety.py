"""AI-first routing — varied languages/styles (no fixed keyword list dependency)."""
from __future__ import annotations

import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
UID = "1167"
MAX_SEC = 45.0
GAP = 3.0

TESTS = [
    ("ood", "yaar mujhe shaadi ke liye ladki chahiye"),
    ("ood", "can u do my homework pls"),
    ("chitchat", "ram ram bhai kaise ho"),
    ("kb", "vapasi policy batao welfog ki"),
    ("wishlist", "jo products maine pasand kiye hain dikhao"),
    ("kb", "delivery me kitna time lagta hai"),
    ("product", "mujhe black jeans chahiye"),
]

def main() -> None:
    for kind, msg in TESTS:
        t0 = time.perf_counter()
        err = ""
        body = ""
        try:
            r = requests.post(
                f"{BASE}/chat?user_id={UID}",
                json={"message": msg, "chat_id": uuid.uuid4().hex},
                timeout=MAX_SEC,
            )
            data = r.json()
            body = (data.get("data") or "").replace("\n", " ")[:200]
        except Exception as exc:
            err = str(exc)
        dt = time.perf_counter() - t0
        ok = dt <= 15 and body and not err
        print(f"[{'PASS' if ok else 'FAIL'}] {kind:8} {dt:4.1f}s | {msg!r}")
        if err:
            print(f"         ERR: {err}")
        else:
            print(f"         -> {body[:160]}")
        time.sleep(GAP)


if __name__ == "__main__":
    main()
