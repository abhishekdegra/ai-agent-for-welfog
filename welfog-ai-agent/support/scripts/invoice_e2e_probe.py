"""One-off E2E probe: invoice routing via POST /chat (fresh chat_id per message)."""
from __future__ import annotations

import re
import sys
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000/chat"
USER_ID = "1167"

CASES: list[tuple[str, str]] = [
    ("hinglish_bill", "2605150 iska bill dede"),
    ("hinglish_involve", "are involve maang rha hu yrr 2605150 is order id ka"),
    ("english_receipt", "2605150 i need recipt for this order"),
    ("english_invoice", "can you give me GST invoice for order 2605270"),
    ("hinglish_chalan", "2606020 ka chalan bhejo mujhe"),
]

INVOICE_MARKERS = (
    "your invoice",
    "download invoice",
    "invoice",
    "bill",
    "receipt",
    "gst",
)
TRACK_MARKERS = (
    "live status",
    "order steps",
    "expected delivery",
    "order ka live status",
    "current status",
    "courier",
)


def classify(html: str) -> str:
    low = (html or "").lower()
    has_inv = any(m in low for m in INVOICE_MARKERS) and "download invoice" in low or (
        "invoice" in low and "download" in low
    )
    has_track = any(m in low for m in TRACK_MARKERS) and "download invoice" not in low
    if has_inv and not has_track:
        return "invoice_ok"
    if has_track and not has_inv:
        return "track_wrong"
    if has_inv and has_track:
        return "mixed"
    return "unknown"


def main() -> int:
    print("=== Invoice E2E probe ===")
    fails = 0
    for label, msg in CASES:
        chat_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                BASE,
                json={"message": msg, "user_id": USER_ID, "chat_id": chat_id},
                timeout=120,
            )
            dt = time.perf_counter() - t0
            resp.raise_for_status()
            payload = resp.json()
            body = str(payload.get("data") or "")
            verdict = classify(body)
            oid = re.search(r"\b(2605150|2605270|2606020)\b", body)
            print(f"\n[{label}] {dt:.2f}s verdict={verdict} order_in_body={bool(oid)}")
            print(f"  msg: {msg!r}")
            snippet = re.sub(r"\s+", " ", body)[:220]
            print(f"  snippet: {snippet}")
            if verdict != "invoice_ok":
                fails += 1
            time.sleep(2)
        except Exception as exc:
            fails += 1
            print(f"\n[{label}] ERROR: {exc}")
            print(f"  msg: {msg!r}")
    print(f"\n=== done: {len(CASES) - fails}/{len(CASES)} passed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
