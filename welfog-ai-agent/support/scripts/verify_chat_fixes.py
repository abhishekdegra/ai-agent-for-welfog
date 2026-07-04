"""Quick E2E verify for KB + delivery routing fixes."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5000/chat"
USER_ID = "1167"
QUERIES = [
    ("welfog krta kya h", ["platform", "commerce", "marketplace", "short", "e-commerce", "integrated"]),
    ("welfog kya h", ["platform", "commerce", "marketplace", "welfog is", "integrated"]),
    (
        "welfog mundiya me de de ga kya apne products ki delivery",
        ["pin", "delivery", "mundiya", "available", "order", "pincode", "service"],
    ),
]


def post_chat(msg: str, chat_id: str | None = None) -> tuple[float, str, dict]:
    payload = {"message": msg, "user_id": USER_ID}
    if chat_id:
        payload["chat_id"] = chat_id
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    ms = (time.perf_counter() - t0) * 1000.0
    data = json.loads(raw) if raw.strip() else {}
    reply = (data.get("reply") or data.get("response") or data.get("message") or raw)[:500]
    return ms, reply, data


def main() -> int:
    chat_id = None
    failed = 0
    for msg, hints in QUERIES:
        print(f"\n=== {msg!r} ===")
        try:
            ms, reply, data = post_chat(msg, chat_id)
            chat_id = data.get("chat_id") or chat_id
            low = reply.lower()
            bad_courier = "courier partners" in low and "what is" not in msg
            bad_returns = "5 days" in low and "return" in low and "kya h" in msg
            ok_hint = any(h in low for h in hints)
            status = "OK" if ok_hint and not bad_courier and not bad_returns else "FAIL"
            if status == "FAIL":
                failed += 1
            print(f"  {status}  {ms:.0f}ms")
            print(f"  reply: {reply[:280]}...")
            if bad_courier:
                print("  !! wrong courier FAQ")
            if bad_returns:
                print("  !! wrong returns FAQ")
        except urllib.error.URLError as e:
            failed += 1
            print(f"  FAIL  server error: {e}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {e}")
        time.sleep(1.5)
    print(f"\nDone: {len(QUERIES) - failed}/{len(QUERIES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
