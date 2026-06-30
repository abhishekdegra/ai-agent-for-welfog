"""Sequential chat API smoke test — one fresh chat_id per message."""
from __future__ import annotations

import json
import time
import uuid

import requests

BASE = "http://127.0.0.1:5000"
UID = "1167"

TESTS = [
    ("chitchat", "heelo bhai kya haal h"),
    ("kb", "return policy kya h welfog ki"),
    ("ood", "bhai meri shaadi krwade"),
    ("wishlist", "mere save products ki list de"),
    ("product", "white color ki shoes dikhao"),
    ("kb", "welfog delivery kitne din me hoti h"),
]

MAX_SEC = 45.0
GAP_SEC = 4.0


def main() -> None:
    results = []
    for kind, msg in TESTS:
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
            body = body.replace("\n", " ")[:220]
        except Exception as exc:
            err = str(exc)
        dt = time.perf_counter() - t0
        ok_time = dt <= 15.0
        row = {
            "kind": kind,
            "msg": msg,
            "sec": round(dt, 2),
            "ok_time": ok_time,
            "status": status,
            "err": err,
            "preview": body,
        }
        results.append(row)
        flag = "PASS" if ok_time and not err and body else "FAIL"
        print(f"[{flag}] {kind:9} {dt:5.1f}s | {msg!r}")
        if err:
            print(f"       ERR: {err}")
        elif body:
            print(f"       -> {body[:180]}")
        else:
            print("       -> (empty body)")
        time.sleep(GAP_SEC)

    print("\n=== SUMMARY ===")
    slow = [r for r in results if not r["ok_time"]]
    empty = [r for r in results if not r["preview"] and not r["err"]]
    errs = [r for r in results if r["err"]]
    print(f"total={len(results)} slow(>{15}s)={len(slow)} errors={len(errs)} empty={len(empty)}")
    out_path = "scripts/test_chat_batch_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
