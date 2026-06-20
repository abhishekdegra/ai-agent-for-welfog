"""Quick routing smoke test — 15 queries, timing + basic relevance checks."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass

import requests

BASE = "http://127.0.0.1:5000/chat"
USER_ID = "1167"
REQUEST_TIMEOUT = float(os.getenv("QA_CHAT_TIMEOUT", "90") or "90")

TESTS = [
    ("api", "meri order history dikhao", ("order", "purchase", "wf-ph", "history")),
    ("api", "2606020 is order ki details bta", ("wf-od", "order", "2606020", "details")),
    ("api", "mujhe iphone dikhao", ("product", "iphone", "wf-product", "search")),
    ("api", "refund status for order 2606020", ("refund", "2606020")),
    ("api", "meri wishlist dikhao", ("wishlist", "wf-wl")),
    ("api", "302001 pincode pe delivery hoti hai kya", ("pin", "302001", "deliver")),
    ("kb", "refund policy kya hai welfog ki", ("refund", "policy", "return")),
    ("kb", "welfog company ke baare me batao", ("welfog", "marketplace", "company")),
    ("kb", "seller registration kaise hoti hai", ("seller", "register", "vendor")),
    ("chat", "hello bhai kaise ho", ("hello", "welfog", "help")),
    ("chat", "thanks yaar bahut help ho gayi", ("thank", "welcome", "shukriya")),
    ("chat", "tum abhi free ho kya", ("welfog", "help", "assist")),
    ("ood", "who won yesterday cricket match", ("welfog", "shopping", "assist", "sorry", "can't", "cannot", "only")),
    ("ood", "write python sorting code for me", ("welfog", "shopping", "assist", "code", "only")),
    ("ood", "amazon pe iphone sasta hai kya", ("welfog", "assist", "amazon", "only", "shopping")),
]


@dataclass
class Result:
    category: str
    query: str
    seconds: float
    ok: bool
    preview: str
    error: str = ""


def run_one(category: str, query: str, markers: tuple[str, ...]) -> Result:
    chat_id = f"qa-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE}?user_id={USER_ID}",
            json={"message": query, "chat_id": chat_id},
            timeout=REQUEST_TIMEOUT,
        )
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return Result(category, query, elapsed, False, "", f"HTTP {r.status_code}")
        data = r.json()
        body = (data.get("data") or "")
        if not isinstance(body, str):
            body = json.dumps(data)[:300]
        low = re.sub(r"<[^>]+>", " ", body).lower()
        ok = any(m.lower() in low for m in markers)
        preview = low[:140].replace("\n", " ")
        return Result(category, query, elapsed, ok, preview)
    except Exception as e:
        return Result(category, query, time.perf_counter() - t0, False, "", str(e))


def main() -> None:
    results: list[Result] = []
    print(f"{'CAT':<5} {'SEC':>6} {'OK':>3}  QUERY")
    print("-" * 72)
    for cat, q, markers in TESTS:
        res = run_one(cat, q, markers)
        results.append(res)
        flag = "Y" if res.ok else "N"
        print(f"{cat:<5} {res.seconds:6.2f} {flag:>3}  {q[:50]}")
        if res.error:
            print(f"      ERR: {res.error}")
        elif not res.ok:
            print(f"      preview: {res.preview!r}")
        time.sleep(1.0)

    slow = [r for r in results if r.seconds > 15]
    bad = [r for r in results if not r.ok]
    avg = sum(r.seconds for r in results) / len(results)
    print("-" * 72)
    print(f"avg={avg:.2f}s  slow(>15s)={len(slow)}  relevance_fail={len(bad)}")


if __name__ == "__main__":
    main()
