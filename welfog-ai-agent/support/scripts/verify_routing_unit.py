"""Unit-level verify — no HTTP server required."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def log_result(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def test_delivery_not_blocked() -> bool:
    from utils.helpers import (
        _turn_blocks_pincode_serviceability_routing,
        turn_is_obvious_product_shopping_turn,
        _text_is_delivery_serviceability_hypothetical,
    )

    msg = "welfog mundiya me de de ga kya apne products ki delivery"
    hypo = _text_is_delivery_serviceability_hypothetical(msg)
    product = turn_is_obvious_product_shopping_turn(msg, msg, "")
    blocked = _turn_blocks_pincode_serviceability_routing(msg)
    ok = hypo and not product and not blocked
    log_result(
        "delivery_mundiya",
        ok,
        f"hypo={hypo} product_shop={product} blocked={blocked}",
    )
    return ok


def test_kb_preflight_about() -> bool:
    from services.ai_first_router import _try_policy_kb_preflight

    t0 = time.perf_counter()
    for msg in ("welfog krta kya h", "welfog kya h"):
        pf = _try_policy_kb_preflight(msg, msg, "hinglish")
        if not pf:
            log_result(f"kb_preflight_{msg[:20]}", False, "preflight returned None")
            return False
        _, route = pf
        keys = route.get("kb_keys") or []
        top = route.get("_preflight_top_file") or (keys[0] if keys else "")
        if top == "faqs" and "company" not in keys:
            log_result(f"kb_preflight_{msg[:20]}", False, f"faqs-only keys={keys}")
            return False
        log_result(
            f"kb_preflight_{msg[:20]}",
            True,
            f"top={top} keys={keys} score={route.get('_preflight_top_score')}",
        )
    log_result("kb_preflight_timing", True, f"{(time.perf_counter()-t0)*1000:.0f}ms total")
    return True


def test_kb_answer_scoped() -> bool:
    from services.kb_service import format_kb_answer_from_brain_keys

    route = {
        "data_channel": "kb",
        "kb_keys": ["company"],
        "_preflight_kb": True,
        "user_meaning": "What is Welfog",
    }
    t0 = time.perf_counter()
    body = format_kb_answer_from_brain_keys(
        "welfog kya h",
        "welfog kya h",
        ["company"],
        reply_lang="hinglish",
        ai_route=route,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    low = (body or "").lower()
    bad = "courier partners" in low or ("5 days" in low and "return" in low)
    good = any(
        x in low
        for x in ("platform", "marketplace", "commerce", "short video", "e-commerce")
    )
    ok = bool(body) and good and not bad
    log_result(
        "kb_answer_company",
        ok,
        f"{ms:.0f}ms bad={bad} snippet={(body or '')[:120]}",
    )
    return ok


def test_delivery_turn() -> bool:
    from services.location_delivery_resolver import turn_requests_delivery_serviceability

    msg = "welfog mundiya me de de ga kya apne products ki delivery"
    t0 = time.perf_counter()
    ok = turn_requests_delivery_serviceability(msg, msg, "", allow_llm=True)
    ms = (time.perf_counter() - t0) * 1000.0
    log_result("delivery_turn_requests", ok, f"{ms:.0f}ms serviceability={ok}")
    return ok


def main() -> int:
    print("=== routing unit verify ===\n")
    results = [
        test_delivery_not_blocked(),
        test_kb_preflight_about(),
        test_kb_answer_scoped(),
        test_delivery_turn(),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
