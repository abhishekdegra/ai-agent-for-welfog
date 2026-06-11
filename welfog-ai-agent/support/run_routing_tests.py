"""
Smoke + integration tests for AI-first routing and grounded replies.
Run: python run_routing_tests.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5000"


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").replace("&nbsp;", " ")


def _post_chat(message: str, user_id: str) -> dict:
    url = f"{BASE}/chat?user_id={user_id}"
    body = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _route_for(message: str) -> tuple[dict, object]:
    from services.ai_first_router import resolve_answer_route_ai_first

    dec, route = resolve_answer_route_ai_first(
        message, message, retrieval_query=message, conv_for_llm="", reply_lang="en"
    )
    return route or {}, dec


def _check(name: str, ok: bool, detail: str) -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return {"name": name, "ok": ok, "detail": detail}


def run_unit_tests() -> list[dict]:
    print("\n=== UNIT: semantic guard + answer plan ===")
    from services.semantic_route_guard import (
        infer_customer_semantic_goal,
        reconcile_ai_route_with_semantic_goal,
    )
    from services.semantic_answer_plan import build_semantic_answer_plan

    results: list[dict] = []

    # Semantic goals — not a phrase router; paraphrases without "wishlist" must still classify.
    cases = [
        ("return kese krte h", "refund_policy", None),
        ("jitne bhi order kiye unki list de de", "order_history_list", None),
        ("track kr dega kya cancel nhi ho gya", "track_single_order", None),
        ("order kiya lekin nahi pahucha", "track_single_order", None),
        ("meri order history dikhao", "order_history_list", None),
        ("tu bta mereko meri wishlist", "wishlist_list", None),
        ("You show me my wishlist", "wishlist_list", None),
        ("meri wishlist dikhao", "wishlist_list", None),
        ("jo maine like kiye woh yahi dikha do", "wishlist_list", None),
        ("heart wale products ki list chahiye", "wishlist_list", None),
        ("jitna bhi mangaya welfog se woh list", "order_history_list", None),
    ]
    for msg, expected_goal, _ in cases:
        goal = infer_customer_semantic_goal(msg, msg)
        ok = goal == expected_goal
        results.append(_check(f"goal:{msg[:35]}", ok, f"got={goal!r} want={expected_goal!r}"))

    ai_ok_route = {
        "intent": "order_history",
        "data_channel": "live_api",
        "needs_order_id": False,
        "answer_strategy": "live_api_only",
    }
    r = reconcile_ai_route_with_semantic_goal(
        ai_ok_route, "jitne bhi order kiye unki list de de", "jitne bhi order kiye unki list de de"
    )
    results.append(
        _check(
            "reconcile:trust AI on list",
            not r.get("_semantic_override"),
            f"override={r.get('_semantic_override')}",
        )
    )

    bad_route = {
        "intent": "order_history",
        "data_channel": "live_api",
        "needs_order_id": True,
        "answer_strategy": "live_api_only",
    }
    r2 = reconcile_ai_route_with_semantic_goal(
        bad_route, "jitne bhi order kiye unki list de de", "jitne bhi order kiye unki list de de"
    )
    results.append(
        _check(
            "reconcile:fix needs_order_id on list",
            r2.get("intent") == "order_history" and not r2.get("needs_order_id"),
            f"intent={r2.get('intent')} needs_oid={r2.get('needs_order_id')} override={r2.get('_semantic_override')}",
        )
    )

    wrong_order_route = {
        "intent": "order_history",
        "data_channel": "live_api",
        "needs_order_id": False,
        "answer_strategy": "live_api_only",
    }
    r3 = reconcile_ai_route_with_semantic_goal(
        wrong_order_route, "tu bta mereko meri wishlist", "tu bta mereko meri wishlist"
    )
    results.append(
        _check(
            "reconcile:wishlist beats order_history AI",
            r3.get("intent") == "wishlist" and r3.get("_semantic_goal") == "wishlist_list",
            f"intent={r3.get('intent')} goal={r3.get('_semantic_goal')} override={r3.get('_semantic_override')}",
        )
    )

    from services.account_list_semantics import promote_account_list_on_route

    paraphrase_route = {
        "intent": "order_history",
        "data_channel": "live_api",
        "user_meaning": "Customer wants to see products they liked and saved on Welfog in chat.",
        "account_list_kind": "wishlist_in_chat",
    }
    r4 = promote_account_list_on_route(
        paraphrase_route,
        "jo maine dil se pasand kiye unki list yahi bhej",
        "jo maine dil se pasand kiye unki list yahi bhej",
    )
    results.append(
        _check(
            "account_list:AI field overrides wrong intent",
            r4.get("intent") == "wishlist"
            and r4.get("account_list_kind") == "wishlist_in_chat",
            f"intent={r4.get('intent')} kind={r4.get('account_list_kind')}",
        )
    )

    plan = build_semantic_answer_plan(
        {"intent": "general", "data_channel": "kb", "answer_strategy": "kb_then_ai"}
    )
    results.append(
        _check(
            "plan:kb_then_ai",
            plan.answer_strategy == "kb_then_ai" and plan.use_ai_synthesis,
            f"strategy={plan.answer_strategy} ai={plan.use_ai_synthesis}",
        )
    )
    return results


def run_routing_tests() -> list[dict]:
    print("\n=== ROUTING: resolve_answer_route_ai_first (Groq) ===")
    results: list[dict] = []

    specs = [
        {
            "msg": "return kese krte h",
            "want_handlers": {"dynamic_kb", "ai_route_and_answer", "warm_feedback"},
            "forbid_handlers": {"order_tracking_api", "order_ai_flow"},
            "want_intents": {"general", "refund", "payment"},
        },
        {
            "msg": "jitne bhi order kiye unki list de de",
            "want_handlers": {"order_ai_flow"},
            "forbid_handlers": {"order_tracking_api"},
            "want_intents": {"order_history"},
        },
        {
            "msg": "track kr dega kya cancel nhi ho gya",
            "want_handlers": {"order_tracking_api"},
            "forbid_handlers": {"order_ai_flow"},
            "want_intents": {"order", "general"},
        },
        {
            "msg": "order kiya lekin nahi pahucha",
            "want_handlers": {"order_tracking_api", "order_ai_flow"},
            "forbid_handlers": set(),
            "want_intents": {"order", "order_history"},
            "forbid_if_intent_only": {"order_history"},
        },
    ]

    for spec in specs:
        msg = spec["msg"]
        try:
            route, dec = _route_for(msg)
            h = dec.handler
            intent = dec.intent
            ok = h in spec["want_handlers"] or (
                not spec["want_handlers"] and h not in spec.get("forbid_handlers", set())
            )
            if spec.get("forbid_handlers") and h in spec["forbid_handlers"]:
                ok = False
            if spec.get("want_intents") and intent not in spec["want_intents"]:
                ok = False
            fi = spec.get("forbid_if_intent_only")
            if fi and intent in fi and h not in spec["want_handlers"]:
                ok = False
            strat = (route or {}).get("answer_strategy", "")
            results.append(
                _check(
                    f"route:{msg[:30]}",
                    ok,
                    f"handler={h} intent={intent} strategy={strat} override={route.get('_semantic_override')}",
                )
            )
        except Exception as e:
            results.append(_check(f"route:{msg[:30]}", False, str(e)[:120]))
        time.sleep(0.5)

    return results


def run_http_tests() -> list[dict]:
    print("\n=== HTTP: POST /chat (live server) ===")
    results: list[dict] = []

    try:
        urllib.request.urlopen(f"{BASE}/", timeout=3)
    except Exception as e:
        results.append(_check("server up", False, f"Cannot reach {BASE}: {e}"))
        return results

    results.append(_check("server up", True, BASE))

    http_cases = [
        {
            "uid": "test_routing_001",
            "msg": "return kese krte h",
            "must_contain_any": ["return", "refund", "pickup", "7", "replacement", "wapas"],
            "must_not_contain": ["No refund record found", "purchase history", "wf-ph-root"],
        },
        {
            "uid": "test_routing_002",
            "msg": "jitne bhi order kiye unki list de de",
            "must_contain_any": ["order", "purchase", "wf-ph", "Order ID", "history"],
            "must_not_contain": ["paste a valid Order ID", "Order ID bhej"],
        },
        {
            "uid": "test_routing_003",
            "msg": "track kr dega kya cancel nhi ho gya",
            "must_contain_any": ["Order ID", "order id", "track", "ID"],
            "must_not_contain": ["wf-ph-root", "purchase history list"],
        },
        {
            "uid": "test_routing_004",
            "msg": "order kiya lekin abhi tak nahi pahucha",
            "must_contain_any": ["Order ID", "order id", "track", "delivery", "pahuch"],
            "must_not_contain": [],
        },
    ]

    for i, spec in enumerate(http_cases):
        uid = spec["uid"]
        msg = spec["msg"]
        try:
            if i:
                time.sleep(2)
            out = _post_chat(msg, uid)
            text = _strip_html(out.get("data", "") or "").lower()
            has_any = any(k.lower() in text for k in spec["must_contain_any"])
            bad = any(k.lower() in text for k in spec["must_not_contain"])
            ok = has_any and not bad
            preview = (text[:140] + "...") if len(text) > 140 else text
            results.append(
                _check(
                    f"http:{msg[:28]}",
                    ok,
                    f"has_keywords={has_any} bad={bad} | {preview}",
                )
            )
        except urllib.error.HTTPError as e:
            results.append(_check(f"http:{msg[:28]}", False, f"HTTP {e.code}"))
        except Exception as e:
            results.append(_check(f"http:{msg[:28]}", False, str(e)[:120]))

    return results


def main() -> int:
    all_results: list[dict] = []
    all_results.extend(run_unit_tests())
    all_results.extend(run_routing_tests())
    all_results.extend(run_http_tests())

    passed = sum(1 for r in all_results if r["ok"])
    total = len(all_results)
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {passed}/{total} passed")
    failed = [r for r in all_results if not r["ok"]]
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  - {r['name']}: {r['detail']}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
