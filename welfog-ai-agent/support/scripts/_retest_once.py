import json, time, uuid, sys
import requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:5000"
UID = "1167"
TIMEOUT = 90.0
TESTS = [
    ("order_hi", "mere purchased orders ka data dikha de"),
    ("product_hi", "bhai redmi ka mobile cover dikha"),
    ("pincode", "302012 pe welfog delivery karta hai kya"),
    ("kb", "refund kitne din me milta hai"),
    ("deals", "aaj ki best deals dikhao welfog pe"),
]
MARKERS = ["wf-ph", "wf-product-rail", "wf-pin-root", "refund"]
pass_n = 0
for name, msg in TESTS:
    t0 = time.perf_counter()
    st = 0
    body = ""
    err = ""
    try:
        r = requests.post(
            f"{BASE}/chat?user_id={UID}",
            json={"message": msg, "chat_id": uuid.uuid4().hex},
            timeout=TIMEOUT,
        )
        st = r.status_code
        try:
            body = (r.json().get("data") or "").strip()
        except Exception:
            body = (r.text or "").strip()
    except Exception as e:
        err = str(e)[:120]
    sec = round(time.perf_counter() - t0, 1)
    low = body.lower()
    mflags = {m: (m in low if m != "refund" else ("refund" in low)) for m in MARKERS}
    ok = sec <= 20 and st == 200 and bool(body.strip())
    if ok:
        pass_n += 1
    line = f"{name}|{sec}s|{st}|len={len(body)}|wf-ph={mflags['wf-ph']}|wf-product-rail={mflags['wf-product-rail']}|wf-pin-root={mflags['wf-pin-root']}|refund={mflags['refund']}|PASS={ok}"
    if err:
        line += f"|err={err}"
    print(line, flush=True)
print(f"PASS_COUNT={pass_n}/5", flush=True)
