"""Re-test previously failing turns."""
import time
import uuid
import requests

BASE = "http://127.0.0.1:5000"
USER = "1167"
CASES = [
    ("hi_about", "वेलफोग क्या है?"),
    ("hinglish_cod", "cash on delivery accept karta hai kya welfog?"),
    ("hinglish_orders", "meri purani orders dikhao na"),
    ("marathi_support", "customer care cha number sang"),
]

for label, msg in CASES:
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE}/chat",
        params={"user_id": USER},
        json={"message": msg, "chat_id": uuid.uuid4().hex},
        timeout=60,
    )
    ms = time.perf_counter() - t0
    body = ""
    if r.status_code == 200:
        d = r.json()
        body = (d.get("data") or "")[:150]
    print(f"{label}: {ms:.1f}s HTTP {r.status_code} | {body.encode('ascii', errors='replace').decode()}")
