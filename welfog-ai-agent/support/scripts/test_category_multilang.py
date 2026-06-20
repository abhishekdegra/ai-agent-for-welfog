"""Quick multilang category browse smoke test against /chat."""
import json
import time
import urllib.request

QUERIES = [
    ("hinglish", "electronics ke products dikha de"),
    ("hinglish", "mens fashion k bhi bta de"),
    ("hinglish", "beauty products dikhao"),
    ("english", "show me women fashion products"),
    ("hinglish", "home kitchen ke products bta"),
    ("sku", "is sku ka product bta INFINIX HOT 10 PLAY-EGL-SP"),
]

BASE = "http://127.0.0.1:5000/chat"


def ask(msg: str, timeout: float = 90.0) -> tuple[float, str, bool]:
    data = json.dumps({"message": msg, "user_id": "1167"}).encode()
    req = urllib.request.Request(
        BASE, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    text = body.get("data") or ""
    has_cards = "wf-product" in text
    busy = "taking longer" in text.lower() or "busy" in text.lower()
    preview = text[:100].replace("\n", " ")
    return dt, preview, has_cards and not busy


def main():
    print("lang\tseconds\tcards\tquery")
    for lang, q in QUERIES:
        try:
            dt, preview, ok = ask(q)
            print(f"{lang}\t{dt:.1f}\t{ok}\t{q[:50]}")
            if not ok and "nahi mila" not in preview.lower():
                print(f"  preview: {preview}")
        except Exception as e:
            print(f"{lang}\tERR\t-\t{q[:50]} ({e})")


if __name__ == "__main__":
    main()
