def log_reasoning(msg: str):
    """Uniform terminal reasoning logs for every chat path."""
    try:
        safe = (msg or "").encode("ascii", errors="replace").decode("ascii")
        print(f"[AI Reasoning] {safe}", flush=True)
    except Exception:
        pass


def chat_log(msg: str):
    """High-level chat pipeline logs (request in/out, errors, timing)."""
    try:
        safe = (msg or "").encode("ascii", errors="replace").decode("ascii")
        print(f"[chat] {safe}", flush=True)
    except Exception:
        pass
