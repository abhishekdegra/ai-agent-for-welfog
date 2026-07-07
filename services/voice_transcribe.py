"""Server-side speech-to-text via Groq Whisper (fallback when browser Web Speech fails)."""
import os

import requests

from utils.reasoning_log import log_reasoning

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _mime_to_ext(mime: str, filename: str) -> str:
    m = (mime or "").lower()
    fn = (filename or "").lower()
    if "webm" in m or fn.endswith(".webm"):
        return "webm"
    if "ogg" in m or fn.endswith(".ogg"):
        return "ogg"
    if "mp4" in m or "m4a" in m or fn.endswith(".m4a"):
        return "m4a"
    if "wav" in m or fn.endswith(".wav"):
        return "wav"
    if "mpeg" in m or fn.endswith(".mp3"):
        return "mp3"
    return "webm"


def transcribe_audio_blob(audio_bytes: bytes, filename: str = "audio.webm", mime: str = "audio/webm") -> dict:
    """
    Transcribe uploaded audio. Language auto-detected (English + Hindi/Hinglish).
    Returns { ok, text, error? }.
    """
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        return {"ok": False, "error": "missing_api_key", "text": ""}
    if not audio_bytes or len(audio_bytes) < 200:
        return {"ok": False, "error": "audio_too_short", "text": ""}

    model = (os.getenv("GROQ_WHISPER_MODEL") or "whisper-large-v3-turbo").strip()
    ext = _mime_to_ext(mime, filename)
    files = {"file": (f"voice.{ext}", audio_bytes, mime or f"audio/{ext}")}
    data = {
        "model": model,
        "response_format": "json",
        "temperature": "0",
        # Hints for Indian English + Roman Hinglish on Welfog support chat
        "prompt": "Welfog customer support. English and Hindi Hinglish roman: order, delivery, product, refund.",
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        res = requests.post(GROQ_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=60)
        if res.status_code != 200:
            log_reasoning(f"Groq Whisper HTTP {res.status_code}: {(res.text or '')[:200]}")
            return {"ok": False, "error": "transcribe_failed", "text": ""}
        body = res.json()
        text = (body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty_transcript", "text": ""}
        log_reasoning(f"Groq Whisper transcript ({len(text)} chars).")
        return {"ok": True, "text": text}
    except requests.Timeout:
        log_reasoning("Groq Whisper request timed out.")
        return {"ok": False, "error": "timeout", "text": ""}
    except Exception as e:
        log_reasoning(f"Groq Whisper error: {e}")
        return {"ok": False, "error": "transcribe_error", "text": ""}
