import os
import threading


def _ensure_hf_hub_token() -> None:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        try:
            from dotenv import load_dotenv

            from support_paths import ENV_FILE

            load_dotenv(ENV_FILE)
            token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
        except Exception:
            pass
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


_ensure_hf_hub_token()

_model = None
_model_init_lock = threading.Lock()
_encode_lock = threading.RLock()


def embeddings_disabled() -> bool:
    return (os.getenv("DISABLE_EMBEDDINGS", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _get_model():
    global _model
    if embeddings_disabled():
        return None
    if _model is not None:
        return _model
    with _model_init_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            print("[embed] loading SentenceTransformer at startup...", flush=True)
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[embed] model ready", flush=True)
    return _model


def encode_texts(texts):
    if not texts or embeddings_disabled():
        return None
    m = _get_model()
    if m is None:
        return None
    try:
        with _encode_lock:
            return m.encode(texts)
    except Exception as exc:
        print(f"[embed] encode failed: {exc}", flush=True)
        return None


class _ModelProxy:
    def encode(self, texts):
        out = encode_texts(texts)
        if out is None:
            import numpy as np

            n = len(texts) if texts else 0
            return np.zeros((n, 384), dtype=np.float32)
        return out


model = _ModelProxy()

GREETINGS = [
    "hi", "hello", "hii", "hey", "namaste",
    "hi there", "hello there", "hey bro", "hi bro", "hello bro",
    "hey bhai", "hello bhai", "suno bhai", "namaste bhai", "namaste bhaiya",
    "ram ram", "ram ram bhai", "radhe radhe", "kaise ho bhai",
    "how are you", "thanks bhai", "shukriya",
]
_GREETINGS_VECS = None


def get_greetings_vecs():
    global _GREETINGS_VECS
    if _GREETINGS_VECS is None:
        vecs = encode_texts(GREETINGS)
        if vecs is None:
            import numpy as np

            _GREETINGS_VECS = np.zeros((len(GREETINGS), 384), dtype=np.float32)
        else:
            _GREETINGS_VECS = vecs
    return _GREETINGS_VECS
