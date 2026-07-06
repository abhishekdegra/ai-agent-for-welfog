import hashlib
import os
import random
import re
import threading
from html import escape as html_escape

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from support_paths import KNOWLEDGE_DIR
from services.embedding import embeddings_disabled, encode_texts
from utils.cache import _cache, _cache_get, _cache_set
from utils.reasoning_log import log_reasoning
from utils import validators
from utils.helpers import (
    _is_plausible_order_id,
    _looks_like_greeting_message,
    _looks_like_light_smalltalk,
    _text_asks_how_to_view_order_history,
    _text_asks_order_history,
    _text_has_order_placement_intent,
    _text_has_refund_or_return_intent,
    _text_is_order_id_help_request,
    _text_is_order_tracking_intent,
    _text_is_tracking_howto_request,
    _text_needs_order_id_for_refund_or_payment,
    _text_needs_order_id_for_tracking,
    pick_warm_chat_reply_key,
    _should_bypass_warm_greeting_fast_path,
)

_KB_SNAPSHOT = ""
_KB_VECTOR_STATUS = "not_loaded"
_KB_REFRESH_LOCK = threading.RLock()

# Unified semantic thresholds (env-tunable; cosine similarity 0–1)
KB_SEMANTIC_MIN_SCORE = float(os.getenv("KB_SEMANTIC_MIN_SCORE", "0.22") or "0.22")
KB_CONTEXT_MIN_SCORE = float(os.getenv("KB_CONTEXT_MIN_SCORE", "0.18") or "0.18")
KB_DIRECT_MIN_SCORE = float(os.getenv("KB_DIRECT_MIN_SCORE", "0.22") or "0.22")
KB_STRONG_MATCH_SCORE = float(os.getenv("KB_STRONG_MATCH_SCORE", "0.32") or "0.32")
KB_ANSWER_MIN_CONFIDENCE = float(os.getenv("KB_ANSWER_MIN_CONFIDENCE", "0.26") or "0.26")
KB_RETRIEVAL_STRONG_SCORE = float(os.getenv("KB_RETRIEVAL_STRONG_SCORE", "0.38") or "0.38")


def get_kb_vector_status() -> str:
    """Debug: last index rebuild / embedding state."""
    return _KB_VECTOR_STATUS


def get_knowledge_version() -> str:
    """MD5 snapshot id — changes when admin updates knowledge files."""
    global _KB_SNAPSHOT
    if not _KB_SNAPSHOT:
        try:
            ensure_knowledge_cache_fresh()
        except Exception:
            pass
    snap = (_KB_SNAPSHOT or "").strip()
    return snap[:48] if snap else "unknown"


def _plain_chunk_for_embed(chunk: str) -> str:
    """Plain text for embeddings — HTML line breaks hurt retrieval quality."""
    t = re.sub(r"<br\s*/?>", "\n", chunk or "", flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:1200]


def _normalize_retrieval_query(query: str) -> str:
    """Normalize user/AI query before embedding (any language)."""
    q = _plain_chunk_for_embed(query or "")
    return q[:900]


def _embed_text_for_chunk(chunk: str) -> str:
    """
    Text fed to the embedding model — FAQ Q+A prefixed for better semantic match.
    Display/storage chunk is unchanged; only vectors use this form.
    """
    raw = (chunk or "").strip()
    if not raw:
        return ""
    parts = re.split(r"<br\s*/?>", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) >= 2:
        q_line = parts[0].strip()
        a_line = parts[1].strip()
        if _looks_like_faq_question_line(q_line) and a_line:
            q_plain = _plain_chunk_for_embed(q_line)
            a_plain = _plain_chunk_for_embed(a_line)
            return f"Question: {q_plain}\nAnswer: {a_plain}"[:1200]
    return _plain_chunk_for_embed(raw)


def _chunk_dedup_key(chunk: str) -> str:
    """Near-duplicate suppression in retrieval results."""
    plain = _plain_chunk_for_embed(chunk)[:320].lower()
    return re.sub(r"\W+", " ", plain).strip()


def _embeddings_ready() -> bool:
    if embeddings_disabled():
        return False
    return all_vectors is not None and len(all_vectors) > 0


def _is_internal_kb_key(key: str) -> bool:
    k = (key or "").lower()
    if k in INTERNAL_KB_KEYS:
        return True
    return k.startswith("welfog_api") or k in ("system_messages", "system_messages_2")


def _build_flat_vectors_from_keys() -> np.ndarray:
    """Stack per-file vectors in the same order as all_chunks / all_chunk_sources."""
    parts: list[np.ndarray] = []
    for k, chunks in (kb_chunks_by_key or {}).items():
        vec = (kb_vectors_by_key or {}).get(k)
        if chunks and vec is not None and len(vec) == len(chunks):
            parts.append(np.atleast_2d(vec))
    if not parts:
        return np.array([])
    return parts[0] if len(parts) == 1 else np.vstack(parts)


def get_runtime_knowledge_files():
    """
    Auto-discovers all .txt files from knowledge folder.
    No hardcoded mapping required.
    """
    runtime = {}
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        return runtime
    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        key_base = os.path.splitext(filename)[0].replace("-", "_").replace(" ", "_").lower()
        key = re.sub(r"[^a-z0-9_]", "", key_base)
        if not key:
            continue
        # Avoid key collisions if two files normalize to same key
        if key in runtime:
            n = 2
            while f"{key}_{n}" in runtime:
                n += 1
            key = f"{key}_{n}"
        runtime[key] = file_path
    return runtime

def get_allowed_knowledge_filenames():
    return sorted({os.path.basename(path) for path in get_runtime_knowledge_files().values() if path.endswith(".txt")})

def _compute_kb_snapshot(runtime_files):
    """Detect any add/update/delete — size, mtime, and content hash (same-size edits)."""
    parts = []
    for key, path in sorted(runtime_files.items()):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            digest = hashlib.md5(raw).hexdigest()[:16]
            parts.append(f"{key}:{len(raw)}:{digest}")
        except OSError:
            parts.append(f"{key}:missing")
    return "|".join(parts)


def _parse_system_messages(text: str):
    """
    Parses lines like: key = value
    Ignores empty lines and headings.
    """
    out = {}
    if not text:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def _looks_like_faq_document(text: str) -> bool:
    """Admin FAQ files: many question lines ending with ? (no code changes when new Q&A added)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    q_lines = sum(1 for ln in lines if ln.endswith("?") and 12 <= len(ln) <= 220)
    return q_lines >= 3


def _looks_like_faq_question_line(line: str) -> bool:
    """FAQ question title — answer should follow without echoing the question."""
    s = re.sub(
        r"^(?:\*\*|__|<b>|</b>|<strong>|</strong>)\s*|\s*(?:\*\*|__|</b>|</strong>)$",
        "",
        (line or "").strip(),
        flags=re.I,
    )
    if not s:
        return False
    if s.endswith("?:") or (s.endswith("?") and 10 <= len(s) <= 240):
        return True
    if 12 <= len(s) <= 120 and re.match(
        r"^(what|how|can|does|do|is|are|will|who|where|when|which)\b",
        s,
        re.I,
    ):
        low = s.lower()
        if any(
            x in low
            for x in (
                " allows ",
                " accepts ",
                " offers ",
                " yes,",
                " no,",
                " welfog has ",
                " welfog does ",
                " welfog will ",
                " to place ",
                " to track ",
                " to initiate ",
            )
        ):
            return False
        return True
    return False


def _chunk_is_agent_instruction_blob(chunk: str) -> bool:
    """Admin/routing instructions in KB files — never show to customers."""
    plain = re.sub(r"<br\s*/?>", "\n", chunk or "")
    low = plain.lower()
    if "support rules" in low or "social media links — support rules" in low:
        return True
    if re.search(
        r"\bnever invent\b|\bonly share welfog\b|\bpolitely decline\b|\brouting rule\b",
        low,
    ):
        return True
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    agent_bullets = sum(
        1
        for ln in lines
        if re.match(r"^-\s+if user", ln, re.I) or re.match(r"^-\s+follow-up", ln, re.I)
    )
    if agent_bullets >= 2:
        return True
    return False


def _looks_like_kb_metadata_label(line: str) -> bool:
    """SEO-style KB labels (e.g. support.txt keyword headings) — not customer-facing prose."""
    s = (line or "").strip().lower()
    if not s:
        return False
    if re.match(r"^welfog customer support.*:\s*$", s):
        return True
    if len(s) < 130 and re.search(
        r"(support mobile number|support email|customer care support|customer support,)",
        s,
    ):
        return True
    return False


def _split_faq_qa_chunks(text: str) -> list[str]:
    """
    One embedding chunk per FAQ pair (question + answer).
    Supports admin format: question line ending with ?, then answer (blank line optional).
    """
    if not (text or "").strip():
        return []

    def _flush(buf: list[str], dest: list[str]) -> None:
        block = "\n".join(buf).strip()
        if len(block) >= 12:
            dest.append(block.replace("\n", "<br>"))

    blocks_raw: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if current:
                _flush(current, blocks_raw)
                current = []
            continue
        is_question = _looks_like_faq_question_line(stripped)
        if is_question and current and any(_looks_like_faq_question_line(l.strip()) for l in current):
            _flush(current, blocks_raw)
            current = [raw.rstrip()]
        else:
            current.append(raw.rstrip())
    if current:
        _flush(current, blocks_raw)

    if not blocks_raw:
        blocks_raw = [b.strip() for b in re.split(r"\n\s*\n+", text.strip()) if b.strip()]

    out: list[str] = []
    for block in blocks_raw:
        html = block if "<br>" in block else block.replace("\n", "<br>")
        if len(html) <= 950:
            out.append(html)
            continue
        i = 0
        while i < len(html):
            piece = html[i : i + 950]
            if len(piece) >= 20:
                out.append(piece)
            i += 780
    return out


def _split_kb_chunks(text: str, file_key: str = ""):
    """
    Split knowledge files into embedding chunks. Short title-only blocks are merged
    with the following paragraph so retrieval returns real content, not just headings.
    FAQ-style files (faqs.txt): one chunk per Q&A block — never merge pairs.
    """
    if not text:
        return []
    fk = (file_key or "").lower()
    if fk == "faqs" or _looks_like_faq_document(text):
        faq_chunks = _split_faq_qa_chunks(text)
        if faq_chunks:
            return faq_chunks

    parts = []
    for c in text.split("\n\n"):
        c = c.strip().replace("\n", "<br>")
        if len(c) < 12:
            continue
        parts.append(c)
    if not parts:
        return []
    merged = []
    acc = parts[0]
    for p in parts[1:]:
        if len(acc) < 160:
            acc = acc + "<br><br>" + p
        else:
            merged.append(acc)
            acc = p
    merged.append(acc)
    out = []
    max_len, step = 950, 780
    for ch in merged:
        if len(ch) <= max_len:
            out.append(ch)
            continue
        i = 0
        while i < len(ch):
            piece = ch[i : i + max_len]
            if len(piece) >= 20:
                out.append(piece)
            i += step
    return out


def load_knowledge_index(runtime_files=None, *, build_vectors: bool = True):
    chunks_by_key = {}
    vectors_by_key = {}
    all_chunks_local = []
    all_sources_local = []

    files_map = runtime_files if runtime_files is not None else get_runtime_knowledge_files()
    for k, path in files_map.items():
        try:
            if not os.path.exists(path):
                print(f"Warning: File missing at {path}")
                chunks_by_key[k] = []
                vectors_by_key[k] = []
                continue

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = [
                c
                for c in _split_kb_chunks(content, file_key=k)
                if not _chunk_is_agent_instruction_blob(c)
            ]
            chunks_by_key[k] = chunks
            if chunks:
                if build_vectors and not embeddings_disabled():
                    embed_inputs = [_embed_text_for_chunk(c) for c in chunks]
                    vecs = encode_texts(embed_inputs)
                    vectors_by_key[k] = vecs if vecs is not None else []
                else:
                    vectors_by_key[k] = []
                all_chunks_local.extend(chunks)
                all_sources_local.extend([k] * len(chunks))
            else:
                vectors_by_key[k] = []
        except Exception as e:
            print(f"Error loading {path}: {e}")
            chunks_by_key[k] = []
            vectors_by_key[k] = []

    if build_vectors and all_chunks_local and not embeddings_disabled():
        stacked: list[np.ndarray] = []
        for k in chunks_by_key:
            vec = vectors_by_key.get(k)
            ch = chunks_by_key.get(k) or []
            if ch and vec is not None and len(vec) == len(ch):
                stacked.append(np.atleast_2d(vec))
        all_vectors_local = np.vstack(stacked) if stacked else np.array([])
    else:
        all_vectors_local = np.array([])
    return chunks_by_key, vectors_by_key, all_chunks_local, all_vectors_local, all_sources_local


_kb_vectors_built = False


def ensure_kb_vectors():
    """Build embedding index on first KB search — keeps server startup fast and chat unblocked."""
    global kb_vectors_by_key, all_vectors, _kb_vectors_built, _KB_VECTOR_STATUS
    if _kb_vectors_built and all_vectors is not None and len(all_vectors) > 0:
        return
    if embeddings_disabled():
        _kb_vectors_built = True
        _KB_VECTOR_STATUS = "embeddings_disabled"
        log_reasoning("[kb-vector] embeddings disabled — semantic search uses degraded mode")
        return
    if not all_chunks:
        _kb_vectors_built = True
        return
    try:
        print("[kb] building vector index (first search)...", flush=True)
        rebuilt = {}
        for k, chunks in (kb_chunks_by_key or {}).items():
            if chunks:
                embed_inputs = [_embed_text_for_chunk(c) for c in chunks]
                _v = encode_texts(embed_inputs)
                rebuilt[k] = _v if _v is not None else []
            else:
                rebuilt[k] = []
        kb_vectors_by_key = rebuilt
        all_vectors = _build_flat_vectors_from_keys()
        _kb_vectors_built = bool(len(all_vectors) > 0)
        _KB_VECTOR_STATUS = (
            f"lazy_build chunks={len(all_chunks)} vectors={len(all_vectors)}"
        )
        print("[kb] vector index ready", flush=True)
    except Exception as exc:
        print(f"[kb] vector index skipped: {exc}", flush=True)
        _kb_vectors_built = True
        _KB_VECTOR_STATUS = f"build_failed:{exc}"


def refresh_knowledge_cache(*, build_vectors: bool = True):
    global kb_chunks_by_key, kb_vectors_by_key, all_chunks, all_vectors, all_chunk_sources, _SYSTEM_MESSAGES, _KB_SNAPSHOT, _cache, _kb_vectors_built, _KB_VECTOR_STATUS
    with _KB_REFRESH_LOCK:
        runtime_files = get_runtime_knowledge_files()
        _kb_vectors_built = False
        kb_chunks_by_key, kb_vectors_by_key, all_chunks, all_vectors, all_chunk_sources = load_knowledge_index(
            runtime_files, build_vectors=build_vectors
        )
        _kb_vectors_built = bool(
            build_vectors
            and not embeddings_disabled()
            and all_vectors is not None
            and len(all_vectors) > 0
        )
        _KB_SNAPSHOT = _compute_kb_snapshot(runtime_files)
        file_n = len([k for k, ch in (kb_chunks_by_key or {}).items() if ch])
        chunk_n = len(all_chunks or [])
        if embeddings_disabled():
            emb_state = "disabled"
        elif _kb_vectors_built:
            emb_state = "ready"
        elif build_vectors:
            emb_state = "empty"
        else:
            emb_state = "text_only"
        _KB_VECTOR_STATUS = (
            f"refreshed files={file_n} chunks={chunk_n} embeddings={emb_state} "
            f"snapshot={_KB_SNAPSHOT[:48]}"
        )
        log_reasoning(f"[kb-vector] vector_update_status={_KB_VECTOR_STATUS}")
        try:
            sys_path = runtime_files.get("system_messages")
            if sys_path and os.path.exists(sys_path):
                with open(sys_path, "r", encoding="utf-8") as f:
                    _SYSTEM_MESSAGES = _parse_system_messages(f.read())
            else:
                _SYSTEM_MESSAGES = {}
        except Exception as e:
            print("System messages reload error:", e)
        try:
            _cache.clear()
        except Exception:
            pass

_KB_VECTORS_STALE = False


def ensure_knowledge_cache_fresh():
    """
    Reload KB text immediately when admin/files change (deleted chunks vanish at once).
    Rebuild embeddings asynchronously — never block /chat on full encode.
    """
    global _KB_SNAPSHOT, _kb_vectors_built, _KB_VECTORS_STALE
    runtime_files = get_runtime_knowledge_files()
    latest_snapshot = _compute_kb_snapshot(runtime_files)
    if latest_snapshot == _KB_SNAPSHOT:
        return
    log_reasoning(
        "[kb-vector] knowledge snapshot changed — text reload now, vectors async"
    )
    acquired = _KB_REFRESH_LOCK.acquire(timeout=2.0)
    if not acquired:
        log_reasoning("[kb-vector] text reload skipped — refresh lock busy")
        _KB_VECTORS_STALE = True
        return
    try:
        refresh_knowledge_cache(build_vectors=False)
    finally:
        _KB_REFRESH_LOCK.release()
    _KB_VECTORS_STALE = True
    try:
        _cache.clear()
    except Exception:
        pass
    _schedule_kb_vector_rebuild_async()


_KB_REFRESH_SCHEDULED = False


def _schedule_kb_vector_rebuild_async() -> None:
    """Background embedding rebuild after admin KB file add/update/delete."""
    global _KB_REFRESH_SCHEDULED, _KB_VECTORS_STALE
    if _KB_REFRESH_SCHEDULED or not _KB_VECTORS_STALE:
        return
    _KB_REFRESH_SCHEDULED = True
    log_reasoning("[kb-vector] async vector rebuild scheduled")

    def _bg_refresh():
        global _KB_REFRESH_SCHEDULED, _KB_VECTORS_STALE
        try:
            with _KB_REFRESH_LOCK:
                refresh_knowledge_cache(build_vectors=True)
            _KB_VECTORS_STALE = False
            try:
                _cache.clear()
            except Exception:
                pass
            log_reasoning("[kb-vector] async vector rebuild complete")
        except Exception as exc:
            log_reasoning(f"[kb-vector] async vector rebuild failed: {exc}")
        finally:
            _KB_REFRESH_SCHEDULED = False

    try:
        import threading

        threading.Thread(
            target=_bg_refresh, daemon=True, name="kb-vector-rebuild"
        ).start()
    except Exception:
        _KB_REFRESH_SCHEDULED = False
        try:
            with _KB_REFRESH_LOCK:
                refresh_knowledge_cache(build_vectors=True)
            _KB_VECTORS_STALE = False
        except Exception as exc:
            log_reasoning(f"[kb-vector] sync vector rebuild fallback failed: {exc}")


def _schedule_kb_refresh_if_stale() -> None:
    """Non-blocking KB sync — text index first, vectors in background."""
    ensure_knowledge_cache_fresh()


_SYSTEM_MESSAGES = {}
# Text index at import; embedding vectors built once in app._warmup_ai_on_startup().
refresh_knowledge_cache(build_vectors=False)

INTERNAL_KB_KEYS = {
    "welfog_api",
    "welfog_api_order_history",
    "welfog_api_order_id",
    "welfog_api_order_tracking",
    "welfog_api_order_tracking_examples",
    "welfog_api_pincode_delivery",
    "welfog_api_product_search",
    "welfog_api_routing_master",
    "system_messages",
}


def get_customer_kb_keys():
    return [k for k in get_runtime_knowledge_files().keys() if not _is_internal_kb_key(k)]


def build_kb_catalog_for_brain_prompt(*, max_files: int = 32, blurb_chars: int = 140) -> str:
    """
    Auto-built catalog of admin knowledge files for the routing LLM.
    Any .txt added/updated via admin panel appears here — no code change for new topics.
    """
    _schedule_kb_refresh_if_stale()
    lines: list[str] = []
    for k in get_customer_kb_keys()[:max_files]:
        blurb = ""
        for chunk in (kb_chunks_by_key.get(k) or [])[:2]:
            plain = _plain_chunk_for_embed(chunk).strip()
            if plain:
                blurb = re.sub(r"\s+", " ", plain)[:blurb_chars]
                break
        if not blurb:
            path = get_runtime_knowledge_files().get(k)
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        blurb = re.sub(r"\s+", " ", f.read().strip())[:blurb_chars]
                except OSError:
                    blurb = ""
        lines.append(f'  "{k}": {blurb or "(admin knowledge — read file at answer time)"}')
    return "\n".join(lines)


def resolve_brain_kb_keys(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    *,
    max_files: int = 4,
    conversation_context: str = "",
) -> list[str]:
    """
    Pick admin KB files: validate brain kb_keys, else semantic search across all customer .txt files.
    Works for any language — admin panel CRUD on knowledge needs no deploy.
    """
    ensure_knowledge_cache_fresh()
    route = route or {}
    customer = set(get_customer_kb_keys())
    raw_brain_keys = [k for k in (route.get("kb_keys") or []) if str(k).strip()]
    brain_keys = [
        k
        for k in raw_brain_keys
        if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS and k in customer
    ]
    if brain_keys:
        merged = list(dict.fromkeys(brain_keys))
        if len(merged) < len(raw_brain_keys):
            extra = resolve_kb_keys_for_question(
                original_msg or (route.get("user_meaning") or ""),
                msg_en or (route.get("user_meaning") or ""),
                suggested_keys=merged,
                max_files=max_files,
                conversation_context=conversation_context,
                ai_route=route,
            )
            if extra:
                merged = list(dict.fromkeys(merged + extra))[:max_files]
        return merged[:max_files]

    # Brain may suggest keys that are not real admin files — semantic pick, not company default.
    if raw_brain_keys:
        semantic_keys = resolve_kb_keys_for_question(
            original_msg or (route.get("user_meaning") or ""),
            msg_en or (route.get("user_meaning") or ""),
            suggested_keys=None,
            max_files=max_files,
            conversation_context=conversation_context,
            ai_route=route,
        )
        if semantic_keys:
            return semantic_keys[:max_files]

    intent = (route.get("intent") or "").strip().lower()
    _intent_files = {
        "payment": ["payment"],
        "refund": ["refund"],
        "general": ["company"],
        "seller": ["company", "faqs"],
    }
    intent_keys = [
        k
        for k in _intent_files.get(intent, [])
        if k not in INTERNAL_KB_KEYS and k in customer
    ]
    if not intent_keys:
        files_map = get_runtime_knowledge_files()
        intent_keys = [
            k
            for k in _intent_files.get(intent, [])
            if k in files_map and k not in INTERNAL_KB_KEYS
        ]
    if intent_keys:
        return list(dict.fromkeys(intent_keys))[:max_files]

    keys = resolve_kb_keys_for_question(
        original_msg or (route.get("user_meaning") or ""),
        msg_en or (route.get("user_meaning") or ""),
        suggested_keys=None,
        max_files=max_files,
        conversation_context=conversation_context,
        ai_route=route,
    )
    if keys:
        return keys[:max_files]
    return list(customer)[: min(3, max_files)]


def get_support_contact_kb_keys():
    """
    Prefer these files when the user asks for customer-care phone/email.
    Matches support.txt, customer_support.txt, contacts.txt, customer_care*.txt, etc.
    (admin-controlled; no hardcoded phone/email values in code).
    """
    out = []
    for k in get_customer_kb_keys():
        nk = k.lower()
        if any(
            x in nk
            for x in (
                "support",
                "helpline",
                "helpdesk",
                "contact",
                "customer_care",
                "cust_care",
            )
        ):
            out.append(k)
        elif "care" in nk and any(y in nk for y in ("customer", "welfog", "user", "buyer")):
            out.append(k)
    # stable order for caching / tests
    return sorted(set(out))


def read_concatenated_kb_file_contents(keys: list) -> str:
    """Raw UTF-8 text from knowledge files — always read live from disk after cache check."""
    ensure_knowledge_cache_fresh()
    if not keys:
        return ""
    files_map = get_runtime_knowledge_files()
    parts = []
    for k in keys:
        path = files_map.get(k)
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                body = (f.read() or "").strip()
            if body:
                parts.append(f"=== FILE key={k} path={os.path.basename(path)} ===\n{body}")
        except OSError:
            continue
    return "\n\n".join(parts).strip()


_KB_QUERY_STOPWORDS = frozenset(
    {
        "kya", "ky", "hai", "hain", "ho", "the", "a", "an", "and", "or", "to", "for",
        "welfog", "me", "mujhe", "mere", "mera", "apna", "apni", "ke", "ki", "ka", "ko",
        "se", "par", "pe", "per", "bta", "bata", "btao", "de", "do", "dena", "dede",
        "please", "pls", "bhai", "sir", "na", "nahi", "nhi", "what", "how", "is", "are",
    }
)


def _query_tokens_for_kb_filter(text: str) -> set[str]:
    tl = f" {(text or '').lower()} "
    tokens = set(re.findall(r"[a-z0-9]{3,}", tl))
    return {t for t in tokens if t not in _KB_QUERY_STOPWORDS}


def _extract_kb_sections_from_blob(blob: str) -> list[tuple[str, str]]:
    """
    Split any admin .txt file into sections by headings (lines ending with ':')
    or numbered blocks — no hardcoded topic names.
    """
    if not (blob or "").strip():
        return []
    sections: list[tuple[str, str]] = []
    title = ""
    lines: list[str] = []
    for raw in blob.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if lines:
                sections.append((title, "\n".join(lines).strip()))
                lines = []
            continue
        if stripped == ":":
            continue
        is_heading = bool(
            re.match(r"^[A-Z0-9][^:\n]{2,100}:\s*$", stripped)
            or re.match(r"^\d+\)\s+", stripped)
            or (stripped.endswith(":") and len(stripped) < 90 and not stripped.startswith("-"))
            or (stripped.endswith("?") and 12 <= len(stripped) <= 220)
        )
        if is_heading:
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = stripped.rstrip(":").strip()
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    if not sections and blob.strip():
        sections.append(("", blob.strip()))
    return sections


def _embedding_similarity(question: str, text: str) -> float:
    """Semantic match between user question and KB section — no topic hardcoding."""
    if not (question or "").strip() or not (text or "").strip():
        return 0.0
    try:
        qv = encode_texts([question[:500]])
        tv = encode_texts([text[:900]])
        return float(cosine_similarity(qv, tv)[0][0])
    except Exception:
        return 0.0


def _question_phrase_hints(question: str) -> list[str]:
    """Phrases from the user's question for line-level filtering (not a fixed topic list)."""
    q = f" {(question or '').lower()} "
    hints: list[str] = []
    words = re.findall(r"[a-z0-9]+", q)
    for t in _query_tokens_for_kb_filter(question):
        if len(t) >= 4:
            hints.append(t)
    for i in range(len(words) - 1):
        if len(words[i]) >= 3 and len(words[i + 1]) >= 3:
            hints.append(f"{words[i]} {words[i + 1]}")
    return hints


def _infer_kb_answer_budget(question: str) -> tuple[int, int, int]:
    """Short question → short answer. Returns (max_chars, max_lines, max_sections)."""
    word_count = len(re.findall(r"\w+", question or ""))
    if word_count <= 8:
        return 480, 3, 1
    if word_count <= 14:
        return 620, 4, 2
    return 780, 5, 3


def _score_kb_section_relevance(question: str, title: str, body: str) -> float:
    """Rank KB sections by token overlap + embedding — works for any admin-added heading."""
    tokens = _query_tokens_for_kb_filter(question)
    title_low = (title or "").lower()
    body_low = (body or "").lower()
    score = 0.0
    for t in tokens:
        if t in title_low:
            score += 2.8
        elif t in body_low:
            score += 1.0
    block = f"{title}: {body}".strip() if title else (body or "").strip()
    if title:
        score += _embedding_similarity(question, title) * 5.0
    if block:
        score += _embedding_similarity(question, block) * 3.5
    return score


def _strip_leading_faq_question(text: str) -> str:
    """Drop echoed FAQ question line — user already asked it; show answer only."""
    raw = (text or "").strip()
    if not raw:
        return ""

    # Inline: "Question?<br>Answer" (common in FAQ chunks)
    inline = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I).strip()
    for pat in (
        r"^(.{10,240}\?)\s*:\s*(.+)$",
        r"^(.{10,240}\?)\s*\n+\s*(.+)$",
        r"^(.{10,240}\?)\s+(.{15,})$",
    ):
        m = re.match(pat, inline, re.DOTALL)
        if m:
            return m.group(2).strip()

    lines = [ln.strip() for ln in inline.splitlines() if ln.strip()]
    if not lines:
        return inline
    first = lines[0]
    if _looks_like_faq_question_line(first) or _looks_like_kb_metadata_label(first):
        rest = "\n".join(lines[1:]).strip()
        if rest:
            return rest
    return inline


def _strip_kb_policy_heading_prefix(text: str) -> str:
    """Drop echoed KB section titles (seller login / create headings) — keep steps only."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return (text or "").strip()
    has_steps = any(
        re.match(r"^\d+\)", ln.strip()) or ln.strip().startswith(("-", "•"))
        for ln in lines
    )
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s == ":":
            continue
        if re.match(r"^Seller\s+.+:\s*$", s, re.I) and len(s) < 120:
            continue
        m = re.match(r"^(Seller\s+[^:]+):\s*(.+)$", s, re.I)
        if m and len(m.group(1)) < 100:
            rest = m.group(2).strip()
            if rest:
                out.append(rest)
                continue
        if _looks_like_faq_question_line(s) and not re.match(r"^\d+\)", s) and not s.startswith(("-", "•")):
            continue
        if has_steps and len(s) < 100 and not re.match(r"^\d+\)", s) and not s.startswith(("-", "•")):
            if re.search(r"\b(supplier|vendor|seller account)\b", s, re.I) or s.endswith(")"):
                continue
        out.append(ln)
    return "\n".join(out).strip() if out else (text or "").strip()


def _strip_trailing_faq_questions(text: str) -> str:
    """Remove trailing orphan FAQ question headings from filtered KB blobs."""
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"(?:\s*[A-Z][^?\n]{8,220}\?:\s*)+$", "", t).strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    while lines and (
        _looks_like_faq_question_line(lines[-1])
        or _looks_like_kb_metadata_label(lines[-1])
        or (lines[-1].endswith(":") and len(lines[-1]) < 90)
    ):
        lines.pop()
    return "\n".join(lines).strip()


def _format_kb_section_block(title: str, body: str) -> str:
    """Join admin KB section — never prefix FAQ questions as 'Question?: answer'."""
    t = (title or "").strip()
    b = (body or "").strip()
    if _looks_like_kb_metadata_label(t):
        t = ""
    if not b:
        if _looks_like_faq_question_line(t):
            return ""
        return t
    if not t or _looks_like_faq_question_line(t):
        return b
    if b.lower().startswith(t.lower()):
        return b
    if len(t) < 80 and not t.endswith("?"):
        return f"{t}: {b}"
    return b


def _should_show_kb_title(title: str, user_msg: str = "") -> bool:
    if not (title or "").strip():
        return False
    if _looks_like_faq_question_line(title) or _looks_like_kb_metadata_label(title):
        return False
    if user_msg and _token_overlap_ratio(title, user_msg) >= 0.42:
        return False
    return True


def _token_overlap_ratio(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def polish_faq_reply_for_customer(reply: str, user_msg: str = "") -> str:
    """
    Remove echoed question from FAQ replies (KB chunk echo or LLM restating the ask).
    Handles **Question?** Answer, <b>Question?</b>, and plain duplicate first sentence.
    """
    text = (reply or "").strip()
    if not text:
        return text

    # Markdown / HTML bold question prefix (with optional colon)
    for _ in range(3):
        new = re.sub(
            r"^(?:\*\*|__)(.+?\?)(?:\*\*|__)\s*:?\s*",
            "",
            text,
            count=1,
            flags=re.DOTALL,
        )
        if new == text:
            break
        text = new.strip()
    text = re.sub(
        r"^(?:<b>|<strong>)\s*(.+?\?)\s*(?:</b>|</strong>)\s*:?\s*",
        "",
        text,
        count=1,
        flags=re.I | re.DOTALL,
    ).strip()

    text = _strip_leading_faq_question(text)
    text = _strip_trailing_faq_questions(text)
    text = _strip_kb_policy_heading_prefix(text)

    # Drop metadata keyword lines leaked from support/contact files
    if "<" in text:
        plain_lines = re.sub(r"<br\s*/?>", "\n", text, flags=re.I).splitlines()
    else:
        plain_lines = text.splitlines()
    kept = [ln for ln in plain_lines if ln.strip() and not _looks_like_kb_metadata_label(ln)]
    if kept:
        if "<" in text:
            text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
            text = "\n".join(
                ln for ln in text.splitlines() if ln.strip() and not _looks_like_kb_metadata_label(ln)
            ).strip()
        else:
            text = "\n".join(kept).strip()

    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    user_plain = re.sub(r"\s+", " ", (user_msg or "").strip())

    # First sentence ending with ? duplicates the user's question
    m = re.match(r"^(.+?\?)\s*(.+)$", plain, re.DOTALL)
    if m and user_plain:
        q_part, rest = m.group(1).strip(), m.group(2).strip()
        if rest and _token_overlap_ratio(q_part, user_plain) >= 0.5:
            # Drop echoed sentence from original text (preserve HTML if any)
            if "<" in text:
                text = re.sub(
                    r"^(.+?\?)\s*",
                    "",
                    re.sub(r"<[^>]+>", " ", text, count=1),
                    count=1,
                    flags=re.DOTALL,
                ).strip()
                if not text:
                    text = rest
            else:
                text = rest

    # "If you're facing X —" restatement without ? but mirrors user ask
    if user_plain and plain.lower().startswith(("if you", "if you're", "if youre")):
        if _token_overlap_ratio(plain.split(".")[0], user_plain) >= 0.45:
            parts = re.split(r"(?<=[.!?])\s+", plain, maxsplit=1)
            if len(parts) > 1 and len(parts[1]) > 20:
                text = parts[1]

    return text.strip()


def _faq_answer_text_from_chunk(chunk: str) -> str:
    return polish_faq_reply_for_customer(_strip_leading_faq_question(chunk), "")


def _kb_excerpt_is_english_for_customer(plain: str, reply_lang: str) -> bool:
    """True when KB excerpt should be grounded/translated for this customer."""
    from services.translation_service import (
        _detect_script_language,
        _hinglish_marker_count,
        _latin_script_only,
        _looks_english_only_latin,
    )

    text = (plain or "").strip()
    text = (
        text.replace("\u2026", "...")
        .replace("—", "-")
        .replace("–", "-")
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("’", "'")
    )
    if not text:
        return False
    rl = (reply_lang or "").strip().lower()
    if rl == "hinglish":
        if not _latin_script_only(text) or _detect_script_language(text):
            return False
        return _hinglish_marker_count(text.lower()) < 2
    return _looks_english_only_latin(text)


def _finalize_kb_customer_reply(body: str, original_msg: str, lang: str) -> str:
    """KB-sourced reply — localize to customer's language/script when KB text is English."""
    from services.translation_service import (
        _looks_english_only_latin,
        finalize_customer_reply,
        resolve_customer_reply_lang,
    )

    if not (body or "").strip():
        return ""
    rl = resolve_customer_reply_lang(original_msg, lang)
    plain = re.sub(r"<[^>]+>", " ", body or "")
    plain = re.sub(r"\s+", " ", plain).strip()
    if (
        _kb_multilingual_answer_enabled()
        and _customer_needs_kb_localization(original_msg, rl)
        and plain
        and _kb_excerpt_is_english_for_customer(plain, rl)
    ):
        grounded = _ground_kb_excerpt_for_customer(
            original_msg,
            plain,
            reply_lang=rl,
        )
        if grounded and grounded.strip():
            return finalize_customer_reply(grounded, original_msg, rl)
    return finalize_customer_reply(body, original_msg, rl)


def _title_from_filtered_kb_text(filtered: str) -> str:
    if not (filtered or "").strip():
        return ""
    first = filtered.splitlines()[0].strip()
    if _looks_like_faq_question_line(first) or _looks_like_kb_metadata_label(first):
        return ""
    if first.endswith(":") and len(first) < 100:
        title = first[:-1].strip()
        if _looks_like_faq_question_line(title) or _looks_like_kb_metadata_label(title):
            return ""
        return title
    return ""


def _kb_focus_profile(question: str) -> tuple[set[str], set[str], int]:
    """
    Generic intent disambiguation only (place order vs return vs track).
    No hardcoded file/topic names — admin KB headings drive content selection.
    """
    q = f" {(question or '').lower()} "
    include: set[str] = set()
    exclude: set[str] = set()
    _, _, max_sections = _infer_kb_answer_budget(question)

    asks_place_order = any(
        x in q
        for x in (
            "how to place order", "place order", "order kaise", "order kese", "checkout",
            "add to cart", "new order", "order karu", "order karna",
        )
    )
    asks_return = any(x in q for x in ("return", "refund", "replacement", "wrong item", "damaged"))
    asks_track = any(x in q for x in ("track", "tracking", "order status", "where is my order"))
    asks_history = any(x in q for x in ("order history", "past order", "my orders", "purane order"))
    asks_refund_timeline = any(
        x in q
        for x in (
            "money back", "get my money", "when will i get", "how many days", "how long",
            "kitne din", "kab milega", "kab aayega", "kab milenge", "timeline", "processing time",
            "business days", "vapasi", "paise wapas", "paise kab",
        )
    )

    if asks_refund_timeline and not (asks_place_order or asks_track):
        include |= {"refund", "processing", "business days", "money back", "returned item"}
        exclude |= {
            "fraud",
            "fake caller",
            "password",
            "otp",
            "cvv",
            "prize",
            "place order",
            "tracking",
            "bank account update",
            "payment methods",
            "payment method does",
        }
        max_sections = min(max_sections, 2)
    elif asks_place_order and not (asks_return or asks_track or asks_history):
        include |= {"place order", "checkout", "add to cart", "new order", "order kaise"}
        exclude |= {"return", "refund", "replacement", "tracking", "track", "order history"}
        max_sections = min(max_sections, 2)
    elif asks_return and not (asks_place_order or asks_track):
        include |= {"return", "refund", "replacement", "damaged", "wrong item"}
        exclude |= {"place order", "checkout", "tracking", "order history"}
        max_sections = min(max_sections, 2)
    elif asks_track and not (asks_place_order or asks_return):
        include |= {"track", "tracking", "order status", "shipment", "courier"}
        exclude |= {"place order", "checkout", "return", "refund", "order history"}
        max_sections = min(max_sections, 2)
    elif asks_history and not (asks_place_order or asks_return or asks_track):
        include |= {"order history", "my orders", "past order", "invoice"}
        exclude |= {"place order", "checkout", "return", "refund", "tracking"}
        max_sections = min(max_sections, 2)

    asks_seller_create = any(
        x in q
        for x in (
            "create seller", "seller account", "become seller", "become a seller",
            "register", "registration", "seller bana", "seller ban", "account create",
            "account bana", "sell on", "new seller", "how to create",
        )
    )
    asks_grievance = any(
        x in q
        for x in (
            "grievance officer",
            "grievance email",
            "chief compliance",
            "formal complaint",
            "legal grievance",
        )
    )
    if asks_grievance:
        include |= {"grievance", "compliance", "tripti", "grievance@"}
        exclude |= {"place order", "track order", "checkout", "my orders"}
        max_sections = min(max_sections, 2)

    asks_seller_login = any(
        x in q
        for x in (
            "login problem", "login fail", "login nahi", "login nhi", "login error",
            "sign in", "log in", "otp", "cannot login", "can't login",
        )
    ) or ("login" in q and "seller" in q and not asks_seller_create)
    if asks_seller_create and not asks_seller_login:
        include |= {"create", "register", "registration", "become seller", "seller account"}
        exclude |= {"login problem", "login fail", "otp", "clear app cache", "same mobile"}
        max_sections = min(max_sections, 2)
    elif asks_seller_login and not asks_seller_create:
        include |= {"login", "otp", "sign in", "login problem"}
        exclude |= {"how to create", "register", "registration", "become seller"}

    return include, exclude, max_sections


def _filter_kb_blob_for_question(blob: str, question: str, *, max_chars: int | None = None) -> str:
    """
    Keep only KB sections/lines that match the user's question.
    Fully driven by live admin .txt content + embeddings — no topic names in code.
    """
    display = _strip_kb_blob_for_display(blob)
    if not display.strip():
        return ""
    budget_chars, budget_lines, budget_sections = _infer_kb_answer_budget(question)
    if max_chars is None:
        max_chars = budget_chars

    sections = _extract_kb_sections_from_blob(display)
    if not sections:
        hit = best_kb_hit(question, min_score=0.20)
        if hit and hit.get("chunk"):
            plain = re.sub(r"<br\s*/?>", "\n", hit["chunk"])
            return _kb_plain_excerpt(plain, max_chars=max_chars)
        return _kb_plain_excerpt(display, max_chars=max_chars)

    include_terms, exclude_terms, max_sections = _kb_focus_profile(question)
    max_sections = min(max_sections, budget_sections)
    scored: list[tuple[float, str, str]] = []
    for title, body in sections:
        if not body and not title:
            continue
        hay = f"{title} {body}".lower()
        score = _score_kb_section_relevance(question, title, body)
        if include_terms:
            hit = sum(1 for t in include_terms if t in hay)
            miss = sum(1 for t in exclude_terms if t in hay)
            score += (2.5 * hit) - (1.7 * miss)
        if score > 0.05:
            scored.append((score, title, body))

    if not scored:
        hit = best_kb_hit(question, min_score=0.18)
        if hit and hit.get("chunk"):
            plain = re.sub(r"<br\s*/?>", "\n", hit["chunk"])
            return _kb_plain_excerpt(plain, max_chars=max_chars)
        if len(sections) == 1:
            t, b = sections[0]
            block = f"{t}:\n{b}" if t else b
            return _kb_plain_excerpt(block, max_chars=max_chars)
        return _kb_plain_excerpt(display, max_chars=max_chars)

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score = scored[0][0]
    # Drop sections far below the best match — avoids dumping unrelated paragraphs.
    scored = [(s, t, b) for s, t, b in scored if s >= top_score * 0.45]

    parts: list[str] = []
    total = 0
    for _, title, body in scored[:max_sections]:
        block = _format_kb_section_block(title, body)
        if not block.strip():
            continue
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    out = "\n\n".join(parts).strip()
    if not out:
        return ""

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return out[:max_chars] if len(out) > max_chars else out

    phrase_hints = _question_phrase_hints(question)
    short_tokens = _query_tokens_for_kb_filter(question)
    scored_lines: list[tuple[float, str]] = []
    for ln in lines:
        if _looks_like_kb_metadata_label(ln):
            continue
        if _looks_like_faq_question_line(ln) and not any(
            x in ln.lower()
            for x in (" allows ", " accepts ", " offers ", " yes,", " welfog has ", " to ")
        ):
            continue
        low = f" {ln.lower()} "
        score = 0.0
        for t in short_tokens:
            if t in low:
                score += 1.2
        for phrase in phrase_hints:
            if phrase in low:
                score += 1.6
        if re.search(r"\b\d{2,}\b", ln) or "@" in ln:
            score += 0.5
        score += _embedding_similarity(question, ln) * 1.5
        if score > 0:
            scored_lines.append((score, ln))

    if not scored_lines:
        return _kb_plain_excerpt(out, max_chars=max_chars)

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    kept: list[str] = []
    total = 0
    for _, ln in scored_lines:
        if ln in kept:
            continue
        if total + len(ln) > max_chars and kept:
            break
        kept.append(ln)
        total += len(ln) + 1
        if len(kept) >= budget_lines:
            break
    compact = "\n".join(kept).strip()
    return compact[:max_chars] if len(compact) > max_chars else compact


def build_kb_retrieval_query(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> str:
    """
    Multilingual retrieval query for embedding search — AI meaning + English gloss + context.
    No keyword lists; admin KB files matched by semantic similarity.
    """
    parts: list[str] = []
    meaning = ((ai_route or {}).get("user_meaning") or "").strip()
    if meaning:
        parts.append(meaning)
    en = (msg_en or "").strip()
    raw = (original_msg or "").strip()
    if en and en.lower() != raw.lower():
        parts.append(en)
    if raw:
        parts.append(raw)

    merged = " — ".join(p for p in parts if p).strip()
    route = ai_route or {}
    cat = (route.get("intent") or "").strip().lower()
    kb_keys = route.get("kb_keys") or []
    if (not cat or cat in ("general", "order_history")) and kb_keys:
        cat = str(kb_keys[0]).strip().lower()
    if cat and cat not in ("general", "order_history", "product", "order"):
        merged = f"{merged} — topic: {cat}".strip(" —")
    if route.get("_preflight_kb"):
        return _normalize_retrieval_query(merged or raw)

    try:
        from services.query_understanding import _is_short_or_vague_message, _last_user_snippets

        if conversation_context and _is_short_or_vague_message(original_msg or msg_en):
            prior_snippets = _last_user_snippets(conversation_context)
            prior = " — ".join(s for s in prior_snippets if s).strip()
            if prior and prior.lower() not in (merged or "").lower():
                merged = f"{prior} — {merged}".strip(" —") if merged else prior
    except ImportError:
        pass
    return _normalize_retrieval_query(merged or raw)


def log_kb_retrieval(
    *,
    query_intent: str = "",
    query_meaning: str = "",
    retrieval_query: str = "",
    matched_file: str = "",
    selected_category: str = "",
    similarity_score: float = 0.0,
    selected_chunks: list[dict] | None = None,
    vector_update_status: str = "",
) -> None:
    """Structured KB retrieval log for debugging wrong-doc selection."""
    from utils.reasoning_log import chat_log

    chunks = selected_chunks or []
    tops = chunks[:4]
    file_set = list(dict.fromkeys([matched_file] + [str(c.get("source") or "") for c in tops if c.get("source")]))
    file_set = [f for f in file_set if f]
    previews: list[str] = []
    for c in tops:
        src = str(c.get("source") or "?")
        sc = float(c.get("score") or 0)
        snippet = re.sub(r"\s+", " ", (c.get("chunk") or "")[:72]).strip()
        previews.append(f"{src}({sc:.3f}):{snippet}")
    vstat = vector_update_status or get_kb_vector_status()
    msg = (
        f"[kb-retrieval] query_meaning={(query_meaning or '-')[:120]} "
        f"query_intent={query_intent or '-'} "
        f"selected_category={selected_category or query_intent or '-'} "
        f"selected_file={','.join(file_set[:3]) or '-'} "
        f"similarity_score={similarity_score:.3f} "
        f"retrieved_chunks={len(tops)} "
        f"vector_update_status={vstat} "
        f"top={' | '.join(previews) if previews else '-'}"
    )
    log_reasoning(msg)
    chat_log(msg)


def log_kb_pipeline_complete(
    *,
    query: str = "",
    language: str = "",
    intent: str = "",
    route: str = "",
    retrieved_chunks: int = 0,
    similarity_score: float = 0.0,
    response_time_ms: float = 0.0,
    source: str = "",
) -> None:
    """Structured KB pipeline metrics — query, retrieval, route, latency."""
    from utils.reasoning_log import chat_log

    msg = (
        f"[kb-pipeline] query={(query or '-')[:120]!r} "
        f"language={language or '-'} intent={intent or '-'} "
        f"route={route or '-'} retrieved_chunks={retrieved_chunks} "
        f"similarity_score={similarity_score:.3f} "
        f"response_time_ms={response_time_ms:.0f} source={source or '-'}"
    )
    log_reasoning(msg)
    chat_log(msg)


def format_direct_reply_from_kb_hit(
    hit: dict,
    original_msg: str,
    *,
    reply_lang: str = "",
    retrieval_query: str = "",
    fast_lane: bool = False,
    conversation_context: str = "",
) -> str:
    """
    Answer from one retrieved KB chunk — no answer-rewrite LLM.
    Translates to customer language when needed.
    """
    if not isinstance(hit, dict):
        return ""
    chunk = (hit.get("chunk") or "").strip()
    if not chunk:
        return ""
    from services.translation_service import customer_reply_language

    rl = reply_lang or customer_reply_language(original_msg)
    score = float(hit.get("score") or 0)
    src = str(hit.get("source") or "general")
    if not fast_lane:
        log_kb_retrieval(
            query_intent=src,
            retrieval_query=_normalize_retrieval_query(retrieval_query or original_msg),
            matched_file=src,
            selected_category=src,
            similarity_score=score,
            selected_chunks=[hit],
        )
    excerpt = _kb_plain_excerpt(
        _faq_answer_text_from_chunk(re.sub(r"<br\s*/?>", "\n", chunk)),
        max_chars=_infer_kb_answer_budget(retrieval_query or original_msg)[0],
    )
    if not excerpt.strip():
        return ""
    from services.translation_service import resolve_customer_reply_lang

    rl_resolved = resolve_customer_reply_lang(original_msg, rl)
    src_list = [src] if src and src != "general" else None
    if _kb_ai_focus_answer_enabled() and excerpt.strip():
        grounded = _ground_kb_excerpt_for_customer(
            original_msg,
            excerpt,
            conversation_context=conversation_context,
            reply_lang=rl_resolved,
            kb_sources=src_list,
        )
        if grounded and grounded.strip():
            return grounded
    body = _plain_text_to_html_body(excerpt)
    if not body:
        return ""
    if fast_lane:
        body = polish_faq_reply_for_customer(body, original_msg)
        return _finalize_kb_customer_reply(body, original_msg, rl_resolved)
    body = polish_faq_reply_for_customer(body, original_msg)
    return _finalize_kb_customer_reply(body, original_msg, rl_resolved)


def format_kb_no_information_reply(original_msg: str, *, reply_lang: str = "") -> str:
    """Honest fallback when KB has no matching chunk."""
    from services.translation_service import customer_reply_language, finalize_customer_reply

    rl = reply_lang or customer_reply_language(original_msg)
    en = (
        "I couldn't find this in Welfog's current knowledge base. "
        "Please contact customer support if you need more help."
    )
    return finalize_customer_reply(en, original_msg, rl)


def _keyword_hit_as_semantic(kw: dict | None) -> dict | None:
    """Map keyword fallback hit-count to a 0–1 pseudo score (never mix raw counts with cosine)."""
    if not kw:
        return None
    hits = float(kw.get("score") or 0)
    pseudo = min(0.58, 0.22 + 0.06 * hits)
    return {
        "source": kw.get("source"),
        "chunk": kw.get("chunk"),
        "score": pseudo,
        "match_type": "keyword_fallback",
    }


def _scoped_kb_index(keys: list[str] | None, *, customer_only: bool = True):
    """Build aligned chunk / vector / source lists for search."""
    ensure_kb_vectors()
    if keys:
        scope_keys = [k for k in keys if k in kb_chunks_by_key]
    elif customer_only:
        scope_keys = get_customer_kb_keys()
    else:
        scope_keys = list((kb_chunks_by_key or {}).keys())

    scoped_chunks: list[str] = []
    scoped_vectors_list: list[np.ndarray] = []
    scoped_sources: list[str] = []
    for k in scope_keys:
        ch = kb_chunks_by_key.get(k) or []
        vec = kb_vectors_by_key.get(k)
        if ch and vec is not None and len(ch) == len(vec):
            scoped_chunks.extend(ch)
            scoped_vectors_list.append(np.atleast_2d(vec))
            scoped_sources.extend([k] * len(ch))
    if not scoped_chunks:
        return [], None, []
    scoped_vectors = (
        scoped_vectors_list[0]
        if len(scoped_vectors_list) == 1
        else np.vstack(scoped_vectors_list)
    )
    return scoped_chunks, scoped_vectors, scoped_sources


def _cached_encode_query(q: str):
    """Per-turn query embedding cache — avoids repeated encode under /chat deadline."""
    try:
        from services.chat_flow_telemetry import _TLS

        if getattr(_TLS, "kb_query_embed_key", None) == q:
            cached = getattr(_TLS, "kb_query_embed_vec", None)
            if cached is not None:
                return cached
    except ImportError:
        pass
    qv = encode_texts([q])
    try:
        from services.chat_flow_telemetry import _TLS

        _TLS.kb_query_embed_key = q
        _TLS.kb_query_embed_vec = qv
    except ImportError:
        pass
    return qv


def semantic_kb_search(
    query: str,
    *,
    keys: list[str] | None = None,
    top_n: int = 4,
    min_score: float | None = None,
    customer_only: bool = True,
    log_retrieval: bool = True,
) -> list[dict]:
    """
    Core semantic retrieval — cosine similarity on embeddings only (no keyword matching).
    Keyword fallback runs only when embeddings are disabled/unavailable.
    """
    ensure_knowledge_cache_fresh()
    ensure_kb_vectors()
    floor = float(min_score if min_score is not None else KB_SEMANTIC_MIN_SCORE)
    q = _normalize_retrieval_query(query)
    if not q:
        return []

    scoped_chunks, scoped_vectors, scoped_sources = _scoped_kb_index(keys, customer_only=customer_only)
    if not scoped_chunks:
        return []

    out: list[dict] = []

    if _embeddings_ready() and scoped_vectors is not None and len(scoped_vectors) > 0:
        query_vec = _cached_encode_query(q)
        if query_vec is not None:
            scores = cosine_similarity(query_vec, scoped_vectors)[0]
            indexed = [
                (float(scores[i]), scoped_sources[i], scoped_chunks[i])
                for i in range(len(scoped_chunks))
            ]
            indexed.sort(key=lambda x: x[0], reverse=True)
            seen_norm: set[str] = set()
            for score, src, chunk in indexed:
                if score < floor:
                    break
                norm = _chunk_dedup_key(chunk)
                if norm and norm in seen_norm:
                    continue
                if norm:
                    seen_norm.add(norm)
                out.append({"source": src, "chunk": chunk, "score": score, "match_type": "semantic"})
                if len(out) >= top_n:
                    break

    if not out:
        kw = keyword_kb_hit(
            q,
            keys=keys or (get_customer_kb_keys() if customer_only else None),
            min_hits=2,
        )
        hit = _keyword_hit_as_semantic(kw)
        if hit and float(hit.get("score") or 0) >= floor:
            out = [hit]

    if log_retrieval and out:
        matched_src = str(out[0].get("source") or "")
        log_kb_retrieval(
            query_intent=matched_src or "general",
            query_meaning="",
            retrieval_query=q,
            matched_file=matched_src,
            selected_category=matched_src or "general",
            similarity_score=float(out[0].get("score") or 0),
            selected_chunks=out,
        )
    return out


def _faq_question_line_from_chunk(chunk: str) -> str:
    plain = re.sub(r"<br\s*/?>", "\n", chunk or "").strip()
    if not plain:
        return ""
    return plain.split("\n", 1)[0].strip()


def _faq_question_similarity_boost(question: str, chunk: str) -> float:
    """Rerank by embedding match to the FAQ question line (not keyword lists)."""
    q_line = _faq_question_line_from_chunk(chunk)
    if not q_line or not _looks_like_faq_question_line(q_line):
        return 0.0
    return _embedding_similarity(question, q_line) * 0.42


def promote_route_from_semantic_kb_match(
    route: dict,
    original_msg: str,
    msg_en: str = "",
    *,
    min_score: float = 0.28,
) -> dict | None:
    """
    Strong admin-KB embedding match → lock welfog_support (any language).
    Prevents brain JSON from misrouting FAQ/policy turns to OOD or chitchat.
    """
    out = dict(route or {})
    if out.get("run_catalog_search") or out.get("_product_catalog_locked"):
        return None
    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return None
    try:
        from services.ai_route_semantics import _brain_route_has_shopping_entities
        from utils.helpers import turn_is_obvious_product_shopping_turn

        if _brain_route_has_shopping_entities(
            out, original_msg=original_msg, msg_en=msg_en
        ) or turn_is_obvious_product_shopping_turn(original_msg, msg_en, ""):
            return None
    except ImportError:
        pass

    q = build_kb_retrieval_query(original_msg, msg_en, "", ai_route=out)
    best = retrieve_best_kb_chunk(
        q or combined,
        ai_route=out,
        min_score=max(0.14, min_score - 0.12),
    )
    score = float((best or {}).get("score") or 0)
    if not best or score < min_score:
        return None

    src = str(best.get("source") or "").strip().lower()
    intent = (out.get("intent") or "general").strip().lower()
    if src == "seller":
        intent = "seller"
    elif src == "refund":
        intent = "refund"
    elif src == "payment" and intent not in ("order", "order_history"):
        intent = "payment"
    elif intent not in ("seller", "refund", "payment"):
        intent = "general"

    keys = resolve_brain_kb_keys(out, original_msg, msg_en)
    if src:
        keys = list(dict.fromkeys([*(keys or []), src]))[:4]

    out["intent"] = intent
    out["data_channel"] = "kb"
    out["conversation_scope"] = "welfog_support"
    out["is_welfog_related"] = True
    out["kb_keys"] = keys
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out["scope_reply"] = ""
    out.pop("category_only_browse", None)
    out.pop("category_browse", None)
    out.pop("_product_catalog_locked", None)
    out["_turn_promotions_done"] = True
    return out


def select_best_kb_hit_for_customer_question(
    original_msg: str,
    msg_en: str = "",
    *,
    conversation_context: str = "",
    preferred_keys: list[str] | None = None,
    ai_route: dict | None = None,
    brain_locked_kb: bool = False,
    min_score: float | None = None,
) -> dict | None:
    """
    Pick the strongest embedding match across brain-hinted files + faqs + related admin KB.
    Brain kb_keys are hints — a higher-scoring chunk in another file always wins.
    """
    floor = float(min_score if min_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    filter_q = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    combined = filter_q or f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return None

    valid = [
        k
        for k in (preferred_keys or [])
        if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS
    ][:4]

    key_sets: list[list[str] | None] = []
    if valid:
        key_sets.append(valid)
    if "faqs" in kb_chunks_by_key:
        key_sets.append(["faqs"])
    for hint in ("seller", "privacy", "company", "payment", "refund", "shipping", "support"):
        if hint in kb_chunks_by_key and hint not in valid:
            key_sets.append([hint])
    if not brain_locked_kb:
        key_sets.append(None)

    best_hit: dict | None = None
    best_score = -1.0
    seen: set[str] = set()
    search_floor = max(0.12, floor - 0.12)

    for keys in key_sets:
        hit = retrieve_best_kb_chunk(
            combined,
            keys=keys,
            ai_route=ai_route,
            min_score=search_floor,
        )
        if not hit:
            continue
        norm = _chunk_dedup_key(hit.get("chunk") or "")
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        sc = float(hit.get("score") or 0)
        if sc > best_score:
            best_score = sc
            best_hit = hit

    if best_hit and best_score >= floor:
        return best_hit
    return None


def retrieve_best_kb_chunk(
    query: str,
    *,
    keys: list[str] | None = None,
    ai_route: dict | None = None,
    min_score: float | None = None,
    conflict_check: bool = True,
) -> dict | None:
    """Semantic search + rerank; None when confidence is below answer threshold."""
    floor = float(min_score if min_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    search_floor = max(0.12, floor - 0.1)
    hits = semantic_kb_search(
        query, keys=keys, top_n=8, min_score=search_floor, log_retrieval=False
    )
    return _pick_best_kb_hit_for_query(
        query,
        hits,
        ai_route=ai_route,
        min_rerank_score=floor,
        conflict_check=conflict_check,
    )


def resolve_kb_keys_for_question(
    original_msg: str,
    msg_en: str = "",
    *,
    suggested_keys: list[str] | None = None,
    max_files: int = 4,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> list[str]:
    """
    Pick which admin knowledge files to use — no hardcoded topic→file map.
    Uses Groq kb_keys when valid, else embedding + keyword search across ALL customer .txt files.
    """
    ensure_knowledge_cache_fresh()
    query = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if not query.strip():
        query = f"{original_msg or ''} {msg_en or ''}".strip()
    customer_keys = get_customer_kb_keys()
    if not customer_keys:
        return []

    try:
        from utils.helpers import _text_asks_customer_care_contact

        if _user_requests_grievance_channel(query):
            prefer = [k for k in ("company", "privacy") if k in customer_keys]
            if prefer:
                return prefer[:max_files]
        if _text_asks_customer_care_contact(query) and not _user_requests_grievance_channel(query):
            sc = get_support_contact_kb_keys()
            if sc:
                return sc[:max_files]
    except ImportError:
        pass

    valid_suggested = [
        k for k in (suggested_keys or []) if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS
    ]

    try:
        from services.query_understanding import (
            filter_kb_keys_for_intent,
            infer_kb_query_category,
            scoped_kb_keys_for_retrieval,
        )

        category = infer_kb_query_category(
            original_msg,
            msg_en,
            ai_route=ai_route,
            conversation_context=conversation_context,
        )
        scoped_keys = scoped_kb_keys_for_retrieval(
            category, ai_route=ai_route, user_meaning=query
        )
        if valid_suggested:
            valid_suggested = filter_kb_keys_for_intent(
                valid_suggested,
                category,
                user_meaning=query,
            )
    except ImportError:
        category = "general"
        scoped_keys = None

    try:
        from services.query_understanding import filter_kb_keys_for_intent

        if valid_suggested and not scoped_keys:
            intent_guess = category if category != "general" else ""
            valid_suggested = filter_kb_keys_for_intent(
                valid_suggested,
                intent_guess or "general",
                user_meaning=query,
            )
    except ImportError:
        pass

    ranked: list[tuple[float, str]] = []
    search_keys = [k for k in (scoped_keys if scoped_keys else customer_keys) if k in customer_keys]
    file_rank_floor = min(0.12, KB_CONTEXT_MIN_SCORE)
    cat_l = (category or "general").strip().lower()
    effective_cat = cat_l
    best_per_key: dict[str, float] = {}
    if search_keys:
        hits = semantic_kb_search(
            query,
            keys=search_keys,
            top_n=min(32, max(len(search_keys) * 4, max_files * 6)),
            min_score=file_rank_floor,
            customer_only=False,
            log_retrieval=False,
        )
        for h in hits:
            src = str(h.get("source") or "")
            sc = float(h.get("score") or 0)
            if src in search_keys and sc >= file_rank_floor:
                best_per_key[src] = max(best_per_key.get(src, 0.0), sc)
    for k, score in best_per_key.items():
        kl = k.lower()
        if effective_cat not in ("", "general"):
            if kl == effective_cat or effective_cat in kl:
                score += 0.1
            elif effective_cat in ("refund", "return") and "refund" in kl:
                score += 0.08
            elif effective_cat == "payment" and "payment" in kl:
                score += 0.08
            elif effective_cat == "shipping" and "shipping" in kl:
                score += 0.08
            elif effective_cat == "seller" and "seller" in kl:
                score += 0.08
        ranked.append((score, k))
    ranked.sort(key=lambda x: x[0], reverse=True)
    keys_from_embed = [k for _, k in ranked[:max_files]]
    if effective_cat not in ("", "general"):
        prim = next((k for k in customer_keys if k == effective_cat), "")
        if not prim:
            prim = next((k for k in customer_keys if effective_cat in k), "")
        if prim and best_per_key.get(prim, 0.0) >= file_rank_floor:
            keys_from_embed = [prim] + [k for k in keys_from_embed if k != prim]
            keys_from_embed = keys_from_embed[:max_files]

    if keys_from_embed:
        try:
            hits_preview = top_kb_hits(
                query, keys=keys_from_embed, min_score=0.14, top_n=3, log_retrieval=False
            )
            log_kb_retrieval(
                query_intent=category,
                query_meaning=((ai_route or {}).get("user_meaning") or "").strip(),
                retrieval_query=query,
                matched_file=keys_from_embed[0],
                selected_category=category,
                similarity_score=float(hits_preview[0].get("score") or 0) if hits_preview else 0.0,
                selected_chunks=hits_preview,
            )
        except Exception:
            pass
        if valid_suggested:
            merged = list(dict.fromkeys(valid_suggested + keys_from_embed))
            return merged[:max_files]
        return keys_from_embed

    if valid_suggested:
        return valid_suggested[:max_files]

    best = retrieve_best_kb_chunk(query, keys=customer_keys, ai_route=ai_route, min_score=0.2)
    if best and best.get("source") in customer_keys:
        return [best["source"]]

    return customer_keys[: min(2, len(customer_keys))]


def _ai_response_looks_like_placeholder(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(r"\[insert\b", low)
        or "[phone number from kb]" in low
        or "from kb]" in low
        or "placeholder" in low
    )


def _kb_multilingual_answer_enabled() -> bool:
    return (os.getenv("ENABLE_KB_MULTILINGUAL_ANSWER", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _kb_ai_focus_answer_enabled() -> bool:
    """RAG answer synthesis — LLM filters KB excerpt to only what the user asked."""
    return (os.getenv("ENABLE_KB_AI_FOCUS_ANSWER", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _customer_needs_kb_localization(original_msg: str, lang: str) -> bool:
    from services.translation_service import _looks_english_only_latin, is_hinglish_message

    if is_hinglish_message(original_msg):
        return True
    norm = (lang or "").strip().lower()
    if norm and norm not in ("en", "english"):
        return True
    return not _looks_english_only_latin(original_msg or "")


def _ground_kb_excerpt_for_customer(
    original_msg: str,
    kb_excerpt: str,
    *,
    conversation_context: str = "",
    reply_lang: str = "",
    kb_sources: list[str] | None = None,
) -> str:
    """
    Answer in the customer's language/script from live KB text (FAQ/policy).
    English KB in admin panel → Hindi/Hinglish/Tamil/etc. reply without code changes.
    """
    excerpt = _strip_leading_faq_question(kb_excerpt)
    if not excerpt:
        return ""
    from services.translation_service import (
        customer_reply_language,
        finalize_customer_reply,
        _looks_english_only_latin,
    )

    rl = reply_lang or customer_reply_language(original_msg)
    src_note = ", ".join(kb_sources or []) or "admin knowledge"
    try:
        from services.ai_service import ai_brain_answer

        ai = ai_brain_answer(
            original_msg,
            (
                f"AUTHORITATIVE KNOWLEDGE from live admin file(s): {src_note}\n"
                "RULES: Answer ONLY from this text. Match the customer's language/script (Hinglish = Roman Hindi+English). "
                "Do NOT repeat or restate the user's question — start directly with the answer. "
                "Use ONLY the part of the knowledge that answers THIS specific question — never paste unrelated "
                "sections, headings, or steps (e.g. refund timeline only when they ask when refund comes; "
                "address update only when they ask to change address; delayed package only when they ask about delay — "
                "not return policy). "
                "Give ONLY what was asked — 1-4 short sentences, no extra topics, no steps they did not ask for. "
                "Seller account steps must NOT be confused with buyer checkout account. "
                "Copy phone/email/addresses exactly.\n\n"
                f"{excerpt}"
            ),
            conversation_context=conversation_context,
            reply_lang=rl,
        )
        resp = (ai.get("response") or "").strip() if ai else ""
        if resp and not _ai_response_looks_like_placeholder(resp):
            resp = polish_faq_reply_for_customer(resp, original_msg)
            if resp.strip().startswith("<"):
                return finalize_customer_reply(resp, original_msg, rl)
            return finalize_customer_reply(
                _plain_text_to_html_body(resp) or resp, original_msg, rl
            )
    except Exception:
        pass
    body = polish_faq_reply_for_customer(_plain_text_to_html_body(excerpt) or excerpt, original_msg)
    return finalize_customer_reply(body, original_msg, rl)


def _chunk_is_order_history_howto_faq(chunk: str) -> bool:
    """FAQ answer with app navigation steps — not live list in chat."""
    c = f" {(chunk or '').lower()} "
    if not ("order history" in c or "my orders" in c or "past order" in c):
        return False
    return any(
        x in c
        for x in (
            "log in to your account",
            "log in to your",
            "user icon",
            "top-right",
            "select my orders",
            "go to my orders",
            "how to view",
            "how can i view",
        )
    )


def _kb_chunk_relevance_adjustment(question: str, chunk: str) -> float:
    """
    Semantic rerank nudge — disambiguate similar FAQ/KB chunks (refund timeline vs return policy;
    seller registration vs login help). Meaning-based token checks, not routing.
    """
    q = f" {(question or '').lower()} "
    c = re.sub(r"<br\s*/?>", " ", (chunk or "").lower())
    adj = 0.0

    asks_refund_time = "refund" in q and any(
        x in q
        for x in (
            "kitne din",
            "kab tak",
            "kab aayega",
            "kab aata",
            "how long",
            "how many day",
            "when will",
            "timeline",
            "business day",
            "processed",
            "aata hai",
            "milega",
        )
    )
    if asks_refund_time:
        if "refund" in c and ("processed" in c or "business day" in c):
            adj += 0.14
        if "return policy" in c or "allows returns within" in c:
            adj -= 0.1
        if "deliver my order" in c or "delivery times" in c:
            adj -= 0.14
        if "bank account" in c and "update" in c and "processed" not in c:
            adj -= 0.16
        if "payment methods does welfog accept" in c or "credit/debit cards" in c:
            adj -= 0.14

    asks_return_policy = "return" in q and any(
        x in q for x in ("policy", "return policy", "returnable", "eligible for return")
    )
    if asks_return_policy:
        if "return policy" in c or "allows returns within" in c:
            adj += 0.14
        if "refund" in c and "processed" in c and "return policy" not in c:
            adj -= 0.08
        if "initiate a return" in c or "click \"return\"" in c or "click return" in c:
            adj -= 0.1

    if any(x in q for x in ("mobile app", "android app", "ios app", "download app")):
        if "mobile app" in c or "browse products" in c:
            adj += 0.14
        if "about welfog" in c and "mobile app" not in c:
            adj -= 0.18

    asks_address_phone_update = any(
        x in q
        for x in (
            "update my address",
            "update address",
            "change address",
            "change phone",
            "update phone",
            "phone number",
            "contact details",
            "my profile",
        )
    ) and any(x in q for x in ("address", "phone", "profile", "contact"))
    if asks_address_phone_update:
        if any(
            x in c
            for x in (
                "update your address",
                "update your phone",
                "my profile",
                "edit your saved address",
                "contact details",
            )
        ):
            adj += 0.22
        if any(x in c for x in ("fraud", "phishing", "suspicious message", "legalsupport@")):
            adj -= 0.28

    asks_delayed_lost = any(
        x in q for x in ("delayed", "delay", "lost", "missing", "not received", "not arrived")
    ) and any(x in q for x in ("package", "order", "delivery", "shipment", "parcel"))
    if asks_delayed_lost:
        if any(
            x in c
            for x in (
                "delayed or lost",
                "package is delayed",
                "tracking",
                "customer support",
            )
        ):
            adj += 0.22
        if "return policy" in c or "initiate a return" in c or "click \"return\"" in c:
            adj -= 0.2

    if any(x in q for x in ("privacy", "personal data", "data safe", "data deletion")):
        if any(x in c for x in ("privacy", "personal data", "data deletion", "personal information")):
            adj += 0.14
        if "refund" in c and "privacy" not in c:
            adj -= 0.1

    asks_delivery_timeline = any(
        x in q
        for x in (
            "kitna time",
            "kitne din",
            "how long",
            "lgta h",
            "lagta h",
            "lagta hai",
            "kab aayega",
            "kab milega",
            "how many day",
            "time lgta",
            "time lagta",
        )
    ) and any(
        x in q
        for x in ("delivery", "deliver", "ship", "order aane", "order aayega", "courier")
    )
    if asks_delivery_timeline:
        if any(
            x in c
            for x in (
                "how long",
                "delivery time",
                "delivery times",
                "typically processes",
                "3-5 days",
                "1-3 business",
                "ship my order",
                "deliver my order",
            )
        ):
            adj += 0.24
        if any(
            x in c
            for x in (
                "not available at the time",
                "attempt redelivery",
                "redelivery",
                "leave a notification",
                "reschedule",
            )
        ):
            adj -= 0.28

    if any(x in q for x in ("deliver", "delivery", "shipping", "courier")):
        if any(x in c for x in ("deliver my order", "delivery times", "delivery time", "courier")):
            adj += 0.14
        if "refund" in c and "deliver" not in c:
            adj -= 0.1

    if any(x in q for x in ("customer care", "contact", "phone", "helpline", "support number", "call")):
        if any(x in c for x in ("9828", "info@", "customer care", "helpline", "contact", "support")):
            adj += 0.1

    asks_seller_create = any(
        x in q
        for x in (
            "banaye",
            "banau",
            "banao",
            "create",
            "register",
            "registration",
            "become seller",
            "become a seller",
            "seller account",
            "seller bane",
            "seller banna",
        )
    )
    if asks_seller_create:
        if any(
            x in c
            for x in (
                "how to create",
                "create a seller",
                "seller registration",
                "become a seller",
                "seller portal",
            )
        ):
            adj += 0.14
        if any(x in c for x in ("login problem", "login fail", "error on login", "cannot login")):
            adj -= 0.12

    asks_seller_login = "seller" in q and any(
        x in q for x in ("login", "log in", "sign in", "otp", "login nahi", "login fail")
    )
    if asks_seller_login:
        if any(x in c for x in ("login problem", "login fail", "otp", "sign in")):
            adj += 0.12
        if "how to create" in c:
            adj -= 0.08

    return adj


def _pick_best_kb_hit_for_query(
    question: str,
    hits: list[dict],
    *,
    ai_route: dict | None = None,
    conflict_check: bool = True,
    min_rerank_score: float | None = None,
) -> dict | None:
    """Rerank embedding hits with relevance adjustments; skip conflicting chunks."""
    best: dict | None = None
    best_score = -1.0
    ranked: list[tuple[float, dict]] = []
    for hit in hits:
        chunk = hit.get("chunk") or ""
        if conflict_check and _faq_chunk_conflicts_with_query(question, chunk, ai_route=ai_route):
            continue
        sc = float(hit.get("score") or 0) + _kb_chunk_relevance_adjustment(question, chunk)
        sc += _faq_question_similarity_boost(question, chunk)
        ranked.append((sc, {**hit, "score": sc}))
        if sc > best_score:
            best_score = sc
            best = {**hit, "score": sc}
    if len(ranked) >= 2:
        ranked.sort(key=lambda x: x[0], reverse=True)
        top_sc, top_hit = ranked[0]
        runner_sc, runner_hit = ranked[1]
        if runner_sc >= top_sc - 0.07:
            top_q = _faq_question_similarity_boost(question, top_hit.get("chunk") or "")
            run_q = _faq_question_similarity_boost(question, runner_hit.get("chunk") or "")
            if run_q > top_q + 0.03:
                best = runner_hit
                best_score = runner_sc
    floor = float(min_rerank_score if min_rerank_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    if best and best_score < floor:
        return None
    return best


def _faq_chunk_conflicts_with_query(
    question: str, chunk: str, ai_route: dict | None = None
) -> bool:
    """
    Drop wrong FAQ matches (e.g. buyer account FAQ on seller questions;
    order-history HOW-TO FAQ when user wants list in chat).
    """
    q = f" {(question or '').lower()} "
    c = f" {(chunk or '').lower()} "
    try:
        from services.account_list_semantics import (
            ai_route_requests_order_history_in_chat,
            ai_route_requests_order_history_howto,
        )

        if ai_route_requests_order_history_in_chat(ai_route) and _chunk_is_order_history_howto_faq(
            chunk
        ):
            return True
        if ai_route_requests_order_history_howto(ai_route) and not _chunk_is_order_history_howto_faq(
            chunk
        ):
            if "order history" in c and "track order" in c:
                return True
    except ImportError:
        pass
    try:
        from utils.helpers import message_is_seller_on_welfog_request

        if message_is_seller_on_welfog_request(question):
            buyer_only = (
                "place an order",
                "place order",
                "track your purchase",
                "track your order",
                "need an account to place",
                "create an account on welfog to place",
            )
            if any(b in c for b in buyer_only):
                return True
            if "seller" not in c and any(x in c for x in ("my orders", "track order", "checkout")):
                return True
    except ImportError:
        pass
    if _user_requests_grievance_channel(question):
        if "grievance" not in c and "compliance" not in c:
            if any(x in c for x in ("place order", "track order", "my orders", "checkout")):
                return True
    if any(x in q for x in (" story", " kahani", " kahaani", "our story")):
        if any(x in c for x in ("place order", "track order", "checkout", "add to cart")):
            return True
    asks_address_phone = any(
        x in q
        for x in (
            "update my address",
            "update address",
            "change address",
            "change phone",
            "update phone",
            "phone number",
            "my profile",
        )
    ) and any(x in q for x in ("address", "phone", "profile"))
    if asks_address_phone:
        if any(x in c for x in ("fraud", "phishing", "suspicious message", "legalsupport@")):
            return True
    asks_delayed_lost = any(
        x in q for x in ("delayed", "delay", "lost", "missing", "not received", "not arrived")
    ) and any(x in q for x in ("package", "order", "delivery", "shipment"))
    if asks_delayed_lost:
        if "return policy" in c and "delayed" not in c and "lost" not in c:
            return True
    return False


def resolve_best_faq_chunk_for_question(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> dict | None:
    """Semantic match against faqs.txt only — auto-updates when admin edits that file."""
    ensure_knowledge_cache_fresh()
    if "faqs" not in kb_chunks_by_key:
        return None
    try:
        from utils.helpers import message_is_seller_on_welfog_request

        if message_is_seller_on_welfog_request(f"{original_msg} {msg_en}"):
            return None
    except ImportError:
        pass
    if _user_requests_grievance_channel(f"{original_msg} {msg_en}"):
        return None
    q = build_kb_retrieval_query(original_msg, msg_en, conversation_context, ai_route=ai_route)
    if not q.strip():
        return None
    hits = top_kb_hits(q, keys=["faqs"], min_score=0.14, top_n=8)
    best = _pick_best_kb_hit_for_query(
        q, hits, ai_route=ai_route, min_rerank_score=KB_ANSWER_MIN_CONFIDENCE
    )
    if best:
        try:
            from services.query_understanding import infer_kb_query_category

            category = infer_kb_query_category(
                original_msg, msg_en, ai_route=ai_route, conversation_context=conversation_context
            )
            log_kb_retrieval(
                query_intent=category,
                query_meaning=((ai_route or {}).get("user_meaning") or "").strip(),
                retrieval_query=q,
                matched_file="faqs",
                selected_category=category,
                similarity_score=float(best.get("score") or 0),
                selected_chunks=hits[:3],
            )
        except Exception:
            pass
        return best
    return None


def _format_kb_brain_gap_reply(
    original_msg: str,
    msg_en: str = "",
    *,
    reply_lang: str = "",
) -> str:
    """Polite KB miss — same language as customer; no invented facts."""
    from services.translation_service import customer_reply_language, finalize_customer_reply

    rl = reply_lang or customer_reply_language(original_msg)
    plain = (
        sysmsg("kb_no_answer")
        or sysmsg("kb_miss")
        or "I don't have that specific detail in Welfog's knowledge base right now. "
        "I can help with orders, delivery, payments, returns, or policies — what do you need?"
    )
    body = _plain_text_to_html_body(plain) or plain
    return finalize_customer_reply(body, original_msg or msg_en, rl) or ""


def format_kb_answer_from_brain_keys(
    original_msg: str,
    msg_en: str,
    keys: list[str],
    *,
    reply_lang: str = "",
    conversation_context: str = "",
    title_hint: str = "",
    user_meaning_en: str = "",
    ai_route: dict | None = None,
) -> str:
    """
    Admin KB answer after brain locks kb_keys: vector RAG first (any language),
    then scoped section filter — grounded LLM only when needed.
    """
    from services.translation_service import (
        _looks_english_only_latin,
        _normalize_language,
        customer_reply_language,
        is_hinglish_message,
    )
    from utils.helpers import _text_asks_customer_care_contact

    ensure_knowledge_cache_fresh()
    ensure_kb_vectors()
    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return ""

    lang = _normalize_language(reply_lang or customer_reply_language(original_msg))
    if lang == "hinglish" or is_hinglish_message(original_msg):
        lang = "hinglish"

    if _text_asks_customer_care_contact(combined) and not _user_requests_grievance_channel(combined):
        cc = format_customer_care_reply_from_kb(original_msg, msg_en)
        if cc:
            return cc

    try:
        from utils.helpers import (
            _text_mentions_social_platform,
            _text_mentions_welfog_brand,
            message_asks_welfog_social_media,
        )

        if message_asks_welfog_social_media(
            combined, conversation_context=conversation_context
        ) or (
            _text_mentions_social_platform(combined.lower())
            and _text_mentions_welfog_brand(combined.lower())
        ):
            social = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
                ai_confirmed=True,
            )
            if social:
                return social
    except ImportError:
        pass

    filter_q = (
        (user_meaning_en or "").strip()
        or ((ai_route or {}).get("user_meaning") or "").strip()
        or (msg_en or "").strip()
        or (original_msg or "").strip()
    )

    route = ai_route or {}
    if not keys:
        keys = resolve_brain_kb_keys(
            route,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
        )

    valid = [k for k in keys if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS][:4]
    if not valid and keys:
        files_map = get_runtime_knowledge_files()
        valid = [
            k for k in keys if k in files_map and k not in INTERNAL_KB_KEYS
        ][:4]
    if not valid:
        keys = resolve_brain_kb_keys(
            route,
            original_msg,
            msg_en,
            conversation_context=conversation_context,
        )
        valid = [k for k in keys if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS][:4]

    brain_raw_keys = {
        str(k).strip().lower()
        for k in (route.get("kb_keys") or [])
        if str(k).strip()
    }
    valid_key_set = {str(k).strip().lower() for k in valid}
    brain_locked_kb = bool(route.get("_preflight_kb")) or (
        (route.get("data_channel") or "").strip().lower() == "kb"
        and bool(brain_raw_keys)
        and bool(valid_key_set)
        and valid_key_set.issubset(brain_raw_keys)
    )

    # Brain-hinted files + cross-file semantic winner (brain keys are hints, not prisons).
    if valid or "faqs" in kb_chunks_by_key:
        semantic_hit = select_best_kb_hit_for_customer_question(
            original_msg,
            msg_en,
            conversation_context=conversation_context,
            preferred_keys=valid,
            ai_route=route,
            brain_locked_kb=brain_locked_kb,
            min_score=KB_ANSWER_MIN_CONFIDENCE,
        )
        if semantic_hit and float(semantic_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
            direct = format_direct_reply_from_kb_hit(
                semantic_hit,
                original_msg,
                reply_lang=lang,
                retrieval_query=filter_q or combined,
                fast_lane=True,
                conversation_context=conversation_context,
            )
            if direct:
                return direct

    if valid:
        scoped_hit = retrieve_best_kb_chunk(
            filter_q or combined,
            keys=list(valid)[:4],
            ai_route=route,
            min_score=KB_ANSWER_MIN_CONFIDENCE,
        )
        if scoped_hit and float(scoped_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
            scoped_direct = format_direct_reply_from_kb_hit(
                scoped_hit,
                original_msg,
                reply_lang=lang,
                retrieval_query=filter_q or combined,
                fast_lane=True,
                conversation_context=conversation_context,
            )
            if scoped_direct:
                return scoped_direct

    if not valid:
        wide = retrieve_best_kb_chunk(
            combined, keys=None, ai_route=route, min_score=KB_ANSWER_MIN_CONFIDENCE
        )
        if wide:
            direct = format_direct_reply_from_kb_hit(
                wide,
                original_msg,
                reply_lang=lang,
                retrieval_query=combined,
                fast_lane=True,
                conversation_context=conversation_context,
            )
            if direct:
                return direct
        gap = _format_kb_brain_gap_reply(original_msg, msg_en, reply_lang=lang)
        return gap or ""

    budget = _infer_kb_answer_budget(combined)[0]
    filtered = ""
    keys_scoped = brain_locked_kb
    search_scope = list(valid)[:4]
    if not keys_scoped:
        search_scope = list(
            dict.fromkeys(
                search_scope
                + [k for k in ("faqs",) if k in kb_chunks_by_key and k not in search_scope]
            )
        )[:4]

    semantic_hit = retrieve_best_kb_chunk(
        filter_q or combined,
        keys=search_scope,
        ai_route=route,
        min_score=KB_ANSWER_MIN_CONFIDENCE,
    )
    if "faqs" in kb_chunks_by_key and not brain_locked_kb:
        faq_hit = resolve_best_faq_chunk_for_question(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=route,
        )
        if faq_hit and float(faq_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
            if not semantic_hit or float(faq_hit.get("score") or 0) >= float(
                semantic_hit.get("score") or 0
            ):
                semantic_hit = faq_hit
    elif "faqs" in kb_chunks_by_key and brain_locked_kb and "faqs" in search_scope:
        faq_hit = resolve_best_faq_chunk_for_question(
            original_msg,
            msg_en,
            "" if keys_scoped else conversation_context,
            ai_route=route,
        )
        if faq_hit and float(faq_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
            if semantic_hit and float(faq_hit.get("score") or 0) < float(
                semantic_hit.get("score") or 0
            ):
                pass
            elif not semantic_hit:
                semantic_hit = faq_hit
    if semantic_hit and float(semantic_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
        direct = format_direct_reply_from_kb_hit(
            semantic_hit,
            original_msg,
            reply_lang=lang,
            retrieval_query=filter_q or combined,
            fast_lane=True,
            conversation_context=conversation_context,
        )
        if direct:
            return direct

    # Section filter within brain-scoped admin files (when vector match is weak).
    if valid:
        live_raw = read_concatenated_kb_file_contents(valid)
        if live_raw:
            blob = re.sub(r"<br\s*/?>", "\n", live_raw)
            filtered = _filter_kb_blob_for_question(blob, filter_q, max_chars=budget)
            if not filtered.strip():
                filtered = _kb_plain_excerpt(
                    _faq_answer_text_from_chunk(blob), max_chars=budget
                )

    if not filtered.strip():
        wide_keys = valid if keys_scoped else None
        wide = retrieve_best_kb_chunk(
            combined,
            keys=wide_keys,
            ai_route=ai_route,
            min_score=KB_ANSWER_MIN_CONFIDENCE,
        )
        if wide:
            direct = format_direct_reply_from_kb_hit(
                wide,
                original_msg,
                reply_lang=lang,
                retrieval_query=filter_q or combined,
                fast_lane=True,
                conversation_context=conversation_context,
            )
            if direct:
                return direct

    if not filtered.strip():
        gap = _format_kb_brain_gap_reply(original_msg, msg_en, reply_lang=lang)
        if gap:
            return gap
        return ""

    use_grounding_llm = (
        _kb_multilingual_answer_enabled()
        and _customer_needs_kb_localization(original_msg, lang)
        and bool((filtered or "").strip())
    )
    if use_grounding_llm:
        plain_filtered = re.sub(r"<[^>]+>", " ", filtered)
        plain_filtered = re.sub(r"\s+", " ", plain_filtered).strip()
        if plain_filtered and _kb_excerpt_is_english_for_customer(plain_filtered, lang):
            grounded = _ground_kb_excerpt_for_customer(
                original_msg,
                plain_filtered,
                conversation_context=conversation_context,
                reply_lang=reply_lang or lang,
                kb_sources=valid,
            )
            if grounded:
                return grounded

    filtered = polish_faq_reply_for_customer(
        _strip_leading_faq_question(filtered), original_msg
    )
    title = (title_hint or "").strip() or _title_from_filtered_kb_text(filtered)
    intro = ""
    if _should_show_kb_title(title, original_msg):
        intro = (
            f"<div style='color:#333;line-height:1.55;margin-bottom:8px;'>"
            f"<b>{html_escape(title)}</b>"
            f"</div>"
        )

    plain_for_body = filtered
    core_html = _plain_text_to_html_body(plain_for_body) or _kb_html_body_from_blob(filtered)
    body = polish_faq_reply_for_customer((intro + core_html) if core_html else "", original_msg)
    if not body:
        return ""
    return _finalize_kb_customer_reply(body, original_msg, lang)


def format_dynamic_kb_answer(
    original_msg: str,
    msg_en: str = "",
    reply_lang: str = "",
    conversation_context: str = "",
    *,
    suggested_keys: list[str] | None = None,
    title_hint: str = "",
    ai_route: dict | None = None,
) -> str:
    """
    Universal admin-KB answer: any .txt file added/updated in admin panel works without code changes.
    Filters sections to match the question; same language as customer; concise; KB-only facts.
    """
    try:
        from services.query_intent_classifier import query_intent_allows_kb

        if not query_intent_allows_kb():
            return ""
    except ImportError:
        pass
    from utils.helpers import _text_asks_customer_care_contact
    from services.translation_service import (
        _normalize_language,
        customer_reply_language,
        finalize_customer_reply,
        is_hinglish_message,
    )

    ensure_knowledge_cache_fresh()
    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return ""

    from utils.helpers import message_asks_other_company_social_media, message_is_conversational_general_talk

    if message_asks_other_company_social_media(combined, conversation_context=conversation_context):
        return ""

    if message_is_conversational_general_talk(original_msg, msg_en, conversation_context):
        return ""

    from utils.helpers import message_is_assistant_identity_question

    if message_is_assistant_identity_question(combined):
        return ""

    lang = _normalize_language(reply_lang or customer_reply_language(original_msg))

    seller_only = False
    try:
        from utils.helpers import message_is_seller_on_welfog_request

        if message_is_seller_on_welfog_request(combined):
            seller_only = True
            suggested_keys = ["seller"]
            log_reasoning("Seller account — seller.txt KB only (skip buyer FAQ chunks).")
    except ImportError:
        pass

    if _user_requests_grievance_channel(combined):
        body = format_grievance_officer_reply_from_kb(
            original_msg, msg_en, reply_lang=reply_lang or lang
        )
        if body:
            log_reasoning("Grievance Officer — company/privacy KB section only.")
            return body

    if _text_asks_customer_care_contact(combined) and not _user_requests_grievance_channel(combined):
        return format_customer_care_reply_from_kb(original_msg, msg_en)

    retrieval_q = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    filter_q = retrieval_q or combined
    if lang == "hinglish" or is_hinglish_message(original_msg):
        lang = "hinglish"

    faq_hit = None
    if not seller_only:
        faq_hit = resolve_best_faq_chunk_for_question(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        )
    if faq_hit and float(faq_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
        excerpt = _faq_answer_text_from_chunk(
            re.sub(r"<br\s*/?>", "\n", faq_hit.get("chunk") or "")
        )
        excerpt = _kb_plain_excerpt(excerpt, max_chars=_infer_kb_answer_budget(filter_q)[0])
        if excerpt.strip():
            if _kb_multilingual_answer_enabled() and _customer_needs_kb_localization(
                original_msg, lang
            ):
                grounded = _ground_kb_excerpt_for_customer(
                    original_msg,
                    excerpt,
                    conversation_context=conversation_context,
                    reply_lang=reply_lang or lang,
                    kb_sources=["faqs"],
                )
                if grounded:
                    log_reasoning(
                        f"FAQ semantic answer (faqs.txt score={float(faq_hit.get('score') or 0):.2f})."
                    )
                    return grounded
            excerpt = polish_faq_reply_for_customer(excerpt, original_msg)
            title = _title_from_filtered_kb_text(excerpt)
            intro = (
                f"<div style='color:#333;line-height:1.55;margin-bottom:8px;'>"
                f"<b>{html_escape(title)}</b></div>"
            ) if _should_show_kb_title(title, original_msg) else ""
            body = intro + (_plain_text_to_html_body(excerpt) or "")
            body = polish_faq_reply_for_customer(body, original_msg)
            return _finalize_kb_customer_reply(body, original_msg, lang)

    keys = resolve_kb_keys_for_question(
        original_msg,
        msg_en,
        suggested_keys=suggested_keys,
        max_files=4,
        conversation_context=conversation_context,
        ai_route=ai_route,
    )
    try:
        from services.query_understanding import infer_kb_query_category

        _kb_cat = infer_kb_query_category(
            original_msg, msg_en, ai_route=ai_route, conversation_context=conversation_context
        )
    except ImportError:
        _kb_cat = "general"
    filtered = ""
    budget = _infer_kb_answer_budget(filter_q)[0]
    seller_keys = [k for k in keys if k == "seller"] if seller_only else keys
    search_keys = seller_keys or keys

    best_chunk = retrieve_best_kb_chunk(
        filter_q,
        keys=search_keys,
        ai_route=ai_route,
        min_score=KB_ANSWER_MIN_CONFIDENCE,
    )
    if best_chunk and best_chunk.get("chunk"):
        raw_hit = re.sub(r"<br\s*/?>", "\n", best_chunk.get("chunk") or "")
        cleaned_hit = polish_faq_reply_for_customer(
            _strip_kb_policy_heading_prefix(_strip_leading_faq_question(raw_hit)),
            original_msg,
        )
        filtered = _kb_plain_excerpt(cleaned_hit or raw_hit, max_chars=budget)
        log_kb_retrieval(
            query_intent=_kb_cat,
            query_meaning=((ai_route or {}).get("user_meaning") or "").strip(),
            retrieval_query=retrieval_q,
            matched_file=str(best_chunk.get("source") or (keys[0] if keys else "")),
            selected_category=_kb_cat,
            similarity_score=float(best_chunk.get("score") or 0),
            selected_chunks=[best_chunk],
        )

    if not filtered.strip():
        return ""

    filtered = polish_faq_reply_for_customer(
        _strip_leading_faq_question(filtered), original_msg
    )

    if lang == "hinglish" or is_hinglish_message(original_msg):
        lang = "hinglish"

    if seller_only and filtered.strip():
        plain_body = filtered
        if lang == "hinglish":
            hb = _hinglish_policy_fallback_plain(filtered)
            if hb:
                plain_body = hb
        core_html = _plain_text_to_html_body(plain_body) or _kb_html_body_from_blob(filtered)
        title = (title_hint or "").strip() or _title_from_filtered_kb_text(filtered) or "Seller account"
        intro = (
            f"<div style='color:#333;line-height:1.55;margin-bottom:8px;'>"
            f"<b>{html_escape(title)}</b></div>"
        ) if _should_show_kb_title(title, original_msg) else ""
        if _kb_multilingual_answer_enabled() and _customer_needs_kb_localization(original_msg, lang):
            grounded = _ground_kb_excerpt_for_customer(
                original_msg,
                filtered,
                conversation_context=conversation_context,
                reply_lang=reply_lang or lang,
                kb_sources=keys,
            )
            if grounded:
                return grounded
        body = polish_faq_reply_for_customer((intro + core_html) if core_html else "", original_msg)
        if body:
            return _finalize_kb_customer_reply(body, original_msg, lang)

    title = (title_hint or "").strip() or _title_from_filtered_kb_text(filtered)
    intro = (
        f"<div style='color:#333;line-height:1.55;margin-bottom:8px;'>"
        f"<b>{html_escape(title)}</b>"
        f"</div>"
    ) if _should_show_kb_title(title, original_msg) else ""

    plain_for_body = filtered
    if lang == "hinglish":
        concise = _hinglish_policy_fallback_plain(filtered)
        if concise:
            plain_for_body = concise
    core_html = _plain_text_to_html_body(plain_for_body) or _kb_html_body_from_blob(filtered)

    if _kb_multilingual_answer_enabled() and _customer_needs_kb_localization(original_msg, lang):
        grounded = _ground_kb_excerpt_for_customer(
            original_msg,
            filtered,
            conversation_context=conversation_context,
            reply_lang=reply_lang or lang,
            kb_sources=keys,
        )
        if grounded:
            if title and title.lower() not in grounded.lower()[:120]:
                return intro + grounded if intro else grounded
            return grounded

    enable_hinglish_llm = (os.getenv("ENABLE_KB_HINGLISH_LLM", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if enable_hinglish_llm and lang == "hinglish" and filtered.strip():
        ai_instruction = (
            f"{original_msg}\n\n"
            "Translate/summarize ONLY the knowledge below into Roman Hinglish. "
            "Copy every number, day count, phone, and email EXACTLY as written — do not change policy facts. "
            "2-6 short bullets or 2-4 sentences; only what was asked."
        )
        try:
            from services.ai_service import ai_brain_answer

            ai = ai_brain_answer(
                ai_instruction,
                f"AUTHORITATIVE KNOWLEDGE (live admin files: {', '.join(keys)}):\n{filtered}",
                conversation_context=conversation_context,
                reply_lang="hinglish",
            )
            resp = (ai.get("response") or "").strip() if ai else ""
            if resp and not _ai_response_looks_like_placeholder(resp):
                core_html = _plain_text_to_html_body(resp)
        except Exception:
            pass

    body = (intro + core_html) if core_html else ""
    if body:
        body = polish_faq_reply_for_customer(body, original_msg)
    if not body:
        return ""
    return _finalize_kb_customer_reply(body, original_msg, lang)


def should_try_admin_kb_answer(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
) -> bool:
    """True when a customer KB file likely has the answer (embedding match)."""
    from utils.helpers import (
        _text_has_product_shopping_intent,
        _text_is_order_tracking_intent,
        message_is_casual_offtopic_not_shopping,
        should_send_warm_greeting_reply,
    )

    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return False
    if should_send_warm_greeting_reply(original_msg, msg_en, conversation_context=conversation_context):
        return False
    from utils.helpers import message_is_conversational_general_talk

    if message_is_conversational_general_talk(original_msg, msg_en, conversation_context):
        return False
    from utils.helpers import message_is_assistant_identity_question

    if message_is_assistant_identity_question(combined):
        return False
    if _text_has_product_shopping_intent(combined):
        return False
    try:
        from utils.helpers import extract_order_id

        if _text_is_order_tracking_intent(combined) and extract_order_id(combined):
            return False
    except ImportError:
        pass
    if message_is_casual_offtopic_not_shopping(combined):
        return False
    from utils.helpers import (
        _text_asks_customer_care_contact,
        _text_asks_welfog_fees_or_charges,
        _text_asks_short_video_content_rules,
        message_asks_other_company_social_media,
        message_asks_welfog_social_media,
        message_is_knowledge_information_request,
    )

    ensure_knowledge_cache_fresh()

    if message_asks_other_company_social_media(combined, conversation_context=conversation_context):
        return False
    if message_asks_welfog_social_media(combined, conversation_context=conversation_context):
        return False
    if _text_asks_customer_care_contact(combined):
        return True
    if message_is_knowledge_information_request(combined, conversation_context):
        return True
    if _text_asks_welfog_fees_or_charges(combined):
        return True
    if _text_asks_short_video_content_rules(combined, conversation_context):
        return True

    retrieval_q = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context
    )
    hit = best_kb_hit(
        retrieval_q or combined,
        keys=get_customer_kb_keys(),
        min_score=float(os.getenv("KNOWLEDGE_SEMANTIC_MIN_SCORE", "0.16") or "0.16"),
    )
    if hit:
        return True

    try:
        from services.knowledge_query_pipeline import analyze_informational_knowledge_turn

        sem = analyze_informational_knowledge_turn(
            original_msg, msg_en, conversation_context
        )
        if sem.is_informational and sem.confidence >= 0.6:
            return True
    except ImportError:
        pass

    hit = best_kb_hit(combined, keys=get_customer_kb_keys(), min_score=0.20)
    return bool(hit)


def _user_requests_grievance_channel(user_text: str) -> bool:
    tl = f" {(user_text or '').lower()} "
    return any(
        p in tl
        for p in (
            "grievance",
            "grievance officer",
            "complaint officer",
            "chief compliance",
            "legal grievance",
            "formal complaint",
            "escalation email",
            "escalate to",
            "serious complaint",
        )
    )


def _is_grievance_style_email(email: str) -> bool:
    e = (email or "").strip().lower()
    if not e or "@" not in e:
        return False
    if "grievance" in e:
        return True
    if e.startswith("legal.") or ".legal." in e:
        return True
    if "legalsupport" in e.replace(".", ""):
        return True
    return False


def extract_contacts_from_plain_text(blob: str, include_grievance: bool):
    """
    Pull phone numbers and emails from arbitrary KB text (digits/spelling come from files only).
    Returns (phones_ordered_unique, routine_emails, grievance_emails).
    """
    if not blob:
        return [], [], []

    phones_raw = []
    # Indian mobiles: 10 digits starting 6-9; optional +91 or leading 0
    for m in re.finditer(r"\+91[\s\-]*([6-9]\d{9})\b", blob):
        phones_raw.append(m.group(1))
    for m in re.finditer(r"\b0([6-9]\d{9})\b", blob):
        phones_raw.append(m.group(1))
    for m in re.finditer(r"\b([6-9]\d{9})\b", blob):
        phones_raw.append(m.group(1))

    seen_p = set()
    phones = []
    for p in phones_raw:
        if p in seen_p:
            continue
        seen_p.add(p)
        phones.append(p)

    emails_all = []
    for m in re.finditer(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", blob):
        emails_all.append(m.group(1).strip())

    seen_e = set()
    routine = []
    griev = []
    for em in emails_all:
        el = em.lower()
        if el in seen_e:
            continue
        seen_e.add(el)
        if _is_grievance_style_email(em):
            griev.append(em)
        else:
            routine.append(em)

    if not include_grievance:
        griev = []
    return phones, routine, griev


def _contact_channel_requested(user_text: str) -> str:
    """
    Detect what user asked for:
    - phone_only, email_only, both, unspecified
    """
    t = f" {(user_text or '').lower()} "
    asks_phone = any(
        x in t
        for x in (
            "phone", "number", "call", "helpline", "mobile", "contact no", "contact number",
            "customer care number", "support number",
        )
    )
    asks_email = any(
        x in t
        for x in ("email", "mail id", "gmail", "e-mail", "support mail", "support email")
    )
    if asks_phone and asks_email:
        return "both"
    if asks_phone:
        return "phone_only"
    if asks_email:
        return "email_only"
    return "unspecified"


def format_customer_care_reply_from_kb(original_msg: str, msg_en: str) -> str:
    """
    When the user asks for customer-care phone/email, build the reply ONLY from
    admin-maintained support/contact knowledge files so we never substitute
    grievance inboxes or generic 'only use Help & Support in the app' as the answer.

    Returns HTML string or "" when no parseable contacts exist in those files.
    """
    from services.translation_service import customer_reply_language, finalize_customer_reply

    keys = get_support_contact_kb_keys()
    if not keys:
        return ""

    blob = read_concatenated_kb_file_contents(keys)
    if not blob:
        return ""

    include_g = _user_requests_grievance_channel(f"{original_msg} {msg_en}")
    phones, emails, griev_emails = extract_contacts_from_plain_text(blob, include_grievance=include_g)

    if include_g and not phones and not emails and not griev_emails:
        return ""

    if not include_g and not phones and not emails:
        return ""

    from services.translation_service import is_hinglish_message

    channel = _contact_channel_requested(f"{original_msg} {msg_en}")
    hinglish = is_hinglish_message(original_msg) or is_hinglish_message(msg_en)
    lines = []
    if hinglish:
        lines.append(
            "Aap pehle se hi <b>Welfog support</b> se is chat mein baat kar rahe ho — "
            "alag se Help menu dhundhne ki zaroorat nahi."
        )
    else:
        lines.append(
            "You are already talking to <b>Welfog support</b> in this chat — no need to hunt for a separate "
            "\"Help & Support\" menu just to reach us."
        )

    if phones and channel in ("phone_only", "both", "unspecified"):
        if hinglish:
            lines.append(
                "Official customer-care number: "
                + ", ".join(f"<b>{p}</b>" for p in phones)
            )
        else:
            lines.append(
                "Official customer-care number: "
                + ", ".join(f"<b>{p}</b>" for p in phones)
            )
    if emails and channel in ("email_only", "both", "unspecified"):
        if hinglish:
            lines.append(
                "Support email: "
                + ", ".join(f"<b>{e}</b>" for e in emails)
            )
        else:
            lines.append(
                "Support email: "
                + ", ".join(f"<b>{e}</b>" for e in emails)
            )

    if include_g and griev_emails:
        if hinglish:
            lines.append(
                "Formal grievance / Grievance Officer ke liye email: "
                + ", ".join(f"<b>{e}</b>" for e in griev_emails)
            )
        else:
            lines.append(
                "For formal grievance / escalation to the Grievance Officer (only if that is what you need): "
                + ", ".join(f"<b>{e}</b>" for e in griev_emails)
            )

    body = "<br><br>".join(lines)
    if not body:
        return ""
    return finalize_customer_reply(body, original_msg, customer_reply_language(original_msg))


def format_grievance_officer_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
) -> str:
    """Grievance / compliance contact — from live admin KB files only (not generic customer-care)."""
    from services.translation_service import customer_reply_language, finalize_customer_reply

    ensure_knowledge_cache_fresh()
    keys = [k for k in ("company", "privacy", "support") if k in kb_chunks_by_key]
    if not keys:
        keys = get_customer_kb_keys()[:3]
    blob = read_concatenated_kb_file_contents(keys)
    if not blob.strip():
        return ""
    filter_q = build_kb_retrieval_query(original_msg, msg_en) or f"{original_msg} {msg_en}"
    filtered = _filter_kb_blob_for_question(blob, filter_q)
    if not filtered.strip():
        for title, body in _extract_kb_sections_from_blob(blob):
            hay = f"{title} {body}".lower()
            if "grievance" in hay or "compliance" in hay:
                filtered = body.strip()
                break
    if not filtered.strip():
        return ""
    rl = reply_lang or customer_reply_language(original_msg)
    if _kb_multilingual_answer_enabled() and _customer_needs_kb_localization(original_msg, rl):
        grounded = _ground_kb_excerpt_for_customer(
            original_msg,
            filtered,
            reply_lang=rl,
            kb_sources=keys,
        )
        if grounded:
            return grounded
    body = _plain_text_to_html_body(_strip_leading_faq_question(filtered)) or ""
    return finalize_customer_reply(body, original_msg, rl) if body else ""


def format_support_escalation_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "en",
) -> str:
    """
    When API/KB cannot resolve payment block, seller upload, account issues, etc. —
    polite escalation with official phone/email from support knowledge files.
    """
    from services.translation_service import (
        customer_reply_language,
        is_hinglish_message,
        localize_for_customer,
    )

    lang = (reply_lang or customer_reply_language(original_msg)).lower().strip()
    if lang == "hinglish" or is_hinglish_message(original_msg):
        intro = sysmsg("support_escalation_hinglish") or sysmsg("support_escalation") or ""
    else:
        intro = sysmsg("support_escalation") or ""
        if intro and lang not in ("en", "hinglish"):
            intro = localize_for_customer(intro, lang)

    cc = format_customer_care_reply_from_kb(original_msg, msg_en)
    if intro and cc:
        return f"{intro}<br><br>{cc}"
    if cc:
        return cc
    return intro or ""


def _kb_section_after_heading(blob: str, heading: str) -> str:
    if not blob or not heading:
        return ""
    # Next section = short line ending with ':' (heading), not inline colons in prose (Address:, Email:).
    m = re.search(
        rf"^{re.escape(heading)}\s*:?\s*\n(.*?)(?=^(?:[^\n]{{0,100}}:\s*$)|\Z)",
        blob,
        flags=re.I | re.M | re.S,
    )
    return (m.group(1).strip() if m else "")


def format_seller_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
    conversation_context: str = "",
) -> str:
    """Seller registration OR login troubleshooting — from seller.txt, not company about story."""
    from utils.helpers import (
        message_is_seller_on_welfog_request,
        _conversation_in_seller_support_flow,
        _text_has_seller_login_problem_intent,
        _user_complains_bot_gave_wrong_topic,
        _user_seller_issue_still_unresolved,
    )
    from services.translation_service import _normalize_language, customer_reply_language

    combined = f"{original_msg} {msg_en}"
    in_seller_thread = (
        message_is_seller_on_welfog_request(combined)
        or _conversation_in_seller_support_flow(conversation_context)
        or _text_has_seller_login_problem_intent(combined, conversation_context)
    )
    if not in_seller_thread:
        return ""

    escalate = (
        _user_seller_issue_still_unresolved(combined, conversation_context)
        or _user_complains_bot_gave_wrong_topic(combined)
    )
    login_issue = _text_has_seller_login_problem_intent(combined, conversation_context) or escalate
    keys = [k for k in ("seller",) if k in kb_chunks_by_key]
    body = format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conversation_context,
        suggested_keys=keys or None,
        title_hint="",
    )
    if escalate and body:
        cc = format_customer_care_reply_from_kb(original_msg, msg_en)
        if cc and cc not in body:
            body = f"{body}<br><br>{cc}"
    return body


def format_welfog_fees_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
    conversation_context: str = "",
) -> str:
    """Delegates to universal admin KB (payment.txt auto-discovered)."""
    from utils.helpers import _text_asks_welfog_fees_or_charges

    combined = f"{original_msg} {msg_en}"
    if not _text_asks_welfog_fees_or_charges(combined):
        return ""
    keys = [k for k in ("payment", "faqs") if k in kb_chunks_by_key]
    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conversation_context,
        suggested_keys=keys or None,
        title_hint="Welfog service charges",
    )


def _short_video_kb_focus(user_msg: str) -> str:
    """
    Narrow KB slice for the current question — not the whole short-video policy every time.
    Returns: all | age | content | seller_promo
    """
    tl = f" {(user_msg or '').lower()} "
    if any(
        x in tl
        for x in (
            "age rule", "age ", "umar", "18 year", "18+", "minor", "bacch", "child",
            "guardian", "parent consent", "verifiable consent", "children",
        )
    ):
        return "age"
    if any(
        x in tl
        for x in (
            "asci", "misleading", "exaggerat", "seller", "supplier", "promotional",
            "claim", "vendor video",
        )
    ):
        return "seller_promo"
    if any(
        x in tl
        for x in (
            "hate speech", "violence", "copyright", "sexual", "fake news",
            "inappropriate", "remove content",
        )
    ):
        return "content"
    return "all"


def _collect_short_video_kb_plain(user_msg: str = "", *, focus: str = "") -> str:
    """Relevant short-video rules only — filtered by what the user asked."""
    focus = (focus or _short_video_kb_focus(user_msg)).strip() or "all"
    parts: list[str] = []
    terms_blob = _strip_kb_blob_for_display(read_concatenated_kb_file_contents(["terms"]))
    if focus in ("all", "content"):
        sec = _kb_section_after_heading(terms_blob, "Short Video Content Rules")
        if sec:
            parts.append(f"Short Video Content Rules:\n{sec.strip()}")
    if focus in ("all", "age"):
        sec = _kb_section_after_heading(terms_blob, "User Eligibility & Age Rules")
        if sec:
            parts.append(f"User Eligibility & Age Rules:\n{sec.strip()}")
        privacy_blob = _strip_kb_blob_for_display(read_concatenated_kb_file_contents(["privacy"]))
        for line in privacy_blob.splitlines():
            low = line.lower()
            if any(
                k in low
                for k in ("short video", "shorts", "children", "18 years", "guardian", "parent")
            ):
                line = line.strip()
                if line:
                    parts.append(line)
    if focus in ("all", "seller_promo"):
        seller_blob = _strip_kb_blob_for_display(read_concatenated_kb_file_contents(["seller"]))
        seller_sec = _kb_section_after_heading(seller_blob, "Supplier Promotional Videos")
        if seller_sec:
            parts.append(f"Supplier Promotional Videos:\n{seller_sec.strip()}")
    if focus == "all" and not parts:
        return _collect_short_video_kb_plain(user_msg, focus="content") + "\n\n" + _collect_short_video_kb_plain(
            user_msg, focus="age"
        )
    return "\n\n".join(parts).strip()


def format_short_video_rules_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
    conversation_context: str = "",
) -> str:
    """Delegates to universal admin KB (terms/seller/privacy files — auto-discovered)."""
    from utils.helpers import _text_asks_short_video_content_rules

    combined = f"{original_msg} {msg_en}"
    if not _text_asks_short_video_content_rules(combined, conversation_context):
        return ""
    keys = [k for k in ("terms", "seller", "privacy") if k in kb_chunks_by_key]
    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conversation_context,
        suggested_keys=keys or None,
        title_hint="Welfog Short Video / Shorts",
    )


def get_welfog_social_links_from_kb() -> list[tuple[str, str, str]]:
    """
    Parse official social URLs from ANY customer knowledge .txt file.
    Format: `Platform Name: https://...` or `Welfog Instagram: https://...`
    Admin can add/change URLs or new platforms without code changes.
    Returns list of (slug, display_label, url) in file order.
    """
    ensure_knowledge_cache_fresh()
    url_line = re.compile(
        r"^\s*(?:Welfog\s+)?(.+?)\s*:\s*(https?://\S+)\s*$",
        re.IGNORECASE,
    )
    found: list[tuple[str, str, str]] = []
    seen_slugs: set[str] = set()

    for key in get_customer_kb_keys():
        path = get_runtime_knowledge_files().get(key)
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or "http" not in line.lower():
                continue
            m = url_line.match(line)
            if not m:
                continue
            label, url = m.group(1).strip(), m.group(2).strip()
            slug = _social_slug_from_kb_label(label)
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            found.append((slug, label, url))
    return found


def _social_slug_from_kb_label(label: str) -> str:
    """Map admin label text to a stable slug — known platforms + generic fallback."""
    low = f" {(label or '').lower()} "
    known = (
        ("instagram", "instagram"),
        ("insta", "instagram"),
        ("linkedin", "linkedin"),
        ("linkdin", "linkedin"),
        ("facebook", "facebook"),
        (" fb ", "facebook"),
        ("youtube", "youtube"),
        ("youtu", "youtube"),
        ("twitter", "twitter"),
        ("twiter", "twitter"),
        ("twittr", "twitter"),
        (" x.com", "twitter"),
        (" x ", "twitter"),
        ("telegram", "telegram"),
        ("whatsapp", "whatsapp"),
        ("threads", "threads"),
        ("pinterest", "pinterest"),
        ("snapchat", "snapchat"),
    )
    for needle, slug in known:
        if needle in low:
            return slug
    words = [w for w in re.findall(r"[a-z0-9]+", label.lower()) if w not in ("welfog", "official", "link", "links")]
    if words:
        return words[-1][:24]
    return ""


def _social_links_as_dict(entries: list[tuple[str, str, str]]) -> dict[str, str]:
    return {slug: url for slug, _label, url in entries}


_SOCIAL_PLATFORM_ORDER = ("instagram", "youtube", "linkedin", "twitter", "facebook")


def _social_platforms_requested(
    text: str, conversation_context: str = "", *, known_links: dict[str, str] | None = None
) -> list[str]:
    """Platforms named in the current user message only — not prior chat turns."""
    _ = conversation_context  # kept for call-site compatibility; never merge old turns
    tl = f" {(text or '').lower()} "
    out: list[str] = []

    if known_links:
        for slug in known_links:
            if slug in tl or slug.replace("_", " ") in tl:
                out.append(slug)

    if (
        re.search(r"\b(?:i?n?stagram|insta(?:gram|grm|gr?am)?)\b", tl)
        or re.search(r"\binstr?gr?am\b", tl)
        or " insta " in tl
        or re.search(r"\binsta\b", tl)
    ):
        out.append("instagram")
    if "linkedin" in tl or "linkdin" in tl or "linked in" in tl:
        out.append("linkedin")
    if "facebook" in tl or " fb " in tl or re.search(r"\bfb\b", tl):
        out.append("facebook")
    if re.search(r"\byou\s*t+u*be\b", tl) or re.search(r"\byout+ube\b", tl) or re.search(r"\byt\b", tl):
        out.append("youtube")
    if (
        "twitter" in tl
        or re.search(r"\b(?:twiter|twittr|twitt?er|twitter|tweet)\b", tl)
        or re.search(r"\bx\.com\b", tl)
        or re.search(r"\bx\s*\(\s*x\s*\)", tl)
    ):
        out.append("twitter")

    seen: set[str] = set()
    ordered: list[str] = []
    for key in list(_SOCIAL_PLATFORM_ORDER) + list(out):
        if key in out and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _wants_all_welfog_social_links(text: str) -> bool:
    tl = f" {(text or '').lower()} "
    return any(
        x in tl
        for x in (
            "saare", "saari", "sabhi", " sab ", " all ", "all social", "har social",
            "saare social", "sab social", "social media account",
            "donon", "dono", "both platform", "har platform",
        )
    ) or any(
        re.search(p, tl)
        for p in (
            r"\bsaare\s+(?:link|social)",
            r"\bsab(?:hi)?\s+(?:link|social)",
            r"\ball\s+(?:official\s+)?(?:link|social)",
        )
    )


def _message_asks_multiple_social_platforms(text: str) -> bool:
    """User wants more than one platform (ya/aur/and) even if typos hide one name."""
    tl = f" {(text or '').lower()} "
    if not re.search(r"\b(ya|aur|and|or|,)\b", tl):
        return False
    keys = _social_platforms_requested(text)
    if len(keys) >= 2:
        return True
    # Connector + two social-ish tokens (e.g. insta aur youttube)
    social_tokens = len(
        re.findall(
            r"\b(?:insta(?:gram)?|instr?gr?am|youtube|yout+ube|twitter|linkedin|linkdin|facebook|fb)\b",
            tl,
        )
    )
    return social_tokens >= 2


def format_welfog_social_media_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
    conversation_context: str = "",
    *,
    user_meaning_en: str = "",
    ai_confirmed: bool = False,
) -> str:
    """Official Welfog social URLs from company knowledge."""
    from services.translation_service import (
        _normalize_language,
        customer_reply_language,
        is_hinglish_message,
    )

    combined = f"{original_msg} {msg_en}"

    if not ai_confirmed:
        from utils.helpers import (
            _is_welfog_social_followup,
            message_asks_welfog_social_media,
        )

        if not (
            message_asks_welfog_social_media(
                combined, conversation_context=conversation_context
            )
            or _is_welfog_social_followup(combined, conversation_context)
        ):
            return ""

    lang = _normalize_language(reply_lang or customer_reply_language(original_msg))
    all_entries = get_welfog_social_links_from_kb()
    if not all_entries:
        if lang == "hinglish" or is_hinglish_message(original_msg):
            return sysmsg("welfog_social_not_available_hinglish") or sysmsg("welfog_social_not_available") or ""
        return sysmsg("welfog_social_not_available") or ""

    all_links = _social_links_as_dict(all_entries)
    labels_from_kb = {slug: label for slug, label, _url in all_entries}

    def _ordered_all_slugs() -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for slug in _SOCIAL_PLATFORM_ORDER:
            if slug in all_links and slug not in seen:
                ordered.append(slug)
                seen.add(slug)
        for slug, _, _ in all_entries:
            if slug not in seen:
                ordered.append(slug)
                seen.add(slug)
        return ordered

    def _pick_slugs(keys: list[str]) -> list[str]:
        return [k for k in keys if k in all_links]

    turn = f"{user_meaning_en or ''} {combined}".strip()
    requested = _social_platforms_requested(turn, known_links=all_links)
    wants_all = _wants_all_welfog_social_links(turn)

    if wants_all:
        picked = _ordered_all_slugs()
    elif len(requested) >= 2:
        picked = _pick_slugs(requested)
    elif len(requested) == 1:
        picked = _pick_slugs(requested)
    else:
        tl = f" {turn.lower()} "
        vague_all = any(
            x in tl
            for x in (
                "social media",
                "social link",
                "social account",
                "official link",
                "official page",
                "saare link",
                "sab link",
                "all link",
            )
        )
        picked = _ordered_all_slugs() if vague_all else []

    if not picked:
        if lang == "hinglish" or is_hinglish_message(original_msg):
            return sysmsg("welfog_social_not_available_hinglish") or sysmsg("welfog_social_not_available") or ""
        return sysmsg("welfog_social_not_available") or ""

    known_labels = {
        "instagram": "Instagram",
        "linkedin": "LinkedIn",
        "facebook": "Facebook",
        "youtube": "YouTube",
        "twitter": "Twitter (X)",
    }
    btn_style = (
        "display:inline-block;text-align:center;margin:6px 8px 6px 0;color:white;"
        "padding:10px 14px;text-decoration:none;border-radius:22px;font-weight:600;font-size:14px;"
    )
    parts = []
    for slug in picked:
        url = all_links[slug]
        label = labels_from_kb.get(slug) or known_labels.get(slug, slug.replace("_", " ").title())
        parts.append(
            f"<a href='{html_escape(url)}' target='_blank' rel='noopener noreferrer' "
            f"style='{btn_style}background:#333;'>{html_escape(label)}</a>"
        )

    if len(picked) == 1:
        only = known_labels.get(picked[0]) or labels_from_kb.get(picked[0]) or picked[0].title()
        if lang == "hinglish" or is_hinglish_message(original_msg):
            intro = (
                f"<div style='color:#333;line-height:1.55;margin-bottom:10px;'>"
                f"<b>Welfog {html_escape(only)}</b>:</div>"
            )
        else:
            intro = (
                f"<div style='color:#333;line-height:1.55;margin-bottom:10px;'>"
                f"<b>Official Welfog {html_escape(only)}</b>:</div>"
            )
    elif lang == "hinglish" or is_hinglish_message(original_msg):
        intro = (
            sysmsg("welfog_social_links_intro_hinglish")
            or "<div style='color:#333;line-height:1.55;margin-bottom:10px;'>"
            "<b>Welfog ke official social media links</b>:</div>"
        )
    else:
        intro = (
            sysmsg("welfog_social_links_intro")
            or "<div style='color:#333;line-height:1.55;margin-bottom:10px;'>"
            "<b>Official Welfog social media</b>:</div>"
        )
    return f"{intro}<div style='margin-top:8px;'>{''.join(parts)}</div>"


def format_welfog_about_reply_from_kb(
    original_msg: str, msg_en: str, reply_lang: str = ""
) -> str:
    """
    Welfog company / story / about — from live admin knowledge files (no hardcoded sections).
    """
    from utils.helpers import message_is_welfog_about_request

    if not message_is_welfog_about_request(f"{original_msg} {msg_en}"):
        return ""

    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
    )


def _knowledge_keys_and_title(text: str) -> tuple[list[str], str]:
    """Soft hints only — file pick is embedding-driven via resolve_kb_keys_for_question."""
    return [], ""


_KB_FILE_MARKER_RE = re.compile(
    r"^===\s*FILE\s+key=\S+\s+path=\S+\s*===\s*\n?",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_kb_blob_for_display(blob: str) -> str:
    """Remove internal KB debug markers before showing text to customers."""
    if not blob:
        return ""
    cleaned = _KB_FILE_MARKER_RE.sub("", blob)
    drop_line = re.compile(
        r"^\s*[A-Z]\)\s+.*$|^\s*→\s*action\s+.*$|"
        r"^\s*action\s*:\s*(ask_order_id|track_live|check_pin).*$|"
        r"EXAMPLES FOR GROQ|wants status but NO id|playbook|specialist|"
        r"support rules|never invent|only share welfog|politely decline",
        re.I | re.M,
    )
    lines = [ln for ln in cleaned.splitlines() if not drop_line.match(ln.strip())]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _hinglish_policy_fallback_plain(plain: str) -> str:
    """Roman Hinglish bullets when LLM paraphrase is unavailable."""
    if not plain.strip():
        return ""
    lines: list[str] = []
    for raw in plain.splitlines():
        line = raw.strip().lstrip("- ").strip()
        if not line:
            continue
        if line.endswith(":") and len(line) < 72:
            continue
        if len(line) > 24:
            lines.append(f"• {line[:280]}")
        if len(lines) >= 4:
            break
    return "\n\n".join(lines)


def _kb_plain_excerpt(blob: str, max_chars: int = 2800) -> str:
    plain = _strip_kb_blob_for_display(blob)
    if len(plain) <= max_chars:
        return plain
    return plain[:max_chars].rsplit("\n", 1)[0].strip() + "…"


def _plain_text_to_html_body(text: str) -> str:
    if not (text or "").strip():
        return ""
    body_parts = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if para.startswith("- "):
            items = [ln.strip()[2:] for ln in para.splitlines() if ln.strip().startswith("- ")]
            if items:
                body_parts.append(
                    "<ul style='margin:8px 0 0 18px;color:#333;line-height:1.55;'>"
                    + "".join(f"<li>{html_escape(it)}</li>" for it in items)
                    + "</ul>"
                )
            continue
        body_parts.append(
            f"<p style='color:#333;line-height:1.55;margin:0 0 10px 0;'>{html_escape(para)}</p>"
        )
    return "".join(body_parts)


def _kb_html_body_from_blob(blob: str, max_chars: int = 3200) -> str:
    blob = _strip_kb_blob_for_display(blob)
    if not blob.strip():
        return ""
    body_parts = []
    for para in re.split(r"\n\s*\n", blob.strip()):
        para = para.strip()
        if not para or para.endswith(":") and len(para) < 80:
            continue
        if para.startswith("- "):
            items = [ln.strip()[2:] for ln in para.splitlines() if ln.strip().startswith("- ")]
            if items:
                body_parts.append(
                    "<ul style='margin:8px 0 0 18px;color:#333;line-height:1.55;'>"
                    + "".join(f"<li>{html_escape(it)}</li>" for it in items)
                    + "</ul>"
                )
            continue
        body_parts.append(
            f"<p style='color:#333;line-height:1.55;margin:0 0 10px 0;'>{html_escape(para)}</p>"
        )
    html = "".join(body_parts)
    if len(blob) > max_chars and not html:
        excerpt = re.sub(r"\s+", " ", blob).strip()[:max_chars]
        html = f"<p style='color:#333;line-height:1.55;'>{html_escape(excerpt)}…</p>"
    return html


def format_knowledge_information_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
    conversation_context: str = "",
) -> str:
    """
    Policy / privacy / terms / FAQ — from knowledge files, never product catalog.
    """
    from utils.helpers import (
        message_asks_other_company_policy,
        message_is_knowledge_information_request,
        message_is_welfog_about_request,
    )
    from services.translation_service import (
        NATIVE_SCRIPT_LANGS,
        _normalize_language,
        customer_reply_language,
        localize_for_customer,
    )

    combined = f"{original_msg} {msg_en}"
    from services.policy_scope import policy_question_is_for_welfog
    from utils.helpers import message_asks_welfog_social_media

    if message_asks_welfog_social_media(combined):
        return ""
    if message_asks_other_company_policy(combined, conversation_context):
        return ""
    if not policy_question_is_for_welfog(original_msg, msg_en, conversation_context):
        return ""
    if not message_is_knowledge_information_request(combined, conversation_context):
        return ""

    keys = resolve_kb_keys_for_question(original_msg, msg_en, max_files=4)
    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conversation_context,
        suggested_keys=keys,
    )


def format_fraud_complaint_reply_from_kb(
    original_msg: str,
    msg_en: str,
    reply_lang: str = "",
) -> str:
    """Fraud/phishing/complaint — filtered from live admin KB (no hardcoded line picking)."""
    q = f" {original_msg} {msg_en} ".lower()
    if not any(x in q for x in ("fraud", "phishing", "fake", "scam", "complaint", "grievance", "report")):
        return ""

    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
    )


def format_policy_help_reply_from_kb(
    original_msg: str, msg_en: str, reply_lang: str = ""
) -> str:
    """
    Structured answers for return/refund/wrong-item questions (often multiple in one message).
    Grounded in faqs/refund KB — not product search. English base → localized per customer language.
    """
    from utils.helpers import (
        _text_has_past_order_complaint_context,
        _text_has_refund_or_return_intent,
    )
    from services.translation_service import (
        customer_reply_language,
        finalize_customer_reply,
        localized_sysmsg_for_customer,
        resolve_customer_reply_lang,
    )

    rl = resolve_customer_reply_lang(original_msg, reply_lang or customer_reply_language(original_msg))
    tl = f" {original_msg} {msg_en} ".lower()

    blocks: list[str] = []

    if _text_has_past_order_complaint_context(tl):
        if any(x in tl for x in ("color", "colour", "rang", "alag", "galat", "wrong")):
            blocks.append(
                "<div style='color:#333;line-height:1.55;'>"
                "Sorry — the colour you received is different from what you ordered. On Welfog this is "
                "treated as a <b>wrong item</b>; return or replacement may apply."
                "</div>"
            )
        else:
            blocks.append(
                "<div style='color:#333;line-height:1.55;'>"
                "For the issue you described (wrong / defective / damaged item), return or replacement "
                "policy may apply."
                "</div>"
            )

    asks_return = "return" in tl and any(
        x in tl
        for x in (
            "kya", "ky ", "ho sk", "kar sk", "kr sk", "sakta", "skta", "milega", "chahiye",
            "possible", "allowed", "kar sakta", "kr sakta",
            "kaise", "kese", "krte", "karte", "karna", "krna", "karu",
        )
    )
    if asks_return or ("return" in tl and _text_has_past_order_complaint_context(tl)):
        blocks.append(
            "<div style='color:#333;line-height:1.55;margin-top:10px;'>"
            "<b>1) Can you return?</b><br>"
            "Yes — within <b>7 days</b> of delivery, raise a return or replacement from "
            "<b>Order History</b>. Clear photos of the wrong colour/item are required."
            "</div>"
        )

    asks_refund_time = "refund" in tl and any(
        x in tl for x in ("kitne din", "kab", "time", "timeline", "days", "din me", "milega", "aayega")
    )
    if asks_refund_time:
        blocks.append(
            "<div style='color:#333;line-height:1.55;margin-top:10px;'>"
            "<b>2) When will the refund arrive?</b><br>"
            "After return is approved and pickup is complete, refund is usually processed in "
            "<b>5–7 business days</b> to your original payment method."
            "</div>"
        )
    elif _text_has_refund_or_return_intent(tl) and _text_has_past_order_complaint_context(tl):
        blocks.append(
            "<div style='color:#333;line-height:1.55;margin-top:10px;'>"
            "<b>Refund:</b> After return approval + pickup, <b>5–7 business days</b> (official policy)."
            "</div>"
        )

    policy_keys = [k for k in ("faqs", "refund", "shipping", "payment") if k in kb_chunks_by_key]
    if policy_keys:
        blob = read_concatenated_kb_file_contents(policy_keys)
        filtered = _filter_kb_blob_for_question(
            blob, f"{original_msg} {msg_en}", max_chars=420
        )
        if filtered.strip():
            excerpt = re.sub(r"\s+", " ", _strip_kb_blob_for_display(filtered)).strip()
            if len(excerpt) > 420:
                excerpt = excerpt[:420].rsplit(" ", 1)[0] + "…"
            blocks.append(
                f"<div style='margin-top:10px;color:#555;font-size:13px;'>"
                f"<b>Policy reference:</b> {html_escape(excerpt)}</div>"
            )

    if not blocks:
        return ""

    if rl == "hinglish":
        blocks.append(
            "<div style='margin-top:12px;color:#555;font-size:13px;'>"
            "Steps: Welfog app/website → <b>My Orders</b> → order select → Return/Replacement. "
            "Order ID chaho to bhej do — status check kar sakte hain."
            "</div>"
        )
    else:
        blocks.append(
            "<div style='margin-top:12px;color:#555;font-size:13px;'>"
            "Steps: Welfog app/website → <b>My Orders</b> → select order → Return/Replacement. "
            "You can paste your Order ID here for status help."
            "</div>"
        )
    combined = "".join(blocks)
    return finalize_customer_reply(combined, original_msg, rl)


def sysmsg(key: str, **kwargs):
    """
    Fetch a user-facing message from knowledge files.
    Supports {placeholders}.
    """
    txt = _SYSTEM_MESSAGES.get(key, "")
    if not txt:
        return ""
    try:
        return txt.format(**kwargs)
    except Exception:
        return txt



def get_knowledge_context(query, keys=None, top_k=3, min_score=None):
    """
    If keys provided, search within those knowledge files only; else customer KB files.
    Returns an HTML string to inject into the system prompt.
    """
    ensure_knowledge_cache_fresh()
    if not all_chunks:
        return ""

    floor = float(min_score if min_score is not None else KB_CONTEXT_MIN_SCORE)
    norm_q = _normalize_retrieval_query(query)
    cache_key = (
        f"kbctx::{_KB_SNAPSHOT}::{norm_q}::{','.join(keys) if keys else 'CUSTOMER'}"
        f"::{top_k}::{floor}"
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    hits = semantic_kb_search(
        norm_q,
        keys=keys,
        top_n=top_k,
        min_score=floor,
        customer_only=not bool(keys),
    )
    if not hits:
        _cache_set(cache_key, "")
        return ""

    out = []
    for h in hits:
        src = h.get("source") or "?"
        sc = float(h.get("score") or 0)
        chunk = h.get("chunk") or ""
        out.append(f"[source={src} score={sc:.2f}] {chunk}")
    result = "<br><br>".join(out)
    _cache_set(cache_key, result)
    return result


# ================= 🚀 1.5 DIRECT KB SEARCH 🚀 =================
def rank_customer_kb_files_by_embedding(
    query: str,
    *,
    keys: list[str] | None = None,
    min_score: float = 0.12,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Rank admin KB files by best chunk score per file (not one global chunk winner).
    New .txt files from admin panel participate automatically — no code deploy.
    """
    if not (query or "").strip() or not all_chunks:
        return []
    customer_keys = [
        k
        for k in (keys or get_customer_kb_keys())
        if k and not str(k).startswith("welfog_api")
    ]
    if not customer_keys:
        return []
    floor = float(min_score)
    hits = semantic_kb_search(
        query,
        keys=customer_keys,
        top_n=min(40, max(len(customer_keys) * 5, top_n * 4)),
        min_score=floor,
        customer_only=False,
        log_retrieval=False,
    )
    best_per_key: dict[str, float] = {}
    for hit in hits:
        src = str(hit.get("source") or "").strip()
        chunk = hit.get("chunk") or ""
        if not src or src not in customer_keys:
            continue
        if _chunk_is_agent_instruction_blob(chunk):
            continue
        sc = float(hit.get("score") or 0.0)
        best_per_key[src] = max(best_per_key.get(src, 0.0), sc)
    ranked = sorted(best_per_key.items(), key=lambda x: x[1], reverse=True)
    return [(k, s) for k, s in ranked[:top_n] if s >= floor]


def best_kb_hit(query, keys=None, min_score=None):
    """
    Returns the best matching KB chunk (raw HTML chunk) with its score and source.
    Uses semantic search + rerank (same as retrieve_best_kb_chunk).
    """
    if not (query or "").strip() or not all_chunks:
        return None
    floor = float(min_score if min_score is not None else KB_SEMANTIC_MIN_SCORE)
    return retrieve_best_kb_chunk(query, keys=keys, min_score=floor)


def top_kb_hits(
    query: str,
    keys=None,
    min_score: float | None = None,
    top_n: int = 4,
    log_retrieval: bool = False,
) -> list[dict]:
    """Top-N semantic KB chunks for reranking (multilingual queries)."""
    if not all_chunks or not (query or "").strip():
        return []
    floor = float(min_score if min_score is not None else KB_CONTEXT_MIN_SCORE)
    return semantic_kb_search(
        query,
        keys=keys,
        top_n=top_n,
        min_score=floor,
        customer_only=not bool(keys),
        log_retrieval=log_retrieval,
    )


def keyword_kb_hit(query: str, keys=None, min_hits: int = 2):
    """
    Fallback when embeddings miss due to phrasing/short queries.
    Scores chunks by keyword overlap (case-insensitive substring match).
    Returns: {source, chunk, score} where score is hit-count.
    """
    if not query:
        return None

    # tokenize: keep alphanum, split, drop short words
    q = re.sub(r"[^a-z0-9 ]+", " ", (query or "").lower())
    raw_tokens = [t for t in q.split() if len(t) >= 4]
    if not raw_tokens:
        return None

    stop = {
        "welfog",
        "about",
        "explain",
        "please",
        "tell",
        "criteria",
        "eligibility",  # keep? removing avoids overfitting on header-only; other tokens still match
        "rules",
    }
    tokens = [t for t in raw_tokens if t not in stop]
    if not tokens:
        tokens = raw_tokens

    if keys:
        keys = [k for k in keys if k in kb_chunks_by_key]
        scoped = []
        scoped_src = []
        for k in keys:
            ch = kb_chunks_by_key.get(k) or []
            scoped.extend(ch)
            scoped_src.extend([k] * len(ch))
    else:
        scoped = all_chunks or []
        scoped_src = all_chunk_sources or []

    best = None
    best_hits = 0
    for i, chunk in enumerate(scoped):
        if not chunk:
            continue
        low = chunk.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits > best_hits:
            best_hits = hits
            best = (scoped_src[i], chunk)
            # early exit if very strong
            if best_hits >= max(min_hits + 2, 5):
                break

    if best and best_hits >= min_hits:
        src, ch = best
        return {"source": src, "chunk": ch, "score": float(best_hits)}
    return None


def direct_kb_search(query, keys=None, min_score=None, *, zero_llm_fast: bool = False):
    if not all_chunks:
        return None
    floor = float(min_score if min_score is not None else KB_DIRECT_MIN_SCORE)

    if zero_llm_fast:
        try:
            from utils.helpers import _text_asks_customer_care_contact

            qcomb = (query or "").strip()
            if _text_asks_customer_care_contact(qcomb):
                cc = format_customer_care_reply_from_kb(qcomb, "")
                if cc:
                    return cc
        except ImportError:
            pass
        hit = retrieve_best_kb_chunk(query, keys=keys, min_score=floor)
        if not hit or float(hit.get("score") or 0) < floor:
            return None
        body = format_direct_reply_from_kb_hit(
            hit,
            query,
            retrieval_query=query,
            fast_lane=True,
        )
        return body if body else None

    try:
        from services.query_intent_classifier import query_intent_allows_kb

        if not query_intent_allows_kb():
            return None
    except ImportError:
        pass
    try:
        from utils.helpers import (
            _text_asks_customer_care_contact,
            _user_asks_order_history_navigation_help,
        )

        qcomb = (query or "").strip()
        if _user_asks_order_history_navigation_help(qcomb):
            body = sysmsg("order_history_help") or ""
            return body if body else None
        if not _text_asks_customer_care_contact(qcomb):
            support_only_keys = get_support_contact_kb_keys()
            if keys and set(keys).issubset(set(support_only_keys or [])):
                return None
    except ImportError:
        pass

    hit = retrieve_best_kb_chunk(query, keys=keys, min_score=floor)
    if not hit or float(hit.get("score") or 0) < floor:
        return None

    category = str(hit.get("source") or "general")

    log_kb_retrieval(
        query_intent=category,
        retrieval_query=_normalize_retrieval_query(query),
        matched_file=str(hit.get("source") or ""),
        selected_category=category,
        similarity_score=float(hit.get("score") or 0),
        selected_chunks=[hit],
    )

    excerpt = _kb_plain_excerpt(
        _faq_answer_text_from_chunk(re.sub(r"<br\s*/?>", "\n", hit.get("chunk") or "")),
        max_chars=_infer_kb_answer_budget(query)[0],
    )
    if not excerpt.strip():
        return None

    body = _plain_text_to_html_body(excerpt)
    if not body:
        return None
    if zero_llm_fast:
        return body
    body = polish_faq_reply_for_customer(body, query)
    try:
        from services.translation_service import customer_reply_language

        return _finalize_kb_customer_reply(body, query, customer_reply_language(query))
    except ImportError:
        return body


# ================= ⚡ ULTRA-FAST INSTANT ROUTER (Dumb & Fast) ⚡ =================
def smart_instant_router(original_msg, english_msg):
    text = f" {original_msg} {english_msg} ".lower()
    words = original_msg.lower().split()

    # 1. INSTANT GREETINGS & LIGHT SMALLTALK (cultural / wellbeing — always in-domain)
    greetings = [
        "hi", "hello", "hii", "heyy", "hey", "hallo", "hiya", "namaste", "bhai", "bhia", "bhisaa",
        "sun", "suno", "brother", "bro", "hiii", "ram", "radhe", "radhey",
    ]
    skip_greeting = _should_bypass_warm_greeting_fast_path(text)
    from utils.helpers import should_use_warm_conversation_reply, build_warm_conversation_reply
    from services.translation_service import customer_reply_language

    if should_use_warm_conversation_reply(original_msg, english_msg, conversation_context=""):
        rl = customer_reply_language(original_msg)
        warm = build_warm_conversation_reply(original_msg, english_msg, reply_lang=rl)
        if warm:
            return {"action": "text", "data": warm}
    if not skip_greeting and len(words) <= 4 and all(w in greetings for w in words):
        rl = customer_reply_language(original_msg)
        warm = build_warm_conversation_reply(original_msg, english_msg, reply_lang=rl)
        if warm:
            return {"action": "text", "data": warm}

    # 2. INSTANT REJECTION & OFF-TOPIC (Static Guardrails)
    competitors = ["amazon", "flipkart", "myntra", "meesho", "ajio", "snapdeal", "shopsy", "glowroad", "zomato", "swiggy", "groww"]
    if any(f" {c} " in text for c in competitors):
        from services.support_scope import build_other_company_support_decline, message_mentions_other_company_support

        if message_mentions_other_company_support(original_msg, english_msg):
            polite = build_other_company_support_decline(
                original_msg, reply_lang=customer_reply_language(original_msg)
            )
            if polite:
                return {"action": "text", "data": polite}
        return {"action": "reject", "data": sysmsg("off_topic_polite") or "I can only help with Welfog."}


    # Off-topic / chit-chat: handled by AI conversation_scope after routing (no keyword lists here).

    # 3z. Place a new order on Welfog (Hinglish / typos) — not product SKU search
    if any(w in text for w in ("welfog", "welfrog", "welkog")):
        if any(
            x in text
            for x in (
                "order kaise", "order kese", "order kes", "oder kaise", "oder kese", " oder ",
                "place order", "order kar", "order kr", "order karna", "order krna", "order dal",
            )
        ):
            if any(x in text for x in ("kaise", "kese", "how", "step", "tarika", "kya", "kru", "kro", "karna", "bhai", " pr ", " pe ", "par ")):
                return {"action": "text", "data": sysmsg("order_placement_help")}
    if (any(w in text for w in ("welfog", "welfrog", "welkog")) or "order" in text) and any(
        x in text for x in ("step", "steps")
    ) and any(x in text for x in ("puchh", "pooch", "puch", "bol diya", "bola tha")):
        return {"action": "text", "data": sysmsg("order_placement_help")}

    from utils.helpers import (
        message_is_welfog_about_request,
        message_needs_policy_answer,
        message_needs_support_not_product,
    )

    if message_is_welfog_about_request(text):
        from services.translation_service import customer_reply_language

        reply_lang = customer_reply_language(original_msg)
        about = format_welfog_about_reply_from_kb(
            original_msg, english_msg, reply_lang=reply_lang
        )
        if about:
            return {"action": "text", "data": about}

    from utils.helpers import message_is_knowledge_information_request

    from utils.helpers import message_asks_other_company_policy

    if message_asks_other_company_policy(text):
        from services.translation_service import customer_reply_language, localize_for_customer

        rl = customer_reply_language(original_msg)
        if rl == "hinglish":
            polite = sysmsg("off_topic_other_company_policy_hinglish")
        else:
            polite = sysmsg("off_topic_other_company_policy") or sysmsg("off_topic_polite")
            if polite and rl not in ("en", "hinglish"):
                polite = localize_for_customer(polite, rl)
        return {"action": "text", "data": polite or sysmsg("out_of_domain")}

    if message_is_knowledge_information_request(text):
        from services.translation_service import customer_reply_language

        reply_lang = customer_reply_language(original_msg)
        info = format_knowledge_information_reply_from_kb(
            original_msg, english_msg, reply_lang=reply_lang
        )
        if info:
            return {"action": "text", "data": info}

    if message_needs_support_not_product(text):
        cc = format_customer_care_reply_from_kb(original_msg, english_msg)
        if cc:
            return {"action": "text", "data": cc}

    # 3. ORDER ID / TRACKING HELP SHORT-CIRCUIT
    if _text_has_refund_or_return_intent(text) and not _text_needs_order_id_for_refund_or_payment(text):
        return {"action": "text", "data": sysmsg("refund_payment_help")}
    if _text_is_tracking_howto_request(text) or (
        _text_is_order_tracking_intent(text) and not _text_needs_order_id_for_tracking(text)
    ):
        from services.translation_service import customer_reply_language, localize_for_customer

        rl = customer_reply_language(original_msg)
        if rl == "hinglish":
            body = sysmsg("tracking_help_hinglish") or sysmsg("tracking_help")
        else:
            body = localize_for_customer(sysmsg("tracking_help") or "", rl)
        return {"action": "text", "data": body}
    
    from utils.helpers import (
        _text_asks_how_to_view_wishlist,
        _text_asks_wishlist,
        _text_is_order_id_help_request,
        _turn_blocks_wishlist_howto_routing,
    )

    if _text_is_order_id_help_request(text):
        from services.translation_service import customer_reply_language, localize_for_customer

        rl = customer_reply_language(original_msg)
        if rl == "hinglish":
            body = sysmsg("order_id_help_hinglish") or sysmsg("order_id_help")
        else:
            body = localize_for_customer(sysmsg("order_id_help") or "", rl)
        if body:
            return {"action": "text", "data": body}

    if _text_asks_how_to_view_wishlist(text) and not _turn_blocks_wishlist_howto_routing(text):
        from services.translation_service import customer_reply_language, localize_for_customer

        rl = customer_reply_language(original_msg)
        if rl == "hinglish":
            body = sysmsg("wishlist_help_hinglish") or sysmsg("wishlist_help")
        else:
            body = sysmsg("wishlist_help") or ""
            if body and rl not in ("en", "hinglish"):
                body = localize_for_customer(body, rl)
        if body:
            return {"action": "text", "data": body}

    if _text_asks_wishlist(text):
        return {"action": "wishlist"}

    # 3b. ORDER HISTORY HELP — order_ai_flow (AI + API KB) decides howto vs list

    # 4. DIRECT ORDER ID ENTRY (4–20 digits or alphanumeric; 6-digit PIN is not an order id)
    from utils.helpers import (
        _text_has_delivery_serviceability_intent,
        _text_is_product_id_lookup_context,
        extract_product_id,
    )

    if _text_has_delivery_serviceability_intent(original_msg, english_msg):
        return None

    if _text_is_product_id_lookup_context(text) or extract_product_id(text):
        return None

    order_match = next((w for w in words if _is_plausible_order_id(w, context=text, shallow=True)), None)
    if order_match:
        if _text_has_delivery_serviceability_intent(original_msg, english_msg):
            return None
        order_text = order_match.lower()
        if not re.fullmatch(r"[1-9]\d{5}", order_text):
            order_context = any(
                x in text
                for x in [
                    "order", "track", "tracking", "refund", "payment", "return", "cancel",
                    "shipment", "delivery status", "order status", "order id", "orderid",
                    " le id", "lo id", "yeh le", "ye le",
                    "kab", "bta", "bata", "aaega", "aayega", "check",
                ]
            ) and not _text_is_product_id_lookup_context(text)
            if len(words) == 1 or order_context:
                return {"action": "direct_order_id", "order_id": order_match.upper()}
        elif re.fullmatch(r"[1-9]\d{5}", order_text) and any(
            x in text for x in ("track", "tracking", "order status", "kab ", "kab aayega", "live status")
        ):
            if not _text_has_delivery_serviceability_intent(original_msg, english_msg):
                return {"action": "direct_order_id", "order_id": order_match}

    # 5. INSTANT SOCIAL MEDIA — same KB formatter as AI-first (multi-platform + typos)
    from utils.helpers import message_asks_other_company_social_media, message_asks_welfog_social_media

    social_comb = f"{original_msg} {english_msg}"
    if message_asks_other_company_social_media(social_comb):
        from services.support_scope import build_other_company_social_decline

        decline = build_other_company_social_decline(original_msg)
        if decline:
            return {"action": "reject", "data": decline}
    if message_asks_welfog_social_media(social_comb):
        body = format_welfog_social_media_reply_from_kb(original_msg, english_msg)
        if body:
            return {"action": "text", "data": body}
    # 🔥 SAB KUCH HATA DIYA! Koi keyword matching nahi. 
    # Ab chahe Telugu me puche ya Hindi me, direct AI Brain decide karega!
    return None
