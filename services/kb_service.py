import hashlib
import os
import random
import re
import threading
from html import escape as html_escape

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
_KB_CATALOG_CACHE: dict[str, str] = {}

# Unified semantic thresholds (env-tunable; cosine similarity 0–1)
KB_SEMANTIC_MIN_SCORE = float(os.getenv("KB_SEMANTIC_MIN_SCORE", "0.16") or "0.16")
KB_CONTEXT_MIN_SCORE = float(os.getenv("KB_CONTEXT_MIN_SCORE", "0.14") or "0.14")
KB_DIRECT_MIN_SCORE = float(os.getenv("KB_DIRECT_MIN_SCORE", "0.16") or "0.16")
KB_STRONG_MATCH_SCORE = float(os.getenv("KB_STRONG_MATCH_SCORE", "0.28") or "0.28")
# Soft answer floor — retrieve/recall above this; gray-band hits still answer via extractive path.
KB_ANSWER_MIN_CONFIDENCE = float(os.getenv("KB_ANSWER_MIN_CONFIDENCE", "0.18") or "0.18")
KB_RETRIEVAL_STRONG_SCORE = float(os.getenv("KB_RETRIEVAL_STRONG_SCORE", "0.34") or "0.34")


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


def _knowledge_key_from_title(title: str) -> str:
    """Normalize MySQL document title to the same key formerly derived from .txt filenames."""
    from services.knowledge_keys import canonical_knowledge_key

    return canonical_knowledge_key(title)


def _is_internal_kb_key(key: str) -> bool:
    from services.knowledge_keys import is_internal_agent_knowledge_key

    return is_internal_agent_knowledge_key(key)


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
    Active knowledge documents from MySQL as {key: content}.
    Production runtime must not read support/knowledge/*.txt.
    """
    runtime: dict[str, str] = {}
    try:
        from services.mysql_service import list_active_knowledge_documents

        docs = list_active_knowledge_documents()
    except Exception as e:
        print(f"get_runtime_knowledge_files MySQL error: {e}")
        return runtime

    for doc in docs:
        title = (doc.get("title") or "").strip()
        if not title:
            continue
        key = _knowledge_key_from_title(title)
        if not key:
            continue
        if key in runtime:
            n = 2
            while f"{key}_{n}" in runtime:
                n += 1
            key = f"{key}_{n}"
        runtime[key] = doc.get("content") or ""
    return runtime


def get_allowed_knowledge_filenames():
    """Admin UI labels — title.txt for URL compatibility; content lives in MySQL."""
    try:
        from services.mysql_service import list_active_knowledge_documents

        titles = [
            (doc.get("title") or "").strip()
            for doc in list_active_knowledge_documents()
            if (doc.get("title") or "").strip()
        ]
    except Exception as e:
        print(f"get_allowed_knowledge_filenames MySQL error: {e}")
        return []
    return sorted({f"{t}.txt" for t in titles})


def _compute_kb_snapshot(runtime_files):
    """Detect any add/update/delete via content hash of MySQL-backed documents."""
    parts = []
    for key, content in sorted((runtime_files or {}).items()):
        raw = (content if isinstance(content, str) else str(content or "")).encode(
            "utf-8", errors="replace"
        )
        digest = hashlib.md5(raw).hexdigest()[:16]
        parts.append(f"{key}:{len(raw)}:{digest}")
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
    # Section headings ("How to return…:") are not FAQ question echoes.
    if s.endswith(":") and not s.endswith("?:") and "?" not in s:
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
    if "social media links" in low and ("support rule" in low or "never invent" in low):
        return True
    if re.search(
        r"\bnever invent\b|\bonly share welfog\b|\bpolitely decline\b|\brouting rule\b|"
        r"\bnever guess handles\b|\bgive only the requested platform\b|"
        r"\bif user asks for welfog social\b|\bfor other brands\b.*\bdecline\b",
        low,
    ):
        return True
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    agent_bullets = sum(
        1
        for ln in lines
        if re.match(r"^-\s+if user", ln, re.I)
        or re.match(r"^-\s+follow-up", ln, re.I)
        or re.match(r"^—\s+only share", ln, re.I)
        or re.match(r"^-\s+only share", ln, re.I)
    )
    if agent_bullets >= 2:
        return True
    # Do NOT drop real customer policy (content rules use never/must).
    # Only agent playbooks with explicit routing shape.
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


def _kb_block_has_identity_markers(text: str) -> bool:
    """Owner/about markers — used during chunk split (must stay above load index)."""
    low = re.sub(r"<br\s*/?>", " ", (text or "").lower())
    return bool(
        re.search(
            r"\b(owner|founder|co-?founder|ceo|about welfog|our story)\b",
            low,
        )
    )


def _kb_block_has_address_markers(text: str) -> bool:
    """Postal address markers without identity facts."""
    if _kb_block_has_identity_markers(text):
        return False
    low = re.sub(r"<br\s*/?>", " ", (text or "").lower())
    return any(
        x in low for x in ("address", "plot", "pin code", "pincode", "street", "colony")
    )


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
        # Never glue identity/owner facts onto a postal address block —
        # that made "who is the owner" miss after address demotion.
        if len(acc) < 160:
            if (
                _kb_block_has_identity_markers(acc)
                and _kb_block_has_address_markers(p)
            ) or (
                _kb_block_has_address_markers(acc)
                and _kb_block_has_identity_markers(p)
            ):
                merged.append(acc)
                acc = p
            else:
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
    for k, content in files_map.items():
        try:
            if content is None:
                print(f"Warning: Empty knowledge document for key={k}")
                chunks_by_key[k] = []
                vectors_by_key[k] = []
                continue

            text = content if isinstance(content, str) else str(content)

            chunks = [
                c
                for c in _split_kb_chunks(text, file_key=k)
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
            print(f"Error loading knowledge key={k}: {e}")
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
            sys_body = runtime_files.get("system_messages")
            if isinstance(sys_body, str) and sys_body.strip():
                _SYSTEM_MESSAGES = _parse_system_messages(sys_body)
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
    global _KB_CATALOG_CACHE
    _KB_CATALOG_CACHE.clear()
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


def build_kb_catalog_for_brain_prompt(*, max_files: int = 500, blurb_chars: int = 140) -> str:
    """
    Auto-built catalog of ALL active Admin knowledge documents for the routing LLM.
    Any document added via Admin appears here — no code change for new topics.
    """
    global _KB_CATALOG_CACHE
    _schedule_kb_refresh_if_stale()
    snap = (_KB_SNAPSHOT or "").strip() or "empty"
    cache_key = f"{snap}|{max_files}|{blurb_chars}"
    cached = _KB_CATALOG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lines: list[str] = []
    for k in get_customer_kb_keys()[: max(1, int(max_files))]:
        blurb = ""
        for chunk in (kb_chunks_by_key.get(k) or [])[:2]:
            plain = _plain_chunk_for_embed(chunk).strip()
            if plain:
                blurb = re.sub(r"\s+", " ", plain)[:blurb_chars]
                break
        if not blurb:
            body = get_runtime_knowledge_files().get(k)
            if isinstance(body, str) and body.strip():
                blurb = re.sub(r"\s+", " ", body.strip())[:blurb_chars]
        lines.append(f'  "{k}": {blurb or "(admin knowledge — read file at answer time)"}')
    out = "\n".join(lines)
    if len(_KB_CATALOG_CACHE) >= 4:
        _KB_CATALOG_CACHE.clear()
    _KB_CATALOG_CACHE[cache_key] = out
    return out


def resolve_brain_kb_keys(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    *,
    max_files: int = 4,
    conversation_context: str = "",
) -> list[str]:
    """
    Soft Brain kb_keys only — never a hardcoded file catalog.

    Valid Brain keys are kept as retrieval hints. Missing/invalid keys fall back
    to semantic file picking across ALL customer Admin documents. Empty result
    means unscoped Qdrant retrieval (pure semantic).
    """
    ensure_knowledge_cache_fresh()
    route = route or {}
    customer = set(get_customer_kb_keys())
    raw_brain_keys = [
        _knowledge_key_from_title(str(k))
        for k in (route.get("kb_keys") or [])
        if str(k).strip()
    ]
    raw_brain_keys = [k for k in raw_brain_keys if k]
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

    # Brain suggested unknown keys — semantic pick across all Admin docs.
    if raw_brain_keys or original_msg or msg_en:
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

    # No hardcoded intent→file map. Empty → unscoped semantic retrieval.
    return []


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
    """Raw UTF-8 text from MySQL knowledge documents — always read live after cache check."""
    ensure_knowledge_cache_fresh()
    if not keys:
        return ""
    files_map = get_runtime_knowledge_files()
    parts = []
    for k in keys:
        body = files_map.get(k)
        if not isinstance(body, str):
            continue
        body = body.strip()
        if body:
            parts.append(f"=== FILE key={k} path={k}.txt ===\n{body}")
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
        # Numbered how-to steps (1) Log in…) are BODY, not section titles.
        # Only treat numbered lines as headings when they are short titles ending with ':'.
        numbered_title = bool(
            re.match(r"^\d+[.)]\s+.+:\s*$", stripped) and len(stripped) < 90
        )
        is_heading = bool(
            re.match(r"^[A-Z0-9][^:\n]{2,100}:\s*$", stripped)
            or numbered_title
            or (
                stripped.endswith(":")
                and len(stripped) < 90
                and not stripped.startswith("-")
                and not re.match(r"^\d+[.)]\s+", stripped)
            )
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


def _brain_focus_text(question: str, ai_route: dict | None = None) -> str:
    """English focus string from Brain meaning (preferred) or retrieval query."""
    if isinstance(ai_route, dict):
        meaning = (ai_route.get("user_meaning") or "").strip()
        if meaning:
            return meaning
    return (question or "").strip()


def _brain_english_meaning_is_narrow_fact(ai_route: dict | None) -> bool:
    """
    Single-fact asks only (owner / contact / social).

    Never treat short policy/how-to meanings as "narrow" — that caused
    "Refund Policy:" stubs and half-step replies. Word-count heuristics are
    forbidden; Brain intent + meaning helpers decide.
    """
    if not isinstance(ai_route, dict):
        return False
    intent = (ai_route.get("intent") or "").strip().lower()
    if intent in ("owner", "contact", "social", "social_media"):
        return True
    if _brain_meaning_asks_contact(ai_route) or _brain_meaning_asks_social(ai_route):
        return True
    if _brain_meaning_asks_owner_name(ai_route):
        return True
    return False


def _infer_kb_answer_budget(
    question: str, *, ai_route: dict | None = None
) -> tuple[int, int, int]:
    """Narrow identity facts → short. Policy/how-to → full section room."""
    if _brain_english_meaning_is_narrow_fact(ai_route):
        return 420, 4, 1
    # Default room for complete FAQ/policy bodies (admin docs may be long).
    return 1100, 14, 2


def _is_kb_heading_only_line(line: str) -> bool:
    """True for bare titles like 'Refund Policy:' with no answer prose."""
    s = (line or "").strip()
    if not s:
        return True
    if _looks_like_kb_metadata_label(s) or _looks_like_faq_question_line(s):
        return True
    if s.endswith(":") and len(s) <= 90:
        return True
    # Short title-like line without sentence punctuation.
    if len(s) <= 48 and not re.search(r"[.!?。]", s) and not re.match(r"^\d+[.)]\s+", s):
        words = re.findall(r"[A-Za-z]+", s)
        if words and sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1):
            return True
    return False


def _looks_like_kb_gap_reply(text: str) -> bool:
    """True for honest 'not in KB' fallbacks — must not short-circuit better paths."""
    plain = re.sub(r"<[^>]+>", " ", text or "")
    plain = re.sub(r"\s+", " ", plain).strip().lower()
    if not plain:
        return True
    markers = (
        "don't have that specific detail",
        "do not have that specific detail",
        "couldn't find this in welfog",
        "could not find this in welfog",
        "not in the available knowledge",
        "not in welfog's current knowledge",
        "current knowledge base right now",
    )
    return any(m in plain for m in markers)


def _kb_answer_body_is_usable(text: str, *, min_chars: int = 28) -> bool:
    """Reject heading-only / empty extractive slices before they reach the customer."""
    plain = re.sub(r"<[^>]+>", " ", text or "")
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain or len(plain) < min_chars:
        return False
    if _looks_like_kb_gap_reply(plain):
        return False
    if _is_kb_heading_only_line(plain):
        return False
    # Title + almost nothing ("Refund Policy: yes") is still useless.
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    body_lines = [ln for ln in lines if not _is_kb_heading_only_line(ln)]
    if not body_lines:
        return False
    body = " ".join(body_lines).strip()
    return len(body) >= min_chars


def _looks_like_step_list(text: str) -> bool:
    """Numbered / bullet how-to blocks — keep contiguous steps in document order."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    numbered = sum(1 for ln in lines if re.match(r"^\d+[.)]\s+\S", ln))
    bullets = sum(1 for ln in lines if re.match(r"^[-•]\s+\S", ln))
    return numbered >= 2 or bullets >= 2


def _excerpt_preserving_structure(text: str, max_chars: int, max_lines: int) -> str:
    """
    Truncate by budget but never leave a heading-only answer, and never scramble
    step order (1) then later 3) without 2).
    """
    plain = (text or "").strip()
    if not plain:
        return ""
    lines = [ln.rstrip() for ln in plain.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Drop leading bare headings with no following body in the kept window.
    while lines and _is_kb_heading_only_line(lines[0]) and len(lines) > 1:
        # Keep heading if next line is body; only drop if heading is alone later.
        break
    if _looks_like_step_list(plain):
        kept: list[str] = []
        total = 0
        started = False
        for ln in lines:
            is_step = bool(
                re.match(r"^\d+[.)]\s+\S", ln) or re.match(r"^[-•]\s+\S", ln)
            )
            if _is_kb_heading_only_line(ln) and not is_step:
                # Skip leading bare titles; stop at the next section once we have body.
                if kept and started:
                    break
                if not kept:
                    continue
                # Mid-block title with no body yet — skip.
                continue
            if total + len(ln) > max_chars and kept:
                break
            kept.append(ln)
            total += len(ln) + 1
            started = True
            if len(kept) >= max_lines:
                break
        out = "\n".join(kept).strip()
        if out and not _is_kb_heading_only_line(out):
            return out
    # Prose / FAQ answer: keep document order, skip orphan titles / next sections.
    kept = []
    total = 0
    started = False
    for i, ln in enumerate(lines):
        if _is_kb_heading_only_line(ln):
            if started and kept:
                # Next FAQ/policy heading — stop (avoid gluing unrelated sections).
                break
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not kept and nxt and not _is_kb_heading_only_line(nxt):
                kept.append(ln)
                total += len(ln) + 1
            continue
        if total + len(ln) > max_chars and kept:
            break
        kept.append(ln)
        total += len(ln) + 1
        started = True
        if len(kept) >= max_lines:
            break
    out = "\n".join(kept).strip()
    # Never ship a heading-only reply.
    if out and _is_kb_heading_only_line(out):
        return ""
    return out


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

    # Never use DOTALL across paragraphs — glued FAQ pairs would lose the first answer
    # (cancel answer eaten because a later "?" begins the next FAQ).
    inline = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I).strip()
    lines = [ln.strip() for ln in inline.splitlines() if ln.strip()]
    if not lines:
        return inline
    first = lines[0]
    if _looks_like_faq_question_line(first) or _looks_like_kb_metadata_label(first):
        rest = "\n".join(lines[1:]).strip()
        if rest:
            return rest
    # Single-line "Question? Answer" on the same line only.
    if "?" in first and "\n" not in first:
        m = re.match(r"^(.{10,240}\?)\s*:?\s+(.{15,})$", first)
        if m and _looks_like_faq_question_line(m.group(1).strip()):
            return m.group(2).strip()
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
        # Only drop seller/vendor label lines — never drop real how-to section titles
        # like "How to return on Welfog (app or website):".
        if has_steps and len(s) < 100 and not re.match(r"^\d+\)", s) and not s.startswith(("-", "•")):
            if re.search(r"\b(supplier|vendor|seller account)\b", s, re.I):
                continue
            if s.endswith(")") and not s.endswith(":"):
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
    """Join admin KB section — never emit a bare heading with empty body."""
    t = (title or "").strip()
    b = (body or "").strip()
    if _looks_like_kb_metadata_label(t):
        t = ""
    if not b:
        # Heading-only is not a customer answer.
        return ""
    if not t or _looks_like_faq_question_line(t):
        return b
    if b.lower().startswith(t.lower()):
        return b
    if len(t) < 80 and not t.endswith("?"):
        return f"{t}:\n{b}"
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


def _finalize_kb_customer_reply(
    body: str, original_msg: str, lang: str, *, ai_route: dict | None = None
) -> str:
    """KB-sourced reply — match customer language via AI polish (no language keyword maps)."""
    from services.translation_service import (
        finalize_customer_reply,
        resolve_customer_reply_lang,
        _reply_needs_style_rewrite,
    )

    if not (body or "").strip():
        return ""
    brain_lang = ""
    if isinstance(ai_route, dict):
        brain_lang = str(ai_route.get("reply_lang") or "").strip()
    rl = resolve_customer_reply_lang(original_msg, brain_lang or lang)
    # Bounded rewrite when English KB ≠ customer style. LLM reads CUSTOMER WROTE
    # and matches language — no Hinglish/English phrase hardcoding here.
    allow_rewrite = False
    try:
        allow_rewrite = bool(_reply_needs_style_rewrite(body, rl, original_msg))
    except Exception:
        allow_rewrite = False
    out = finalize_customer_reply(
        body, original_msg, rl, allow_llm_style_rewrite=allow_rewrite
    )
    if allow_rewrite and out and not _rewrite_preserves_kb_facts(body, out):
        return finalize_customer_reply(
            body, original_msg, rl, allow_llm_style_rewrite=False
        )
    return out


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


def _kb_focus_profile(
    question: str, *, ai_route: dict | None = None
) -> tuple[set[str], set[str], int]:
    """
    Section budget only — no topic include/exclude keyword maps.
    Matching is Brain meaning + embeddings over live admin KB text.
    """
    _, _, max_sections = _infer_kb_answer_budget(question, ai_route=ai_route)
    return set(), set(), max_sections


def _filter_kb_blob_fast(
    blob: str,
    question: str,
    *,
    max_chars: int | None = None,
    ai_route: dict | None = None,
) -> str:
    """
    Fast extractive slice after Qdrant ranked the chunk.

    Prefer best section by Brain meaning; keep step lists / policy bodies intact.
    Never return a bare heading. Aggressive sentence trim only for true narrow facts.
    """
    if _chunk_is_agent_instruction_blob(blob):
        return ""
    display = _strip_kb_blob_for_display(blob)
    if not display.strip() or _chunk_is_agent_instruction_blob(display):
        return ""
    budget_chars, budget_lines, budget_sections = _infer_kb_answer_budget(
        question, ai_route=ai_route
    )
    if max_chars is None:
        max_chars = budget_chars

    plain = _strip_leading_faq_question(display)
    plain = _strip_kb_policy_heading_prefix(plain)
    identity = _brain_meaning_asks_company_identity(ai_route)
    focus = _brain_focus_text(question, ai_route)
    narrow = _brain_english_meaning_is_narrow_fact(ai_route)

    # Narrow facts: pick best sentence(s). How-to / policy: keep structure.
    if narrow:
        focused_plain = _prefer_fact_lines_for_question(
            plain, question, ai_route=ai_route, force=True
        )
        if focused_plain and focused_plain.strip() and not _is_kb_heading_only_line(
            focused_plain
        ):
            return _kb_plain_excerpt(focused_plain, max_chars=max_chars)

    sections = _extract_kb_sections_from_blob(plain)
    if not sections:
        if identity and _chunk_looks_like_address_block(plain):
            return ""
        structured = _excerpt_preserving_structure(plain, max_chars, budget_lines)
        return structured or _kb_plain_excerpt(plain, max_chars=max_chars)

    scored: list[tuple[float, str, str]] = []
    for title, body in sections:
        if not body and not title:
            continue
        # Skip empty-body headings — they produce "Refund Policy:" only replies.
        if not (body or "").strip():
            continue
        hay = f"{title} {body}".lower()
        if identity and any(
            x in hay for x in ("plot-", "pin code", "pincode", "sirsi", "sukhijha")
        ) and not _chunk_has_identity_fact(hay):
            continue
        score = _score_kb_section_relevance(focus or question, title, body)
        # How-to meanings should prefer sections that actually contain step lists.
        fl = (focus or question or "").lower()
        if re.search(r"\b(how to|how do|how can|steps?|procedure)\b", fl):
            if _looks_like_step_list(body) or _looks_like_step_list(f"{title}\n{body}"):
                score += 2.8
        if score > 0.05 or len(sections) == 1:
            scored.append((score if score > 0 else 0.1, title, body))

    if not scored:
        # Fall back to whole blob with structure (e.g. FAQ Q&A without heading sections).
        structured = _excerpt_preserving_structure(plain, max_chars, budget_lines)
        if structured:
            return structured
        focused = _prefer_fact_lines_for_question(
            plain, question, ai_route=ai_route, force=True
        )
        if focused and not _is_kb_heading_only_line(focused):
            return _kb_plain_excerpt(focused, max_chars=max_chars)
        return _kb_plain_excerpt(plain, max_chars=max_chars)

    scored.sort(key=lambda x: x[0], reverse=True)
    fl = (focus or question or "").lower()
    howto_ask = bool(
        re.search(r"\b(how to|how do|how can|steps?|procedure|guide)\b", fl)
    )
    # How-to asks: pick the best step-list section as the answer (avoid pairing
    # a short related policy blurb that would bury or truncate steps).
    if howto_ask:
        step_ranked = [
            s
            for s in scored
            if _looks_like_step_list(s[2])
            or _looks_like_step_list(f"{s[1]}\n{s[2]}")
        ]
        if step_ranked:
            scored = step_ranked
    # Prefer one clear best section. Only merge a second when scores are close
    # and the top section is not already a full how-to / policy body.
    take_n = 1
    if len(scored) >= 2 and budget_sections >= 2:
        top_sc, top_title, top_body = scored[0]
        second_sc = scored[1][0]
        top_block = f"{top_title}\n{top_body}"
        top_is_complete = (
            _looks_like_step_list(top_body)
            or _looks_like_step_list(top_block)
            or len((top_body or "").strip()) >= 120
        )
        if (not top_is_complete) and (top_sc - second_sc) < 1.2:
            take_n = 2
    parts: list[str] = []
    total = 0
    for _, title, body in scored[:take_n]:
        block = _format_kb_section_block(title, body)
        if not block.strip() or _is_kb_heading_only_line(block):
            # Body alone if title formatting failed.
            block = (body or "").strip()
        if not block.strip():
            continue
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    out = "\n\n".join(parts).strip() or plain

    # Never scramble how-to steps with line re-ranking.
    if narrow and not _looks_like_step_list(out):
        focused = _prefer_fact_lines_for_question(
            out, question, ai_route=ai_route, force=True
        )
        if focused and not _is_kb_heading_only_line(focused):
            out = focused

    structured = _excerpt_preserving_structure(out, max_chars, budget_lines)
    if structured and not _is_kb_heading_only_line(structured):
        return structured
    # Last resort: full best body clipped (still better than a bare title).
    for _, title, body in scored:
        b = (body or "").strip()
        if b and not _is_kb_heading_only_line(b):
            return _kb_plain_excerpt(b, max_chars=max_chars)
    return _kb_plain_excerpt(plain, max_chars=max_chars)


def _meaning_line_score(focus: str, line: str, *, emb: float | None = None) -> float:
    """Score a KB line against Brain meaning — embeddings + dynamic token overlap."""
    if not (focus or "").strip() or not (line or "").strip():
        return 0.0
    if _looks_like_faq_question_line(line) or _looks_like_kb_metadata_label(line):
        return -1.0
    if emb is None:
        emb = _embedding_similarity(focus[:400], line[:500])
    overlap = _token_overlap_ratio(focus, line)
    return (float(emb) * 4.0) + (overlap * 2.5)


def _batch_meaning_line_scores(focus: str, lines: list[str]) -> list[tuple[float, str]]:
    """Score many lines with one embed batch when possible."""
    usable = [
        ln
        for ln in lines
        if ln.strip()
        and not _looks_like_faq_question_line(ln)
        and not _looks_like_kb_metadata_label(ln)
    ]
    if not usable or not (focus or "").strip():
        return []
    emb_map: dict[str, float] = {}
    try:
        vecs = encode_texts([focus[:400]] + [ln[:500] for ln in usable])
        if vecs is not None and len(vecs) == len(usable) + 1:
            sims = cosine_similarity(vecs[0:1], vecs[1:])[0]
            for ln, sim in zip(usable, sims):
                emb_map[ln] = float(sim)
    except Exception:
        emb_map = {}
    return [
        (_meaning_line_score(focus, ln, emb=emb_map.get(ln)), ln) for ln in usable
    ]


def _prefer_fact_lines_for_question(
    text: str,
    question: str,
    *,
    ai_route: dict | None = None,
    force: bool = False,
) -> str:
    """
    Keep sentences that best match Brain English meaning (embedding + overlap).
    Works for any admin-added topic — no cancel/return/shipping keyword maps.
    """
    plain = (text or "").strip()
    if not plain:
        return plain
    focus = _brain_focus_text(question, ai_route) or (question or "").strip()
    if not force and not _brain_english_meaning_is_narrow_fact(ai_route):
        if not focus or len(re.split(r"(?<=[.!?])\s+|\n+", plain)) < 2:
            return plain

    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    sents: list[str] = []
    for ln in lines or [plain]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if len(s.strip()) > 12]
        sents.extend(parts if parts else ([ln] if ln else []))
    if not sents:
        return plain
    if len(sents) == 1:
        return sents[0]

    ranked = sorted(_batch_meaning_line_scores(focus, sents), key=lambda x: x[0], reverse=True)
    if not ranked:
        return plain
    # Never choose a bare heading as the "best fact".
    ranked = [(sc, s) for sc, s in ranked if not _is_kb_heading_only_line(s)]
    if not ranked:
        return plain
    best_sc, best = ranked[0]
    if best_sc < 0.35:
        return plain
    margin = 0.12
    top = [s for sc, s in ranked if sc >= best_sc - margin and sc >= 0.35][:2]
    if _brain_meaning_asks_owner_name(ai_route):
        id_lines = [s for s in top if _chunk_has_identity_fact(s)]
        if id_lines:
            top = id_lines[:1]
    return "\n".join(top).strip() if top else plain


def _filter_kb_blob_for_question(
    blob: str,
    question: str,
    *,
    max_chars: int | None = None,
    ai_route: dict | None = None,
) -> str:
    """
    Keep only KB sections/lines that match the user's question.
    Fully driven by live admin .txt content + embeddings — no topic names in code.
    """
    if _chunk_is_agent_instruction_blob(blob):
        return ""
    display = _strip_kb_blob_for_display(blob)
    if not display.strip():
        return ""
    budget_chars, budget_lines, budget_sections = _infer_kb_answer_budget(
        question, ai_route=ai_route
    )
    if max_chars is None:
        max_chars = budget_chars

    sections = _extract_kb_sections_from_blob(display)
    if not sections:
        hit = best_kb_hit(question, min_score=0.20, ai_route=ai_route)
        if hit and hit.get("chunk"):
            plain = re.sub(r"<br\s*/?>", "\n", hit["chunk"])
            return _kb_plain_excerpt(plain, max_chars=max_chars)
        return _kb_plain_excerpt(display, max_chars=max_chars)

    _, _, max_sections = _kb_focus_profile(question, ai_route=ai_route)
    max_sections = min(max_sections, budget_sections)
    focus = _brain_focus_text(question, ai_route) or question
    scored: list[tuple[float, str, str]] = []
    for title, body in sections:
        if not body and not title:
            continue
        score = _score_kb_section_relevance(focus, title, body)
        if score > 0.05:
            scored.append((score, title, body))

    if not scored:
        focused = _prefer_fact_lines_for_question(
            display, question, ai_route=ai_route, force=True
        )
        return _kb_plain_excerpt(focused or display, max_chars=max_chars)

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score = scored[0][0]
    # Drop sections far below the best match — avoids dumping unrelated paragraphs
    # (discounts / social links glued onto a narrow reels/content question).
    scored = [(s, t, b) for s, t, b in scored if s >= top_score * 0.72]
    # Prefer a single best section for short / focused turns.
    if max_sections > 1 and len(scored) > 1 and scored[0][0] >= (scored[1][0] + 0.8):
        max_sections = 1

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

    # Keep full best section for policy/how-to. Aggressive 1–2 line trim
    # only for true narrow identity/contact facts (never for "Refund Policy").
    if _brain_english_meaning_is_narrow_fact(ai_route) and not _looks_like_step_list(out):
        focused = _prefer_fact_lines_for_question(
            out, question, ai_route=ai_route, force=True
        )
        if focused and _kb_answer_body_is_usable(focused):
            return _kb_plain_excerpt(focused, max_chars=max_chars)

    structured = _excerpt_preserving_structure(out, max_chars, budget_lines)
    if structured and _kb_answer_body_is_usable(structured):
        return structured
    return _kb_plain_excerpt(out, max_chars=max_chars)


def build_kb_retrieval_query(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    ai_route: dict | None = None,
) -> str:
    """
    Multilingual retrieval query for embedding search.

    Always includes an English gloss (Brain meaning or retrieval translate) so
    Hinglish/Hindi/other Indian-language questions align with English KB chunks.
    Distinct parts joined with ' — ' for multi-query RRF; no keyword lists.
    """
    from services.translation_service import (
        resolve_customer_reply_lang,
        text_usable_as_english_retrieval,
        to_en_for_retrieval,
        _latin_script_only,
        is_hinglish_message,
    )

    raw = (original_msg or "").strip()
    meaning = ((ai_route or {}).get("user_meaning") or "").strip()
    en = (msg_en or "").strip()
    reply_lang = resolve_customer_reply_lang(
        raw, str((ai_route or {}).get("reply_lang") or "")
    )
    preflight = bool((ai_route or {}).get("_preflight_kb"))
    ctx_tail = (conversation_context or "")[-800:]
    cache_key = f"{raw}|{en}|{meaning}|{reply_lang}|{preflight}|{ctx_tail}"
    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        qcache = get_kb_turn_cache("kb_retrieval_query_cache") or {}
        hit = qcache.get(cache_key)
        if isinstance(hit, str) and hit.strip():
            return hit
    except ImportError:
        qcache = None

    english_gloss = ""
    # Prefer Brain meaning first — never wait on Google when brain already glossed.
    # Trust Latin non-Hinglish brain paraphrases even if conversational detectors miss them.
    if meaning:
        if text_usable_as_english_retrieval(meaning):
            english_gloss = meaning
        elif (
            _latin_script_only(meaning)
            and not is_hinglish_message(meaning)
            and meaning.lower() != raw.lower()
        ):
            english_gloss = meaning
        else:
            meaning_en = to_en_for_retrieval(meaning, reply_lang)
            english_gloss = meaning_en or meaning
    elif en and text_usable_as_english_retrieval(en) and en.lower() != raw.lower():
        english_gloss = en
    elif raw:
        english_gloss = to_en_for_retrieval(raw, reply_lang)

    parts: list[str] = []
    seen: set[str] = set()

    def _add(part: str) -> None:
        p = (part or "").strip()
        if not p:
            return
        key = p.lower()
        if key in seen:
            return
        seen.add(key)
        parts.append(p)

    _add(english_gloss)
    # Keep raw only when it differs — fulltext/hybrid can still use customer wording.
    if raw and raw.lower() != (english_gloss or "").lower():
        _add(raw)

    merged = " — ".join(parts).strip()
    route = ai_route or {}
    # Do NOT append "topic: …" into the embed string — it creates fake variants and
    # dilutes similarity. Soft kb_keys already rerank in the retriever.

    if route.get("_preflight_kb"):
        out = _normalize_retrieval_query(merged or english_gloss or raw)
        try:
            from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

            qcache = get_kb_turn_cache("kb_retrieval_query_cache") or {}
            qcache[cache_key] = out
            set_kb_turn_cache("kb_retrieval_query_cache", qcache)
        except ImportError:
            pass
        return out

    try:
        from services.query_understanding import _is_short_or_vague_message, _last_user_snippets

        scope = (route.get("conversation_scope") or "").strip().lower()
        intent = (route.get("intent") or "").strip().lower()
        # Never glue prior-turn topics onto greetings / locked non-KB intents —
        # that is what turns "hyee" into a return-policy or charger retrieval.
        allow_prior_expand = (
            conversation_context
            and _is_short_or_vague_message(original_msg or msg_en)
            and scope not in ("general_chitchat", "out_of_domain", "harm_sensitive")
            and intent
            not in (
                "order_history",
                "wishlist",
                "pincode_check",
                "out_of_domain",
                "deals",
                "categories",
            )
            and route.get("continue_previous_topic") is not False
        )
        if allow_prior_expand:
            prior_snippets = _last_user_snippets(conversation_context)
            prior = " — ".join(s for s in prior_snippets if s).strip()
            if prior and prior.lower() not in (merged or "").lower():
                # Prefer English gloss of prior+current for short follow-ups.
                prior_en = to_en_for_retrieval(prior, reply_lang)
                follow = english_gloss or raw
                merged = " — ".join(
                    dict.fromkeys(
                        p for p in (prior_en, follow, raw if raw.lower() != (follow or "").lower() else "") if p
                    )
                )
    except ImportError:
        pass
    out = _normalize_retrieval_query(merged or english_gloss or raw)
    try:
        from services.chat_flow_telemetry import get_kb_turn_cache, set_kb_turn_cache

        qcache = get_kb_turn_cache("kb_retrieval_query_cache") or {}
        qcache[cache_key] = out
        set_kb_turn_cache("kb_retrieval_query_cache", qcache)
    except ImportError:
        pass
    return out


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


def _expand_hit_with_neighbor_chunks(hit: dict, *, max_chars: int = 1400) -> str:
    """
    Merge same-document neighbor chunks when the top hit is a heading stub or
    mid-sentence char window (common with Qdrant char chunking).
    Uses other same-turn hits when available — no keyword maps.
    """
    if not isinstance(hit, dict):
        return ""
    base = (hit.get("chunk") or "").strip()
    if not base:
        return ""
    try:
        from services.chat_flow_telemetry import get_kb_grounding_context

        prior_hits, _ = get_kb_grounding_context()
    except ImportError:
        prior_hits = []
    if not prior_hits:
        return base

    doc_id = hit.get("doc_id")
    chunk_no = hit.get("chunk_no")
    src = str(hit.get("source") or "").strip().lower()
    neighbors: list[tuple[int, str]] = []
    for h in prior_hits or []:
        if not isinstance(h, dict):
            continue
        c = (h.get("chunk") or "").strip()
        if not c:
            continue
        same_doc = False
        if doc_id is not None and h.get("doc_id") is not None:
            same_doc = int(h.get("doc_id") or -1) == int(doc_id)
        elif src and str(h.get("source") or "").strip().lower() == src:
            same_doc = True
        if not same_doc:
            continue
        try:
            n = int(h.get("chunk_no") if h.get("chunk_no") is not None else -1)
        except (TypeError, ValueError):
            n = -1
        neighbors.append((n, c))
    if not neighbors:
        return base

    neighbors.sort(key=lambda x: x[0] if x[0] >= 0 else 10**9)
    # Prefer contiguous window around the winning chunk_no.
    if chunk_no is not None:
        try:
            center = int(chunk_no)
            neighbors = [
                (n, c)
                for n, c in neighbors
                if n < 0 or abs(n - center) <= 2
            ]
        except (TypeError, ValueError):
            pass

    parts: list[str] = []
    seen: set[str] = set()
    total = 0
    for _, c in neighbors:
        key = _chunk_dedup_key(c)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        if total + len(c) > max_chars and parts:
            break
        parts.append(c)
        total += len(c) + 2
    merged = "\n\n".join(parts).strip()
    return merged if len(merged) > len(base) else base


def format_direct_reply_from_kb_hit(
    hit: dict,
    original_msg: str,
    *,
    reply_lang: str = "",
    retrieval_query: str = "",
    fast_lane: bool = False,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """
    Answer from one retrieved KB chunk (optionally expanded with neighbors).
    Grounds via LLM when extractive is thin or customer needs localization.
    """
    if not isinstance(hit, dict):
        return ""
    chunk = (hit.get("chunk") or "").strip()
    if not chunk:
        return ""
    if _chunk_is_agent_instruction_blob(chunk):
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
    q_for_budget = (retrieval_query or original_msg or "").strip()
    budget_chars, _, _ = _infer_kb_answer_budget(q_for_budget, ai_route=ai_route)
    # Expand heading-only / mid-window char chunks using same-doc neighbors.
    if not _kb_answer_body_is_usable(chunk, min_chars=40):
        chunk = _expand_hit_with_neighbor_chunks(hit, max_chars=budget_chars + 400)
    plain_chunk = _faq_answer_text_from_chunk(re.sub(r"<br\s*/?>", "\n", chunk))
    if not _kb_answer_body_is_usable(plain_chunk, min_chars=24):
        # Retry expansion before polish stripped too much.
        chunk = _expand_hit_with_neighbor_chunks(hit, max_chars=budget_chars + 400)
        plain_chunk = _strip_leading_faq_question(
            re.sub(r"<br\s*/?>", "\n", chunk)
        )
    # Fast lane: Qdrant already ranked — skip MiniLM section/line rescoring.
    if fast_lane:
        focused = _filter_kb_blob_fast(
            plain_chunk, q_for_budget, max_chars=budget_chars, ai_route=ai_route
        )
    else:
        focused = _filter_kb_blob_for_question(
            plain_chunk,
            q_for_budget,
            max_chars=budget_chars,
            ai_route=ai_route,
        )
    excerpt = _kb_plain_excerpt(focused or plain_chunk, max_chars=budget_chars)
    if not excerpt.strip() or _chunk_is_agent_instruction_blob(excerpt):
        return ""
    # Never ship heading-only / stub extractive replies.
    if not _kb_answer_body_is_usable(excerpt):
        fallback = _kb_plain_excerpt(plain_chunk, max_chars=budget_chars)
        if _kb_answer_body_is_usable(fallback):
            excerpt = fallback
        else:
            return ""
    from services.translation_service import resolve_customer_reply_lang

    rl_resolved = resolve_customer_reply_lang(original_msg, rl)
    src_list = [src] if src and src != "general" else None
    needs_ground = (
        _kb_ai_focus_answer_enabled()
        and excerpt.strip()
        and (
            (not fast_lane)
            or _customer_needs_kb_localization(original_msg, rl_resolved)
            or (
                not _brain_english_meaning_is_narrow_fact(ai_route)
                and len(excerpt) < 120
            )
        )
    )
    if needs_ground:
        grounded = _ground_kb_excerpt_for_customer(
            original_msg,
            excerpt,
            conversation_context=conversation_context,
            reply_lang=rl_resolved,
            kb_sources=src_list,
        )
        if grounded and _kb_answer_body_is_usable(grounded, min_chars=20):
            return grounded
    body = _plain_text_to_html_body(excerpt)
    if not body or not _kb_answer_body_is_usable(body, min_chars=20):
        return ""
    body = polish_faq_reply_for_customer(body, original_msg)
    if not _kb_answer_body_is_usable(body, min_chars=20):
        return ""
    # Match customer language with bounded AI polish (customer message → LLM).
    return _finalize_kb_customer_reply(
        body, original_msg, rl_resolved, ai_route=ai_route
    )


def format_kb_no_information_reply(original_msg: str, *, reply_lang: str = "") -> str:
    """Honest fallback when KB has no matching chunk."""
    from services.translation_service import customer_reply_language, finalize_customer_reply

    rl = reply_lang or customer_reply_language(original_msg)
    en = (
        "I couldn't find this in Welfog's current knowledge base. "
        "Please contact customer support if you need more help."
    )
    return finalize_customer_reply(en, original_msg, rl, allow_llm_style_rewrite=False)


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
    ai_route: dict | None = None,
    kb_intent: bool = False,
) -> list[dict]:
    """
    Core semantic retrieval — cosine similarity on embeddings only (no keyword matching).
    Keyword fallback runs only when embeddings are disabled/unavailable.
    When KB_RETRIEVAL_BACKEND=qdrant and intent is Knowledge Base, uses Qdrant retrieval.
    """
    floor = float(min_score if min_score is not None else KB_SEMANTIC_MIN_SCORE)
    q = _normalize_retrieval_query(query)
    if not q:
        return []

    try:
        from services.knowledge_retriever import (
            is_qdrant_kb_retrieval_enabled,
            kb_retrieval_backend,
            retrieve_knowledge_hits,
            should_use_qdrant_retrieval,
        )

        if should_use_qdrant_retrieval(ai_route=ai_route, kb_intent=kb_intent):
            scoped_keys = keys
            if scoped_keys is None and customer_only:
                scoped_keys = get_customer_kb_keys()
            return retrieve_knowledge_hits(
                q,
                keys=scoped_keys,
                top_n=top_n,
                min_score=floor,
                ai_route=ai_route,
                kb_intent=kb_intent,
                log_retrieval=log_retrieval,
            )
        # Production: when Qdrant is the configured backend, never fall back to
        # building/loading MiniLM mid-request (that caused 30–90s hangs / timeouts).
        if kb_retrieval_backend() == "qdrant" and kb_intent:
            log_reasoning(
                "Qdrant KB backend configured but retrieval gate closed — "
                "return empty (skip MiniLM fallback)."
            )
            return []
        if is_qdrant_kb_retrieval_enabled() and not should_use_qdrant_retrieval(
            ai_route=ai_route, kb_intent=kb_intent
        ):
            # Qdrant enabled but non-KB probe — fall through to in-memory path.
            pass
    except ImportError:
        pass

    ensure_knowledge_cache_fresh()
    ensure_kb_vectors()

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
    Strong Admin-KB semantic match (Qdrant SoT) → lock welfog_support.

    Used to rescue Brain OOD misroutes when an active+indexed Admin
    document already answers an informational question. Never steals
    pure chitchat / affection turns into random FAQ hits.
    """
    out = dict(route or {})
    if out.get("run_catalog_search") or out.get("_product_catalog_locked"):
        return None
    scope_now = (out.get("conversation_scope") or "").strip().lower()
    if scope_now == "general_chitchat":
        return None
    # Brain English meaning already said casual talk / no ask — never FAQ-hijack.
    try:
        from services.ai_first_router import _brain_user_meaning_indicates_chitchat

        um = f"{out.get('user_meaning') or ''} {out.get('reasoning') or ''}"
        if _brain_user_meaning_indicates_chitchat(um):
            return None
    except ImportError:
        pass
    try:
        from services.ai_route_semantics import brain_route_indicates_product_catalog

        if brain_route_indicates_product_catalog(out):
            return None
    except ImportError:
        pass
    # Never steal live transactional intents.
    intent_now = (out.get("intent") or "").strip().lower()
    channel_now = (out.get("data_channel") or "").strip().lower()
    if intent_now in (
        "order",
        "order_history",
        "wishlist",
        "product",
        "pincode_check",
        "deals",
        "categories",
        "category_feed",
    ) or channel_now in ("live_api", "catalog"):
        return None
    if out.get("_pincode_delivery_locked") or out.get("_pincode_delivery_fast"):
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

    # Stricter floor when Brain said unrelated — weak FAQ hits (0.3–0.5) were
    # stealing affection / small-talk into invoice/grievance chunks.
    floor = float(min_score)
    if out.get("is_welfog_related") is False:
        floor = max(floor, 0.58)

    # Probe with a temporary KB intent so Qdrant (not OOD-blocked memory probes) runs.
    probe = dict(out)
    probe["data_channel"] = "kb"
    probe["is_welfog_related"] = True
    probe["conversation_scope"] = "welfog_support"
    probe["kb_keys"] = []  # unscoped — full Admin catalog

    q = build_kb_retrieval_query(original_msg, msg_en, "", ai_route=probe)
    best = retrieve_best_kb_chunk(
        q or combined,
        keys=get_customer_kb_keys() or None,
        ai_route=probe,
        min_score=max(0.14, floor - 0.12),
        grounded=True,
    )
    score = float((best or {}).get("score") or 0)
    if not best or score < floor:
        return None

    src = str(best.get("source") or "").strip()
    try:
        from services.knowledge_keys import canonical_knowledge_key

        src = canonical_knowledge_key(src) or src
    except ImportError:
        src = (src or "").lower()

    keys: list[str] = []
    if src:
        keys = [src]
    # Soft-merge any prior Brain hints after the semantic winner.
    for k in resolve_brain_kb_keys(out, original_msg, msg_en) or []:
        if k and k not in keys:
            keys.append(k)
    keys = keys[:4]

    out["intent"] = "general"
    out["data_channel"] = "kb"
    out["conversation_scope"] = "welfog_support"
    out["is_welfog_related"] = True
    out["kb_keys"] = keys
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out["scope_reply"] = ""
    out["needs_order_id"] = False
    out.pop("category_only_browse", None)
    out.pop("category_browse", None)
    out.pop("_product_catalog_locked", None)
    out["_turn_promotions_done"] = True
    out["_semantic_kb_rescue"] = True
    try:
        from services.chat_flow_telemetry import lock_authoritative_kb_route_from_retrieval

        lock_authoritative_kb_route_from_retrieval(
            chunks=1,
            top_score=score,
            ai_route=out,
        )
    except ImportError:
        pass
    log_reasoning(
        f"Admin KB semantic rescue — source={src or '-'} score={score:.2f} "
        f"(Brain OOD/misroute overridden by indexed knowledge)."
    )
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
    Strongest semantic match across ALL customer Admin docs.
    preferred_keys / Brain hints are soft only — higher raw score always wins.
    """
    floor = float(min_score if min_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    filter_q = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    combined = filter_q or f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return None

    route = dict(ai_route or {})
    valid = [
        k
        for k in (preferred_keys or [])
        if k in kb_chunks_by_key and k not in INTERNAL_KB_KEYS
    ][:4]
    if valid and not route.get("kb_keys"):
        route["kb_keys"] = valid

    # One catalog-wide search — never a predefined classic-file loop.
    if brain_locked_kb and valid:
        search_keys = valid
    else:
        search_keys = get_customer_kb_keys() or None

    hit = retrieve_best_kb_chunk(
        combined,
        keys=search_keys,
        ai_route=route,
        min_score=max(0.12, floor - 0.12),
    )
    if hit and float(hit.get("score") or 0) >= floor:
        return hit
    return None


def retrieve_best_kb_chunk(
    query: str,
    *,
    keys: list[str] | None = None,
    ai_route: dict | None = None,
    min_score: float | None = None,
    conflict_check: bool = True,
    grounded: bool = True,
) -> dict | None:
    """
    Semantic search + rerank.
    grounded=True (default): when Qdrant KB backend is on, use Qdrant only for answer chunks.
    grounded=False: MiniLM memory path allowed (routing / intent assists).
    """
    floor = float(min_score if min_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    search_floor = max(0.12, floor - 0.1)
    kb_intent = False
    if grounded:
        try:
            from services.knowledge_retriever import is_qdrant_kb_retrieval_enabled

            kb_intent = bool(is_qdrant_kb_retrieval_enabled())
        except ImportError:
            kb_intent = False
    hits = semantic_kb_search(
        query,
        keys=keys,
        top_n=8,
        min_score=search_floor,
        log_retrieval=False,
        ai_route=ai_route,
        kb_intent=kb_intent,
    )
    return _pick_best_kb_hit_for_query(
        query,
        hits,
        ai_route=ai_route,
        min_rerank_score=floor,
        conflict_check=conflict_check,
        light_rerank=bool(kb_intent),
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
    Soft file hints via semantic ranking across ALL active Admin customer docs.

    No predefined topic→file map, no contact keyword force, no classic-name boost.
    Empty return → callers run unscoped Qdrant retrieval.
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
        from services.knowledge_keys import canonical_knowledge_key
    except ImportError:
        canonical_knowledge_key = lambda t: (t or "").strip().lower()  # noqa: E731

    valid_suggested = []
    for k in suggested_keys or []:
        ck = canonical_knowledge_key(str(k or ""))
        if ck and ck in kb_chunks_by_key and ck not in INTERNAL_KB_KEYS and ck in customer_keys:
            valid_suggested.append(ck)
    valid_suggested = list(dict.fromkeys(valid_suggested))

    file_rank_floor = min(0.12, KB_CONTEXT_MIN_SCORE)
    best_per_key: dict[str, float] = {}
    hits = semantic_kb_search(
        query,
        keys=customer_keys,
        top_n=min(32, max(len(customer_keys) * 4, max_files * 6)),
        min_score=file_rank_floor,
        customer_only=False,
        log_retrieval=False,
        ai_route=ai_route,
        kb_intent=True,
    )
    for h in hits:
        src = str(h.get("source") or "")
        sc = float(h.get("score") or 0)
        if src in customer_keys and sc >= file_rank_floor:
            best_per_key[src] = max(best_per_key.get(src, 0.0), sc)

    ranked = sorted(best_per_key.items(), key=lambda x: x[1], reverse=True)
    keys_from_embed = [k for k, _ in ranked[:max_files]]

    if keys_from_embed:
        try:
            hits_preview = top_kb_hits(
                query, keys=keys_from_embed, min_score=0.14, top_n=3, log_retrieval=False
            )
            log_kb_retrieval(
                query_intent="semantic",
                query_meaning=((ai_route or {}).get("user_meaning") or "").strip(),
                retrieval_query=query,
                matched_file=keys_from_embed[0],
                selected_category="semantic",
                similarity_score=float(hits_preview[0].get("score") or 0) if hits_preview else 0.0,
                selected_chunks=hits_preview,
            )
        except Exception:
            pass
        if valid_suggested:
            # Soft merge: semantic winners first, then Brain hints.
            merged = list(dict.fromkeys(keys_from_embed + valid_suggested))
            return merged[:max_files]
        return keys_from_embed

    if valid_suggested:
        return valid_suggested[:max_files]

    best = retrieve_best_kb_chunk(query, keys=customer_keys, ai_route=ai_route, min_score=0.2)
    if best and best.get("source") in customer_keys:
        return [best["source"]]

    return []


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
                "RULES: Answer ONLY from this text. Match the customer's language/script "
                "(Hinglish = Roman Hindi+English; also Hindi/Tamil/Punjabi/Kannada/Marathi/etc.). "
                "Do NOT repeat or restate the user's question — start directly with the answer. "
                "Use ONLY the part of the knowledge that answers THIS specific question — never paste "
                "unrelated sections (cancel ≠ return; refund timeline ≠ return steps; shipping SLA ≠ "
                "pincode instructions; owner ≠ office address). "
                "Be COMPLETE for what was asked: if they ask a policy, give the full relevant policy "
                "facts; if they ask how-to, give the full numbered steps from the text — never return "
                "only a heading like 'Refund Policy:'. "
                "Do not invent facts. Seller account steps must NOT be confused with buyer checkout. "
                "Copy phone/email/addresses exactly.\n\n"
                f"{excerpt}"
            ),
            conversation_context=conversation_context,
            reply_lang=rl,
        )
        resp = (ai.get("response") or "").strip() if ai else ""
        if resp and not _ai_response_looks_like_placeholder(resp):
            try:
                from services.knowledge_grounding_validator import ground_kb_llm_response

                resp = ground_kb_llm_response(
                    resp,
                    kb_context=excerpt,
                    original_msg=original_msg,
                    msg_en="",
                )
            except ImportError:
                pass
            if not resp:
                resp = (ai.get("response") or "").strip() if ai else ""
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


def _kb_chunk_relevance_adjustment(
    question: str, chunk: str, *, ai_route: dict | None = None
) -> float:
    """
    Soft semantic nudge: Brain meaning vs chunk embedding only.
    No topic keyword lists — new admin FAQs need no code changes.
    """
    focus = _brain_focus_text(question, ai_route)
    if not focus or not (chunk or "").strip():
        return 0.0
    plain = re.sub(r"<br\s*/?>", " ", chunk or "")
    plain = re.sub(r"<[^>]+>", " ", plain)
    sim = _embedding_similarity(focus[:400], plain[:900])
    return max(-0.12, min(0.12, (sim - 0.35) * 0.35))


def _chunk_looks_like_contact_card(chunk: str) -> bool:
    """True when chunk is mostly phone/email contact lines (not a policy narrative)."""
    plain = re.sub(r"<br\s*/?>", "\n", chunk or "", flags=re.I)
    plain = re.sub(r"<[^>]+>", " ", plain)
    phones = re.findall(r"\b\d{10}\b", plain)
    emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", plain, flags=re.I)
    if not phones and not emails:
        return False
    words = re.findall(r"[A-Za-z]{3,}", plain)
    # Short contact directory vs descriptive About/policy prose.
    return len(words) <= 48 or (len(phones) + len(emails) >= 2 and len(words) <= 80)


def _chunk_has_identity_fact(chunk: str) -> bool:
    """True when chunk mentions owner/founder/about — keep even if address co-occurs."""
    low = re.sub(r"<br\s*/?>", " ", (chunk or "").lower())
    low = re.sub(r"<[^>]+>", " ", low)
    return bool(
        re.search(
            r"\b(owner|founder|co-?founder|ceo|about welfog|our story)\b",
            low,
        )
    )


def _chunk_looks_like_address_block(chunk: str) -> bool:
    """True when chunk is mostly a postal/office address listing."""
    plain = re.sub(r"<br\s*/?>", "\n", chunk or "", flags=re.I)
    plain = re.sub(r"<[^>]+>", " ", plain)
    low = plain.lower()
    if not any(x in low for x in ("address", "plot", "pin code", "pincode", "street", "colony")):
        return False
    # Owner/about facts glued next to an address are NOT pure address blocks.
    if _chunk_has_identity_fact(plain):
        return False
    words = re.findall(r"[A-Za-z]{3,}", plain)
    # Short address block — not a long "About Welfog" narrative.
    return len(words) <= 60 and "integrated digital platform" not in low and "ecosystem" not in low


def _brain_meaning_asks_contact(ai_route: dict | None) -> bool:
    """Use Brain English meaning — not customer keyword maps — to detect contact asks."""
    r = ai_route or {}
    blob = f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".strip()
    if not blob:
        return False
    try:
        from utils.helpers import _text_asks_customer_care_contact

        return bool(_text_asks_customer_care_contact(blob))
    except ImportError:
        low = blob.lower()
        return any(
            x in low
            for x in (
                "customer care",
                "support number",
                "phone number",
                "contact number",
                "helpline",
                "email",
            )
        )


def _brain_meaning_asks_company_identity(ai_route: dict | None) -> bool:
    """Brain meaning / intent: what is / about / owner — not address or phone."""
    r = ai_route or {}
    intent = (r.get("intent") or "").strip().lower()
    if intent in ("owner", "about"):
        return True
    um = f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".strip().lower()
    if not um:
        return False
    if any(x in um for x in ("address", "phone", "helpline", "customer care", "contact number")):
        # Still allow owner asks that mention contact accidentally.
        if not re.search(r"\b(owner|founder|who owns)\b", um):
            return False
    return bool(
        re.search(
            r"\b("
            r"what is welfog|what welfog is|about welfog|who owns|owner of|owner name|"
            r"founder|asking what welfog|user is asking what welfog|"
            r"who is the owner|welfog'?s? owner|owner of welfog|name of (?:the )?owner|"
            r"who (?:is|owns) (?:the )?owner"
            r")\b",
            um,
        )
        or re.search(r"\b(owner|founder)\b", um)
    )


def _brain_meaning_asks_owner_name(ai_route: dict | None) -> bool:
    """Brain English meaning: who owns / owner name (narrower than general about)."""
    r = ai_route or {}
    intent = (r.get("intent") or "").strip().lower()
    if intent == "owner":
        return True
    um = f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".strip().lower()
    if not um:
        return False
    return bool(
        re.search(
            r"\b(who owns|owner of|owner name|founder|who is the owner|"
            r"welfog'?s? owner|name of (?:the )?owner)\b",
            um,
        )
        or (re.search(r"\bowner\b", um) and not re.search(r"\b(account owner|order owner)\b", um))
    )


def _brain_meaning_asks_social(ai_route: dict | None, combined: str = "") -> bool:
    """Brain / text: official social URL ask — not Reels/content policy."""
    r = ai_route or {}
    rh = (r.get("route_handler") or "").strip().lower()
    um = f"{r.get('user_meaning') or ''} {r.get('reasoning') or ''}".strip().lower()
    # Content / reels *system* is KB policy, never a social-URL answer.
    if re.search(
        r"\b(reel|reels|shorts|content (?:rules?|polic)|allowed content|"
        r"prohibited content|what content)\b",
        um,
    ) and not re.search(r"\b(instagram|facebook|youtube|linkedin|twitter)\b.*\b(link|url|handle)\b", um):
        return False
    if "social" in rh and re.search(r"\b(link|url|handle|page|profile|follow)\b", um or combined.lower()):
        return True
    intent = (r.get("intent") or "").strip().lower()
    if intent in ("social", "social_media") and not re.search(r"\b(reel|reels|shorts)\b", um):
        return True
    blob = f"{um} {combined or ''}".strip().lower()
    if not blob:
        return False
    if re.search(r"\b(reel|reels|shorts)\b", blob) and not re.search(
        r"\b(instagram|facebook|youtube|linkedin|twitter)\b", blob
    ):
        return False
    has_platform = bool(
        re.search(
            r"\b(?:instagram|insta|facebook|fb|youtube|linkedin|twitter|social media|"
            r"social link|social account)\b",
            blob,
        )
    )
    if not has_platform:
        return False
    return bool(
        re.search(
            r"\b(?:link|url|handle|page|follow|id|account|profile|de\s*do|dedo|"
            r"chahiye|want|need|give|share|send)\b",
            blob,
        )
    )


def _rewrite_preserves_kb_facts(source: str, rewritten: str) -> bool:
    """Reject style rewrite that invents phone/email/URL not present in the KB excerpt."""
    src = re.sub(r"<[^>]+>", " ", source or "")
    dst = re.sub(r"<[^>]+>", " ", rewritten or "")
    if not dst.strip():
        return False
    src_phones = set(re.findall(r"\b\d{10}\b", src))
    dst_phones = set(re.findall(r"\b\d{10}\b", dst))
    if dst_phones - src_phones:
        return False
    src_mails = {m.lower() for m in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", src, flags=re.I)}
    dst_mails = {m.lower() for m in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", dst, flags=re.I)}
    if dst_mails - src_mails:
        return False
    return True


def _pick_best_kb_hit_for_query(
    question: str,
    hits: list[dict],
    *,
    ai_route: dict | None = None,
    conflict_check: bool = True,
    min_rerank_score: float | None = None,
    light_rerank: bool = False,
) -> dict | None:
    """
    Rerank embedding hits; skip conflicting chunks.

    Raw vector similarity is primary. FAQ-question / relevance nudges are
    tie-breakers only — they must never let a weaker chunk leapfrog a stronger
    semantic hit (e.g. seller howto FAQ stealing a company-owner fact).
    Brain kb_keys act as a soft preference within a small raw-score margin.

    light_rerank=True (Qdrant answer path): skip MiniLM FAQ re-embeds — Qdrant
    + RRF already ranked; secondary nudges stay token/structure cheap.
    """
    hint_keys: set[str] = set()
    try:
        for k in (ai_route or {}).get("kb_keys") or []:
            ck = str(k or "").strip().lower()
            if ck:
                hint_keys.add(ck)
    except Exception:
        hint_keys = set()

    ranked: list[tuple[float, float, float, int, dict]] = []
    meaning_wants_contact = _brain_meaning_asks_contact(ai_route)
    meaning_identity = _brain_meaning_asks_company_identity(ai_route)
    meaning_owner = _brain_meaning_asks_owner_name(ai_route)
    meaning_blob = ""
    if isinstance(ai_route, dict):
        meaning_blob = (
            f"{ai_route.get('user_meaning') or ''} {ai_route.get('reasoning') or ''}"
        ).strip().lower()
    focus_q = f" {meaning_blob} {(question or '').lower()} "
    for hit in hits:
        chunk = hit.get("chunk") or ""
        if _chunk_is_agent_instruction_blob(chunk):
            continue
        if conflict_check and _faq_chunk_conflicts_with_query(
            question, chunk, ai_route=ai_route
        ):
            continue
        # Contact / address cards must not answer "what is Welfog / who owns …".
        if not meaning_wants_contact and _chunk_looks_like_contact_card(chunk):
            if not (meaning_owner and _chunk_has_identity_fact(chunk)):
                continue
        if meaning_identity and _chunk_looks_like_address_block(chunk):
            continue
        raw = float(hit.get("score") or 0)
        adj = float(_kb_chunk_relevance_adjustment(question, chunk, ai_route=ai_route) or 0)
        # Strong facet conflicts — allow larger secondary so cancel≠return etc.
        if abs(adj) >= 0.18:
            secondary_adj = max(-0.22, min(0.22, adj))
        else:
            secondary_adj = max(-0.08, min(0.10, adj))
        faq_b = (
            0.0
            if light_rerank
            else float(_faq_question_similarity_boost(question, chunk) or 0)
        )
        secondary = secondary_adj + max(0.0, min(0.06, faq_b))
        src = str(hit.get("source") or "").strip().lower()
        hint_match = 1 if (hint_keys and src in hint_keys) else 0
        # Soft-penalize support/contact keys when Brain did not ask for contact.
        if (
            not meaning_wants_contact
            and src in ("support", "customer_support", "customer_care", "contacts")
        ):
            raw = max(0.0, raw - 0.08)
        if meaning_owner and _chunk_has_identity_fact(chunk):
            secondary += 0.06
            if src in ("company", "about", "faqs"):
                hint_match = max(hint_match, 1)
        # Display score keeps a light secondary signal for logging/thresholds.
        display = raw + secondary + (0.02 if hint_match else 0.0)
        ranked.append((raw, secondary, float(hint_match), display, {**hit, "score": display}))

    if not ranked:
        return None

    # Primary: raw semantic. Secondary: brain key match, then FAQ/relevance nudge.
    ranked.sort(key=lambda x: (x[0], x[2], x[1]), reverse=True)
    best_raw, _, _, best_display, best = ranked[0]

    # Within a tight raw margin, allow FAQ-line or key-hint preference to break ties.
    if len(ranked) >= 2:
        runner_raw, runner_sec, runner_hint, runner_display, runner = ranked[1]
        if runner_raw >= best_raw - 0.06:
            if light_rerank:
                top_faq = run_faq = 0.0
            else:
                top_faq = _faq_question_similarity_boost(question, best.get("chunk") or "")
                run_faq = _faq_question_similarity_boost(question, runner.get("chunk") or "")
            if runner_hint > (1 if str(best.get("source") or "").strip().lower() in hint_keys else 0):
                best = runner
                best_display = runner_display
            elif run_faq > top_faq + 0.04 and runner_raw + 0.01 >= best_raw:
                best = runner
                best_display = runner_display

    # If a brain-hinted source is close to the raw leader, prefer it (soft lock).
    if hint_keys and ranked:
        hinted = [r for r in ranked if str(r[4].get("source") or "").strip().lower() in hint_keys]
        if hinted:
            h_raw, _, _, h_display, h_hit = max(hinted, key=lambda x: (x[0], x[1]))
            if h_raw >= best_raw - 0.12 and str(best.get("source") or "").strip().lower() not in hint_keys:
                best = h_hit
                best_display = h_display

    floor = float(min_rerank_score if min_rerank_score is not None else KB_ANSWER_MIN_CONFIDENCE)
    if best_display < floor and best_raw < floor:
        return None
    return best


def _faq_chunk_conflicts_with_query(
    question: str, chunk: str, ai_route: dict | None = None
) -> bool:
    """
    Drop clearly wrong channel matches using Brain meaning + embeddings.

    No admin-topic keyword include/exclude maps — new KB docs need no code edits.
    Structure/channel helpers (order-history live list vs how-to) stay AI-route based.
    """
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
            # How-to ask vs status/track narrative — meaning gate, not keyword lists.
            if _chunk_is_order_history_howto_faq(chunk) is False and (
                "order history" in c or "my orders" in c
            ):
                # Keep; conflict only when clearly not a how-to blob for a how-to ask.
                pass
    except ImportError:
        pass

    focus = _brain_focus_text(question, ai_route)
    if not focus or not (chunk or "").strip():
        return False
    try:
        from utils.helpers import message_is_seller_on_welfog_request

        if message_is_seller_on_welfog_request(question):
            # Seller meaning vs buyer-checkout FAQ — embedding distance only.
            buyer_probe = (
                "create an account on Welfog to place an order and track purchases checkout"
            )
            seller_probe = (
                "become a seller supplier registration seller portal vendor account"
            )
            plain = re.sub(r"<[^>]+>", " ", chunk or "")
            sim_buyer = _embedding_similarity(buyer_probe, plain[:700])
            sim_seller = _embedding_similarity(seller_probe, plain[:700])
            sim_focus = _embedding_similarity(focus[:400], plain[:700])
            if sim_buyer >= sim_seller + 0.08 and sim_focus < sim_buyer:
                return True
    except ImportError:
        pass

    # Soft meaning vs chunk: reject chunks far below relevance to Brain focus
    # (handles cancel↔return / owner↔address without topic keyword lists).
    plain = re.sub(r"<br\s*/?>", " ", chunk or "")
    plain = re.sub(r"<[^>]+>", " ", plain)
    sim = _embedding_similarity(focus[:400], plain[:900])
    if sim < 0.12:
        return True
    if _brain_meaning_asks_owner_name(ai_route) and not _chunk_has_identity_fact(chunk):
        if _chunk_looks_like_address_block(chunk) or _chunk_looks_like_contact_card(chunk):
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
    return finalize_customer_reply(
        body, original_msg or msg_en, rl, allow_llm_style_rewrite=False
    ) or ""


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
    Admin KB answer after brain locks kb_keys: vector RAG across all customer
    docs (keys = soft hints) → grounded answer in the customer's language.
    """
    try:
        from services.chat_resilience import chat_turn_abandoned

        if chat_turn_abandoned():
            return ""
    except ImportError:
        pass
    from services.translation_service import (
        _normalize_language,
        customer_reply_language,
        is_hinglish_message,
    )
    from utils.helpers import _text_asks_customer_care_contact

    # Qdrant answers use chunk payload text — do NOT call ensure_knowledge_cache_fresh()
    # here (MySQL can sit on MYSQL_READ_TIMEOUT and blank the UI until AbortError).
    try:
        from services.knowledge_retriever import kb_retrieval_backend

        if kb_retrieval_backend() != "qdrant":
            ensure_knowledge_cache_fresh()
            ensure_kb_vectors()
    except ImportError:
        ensure_knowledge_cache_fresh()

    # Direct call — never nest ThreadPoolExecutor under chat-deadline worker
    # (that deadlocked until CHAT_MAX and clients saw AbortError / busy).
    combined = f"{original_msg or ''} {msg_en or ''}".strip()
    if not combined:
        return ""

    lang = _normalize_language(reply_lang or customer_reply_language(original_msg))
    if lang == "hinglish" or is_hinglish_message(original_msg):
        lang = "hinglish"

    route = dict(ai_route or {})
    local_keys = list(keys or [])
    if local_keys and not route.get("kb_keys"):
        route["kb_keys"] = list(local_keys)
    if user_meaning_en and not (route.get("user_meaning") or "").strip():
        route["user_meaning"] = user_meaning_en.strip()
    meaning_for_route = (
        (user_meaning_en or "").strip()
        or (route.get("user_meaning") or "").strip()
    )
    if meaning_for_route:
        route["user_meaning"] = meaning_for_route

    if local_keys and not _brain_meaning_asks_contact(route):
        contactish = {
            "support",
            "customer_support",
            "customer_care",
            "contacts",
            "contact",
        }
        normed = [
            str(k or "").strip().lower() for k in local_keys if str(k or "").strip()
        ]
        if normed and all(k in contactish for k in normed):
            log_reasoning(
                "KB brain-keys — drop support-only hints; meaning is not a contact ask."
            )
            local_keys = []
            route["kb_keys"] = []

    # Owner/founder facts often live in a dedicated admin doc (key=owner), not
    # company About — soft-expand keys from Brain meaning (no customer keywords).
    if _brain_meaning_asks_owner_name(route):
        customer = list(get_customer_kb_keys() or [])
        expanded = list(local_keys or [])
        for k in customer:
            kl = str(k or "").strip().lower()
            if kl in ("owner", "about", "company", "faqs") or "owner" in kl:
                if k not in expanded:
                    expanded.append(k)
        if "owner" in customer:
            expanded = ["owner"] + [k for k in expanded if k != "owner"]
        if expanded != local_keys:
            log_reasoning(
                f"KB brain-keys — owner meaning; soft keys={','.join(expanded[:5])}"
            )
        local_keys = expanded
        route["kb_keys"] = list(local_keys)

    if not _user_requests_grievance_channel(combined):
        msg_wants_contact = _text_asks_customer_care_contact(combined)
        meaning_wants_contact = _brain_meaning_asks_contact(route)
        if msg_wants_contact and (meaning_wants_contact or not meaning_for_route):
            cc = format_customer_care_reply_from_kb(original_msg, msg_en)
            if cc:
                return cc

    # Social URLs before vector retrieve — never dump Support Rules / agent meta.
    try_social = _brain_meaning_asks_social(route, combined)
    if not try_social:
        try:
            from utils.helpers import (
                message_asks_welfog_social_media,
                _is_welfog_social_followup,
            )

            try_social = bool(
                message_asks_welfog_social_media(
                    combined, conversation_context=conversation_context
                )
                or _is_welfog_social_followup(combined, conversation_context)
            )
        except ImportError:
            try_social = False
    if try_social:
        try:
            social = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
                user_meaning_en=meaning_for_route,
                ai_confirmed=True,
            )
            if social:
                log_reasoning("KB brain-keys — official social URLs from admin KB.")
                return social
        except Exception as exc:
            log_reasoning(f"KB social formatter skipped: {exc}")

    filter_q = build_kb_retrieval_query(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=route,
    ) or (
        meaning_for_route
        or (msg_en or "").strip()
        or (original_msg or "").strip()
    )

    # Soft keys only — never hard-lock to one/two files. Search full customer corpus
    # so faqs / shipping / owner / seller docs can all match by vectors.
    search_keys = local_keys or None
    try:
        from services.knowledge_answer_service import (
            generate_grounded_kb_answer_from_qdrant,
            should_generate_qdrant_kb_answer,
        )

        if (route.get("data_channel") or "").strip().lower() != "kb":
            route["data_channel"] = "kb"
        if should_generate_qdrant_kb_answer(route):
            grounded = generate_grounded_kb_answer_from_qdrant(
                original_msg,
                msg_en=msg_en,
                keys=search_keys,
                reply_lang=lang,
                conversation_context=conversation_context,
                user_meaning_en=meaning_for_route,
                ai_route=route,
            )
            if grounded and _kb_answer_body_is_usable(grounded, min_chars=20):
                log_reasoning(
                    "KB brain-keys — Qdrant grounded multi-chunk answer (all matching docs)."
                )
                return grounded
    except Exception as exc:
        log_reasoning(f"KB brain-keys grounded path skipped: {exc}")

    gray_floor = max(float(KB_DIRECT_MIN_SCORE), 0.16)
    try:
        hit = retrieve_best_kb_chunk(
            filter_q,
            keys=search_keys,
            min_score=gray_floor,
            ai_route=route,
        )
        # Miss under soft keys → retry full customer corpus (admin may have added
        # the answer under any title).
        if (not hit) or float((hit or {}).get("score") or 0) < gray_floor:
            hit_all = retrieve_best_kb_chunk(
                filter_q,
                keys=None,
                min_score=gray_floor,
                ai_route=route,
            )
            if hit_all and float(hit_all.get("score") or 0) >= gray_floor:
                hit = hit_all
                log_reasoning(
                    "KB brain-keys — soft keys missed; full-corpus vector hit used."
                )
        # Owner asks: About narrative without owner fact is a miss — retry owner/unscoped.
        if (
            hit
            and _brain_meaning_asks_owner_name(route)
            and not _chunk_has_identity_fact(hit.get("chunk") or "")
        ):
            log_reasoning(
                "KB brain-keys — About hit lacks owner fact; retry owner scope."
            )
            owner_keys = [
                k
                for k in (get_customer_kb_keys() or [])
                if str(k).lower() == "owner" or "owner" in str(k).lower()
            ]
            hit2 = retrieve_best_kb_chunk(
                filter_q,
                keys=owner_keys or None,
                min_score=gray_floor,
                ai_route=route,
            )
            if hit2 and _chunk_has_identity_fact(hit2.get("chunk") or ""):
                hit = hit2
            else:
                hit3 = retrieve_best_kb_chunk(
                    filter_q,
                    keys=None,
                    min_score=gray_floor,
                    ai_route=route,
                )
                if hit3 and _chunk_has_identity_fact(hit3.get("chunk") or ""):
                    hit = hit3
        score = float((hit or {}).get("score") or 0)
        if hit and score >= gray_floor:
            body = format_direct_reply_from_kb_hit(
                hit,
                original_msg,
                reply_lang=lang,
                retrieval_query=filter_q,
                fast_lane=False,
                conversation_context=conversation_context,
                ai_route=route,
            )
            if body and _kb_answer_body_is_usable(body, min_chars=20):
                log_reasoning(
                    f"KB brain-keys — vector hit score={score:.2f} "
                    f"src={(hit.get('source') or '?')}"
                )
                return body
    except Exception as exc:
        log_reasoning(f"KB brain-keys extractive path skipped: {exc}")

    log_reasoning("KB brain-keys — no usable vector answer; honest gap.")
    return _format_kb_brain_gap_reply(original_msg, msg_en, reply_lang=lang) or (
        format_kb_no_information_reply(original_msg, reply_lang=lang) or ""
    )

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

    try:
        from utils.helpers import (
            message_asks_welfog_social_media,
            _is_welfog_social_followup,
        )

        if (
            _brain_meaning_asks_social(ai_route, combined)
            or message_asks_welfog_social_media(
                combined, conversation_context=conversation_context
            )
            or _is_welfog_social_followup(combined, conversation_context)
        ):
            social = format_welfog_social_media_reply_from_kb(
                original_msg,
                msg_en,
                reply_lang=lang,
                conversation_context=conversation_context,
                user_meaning_en=(
                    ((ai_route or {}).get("user_meaning") or "").strip()
                    if isinstance(ai_route, dict)
                    else ""
                ),
                ai_confirmed=True,
            )
            if social:
                return social
    except Exception:
        pass

    retrieval_q = build_kb_retrieval_query(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    filter_q = retrieval_q or combined
    if lang == "hinglish" or is_hinglish_message(original_msg):
        lang = "hinglish"

    # Prefer full-corpus retrieval first. faqs.txt is not a privileged silo —
    # any admin doc (faqs/shipping/owner/…) may hold the answer.
    route_for_ans = dict(ai_route or {})
    if (route_for_ans.get("data_channel") or "").strip().lower() != "kb":
        route_for_ans["data_channel"] = "kb"
    try:
        from services.knowledge_answer_service import (
            generate_grounded_kb_answer_from_qdrant,
            should_generate_qdrant_kb_answer,
        )

        if should_generate_qdrant_kb_answer(route_for_ans):
            grounded_all = generate_grounded_kb_answer_from_qdrant(
                original_msg,
                msg_en=msg_en,
                keys=None if not seller_only else ["seller"],
                reply_lang=lang,
                conversation_context=conversation_context,
                ai_route=route_for_ans,
            )
            if grounded_all and _kb_answer_body_is_usable(grounded_all, min_chars=20):
                log_reasoning("Dynamic KB — Qdrant grounded answer across all matching docs.")
                return grounded_all
    except Exception as exc:
        log_reasoning(f"Dynamic KB grounded path skipped: {exc}")

    # Memory / MiniLM backend: multi-file vector retrieve → LLM ground (same idea as Qdrant).
    try:
        corpus_keys = ["seller"] if seller_only else None
        multi_hits = top_kb_hits(
            filter_q,
            keys=corpus_keys,
            min_score=max(0.14, float(KB_ANSWER_MIN_CONFIDENCE) - 0.08),
            top_n=8,
            ai_route=route_for_ans,
            kb_intent=True,
        )
        if multi_hits:
            top_sc = float(multi_hits[0].get("score") or 0)
            band = [
                h
                for h in multi_hits
                if float(h.get("score") or 0) >= max(0.14, top_sc * 0.72)
            ][:5]
            best = _pick_best_kb_hit_for_query(
                filter_q, band, ai_route=route_for_ans
            ) or band[0]
            # Keep only chunks near the best meaning match — avoid FAQ dumps.
            best_sc = float(best.get("score") or 0)
            focused_hits = [
                h
                for h in band
                if float(h.get("score") or 0) >= max(0.14, best_sc * 0.85)
            ][:3]
            ctx_parts: list[str] = []
            srcs: list[str] = []
            for h in focused_hits:
                c = _filter_kb_blob_fast(
                    re.sub(r"<br\s*/?>", "\n", h.get("chunk") or ""),
                    filter_q,
                    max_chars=700,
                    ai_route=route_for_ans,
                ) or (h.get("chunk") or "")
                if _kb_answer_body_is_usable(c, min_chars=20):
                    ctx_parts.append(c.strip())
                    src = str(h.get("source") or "").strip()
                    if src and src not in srcs:
                        srcs.append(src)
            if ctx_parts:
                grounded_mem = _ground_kb_excerpt_for_customer(
                    original_msg,
                    "\n\n---\n\n".join(ctx_parts),
                    conversation_context=conversation_context,
                    reply_lang=reply_lang or lang,
                    kb_sources=srcs or None,
                )
                if grounded_mem and _kb_answer_body_is_usable(grounded_mem, min_chars=20):
                    log_reasoning(
                        f"Dynamic KB — multi-file grounded "
                        f"sources={','.join(srcs[:4]) or 'corpus'} "
                        f"score={best_sc:.2f}."
                    )
                    return grounded_mem
                # Extractive from best usable chunk if grounding empty.
                direct = format_direct_reply_from_kb_hit(
                    best,
                    original_msg,
                    reply_lang=lang,
                    retrieval_query=filter_q,
                    fast_lane=False,
                    conversation_context=conversation_context,
                    ai_route=route_for_ans,
                )
                if direct and _kb_answer_body_is_usable(direct, min_chars=20):
                    log_reasoning(
                        f"Dynamic KB — best corpus chunk "
                        f"src={best.get('source')} score={best_sc:.2f}."
                    )
                    return direct
    except Exception as exc:
        log_reasoning(f"Dynamic KB multi-file path skipped: {exc}")

    faq_hit = None
    if not seller_only:
        faq_hit = resolve_best_faq_chunk_for_question(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        )
    if faq_hit and float(faq_hit.get("score") or 0) >= KB_ANSWER_MIN_CONFIDENCE:
        # Only short-circuit on faqs when the excerpt itself is a usable answer
        # (not a heading stub / half line).
        excerpt = _faq_answer_text_from_chunk(
            re.sub(r"<br\s*/?>", "\n", faq_hit.get("chunk") or "")
        )
        excerpt = _kb_plain_excerpt(excerpt, max_chars=_infer_kb_answer_budget(filter_q, ai_route=ai_route)[0])
        if _kb_answer_body_is_usable(excerpt):
            grounded = _ground_kb_excerpt_for_customer(
                original_msg,
                excerpt,
                conversation_context=conversation_context,
                reply_lang=reply_lang or lang,
                kb_sources=["faqs"],
            )
            if grounded and _kb_answer_body_is_usable(grounded, min_chars=20):
                log_reasoning(
                    f"FAQ semantic answer (faqs score={float(faq_hit.get('score') or 0):.2f})."
                )
                return grounded
            if not _customer_needs_kb_localization(original_msg, lang):
                excerpt = polish_faq_reply_for_customer(excerpt, original_msg)
                if _kb_answer_body_is_usable(excerpt):
                    body = _plain_text_to_html_body(excerpt) or ""
                    body = polish_faq_reply_for_customer(body, original_msg)
                    if _kb_answer_body_is_usable(body, min_chars=20):
                        return _finalize_kb_customer_reply(body, original_msg, lang)

    keys = resolve_kb_keys_for_question(
        original_msg,
        msg_en,
        suggested_keys=suggested_keys,
        max_files=8,
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
    budget = _infer_kb_answer_budget(filter_q, ai_route=ai_route)[0]
    seller_keys = [k for k in keys if k == "seller"] if seller_only else keys
    # Soft file hints — fall back to full customer corpus when scoped miss.
    search_keys = seller_keys or keys or None

    best_chunk = retrieve_best_kb_chunk(
        filter_q,
        keys=search_keys,
        ai_route=ai_route,
        min_score=KB_ANSWER_MIN_CONFIDENCE,
    )
    if (not best_chunk or not best_chunk.get("chunk")) and not seller_only:
        best_chunk = retrieve_best_kb_chunk(
            filter_q,
            keys=None,
            ai_route=ai_route,
            min_score=KB_ANSWER_MIN_CONFIDENCE,
        )
    if best_chunk and best_chunk.get("chunk"):
        raw_hit = re.sub(r"<br\s*/?>", "\n", best_chunk.get("chunk") or "")
        # Prefer focused section filter over polish-first (polish can leave headings).
        cleaned_hit = _filter_kb_blob_fast(
            raw_hit, filter_q, max_chars=budget, ai_route=ai_route
        ) or _filter_kb_blob_for_question(
            raw_hit, filter_q, max_chars=budget, ai_route=ai_route
        )
        if not _kb_answer_body_is_usable(cleaned_hit or ""):
            cleaned_hit = polish_faq_reply_for_customer(
                _strip_leading_faq_question(raw_hit),
                original_msg,
            )
        filtered = _kb_plain_excerpt(cleaned_hit or raw_hit, max_chars=budget)
        if not _kb_answer_body_is_usable(filtered):
            filtered = ""
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
                try:
                    from services.knowledge_grounding_validator import ground_kb_llm_response

                    kb_ctx = (
                        f"AUTHORITATIVE KNOWLEDGE (live admin files: {', '.join(keys)}):\n{filtered}"
                    )
                    resp = ground_kb_llm_response(
                        resp,
                        kb_context=kb_ctx,
                        original_msg=original_msg,
                        msg_en=msg_en,
                    )
                except ImportError:
                    pass
                if resp:
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
    """
    Answer short-video / reels questions via universal Admin semantic KB.
    No hardcoded terms/seller/privacy keys — any indexed Admin doc can win.
    """
    return format_dynamic_kb_answer(
        original_msg,
        msg_en,
        reply_lang=reply_lang,
        conversation_context=conversation_context,
        suggested_keys=None,
        title_hint="",
    )


def get_welfog_social_links_from_kb() -> list[tuple[str, str, str]]:
    """
    Parse official social URLs from ANY customer knowledge .txt file.
    Format: `Platform Name: https://...` or `Welfog Instagram: https://...`
    Also accepts `Platform https://...` and bare platform-host URLs.
    Admin can add/change URLs or new platforms without code changes.
    Returns list of (slug, display_label, url) in file order.
    """
    ensure_knowledge_cache_fresh()
    url_line = re.compile(
        r"^\s*(?:Welfog\s+)?(.+?)\s*:\s*(https?://\S+)\s*$",
        re.IGNORECASE,
    )
    url_line_loose = re.compile(
        r"^\s*(?:Welfog\s+)?([A-Za-z][A-Za-z0-9+ ._-]{1,40}?)\s+(https?://\S+)\s*$",
        re.IGNORECASE,
    )
    bare_url = re.compile(r"(https?://[^\s<>'\"]+)", re.IGNORECASE)
    found: list[tuple[str, str, str]] = []
    seen_slugs: set[str] = set()

    def _add(label: str, url: str) -> None:
        url = (url or "").strip().rstrip(").,];")
        label = (label or "").strip()
        slug = _social_slug_from_kb_label(label) or _social_slug_from_kb_label(url)
        if not slug or slug in seen_slugs:
            return
        if not re.match(r"^https?://", url, re.I):
            return
        seen_slugs.add(slug)
        found.append((slug, label or slug.title(), url))

    for key in get_customer_kb_keys():
        text = get_runtime_knowledge_files().get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or "http" not in line.lower():
                continue
            if _chunk_is_agent_instruction_blob(line):
                continue
            m = url_line.match(line)
            if m:
                _add(m.group(1), m.group(2))
                continue
            m2 = url_line_loose.match(line)
            if m2:
                _add(m2.group(1), m2.group(2))
                continue
            # URL on line with platform hostname → infer slug from host
            um = bare_url.search(line)
            if not um:
                continue
            url = um.group(1)
            host = re.sub(r"^https?://(www\.)?", "", url, flags=re.I).split("/")[0].lower()
            label_guess = ""
            for needle, slug in (
                ("instagram.com", "Instagram"),
                ("facebook.com", "Facebook"),
                ("fb.com", "Facebook"),
                ("youtube.com", "YouTube"),
                ("youtu.be", "YouTube"),
                ("linkedin.com", "LinkedIn"),
                ("twitter.com", "Twitter"),
                ("x.com", "Twitter"),
            ):
                if needle in host:
                    label_guess = slug
                    break
            if label_guess:
                _add(label_guess, url)
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
        # Not a clear platform-URL ask (e.g. Reels system) — fall through to KB.
        if not requested and not wants_all:
            return ""
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
            f"<a href='{html_escape(url)}'  "
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



def get_knowledge_context(query, keys=None, top_k=3, min_score=None, ai_route=None):
    """
    If keys provided, search within those knowledge files only; else customer KB files.
    Returns an HTML string to inject into the system prompt.
    When Qdrant KB backend is enabled, uses Qdrant only (no MiniLM context cache).
    """
    floor = float(min_score if min_score is not None else KB_CONTEXT_MIN_SCORE)
    norm_q = _normalize_retrieval_query(query)
    if not norm_q:
        return ""

    try:
        from services.knowledge_retriever import (
            format_knowledge_context_string,
            is_qdrant_kb_retrieval_enabled,
            should_use_qdrant_retrieval,
            retrieve_knowledge_context,
        )

        if is_qdrant_kb_retrieval_enabled():
            result = retrieve_knowledge_context(
                norm_q,
                top_k=top_k,
                min_score=floor,
                categories=keys,
                ai_route=ai_route,
                kb_intent=True,
                log=True,
            )
            return format_knowledge_context_string(result.hits)
        if should_use_qdrant_retrieval(ai_route=ai_route, kb_intent=True):
            result = retrieve_knowledge_context(
                norm_q,
                top_k=top_k,
                min_score=floor,
                categories=keys,
                ai_route=ai_route,
                kb_intent=True,
                log=True,
            )
            return format_knowledge_context_string(result.hits)
    except ImportError:
        pass

    ensure_knowledge_cache_fresh()
    if not all_chunks:
        return ""

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
        ai_route=ai_route,
        kb_intent=True,
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


def best_kb_hit(query, keys=None, min_score=None, ai_route=None):
    """
    Best matching KB chunk for grounded answers.
    Qdrant backend: does not require MiniLM in-memory index.
    """
    if not (query or "").strip():
        return None
    floor = float(min_score if min_score is not None else KB_SEMANTIC_MIN_SCORE)
    try:
        from services.knowledge_retriever import is_qdrant_kb_retrieval_enabled

        if is_qdrant_kb_retrieval_enabled():
            return retrieve_best_kb_chunk(
                query, keys=keys, min_score=floor, ai_route=ai_route
            )
    except ImportError:
        pass
    if not all_chunks:
        return None
    return retrieve_best_kb_chunk(query, keys=keys, min_score=floor, ai_route=ai_route)


def top_kb_hits(
    query: str,
    keys=None,
    min_score: float | None = None,
    top_n: int = 4,
    log_retrieval: bool = False,
    ai_route: dict | None = None,
    kb_intent: bool = False,
) -> list[dict]:
    """Top-N semantic KB chunks for reranking (multilingual queries)."""
    if not (query or "").strip():
        return []
    try:
        from services.knowledge_retriever import should_use_qdrant_retrieval

        if not should_use_qdrant_retrieval(ai_route=ai_route, kb_intent=kb_intent):
            ensure_knowledge_cache_fresh()
            if not all_chunks:
                return []
    except ImportError:
        ensure_knowledge_cache_fresh()
        if not all_chunks:
            return []
    floor = float(min_score if min_score is not None else KB_CONTEXT_MIN_SCORE)
    return semantic_kb_search(
        query,
        keys=keys,
        top_n=top_n,
        min_score=floor,
        customer_only=not bool(keys),
        log_retrieval=log_retrieval,
        ai_route=ai_route,
        kb_intent=kb_intent,
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
