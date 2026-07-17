"""
Welfog support application entrypoint.

- Public chat + MySQL history: routes.chat_routes
- Admin + SQLite (knowledge files): routes.admin_routes, admin_models, extensions
- AI / KB / APIs: services/* and utils/* (not duplicated here)
"""
import os
import signal
import socket
import subprocess
import sys
from secrets import token_hex

from dotenv import load_dotenv
from flask import Flask

from admin_models import AdminUser, AgentSettings  # noqa: F401 — register model for create_all
from extensions import db, login_manager
from routes.admin_routes import register_admin_routes
from routes.chat_routes import register_chat_routes
from services.mysql_service import (
    ensure_knowledge_mysql_ready,
    init_mysql_chat_schema,
    init_mysql_knowledge_chunks_schema,
    init_mysql_knowledge_documents_schema,
)
from services.qdrant_service import init_qdrant_on_startup
from support_paths import BASE_DIR, ENV_FILE

load_dotenv(ENV_FILE)

_hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)


def _warmup_ai_on_startup() -> None:
    """Load SentenceTransformer + KB vectors at startup (before first /chat)."""
    if (os.getenv("DISABLE_EMBEDDINGS", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        print("[startup] embeddings disabled (DISABLE_EMBEDDINGS=1)", flush=True)
        return
    try:
        print("[startup] Loading embedding weights + KB text index...", flush=True)
        from services.embedding import encode_texts, get_greetings_vecs
        from services.kb_service import ensure_kb_vectors, refresh_knowledge_cache

        refresh_knowledge_cache(build_vectors=True)
        encode_texts(["welfog startup warmup"])
        get_greetings_vecs()
        ensure_kb_vectors()
        print("[startup] Warming catalog + order fast-path modules...", flush=True)
        from services.brain_direct_dispatch import try_structural_product_catalog_reply

        try_structural_product_catalog_reply  # noqa: B018 — force import chain at startup
        from services.product_search_flow import run_product_search_ai_flow

        run_product_search_ai_flow  # noqa: B018
        from services.order_details_flow import (  # noqa: F401
            message_wants_order_details_or_invoice,
        )
        from utils import helpers  # noqa: F401

        helpers.turn_is_obvious_product_shopping_turn(
            "warmup product browse", "warmup product browse", ""
        )
        try:
            from services.welfog_api import warmup_welfog_category_caches

            warmup_welfog_category_caches()
            print("[startup] category cache warmup done", flush=True)
        except Exception as cat_exc:
            print(f"[startup] category cache warmup skip: {cat_exc}", flush=True)
        try:
            from services.ai_service import ai_brain_route

            ai_brain_route("show electronics products", "", "en")
            print("[startup] routing LLM warmup done", flush=True)
        except Exception as route_exc:
            print(f"[startup] routing LLM warmup skip: {route_exc}", flush=True)
        print("[startup] AI embeddings + KB vector index ready", flush=True)
    except Exception as exc:
        print(f"[startup] embedding warmup failed (lazy load on first use): {exc}", flush=True)


def _resolve_sqlalchemy_database_uri() -> str:
    """
    Admin SQLite URI from .env, always rooted at support/ BASE_DIR.

    Flask turns bare relative ``sqlite:///welfog_v2.db`` into
    ``support/instance/welfog_v2.db`` (often empty) — that caused
    "Invalid username or password" after DATABASE_URL was added to .env.
    """
    raw = (os.getenv("DATABASE_URL") or "").strip()
    default_path = os.path.join(BASE_DIR, "welfog_v2.db")
    if not raw:
        return "sqlite:///" + default_path.replace("\\", "/")
    if raw.startswith("sqlite:///"):
        path_part = raw[len("sqlite:///") :]
        # Absolute POSIX or Windows (C:/...) — keep as-is.
        if path_part.startswith("/") or (
            len(path_part) >= 3 and path_part[1] == ":" and path_part[2] in "/\\"
        ):
            return "sqlite:///" + path_part.replace("\\", "/")
        # Relative path → support/ (never Flask instance/)
        abs_path = os.path.normpath(os.path.join(BASE_DIR, path_part))
        return "sqlite:///" + abs_path.replace("\\", "/")
    return raw


def create_app():
    app = Flask(__name__)
    # Secrets only from support/.env — never hardcode FLASK_SECRET_KEY / SECRET_KEY.
    secret = (os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "").strip()
    if not secret:
        secret = token_hex(32)
        print(
            "[startup] FLASK_SECRET_KEY missing in .env — using ephemeral key "
            "(set FLASK_SECRET_KEY in support/.env for stable sessions).",
            flush=True,
        )
    app.secret_key = secret
    db_url = _resolve_sqlalchemy_database_uri()
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    print(f"[startup] Admin SQLite: {db_url}", flush=True)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["REMEMBER_COOKIE_HTTPONLY"] = True
    app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "admin_login"

    @login_manager.user_loader
    def load_user(user_id):
        return AdminUser.query.get(int(user_id))

    init_mysql_chat_schema()
    init_mysql_knowledge_documents_schema()
    init_mysql_knowledge_chunks_schema()
    try:
        ensure_knowledge_mysql_ready()
    except Exception as kb_ready_err:
        print(f"[startup] knowledge MySQL ready check failed: {kb_ready_err}", flush=True)
    init_qdrant_on_startup()
    try:
        from services.knowledge_embedding_indexer import (
            reconcile_knowledge_vectors_after_mysql_recovery,
        )

        reconcile_knowledge_vectors_after_mysql_recovery()
    except Exception as kb_vec_err:
        print(f"[startup] knowledge vector reconcile failed: {kb_vec_err}", flush=True)
    register_chat_routes(app)
    register_admin_routes(app)

    @app.route("/health")
    def health():
        return {"status": "ok"}, 200
    _warmup_ai_on_startup()
    _register_request_logging(app)
    return app


def _register_request_logging(app: Flask) -> None:
    @app.after_request
    def _log_request(response):
        try:
            path = (request.path or "").strip()
            if path in ("/chat", "/") or path.startswith("/api/"):
                print(
                    f"[http] {request.method} {path} -> {response.status_code}",
                    flush=True,
                )
        except Exception:
            pass
        return response


app = create_app()


def _pids_listening_on_port(port: int) -> list[int]:
    """Return PIDs bound to 127.0.0.1:port (Windows netstat)."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True,
            errors="replace",
            timeout=8,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []
    needle = f":{port}"
    pids: list[int] = []
    for line in out.splitlines():
        if "LISTENING" not in line or needle not in line:
            continue
        parts = line.split()
        if parts and parts[-1].isdigit():
            pids.append(int(parts[-1]))
    return list(dict.fromkeys(pids))


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _ensure_port_available(port: int) -> None:
    """Refuse to start if another server is already on this port (orphan after Ctrl+C)."""
    if _port_is_free(port):
        return
    pids = _pids_listening_on_port(port)
    pid_hint = ", ".join(str(p) for p in pids) if pids else "unknown"
    print(
        f"[startup] Port {port} is already in use (PID {pid_hint}).\n"
        f"  Another python app.py is still running — kill it first:\n"
        f"    taskkill /F /PID {pids[0] if pids else '<pid>'}\n"
        f"  Or in PowerShell: Stop-Process -Id {pids[0] if pids else '<pid>'} -Force",
        flush=True,
    )
    sys.exit(1)


def _install_shutdown_handlers() -> None:
    """Ctrl+C must stop the server; hard exit avoids Windows forrtl hang."""

    def _stop(_signum, _frame):
        print("\n[shutdown] Stopping server (Ctrl+C)...", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)


if __name__ == "__main__":
    _install_shutdown_handlers()
    port = int((os.getenv("FLASK_PORT") or "5000").strip() or "5000")
    _ensure_port_available(port)

    with app.app_context():
        db.create_all()

    # Windows: never use reloader — spawns orphan child that survives Ctrl+C.
    reloader_env = (os.getenv("FLASK_USE_RELOADER", "0") or "0").strip().lower()
    use_reloader = reloader_env not in ("0", "false", "no", "off") and sys.platform != "win32"
    debug = (os.getenv("FLASK_DEBUG", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    print(
        f"[startup] Welfog support server PID {os.getpid()} on http://127.0.0.1:{port} "
        f"(Ctrl+C to stop, reloader={'on' if use_reloader else 'off'})",
        flush=True,
    )
    # Sync handler — one /chat at a time per worker; async threads caused encode-lock pileups.
    os.environ.setdefault("CHAT_ASYNC_HANDLER", "0")
    app.run(port=port, debug=debug, use_reloader=use_reloader, threaded=True)
