"""
Load support/.env once and read connection settings without baking secrets into code.

Clone workflow: copy .env.example → .env (or receive .env from teammate). Never commit .env.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def ensure_dotenv_loaded() -> bool:
    """Load support/.env if present. Safe to call from any module."""
    try:
        from dotenv import load_dotenv

        from support_paths import ENV_FILE

        if os.path.isfile(ENV_FILE):
            load_dotenv(ENV_FILE)
            return True
    except Exception:
        pass
    return False


def env_str(key: str, default: str | None = None) -> str:
    ensure_dotenv_loaded()
    raw = os.getenv(key)
    if raw is None:
        return "" if default is None else str(default)
    return str(raw).strip()


def env_required(key: str) -> str | None:
    """Return stripped value, or None when missing/blank (callers should fail closed)."""
    ensure_dotenv_loaded()
    val = (os.getenv(key) or "").strip()
    return val or None


def mysql_connect_kwargs(*, with_database: bool = True) -> dict[str, Any] | None:
    """
    Build pymysql.connect kwargs from env only.

    Required in .env: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    (MYSQL_PASSWORD may be empty for local XAMPP — key must still be present).
    """
    ensure_dotenv_loaded()

    missing: list[str] = []
    for key in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"):
        if key not in os.environ:
            missing.append(key)
    if missing:
        print(
            "MySQL: missing env key(s) "
            f"{', '.join(missing)} — copy support/.env.example to support/.env "
            "and fill credentials (do not commit .env)."
        )
        return None

    host = (os.getenv("MYSQL_HOST") or "").strip()
    user = (os.getenv("MYSQL_USER") or "").strip()
    database = (os.getenv("MYSQL_DATABASE") or "").strip()
    port_raw = (os.getenv("MYSQL_PORT") or "").strip()
    # Explicit empty password is allowed (key present); unset already handled above.
    password = os.getenv("MYSQL_PASSWORD")
    if password is None:
        password = ""

    if not host or not user or not port_raw:
        print(
            "MySQL: MYSQL_HOST, MYSQL_PORT, MYSQL_USER must be non-empty in support/.env"
        )
        return None
    if with_database and not database:
        print("MySQL: MYSQL_DATABASE must be non-empty in support/.env")
        return None

    try:
        port = int(port_raw)
    except ValueError:
        print(f"MySQL: invalid MYSQL_PORT={port_raw!r}")
        return None

    try:
        connect_timeout = float(os.getenv("MYSQL_CONNECT_TIMEOUT") or "3")
        read_timeout = float(os.getenv("MYSQL_READ_TIMEOUT") or "15")
        write_timeout = float(os.getenv("MYSQL_WRITE_TIMEOUT") or "15")
    except ValueError:
        connect_timeout, read_timeout, write_timeout = 3.0, 15.0, 15.0

    collation = (os.getenv("MYSQL_COLLATION") or "utf8mb4_unicode_ci").strip()

    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "charset": "utf8mb4",
        "collation": collation,
        "init_command": f"SET NAMES utf8mb4 COLLATE {collation}",
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
        "write_timeout": write_timeout,
    }
    if with_database:
        kwargs["database"] = database
    return kwargs


def mysql_database_name() -> str:
    ensure_dotenv_loaded()
    return (os.getenv("MYSQL_DATABASE") or "").strip()
