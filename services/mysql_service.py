import json
import os
import re
import hashlib
from pathlib import Path
from secrets import token_hex

import pymysql

# One collation everywhere — avoids MySQL 8 (utf8mb4_0900_ai_ci) vs XAMPP/legacy (general_ci) mix errors.
MYSQL_COLLATION = (os.getenv("MYSQL_COLLATION") or "utf8mb4_unicode_ci").strip()


def sql_collate(expr: str) -> str:
    """Wrap a SQL expression so string comparisons use MYSQL_COLLATION."""
    return f"{expr} COLLATE {MYSQL_COLLATION}"


def get_mysql_connection():
    """Public chat history MySQL (XAMPP). Override via .env: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE."""
    try:
        connect_timeout = float(os.getenv("MYSQL_CONNECT_TIMEOUT", "10") or "10")
        read_timeout = float(os.getenv("MYSQL_READ_TIMEOUT", "30") or "30")
        write_timeout = float(os.getenv("MYSQL_WRITE_TIMEOUT", "30") or "30")
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "welfog_ai"),
            charset="utf8mb4",
            collation=MYSQL_COLLATION,
            init_command=f"SET NAMES utf8mb4 COLLATE {MYSQL_COLLATION}",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
    except Exception as e:
        # Avoid emoji in logs (Windows console encoding can crash on unicode)
        print(f"MySQL Connection Error: {e}")
        return None


def _ensure_mysql_database_exists() -> bool:
    """Create welfog_ai (or MYSQL_DATABASE) if missing — common fresh XAMPP setup."""
    db_name = (os.getenv("MYSQL_DATABASE") or "welfog_ai").strip()
    if not db_name:
        return False
    try:
        connect_timeout = float(os.getenv("MYSQL_CONNECT_TIMEOUT", "10") or "10")
        read_timeout = float(os.getenv("MYSQL_READ_TIMEOUT", "30") or "30")
        write_timeout = float(os.getenv("MYSQL_WRITE_TIMEOUT", "30") or "30")
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            charset="utf8mb4",
            collation=MYSQL_COLLATION,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                    f"CHARACTER SET utf8mb4 COLLATE {MYSQL_COLLATION}"
                )
            return True
        finally:
            conn.close()
    except Exception as e:
        print(f"MySQL ensure database `{db_name}`: {e}")
        return False


def generate_chat_token():
    return token_hex(16)


def _is_mysql_table_broken(exc: BaseException) -> bool:
    """1932 ghost table, 1146 missing, 1813 orphan tablespace (XAMPP/MySQL)."""
    msg = str(exc).lower()
    return (
        "1932" in msg
        or "1146" in msg
        or "1813" in msg
        or "doesn't exist in engine" in msg
        or "does not exist in engine" in msg
        or "tablespace" in msg
    )


def _force_drop_table(cur, table_name: str) -> None:
    """Drop table even when InnoDB metadata/data files are out of sync."""
    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    try:
        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")
    except Exception:
        pass
    try:
        cur.execute(f"CREATE TABLE `{table_name}` (`_drop` INT) ENGINE=InnoDB")
        cur.execute(f"ALTER TABLE `{table_name}` DISCARD TABLESPACE")
        cur.execute(f"DROP TABLE `{table_name}`")
    except Exception:
        try:
            cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")
        except Exception:
            pass
    cur.execute("SET FOREIGN_KEY_CHECKS = 1")


def _drop_chat_tables(cur) -> None:
    _force_drop_table(cur, "chats")
    _force_drop_table(cur, "chat_sessions")


def _create_chat_tables(cur) -> None:
    cur.execute(
        """
        CREATE TABLE chat_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_token VARCHAR(32) NOT NULL,
            user_id VARCHAR(128) NOT NULL,
            title VARCHAR(512) NOT NULL,
            customer_id VARCHAR(128) NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_user_created (user_id, created_at),
            UNIQUE KEY uq_chat_token (chat_token)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
        """
    )
    cur.execute(
        """
        CREATE TABLE chats (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_token VARCHAR(32) NOT NULL,
            chat_id VARCHAR(128) NOT NULL,
            user_id VARCHAR(128) NULL,
            chat_data MEDIUMTEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_chat_token (chat_token),
            INDEX idx_chat_messages (chat_token, id),
            INDEX idx_chat_id (chat_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
        """
    )


def _align_chat_collations(cur) -> None:
    """Normalize existing tables so JOIN/CAST comparisons do not mix collations."""
    db_name = os.getenv("MYSQL_DATABASE", "welfog_ai")
    try:
        cur.execute(
            f"ALTER DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE {MYSQL_COLLATION}"
        )
    except Exception as e:
        print(f"MySQL collation align (database): {e}")
    for table in ("chat_sessions", "chats"):
        try:
            cur.execute(f"SELECT 1 FROM `{table}` LIMIT 1")
            cur.execute(
                f"ALTER TABLE `{table}` CONVERT TO CHARACTER SET utf8mb4 COLLATE {MYSQL_COLLATION}"
            )
        except Exception as e:
            if "1146" not in str(e):
                print(f"MySQL collation align ({table}): {e}")


def _remove_orphan_innodb_files() -> bool:
    """
    XAMPP #1813: .ibd files left on disk without valid table metadata.
    Safe only for chat tables in welfog_ai (no other tables in that folder).
    """
    db_name = os.getenv("MYSQL_DATABASE", "welfog_ai")
    data_dirs = [
        Path(os.getenv("MYSQL_DATA_DIR", "")),
        Path(f"C:/xampp/mysql/data/{db_name}"),
        Path(f"C:/XAMPP/mysql/data/{db_name}"),
    ]
    removed = False
    for folder in data_dirs:
        if not folder.is_dir():
            continue
        for fname in ("chats.ibd", "chat_sessions.ibd"):
            path = folder / fname
            frm_path = folder / fname.replace(".ibd", ".frm")
            if not path.is_file():
                continue
            # Orphan: .ibd on disk but no table metadata (.frm) — classic XAMPP #1813
            if frm_path.is_file():
                continue
            try:
                path.unlink()
                print(f"MySQL: removed orphan file {path}")
                removed = True
            except OSError as e:
                print(f"MySQL: could not remove {path} ({e}). Stop XAMPP MySQL and delete manually.")
        break
    return removed


def _recreate_welfog_ai_database() -> None:
    """Drop + recreate DB when InnoDB tablespace files are orphaned (XAMPP #1813 / #1932)."""
    db_name = os.getenv("MYSQL_DATABASE", "welfog_ai")
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        charset="utf8mb4",
        collation=MYSQL_COLLATION,
        init_command=f"SET NAMES utf8mb4 COLLATE {MYSQL_COLLATION}",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            cur.execute(
                f"CREATE DATABASE `{db_name}` "
                f"CHARACTER SET utf8mb4 COLLATE {MYSQL_COLLATION}"
            )
            cur.execute(f"USE `{db_name}`")
            _create_chat_tables(cur)
        print(f"MySQL: database `{db_name}` recreated with fresh chat tables.")
    finally:
        conn.close()


def repair_mysql_chat_tables_if_broken() -> bool:
    """
    Detect XAMPP/MySQL #1932 ghost tables and recreate empty working tables.
    Returns True if tables were recreated.
    """
    conn = get_mysql_connection()
    if not conn:
        return False
    recreated = False
    try:
        with conn.cursor() as cur:
            broken = False
            for table in ("chats", "chat_sessions"):
                try:
                    cur.execute(f"SELECT 1 FROM `{table}` LIMIT 1")
                except Exception as e:
                    if _is_mysql_table_broken(e):
                        broken = True
                        print(f"MySQL: table `{table}` broken — will recreate chat tables.")
                    else:
                        raise
            if broken:
                _remove_orphan_innodb_files()
                try:
                    _drop_chat_tables(cur)
                    _create_chat_tables(cur)
                    conn.commit()
                    recreated = True
                    print("MySQL: chat tables recreated successfully.")
                except Exception as drop_err:
                    if _is_mysql_table_broken(drop_err):
                        conn.rollback()
                        conn.close()
                        conn = None
                        _remove_orphan_innodb_files()
                        conn2 = get_mysql_connection()
                        if conn2:
                            with conn2.cursor() as cur2:
                                _drop_chat_tables(cur2)
                                _create_chat_tables(cur2)
                            conn2.commit()
                            conn2.close()
                            recreated = True
                            print("MySQL: chat tables recreated after orphan file cleanup.")
                    else:
                        raise
        return recreated
    except Exception as e:
        print(f"MySQL repair error: {e}")
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if conn:
            conn.close()


def init_mysql_chat_schema():
    """chat_sessions = sidebar / chat id; chats = har message ka row (chat_id, chat_data JSON)."""
    _ensure_mysql_database_exists()
    repair_mysql_chat_tables_if_broken()

    conn = get_mysql_connection()
    if not conn:
        print("MySQL unreachable — chat save/load will not work until DB is running.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_token VARCHAR(32) NOT NULL,
                    user_id VARCHAR(128) NOT NULL,
                    title VARCHAR(512) NOT NULL,
                    customer_id VARCHAR(128) NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user_created (user_id, created_at),
                    UNIQUE KEY uq_chat_token (chat_token)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_token VARCHAR(32) NOT NULL,
                    chat_id VARCHAR(128) NOT NULL DEFAULT '',
                    user_id VARCHAR(128) NULL,
                    chat_data MEDIUMTEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_chat_token (chat_token),
                    INDEX idx_chat_messages (chat_token, id),
                    INDEX idx_chat_id (chat_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci
                """
            )
            _align_chat_collations(cur)
            cur.execute("SHOW COLUMNS FROM chat_sessions LIKE 'customer_id'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chat_sessions ADD COLUMN customer_id VARCHAR(128) NULL")
            cur.execute("SHOW COLUMNS FROM chats LIKE 'chat_id'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chats ADD COLUMN chat_id VARCHAR(128) NOT NULL DEFAULT ''")
            cur.execute("SHOW COLUMNS FROM chats LIKE 'user_id'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chats ADD COLUMN user_id VARCHAR(128) NULL")

            # Backfill random chat tokens for older tables
            cur.execute("SHOW COLUMNS FROM chat_sessions LIKE 'chat_token'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chat_sessions ADD COLUMN chat_token VARCHAR(32) NULL")
                cur.execute("SELECT id FROM chat_sessions")
                for row in cur.fetchall():
                    cur.execute(
                        "UPDATE chat_sessions SET chat_token = %s WHERE id = %s",
                        (token_hex(16), row["id"]),
                    )
                cur.execute("ALTER TABLE chat_sessions MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chat_sessions WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_token (chat_token)")
            else:
                cur.execute("SELECT id FROM chat_sessions WHERE chat_token IS NULL OR chat_token = ''")
                for row in cur.fetchall():
                    cur.execute(
                        "UPDATE chat_sessions SET chat_token = %s WHERE id = %s",
                        (token_hex(16), row["id"]),
                    )
                cur.execute("ALTER TABLE chat_sessions MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chat_sessions WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_token (chat_token)")

            cur.execute("SHOW COLUMNS FROM chats LIKE 'chat_token'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chats ADD COLUMN chat_token VARCHAR(32) NULL")
                cur.execute(
                    "UPDATE chats c JOIN chat_sessions s ON c.chat_id = s.id SET c.chat_token = s.chat_token"
                )
                cur.execute("UPDATE chats SET chat_id = chat_token WHERE chat_id IS NULL OR chat_id = ''")
                cur.execute("ALTER TABLE chats MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chats WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chats ADD UNIQUE KEY uq_chat_token (chat_token)")
            else:
                cur.execute("UPDATE chats c JOIN chat_sessions s ON c.chat_id = s.id SET c.chat_token = s.chat_token WHERE c.chat_token IS NULL OR c.chat_token = ''")
                cur.execute("UPDATE chats SET chat_id = chat_token WHERE chat_id IS NULL OR chat_id = ''")
                cur.execute("ALTER TABLE chats MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chats WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chats ADD UNIQUE KEY uq_chat_token (chat_token)")
        conn.commit()
    except Exception as e:
        print(f"MySQL schema init error: {e}")
        if _is_mysql_table_broken(e):
            _remove_orphan_innodb_files()
            if repair_mysql_chat_tables_if_broken():
                print("MySQL: schema init recovered after chat table repair.")
    finally:
        conn.close()


def init_mysql_knowledge_documents_schema():
    """
    Step-1 knowledge schema migration.
    Adds MySQL `knowledge_documents` without touching existing tables/logic.
    """
    _ensure_mysql_database_exists()
    conn = get_mysql_connection()
    if not conn:
        print("MySQL unreachable — knowledge_documents schema init skipped.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(512) NOT NULL,
                    category VARCHAR(128) NOT NULL,
                    content LONGTEXT NOT NULL,
                    language VARCHAR(32) NOT NULL DEFAULT 'en',
                    status VARCHAR(32) NOT NULL DEFAULT 'active',
                    version INT NOT NULL DEFAULT 1,
                    index_status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_kd_category_status_lang (category, status, language),
                    INDEX idx_kd_updated_at (updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE {MYSQL_COLLATION}
                """
            )
        conn.commit()
    except Exception as e:
        print(f"MySQL knowledge_documents schema init error: {e}")
    finally:
        conn.close()


def _infer_knowledge_category_from_title(title: str) -> str:
    """
    Soft taxonomy label for Admin docs (never used to hard-exclude customer knowledge).

    Internal agent docs (welfog-api*, system-messages) are tagged system/api only by
    exact canonical key — never by loose substring like "api" inside "application".
    """
    from services.knowledge_keys import canonical_knowledge_key, is_internal_agent_knowledge_key

    t = (title or "").strip().lower()
    if not t:
        return "general"
    key = canonical_knowledge_key(t)
    if key.startswith("welfog_api"):
        return "api"
    if key in ("system_messages", "system_messages_2", "system_message"):
        return "system"
    if is_internal_agent_knowledge_key(key):
        return "system"
    if any(k in t for k in ("refund", "return")):
        return "refund"
    if any(k in t for k in ("payment", "invoice")):
        return "payment"
    if any(k in t for k in ("shipping", "delivery")):
        return "shipping"
    if "seller" in t:
        return "seller"
    if any(k in t for k in ("support", "faq", "faqs")):
        return "support"
    if any(k in t for k in ("policy", "privacy", "terms")):
        return "policy"
    if "company" in t:
        return "company"
    return "general"


def import_knowledge_txt_files_to_mysql(knowledge_dir: str) -> dict:
    """
    One-time migration helper:
    imports each *.txt from support/knowledge into knowledge_documents.
    Duplicate policy: if same `title` already exists, skip.
    """
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return {"imported": 0, "skipped_duplicates": 0, "total_rows": 0, "errors": 1}

    imported = 0
    skipped_duplicates = 0
    errors = 0
    kdir = Path(knowledge_dir)
    txt_files = sorted(kdir.glob("*.txt"))
    try:
        with conn.cursor() as cur:
            for fpath in txt_files:
                try:
                    title = fpath.stem.strip()
                    if not title:
                        continue
                    cur.execute(
                        "SELECT id FROM knowledge_documents WHERE title = %s LIMIT 1",
                        (title,),
                    )
                    if cur.fetchone():
                        skipped_duplicates += 1
                        continue

                    body = fpath.read_text(encoding="utf-8")
                    cur.execute(
                        """
                        INSERT INTO knowledge_documents
                            (title, category, content, language, status, version, index_status)
                        VALUES
                            (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            title,
                            _infer_knowledge_category_from_title(title),
                            body,
                            "auto",
                            "active",
                            1,
                            "pending",
                        ),
                    )
                    imported += 1
                except Exception as row_err:
                    errors += 1
                    print(f"knowledge import row error ({fpath.name}): {row_err}")

            conn.commit()
            cur.execute("SELECT COUNT(*) AS c FROM knowledge_documents")
            total_rows = int((cur.fetchone() or {}).get("c") or 0)
    except Exception as e:
        print(f"knowledge import error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        errors += 1
        total_rows = 0
    finally:
        conn.close()

    return {
        "imported": imported,
        "skipped_duplicates": skipped_duplicates,
        "total_rows": total_rows,
        "errors": errors,
    }


def init_mysql_knowledge_chunks_schema():
    """Step-3 schema: deterministic cleaned/chunked documents (no vectors yet)."""
    _ensure_mysql_database_exists()
    conn = get_mysql_connection()
    if not conn:
        print("MySQL unreachable — knowledge_document_chunks schema init skipped.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_document_chunks (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    chunk_id VARCHAR(128) NOT NULL,
                    doc_id BIGINT NOT NULL,
                    version INT NOT NULL,
                    chunk_no INT NOT NULL,
                    title VARCHAR(512) NOT NULL,
                    category VARCHAR(128) NOT NULL,
                    language VARCHAR(32) NOT NULL,
                    content LONGTEXT NOT NULL,
                    content_hash CHAR(64) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_kdc_chunk_id (chunk_id),
                    UNIQUE KEY uq_kdc_doc_ver_chunk (doc_id, version, chunk_no),
                    INDEX idx_kdc_doc_ver (doc_id, version),
                    INDEX idx_kdc_category_lang (category, language)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE {MYSQL_COLLATION}
                """
            )
        conn.commit()
    except Exception as e:
        print(f"MySQL knowledge_document_chunks schema init error: {e}")
    finally:
        conn.close()


def clean_knowledge_document_content(raw: str) -> str:
    """
    UTF-8-safe normalization:
    - normalize line endings
    - trim trailing spaces
    - collapse duplicate blank lines
    - normalize intra-line whitespace (spaces/tabs)
    """
    text = (raw or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    blank_run = 0
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if not ln:
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        lines.append(ln)
    return "\n".join(lines).strip()


def chunk_knowledge_document_content(cleaned: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """
    Deterministic chunking with paragraph preference.
    chunk_size/overlap are character-based.
    """
    content = (cleaned or "").strip()
    if not content:
        return []
    size = max(200, int(chunk_size or 900))
    ov = max(0, min(int(overlap or 120), size // 2))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for para in paragraphs:
        if len(para) > size:
            words = para.split()
            seg = ""
            for w in words:
                cand = f"{seg} {w}".strip()
                if len(cand) <= size:
                    seg = cand
                else:
                    if seg:
                        if current:
                            _flush()
                        chunks.append(seg.strip())
                    seg = w
            if seg:
                if current:
                    _flush()
                chunks.append(seg.strip())
            continue

        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= size:
            current = candidate
        else:
            _flush()
            current = para
    _flush()

    if ov == 0 or len(chunks) <= 1:
        return chunks

    out: list[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            out.append(ch)
            continue
        tail = chunks[i - 1][-ov:]
        merged = f"{tail}\n{ch}".strip()
        out.append(merged[: size + ov])
    return out


def ingest_knowledge_documents_for_chunking(chunk_size: int = 900, overlap: int = 120) -> dict:
    """
    Step-3 ingestion pipeline (no vectors):
    pending -> processing -> pending_ready
    """
    init_mysql_knowledge_documents_schema()
    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return {"documents": 0, "chunks": 0, "avg_chunk_size": 0.0, "errors": 1}

    total_docs = 0
    total_chunks = 0
    chunk_chars = 0
    errors = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, category, content, language, version
                FROM knowledge_documents
                WHERE status = 'active' AND index_status = 'pending'
                ORDER BY id ASC
                """
            )
            docs = cur.fetchall() or []

            for doc in docs:
                doc_id = int(doc["id"])
                version = int(doc.get("version") or 1)
                title = (doc.get("title") or "").strip()
                category = (doc.get("category") or "general").strip() or "general"
                language = (doc.get("language") or "auto").strip() or "auto"
                try:
                    cur.execute(
                        "UPDATE knowledge_documents SET index_status = 'processing' WHERE id = %s",
                        (doc_id,),
                    )
                    cleaned = clean_knowledge_document_content(doc.get("content") or "")
                    chunks = chunk_knowledge_document_content(cleaned, chunk_size=chunk_size, overlap=overlap)

                    cur.execute(
                        "DELETE FROM knowledge_document_chunks WHERE doc_id = %s AND version = %s",
                        (doc_id, version),
                    )

                    for i, text in enumerate(chunks, start=1):
                        content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                        chunk_id = f"{doc_id}:{version}:{i}"
                        cur.execute(
                            """
                            INSERT INTO knowledge_document_chunks
                                (chunk_id, doc_id, version, chunk_no, title, category, language, content, content_hash)
                            VALUES
                                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (chunk_id, doc_id, version, i, title, category, language, text, content_hash),
                        )
                        total_chunks += 1
                        chunk_chars += len(text)

                    cur.execute(
                        "UPDATE knowledge_documents SET index_status = 'pending_ready' WHERE id = %s",
                        (doc_id,),
                    )
                    total_docs += 1
                except Exception as row_err:
                    errors += 1
                    cur.execute(
                        "UPDATE knowledge_documents SET index_status = 'pending' WHERE id = %s",
                        (doc_id,),
                    )
                    print(f"knowledge chunking row error (doc_id={doc_id}): {row_err}")
            conn.commit()
    except Exception as e:
        errors += 1
        print(f"knowledge chunking pipeline error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()

    avg = float(chunk_chars / total_chunks) if total_chunks else 0.0
    return {"documents": total_docs, "chunks": total_chunks, "avg_chunk_size": avg, "errors": errors}


def get_knowledge_document_by_title(title: str) -> dict | None:
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, category, content, language, status, version, index_status "
                "FROM knowledge_documents WHERE title = %s LIMIT 1",
                ((title or "").strip(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_knowledge_document_by_id(doc_id: int) -> dict | None:
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, category, content, language, status, version, index_status "
                "FROM knowledge_documents WHERE id = %s LIMIT 1",
                (int(doc_id),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def list_active_knowledge_documents() -> list[dict]:
    """Active knowledge documents from MySQL (production source of truth for originals)."""
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, category, content, language, status, version, index_status,
                       created_at, updated_at
                FROM knowledge_documents
                WHERE status = 'active'
                ORDER BY title ASC
                """
            )
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"list_active_knowledge_documents error: {e}")
        return []
    finally:
        conn.close()


def create_knowledge_document_record(
    title: str,
    content: str,
    *,
    category: str | None = None,
    language: str = "auto",
) -> int | None:
    """Insert active knowledge document with index_status=pending."""
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return None
    t = (title or "").strip()
    if not t:
        return None
    cat = (category or _infer_knowledge_category_from_title(t)).strip() or "general"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_documents
                    (title, category, content, language, status, version, index_status)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """,
                (t, cat, content or "", language or "auto", "active", 1, "pending"),
            )
            doc_id = int(cur.lastrowid)
        conn.commit()
        return doc_id
    except Exception as e:
        print(f"create_knowledge_document_record error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        conn.close()


def delete_knowledge_document_record(doc_id: int) -> bool:
    init_mysql_knowledge_documents_schema()
    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_document_chunks WHERE doc_id = %s", (int(doc_id),))
            cur.execute("DELETE FROM knowledge_documents WHERE id = %s", (int(doc_id),))
        conn.commit()
        return True
    except Exception as e:
        print(f"delete_knowledge_document_record error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()


def delete_knowledge_document_chunks(doc_id: int, *, version: int | None = None) -> int:
    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return 0
    try:
        with conn.cursor() as cur:
            if version is not None:
                cur.execute(
                    "DELETE FROM knowledge_document_chunks WHERE doc_id = %s AND version = %s",
                    (int(doc_id), int(version)),
                )
            else:
                cur.execute(
                    "DELETE FROM knowledge_document_chunks WHERE doc_id = %s",
                    (int(doc_id),),
                )
            deleted = int(cur.rowcount or 0)
        conn.commit()
        return deleted
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        conn.close()


def delete_knowledge_document_chunks_except_version(doc_id: int, keep_version: int) -> int:
    """Remove MySQL chunks for all versions other than keep_version."""
    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_document_chunks WHERE doc_id = %s AND version <> %s",
                (int(doc_id), int(keep_version)),
            )
            deleted = int(cur.rowcount or 0)
        conn.commit()
        return deleted
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        conn.close()


def set_knowledge_document_index_status(doc_id: int, status: str) -> None:
    conn = get_mysql_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE knowledge_documents SET index_status = %s WHERE id = %s",
                (status, int(doc_id)),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def process_knowledge_document_chunks(
    doc_id: int,
    *,
    chunk_size: int = 900,
    overlap: int = 120,
    replace_all_versions: bool = False,
) -> dict:
    """
    Clean + chunk one document.
    index_status: processing -> pending_ready (or failed).
    """
    init_mysql_knowledge_documents_schema()
    init_mysql_knowledge_chunks_schema()
    conn = get_mysql_connection()
    if not conn:
        return {"ok": False, "doc_id": doc_id, "chunks": 0, "error": "mysql_unreachable"}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, category, content, language, version, status
                FROM knowledge_documents
                WHERE id = %s
                LIMIT 1
                FOR UPDATE
                """,
                (int(doc_id),),
            )
            doc = cur.fetchone()
            if not doc:
                conn.rollback()
                return {"ok": False, "doc_id": doc_id, "chunks": 0, "error": "document_not_found"}

            if (doc.get("status") or "").strip().lower() != "active":
                conn.rollback()
                return {"ok": False, "doc_id": doc_id, "chunks": 0, "error": "document_not_active"}

            version = int(doc.get("version") or 1)
            title_raw = (doc.get("title") or "").strip()
            try:
                from services.knowledge_keys import canonical_knowledge_key

                title = canonical_knowledge_key(title_raw) or title_raw
            except ImportError:
                title = title_raw
            category = (doc.get("category") or "general").strip() or "general"
            language = (doc.get("language") or "auto").strip() or "auto"

            cur.execute(
                "UPDATE knowledge_documents SET index_status = 'processing' WHERE id = %s",
                (doc_id,),
            )

            cleaned = clean_knowledge_document_content(doc.get("content") or "")
            chunks = chunk_knowledge_document_content(cleaned, chunk_size=chunk_size, overlap=overlap)

            if replace_all_versions:
                cur.execute(
                    "DELETE FROM knowledge_document_chunks WHERE doc_id = %s",
                    (doc_id,),
                )
            else:
                cur.execute(
                    "DELETE FROM knowledge_document_chunks WHERE doc_id = %s AND version = %s",
                    (doc_id, version),
                )

            for i, text in enumerate(chunks, start=1):
                content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                chunk_id = f"{doc_id}:{version}:{i}"
                cur.execute(
                    """
                    INSERT INTO knowledge_document_chunks
                        (chunk_id, doc_id, version, chunk_no, title, category, language, content, content_hash)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (chunk_id, doc_id, version, i, title, category, language, text, content_hash),
                )

            cur.execute(
                "UPDATE knowledge_documents SET index_status = 'pending_ready' WHERE id = %s",
                (doc_id,),
            )
        conn.commit()
        return {
            "ok": True,
            "doc_id": int(doc_id),
            "version": version,
            "chunks": len(chunks),
        }
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        set_knowledge_document_index_status(doc_id, "failed")
        return {"ok": False, "doc_id": doc_id, "chunks": 0, "error": str(exc)}
    finally:
        conn.close()


def bump_knowledge_document_version(
    doc_id: int,
    content: str,
    *,
    category: str | None = None,
) -> dict:
    """
    Update document content and increment version.
    Sets index_status=processing (caller runs chunk + embed).
    """
    init_mysql_knowledge_documents_schema()
    conn = get_mysql_connection()
    if not conn:
        return {"ok": False, "error": "mysql_unreachable"}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT version, category FROM knowledge_documents WHERE id = %s FOR UPDATE",
                (int(doc_id),),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return {"ok": False, "error": "document_not_found"}
            old_version = int(row.get("version") or 1)
            new_version = old_version + 1
            cat = (category or row.get("category") or "general").strip() or "general"
            cur.execute(
                """
                UPDATE knowledge_documents
                SET content = %s, category = %s, version = %s, index_status = 'processing', updated_at = NOW()
                WHERE id = %s
                """,
                (content or "", cat, new_version, int(doc_id)),
            )
        conn.commit()
        return {
            "ok": True,
            "doc_id": int(doc_id),
            "old_version": old_version,
            "new_version": new_version,
        }
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        set_knowledge_document_index_status(doc_id, "failed")
        return {"ok": False, "error": str(exc)}
    finally:
        conn.close()


def _ensure_chat_session_row(cursor, chat_id: str, user_id, user_message: str) -> None:
    """Sidebar row must exist before/with chats — history INNER JOIN depends on it."""
    if not chat_id:
        return
    title = (user_message[:30] + "...") if len(user_message or "") > 30 else (user_message or "New chat")
    uid = str(user_id or "").strip() or "0"
    cursor.execute(
        "SELECT id FROM chat_sessions WHERE chat_token = %s LIMIT 1",
        (chat_id,),
    )
    if cursor.fetchone():
        return
    try:
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, title, chat_token, customer_id) VALUES (%s, %s, %s, %s)",
            (uid, title, chat_id, uid),
        )
    except Exception as e:
        if "customer_id" in str(e).lower():
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, title, chat_token) VALUES (%s, %s, %s)",
                (uid, title, chat_id),
            )
        else:
            raise


def db_store_turn_pair(chat_id, user_message, bot_message, user_id=None):
    """Atomically append user + bot messages — avoids parallel thread lost updates."""
    if not chat_id:
        return
    conn = get_mysql_connection()
    if not conn:
        return
    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cursor:
            if user_id:
                _ensure_chat_session_row(cursor, chat_id, user_id, user_message)
            cursor.execute("SHOW COLUMNS FROM chats LIKE 'user_id'")
            has_user_id = cursor.fetchone() is not None

            if numeric_chat_id is not None:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                    (chat_id, numeric_chat_id),
                )
            else:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1",
                    (chat_id,),
                )
            row = cursor.fetchone()

            new_rows = [
                {"sender": "user", "text": user_message},
                {"sender": "bot", "text": bot_message},
            ]
            if row and row.get("chat_data"):
                try:
                    chat_data = json.loads(row["chat_data"])
                except (TypeError, json.JSONDecodeError):
                    chat_data = []
                if isinstance(chat_data, dict):
                    chat_data = [chat_data]
                elif not isinstance(chat_data, list):
                    chat_data = []
                chat_data.extend(new_rows)
                chat_data_json = json.dumps(chat_data, ensure_ascii=False)
                if has_user_id:
                    cursor.execute(
                        "UPDATE chats SET chat_data = %s, chat_id = %s, user_id = %s, updated_at = NOW() WHERE chat_token = %s OR chat_id = %s",
                        (
                            chat_data_json,
                            chat_id,
                            user_id,
                            chat_id,
                            numeric_chat_id if numeric_chat_id is not None else -1,
                        ),
                    )
                else:
                    cursor.execute(
                        "UPDATE chats SET chat_data = %s, chat_id = %s, updated_at = NOW() WHERE chat_token = %s OR chat_id = %s",
                        (
                            chat_data_json,
                            chat_id,
                            chat_id,
                            numeric_chat_id if numeric_chat_id is not None else -1,
                        ),
                    )
            else:
                chat_data_json = json.dumps(new_rows, ensure_ascii=False)
                if has_user_id:
                    cursor.execute(
                        "INSERT INTO chats (chat_token, chat_id, chat_data, user_id, created_at, updated_at) VALUES (%s, %s, %s, %s, NOW(), NOW())",
                        (chat_id, chat_id, chat_data_json, user_id),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO chats (chat_token, chat_id, chat_data, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())",
                        (chat_id, chat_id, chat_data_json),
                    )
        conn.commit()
    except Exception as e:
        print(f"MySQL turn-pair store error: {e}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def db_store_message(chat_id, sender, message, user_id=None):
    conn = get_mysql_connection()
    if not conn:
        return

    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM chats LIKE 'user_id'")
            has_user_id = cursor.fetchone() is not None

            if numeric_chat_id is not None:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                    (chat_id, numeric_chat_id),
                )
            else:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1",
                    (chat_id,),
                )
            row = cursor.fetchone()

            new_message = {"sender": sender, "text": message}
            if row and row.get("chat_data"):
                try:
                    chat_data = json.loads(row["chat_data"])
                except (TypeError, json.JSONDecodeError):
                    chat_data = []

                if isinstance(chat_data, dict):
                    chat_data = [chat_data]
                elif not isinstance(chat_data, list):
                    chat_data = []

                chat_data.append(new_message)
                chat_data_json = json.dumps(chat_data, ensure_ascii=False)
                if has_user_id:
                    cursor.execute(
                        "UPDATE chats SET chat_data = %s, chat_id = %s, user_id = %s, updated_at = NOW() WHERE chat_token = %s OR chat_id = %s",
                        (chat_data_json, chat_id, user_id, chat_id, numeric_chat_id if numeric_chat_id is not None else -1),
                    )
                else:
                    cursor.execute(
                        "UPDATE chats SET chat_data = %s, chat_id = %s, updated_at = NOW() WHERE chat_token = %s OR chat_id = %s",
                        (chat_data_json, chat_id, chat_id, numeric_chat_id if numeric_chat_id is not None else -1),
                    )
            else:
                chat_data_json = json.dumps([new_message], ensure_ascii=False)
                if has_user_id:
                    cursor.execute(
                        "INSERT INTO chats (chat_token, chat_id, chat_data, user_id, created_at, updated_at) VALUES (%s, %s, %s, %s, NOW(), NOW())",
                        (chat_id, chat_id, chat_data_json, user_id),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO chats (chat_token, chat_id, chat_data, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())",
                        (chat_id, chat_id, chat_data_json),
                    )

        conn.commit()
    except Exception as e:
        print(f"MySQL Insert/Update Error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def db_get_recent_messages(chat_id, limit=12):
    """Last N messages for this chat (includes current user message once stored)."""
    if not chat_id or limit <= 0:
        return []
    conn = get_mysql_connection()
    if not conn:
        return []
    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cur:
            if numeric_chat_id is not None:
                cur.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                    (chat_id, numeric_chat_id),
                )
            else:
                cur.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1",
                    (chat_id,),
                )
            row = cur.fetchone()

        if not row or not row.get("chat_data"):
            return []

        try:
            chat_data = json.loads(row["chat_data"])
        except (TypeError, json.JSONDecodeError):
            return []

        if isinstance(chat_data, dict):
            chat_data = [chat_data]
        elif not isinstance(chat_data, list):
            return []

        recent = chat_data[-limit:]
        out = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            out.append({
                "sender": item.get("sender"),
                "message": item.get("text") if item.get("text") is not None else item.get("message"),
            })
        return out
    except Exception as e:
        print(f"db_get_recent_messages: {e}")
        return []
    finally:
        conn.close()
