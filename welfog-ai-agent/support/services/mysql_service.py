import json
import os
from pathlib import Path
from secrets import token_hex

import pymysql


def get_mysql_connection():
    """Public chat history MySQL (XAMPP). Override via .env: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE."""
    try:
        connect_timeout = float(os.getenv("MYSQL_CONNECT_TIMEOUT", "3") or "3")
        read_timeout = float(os.getenv("MYSQL_READ_TIMEOUT", "8") or "8")
        write_timeout = float(os.getenv("MYSQL_WRITE_TIMEOUT", "8") or "8")
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "welfog_ai"),
            charset="utf8mb4",
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )


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
            if path.is_file():
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
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            cur.execute(
                f"CREATE DATABASE `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
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
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
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
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
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
