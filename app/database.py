import sqlite3
import bcrypt
import os
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "data/users.db")


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name  TEXT NOT NULL,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER REFERENCES users(id),
            identity   TEXT,
            confidence REAL,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def register_user(first_name: str, last_name: str, username: str, password: str) -> bool:
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO users (first_name, last_name, username, password) VALUES (?, ?, ?, ?)",
            (first_name, last_name, username, hashed),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(username: str, password: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row["password"].encode()):
        return dict(row)
    return None


def log_recognition(user_id: int, identity: str, confidence: float):
    conn = get_connection()
    conn.execute(
        "INSERT INTO recognition_logs (user_id, identity, confidence) VALUES (?, ?, ?)",
        (user_id, identity, confidence),
    )
    conn.commit()
    conn.close()


def get_recognition_logs(user_id: int) -> list[tuple]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT identity, confidence, timestamp FROM recognition_logs "
        "WHERE user_id = ? ORDER BY timestamp DESC LIMIT 200",
        (user_id,),
    ).fetchall()
    conn.close()
    return [tuple(r) for r in rows]
