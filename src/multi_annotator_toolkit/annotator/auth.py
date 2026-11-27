"""SQLite user store (bcrypt passwords) and HMAC-signed session cookies for Annotator UI."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any

import bcrypt

VALID_ROLES = ("annotator", "admin")
SESSION_COOKIE_NAME = "mat_session"


class UserStore:
    """Minimal fail-fast SQLite user store kept outside FiftyOne."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('annotator', 'admin')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'cookie_secret'"
            ).fetchone()
            if row is None:
                secret = os.environ.get("MAT_SECRET_KEY") or secrets.token_hex(32)
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('cookie_secret', ?)",
                    (secret,),
                )

    def get_cookie_secret(self) -> str:
        env_secret = os.environ.get("MAT_SECRET_KEY")
        if env_secret:
            return env_secret
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'cookie_secret'"
            ).fetchone()
            if row is None:
                raise RuntimeError("Missing cookie_secret in user database.")
            return str(row["value"])

    def create_user(
        self, username: str, password: str, role: str = "annotator"
    ) -> dict[str, Any]:
        clean_user = username.strip()
        if not clean_user:
            raise ValueError("Username must be non-empty.")
        if len(password) < 4:
            raise ValueError("Password must be at least 4 characters.")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}; expected one of {VALID_ROLES}")

        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, active, created_at) VALUES (?, ?, ?, 1, ?)",
                    (clean_user, pw_hash, role, created_at),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"User {clean_user!r} already exists.") from exc
        return {
            "username": clean_user,
            "role": role,
            "active": True,
            "created_at": created_at,
        }

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, role, active, created_at FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not row["active"]:
            return None
        if not bcrypt.checkpw(
            password.encode("utf-8"), str(row["password_hash"]).encode("utf-8")
        ):
            return None
        return {
            "username": row["username"],
            "role": row["role"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }

    def get_user(self, username: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, role, active, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {
            "username": row["username"],
            "role": row["role"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username, role, active, created_at FROM users ORDER BY username"
            ).fetchall()
        return [
            {
                "username": r["username"],
                "role": r["role"],
                "active": bool(r["active"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def update_user(
        self,
        username: str,
        *,
        active: bool | None = None,
        role: str | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_user(username)
        if existing is None:
            raise ValueError(f"User {username!r} does not exist.")
        if role is not None and role not in VALID_ROLES:
            raise ValueError(f"Invalid role {role!r}; expected one of {VALID_ROLES}")
        if password is not None and len(password) < 4:
            raise ValueError("Password must be at least 4 characters.")

        with self._connect() as conn:
            if active is not None:
                conn.execute(
                    "UPDATE users SET active = ? WHERE username = ?",
                    (1 if active else 0, username),
                )
            if role is not None:
                conn.execute(
                    "UPDATE users SET role = ? WHERE username = ?",
                    (role, username),
                )
            if password is not None:
                pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (pw_hash, username),
                )
        updated = self.get_user(username)
        assert updated is not None
        return updated


def sign_session_token(username: str, role: str, secret: str) -> str:
    payload = json.dumps({"u": username, "r": role}, separators=(",", ":")).encode("utf-8")
    b64 = base64.urlsafe_b64encode(payload).decode("ascii")
    sig = hmac.new(secret.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def verify_session_token(token: str, secret: str) -> dict[str, str] | None:
    if not token or "." not in token:
        return None
    b64, sig = token.rsplit(".", 1)
    expected = hmac.new(secret.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8"))
        return {"username": str(data["u"]), "role": str(data["r"])}
    except Exception:
        return None


def admin_cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap and manage Annotator UI user accounts."
    )
    parser.add_argument(
        "--users-db",
        default=".mat_users.db",
        help="Path to SQLite users database (default: .mat_users.db).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create-user", help="Create a new user.")
    p_create.add_argument("--username", required=True)
    p_create.add_argument("--password", required=True)
    p_create.add_argument("--role", default="annotator", choices=VALID_ROLES)

    sub.add_parser("list-users", help="List all users.")

    args = parser.parse_args()
    store = UserStore(args.users_db)
    if args.cmd == "create-user":
        u = store.create_user(args.username, args.password, role=args.role)
        print(f"Created {u['role']} user '{u['username']}' in {store.db_path}.")
    elif args.cmd == "list-users":
        for u in store.list_users():
            status = "active" if u["active"] else "disabled"
            print(f"  {u['username']:20s}  role={u['role']:10s}  ({status})")
