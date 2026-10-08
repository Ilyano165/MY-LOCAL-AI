"""Gesprächsverlauf (SQLite, ``<data_dir>/conversations.db``).

Nachrichten speichern zusätzlich ``meta`` (Modell, Routing, Messwerte – nur was tatsächlich
vorlag) und Anhänge (Text-Inhalt bzw. Bild-Datei im Upload-Verzeichnis).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    attachments TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class NotFoundError(KeyError):
    pass


@dataclass
class StoredMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_attachment_text: bool = False) -> dict[str, Any]:
        attachments = [
            {k: v for k, v in a.items() if include_attachment_text or k != "text"}
            for a in self.attachments
        ]
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "meta": self.meta,
            "attachments": attachments,
        }


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ Gespräche

    def create(self, title: str = "Neuer Chat") -> dict[str, Any]:
        cid = uuid.uuid4().hex[:16]
        now = _now()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)", (cid, title[:120], now, now)
            )
        return {"id": cid, "title": title[:120], "created_at": now, "updated_at": now}

    def list_all(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) "
                "AS message_count FROM conversations c ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, cid: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        if row is None:
            raise NotFoundError(cid)
        return dict(row)

    def rename(self, cid: str, title: str) -> dict[str, Any]:
        title = title.strip()[:120]
        if not title:
            raise ValueError("Titel darf nicht leer sein")
        self.get(cid)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), cid),
            )
        return self.get(cid)

    def delete(self, cid: str) -> None:
        self.get(cid)
        with self._lock, self._db:
            self._db.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    # ------------------------------------------------------------------ Nachrichten

    def add_message(
        self,
        cid: str,
        role: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> StoredMessage:
        self.get(cid)
        msg = StoredMessage(
            uuid.uuid4().hex[:16], cid, role, content, _now(), meta or {}, attachments or []
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    msg.id,
                    cid,
                    role,
                    content,
                    msg.created_at,
                    json.dumps(msg.meta),
                    json.dumps(msg.attachments),
                ),
            )
            self._db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (msg.created_at, cid)
            )
        return msg

    def messages(self, cid: str) -> list[StoredMessage]:
        self.get(cid)
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                (cid,),
            ).fetchall()
        return [
            StoredMessage(
                r["id"],
                r["conversation_id"],
                r["role"],
                r["content"],
                r["created_at"],
                json.loads(r["meta"]),
                json.loads(r["attachments"]),
            )
            for r in rows
        ]
