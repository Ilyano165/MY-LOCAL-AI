"""Persistenter Memory-Store auf SQLite mit FTS5-Volltextindex.

Eine Datenbankdatei (Default ``~/.nova/memory.db``) hält Session-, Projekt- und
Langzeit-Erinnerungen. FTS5 liefert die Kandidaten-Vorauswahl (Präfixsuche, diakritik-
unempfindlich); das Ranking erfolgt einheitlich in :mod:`memory.base`.

SQLite-Aufrufe laufen in einem Worker-Thread (``asyncio.to_thread``), serialisiert über einen
Lock; WAL-Modus erlaubt parallele Leser anderer Prozesse.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from memory.base import (
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    MemoryNotFoundError,
    MemoryStore,
    MemoryStoreError,
    tokenize,
)

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    importance REAL NOT NULL,
    source TEXT NOT NULL,
    tags TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL,
    expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(layer, scope, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(layer, scope, content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories(expires_at);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED, content, tags, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
"""
SCHEMA_VERSION = 1


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] != SCHEMA_VERSION:
                raise MemoryStoreError(
                    f"Unbekannte Schema-Version {row['version']} (erwartet {SCHEMA_VERSION})"
                )

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def locked() -> T:
            with self._lock, self._conn:  # Transaktion: commit bei Erfolg, rollback bei Fehler
                return fn(self._conn)

        return await asyncio.to_thread(locked)

    # ------------------------------------------------------------------ Mapping

    @staticmethod
    def _row(item: MemoryItem) -> tuple[Any, ...]:
        return (
            item.id,
            item.layer.value,
            item.scope,
            item.kind.value,
            item.content,
            item.content_hash,
            item.importance,
            item.source.value,
            json.dumps(item.tags, ensure_ascii=False),
            json.dumps(item.metadata, ensure_ascii=False),
            item.created_at.isoformat(),
            item.updated_at.isoformat(),
            item.last_accessed_at.isoformat(),
            item.access_count,
            item.expires_at.isoformat() if item.expires_at else None,
        )

    @staticmethod
    def _item(row: sqlite3.Row) -> MemoryItem:
        data = dict(row)
        data["tags"] = json.loads(data["tags"])
        data["metadata"] = json.loads(data["metadata"])
        return MemoryItem.from_dict(data)

    @staticmethod
    def _index(conn: sqlite3.Connection, item: MemoryItem) -> None:
        conn.execute("DELETE FROM memories_fts WHERE id = ?", (item.id,))
        conn.execute(
            "INSERT INTO memories_fts (id, content, tags) VALUES (?, ?, ?)",
            (item.id, item.content, " ".join(item.tags)),
        )

    @staticmethod
    def _filters(
        layer: MemoryLayer, scope: str | None, kinds: Iterable[MemoryKind] | None
    ) -> tuple[str, list[Any]]:
        clauses, params = ["m.layer = ?"], [layer.value]
        if scope is not None:
            clauses.append("m.scope = ?")
            params.append(scope)
        if kinds is not None:
            kind_list = [MemoryKind(k).value for k in kinds]
            if not kind_list:
                clauses.append("0")
            else:
                clauses.append(f"m.kind IN ({', '.join('?' for _ in kind_list)})")
                params += kind_list
        return " AND ".join(clauses), params

    # ------------------------------------------------------------------ API

    async def add(self, item: MemoryItem) -> MemoryItem:
        def op(conn: sqlite3.Connection) -> None:
            try:
                conn.execute(
                    f"INSERT INTO memories VALUES ({', '.join('?' * 15)})", self._row(item)
                )
            except sqlite3.IntegrityError as exc:
                raise MemoryStoreError(f"Memory {item.id} existiert bereits") from exc
            self._index(conn, item)

        await self._run(op)
        return item

    async def get(self, item_id: str) -> MemoryItem | None:
        def op(conn: sqlite3.Connection) -> MemoryItem | None:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
            return self._item(row) if row else None

        return await self._run(op)

    async def save(self, item: MemoryItem) -> MemoryItem:
        def op(conn: sqlite3.Connection) -> None:
            row = self._row(item)
            cur = conn.execute(
                "UPDATE memories SET layer=?, scope=?, kind=?, content=?, content_hash=?, "
                "importance=?, source=?, tags=?, metadata=?, created_at=?, updated_at=?, "
                "last_accessed_at=?, access_count=?, expires_at=? WHERE id=?",
                (*row[1:], item.id),
            )
            if cur.rowcount == 0:
                raise MemoryNotFoundError(item.id)
            self._index(conn, item)

        await self._run(op)
        return item

    async def delete(self, item_id: str) -> bool:
        def op(conn: sqlite3.Connection) -> bool:
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (item_id,))
            return conn.execute("DELETE FROM memories WHERE id = ?", (item_id,)).rowcount > 0

        return await self._run(op)

    async def list(
        self,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        where, params = self._filters(layer, scope, kinds)
        sql = f"SELECT * FROM memories m WHERE {where} ORDER BY m.created_at DESC, m.rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            return [self._item(r) for r in conn.execute(sql, params)]

        return await self._run(op)

    async def candidates(
        self,
        query: str,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 200,
    ) -> builtins.list[MemoryItem]:
        tokens = list(dict.fromkeys(tokenize(query)))
        if not tokens:
            return []
        # Tokens enthalten nur Wortzeichen und -./ → sicher in Anführungszeichen; Präfixsuche
        # findet Komposita („datenbank“ → „datenbankschema“).
        match = " OR ".join(f'"{t}"*' if len(t) >= 4 else f'"{t}"' for t in tokens)
        where, params = self._filters(layer, scope, kinds)
        sql = (
            "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.id "
            f"WHERE memories_fts MATCH ? AND {where} ORDER BY bm25(memories_fts) LIMIT ?"
        )

        def op(conn: sqlite3.Connection) -> list[MemoryItem]:
            return [self._item(r) for r in conn.execute(sql, [match, *params, limit])]

        return await self._run(op)

    async def find_by_hash(self, layer: MemoryLayer, scope: str, digest: str) -> MemoryItem | None:
        def op(conn: sqlite3.Connection) -> MemoryItem | None:
            row = conn.execute(
                "SELECT * FROM memories WHERE layer=? AND scope=? AND content_hash=? LIMIT 1",
                (layer.value, scope, digest),
            ).fetchone()
            return self._item(row) if row else None

        return await self._run(op)

    async def delete_expired(self, now: datetime) -> int:
        def op(conn: sqlite3.Connection) -> int:
            ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now.isoformat(),),
                )
            ]
            for item_id in ids:
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (item_id,))
                conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
            return len(ids)

        return await self._run(op)

    async def count(self, layer: MemoryLayer, scope: str | None = None) -> int:
        where, params = self._filters(layer, scope, None)

        def op(conn: sqlite3.Connection) -> int:
            return int(
                conn.execute(f"SELECT COUNT(*) FROM memories m WHERE {where}", params).fetchone()[0]
            )

        return await self._run(op)

    async def close(self) -> None:
        with self._lock:
            self._conn.close()
