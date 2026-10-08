"""Vertragstests: jedes MemoryStore-Backend muss sich identisch verhalten."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory.base import (
    InMemoryStore,
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    MemoryNotFoundError,
    MemoryStore,
    MemoryStoreError,
    content_hash,
)
from memory.sqlite_store import SQLiteMemoryStore

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def item(
    content: str,
    *,
    layer: MemoryLayer = MemoryLayer.PROJECT,
    scope: str = "p1",
    kind: MemoryKind = MemoryKind.FACT,
    minutes: int = 0,
    **kw: object,
) -> MemoryItem:
    t = T0 + timedelta(minutes=minutes)
    return MemoryItem(
        content, layer, scope, kind=kind, created_at=t, updated_at=t, last_accessed_at=t, **kw
    )  # type: ignore[arg-type]


@pytest.fixture(params=["memory", "sqlite"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[MemoryStore]:
    s: MemoryStore = (
        InMemoryStore() if request.param == "memory" else SQLiteMemoryStore(tmp_path / "m.db")
    )
    yield s
    await s.close()


async def test_add_get_save_delete(store: MemoryStore) -> None:
    a = await store.add(item("Wir nutzen SQLite", tags=["db"], metadata={"k": 1}))
    loaded = await store.get(a.id)
    assert loaded == a
    loaded.content = "Wir nutzen PostgreSQL"
    loaded.importance = 0.9
    await store.save(loaded)
    assert (await store.get(a.id)).content == "Wir nutzen PostgreSQL"  # type: ignore[union-attr]
    assert await store.delete(a.id) is True
    assert await store.get(a.id) is None
    assert await store.delete(a.id) is False


async def test_errors(store: MemoryStore) -> None:
    a = await store.add(item("x y"))
    with pytest.raises(MemoryStoreError):
        await store.add(a)
    with pytest.raises(MemoryNotFoundError):
        await store.save(item("nie gespeichert"))


async def test_returned_items_are_copies(store: MemoryStore) -> None:
    a = await store.add(item("original text"))
    copy = await store.get(a.id)
    assert copy is not None
    copy.content = "lokal verändert"
    assert (await store.get(a.id)).content == "original text"  # type: ignore[union-attr]


async def test_list_filters_and_order(store: MemoryStore) -> None:
    await store.add(item("alt", minutes=1))
    await store.add(item("neu", minutes=2, kind=MemoryKind.DECISION))
    await store.add(item("anderes projekt", scope="p2"))
    await store.add(item("session", layer=MemoryLayer.SESSION, scope="s1"))
    assert [i.content for i in await store.list(MemoryLayer.PROJECT, "p1")] == ["neu", "alt"]
    assert [
        i.content for i in await store.list(MemoryLayer.PROJECT, "p1", kinds=[MemoryKind.DECISION])
    ] == ["neu"]
    assert await store.list(MemoryLayer.PROJECT, "p1", kinds=[]) == []
    assert len(await store.list(MemoryLayer.PROJECT)) == 3
    assert [i.content for i in await store.list(MemoryLayer.PROJECT, "p1", limit=1)] == ["neu"]
    assert await store.count(MemoryLayer.PROJECT, "p1") == 2
    assert await store.count(MemoryLayer.SESSION) == 1


async def test_candidates_find_terms_prefixes_and_tags(store: MemoryStore) -> None:
    await store.add(item("Das Datenbankschema liegt in schema.sql"))
    await store.add(item("Deployment über Docker", tags=["infrastruktur"]))
    await store.add(item("Größe der Bilder begrenzen"))
    await store.add(item("Datenbank in anderem Projekt", scope="p2"))

    def contents(items: list[MemoryItem]) -> set[str]:
        return {i.content for i in items}

    assert contents(await store.candidates("Datenbank", MemoryLayer.PROJECT, "p1")) == {
        "Das Datenbankschema liegt in schema.sql"
    }
    assert contents(await store.candidates("Infrastruktur?", MemoryLayer.PROJECT, "p1")) == {
        "Deployment über Docker"
    }
    assert await store.candidates("und oder", MemoryLayer.PROJECT, "p1") == []
    assert len(await store.candidates("datenbank", MemoryLayer.PROJECT)) == 2


async def test_find_by_hash_and_expiry(store: MemoryStore) -> None:
    a = await store.add(item("Wir nutzen SQLite", expires_at=T0 + timedelta(days=1)))
    await store.add(item("bleibt"))
    found = await store.find_by_hash(MemoryLayer.PROJECT, "p1", content_hash("wir  NUTZEN sqlite"))
    assert found is not None and found.id == a.id
    assert await store.find_by_hash(MemoryLayer.PROJECT, "p2", a.content_hash) is None
    assert await store.delete_expired(T0) == 0
    assert await store.delete_expired(T0 + timedelta(days=2)) == 1
    assert [i.content for i in await store.list(MemoryLayer.PROJECT)] == ["bleibt"]


async def test_touch_updates_access(store: MemoryStore) -> None:
    a = await store.add(item("x y z"))
    later = T0 + timedelta(hours=5)
    await store.touch([a], later)
    loaded = await store.get(a.id)
    assert loaded is not None and loaded.access_count == 1 and loaded.last_accessed_at == later


async def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.db"
    first = SQLiteMemoryStore(path)
    a = await first.add(item("Persistente Erinnerung über Datenbanken"))
    await first.close()
    second = SQLiteMemoryStore(path)
    assert (await second.get(a.id)) == a
    assert len(await second.candidates("datenbanken", MemoryLayer.PROJECT)) == 1
    await second.close()


async def test_sqlite_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "v.db"
    await SQLiteMemoryStore(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE schema_version SET version = 99")
    with pytest.raises(MemoryStoreError, match="Schema-Version"):
        SQLiteMemoryStore(path)


async def test_sqlite_fts_query_is_injection_safe(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "f.db")
    await store.add(item("normaler Inhalt"))
    for query in [
        '"',
        "inhalt OR 1=1",
        "NEAR(inhalt",
        "*",
        "inhalt) --",
        "'; DROP TABLE memories;",
    ]:
        await store.candidates(query, MemoryLayer.PROJECT)
    assert await store.count(MemoryLayer.PROJECT) == 1
    await store.close()
