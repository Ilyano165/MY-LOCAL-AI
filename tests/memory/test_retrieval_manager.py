"""MemoryManager: Routing, CRUD, Suche, Abruf für neue Aufgaben, Lebenszyklus."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory import (
    GLOBAL_SCOPE,
    MemoryContext,
    MemoryKind,
    MemoryLayer,
    MemoryManager,
    MemoryNotFoundError,
    MemorySource,
    MemoryStoreError,
)
from tests.memory.conftest import FakeClock

# --------------------------------------------------------------------------- Speichern


async def test_remember_routes_automatically(manager: MemoryManager, ctx: MemoryContext) -> None:
    pref = await manager.remember("Ich bevorzuge kurze Antworten", ctx)
    decision = await manager.remember("Wir verwenden FastAPI für die lokale API", ctx)
    chat = await manager.remember("Der Build ist gerade rot", ctx)
    trivial = await manager.remember("ok", ctx)
    secret = await manager.remember("Merk dir: api_key = sk-abcdefghijklmnopqrstuvwx", ctx)

    assert (pref.stored, pref.layer) == (True, MemoryLayer.LONG_TERM)
    assert (decision.stored, decision.layer) == (True, MemoryLayer.PROJECT)
    assert (chat.stored, chat.layer) == (True, MemoryLayer.SESSION)
    assert not trivial.stored and not secret.stored
    assert "Secret" in " ".join(secret.reasons)
    assert await manager.list(MemoryLayer.LONG_TERM, ctx) == [pref.item]


async def test_forced_layer_and_overrides(manager: MemoryManager, ctx: MemoryContext) -> None:
    result = await manager.remember(
        "Der Build ist gerade rot",
        ctx,
        layer=MemoryLayer.PROJECT,
        kind=MemoryKind.OBSERVATION,
        importance=0.7,
        tags=["ci"],
    )
    assert result.layer is MemoryLayer.PROJECT and result.item is not None
    assert (result.item.kind, result.item.importance, result.item.tags) == (
        MemoryKind.OBSERVATION,
        0.7,
        ["ci"],
    )
    secret = await manager.remember("passwort: hunter22", ctx, layer=MemoryLayer.PROJECT)
    assert not secret.stored  # auch erzwungen nie Secrets


async def test_missing_context_falls_back_or_refuses(manager: MemoryManager) -> None:
    session_only = MemoryContext(session_id="s")
    result = await manager.remember(
        "Merk dir: unser Projekt nutzt uv", session_only, layer=MemoryLayer.PROJECT
    )
    assert result.layer is MemoryLayer.SESSION and "kein Kontext für project" in result.reasons
    nothing = await manager.remember(
        "Wir verwenden SQLite", MemoryContext(), layer=MemoryLayer.PROJECT
    )
    assert not nothing.stored


async def test_duplicate_is_refreshed_not_duplicated(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    first = await manager.remember("Antworte immer auf Deutsch", ctx)
    second = await manager.remember("antworte immer auf deutsch", ctx)
    assert first.created and not second.created and second.item.id == first.item.id  # type: ignore[union-attr]
    assert len(await manager.list(MemoryLayer.LONG_TERM, ctx)) == 1


async def test_observe_message_logs_everything_but_promotes_selectively(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    messages = [
        "Hallo",
        "Wie spät ist es?",
        "Ab jetzt bitte immer mit Quellen antworten",
        "Der Kaffee ist alle",
        "Mein Passwort: hunter22",
    ]
    results = [await manager.observe_message("user", m, ctx) for m in messages]
    assert [r.stored for r in results] == [False, False, True, False, False]
    log = await manager.session.history(ctx.session_id or "")
    assert len(log) == 5 and "hunter22" not in " ".join(i.content for i in log)
    durable = await manager.list(MemoryLayer.LONG_TERM, ctx)
    assert [i.content for i in durable] == ["Ab jetzt bitte immer mit Quellen antworten"]
    assert await manager.list(MemoryLayer.PROJECT, ctx) == []


async def test_note_requires_task_context(manager: MemoryManager, ctx: MemoryContext) -> None:
    item = await manager.note("Zwischenergebnis: 3 Dateien betroffen", ctx)
    assert item.layer is MemoryLayer.WORKING and item.scope == ctx.task_id
    with pytest.raises(MemoryStoreError, match="Kontext"):
        await manager.note("x", MemoryContext(session_id="s"))


# --------------------------------------------------------------------------- Verwalten


async def test_get_update_delete_move_across_layers(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    note = await manager.note("Die Datei config.toml enthält die Ports", ctx)
    stored = await manager.remember("Ich bevorzuge Tabs", ctx)
    assert stored.item is not None

    assert await manager.get(note.id) == note
    updated = await manager.update(stored.item.id, content="Ich bevorzuge Spaces", importance=0.95)
    assert updated.content == "Ich bevorzuge Spaces" and updated.layer is MemoryLayer.LONG_TERM

    moved = await manager.move(note.id, MemoryLayer.PROJECT, ctx)
    assert moved.layer is MemoryLayer.PROJECT and await manager.get(note.id) is None

    assert await manager.delete(stored.item.id) is True
    assert await manager.delete(stored.item.id) is False
    with pytest.raises(MemoryNotFoundError):
        await manager.update("unbekannt", importance=0.1)
    with pytest.raises(MemoryNotFoundError):
        await manager.move("unbekannt", MemoryLayer.PROJECT, ctx)


async def test_search_has_no_side_effects(manager: MemoryManager, ctx: MemoryContext) -> None:
    stored = await manager.remember("Wir verwenden PostgreSQL in Produktion", ctx)
    hits = await manager.search("PostgreSQL", ctx)
    assert [h.item.id for h in hits] == [stored.item.id]  # type: ignore[union-attr]
    assert (await manager.get(stored.item.id)).access_count == 0  # type: ignore[union-attr]
    assert await manager.search("PostgreSQL", ctx, layer=MemoryLayer.LONG_TERM) == []


# --------------------------------------------------------------------------- Abruf


async def seed(manager: MemoryManager, ctx: MemoryContext) -> None:
    await manager.remember("Antworte immer auf Deutsch", ctx)
    await manager.remember("Wir verwenden SQLite mit FTS5 für die Volltextsuche", ctx)
    await manager.remember("Wir verwenden pytest mit asyncio_mode auto für Tests", ctx)
    await manager.remember("Merk dir: die Datenbank-Migrationen liegen in db/migrations", ctx)
    other = MemoryContext(session_id="anders", project_id="anderes-projekt")
    await manager.remember("Wir verwenden MongoDB als Datenbank", other)


async def test_recall_finds_relevant_memories_for_new_task(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    await seed(manager, ctx)
    result = await manager.recall(
        "Füge eine neue Datenbank-Migration für die Volltextsuche hinzu", ctx
    )
    contents = [m.item.content for m in result.memories]
    assert "Wir verwenden SQLite mit FTS5 für die Volltextsuche" in contents
    assert "die Datenbank-Migrationen liegen in db/migrations" in contents
    assert all("pytest" not in c for c in contents)  # irrelevant
    assert all("MongoDB" not in c for c in contents)  # anderes Projekt
    assert [g.content for g in result.guidance] == ["Antworte immer auf Deutsch"]
    prompt = result.as_prompt()
    assert "standing instructions" in prompt and "may be outdated" in prompt
    assert all(m.item.access_count == 1 for m in result.memories)  # Zugriff vermerkt


async def test_recall_without_matches_returns_only_guidance(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    await seed(manager, ctx)
    result = await manager.recall("Schreibe ein Gedicht über Herbstlaub", ctx)
    assert result.memories == [] and len(result.guidance) == 1
    assert (await manager.recall("Gedicht", ctx, include_guidance=False)).empty


async def test_recall_prefers_recent_and_important(
    manager: MemoryManager, ctx: MemoryContext, clock: FakeClock
) -> None:
    old = await manager.remember("Wir verwenden Docker für das Deployment", ctx, importance=0.7)
    clock.advance(days=120)
    new = await manager.remember("Wir verwenden Podman für das Deployment", ctx, importance=0.7)
    result = await manager.recall("Wie läuft das Deployment?", ctx)
    assert [m.item.id for m in result.memories] == [new.item.id, old.item.id]  # type: ignore[union-attr]
    assert result.memories[0].recency > result.memories[1].recency


async def test_recall_filters_layers_and_kinds(manager: MemoryManager, ctx: MemoryContext) -> None:
    await seed(manager, ctx)
    only_long = await manager.recall(
        "Volltextsuche Datenbank", ctx, layers=[MemoryLayer.LONG_TERM], include_guidance=False
    )
    assert only_long.memories == []
    decisions = await manager.recall("Volltextsuche Datenbank", ctx, kinds=[MemoryKind.DECISION])
    assert {m.item.kind for m in decisions.memories} == {MemoryKind.DECISION}


async def test_same_content_in_two_layers_is_returned_once(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    await manager.remember("Wir verwenden SQLite", ctx)  # Projekt
    await manager.note("Wir verwenden SQLite", ctx)  # Working
    result = await manager.recall("SQLite", ctx, include_guidance=False)
    # Einmal zurückgegeben – der höher bewertete Eintrag (Projektentscheidung, Wichtigkeit 0.65)
    assert len(result.memories) == 1 and result.memories[0].item.layer is MemoryLayer.PROJECT


# --------------------------------------------------------------------------- Lebenszyklus


async def test_record_outcome_only_keeps_lessons_from_verified_tasks(
    manager: MemoryManager, ctx: MemoryContext
) -> None:
    failed = await manager.record_outcome(
        "Migration bauen",
        "failed",
        "Tests rot",
        ctx,
        lessons=["Vermutung: Treiber fehlt"],
        verified=False,
    )
    assert [i.kind for i in failed] == [MemoryKind.TASK_RESULT]
    ok = await manager.record_outcome(
        "Migration bauen",
        "success",
        "Tests grün",
        ctx,
        lessons=["Migrationen brauchen eine eindeutige Versionsnummer"],
        verified=True,
    )
    assert [i.kind for i in ok] == [MemoryKind.TASK_RESULT, MemoryKind.LESSON]
    assert all(i.layer is MemoryLayer.PROJECT for i in ok)
    assert await manager.list(MemoryLayer.LONG_TERM, ctx) == []  # nie automatisch langfristig
    no_ctx = await manager.record_outcome("x", "success", "y", MemoryContext(), verified=True)
    assert no_ctx == []


async def test_end_task_and_maintenance(
    manager: MemoryManager, ctx: MemoryContext, clock: FakeClock
) -> None:
    await manager.note("temporär", ctx)
    await manager.remember("Der Build ist gerade rot", ctx)  # Session, TTL 7 Tage
    await manager.remember("Antworte immer auf Deutsch", ctx)  # Langzeit, kein Ablauf
    assert await manager.end_task(ctx) == 1
    assert await manager.end_task(MemoryContext()) == 0
    clock.advance(days=8)
    assert (await manager.maintenance())["expired"] >= 1
    assert await manager.list(MemoryLayer.SESSION, ctx) == []
    assert len(await manager.list(MemoryLayer.LONG_TERM, ctx)) == 1


async def test_persistent_manager_survives_restart(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "memory.db"
    ctx = MemoryContext(session_id="s", project_id="p")
    first = MemoryManager.open(path, clock=clock)
    await first.remember("Antworte immer auf Deutsch", ctx)
    await first.remember("Wir verwenden uv für Abhängigkeiten", ctx)
    await first.note("nur im Prozess", MemoryContext(task_id="t"))
    await first.close()

    second = MemoryManager.open(path, clock=clock)
    result = await second.recall("Abhängigkeiten installieren mit uv", ctx)
    assert [m.item.content for m in result.memories] == ["Wir verwenden uv für Abhängigkeiten"]
    assert [g.content for g in result.guidance] == ["Antworte immer auf Deutsch"]
    assert await second.list(MemoryLayer.WORKING, MemoryContext(task_id="t")) == []
    assert await second.list(MemoryLayer.LONG_TERM, ctx) != []
    await second.close()


def test_scope_mapping() -> None:
    ctx = MemoryContext(session_id="s", project_id="p", task_id="t")
    assert MemoryManager.scope_for(MemoryLayer.LONG_TERM, ctx) == GLOBAL_SCOPE
    assert MemoryManager.scope_for(MemoryLayer.WORKING, MemoryContext()) is None
    assert MemorySource.USER.value == "user"
