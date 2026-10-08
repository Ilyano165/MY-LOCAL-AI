from __future__ import annotations

from pathlib import Path

import pytest

from memory import (
    GLOBAL_SCOPE,
    InMemoryStore,
    LongTermMemory,
    MemoryKind,
    MemoryNotFoundError,
    MemorySource,
    MemoryStoreError,
    ProjectMemory,
    SessionMemory,
    WorkingMemory,
    project_id_for,
)
from tests.memory.conftest import FakeClock

# --------------------------------------------------------------------------- gemeinsame Operationen


async def test_add_deduplicates_and_refreshes(clock: FakeClock) -> None:
    mem = ProjectMemory(InMemoryStore(), clock=clock)
    first, created = await mem.add("Wir nutzen SQLite", "p1", importance=0.5, tags=["db"])
    clock.advance(hours=2)
    again, created_again = await mem.add(
        "wir  nutzen sqlite", "p1", importance=0.8, tags=["persistenz"]
    )
    assert created and not created_again and again.id == first.id
    assert again.importance == 0.8 and again.tags == ["db", "persistenz"]
    assert again.access_count == 1 and again.last_accessed_at == clock.now
    _, other_scope = await mem.add("Wir nutzen SQLite", "p2")
    assert other_scope  # anderer Scope → eigener Eintrag


async def test_update_get_delete(clock: FakeClock) -> None:
    mem = LongTermMemory(InMemoryStore(), clock=clock)
    item, _ = await mem.add("Ich mag kurze Antworten", GLOBAL_SCOPE, kind=MemoryKind.PREFERENCE)
    clock.advance(minutes=5)
    updated = await mem.update(
        item.id,
        content="Ich mag sehr kurze Antworten",
        importance=0.9,
        tags=["stil"],
        metadata={"quelle": "chat"},
    )
    assert updated.content == "Ich mag sehr kurze Antworten" and updated.updated_at == clock.now
    assert (await mem.get(item.id)) == updated
    with pytest.raises(ValueError):
        await mem.update(item.id, importance=2.0)
    with pytest.raises(MemoryStoreError, match="Secret"):
        await mem.update(item.id, content="passwort: hunter22")
    with pytest.raises(MemoryNotFoundError):
        await mem.update("gibt-es-nicht", importance=0.1)
    assert await mem.delete(item.id) is True
    assert await mem.get(item.id) is None


async def test_layers_do_not_see_each_others_items(clock: FakeClock) -> None:
    store = InMemoryStore()
    project = ProjectMemory(store, clock=clock)
    long_term = LongTermMemory(store, clock=clock)
    item, _ = await project.add("Projektwissen", "p1")
    assert await long_term.get(item.id) is None
    assert await long_term.delete(item.id) is False
    assert await project.get(item.id) is not None


async def test_secrets_are_refused(clock: FakeClock) -> None:
    mem = LongTermMemory(InMemoryStore(), clock=clock)
    with pytest.raises(MemoryStoreError, match="Secret"):
        await mem.add("Mein Token: abcd1234efgh5678", GLOBAL_SCOPE)
    assert await mem.list(GLOBAL_SCOPE) == []


async def test_search_ranks_by_relevance_importance_recency(clock: FakeClock) -> None:
    mem = ProjectMemory(InMemoryStore(), clock=clock)
    await mem.add("Die API nutzt FastAPI und läuft lokal", "p1", importance=0.3)
    await mem.add("Die API Authentifizierung nutzt Tokens aus der Umgebung", "p1", importance=0.9)
    await mem.add("Das Frontend nutzt React", "p1", importance=1.0)
    hits = await mem.search("API Authentifizierung", "p1")
    assert [h.item.content for h in hits][:2] == [
        "Die API Authentifizierung nutzt Tokens aus der Umgebung",
        "Die API nutzt FastAPI und läuft lokal",
    ]
    assert all("React" not in h.item.content for h in hits)  # irrelevant trotz hoher Wichtigkeit
    assert hits[0].relevance == 1.0


async def test_recency_decays_with_time_since_access(clock: FakeClock) -> None:
    mem = ProjectMemory(InMemoryStore(), clock=clock, half_life_hours=24)
    await mem.add("Build läuft über make", "p1")
    clock.advance(hours=24)
    hit = (await mem.search("make build", "p1"))[0]
    assert hit.recency == pytest.approx(0.5)


async def test_expired_items_are_invisible(clock: FakeClock) -> None:
    mem = SessionMemory(InMemoryStore(), clock=clock)
    item, _ = await mem.add("Kurzlebige Notiz über Docker", "s1")
    clock.advance(days=8)  # TTL 7 Tage
    assert await mem.get(item.id) is None
    assert await mem.list("s1") == []
    assert await mem.search("docker", "s1") == []


async def test_prune_keeps_most_valuable(clock: FakeClock) -> None:
    mem = LongTermMemory(InMemoryStore(), clock=clock, capacity=3)
    keep, _ = await mem.add("wichtig eins", GLOBAL_SCOPE, importance=0.9)
    await mem.add("unwichtig alt", GLOBAL_SCOPE, importance=0.1)
    clock.advance(days=1)
    await mem.add("wichtig zwei", GLOBAL_SCOPE, importance=0.8)
    await mem.add("mittel neu", GLOBAL_SCOPE, importance=0.5)
    contents = {i.content for i in await mem.list(GLOBAL_SCOPE)}
    assert contents == {"wichtig eins", "wichtig zwei", "mittel neu"}
    assert await mem.get(keep.id) is not None


async def test_frequently_used_items_survive_pruning(clock: FakeClock) -> None:
    mem = ProjectMemory(InMemoryStore(), clock=clock, capacity=1)
    used, _ = await mem.add("oft genutzt", "p1", importance=0.4)
    for _ in range(20):
        await mem.add("oft genutzt", "p1", importance=0.4)  # Dedupe zählt als Zugriff
    await mem.add("neu aber gleich wichtig", "p1", importance=0.4)
    assert [i.id for i in await mem.list("p1")] == [used.id]


# --------------------------------------------------------------------------- schichtspezifisch


async def test_working_memory_is_in_process_and_clearable(clock: FakeClock) -> None:
    working = WorkingMemory(clock=clock)
    assert isinstance(working.store, InMemoryStore) and working.capacity == 100
    await working.add("Zwischenergebnis A", "task-1")
    await working.add("Zwischenergebnis B", "task-2")
    assert await working.clear("task-1") == 1
    assert [i.content for i in await working.list("task-2")] == ["Zwischenergebnis B"]


async def test_session_messages_history_and_redaction(clock: FakeClock) -> None:
    session = SessionMemory(InMemoryStore(), clock=clock)
    for text in ("Hallo", "Mein Passwort: hunter22 bitte nicht speichern", "Hallo"):
        clock.advance(seconds=1)
        await session.add_message("s1", "user", text)
    clock.advance(seconds=1)
    await session.add_message("s1", "assistant", "Verstanden.")
    assert await session.add_message("s1", "user", "   ") is None
    history = await session.history("s1")
    assert [h.content for h in history] == [
        "Hallo",
        "Mein [REDACTED] bitte nicht speichern",
        "Hallo",
        "Verstanden.",
    ]
    assert history[-1].source is MemorySource.AGENT and history[0].metadata["role"] == "user"
    assert [h.content for h in await session.history("s1", limit=2)] == ["Hallo", "Verstanden."]


def test_project_id_is_stable_and_path_specific(tmp_path: Path) -> None:
    a = tmp_path / "Mein Projekt"
    a.mkdir()
    pid = project_id_for(a)
    assert pid.startswith("mein-projekt-") and pid == project_id_for(str(a) + "/")
    other = tmp_path / "x" / "Mein Projekt"
    other.mkdir(parents=True)
    assert project_id_for(other) != pid


async def test_long_term_active_guidance(clock: FakeClock) -> None:
    lt = LongTermMemory(InMemoryStore(), clock=clock)
    await lt.add("Antworte auf Deutsch", GLOBAL_SCOPE, kind=MemoryKind.INSTRUCTION, importance=0.8)
    await lt.add("Ich mag Tabs", GLOBAL_SCOPE, kind=MemoryKind.PREFERENCE, importance=0.9)
    await lt.add("Berlin ist die Hauptstadt", GLOBAL_SCOPE, kind=MemoryKind.FACT, importance=1.0)
    guidance = await lt.active_guidance()
    assert [g.content for g in guidance] == ["Ich mag Tabs", "Antworte auf Deutsch"]
