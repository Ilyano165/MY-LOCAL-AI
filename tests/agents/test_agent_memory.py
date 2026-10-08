"""Agent + Memory: automatischer Abruf bei neuen Aufgaben, Protokoll des Ergebnisses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agents.agent import Agent
from agents.state import InMemoryTaskStore
from agents.task import Constraints, FinalStatus
from memory import MemoryContext, MemoryKind, MemoryLayer, MemoryManager, MemoryStoreError
from tests.agents.fakes import ScriptedProvider, engine_for, plan_json
from tools import default_tools
from tools.registry import ToolRegistry

PYTEST = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]


@pytest.fixture
async def memory() -> AsyncIterator[MemoryManager]:
    m = MemoryManager.in_memory()
    yield m
    await m.close()


def build(tmp_path: Path, provider: ScriptedProvider, memory: MemoryManager) -> Agent:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return Agent(
        engine_for(provider),
        ToolRegistry(default_tools()),
        InMemoryTaskStore(),
        workspace,
        memory=memory,
        session_id="sess-1",
    )


async def test_relevant_memories_reach_the_model(tmp_path: Path, memory: MemoryManager) -> None:
    provider = ScriptedProvider(execute=["Erledigt."])
    agent = build(tmp_path, provider, memory)
    ctx = MemoryContext(session_id="sess-1", project_id=agent.project_id)
    await memory.remember("Antworte immer auf Deutsch", ctx)
    await memory.remember("Wir verwenden SQLite mit FTS5 für die Volltextsuche", ctx)
    await memory.remember("Wir verwenden React im Frontend", ctx)

    task = await agent.run("Was nutzen wir für die Volltextsuche?")

    contents = [m["content"] for m in task.recalled_memories]
    assert "Wir verwenden SQLite mit FTS5 für die Volltextsuche" in contents
    assert "Antworte immer auf Deutsch" in contents
    assert all("React" not in c for c in contents)
    prompt = provider.requests["execute"][0].messages[1].content
    assert "SQLite mit FTS5" in prompt and "Antworte immer auf Deutsch" in prompt
    assert any(o.source == "memory" for o in task.observations)


async def test_preference_in_task_is_remembered(tmp_path: Path, memory: MemoryManager) -> None:
    agent = build(tmp_path, ScriptedProvider(execute=["4"]), memory)
    await agent.run("Was ist 2+2? Und antworte ab jetzt immer auf Deutsch.")
    stored = await memory.list(MemoryLayer.LONG_TERM, MemoryContext())
    assert [i.kind for i in stored] == [MemoryKind.INSTRUCTION]
    history = await memory.session.history("sess-1")
    assert history[0].content.startswith("Was ist 2+2?")


async def test_plain_task_is_not_stored_durably(tmp_path: Path, memory: MemoryManager) -> None:
    agent = build(tmp_path, ScriptedProvider(execute=["4"]), memory)
    await agent.run("Was ist 2+2?")
    assert await memory.list(MemoryLayer.LONG_TERM, MemoryContext()) == []


async def test_verified_outcome_and_lesson_are_recorded(
    tmp_path: Path, memory: MemoryManager
) -> None:
    broken = "def add(a, b)\n    return a + b\n"
    fixed = "def add(a, b):\n    return a + b\n"
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {
                    "id": "s1",
                    "description": "add() implementieren",
                    "verification": [{"type": "command", "command": PYTEST}],
                },
                {"id": "s2", "description": "Fertig melden"},
            )
        ],
        execute=[
            [("write_file", {"path": "calc.py", "content": broken})],
            "fertig",
            [("write_file", {"path": "calc.py", "content": fixed})],
            "korrigiert",
            "gemeldet",
        ],
        analyze=["Nach der Funktionssignatur fehlte der Doppelpunkt."],
        final=["add() implementiert."],
    )
    agent = build(tmp_path, provider, memory)
    (agent.workspace / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    task = await agent.run(
        "Implementiere add in calc.py und dann teste es", Constraints(test_command=PYTEST)
    )
    assert task.final_result is not None
    # s2 hatte keine eigene Prüfung, aber die Gesamtprüfung (Syntax, Typen, Tests) bestätigt
    assert task.final_result.status == FinalStatus.SUCCESS
    assert task.verification_report is not None
    assert task.verification_report["strategy"] == "code"

    ctx = MemoryContext(project_id=agent.project_id)
    items = await memory.list(MemoryLayer.PROJECT, ctx)
    assert MemoryKind.TASK_RESULT in {i.kind for i in items}
    lessons = [i.content for i in items if i.kind == MemoryKind.LESSON]
    assert lessons == ["Nach der Funktionssignatur fehlte der Doppelpunkt."]
    assert await memory.list(MemoryLayer.WORKING, MemoryContext(task_id=task.id)) == []


async def test_memory_failure_does_not_break_agent(tmp_path: Path, memory: MemoryManager) -> None:
    async def broken(*args: object, **kwargs: object) -> None:
        raise MemoryStoreError("Datenbank gesperrt")

    memory.recall = broken  # type: ignore[method-assign]
    agent = build(tmp_path, ScriptedProvider(execute=["Antwort"]), memory)
    task = await agent.run("Erkläre kurz Rekursion")
    assert task.final_result is not None and task.final_result.answer == "Antwort"
    assert any(e.kind == "memory" and "gesperrt" in e.message for e in task.errors)


def test_project_id_defaults_to_workspace(tmp_path: Path, memory: MemoryManager) -> None:
    agent = build(tmp_path, ScriptedProvider(), memory)
    assert agent.project_id is not None and agent.project_id.startswith("ws-")
    no_memory = Agent(
        engine_for(ScriptedProvider()), ToolRegistry(), InMemoryTaskStore(), agent.workspace
    )
    assert no_memory.project_id is None


async def test_unverified_outcome_records_no_lesson(tmp_path: Path, memory: MemoryManager) -> None:
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {
                    "id": "s1",
                    "description": "notes.md anlegen",
                    "verification": [{"type": "file_contains", "path": "notes.md", "text": "Nova"}],
                },
                {"id": "s2", "description": "Fertig melden"},
            )
        ],
        execute=[
            [("write_file", {"path": "notes.md", "content": "leer"})],
            "fertig",
            [("write_file", {"path": "notes.md", "content": "Nova"})],
            "korrigiert",
            "gemeldet",
        ],
        analyze=["Das Wort Nova fehlte."],
        final=["Erledigt."],
    )
    agent = build(tmp_path, provider, memory)
    task = await agent.run(
        "Lege notes.md an und dann melde dich", Constraints(verification_strategy="none")
    )
    assert task.final_result is not None
    assert task.final_result.status == FinalStatus.UNVERIFIED  # s2 ohne Prüfung, Engine aus
    items = await memory.list(MemoryLayer.PROJECT, MemoryContext(project_id=agent.project_id))
    assert MemoryKind.LESSON not in {i.kind for i in items}
