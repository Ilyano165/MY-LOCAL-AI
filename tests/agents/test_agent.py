"""End-to-End-Tests des Agent-Loops (Modell gescriptet, Tools/Verifier/Persistenz echt)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent import Agent
from agents.state import JsonFileTaskStore
from agents.task import Constraints, FinalStatus, Phase, SubtaskStatus, Task, TaskStatus, Verdict
from models.base import ChatRequest
from tests.agents.fakes import ScriptedProvider, engine_for, plan_json
from tools import default_tools
from tools.registry import ToolRegistry

PYTEST = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
BROKEN = "def add(a, b)\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def build(tmp_path: Path, provider: ScriptedProvider) -> tuple[Agent, Path, list[tuple[str, str]]]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    events: list[tuple[str, str]] = []
    agent = Agent(
        engine_for(provider),
        ToolRegistry(default_tools()),
        JsonFileTaskStore(tmp_path / "state"),
        workspace,
        on_event=lambda task, phase, msg: events.append((phase.value, msg)),
    )
    return agent, workspace, events


async def test_simple_question_is_answered_but_marked_unverified(tmp_path: Path) -> None:
    provider = ScriptedProvider(execute=["4"])
    agent, _, events = build(tmp_path, provider)
    task = await agent.run("Was ist 2+2?")

    assert task.status == TaskStatus.COMPLETED
    assert task.final_result is not None
    assert task.final_result.status == FinalStatus.UNVERIFIED
    assert task.final_result.verified is False
    assert task.final_result.answer == "4"
    assert "plan" not in provider.requests  # einfacher Task: kein Planungsaufruf
    assert provider.requests["execute"][0].tools == ()  # Wissensfrage: keine Tools angeboten
    assert [p for p, _ in events][:2] == ["analyze", "plan"]
    assert events[-1][0] == "finalize"


async def test_code_task_full_loop_with_correction(tmp_path: Path) -> None:
    """Syntaxfehler → Korrektur → Testfehler → Korrektur → Tests grün → SUCCESS."""
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {
                    "id": "s1",
                    "description": "README.md mit Kurzbeschreibung anlegen",
                    "verification": [{"type": "file_contains", "path": "README.md", "text": "add"}],
                },
                {
                    "id": "s2",
                    "description": "add() in calc.py implementieren",
                    "depends_on": ["s1"],
                    "verification": [{"type": "command", "command": PYTEST}],
                },
            )
        ],
        execute=[
            [("write_file", {"path": "README.md", "content": "calc: add(a, b)"})],
            "README angelegt.",
            [("write_file", {"path": "calc.py", "content": BROKEN})],
            "calc.py fertig.",
            [("write_file", {"path": "calc.py", "content": WRONG})],
            "Syntax korrigiert.",
            [("write_file", {"path": "calc.py", "content": FIXED})],
            "Vorzeichen korrigiert.",
        ],
        analyze=["Doppelpunkt fehlt nach der Signatur.", "add subtrahiert statt zu addieren."],
        final=["calc.py implementiert, Tests laufen grün."],
    )
    agent, workspace, _ = build(tmp_path, provider)
    (workspace / "test_calc.py").write_text(TEST)  # vorhandene Tests im Repo
    task = await agent.run(
        "Schreibe eine Funktion add in calc.py und teste sie mit pytest",
        Constraints(test_command=PYTEST),
    )

    assert task.final_result is not None, task.errors
    assert task.final_result.status == FinalStatus.SUCCESS, task.final_result.summary
    assert task.final_result.verified
    assert task.status == TaskStatus.COMPLETED
    assert (workspace / "calc.py").read_text() == FIXED
    s2 = task.subtask("s2")
    assert s2.attempts == 3 and s2.verdict == Verdict.PASSED and len(s2.feedback) == 2

    # Fehler wurden erkannt, nicht übergangen
    failed = [
        (r.attempt, r.check)
        for r in task.verification_results
        if r.subtask_id == "s2" and r.verdict == Verdict.FAILED
    ]
    assert (1, "python_syntax(calc.py)") in failed
    assert any(a == 2 and c.startswith("test_command") for a, c in failed)
    assert [e.analysis for e in task.errors if e.phase == Phase.CORRECT] == [
        "Doppelpunkt fehlt nach der Signatur.",
        "add subtrahiert statt zu addieren.",
    ]
    # Korrekturhinweis landete im Prompt des nächsten Versuchs
    third_attempt_prompt = provider.requests["execute"][6].messages[1].content
    assert "add subtrahiert statt zu addieren" in third_attempt_prompt
    assert provider.remaining() == {}


async def test_persistent_tool_failure_leads_to_replan_and_honest_failure(tmp_path: Path) -> None:
    escape = [("write_file", {"path": "../outside.txt", "content": "x"})]
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {"id": "s1", "description": "Datei außerhalb schreiben"},
                {"id": "s2", "description": "Bericht", "depends_on": ["s1"]},
            ),
            plan_json({"id": "alt", "description": "Anderer Weg"}),
        ],
        # s1: 2 Versuche, danach Neuplanung → r1-alt: 2 Versuche
        execute=[escape, "Erledigt!"] * 4,
        analyze=["Pfad liegt außerhalb."] * 4,
        final=["Die Aufgabe konnte nicht erledigt werden."],
    )
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run(
        "Schreibe die Datei outside.txt und dann einen Bericht",
        Constraints(max_attempts_per_subtask=2, max_replans=1),
    )

    assert task.final_result is not None
    assert task.final_result.status == FinalStatus.FAILED
    assert task.final_result.verified is False
    assert task.status == TaskStatus.FAILED
    assert task.replans == 1 and task.plan_version == 2
    assert [s.id for s in task.subtasks] == ["s1", "r1-alt"]
    assert task.subtask("s1").superseded and task.subtask("s1").status == SubtaskStatus.FAILED
    assert task.subtask("r1-alt").status == SubtaskStatus.FAILED
    assert task.final_result.failed_subtasks == ["r1-alt"]
    tool_errors = [e for e in task.errors if e.kind == "tool"]
    assert len(tool_errors) == 4 and all("außerhalb" in e.message for e in tool_errors)
    # Das Modell behauptete "Erledigt!" – der Verifier hat es trotzdem abgelehnt
    assert all(s.verdict == Verdict.FAILED for s in task.subtasks)
    assert provider.remaining() == {}


async def test_dependent_subtasks_are_skipped_and_result_is_partial(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {
                    "id": "s1",
                    "description": "a.txt anlegen",
                    "verification": [{"type": "file_exists", "path": "a.txt"}],
                },
                {"id": "s2", "description": "b.txt anlegen", "depends_on": ["s1"]},
                {
                    "id": "s3",
                    "description": "c.txt anlegen",
                    "verification": [{"type": "file_exists", "path": "c.txt"}],
                },
            )
        ],
        execute=["vergessen", [("write_file", {"path": "c.txt", "content": "c"})], "c fertig"],
        analyze=["Datei wurde nie geschrieben."],
        final=["Teilweise erledigt."],
    )
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run(
        "Lege a.txt, b.txt und c.txt an.\n- a\n- b\n- c",
        Constraints(max_attempts_per_subtask=1, max_replans=0),
    )
    assert task.final_result is not None
    assert task.final_result.status == FinalStatus.PARTIAL
    assert [s.status for s in task.subtasks] == [
        SubtaskStatus.FAILED,
        SubtaskStatus.SKIPPED,
        SubtaskStatus.DONE,
    ]
    assert task.final_result.failed_subtasks == ["s1", "s2"]
    assert task.status == TaskStatus.FAILED


async def test_step_budget_aborts(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {
                    "id": "s1",
                    "description": "x.txt anlegen",
                    "verification": [{"type": "file_exists", "path": "x.txt"}],
                }
            )
        ],
        execute=["erledigt (gelogen)"] * 2,
        analyze=["Datei fehlt."] * 2,
        final=["Abgebrochen."],
    )
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run(
        "Lege x.txt an und dann prüfe sie", Constraints(max_steps=2, max_attempts_per_subtask=5)
    )

    assert task.final_result is not None
    assert task.final_result.status == FinalStatus.ABORTED
    assert task.status == TaskStatus.ABORTED
    assert task.steps_taken == 2
    assert task.errors[-1].kind == "budget"
    assert task.final_result.failed_subtasks == ["s1"]


async def test_state_is_persisted_and_resumable(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {"id": "s1", "description": "a.txt anlegen"},
                {"id": "s2", "description": "b.txt anlegen"},
            )
        ],
        execute=[[("write_file", {"path": "a.txt", "content": "a"})], "a fertig"],
    )
    agent, workspace, _ = build(tmp_path, provider)

    class Crash(Exception):
        pass

    original = agent.executor.execute
    calls = 0

    async def crash_on_second(task: Task, sub: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Crash("Stromausfall")
        return await original(task, sub)  # type: ignore[arg-type]

    agent.executor.execute = crash_on_second  # type: ignore[method-assign]
    with pytest.raises(Crash):
        await agent.run("Lege a.txt an und dann b.txt")

    task_id = (await agent.store.list_ids())[0]
    saved = await agent.store.load(task_id)
    assert saved.subtask("s1").status == SubtaskStatus.DONE
    assert saved.status == TaskStatus.FAILED and saved.errors[-1].kind == "internal"

    # Wiederaufnahme: nach dem Absturz neu laden (Status zurück auf RUNNING setzen)
    saved.status = TaskStatus.RUNNING
    saved.final_result = None
    await agent.store.save(saved)
    agent.executor.execute = original  # type: ignore[method-assign]
    provider.scripts["execute"] = [[("write_file", {"path": "b.txt", "content": "b"})], "b fertig"]
    provider.scripts["final"] = ["a.txt und b.txt angelegt."]
    resumed = await agent.resume(task_id)

    assert resumed.final_result is not None
    assert resumed.final_result.status == FinalStatus.SUCCESS
    assert (workspace / "b.txt").read_text() == "b"
    assert resumed.subtask("s1").attempts == 1  # nicht erneut ausgeführt
    assert resumed.subtask("s2").attempts == 2  # abgebrochener Versuch zählt


async def test_resume_of_finished_task_is_noop(tmp_path: Path) -> None:
    provider = ScriptedProvider(execute=["4"])
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run("Was ist 2+2?")
    again = await agent.resume(task.id)
    assert again.to_dict() == task.to_dict()


async def test_planning_failure_falls_back_to_single_step(tmp_path: Path) -> None:
    provider = ScriptedProvider(plan=["kein plan", "wieder nichts"], execute=["Antwort"])
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run("Schreibe einen Plan für meine Woche und dann eine Einkaufsliste")
    assert [s.description for s in task.subtasks] == [task.objective]
    assert task.errors[0].kind == "planning"
    assert task.final_result is not None and task.final_result.status == FinalStatus.UNVERIFIED


async def test_final_answer_prompt_carries_authoritative_status(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        plan=[plan_json({"id": "s1", "description": "a"}, {"id": "s2", "description": "b"})],
        execute=["A", "B"],
        final=["Zusammenfassung"],
    )
    agent, _, _ = build(tmp_path, provider)
    task = await agent.run("Mach a und dann b")
    request: ChatRequest = provider.requests["final"][0]
    assert "never claim more success" in request.messages[0].content
    assert "Status: unverified" in request.messages[1].content
    assert task.final_result is not None and task.final_result.answer == "Zusammenfassung"


def test_workspace_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Agent(
            engine_for(ScriptedProvider()),
            ToolRegistry(),
            JsonFileTaskStore(tmp_path),
            tmp_path / "nope",
        )
