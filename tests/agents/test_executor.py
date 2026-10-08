from __future__ import annotations

from pathlib import Path

from agents.executor import ExecutionOutcome, Executor, FailureAnalyzer, reset_interrupted
from agents.task import (
    Constraints,
    Phase,
    Subtask,
    SubtaskStatus,
    Task,
    Verdict,
    VerificationRecord,
)
from models.base import ProviderUnavailableError
from tests.agents.fakes import ScriptedProvider, engine_for, tool_messages
from tools.base import ToolRegistry
from tools.filesystem import default_tools


def make(
    tmp_path: Path, provider: ScriptedProvider, **constraints: object
) -> tuple[Executor, Task, Subtask]:
    executor = Executor(engine_for(provider), ToolRegistry(default_tools()), tmp_path)
    task = Task("Schreibe hello.txt", Constraints(**constraints))  # type: ignore[arg-type]
    sub = Subtask("s1", "hello.txt mit Inhalt 'hi' anlegen", attempts=1)
    task.subtasks = [sub]
    return executor, task, sub


async def test_tool_loop_records_results_and_observations(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        execute=[
            [("write_file", {"path": "hello.txt", "content": "hi"})],
            [("read_file", {"path": "hello.txt"})],
            "hello.txt angelegt und geprüft.",
        ]
    )
    executor, task, sub = make(tmp_path, provider)
    outcome = await executor.execute(task, sub)

    assert outcome.completed and outcome.output == "hello.txt angelegt und geprüft."
    assert outcome.tool_calls == 2 and outcome.errors == []
    assert (tmp_path / "hello.txt").read_text() == "hi"
    assert [r.tool for r in task.tool_results] == ["write_file", "read_file"]
    assert all(r.ok and r.attempt == 1 and r.subtask_id == "s1" for r in task.tool_results)
    assert [o.source for o in task.observations] == ["tool:write_file", "tool:read_file", "model"]
    # Tool-Ergebnis wurde dem Modell zurückgegeben
    assert tool_messages(provider.requests["execute"][2])[-1] == "hi"


async def test_tool_failure_is_recorded_and_fed_back(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        execute=[
            [("read_file", {"path": "fehlt.txt"})],
            [("write_file", {"path": "fehlt.txt", "content": "neu"})],
            "Datei existierte nicht, daher neu angelegt.",
        ]
    )
    executor, task, sub = make(tmp_path, provider)
    outcome = await executor.execute(task, sub)

    assert outcome.completed
    assert outcome.errors == ["read_file fehlgeschlagen: Datei nicht gefunden: fehlt.txt"]
    assert task.errors[0].phase == Phase.OBSERVE and task.errors[0].kind == "tool"
    fed_back = tool_messages(provider.requests["execute"][1])[0]
    assert fed_back.startswith("ERROR:") and "Do not claim success" in fed_back
    assert [r.ok for r in task.tool_results] == [False, True]


async def test_disallowed_tool_is_rejected(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        execute=[[("write_file", {"path": "x", "content": "y"})], "aufgegeben"]
    )
    executor, task, sub = make(tmp_path, provider, allowed_tools=["read_file"])
    outcome = await executor.execute(task, sub)
    assert not (tmp_path / "x").exists()
    assert "nicht erlaubt" in outcome.errors[0]
    assert [s.name for s in provider.requests["execute"][0].tools] == ["read_file"]


async def test_tool_budget_stops_attempt(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        execute=[[("list_dir", {})], [("list_dir", {})], [("list_dir", {})]]
    )
    executor, task, sub = make(tmp_path, provider, max_tool_calls_per_attempt=2)
    outcome = await executor.execute(task, sub)
    assert not outcome.completed and outcome.tool_calls == 2
    assert task.errors[-1].kind == "budget"


async def test_model_failure_and_empty_answer(tmp_path: Path) -> None:
    executor, task, sub = make(
        tmp_path, ScriptedProvider(execute=[ProviderUnavailableError("down")])
    )
    outcome = await executor.execute(task, sub)
    assert not outcome.completed and task.errors[-1].kind == "model"

    executor, task, sub = make(tmp_path, ScriptedProvider(execute=["   "]))
    outcome = await executor.execute(task, sub)
    assert not outcome.completed and "weder Tool-Calls" in outcome.errors[0]


async def test_prompt_contains_feedback_dependencies_and_checks(tmp_path: Path) -> None:
    provider = ScriptedProvider(execute=["ok"])
    executor, task, sub = make(tmp_path, provider, notes=["nur stdlib"])
    task.subtasks.insert(
        0, Subtask("s0", "vorher", status=SubtaskStatus.DONE, output="Ergebnis-S0")
    )
    sub.depends_on = ["s0"]
    sub.feedback = ["Syntaxfehler in Zeile 3"]
    from agents.task import VerificationSpec

    sub.verification = [VerificationSpec("file_exists", {"path": "hello.txt"})]
    await executor.execute(task, sub)
    prompt = provider.requests["execute"][0].messages[1].content
    for fragment in (
        "Ergebnis-S0",
        "Syntaxfehler in Zeile 3",
        "file_exists",
        "nur stdlib",
        "attempt 1",
    ):
        assert fragment in prompt


async def test_failure_analyzer_with_and_without_model(tmp_path: Path) -> None:
    task = Task("x")
    sub = Subtask("s1", "y", attempts=2)
    task.verification_results.append(
        VerificationRecord("s1", 2, "python_syntax(a.py)", Verdict.FAILED, "Zeile 1")
    )
    outcome = ExecutionOutcome("fertig", completed=True)

    feedback = await FailureAnalyzer(None).analyze(task, sub, outcome)
    assert "python_syntax(a.py): Zeile 1" in feedback and task.errors[-1].phase == Phase.CORRECT

    provider = ScriptedProvider(
        analyze=["Ursache: fehlende Klammer. Nächstes Mal Datei komplett neu schreiben."]
    )
    feedback = await FailureAnalyzer(engine_for(provider)).analyze(task, sub, outcome)
    assert "fehlende Klammer" in feedback and "fehlende Klammer" in task.errors[-1].analysis

    provider = ScriptedProvider(analyze=[ProviderUnavailableError("down")])
    feedback = await FailureAnalyzer(engine_for(provider)).analyze(task, sub, outcome)
    assert "Analyse" not in feedback and task.errors[-1].kind == "model"


def test_reset_interrupted() -> None:
    task = Task("x")
    task.subtasks = [
        Subtask("a", "a", status=SubtaskStatus.IN_PROGRESS),
        Subtask("b", "b", status=SubtaskStatus.DONE),
    ]
    reset_interrupted(task)
    assert [s.status for s in task.subtasks] == [SubtaskStatus.PENDING, SubtaskStatus.DONE]
