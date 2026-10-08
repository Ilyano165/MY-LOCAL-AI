from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.state import InMemoryTaskStore, JsonFileTaskStore, TaskStateError
from agents.task import (
    Complexity,
    Constraints,
    FinalResult,
    FinalStatus,
    Phase,
    Subtask,
    SubtaskStatus,
    Task,
    TaskAnalysis,
    TaskStatus,
    ToolResultRecord,
    Verdict,
    VerificationRecord,
    VerificationSpec,
)
from models.capabilities import TaskType


def full_task() -> Task:
    task = Task(
        "Schreibe app.py", Constraints(max_steps=5, test_command=["python", "-m", "pytest"])
    )
    task.analysis = TaskAnalysis(
        TaskType.CODING, Complexity.MULTI_STEP, True, ["tests grün"], ["coding"]
    )
    task.subtasks = [
        Subtask(
            "s1",
            "Datei schreiben",
            verification=[VerificationSpec("python_syntax", {"path": "app.py"})],
            status=SubtaskStatus.DONE,
            attempts=2,
            output="ok",
            feedback=["fix"],
            verdict=Verdict.PASSED,
        ),
        Subtask("s2", "Tests", depends_on=["s1"], superseded=True),
    ]
    task.observe("tool:write_file", "geschrieben", "s1", 1)
    task.tool_results.append(
        ToolResultRecord(
            "s1", 1, "write_file", {"path": "app.py"}, True, "ok", None, {"sha256": "abc"}
        )
    )
    err = task.record_error(Phase.VERIFY, "verification", "Syntaxfehler", "s1", 1)
    err.analysis = "Klammer fehlt"
    task.verification_results.append(
        VerificationRecord("s1", 2, "python_syntax(app.py)", Verdict.PASSED, "ok")
    )
    task.final_result = FinalResult(FinalStatus.SUCCESS, "fertig", True, "alles gut")
    task.status = TaskStatus.COMPLETED
    task.phase = Phase.FINALIZE
    return task


def test_task_roundtrip_is_lossless() -> None:
    task = full_task()
    data = json.loads(json.dumps(task.to_dict()))
    restored = Task.from_dict(data)
    assert restored == task
    assert restored.subtasks[0].verification[0].params == {"path": "app.py"}
    assert restored.errors[0].analysis == "Klammer fehlt"


def test_task_helpers() -> None:
    task = full_task()
    assert task.subtask("s2").depends_on == ["s1"]
    with pytest.raises(KeyError):
        task.subtask("x")
    task.current_step = 1
    assert task.current_subtask is task.subtasks[1]
    task.current_step = 9
    assert task.current_subtask is None
    with pytest.raises(ValueError):
        Task("   ")


def test_status_terminal() -> None:
    assert TaskStatus.COMPLETED.terminal and TaskStatus.ABORTED.terminal
    assert not TaskStatus.RUNNING.terminal


@pytest.mark.parametrize(
    "kwargs",
    [{"max_steps": 0}, {"max_attempts_per_subtask": 0}, {"max_replans": -1}, {"tool_timeout_s": 0}],
)
def test_constraints_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Constraints(**kwargs)  # type: ignore[arg-type]


def test_verification_spec_flat_and_nested() -> None:
    flat = VerificationSpec.from_dict({"type": "file_exists", "path": "a"})
    nested = VerificationSpec.from_dict(
        {"type": "file_exists", "params": {"path": "a"}, "auto": True}
    )
    assert flat.params == nested.params == {"path": "a"}
    assert nested.auto
    with pytest.raises(ValueError):
        VerificationSpec.from_dict({"path": "a"})


async def test_json_store_roundtrip_and_listing(tmp_path: Path) -> None:
    store = JsonFileTaskStore(tmp_path / "tasks")
    assert await store.list_ids() == []
    task = full_task()
    await store.save(task)
    assert await store.list_ids() == [task.id]
    loaded = await store.load(task.id)
    assert loaded.to_dict() == task.to_dict()
    assert not list((tmp_path / "tasks").glob("*.tmp"))  # atomar, keine Reste


async def test_json_store_errors(tmp_path: Path) -> None:
    store = JsonFileTaskStore(tmp_path)
    with pytest.raises(TaskStateError, match="nicht gefunden"):
        await store.load("missing")
    with pytest.raises(TaskStateError, match="Ungültige Task-ID"):
        await store.load("../etc/passwd")
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(TaskStateError, match="beschädigt"):
        await store.load("broken")


async def test_in_memory_store_isolates_copies() -> None:
    store = InMemoryTaskStore()
    task = full_task()
    await store.save(task)
    task.objective = "geändert nach dem Speichern"
    assert (await store.load(task.id)).objective == "Schreibe app.py"
    assert await store.list_ids() == [task.id]
    with pytest.raises(TaskStateError):
        await store.load("nope")
