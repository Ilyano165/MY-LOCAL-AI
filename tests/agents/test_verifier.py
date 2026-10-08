from __future__ import annotations

from pathlib import Path

import pytest

from agents.executor import ExecutionOutcome
from agents.task import Constraints, Subtask, Task, ToolResultRecord, Verdict, VerificationSpec
from agents.verifier import Verifier
from tools.filesystem import sha256_text

PYTEST = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def setup(
    tmp_path: Path, *specs: VerificationSpec, test_command: list[str] | None = None
) -> tuple[Task, Subtask]:
    task = Task("x", Constraints(test_command=test_command))
    sub = Subtask("s1", "y", verification=list(specs), attempts=1)
    task.subtasks = [sub]
    return task, sub


def wrote(task: Task, path: str, content: str, ok: bool = True) -> None:
    task.tool_results.append(
        ToolResultRecord(
            "s1",
            1,
            "write_file",
            {"path": path, "content": content},
            ok,
            "",
            None if ok else "kaputt",
            {"sha256": sha256_text(content)} if ok else {},
        )
    )


DONE = ExecutionOutcome("Antwort mit Ergebnis 42", completed=True)


async def test_no_checks_is_unverified_not_success(tmp_path: Path) -> None:
    task, sub = setup(tmp_path)
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.UNVERIFIED
    assert task.verification_results[-1].check == "none"


async def test_incomplete_execution_always_fails(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("x")
    task, sub = setup(tmp_path, VerificationSpec("file_exists", {"path": "a"}))
    verdict = await Verifier(tmp_path).verify(
        task, sub, ExecutionOutcome("", completed=False, errors=["Budget"])
    )
    assert verdict == Verdict.FAILED
    assert task.verification_results[0].check == "execution_completed"


async def test_written_file_is_reread_and_compared(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hallo")
    task, sub = setup(tmp_path)
    wrote(task, "a.txt", "hallo")
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.PASSED
    record = task.verification_results[0]
    assert record.auto and record.check == "file_matches_written(a.txt)"

    (tmp_path / "a.txt").write_text("von außen verändert")
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.FAILED


async def test_python_write_triggers_syntax_check_and_tests(tmp_path: Path) -> None:
    code = "def add(a, b):\n    return a + b\n"
    tests = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    (tmp_path / "calc.py").write_text(code)
    (tmp_path / "test_calc.py").write_text(tests)
    task, sub = setup(tmp_path, test_command=PYTEST)
    wrote(task, "calc.py", code)

    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.PASSED
    checks = [r.check for r in task.verification_results]
    assert checks == ["file_matches_written(calc.py)", "python_syntax(calc.py)", "test_command"]
    assert "1 passed" in task.verification_results[-1].detail


async def test_syntax_error_is_reported_with_location(tmp_path: Path) -> None:
    broken = "def add(a, b)\n    return a + b\n"
    (tmp_path / "calc.py").write_text(broken)
    task, sub = setup(tmp_path)
    wrote(task, "calc.py", broken)
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.FAILED
    syntax = task.verification_results[1]
    assert syntax.verdict == Verdict.FAILED and "calc.py:1" in syntax.detail


async def test_failing_tests_fail_verification(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": PYTEST}))
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.FAILED
    assert "1 failed" in task.verification_results[0].detail


async def test_command_not_in_allowlist_is_not_executed(tmp_path: Path) -> None:
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": ["rm", "-rf", "."]}))
    (tmp_path / "keep.txt").write_text("x")
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.UNVERIFIED
    assert (tmp_path / "keep.txt").exists()
    assert "Allowlist" in task.verification_results[0].detail


async def test_command_timeout(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(5)\n")
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": PYTEST}))
    verifier = Verifier(tmp_path, command_timeout_s=0.5)
    assert await verifier.verify(task, sub, DONE) == Verdict.FAILED
    assert "Zeitlimit" in task.verification_results[0].detail


async def test_failed_write_without_retry_fails(tmp_path: Path) -> None:
    task, sub = setup(tmp_path)
    wrote(task, "a.txt", "x", ok=False)
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.FAILED
    assert "a.txt" in task.verification_results[-1].detail

    task, sub = setup(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    wrote(task, "a.txt", "x", ok=False)
    wrote(task, "a.txt", "x", ok=True)  # erfolgreich wiederholt
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.PASSED


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (VerificationSpec("output_contains", {"text": "ergebnis 42"}), Verdict.PASSED),
        (VerificationSpec("output_contains", {"text": "43"}), Verdict.FAILED),
        (VerificationSpec("file_exists", {"path": "nope"}), Verdict.FAILED),
        (VerificationSpec("file_contains", {"path": "f.txt", "text": "abc"}), Verdict.PASSED),
        (VerificationSpec("file_contains", {"path": "f.txt", "text": "xyz"}), Verdict.FAILED),
        (VerificationSpec("file_contains", {"path": "nope", "text": "x"}), Verdict.FAILED),
        (VerificationSpec("python_syntax", {"path": "."}), Verdict.UNVERIFIED),  # keine .py-Dateien
        (VerificationSpec("tool_succeeded", {"tool": "write_file"}), Verdict.FAILED),
        (VerificationSpec("file_exists", {"path": "../x"}), Verdict.FAILED),  # Workspace-Ausbruch
        (VerificationSpec("file_exists", {}), Verdict.UNVERIFIED),  # fehlender Parameter
        (VerificationSpec("magic", {}), Verdict.UNVERIFIED),
        (VerificationSpec("command", {"command": "python -m pytest"}), Verdict.UNVERIFIED),
    ],
)
async def test_individual_checks(tmp_path: Path, spec: VerificationSpec, expected: Verdict) -> None:
    (tmp_path / "f.txt").write_text("abcdef")
    task, sub = setup(tmp_path, spec)
    assert await Verifier(tmp_path).verify(task, sub, DONE) == expected


def test_known_checks_and_registration(tmp_path: Path) -> None:
    verifier = Verifier(tmp_path)
    assert "command" in verifier.known_checks()
    assert "file_matches_written" not in verifier.known_checks()  # nur automatisch
    with pytest.raises(ValueError):
        verifier.register_check("command", verifier._command)


async def test_pytest_without_tests_is_unverified(tmp_path: Path) -> None:
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": PYTEST}))
    assert await Verifier(tmp_path).verify(task, sub, DONE) == Verdict.UNVERIFIED
    assert "keine Tests" in task.verification_results[0].detail


async def test_same_size_fix_within_same_second_is_tested_freshly(tmp_path: Path) -> None:
    """Regression: veralteter Bytecode darf eine Korrektur nicht verdecken."""
    import os

    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    calc = tmp_path / "calc.py"
    calc.write_text("def add(a, b):\n    return a - b\n")
    stat = calc.stat()
    verifier = Verifier(tmp_path)
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": PYTEST}))
    assert await verifier.verify(task, sub, DONE) == Verdict.FAILED

    calc.write_text("def add(a, b):\n    return a + b\n")  # gleiche Größe
    os.utime(calc, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # gleiche Sekunde
    task, sub = setup(tmp_path, VerificationSpec("command", {"command": PYTEST}))
    assert await verifier.verify(task, sub, DONE) == Verdict.PASSED
