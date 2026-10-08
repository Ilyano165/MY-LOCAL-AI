"""TestRunner (strukturiert über JUnit-XML) und Code-Checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import CheckStatus, VerificationContext
from evaluation.test_runner import (
    IntegrationTestCheck,
    PythonSyntaxCheck,
    TestRunner,
    TypeCheck,
    UnitTestCheck,
    python_files,
)

TESTS = (
    "import pytest\n\n"
    "def test_ok():\n    assert 1 == 1\n\n"
    "def test_fail():\n    assert 1 == 2, 'eins ist nicht zwei'\n\n"
    "@pytest.mark.skip(reason='später')\ndef test_skip():\n    pass\n\n"
    "@pytest.fixture\ndef broken():\n    raise RuntimeError('fixture kaputt')\n\n"
    "def test_error(broken):\n    pass\n"
)


async def test_pytest_results_are_parsed_from_junit(tmp_path: Path) -> None:
    (tmp_path / "test_mix.py").write_text(TESTS)
    report = await TestRunner(tmp_path).run_pytest()
    assert report.structured
    assert (report.total, report.passed, report.failed, report.errors, report.skipped) == (
        4,
        1,
        1,
        1,
        1,
    )
    assert not report.ok
    names = {f.name for f in report.failures}
    assert names == {"test_mix::test_fail", "test_mix::test_error"}
    assert any("eins ist nicht zwei" in f.message for f in report.failures)


async def test_no_tests_collected(tmp_path: Path) -> None:
    report = await TestRunner(tmp_path).run_pytest()
    assert not report.collected and report.exit_code == 5


async def test_non_pytest_command_uses_exit_code(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import sys\nsys.exit(3)\n")
    report = await TestRunner(tmp_path).run_command_as_tests(["python", "check.py"])
    assert not report.structured and report.exit_code == 3 and not report.ok


async def test_pytest_command_is_parsed_structured(tmp_path: Path) -> None:
    (tmp_path / "test_a.py").write_text("def test_a():\n    assert True\n")
    report = await TestRunner(tmp_path).run_command_as_tests(["python", "-m", "pytest", "-q"])
    assert report.structured and report.passed == 1 and report.ok


def test_module_available() -> None:
    assert TestRunner.module_available("pytest")
    assert not TestRunner.module_available("gibt_es_nicht_123")


# --------------------------------------------------------------------------- Syntax


@pytest.mark.parametrize(
    ("source", "status"),
    [
        ("x = 1\n", CheckStatus.PASSED),
        ("def f(:\n    pass\n", CheckStatus.FAILED),
        ("return 5\n", CheckStatus.FAILED),  # nur compile() erkennt das, ast.parse nicht
        ("import os\nos.remove('/tmp/nie')\n", CheckStatus.PASSED),  # wird NICHT ausgeführt
    ],
)
async def test_python_syntax(tmp_path: Path, source: str, status: CheckStatus) -> None:
    (tmp_path / "m.py").write_text(source)
    result = await PythonSyntaxCheck().run(VerificationContext(tmp_path, artifacts=["m.py"]))
    assert result.status == status, result.detail


async def test_python_syntax_file_selection(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bad.py").write_text("def (\n")
    ctx = VerificationContext(tmp_path)
    assert [p.name for p in python_files(ctx, None)] == ["a.py"]
    assert (await PythonSyntaxCheck().run(ctx)).status == CheckStatus.PASSED
    missing = await PythonSyntaxCheck(["fehlt.py"]).run(ctx)
    assert missing.status == CheckStatus.FAILED
    assert python_files(VerificationContext(tmp_path), ["../ausserhalb.py"]) == []


# --------------------------------------------------------------------------- Typen


async def test_type_check_pass_and_fail(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("def f(x: int) -> int:\n    return x + 1\n")
    (tmp_path / "bad.py").write_text("def f(x: int) -> int:\n    return str(x)\n")
    ok = await TypeCheck().run(VerificationContext(tmp_path, artifacts=["ok.py"]))
    bad = await TypeCheck().run(VerificationContext(tmp_path, artifacts=["bad.py"]))
    assert ok.status == CheckStatus.PASSED
    assert bad.status == CheckStatus.FAILED and "Typfehler" in bad.detail
    assert any("bad.py:2" in e for e in bad.evidence["errors"])


async def test_type_check_without_mypy_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    monkeypatch.setattr(TestRunner, "module_available", staticmethod(lambda m: False))
    result = await TypeCheck().run(VerificationContext(tmp_path))
    assert result.status == CheckStatus.UNVERIFIABLE


async def test_custom_type_command(tmp_path: Path) -> None:
    (tmp_path / "t.py").write_text("import sys\nsys.exit(1)\n")
    ctx = VerificationContext(tmp_path, type_command=["python", "t.py"])
    assert (await TypeCheck().run(ctx)).status == CheckStatus.FAILED


# --------------------------------------------------------------------------- Tests


async def test_unit_test_check(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = await UnitTestCheck().run(VerificationContext(tmp_path))
    assert result.status == CheckStatus.PASSED and result.evidence["passed"] == 1
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    failed = await UnitTestCheck().run(VerificationContext(tmp_path))
    assert failed.status == CheckStatus.FAILED and "test_bad" in failed.detail


async def test_unit_test_check_without_tests_is_unverifiable(tmp_path: Path) -> None:
    result = await UnitTestCheck().run(VerificationContext(tmp_path))
    assert result.status == CheckStatus.UNVERIFIABLE


async def test_integration_tests_by_marker_or_command(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(
        "def pytest_configure(config):\n    config.addinivalue_line('markers', 'integration: x')\n"
    )
    (tmp_path / "test_it.py").write_text(
        "import pytest\n\n@pytest.mark.integration\ndef test_it():\n    assert True\n\n"
        "def test_unit():\n    assert False\n"
    )
    marker = await IntegrationTestCheck().run(VerificationContext(tmp_path))
    assert marker.status == CheckStatus.PASSED and marker.evidence["total"] == 1
    none = await IntegrationTestCheck(marker=None).run(VerificationContext(tmp_path))
    assert none.status == CheckStatus.UNVERIFIABLE
    (tmp_path / "it.py").write_text("raise SystemExit(2)\n")
    cmd = await IntegrationTestCheck().run(
        VerificationContext(tmp_path, integration_command=["python", "it.py"])
    )
    assert cmd.status == CheckStatus.FAILED
