"""Testausführung und Code-Checks.

:class:`TestRunner` führt Prüfbefehle kontrolliert aus (gemeinsame Prozess-Sandbox aus
:mod:`tools.process`) und wertet pytest **strukturiert** über JUnit-XML aus – nicht über
Textparsing der Konsolenausgabe.

Code-Checks: :class:`PythonSyntaxCheck`, :class:`TypeCheck`, :class:`UnitTestCheck`,
:class:`IntegrationTestCheck`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.verifier import Check, CheckResult, CheckStatus, VerificationContext
from tools.base import ToolError
from tools.process import ProcessResult, run_process

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".tox",
    }
)
PYTEST_NO_TESTS = 5


@dataclass(frozen=True)
class TestFailure:
    name: str
    message: str


@dataclass
class TestReport:
    """Strukturiertes Ergebnis eines Testlaufs."""

    __test__ = False  # kein pytest-Testfall

    command: list[str]
    exit_code: int | None
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list[TestFailure] = field(default_factory=list)
    output_tail: str = ""
    timed_out: bool = False
    collected: bool = True
    structured: bool = False
    """True, wenn die Zahlen aus JUnit-XML stammen (sonst nur Exit-Code)."""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.failed == 0 and self.errors == 0


class TestRunner:
    __test__ = False

    def __init__(
        self, workspace: Path, *, timeout_s: float = 300.0, max_output_bytes: int = 200_000
    ) -> None:
        self.workspace = workspace
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes

    async def run(self, argv: Sequence[str], *, timeout_s: float | None = None) -> ProcessResult:
        return await run_process(
            argv,
            self.workspace,
            timeout_s=timeout_s or self.timeout_s,
            max_output_bytes=self.max_output_bytes,
        )

    @staticmethod
    def module_available(module: str) -> bool:
        """Ist ein Python-Werkzeug (pytest, mypy …) im NOVA-Interpreter installiert?"""
        return importlib.util.find_spec(module) is not None

    async def run_pytest(
        self, extra_args: Sequence[str] = (), *, marker: str | None = None
    ) -> TestReport:
        junit_dir = Path(tempfile.mkdtemp(prefix="nova-junit-"))
        junit = junit_dir / "report.xml"
        argv = [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            *extra_args,
        ]
        if marker:
            argv += ["-m", marker]
        try:
            result = await self.run(argv)
            report = TestReport(
                command=argv,
                exit_code=result.exit_code,
                output_tail=result.output[-3000:],
                timed_out=result.timed_out,
                collected=result.exit_code != PYTEST_NO_TESTS,
            )
            if junit.exists():
                self._parse_junit(junit, report)
            return report
        finally:
            await asyncio.to_thread(shutil.rmtree, junit_dir, True)

    async def run_command_as_tests(self, argv: Sequence[str]) -> TestReport:
        """Beliebiger Testbefehl. Ist es pytest, wird strukturiert ausgewertet."""
        if _is_pytest(argv):
            return await self.run_pytest(_pytest_args(argv))
        result = await self.run(argv)
        return TestReport(
            command=list(argv),
            exit_code=result.exit_code,
            output_tail=result.output[-3000:],
            timed_out=result.timed_out,
        )

    @staticmethod
    def _parse_junit(path: Path, report: TestReport) -> None:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for suite in suites:
            report.total += int(suite.get("tests", 0))
            report.failed += int(suite.get("failures", 0))
            report.errors += int(suite.get("errors", 0))
            report.skipped += int(suite.get("skipped", 0))
            for case in suite.iter("testcase"):
                for tag in ("failure", "error"):
                    node = case.find(tag)
                    if node is not None:
                        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
                        message = (node.get("message") or node.text or "").strip()
                        report.failures.append(TestFailure(name, message[:500]))
        report.passed = max(report.total - report.failed - report.errors - report.skipped, 0)
        report.structured = True


def _is_pytest(argv: Sequence[str]) -> bool:
    joined = " ".join(argv[:3])
    return bool(argv) and (Path(argv[0]).name == "pytest" or "-m pytest" in joined)


def _pytest_args(argv: Sequence[str]) -> list[str]:
    if Path(argv[0]).name == "pytest":
        return list(argv[1:])
    index = list(argv).index("pytest")
    return list(argv[index + 1 :])


def python_files(ctx: VerificationContext, paths: Sequence[str] | None) -> list[Path]:
    """Explizite Pfade, sonst geänderte .py-Artefakte, sonst alle .py-Dateien im Workspace."""
    root = ctx.workspace.resolve()
    selected = list(paths) if paths else [a for a in ctx.artifacts if a.endswith(".py")]
    if selected:
        files = []
        for p in selected:
            candidate = (root / p).resolve()
            if candidate.is_relative_to(root):
                files += sorted(candidate.rglob("*.py")) if candidate.is_dir() else [candidate]
        return files
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        found += [Path(dirpath) / f for f in filenames if f.endswith(".py")]
    return sorted(found)


# --------------------------------------------------------------------------- Checks


class PythonSyntaxCheck(Check):
    name = "python_syntax"
    category = "code"
    timeout_s = 60.0

    def __init__(self, paths: Sequence[str] | None = None) -> None:
        self.paths = list(paths) if paths else None

    async def run(self, ctx: VerificationContext) -> CheckResult:
        files = python_files(ctx, self.paths)
        if not files:
            return self.unverifiable("keine Python-Dateien gefunden")
        missing = [f for f in files if not f.exists()]
        if missing:
            return self.failed(f"Datei fehlt: {missing[0].name}", missing=[str(m) for m in missing])
        errors = await asyncio.to_thread(self._compile_all, files, ctx.workspace.resolve())
        if errors:
            return self.failed(
                "Syntaxfehler: " + "; ".join(errors[:5]), errors=errors, files=len(files)
            )
        return self.passed(f"Syntax ok ({len(files)} Datei(en))", files=len(files))

    @staticmethod
    def _compile_all(files: Sequence[Path], root: Path) -> list[str]:
        errors = []
        for file in files:
            source = file.read_text(encoding="utf-8", errors="replace")
            try:
                # compile (nicht nur ast.parse) erkennt auch z. B. 'return' außerhalb einer
                # Funktion; der Code wird dabei NICHT ausgeführt.
                compile(source, str(file), "exec", dont_inherit=True)
            except SyntaxError as exc:
                rel = file.relative_to(root) if file.is_relative_to(root) else file
                errors.append(f"{rel}:{exc.lineno}:{exc.offset}: {exc.msg}")
        return errors


_MYPY_ERROR = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)? error: (?P<msg>.+)$", re.MULTILINE
)


class TypeCheck(Check):
    name = "type_check"
    category = "code"
    requires = ("python_syntax",)

    def __init__(self, paths: Sequence[str] | None = None) -> None:
        self.paths = list(paths) if paths else None

    async def run(self, ctx: VerificationContext) -> CheckResult:
        runner = TestRunner(ctx.workspace)
        if ctx.type_command:
            argv = list(ctx.type_command)
        else:
            if not runner.module_available("mypy"):
                return self.unverifiable("kein Typchecker verfügbar (mypy nicht installiert)")
            files = python_files(ctx, self.paths)
            if not files:
                return self.unverifiable("keine Python-Dateien gefunden")
            root = ctx.workspace.resolve()
            argv = [
                "python",
                "-m",
                "mypy",
                "--ignore-missing-imports",
                "--no-color-output",
                "--no-error-summary",
                "--cache-dir",
                os.devnull,
                *[str(f.relative_to(root)) for f in files],
            ]
        try:
            result = await runner.run(argv)
        except ToolError as exc:
            return self.unverifiable(f"Typchecker nicht startbar: {exc}")
        if result.timed_out:
            return self.result(CheckStatus.ERROR, "Typprüfung: Zeitlimit überschritten")
        errors = [m.group(0) for m in _MYPY_ERROR.finditer(result.output)]
        if result.exit_code == 0:
            return self.passed("Typprüfung ohne Fehler", command=list(result.argv))
        return self.failed(
            f"{len(errors) or 'unbekannte Anzahl'} Typfehler: " + "; ".join(errors[:3]),
            errors=errors[:20],
            exit_code=result.exit_code,
        )


class _TestCheck(Check):
    category = "code"
    requires = ("python_syntax",)

    def _evaluate(self, report: TestReport, label: str) -> CheckResult:
        evidence = {
            "command": report.command,
            "exit_code": report.exit_code,
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "errors": report.errors,
            "skipped": report.skipped,
            "structured": report.structured,
            "failures": [f.__dict__ for f in report.failures[:10]],
        }
        if report.timed_out:
            return self.result(CheckStatus.ERROR, f"{label}: Zeitlimit überschritten", **evidence)
        if not report.collected or (report.structured and report.total == 0):
            return self.unverifiable(
                f"{label}: keine Tests gefunden – nichts verifiziert", **evidence
            )
        if report.ok:
            counts = (
                f"{report.passed} bestanden, {report.skipped} übersprungen"
                if report.structured
                else "Exit-Code 0"
            )
            return self.passed(f"{label}: {counts}", **evidence)
        if report.structured:
            names = ", ".join(f.name for f in report.failures[:3])
            return self.failed(
                f"{label}: {report.failed} fehlgeschlagen, {report.errors} Fehler"
                + (f" ({names})" if names else ""),
                **evidence,
            )
        return self.failed(
            f"{label}: Exit-Code {report.exit_code}\n{report.output_tail[-800:]}", **evidence
        )


class UnitTestCheck(_TestCheck):
    name = "unit_tests"

    async def run(self, ctx: VerificationContext) -> CheckResult:
        runner = TestRunner(ctx.workspace)
        if ctx.test_command:
            report = await runner.run_command_as_tests(ctx.test_command)
        elif runner.module_available("pytest"):
            report = await runner.run_pytest()
        else:
            return self.unverifiable("kein Testbefehl konfiguriert und pytest nicht installiert")
        return self._evaluate(report, "Unit-Tests")


class IntegrationTestCheck(_TestCheck):
    name = "integration_tests"
    weight = 1.5

    def __init__(self, marker: str | None = "integration") -> None:
        self.marker = marker

    async def run(self, ctx: VerificationContext) -> CheckResult:
        runner = TestRunner(ctx.workspace)
        if ctx.integration_command:
            report = await runner.run_command_as_tests(ctx.integration_command)
        elif self.marker and runner.module_available("pytest"):
            report = await runner.run_pytest(marker=self.marker)
        else:
            return self.unverifiable("keine Integrationstests konfiguriert")
        return self._evaluate(report, "Integrationstests")
