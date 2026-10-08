"""VERIFY: deterministische Prüfung der Ergebnisse einer Teilaufgabe.

Der Verifier entscheidet – nicht das Modell –, ob eine Teilaufgabe als erfolgreich gilt.

Prüfquellen:
1. Vom Planner deklarierte Checks (``Subtask.verification``).
2. Automatische Checks aus den Tool-Ergebnissen des Versuchs:
   * Datei geschrieben → Datei erneut lesen, Inhalt per SHA-256 mit dem Geschriebenen vergleichen
   * Python-Datei geschrieben → Syntaxprüfung
   * Python-Datei geschrieben und ``constraints.test_command`` gesetzt → Tests ausführen
   * Schreibvorgang fehlgeschlagen und nicht erfolgreich wiederholt → Fehler

Ergebnis: ``PASSED`` nur, wenn mindestens eine Prüfung lief und alle bestanden.
``UNVERIFIED``, wenn keine Prüfung anwendbar war – das ist ausdrücklich kein Erfolg.

Befehle (``command``) laufen nur, wenn sie mit einem Eintrag der Allowlist beginnen,
ohne Shell, im Workspace, mit Timeout und ohne Secrets in der Umgebung.
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.executor import ExecutionOutcome
from agents.task import (
    Subtask,
    Task,
    ToolResultRecord,
    Verdict,
    VerificationRecord,
    VerificationSpec,
)
from tools.base import ToolError
from tools.filesystem import resolve_in_workspace, sha256_text

DEFAULT_ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest"),
    ("python", "-m", "py_compile"),
    ("pytest",),
)
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_WRITE_TOOLS = frozenset({"write_file"})
_PYTEST_NO_TESTS = 5  # pytest: "no tests collected"


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    verdict: Verdict
    detail: str


@dataclass
class VerifyContext:
    task: Task
    subtask: Subtask
    outcome: ExecutionOutcome
    attempt_results: list[ToolResultRecord]


CheckFn = Callable[[Mapping[str, Any], VerifyContext], Awaitable[CheckOutcome]]


def _passed(detail: str) -> CheckOutcome:
    return CheckOutcome(Verdict.PASSED, detail)


def _failed(detail: str) -> CheckOutcome:
    return CheckOutcome(Verdict.FAILED, detail)


def _unverified(detail: str) -> CheckOutcome:
    return CheckOutcome(Verdict.UNVERIFIED, detail)


def _safe_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}
    # Python prüft gecachten Bytecode nur über mtime (Sekunden) + Größe: eine Korrektur gleicher
    # Länge in derselben Sekunde würde sonst den alten Code ausführen.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class Verifier:
    def __init__(
        self,
        workspace: Path,
        *,
        allowed_commands: Sequence[Sequence[str]] = DEFAULT_ALLOWED_COMMANDS,
        command_timeout_s: float = 300.0,
        max_output_chars: int = 3000,
    ) -> None:
        self.workspace = workspace
        self.allowed_commands = [tuple(c) for c in allowed_commands]
        self.command_timeout_s = command_timeout_s
        self.max_output_chars = max_output_chars
        self._checks: dict[str, CheckFn] = {
            "file_exists": self._file_exists,
            "file_contains": self._file_contains,
            "file_matches_written": self._file_matches_written,
            "python_syntax": self._python_syntax,
            "command": self._command,
            "test_command": self._test_command,
            "output_contains": self._output_contains,
            "tool_succeeded": self._tool_succeeded,
            "writes_succeeded": self._writes_succeeded,
        }

    #: Checks, die der Planner verwenden darf (die übrigen werden nur automatisch erzeugt).
    PLANNABLE = (
        "file_exists",
        "file_contains",
        "python_syntax",
        "command",
        "output_contains",
        "tool_succeeded",
    )

    def known_checks(self) -> list[str]:
        return list(self.PLANNABLE)

    def register_check(self, name: str, fn: CheckFn) -> None:
        if name in self._checks:
            raise ValueError(f"Check {name!r} existiert bereits")
        self._checks[name] = fn

    # ------------------------------------------------------------------ Ablauf

    def auto_checks(
        self, task: Task, attempt_results: list[ToolResultRecord]
    ) -> list[VerificationSpec]:
        specs: list[VerificationSpec] = []
        last_write: dict[str, ToolResultRecord] = {}
        failed_writes: set[str] = set()
        for rec in attempt_results:
            if rec.tool not in _WRITE_TOOLS:
                continue
            path = str(rec.arguments.get("path", ""))
            if rec.ok:
                last_write[path] = rec
                failed_writes.discard(path)
            else:
                failed_writes.add(path)
        for path, rec in last_write.items():
            specs.append(
                VerificationSpec(
                    "file_matches_written",
                    {"path": path, "sha256": rec.data.get("sha256", "")},
                    auto=True,
                )
            )
            if path.endswith(".py"):
                specs.append(VerificationSpec("python_syntax", {"path": path}, auto=True))
        if task.constraints.test_command and any(p.endswith(".py") for p in last_write):
            specs.append(VerificationSpec("test_command", {}, auto=True))
        if failed_writes:
            specs.append(
                VerificationSpec("writes_succeeded", {"paths": sorted(failed_writes)}, auto=True)
            )
        return specs

    async def verify(self, task: Task, subtask: Subtask, outcome: ExecutionOutcome) -> Verdict:
        attempt = subtask.attempts
        attempt_results = [
            r for r in task.tool_results if r.subtask_id == subtask.id and r.attempt == attempt
        ]
        ctx = VerifyContext(task, subtask, outcome, attempt_results)
        specs = self.auto_checks(task, attempt_results) + list(subtask.verification)
        if not outcome.completed:
            # Nicht regulär beendet: Ergebnis zählt nicht, unabhängig von Einzelchecks.
            specs = [VerificationSpec("execution_completed", {}, auto=True), *specs]

        verdicts: list[Verdict] = []
        for spec in specs:
            result = await self._run_check(spec, ctx)
            verdicts.append(result.verdict)
            task.verification_results.append(
                VerificationRecord(
                    subtask.id, attempt, self._label(spec), result.verdict, result.detail, spec.auto
                )
            )
        if not verdicts:
            task.verification_results.append(
                VerificationRecord(
                    subtask.id,
                    attempt,
                    "none",
                    Verdict.UNVERIFIED,
                    "Keine anwendbare Prüfung – Ergebnis ist nicht verifiziert",
                    True,
                )
            )
            return Verdict.UNVERIFIED
        if Verdict.FAILED in verdicts:
            return Verdict.FAILED
        if all(v == Verdict.PASSED for v in verdicts):
            return Verdict.PASSED
        return Verdict.UNVERIFIED

    @staticmethod
    def _label(spec: VerificationSpec) -> str:
        target = spec.params.get("path") or spec.params.get("tool") or spec.params.get("command")
        if isinstance(target, list):
            target = " ".join(map(str, target))
        return f"{spec.type}({target})" if target else spec.type

    async def _run_check(self, spec: VerificationSpec, ctx: VerifyContext) -> CheckOutcome:
        if spec.type == "execution_completed":
            return _failed("Ausführung nicht abgeschlossen: " + "; ".join(ctx.outcome.errors[-3:]))
        fn = self._checks.get(spec.type)
        if fn is None:
            return _unverified(f"Unbekannter Check-Typ {spec.type!r}")
        try:
            return await fn(spec.params, ctx)
        except ToolError as exc:
            return _failed(str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            return _unverified(f"Ungültige Check-Parameter für {spec.type}: {exc}")

    # ------------------------------------------------------------------ Checks

    def _path(self, params: Mapping[str, Any]) -> Path:
        return resolve_in_workspace(self.workspace, str(params["path"]))

    async def _file_exists(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        path = self._path(params)
        return (
            _passed(f"{params['path']} existiert")
            if path.exists()
            else _failed(f"{params['path']} existiert nicht")
        )

    async def _read(self, params: Mapping[str, Any]) -> str | None:
        path = self._path(params)
        if not path.is_file():
            return None
        return await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")

    async def _file_contains(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        content = await self._read(params)
        if content is None:
            return _failed(f"{params['path']} existiert nicht")
        text = str(params["text"])
        return (
            _passed(f"{params['path']} enthält {text!r}")
            if text in content
            else _failed(f"{params['path']} enthält {text!r} nicht")
        )

    async def _file_matches_written(
        self, params: Mapping[str, Any], ctx: VerifyContext
    ) -> CheckOutcome:
        content = await self._read(params)
        if content is None:
            return _failed(f"{params['path']} nach dem Schreiben nicht lesbar")
        expected = str(params.get("sha256", ""))
        if not expected:
            return _unverified("Kein Hash des geschriebenen Inhalts vorhanden")
        if sha256_text(content) != expected:
            return _failed(f"{params['path']}: Inhalt weicht vom geschriebenen Inhalt ab")
        return _passed(f"{params['path']} erneut gelesen, Inhalt stimmt ({len(content)} Zeichen)")

    async def _python_syntax(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        root = self._path(params)
        if root.is_dir():
            files = sorted(p for p in root.rglob("*.py") if ".venv" not in p.parts)
        elif root.is_file():
            files = [root]
        else:
            return _failed(f"{params['path']} existiert nicht")
        if not files:
            return _unverified(f"Keine Python-Dateien unter {params['path']}")
        errors = []
        for file in files:
            source = await asyncio.to_thread(file.read_text, encoding="utf-8", errors="replace")
            try:
                ast.parse(source, filename=str(file))
            except SyntaxError as exc:
                rel = file.relative_to(self.workspace.resolve())
                errors.append(f"{rel}:{exc.lineno}:{exc.offset}: {exc.msg}")
        if errors:
            return _failed("Syntaxfehler: " + "; ".join(errors[:5]))
        return _passed(f"Syntax ok ({len(files)} Datei(en))")

    def _allowed(self, command: Sequence[str]) -> bool:
        return any(tuple(command[: len(a)]) == a for a in self.allowed_commands)

    async def _run_command(
        self, command: Sequence[str], expect_exit: int, timeout_s: float
    ) -> CheckOutcome:
        argv = [sys.executable if command[0] == "python" else command[0], *command[1:]]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.workspace,
                env=_safe_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return _failed(f"Befehl nicht startbar: {exc}")
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return _failed(f"Zeitlimit {timeout_s:.0f}s überschritten: {' '.join(command)}")
        output = stdout.decode("utf-8", errors="replace").strip()
        tail = output[-self.max_output_chars :]
        label = " ".join(command)
        if proc.returncode == expect_exit:
            return _passed(f"{label} → exit {proc.returncode}\n{tail[-500:]}")
        if proc.returncode == _PYTEST_NO_TESTS and "pytest" in command:
            return _unverified(f"{label}: keine Tests gefunden – nichts verifiziert\n{tail[-500:]}")
        return _failed(f"{label} → exit {proc.returncode} (erwartet {expect_exit})\n{tail}")

    async def _command(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        command = params["command"]
        if isinstance(command, str) or not command or not all(isinstance(c, str) for c in command):
            raise ValueError("command muss eine nicht-leere Liste von Strings sein")
        if not self._allowed(command):
            return _unverified(
                f"Befehl nicht in der Allowlist, nicht ausgeführt: {' '.join(command)}"
            )
        timeout = min(
            float(params.get("timeout_s", self.command_timeout_s)), self.command_timeout_s
        )
        return await self._run_command(command, int(params.get("expect_exit", 0)), timeout)

    async def _test_command(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        # Vom Nutzer konfiguriert (Constraints) – nicht vom Modell → keine Allowlist nötig.
        command = ctx.task.constraints.test_command
        if not command:
            return _unverified("Kein test_command konfiguriert")
        return await self._run_command(command, 0, self.command_timeout_s)

    async def _output_contains(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        text = str(params["text"])
        if text.lower() in ctx.outcome.output.lower():
            return _passed(f"Antwort enthält {text!r}")
        return _failed(f"Antwort enthält {text!r} nicht")

    async def _tool_succeeded(self, params: Mapping[str, Any], ctx: VerifyContext) -> CheckOutcome:
        tool = str(params["tool"])
        calls = [r for r in ctx.attempt_results if r.tool == tool]
        if any(r.ok for r in calls):
            return _passed(f"{tool} erfolgreich ausgeführt")
        if calls:
            return _failed(f"{tool} {len(calls)}x aufgerufen, nie erfolgreich: {calls[-1].error}")
        return _failed(f"{tool} wurde nicht aufgerufen")

    async def _writes_succeeded(
        self, params: Mapping[str, Any], ctx: VerifyContext
    ) -> CheckOutcome:
        return _failed(
            "Schreiben fehlgeschlagen und nicht erfolgreich wiederholt: "
            + ", ".join(params["paths"])
        )
