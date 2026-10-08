"""execute_command: kontrollierte Prozessausführung.

* **Keine Shell.** Der Befehl wird als Argumentliste ausgeführt; Shell-Operatoren
  (``|``, ``&&``, ``>`` …) werden abgelehnt statt stillschweigend falsch interpretiert.
* **Klassifizierung** bestimmt die Berechtigung je Aufruf:
  gesperrt (``sudo``, ``mkfs`` …) → nie; Netzwerk (``curl``, ``git push``, ``pip install`` …)
  → ``NETWORK``; Shells/Interpreter-Code/Löschen → ``DANGEROUS``; Allowlist →
  ``EXECUTE_ALLOWLISTED``; alles andere → ``EXECUTE`` (Default: Bestätigung nötig).
* **Grenzen:** Arbeitsverzeichnis nur in erlaubten Verzeichnissen, Timeout (Prozessgruppe wird
  beendet), Ausgabe-Limit beim Einlesen, kein stdin, Umgebung ohne Secrets.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Any, ClassVar

from tools.base import Permission, Tool, ToolContext, ToolError, ToolResult, safe_environment

DEFAULT_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest"),
    ("python", "-m", "py_compile"),
    ("python", "-m", "mypy"),
    ("python", "-m", "ruff", "check"),
    ("python", "-m", "ruff", "format", "--check"),
    ("pytest",),
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("mypy",),
    ("ls",),
    ("pwd",),
    ("cat",),
    ("head",),
    ("tail",),
    ("wc",),
)

BLOCKED = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "pkexec",
        "runas",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "mkfs",
        "fdisk",
        "parted",
        "diskutil",
        "format",
        "diskpart",
        "dd",
        "chown",
        "crontab",
        "systemctl",
        "launchctl",
        "passwd",
        "useradd",
        "userdel",
        "visudo",
    }
)
NETWORK = frozenset(
    {
        "curl",
        "wget",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "nc",
        "ncat",
        "netcat",
        "telnet",
        "ftp",
        "ping",
        "nslookup",
        "dig",
        "socat",
        "aria2c",
    }
)
NETWORK_SUBCOMMANDS: Mapping[str, frozenset[str]] = {
    "git": frozenset({"push", "pull", "fetch", "clone", "remote", "submodule", "ls-remote"}),
    "pip": frozenset({"install", "download", "wheel"}),
    "uv": frozenset({"add", "sync", "lock", "pip", "tool", "run"}),
    "npm": frozenset({"install", "i", "ci", "add", "publish", "update", "exec"}),
    "npx": frozenset({"*"}),
    "yarn": frozenset({"add", "install", "upgrade", "dlx"}),
    "pnpm": frozenset({"add", "install", "i", "update", "dlx"}),
    "cargo": frozenset({"install", "add", "update", "fetch", "publish"}),
    "go": frozenset({"get", "install", "mod"}),
    "docker": frozenset({"pull", "push", "run", "build", "login"}),
    "ollama": frozenset({"pull", "push"}),
}
SHELLS = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "fish",
        "dash",
        "ksh",
        "csh",
        "tcsh",
        "pwsh",
        "powershell",
        "cmd",
        "env",
        "xargs",
        "nohup",
        "eval",
        "exec",
        "time",
        "timeout",
        "nice",
    }
)
INLINE_CODE_FLAGS: Mapping[str, frozenset[str]] = {
    "python": frozenset({"-c"}),
    "python3": frozenset({"-c"}),
    "node": frozenset({"-e", "--eval"}),
    "perl": frozenset({"-e", "-E"}),
    "ruby": frozenset({"-e"}),
    "php": frozenset({"-r"}),
}
DESTRUCTIVE = frozenset(
    {"rm", "rmdir", "shred", "truncate", "mv", "chmod", "kill", "pkill", "killall"}
)
_OPERATOR_CHARS = ";|&<>"
SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "&>"})


def _exe(argv: Sequence[str]) -> str:
    name = PurePath(argv[0]).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        name = name.removesuffix(suffix)
    if name.startswith("python") and name[6:].replace(".", "").isdigit():
        return "python"
    return name


def parse_command(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        # punctuation_chars trennt ungequotete Operatoren (;|&<>) als eigene Tokens ab,
        # gequotete Argumente wie "a|b" bleiben unverändert.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=_OPERATOR_CHARS)
        lexer.whitespace_split = True
        try:
            argv = list(lexer)
        except ValueError as exc:
            raise ToolError(f"Befehl nicht parsebar: {exc}") from exc
        operators = [a for a in argv if a and set(a) <= set(_OPERATOR_CHARS)]
    else:
        argv = [str(c) for c in command]
        operators = [a for a in argv if a in SHELL_OPERATORS]
    if not argv:
        raise ToolError("Leerer Befehl")
    if operators or any(a.startswith(("$(", "`")) for a in argv):
        raise ToolError(
            "Shell-Operatoren werden nicht unterstützt (keine Shell). Führe Befehle einzeln aus "
            f"und nutze Tools statt Umleitungen. Gefunden: {operators or ['$(…)/`…`']}"
        )
    return argv


def classify(argv: Sequence[str], allowlist: Sequence[Sequence[str]]) -> frozenset[Permission]:
    exe = _exe(argv)
    args = [a.lower() for a in argv[1:]]
    if exe in BLOCKED or exe.startswith("mkfs"):
        raise ToolError(f"Befehl {exe!r} ist grundsätzlich gesperrt")
    if exe in NETWORK:
        return frozenset({Permission.EXECUTE, Permission.NETWORK})
    subcommands = NETWORK_SUBCOMMANDS.get(exe)
    if subcommands and ("*" in subcommands or (args and args[0] in subcommands)):
        return frozenset({Permission.EXECUTE, Permission.NETWORK})
    if (
        exe == "python"
        and args[:2] in (["-m", "pip"],)
        and len(args) > 2
        and args[2] in NETWORK_SUBCOMMANDS["pip"]
    ):
        return frozenset({Permission.EXECUTE, Permission.NETWORK})
    if exe in SHELLS or exe in DESTRUCTIVE:
        return frozenset({Permission.EXECUTE, Permission.DANGEROUS})
    if any(a in INLINE_CODE_FLAGS.get(exe, frozenset()) for a in args):
        return frozenset({Permission.EXECUTE, Permission.DANGEROUS})
    normalized = [exe, *argv[1:]]
    if any(tuple(normalized[: len(a)]) == tuple(a) for a in allowlist):
        return frozenset({Permission.EXECUTE_ALLOWLISTED})
    return frozenset({Permission.EXECUTE})


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, int]:
    """Liest bis EOF, behält aber höchstens ``limit`` Bytes (Rest wird verworfen, damit der
    Prozess nicht an einer vollen Pipe hängen bleibt)."""
    kept = bytearray()
    total = 0
    while chunk := await stream.read(65536):
        total += len(chunk)
        if len(kept) < limit:
            kept += chunk[: limit - len(kept)]
    return bytes(kept), total


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover – Windows
            proc.kill()


class ExecuteCommandTool(Tool):
    name = "execute_command"
    description = (
        "Run a program WITHOUT a shell (no pipes, redirects, &&). Pass the command as a list of "
        "arguments or a simple string. Non-interactive (no stdin). Prefer the dedicated file and "
        "git tools where possible. Some commands require user approval or are blocked."
    )
    permissions = frozenset({Permission.EXECUTE})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": ["array", "string"],
                "items": {"type": "string"},
                "description": 'e.g. ["python", "-m", "pytest", "-q"]',
            },
            "cwd": {"type": "string", "description": "Working directory (default: workspace)"},
            "timeout_s": {"type": "number", "minimum": 1},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        allowlist: Sequence[Sequence[str]] = DEFAULT_ALLOWLIST,
        default_timeout_s: float = 60.0,
        max_timeout_s: float = 600.0,
        max_output_bytes: int = 200_000,
    ) -> None:
        self.allowlist = [tuple(a) for a in allowlist]
        self.default_timeout_s = default_timeout_s
        self.max_timeout_s = max_timeout_s
        self.max_output_bytes = max_output_bytes

    def required_permissions(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> frozenset[Permission]:
        return classify(parse_command(args["command"]), self.allowlist)

    def describe(self, args: Mapping[str, Any]) -> str:
        try:
            argv = parse_command(args["command"])
        except ToolError:
            return f"execute_command {args.get('command')!r}"
        cwd = f" (in {args['cwd']})" if args.get("cwd") else ""
        return f"$ {shlex.join(argv)}{cwd}"

    def _timeout(self, args: Mapping[str, Any]) -> float:
        return min(float(args.get("timeout_s", self.default_timeout_s)), self.max_timeout_s)

    def timeout_for(self, args: Mapping[str, Any], ctx: ToolContext) -> float:
        return self._timeout(args) + 10.0  # eigener, präziser Timeout + Puffer fürs Aufräumen

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        argv = parse_command(args["command"])
        cwd = ctx.resolve(str(args.get("cwd", ".")))
        if not cwd.is_dir():
            raise ToolError(f"Arbeitsverzeichnis existiert nicht: {args.get('cwd')}")
        if _exe(argv) == "python":
            argv = [sys.executable, *argv[1:]]
        timeout = self._timeout(args)
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=safe_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise ToolError(f"Programm nicht gefunden: {argv[0]}") from None
        except PermissionError:
            raise ToolError(f"Programm nicht ausführbar: {argv[0]}") from None

        assert proc.stdout is not None
        timed_out = False
        try:
            data, total = await asyncio.wait_for(
                _read_limited(proc.stdout, self.max_output_bytes), timeout
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            timed_out = True
            _kill_group(proc)
            await proc.wait()
            data, total = b"", 0
        duration = round(time.perf_counter() - started, 3)
        text = data.decode("utf-8", errors="replace")
        meta: dict[str, Any] = {
            "command": argv[:1] + list(argv[1:]),
            "cwd": ctx.display(cwd),
            "exit_code": None if timed_out else proc.returncode,
            "duration_s": duration,
            "output_bytes": total,
            "truncated": total > self.max_output_bytes,
        }
        if timed_out:
            return ToolResult.fail(
                f"Zeitlimit {timeout:.0f}s überschritten – Prozess beendet", **meta
            )
        if total > self.max_output_bytes:
            text += f"\n…[Ausgabe gekürzt: {total} Bytes gesamt]"
        if proc.returncode == 0:
            return ToolResult.ok(text, **meta)
        # Bei Fehlschlag ist die Ausgabe (z. B. Testfehler) für die Analyse entscheidend.
        return ToolResult.fail(f"Exit-Code {proc.returncode}", stdout=text, **meta)
