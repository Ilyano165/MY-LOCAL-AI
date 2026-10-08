"""Gemeinsame, kontrollierte Prozessausführung (Terminal-Tool, Verification Engine).

Ohne Shell, ohne stdin, Umgebung ohne Secrets, eigene Prozessgruppe (Timeout beendet auch
Kindprozesse), Ausgabe wird begrenzt eingelesen, ohne dass der Prozess an einer vollen
Pipe hängen bleibt.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

from tools.base import ToolError, safe_environment


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    exit_code: int | None
    """``None`` bei Timeout."""
    output: str
    output_bytes: int
    truncated: bool
    timed_out: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, int]:
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


def resolve_python(argv: Sequence[str]) -> list[str]:
    """``python``/``python3`` → aktueller Interpreter (gleiche venv wie NOVA)."""
    name = PurePath(argv[0]).name.lower() if argv else ""
    if name in ("python", "python3") or (
        name.startswith("python") and name[6:].replace(".", "").isdigit()
    ):
        return [sys.executable, *argv[1:]]
    return list(argv)


async def run_process(
    argv: Sequence[str],
    cwd: Path,
    *,
    timeout_s: float,
    max_output_bytes: int = 200_000,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Führt ``argv`` aus. Wirft :class:`ToolError`, wenn das Programm nicht startbar ist."""
    command = resolve_python(argv)
    if not command:
        raise ToolError("Leerer Befehl")
    started = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=dict(env) if env is not None else safe_environment(),
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
            _read_limited(proc.stdout, max_output_bytes), timeout_s
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        timed_out = True
        _kill_group(proc)
        await proc.wait()
        data, total = b"", 0
    text = data.decode("utf-8", errors="replace")
    if total > max_output_bytes:
        text += f"\n…[Ausgabe gekürzt: {total} Bytes gesamt]"
    return ProcessResult(
        argv=tuple(command),
        exit_code=None if timed_out else proc.returncode,
        output=text,
        output_bytes=total,
        truncated=total > max_output_bytes,
        timed_out=timed_out,
        duration_s=round(time.perf_counter() - started, 3),
    )
