"""execute_command: Parsing, Klassifizierung, Berechtigungen, Grenzen."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.tools.conftest import Harness
from tools import Permission, ToolContext, ToolError
from tools.terminal import DEFAULT_ALLOWLIST, ExecuteCommandTool, classify, parse_command

PY = sys.executable


def perms(command: str | list[str]) -> set[str]:
    return {p.value for p in classify(parse_command(command), DEFAULT_ALLOWLIST)}


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python -m pytest -q", {"execute_allowlisted"}),
        ("pytest tests/", {"execute_allowlisted"}),
        ("ruff check .", {"execute_allowlisted"}),
        ("ls -la", {"execute_allowlisted"}),
        ("python3.12 -m pytest", {"execute_allowlisted"}),
        ("make build", {"execute"}),
        ("python script.py", {"execute"}),
        ("ruff format .", {"execute"}),  # nur --check ist allowlisted
        ("curl https://example.com", {"execute", "network"}),
        ("git push origin main", {"execute", "network"}),
        ("pip install requests", {"execute", "network"}),
        ("python -m pip install x", {"execute", "network"}),
        ("npx something", {"execute", "network"}),
        ("bash -c 'ls'", {"execute", "dangerous"}),
        ("rm -rf build", {"execute", "dangerous"}),
        ("python -c 'print(1)'", {"execute", "dangerous"}),
        ("node -e '1'", {"execute", "dangerous"}),
        ("env FOO=1 ls", {"execute", "dangerous"}),
    ],
)
def test_classification(command: str, expected: set[str]) -> None:
    assert perms(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "sudo ls",
        "/usr/bin/sudo -i",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero",
        "shutdown now",
        "SUDO.exe ls",
    ],
)
def test_blocked_commands(command: str) -> None:
    with pytest.raises(ToolError, match="gesperrt"):
        perms(command)


@pytest.mark.parametrize(
    "command",
    [
        "ls | grep x",
        "make && make install",
        "echo hi > f",
        "cat a; rm b",
        ["ls", "$(whoami)"],
        "",
        "echo 'unclosed",
    ],
)
def test_shell_syntax_rejected(command: str | list[str]) -> None:
    with pytest.raises(ToolError):
        parse_command(command)


async def test_allowlisted_command_runs_and_returns_structured_result(
    harness: Harness, ws: Path
) -> None:
    (ws / "a.txt").write_text("x")
    res = await harness.call("execute_command", command=["ls"])
    assert res.success and res.output == "a.txt\n" and res.error is None
    meta = res.metadata
    assert meta["exit_code"] == 0 and meta["cwd"] == "." and meta["decision"] == "allow"
    assert meta["permissions"] == ["execute_allowlisted"]


async def test_failing_command_keeps_output_in_metadata(harness: Harness, ws: Path) -> None:
    (ws / "test_x.py").write_text("def test_x():\n    assert 1 == 2\n")
    res = await harness.call("execute_command", command="python -m pytest -q -p no:cacheprovider")
    assert not res.success and res.output is None and res.error == "Exit-Code 1"
    assert "1 failed" in res.metadata["stdout"] and res.metadata["exit_code"] == 1


async def test_unlisted_command_needs_approval(
    make_harness: Callable[..., Harness], ws: Path
) -> None:
    script = ws / "hello.py"
    script.write_text("print('hallo')")

    no_channel = make_harness()
    res = await no_channel.call("execute_command", command=["python", "hello.py"])
    assert not res.success and "Bestätigung erforderlich" in (res.error or "")

    denied = make_harness(approve=False)
    res = await denied.call(
        "execute_command", reason="Skript testen", command=["python", "hello.py"]
    )
    assert not res.success and "abgelehnt" in (res.error or "")
    request = denied.approvals[0]
    assert request.invocation.reason == "Skript testen"
    assert request.description.startswith("$ python hello.py")
    assert request.permissions == {Permission.EXECUTE}

    approved = make_harness(approve=True)
    res = await approved.call("execute_command", command=["python", "hello.py"])
    assert res.success and res.output == "hallo\n"


async def test_network_denied_even_with_approval(make_harness: Callable[..., Harness]) -> None:
    harness = make_harness(approve=True)
    res = await harness.call("execute_command", command="curl https://example.com")
    assert not res.success and "Berechtigungsrichtlinie" in (res.error or "")
    assert harness.approvals == []  # DENY fragt gar nicht erst


async def test_timeout_kills_process_group(make_harness: Callable[..., Harness], ws: Path) -> None:
    (ws / "slow.py").write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "time.sleep(30)\n"
    )
    harness = make_harness(approve=True)
    started = time.monotonic()
    res = await harness.call("execute_command", command=["python", "slow.py"], timeout_s=1)
    assert time.monotonic() - started < 10
    assert (
        not res.success and "Zeitlimit" in (res.error or "") and res.metadata["exit_code"] is None
    )


async def test_output_limit(make_harness: Callable[..., Harness], ws: Path) -> None:
    from tools import ToolRegistry
    from tools.registry import MemoryAuditLog

    (ws / "loud.py").write_text("print('x' * 100_000)")
    harness = make_harness(approve=True)
    harness.registry = ToolRegistry(
        [ExecuteCommandTool(max_output_bytes=1000)],
        approval=harness.registry.approval,
        audit=MemoryAuditLog(),
    )
    res = await harness.call("execute_command", command=["python", "loud.py"])
    assert res.success and res.metadata["truncated"] and res.metadata["output_bytes"] > 100_000
    assert len(res.output or "") < 1200


async def test_cwd_must_be_allowed(harness: Harness, ws: Path) -> None:
    (ws / "sub").mkdir()
    (ws / "sub" / "f.txt").write_text("")
    ok = await harness.call("execute_command", command=["ls"], cwd="sub")
    assert ok.output == "f.txt\n" and ok.metadata["cwd"] == "sub"
    for cwd, fragment in (("..", "außerhalb"), ("fehlt", "existiert nicht")):
        res = await harness.call("execute_command", command=["ls"], cwd=cwd)
        assert fragment in (res.error or "")


async def test_no_stdin_and_secrets_not_inherited(
    make_harness: Callable[..., Harness], ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVA_SECRET_TOKEN", "geheim")
    (ws / "probe.py").write_text(
        "import os, sys\n"
        "print(os.environ.get('NOVA_SECRET_TOKEN', 'none'))\n"
        "print(repr(sys.stdin.read()))\n"
    )
    res = await make_harness(approve=True).call("execute_command", command=["python", "probe.py"])
    assert res.output == "none\n''\n"


async def test_missing_program(make_harness: Callable[..., Harness]) -> None:
    res = await make_harness(approve=True).call("execute_command", command=["nova-gibts-nicht-123"])
    assert not res.success and "nicht gefunden" in (res.error or "")


def test_timeout_is_capped(ws: Path) -> None:
    tool = ExecuteCommandTool(default_timeout_s=60, max_timeout_s=5)
    assert tool._timeout({"timeout_s": 9999}) == 5
    assert tool._timeout({}) == 5
    # Registry-Timeout = eigenes Limit + Puffer zum Aufräumen der Prozessgruppe
    assert tool.timeout_for({}, ToolContext(ws)) == 15


def test_quoted_operators_are_plain_arguments() -> None:
    assert parse_command('grep -E "a|b" file.txt') == ["grep", "-E", "a|b", "file.txt"]
    assert parse_command("echo 'x; y'") == ["echo", "x; y"]
