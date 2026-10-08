from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tools.base import Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, validate_arguments
from tools.filesystem import (
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    default_tools,
    resolve_in_workspace,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer"},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "flag": {"type": "boolean"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_validate_arguments() -> None:
    assert validate_arguments(SCHEMA, {"path": "x", "count": 2, "mode": "a", "flag": True}) == []
    problems = validate_arguments(SCHEMA, {"count": True, "mode": "z", "extra": 1})
    assert "Pflichtargument fehlt: path" in problems
    assert any("count" in p and "integer" in p for p in problems)  # bool ist kein integer
    assert any("mode" in p for p in problems)
    assert "Unbekanntes Argument: extra" in problems
    assert validate_arguments(SCHEMA, "nope") == ["Argumente müssen ein JSON-Objekt sein"]  # type: ignore[arg-type]


class _Behaving(Tool):
    name = "behave"
    description = "test tool"
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"how": {"type": "string"}},
        "required": ["how"],
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        how = args["how"]
        if how == "slow":
            await asyncio.sleep(1)
        if how == "tool_error":
            raise ToolError("fachlicher Fehler")
        if how == "crash":
            raise RuntimeError("bug")
        if how == "big":
            return ToolOutput(ok=True, output="x" * 500)
        return ToolOutput(ok=True, output="fine", data={"k": 1})


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, timeout_s=0.1)


async def test_registry_success_and_failures(ctx: ToolContext) -> None:
    reg = ToolRegistry([_Behaving()], max_output_chars=100)
    ok = await reg.execute("behave", {"how": "ok"}, ctx)
    assert ok.ok and ok.output == "fine" and ok.data == {"k": 1}

    for how, fragment in [
        ("slow", "Zeitlimit"),
        ("tool_error", "fachlicher Fehler"),
        ("crash", "RuntimeError"),
    ]:
        res = await reg.execute("behave", {"how": how}, ctx)
        assert not res.ok and fragment in (res.error or "")

    big = await reg.execute("behave", {"how": "big"}, ctx)
    assert big.truncated and len(big.output) < 200

    unknown = await reg.execute("nope", {}, ctx)
    assert not unknown.ok and "Unbekanntes Tool" in (unknown.error or "")
    invalid = await reg.execute("behave", {}, ctx)
    assert not invalid.ok and "Pflichtargument" in (invalid.error or "")


def test_registry_rejects_duplicates_and_lists_specs() -> None:
    reg = ToolRegistry(default_tools())
    with pytest.raises(ValueError):
        reg.register(ReadFileTool())
    assert reg.names() == ["list_dir", "read_file", "write_file"]
    assert [s.name for s in reg.specs(["write_file", "ghost"])] == ["write_file"]
    with pytest.raises(KeyError):
        reg.get("ghost")


async def test_write_read_list(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path)
    reg = ToolRegistry(default_tools())
    written = await reg.execute("write_file", {"path": "pkg/a.py", "content": "x = 1\n"}, ctx)
    assert written.ok and (tmp_path / "pkg" / "a.py").read_text() == "x = 1\n"
    read = await reg.execute("read_file", {"path": "pkg/a.py"}, ctx)
    assert read.output == "x = 1\n" and read.data["sha256"] == written.data["sha256"]
    listing = await reg.execute("list_dir", {"path": "pkg"}, ctx)
    assert listing.output == "a.py"
    missing = await reg.execute("read_file", {"path": "nope.txt"}, ctx)
    assert not missing.ok and "nicht gefunden" in (missing.error or "")


@pytest.mark.parametrize("bad", ["../outside.txt", "/etc/passwd", "a/../../x"])
def test_workspace_escape_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ToolError, match="außerhalb"):
        resolve_in_workspace(tmp_path, bad)


async def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside)
    res = await ToolRegistry([ReadFileTool()]).execute(
        "read_file", {"path": "link/secret.txt"}, ToolContext(workspace)
    )
    assert not res.ok and "außerhalb" in (res.error or "")


async def test_write_to_directory_and_binary_read_fail(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00")
    reg = ToolRegistry([WriteFileTool(), ReadFileTool(), ListDirTool()])
    ctx = ToolContext(tmp_path)
    assert not (await reg.execute("write_file", {"path": "d", "content": ""}, ctx)).ok
    assert not (await reg.execute("read_file", {"path": "bin.dat"}, ctx)).ok
    assert not (await reg.execute("list_dir", {"path": "bin.dat"}, ctx)).ok


async def test_writing_python_file_removes_its_stale_bytecode(tmp_path: Path) -> None:
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    own = cache / "calc.cpython-313.pyc"
    other = cache / "calculator.cpython-313.pyc"
    own.write_bytes(b"stale")
    other.write_bytes(b"keep")
    res = await ToolRegistry([WriteFileTool()]).execute(
        "write_file", {"path": "calc.py", "content": "x = 1\n"}, ToolContext(tmp_path)
    )
    assert res.ok
    assert not own.exists() and other.exists()
