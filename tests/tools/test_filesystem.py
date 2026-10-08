"""list_directory, read_file, write_file, edit_file, search_files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.tools.conftest import Harness
from tools import ToolContext
from tools.base import sha256_text
from tools.registry import PermissionPolicy

# --------------------------------------------------------------------------- list_directory


async def test_list_directory_flat_and_recursive(harness: Harness, ws: Path) -> None:
    (ws / "src" / "pkg").mkdir(parents=True)
    (ws / "src" / "pkg" / "mod.py").write_text("x")
    (ws / "README.md").write_text("hi")
    (ws / ".hidden").write_text("h")
    (ws / "node_modules" / "dep").mkdir(parents=True)

    flat = await harness.call("list_directory")
    assert flat.success and flat.output == "node_modules/\nsrc/\nREADME.md (2 B)"
    assert flat.metadata["entries"] == 3

    tree = await harness.call("list_directory", recursive=True, max_depth=3)
    assert "    mod.py (1 B)" in (tree.output or "")
    assert "dep/" not in (tree.output or "")  # node_modules wird nicht betreten

    hidden = await harness.call("list_directory", include_hidden=True)
    assert ".hidden" in (hidden.output or "")


async def test_list_directory_limits_and_errors(ws: Path) -> None:
    from tools import ListDirectoryTool, ToolRegistry

    for i in range(5):
        (ws / f"f{i}.txt").write_text("")
    reg = ToolRegistry([ListDirectoryTool(max_entries=3)])
    from tools import ToolInvocation

    res = await reg.execute(ToolInvocation("list_directory", {}, reason="x"), ToolContext(ws))
    assert res.metadata["truncated"] and res.metadata["entries"] == 3
    harness = Harness(ws)
    assert not (await harness.call("list_directory", path="f0.txt")).success
    assert not (await harness.call("list_directory", path="../")).success


# --------------------------------------------------------------------------- read_file


async def test_read_file_full_and_range(harness: Harness, ws: Path) -> None:
    content = "".join(f"zeile {i}\n" for i in range(1, 13))
    (ws / "a.txt").write_text(content)
    full = await harness.call("read_file", path="a.txt")
    assert full.success and (full.output or "").startswith(" 1| zeile 1")
    assert full.metadata["total_lines"] == 12 and full.metadata["sha256"] == sha256_text(content)

    part = await harness.call("read_file", path="a.txt", start_line=10, end_line=11)
    assert part.output == "10| zeile 10\n11| zeile 11"
    clipped = await harness.call("read_file", path="a.txt", start_line=11, end_line=99)
    assert clipped.metadata["end_line"] == 12


@pytest.mark.parametrize(
    ("setup", "args", "fragment"),
    [
        (None, {"path": "fehlt.txt"}, "nicht gefunden"),
        ("bin", {"path": "bin.dat"}, "Binärdatei"),
        ("latin1", {"path": "latin.txt"}, "UTF-8"),
        ("text", {"path": "a.txt", "start_line": 5}, "Zeilenbereich"),
        ("text", {"path": "a.txt", "start_line": 2, "end_line": 1}, "Zeilenbereich"),
        (None, {"path": "../../etc/passwd"}, "außerhalb"),
        ("env", {"path": ".env"}, "gesperrt"),
    ],
)
async def test_read_file_errors(
    harness: Harness, ws: Path, setup: str | None, args: dict[str, object], fragment: str
) -> None:
    (ws / "bin.dat").write_bytes(b"\x00\x01\x02")
    (ws / "latin.txt").write_bytes("Größe".encode("latin-1"))
    (ws / "a.txt").write_text("eins\nzwei\n")
    (ws / ".env").write_text("API_KEY=geheim")
    res = await harness.call("read_file", **args)
    assert not res.success and res.output is None and fragment in (res.error or "")


async def test_read_file_size_limit(ws: Path) -> None:
    from tools import ReadFileTool, ToolInvocation, ToolRegistry

    (ws / "big.txt").write_text("x" * 200)
    reg = ToolRegistry([ReadFileTool(max_bytes=100)])
    res = await reg.execute(
        ToolInvocation("read_file", {"path": "big.txt"}, reason="x"), ToolContext(ws)
    )
    assert not res.success and "zu groß" in (res.error or "")


async def test_read_empty_file(harness: Harness, ws: Path) -> None:
    (ws / "leer.txt").write_text("")
    res = await harness.call("read_file", path="leer.txt")
    assert res.success and res.metadata["total_lines"] == 0


# --------------------------------------------------------------------------- write_file


async def test_write_file_create_overwrite_and_backup(ws: Path, tmp_path: Path) -> None:
    harness = Harness(ws)
    ctx = ToolContext(ws, backup_dir=tmp_path / "backups")
    created = await harness.call("write_file", ctx=ctx, path="sub/dir/a.txt", content="eins")
    assert (
        created.success
        and created.metadata["created"] is True
        and created.metadata["backup"] is None
    )
    assert (ws / "sub/dir/a.txt").read_text() == "eins"

    replaced = await harness.call("write_file", ctx=ctx, path="sub/dir/a.txt", content="zwei")
    assert replaced.metadata["created"] is False
    backup = Path(replaced.metadata["backup"])
    assert backup.read_text() == "eins" and backup.is_relative_to(tmp_path / "backups")
    assert replaced.metadata["sha256"] == sha256_text("zwei")
    assert not list(ws.rglob("*.nova-tmp"))  # atomar, keine Reste


async def test_write_file_refusals(harness: Harness, ws: Path) -> None:
    (ws / "d").mkdir()
    (ws / "exists.txt").write_text("x")
    cases = [
        ({"path": "d", "content": ""}, "Verzeichnis"),
        ({"path": "exists.txt", "content": "y", "overwrite": False}, "existiert bereits"),
        ({"path": ".git/hooks/pre-commit", "content": "evil"}, ".git"),
        ({"path": "../escape.txt", "content": "x"}, "außerhalb"),
        ({"path": "id_rsa", "content": "x"}, "gesperrt"),
    ]
    for args, fragment in cases:
        res = await harness.call("write_file", **args)
        assert not res.success and fragment in (res.error or ""), args
    assert (ws / "exists.txt").read_text() == "x"
    assert not (ws.parent / "escape.txt").exists()


async def test_write_file_drops_stale_bytecode(harness: Harness, ws: Path) -> None:
    cache = ws / "__pycache__"
    cache.mkdir()
    (cache / "calc.cpython-313.pyc").write_bytes(b"stale")
    (cache / "calculator.cpython-313.pyc").write_bytes(b"keep")
    assert (await harness.call("write_file", path="calc.py", content="x = 1\n")).success
    assert not (cache / "calc.cpython-313.pyc").exists()
    assert (cache / "calculator.cpython-313.pyc").exists()


async def test_write_denied_by_read_only_policy(ws: Path) -> None:
    harness = Harness(ws, policy=PermissionPolicy.read_only())
    res = await harness.call("write_file", path="a.txt", content="x")
    assert not res.success and "Berechtigungsrichtlinie" in (res.error or "")
    assert not (ws / "a.txt").exists()


# --------------------------------------------------------------------------- edit_file


async def test_edit_file_unique_replacement_with_diff(ws: Path, tmp_path: Path) -> None:
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    harness = Harness(ws)
    ctx = ToolContext(ws, backup_dir=tmp_path / "bk")
    res = await harness.call(
        "edit_file", ctx=ctx, path="calc.py", old_text="return a - b", new_text="return a + b"
    )
    assert res.success and res.metadata["replacements"] == 1
    assert (ws / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert "-    return a - b" in (res.output or "") and "+    return a + b" in (res.output or "")
    assert Path(res.metadata["backup"]).read_text().endswith("a - b\n")


async def test_edit_file_ambiguity_and_replace_all(harness: Harness, ws: Path) -> None:
    (ws / "a.txt").write_text("x = 1\nx = 1\n")
    ambiguous = await harness.call("edit_file", path="a.txt", old_text="x = 1", new_text="x = 2")
    assert not ambiguous.success and "2x" in (ambiguous.error or "")
    assert (ws / "a.txt").read_text() == "x = 1\nx = 1\n"  # unverändert
    every = await harness.call(
        "edit_file", path="a.txt", old_text="x = 1", new_text="x = 2", replace_all=True
    )
    assert every.metadata["replacements"] == 2 and (ws / "a.txt").read_text() == "x = 2\nx = 2\n"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"path": "a.txt", "old_text": "", "new_text": "y"}, "nicht leer"),
        ({"path": "a.txt", "old_text": "same", "new_text": "same"}, "identisch"),
        ({"path": "a.txt", "old_text": "fehlt", "new_text": "y"}, "nicht gefunden"),
        ({"path": "a.txt", "old_text": "def  f():", "new_text": "y"}, "nicht gefunden"),
        ({"path": "neu.txt", "old_text": "x", "new_text": "y"}, "nicht gefunden"),
    ],
)
async def test_edit_file_errors(
    harness: Harness, ws: Path, args: dict[str, object], fragment: str
) -> None:
    (ws / "a.txt").write_text("def f():\n    pass\n")
    res = await harness.call("edit_file", **args)
    assert not res.success and fragment in (res.error or "")


async def test_edit_file_whitespace_hint(harness: Harness, ws: Path) -> None:
    (ws / "a.txt").write_text("    value = 1\n")
    res = await harness.call(
        "edit_file", path="a.txt", old_text="value = 1  ", new_text="value = 2"
    )
    assert "anderem Leerraum" in (res.error or "")


# --------------------------------------------------------------------------- search_files


@pytest.fixture
def corpus(ws: Path) -> Path:
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text("def main():\n    return TODO_fix()\n")
    (ws / "src" / "util.py").write_text("# todo: refactor\nx = 1\n")
    (ws / "notes.md").write_text("TODO in docs\n")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "lib.js").write_text("TODO vendor\n")
    (ws / "image.bin").write_bytes(b"TODO\x00\x00")
    (ws / ".env").write_text("TODO=secret\n")
    return ws


async def test_search_literal_case_insensitive(harness: Harness, corpus: Path) -> None:
    res = await harness.call("search_files", pattern="todo")
    out = (res.output or "").splitlines()
    assert out == [
        "notes.md:1: TODO in docs",
        "src/app.py:2: return TODO_fix()",
        "src/util.py:1: # todo: refactor",
    ]
    assert res.metadata["matches"] == 3 and res.metadata["files_skipped"] == 1  # Binärdatei


async def test_search_options(harness: Harness, corpus: Path) -> None:
    cs = await harness.call("search_files", pattern="TODO", case_sensitive=True, glob="*.py")
    assert cs.output == "src/app.py:2: return TODO_fix()"
    rx = await harness.call("search_files", pattern=r"^def \w+\(", regex=True, path="src")
    assert rx.output == "src/app.py:1: def main():"
    literal = await harness.call("search_files", pattern="fix()")  # Klammern literal
    assert literal.metadata["matches"] == 1
    limited = await harness.call("search_files", pattern="todo", max_results=1)
    assert limited.metadata["truncated"] and limited.metadata["matches"] == 1
    none = await harness.call("search_files", pattern="gibtsnicht")
    assert none.success and none.output == "Keine Treffer"


async def test_search_errors(harness: Harness, corpus: Path) -> None:
    assert "regulärer Ausdruck" in (
        (await harness.call("search_files", pattern="(", regex=True)).error or ""
    )
    assert not (await harness.call("search_files", pattern="")).success
    assert not (await harness.call("search_files", pattern="x", path="fehlt")).success
    assert not (await harness.call("search_files", pattern="x", path="/")).success


async def test_search_skips_symlinks_outside(harness: Harness, ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("NEEDLE")
    os.symlink(outside / "secret.txt", ws / "link.txt")
    res = await harness.call("search_files", pattern="NEEDLE")
    assert res.output == "Keine Treffer"
