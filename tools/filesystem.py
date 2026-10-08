"""Dateisystem-Tools: list_directory, read_file, write_file, edit_file, search_files.

Alle Pfade laufen über :meth:`ToolContext.resolve` (erlaubte Verzeichnisse, Sperrliste,
Symlink-Schutz, kein Schreiben in ``.git/``). Schreibende Tools sichern auf Wunsch den alten
Inhalt (``ToolContext.backup_dir``) und entfernen veralteten Python-Bytecode.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from tools.base import Permission, Tool, ToolContext, ToolError, ToolResult, sha256_text

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)
MAX_READ_BYTES = 1_000_000
MAX_SEARCH_FILE_BYTES = 2_000_000


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _read_text(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise ToolError(
            f"Datei zu groß ({size} Bytes > {max_bytes}); "
            "nutze start_line/end_line oder search_files"
        )
    data = path.read_bytes()
    if _is_binary(data[:8192]):
        raise ToolError("Binärdatei – nur Textdateien werden gelesen")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("Keine UTF-8-Textdatei") from exc


def _drop_stale_bytecode(path: Path) -> None:
    # Python erkennt Änderungen gleicher Größe innerhalb derselben Sekunde sonst nicht.
    if path.suffix == ".py":
        for stale in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
            stale.unlink(missing_ok=True)


def _backup(path: Path, ctx: ToolContext) -> str | None:
    if ctx.backup_dir is None or not path.is_file():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = ctx.backup_dir / stamp / ctx.display(path).lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return str(target)


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.nova-tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- list_directory


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = (
        "List files and directories. Directories end with '/'. Use recursive=true with "
        "max_depth to see a tree; dependency/cache folders are listed but not descended into."
    )
    permissions = frozenset({Permission.READ})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory (default: workspace root)"},
            "recursive": {"type": "boolean"},
            "max_depth": {"type": "integer", "minimum": 1, "maximum": 10},
            "include_hidden": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.resolve(str(args.get("path", ".")))
        if not root.is_dir():
            raise ToolError(f"Kein Verzeichnis: {args.get('path', '.')}")
        depth = int(args.get("max_depth", 3)) if args.get("recursive") else 1
        hidden = bool(args.get("include_hidden", False))
        return await asyncio.to_thread(self._list, root, depth, hidden, ctx)

    def _list(self, root: Path, max_depth: int, hidden: bool, ctx: ToolContext) -> ToolResult:
        lines: list[str] = []
        count = 0
        truncated = False

        def walk(directory: Path, depth: int) -> None:
            nonlocal count, truncated
            try:
                children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError:
                lines.append(f"{'  ' * (depth - 1)}[kein Zugriff]")
                return
            for child in children:
                if not hidden and child.name.startswith("."):
                    continue
                if count >= self.max_entries:
                    truncated = True
                    return
                count += 1
                indent = "  " * (depth - 1)
                if child.is_dir():
                    lines.append(f"{indent}{child.name}/")
                    descend = child.name not in SKIP_DIRS and not child.is_symlink()
                    if depth < max_depth and descend:
                        walk(child, depth + 1)
                else:
                    lines.append(f"{indent}{child.name} ({child.stat().st_size} B)")

        walk(root, 1)
        return ToolResult.ok(
            "\n".join(lines) if lines else "(leer)",
            path=ctx.display(root),
            entries=count,
            truncated=truncated,
        )


# --------------------------------------------------------------------------- read_file


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file. Optionally only lines start_line..end_line (1-based, inclusive). "
        "Output lines are prefixed with their line number and '| '."
    )
    permissions = frozenset({Permission.READ})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, max_bytes: int = MAX_READ_BYTES) -> None:
        self.max_bytes = max_bytes

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(str(args["path"]))
        if not path.is_file():
            raise ToolError(f"Datei nicht gefunden: {args['path']}")
        content = await asyncio.to_thread(_read_text, path, self.max_bytes)
        lines = content.splitlines()
        start = int(args.get("start_line", 1))
        end = min(int(args.get("end_line", len(lines))), len(lines))
        if lines and (start > len(lines) or end < start):
            raise ToolError(
                f"Ungültiger Zeilenbereich {start}..{args.get('end_line', end)} "
                f"(Datei hat {len(lines)} Zeilen)"
            )
        width = len(str(max(end, 1)))
        numbered = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))
        return ToolResult.ok(
            numbered if lines else "(leere Datei)",
            path=ctx.display(path),
            sha256=sha256_text(content),
            size_bytes=len(content.encode("utf-8")),
            total_lines=len(lines),
            start_line=start if lines else 0,
            end_line=end,
        )


# --------------------------------------------------------------------------- write_file


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create a file or replace its ENTIRE content. For partial changes use edit_file."
    permissions = frozenset({Permission.WRITE})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Complete new file content"},
            "overwrite": {"type": "boolean", "description": "Default true"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def describe(self, args: Mapping[str, Any]) -> str:
        return f"write_file {args.get('path')!r} ({len(str(args.get('content', '')))} Zeichen)"

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(str(args["path"]), write=True)
        content = str(args["content"])
        if path.is_dir():
            raise ToolError(f"{args['path']} ist ein Verzeichnis")
        existed = path.exists()
        if existed and args.get("overwrite") is False:
            raise ToolError(f"{args['path']} existiert bereits (overwrite=false)")

        def _write() -> str | None:
            backup = _backup(path, ctx)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, content)
            _drop_stale_bytecode(path)
            return backup

        backup = await asyncio.to_thread(_write)
        size = len(content.encode("utf-8"))
        rel = ctx.display(path)
        return ToolResult.ok(
            f"{'Überschrieben' if existed else 'Angelegt'}: {rel} ({size} Bytes)",
            path=rel,
            sha256=sha256_text(content),
            bytes=size,
            created=not existed,
            backup=backup,
        )


# --------------------------------------------------------------------------- edit_file


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact text snippet in a file. old_text must match exactly (including "
        "whitespace and indentation) and be unique unless replace_all=true. Read the file first."
    )
    permissions = frozenset({Permission.WRITE})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(self, max_bytes: int = MAX_READ_BYTES, max_diff_chars: int = 4000) -> None:
        self.max_bytes = max_bytes
        self.max_diff_chars = max_diff_chars

    def describe(self, args: Mapping[str, Any]) -> str:
        return f"edit_file {args.get('path')!r}"

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(str(args["path"]), write=True)
        old, new = str(args["old_text"]), str(args["new_text"])
        if not old:
            raise ToolError("old_text darf nicht leer sein (neue Dateien: write_file)")
        if old == new:
            raise ToolError("old_text und new_text sind identisch")
        if not path.is_file():
            raise ToolError(f"Datei nicht gefunden: {args['path']}")
        content = await asyncio.to_thread(_read_text, path, self.max_bytes)
        count = content.count(old)
        if count == 0:
            hint = ""
            if old.strip() and old.strip() in content:
                hint = " (Text existiert mit anderem Leerraum – exakt übernehmen)"
            raise ToolError(f"old_text nicht gefunden in {args['path']}{hint}")
        replace_all = bool(args.get("replace_all"))
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_text kommt {count}x vor – mehr Kontext angeben oder replace_all=true"
            )
        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)

        def _write() -> str | None:
            backup = _backup(path, ctx)
            _write_atomic(path, updated)
            _drop_stale_bytecode(path)
            return backup

        backup = await asyncio.to_thread(_write)
        rel = ctx.display(path)
        diff = "".join(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        if len(diff) > self.max_diff_chars:
            diff = diff[: self.max_diff_chars] + "\n…[Diff gekürzt]"
        replacements = count if replace_all else 1
        return ToolResult.ok(
            f"{replacements} Ersetzung(en) in {rel}\n{diff}",
            path=rel,
            sha256=sha256_text(updated),
            replacements=replacements,
            backup=backup,
        )


# --------------------------------------------------------------------------- search_files


class SearchFilesTool(Tool):
    name = "search_files"
    description = (
        "Search text in files below a directory. Returns 'path:line: text'. Literal search by "
        "default, regex=true for regular expressions; glob filters file names (e.g. '*.py')."
    )
    permissions = frozenset({Permission.READ})
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "regex": {"type": "boolean"},
            "case_sensitive": {"type": "boolean"},
            "glob": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        if not pattern:
            raise ToolError("pattern darf nicht leer sein")
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        source = pattern if args.get("regex") else re.escape(pattern)
        try:
            compiled = re.compile(source, flags)
        except re.error as exc:
            raise ToolError(f"Ungültiger regulärer Ausdruck: {exc}") from exc
        root = ctx.resolve(str(args.get("path", ".")))
        if not root.exists():
            raise ToolError(f"Pfad nicht gefunden: {args.get('path', '.')}")
        glob = args.get("glob")
        limit = int(args.get("max_results", 100))
        return await asyncio.to_thread(self._search, compiled, root, glob, limit, ctx)

    @staticmethod
    def _files(root: Path, glob: str | None, ctx: ToolContext) -> Iterator[Path]:
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                path = Path(dirpath) / name
                try:
                    ctx.resolve(str(path))  # Sperrliste / Symlinks nach außen
                except ToolError:
                    continue
                yield path

    def _search(
        self, compiled: re.Pattern[str], root: Path, glob: str | None, limit: int, ctx: ToolContext
    ) -> ToolResult:
        lines: list[str] = []
        scanned = skipped = 0
        truncated = False
        for path in self._files(root, glob, ctx):
            try:
                if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    skipped += 1
                    continue
                data = path.read_bytes()
            except OSError:
                skipped += 1
                continue
            if _is_binary(data[:8192]):
                skipped += 1
                continue
            scanned += 1
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if compiled.search(line):
                    if len(lines) >= limit:
                        truncated = True
                        break
                    lines.append(f"{ctx.display(path)}:{number}: {line.strip()[:300]}")
            if truncated:
                break
        return ToolResult.ok(
            "\n".join(lines) if lines else "Keine Treffer",
            matches=len(lines),
            files_scanned=scanned,
            files_skipped=skipped,
            truncated=truncated,
        )


def filesystem_tools() -> list[Tool]:
    return [ListDirectoryTool(), ReadFileTool(), WriteFileTool(), EditFileTool(), SearchFilesTool()]
