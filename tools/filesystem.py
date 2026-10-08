"""Dateisystem-Tools, strikt auf den Workspace beschränkt (siehe docs/security.md §3)."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from tools.base import Tool, ToolContext, ToolError, ToolOutput


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Löst ``relative`` im Workspace auf; Ausbruch (``..``, absolute Pfade, Symlinks) → Fehler."""
    root = workspace.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError(f"Pfad {relative!r} liegt außerhalb des Workspace")
    return candidate


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to workspace"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        path = resolve_in_workspace(ctx.workspace, str(args["path"]))
        if not path.is_file():
            raise ToolError(f"Datei nicht gefunden: {args['path']}")
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"Keine UTF-8-Textdatei: {args['path']}") from exc
        return ToolOutput(
            ok=True,
            output=content,
            data={"path": str(args["path"]), "sha256": sha256_text(content), "size": len(content)},
        )


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file in the workspace."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace"},
            "content": {"type": "string", "description": "Full new file content"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        rel = str(args["path"])
        content = str(args["content"])
        path = resolve_in_workspace(ctx.workspace, rel)
        if path.is_dir():
            raise ToolError(f"{rel} ist ein Verzeichnis")

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if path.suffix == ".py":
                # Veralteten Bytecode entfernen: Python erkennt Änderungen gleicher Größe innerhalb
                # derselben Sekunde sonst nicht und würde den alten Code ausführen.
                for stale in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
                    stale.unlink(missing_ok=True)

        await asyncio.to_thread(_write)
        return ToolOutput(
            ok=True,
            output=f"{len(content.encode('utf-8'))} Bytes nach {rel} geschrieben",
            data={"path": rel, "sha256": sha256_text(content), "bytes": len(content.encode())},
        )


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories in a workspace directory."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory, default '.'"}},
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        rel = str(args.get("path", "."))
        path = resolve_in_workspace(ctx.workspace, rel)
        if not path.is_dir():
            raise ToolError(f"Verzeichnis nicht gefunden: {rel}")
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in path.iterdir() if p.name != ".git"
        )
        return ToolOutput(ok=True, output="\n".join(entries), data={"entries": entries})


def default_tools() -> list[Tool]:
    return [ReadFileTool(), WriteFileTool(), ListDirTool()]
