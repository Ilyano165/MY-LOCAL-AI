"""Git-Lesetools: git_status, git_diff, git_log.

Sicherheit: Auch „lesende“ Git-Befehle können über die Konfiguration eines Repos Code
ausführen (``core.fsmonitor``, ``diff.external``, Textconv, ``filter.<x>.clean`` …;
vgl. CVE-2026-45033). Zwei Schutzebenen:

1. Jeder Aufruf erzwingt sichere Overrides (``-c core.fsmonitor=false``,
   ``safe.bareRepository=explicit``, ``--no-ext-diff``, ``--no-textconv`` …).
2. Vor jedem Befehl wird die repo-lokale Konfiguration gelesen (reines Lesen, führt nichts
   aus). Enthält sie ausführbare Einträge, verweigern die Tools die Arbeit, statt zu hoffen,
   dass die Overrides alles abdecken (``filter.*`` lässt sich nicht pauschal abschalten).

Zusätzlich muss die Repository-Wurzel in einem erlaubten Verzeichnis liegen.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from tools.base import Permission, Tool, ToolContext, ToolError, ToolResult, safe_environment

SAFE_OVERRIDES: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    "core.pager=cat",
    "-c",
    "safe.bareRepository=explicit",
    "-c",
    "color.ui=false",
    "-c",
    "core.quotePath=false",
)
_DANGEROUS_KEY = re.compile(
    r"^(core\.fsmonitor|core\.hookspath|core\.pager|core\.editor|core\.sshcommand|core\.gitproxy"
    r"|core\.askpass|diff\.external|diff\..+\.(textconv|command)|filter\..+\.(clean|smudge|process)"
    r"|merge\..+\.driver|pager\..+|sequence\.editor|gpg\..*program|credential\..*helper)$",
    re.IGNORECASE,
)
_BOOLEAN = frozenset({"true", "false", "yes", "no", "on", "off", "1", "0", ""})
_REPO_SCOPES = frozenset({"local", "worktree"})
_STRIP_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_EXTERNAL_DIFF",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_ASKPASS",
    "GIT_EDITOR",
    "GIT_OBJECT_DIRECTORY",
)


def dangerous_config(entries: Sequence[tuple[str, str, str]]) -> list[str]:
    """``entries``: (scope, key, value). Liefert repo-lokale Schlüssel, die Befehle ausführen."""
    found = []
    for scope, key, value in entries:
        if scope not in _REPO_SCOPES or not _DANGEROUS_KEY.match(key):
            continue
        if key.lower() == "core.fsmonitor" and value.strip().lower() in _BOOLEAN:
            continue  # true/false = eingebauter Daemon bzw. aus, kein Fremdbefehl
        if key.lower() == "core.hookspath":
            continue  # Hooks laufen bei Lesebefehlen nicht; wird zudem überschrieben
        found.append(key)
    return found


def parse_config_z(raw: str) -> list[tuple[str, str, str]]:
    """Parst ``git config --list --show-scope -z``: ``scope\\0key\\nvalue\\0``."""
    parts = raw.split("\0")
    entries = []
    for scope, item in zip(parts[0::2], parts[1::2], strict=False):
        key, _, value = item.partition("\n")
        if key:
            entries.append((scope, key, value))
    return entries


class _GitTool(Tool):
    permissions = frozenset({Permission.READ})
    timeout_s: ClassVar[float] = 30.0

    def __init__(self, max_output_chars: int = 50_000) -> None:
        self.max_output_chars = max_output_chars
        self._last_error = ""

    def timeout_for(self, args: Mapping[str, Any], ctx: ToolContext) -> float:
        return self.timeout_s * 3 + 5  # bis zu drei Git-Aufrufe

    @staticmethod
    def _env() -> dict[str, str]:
        env = safe_environment({"GIT_OPTIONAL_LOCKS": "0", "GIT_PAGER": "cat", "LC_ALL": "C"})
        for key in _STRIP_ENV:
            env.pop(key, None)
        return env

    async def _git(self, repo: Path, *args: str, check: bool = True) -> tuple[int, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *SAFE_OVERRIDES,
                "-C",
                str(repo),
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
            )
        except FileNotFoundError:
            raise ToolError("git ist nicht installiert") from None
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolError(
                f"git {args[0]}: Zeitlimit {self.timeout_s:.0f}s überschritten"
            ) from None
        stdout = out.decode("utf-8", errors="replace")
        self._last_error = err.decode("utf-8", errors="replace").strip()[:300]
        if check and proc.returncode != 0:
            message = self._last_error or stdout.strip()
            raise ToolError(f"git {args[0]} fehlgeschlagen: {message[:500]}")
        return proc.returncode or 0, stdout

    async def _repo(self, args: Mapping[str, Any], ctx: ToolContext) -> Path:
        path = ctx.resolve(str(args.get("path", ".")))
        directory = path if path.is_dir() else path.parent
        code, out = await self._git(directory, "rev-parse", "--show-toplevel", check=False)
        if code != 0:
            raise ToolError(f"Kein Git-Repository: {args.get('path', '.')} ({self._last_error})")
        top = await asyncio.to_thread(Path(out.strip()).resolve)
        if not any(top.is_relative_to(root) for root in ctx.allowed_roots):
            raise ToolError(f"Repository-Wurzel {top} liegt außerhalb der erlaubten Verzeichnisse")
        _, raw = await self._git(top, "config", "--list", "--show-scope", "-z")
        risky = dangerous_config(parse_config_z(raw))
        if risky:
            raise ToolError(
                "Repository-Konfiguration enthält Einträge, die Befehle ausführen können "
                f"({', '.join(sorted(set(risky)))}). Git-Tools verweigert – Repository prüfen."
            )
        return top

    def _clip(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        return text[: self.max_output_chars] + "\n…[gekürzt]", True

    def _pathspec(self, args: Mapping[str, Any], ctx: ToolContext, repo: Path) -> list[str]:
        files = args.get("files") or []
        specs = []
        for f in files:
            resolved = ctx.resolve(str(f))
            if not resolved.is_relative_to(repo):
                raise ToolError(f"{f} liegt nicht im Repository")
            specs.append(str(resolved.relative_to(repo)) or ".")
        return ["--", *specs] if specs else []


_PATH_PROP = {
    "type": "string",
    "description": "Directory inside the repository (default: workspace)",
}


class GitStatusTool(_GitTool):
    name = "git_status"
    description = "Show branch and changed/staged/untracked files of a Git repository."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"path": _PATH_PROP},
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        repo = await self._repo(args, ctx)
        _, raw = await self._git(
            repo, "status", "--porcelain=v1", "-b", "-z", "--untracked-files=all"
        )
        records = [r for r in raw.split("\0") if r]
        branch = ""
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        conflicts: list[str] = []
        i = 0
        while i < len(records):
            rec = records[i]
            if rec.startswith("## "):
                branch = rec[3:]
            else:
                x, y, name = rec[0], rec[1], rec[3:]
                if x in "RC":
                    i += 1  # nächster Eintrag ist der ursprüngliche Pfad
                if x == "?" and y == "?":
                    untracked.append(name)
                elif "U" in (x, y) or (x, y) in (("A", "A"), ("D", "D")):
                    conflicts.append(name)
                else:
                    if x not in " ?":
                        staged.append(f"{x} {name}")
                    if y not in " ?":
                        unstaged.append(f"{y} {name}")
            i += 1
        sections = [f"Branch: {branch or '?'}"]
        for title, items in (
            ("Staged", staged),
            ("Nicht gestaged", unstaged),
            ("Untracked", untracked),
            ("Konflikte", conflicts),
        ):
            if items:
                sections.append(f"{title} ({len(items)}):\n" + "\n".join(f"  {s}" for s in items))
        clean = not (staged or unstaged or untracked or conflicts)
        if clean:
            sections.append("Arbeitsverzeichnis sauber")
        output, truncated = self._clip("\n".join(sections))
        return ToolResult.ok(
            output,
            repository=str(repo),
            branch=branch,
            clean=clean,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            conflicts=conflicts,
            truncated=truncated,
        )


class GitDiffTool(_GitTool):
    name = "git_diff"
    description = (
        "Show changes as unified diff. staged=true shows the index instead of the working tree; "
        "stat=true shows only a summary; files limits the diff to given paths."
    )
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "staged": {"type": "boolean"},
            "stat": {"type": "boolean"},
            "files": {"type": "array", "items": {"type": "string"}},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
        },
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        repo = await self._repo(args, ctx)
        cmd = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            f"-U{int(args.get('context_lines', 3))}",
        ]
        if args.get("staged"):
            cmd.append("--staged")
        if args.get("stat"):
            cmd.append("--stat")
        pathspec = self._pathspec(args, ctx, repo)
        _, out = await self._git(repo, *cmd, *pathspec)
        numstat_cmd = ["diff", "--no-ext-diff", "--no-textconv", "--numstat"]
        if args.get("staged"):
            numstat_cmd.append("--staged")
        _, numstat = await self._git(repo, *numstat_cmd, *pathspec)
        files = []
        for line in numstat.splitlines():
            added, removed, name = [*line.split("\t", 2), "", "", ""][:3]
            files.append({"file": name, "added": added, "removed": removed})
        output, truncated = self._clip(out or "Keine Änderungen")
        return ToolResult.ok(
            output,
            repository=str(repo),
            staged=bool(args.get("staged")),
            files_changed=len(files),
            files=files,
            truncated=truncated,
        )


class GitLogTool(_GitTool):
    name = "git_log"
    description = "Show recent commits (hash, author, date, subject), optionally for one file."
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "max_count": {"type": "integer", "minimum": 1, "maximum": 200},
            "file": {"type": "string"},
        },
        "additionalProperties": False,
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        repo = await self._repo(args, ctx)
        cmd = [
            "log",
            "--no-color",
            f"--max-count={int(args.get('max_count', 20))}",
            "--format=%H%x1f%an%x1f%aI%x1f%s%x1e",
        ]
        if args.get("file"):
            cmd += self._pathspec({"files": [args["file"]]}, ctx, repo)
        code, out = await self._git(repo, *cmd, check=False)
        if code != 0:
            # Leeres Repository (noch kein Commit) ist kein Fehler.
            _, head = await self._git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False)
            if not head.strip():
                return ToolResult.ok("Noch keine Commits", repository=str(repo), commits=[])
            raise ToolError("git log fehlgeschlagen")
        commits = []
        for record in out.split("\x1e"):
            fields = record.strip().split("\x1f")
            if len(fields) == 4:
                commits.append(
                    dict(zip(("hash", "author", "date", "subject"), fields, strict=True))
                )
        lines = [
            f"{c['hash'][:10]}  {c['date'][:10]}  {c['author']}: {c['subject']}" for c in commits
        ]
        output, truncated = self._clip("\n".join(lines) or "Keine Commits")
        return ToolResult.ok(output, repository=str(repo), commits=commits, truncated=truncated)


def git_tools() -> list[Tool]:
    return [GitStatusTool(), GitDiffTool(), GitLogTool()]
