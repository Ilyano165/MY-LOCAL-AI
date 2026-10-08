"""Persistenz des Task State.

``JsonFileTaskStore`` schreibt jeden Task als eigene JSON-Datei – atomar (temporäre Datei
+ ``os.replace``), damit ein Absturz mitten im Schreiben nie einen halben Zustand hinterlässt.
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

from agents.task import Task, utc_now

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class TaskStateError(Exception):
    """Task State konnte nicht gelesen oder geschrieben werden."""


class TaskStore(abc.ABC):
    @abc.abstractmethod
    async def save(self, task: Task) -> None: ...

    @abc.abstractmethod
    async def load(self, task_id: str) -> Task: ...

    @abc.abstractmethod
    async def list_ids(self) -> list[str]: ...


class InMemoryTaskStore(TaskStore):
    """Hält Tasks serialisiert im Speicher (gleiche Semantik wie die Dateivariante)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def save(self, task: Task) -> None:
        task.updated_at = utc_now()
        self._data[task.id] = json.dumps(task.to_dict())

    async def load(self, task_id: str) -> Task:
        try:
            return Task.from_dict(json.loads(self._data[task_id]))
        except KeyError:
            raise TaskStateError(f"Task {task_id!r} nicht gefunden") from None

    async def list_ids(self) -> list[str]:
        return sorted(self._data)


class JsonFileTaskStore(TaskStore):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    def _path(self, task_id: str) -> Path:
        if not _ID_RE.match(task_id):
            raise TaskStateError(f"Ungültige Task-ID: {task_id!r}")
        return self.directory / f"{task_id}.json"

    async def save(self, task: Task) -> None:
        task.updated_at = utc_now()
        payload = json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
        path = self._path(task.id)

        def _write() -> None:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{task.id}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

        await asyncio.to_thread(_write)

    async def load(self, task_id: str) -> Task:
        path = self._path(task_id)
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            raise TaskStateError(f"Task {task_id!r} nicht gefunden ({path})") from None
        try:
            return Task.from_dict(json.loads(raw))
        except (ValueError, KeyError, TypeError) as exc:
            raise TaskStateError(f"Task-Datei {path} ist beschädigt: {exc}") from exc

    async def list_ids(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))
