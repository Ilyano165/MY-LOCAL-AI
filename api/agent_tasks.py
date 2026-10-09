"""Agent-Aufgaben für Integrationen: Task-ID, Status, Ergebnis, Abbruch, Tool-Freigabe.

Lebenszyklus: ``queued`` → ``running`` → ``succeeded`` | ``failed`` | ``cancelled`` |
``timed_out``. ``succeeded`` heißt: der Agent hat ein Ergebnis geliefert – ob es verifiziert
ist, steht separat in ``result.verification`` (nie vermischt).

Sicherheit:
* Tools nur, wenn die Integration ``agent:tool:<name>`` besitzt **und** der Task sie anfordert.
* Ohne freigegebene Tools arbeitet der Agent ohne Werkzeuge (reine Antwort).
* Aktionen, die im Tool-System eine Bestätigung verlangen (z. B. nicht freigelistete
  Befehle), werden für API-Tasks **immer abgelehnt** – es sitzt niemand davor.
* Arbeitsverzeichnis: das der Integration zugewiesene oder ein leeres, task-eigenes.
* Abfragen und Abbruch nur durch die Integration, die den Task angelegt hat.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from agents.agent import Agent
from agents.state import JsonFileTaskStore
from agents.task import Phase, Task
from models.base import ModelError
from models.inference import InferenceEngine
from tools import default_tools
from tools.registry import ApprovalRequest, ToolRegistry


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def final(self) -> bool:
        return self not in (TaskState.QUEUED, TaskState.RUNNING)


class TaskPermissionError(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class AgentTaskRecord:
    id: str
    owner: str
    """Integrations-ID (oder ``ui``)."""
    objective: str
    status: TaskState = TaskState.QUEUED
    allowed_tools: list[str] = field(default_factory=list)
    timeout_s: float = 600.0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    events: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self, *, include_objective: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        if not include_objective:
            data.pop("objective")
        return data


class AgentTaskManager:
    def __init__(
        self,
        engine: InferenceEngine,
        state_dir: Path,
        *,
        max_concurrent: int = 2,
        routing_lookup: Callable[[str], dict[str, Any] | None] | None = None,
        on_event: Callable[[str, AgentTaskRecord], object] | None = None,
    ) -> None:
        self.engine = engine
        self.state_dir = state_dir
        self.tasks: dict[str, AgentTaskRecord] = {}
        self._runners: dict[str, asyncio.Task[None]] = {}
        self._slots = asyncio.Semaphore(max_concurrent)
        self.routing_lookup = routing_lookup
        self.on_event = on_event
        self._load()

    # ------------------------------------------------------------------ Persistenz

    def _path(self, task_id: str) -> Path:
        return self.state_dir / f"{task_id}.json"

    def _save(self, record: AgentTaskRecord) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(record.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(record.id))

    def _load(self) -> None:
        if not self.state_dir.is_dir():
            return
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = AgentTaskRecord(**{**data, "status": TaskState(data["status"])})
            except (ValueError, TypeError, KeyError):
                continue
            if not record.status.final:  # Server wurde während des Laufs beendet
                record.status = TaskState.FAILED
                record.error = {"code": "interrupted", "message": "NOVA was restarted"}
                record.finished_at = _now()
                self._save(record)
            self.tasks[record.id] = record

    # ------------------------------------------------------------------ API

    def create(
        self,
        owner: str,
        objective: str,
        *,
        granted_tools: set[str],
        requested_tools: list[str],
        workspace: str | None,
        timeout_s: float,
    ) -> AgentTaskRecord:
        known = {t.name for t in default_tools()}
        unknown = sorted(set(requested_tools) - known)
        if unknown:
            raise TaskPermissionError(f"Unknown tools: {', '.join(unknown)}")
        denied = sorted(set(requested_tools) - granted_tools)
        if denied:
            raise TaskPermissionError(
                "Tools not granted to this integration: "
                + ", ".join(denied)
                + " (requires scope agent:tool:<name>)"
            )
        record = AgentTaskRecord(
            id="task_" + uuid.uuid4().hex[:20],
            owner=owner,
            objective=objective,
            allowed_tools=sorted(set(requested_tools)),
            timeout_s=timeout_s,
        )
        self.tasks[record.id] = record
        self._save(record)
        self._runners[record.id] = asyncio.create_task(self._run(record, workspace))
        return record

    def get(self, task_id: str, owner: str) -> AgentTaskRecord | None:
        record = self.tasks.get(task_id)
        return record if record is not None and record.owner == owner else None

    def cancel(self, task_id: str, owner: str) -> AgentTaskRecord | None:
        record = self.get(task_id, owner)
        if record is None:
            return None
        runner = self._runners.get(task_id)
        if not record.status.final and runner is not None:
            runner.cancel()
        return record

    async def wait(self, task_id: str) -> None:
        runner = self._runners.get(task_id)
        if runner is not None:
            await asyncio.gather(runner, return_exceptions=True)

    async def shutdown(self) -> None:
        for runner in list(self._runners.values()):
            runner.cancel()
        await asyncio.gather(*self._runners.values(), return_exceptions=True)

    # ------------------------------------------------------------------ Ausführung

    def _event(self, record: AgentTaskRecord, phase: str, message: str) -> None:
        record.events = [
            *record.events[-49:],
            {"ts": _now(), "phase": phase, "message": message[:300]},
        ]

    async def _run(self, record: AgentTaskRecord, workspace: str | None) -> None:
        try:
            async with self._slots:
                record.status = TaskState.RUNNING
                record.started_at = _now()
                self._save(record)
                if self.on_event:
                    self.on_event("agent_task_started", record)
                work = Path(workspace) if workspace else self.state_dir / record.id / "workspace"
                work.mkdir(parents=True, exist_ok=True)
                tools = [t for t in default_tools() if t.name in record.allowed_tools]

                async def deny_confirmation(_request: ApprovalRequest) -> bool:
                    return False  # API-Tasks haben keinen Menschen für Bestätigungen

                agent = Agent(
                    self.engine,
                    ToolRegistry(tools, approval=deny_confirmation),
                    JsonFileTaskStore(self.state_dir / record.id / "state"),
                    work,
                    on_event=lambda _t, phase, msg: self._event(record, phase.value, msg),
                )
                started = time.perf_counter()
                task = await asyncio.wait_for(agent.run(record.objective), record.timeout_s)
                record.result = self._result(task, time.perf_counter() - started)
                record.status = TaskState.SUCCEEDED
        except TimeoutError:
            record.status = TaskState.TIMED_OUT
            record.error = {
                "code": "timeout",
                "message": f"Agent task exceeded {record.timeout_s:g} s",
            }
        except asyncio.CancelledError:
            record.status = TaskState.CANCELLED
            record.error = {"code": "cancelled", "message": "Cancelled by request"}
        except ModelError as exc:
            record.status = TaskState.FAILED
            record.error = {"code": "model_error", "message": str(exc)[:500]}
        except Exception as exc:  # unerwartet: sauber als Fehler melden, nicht abstürzen
            record.status = TaskState.FAILED
            record.error = {"code": "internal_error", "message": type(exc).__name__}
        finally:
            record.finished_at = _now()
            self._save(record)
            self._runners.pop(record.id, None)
            if self.on_event:
                self.on_event("agent_task_finished", record)

    def _result(self, task: Task, seconds: float) -> dict[str, Any]:
        result: dict[str, Any] = {"duration_s": round(seconds, 3)}
        if task.final_result is not None:
            r = task.final_result
            result["answer"] = r.answer
            result["verification"] = {
                "status": r.status.value,
                "verified": r.verified,
                "summary": r.summary,
            }
            if r.quality:
                result["verification"]["quality"] = r.quality.get("overall")
        if self.routing_lookup is not None:
            decisions = [d for d in (self.routing_lookup(i) for i in task.routing_ids) if d]
            if decisions:
                result["models_used"] = sorted({str(d.get("selected_model")) for d in decisions})
        return result


__all__ = ["AgentTaskManager", "AgentTaskRecord", "Phase", "TaskPermissionError", "TaskState"]
