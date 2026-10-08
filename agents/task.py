"""Datenmodell eines Agent-Tasks.

Der Task ist der vollständige, serialisierbare Zustand eines Agent-Laufs. Er wird nach
jedem Phasenwechsel persistiert (siehe :mod:`agents.state`) und erlaubt damit Nachvollziehbarkeit
und Wiederaufnahme. Alle Enums sind ``StrEnum`` → direkt JSON-serialisierbar.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from models.capabilities import TaskType


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Phase(StrEnum):
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"
    VERIFY = "verify"
    CORRECT = "correct"
    FINALIZE = "finalize"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"  # alle Teilaufgaben erledigt (Verifikationsgrad: siehe FinalResult)
    FAILED = "failed"
    ABORTED = "aborted"  # Budget erschöpft oder abgebrochen

    @property
    def terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ABORTED)


class SubtaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Complexity(StrEnum):
    SIMPLE = "simple"
    MULTI_STEP = "multi_step"


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"  # keine anwendbare Prüfung – ausdrücklich KEIN Erfolg


class FinalStatus(StrEnum):
    SUCCESS = "success"  # alles erledigt und verifiziert
    UNVERIFIED = "unverified"  # alles erledigt, aber mind. ein Schritt ohne Verifikation
    PARTIAL = "partial"  # einige Teilaufgaben gescheitert/übersprungen
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Constraints:
    max_steps: int = 30
    """Maximale Anzahl Ausführungsversuche über alle Teilaufgaben."""
    max_attempts_per_subtask: int = 3
    max_replans: int = 1
    max_tool_calls_per_attempt: int = 12
    max_subtasks: int = 10
    tool_timeout_s: float = 60.0
    allowed_tools: list[str] | None = None
    test_command: list[str] | None = None
    """Wird nach Änderungen an Python-Dateien automatisch als Verifikation ausgeführt."""
    notes: list[str] = field(default_factory=list)
    """Freitext-Vorgaben des Nutzers (z. B. 'keine neuen Abhängigkeiten')."""

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_attempts_per_subtask",
            "max_tool_calls_per_attempt",
            "max_subtasks",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} muss >= 1 sein")
        if self.max_replans < 0:
            raise ValueError("max_replans darf nicht negativ sein")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s muss > 0 sein")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraints:
        return cls(**data)


@dataclass
class VerificationSpec:
    """Deklarative Prüfung, z. B. ``{"type": "python_syntax", "path": "app.py"}``."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    auto: bool = False  # vom Verifier automatisch ergänzt (nicht vom Planner)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationSpec:
        data = dict(data)
        if "type" not in data:
            raise ValueError("Verifikation ohne 'type'")
        type_ = str(data.pop("type"))
        auto = bool(data.pop("auto", False))
        params = data.pop("params", None)
        if params is None:
            params = data  # flache Form aus dem Planner: {"type": ..., "path": ...}
        return cls(type=type_, params=dict(params), auto=auto)


@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: Complexity
    requires_tools: bool
    success_criteria: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskAnalysis:
        return cls(
            task_type=TaskType(data["task_type"]),
            complexity=Complexity(data["complexity"]),
            requires_tools=bool(data["requires_tools"]),
            success_criteria=list(data.get("success_criteria", [])),
            signals=list(data.get("signals", [])),
        )


@dataclass
class Subtask:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    verification: list[VerificationSpec] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = 0
    output: str = ""
    feedback: list[str] = field(default_factory=list)
    """Korrekturhinweise aus fehlgeschlagenen Versuchen – fließen in den nächsten Versuch."""
    verdict: Verdict | None = None
    superseded: bool = False
    """Durch eine Neuplanung ersetzt – zählt nicht mehr für das Endergebnis."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subtask:
        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            depends_on=[str(d) for d in data.get("depends_on", [])],
            verification=[VerificationSpec.from_dict(v) for v in data.get("verification", [])],
            status=SubtaskStatus(data.get("status", SubtaskStatus.PENDING)),
            attempts=int(data.get("attempts", 0)),
            output=str(data.get("output", "")),
            feedback=list(data.get("feedback", [])),
            verdict=Verdict(data["verdict"]) if data.get("verdict") else None,
            superseded=bool(data.get("superseded", False)),
        )


@dataclass
class Observation:
    subtask_id: str | None
    source: str  # "tool:<name>" | "model" | "verifier" | "agent"
    content: str
    attempt: int = 0
    timestamp: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**data)


@dataclass
class ToolResultRecord:
    """Ein Tool-Aufruf im Task State: was (tool, arguments), warum (reason), mit welchem
    strukturierten Ergebnis (success, output, error, metadata)."""

    subtask_id: str
    attempt: int
    tool: str
    arguments: dict[str, Any]
    success: bool
    output: str | None
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    invocation_id: str = ""
    timestamp: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResultRecord:
        return cls(**data)


@dataclass
class ErrorRecord:
    phase: Phase
    kind: str  # "tool" | "model" | "verification" | "planning" | "budget" | "internal"
    message: str
    subtask_id: str | None = None
    attempt: int = 0
    analysis: str = ""
    """Ursachenanalyse (aus der CORRECT-Phase)."""
    timestamp: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorRecord:
        return cls(**{**data, "phase": Phase(data["phase"])})


@dataclass
class VerificationRecord:
    subtask_id: str
    attempt: int
    check: str
    verdict: Verdict
    detail: str
    auto: bool = False
    timestamp: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationRecord:
        return cls(**{**data, "verdict": Verdict(data["verdict"])})


@dataclass
class FinalResult:
    status: FinalStatus
    answer: str
    verified: bool
    summary: str
    unverified_subtasks: list[str] = field(default_factory=list)
    failed_subtasks: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalResult:
        return cls(**{**data, "status": FinalStatus(data["status"])})


@dataclass
class Task:
    objective: str
    constraints: Constraints = field(default_factory=Constraints)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    phase: Phase = Phase.ANALYZE
    analysis: TaskAnalysis | None = None
    subtasks: list[Subtask] = field(default_factory=list)
    current_step: int = 0
    """Index der aktuell bearbeiteten Teilaufgabe."""
    steps_taken: int = 0
    """Verbrauchte Ausführungsversuche (Budget: ``constraints.max_steps``)."""
    replans: int = 0
    plan_version: int = 0
    observations: list[Observation] = field(default_factory=list)
    tool_results: list[ToolResultRecord] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    verification_results: list[VerificationRecord] = field(default_factory=list)
    final_result: FinalResult | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("objective darf nicht leer sein")

    # ------------------------------------------------------------------ Zugriffshilfen

    def subtask(self, subtask_id: str) -> Subtask:
        for sub in self.subtasks:
            if sub.id == subtask_id:
                return sub
        raise KeyError(subtask_id)

    @property
    def current_subtask(self) -> Subtask | None:
        if 0 <= self.current_step < len(self.subtasks):
            return self.subtasks[self.current_step]
        return None

    def observe(
        self, source: str, content: str, subtask_id: str | None = None, attempt: int = 0
    ) -> None:
        self.observations.append(Observation(subtask_id, source, content, attempt))

    def record_error(
        self,
        phase: Phase,
        kind: str,
        message: str,
        subtask_id: str | None = None,
        attempt: int = 0,
    ) -> ErrorRecord:
        record = ErrorRecord(phase, kind, message, subtask_id, attempt)
        self.errors.append(record)
        return record

    # ------------------------------------------------------------------ Serialisierung

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            objective=data["objective"],
            constraints=Constraints.from_dict(data.get("constraints", {})),
            id=data["id"],
            status=TaskStatus(data["status"]),
            phase=Phase(data["phase"]),
            analysis=TaskAnalysis.from_dict(data["analysis"]) if data.get("analysis") else None,
            subtasks=[Subtask.from_dict(s) for s in data.get("subtasks", [])],
            current_step=int(data.get("current_step", 0)),
            steps_taken=int(data.get("steps_taken", 0)),
            replans=int(data.get("replans", 0)),
            plan_version=int(data.get("plan_version", 0)),
            observations=[Observation.from_dict(o) for o in data.get("observations", [])],
            tool_results=[ToolResultRecord.from_dict(t) for t in data.get("tool_results", [])],
            errors=[ErrorRecord.from_dict(e) for e in data.get("errors", [])],
            verification_results=[
                VerificationRecord.from_dict(v) for v in data.get("verification_results", [])
            ],
            final_result=FinalResult.from_dict(data["final_result"])
            if data.get("final_result")
            else None,
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )
