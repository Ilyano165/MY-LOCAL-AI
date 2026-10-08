"""Routing Logs.

Jede Entscheidung wird nachvollziehbar protokolliert (Format siehe :func:`format_decision`).
Zusätzlich lässt sich das **Ergebnis** einer gerouteten Aufgabe nachtragen
(:meth:`RoutingLog.record_outcome`: Urteil der Verification Engine, Quality Score). Ein Router
sieht nur das Ergebnis des Modells, das er gewählt hat – diese Paare aus Entscheidung und Ergebnis
sind die Grundlage, um die Regeln auszuwerten oder später einen Router zu trainieren.
"""

from __future__ import annotations

import abc
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from router.base import (
        NoModelAvailableError,
        RoutingDecision,
        RoutingRequest,
        TaskClassification,
    )

logger = logging.getLogger("nova.router")


def format_decision(decision: RoutingDecision) -> str:
    c = decision.classification
    task = decision.request.task if decision.request else ""
    task = task if len(task) <= 200 else task[:200] + "…"
    lines = [
        "TASK:",
        f'"{task}"',
        "ROUTING:",
        f"category = {c.category.name}" + (f" ({c.secondary.name})" if c.secondary else ""),
        f"complexity = {c.complexity.name}",
        f"selected_model = {decision.model.name}",
        f"data = {decision.data_status.get(decision.model.name, 'UNMEASURED')}",
        f'reason = "{decision.reason}"',
        f"classifier = {c.classifier} (confidence {c.confidence:.2f})",
    ]
    if c.signals:
        lines.append(f"signals = {'; '.join(c.signals)}")
    if decision.fallbacks:
        lines.append(f"fallbacks = {', '.join(m.name for m in decision.fallbacks)}")
    for rejection in decision.rejected:
        lines.append(f"rejected = {rejection.model}: {', '.join(rejection.reasons)}")
    for note in decision.notes:
        lines.append(f"note = {note}")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RoutingLog(abc.ABC):
    @abc.abstractmethod
    def _write(self, entry: dict[str, Any]) -> None: ...

    def record(self, decision: RoutingDecision) -> None:
        logger.info("\n%s", format_decision(decision))
        self._write({"type": "decision", **decision.to_dict()})

    def record_failure(
        self,
        request: RoutingRequest,
        classification: TaskClassification,
        error: NoModelAvailableError,
    ) -> None:
        logger.warning("Routing fehlgeschlagen: %s", error)
        self._write(
            {
                "type": "failure",
                "timestamp": _now(),
                "task": request.task[:500],
                "category": classification.category.value,
                "complexity": classification.complexity.name,
                "error": str(error),
                "rejected": error.rejected,
            }
        )

    def record_outcome(
        self,
        decision_id: str,
        *,
        success: bool,
        verdict: str | None = None,
        quality: float | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        self._write(
            {
                "type": "outcome",
                "timestamp": _now(),
                "decision_id": decision_id,
                "success": success,
                "verdict": verdict,
                "quality": quality,
                "latency_ms": latency_ms,
                "error": error,
            }
        )


class MemoryRoutingLog(RoutingLog):
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def _write(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)

    def decisions(self) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["type"] == "decision"]


class JsonlRoutingLog(RoutingLog):
    """Append-only JSON-Lines-Datei (Default-Ort: ``~/.nova/routing.jsonl``)."""

    def __init__(self, path: str | Path = "~/.nova/routing.jsonl") -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def _write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def joined(self) -> list[dict[str, Any]]:
        """Entscheidungen mit nachgetragenem Ergebnis (Auswertung/Trainingsdaten)."""
        entries = self.read()
        outcomes = {e["decision_id"]: e for e in entries if e["type"] == "outcome"}
        return [{**d, "outcome": outcomes.get(d["id"])} for d in entries if d["type"] == "decision"]
