"""Verification Engine: prüft Ergebnisse unabhängig von den Behauptungen des Agenten.

Grundsätze (vgl. Forschung zu LLM-Selbstverifikation: Selbstkritik ist unzuverlässig,
externe, fundierte Prüfung wirkt – Stechly et al. 2024, Huang et al. 2023):

* Die Behauptung des Agenten (``VerificationContext.claim``) ist **Prüfgegenstand, nie Beweis**.
* Ein Check ist *unabhängig*, wenn er Evidenz außerhalb der Behauptung erzeugt (Datei lesen,
  Code ausführen, nachrechnen, mit Quellen abgleichen). Nur unabhängige Checks können ein
  ``PASSED`` begründen.
* ``PASSED`` verlangt mindestens einen bestandenen unabhängigen Check und keinen
  fehlgeschlagenen; was nicht prüfbar ist, wird als ``UNVERIFIABLE`` ausgewiesen – nie als Erfolg.
* Kein LLM als Richter in der Engine: ein Prüfer, der schwächer ist als der Generator, kann das
  Ergebnis verschlechtern. Modellbasierte Kritik gehört, falls überhaupt, als schwaches Signal
  außerhalb dieser Engine.

Bausteine: :class:`Check` (eine Prüfung), :class:`VerificationStrategy` (Bündel von Checks für
eine Aufgabenart), :class:`VerificationEngine` (führt aus, entscheidet, bewertet Qualität).
"""

from __future__ import annotations

import abc
import asyncio
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- Typen


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"
    """Konnte mangels Evidenz/Werkzeug nicht geprüft werden – kein Erfolg, kein Fehlschlag."""
    SKIPPED = "skipped"
    """Nicht ausgeführt, weil eine Voraussetzung nicht erfüllt war."""
    ERROR = "error"
    """Der Check selbst ist abgestürzt oder hat das Zeitlimit überschritten."""


class Aspect(StrEnum):
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    REQUIREMENTS = "requirements"


class Verdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ToolEvent:
    """Ein Tool-Aufruf des Agenten (aus dem Task State), als *Hinweis*, nicht als Beweis."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    success: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class VerificationContext:
    workspace: Path
    task: str = ""
    """Aufgabenstellung."""
    claim: str = ""
    """Antwort/Behauptung des Agenten – wird geprüft, nie geglaubt."""
    artifacts: list[str] = field(default_factory=list)
    """Pfade, die der Agent angelegt/geändert haben will (relativ zum Workspace)."""
    tool_events: list[ToolEvent] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    """Ausdrückliche Anforderungen (zusätzlich zu denen aus ``task``)."""
    sources: dict[str, str] = field(default_factory=dict)
    """Bereitgestellte Quelltexte (URL/ID → Text) für Recherche-Prüfungen."""
    test_command: list[str] | None = None
    type_command: list[str] | None = None
    integration_command: list[str] | None = None
    reported_errors: int = 0
    """Vom Agenten während der Ausführung gemeldete Fehler (fließen in den Quality Score)."""
    params: dict[str, Any] = field(default_factory=dict)
    """Strategiespezifische Parameter (z. B. ``expression`` für Mathe)."""
    results: list[CheckResult] = field(default_factory=list)
    """Bisherige Ergebnisse dieses Laufs (von der Engine befüllt)."""

    def result(self, check_name: str) -> CheckResult | None:
        return next((r for r in self.results if r.check == check_name), None)


@dataclass
class CheckResult:
    check: str
    category: str
    aspect: Aspect
    status: CheckStatus
    detail: str
    independent: bool
    weight: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    can_confirm: bool = True
    """False bei reinen Problemdetektoren: „nichts gefunden“ ist keine Bestätigung."""

    @property
    def executed(self) -> bool:
        return self.status in (CheckStatus.PASSED, CheckStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "category": self.category,
            "aspect": self.aspect.value,
            "status": self.status.value,
            "detail": self.detail,
            "independent": self.independent,
            "weight": self.weight,
            "evidence": self.evidence,
            "duration_ms": round(self.duration_ms, 1),
            "can_confirm": self.can_confirm,
        }


# --------------------------------------------------------------------------- Check


class Check(abc.ABC):
    """Eine einzelne Prüfung. Unterklassen setzen Name, Kategorie und Aspekt."""

    name: str = "check"
    category: str = "general"
    aspect: Aspect = Aspect.CORRECTNESS
    independent: bool = True
    can_confirm: bool = True
    """Kann ein bestandener Check das Ergebnis *bestätigen*? Problemdetektoren (z. B.
    Widerspruchssuche) können nur widerlegen – „nichts gefunden“ belegt keine Korrektheit."""
    weight: float = 1.0
    requires: tuple[str, ...] = ()
    """Namen von Checks, die vorher bestanden sein müssen (sonst SKIPPED)."""
    timeout_s: float = 600.0

    @abc.abstractmethod
    async def run(self, ctx: VerificationContext) -> CheckResult: ...

    def result(self, status: CheckStatus, detail: str, **evidence: Any) -> CheckResult:
        return CheckResult(
            self.name,
            self.category,
            self.aspect,
            status,
            detail,
            self.independent,
            self.weight,
            dict(evidence),
            can_confirm=self.can_confirm,
        )

    def passed(self, detail: str, **evidence: Any) -> CheckResult:
        return self.result(CheckStatus.PASSED, detail, **evidence)

    def failed(self, detail: str, **evidence: Any) -> CheckResult:
        return self.result(CheckStatus.FAILED, detail, **evidence)

    def unverifiable(self, detail: str, **evidence: Any) -> CheckResult:
        return self.result(CheckStatus.UNVERIFIABLE, detail, **evidence)


ChecksFactory = Callable[[VerificationContext], Sequence[Check]]


@dataclass
class VerificationStrategy:
    """Bündel von Checks für eine Aufgabenart. ``checks`` kann vom Kontext abhängen."""

    name: str
    description: str
    checks: ChecksFactory

    def build(self, ctx: VerificationContext) -> list[Check]:
        return list(self.checks(ctx))


# --------------------------------------------------------------------------- Bericht


@dataclass
class VerificationReport:
    strategy: str
    verdict: Verdict
    results: list[CheckResult]
    quality: Any  # QualityScore (zirkuläre Typabhängigkeit vermieden)
    reasons: list[str] = field(default_factory=list)

    def by_status(self, status: CheckStatus) -> list[CheckResult]:
        return [r for r in self.results if r.status == status]

    def summary(self) -> str:
        counts = {s.value: len(self.by_status(s)) for s in CheckStatus if self.by_status(s)}
        lines = [f"Verifikation ({self.strategy}): {self.verdict.value} – {counts}"]
        lines += [f"  [{r.status.value}] {r.check}: {r.detail}" for r in self.results]
        if self.quality is not None:
            lines.append(f"  {self.quality.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "results": [r.to_dict() for r in self.results],
            "quality": self.quality.to_dict() if self.quality is not None else None,
        }


def decide(results: Sequence[CheckResult]) -> tuple[Verdict, list[str]]:
    """Gesamturteil aus Einzelergebnissen – die Behauptung des Agenten spielt keine Rolle."""
    failed = [r for r in results if r.status == CheckStatus.FAILED]
    if failed:
        return Verdict.FAILED, [f"{r.check}: {r.detail}" for r in failed]
    independent_passed = [
        r for r in results if r.status == CheckStatus.PASSED and r.independent and r.can_confirm
    ]
    if not independent_passed:
        return Verdict.UNVERIFIED, [
            "kein bestätigender unabhängiger Check bestanden – Ergebnis nicht belegt"
        ]
    reasons = [f"{len(independent_passed)} unabhängige(r) Check(s) bestanden"]
    open_ = [
        r
        for r in results
        if r.status in (CheckStatus.UNVERIFIABLE, CheckStatus.ERROR, CheckStatus.SKIPPED)
    ]
    if open_:
        reasons.append(f"{len(open_)} Check(s) nicht prüfbar – geringere Konfidenz")
    return Verdict.PASSED, reasons


# --------------------------------------------------------------------------- Engine


class VerificationEngine:
    def __init__(
        self, strategies: Iterable[VerificationStrategy] | None = None, *, scorer: Any = None
    ) -> None:
        from evaluation.quality import QualityScorer  # lokal: quality importiert verifier
        from evaluation.strategies import default_strategies

        self.strategies: dict[str, VerificationStrategy] = {}
        for strategy in strategies if strategies is not None else default_strategies():
            self.register(strategy)
        self.scorer = scorer or QualityScorer()

    def register(self, strategy: VerificationStrategy) -> None:
        if strategy.name in self.strategies:
            raise ValueError(f"Strategie {strategy.name!r} existiert bereits")
        self.strategies[strategy.name] = strategy

    def strategy(self, name: str) -> VerificationStrategy:
        try:
            return self.strategies[name]
        except KeyError:
            known = ", ".join(sorted(self.strategies))
            raise ValueError(f"Unbekannte Strategie {name!r} (bekannt: {known})") from None

    async def verify(
        self, ctx: VerificationContext, strategy: str | VerificationStrategy = "auto"
    ) -> VerificationReport:
        if isinstance(strategy, str):
            name = infer_strategy(ctx) if strategy == "auto" else strategy
            chosen = self.strategy(name)
        else:
            chosen = strategy
        ctx.results = []
        for check in chosen.build(ctx):
            ctx.results.append(await self._run(check, ctx))
        verdict, reasons = decide(ctx.results)
        report = VerificationReport(chosen.name, verdict, list(ctx.results), None, reasons)
        report.quality = self.scorer.score(report, ctx)
        return report

    @staticmethod
    async def _run(check: Check, ctx: VerificationContext) -> CheckResult:
        blocked = [
            n
            for n in check.requires
            if (r := ctx.result(n)) is not None and r.status != CheckStatus.PASSED
        ]
        if blocked:
            return check.result(CheckStatus.SKIPPED, f"Voraussetzung nicht erfüllt: {blocked[0]}")
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(check.run(ctx), check.timeout_s)
        except TimeoutError:
            result = check.result(
                CheckStatus.ERROR, f"Zeitlimit {check.timeout_s:.0f}s überschritten"
            )
        except Exception as exc:
            result = check.result(
                CheckStatus.ERROR, f"Check abgestürzt: {type(exc).__name__}: {exc}"
            )
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result


# --------------------------------------------------------------------------- Strategiewahl

_MATH_TASK = re.compile(
    r"(\d\s*[-+*/×÷·^:]\s*[\d(])|\b(berechne|rechne|calculate|compute|wie ?viel ist|what is)\b.*\d"
    r"|\b(sqrt|wurzel|prozent|percent)\b.*\d|\d\s*%\s*(von|of)\s*\d",
    re.IGNORECASE,
)
_RESEARCH_TASK = re.compile(
    r"\b(recherch\w*|research|quellen?|sources?|belege|zitier\w*|cite|citation|studien?|studies)\b",
    re.IGNORECASE,
)


def infer_strategy(ctx: VerificationContext) -> str:
    """Wählt eine Strategie aus Artefakten und Aufgabentext (deterministisch, erklärbar)."""
    if any(a.endswith(".py") for a in ctx.artifacts) or ctx.test_command:
        return "code"
    if ctx.artifacts:
        return "files"
    if ctx.sources or _RESEARCH_TASK.search(ctx.task):
        return "research"
    if "expression" in ctx.params or _MATH_TASK.search(ctx.task):
        return "math"
    return "text"


def strategy_names() -> list[str]:
    return ["code", "files", "research", "math", "text"]
