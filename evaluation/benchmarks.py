"""Benchmarks für die Verification Engine selbst.

Ein Benchmark-Fall beschreibt eine Aufgabe, die Antwort eines (simulierten) Agenten, die
Dateien im Workspace und das **erwartete** Urteil. Der Runner legt pro Fall einen frischen
Workspace an, prüft mit der Engine und vergleicht.

Wichtigste Kennzahl ist die **False-Accept-Rate**: Wie oft wird ein falsches Ergebnis als
``passed`` durchgewunken? Sie muss 0 sein – ein Verifier, der Fehler durchlässt, ist schädlicher
als einer, der zu streng ist.

Eigene Fälle lassen sich als JSON Lines laden (:func:`load_cases`).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.verifier import (
    ToolEvent,
    Verdict,
    VerificationContext,
    VerificationEngine,
    VerificationReport,
)


@dataclass
class BenchmarkCase:
    id: str
    description: str
    strategy: str
    expected: Verdict
    task: str = ""
    claim: str = ""
    files: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    test_command: list[str] | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCase:
        data = dict(data)
        data["expected"] = Verdict(data["expected"])
        return cls(**data)

    def context(self, workspace: Path) -> VerificationContext:
        return VerificationContext(
            workspace=workspace,
            task=self.task,
            claim=self.claim,
            artifacts=list(self.artifacts),
            requirements=list(self.requirements),
            sources=dict(self.sources),
            params=dict(self.params),
            test_command=self.test_command,
            tool_events=[ToolEvent(**e) for e in self.tool_events],
        )


@dataclass
class CaseOutcome:
    case: BenchmarkCase
    actual: Verdict
    report: VerificationReport
    duration_ms: float

    @property
    def correct(self) -> bool:
        return self.actual == self.case.expected

    @property
    def false_accept(self) -> bool:
        return self.actual == Verdict.PASSED and self.case.expected != Verdict.PASSED

    @property
    def false_reject(self) -> bool:
        return self.actual == Verdict.FAILED and self.case.expected == Verdict.PASSED


@dataclass
class BenchmarkResult:
    outcomes: list[CaseOutcome]

    @property
    def accuracy(self) -> float:
        return sum(o.correct for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def false_accepts(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.false_accept]

    @property
    def false_rejects(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.false_reject]

    @property
    def false_accept_rate(self) -> float:
        negatives = [o for o in self.outcomes if o.case.expected != Verdict.PASSED]
        return len(self.false_accepts) / len(negatives) if negatives else 0.0

    def by_strategy(self) -> dict[str, tuple[int, int]]:
        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for o in self.outcomes:
            stats[o.case.strategy][0] += o.correct
            stats[o.case.strategy][1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(stats.items())}

    def to_markdown(self) -> str:
        lines = [
            "# Verification-Benchmark",
            "",
            f"- Fälle: {len(self.outcomes)}",
            f"- Genauigkeit: {self.accuracy:.1%}",
            f"- False Accepts (Fehler durchgelassen): {len(self.false_accepts)} "
            f"(Rate {self.false_accept_rate:.1%})",
            f"- False Rejects (Korrektes abgelehnt): {len(self.false_rejects)}",
            "",
            "| Fall | Strategie | erwartet | tatsächlich | ok | Qualität (intern) |",
            "|---|---|---|---|---|---|",
        ]
        for o in self.outcomes:
            q = o.report.quality
            overall = "–" if q is None or q.overall is None else f"{q.overall:.2f}"
            lines.append(
                f"| {o.case.id} | {o.case.strategy} | {o.case.expected.value} | "
                f"{o.actual.value} | {'✓' if o.correct else '✗'} | {overall} |"
            )
        return "\n".join(lines)


class BenchmarkRunner:
    def __init__(self, engine: VerificationEngine | None = None) -> None:
        self.engine = engine or VerificationEngine()

    async def run_case(self, case: BenchmarkCase, base_dir: Path) -> CaseOutcome:
        workspace = base_dir / case.id
        workspace.mkdir(parents=True)
        for rel, content in case.files.items():
            target = (workspace / rel).resolve()
            if not target.is_relative_to(workspace.resolve()):
                raise ValueError(f"{case.id}: Dateipfad außerhalb des Workspace: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        started = time.perf_counter()
        report = await self.engine.verify(case.context(workspace), case.strategy)
        return CaseOutcome(case, report.verdict, report, (time.perf_counter() - started) * 1000)

    async def run(
        self, cases: Sequence[BenchmarkCase], base_dir: Path | None = None
    ) -> BenchmarkResult:
        ids = [c.id for c in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Benchmark-IDs müssen eindeutig sein")
        own_dir = base_dir is None
        root = Path(tempfile.mkdtemp(prefix="nova-bench-")) if own_dir else base_dir
        assert root is not None
        try:
            return BenchmarkResult([await self.run_case(c, root) for c in cases])
        finally:
            if own_dir:
                shutil.rmtree(root, ignore_errors=True)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    with Path(path).open(encoding="utf-8") as fh:
        return [BenchmarkCase.from_dict(json.loads(line)) for line in fh if line.strip()]


def dump_cases(cases: Iterable[BenchmarkCase], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for c in cases:
            data = {**c.__dict__, "expected": c.expected.value}
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- Standardfälle

_ADD_OK = "def add(a: int, b: int) -> int:\n    return a + b\n"
_ADD_WRONG = "def add(a: int, b: int) -> int:\n    return a - b\n"
_ADD_SYNTAX = "def add(a: int, b: int) -> int\n    return a + b\n"
_ADD_TYPE = "def add(a: int, b: int) -> int:\n    return str(a + b)\n"
_ADD_TEST = "from calc import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n"
_SOURCE = (
    "Die Stadt Beispielstadt hat 52000 Einwohner. Der Fluss Nova ist 120 km lang. "
    "Die Brücke wurde 1910 gebaut."
)


def builtin_cases() -> list[BenchmarkCase]:
    """Fälle mit bekannter Wahrheit; jede Kategorie hat korrekte und falsche Beispiele."""
    calc = {"calc.py": _ADD_OK, "test_calc.py": _ADD_TEST}
    return [
        # CODE
        BenchmarkCase(
            "code-ok",
            "korrekter Code, Tests grün",
            "code",
            Verdict.PASSED,
            task="Implementiere add",
            claim="calc.py erstellt, Tests bestanden.",
            files=calc,
            artifacts=["calc.py"],
        ),
        BenchmarkCase(
            "code-syntax",
            "Syntaxfehler",
            "code",
            Verdict.FAILED,
            claim="Fertig, alles funktioniert.",
            files={**calc, "calc.py": _ADD_SYNTAX},
            artifacts=["calc.py"],
        ),
        BenchmarkCase(
            "code-wrong",
            "Logikfehler, Agent behauptet grüne Tests",
            "code",
            Verdict.FAILED,
            claim="Implementiert. Alle Tests bestanden.",
            files={**calc, "calc.py": _ADD_WRONG},
            artifacts=["calc.py"],
        ),
        BenchmarkCase(
            "code-types",
            "Typfehler bei grünen Tests? (str statt int)",
            "code",
            Verdict.FAILED,
            claim="Fertig.",
            files={
                "calc.py": _ADD_TYPE,
                "test_calc.py": "from calc import add\n\n\ndef test_add() -> None:\n"
                "    assert add(2, 3) == '5'\n",
            },
            artifacts=["calc.py"],
        ),
        BenchmarkCase(
            "code-claimed-file-missing",
            "behauptete Datei existiert nicht",
            "code",
            Verdict.FAILED,
            claim="Ich habe utils.py erstellt und calc.py angepasst.",
            files=calc,
            artifacts=["calc.py"],
        ),
        # FILES
        BenchmarkCase(
            "files-ok",
            "Datei mit gefordertem Inhalt",
            "files",
            Verdict.PASSED,
            claim="config.json angelegt.",
            files={"config.json": '{"port": 8080}'},
            params={"file_expectations": {"config.json": {"contains": ["8080"]}}},
            artifacts=["config.json"],
        ),
        BenchmarkCase(
            "files-missing",
            "Datei fehlt",
            "files",
            Verdict.FAILED,
            claim="report.md wurde erstellt.",
            params={"expected_files": ["report.md"]},
        ),
        BenchmarkCase(
            "files-invalid-json",
            "ungültiges JSON",
            "files",
            Verdict.FAILED,
            claim="config.json angelegt.",
            files={"config.json": '{"port": 8080,}'},
            artifacts=["config.json"],
        ),
        # MATH
        BenchmarkCase(
            "math-ok",
            "richtig gerechnet",
            "math",
            Verdict.PASSED,
            task="Was ist 17 * 23?",
            claim="17 * 23 = 391",
        ),
        BenchmarkCase(
            "math-wrong",
            "falsch gerechnet",
            "math",
            Verdict.FAILED,
            task="Was ist 17 * 23?",
            claim="Das Ergebnis ist 381.",
        ),
        BenchmarkCase(
            "math-percent",
            "Prozentrechnung",
            "math",
            Verdict.PASSED,
            task="Berechne 15% von 240.",
            claim="15 % von 240 sind 36.",
        ),
        BenchmarkCase(
            "math-rounded",
            "gerundete Wurzel",
            "math",
            Verdict.PASSED,
            task="Wie viel ist sqrt(2) * 3?",
            claim="Das sind etwa 4,243.",
        ),
        BenchmarkCase(
            "math-no-expression",
            "nichts nachrechenbar",
            "math",
            Verdict.UNVERIFIED,
            task="Ist die Riemannsche Vermutung wahr?",
            claim="Das ist offen.",
        ),
        # RESEARCH
        BenchmarkCase(
            "research-ok",
            "korrektes Zitat aus bereitgestellter Quelle",
            "research",
            Verdict.PASSED,
            task="Recherchiere die Einwohnerzahl.",
            claim="Laut https://example.org/stadt gilt: „Die Stadt Beispielstadt hat "
            "52000 Einwohner.“",
            sources={"https://example.org/stadt": _SOURCE},
        ),
        BenchmarkCase(
            "research-fabricated-quote",
            "erfundenes Zitat",
            "research",
            Verdict.FAILED,
            claim="Laut https://example.org/stadt: „Die Brücke wurde 1850 von Römern gebaut.“",
            sources={"https://example.org/stadt": _SOURCE},
        ),
        BenchmarkCase(
            "research-fabricated-source",
            "Quelle nicht im Material",
            "research",
            Verdict.FAILED,
            claim="Siehe https://erfunden.example.com/studie.",
            sources={"https://example.org/stadt": _SOURCE},
        ),
        BenchmarkCase(
            "research-no-source",
            "Behauptung ohne Quelle",
            "research",
            Verdict.FAILED,
            task="Recherchiere die Flusslänge.",
            claim="Der Fluss ist 120 km lang.",
        ),
        BenchmarkCase(
            "research-contradiction",
            "widerspricht der Quelle",
            "research",
            Verdict.FAILED,
            claim="Gemäß https://example.org/stadt ist der Fluss Nova 95 km lang.",
            sources={"https://example.org/stadt": _SOURCE},
        ),
        # TEXT
        BenchmarkCase(
            "text-ok",
            "Anforderungen erfüllt",
            "text",
            Verdict.PASSED,
            task="Fasse in höchstens 20 Wörtern zusammen und erwähne „Nova“.",
            claim="Nova ist eine lokale KI-Plattform mit Agent, Tools und Gedächtnis.",
        ),
        BenchmarkCase(
            "text-too-long",
            "Wortlimit verletzt",
            "text",
            Verdict.FAILED,
            task="Antworte in maximal 5 Wörtern.",
            claim="Das ist eine deutlich zu lange Antwort auf die Frage.",
        ),
        BenchmarkCase(
            "text-json",
            "JSON gefordert, Prosa geliefert",
            "text",
            Verdict.FAILED,
            task="Gib das Ergebnis als JSON zurück.",
            claim="Das Ergebnis ist positiv.",
        ),
        BenchmarkCase(
            "text-unverifiable",
            "keine prüfbaren Anforderungen",
            "text",
            Verdict.UNVERIFIED,
            task="Schreib ein schönes Gedicht.",
            claim="Herbstlaub fällt leise.",
        ),
    ]
