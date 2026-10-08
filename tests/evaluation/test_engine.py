"""Engine-Regeln: Behauptungen allein bestätigen nie etwas."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evaluation import (
    Aspect,
    Check,
    CheckResult,
    CheckStatus,
    Verdict,
    VerificationContext,
    VerificationEngine,
    VerificationStrategy,
    decide,
    infer_strategy,
)


class Fixed(Check):
    def __init__(
        self,
        name: str,
        status: CheckStatus,
        *,
        independent: bool = True,
        can_confirm: bool = True,
        requires: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._status = status
        self.independent = independent
        self.can_confirm = can_confirm
        self.requires = requires
        self.ran = False

    async def run(self, ctx: VerificationContext) -> CheckResult:
        self.ran = True
        return self.result(self._status, f"{self.name} → {self._status.value}")


class Crashing(Check):
    name = "crash"

    async def run(self, ctx: VerificationContext) -> CheckResult:
        raise RuntimeError("kaputt")


class Slow(Check):
    name = "slow"
    timeout_s = 0.05

    async def run(self, ctx: VerificationContext) -> CheckResult:
        await asyncio.sleep(1)
        return self.passed("nie")


def strategy(*checks: Check) -> VerificationStrategy:
    return VerificationStrategy("test", "Test", lambda ctx: list(checks))


def results(*specs: tuple[CheckStatus, bool, bool]) -> list[CheckResult]:
    return [
        CheckResult(f"c{i}", "x", Aspect.CORRECTNESS, s, "", ind, can_confirm=conf)
        for i, (s, ind, conf) in enumerate(specs)
    ]


P, F, U = CheckStatus.PASSED, CheckStatus.FAILED, CheckStatus.UNVERIFIABLE


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        ([(P, True, True)], Verdict.PASSED),
        ([(P, True, True), (F, True, True)], Verdict.FAILED),
        ([(P, False, True)], Verdict.UNVERIFIED),  # nur Behauptung/abhängiger Check
        ([(P, True, False)], Verdict.UNVERIFIED),  # Problemdetektor ohne Fund ≠ Bestätigung
        ([(U, True, True), (U, True, True)], Verdict.UNVERIFIED),
        ([(F, False, True)], Verdict.FAILED),  # Widerlegung zählt immer
        ([(P, True, True), (U, True, True)], Verdict.PASSED),
        ([], Verdict.UNVERIFIED),
    ],
)
def test_decide(specs: list[tuple[CheckStatus, bool, bool]], expected: Verdict) -> None:
    verdict, reasons = decide(results(*specs))
    assert verdict == expected and reasons


async def test_engine_runs_checks_and_scores(tmp_path: Path) -> None:
    engine = VerificationEngine([strategy(Fixed("a", P), Fixed("b", U))])
    report = await engine.verify(VerificationContext(tmp_path), "test")
    assert report.verdict == Verdict.PASSED
    assert [r.check for r in report.results] == ["a", "b"]
    assert report.quality is not None and report.quality.confidence == 0.5
    data = report.to_dict()
    assert data["verdict"] == "passed" and "disclaimer" in data["quality"]
    assert "Verifikation (test): passed" in report.summary()


async def test_prerequisites_skip_dependent_checks(tmp_path: Path) -> None:
    dependent = Fixed("tests", P, requires=("syntax",))
    report = await VerificationEngine([strategy(Fixed("syntax", F), dependent)]).verify(
        VerificationContext(tmp_path), "test"
    )
    assert not dependent.ran
    assert report.results[1].status == CheckStatus.SKIPPED
    assert report.verdict == Verdict.FAILED


async def test_crash_and_timeout_become_errors(tmp_path: Path) -> None:
    report = await VerificationEngine([strategy(Crashing(), Slow(), Fixed("ok", P))]).verify(
        VerificationContext(tmp_path), "test"
    )
    statuses = [r.status for r in report.results]
    assert statuses == [CheckStatus.ERROR, CheckStatus.ERROR, CheckStatus.PASSED]
    assert "RuntimeError" in report.results[0].detail and "Zeitlimit" in report.results[1].detail
    assert all(r.duration_ms >= 0 for r in report.results)


def test_strategy_registry() -> None:
    engine = VerificationEngine()
    assert sorted(engine.strategies) == ["code", "files", "math", "research", "text"]
    with pytest.raises(ValueError, match="existiert bereits"):
        engine.register(VerificationStrategy("code", "x", lambda c: []))
    with pytest.raises(ValueError, match="Unbekannte Strategie"):
        engine.strategy("magie")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"artifacts": ["app.py"]}, "code"),
        ({"test_command": ["pytest"]}, "code"),
        ({"artifacts": ["report.md"]}, "files"),
        ({"task": "Recherchiere Quellen zu X"}, "research"),
        ({"sources": {"u": "t"}}, "research"),
        ({"task": "Was ist 17 * 23?"}, "math"),
        ({"task": "Berechne 15% von 200"}, "math"),
        ({"params": {"expression": "1+1"}}, "math"),
        ({"task": "Schreib ein Gedicht"}, "text"),
    ],
)
def test_infer_strategy(tmp_path: Path, kwargs: dict[str, object], expected: str) -> None:
    assert infer_strategy(VerificationContext(tmp_path, **kwargs)) == expected  # type: ignore[arg-type]


async def test_auto_strategy_is_resolved(tmp_path: Path) -> None:
    ctx = VerificationContext(tmp_path, task="Was ist 6 * 7?", claim="42")
    report = await VerificationEngine().verify(ctx)
    assert report.strategy == "math" and report.verdict == Verdict.PASSED
