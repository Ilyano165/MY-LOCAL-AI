"""Benchmark der Verification Engine – zugleich Regressions-Gate.

Jede Änderung an Checks oder Strategien muss diese Fälle weiterhin korrekt beurteilen; ein
einziger False Accept (Fehler durchgelassen) lässt den Test scheitern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import BenchmarkCase, BenchmarkRunner, Verdict, builtin_cases
from evaluation.benchmarks import dump_cases, load_cases


async def test_builtin_benchmark_has_no_false_accepts(tmp_path: Path) -> None:
    result = await BenchmarkRunner().run(builtin_cases(), tmp_path)
    wrong = [
        f"{o.case.id}: erwartet {o.case.expected.value}, erhalten {o.actual.value}\n"
        f"{o.report.summary()}"
        for o in result.outcomes
        if not o.correct
    ]
    assert result.false_accepts == [], "\n".join(wrong)
    assert result.accuracy == 1.0, "\n".join(wrong)
    assert result.false_accept_rate == 0.0


def test_builtin_cases_cover_every_strategy_with_positive_and_negative_cases() -> None:
    cases = builtin_cases()
    assert len({c.id for c in cases}) == len(cases)
    for strategy in ("code", "files", "research", "math", "text"):
        verdicts = {c.expected for c in cases if c.strategy == strategy}
        assert Verdict.PASSED in verdicts and Verdict.FAILED in verdicts, strategy


async def test_metrics_and_markdown(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase("ok", "korrekt", "math", Verdict.PASSED, task="Was ist 2 + 2?", claim="4"),
        BenchmarkCase(
            "falsch-erwartet",
            "absichtlich falsche Erwartung → False Accept",
            "math",
            Verdict.FAILED,
            task="Was ist 2 + 2?",
            claim="4",
        ),
        BenchmarkCase(
            "reject",
            "absichtlich falsche Erwartung → False Reject",
            "math",
            Verdict.PASSED,
            task="Was ist 2 + 2?",
            claim="5",
        ),
    ]
    result = await BenchmarkRunner().run(cases, tmp_path)
    assert result.accuracy == pytest.approx(1 / 3)
    assert [o.case.id for o in result.false_accepts] == ["falsch-erwartet"]
    assert [o.case.id for o in result.false_rejects] == ["reject"]
    assert result.false_accept_rate == 1.0
    assert result.by_strategy() == {"math": (1, 3)}
    md = result.to_markdown()
    assert "False Accepts (Fehler durchgelassen): 1" in md and "| ok | math |" in md


async def test_runner_validation(tmp_path: Path) -> None:
    dup = [BenchmarkCase("x", "", "text", Verdict.UNVERIFIED)] * 2
    with pytest.raises(ValueError, match="eindeutig"):
        await BenchmarkRunner().run(dup, tmp_path)
    escape = BenchmarkCase("esc", "", "files", Verdict.FAILED, files={"../raus.txt": "x"})
    with pytest.raises(ValueError, match="außerhalb"):
        await BenchmarkRunner().run([escape], tmp_path / "b")
    assert not (tmp_path / "raus.txt").exists()


async def test_own_temp_dir_is_cleaned_up() -> None:
    result = await BenchmarkRunner().run(
        [BenchmarkCase("m", "", "math", Verdict.PASSED, task="Was ist 3 * 3?", claim="9")]
    )
    assert result.accuracy == 1.0


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    dump_cases(builtin_cases(), path)
    loaded = load_cases(path)
    assert [c.id for c in loaded] == [c.id for c in builtin_cases()]
    assert loaded[0].expected is Verdict.PASSED and loaded[0].files == builtin_cases()[0].files
