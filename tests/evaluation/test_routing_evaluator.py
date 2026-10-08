from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation.routing_evaluator import (
    DEFAULT_DATASET,
    WEIGHTS,
    DatasetError,
    Fleet,
    RoutingEvaluator,
    RoutingTask,
    Scenario,
    ScenarioReport,
    load_dataset,
    missing_capabilities,
    render_report,
    write_json,
)
from models.capabilities import ModelMetadata
from models.model_registry import ModelRegistry
from router.availability import StaticAvailability
from router.base import (
    Complexity,
    ModelRouter,
    RoutingCategory,
    RoutingDecision,
    RoutingRequest,
    TaskClassification,
)
from tests.router.conftest import model

# ---------------------------------------------------------------------- Datensatz


@pytest.fixture(scope="module")
def dataset() -> list[RoutingTask]:
    return load_dataset()


def test_dataset_size_and_distribution(dataset: list[RoutingTask]) -> None:
    assert len(dataset) >= 150
    counts = Counter(t.set for t in dataset)
    assert counts == {
        "FAST": 25,
        "GENERAL": 25,
        "CODING": 30,
        "REASONING": 25,
        "LONG_CONTEXT": 20,
        "VISION": 15,
        "MULTI_STEP": 10,
    }


def test_required_coverage(dataset: list[RoutingTask]) -> None:
    def tags(set_name: str) -> set[str]:
        return {tag for t in dataset if t.set == set_name for tag in t.tags}

    assert {
        "debugging",
        "refactoring",
        "architecture",
        "tests",
        "multi_file",
        "error_search",
        "comprehension",
    } <= tags("CODING")
    assert {"proof", "multi_step_logic", "contradiction", "planning", "complex_analysis"} <= tags(
        "REASONING"
    )
    assert {"large_document", "multi_source", "distractors", "cross_reference"} <= tags(
        "LONG_CONTEXT"
    )
    assert {"calculation", "fact", "transformation", "classification"} <= tags("FAST")
    assert all(t.requires_vision for t in dataset if t.set == "VISION")
    assert all(t.requires_tools for t in dataset if t.set == "MULTI_STEP")


def test_labels_are_consistent(dataset: list[RoutingTask]) -> None:
    for t in dataset:
        if t.set == "LONG_CONTEXT":
            assert t.context_tokens > 24_000 and t.estimated_context in ("large", "very_large")
        if t.set == "FAST":
            assert t.expected_complexity is Complexity.LOW
        assert t.prompt.strip() and len(t.prompt) < 2000
    assert len({t.id for t in dataset}) == len(dataset)
    assert len({t.prompt for t in dataset}) == len(dataset)
    assert {"de", "en"} <= {t.language for t in dataset}
    assert sum(t.difficulty == "hard" for t in dataset) >= 10  # nicht nur triviale Beispiele
    assert sum(not t.expected_routable for t in dataset) >= 1


def test_dataset_validation_rejects_bad_records(tmp_path: Path) -> None:
    good = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0])
    cases = [
        {k: v for k, v in good.items() if k != "prompt"},
        {**good, "expected_category": "MAGIC"},
        {**good, "required_capabilities": ["telepathy"]},
        {**good, "requires_vision": True},  # widerspricht required_capabilities
        {**good, "estimated_context": "huge"},
    ]
    for record in cases:
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps(record) + "\n")
        with pytest.raises(DatasetError):
            load_dataset(path)
    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(DatasetError, match="doppelte ID"):
        load_dataset(path)


def test_request_contains_only_runtime_facts(dataset: list[RoutingTask]) -> None:
    task = next(t for t in dataset if t.requires_vision and t.requires_tools)
    request = task.to_request()
    assert request.needs_vision and request.required_tools
    assert request.category_hint is None and request.complexity is None  # nicht vorsagen


# ---------------------------------------------------------------------- Bewertung


def task(**overrides: object) -> RoutingTask:
    data: dict[str, object] = {
        "id": "t-1",
        "set": "VISION",
        "prompt": "Turn this sketch into HTML.",
        "expected_category": "VISION",
        "expected_complexity": "MEDIUM",
        "required_capabilities": ["vision", "coding"],
        "requires_tools": False,
        "requires_vision": True,
        "estimated_context": "small",
        "difficulty": "easy",
    }
    data.update(overrides)
    return RoutingTask.from_dict(data)


def test_missing_capabilities() -> None:
    seer = model(
        "seer",
        vision_capability="good",
        coding_capability="none",
        tool_calling=False,
        context_length=8192,
    )
    assert missing_capabilities(seer, task()) == ["keine coding-Fähigkeit"]
    t = task(
        required_capabilities=["vision", "tool_calling"], requires_tools=True, context_tokens=20_000
    )
    found = missing_capabilities(seer, t)
    assert "kein Tool-Calling" in found and any("Kontext" in f for f in found)


class FixedRouter(ModelRouter):
    """Wählt immer dasselbe Modell (zum Testen der Bewertung)."""

    def __init__(
        self,
        registry: ModelRegistry,
        availability: StaticAvailability,
        pick: str,
        category: RoutingCategory = RoutingCategory.VISION,
        confidence: float = 0.95,
    ) -> None:
        self.registry, self.availability = registry, availability
        self.pick, self.category, self.confidence = pick, category, confidence

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        c = TaskClassification(self.category, Complexity.MEDIUM, False, 100, self.confidence)
        return RoutingDecision(self.registry.get(self.pick), c, "fix")


def fleet(*models: ModelMetadata, unavailable: tuple[str, ...] = ()) -> Fleet:
    return Fleet(list(models), [Scenario("s", unavailable=unavailable)])


SEER = model("seer", vision_capability="good", coding_capability="none", tool_calling=False)
OMNI = model("omni", vision_capability="good", coding_capability="good", speed="slow")


async def test_unsuitable_model_is_critical_and_outweighs_category_error() -> None:
    f = fleet(SEER, OMNI)
    wrong_model = RoutingEvaluator(
        f, router_factory=lambda r, a: FixedRouter(r, a, "seer"), latency_repeats=1
    )
    report = await wrong_model.evaluate([task()], f.scenarios[0])
    o = report.outcomes[0]
    assert o.critical and o.penalty == WEIGHTS["critical"]
    assert report.capability_violations == 1
    wrong_category = RoutingEvaluator(
        f,
        router_factory=lambda r, a: FixedRouter(r, a, "omni", RoutingCategory.CODING),
        latency_repeats=1,
    )
    report2 = await wrong_category.evaluate([task()], f.scenarios[0])
    assert not report2.outcomes[0].critical
    assert report2.outcomes[0].penalty == WEIGHTS["category"]
    assert o.penalty >= 10 * report2.outcomes[0].penalty


async def test_unavailable_model_violation() -> None:
    f = fleet(SEER, OMNI, unavailable=("omni",))
    ev = RoutingEvaluator(
        f, router_factory=lambda r, a: FixedRouter(r, a, "omni"), latency_repeats=1
    )
    report = await ev.evaluate([task()], f.scenarios[0])
    assert report.unavailable_violations == 1 and report.critical_errors == 1


async def test_tool_task_on_model_without_tools_is_critical() -> None:
    f = fleet(SEER, OMNI)
    t = task(required_capabilities=["vision", "tool_calling"], requires_tools=True)
    ev = RoutingEvaluator(
        f, router_factory=lambda r, a: FixedRouter(r, a, "seer"), latency_repeats=1
    )
    o = (await ev.evaluate([t], f.scenarios[0])).outcomes[0]
    assert o.critical and "kein Tool-Calling" in o.issues["critical"][0]


async def test_routing_failure_vs_correct_refusal() -> None:
    small = model("small", context_length=8192)
    f = fleet(small)
    t_ok = task(
        set="GENERAL",
        expected_category="GENERAL",
        required_capabilities=[],
        requires_vision=False,
        prompt="Write a haiku.",
    )
    t_huge = task(
        id="t-2",
        set="LONG_CONTEXT",
        expected_category="LONG_CONTEXT",
        expected_secondary="GENERAL",
        required_capabilities=["long_context"],
        requires_vision=False,
        context_tokens=500_000,
        estimated_context="very_large",
        expected_routable=False,
    )
    report = await RoutingEvaluator(f, latency_repeats=1).evaluate([t_ok, t_huge], f.scenarios[0])
    assert report.routing_failures == 0 and report.correct_refusals == 1
    assert report.outcomes[1].category is RoutingCategory.LONG_CONTEXT  # trotzdem klassifiziert
    assert report.critical_errors == 0


async def test_underpowered_model_is_major() -> None:
    weak = model("weak", reasoning_capability="basic", coding_capability="basic", speed="fast")
    strong = model("strong", reasoning_capability="strong")
    f = fleet(weak, strong)
    t = task(
        set="REASONING",
        expected_category="REASONING",
        expected_complexity="HIGH",
        required_capabilities=["reasoning"],
        requires_vision=False,
        prompt="Prove it.",
    )
    ev = RoutingEvaluator(
        f,
        router_factory=lambda r, a: FixedRouter(r, a, "weak", RoutingCategory.REASONING),
        latency_repeats=1,
    )
    o = (await ev.evaluate([t], f.scenarios[0])).outcomes[0]
    assert o.issues["major"] and not o.critical
    assert o.penalty == WEIGHTS["major"] + WEIGHTS["complexity"]


async def test_calibration_and_overconfidence() -> None:
    f = fleet(SEER, OMNI)
    ev = RoutingEvaluator(
        f,
        router_factory=lambda r, a: FixedRouter(
            r, a, "omni", RoutingCategory.CODING, confidence=0.9
        ),
        latency_repeats=1,
    )
    report = await ev.evaluate([task()], f.scenarios[0])
    _bins, ece, brier = report.calibration()
    assert ece == pytest.approx(0.9) and brier == pytest.approx(0.81)
    assert len(report.overconfident_errors()) == 1


# ---------------------------------------------------------------------- echter Router


@pytest.fixture(scope="module")
def baseline(dataset: list[RoutingTask]) -> list[ScenarioReport]:
    f = Fleet.from_toml()
    return asyncio.run(RoutingEvaluator(f, latency_repeats=1).evaluate_all(dataset))


async def test_baseline_is_reproducible(
    dataset: list[RoutingTask], baseline: list[ScenarioReport]
) -> None:
    again = await RoutingEvaluator(Fleet.from_toml(), latency_repeats=1).evaluate_all(dataset)
    for a, b in zip(baseline, again, strict=True):
        sa, sb = a.summary(), b.summary()
        sa.pop("latency"), sb.pop("latency")
        assert sa == sb


async def test_router_never_picks_unavailable_models(baseline: list[ScenarioReport]) -> None:
    for report in baseline:
        assert report.unavailable_violations == 0
        assert report.routing_failures == 0
        assert report.correct_refusals == 2
        assert report.latency()["mean_ms"] > 0


async def test_fallback_rate_only_in_degraded_scenario(baseline: list[ScenarioReport]) -> None:
    full, degraded = baseline
    assert full.fallback_rate == 0.0
    assert degraded.fallback_rate > 0.0


async def test_report_rendering(baseline: list[ScenarioReport], tmp_path: Path) -> None:
    from evaluation.routing_evaluator import DEFAULT_FLEET

    text = render_report(
        baseline,
        dataset=DEFAULT_DATASET,
        fleet=DEFAULT_FLEET,
        router_name="rules",
        generated="2026-01-01",
        recommendations=["eins", "zwei"],
    )
    for heading in (
        "Gesamtergebnis",
        "Kritische Fehler",
        "Häufigste Fehlklassifikationen",
        "Problematische Aufgaben",
        "Confidence",
        "Verbesserungsvorschläge",
    ):
        assert heading in text
    write_json(baseline, tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert [s["scenario"] for s in data["scenarios"]] == ["all_available", "degraded"]
    assert len(data["outcomes"]["degraded"]) == 150
