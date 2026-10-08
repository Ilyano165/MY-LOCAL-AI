"""Learned Router: harte Regeln bleiben vorrangig, kein Leakage, Persistenz, Trainingsdaten."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from models.capabilities import ModelMetadata
from models.model_registry import ModelRegistry
from router import (
    LearnedRanker,
    LearnedRouter,
    LearnedRoutingError,
    MemoryRoutingLog,
    NoModelAvailableError,
    RoutingRequest,
    RuleBasedRouter,
    StaticAvailability,
)
from router.base import Complexity, RoutingCategory
from router.feature_extractor import FeatureExtractor, HistoryStats, TaskView
from router.ranking_model import Example, RankingModel, TrainingConfig
from router.routing_log import JsonlRoutingLog
from router.training_data import (
    OutcomeRecord,
    build_examples,
    from_routing_log,
    split_by_task,
    task_key,
)
from tests.router.conftest import fleet, model


def view(text: str = "Fix the failing test in parser.py", **kw: object) -> TaskView:
    data: dict[str, object] = {
        "text": text,
        "category": RoutingCategory.CODING,
        "complexity": Complexity.MEDIUM,
        "confidence": 0.7,
        "needs_tools": False,
        "needs_vision": False,
        "context_tokens": 0,
    }
    data.update(kw)
    return TaskView(**data)  # type: ignore[arg-type]


class FixedRanker(LearnedRanker):
    """Ranker mit vorgegebenen Scores – simuliert auch fehlerhafte Vorschläge."""

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__(model=RankingModel(names=[]))
        self.fixed = scores
        self.seen: list[list[str]] = []

    def predict(self, task: TaskView, candidates: Sequence[ModelMetadata]) -> dict[str, float]:
        self.seen.append([m.name for m in candidates])
        return dict(self.fixed)


def learned(
    models: list[ModelMetadata],
    ranker: LearnedRanker | None,
    available: Sequence[str],
    on_ranker_error: Literal["raise", "rules"] = "rules",
) -> LearnedRouter:
    return LearnedRouter(
        ModelRegistry(models),
        StaticAvailability(available),
        ranker,
        log=MemoryRoutingLog(),
        on_ranker_error=on_ranker_error,
    )


CODE_TASK = RoutingRequest("Refactor the parser module and fix the failing unit tests")


# ---------------------------------------------------------------------- harte Regeln


async def test_ranker_only_sees_valid_candidates() -> None:
    models = fleet()
    ranker = FixedRanker({"coder": 1.0})
    names = [m.name for m in models if m.name != "thinker"]
    router = learned(models, ranker, names)
    decision = await router.route(RoutingRequest("Describe this", needs_vision=True))
    assert decision.model.name == "seer"  # einziges Vision-Modell
    assert ranker.seen == [["seer"]]  # Ranker bekam nur den gültigen Kandidaten


async def test_invalid_proposal_is_rejected_and_logged() -> None:
    models = fleet()
    available = [m.name for m in models if m.name != "coder"]
    # Ranker bevorzugt ein nicht verfügbares Modell und ein unbekanntes
    ranker = FixedRanker({"coder": 9.0, "ghost": 8.0, "small-fast": 1.0, "thinker": 0.5})
    router = learned(models, ranker, available)
    decision = await router.route(CODE_TASK)
    assert decision.model.name != "coder" and decision.model.name in available
    assert router.guard_interventions == 2
    assert any("coder verworfen" in n and "nicht verfügbar" in n for n in decision.notes)
    assert any("ghost verworfen" in n for n in decision.notes)
    assert decision.ranker == "learned"
    log = router.log
    assert isinstance(log, MemoryRoutingLog)
    assert any("verworfen" in n for n in log.decisions()[0]["notes"])


async def test_proposal_without_vision_rejected_for_vision_task() -> None:
    models = fleet()
    ranker = FixedRanker({"coder": 5.0, "seer": 1.0})
    router = learned(models, ranker, [m.name for m in models])
    decision = await router.route(RoutingRequest("What is in this picture?", needs_vision=True))
    assert decision.model.name == "seer"
    assert any("coder verworfen" in n for n in decision.notes)


async def test_no_valid_candidate_raises_clear_error() -> None:
    models = [model("only", vision_capability="none")]
    router = learned(models, FixedRanker({"only": 1.0}), ["only"])
    with pytest.raises(NoModelAvailableError, match="keine Vision-Fähigkeit"):
        await router.route(RoutingRequest("look", needs_vision=True))


async def test_missing_ranker_is_explicit_not_silent() -> None:
    models = fleet()
    names = [m.name for m in models]
    decision = await learned(models, None, names).route(CODE_TASK)
    rule = await RuleBasedRouter(ModelRegistry(models), StaticAvailability(names)).route(CODE_TASK)
    assert decision.model.name == rule.model.name
    assert decision.ranker.startswith("rules (learned unavailable")
    assert any("Regelrangfolge verwendet" in n for n in decision.notes)
    with pytest.raises(LearnedRoutingError):
        await learned(models, None, names, on_ranker_error="raise").route(CODE_TASK)


async def test_non_finite_scores_are_caught() -> None:
    class Broken(FixedRanker):
        def predict(self, task: TaskView, candidates: Sequence[ModelMetadata]) -> dict[str, float]:
            raise ValueError("Score ist nicht endlich")

    models = fleet()
    names = [m.name for m in models]
    decision = await learned(models, Broken({}), names).route(CODE_TASK)
    assert decision.ranker.startswith("rules (")
    with pytest.raises(LearnedRoutingError):
        await learned(models, Broken({}), names, on_ranker_error="raise").route(CODE_TASK)


async def test_partial_scores_keep_all_candidates() -> None:
    models = fleet()
    names = [m.name for m in models]
    decision = await learned(models, FixedRanker({"thinker": 1.0}), names).route(CODE_TASK)
    assert decision.model.name == "thinker"
    assert any("ohne Learned-Score" in n for n in decision.notes)
    assert len(decision.considered) > 1


# ---------------------------------------------------------------------- Ranking-Modell


def toy_examples() -> list[Example]:
    out = []
    for g in range(20):
        for name, value, util in (("good", 1.0, 0.9), ("bad", 0.0, 0.1)):
            out.append(
                Example(
                    f"g{g}",
                    f"t{g}",
                    name,
                    {"a": value + 0.01 * g, "b": 0.5},
                    util,
                    util > 0.5,
                    "test",
                )
            )
    return out


def test_ranking_model_learns_and_round_trips(tmp_path: Path) -> None:
    rm = RankingModel(names=["a", "b"], config=TrainingConfig(epochs=30))
    stats = rm.fit(toy_examples())
    assert stats["train_pair_accuracy"] == 1.0
    assert rm.score({"a": 1.0}) > rm.score({"a": 0.0})
    assert 0 < rm.success_probability({"a": 1.0}) <= 1
    rm.save(tmp_path / "m.json")
    again = RankingModel.load(tmp_path / "m.json")
    assert again.score({"a": 0.3, "b": 0.5}) == pytest.approx(rm.score({"a": 0.3, "b": 0.5}))
    assert rm.top_weights(1)[0][0] == "a"


def test_ranking_model_is_deterministic() -> None:
    a, b = RankingModel(names=["a", "b"]), RankingModel(names=["a", "b"])
    a.fit(toy_examples())
    b.fit(toy_examples())
    assert a.weights == b.weights


def test_ranking_model_errors(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        RankingModel(names=["a"]).score({"a": 1})
    with pytest.raises(ValueError):
        RankingModel(names=["a"]).fit([])
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError):
        RankingModel.load(path)


# ---------------------------------------------------------------------- Trainingsdaten


def records(task_ids: Sequence[str]) -> list[OutcomeRecord]:
    out = []
    for tid in task_ids:
        v = view(text=f"Implement the parser function for case {tid}")
        out.append(OutcomeRecord(f"{tid}@s", tid, v, "coder", True, 0.9, 20.0))
        out.append(OutcomeRecord(f"{tid}@s", tid, v, "small-fast", False, 0.4, 3.0))
    return out


def test_split_by_task_is_disjoint_and_stable() -> None:
    recs = records([f"t{i}" for i in range(30)])
    train, test = split_by_task(recs)
    assert not {r.task_id for r in train} & {r.task_id for r in test}
    assert {r.task_id for r in test} == {r.task_id for r in split_by_task(recs)[1]}
    fixed_train, fixed_test = split_by_task(recs, test_task_ids=["t1", "t2"])
    assert {r.task_id for r in fixed_test} == {"t1", "t2"}
    assert len(fixed_train) == len(recs) - 4


def test_leave_one_task_out_history_excludes_own_outcome() -> None:
    models = {m.name: m for m in fleet()}
    single = records(["only"])
    examples, _ = build_examples(single, models)
    coder = next(e for e in examples if e.model == "coder")
    prior = HistoryStats().features("coder", RoutingCategory.CODING)
    assert coder.features["hist_success"] == pytest.approx(prior["hist_success"])
    assert coder.features["hist_count"] == 0


def test_utility_prefers_quality_then_latency() -> None:
    v = view()
    fast = OutcomeRecord("g", "t", v, "a", True, 0.9, 5.0)
    slow = OutcomeRecord("g", "t", v, "b", True, 0.9, 200.0)
    weak = OutcomeRecord("g", "t", v, "c", False, 0.6, 1.0)
    broken = OutcomeRecord("g", "t", v, "d", False, 0.6, 1.0, hard_failure=True)
    assert fast.utility > slow.utility > weak.utility > broken.utility
    with pytest.raises(ValueError):
        OutcomeRecord("g", "t", v, "a", True, 1.5)


async def test_routing_log_round_trip_to_training_records(tmp_path: Path) -> None:
    models = fleet()
    log = JsonlRoutingLog(tmp_path / "routing.jsonl")
    router = RuleBasedRouter(
        ModelRegistry(models), StaticAvailability([m.name for m in models]), log=log
    )
    done = await router.route(CODE_TASK)
    await router.route(RoutingRequest("hello"))  # ohne Ergebnis
    router.record_outcome(done.id, success=True, verdict="success", quality=0.8, latency_ms=4200.0)
    recs, skipped = from_routing_log(log.joined())
    assert skipped == 1 and len(recs) == 1
    r = recs[0]
    assert r.model == done.model.name and r.quality == 0.8 and r.latency_s == 4.2
    assert r.task_id == task_key(CODE_TASK.task) and r.source == "routing_log"
    assert r.task.category is done.classification.category


def test_task_key_normalizes_whitespace_and_case() -> None:
    assert task_key("Fix  the Bug") == task_key("fix the bug ")


# ---------------------------------------------------------------------- LearnedRanker


def trained_ranker() -> tuple[LearnedRanker, dict[str, ModelMetadata]]:
    models = {m.name: m for m in fleet()}
    ranker = LearnedRanker.train(
        records([f"t{i}" for i in range(12)]), models, config=TrainingConfig(epochs=10)
    )
    return ranker, models


def test_learned_ranker_train_predict_evaluate_save_load(tmp_path: Path) -> None:
    ranker, models = trained_ranker()
    v = view(text="Implement the parser function for case new")
    scores = ranker.predict(v, [models["coder"], models["small-fast"]])
    assert set(scores) == {"coder", "small-fast"} and scores["coder"] > scores["small-fast"]
    metrics = ranker.evaluate(records(["held-out-1", "held-out-2"]), models)
    assert metrics.groups == 2 and metrics.pair_accuracy == 1.0
    ranker.save(tmp_path / "r.json")
    loaded = LearnedRanker.load(tmp_path / "r.json")
    again = loaded.predict(v, [models["coder"], models["small-fast"]])
    assert again == pytest.approx(scores)


def test_evaluate_refuses_leaked_tasks() -> None:
    ranker, models = trained_ranker()
    with pytest.raises(ValueError, match="Data Leakage"):
        ranker.evaluate(records(["t1"]), models)


def test_load_rejects_incompatible_features(tmp_path: Path) -> None:
    ranker, _ = trained_ranker()
    ranker.save(tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    data["model"]["names"][0] = "renamed"
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Merkmalsnamen"):
        LearnedRanker.load(tmp_path / "r.json")
    data["feature_version"] = 999
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Merkmalsversion"):
        LearnedRanker.load(tmp_path / "r.json")


async def test_trained_ranker_in_router_respects_hard_rules() -> None:
    ranker, _ = trained_ranker()
    models = fleet()
    router = learned(models, ranker, [m.name for m in models if m.name != "coder"])
    decision = await router.route(CODE_TASK)
    assert decision.ranker == "learned" and decision.model.name != "coder"
    assert set(decision.scores) == set(decision.considered)
    assert all(math.isfinite(s) for s in decision.scores.values())


# ---------------------------------------------------------------------- Merkmale


def test_features_use_runtime_knowledge_only() -> None:
    fields = set(TaskView.__dataclass_fields__)
    assert not {f for f in fields if "expected" in f or "label" in f}
    extractor = FeatureExtractor()
    names = extractor.names()
    assert len(names) == len(set(names)) and names == extractor.names()
    m = fleet()[0]
    f1 = extractor.pair_features(m, view(), HistoryStats())
    f2 = extractor.pair_features(m, view(), HistoryStats())
    assert f1 == f2 and list(f1) == names
    assert all(math.isfinite(v) for v in f1.values())


def test_measured_values_become_features() -> None:
    base = model(
        "m", measured_tokens_per_second=50.0, measured_latency_s=0.4, data_status="MEASURED"
    )  # unbekannte Felder landen in ``extra``
    f = FeatureExtractor().model_features(base, view(), HistoryStats())
    assert f["m_tps_known"] == 1.0 and f["m_tps"] > 0 and f["m_latency_known"] == 1.0
    assert f["m_measured"] == 1.0
    plain = FeatureExtractor().model_features(model("n"), view(), HistoryStats())
    assert plain["m_tps_known"] == 0.0 and plain["m_measured"] == 0.0


def test_history_stats_round_trip_and_smoothing() -> None:
    h = HistoryStats()
    h.add("a", RoutingCategory.CODING, True, 1.0, False)
    f = h.features("a", RoutingCategory.CODING)
    assert 0.5 < f["hist_success"] < 1.0  # ein Erfolg macht ein Modell nicht „perfekt“
    again = HistoryStats.from_dict(json.loads(json.dumps(h.to_dict())))
    assert again.features("a", RoutingCategory.CODING) == f


def test_record_outcome_passes_latency() -> None:
    log = MemoryRoutingLog()
    router = RuleBasedRouter(ModelRegistry(fleet()), StaticAvailability([]), log=log)
    router.record_outcome("x", success=True, quality=0.7, latency_ms=12.5)
    assert log.entries[-1]["latency_ms"] == 12.5
