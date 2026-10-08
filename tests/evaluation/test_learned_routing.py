from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from evaluation.learned_routing import (
    compare,
    learned_builder,
    load_split,
    make_split,
    rule_builder,
    simulate_outcome,
    simulated_records,
    split_tasks,
)
from evaluation.routing_evaluator import Fleet, RoutingTask, load_dataset
from router.learned_router import LearnedRanker
from router.ranking_model import TrainingConfig


@pytest.fixture(scope="module")
def tasks() -> list[RoutingTask]:
    return load_dataset()


def test_fixed_split_file_is_disjoint_stratified_and_reproducible(
    tasks: list[RoutingTask],
) -> None:
    test_ids = load_split()
    assert test_ids == make_split(tasks)
    train, test = split_tasks(tasks, test_ids)
    assert not {t.id for t in train} & {t.id for t in test}
    assert len(train) + len(test) == len(tasks)
    sets = Counter(t.set for t in test)
    assert set(sets) == {t.set for t in tasks}  # jedes Set im Test vertreten
    assert {t.prompt for t in train}.isdisjoint({t.prompt for t in test})


def test_split_rejects_unknown_ids(tasks: list[RoutingTask]) -> None:
    with pytest.raises(ValueError):
        split_tasks(tasks, ["does-not-exist"])


def test_simulation_rules(tasks: list[RoutingTask]) -> None:
    fleet = {m.name: m for m in Fleet.from_toml().models}
    vision_code = next(t for t in tasks if t.id == "vis-003")
    assert simulate_outcome(vision_code, fleet["vision-mid"]).hard_failure
    assert simulate_outcome(vision_code, fleet["omni-large"]).success
    hard_reasoning = next(t for t in tasks if t.id == "reas-003")
    weak = simulate_outcome(hard_reasoning, fleet["fast-small"])
    strong = simulate_outcome(hard_reasoning, fleet["reasoner-large"])
    assert not weak.success and strong.success and strong.quality > weak.quality
    fast = next(t for t in tasks if t.id == "fast-001")
    assert (
        simulate_outcome(fast, fleet["fast-small"]).latency_s
        < simulate_outcome(fast, fleet["reasoner-large"]).latency_s
    )


def test_training_and_comparison_respect_hard_rules(tasks: list[RoutingTask]) -> None:
    async def run() -> list[dict[str, object]]:
        fleet = Fleet.from_toml()
        train, test = split_tasks(tasks, load_split())
        records = await simulated_records(train[:40], fleet)
        assert {r.source for r in records} == {"simulated"}
        assert not {r.task_id for r in records} & {t.id for t in test}
        ranker = LearnedRanker.train(
            records, {m.name: m for m in fleet.models}, config=TrainingConfig(epochs=5)
        )
        reports = await compare(
            test[:15], fleet, {"rule": rule_builder, "learned": learned_builder(ranker)}
        )
        return [r.metrics() for r in reports]

    metrics = asyncio.run(run())
    learned = [m for m in metrics if m["router"] == "learned"]
    assert len(learned) == 2
    for m in learned:
        assert m["routing_failures"] == 0
        assert m["ranker_unavailable"] == 0
        assert 0.0 <= m["routing_accuracy"] <= 1.0  # type: ignore[operator]
