"""Trainings-Grundlagen: Datensätze (Filter, Splits, Lecks, Versionen), Probe, Plan."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training.dataset import (
    DatasetError,
    DatasetStore,
    filter_examples,
    from_findings,
    from_jsonl,
    leak_report,
    make_example,
)
from training.plan import TrainingConfig, make_plan
from training.probe import probe


def finding(i: int, statement: str, question: str, **kw: Any) -> dict[str, Any]:
    return {
        "id": f"r1:F{i}",
        "run_id": "r1",
        "subquestion": question,
        "statement": statement,
        "classification": kw.get("classification", "supported"),
        "stale": kw.get("stale", False),
        "sources": kw.get(
            "sources",
            [
                {
                    "url": f"https://docs.example.org/{i}",
                    "title": f"Doc {i}",
                    "fetched_at": "2026-10-10T00:00:00+00:00",
                    "via": "search:searxng",
                }
            ],
        ),
    }


def many(n: int, offset: int = 0) -> list[dict[str, Any]]:
    return [
        finding(
            i, f"Fact number {i} about topic {i} is well documented here.", f"What about topic {i}?"
        )
        for i in range(offset, offset + n)
    ]


# ---------------------------------------------------------------------- Quellen & Filter


def test_from_findings_only_supported_and_allowed_sources() -> None:
    rows = [
        finding(1, "The NX-1 battery has 5000 mAh capacity.", "NX-1 battery capacity?"),
        finding(2, "Disputed claim.", "Q2?", classification="contested"),
        finding(3, "Old fact.", "Q3?", stale=True),
        finding(
            4, "Brave fact here.", "Q4?", sources=[{"url": "https://x.org", "via": "search:brave"}]
        ),
        finding(5, "No source.", "Q5?", sources=[]),
    ]
    examples, skipped = from_findings(rows)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.messages[1]["content"] == "NX-1 battery capacity?"
    assert "Sources: Doc 1 (https://docs.example.org/1)" in ex.assistant
    assert ex.provenance["type"] == "research_finding" and ex.provenance["finding_id"] == "r1:F1"
    assert skipped == {
        "classification:contested": 1,
        "stale": 1,
        "source_terms_forbid_training": 1,
        "no_source": 1,
    }


def test_quality_filters() -> None:
    examples = [
        make_example("What is X?", "X is a thing that exists.", {"type": "manual"}),
        make_example("What is X?", "X is a thing that exists.", {"type": "manual"}),  # Dublette
        make_example("Hi", "Hello", {}),  # zu kurz
        make_example("What is my key?", "api_key = abcdef1234567890", {}),
        make_example("Contact?", "Write to max.mustermann@example.com please", {}),
        make_example("Phone number?", "Call +49 170 1234567 tomorrow", {}),
        make_example("Same question here?", "Answer A is correct.", {}),
        make_example("Same question here?", "Answer B is correct.", {}),  # Widerspruch
    ]
    kept, dropped = filter_examples(examples)
    assert [e.user for e in kept] == ["What is X?"]
    assert dropped == {
        "duplicate": 1,
        "too_short": 1,
        "secret": 1,
        "personal_data": 2,
        "conflicting_answers": 2,
    }


def test_from_jsonl(tmp_path: Path) -> None:
    good = tmp_path / "manual.jsonl"
    good.write_text(
        json.dumps({"question": "What is NOVA?", "answer": "A local AI platform."}) + "\n\n"
    )
    assert from_jsonl(good)[0].provenance == {
        "type": "manual",
        "file": "manual.jsonl",
        "line": 1,
        "note": "",
    }
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "x"}\n')
    with pytest.raises(DatasetError, match=r"bad\.jsonl:1"):
        from_jsonl(bad)


# ---------------------------------------------------------------------- Versionen & Splits


def test_build_versions_are_immutable_and_splits_stable(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    examples, skipped = from_findings(many(60))
    m1 = store.build("nova-knowledge", examples, skipped=skipped, description="first")
    assert m1["version"] == "0.1.0" and m1["parent"] is None
    assert sum(m1["counts"].values()) == 60 and m1["counts"]["train"] > m1["counts"]["test"] > 0
    assert store.verify("nova-knowledge", "0.1.0") == []
    with pytest.raises(DatasetError, match="immutable"):
        store.build("nova-knowledge", examples, version="0.1.0")
    first = {
        s: {e.group for e in store.load("nova-knowledge", "0.1.0", s)}
        for s in ("train", "val", "test")
    }
    more, _ = from_findings(many(80))  # alte 60 + 20 neue
    m2 = store.build("nova-knowledge", more)
    assert m2["version"] == "0.2.0" and m2["parent"] == "0.1.0"
    second = {
        s: {e.group for e in store.load("nova-knowledge", "0.2.0", s)}
        for s in ("train", "val", "test")
    }
    for split in first:
        assert first[split] <= second[split]  # vergebene Gruppen behalten ihren Split
    assert not (second["train"] & second["test"])
    assert [d["version"] for d in store.list_all()] == ["0.1.0", "0.2.0"]
    # Manipulation wird erkannt
    path = tmp_path / "datasets" / "nova-knowledge" / "0.1.0" / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    assert store.verify("nova-knowledge", "0.1.0") == ["train.jsonl was modified"]


def test_leaks_are_removed_from_evaluation(tmp_path: Path) -> None:
    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    train = make_example("Tell me about foxes?", base, {})
    test = make_example("Different question on foxes?", base + " indeed", {})
    report = leak_report({"train": [train], "val": [], "test": [test]})
    assert report and report[0]["split"] == "test" and report[0]["overlaps_with"] == [train.id]


def test_build_rejects_bad_names_and_empty(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    with pytest.raises(DatasetError):
        store.build("Bad Name", [])
    with pytest.raises(DatasetError, match="no training examples"):
        store.build("empty", [make_example("Hi", "Yo", {})])


# ---------------------------------------------------------------------- Hardware-Probe


def fake_torch(
    capability: tuple[int, int], vram_gb: float, arch: list[str], name: str = "GPU"
) -> Any:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda i: capability,
        get_device_properties=lambda i: SimpleNamespace(name=name, total_memory=vram_gb * 1024**3),
        is_bf16_supported=lambda: capability[0] >= 8,
        get_arch_list=lambda: arch,
    )
    return SimpleNamespace(
        __version__="2.14.0+cu126", version=SimpleNamespace(cuda="12.6"), cuda=cuda
    )


def importer_for(torch: Any, missing: tuple[str, ...] = ()) -> Any:
    def imp(name: str) -> Any:
        if name == "torch":
            if torch is None:
                raise ImportError(name)
            return torch
        if name in missing:
            raise ImportError(name)
        return SimpleNamespace(__version__="1.0")

    return imp


def test_probe_gtx_1080_ti_with_pascal_kernels() -> None:
    r = probe(
        importer_for(
            fake_torch((6, 1), 11.0, ["sm_60", "sm_70", "sm_80"], "NVIDIA GeForce GTX 1080 Ti")
        )
    )
    assert r.ready and r.compute_capability == "6.1" and r.compute_dtype == "float32"
    assert r.vram_gb == 11.0 and r.bf16 is False and r.max_model_b_qlora == 12.7
    assert any("Pascal" in n for n in r.notes)


def test_probe_detects_pytorch_build_without_pascal_kernels() -> None:
    r = probe(importer_for(fake_torch((6, 1), 11.0, ["sm_75", "sm_80", "sm_90"])))
    assert not r.ready and any("cu126" in p for p in r.problems)


def test_probe_modern_gpu_and_missing_packages() -> None:
    r = probe(importer_for(fake_torch((8, 6), 24.0, ["sm_86"]), missing=("trl", "bitsandbytes")))
    assert r.compute_dtype == "bfloat16" and r.bf16
    assert "Python package 'trl' is missing" in r.problems
    assert any("bitsandbytes missing" in n for n in r.notes)
    assert probe(importer_for(None)).problems == ["PyTorch is not installed in this environment"]


# ---------------------------------------------------------------------- Plan


def config(**kw: Any) -> TrainingConfig:
    base = dict(
        name="nova-qwen3.5-4b",
        base_model="Qwen/Qwen3.5-4B-Instruct",
        base_revision="abc123",
        base_license="apache-2.0",
        dataset="nova-knowledge@0.1.0",
    )
    base.update(kw)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def test_plan_ok_on_1080_ti(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    examples, _ = from_findings(many(300))
    store.build("nova-knowledge", examples)
    hw = probe(importer_for(fake_torch((6, 1), 11.0, ["sm_60"]))).to_dict()
    plan = make_plan(config(), store, hw)
    assert plan.ok, plan.problems
    assert plan.estimated_vram_gb is not None and plan.estimated_vram_gb < 11
    assert any("fp32" in w for w in plan.warnings)
    assert plan.config_hash == config().config_hash() and len(plan.config_hash) == 16
    assert config(seed=1).config_hash() != plan.config_hash


def test_plan_problems(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    hw = probe(importer_for(fake_torch((6, 1), 11.0, ["sm_60"]))).to_dict()
    plan = make_plan(config(base_revision="main", base_model="Qwen/Qwen3.5-27B"), store, hw)
    assert not plan.ok
    text = " | ".join(plan.problems)
    assert "pin an exact revision" in text and "not found" in text and "GB VRAM needed" in text
    no_hw = make_plan(config(), store, None)
    assert any("hardware probe" in w for w in no_hw.warnings)


def test_config_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        'name = "n"\nbase_model = "Qwen/Qwen3.5-4B"\nbase_revision = "r1"\n'
        'base_license = "apache-2.0"\ndataset = "d@0.1.0"\nlora_r = 8\n'
    )
    assert TrainingConfig.from_toml(path).lora_r == 8
    path.write_text('name = "n"\nunknown = 1\n')
    with pytest.raises(ValueError):
        TrainingConfig.from_toml(path)
