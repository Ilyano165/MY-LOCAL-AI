from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evaluation.benchmark_results import MeasurementStatus, ProfileStore
from evaluation.benchmark_runner import ModelBenchmarkRunner, attach_profiles, render_profile
from evaluation.benchmark_tasks import suite_version
from evaluation.hardware import HardwareProfile, ResourceSampler, Sample
from evaluation.model_benchmark import BenchmarkOptions, ModelBenchmark, ServerLauncher
from models.capabilities import CapabilityLevel, Speed
from models.inference import InferenceEngine, ProviderRegistry
from models.measured import DataStatus
from models.model_registry import ModelRegistry
from tests.evaluation.bench_fakes import FakeBenchServer
from tests.evaluation.test_gguf import write_gguf
from tests.models.fakes import make_meta

M = MeasurementStatus
FAST = BenchmarkOptions(throughput_runs=1, throughput_tokens=32, long_context_max_tokens=2000)


def hardware() -> HardwareProfile:
    return HardwareProfile(
        os="Linux test",
        arch="x86_64",
        python="3.13",
        cpu_model="Test CPU",
        cpu_cores_logical=8,
        ram_total_gb=32.0,
    )


def fake_sampler(vram: list[float | None], ram: list[float]) -> ResourceSampler:
    values = iter(list(zip(vram, ram, strict=True)))
    last = (vram[-1], ram[-1])

    async def probe() -> Sample:
        v, r = next(values, last)
        return Sample(v, r)

    return ResourceSampler(interval_s=0.01, probe=probe, gpu_available=vram[0] is not None)


async def bench(
    server: FakeBenchServer, tmp_path: Path, options: BenchmarkOptions = FAST, **kw: object
) -> ModelBenchmark:
    kw.setdefault("sampler", fake_sampler([1000, 7000, 6500], [8000, 9000, 8500]))
    return ModelBenchmark(
        server.provider(),
        make_meta(),
        hardware(),
        workdir=tmp_path,
        options=options,
        **kw,  # type: ignore[arg-type]
    )


async def test_full_profile_good_model(tmp_path: Path) -> None:
    server = FakeBenchServer(skill="good", per_token_s=0.002, vision_enabled=True)
    profile = await (await bench(server, tmp_path)).run()
    caps = profile.capabilities
    for key in (
        "general",
        "reasoning",
        "coding",
        "tool_calling",
        "agentic",
        "long_context",
        "vision",
    ):
        assert caps[key].status is M.MEASURED, (key, caps[key])
        assert caps[key].value == 1.0, (key, caps[key].note)
    perf = profile.performance
    assert perf["tokens_per_second"].status is M.MEASURED
    assert 250 < float(perf["tokens_per_second"].value or 0) < 800  # Soll 500 tok/s
    assert perf["tokens_per_second_runtime"].status is M.REPORTED
    assert perf["tokens_per_second_runtime"].value == pytest.approx(500)
    assert perf["prompt_tokens_per_second"].status is M.REPORTED
    assert perf["first_token_latency"].status is M.MEASURED
    assert perf["load_time"].status is M.NOT_MEASURABLE  # llama.cpp lädt beim Serverstart
    assert perf["peak_vram"].value == pytest.approx(7000 / 1024, abs=0.01)
    assert perf["peak_ram"].value == pytest.approx(9000 / 1024, abs=0.01)
    assert perf["memory_footprint"].status is M.NOT_MEASURABLE  # nur systemweit, kein Ausgang
    info = profile.model_info
    assert info["context_length"].status is M.REPORTED and info["context_length"].value == 32768
    assert info["parameter_count"].status is M.REPORTED
    assert info["model_size"].status is M.REPORTED
    assert info["quantization"].status is M.CONFIGURED
    assert profile.runtime["runtime"] == "llama.cpp"
    assert profile.benchmark_version == suite_version()
    assert profile.configured["status"] == "CONFIGURED"
    assert "MODELLFÄHIGKEIT" in render_profile(profile)


async def test_bad_model_scores_low(tmp_path: Path) -> None:
    profile = await (await bench(FakeBenchServer(skill="bad"), tmp_path)).run()
    assert float(profile.capabilities["reasoning"].value or 0) < 0.5
    assert float(profile.capabilities["coding"].value or 0) < 0.7
    assert profile.capabilities["vision"].status is M.NOT_SUPPORTED


async def test_slow_model_is_not_a_bad_model(tmp_path: Path) -> None:
    """Kernanforderung: Fähigkeit und Hardware-Leistung strikt getrennt."""
    server = FakeBenchServer(skill="good", per_token_s=0.08, overhead_s=0.0)
    options = BenchmarkOptions(
        throughput_runs=1,
        throughput_tokens=8,
        tasks=["chat", "complex_analysis", "coding", "debugging"],
    )
    profile = await (await bench(server, tmp_path, options)).run()
    overrides = profile.to_overrides(
        hardware_fingerprint=hardware().fingerprint, benchmark_version=suite_version()
    )
    assert overrides.speed is Speed.SLOW  # ≈ 12 tok/s → langsam
    assert overrides.coding_capability is CapabilityLevel.STRONG  # trotzdem stark
    assert overrides.reasoning_capability is CapabilityLevel.STRONG


async def test_unsupported_tools_marked_not_supported(tmp_path: Path) -> None:
    server = FakeBenchServer(tools_enabled=False)
    options = BenchmarkOptions(tasks=["tool_calling", "agent_multistep"], performance=False)
    profile = await (await bench(server, tmp_path, options)).run()
    assert profile.capabilities["tool_calling"].status is M.NOT_SUPPORTED
    o = profile.to_overrides(hardware_fingerprint=None, benchmark_version=None)
    assert o.tool_calling is False
    assert "tokens_per_second" not in profile.performance


async def test_runtime_errors_never_become_measurements(tmp_path: Path) -> None:
    profile = await (await bench(FakeBenchServer(fail_chat=True), tmp_path)).run()
    assert not any(m.is_fact for m in profile.capabilities.values())
    assert profile.capabilities["coding"].status is M.FAILED
    assert profile.performance["tokens_per_second"].status is M.FAILED
    o = profile.to_overrides(hardware_fingerprint=None, benchmark_version=None)
    # nur, was die Runtime selbst meldet: Kontextfenster und „keine Bild-Unterstützung“
    assert o.applied_fields() == ["vision_capability", "context_length"]


async def test_gguf_quantization_detected(tmp_path: Path) -> None:
    gguf = write_gguf(tmp_path / "model.gguf", file_type=17)
    server = FakeBenchServer(model_path=str(gguf))
    options = BenchmarkOptions(tasks=["chat"], performance=False)
    profile = await (await bench(server, tmp_path / "w", options)).run()
    q = profile.model_info["quantization"]
    assert q.status is M.DETECTED and q.value == "Q5_K_M"
    assert "Konfiguration sagt Q4_K_M" in q.note
    assert profile.model_info["architecture"].value == "llama"


async def test_unknown_task_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unbekannte Aufgaben"):
        await bench(FakeBenchServer(), tmp_path, BenchmarkOptions(tasks=["nope"]))


class TestOllama:
    OPTIONS = BenchmarkOptions(tasks=["chat"], throughput_runs=1, throughput_tokens=16)

    async def test_cold_model_load_is_measured(self, tmp_path: Path) -> None:
        server = FakeBenchServer(runtime="ollama", ollama_loaded=False)
        profile = await (await bench(server, tmp_path, self.OPTIONS)).run()
        load = profile.performance["load_time"]
        assert load.status is M.MEASURED and float(load.value or 0) >= 0.05
        assert "load_duration 0.05" in load.note
        assert server.loads == 1 and server.unloads == 0
        assert profile.performance["memory_footprint"].status is M.REPORTED
        assert profile.performance["vram_reported"].value == pytest.approx(5.122, abs=0.01)
        assert profile.model_info["context_length"].value == 8192
        assert profile.runtime["runtime_version"] == "0.12.3"
        assert "tokens_per_second_runtime" not in profile.performance  # Ollama: keine timings

    async def test_preloaded_model_not_measurable_without_opt_in(self, tmp_path: Path) -> None:
        server = FakeBenchServer(runtime="ollama", ollama_loaded=True)
        profile = await (await bench(server, tmp_path, self.OPTIONS)).run()
        assert profile.performance["load_time"].status is M.NOT_MEASURABLE
        assert server.unloads == 0

    async def test_measure_load_reloads(self, tmp_path: Path) -> None:
        server = FakeBenchServer(runtime="ollama", ollama_loaded=True)
        options = BenchmarkOptions(tasks=["chat"], performance=False, measure_load=True)
        profile = await (await bench(server, tmp_path, options)).run()
        assert profile.performance["load_time"].status is M.MEASURED
        assert server.unloads == 1 and server.loads == 1


async def test_launcher_measures_load_time(tmp_path: Path) -> None:
    server = FakeBenchServer(ready_after=3)
    launcher = ServerLauncher([sys.executable, "-c", "import time; time.sleep(60)"])
    options = BenchmarkOptions(tasks=["chat"], throughput_runs=1, throughput_tokens=16)
    sampler = fake_sampler([None, None, None], [4000, 10000, 9000])
    profile = await (
        await bench(server, tmp_path, options, launcher=launcher, sampler=sampler)
    ).run()
    load = profile.performance["load_time"]
    assert load.status is M.MEASURED and "von NOVA gestartet" in load.method
    assert float(load.value or 0) >= 0.2  # drei „lädt noch“-Antworten à 0.1 s
    footprint = profile.performance["memory_footprint"]
    assert footprint.status is M.MEASURED and "System-Delta" in footprint.method
    assert footprint.value == pytest.approx(6000 / 1024, abs=0.01)
    assert launcher.process is not None and launcher.process.returncode is not None  # beendet
    assert profile.performance["peak_vram"].status is M.NOT_MEASURABLE


async def test_launcher_detects_crash(tmp_path: Path) -> None:
    from models.base import ModelError

    launcher = ServerLauncher([sys.executable, "-c", "raise SystemExit(3)"])
    server = FakeBenchServer(ready_after=10_000)
    with pytest.raises(ModelError, match="Code 3"):
        await launcher.start(server.provider())


def test_launcher_rejects_empty_command() -> None:
    with pytest.raises(ValueError):
        ServerLauncher("   ")


class TestRunner:
    def engine(self, server: FakeBenchServer) -> InferenceEngine:
        return InferenceEngine(ModelRegistry([make_meta()]), ProviderRegistry([server.provider()]))

    async def test_runs_and_saves_profile(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles")
        runner = ModelBenchmarkRunner(
            self.engine(FakeBenchServer()),
            store,
            hardware=hardware(),
            options=BenchmarkOptions(
                tasks=["chat", "coding"], throughput_runs=1, throughput_tokens=16
            ),
        )
        outcomes = await runner.run()
        assert [o.ok for o in outcomes] == [True]
        assert outcomes[0].path == store.path_for("test-model")
        assert store.load("test-model") is not None

    async def test_router_uses_saved_profile(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        engine = self.engine(FakeBenchServer(per_token_s=0.0005))
        await ModelBenchmarkRunner(engine, store, hardware=hardware(), options=FAST).run()
        assert engine.models.data_status("test-model").status is DataStatus.UNMEASURED
        await attach_profiles(engine.models, tmp_path, hardware=hardware())
        status = engine.models.data_status("test-model")
        assert status.status is DataStatus.MEASURED
        effective = engine.models.effective("test-model")
        assert effective.coding_capability is CapabilityLevel.STRONG  # konfiguriert: good
        assert effective.speed is Speed.FAST  # konfiguriert: medium

    async def test_unready_runtime_saves_nothing(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        server = FakeBenchServer(ready_after=10_000)
        outcomes = await ModelBenchmarkRunner(self.engine(server), store, hardware=hardware()).run()
        assert outcomes[0].error and "nicht bereit" in outcomes[0].error
        assert store.list() == []

    async def test_failed_run_saves_nothing(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        server = FakeBenchServer(fail_chat=True)
        options = BenchmarkOptions(tasks=["chat"], throughput_runs=1, throughput_tokens=16)
        runner = ModelBenchmarkRunner(
            self.engine(server), store, hardware=hardware(), options=options
        )
        # Runtime meldet Kontext (REPORTED in model_info), aber kein Fähigkeits-/Leistungswert
        outcomes = await runner.run()
        assert outcomes[0].error and "kein einziger Messwert" in outcomes[0].error
        assert store.list() == []

    async def test_unknown_model(self, tmp_path: Path) -> None:
        runner = ModelBenchmarkRunner(
            self.engine(FakeBenchServer()), ProfileStore(tmp_path), hardware=hardware()
        )
        outcomes = await runner.run(["ghost"])
        assert outcomes[0].error and "ghost" in outcomes[0].error
