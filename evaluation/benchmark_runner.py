"""Benchmark-Lauf über ein oder mehrere Modelle (sequenziell) und Speicherung der Profile.

Sequenziell, weil parallele Läufe sich Speicher und Rechenzeit teilen und damit jede
Leistungsmessung verfälschen würden. Profile werden nur gespeichert, wenn mindestens ein
echter Messwert entstanden ist – ein fehlgeschlagener Lauf erzeugt kein „gemessenes“ Profil.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluation.benchmark_results import ModelProfile, ProfileStore
from evaluation.benchmark_tasks import BenchmarkTask, standard_tasks, suite_version
from evaluation.hardware import HardwareProfile, detect_hardware
from evaluation.model_benchmark import BenchmarkOptions, ModelBenchmark, ServerLauncher
from models.base import ModelError
from models.inference import InferenceEngine
from models.model_registry import ModelRegistry


@dataclass
class BenchmarkOutcome:
    model: str
    profile: ModelProfile | None = None
    path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.profile is not None and self.error is None


class ModelBenchmarkRunner:
    def __init__(
        self,
        engine: InferenceEngine,
        store: ProfileStore,
        *,
        options: BenchmarkOptions | None = None,
        hardware: HardwareProfile | None = None,
        tasks: Sequence[BenchmarkTask] | None = None,
        launcher_factory: Callable[[str], ServerLauncher | None] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.options = options or BenchmarkOptions()
        self.hardware = hardware
        self.tasks = list(tasks) if tasks is not None else standard_tasks()
        self.launcher_factory = launcher_factory
        self.progress = progress or (lambda _msg: None)

    @property
    def version(self) -> str:
        return suite_version(self.tasks)

    async def run(self, model_names: Sequence[str] | None = None) -> list[BenchmarkOutcome]:
        hardware = self.hardware or await detect_hardware()
        self.hardware = hardware
        names = list(model_names) if model_names else [m.name for m in self.engine.models.list()]
        outcomes = []
        for name in names:
            self.progress(f"Benchmark {name} …")
            outcomes.append(await self.run_one(name, hardware))
        return outcomes

    async def run_one(self, name: str, hardware: HardwareProfile) -> BenchmarkOutcome:
        try:
            model = self.engine.models.get(name)
            provider = self.engine.providers.get(model.provider)
        except ModelError as exc:
            return BenchmarkOutcome(name, error=str(exc))
        launcher = self.launcher_factory(name) if self.launcher_factory else None
        if launcher is None:
            health = await provider.health()
            if not health.ready:
                return BenchmarkOutcome(name, error=f"Runtime nicht bereit: {health.detail}")
        with tempfile.TemporaryDirectory(prefix="nova-bench-") as tmp:
            bench = ModelBenchmark(
                provider,
                model,
                hardware,
                workdir=Path(tmp),
                options=self.options,
                tasks=self.tasks,
                launcher=launcher,
            )
            try:
                profile = await bench.run()
            except ModelError as exc:
                return BenchmarkOutcome(name, error=f"{type(exc).__name__}: {exc}")
        if not _has_facts(profile):
            return BenchmarkOutcome(
                name, profile=profile, error="kein einziger Messwert – Profil nicht gespeichert"
            )
        path = self.store.save(profile)
        return BenchmarkOutcome(name, profile=profile, path=path)


_ACTIVE_PERFORMANCE = (
    "load_time",
    "first_token_latency",
    "tokens_per_second",
    "prompt_tokens_per_second",
)


def _has_facts(profile: ModelProfile) -> bool:
    """Mindestens ein Fähigkeits- oder aktiver Leistungswert aus einem echten Lauf.

    Passives Speicher-Sampling allein zählt nicht: Ohne erfolgreiche Inferenz sagt es nichts
    über das Modell aus.
    """
    values = [
        *profile.capabilities.values(),
        *(profile.performance[k] for k in _ACTIVE_PERFORMANCE if k in profile.performance),
    ]
    return any(m.is_fact for m in values)


# ---------------------------------------------------------------------- Bericht


def render_profile(profile: ModelProfile) -> str:
    lines = [
        f"Modell: {profile.model}   (Benchmark {profile.benchmark_version}, {profile.created_at})",
        f"Runtime: {profile.runtime.get('runtime', '?')}"
        + (
            f" {profile.runtime['runtime_version']}"
            if profile.runtime.get("runtime_version")
            else ""
        ),
        "",
        "MODELLFÄHIGKEIT (hardwareunabhängig, Score 0–1 aus unabhängig geprüften Aufgaben)",
    ]
    for key, m in profile.capabilities.items():
        lines.append(f"  {key:<14} {m!s:<34} {m.note if m.value is not None else ''}".rstrip())
    lines += ["", "HARDWARE-LEISTUNG (nur gültig für diese Hardware/Runtime)"]
    for key, m in profile.performance.items():
        lines.append(f"  {key:<26} {m!s:<30} {m.method}".rstrip())
    lines += ["", "MODELL-INFO"]
    for key, m in profile.model_info.items():
        lines.append(f"  {key:<22} {m!s:<30} {m.method}")
    lines += ["", "AUFGABEN"]
    for t in profile.tasks:
        score = f"{t.score:.2f}" if t.score is not None else "–"
        lines.append(
            f"  {t.task_id:<17} {t.status.value:<11} {score:>5}  {t.duration_s:7.1f} s  "
            f"{t.detail[:70]}"
        )
    if profile.notes:
        lines += ["", "HINWEISE", *(f"  - {n}" for n in profile.notes)]
    return "\n".join(lines)


async def attach_profiles(
    registry: ModelRegistry,
    root: Path | None = None,
    *,
    hardware: HardwareProfile | None = None,
) -> ProfileStore:
    """Verbindet den Registry mit gespeicherten Profilen für *diese* Hardware und Suite.

    Danach liefern ``registry.effective()``/``list_effective()`` gemessene Werte; Modelle
    ohne Profil bleiben ausdrücklich ``UNMEASURED``.
    """
    hw = hardware or await detect_hardware()
    store = ProfileStore(
        root, hardware_fingerprint=hw.fingerprint, benchmark_version=suite_version()
    )
    registry.set_measurements(store)
    return store
