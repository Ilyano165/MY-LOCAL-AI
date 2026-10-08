"""Benchmark eines einzelnen Modells: Modellfähigkeit und Hardware-Leistung – getrennt.

**Modellfähigkeit** (``profile.capabilities``): Scores aus den Standardaufgaben
(:mod:`evaluation.benchmark_tasks`), unabhängig bewertet. Hängt vom Modell (inkl.
Quantisierung) ab, nicht von der Geschwindigkeit der Hardware.

**Hardware-Leistung** (``profile.performance``), nur aus echten Läufen:

* ``load_time`` – MEASURED, wenn NOVA den Ladevorgang beobachtet: Server selbst gestartet
  (Prozessstart → ``/health`` bereit) oder Ollama-Lade-API; sonst NOT_MEASURABLE (bei
  llama.cpp lädt der Server das Modell beim Start, bevor NOVA es sieht).
* ``first_token_latency`` – Wanduhr für eine 1-Token-Antwort auf einen neuen Prompt.
* ``tokens_per_second`` – Dekodierrate, Differenzmethode: (N−1) / (t(N Tokens) − t(1 Token))
  bei gleichem Prompt; Median über mehrere Läufe. Funktioniert runtime-unabhängig ohne
  Streaming. Zusätzlich ``tokens_per_second_runtime`` (REPORTED, llama.cpp ``timings``).
* ``prompt_tokens_per_second`` – Prompt-Verarbeitung (langer, neuer Prompt, 1 Token Ausgabe).
* ``peak_vram``/``peak_ram`` – Sampling während aller Läufe (:class:`ResourceSampler`).
* ``memory_footprint`` – nur, wenn belastbar: Ollama-``size`` (REPORTED), Prozess-Sampling
  oder System-Delta ab einem Ausgangswert *vor* dem Laden (selbst gestarteter Server).
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evaluation.benchmark_results import (
    Measurement,
    MeasurementStatus,
    ModelProfile,
    TaskResult,
    TaskStatus,
    configured_snapshot,
)
from evaluation.benchmark_tasks import (
    BenchmarkTask,
    TaskContext,
    params,
    standard_tasks,
    suite_version,
)
from evaluation.gguf import GGUFError, read_metadata
from evaluation.hardware import (
    HardwareProfile,
    PeakUsage,
    ResourceSampler,
    find_listening_pid,
)
from models.base import ChatRequest, Message, ModelError, ModelProvider
from models.capabilities import ModelMetadata

CAPABILITY_GROUPS: dict[str, tuple[str, ...]] = {
    "general": ("chat", "short_analysis"),
    "reasoning": ("short_analysis", "complex_analysis"),
    "coding": ("coding", "debugging"),
    "tool_calling": ("tool_calling",),
    "agentic": ("agent_multistep",),
    "long_context": ("long_context",),
    "vision": ("vision",),
}


@dataclass
class BenchmarkOptions:
    tasks: Sequence[str] | None = None
    """Teilmenge der Aufgaben-IDs (``None`` = alle)."""
    timeout_s: float = 300.0
    execute_code: bool = True
    capabilities: bool = True
    performance: bool = True
    throughput_runs: int = 3
    throughput_tokens: int = 128
    measure_load: bool = False
    """Ollama: Modell entladen und neu laden, um die Ladezeit zu messen (unterbricht Nutzung)."""
    server_pid: int | None = None
    """PID der Runtime für Prozess-Sampling; sonst automatische Erkennung über den Port."""
    sample_interval_s: float = 0.2
    long_context_max_tokens: int = 16_384

    def __post_init__(self) -> None:
        if self.throughput_runs < 1 or self.throughput_tokens < 8:
            raise ValueError("throughput_runs ≥ 1 und throughput_tokens ≥ 8 erforderlich")


class ServerLauncher:
    """Startet eine Runtime selbst (z. B. ``llama-server -m …``), um die Ladezeit zu messen.

    Kein Shell-Interpreter: der Befehl wird mit :func:`shlex.split` zerlegt.
    """

    def __init__(self, command: str | Sequence[str], *, ready_timeout_s: float = 600.0) -> None:
        self.argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.argv:
            raise ValueError("Leerer Server-Befehl")
        self.ready_timeout_s = ready_timeout_s
        self.process: asyncio.subprocess.Process | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    async def start(self, provider: ModelProvider) -> float:
        """Startet den Prozess und wartet, bis die Runtime bereit ist; liefert Sekunden."""
        started = time.perf_counter()
        self.process = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        while time.perf_counter() - started < self.ready_timeout_s:
            if self.process.returncode is not None:
                raise ModelError(f"Server beendet mit Code {self.process.returncode}")
            health = await provider.health()
            if health.ready:
                return time.perf_counter() - started
            await asyncio.sleep(0.1)
        raise ModelError(f"Server nach {self.ready_timeout_s:.0f} s nicht bereit")

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), 15)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


def _gb(mb: float | None) -> float | None:
    return round(mb / 1024, 3) if mb is not None else None


class ModelBenchmark:
    def __init__(
        self,
        provider: ModelProvider,
        model: ModelMetadata,
        hardware: HardwareProfile,
        *,
        workdir: Path,
        options: BenchmarkOptions | None = None,
        tasks: Sequence[BenchmarkTask] | None = None,
        launcher: ServerLauncher | None = None,
        sampler: ResourceSampler | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.hardware = hardware
        self.workdir = workdir
        self.options = options or BenchmarkOptions()
        all_tasks = list(tasks) if tasks is not None else standard_tasks()
        self.version = suite_version(all_tasks)
        if self.options.tasks is not None:
            wanted = set(self.options.tasks)
            unknown = wanted - {t.id for t in all_tasks}
            if unknown:
                raise ValueError(f"Unbekannte Aufgaben: {sorted(unknown)}")
            all_tasks = [t for t in all_tasks if t.id in wanted]
        self.tasks = all_tasks
        self.launcher = launcher
        self._sampler = sampler
        self.notes: list[str] = []

    # ------------------------------------------------------------------ Ablauf

    async def run(self) -> ModelProfile:
        performance: dict[str, Measurement] = {}
        sampler = self._sampler
        info: dict[str, Any] = {}
        tasks: list[TaskResult] = []
        try:
            if self.launcher is not None:
                # Ausgangswert vor dem Laden: Delta = Speicherbedarf des Modells
                sampler = sampler or ResourceSampler(interval_s=self.options.sample_interval_s)
                async with _maybe(sampler):
                    performance["load_time"] = await self._launch()
                    info = await self._runtime_info()
                    tasks, perf = await self._measure(info)
                    performance |= perf
            else:
                info = await self._runtime_info()
                performance["load_time"] = await self._load_time(info)
                if performance["load_time"].is_fact:
                    info = await self._runtime_info()  # nach dem Laden: /api/ps-Angaben
                sampler = sampler or self._default_sampler(info)
                async with _maybe(sampler):
                    tasks, perf = await self._measure(info)
                performance |= perf
        finally:
            if self.launcher is not None:
                await self.launcher.stop()
        usage = sampler.result if sampler is not None else None
        performance |= self._memory(usage, info)
        return ModelProfile(
            model=self.model.name,
            benchmark_version=self.version,
            runtime=info,
            hardware=self.hardware.to_dict(),
            hardware_fingerprint=self.hardware.fingerprint,
            model_info=self._model_info(info),
            capabilities=self._capabilities(tasks),
            performance=performance,
            tasks=tasks,
            configured=configured_snapshot(self.model),
            notes=self.notes,
        )

    async def _measure(
        self, info: dict[str, Any]
    ) -> tuple[list[TaskResult], dict[str, Measurement]]:
        results: list[TaskResult] = []
        if self.options.capabilities:
            ctx = TaskContext(
                provider=self.provider,
                model=self.model,
                workdir=self.workdir,
                timeout_s=self.options.timeout_s,
                context_length=info.get("n_ctx") if isinstance(info.get("n_ctx"), int) else None,
                vision_supported=(info.get("modalities") or {}).get("vision"),
                execute_code=self.options.execute_code,
                long_context_max_tokens=self.options.long_context_max_tokens,
            )
            for task in self.tasks:
                results.append(await task.run(ctx))
        perf = await self._throughput() if self.options.performance else {}
        return results, perf

    # ------------------------------------------------------------------ Runtime

    async def _runtime_info(self) -> dict[str, Any]:
        try:
            info = await self.provider.runtime_info(self.model)
        except ModelError as exc:
            self.notes.append(f"Runtime-Info nicht abrufbar: {exc}")
            info = {}
        info.setdefault("provider", self.provider.name)
        info["served_name"] = self.model.runtime_name
        return info

    def _default_sampler(self, info: dict[str, Any]) -> ResourceSampler:
        pid = self.options.server_pid
        if pid is None and info.get("runtime") != "ollama":
            # Ollama: Modell läuft in einem Kindprozess („runner“) – Port-PID wäre falsch
            port = urlsplit(getattr(self.provider, "base_url", "")).port
            if port is not None:
                pid = find_listening_pid(port)
        if pid is None:
            self.notes.append("Server-PID unbekannt – Speicher systemweit gemessen")
        return ResourceSampler(pid=pid, interval_s=self.options.sample_interval_s)

    async def _launch(self) -> Measurement:
        assert self.launcher is not None
        try:
            seconds = await self.launcher.start(self.provider)
        except ModelError as exc:
            raise ModelError(f"Server-Start fehlgeschlagen: {exc}") from exc
        return Measurement.measured(
            round(seconds, 3), "s", "Prozessstart → Runtime bereit (/health), von NOVA gestartet"
        )

    async def _load_time(self, info: dict[str, Any]) -> Measurement:
        if info.get("runtime") == "ollama":
            loaded = bool(info.get("loaded"))
            if loaded and not self.options.measure_load:
                return Measurement.missing(
                    MeasurementStatus.NOT_MEASURABLE,
                    "Modell war bereits geladen; --measure-load entlädt und lädt neu",
                    "s",
                )
            try:
                if loaded:
                    await self.provider.unload(self.model)
                    await asyncio.sleep(1.0)
                started = time.perf_counter()
                reported = await self.provider.load(self.model)
                seconds = time.perf_counter() - started
            except (ModelError, NotImplementedError) as exc:
                return Measurement.missing(
                    MeasurementStatus.FAILED, f"Laden fehlgeschlagen: {exc}", "s"
                )
            note = (
                f"Runtime meldet load_duration {reported['load_duration_s']:.2f} s"
                if "load_duration_s" in reported
                else ""
            )
            return Measurement.measured(
                round(seconds, 3), "s", "Ollama /api/generate (kalt geladen), Wanduhr", note
            )
        return Measurement.missing(
            MeasurementStatus.NOT_MEASURABLE,
            "Runtime lädt das Modell beim Serverstart; mit --server-cmd startet NOVA den Server "
            "selbst und misst die Ladezeit",
            "s",
        )

    # ------------------------------------------------------------------ Durchsatz

    async def _timed(self, prompt: str, max_tokens: int) -> tuple[float, Any]:
        request = ChatRequest(
            (Message.user(prompt),), params=params(max_tokens), timeout_s=self.options.timeout_s
        )
        started = time.perf_counter()
        response = await self.provider.chat(self.model, request)
        return time.perf_counter() - started, response

    async def _throughput(self) -> dict[str, Measurement]:
        out: dict[str, Measurement] = {}
        n = self.options.throughput_tokens
        count_prompt = (
            "Count upward from 1 in English words, separated by commas, and never stop: "
            "one, two, three,"
        )
        try:
            ttft, _ = await self._timed(f"[{uuid.uuid4().hex}] Say OK.", 1)
            out["first_token_latency"] = Measurement.measured(
                round(ttft, 3), "s", "Wanduhr, 1-Token-Antwort auf neuen Prompt (inkl. HTTP)"
            )
        except ModelError as exc:
            out["first_token_latency"] = Measurement.missing(
                MeasurementStatus.FAILED, str(exc), "s"
            )
        rates: list[float] = []
        reported: list[float] = []
        failures: list[str] = []
        for _ in range(self.options.throughput_runs):
            try:
                t_n, long = await self._timed(count_prompt, n)  # zuerst N → Prompt im Cache
                t_1, _ = await self._timed(count_prompt, 1)
            except ModelError as exc:
                failures.append(str(exc))
                continue
            produced = long.usage.completion_tokens
            if produced > 1 and t_n > t_1:
                rates.append((produced - 1) / (t_n - t_1))
            else:
                failures.append(f"nur {produced} Tokens erzeugt")
            if (value := long.runtime_stats.get("predicted_per_second")) is not None:
                reported.append(value)
        if rates:
            out["tokens_per_second"] = Measurement.measured(
                round(statistics.median(rates), 2),
                "tok/s",
                f"Differenzmethode (N−1)/(t_N − t_1), N={n}, Median aus {len(rates)} Läufen",
                f"Einzelwerte: {', '.join(f'{r:.1f}' for r in rates)}",
            )
        else:
            out["tokens_per_second"] = Measurement.missing(
                MeasurementStatus.FAILED, "; ".join(failures) or "keine Messung", "tok/s"
            )
        if reported:
            out["tokens_per_second_runtime"] = Measurement.reported(
                round(statistics.median(reported), 2),
                "tok/s",
                "llama.cpp timings.predicted_per_second",
            )
        out["prompt_tokens_per_second"] = await self._prompt_rate()
        return out

    async def _prompt_rate(self) -> Measurement:
        filler = " ".join(f"Item {i} is stored in box {i * 7 % 101}." for i in range(250))
        prompt = f"[{uuid.uuid4().hex}] {filler}\nReply with OK."
        try:
            seconds, response = await self._timed(prompt, 1)
        except ModelError as exc:
            return Measurement.missing(MeasurementStatus.FAILED, str(exc), "tok/s")
        reported = response.runtime_stats.get("prompt_per_second")
        if reported is not None:
            return Measurement.reported(
                round(reported, 2),
                "tok/s",
                "llama.cpp timings.prompt_per_second",
                f"Wanduhr inkl. HTTP: {response.usage.prompt_tokens / seconds:.1f} tok/s"
                if response.usage.prompt_tokens
                else "",
            )
        if response.usage.prompt_tokens <= 0:
            return Measurement.missing(
                MeasurementStatus.FAILED, "Runtime meldet keine Prompt-Tokens", "tok/s"
            )
        return Measurement.measured(
            round(response.usage.prompt_tokens / seconds, 2),
            "tok/s",
            f"Wanduhr, {response.usage.prompt_tokens} Prompt-Tokens + 1 Ausgabetoken (inkl. HTTP)",
        )

    # ------------------------------------------------------------------ Speicher

    def _memory(self, usage: PeakUsage | None, info: dict[str, Any]) -> dict[str, Measurement]:
        out: dict[str, Measurement] = {}
        if usage is None:
            missing = Measurement.missing(MeasurementStatus.UNMEASURED, "kein Sampling", "GB")
            return {"peak_vram": missing, "peak_ram": missing}
        detail = f"{usage.samples} Samples à {usage.interval_s:g} s über {usage.duration_s:.0f} s"
        if usage.peak_vram_mb is not None:
            out["peak_vram"] = Measurement.measured(
                _gb(usage.peak_vram_mb),
                "GB",
                f"Sampling {usage.vram_method}",
                f"{detail}; Ausgangswert {_gb(usage.baseline_vram_mb)} GB",
            )
        else:
            reason = (
                "Unified Memory – siehe peak_ram"
                if self.hardware.unified_memory
                else "keine NVIDIA-GPU bzw. nvidia-smi nicht verfügbar"
            )
            out["peak_vram"] = Measurement.missing(MeasurementStatus.NOT_MEASURABLE, reason, "GB")
        if usage.peak_ram_mb is not None:
            out["peak_ram"] = Measurement.measured(
                _gb(usage.peak_ram_mb),
                "GB",
                f"Sampling {usage.ram_method}",
                f"{detail}; Ausgangswert {_gb(usage.baseline_ram_mb)} GB",
            )
        else:
            out["peak_ram"] = Measurement.missing(
                MeasurementStatus.FAILED, "RAM nicht lesbar", "GB"
            )
        if isinstance(info.get("size_vram_bytes"), int):
            out["vram_reported"] = Measurement.reported(
                round(info["size_vram_bytes"] / 1024**3, 3), "GB", "Ollama /api/ps size_vram"
            )
        out["memory_footprint"] = self._footprint(usage, info)
        return out

    def _footprint(self, usage: PeakUsage, info: dict[str, Any]) -> Measurement:
        if (
            info.get("runtime") == "ollama"
            and isinstance(info.get("model_size_bytes"), int)
            and info.get("loaded")
        ):
            return Measurement.reported(
                round(info["model_size_bytes"] / 1024**3, 3),
                "GB",
                "Ollama /api/ps size (geladenes Modell inkl. KV-Cache)",
            )
        vram, ram = usage.peak_vram_mb, usage.peak_ram_mb
        if usage.per_process and ram is not None:
            total = (vram or 0.0) + ram
            return Measurement.measured(
                _gb(total), "GB", "Prozess-Spitze VRAM + RSS (konservativ, mmap-Seiten zählen mit)"
            )
        if self.launcher is not None and ram is not None and usage.baseline_ram_mb is not None:
            delta = (ram - usage.baseline_ram_mb) + (
                (vram - (usage.baseline_vram_mb or 0.0)) if vram is not None else 0.0
            )
            if delta > 0:
                return Measurement.measured(
                    _gb(delta),
                    "GB",
                    "System-Delta Spitze − Ausgangswert vor Serverstart (inkl. Fremdprozesse)",
                )
        return Measurement.missing(
            MeasurementStatus.NOT_MEASURABLE,
            "nur systemweite Werte ohne Ausgangswert vor dem Laden – Modellanteil nicht trennbar",
            "GB",
        )

    # ------------------------------------------------------------------ Modell-Info

    def _model_info(self, info: dict[str, Any]) -> dict[str, Measurement]:
        out: dict[str, Measurement] = {}
        runtime = info.get("runtime", "runtime")
        if isinstance(info.get("n_ctx"), int):
            source = "/api/ps context_length" if runtime == "ollama" else "/props n_ctx"
            out["context_length"] = Measurement.reported(info["n_ctx"], "Tokens", source)
        else:
            out["context_length"] = Measurement.configured(self.model.context_length, "Tokens")
        if isinstance(info.get("n_ctx_train"), int):
            out["context_length_train"] = Measurement.reported(
                info["n_ctx_train"], "Tokens", "/v1/models meta.n_ctx_train"
            )
        if isinstance(info.get("n_params"), int):
            out["parameter_count"] = Measurement.reported(info["n_params"], "", "meta.n_params")
        else:
            out["parameter_count"] = Measurement.configured(self.model.parameter_count)
        path = self._model_file(info)
        gguf: dict[str, Any] = {}
        if path is not None:
            try:
                gguf = read_metadata(path)
            except (GGUFError, OSError) as exc:
                self.notes.append(f"GGUF-Header nicht lesbar ({path}): {exc}")
        if isinstance(info.get("model_size_bytes"), int) and runtime != "ollama":
            out["model_size"] = Measurement.reported(
                round(info["model_size_bytes"] / 1024**3, 3), "GB", "meta.size (Gewichte)"
            )
        elif path is not None:
            out["model_size"] = Measurement.detected(
                round(path.stat().st_size / 1024**3, 3), "GB", f"Dateigröße {path.name}"
            )
        else:
            out["model_size"] = Measurement.missing(
                MeasurementStatus.UNMEASURED, "Modelldatei nicht zugänglich", "GB"
            )
        if "quantization" in gguf:
            out["quantization"] = Measurement.detected(
                gguf["quantization"],
                "",
                "GGUF general.file_type",
                ""
                if gguf["quantization"].upper() == self.model.quantization.upper()
                else f"Konfiguration sagt {self.model.quantization}",
            )
        else:
            out["quantization"] = Measurement.configured(self.model.quantization)
        if isinstance(gguf.get("general.architecture"), str):
            out["architecture"] = Measurement.detected(
                gguf["general.architecture"], "", "GGUF general.architecture"
            )
        if "context_length_train" not in out and isinstance(gguf.get("context_length_train"), int):
            out["context_length_train"] = Measurement.detected(
                gguf["context_length_train"], "Tokens", "GGUF <arch>.context_length"
            )
        return out

    def _model_file(self, info: dict[str, Any]) -> Path | None:
        for candidate in (self.model.local_path, info.get("model_path")):
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return None

    # ------------------------------------------------------------------ Fähigkeiten

    def _capabilities(self, tasks: Sequence[TaskResult]) -> dict[str, Measurement]:
        by_id = {t.task_id: t for t in tasks}
        out: dict[str, Measurement] = {}
        for group, ids in CAPABILITY_GROUPS.items():
            results = [by_id[i] for i in ids if i in by_id]
            if not results:
                continue
            done = [r for r in results if r.status is TaskStatus.COMPLETED and r.score is not None]
            if done:
                score = sum(r.score or 0.0 for r in done) / len(done)
                checks = sum(len(r.checks) for r in done)
                note = ", ".join(f"{r.task_id}={r.score:.2f}" for r in done)
                if len(done) < len(results):
                    note += "; nicht bewertet: " + ", ".join(
                        f"{r.task_id} ({r.status.value})" for r in results if r not in done
                    )
                out[group] = Measurement.measured(
                    round(score, 4),
                    "score",
                    f"NOVA-Benchmark {self.version}: {len(done)} Aufgabe(n), {checks} Kriterien",
                    note,
                )
            elif all(r.status is TaskStatus.UNSUPPORTED for r in results):
                out[group] = Measurement.missing(
                    MeasurementStatus.NOT_SUPPORTED, "; ".join(r.detail for r in results)[:300]
                )
            elif all(r.status in (TaskStatus.SKIPPED, TaskStatus.UNSUPPORTED) for r in results):
                out[group] = Measurement.missing(
                    MeasurementStatus.NOT_MEASURABLE, "; ".join(r.detail for r in results)[:300]
                )
            else:
                out[group] = Measurement.missing(
                    MeasurementStatus.FAILED, "; ".join(r.detail for r in results)[:300]
                )
        return out


@contextlib.asynccontextmanager
async def _maybe(sampler: ResourceSampler | None) -> Any:
    if sampler is None:
        yield None
        return
    async with sampler:
        yield sampler
