"""Training im Core: Umgebung einrichten, Jobs starten/abbrechen, Evaluation, Freigabe, Aktivierung.

* Die **Trainingsumgebung** ist getrennt von der App (``<Daten>/train-env``). NOVA richtet sie
  selbst ein: ``uv`` (GitHub-Release mit SHA-256), Python 3.12, PyTorch (CUDA 12.6 für NVIDIA –
  enthält Pascal-Kernels – sonst CPU), transformers/peft/accelerate (+ bitsandbytes auf GPU).
* Ein Job = Datensatz@Version + Basismodell (feste Revision aus der Hugging-Face-API) +
  Hyperparameter. Vorher prüft der Trainingsplan Datensatz, Hardware und VRAM.
* GPU-Ressourcen: Läuft die verwaltete Runtime auf der GPU, wird sie für die Dauer des Trainings
  angehalten (der Chat pausiert) und danach wieder gestartet.
* Nach dem Training läuft automatisch die unabhängige Evaluation. Aktiviert werden kann nur ein
  Checkpoint, dessen Freigabe-Gate bestanden ist – und nur nach ausdrücklicher Bestätigung.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from api.paths import DataLayout
from api.runtime import (
    HF_API,
    Artifact,
    RuntimeError_,
    RuntimeManager,
    fetch_verified,
    no_window,
    recommended_models,
    safe_extract,
)
from training.dataset import DatasetError, DatasetStore
from training.plan import TrainingConfig, make_plan

log = logging.getLogger("nova.training")

UV_RELEASES = "https://api.github.com/repos/astral-sh/uv/releases/latest"
LLAMA_SRC = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip"
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
TRAIN_PACKAGES = [
    "transformers>=4.57",
    "peft>=0.13",
    "accelerate>=1.0",
    "huggingface_hub>=0.25",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "numpy",
    # llama.cpp-Konverter (convert_lora_to_gguf.py + gguf-py)
    "pyyaml",
    "tqdm",
    "requests",
]
MIN_TRAIN_VRAM_GB = 6.0
SCRIPTS = ("runner.py", "evaluate.py", "export.py", "probe.py")
_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


def trainer_source() -> Path:
    """Ordner mit den Trainer-Skripten: im Paket ``trainer/``, im Quellbaum ``training/``."""
    root = Path(__file__).resolve().parents[1]
    for candidate in (root / "trainer", root / "training"):
        if (candidate / "runner.py").is_file():
            return candidate
    raise RuntimeError_(
        "trainer_missing", "Trainer scripts are missing from this installation", 500
    )


def converter_files(name: PurePosixPath) -> bool:
    """Aus dem llama.cpp-Quellarchiv nur Konverter, ``conversion/`` und ``gguf-py/`` entpacken."""
    parts = name.parts[1:]  # erste Ebene: llama.cpp-<tag>/
    if len(parts) == 1:
        return parts[0].startswith("convert_") and parts[0].endswith(".py")
    return len(parts) > 1 and parts[0] in ("conversion", "gguf-py")


def uv_asset_pattern(platform: str = sys.platform) -> re.Pattern[str]:
    if platform == "win32":
        return re.compile(r"^uv-x86_64-pc-windows-msvc\.zip$")
    if platform == "darwin":
        return re.compile(r"^uv-(aarch64|x86_64)-apple-darwin\.tar\.gz$")
    return re.compile(r"^uv-x86_64-unknown-linux-gnu\.tar\.gz$")


def safe_untar(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise RuntimeError_("unsafe_archive", f"unsafe path in archive: {member.name}")
        tf.extractall(root, filter="data")


class TrainingManager:
    def __init__(
        self,
        layout: DataLayout,
        runtime: RuntimeManager,
        recommended_path: Path,
        *,
        on_runtime_change: Callable[[], None] = lambda: None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        platform: str = sys.platform,
    ) -> None:
        self.layout = layout
        self.runtime = runtime
        self.recommended_path = recommended_path
        self.on_runtime_change = on_runtime_change
        self.client_factory = client_factory or runtime.client_factory
        self.platform = platform
        self.datasets = DatasetStore(layout.root / "datasets")
        self.env_dir = layout.root / "train-env"
        self.tools = layout.root / "tools"
        self.trainer_dir = layout.root / "trainer"
        self.jobs_dir = layout.root / "training-jobs"
        self.setup_progress: dict[str, Any] = {"status": "idle"}
        self._setup_task: asyncio.Task[None] | None = None
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    # ------------------------------------------------------------------ Umgebung

    def env_python(self) -> Path:
        override = os.environ.get("NOVA_TRAIN_PYTHON")
        if override:
            return Path(override)
        if self.platform == "win32":
            return self.env_dir / "Scripts" / "python.exe"
        return self.env_dir / "bin" / "python"

    def probe_result(self) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads(
                (self.env_dir / "probe.json").read_text(encoding="utf-8")
            )
            return data
        except (OSError, ValueError):
            return None

    def env_status(self) -> dict[str, Any]:
        python = self.env_python()
        return {
            "ready": python.is_file() and (self.trainer_dir / "runner.py").is_file(),
            "python": str(python),
            "probe": self.probe_result(),
            "setup": self.setup_progress,
        }

    def _log(self, line: str) -> None:
        path = self.layout.logs / "training-setup.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line + "\n")

    def _run(self, args: list[str], step: str) -> None:
        self.setup_progress.update({"status": "installing", "step": step})
        self._log(f"$ {' '.join(args)}")
        result = subprocess.run(args, capture_output=True, text=True, check=False, **no_window())
        self._log(result.stdout[-4000:] + result.stderr[-4000:])
        if result.returncode != 0:
            raise RuntimeError_(
                "setup_failed",
                f"{step} failed (exit {result.returncode}) – see logs/training-setup.log",
                500,
            )

    def install_scripts(self) -> None:
        self.trainer_dir.mkdir(parents=True, exist_ok=True)
        source = trainer_source()
        for name in SCRIPTS:
            shutil.copy2(source / name, self.trainer_dir / name)

    def hardware(self) -> dict[str, Any]:
        """Kann dieser PC trainieren? Ehrliche Antwort vor mehreren GB Download."""
        gpu = self.runtime.gpu()
        info: dict[str, Any] = {"gpu": None, "trainable": False}
        if gpu is None:
            info["reason"] = (
                "This PC has no NVIDIA GPU. Training a language model needs an NVIDIA GPU with "
                f"at least {MIN_TRAIN_VRAM_GB:g} GB VRAM (for example a GTX 1080 Ti) – on the CPU "
                "it would take days and run out of memory. Chat and research work normally here; "
                "train on a PC with a suitable GPU."
            )
            return info
        info["gpu"] = {"name": gpu.name, "vram_gb": gpu.vram_gb}
        if gpu.vram_gb < MIN_TRAIN_VRAM_GB:
            info["reason"] = (
                f"{gpu.name} has {gpu.vram_gb:g} GB VRAM; training needs at least "
                f"{MIN_TRAIN_VRAM_GB:g} GB."
            )
            return info
        info["trainable"] = True
        return info

    async def setup_env(self) -> dict[str, Any]:
        if self._setup_task is not None and not self._setup_task.done():
            raise RuntimeError_("already_running", "Training setup is already running", 409)
        hardware = self.hardware()
        if not hardware["trainable"] and not os.environ.get("NOVA_TRAIN_ALLOW_CPU"):
            raise RuntimeError_("no_training_gpu", hardware["reason"], 409)
        self.setup_progress = {"status": "starting", "error": None}
        self._setup_task = asyncio.create_task(self._setup())
        return self.setup_progress

    async def _setup(self) -> None:
        try:
            gpu = self.runtime.gpu()
            async with self.client_factory() as client:
                self.setup_progress["step"] = "uv"
                release = (await client.get(UV_RELEASES)).json()
                pattern = uv_asset_pattern(self.platform)
                asset = next(
                    (a for a in release.get("assets", []) if pattern.match(str(a.get("name", "")))),
                    None,
                )
                digest = str((asset or {}).get("digest") or "")
                if asset is None or not digest.startswith("sha256:"):
                    raise RuntimeError_(
                        "setup_failed", "uv release asset with SHA-256 not found", 502
                    )
                archive = await fetch_verified(
                    client,
                    Artifact(
                        asset["name"],
                        asset["browser_download_url"],
                        digest[7:],
                        int(asset.get("size") or 0),
                    ),
                    self.tools / ".downloads" / asset["name"],
                )
            uv_dir = self.tools / f"uv-{release.get('tag_name', 'latest')}"
            if archive.suffix == ".zip":
                await asyncio.to_thread(safe_extract, archive, uv_dir)
            else:
                await asyncio.to_thread(safe_untar, archive, uv_dir)
            uv = next(
                iter(sorted(uv_dir.rglob("uv.exe" if self.platform == "win32" else "uv"))), None
            )
            if uv is None:
                raise RuntimeError_("setup_failed", "uv executable not found in release", 502)
            if self.platform != "win32":
                uv.chmod(0o755)
            await asyncio.to_thread(
                self._run, [str(uv), "python", "install", "3.12"], "Python 3.12"
            )
            await asyncio.to_thread(
                self._run,
                [str(uv), "venv", str(self.env_dir), "--python", "3.12", "--allow-existing"],
                "virtual environment",
            )
            py = str(self.env_python())
            index = TORCH_INDEX["cuda" if gpu else "cpu"]
            await asyncio.to_thread(
                self._run,
                [str(uv), "pip", "install", "--python", py, "torch", "--index-url", index],
                "PyTorch (CUDA 12.6)" if gpu else "PyTorch (CPU)",
            )
            packages = [*TRAIN_PACKAGES, *(["bitsandbytes"] if gpu else [])]
            await asyncio.to_thread(
                self._run,
                [str(uv), "pip", "install", "--python", py, *packages],
                "training libraries",
            )
            self.install_scripts()
            await asyncio.to_thread(self.run_probe)
            self.setup_progress = {"status": "completed", "error": None}
        except RuntimeError_ as exc:
            self.setup_progress.update({"status": "failed", "error": exc.message})
        except Exception as exc:
            log.exception("training setup failed")
            self.setup_progress.update(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )

    def run_probe(self) -> dict[str, Any]:
        self.install_scripts()
        result = subprocess.run(
            [str(self.env_python()), str(self.trainer_dir / "probe.py")],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(self.trainer_dir),
            **no_window(),
        )
        try:
            data: dict[str, Any] = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError_(
                "probe_failed", f"hardware probe failed: {result.stderr[-500:]}", 500
            ) from exc
        self.env_dir.mkdir(parents=True, exist_ok=True)
        (self.env_dir / "probe.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    # ------------------------------------------------------------------ Jobs

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.match(job_id):
            raise RuntimeError_("not_found", f"Training job {job_id} not found", 404)
        path = self.jobs_dir / job_id
        if not (path / "job.json").is_file():
            raise RuntimeError_("not_found", f"Training job {job_id} not found", 404)
        return path

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return data
        except (OSError, ValueError):
            return None

    def job(self, job_id: str) -> dict[str, Any]:
        path = self._job_dir(job_id)
        job = self._read(path / "job.json") or {}
        report = self._read(path / "eval" / "report.json")
        return {
            "id": job_id,
            "name": job.get("name"),
            "base_model": job.get("base_model"),
            "base_choice": job.get("base_choice"),
            "dataset": job.get("dataset"),
            "method": job.get("method"),
            "created_at": job.get("created_at"),
            "training": self._read(path / "status.json"),
            "evaluation": self._read(path / "eval" / "status.json"),
            "report": {k: report[k] for k in ("base", "tuned", "gate", "test_examples")}
            if report
            else None,
            "approval": self._read(path / "approval.json"),
            "running": job_id in self._processes and self._processes[job_id].poll() is None,
            "runtime_paused": bool(job.get("runtime_paused")),
        }

    def jobs(self) -> list[dict[str, Any]]:
        if not self.jobs_dir.is_dir():
            return []
        return [
            self.job(p.name)
            for p in sorted(self.jobs_dir.iterdir(), reverse=True)
            if (p / "job.json").is_file()
        ]

    def choices(self) -> list[dict[str, Any]]:
        return [m for m in recommended_models(self.recommended_path) if m.get("training_base")]

    async def start_job(
        self,
        dataset: str,
        choice: str,
        *,
        epochs: float = 2.0,
        max_steps: int | None = None,
        local_model_dir: str | None = None,
    ) -> dict[str, Any]:
        if any(p.poll() is None for p in self._processes.values()):
            raise RuntimeError_("busy", "Another training or evaluation is running", 409)
        if not self.env_status()["ready"]:
            raise RuntimeError_("env_not_ready", "Set up the training environment first", 409)
        probe = self.probe_result()
        if not local_model_dir and not (probe or {}).get("cuda_available"):
            raise RuntimeError_(
                "no_training_gpu",
                self.hardware().get("reason")
                or "PyTorch cannot use the GPU (see the hardware check) – training on the CPU "
                "is not offered for real models.",
                409,
            )
        spec = next((m for m in self.choices() if m["id"] == choice), None)
        if spec is None:
            raise RuntimeError_("invalid_request", f"Unknown base model choice {choice!r}")
        base = spec["training_base"]
        if local_model_dir:
            revision = "local"
        else:
            async with self.client_factory() as client:
                try:
                    info = (await client.get(HF_API.format(repo=base["repo"]))).json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise RuntimeError_(
                        "hf_unreachable", f"Hugging Face not reachable: {exc}", 502
                    ) from exc
            revision = str(info.get("sha") or "")
            if not re.match(r"^[0-9a-f]{40}$", revision):
                raise RuntimeError_(
                    "hf_unreachable", f"{base['repo']}: no revision in API answer", 502
                )
        gpu = bool(probe and probe.get("cuda_available"))
        has_bnb = bool(probe and (probe.get("packages") or {}).get("bitsandbytes"))
        method = "qlora" if gpu and has_bnb else "lora"
        config = TrainingConfig(
            name=f"nova-{choice}",
            base_model=base["repo"],
            base_revision=revision,
            base_license=str(spec.get("base_license", "")),
            dataset=dataset,
            method=method,
            epochs=epochs,
            params_b=float(base.get("params_b") or 0) or None,
            max_seq_len=1024 if gpu else 512,
            batch_size=1,
            grad_accum=16 if gpu else 4,
        )
        plan = make_plan(config, self.datasets, probe if gpu else None)
        problems = [p for p in plan.problems if not (local_model_dir and "revision" in p)]
        if problems:
            raise RuntimeError_("plan_failed", "Training plan not possible: " + "; ".join(problems))
        name, version = dataset.split("@", 1)
        manifest = self.datasets.manifest(name, version)
        job_id = time.strftime("%Y%m%d-%H%M%S-") + choice
        out = self.jobs_dir / job_id
        out.mkdir(parents=True)
        job = {
            "job_id": job_id,
            "name": config.name,
            "base_choice": choice,
            "base_model": base["repo"],
            "base_revision": revision,
            "method": method,
            "dataset": dataset,
            "dataset_dir": str(self.datasets.root / name / version),
            "dataset_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "output_dir": str(out),
            "compute_dtype": (probe or {}).get("compute_dtype") or "float32",
            "config_hash": plan.config_hash,
            "plan_warnings": plan.warnings,
            "hyper": {
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "epochs": epochs,
                "max_seq_len": config.max_seq_len,
                "batch_size": config.batch_size,
                "grad_accum": config.grad_accum,
                "seed": config.seed,
                "save_steps": 50,
                **({"max_steps": max_steps} if max_steps else {}),
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if local_model_dir:
            job["local_model_dir"] = local_model_dir
        if gpu and self.runtime.process.running:  # GPU teilen geht auf 11 GB nicht: Chat pausieren
            await asyncio.to_thread(self.runtime.stop)
            job["runtime_paused"] = True
        (out / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        self._job_tasks[job_id] = asyncio.create_task(self._pipeline(job_id))
        return self.job(job_id)

    def _spawn(self, job_id: str, script: str, *extra: str) -> subprocess.Popen[bytes]:
        out = self.jobs_dir / job_id
        log_file = (out / f"{script.removesuffix('.py')}.out.log").open("ab")
        kwargs: dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "cwd": str(self.trainer_dir),
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            [str(self.env_python()), str(self.trainer_dir / script), str(out / "job.json"), *extra],
            **kwargs,
        )
        self._processes[job_id] = process
        return process

    async def _wait(self, process: subprocess.Popen[bytes]) -> int:
        return int(await asyncio.to_thread(process.wait))

    async def _pipeline(self, job_id: str) -> None:
        out = self.jobs_dir / job_id
        try:
            self.install_scripts()
            code = await self._wait(self._spawn(job_id, "runner.py"))
            if code == 0 and (out / "adapter").is_dir():
                await self._wait(self._spawn(job_id, "evaluate.py"))
        finally:
            self._processes.pop(job_id, None)
            job = self._read(out / "job.json") or {}
            if job.get("runtime_paused"):
                with contextlib.suppress(Exception):
                    if await asyncio.to_thread(self.runtime.start_runtime):
                        self.on_runtime_change()
                job["runtime_paused"] = False
                (out / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")

    def cancel(self, job_id: str) -> dict[str, Any]:
        out = self._job_dir(job_id)
        (out / "STOP").write_text("stop requested", encoding="utf-8")
        return self.job(job_id)

    async def wait(self, job_id: str, timeout_s: float | None = None) -> dict[str, Any]:
        task = self._job_tasks.get(job_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        return self.job(job_id)

    # ------------------------------------------------------------------ Freigabe

    async def ensure_llama_source(self) -> Path:
        """llama.cpp-Quellen (Konverter + gguf-py) zum Release-Tag der installierten Runtime."""
        tag = self.runtime.state().get("runtime_tag")
        if not tag:
            raise RuntimeError_(
                "not_installed", "Set up the model runtime first (one-click setup)", 409
            )
        target = self.tools / f"llama.cpp-src-{tag}"
        found = sorted(target.glob("*/convert_lora_to_gguf.py"))
        if found:
            return found[0].parent
        archive = self.tools / ".downloads" / f"llama.cpp-{tag}-src.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        async with (
            self.client_factory() as client,
            client.stream("GET", LLAMA_SRC.format(tag=tag)) as r,
        ):
            if r.status_code != 200 or not str(r.url).startswith("https://"):
                raise RuntimeError_(
                    "download_failed", f"llama.cpp sources: HTTP {r.status_code}", 502
                )
            digest = hashlib.sha256()
            with archive.open("wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)
                    digest.update(chunk)
        # GitHub-Quellarchive haben keine veröffentlichte Prüfsumme: Tag + HTTPS, Hash protokolliert
        self._log(f"llama.cpp source {tag}: sha256 {digest.hexdigest()}")
        await asyncio.to_thread(safe_extract, archive, target, converter_files)
        found = sorted(target.glob("*/convert_lora_to_gguf.py"))
        if not found:
            raise RuntimeError_("download_failed", "converter not found in llama.cpp sources", 502)
        return found[0].parent

    async def approve(self, job_id: str, *, llama_src: Path | None = None) -> dict[str, Any]:
        out = self._job_dir(job_id)
        info = self.job(job_id)
        report = info.get("report")
        if not report or not report["gate"]["eligible"]:
            raise RuntimeError_(
                "not_eligible",
                "This checkpoint did not pass the evaluation gate and cannot be activated",
                409,
            )
        installed = (self.runtime.state().get("model") or {}).get("id")
        if installed != info["base_choice"]:
            raise RuntimeError_(
                "base_mismatch",
                f"The adapter was trained for {info['base_choice']}, but the runtime runs "
                f"{installed}. Install the matching model first.",
                409,
            )
        source = llama_src or await self.ensure_llama_source()
        code = await self._wait(self._spawn(job_id, "export.py", str(source)))
        self._processes.pop(job_id, None)
        adapter = out / "adapter.gguf"
        if code != 0 or not adapter.is_file():
            raise RuntimeError_(
                "export_failed", "Converting the adapter to GGUF failed – see export.log", 500
            )
        sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
        approval = {
            "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "approved_by": "local user (explicit confirmation in NOVA)",
            "gate": report["gate"],
            "adapter_gguf_sha256": sha,
        }
        (out / "approval.json").write_text(json.dumps(approval, indent=2), encoding="utf-8")
        name = f"nova-{info['base_choice']}-{job_id[:15]}"
        self.runtime.set_lora([str(adapter)], name)
        await self._restart_runtime()
        return self.job(job_id)

    async def deactivate(self) -> dict[str, Any]:
        self.runtime.set_lora([])
        await self._restart_runtime()
        return {"active": None}

    async def _restart_runtime(self) -> None:
        await asyncio.to_thread(self.runtime.stop)
        if await asyncio.to_thread(self.runtime.start_runtime):
            self.on_runtime_change()

    def status(self) -> dict[str, Any]:
        state = self.runtime.state().get("model") or {}
        try:
            datasets = self.datasets.list_all()
        except DatasetError:
            datasets = []
        return {
            "hardware": self.hardware(),
            "env": self.env_status(),
            "choices": [
                {"id": c["id"], "label": c["label"], "repo": c["training_base"]["repo"]}
                for c in self.choices()
            ],
            "installed_model": state.get("id"),
            "active_adapter": state.get("active_name"),
            "datasets": datasets,
            "jobs": self.jobs(),
        }

    async def shutdown(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
