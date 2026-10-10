"""Trainingsplan (Dry-Run): prüft Konfiguration, Datensatz und Hardware, bevor trainiert wird.

Ein Plan erzeugt kein Modell. Er hält fest, *was* trainiert würde (Basismodell + Revision,
Datensatz@Version mit Hashes, Hyperparameter, Seed) und ob das auf der Hardware machbar ist.
Der Plan-Hash ist Teil der späteren Lineage eines NOVA-Checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from training.dataset import DatasetError, DatasetStore

_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


@dataclass
class TrainingConfig:
    name: str
    base_model: str  # z. B. Hugging-Face-ID
    base_revision: str  # Commit-Hash/Tag – Pflicht für Reproduzierbarkeit
    base_license: str
    dataset: str  # name@version
    method: str = "qlora"  # qlora | lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 2.0
    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 16
    seed: int = 42
    params_b: float | None = None  # Modellgröße in Mrd., sonst aus dem Namen geschätzt

    @classmethod
    def from_toml(cls, path: Path) -> TrainingConfig:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    def validate(self) -> list[str]:
        problems = []
        if self.method not in ("qlora", "lora"):
            problems.append("method must be qlora or lora")
        if not self.base_revision or self.base_revision in ("main", "latest"):
            problems.append("base_revision must pin an exact revision (commit hash or tag)")
        if not self.base_license:
            problems.append("base_license is required")
        if "@" not in self.dataset:
            problems.append("dataset must be name@version")
        if not 1 <= self.lora_r <= 256 or not 0 <= self.lora_dropout < 1:
            problems.append("invalid LoRA parameters")
        if not 1e-6 <= self.learning_rate <= 1e-2:
            problems.append("learning_rate out of range")
        if not 128 <= self.max_seq_len <= 32768:
            problems.append("max_seq_len out of range")
        return problems

    @property
    def model_size_b(self) -> float | None:
        if self.params_b:
            return self.params_b
        match = _PARAMS.search(self.base_model)
        return float(match.group(1)) if match else None

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TrainingPlan:
    config: dict[str, Any]
    config_hash: str
    dataset_manifest: dict[str, Any] | None
    hardware: dict[str, Any] | None
    estimated_vram_gb: float | None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def estimate_vram_gb(size_b: float, method: str, seq_len: int) -> float:
    """Grobe Schätzung (Gewichte + LoRA-Zustände + Aktivierungen); bewusst konservativ."""
    weights = size_b * (0.55 if method == "qlora" else 2.0)
    activations = 0.6 * (seq_len / 1024) * max(size_b / 4, 0.5)
    return round(weights + activations + 1.2, 1)


def make_plan(
    config: TrainingConfig, datasets: DatasetStore, hardware: dict[str, Any] | None
) -> TrainingPlan:
    problems = config.validate()
    warnings: list[str] = []
    manifest = None
    if "@" in config.dataset:
        name, version = config.dataset.split("@", 1)
        try:
            manifest = datasets.manifest(name, version)
            problems += [f"dataset {config.dataset}: {p}" for p in datasets.verify(name, version)]
            if manifest["counts"].get("test", 0) == 0:
                problems.append("dataset has no test split – independent evaluation impossible")
            if manifest["counts"].get("train", 0) < 200:
                warnings.append(
                    f"only {manifest['counts'].get('train', 0)} training examples – "
                    "aim for several hundred checked examples"
                )
        except DatasetError as exc:
            problems.append(str(exc))
    size = config.model_size_b
    estimate = estimate_vram_gb(size, config.method, config.max_seq_len) if size else None
    if size is None:
        warnings.append("model size unknown – set params_b for a VRAM estimate")
    if hardware is None:
        warnings.append(
            "no hardware probe – run `python -m training.probe` in the training environment"
        )
    else:
        problems += [f"hardware: {p}" for p in hardware.get("problems", [])]
        vram = hardware.get("vram_gb")
        if estimate and vram and estimate > vram:
            problems.append(f"estimated {estimate} GB VRAM needed, GPU has {vram} GB")
        if config.method == "qlora" and not (hardware.get("packages") or {}).get("bitsandbytes"):
            problems.append("qlora needs bitsandbytes")
        if hardware.get("compute_dtype") == "float32":
            warnings.append("fp32 compute (Pascal GPU): expect slow training")
    return TrainingPlan(
        asdict(config), config.config_hash(), manifest, hardware, estimate, problems, warnings
    )
