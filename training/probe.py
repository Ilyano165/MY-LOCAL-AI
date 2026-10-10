"""Hardware-/Software-Probe für das Training (läuft in der Trainingsumgebung des Nutzers).

    python -m training.probe            # JSON-Ergebnis + Empfehlung

Prüft PyTorch/CUDA, GPU, Compute Capability, VRAM, bf16, bitsandbytes/peft/trl und leitet
konkrete Einstellungen ab. Für Pascal-GPUs (Compute Capability 6.x, z. B. GTX 1080 Ti) gilt:
kein bf16, fp16 stark gedrosselt → fp32-Compute; PyTorch ≤ 2.14 mit CUDA 12.6 (letzte Builds mit
Pascal-Kernels). Nichts wird installiert oder heruntergeladen.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProbeResult:
    python: str
    torch: str | None = None
    torch_cuda: str | None = None
    cuda_available: bool = False
    gpu: str | None = None
    compute_capability: str | None = None
    vram_gb: float | None = None
    bf16: bool = False
    packages: dict[str, str | None] = field(default_factory=dict)
    compute_dtype: str | None = None
    max_model_b_qlora: float | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ready": self.ready}


def _version(module: str, importer: Any) -> str | None:
    try:
        return str(getattr(importer(module), "__version__", "unknown"))
    except Exception:
        return None


def probe(importer: Any = importlib.import_module) -> ProbeResult:
    result = ProbeResult(python=sys.version.split()[0])
    try:
        torch = importer("torch")
    except Exception:
        result.problems.append("PyTorch is not installed in this environment")
        return result
    result.torch = str(torch.__version__)
    result.torch_cuda = getattr(torch.version, "cuda", None)
    for pkg in ("transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        result.packages[pkg] = _version(pkg, importer)
    if not torch.cuda.is_available():
        result.problems.append(
            "CUDA is not available to PyTorch (CPU-only build or driver problem)"
        )
        return result
    result.cuda_available = True
    major, minor = torch.cuda.get_device_capability(0)
    result.compute_capability = f"{major}.{minor}"
    props = torch.cuda.get_device_properties(0)
    result.gpu = str(props.name)
    result.vram_gb = round(props.total_memory / 1024**3, 1)
    result.bf16 = bool(torch.cuda.is_bf16_supported()) and major >= 8
    if major < 6:
        result.problems.append(f"GPU compute capability {major}.{minor} is too old (need ≥ 6.0)")
    elif major < 7:
        result.compute_dtype = "float32"
        result.notes.append(
            "Pascal GPU: no tensor cores, no bf16, slow fp16 – training runs in fp32 and is slow."
        )
        arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        if arch and not any(a.startswith("sm_6") for a in arch):
            result.problems.append(
                f"This PyTorch build has no Pascal kernels ({', '.join(arch)}); "
                "install PyTorch ≤ 2.14 built for CUDA 12.6 (cu126)"
            )
    else:
        result.compute_dtype = "bfloat16" if result.bf16 else "float16"
    if result.vram_gb is not None:
        # QLoRA-Faustregel (Unsloth-Untergrenzen + Puffer): ~0,75 GB VRAM je Mrd. Parameter + 1,5 GB
        result.max_model_b_qlora = round(max((result.vram_gb - 1.5) / 0.75, 0), 1)
    for pkg in ("transformers", "peft", "trl"):
        if result.packages.get(pkg) is None:
            result.problems.append(f"Python package '{pkg}' is missing")
    if result.packages.get("bitsandbytes") is None:
        result.notes.append(
            "bitsandbytes missing – 4-bit QLoRA not possible (LoRA in 16/32 bit only)"
        )
    return result


def main() -> None:
    print(json.dumps(probe().to_dict(), indent=2))


if __name__ == "__main__":
    main()
