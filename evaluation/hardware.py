"""Hardware-Erkennung und Ressourcen-Sampling für das Model-Benchmarking.

**Erkennung** (Status ``DETECTED``, jede Angabe mit Quelle):

* CPU: ``/proc/cpuinfo`` (Modell, physische/logische Kerne, SIMD-Flags), macOS ``sysctl``
* RAM: ``/proc/meminfo`` bzw. ``sysctl hw.memsize``
* NVIDIA: ``nvidia-smi --query-gpu`` · AMD: ``rocm-smi --json`` · Apple Silicon: Metal,
  Unified Memory

**Sampling** (Status ``MEASURED``): :class:`ResourceSampler` fragt während eines Laufs
periodisch VRAM/RAM ab und hält Ausgangs- und Spitzenwert fest.

* mit bekannter Server-PID: VRAM pro Prozess (``nvidia-smi --query-compute-apps``) und RSS
  (``/proc/<pid>/status``) – Methode „per-process“
* ohne PID: systemweite Belegung – Methode „system-wide“; der Spitzenwert enthält dann auch
  andere Prozesse, das Delta zum Ausgangswert ist eine Näherung (im Profil vermerkt)

Grenzen (bewusst dokumentiert): Sampling verpasst Spitzen kürzer als das Intervall; RSS
umfasst bei ``mmap`` nur tatsächlich berührte Seiten der Modelldatei.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.base import ToolError
from tools.process import run_process

CommandRunner = Callable[[Sequence[str]], Awaitable[str | None]]
"""Führt ein Diagnoseprogramm aus; ``None``, wenn es fehlt oder fehlschlägt."""


async def default_runner(argv: Sequence[str]) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        result = await run_process(list(argv), Path.cwd(), timeout_s=15, max_output_bytes=500_000)
    except ToolError:
        return None
    return result.output if result.ok else None


@dataclass
class GpuInfo:
    name: str
    vendor: str
    vram_total_mb: float | None = None
    vram_free_mb: float | None = None
    driver: str | None = None
    index: int = 0
    source: str = ""


@dataclass
class HardwareProfile:
    os: str
    arch: str
    python: str
    cpu_model: str | None = None
    cpu_cores_logical: int | None = None
    cpu_cores_physical: int | None = None
    cpu_flags: list[str] = field(default_factory=list)
    """Für Inferenz relevante SIMD-Erweiterungen (avx2, avx512f, neon, …)."""
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    unified_memory: bool = False
    gpus: list[GpuInfo] = field(default_factory=list)
    accelerators: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    status: str = "DETECTED"
    detected_at: str = ""

    @property
    def vram_total_gb(self) -> float | None:
        totals = [g.vram_total_mb for g in self.gpus if g.vram_total_mb is not None]
        return sum(totals) / 1024 if totals else None

    @property
    def fingerprint(self) -> str:
        """Stabile Kennung der Hardware (ohne flüchtige Werte wie freien Speicher).

        Leistungsmessungen gelten nur für dieselbe Kennung.
        """
        stable = {
            "os": self.os,
            "arch": self.arch,
            "cpu": self.cpu_model,
            "cores": self.cpu_cores_logical,
            "ram_gb": round(self.ram_total_gb) if self.ram_total_gb else None,
            "gpus": sorted(
                f"{g.vendor}:{g.name}:{round((g.vram_total_mb or 0) / 1024)}" for g in self.gpus
            ),
        }
        raw = json.dumps(stable, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vram_total_gb"] = round(self.vram_total_gb, 2) if self.vram_total_gb else None
        data["fingerprint"] = self.fingerprint
        return data

    def summary(self) -> str:
        gpu = (
            ", ".join(
                f"{g.name} ({g.vram_total_mb / 1024:.1f} GB)" if g.vram_total_mb else g.name
                for g in self.gpus
            )
            or "keine dedizierte GPU erkannt"
        )
        ram = f"{self.ram_total_gb:.1f} GB" if self.ram_total_gb else "?"
        return (
            f"CPU {self.cpu_model or '?'} ({self.cpu_cores_physical or '?'}C/"
            f"{self.cpu_cores_logical or '?'}T) · RAM {ram} · GPU {gpu} · "
            f"Beschleuniger: {', '.join(self.accelerators) or 'keine'}"
        )


_RELEVANT_FLAGS = (
    "avx",
    "avx2",
    "avx512f",
    "avx512_vnni",
    "avx512_bf16",
    "amx_tile",
    "fma",
    "f16c",
    "neon",
    "asimd",
    "sve",
    "dotprod",
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def parse_cpuinfo(text: str) -> tuple[str | None, int | None, int | None, list[str]]:
    """(Modell, logische Kerne, physische Kerne, relevante Flags) aus ``/proc/cpuinfo``."""
    model: str | None = None
    logical = 0
    cores: set[tuple[str, str]] = set()
    flags: set[str] = set()
    physical_id = core_id = ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
        elif key in ("model name", "Hardware", "Model") and model is None and value:
            model = value
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
            cores.add((physical_id, core_id))
        elif key in ("flags", "Features"):
            flags |= set(value.split())
    relevant = [f for f in _RELEVANT_FLAGS if f in flags]
    return model, logical or None, len(cores) or None, relevant


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) / 1024 / 1024  # kB → GB
    return values.get("MemTotal"), values.get("MemAvailable")


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor="nvidia",
                    vram_total_mb=float(parts[2]),
                    vram_free_mb=float(parts[3]),
                    driver=parts[4] or None,
                    source="nvidia-smi",
                )
            )
        except ValueError:
            continue
    return gpus


def parse_rocm_smi(text: str) -> list[GpuInfo]:
    """``rocm-smi --showproductname --showmeminfo vram --json`` (Schlüssel je nach Version)."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    gpus = []
    if not isinstance(data, dict):
        return []
    for index, (card, info) in enumerate(sorted(data.items())):
        if not card.startswith("card") or not isinstance(info, dict):
            continue

        def find(*needles: str, info: dict[str, Any] = info) -> Any:
            for key, value in info.items():
                if all(n in key.lower() for n in needles):
                    return value
            return None

        total = find("vram", "total", "memory")
        used = find("vram", "used")
        name = find("card series") or find("card model") or find("product name") or card
        total_mb = float(total) / 1024**2 if str(total or "").isdigit() else None
        used_mb = float(used) / 1024**2 if str(used or "").isdigit() else None
        gpus.append(
            GpuInfo(
                index=index,
                name=str(name),
                vendor="amd",
                vram_total_mb=total_mb,
                vram_free_mb=(total_mb - used_mb) if total_mb is not None and used_mb else None,
                source="rocm-smi",
            )
        )
    return gpus


async def detect_hardware(
    runner: CommandRunner = default_runner, proc: Path = Path("/proc")
) -> HardwareProfile:
    hw = HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python=sys.version.split()[0],
        detected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if (cpuinfo := _read(proc / "cpuinfo")) is not None:
        hw.cpu_model, hw.cpu_cores_logical, hw.cpu_cores_physical, hw.cpu_flags = parse_cpuinfo(
            cpuinfo
        )
        hw.sources["cpu"] = "/proc/cpuinfo"
    if (meminfo := _read(proc / "meminfo")) is not None:
        hw.ram_total_gb, hw.ram_available_gb = parse_meminfo(meminfo)
        hw.sources["ram"] = "/proc/meminfo"
    if platform.system() == "Darwin":
        if hw.cpu_model is None and (
            out := await runner(["sysctl", "-n", "machdep.cpu.brand_string"])
        ):
            hw.cpu_model = out.strip()
            hw.sources["cpu"] = "sysctl"
        if hw.ram_total_gb is None and (out := await runner(["sysctl", "-n", "hw.memsize"])):
            with contextlib.suppress(ValueError):
                hw.ram_total_gb = int(out.strip()) / 1024**3
                hw.sources["ram"] = "sysctl hw.memsize"
        if hw.cpu_cores_physical is None and (
            out := await runner(["sysctl", "-n", "hw.physicalcpu"])
        ):
            with contextlib.suppress(ValueError):
                hw.cpu_cores_physical = int(out.strip())
        if hw.arch == "arm64":
            hw.unified_memory = True
            hw.accelerators.append("metal")
            hw.sources["accelerators"] = "Apple Silicon (arm64/Darwin)"
    if hw.cpu_cores_logical is None:
        hw.cpu_cores_logical = os.cpu_count()
    nvidia = await runner(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia:
        hw.gpus += parse_nvidia_smi(nvidia)
        if hw.gpus:
            hw.accelerators.append("cuda")
            hw.sources["gpu"] = "nvidia-smi"
    rocm = await runner(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if rocm:
        amd = parse_rocm_smi(rocm)
        if amd:
            hw.gpus += amd
            hw.accelerators.append("rocm")
            hw.sources["gpu"] = ", ".join(filter(None, [hw.sources.get("gpu"), "rocm-smi"]))
    for flag in ("avx512f", "avx2", "neon", "asimd"):
        if flag in hw.cpu_flags:
            hw.accelerators.append(f"cpu-{flag}")
            break
    return hw


# ---------------------------------------------------------------------- Prozesse


def find_listening_pid(port: int, proc: Path = Path("/proc")) -> int | None:
    """PID des Prozesses, der auf ``port`` lauscht (Linux, nur eigene Prozesse/root).

    ``/proc/net/tcp{,6}`` → Socket-Inode → ``/proc/<pid>/fd/*``.
    """
    inodes: set[str] = set()
    for table in ("tcp", "tcp6"):
        text = _read(proc / "net" / table)
        if not text:
            continue
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":  # 0A = LISTEN
                continue
            try:
                if int(parts[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(parts[9])
            except (ValueError, IndexError):
                continue
    if not inodes:
        return None
    targets = {f"socket:[{i}]" for i in inodes}
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            for fd in (pid_dir / "fd").iterdir():
                with contextlib.suppress(OSError):
                    if os.readlink(fd) in targets:
                        return int(pid_dir.name)
    return None


def process_rss_mb(pid: int, proc: Path = Path("/proc")) -> float | None:
    text = _read(proc / str(pid) / "status")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            with contextlib.suppress(ValueError, IndexError):
                return int(line.split()[1]) / 1024
    return None


def system_ram_used_mb(proc: Path = Path("/proc")) -> float | None:
    text = _read(proc / "meminfo")
    if text is None:
        return None
    total, available = parse_meminfo(text)
    if total is None or available is None:
        return None
    return (total - available) * 1024


# ---------------------------------------------------------------------- Sampling


@dataclass(frozen=True)
class Sample:
    vram_mb: float | None
    ram_mb: float | None


@dataclass
class PeakUsage:
    vram_method: str
    ram_method: str
    baseline_vram_mb: float | None = None
    peak_vram_mb: float | None = None
    baseline_ram_mb: float | None = None
    peak_ram_mb: float | None = None
    samples: int = 0
    interval_s: float = 0.0
    duration_s: float = 0.0

    @property
    def per_process(self) -> bool:
        return self.vram_method.startswith("per-process") or self.ram_method.startswith(
            "per-process"
        )


class ResourceSampler:
    """Periodisches Sampling während eines Benchmark-Laufs.

    ``async with ResourceSampler(...) as sampler: ...`` – danach ``sampler.result``.
    """

    def __init__(
        self,
        *,
        pid: int | None = None,
        interval_s: float = 0.2,
        runner: CommandRunner = default_runner,
        proc: Path = Path("/proc"),
        probe: Callable[[], Awaitable[Sample]] | None = None,
        gpu_available: bool | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s muss > 0 sein")
        self.pid = pid
        self.interval_s = interval_s
        self.runner = runner
        self.proc = proc
        self._probe = probe or self._default_probe
        self._gpu = gpu_available
        self._task: asyncio.Task[None] | None = None
        self._samples: list[Sample] = []
        self._started = 0.0
        self.result: PeakUsage | None = None

    @property
    def vram_method(self) -> str:
        if self._gpu is False:
            return "unavailable (keine NVIDIA-GPU / nvidia-smi)"
        if self.pid is not None:
            return f"per-process (nvidia-smi compute-apps, PID {self.pid})"
        return "system-wide (nvidia-smi memory.used, inkl. anderer Prozesse)"

    @property
    def ram_method(self) -> str:
        if self.pid is not None:
            return f"per-process (VmRSS, PID {self.pid})"
        return "system-wide (MemTotal − MemAvailable, inkl. anderer Prozesse)"

    async def _vram(self) -> float | None:
        if self._gpu is False:
            return None
        if self.pid is not None:
            out = await self.runner(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            if out is None:
                return None
            total = 0.0
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(self.pid):
                    with contextlib.suppress(ValueError):
                        total += float(parts[1])
            return total
        out = await self.runner(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        if out is None:
            return None
        try:
            return sum(float(x) for x in out.split())
        except ValueError:
            return None

    async def _default_probe(self) -> Sample:
        vram = await self._vram()
        if self.pid is not None:
            ram = await asyncio.to_thread(process_rss_mb, self.pid, self.proc)
        else:
            ram = await asyncio.to_thread(system_ram_used_mb, self.proc)
        return Sample(vram, ram)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            self._samples.append(await self._probe())

    async def __aenter__(self) -> ResourceSampler:
        if self._gpu is None:
            self._gpu = (await self.runner(["nvidia-smi", "-L"])) is not None
        self._started = time.perf_counter()
        self._samples = [await self._probe()]  # Ausgangswert
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._samples.append(await self._probe())  # Endwert
        self.result = self._summarize()

    def _summarize(self) -> PeakUsage:
        base = self._samples[0]
        vram = [s.vram_mb for s in self._samples if s.vram_mb is not None]
        ram = [s.ram_mb for s in self._samples if s.ram_mb is not None]
        return PeakUsage(
            vram_method=self.vram_method,
            ram_method=self.ram_method,
            baseline_vram_mb=base.vram_mb,
            peak_vram_mb=max(vram) if vram else None,
            baseline_ram_mb=base.ram_mb,
            peak_ram_mb=max(ram) if ram else None,
            samples=len(self._samples),
            interval_s=self.interval_s,
            duration_s=round(time.perf_counter() - self._started, 3),
        )
