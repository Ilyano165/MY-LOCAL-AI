"""Erkennung verfügbarer Ressourcen (VRAM/RAM) für die Modellauswahl.

* RAM: ``/proc/meminfo`` (MemAvailable, Linux), sonst Gesamtspeicher über ``os.sysconf``
  (macOS/BSD). Auf Apple Silicon ist der Speicher geteilt („unified memory“) – dort zählt RAM.
* VRAM: ``nvidia-smi`` (frei, über alle GPUs summiert), falls vorhanden.

Unbekannte Werte bleiben ``None``: Der Router filtert dann nicht nach Speicher, vermerkt das
aber im Routing-Log. Bereits geladene Modelle belegen Speicher, der hier als belegt erscheint –
die Erkennung ist eine Momentaufnahme, keine Reservierung.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.base import ToolError
from tools.process import run_process


@dataclass(frozen=True)
class ResourceBudget:
    vram_gb: float | None = None
    ram_gb: float | None = None
    source: str = "unbekannt"
    ram_usable_fraction: float = 0.8
    """Anteil des freien RAMs, den ein Modell belegen darf (Rest für System/KV-Cache-Spitzen)."""

    @property
    def known(self) -> bool:
        return self.vram_gb is not None or self.ram_gb is not None

    def placement(self, memory_gb: float) -> str | None:
        """``"gpu"``, ``"cpu"`` (passt nur in RAM, langsamer) oder ``None`` (passt nicht)."""
        if self.vram_gb is not None and memory_gb <= self.vram_gb:
            return "gpu"
        if self.ram_gb is not None and memory_gb <= self.ram_gb * self.ram_usable_fraction:
            return "cpu" if self.vram_gb is not None else "ram"
        return None


def _meminfo_available_gb(path: Path = Path("/proc/meminfo")) -> float | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _sysconf_total_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return pages * size / 1024**3 if pages > 0 and size > 0 else None


def windows_memory_gb() -> tuple[float, float] | None:
    """(verfügbar, gesamt) in GB über ``GlobalMemoryStatusEx`` – nur unter Windows."""
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 – nur unter Windows vorhanden
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 1024**3, status.ullTotalPhys / 1024**3


async def _nvidia_free_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = await run_process(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            Path.cwd(),
            timeout_s=10,
        )
    except ToolError:
        return None
    if not result.ok:
        return None
    try:
        return sum(float(line) for line in result.output.split() if line.strip()) / 1024
    except ValueError:
        return None


async def detect_resources() -> ResourceBudget:
    vram = await _nvidia_free_gb()
    ram = _meminfo_available_gb()
    source = "proc/meminfo" if ram is not None else ""
    if ram is None and (win := windows_memory_gb()) is not None:
        ram, source = win[0], "GlobalMemoryStatusEx"
    if ram is None:
        ram = _sysconf_total_gb()
        source = "sysconf (Gesamt-RAM)" if ram is not None else ""
    if vram is not None:
        source += " + nvidia-smi"
    return ResourceBudget(vram_gb=vram, ram_gb=ram, source=source or "unbekannt")
