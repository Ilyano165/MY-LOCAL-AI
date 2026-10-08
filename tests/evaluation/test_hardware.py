from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from evaluation.hardware import (
    ResourceSampler,
    Sample,
    detect_hardware,
    find_listening_pid,
    parse_cpuinfo,
    parse_meminfo,
    parse_nvidia_smi,
    parse_rocm_smi,
    process_rss_mb,
    system_ram_used_mb,
)

CPUINFO = """processor\t: 0
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse avx avx2 fma f16c avx512f avx512_vnni
processor\t: 1
model name\t: AMD Ryzen 9 7950X 16-Core Processor
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse avx avx2 fma f16c avx512f avx512_vnni
processor\t: 2
physical id\t: 0
core id\t\t: 1
"""
MEMINFO = "MemTotal:       65536000 kB\nMemFree: 1000 kB\nMemAvailable:   32768000 kB\n"
NVIDIA = "0, NVIDIA GeForce RTX 4090, 24564, 23000, 560.35.03\n"


def fake_proc(tmp_path: Path, meminfo: str = MEMINFO) -> Path:
    (tmp_path / "cpuinfo").write_text(CPUINFO)
    (tmp_path / "meminfo").write_text(meminfo)
    return tmp_path


def runner_for(outputs: dict[str, str | None]):  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    async def run(argv: Sequence[str]) -> str | None:
        calls.append(list(argv))
        for prefix, out in outputs.items():
            if " ".join(argv).startswith(prefix):
                return out
        return None

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_parse_cpuinfo() -> None:
    model, logical, physical, flags = parse_cpuinfo(CPUINFO)
    assert model == "AMD Ryzen 9 7950X 16-Core Processor"
    assert logical == 3 and physical == 2
    assert {"avx2", "avx512f", "avx512_vnni", "fma"} <= set(flags)
    assert "sse" not in flags


def test_parse_meminfo() -> None:
    total, available = parse_meminfo(MEMINFO)
    assert total == pytest.approx(62.5) and available == pytest.approx(31.25)


def test_parse_nvidia_smi() -> None:
    gpus = parse_nvidia_smi(NVIDIA + "garbage line\n")
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 4090" and gpus[0].vram_total_mb == 24564
    assert gpus[0].driver == "560.35.03"


def test_parse_rocm_smi() -> None:
    text = json.dumps(
        {
            "card0": {
                "Card series": "Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": str(24 * 1024**3),
                "VRAM Total Used Memory (B)": str(1024**3),
            },
            "system": {"Driver version": "6.8"},
        }
    )
    gpus = parse_rocm_smi(text)
    assert len(gpus) == 1 and gpus[0].vendor == "amd"
    assert gpus[0].vram_total_mb == pytest.approx(24 * 1024)
    assert gpus[0].vram_free_mb == pytest.approx(23 * 1024)
    assert parse_rocm_smi("not json") == []


async def test_detect_hardware_with_nvidia(tmp_path: Path) -> None:
    hw = await detect_hardware(runner_for({"nvidia-smi": NVIDIA}), fake_proc(tmp_path))
    assert hw.cpu_model and "7950X" in hw.cpu_model
    assert hw.ram_total_gb == pytest.approx(62.5)
    assert hw.gpus[0].vendor == "nvidia"
    assert "cuda" in hw.accelerators and "cpu-avx512f" in hw.accelerators
    assert hw.sources["gpu"] == "nvidia-smi" and hw.sources["cpu"] == "/proc/cpuinfo"
    assert hw.vram_total_gb == pytest.approx(24564 / 1024)
    data = hw.to_dict()
    assert data["fingerprint"] == hw.fingerprint and data["status"] == "DETECTED"
    assert "RTX 4090" in hw.summary()


async def test_detect_hardware_without_gpu(tmp_path: Path) -> None:
    hw = await detect_hardware(runner_for({}), fake_proc(tmp_path))
    assert hw.gpus == [] and "cuda" not in hw.accelerators
    assert "keine dedizierte GPU" in hw.summary()


async def test_fingerprint_ignores_free_memory(tmp_path: Path) -> None:
    a = await detect_hardware(runner_for({"nvidia-smi": NVIDIA}), fake_proc(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    b = await detect_hardware(
        runner_for({"nvidia-smi": NVIDIA.replace("23000", "100")}),
        fake_proc(other, MEMINFO.replace("32768000", "1000")),
    )
    assert a.fingerprint == b.fingerprint
    c = await detect_hardware(runner_for({}), fake_proc(tmp_path))
    assert c.fingerprint != a.fingerprint  # GPU fehlt → andere Hardware


def test_find_listening_pid(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    header = "  sl  local_address rem_address   st tx_queue rx_queue ... uid timeout inode\n"
    tail = " 00000000:00000000 00:00000000 00000000  1000 0"
    # Port 8080 = 0x1F90, LISTEN (0A), Inode 4242; Port 9090 nicht lauschend (01)
    net.joinpath("tcp").write_text(
        header
        + f"   0: 0100007F:1F90 00000000:0000 0A{tail} 4242\n"
        + f"   1: 0100007F:2382 00000000:0000 01{tail} 777\n"
    )
    fd = tmp_path / "1234" / "fd"
    fd.mkdir(parents=True)
    os.symlink("socket:[4242]", fd / "3")
    os.symlink("/dev/null", fd / "0")
    (tmp_path / "self").mkdir()
    assert find_listening_pid(8080, tmp_path) == 1234
    assert find_listening_pid(9090, tmp_path) is None
    assert find_listening_pid(1, tmp_path / "missing") is None


def test_process_and_system_ram(tmp_path: Path) -> None:
    (tmp_path / "77").mkdir()
    (tmp_path / "77" / "status").write_text("Name: llama-server\nVmRSS:\t 2097152 kB\n")
    assert process_rss_mb(77, tmp_path) == pytest.approx(2048)
    assert process_rss_mb(78, tmp_path) is None
    fake_proc(tmp_path)
    assert system_ram_used_mb(tmp_path) == pytest.approx((62.5 - 31.25) * 1024)


async def test_sampler_tracks_baseline_and_peak() -> None:
    values = iter([Sample(1000, 4000), Sample(5000, 6000), Sample(7000, 9000)])
    last = Sample(2000, 5000)

    async def probe() -> Sample:
        return next(values, last)

    import asyncio

    async with ResourceSampler(interval_s=0.01, probe=probe, gpu_available=True) as sampler:
        await asyncio.sleep(0.05)
    usage = sampler.result
    assert usage is not None
    assert usage.baseline_vram_mb == 1000 and usage.baseline_ram_mb == 4000
    assert usage.peak_vram_mb == 7000 and usage.peak_ram_mb == 9000
    assert usage.samples >= 4
    assert usage.vram_method.startswith("system-wide")
    assert not usage.per_process


async def test_sampler_per_process_vram(tmp_path: Path) -> None:
    (tmp_path / "55").mkdir()
    (tmp_path / "55" / "status").write_text("VmRSS:\t1048576 kB\n")
    runner = runner_for(
        {
            "nvidia-smi -L": "GPU 0: RTX",
            "nvidia-smi --query-compute-apps": "55, 6000\n99, 3000\n55, 500\n",
        }
    )
    sampler = ResourceSampler(pid=55, interval_s=0.01, runner=runner, proc=tmp_path)
    async with sampler:
        pass
    usage = sampler.result
    assert usage is not None and usage.per_process
    assert usage.peak_vram_mb == 6500  # nur PID 55, beide Kontexte
    assert usage.peak_ram_mb == pytest.approx(1024)


async def test_sampler_without_gpu(tmp_path: Path) -> None:
    fake_proc(tmp_path)
    sampler = ResourceSampler(interval_s=0.01, runner=runner_for({}), proc=tmp_path)
    async with sampler:
        pass
    assert sampler.result is not None
    assert sampler.result.peak_vram_mb is None
    assert sampler.result.vram_method.startswith("unavailable")


def test_sampler_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        ResourceSampler(interval_s=0)
