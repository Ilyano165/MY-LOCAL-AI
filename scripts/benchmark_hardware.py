"""Erkennt die Hardware (CPU, RAM, GPUs, Beschleuniger) – Grundlage jedes Benchmark-Profils.

Verwendung:
    python -m scripts.benchmark_hardware [--json] [--sample SEKUNDEN] [--pid PID]

``--sample`` misst zusätzlich für die angegebene Dauer die Speicherbelegung (Spitzenwerte),
systemweit oder für einen Prozess (``--pid``). Alle Angaben stammen aus Systemabfragen
(Status DETECTED) bzw. Sampling (MEASURED) – nichts wird geschätzt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from evaluation.hardware import ResourceSampler, detect_hardware


async def _main(args: argparse.Namespace) -> int:
    hw = await detect_hardware()
    data = hw.to_dict()
    if args.sample:
        async with ResourceSampler(pid=args.pid, interval_s=0.25) as sampler:
            await asyncio.sleep(args.sample)
        assert sampler.result is not None
        data["sampling"] = {"status": "MEASURED", **vars(sampler.result)}
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(hw.summary())
    print(f"Fingerprint: {hw.fingerprint}  (Leistungsprofile gelten nur für diese Kennung)")
    print(f"OS: {hw.os} ({hw.arch}), Python {hw.python}")
    if hw.ram_available_gb is not None:
        print(f"RAM verfügbar: {hw.ram_available_gb:.1f} GB")
    for gpu in hw.gpus:
        free = f", frei {gpu.vram_free_mb / 1024:.1f} GB" if gpu.vram_free_mb is not None else ""
        print(
            f"GPU {gpu.index}: {gpu.name} [{gpu.vendor}] {(gpu.vram_total_mb or 0) / 1024:.1f} GB"
            f"{free}, Treiber {gpu.driver or '?'}"
        )
    if hw.cpu_flags:
        print(f"CPU-Erweiterungen: {', '.join(hw.cpu_flags)}")
    print("Quellen: " + ", ".join(f"{k}={v}" for k, v in hw.sources.items()))
    if "sampling" in data:
        s = data["sampling"]
        print(f"\nSampling ({s['samples']} Samples, {s['duration_s']} s) [MEASURED]")
        print(
            f"  RAM  {s['ram_method']}: Spitze {s['peak_ram_mb']} MB, "
            f"Ausgang {s['baseline_ram_mb']} MB"
        )
        print(f"  VRAM {s['vram_method']}: Spitze {s['peak_vram_mb']} MB")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Hardware-Erkennung")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sample", type=float, default=0.0, help="Sekunden Speicher-Sampling")
    parser.add_argument("--pid", type=int, help="Prozess für Sampling (z. B. llama-server)")
    sys.exit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
