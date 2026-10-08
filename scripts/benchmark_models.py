"""Reales Model-Benchmarking: misst Fähigkeit und Hardware-Leistung konfigurierter Modelle.

Verwendung:
    python -m scripts.benchmark_models --config ~/.nova/models.toml [--model NAME ...]
        [--tasks chat,coding] [--no-exec] [--performance-only | --capabilities-only]
        [--measure-load] [--server-cmd "llama-server -m … --port 8080"] [--server-pid PID]
        [--profiles DIR] [--json] [--list]

Ergebnisse: ``~/.nova/benchmarks/<modell>.json`` (bzw. ``--profiles``/``NOVA_BENCHMARK_DIR``).
Der Router nutzt diese Profile automatisch (``attach_profiles``); Modelle ohne Profil sind
``UNMEASURED``.

Achtung: Coding-Aufgaben führen vom Modell erzeugten Code in einem Subprozess aus (ohne
Shell, Zeitlimit, temporäres Verzeichnis, Umgebung ohne Secrets). ``--no-exec`` überspringt sie.
Exit-Code 0 = alle Läufe ergaben ein Profil, 1 = mindestens ein Lauf ohne Messwerte.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evaluation.benchmark_results import ProfileStore
from evaluation.benchmark_runner import ModelBenchmarkRunner, render_profile
from evaluation.benchmark_tasks import standard_tasks, suite_version, task_ids
from evaluation.hardware import detect_hardware
from evaluation.model_benchmark import BenchmarkOptions, ServerLauncher
from models.inference import InferenceEngine


def _list(store: ProfileStore, engine: InferenceEngine) -> int:
    print(f"Profile in {store.root} (aktuelle Suite {suite_version()}):")
    for model in engine.models.list():
        status = engine.models.data_status(model.name)
        print(f"  {model.name:<24} {status.describe()}")
    return 0


async def _main(args: argparse.Namespace) -> int:
    engine = InferenceEngine.from_toml(args.config)
    try:
        hardware = await detect_hardware()
        store = ProfileStore(
            args.profiles,
            hardware_fingerprint=hardware.fingerprint,
            benchmark_version=suite_version(),
        )
        engine.models.set_measurements(store)
        if args.list:
            return _list(store, engine)
        if args.server_cmd and len(args.model or []) != 1:
            print("--server-cmd erfordert genau ein --model", file=sys.stderr)
            return 2
        options = BenchmarkOptions(
            tasks=args.tasks.split(",") if args.tasks else None,
            timeout_s=args.timeout,
            execute_code=not args.no_exec,
            capabilities=not args.performance_only,
            performance=not args.capabilities_only,
            throughput_runs=args.runs,
            throughput_tokens=args.tokens,
            measure_load=args.measure_load,
            server_pid=args.server_pid,
            long_context_max_tokens=args.long_context_tokens,
        )
        runner = ModelBenchmarkRunner(
            engine,
            store,
            options=options,
            hardware=hardware,
            tasks=standard_tasks(),
            launcher_factory=(lambda _name: ServerLauncher(args.server_cmd))
            if args.server_cmd
            else None,
            progress=lambda msg: print(msg, file=sys.stderr),
        )
        print(f"Hardware: {hardware.summary()} [DETECTED]", file=sys.stderr)
        outcomes = await runner.run(args.model)
    finally:
        await engine.aclose()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "model": o.model,
                        "error": o.error,
                        "path": str(o.path) if o.path else None,
                        "profile": o.profile.to_dict() if o.profile else None,
                    }
                    for o in outcomes
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for o in outcomes:
            print("\n" + "=" * 78)
            if o.profile is not None:
                print(render_profile(o.profile))
            if o.path:
                print(f"\nGespeichert: {o.path}")
            if o.error:
                print(f"\nFEHLER {o.model}: {o.error}")
    return 0 if all(o.ok for o in outcomes) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Model-Benchmarking (echte Messläufe)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", action="append", help="Registry-Modell (mehrfach möglich)")
    parser.add_argument("--tasks", help=f"Teilmenge, kommagetrennt: {','.join(task_ids())}")
    parser.add_argument("--no-exec", action="store_true", help="generierten Code nicht ausführen")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--performance-only", action="store_true")
    mode.add_argument("--capabilities-only", action="store_true")
    parser.add_argument("--runs", type=int, default=3, help="Durchsatz-Wiederholungen")
    parser.add_argument("--tokens", type=int, default=128, help="Tokens je Durchsatzlauf")
    parser.add_argument("--timeout", type=float, default=300.0, help="Zeitlimit je Anfrage (s)")
    parser.add_argument("--long-context-tokens", type=int, default=16_384)
    parser.add_argument(
        "--measure-load", action="store_true", help="Ollama: Modell entladen und Ladezeit messen"
    )
    parser.add_argument("--server-cmd", help="Runtime selbst starten (misst Ladezeit)")
    parser.add_argument("--server-pid", type=int, help="PID der Runtime für Speicher-Sampling")
    parser.add_argument("--profiles", type=Path, help="Profilverzeichnis")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="Messstatus aller Modelle anzeigen")
    args = parser.parse_args()
    args.config = args.config.expanduser()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
