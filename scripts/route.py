"""Routing-Trockenlauf: Welches Modell würde NOVA für eine Aufgabe wählen – und warum?

Verwendung:
    python -m scripts.route --config ~/.nova/models.toml "Debug this Python project"
    python -m scripts.route --config models.toml --context-tokens 60000 --latency batch "…"

Prüft die echte Verfügbarkeit (Runtime gefragt) und erkennt VRAM/RAM automatisch.
Führt keine Inferenz aus.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from models.inference import InferenceEngine
from router import (
    Latency,
    NoModelAvailableError,
    ProviderAvailability,
    RoutingRequest,
    RuleBasedRouter,
    detect_resources,
    format_decision,
)


async def _main(args: argparse.Namespace) -> int:
    engine = InferenceEngine.from_toml(args.config)
    try:
        resources = await detect_resources()

        def gb(value: float | None) -> str:
            return "unbekannt" if value is None else f"{value:.1f} GB"

        print(
            f"Ressourcen: VRAM {gb(resources.vram_gb)}, RAM frei {gb(resources.ram_gb)} "
            f"({resources.source})\n"
        )
        router = RuleBasedRouter(
            engine.models, ProviderAvailability(engine.providers), resources=resources
        )
        request = RoutingRequest(
            args.task,
            conversation_tokens=args.context_tokens,
            needs_vision=args.vision,
            latency=Latency(args.latency),
            required_tools=("*",) if args.tools else (),
        )
        try:
            print(format_decision(await router.route(request)))
        except NoModelAvailableError as exc:
            print(f"Kein Modell wählbar: {exc}")
            return 1
    finally:
        await engine.aclose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Routing-Trockenlauf")
    parser.add_argument("task")
    parser.add_argument("--config", required=True)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--tools", action="store_true", help="Aufgabe braucht Tool-Calling")
    parser.add_argument("--latency", choices=[x.value for x in Latency], default="normal")
    args = parser.parse_args()
    args.config = Path(args.config).expanduser()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
