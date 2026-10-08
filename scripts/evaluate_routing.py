"""Routing-Evaluation: misst den Router gegen den gelabelten Datensatz und schreibt den Bericht.

Verwendung:
    python -m scripts.evaluate_routing [--dataset PFAD] [--fleet PFAD]
        [--report evaluation/reports/routing_baseline.md] [--json PFAD] [--no-report]

Alle Kennzahlen außer der Latenz sind deterministisch (gleicher Datensatz, gleiche Flotte,
gleicher Router → gleiches Ergebnis).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from evaluation.hardware import detect_hardware
from evaluation.routing_evaluator import (
    DEFAULT_DATASET,
    DEFAULT_FLEET,
    Fleet,
    RoutingEvaluator,
    load_dataset,
    render_report,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "evaluation" / "reports" / "routing_baseline.md"
RECOMMENDATIONS_FILE = ROOT / "evaluation" / "reports" / "routing_baseline_recommendations.md"


def _recommendations() -> list[str]:
    """Verbesserungsvorschläge stammen aus der Analyse des Laufs (separate, gepflegte Datei)."""
    if not RECOMMENDATIONS_FILE.is_file():
        return []
    items = []
    for line in RECOMMENDATIONS_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("  ") and items:
            items[-1] += " " + line.strip()
    return items


async def _main(args: argparse.Namespace) -> int:
    tasks = load_dataset(args.dataset)
    fleet = Fleet.from_toml(args.fleet)
    reports = await RoutingEvaluator(fleet).evaluate_all(tasks)
    for r in reports:
        s = r.summary()
        print(
            f"[{s['scenario']}] Score {s['routing_score']:.3f} · Kategorie "
            f"{s['category_accuracy']:.1%} · Komplexität {s['complexity_accuracy']:.1%} · "
            f"kritisch {s['critical_errors']} · Ø {s['latency']['mean_ms']:.2f} ms"
        )
    if args.json:
        write_json(reports, args.json)
    if not args.no_report:
        hw = await detect_hardware()
        text = render_report(
            reports,
            dataset=args.dataset,
            fleet=args.fleet,
            router_name="RuleBasedRouter + RuleBasedClassifier",
            generated=datetime.now(UTC).strftime("%Y-%m-%d"),
            environment=hw.summary(),
            recommendations=_recommendations(),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"Bericht: {args.report}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Routing-Evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fleet", type=Path, default=DEFAULT_FLEET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, help="Rohdaten (Kennzahlen + Einzelergebnisse)")
    parser.add_argument("--no-report", action="store_true")
    sys.exit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
