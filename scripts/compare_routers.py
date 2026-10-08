"""Trainiert den Learned Router auf dem festen Trainings-Split und vergleicht ihn mit dem
Regel-Router auf dem Test-Split.

Verwendung:
    python -m scripts.compare_routers [--out ~/.nova/router/learned_ranker.json]
        [--report evaluation/reports/learned_vs_rule.md] [--json PFAD]
        [--routing-log ~/.nova/routing.jsonl]

Ergebnisse auf dem Datensatz sind SIMULIERT (aus Labels abgeleitet, siehe
``evaluation/learned_routing.py``). Echte Ergebnisse aus ``--routing-log`` werden – falls
vorhanden – zusätzlich ins Training aufgenommen und im Bericht getrennt ausgewiesen.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.learned_routing import (
    SPLIT_FILE,
    compare,
    learned_builder,
    load_split,
    make_split,
    rule_builder,
    simulated_records,
    split_tasks,
)
from evaluation.routing_evaluator import (
    DEFAULT_DATASET,
    DEFAULT_FLEET,
    Fleet,
    RoutingEvaluator,
    ScenarioReport,
    file_hash,
    load_dataset,
)
from router.learned_router import LearnedRanker, LearnedRouter
from router.routing_log import JsonlRoutingLog
from router.training_data import from_routing_log, split_by_task

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "evaluation" / "reports" / "learned_vs_rule.md"
DEFAULT_OUT = Path("~/.nova/router/learned_ranker.json").expanduser()
ASSESSMENT_FILE = ROOT / "evaluation" / "reports" / "learned_vs_rule_assessment.md"
"""Gepflegte, menschenlesbare Bewertung des Laufs (wird an den Bericht angehängt)."""

LABELS = {
    "routing_accuracy": "Routing-Genauigkeit (= bestes Modell laut Simulation)",
    "acceptable_rate": "Akzeptable Wahl (Nutzen ≤ 0.02 unter dem besten)",
    "verification_success": "Verifikationserfolg (simuliert)",
    "avg_quality": "Ø Quality Score (simuliert)",
    "avg_latency_s": "Ø Latenz in s (simuliert)",
    "avg_utility": "Ø Nutzen (Qualität − Latenzkosten)",
    "critical_violations": "Kritische Capability-Verletzungen",
    "routing_failures": "Unberechtigte Routing-Fehlschläge",
    "fallback_rate": "Fallback-Rate (Wahl durch Ausfall geändert)",
    "guard_interventions": "Wächter: verworfene ungültige Vorschläge",
    "ranker_unavailable": "Learned Ranking nicht nutzbar (Regelrangfolge)",
}
LOWER_IS_BETTER = {
    "avg_latency_s",
    "critical_violations",
    "routing_failures",
    "fallback_rate",
    "guard_interventions",
    "ranker_unavailable",
}


def _fmt(key: str, value: Any) -> str:
    if isinstance(value, float) and key not in ("avg_latency_s", "avg_utility", "avg_quality"):
        return f"{value * 100:.1f} %"
    if isinstance(value, float):
        return f"{value:.3f}" if key != "avg_latency_s" else f"{value:.1f}"
    return str(value)


def _differences(reports: list[Any]) -> list[tuple[str, Any, Any]]:
    rule = next(r for r in reports if r.router == "rule" and r.scenario == "all_available")
    learned = next(r for r in reports if r.router == "learned" and r.scenario == "all_available")
    pairs = zip(rule.results, learned.results, strict=True)
    return [
        (a.task_id, a, b)
        for a, b in pairs
        if a.selected != b.selected and a.utility is not None and b.utility is not None
    ]


def _verdict(key: str, rule: Any, learned: Any) -> str:
    if rule == learned:
        return "gleich"
    better = (learned < rule) if key in LOWER_IS_BETTER else (learned > rule)
    return "Learned günstiger" if better else "Rule günstiger"


async def _main(args: argparse.Namespace) -> int:
    tasks = load_dataset(args.dataset)
    fleet = Fleet.from_toml(args.fleet)
    test_ids = load_split(args.split)
    if sorted(test_ids) != make_split(tasks):
        print(
            "WARNUNG: Split-Datei weicht von make_split() ab – Datei hat Vorrang", file=sys.stderr
        )
    train_tasks, test_tasks = split_tasks(tasks, test_ids)
    overlap = {t.id for t in train_tasks} & {t.id for t in test_tasks}
    assert not overlap, overlap

    train_records = await simulated_records(train_tasks, fleet)
    test_records = await simulated_records(test_tasks, fleet)
    real_train: list[Any] = []
    real_test: list[Any] = []
    if args.routing_log and await asyncio.to_thread(args.routing_log.expanduser().exists):
        real, skipped = from_routing_log(JsonlRoutingLog(args.routing_log).joined())
        real_train, real_test = split_by_task(real)
        print(f"Routing-Log: {len(real)} echte Ergebnisse ({skipped} ohne Ergebnis übersprungen)")
    models = {m.name: m for m in fleet.models}
    ranker = LearnedRanker.train(
        train_records + real_train,
        models,
        metadata={
            "dataset": f"{args.dataset.name}@{file_hash(args.dataset)}",
            "fleet": f"{args.fleet.name}@{file_hash(args.fleet)}",
            "trained": datetime.now(UTC).isoformat(timespec="seconds"),
            "data": "simulated" + (" + routing_log" if real_train else ""),
        },
    )
    ranker.save(args.out)
    weights_hash = hashlib.sha256(json.dumps(ranker.model.weights).encode()).hexdigest()[:12]
    offline = ranker.evaluate(test_records, models)
    offline_real = ranker.evaluate(real_test, models) if real_test else None

    reports = await compare(
        test_tasks, fleet, {"rule": rule_builder, "learned": learned_builder(ranker)}
    )
    rule_eval = await RoutingEvaluator(fleet).evaluate_all(test_tasks)
    learned_eval = await RoutingEvaluator(
        fleet, router_factory=lambda r, a: LearnedRouter(r, a, ranker, on_ranker_error="raise")
    ).evaluate_all(test_tasks)

    metrics = [r.metrics() for r in reports]
    for m in metrics:
        print(
            f"[{m['router']}/{m['scenario']}] Genauigkeit {m['routing_accuracy']:.1%} · "
            f"Erfolg {m['verification_success']:.1%} · Qualität {m['avg_quality']:.3f} · "
            f"Latenz {m['avg_latency_s']:.1f} s · kritisch {m['critical_violations']}"
        )
    text = render(
        args,
        tasks,
        train_tasks,
        test_tasks,
        ranker,
        offline,
        offline_real,
        metrics,
        rule_eval,
        learned_eval,
        weights_hash,
        len(real_train),
        len(real_test),
        _differences(reports),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(f"Bericht: {args.report}\nRanker: {args.out}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "comparison": metrics,
                    "offline_simulated": offline.to_dict(),
                    "offline_routing_log": offline_real.to_dict() if offline_real else None,
                    "evaluator": {
                        "rule": [r.summary() for r in rule_eval],
                        "learned": [r.summary() for r in learned_eval],
                    },
                    "results": {
                        f"{r.router}/{r.scenario}": [
                            vars(x) | {"outcome": vars(x.outcome) if x.outcome else None}
                            for x in r.results
                        ]
                        for r in reports
                    },
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def render(
    args: argparse.Namespace,
    tasks: list[Any],
    train: list[Any],
    test: list[Any],
    ranker: LearnedRanker,
    offline: Any,
    offline_real: Any,
    metrics: list[dict[str, Any]],
    rule_eval: list[ScenarioReport],
    learned_eval: list[ScenarioReport],
    weights_hash: str,
    n_real_train: int,
    n_real_test: int,
    differences: list[Any],
) -> str:
    t = ranker.model.trained_on
    lines = [
        "# NOVA – Vergleich Rule Router vs. Learned Router",
        "",
        f"Erzeugt: {datetime.now(UTC).strftime('%Y-%m-%d')} · Datensatz `{args.dataset.name}` "
        f"(sha256 {file_hash(args.dataset)}) · Flotte `{args.fleet.name}` "
        f"(sha256 {file_hash(args.fleet)}) · Split `{args.split.name}` · Ranker-Gewichte "
        f"sha256 {weights_hash}",
        "",
        "Reproduzieren: `python -m scripts.compare_routers` (deterministisch, außer Datum).",
        "",
        "> **Achtung – simulierte Ergebnisse.** Für die Datensatzaufgaben gibt es noch keine "
        "Läufe mit echten Modellen. Erfolg, Qualität und Latenz sind aus den Labels und der "
        "Flottenkonfiguration abgeleitet (Regeln in `evaluation/learned_routing.py`). Die Werte "
        "zeigen, wie gut ein Router die Label-Regeln trifft – **nicht**, welcher mit echten "
        "Modellen bessere Antworten liefert. „Günstiger“ in den Tabellen bezieht sich nur auf "
        "diese Simulation.",
        "",
        "## Daten und Split",
        "",
        f"* Aufgaben: {len(tasks)} · Training {len(train)} · Test {len(test)} (fester, nach Set "
        "geschichteter Split nach Aufgaben-ID; keine Aufgabe in beiden Teilen – im Skript per "
        "Assertion und im Test geprüft)",
        f"* Trainingsbeispiele: {t.get('examples')} (Aufgabe × Szenario × gültiger Kandidat), "
        f"{t.get('pairs')} Paare, Quellen: {', '.join(t.get('sources', []))}",
        f"* Echte Ergebnisse aus dem Routing-Log: Training {n_real_train}, Test {n_real_test}"
        + (" – noch keine vorhanden" if not (n_real_train or n_real_test) else ""),
        "* Historische Merkmale (Erfolg/Qualität je Modell): nur aus Trainingsdaten, für "
        "Trainingsbeispiele leave-one-task-out",
        "",
        "## Offline-Ranking-Güte (Test-Split, simuliert)",
        "",
        "| Kennzahl | Wert |",
        "|---|---|",
        f"| Paargenauigkeit Training | {t.get('train_pair_accuracy', 0) * 100:.1f} % |",
        f"| Paargenauigkeit Test | {offline.pair_accuracy * 100:.1f} % |",
        f"| Top-1 = bester Kandidat (Test) | {offline.top1_match * 100:.1f} % |",
        f"| Ø Nutzen gewählt / bester (Test) | {offline.mean_selected_utility:.3f} / "
        f"{offline.mean_best_utility:.3f} |",
    ]
    if offline_real is not None:
        lines.append(f"| Paargenauigkeit echte Logs (Test) | {offline_real.pair_accuracy:.3f} |")
    lines += [
        "",
        "Der Abstand Training ↔ Test bei der Paargenauigkeit zeigt Überanpassung an die "
        "Trainingsaufgaben.",
        "",
        "## Vergleich auf dem Test-Split",
        "",
    ]
    for scenario in dict.fromkeys(m["scenario"] for m in metrics):
        rule = next(m for m in metrics if m["router"] == "rule" and m["scenario"] == scenario)
        learned = next(m for m in metrics if m["router"] == "learned" and m["scenario"] == scenario)
        lines += [
            f"### Szenario `{scenario}` ({rule['tasks']} Aufgaben)",
            "",
            "| Kennzahl | Rule | Learned | Einordnung (Simulation) |",
            "|---|---|---|---|",
        ]
        for key, label in LABELS.items():
            lines.append(
                f"| {label} | {_fmt(key, rule[key])} | {_fmt(key, learned[key])} | "
                f"{_verdict(key, rule[key], learned[key])} |"
            )
        lines.append("")
    lines += [
        "## Routing-Evaluator (Test-Split, gleiche Bewertung wie die Baseline)",
        "",
        "| Kennzahl | "
        + " | ".join(f"Rule/{r.scenario.name}" for r in rule_eval)
        + " | "
        + " | ".join(f"Learned/{r.scenario.name}" for r in learned_eval)
        + " |",
        "|---|" + "---|" * (len(rule_eval) + len(learned_eval)),
    ]
    rows: list[tuple[str, Callable[[ScenarioReport], str]]] = [
        ("Routing-Score (gewichtet)", lambda r: f"{r.routing_score:.3f}"),
        ("Kritische Fehler", lambda r: str(r.critical_errors)),
        ("Schwere Fehler (zu schwach)", lambda r: str(r.count("major"))),
        ("Überdimensioniert", lambda r: str(r.count("over_provisioned"))),
        ("Nicht verfügbares Modell gewählt", lambda r: str(r.unavailable_violations)),
        ("Failover-Sicherheit", lambda r: f"{(r.failover_safety or 0) * 100:.1f} %"),
        ("Routing-Latenz Ø (ms, gemessen)", lambda r: f"{r.latency()['mean_ms']:.2f}"),
    ]
    for label, fn in rows:
        lines.append(
            f"| {label} | " + " | ".join(fn(r) for r in [*rule_eval, *learned_eval]) + " |"
        )
    lines += [
        "",
        "Kategorie- und Komplexitätsgenauigkeit sind für beide Router identisch "
        "(gleicher Klassifikator); der Learned Router ändert nur die Rangfolge.",
        "",
    ]
    lines += [
        "## Unterschiedliche Entscheidungen (Szenario `all_available`)",
        "",
        "| Aufgabe | Rule → Nutzen | Learned → Nutzen | bestes laut Simulation |",
        "|---|---|---|---|",
    ]
    for task_id, rule_x, learned_x in differences:
        lines.append(
            f"| {task_id} | {rule_x.selected} → {rule_x.utility:.3f} | "
            f"{learned_x.selected} → {learned_x.utility:.3f} | {', '.join(rule_x.oracle)} |"
        )
    lines += [
        "",
        "## Einflussreichste Merkmale (standardisierte Gewichte)",
        "",
        "| Merkmal | Gewicht |",
        "|---|---|",
    ]
    for name, weight in ranker.model.top_weights(12):
        lines.append(f"| `{name}` | {weight:+.3f} |")
    if ASSESSMENT_FILE.is_file():
        lines += ["", ASSESSMENT_FILE.read_text(encoding="utf-8").strip()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule Router vs. Learned Router")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fleet", type=Path, default=DEFAULT_FLEET)
    parser.add_argument("--split", type=Path, default=SPLIT_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ranker-Datei")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--routing-log", type=Path, help="echte Routing-Ergebnisse (JSONL)")
    args = parser.parse_args()
    args.out = args.out.expanduser()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
