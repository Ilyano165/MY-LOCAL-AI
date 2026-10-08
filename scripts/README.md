# scripts

Entwickler- und Betriebsskripte (`python -m scripts.<name> --help`).

| Skript | Zweck |
|---|---|
| `model_health.py` | Health Check konfigurierter Modelle (erreichbar, ladbar, Inferenz, Kontext, Tools) |
| `route.py` | Routing-Trockenlauf; nutzt Benchmark-Profile (`--no-profiles` = nur Konfiguration) |
| `verification_benchmark.py` | Benchmark der Verification Engine (False-Accept-Gate) |
| `benchmark_hardware.py` | Hardware-Erkennung mit Quellen und Fingerprint; optional Speicher-Sampling |
| `evaluate_routing.py` | Routing-Evaluation gegen den Datensatz, schreibt `evaluation/reports/routing_baseline.md` |
| `compare_routers.py` | Learned Router auf Trainings-Split trainieren, gegen Regel-Router auf Test-Split vergleichen (`evaluation/reports/learned_vs_rule.md`) |
| `benchmark_models.py` | Reales Model-Benchmarking: Fähigkeit + Leistung messen, Profile speichern (`docs/benchmarking.md`) |
