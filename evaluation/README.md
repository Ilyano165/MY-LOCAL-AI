# evaluation/

**Verification Engine**: prüft Ergebnisse des Agenten unabhängig von dessen Behauptungen und
bewertet sie mit einem internen Quality Score.

**Model-Benchmarking**: misst Modellfähigkeit und Hardware-Leistung in echten Läufen und
speichert Profile, die der Router den Konfigurationswerten vorzieht (siehe
`docs/benchmarking.md`).

| Datei | Inhalt |
|---|---|
| `verifier.py` | `VerificationContext`, `Check`, `CheckResult`, `VerificationStrategy`, `VerificationEngine`, Urteilsregel `decide`, Strategiewahl `infer_strategy` |
| `strategies.py` | Standardstrategien `code`, `files`, `research`, `math`, `text` |
| `test_runner.py` | `TestRunner` (pytest strukturiert über JUnit-XML), `PythonSyntaxCheck`, `TypeCheck`, `UnitTestCheck`, `IntegrationTestCheck` |
| `checks.py` | Datei-, Behauptungs-, Quellen-, Widerspruchs-, Rechen- und Anforderungs-Checks |
| `quality.py` | `QualityScore` (correctness, completeness, requirements, errors, confidence) |
| `benchmarks.py` | Benchmark der Engine mit bekannter Wahrheit; Kennzahl: False-Accept-Rate |
| `benchmark_tasks.py` | 8 Standardaufgaben + Vision-Probe (deterministisch, unabhängig bewertet), Suite-Version mit Inhalts-Hash |
| `model_benchmark.py` | `ModelBenchmark`: Fähigkeit (Scores) und Leistung (Ladezeit, tok/s, Spitzen-VRAM/RAM) getrennt; `ServerLauncher` |
| `benchmark_runner.py` | `ModelBenchmarkRunner` (sequenziell, speichert nur Profile mit echten Messwerten), `attach_profiles`, Bericht |
| `benchmark_results.py` | `Measurement` (Wert + Status + Methode), `ModelProfile`, `ProfileStore` (JSON, `~/.nova/benchmarks`) |
| `hardware.py` | Hardware-Erkennung (CPU, RAM, NVIDIA/AMD/Apple, Fingerprint), `ResourceSampler`, Server-PID über Port |
| `gguf.py` | GGUF-Header-Leser (Quantisierung, Architektur, Trainingskontext) |
| `routing_evaluator.py` | Router-Evaluation gegen `datasets/routing_tasks.jsonl`: Genauigkeit, kritische Fehler (ungeeignetes Modell), Fallbacks, Kalibrierung, Latenz; Bericht |
| `datasets/` | Routing-Datensatz (150 Aufgaben) und Evaluationsflotte – siehe `datasets/README.md` |
| `reports/` | `routing_baseline.md` (+ `.json`) und gepflegte Verbesserungsvorschläge |

## Strategien

| Strategie | Prüfungen |
|---|---|
| **code** | Syntax (`compile`, ohne Ausführung) · Typprüfung (mypy) · Unit-Tests · Integrationstests (falls konfiguriert) · Behauptungsabgleich |
| **files** | Existenz · Inhalt (Pflicht-/Verbotstexte, Regex, Größe, JSON/TOML/Python-Gültigkeit) · Behauptungsabgleich |
| **research** | Quellenvalidierung (Quelle angegeben? lokal vorhanden? im bereitgestellten Material? Zitate wörtlich belegt?) · Widerspruchserkennung (Zahlen, Negation; intern und gegenüber Quellen) · Anforderungen |
| **math** | unabhängige Nachrechnung (eigener sicherer Auswerter, exakte Brüche, keine `eval`) · Anforderungen |
| **text** | Anforderungsprüfung (Wort-/Satz-/Zeichenlimits, Liste, Sprache, JSON, Überschrift, Pflichtbegriffe) · Widerspruchserkennung |

`auto` wählt anhand von Artefakten und Aufgabentext. Jede Agent-Aufgabe kann die Strategie über
`Constraints.verification_strategy` festlegen (`none` schaltet die Gesamtprüfung ab).

## Urteilsregel (`decide`)

1. Ein fehlgeschlagener Check → **failed** (auch Widerlegungen von Behauptungen).
2. **passed** nur mit mindestens einem bestandenen Check, der *unabhängig* ist (eigene Evidenz,
   nicht die Aussage des Agenten) **und** *bestätigen kann*. Reine Problemdetektoren wie die
   Widerspruchssuche können nur widerlegen – „nichts gefunden“ belegt keine Korrektheit.
3. Sonst **unverified**. Nicht prüfbare Checks werden ausgewiesen, nie als Erfolg gezählt.

Kein LLM als Richter in der Engine: Selbstverifikation von LLMs ist nachweislich unzuverlässig,
und ein schwächerer Prüfer kann Ergebnisse verschlechtern.

## Quality Score

Internes, heuristisches Qualitätsmaß – **keine Wahrscheinlichkeit** und nicht kalibriert.
`confidence` beschreibt, wie stark die Bewertung auf unabhängiger Evidenz beruht, nicht wie
sicher das Ergebnis stimmt. Details und Formeln: `quality.py`.

## Benchmark

`python -m scripts.verification_benchmark` – 22 eingebaute Fälle (korrekte und fehlerhafte
Ergebnisse je Strategie). Der Test `tests/evaluation/test_benchmarks.py` ist ein
Regressions-Gate: Jeder False Accept lässt ihn scheitern.

## Model-Benchmarking

`python -m scripts.benchmark_models --config ~/.nova/models.toml` – jeder Wert im Profil hat
einen Status (`MEASURED`, `REPORTED`, `DETECTED`, `CONFIGURED`, `NOT_SUPPORTED`,
`NOT_MEASURABLE`, `FAILED`, `UNMEASURED`). Nur echte Messwerte gelangen in den Router.
Details, Methoden und Grenzen: `docs/benchmarking.md`.

## Routing-Evaluation

`python -m scripts.evaluate_routing` misst den Router gegen 150 gelabelte Aufgaben in zwei
Szenarien (alle Modelle verfügbar / Ausfälle). Ein ungeeignetes Modell (Vision-Aufgabe ohne
Vision, Coding ohne Coding, Tool-Aufgabe ohne Tool-Calling, Kontext zu klein) ist ein
**kritischer Fehler** und wiegt 10× so schwer wie eine falsche Kategorie. Baseline:
`reports/routing_baseline.md`.

## Grenzen

* Recherche-Prüfung arbeitet nur mit bereitgestellten Quelltexten; URLs werden nicht geladen.
* Widerspruchserkennung und Anforderungsextraktion sind Heuristiken (DE/EN).
* Mathematik: Arithmetik und Standardfunktionen; keine Gleichungssysteme, Symbolik, Einheiten.
* Code: Python. Andere Sprachen über `test_command`/`type_command` (nur Exit-Code).
* Die eingebauten Benchmark-Fälle sind klein und selbst erstellt – kein Nachweis der
  Verallgemeinerung.
