# NOVA – Evaluationskonzept

Status: Entwurf v0.1 · Stand: 2026-10-08

Ohne Messung ist jede Verbesserung eine Vermutung. NOVA trennt strikt:

- **Tests** (`tests/`): Ist der Code korrekt? Deterministisch, schnell, ohne Modell.
- **Evaluation** (`evaluation/`): Wie gut löst das System Aufgaben? Mit echten Modellen,
  statistisch, langsamer.

---

## 0. Umsetzungsstand (2026-10-08)

Die **Verification Engine** (`evaluation/`, siehe `evaluation/README.md`) prüft Agent-Ergebnisse
unabhängig (Code ausführen, nachrechnen, Quellen abgleichen, Anforderungen zählen) und liefert
einen internen Quality Score. Sie ist in den Agenten integriert (FINALIZE) und wird über einen
eigenen Benchmark mit False-Accept-Gate abgesichert. Noch offen: Eval-Suites für den
*Agenten* mit echtem Modell (E2/E3) und Kalibrierung des Quality Scores gegen menschliche Urteile.

Das **Model-Benchmarking** (`docs/benchmarking.md`) misst pro Modell Fähigkeit (8 Standardaufgaben,
unabhängig bewertet) und Hardware-Leistung (Ladezeit, tok/s, Spitzen-VRAM/RAM) in echten Läufen.
Der Router bevorzugt diese Profile; Modelle ohne Profil sind ausdrücklich `UNMEASURED`.

## 1. Ebenen

| Ebene | Was wird gemessen | Beispiele für Metriken | Ab Phase |
|------|-------------------|------------------------|----------|
| E1 Komponenten | einzelne Bausteine isoliert | Router-Accuracy, Retrieval Recall@k/MRR, Tool-Call-Validität, Plan-Schema-Validität | 3 / 6 |
| E2 Fähigkeiten | eine Fähigkeit des Agenten | Code-Edit gelingt + Tests grün; Frage mit Quellenzitat korrekt beantwortet | 5 |
| E3 End-to-End | mehrstufige realistische Aufgaben | Aufgabe erfüllt, Schritte, Zeit, Tokens, Nutzereingriffe | 5 / 7 |
| E4 Regression | Vergleich zu vorherigem Stand | Δ Erfolgsquote, Δ Latenz, neue Fehlschläge | 8 |

## 2. Metriken

**Qualität**
- Erfolgsquote (pass@1; bei Stichproben pass@k mit dokumentiertem k)
- Teilpunkte pro Check (für Aufgaben mit mehreren Kriterien)
- Zitationsgenauigkeit (RAG): Anteil Aussagen, die durch den zitierten Chunk gedeckt sind
- Router: Accuracy, Konfusionsmatrix pro Rolle

**Effizienz**
- Wall-Clock-Zeit, Time-to-first-token, Tokens (Prompt/Completion), Anzahl Modellaufrufe,
  Anzahl Tool-Aufrufe, Spitzen-Speicherbedarf (sofern messbar)

**Robustheit**
- Anteil ungültiger Tool-Calls / ungültiger strukturierter Ausgaben
- Abbrüche nach Ursache (Budget, Schleife, Fehler)
- Varianz über Wiederholungen

## 3. Bewertungsmethoden (in Reihenfolge der Verlässlichkeit)

1. **Ausführbare Checks:** Tests laufen grün, Programm-Output stimmt, Datei hat erwarteten Inhalt.
2. **Deterministische Vergleiche:** exakte/normalisierte Übereinstimmung, Regex, JSON-Schema, numerische Toleranz.
3. **Referenzbasierte Checks:** erwartete Fakten/Schlüsselbegriffe enthalten, Quellen korrekt.
4. **LLM-as-Judge (lokal):** nur für offene Aufgaben (Schreiben, Analyse). Pflicht:
   feste Rubrik, Judge-Modell ≠ geprüftes Modell wenn möglich, Kalibrierung gegen
   ≥ 30 manuell bewertete Beispiele, Übereinstimmung wird berichtet. Judge-Ergebnisse
   gelten als schwaches Signal.
5. **Manuelle Bewertung:** Stichproben, insbesondere bei neuen Aufgabentypen.

## 4. Datensätze

Format: JSON Lines unter `evaluation/datasets/<suite>.jsonl`, versioniert im Repo.

```json
{"id": "code-0001",
 "suite": "coding",
 "tags": ["python", "bugfix"],
 "instruction": "Behebe den Fehler, wegen dem test_parse_date fehlschlägt.",
 "fixture": "fixtures/code-0001/",
 "checks": [{"type": "command", "cmd": ["pytest", "-q"], "expect_exit": 0},
            {"type": "file_unchanged", "path": "tests/test_dates.py"}],
 "budget": {"max_steps": 20, "max_wall_seconds": 300}}
```

Geplante Suites:
- `router` – gelabelte Anfragen → erwartete Rolle (Phase 3)
- `tools` – Aufgaben, die korrekte Tool-Calls erfordern (Phase 4/5)
- `coding` – kleine Repos mit fehlschlagenden Tests / Feature-Wünschen + verdeckten Tests (Phase 5/7)
- `rag` – Dokumentkorpus + Fragen mit Goldstellen (Phase 6)
- `planning` – mehrstufige Aufgaben mit prüfbaren Zwischenergebnissen (Phase 5)
- `security` – Prompt-Injection- und Missbrauchsszenarien; Erwartung: keine unerlaubte Aktion (Phase 4)

Regeln:
- Eigene Aufgaben haben Vorrang vor öffentlichen Benchmarks (Kontaminationsrisiko,
  Bezug zu echten Nutzungsfällen).
- Öffentliche Benchmarks (z. B. Teilmengen von SWE-bench-artigen Aufgaben, LiveCodeBench)
  nur als Zusatz-Referenz; Harness-Unterschiede werden dokumentiert.
- Ein Holdout-Teil jeder Suite wird nicht zur Prompt-/Router-Optimierung benutzt.

## 5. Reproduzierbarkeit

Jeder Eval-Lauf speichert:
- Git-Commit von NOVA, vollständige aufgelöste Konfiguration (ohne Secrets),
- pro Modell: Name, Quantisierung, Datei-Hash bzw. Runtime-Version, Sampling-Parameter, Seed,
- Hardware-Beschreibung,
- pro Fall: Trace-ID (vollständiger Trace abrufbar), Ergebnis, Metriken.

Default-Sampling für Evals: `temperature=0` bzw. fester Seed; Aufgaben mit Zufallsanteil
werden n-mal wiederholt und mit Mittelwert ± Streuung berichtet.

## 6. Ablauf

```
nova eval run --suite coding --config config/experiments/<name>.toml
nova eval compare <run_a> <run_b>
```

- Ergebnisse unter `~/.nova/evals/<run_id>/` (JSONL + Markdown-Report).
- **Regression-Gate** (ab Phase 8): Änderungen an Prompts, Routing, Kontextverwaltung oder
  Modellwahl werden nur übernommen, wenn die betroffene Suite nicht signifikant schlechter
  wird. Kleine Suites → Unterschiede mit Konfidenzintervall bzw. gepaartem Vergleich
  pro Fall berichten statt nur Prozentzahlen.

## 7. `Evaluator`-Implementierungen

| Evaluator | Zweck |
|----------|-------|
| `CommandCheckEvaluator` | führt Befehle in der Fixture aus (Sandbox-Regeln aus `security.md`) |
| `ExactMatchEvaluator` / `SchemaEvaluator` | deterministische Vergleiche |
| `RetrievalEvaluator` | Recall@k, MRR, nDCG gegen Goldstellen |
| `RouterEvaluator` | Accuracy, Konfusionsmatrix |
| `RubricJudgeEvaluator` | LLM-as-Judge mit Rubrik und Kalibrierungsbericht |
| `CompositeEvaluator` | kombiniert mehrere Checks mit Gewichten |

## 8. Von der Evaluation zum Lernen

Traces + Eval-Ergebnisse bilden später die Datengrundlage für: Router-Training,
Few-Shot-Beispielauswahl, Fehleranalyse und ggf. Fine-Tuning (siehe `model-strategy.md` §8).
Fehlgeschlagene Fälle werden kategorisiert (falsches Modell, falsches Tool, Kontext fehlte,
Verifikation übersehen, Modell-Fähigkeit), damit klar ist, *welche* Komponente verbessert
werden muss.
