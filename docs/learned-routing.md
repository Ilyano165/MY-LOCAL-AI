# NOVA – Learned Routing

Status: erste Version (Ranker-Format 1, Merkmalsversion 1) · Stand: 2026-10-08 ·
**nicht** als Standard-Router aktiviert (Begründung in §6).

## 1. Ziel und Architektur

Frage, die der Learned Router beantwortet: *Welches der aktuell gültigen Modelle liefert für
diese konkrete Aufgabe voraussichtlich das beste Ergebnis bei vertretbarer Latenz?* –
nicht: welches Modell ist allgemein das beste.

```
Aufgabe
  → Klassifikation (RuleBasedClassifier: Kategorie, Komplexität, Confidence)
  → HARTE REGELN  (RuleBasedRouter.candidates)
       Modell nicht verfügbar · VRAM/RAM reicht nicht · Kontext zu klein ·
       Vision fehlt · Coding/Reasoning fehlt · Tool-Calling fehlt   → ausgeschlossen
  → gültige Kandidaten
  → LEARNED RANKING (LearnedRanker.predict – sieht nur diese Kandidaten)
  → WÄCHTER (RuleBasedRouter.is_valid – jeder Vorschlag wird erneut geprüft)
  → Entscheidung (ranker = "learned", Scores je Kandidat, Begründung, Notizen)
  → Ausführung → Verification Engine → Ergebnis
  → Routing-Log (Urteil, Quality Score, Latenz) → Trainingsdaten
```

Garantien (durch Tests abgesichert, `tests/router/test_learned_router.py`):

* Der Ranker erhält ausschließlich die Kandidaten der Hard-Constraint-Stufe.
* Schlägt ein (fehlerhafter oder ausgetauschter) Ranker ein ungültiges oder unbekanntes Modell
  vor, wird der Vorschlag verworfen, `guard_interventions` erhöht und in den Notizen der
  Entscheidung (also im Routing-Log) vermerkt: `Learned-Vorschlag X verworfen: <Grund>`.
* Kein gültiger Kandidat → `NoModelAvailableError` mit den Ausschlussgründen je Modell.
* Ranking nicht nutzbar (kein Ranker, ungültige Scores): `on_ranker_error="raise"` →
  `LearnedRoutingError`; `"rules"` → Regelrangfolge **mit** Vermerk
  `ranker = "rules (learned unavailable: …)"`. Es gibt keinen stillen Fallback.

## 2. Daten

| Quelle | Inhalt | Status |
|---|---|---|
| Routing-Log (`~/.nova/routing.jsonl`) | Entscheidung + nachgetragenes Ergebnis: Verifikationsurteil, Erfolg, Quality Score, Latenz (vom Agenten je Routing-ID summiert) | Pipeline vollständig; **noch keine echten Daten** (keine Modelle installiert) |
| Routing-Datensatz (150 Aufgaben) | Ergebnisse aus Labels + Flottenkonfiguration abgeleitet (`evaluation/learned_routing.py`) | `SIMULATED` – überall so gekennzeichnet |
| Benchmark-Profile | gemessene tok/s, Antwortlatenz, Speicherbedarf (über `ModelRegistry.effective`) | genutzt, sobald Profile existieren |

Ein Trainingsdatensatz (`OutcomeRecord`) ist: *Aufgabe (Laufzeitwissen) → gewähltes Modell →
Verifikationsurteil → Quality Score → Latenz → Erfolg/Misserfolg*. Entscheidungen ohne
nachgetragenes Ergebnis werden übersprungen (fehlendes Ergebnis ≠ Misserfolg).

**Nutzen** eines Ergebnisses: `Qualität − 0.1 · min(Latenz, 300 s)/300`; Misserfolg höchstens
0.5, harter Fehlschlag (ungeeignetes Modell, Fehler) 0.

**Simulationsregeln** für den Datensatz: fehlende Pflichtfähigkeit/zu kleiner Kontext → harter
Fehlschlag; Stärke (schwächste geforderte Fähigkeit) ≥ benötigte Stufe (LOW 1 / MEDIUM 2 /
HIGH 3) → Erfolg, Qualität 0.9; sonst 0.9 − 0.35·Defizit. Latenz aus angenommenen Raten je
Geschwindigkeitsklasse (gemessene tok/s haben Vorrang).

## 3. Merkmale (`router/feature_extractor.py`)

Nur Laufzeitwissen – `TaskView` hat keine Label-Felder (Test erzwingt das).

* **Aufgabe:** Kategorie und Inhaltskategorie (Klassifikator), geschätzte Komplexität,
  Confidence, Kontextgröße (log), Wortzahl, Zahlen, Aufzählungen, Code-/Reasoning-Signale,
  mehrstufig, formal, tiefe Arbeit, großer Umfang, Tools angeboten, Bild angehängt.
* **Modell:** Reasoning-/Coding-/Vision-Stufe, Tool-Calling, Geschwindigkeit, Speicherbedarf,
  Kontextreserve, gemessene tok/s und Latenz (+ „bekannt“-Flags), gemessen ja/nein,
  relevante Fähigkeit für die Kategorie und Abstand zur geschätzten Komplexität.
* **Historie** (geglättet mit Prior): Erfolgsrate, Ø Qualität, harte Fehlerrate, Anzahl – je
  Modell und je (Modell, Kategorie).
* **Interaktionen:** jedes Aufgabenmerkmal × {Reasoning, Coding, Geschwindigkeit} sowie
  Komplexität × relevante Fähigkeit, Confidence × Fähigkeitsabstand, Kontext × Reserve.
  So kann ein lineares Modell lernen, *für welche Aufgaben* eine Modelleigenschaft zählt.

## 4. Training (`router/ranking_model.py`, `LearnedRanker.train`)

* Lineares Modell `s = w·x̂ + b`, Merkmale standardisiert mit Statistik der Trainingsdaten.
* Verlust: paarweise logistisch (RankNet, linear) innerhalb einer Anfrage, gewichtet mit dem
  Nutzenabstand (Paare ab 0.01 Abstand), plus punktweise logistische Regression auf Erfolg
  (für Logs mit nur einem beobachteten Modell). L2 = 1e-3, 60 Epochen, SGD mit fallender
  Lernrate, fester Seed → deterministisch.
* Reines Python, keine zusätzliche Abhängigkeit; Training auf dem Datensatz ≈ 4 s.
* Persistenz: JSON mit Gewichten, Normalisierung, Historie, Metadaten (Datensatz-/Flotten-Hash,
  Trainingsaufgaben). `load()` verweigert Dateien mit anderer Merkmalsversion oder anderen
  Merkmalsnamen.

**Data Leakage verhindert durch:**

1. Fester Split nach **Aufgabe** (`evaluation/datasets/routing_split.json`, 46 Test- /
   104 Trainingsaufgaben, je Set geschichtet, gesalzener Hash). Log-Daten: `task_id` = Hash
   des normalisierten Aufgabentexts, Split ebenfalls nach `task_id`.
2. Historische Merkmale nur aus Trainingsdaten, für Trainingsbeispiele leave-one-task-out.
3. `LearnedRanker.evaluate()` bricht ab, wenn eine Testaufgabe im Training war.
4. Hyperparameter vor dem ersten Lauf festgelegt; nicht nach Testergebnissen verändert.

## 5. Evaluation (`python -m scripts.compare_routers`)

Trainiert auf dem Trainings-Split, speichert den Ranker (`~/.nova/router/learned_ranker.json`)
und vergleicht auf dem Test-Split in beiden Szenarien (alle verfügbar / Ausfälle):
Routing-Genauigkeit (bestes Modell laut Simulation), akzeptable Wahl, Verifikationserfolg,
Ø Quality Score, Ø Latenz, Ø Nutzen, kritische Capability-Verletzungen, unberechtigte
Fehlschläge, Fallback-Rate, Wächter-Eingriffe – plus der Routing-Evaluator der Baseline.
Bericht: `evaluation/reports/learned_vs_rule.md`.

## 6. Ergebnis des ersten Laufs (Kurzfassung, simuliert)

Learned vs. Rule auf 46 Testaufgaben, Szenario `all_available`: Verifikationserfolg 87 % vs.
60 %, kritische Verletzungen 0 vs. 1, zu schwache Modelle 1 vs. 12 – aber exakte
Routing-Genauigkeit 49 % vs. 62 %, überdimensionierte Wahl 17 vs. 2, Latenz +14 s, im
Ausfallszenario höhere Fallback-Rate (44 % vs. 24 %). Damit ist **kein** „besser“ im Sinne der
Zielfunktion nachgewiesen: sicherer, aber langsamer und unpräziser. Ursache und Korrektur siehe
Bericht (u. a. `pair_margin` größer als die Latenzauflösung → Geschwindigkeit wird nie gelernt).

## 7. Grenzen

* **Keine echten Ergebnisse:** Alle Vergleichszahlen beruhen auf aus Labels abgeleiteten
  Ergebnissen. Sie messen, wie gut ein Router die eigenen Label-Regeln trifft.
* **Kleine Datenbasis:** 104 Trainingsaufgaben, ein Labeler; deutliche Überanpassung
  (Paargenauigkeit Training 87 % vs. Test 69 %).
* **Bandit-Problem bei echten Logs:** Pro Anfrage wird nur das gewählte Modell beobachtet.
  Ohne Exploration lernt der Ranker nur über die Modelle, die er ohnehin wählt.
* **Abhängig vom Klassifikator:** Fehler der Komplexitätsschätzung kann der Ranker nur über
  Textmerkmale teilweise ausgleichen.
* Lineares Modell: keine komplexen Wechselwirkungen jenseits der expliziten Interaktionen.
* Fiktive Flotte ohne Messwerte: tok/s-, Latenz- und Speichermerkmale sind in der Evaluation
  konstant „unbekannt“; ihr Nutzen ist noch unbelegt.
* Kein Fine-Tuning, kein neuronales Netz – bewusst.

## 8. Nächste Schritte

1. `pair_margin`/Latenzgewicht korrigieren, Auswahl per Kreuzvalidierung **nur auf dem
   Training**, dann ein neuer, dokumentierter Testlauf.
2. Echte Routing-Logs sammeln (Pipeline steht); Exploration nur innerhalb gültiger Kandidaten
   und nur mit Zustimmung.
3. Zusatzfähigkeiten (Vision + Coding) als harte Regel – Befund der Baseline.
