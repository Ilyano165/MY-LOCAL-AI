# NOVA – Vergleich Rule Router vs. Learned Router

Erzeugt: 2026-10-08 · Datensatz `routing_tasks.jsonl` (sha256 a2d6cecacf4f) · Flotte `routing_fleet.toml` (sha256 47c585ba42ed) · Split `routing_split.json` · Ranker-Gewichte sha256 bc464bd129a1

Reproduzieren: `python -m scripts.compare_routers` (deterministisch, außer Datum).

> **Achtung – simulierte Ergebnisse.** Für die Datensatzaufgaben gibt es noch keine Läufe mit echten Modellen. Erfolg, Qualität und Latenz sind aus den Labels und der Flottenkonfiguration abgeleitet (Regeln in `evaluation/learned_routing.py`). Die Werte zeigen, wie gut ein Router die Label-Regeln trifft – **nicht**, welcher mit echten Modellen bessere Antworten liefert. „Günstiger“ in den Tabellen bezieht sich nur auf diese Simulation.

## Daten und Split

* Aufgaben: 150 · Training 104 · Test 46 (fester, nach Set geschichteter Split nach Aufgaben-ID; keine Aufgabe in beiden Teilen – im Skript per Assertion und im Test geprüft)
* Trainingsbeispiele: 987 (Aufgabe × Szenario × gültiger Kandidat), 1319 Paare, Quellen: simulated
* Echte Ergebnisse aus dem Routing-Log: Training 0, Test 0 – noch keine vorhanden
* Historische Merkmale (Erfolg/Qualität je Modell): nur aus Trainingsdaten, für Trainingsbeispiele leave-one-task-out

## Offline-Ranking-Güte (Test-Split, simuliert)

| Kennzahl | Wert |
|---|---|
| Paargenauigkeit Training | 87.4 % |
| Paargenauigkeit Test | 69.4 % |
| Top-1 = bester Kandidat (Test) | 41.0 % |
| Ø Nutzen gewählt / bester (Test) | 0.818 / 0.827 |

Der Abstand Training ↔ Test bei der Paargenauigkeit zeigt Überanpassung an die Trainingsaufgaben.

## Vergleich auf dem Test-Split

### Szenario `all_available` (46 Aufgaben)

| Kennzahl | Rule | Learned | Einordnung (Simulation) |
|---|---|---|---|
| Routing-Genauigkeit (= bestes Modell laut Simulation) | 62.2 % | 48.9 % | Rule günstiger |
| Akzeptable Wahl (Nutzen ≤ 0.02 unter dem besten) | 66.7 % | 93.3 % | Learned günstiger |
| Verifikationserfolg (simuliert) | 60.0 % | 86.7 % | Learned günstiger |
| Ø Quality Score (simuliert) | 0.721 | 0.857 | Learned günstiger |
| Ø Latenz in s (simuliert) | 60.3 | 73.9 | Rule günstiger |
| Ø Nutzen (Qualität − Latenzkosten) | 0.685 | 0.825 | Learned günstiger |
| Kritische Capability-Verletzungen | 1 | 0 | Learned günstiger |
| Unberechtigte Routing-Fehlschläge | 0 | 0 | gleich |
| Fallback-Rate (Wahl durch Ausfall geändert) | 0.0 % | 0.0 % | gleich |
| Wächter: verworfene ungültige Vorschläge | 0 | 0 | gleich |
| Learned Ranking nicht nutzbar (Regelrangfolge) | 0 | 0 | gleich |

### Szenario `degraded` (46 Aufgaben)

| Kennzahl | Rule | Learned | Einordnung (Simulation) |
|---|---|---|---|
| Routing-Genauigkeit (= bestes Modell laut Simulation) | 66.7 % | 42.2 % | Rule günstiger |
| Akzeptable Wahl (Nutzen ≤ 0.02 unter dem besten) | 71.1 % | 93.3 % | Learned günstiger |
| Verifikationserfolg (simuliert) | 55.6 % | 80.0 % | Learned günstiger |
| Ø Quality Score (simuliert) | 0.705 | 0.834 | Learned günstiger |
| Ø Latenz in s (simuliert) | 60.3 | 86.5 | Rule günstiger |
| Ø Nutzen (Qualität − Latenzkosten) | 0.668 | 0.794 | Learned günstiger |
| Kritische Capability-Verletzungen | 1 | 0 | Learned günstiger |
| Unberechtigte Routing-Fehlschläge | 0 | 0 | gleich |
| Fallback-Rate (Wahl durch Ausfall geändert) | 24.4 % | 44.4 % | Rule günstiger |
| Wächter: verworfene ungültige Vorschläge | 0 | 0 | gleich |
| Learned Ranking nicht nutzbar (Regelrangfolge) | 0 | 0 | gleich |

## Routing-Evaluator (Test-Split, gleiche Bewertung wie die Baseline)

| Kennzahl | Rule/all_available | Rule/degraded | Learned/all_available | Learned/degraded |
|---|---|---|---|---|
| Routing-Score (gewichtet) | 0.847 | 0.853 | 0.923 | 0.929 |
| Kritische Fehler | 1 | 1 | 0 | 0 |
| Schwere Fehler (zu schwach) | 12 | 11 | 1 | 0 |
| Überdimensioniert | 2 | 2 | 17 | 17 |
| Nicht verfügbares Modell gewählt | 0 | 0 | 0 | 0 |
| Failover-Sicherheit | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Routing-Latenz Ø (ms, gemessen) | 0.49 | 0.45 | 1.77 | 1.44 |

Kategorie- und Komplexitätsgenauigkeit sind für beide Router identisch (gleicher Klassifikator); der Learned Router ändert nur die Rangfolge.

## Unterschiedliche Entscheidungen (Szenario `all_available`)

| Aufgabe | Rule → Nutzen | Learned → Nutzen | bestes laut Simulation |
|---|---|---|---|
| fast-002 | fast-small → 0.899 | reasoner-large → 0.895 | fast-small |
| fast-005 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| fast-006 | fast-small → 0.899 | reasoner-large → 0.895 | fast-small |
| fast-009 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| fast-014 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| fast-018 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| fast-021 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| fast-022 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| gen-002 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| gen-005 | fast-small → 0.899 | omni-large → 0.895 | fast-small |
| gen-009 | fast-small → 0.899 | omni-large → 0.895 | fast-small |
| gen-011 | coder-large → 0.898 | omni-large → 0.895 | fast-small |
| gen-012 | fast-small → 0.899 | omni-large → 0.895 | fast-small |
| gen-015 | fast-small → 0.899 | coder-large → 0.898 | fast-small |
| gen-018 | fast-small → 0.899 | omni-large → 0.895 | fast-small |
| gen-021 | general-mid → 0.898 | omni-large → 0.895 | fast-small |
| code-003 | fast-small → 0.193 | coder-large → 0.882 | coder-large |
| code-018 | general-mid → 0.892 | reasoner-large → 0.880 | coder-large, general-mid |
| code-030 | reasoner-large → 0.460 | coder-large → 0.484 | coder-large, general-mid |
| reas-004 | fast-small → 0.497 | coder-large → 0.893 | coder-large, general-mid |
| reas-005 | fast-small → 0.193 | omni-large → 0.860 | omni-large, reasoner-large |
| reas-008 | fast-small → 0.193 | omni-large → 0.860 | omni-large, reasoner-large |
| reas-009 | fast-small → 0.193 | reasoner-large → 0.858 | omni-large, reasoner-large |
| reas-010 | fast-small → 0.497 | omni-large → 0.883 | coder-large, general-mid |
| reas-014 | general-mid → 0.484 | omni-large → 0.860 | omni-large, reasoner-large |
| reas-024 | fast-small → 0.497 | omni-large → 0.883 | coder-large, general-mid |
| long-015 | longctx-mid → 0.400 | coder-large → 0.861 | coder-large |
| vis-002 | vision-mid → 0.898 | omni-large → 0.895 | vision-mid |
| vis-004 | vision-mid → 0.493 | omni-large → 0.883 | omni-large |
| vis-013 | vision-mid → -0.007 | omni-large → 0.883 | omni-large |
| multi-006 | general-mid → 0.484 | omni-large → 0.860 | omni-large, reasoner-large |
| multi-008 | coder-large → 0.484 | omni-large → 0.460 | coder-large, general-mid |

## Einflussreichste Merkmale (standardisierte Gewichte)

| Merkmal | Gewicht |
|---|---|
| `m_coding` | +0.988 |
| `x_confidence*cap_gap` | +0.894 |
| `x_cat_fast*coding` | +0.844 |
| `m_cap_gap` | +0.795 |
| `m_relevant_cap` | +0.745 |
| `hist_cat_count` | +0.713 |
| `x_reasoning_signals*speed` | -0.699 |
| `x_complexity*coding` | +0.624 |
| `x_words*reasoning` | -0.612 |
| `x_cat_reasoning*coding` | +0.572 |
| `x_multi_step*reasoning` | -0.553 |
| `x_code_signals*reasoning` | -0.534 |

## Bewertung

Alle Aussagen gelten nur für die **Simulation** auf dem Test-Split (46 Aufgaben, nie im
Training). Hyperparameter wurden vor dem ersten Lauf festgelegt und danach nicht verändert –
auch nicht nach Sichtung der Testergebnisse.

**Nachgewiesen (in dieser Simulation):**

* Weniger Fehlgriffe nach unten: kritische Capability-Verletzungen 1 → 0 (vis-013: Screenshot +
  CSS landet beim Learned Router auf dem multimodalen Modell mit Coding-Fähigkeit), schwere
  Fehler „zu schwaches Modell“ 12 → 1, simulierter Verifikationserfolg 60 % → 87 %
  (`all_available`) bzw. 56 % → 80 % (`degraded`).
* Die harten Regeln hielten: 0 nicht verfügbare Modelle, 0 unberechtigte Fehlschläge,
  0 verworfene Vorschläge des Wächters (der Ranker bekommt nur gültige Kandidaten zu sehen).

**Nicht besser bzw. schlechter:**

* Exakte Routing-Genauigkeit sinkt (62 % → 49 %, `degraded` 67 % → 42 %): Der Learned Router
  überdimensioniert systematisch (17 statt 2 Fälle) – FAST- und GENERAL-Aufgaben gehen an große,
  langsamere Modelle. Simulierte Latenz +14 s bzw. +26 s im Mittel.
* Höhere Fallback-Rate im Ausfallszenario (24 % → 44 %): Er bevorzugt genau die großen Modelle,
  die in `degraded` ausfallen.
* Routing-Zeit 0.5 → 2–3 ms (Merkmalsberechnung) – für lokale Inferenz vernachlässigbar, aber
  messbar höher.
* Paargenauigkeit 87 % (Training) vs. 69 % (Test): deutliche Überanpassung bei 103
  Trainingsaufgaben.

**Ursachen (aus Gewichten und Entscheidungen abgeleitet):**

1. *Konstruktionsfehler, im Lauf entdeckt:* Paare entstehen erst ab Nutzenabstand 0.01
   (`pair_margin`), Latenzunterschiede gleich guter Modelle kosten aber nur 0.001–0.004. Der
   Ranker sieht daher nie ein Paar „schneller ist besser“ und lernt nur „stärker ist sicherer“.
   Korrektur für den nächsten Lauf: `pair_margin` unter die Latenzauflösung senken bzw. Latenz
   stärker gewichten – Auswahl per Kreuzvalidierung **auf dem Trainings-Split**.
2. Der Klassifikator schätzt Komplexität oft zu niedrig (Baseline); der Ranker kompensiert das,
   indem er Textmerkmale (Länge, Zahlen, Reasoning-/Code-Signale) mit Modellstärke verknüpft –
   und übertreibt dabei bei echten FAST-Aufgaben (`x_cat_fast*coding` +0.84).

**Gesamturteil:** Der Learned Router ist in dieser Simulation **sicherer** (weniger zu schwache
und ungeeignete Modelle), aber **langsamer und weniger präzise** in der Wahl des *passenden*
Modells. Ein „besser“ im Sinne der Zielfunktion (bestes Ergebnis bei vertretbarer Latenz) ist
**nicht** nachgewiesen. Er sollte nicht als Standard-Router aktiviert werden, bevor (a) der
Konstruktionsfehler behoben, (b) per Kreuzvalidierung auf dem Training neu bewertet und
(c) mit echten Routing-Log-Daten bestätigt ist.
