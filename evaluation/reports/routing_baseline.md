# NOVA – Routing-Baseline

Erzeugt: 2026-10-08 · Router: `RuleBasedRouter + RuleBasedClassifier` · Datensatz: `routing_tasks.jsonl` (sha256 a2d6cecacf4f, 150 Aufgaben) · Flotte: `routing_fleet.toml` (sha256 47c585ba42ed)

Reproduzieren: `python -m scripts.evaluate_routing` – alle Werte außer der Latenz sind deterministisch.

Verteilung: FAST 25, GENERAL 25, CODING 30, REASONING 25, LONG_CONTEXT 20, VISION 15, MULTI_STEP 10

## 1. Gesamtergebnis

| Kennzahl | all_available | degraded |
|---|---|---|
| Routing-Score (gewichtet, 1 = fehlerfrei) | 0.837 | 0.843 |
| Kategorie-Genauigkeit | 72.0 % | 72.0 % |
| Sekundärkategorie (LONG_CONTEXT) | 40.0 % | 40.0 % |
| Komplexitäts-Genauigkeit | 46.7 % | 46.7 % |
| **Kritische Fehler** | **2** | **2** |
| – Capability-Verletzungen | 2 | 2 |
| – nicht verfügbares Modell gewählt | 0 | 0 |
| – unberechtigte Routing-Fehlschläge | 0 | 0 |
| Korrekte Ablehnungen (kein Modell geeignet) | 2 | 2 |
| Schwere Fehler (Modell zu schwach) | 47 | 44 |
| Überdimensioniert (einfache Aufgabe, langsames Modell) | 5 | 5 |
| Fallback-Rate (Wahl durch Ausfall geändert) | 0.0 % | 20.3 % |
| Fallback-Abdeckung (Ersatzmodell vorhanden) | 99.3 % | 87.2 % |
| Failover-Sicherheit (Ersatzwahl geeignet) | 100.0 % | 100.0 % |
| ECE (Confidence-Kalibrierung, 0 = ideal) | 0.045 | 0.045 |
| Brier-Score | 0.155 | 0.155 |
| Überzeugte Fehler (Confidence ≥ 0.85) | 2 | 2 |
| Routing-Latenz Ø / p95 | 0.47 / 0.76 ms | 0.48 / 0.80 ms |

Latenz gemessen auf: CPU Intel(R) Xeon(R) Processor @ 2.80GHz (4C/4T) · RAM 15.7 GB · GPU keine dedizierte GPU erkannt · Beschleuniger: cpu-avx512f (einziger hardwareabhängiger Wert).

## 2. Ergebnis pro Aufgabenset (all_available)

| Set | n | Kategorie | Komplexität | kritisch | Score |
|---|---|---|---|---|---|
| FAST | 25 | 92.0 % | 96.0 % | 0 | 0.988 |
| GENERAL | 25 | 52.0 % | 64.0 % | 0 | 0.856 |
| CODING | 30 | 76.7 % | 36.7 % | 0 | 0.863 |
| REASONING | 25 | 32.0 % | 8.0 % | 0 | 0.610 |
| LONG_CONTEXT | 20 | 100.0 % | 30.0 % | 0 | 0.935 |
| VISION | 15 | 100.0 % | 40.0 % | 2 | 0.763 |
| MULTI_STEP | 10 | 60.0 % | 50.0 % | 0 | 0.815 |

Kategorie-Genauigkeit nach Schwierigkeit: easy 90.4 %, medium 64.0 %, hard 5.9 %

## 3. Kritische Fehler

### Szenario `all_available` (2)

| ID | erwartet | erkannt | gewählt | Befund |
|---|---|---|---|---|
| vis-003 | VISION | VISION | vision-mid | ungeeignet: vision-mid – keine coding-Fähigkeit |
| vis-013 | VISION | VISION | vision-mid | ungeeignet: vision-mid – keine coding-Fähigkeit |

### Szenario `degraded` (2)

| ID | erwartet | erkannt | gewählt | Befund |
|---|---|---|---|---|
| vis-003 | VISION | VISION | vision-mid | ungeeignet: vision-mid – keine coding-Fähigkeit |
| vis-013 | VISION | VISION | vision-mid | ungeeignet: vision-mid – keine coding-Fähigkeit |

## 4. Häufigste Fehlklassifikationen

| erwartet → erkannt | Anzahl | Beispiele |
|---|---|---|
| REASONING → GENERAL | 16 | reas-002, reas-005, reas-006, reas-007, reas-008 |
| GENERAL → FAST | 11 | gen-002, gen-007, gen-008, gen-010, gen-014 |
| CODING → GENERAL | 5 | code-004, code-021, code-027, code-029, multi-008 |
| CODING → REASONING | 3 | code-018, code-030, multi-001 |
| GENERAL → REASONING | 2 | gen-021, multi-005 |
| FAST → REASONING | 1 | fast-003 |
| FAST → GENERAL | 1 | fast-024 |
| CODING → FAST | 1 | code-020 |
| REASONING → FAST | 1 | reas-004 |
| REASONING → CODING | 1 | multi-010 |

Komplexität (erwartet → erkannt): HIGH→MEDIUM: 35, MEDIUM→LOW: 27, HIGH→LOW: 10, LOW→MEDIUM: 5, MEDIUM→HIGH: 3

## 5. Problematische Aufgaben

Höchste Strafpunkte im Szenario `all_available` (kritisch 10 · schwer 3 · Kategorie 1 · Komplexität 0.5 · überdimensioniert 0.5):

| ID | Strafe | Befunde | Prompt (Anfang) |
|---|---|---|---|
| vis-003 | 10 | ungeeignet: vision-mid – keine coding-Fähigkeit; MEDIUM erwartet, LOW erkannt | Wandle diese fotografierte Whiteboard-Skizze eines Login-Formulars in … |
| vis-013 | 10 | ungeeignet: vision-mid – keine coding-Fähigkeit; MEDIUM erwartet, LOW erkannt | The attached screenshot shows our dashboard with a broken layout on mo… |
| code-021 | 4.5 | fast-small zu schwach (Stufe 1 < 3 für HIGH; verfügbar bis 2); CODING erwartet, GENERAL erkannt; HIGH erwartet, LOW erkannt | Design the database schema for a multi-tenant SaaS invoicing app (tena… |
| gen-007 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 2.5); GENERAL erwartet, FAST erkannt; MEDIUM erwartet, LOW erkannt | Was sind die Vor- und Nachteile von Wärmepumpen in einem unsanierten A… |
| gen-008 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 2.5); GENERAL erwartet, FAST erkannt; MEDIUM erwartet, LOW erkannt | Erstelle einen Wochenplan für vegetarische Abendessen für eine vierköp… |
| gen-010 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 2.5); GENERAL erwartet, FAST erkannt; MEDIUM erwartet, LOW erkannt | Explain the difference between a Roth IRA and a traditional IRA in sim… |
| gen-024 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 2.5); GENERAL erwartet, FAST erkannt; MEDIUM erwartet, LOW erkannt | Summarize the main arguments for and against a four-day work week.… |
| multi-005 | 4.5 | general-mid zu schwach (Stufe 2 < 3 für HIGH; verfügbar bis 2.5); GENERAL erwartet, REASONING erkannt; HIGH erwartet, MEDIUM erkannt | Plane eine dreitägige Reise nach Lissabon: Suche zuerst Flüge im Budge… |
| multi-010 | 4.5 | coder-large zu schwach (Stufe 2 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, CODING erkannt; HIGH erwartet, MEDIUM erkannt | Berechne aus experiment.csv Mittelwert und Standardabweichung pro Vers… |
| reas-002 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; MEDIUM erwartet, LOW erkannt | Zeige per vollständiger Induktion, dass für alle n ≥ 1 gilt: 1³ + 2³ +… |
| reas-004 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, FAST erkannt; MEDIUM erwartet, LOW erkannt | Zeige, dass jeder Baum mit n Knoten genau n−1 Kanten hat.… |
| reas-005 | 4.5 | fast-small zu schwach (Stufe 1 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; HIGH erwartet, LOW erkannt | Fünf Personen sitzen in einer Reihe. Anna sitzt nicht am Rand. Ben sit… |
| reas-006 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; MEDIUM erwartet, LOW erkannt | A bat and a ball cost $1.10 in total. The bat costs $1.00 more than th… |
| reas-007 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; MEDIUM erwartet, LOW erkannt | Auf einer Insel lügt jeder Bewohner immer oder sagt immer die Wahrheit… |
| reas-008 | 4.5 | fast-small zu schwach (Stufe 1 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; HIGH erwartet, LOW erkannt | Drei Zeugen beschreiben einen Unfall. Zeuge 1: Ein rotes Auto kam von … |
| reas-009 | 4.5 | fast-small zu schwach (Stufe 1 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; HIGH erwartet, LOW erkannt | Zwei Studien kommen zu gegensätzlichen Ergebnissen zur Produktivität i… |
| reas-010 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; MEDIUM erwartet, LOW erkannt | My calendar says the meeting is Tuesday 3pm CET, the invite email says… |
| reas-012 | 4.5 | fast-small zu schwach (Stufe 1 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; HIGH erwartet, LOW erkannt | We have four developers and six features: A (5 days), B (3 days, needs… |
| reas-013 | 4.5 | coder-large zu schwach (Stufe 2 < 3 für HIGH; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; HIGH erwartet, MEDIUM erkannt | Ich habe 25.000 € Erspartes und 8.000 € Konsumkredit zu 9 % Zinsen. So… |
| reas-017 | 4.5 | fast-small zu schwach (Stufe 1 < 2 für MEDIUM; verfügbar bis 3); REASONING erwartet, GENERAL erkannt; MEDIUM erwartet, LOW erkannt | Ein Zug fährt um 9:15 Uhr in A mit 80 km/h los, ein zweiter um 9:45 Uh… |

## 6. Confidence / Sicherheit

| Confidence | n | Ø Confidence | Treffer |
|---|---|---|---|
| 0.00–0.60 | 72 | 0.53 | 51.4 % |
| 0.60–0.80 | 25 | 0.68 | 76.0 % |
| 0.80–0.90 | 10 | 0.83 | 90.0 % |
| 0.90–1.00 | 43 | 0.93 | 100.0 % |

ECE 0.045, Brier 0.155. Überzeugte Fehler (Confidence ≥ 0.85, falsche Kategorie oder kritisch): 2

* vis-003: 0.95 → VISION (erwartet VISION) **kritisch**
* vis-013: 0.95 → VISION (erwartet VISION) **kritisch**

## 7. Verbesserungsvorschläge

1. **Kritisch – Zusatzfähigkeiten bei VISION ignoriert (vis-003, vis-013):** Bei Bildaufgaben rankt der Router nur nach Vision-Fähigkeit und wählt ein Vision-Modell ohne Coding-Fähigkeit für „Skizze → HTML/CSS“ bzw. „CSS-Fehler im Screenshot“. Der Klassifikator sollte eine *Menge* benötigter Fähigkeiten liefern (z. B. vision + coding), und der Router muss alle als harte Filter anwenden – nicht nur die Hauptkategorie. Gleiches Muster droht bei LONG_CONTEXT + coding.
2. **Komplexität wird systematisch unterschätzt (46.7 % korrekt; 37 MEDIUM/HIGH-Aufgaben als LOW):** Dadurch landen 72 von 150 Aufgaben beim kleinsten Modell; insgesamt 47 Aufgaben bekommen ein zu schwaches Modell, obwohl ein stärkeres geeignetes verfügbar war (schwerer Fehler). Die Punktevergabe hängt an Signalwörtern und Länge; kurze, dichte Aufgaben (Beweise, Logikrätsel, Schema-Design, Zeitplanung) erhalten 0 Punkte. Vorschläge: Dichte-Signale (mehrere Zahlen/Bedingungen, nummerierte Aussagen, „und dann“-Ketten), Untergrenze MEDIUM für erkannte CODING-Aufgaben außer klaren Einzeilern, HIGH bei Entwurfs-/Design-Verben (entwirf, design, schema, migrationspfad).
3. **REASONING-Erkennung nur 32 %:** 16 von 25 Reasoning-Aufgaben werden GENERAL. Fehlende Signale: „zeige (dass)“, „begründe“, „leite … her“, „widersprech*“, „konsistent/consistent“, „Hypothese*“, „optimal*“, „Reihenfolge“, „Annahmen“, „valid/fallacy“, „schedule“, „explain your answer“. Zusätzlich strukturelle Signale: Rechenaufgaben mit mehreren Größen, Bedingungslisten, gegensätzliche Zahlen.
4. **GENERAL → FAST (11 Fälle):** Die Regel „≤ 15 Wörter und LOW“ ist zu grob. Schreib-, Erklär- und Beratungsaufgaben (schreib, erkläre, formuliere, gib Ideen, wie kann ich) sind GENERAL; FAST sollte positive Muster brauchen (Rechnung, Faktfrage, Umwandlung, Ein-Wort-Klassifikation). Bei LOW meist folgenlos, bei MEDIUM (gen-007, gen-008, gen-010, gen-024) ein schwerer Fehler.
5. **CODING ohne Code-Signalwort (code-004, code-021, code-027, code-029):** Architektur-/Schema-/Repository-Aufgaben ohne Wörter wie „code“ oder Dateiendungen werden GENERAL. Ergänzen: schema, datenbank/database, index, middleware, endpoint, regex/regulärer ausdruck, repository-struktur, django/react/express/fastapi/pandas (Framework-Namen sind keine Modellnamen).
6. **LONG_CONTEXT-Inhaltskategorie nur 40 % korrekt:** Die Sekundärkategorie steuert die Rangfolge innerhalb der Langkontext-Modelle. Sie nutzt dieselben schwachen Regeln; profitiert direkt von den Reasoning-/Coding-Ergänzungen oben.
7. **Confidence misst nur die Kategorie, nicht die Eignung:** Kalibrierung der Kategorie ist gut (ECE 0.045; Confidence < 0.6 trifft nur 51 %), aber beide kritischen Fehler hatten Confidence 0.95, weil die Kategorie stimmte und die Modellwahl nicht. Vorschlag: zusätzliche Entscheidungs-Confidence (alle benötigten Fähigkeiten erfüllt? Abstand zum nächsten Kandidaten?) und den Hybrid-Klassifikator ab Confidence < 0.6 einsetzen – genau dort liegen 72 Aufgaben mit 51 % Trefferquote.
8. **Fallback-Abdeckung im Ausfallszenario 87 %:** Für rund jede achte Aufgabe gibt es im Szenario `degraded` kein Ersatzmodell. Das Routing-Log sollte das als Warnung ausweisen („single point of failure“), damit Betrieb und Flottenplanung es sehen.
9. **Datensatz vor einem lernenden Router ausbauen:** 150 Aufgaben, gelabelt von einer Person (bzw. einem Modell) – Labels sind nicht durch einen zweiten Annotator geprüft. Vor einem gelernten Router: ≥ 300 Aufgaben, Zweitlabel mit Übereinstimmungsmaß (Cohen's κ), getrennte Dev-/Test-Splits, damit Regelverbesserungen nicht auf den Testdaten überangepasst werden.

## Methodik

* Der Router erhält nur, was das System zur Laufzeit weiß: Prompt, angehängte Kontextgröße, ob Bilder anhängen und ob Tools angeboten werden. Kategorie und Komplexität schätzt er selbst.
* Eignung (kritisch) wird unabhängig vom Router geprüft: geforderte Fähigkeiten (Stufe > none), Tool-Calling, Kontextfenster ≥ Material + Prompt + 2048.
* Fallback-Rate: Anteil der Aufgaben, deren Wahl sich gegenüber voller Verfügbarkeit ändert. Failover-Sicherheit: erste Wahl wird entfernt, ist die neue Wahl noch geeignet?
* Labels: `evaluation/datasets/README.md`.
