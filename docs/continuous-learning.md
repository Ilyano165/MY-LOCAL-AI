# NOVA – Research Engine und kontinuierliches Lernen

Stand: 2026-10-10 · Status: Research Engine und Knowledge Store **implementiert** (`research/`);
Lernpipeline (Datensätze, Training) **noch nicht** – siehe §6.

## 1. Grundsatz

Recherche verändert **nie direkt** die Modellgewichte. Es gibt zwei getrennte Wege:

```
RESEARCH → SOURCE VALIDATION → KNOWLEDGE EXTRACTION ─┬─► Knowledge Store (Retrieval)   Weg 1
                                                     │      sofort nutzbar, mit Quellen
                                                     └─► DATASET CANDIDATES              Weg 2
                                                          → QUALITY FILTERING → DEDUPLICATION
                                                          → TRAINING DATASET (versioniert)
                                                          → CONTROLLED TRAINING (explizit)
                                                          → INDEPENDENT EVALUATION
                                                          → MODEL VERSION → APPROVAL
```

## 2. Research Engine (`research/`)

### Arbeitszyklus (kein Dauer-Token-Generieren)

```
plan ─► search ─► fetch ─► assess ─► extract ─► reconcile ─► store ─► checkpoint ─► replan
 ▲                                                                                   │
 └───────────────────────────── bis Zeit-/Quellenbudget erschöpft ───────────────────┘
```

Das Modell wird nur für Planung, Extraktion und Abgleich gebraucht; Suche, Abruf,
Deduplizierung, Bewertung nach Metadaten und Speicherung sind deterministischer Code.
Stundenlange Recherche heißt: viele kurze Zyklen mit Pausen (Rate Limits), nicht ein langer
Generierungslauf.

### Bausteine

| Baustein | Aufgabe |
|---|---|
| Research Planner | Auftrag → Teilfragen, Suchanfragen, Budget (Zeit, Quellen, Requests) |
| Search-Adapter | austauschbar: SearXNG (selbst gehostet, AGPL), Brave Search API, weitere; **ohne konfigurierten Anbieter keine Websuche** |
| Fetcher | robots.txt beachten, eigener User-Agent, Rate Limit je Domain, Größen-/Typlimit, keine Logins, keine Paywall-Umgehung |
| Quellenverwaltung | URL-Normalisierung, Inhalts-Hash, Snapshot mit Abrufzeit, dauerhafte ID |
| Deduplizierung | URL-kanonisch + Inhalts-Hash + Nahdubletten (Shingling) |
| Quellenbewertung | Art (Primärquelle, Doku, Paper, News, Forum), Domain, Datum, Autor, Zitierungen – nachvollziehbare Punktzahl mit Begründung |
| Aussagen-/Belegextraktion | Aussage + wörtliches Zitat + Quellen-ID + Fundstelle |
| Widerspruchserkennung | gleiche Teilfrage, unvereinbare Aussagen → markiert, nicht aufgelöst durch Raten |
| Klassifikation | **belegt** (≥ 2 unabhängige Quellen oder Primärquelle) · **Hypothese** · **Meinung** · **ungeklärt/widersprüchlich** |
| Checkpointing | nach jedem Zyklus: Plan, offene Teilfragen, Quellen, Aussagen (atomar) → Fortsetzen nach Abbruch/Absturz |
| Audit-Log | jede Suche, jeder Abruf (URL, Status, Hash, Zeit), jede Modellentscheidung |
| Bericht | Markdown + JSON: Erkenntnisse nach Klassifikation, Widersprüche, Quellenliste mit IDs, offene Fragen |

### Sicherheit

* Webseiteninhalt ist **nicht vertrauenswürdige Eingabe**. Er wird als Daten in klar
  abgegrenzten Blöcken an das Modell gegeben; Anweisungen darin werden ignoriert und können
  weder Systemregeln, Berechtigungen noch Werkzeugfreigaben ändern. Die Research Engine hat
  **keine** Schreib-/Ausführungswerkzeuge außer ihrem eigenen Speicher.
* Rechtliches: robots.txt, Nutzungsbedingungen des Suchanbieters (z. B. untersagen manche
  Such-APIs das Training mit ihren Ergebnissen – solche Ergebnisse dürfen nur für Weg 1
  genutzt werden; Lizenz je Quelle wird gespeichert), Urheberrecht (nur Auszüge/Zitate
  speichern, Volltexte nur lokal als Snapshot), keine personenbezogenen Massendaten.

## 3. Weg 1 – Wissensverbesserung ohne Retraining

* Knowledge Store (SQLite/FTS5, später Embeddings): Erkenntnis, Klassifikation, Quellen-IDs,
  Gültigkeit (Abrufdatum, „veraltet“-Markierung), Widerspruchsgruppen.
* Retrieval im Chat: relevante Erkenntnisse mit Quellenangaben in den Kontext; Antworten
  zitieren Quellen-IDs.
* Pflege: aktualisieren, als veraltet markieren, löschen (auch kaskadierend aus Datensatz-
  kandidaten, solange noch nicht trainiert).

## 4. Weg 2 – Modellverbesserung durch Training (`training/`, geplant)

| Schritt | Regel |
|---|---|
| Datensatzkandidaten | nur aus **belegten** Erkenntnissen, verifiziert erfolgreichen Agent-Traces oder manuell geprüften Beispielen; Herkunft je Beispiel |
| Qualitätsfilter | deterministisch: Format, Länge, Sprache, Quellenpflicht, keine Secrets/PII, keine widersprüchlichen Paare; optional menschliche Prüfung |
| Deduplizierung | exakt + Nahdubletten, auch gegen ältere Versionen |
| Versionierung | `datasets/<name>/<semver>/` mit `manifest.json` (Hashes je Datei, Herkunft, Filterstatistik) – unveränderlich |
| Splits | train/val/test nach **Gruppen** (gleiche Quelle/gleiches Thema nie über Splits verteilt); festes Testset pro Fähigkeit |
| Leck-Prüfung | n-Gramm-/Hash-Abgleich Testset ↔ Trainingsdaten und ↔ öffentliche Benchmark-Testsets |
| Training | nur explizit (`nova train --dataset … --base …`), Ressourcenbudget, reproduzierbar (Seed, Config-Hash, Code-Commit, Basis-Revision) |
| Evaluation | Basis vs. Checkpoint auf identischen Suites; deterministische Checks und Referenzantworten; Regressionserkennung |
| Freigabe | Gate aus `model-development.md` §6, menschliche Bestätigung |

**Ein Modell bewertet seine eigene Verbesserung nie allein.** LLM-als-Richter nur ergänzend,
mit einem anderen Modell als dem trainierten, und nie als alleiniges Freigabekriterium.

## 5. Betrieb

* Chat-Anfragen starten weder Research noch Training. Research startet auf Auftrag oder
  nach konfiguriertem Zeitplan; Training nur explizit.
* Ressourcensteuerung (Resource Governor, geplant): Prioritäten Chat > Agent > Research >
  Training; Hintergrundjobs pausieren, wenn der Chat den einzigen Modell-Slot braucht.
* Jeder Lauf hat eine ID, Status, Budget, Abbruch, Fortsetzen und ein Audit-Log.

## 6. Status

| Teil | Status | Nachweis |
|---|---|---|
| Architektur (dieses Dokument) | ✅ | – |
| Research Engine: Planer, Such-Adapter (SearXNG, Brave), höflicher Abruf (robots.txt, Rate Limit, Größen-/Typlimit, keine Paywall-Umgehung), Deduplizierung (URL + Hash + Nahdubletten), Quellenbewertung mit Begründung, Extraktion mit **wörtlicher Zitatprüfung**, Gleich-/Widerspruchserkennung, Klassifikation, Budget, Checkpoint, Abbruch/Pause/Fortsetzen, Audit, Bericht | ✅ implementiert + getestet | `tests/research/` (simuliertes Web inkl. Prompt-Injection), Browser-E2E, echter Lauf gegen erreichbare Webseiten (§7) |
| Bedienung: UI (Research-Bereich), CLI `nova research`, API `/research/*` | ✅ | Tests, E2E |
| Knowledge Store (SQLite/FTS5): speichern, suchen, veraltet markieren, löschen | ✅ | Tests |
| Retrieval der Erkenntnisse **im Chat** mit Quellenangaben | ⬜ nächster Schritt |
| Datensatzpipeline, Splits, Leak-Check | ⬜ |
| Trainings-Adapter (TRL/PEFT), Hardware-Probelauf | ⬜ |
| Eval-Vergleich Basis vs. Checkpoint, Freigabe-Gate | ⬜ (Evaluationsrahmen vorhanden) |
| Zeitgesteuerte Recherche („nach Plan“) | ⬜ (heute nur auf Auftrag) |
| Zentrale Ressourcensteuerung | ⬜ heute: höchstens 1 Research-Lauf gleichzeitig; Modellaufrufe teilen sich die Engine mit dem Chat |

## 7. Bedienung und tatsächlich ausgeführte Läufe

```powershell
nova research config --provider searxng --url http://127.0.0.1:8888   # oder: --provider brave (Key in NOVA_BRAVE_API_KEY)
nova research start "Frage …" --minutes 120 --max-sources 40 [--url https://… ]
nova research list | show <id> | report <id> | resume <id>
nova research knowledge "suchbegriff"
```

Strg+C pausiert einen Lauf (fortsetzbar). In der UI: Seitenleiste → Lupe → Research.

Ehrlicher Stand der ausgeführten Läufe (2026-10-10):
* **Mit Modell** (Planung, Extraktion, Abgleich): nur gegen ein **simuliertes** Web mit einem
  skriptbaren Test-Modell. Ein echter Lauf mit Sprachmodell braucht ein lokales Modell auf dem
  PC des Nutzers – in der Entwicklungsumgebung gibt es keins, und deren Netzwerkrichtlinie
  sperrt u. a. Hugging Face.
* **Echtes Web, ohne Modell und ohne Suchanbieter:** Start-URLs auf pypi.org und
  raw.githubusercontent.com wurden echt abgerufen (robots.txt geprüft, Snapshots, Bewertung,
  Tracking-Variante als bekannt erkannt); gesperrte Hosts wurden korrekt als „robots.txt not
  reachable – not fetched“ protokolliert.
* **Ohne Suchanbieter** sucht NOVA nicht im Web. Für echte Websuche braucht es eine SearXNG-
  Instanz oder einen Brave-API-Schlüssel (Bedingungen beachten: Brave untersagt Training mit
  den Ergebnissen – solche Quellen sind für Weg 2 auszuschließen; die Herkunft wird gespeichert).
