# evaluation/datasets

## `routing_tasks.jsonl` – Routing-Evaluationsdatensatz

150 realistische Aufgaben (DE/EN), mit denen gemessen wird, ob der Router die richtige
Modellklasse wählt. Eine Zeile = ein JSON-Objekt. Auswertung:
`python -m scripts.evaluate_routing` → `evaluation/reports/routing_baseline.md`.

| Set | Anzahl | Inhalt |
|---|---|---|
| FAST | 25 | einfache Berechnungen, kurze Fakten, Umwandlungen, einfache Klassifikation |
| GENERAL | 25 | Schreiben, Erklären, Zusammenfassen, Beraten, Übersetzen von Absätzen |
| CODING | 30 | Debugging, Refactoring, Architektur, Tests, mehrere Dateien, Fehlersuche, Codeverständnis |
| REASONING | 25 | Beweise, mehrstufige Logik, widersprüchliche Informationen, Planung, komplexe Analyse |
| LONG_CONTEXT | 20 | große Dokumente, mehrere Quellen, irrelevante Abschnitte (Distraktoren), Querverbindungen |
| VISION | 15 | Aufgaben, die ohne das Bild unlösbar sind (Belege, Screenshots, Diagramme, Fotos) |
| MULTI_STEP | 10 | mehrere voneinander abhängige Schritte mit Tools |

### Felder

| Feld | Bedeutung |
|---|---|
| `id` | eindeutig, `<set>-<nr>` |
| `set` | Aufgabenset (Tabelle oben) – Gliederung des Datensatzes |
| `prompt` | Aufgabentext, wie ihn der Nutzer schreibt |
| `expected_category` | richtige Router-Kategorie: `FAST`, `GENERAL`, `CODING`, `REASONING`, `VISION`, `LONG_CONTEXT` |
| `expected_secondary` | nur bei `LONG_CONTEXT`: inhaltliche Kategorie |
| `expected_complexity` | `LOW`, `MEDIUM`, `HIGH` (Regeln unten) |
| `required_capabilities` | harte Anforderungen: `coding`, `reasoning`, `vision`, `tool_calling`, `long_context` |
| `requires_tools` / `requires_vision` | Tools werden angeboten bzw. Bild hängt an (zur Laufzeit bekannt) |
| `estimated_context` | `small` < 2k · `medium` < 24k · `large` < 100k · `very_large` ≥ 100k Tokens |
| `context_tokens` | Umfang des angehängten Materials (Dokumente, Code, Logs) in Tokens, ohne Prompt |
| `difficulty` | wie schwer die **richtige Einordnung** ist: `easy` eindeutige Signale · `medium` Signale gemischt · `hard` irreführende oder fehlende Signalwörter |
| `expected_routable` | `false`, wenn kein Modell der Flotte geeignet ist – richtig ist dann ein begründeter Fehlschlag |
| `language`, `tags`, `note` | Sprache, Abdeckungsmerkmale (z. B. `debugging`, `proof`, `distractors`), Begründung bei Grenzfällen |

### Warum MULTI_STEP keine Router-Kategorie ist

Der Router wählt ein *Modell*; „mehrstufig“ ist eine Eigenschaft der Ausführung (Agent-Loop).
MULTI_STEP-Aufgaben erwarten deshalb ihre inhaltliche Kategorie (meist CODING), Komplexität
HIGH (bzw. MEDIUM bei einfachen Schritten) und Tool-Calling als harte Anforderung.

### Label-Regeln

* **Kategorie** nach dem, was die Aufgabe *braucht*, nicht nach Signalwörtern: Bild hängt an
  → VISION; Material > 24k Tokens → LONG_CONTEXT; Quelltext lesen/schreiben/entwerfen → CODING
  (auch Software-Architektur und DB-Schemata); mehrstufiges Schließen, Beweise, Planung mit
  Abhängigkeiten, Abwägen widersprüchlicher Angaben → REASONING; triviale Einzelfakten,
  Rechnungen, Umwandlungen → FAST; alles andere → GENERAL.
* **Komplexität**: LOW = ein Schritt, Allgemeinwissen · MEDIUM = mehrere Schritte oder
  Fachwissen, begrenzter Umfang · HIGH = abhängige Schritte, großer Umfang,
  Entwurfsentscheidungen, Debugging mit unbekannter Ursache oder anspruchsvoller Beweis.
* **Eignung**: Eine geforderte Fähigkeit ist erfüllt, wenn das Modell sie überhaupt hat
  (Stufe > none bzw. Tool-Calling) und das Kontextfenster für Material + Prompt + 2048 Tokens
  Antwortreserve reicht. Verstöße sind **kritische Fehler**.

### Herkunft und Grenzen

* Aufgaben selbst verfasst (keine Kopien aus öffentlichen Benchmarks), damit sie nicht in
  Trainingsdaten stecken. Labels nach den Regeln oben **vor** dem ersten Lauf vergeben und
  danach nicht an die Router-Ergebnisse angepasst. Offenlegung: Beim Verfassen war der
  Regel-Klassifikator bekannt; `difficulty: hard` markiert bewusst Fälle, die seine Signalwörter
  ausnutzen oder umgehen.
* Ein Labeler, keine Zweitannotation – Grenzfälle sind mit `note` begründet. Vor einem
  lernenden Router: Ausbau und Zweitlabel (siehe Bericht, Abschnitt 7).
* Angehängtes Material ist nicht enthalten, nur sein Umfang (`context_tokens`) – genau das,
  was der Router zur Laufzeit sieht.

## `routing_fleet.toml` – Evaluationsflotte

Sieben fiktive Modellklassen (klein/schnell, Allround, Coder, Denker, Vision, Langkontext,
multimodal) mit konfigurierten Fähigkeiten und zwei Szenarien (`all_available`, `degraded`).
Fiktiv, damit die Baseline reproduzierbar und hardwareunabhängig ist; keine Messwerte.
