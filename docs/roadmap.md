# NOVA – Roadmap

Status: Entwurf v0.1 · Stand: 2026-10-08

Grundsatz: Jede Phase endet mit **lauffähigem, getestetem Code** und messbaren
Abnahmekriterien. Keine Phase beginnt, bevor die vorherige abgenommen ist.
Reihenfolge ist so gewählt, dass ab Phase 5 ein nutzbarer Agent existiert und jede
spätere Phase ihn messbar verbessert.

Arbeitsablauf pro Änderung: Zustand prüfen → Architektur prüfen → begründen →
implementieren → Tests → Fehler beheben → dokumentieren.

---

## Phase 0 – Analyse & Architektur ✅ (dieser Schritt)

- Dokumente: `architecture.md`, `roadmap.md`, `model-strategy.md`, `security.md`, `evaluation.md`
- Verzeichnisstruktur angelegt
- **Abnahme:** Dokumente reviewed, offene Fragen beantwortet (Hardware, OS, ADR-001)

## Phase 1 – Foundation: Projekt-Setup & Kern-Verträge

- `pyproject.toml`, `uv`-Lockfile, `ruff`, `mypy --strict`, `pytest`, `pre-commit` (inkl. Secret-Scan)
- `core/`: Datentypen, alle Interfaces (§6 Architektur), Fehler-Hierarchie
- `core/config`: TOML + Env-Overrides + Validierung
- `core/logging` + Tracing (JSONL, trace/span-IDs, Secret-Redaction)
- Test-Doubles in `tests/`: `ScriptedModelProvider`, `InMemoryMemoryStore` (als Referenz-Impl. für Vertragstests)
- Architekturtest für Importregeln
- **Abnahme:** `ruff`, `mypy`, `pytest` grün; Config lädt; Trace wird geschrieben; Importregeln erzwungen

## Phase 2 – Model Provider

> **Stand 2026-10-08: Kern umgesetzt** (vorgezogen vor Phase 1): `models/` mit
> `ModelProvider`, `OpenAICompatibleProvider`, `ModelRegistry` inkl. `find_best_for`,
> `InferenceEngine` (Timeout, Fallback), `ModelHealthChecker`, 116 Unit-Tests.
> Offen: Streaming, Token-Zählung, nativer Ollama-Provider, Smoke-Test gegen echte Runtime.

- `OpenAICompatibleProvider` (chat, stream, tools, JSON-Schema-Output, health)
- `OllamaProvider` (native API, Embeddings, Modellliste)
- Robuste Fehlerbehandlung: Timeouts, Retries mit Backoff, `ContextOverflow`, ungültiges JSON
- Modell-Registry aus Config, Capability-Erkennung
- Tests über `httpx.MockTransport` mit realistischen Antwort-Fixtures; Integrationstest gegen kleines lokales Modell
- **Abnahme:** Vertragstests für beide Provider grün; manueller Smoke-Test gegen echte Runtime dokumentiert

## Phase 3 – Model Router

- Regelbasierter Klassifikator (Aufgabe → Rolle), Auflösung (Rolle → Modell) mit Capability-Filter und Fallback-Kette
- Routing-Entscheidungen im Trace
- Gelabelter Router-Testdatensatz (≥ 100 Beispiele) in `evaluation/datasets/`
- **Abnahme:** Router-Genauigkeit gemessen und dokumentiert; Fallback greift bei ausgefallenem Modell

## Phase 4 – Tool-System

> **Stand 2026-10-08: umgesetzt** (`tools/`): 9 Tools (Dateisystem, Terminal, Git), Policy,
> Bestätigung, Audit-Log, strukturierte Ergebnisse, Begründungspflicht. Offen: echte Sandbox,
> Policy-Konfiguration aus `config/`, Bestätigungsdialog in CLI/UI.

- `ToolRegistry`: Schema-Validierung, Permissions, Timeouts, Output-Kürzung
- Erste Tools: `read_file`, `list_dir`, `search_text`, `write_file`/`edit_file` (Workspace-gebunden), `run_command` (Allowlist, kein Shell-Interpreter), `remember`/`recall`
- Bestätigungsmechanismus für Tools mit Seiteneffekten
- **Abnahme:** Sicherheitstests aus `security.md` (Path-Traversal, Symlink-Escape, Timeout, Permission-Denial) grün

## Phase 5 – Agent-Kern (erster nutzbarer Agent)

> **Stand 2026-10-08: Kern umgesetzt** (vorgezogen): Agent-Loop ANALYZE→…→FINALIZE mit
> persistentem Task State, Planner (JSON, Retry, Replan), Executor (Tool-Loop), Verifier
> (deterministisch + automatische Checks nach Dateiänderungen), Korrekturkaskade, Budgets.
> Minimal-Tools `read_file`/`write_file`/`list_dir`. Offen: ContextManager mit Token-Budget,
> CLI, Eval-Suite mit echtem Modell.

- Agent-Loop mit Budget, Schleifenerkennung, Abbruchstatus
- `ContextManager` (Budgetierung, Kürzung, Zusammenfassung)
- `Planner` (strukturierter Plan, Replan)
- `Verifier` (Stufen 1–2 aus Architektur §9)
- CLI: `nova run "<aufgabe>"`, `nova chat`
- **Abnahme:** End-to-End-Lauf mit lokalem Modell; erste Eval-Suite (≥ 20 Aufgaben) mit Baseline-Ergebnis

## Phase 6 – Memory & RAG

> **Stand 2026-10-08: Memory umgesetzt** (`memory/`): Working/Session/Project/Long-Term,
> automatisches Routing nach Relevanz, Secret-Schutz, SQLite+FTS5, Abruf mit
> Relevanz×Wichtigkeit×Aktualität, Agent-Integration. Offen: Embeddings/semantische Suche,
> Widerspruchserkennung, RAG-Pipeline.

- `SQLiteMemoryStore` (FTS5, optional Vektoren)
- RAG-Pipeline: Loader (Markdown, Text, PDF, Code), Chunker (struktur-/codebewusst), Embedder, Index
- Hybrid-Retrieval (BM25 + Vektor, RRF), optionaler Cross-Encoder-Reranker
- Entscheidung ADR-006 (Vektorindex) per Messung
- **Abnahme:** Retrieval-Metriken (Recall@k, MRR) auf eigenem Datensatz dokumentiert; Zitate im Agent-Output

## Phase 7 – Coding-Agent

- Repo-Verständnis: Datei-Baum, Symbol-Index (z. B. tree-sitter), relevante-Datei-Suche
- Präzise Edits (Search/Replace-Blöcke bzw. Diffs, validiert vor Anwendung)
- Verifikationsschleife: Tests/Linter ausführen → Fehler lesen → korrigieren
- Ausführung in Sandbox (siehe `security.md`)
- **Abnahme:** Coding-Eval-Suite (eigene Aufgaben mit Tests) – Erfolgsquote gegenüber Phase-5-Baseline messbar verbessert

## Phase 8 – Evaluation-Ausbau

- Eval-Runner, Reports (Markdown/HTML), Vergleich zwischen Läufen, Regression-Gate
- Modellvergleich: dieselbe Suite gegen verschiedene Modelle/Router-Konfigurationen
- Optional: lokaler LLM-Judge mit kalibrierter Rubrik
- **Abnahme:** Ein Befehl erzeugt einen Vergleichsreport zweier Konfigurationen

## Phase 9 – UI

- Lokale API (FastAPI, nur `127.0.0.1`, Token-Auth)
- Web-UI: Chat, Live-Trace (Schritte, Tools, Routing), Tool-Bestätigungen, Memory-Verwaltung, Wissensbasis-Verwaltung
- **Abnahme:** Alle CLI-Funktionen in der UI nutzbar; Bestätigungsdialoge funktionieren

## Phase 10+ – Ausbau (nach Bedarf, datengetrieben)

- Vision-Rolle produktiv, Bild-Inputs in Tools
- Lernender Router (aus Traces + Eval-Ergebnissen)
- Mehr-Agenten-Muster (Planner/Worker/Reviewer) – nur, wenn Evals einen Gewinn zeigen
- Web-Recherche-Tool (bewusst freigegeben, Netzwerk-Permission)
- Fine-Tuning / LoRA auf eigenen Traces – **frühestens**, wenn genug kuratierte Daten und eine stabile Eval-Suite existieren

---

## Bewusst ausgeschlossen (vorerst)

- Download großer Modelle, Fine-Tuning, verteilte Infrastruktur, Cloud-APIs im Kern.

## Offene Fragen, die die Roadmap beeinflussen

1. Zielhardware (VRAM/RAM) → bestimmt, ob CODING/REASONING getrennte Modelle sein können
   oder ein Modell mehrere Rollen abdeckt (Speicherdruck durch gleichzeitig geladene Modelle).
2. Betriebssystem → Sandbox-Technik in Phase 4/7.
3. ADR-001 (Paket-Namespace) → muss vor Phase 1 entschieden sein.
