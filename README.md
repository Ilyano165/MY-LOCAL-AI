# NOVA

Vollständig lokale, persönliche KI-Plattform: lokales Open-Weight-Modell + Model Router +
Planung + Tools + Memory + RAG + Context Management + Self-Verification + Evaluation + UI.

**Status:** Phase 0 – Architektur & Planung. Noch kein ausführbarer Code.

## Dokumentation

- [Architektur](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Modellstrategie](docs/model-strategy.md)
- [Sicherheit](docs/security.md)
- [Evaluation](docs/evaluation.md)

## Struktur

| Verzeichnis | Inhalt |
|---|---|
| `app/` | CLI, lokale API, UI, Verdrahtung |
| `core/` | Interfaces, Datentypen, Fehler, Config, Logging |
| `models/` | ModelProvider-Adapter, Modell-Registry (keine Gewichte) |
| `router/` | Model Router |
| `agents/` | Agenten, Planner, Context Manager, Verifier |
| `tools/` | Tools, Registry, Permissions, Sandbox |
| `memory/` | Langzeitgedächtnis |
| `rag/` | Ingestion & Retrieval |
| `evaluation/` | Evaluatoren, Datensätze, Reports |
| `tests/` | Tests |
| `config/` | Konfiguration (ohne Secrets) |
| `scripts/` | Entwickler-Skripte |
