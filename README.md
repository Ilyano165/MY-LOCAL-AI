# NOVA

Vollständig lokale, persönliche KI-Plattform: lokales Open-Weight-Modell + Model Router +
Planung + Tools + Memory + RAG + Context Management + Self-Verification + Evaluation + UI.

**Status:** Phase 0 abgeschlossen; Model-Engine-Layer (`models/`), Agent Core (`agents/`), Tool-System (`tools/`), Memory (`memory/`), Verification Engine (`evaluation/`) Model Router (`router/`), Model-Benchmarking, Learned Router (nicht aktiviert) sowie lokale API
und Web-Oberfläche (`api/`) implementiert und getestet.

## Benutzen

```bash
uv pip install --python .venv/bin/python -e ".[ui]"
.venv/bin/python -m api --config ~/.nova/models.toml   # → http://127.0.0.1:8765
.venv/bin/python -m api --dev                           # ohne Modell (zeigt „No local model available.“)
```

## Entwicklung

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest          # Unit-Tests (ohne Modell)
.venv/bin/ruff check . && .venv/bin/mypy
# Health Check gegen laufende lokale Runtime:
.venv/bin/python -m scripts.model_health --config config/models.toml
# Integrationstests: NOVA_IT_CONFIG=config/models.toml .venv/bin/python -m pytest -m integration
```

## Dokumentation

- [Architektur](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Modellstrategie](docs/model-strategy.md)
- [Sicherheit](docs/security.md)
- [Evaluation](docs/evaluation.md)

## Struktur

| Verzeichnis | Inhalt |
|---|---|
| `api/` | Lokale NOVA API (FastAPI) und Web-Oberfläche |
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
