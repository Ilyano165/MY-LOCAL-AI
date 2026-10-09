# NOVA

Vollständig lokale, persönliche KI-Plattform: lokales Open-Weight-Modell + Model Router +
Planung + Tools + Memory + RAG + Context Management + Self-Verification + Evaluation + UI.

**Status:** Phase 0 abgeschlossen; Model-Engine-Layer (`models/`), Agent Core (`agents/`), Tool-System (`tools/`), Memory (`memory/`), Verification Engine (`evaluation/`) Model Router (`router/`), Model-Benchmarking, Learned Router (nicht aktiviert) sowie lokale API
und Web-Oberfläche (`api/`), Integrations-API (`/api/v1`, OpenAI-kompatibel `/v1`),
Kommandozeile `nova`, Hintergrunddienst, Modell-Setup und Windows-Installer (MSI, CI-Pipeline)
implementiert und getestet. Was davon wo geprüft ist: [Testbericht](docs/test-report.md).

## Benutzen

**Windows:** MSI installieren → [Anleitung](docs/install-windows.md).

**Aus dem Quellcode:**

```bash
uv sync --extra ui
.venv/bin/nova serve                       # → http://127.0.0.1:8765 (ohne models.toml: Setup-Modus)
.venv/bin/nova serve --config ~/.nova/models.toml
.venv/bin/nova service start|stop|status   # als Hintergrundprozess
.venv/bin/nova integrations create --name meine-app --scope chat:complete
```

Andere Programme anbinden: [Integrations-API](docs/integration-api.md) ·
Beispiel [IC WARE HQ](examples/ic-ware-hq-integration/README.md).

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
- [Integrations-API / OpenAI-Kompatibilität](docs/integration-api.md)
- [Windows-Installation](docs/install-windows.md) · [Release-Plan](docs/windows-release-plan.md)
- [Testbericht](docs/test-report.md)

## Struktur

| Verzeichnis | Inhalt |
|---|---|
| `api/` | Lokale NOVA API (FastAPI), Integrations-API, CLI `nova`, Web-Oberfläche |
| `packaging/` | PyInstaller-Build, WiX-Installer, Installationstest |
| `examples/` | Integrationsbeispiele (IC WARE HQ) |
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
