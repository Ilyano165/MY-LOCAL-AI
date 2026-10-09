# NOVA – Produktarchitektur (Desktop-KI mit eigener Modell- und Lernpipeline)

Stand: 2026-10-09 · ersetzt keine bestehenden Dokumente, sondern ordnet sie ein
(`architecture.md` = technische Kernarchitektur, `integration-api.md`, `windows-release-plan.md`).

Status-Legende: **✅ implementiert + getestet** · **🟡 teilweise** · **⬜ geplant**.
Nichts in diesem Dokument behauptet einen Stand, der nicht in `docs/test-report.md` belegt ist.

## 1. Zielbild

NOVA ist eine installierbare Windows-KI-Plattform, kein Browser-Tab und kein bloßer
Wrapper um ein fremdes Modell:

1. **Desktop-App** mit eigenem Fenster, Branding, Startmenü-Eintrag – kein manueller URL-Aufruf.
2. **Ein Core-Dienst**, den Desktop-App, Integrations-API (IC WARE HQ) und Research Engine
   gemeinsam nutzen.
3. **Eigene Modellentwicklung**: Open-Weight-Basismodell → eigene Daten → SFT → (Präferenz-
   optimierung) → unabhängige Evaluation → versionierte NOVA-Checkpoints
   (`docs/model-development.md`).
4. **Research → Wissen → (geprüft) Training**, nie ungeprüft in die Gewichte
   (`docs/continuous-learning.md`).

## 2. Bestandsaufnahme (2026-10-09)

| Komponente | Verzeichnis | Status |
|---|---|---|
| Model Runtime-Anbindung (OpenAI-kompatibel, Ollama) | `models/` | ✅ |
| Model Router (Regeln) / Learned Router (inaktiv) | `router/` | ✅ / ✅ nicht aktiviert |
| Agent Core, Planner, Verifier | `agents/`, `evaluation/` | ✅ |
| Tools (Dateisystem, Git, Terminal mit Freigaben) | `tools/` | ✅ |
| Memory (4 Schichten, SQLite/FTS5) | `memory/` | ✅ Bibliothek · ⬜ nicht an Core/UI angebunden |
| RAG / Retrieval über Dokumente | `rag/` | ⬜ (leer) |
| Core-Dienst + Web-UI + Integrations-API + CLI | `api/` | ✅ |
| Windows-Installer (MSI), CI mit Installationstest | `packaging/`, `.github/` | ✅ (Release v0.1.0) |
| Desktop-Fenster | `desktop/` | ✅ dieser Schritt (siehe §5) |
| Research Engine | `research/` | ⬜ nächster Schritt |
| Trainings-/Datensatzpipeline | `training/` | ⬜ übernächster Schritt |
| Zentrale Ressourcensteuerung | `core/` | ⬜ |
| Web-Zugriff (Suche/Fetch) | – | ⬜ existiert nicht; Agent hat keine Web-Tools |

Wichtigste Lücke gegenüber dem Zielbild: Es gibt bisher **keine** Web-Recherche, **keine**
Trainingspipeline und **kein** eigenes Modell. Alle Chat-Tests laufen gegen eine simulierte
Runtime.

## 3. Komponenten und Grenzen

```
┌──────────────── Desktop UI (desktop/, nova-desktop.exe) ───────────────┐
│ natives Fenster (WebView2) → lädt die UI des Core über 127.0.0.1        │
│ eigene Bridge nur für Fenster-/Dienststeuerung (Start/Stopp/Neustart)   │
└───────────────┬─────────────────────────────────────────────────────────┘
                │ HTTP (gleiche API wie Browser-UI)
┌───────────────▼──────────────── Core Service (api/, nova.exe serve) ────┐
│ NovaService: Chat, Agent, Verlauf, Settings, Setup, Integrations-API    │
│   ├─ Model Router (router/) ── Model Runtime-Adapter (models/)          │
│   ├─ Agent Core (agents/) + Tools (tools/) mit Freigaben                │
│   ├─ Memory & Retrieval (memory/, rag/)              ⬜ Anbindung        │
│   ├─ Research Engine (research/)                     ⬜                  │
│   ├─ Training Pipeline (training/) – separater Prozess ⬜               │
│   ├─ Evaluation (evaluation/)                                           │
│   └─ Resource Governor (core/) – ein Platz pro Modell-/GPU-Job  ⬜       │
└───────────────┬─────────────────────────────┬───────────────────────────┘
                │ /api/v1, /v1 (Schlüssel)    │ HTTP
        IC WARE HQ / andere Apps     lokale Runtime (llama.cpp, Ollama)
```

Regeln:

* **Desktop UI enthält keine Geschäftslogik.** Sie ist ein Fenster auf die Core-UI plus
  Dienststeuerung. Der Core ist ohne Desktop-App vollständig nutzbar (Browser, API, CLI).
* **Ein Core, eine InferenceEngine.** Desktop, Browser-UI, Integrations-API und Research
  Engine rufen dieselben Dienste.
* **Chat startet nie Training.** Training ist ein expliziter, eigener Prozess mit
  Ressourcenlimit (`docs/continuous-learning.md` §5).
* **Research und Training blockieren den Chat nicht.** Der Resource Governor vergibt
  Modell-Slots nach Priorität: interaktiver Chat > Agent > Research > Training. Reicht die
  Hardware nicht für parallele Modellprozesse, wartet Hintergrundarbeit (pausierbar,
  checkpointbasiert).

## 4. Datenablage (Benutzerdaten, `%LOCALAPPDATA%\NOVA`)

| Pfad | Inhalt | Status |
|---|---|---|
| `models.toml`, `models/` | Runtime-Konfiguration, GGUF-Dateien | ✅ |
| `conversations/`, `settings.json` | Verlauf, Einstellungen | ✅ |
| `integrations.db`, `logs/audit.jsonl` | Integrationsschlüssel (Hash), Audit | ✅ |
| `desktop.json` | Fenstergröße, „Core beim Schließen weiterlaufen lassen“ | ✅ dieser Schritt |
| `memory.db` | Gedächtnis | ⬜ Anbindung |
| `research/<run-id>/` | Plan, Checkpoints, Quellen-Snapshots, Audit, Bericht | ⬜ |
| `knowledge.db` | geprüfte Erkenntnisse mit Quellen (Retrieval) | ⬜ |
| `datasets/<name>/<version>/` | Trainingsdaten (train/val/test, Manifest, Hashes) | ⬜ |
| `checkpoints/nova-<version>/` | LoRA-Adapter / gemergte Gewichte, Model Card, Eval-Report | ⬜ |

## 5. Desktop-Packaging – Entscheidung

| Kriterium | **pywebview (WebView2)** | Tauri 2 + Python-Sidecar | Electron + Python-Sidecar |
|---|---|---|---|
| zusätzliche Toolchain | keine (Python, bereits PyInstaller) | Rust, Cargo, Sidecar-Verwaltung | Node, electron-builder |
| Renderer | System-WebView2 (Chromium) | System-WebView2 | gebündeltes Chromium |
| Paketgröße zusätzlich | wenige MB (pywebview, pythonnet) | Shell klein, Python-Runtime bleibt | ~300 MB laut Vergleichsmessung |
| Prozesse | Fenster-Prozess + Core-Prozess | Rust-Shell + Core-Prozess | Electron (mehrere) + Core |
| Wiederverwendung UI | 1:1 (lädt Core-UI) | 1:1 | 1:1 |
| Wartung | ein Build (PyInstaller + MSI) | zwei Build-Systeme | zwei Build-Systeme, große Updates |
| Lizenz | BSD-3 (pywebview), MIT (pythonnet) | MIT/Apache | MIT |

**Entscheidung: pywebview 6.x mit Edge-Chromium/WebView2-Backend.** Begründung: Das Frontend
hat keinen Build-Schritt, der Core ist Python und wird bereits mit PyInstaller gepackt; ein
zusätzliches Rust- oder Node-Ökosystem brächte Wartungsaufwand ohne funktionalen Gewinn.
WebView2 ist derselbe Chromium-Renderer, den Tauri unter Windows nutzt.

Voraussetzungen und Grenzen (Windows):
* WebView2 Evergreen Runtime (bei Windows 11 vorinstalliert, bei Windows 10 über Edge in der
  Regel vorhanden) und .NET Framework ≥ 4.6.2 (Windows 10/11 enthalten). Fehlt WebView2,
  meldet die App das klar und bietet an, NOVA im Standardbrowser zu öffnen. Eine stille
  Rückfallstufe auf den alten IE-Renderer (MSHTML) wird **verhindert** (die UI benötigt
  moderne Web-Standards).
* Zwei Prozesse bewusst getrennt: `nova-desktop.exe` (Fenster) und `nova.exe serve` (Core).
  Schließen des Fensters stoppt den Core nur, wenn das Fenster ihn gestartet hat und die
  Einstellung „Core im Hintergrund weiterlaufen lassen“ aus ist (Integrationen wie HQ
  brauchen den Core unabhängig vom Fenster).

Quellen: [pywebview Web-Engine-Doku](https://pywebview.flowrl.com/guide/web_engine),
[pywebview Releases](https://github.com/r0x0r/pywebview/releases),
[Microsoft WebView2](https://learn.microsoft.com/en-us/microsoft-edge/webview2/),
[Tauri/Electron-Vergleich (Better Stack)](https://betterstack.com/community/guides/scaling-nodejs/tauri-vs-electron-vs-deno-vs-electrobun/),
PyPI-Metadaten pywebview 6.2.1 / pythonnet 3.2.1 (abgerufen 2026-10-09).

## 6. UI-Funktionsumfang (Desktop = gleiche UI wie Browser)

| Bereich | Status |
|---|---|
| Chat mit Streaming, Stop, Markdown, Anhänge | ✅ |
| Conversation History | ✅ |
| Modellverwaltung (Status, Katalog, Download mit Prüfungen) | ✅ |
| Agent Mode | ✅ |
| Einstellungen, System-/Modell-/Router-Status (Diagnose) | ✅ |
| Core-Dienst starten/stoppen/neu starten aus der App | ✅ dieser Schritt (Desktop-Bridge) |
| Research Mode | ⬜ mit Research Engine |
| Memory-Verwaltung | ⬜ mit Memory-Anbindung |
| Trainings- und Evaluationsstatus | ⬜ mit Trainingspipeline |

Nicht vorhandene Bereiche werden in der UI **nicht** als Attrappe angezeigt.

## 7. Umsetzungsreihenfolge

| # | Abschnitt | Status |
|---|---|---|
| 1 | Bestandsaufnahme, dieses Dokument, Modell- und Lern-Architektur | ✅ |
| 2 | Desktop-App (Fenster, Icon, Core-Steuerung, MSI-Integration, CI-Test) | ✅ |
| 3 | Research Engine (minimaler echter Durchlauf mit Checkpointing, Audit, Bericht) | ⬜ nächster Schritt |
| 4 | Knowledge Store + Retrieval (Wissensverbesserung ohne Training), Memory-Anbindung | ⬜ |
| 5 | Datensatz- und Trainingspipeline (Versionierung, Splits, Leak-Check, Trainer-Adapter) | ⬜ |
| 6 | Resource Governor, Research/Training-Status in der UI | ⬜ |
| 7 | Erstes Fine-Tuning auf geeigneter Hardware + unabhängige Evaluation | ⬜ hardwareabhängig |
