# NOVA – Windows-Release-Plan und Bestandsaufnahme

Stand: 2026-10-09 · Version im Repository: siehe `pyproject.toml`

Dieses Dokument trennt strikt: **implementiert + getestet** · **implementiert, nicht auf
Windows geprüft** · **geplant**. Der aktuelle Prüfstatus des Installers steht in §7.

## 1. Bestandsaufnahme (vor diesem Schritt)

| Bereich | Status | Nachweis |
|---|---|---|
| Model-Layer (`models/`): Provider, Registry, InferenceEngine, Health, Streaming | implementiert + getestet (Linux) | `tests/models`, `tests/api/test_streaming_core.py` |
| Model Router (`router/`), Routing-Evaluation | implementiert + getestet | `tests/router`, `evaluation/reports/routing_baseline.md` |
| Learned Router | implementiert + getestet, **nicht aktiviert** (Vorteil nicht nachgewiesen) | `docs/learned-routing.md` |
| Agent Core (`agents/`), Tools (`tools/`), Verification (`evaluation/`) | implementiert + getestet | `tests/agents`, `tests/tools`, `tests/evaluation` |
| Memory (`memory/`) | implementiert + getestet, **nicht an die API angebunden** | `tests/memory` |
| Model-Benchmarking | implementiert + getestet, nie gegen echtes Modell gelaufen | `docs/benchmarking.md` |
| Lokale API + Web-UI (`api/`) | implementiert + getestet (inkl. Browser-E2E) | `tests/api`, `tests/ui` |
| RAG (`rag/`), `app/`, `core/` | **nur geplant** (leere Verzeichnisse) | – |
| CI, Packaging, Installer | **nicht vorhanden** | – |
| Windows-Lauffähigkeit | **nie geprüft**; Ressourcenerkennung nutzt `/proc`/`sysconf` (auf Windows: „unbekannt“) | Code-Review |

Build-System: `pyproject.toml` (setuptools), `uv.lock`; Laufzeitabhängigkeiten `httpx`, optional
`fastapi` + `uvicorn` (UI/API). Kein Node.js zur Laufzeit (Frontend ohne Build-Schritt).

## 2. Technologieentscheidungen

### Ausführbares Paket: PyInstaller (onedir)
* Bündelt Python-Interpreter und Abhängigkeiten → **kein Python auf dem Zielrechner nötig**.
* `onedir` statt `onefile`: schneller Start, kein Entpacken nach `%TEMP%`, weniger
  Fehlalarme von Virenscannern, Dateien für den MSI-Installer einzeln versionierbar.
* Zwei Programme aus einer Analyse: `nova.exe` (Konsole: CLI, Server) und `nova-launcher.exe`
  (ohne Konsolenfenster: Startmenü-Verknüpfungen „NOVA öffnen“, Dienst starten/stoppen).
* Version 6.x mit Python 3.13 (PyInstaller unterstützt 3.8–3.15).

### Installer: MSI mit WiX Toolset **v5.0.2** (gepinnt)
* MSI ist das Standardformat für Unternehmen (Verteilung über Intune/GPO, `msiexec`-Logs,
  **Reparatur** und **Major Upgrade** eingebaut) – wichtig für den Einsatz bei IC WARE.
* **Lizenz:** WiX v6 und v7 stehen unter einer „Open Source Maintenance Fee“-EULA
  (Gebühr für Organisationen mit Umsatz, v7 ab 10 000 USD/Jahr). WiX **v5** steht unter MS-RL
  ohne Gebühr und bietet bereits das `Files`-Element (automatisches Einsammeln der
  PyInstaller-Ausgabe). Ein späterer Wechsel auf v6/v7 setzt eine Prüfung der EULA voraus.
* Alternativen geprüft: Inno Setup/NSIS (EXE-Installer, kein MSI-Repair/Upgrade-Standard,
  schlechter für Unternehmensverteilung), MSIX (Sandboxing erschwert einen lokalen
  Hintergrunddienst mit Dateizugriff, Signaturpflicht).

### Installationsart: pro Benutzer, ohne Administratorrechte
* Programmdateien: `%LOCALAPPDATA%\Programs\NOVA\` (vom MSI verwaltet).
* Benutzerdaten: `%LOCALAPPDATA%\NOVA\` – Konfiguration, Chatverlauf, Memory, Logs, Modelle,
  Einstellungen, Integrationsschlüssel (gehasht). **Vom Installer nie angefasst** →
  bleiben bei Update, Reparatur und Deinstallation erhalten.
* Begründung: NOVA ist ein persönlicher Assistent; Modelle, Verlauf und der lokale Dienst
  gehören zum Benutzerkonto. Ein Windows-Systemdienst (LocalSystem) hätte ein anderes Profil
  und bräuchte Administratorrechte.

### „Hintergrunddienst“: Benutzerprozess statt Windows-Service
* `nova service start|stop|status` startet den API-Server als abgelösten Benutzerprozess
  (nur `127.0.0.1`), PID-Datei + Health-Check, Log in `%LOCALAPPDATA%\NOVA\logs\`.
* Startmenü: „NOVA“ (startet bei Bedarf den Dienst und öffnet die Oberfläche),
  „NOVA-Dienst stoppen“. Optionaler Autostart bei Anmeldung (Feature im Installer).
* Vor Upgrade/Deinstallation stoppt der Installer den Dienst (Custom Action).

### Systemvoraussetzungen
* 64-Bit-Windows 10/11 (MSI-Startbedingung `VersionNT64`, Mindestversion `603`; Windows 10
  meldet MSI aus Kompatibilitätsgründen `603`). Abhängigkeiten: Python 3.13 (über PyInstaller
  gebündelt) unterstützt Windows 10+.
* Modelle sind **nicht** im Installer: Download separat mit Größen-, Speicher-, Lizenz- und
  Prüfsummenprüfung (`nova models …`, API, Setup-Bildschirm).

## 3. Integrationsplattform (Phasen 3–6)

* Versionierte API `/api/v1/*` und OpenAI-kompatible Chat-Completions unter `/v1/*` – beide
  auf **derselben** Backend-Funktion (`NovaService`), keine zweite InferenceEngine.
* Integrationsschlüssel (gehasht gespeichert), Scopes je Integration, Rate Limits,
  Größenlimits, Audit-Log, restriktives CORS (Standard: keine Cross-Origin-Freigabe).
* Agent-Tasks mit Task-ID, Status, Abbruch; Tools nur bei expliziter Freigabe pro Scope.
* Netzwerk: Standard `127.0.0.1`; Fernzugriff nur bewusst mit TLS + Schlüsseln.
* Details: `docs/integration-api.md`, Beispiel `examples/ic-ware-hq-integration/`.

## 4. Release-Prozess

1. Version in `pyproject.toml` setzen (einzige Quelle; MSI-Version wird daraus abgeleitet).
2. Tag `vX.Y.Z` pushen → GitHub Actions `windows-release.yml`:
   Tests (Windows) → PyInstaller → Smoke-Test der EXE → MSI bauen → **Installationstest**
   (Installieren, Dienst starten, Health-Check, Daten anlegen, Upgrade auf höhere Version,
   Datenerhalt prüfen, Reparatur, Deinstallation, Datenerhalt prüfen) → SHA-256 → Artefakte.
3. Bei Tags zusätzlich GitHub-Release mit MSI, Prüfsummen und Testprotokoll.
4. Der Workflow unterscheidet **gebaut** (Artefakt existiert) und **getestet** (Installations-
   test bestanden); das Testprotokoll ist Teil der Artefakte.

## 5. Arbeitsplan

| Schritt | Inhalt | Abnahme |
|---|---|---|
| 1 | Plattformpfade, Windows-RAM-Erkennung, Versionsquelle | Tests |
| 2 | Integrationsschlüssel, Scopes, Rate Limit, Audit | Tests |
| 3 | `/api/v1` + OpenAI-kompatibel, gemeinsame Generierungsfunktion | Tests + E2E |
| 4 | Agent-Tasks (ID, Status, Abbruch, Tool-Freigabe) | Tests |
| 5 | CLI `nova` (serve, service, integrations, models, data), Netzwerkschutz | Tests |
| 6 | Modellverwaltung (Download, Prüfsumme, Fortsetzen, Lizenz, Speicher) + Setup-Ansicht | Tests |
| 7 | HQ-Integrationsbeispiel | Tests gegen echten Server |
| 8 | PyInstaller-Spec, WiX-Quelle, CI-Workflow mit Installationstest | CI-Lauf auf `windows-latest` |
| 9 | Dokumentation, Testbericht | – |

## 6. Bewusst nicht Teil dieses Releases

Codesignatur (benötigt ein Zertifikat des Unternehmens – ohne Signatur zeigt Windows
SmartScreen eine Warnung), automatische Updates im Programm, Windows-Systemdienst,
RAG, Memory-Anbindung an die API.

## 7. Prüfstatus

Wird nach jedem CI-Lauf aktualisiert – siehe `docs/test-report.md`.
