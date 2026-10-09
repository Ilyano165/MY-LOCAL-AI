# NOVA – Testbericht (Windows-Release + Integrationsplattform)

Stand: 2026-10-09 · Branch `claude/beautiful-faraday-oqvyqv`

Dieser Bericht trennt drei Dinge strikt:

* **ausgeführt und bestanden**: mit Nachweis;
* **gebaut, aber nicht so geprüft, wie ein Benutzer es täte**;
* **nicht ausgeführt**: mit Grund.

## 1. Automatische Tests

| Umgebung | Umfang | Ergebnis |
|---|---|---|
| Linux (lokal, Python 3.13) | komplette Suite inkl. Browser-E2E (Playwright/Chromium) | **922 bestanden**, 2 abgewählt (Marker `integration`, s. §4) |
| Linux (GitHub Actions `ci`) | ruff, ruff format, mypy (Linux **und** `--platform win32`), komplette Suite inkl. E2E | bestanden |
| Windows (GitHub Actions `windows-release`, `windows-latest`) | ruff, mypy, komplette Suite **ohne** Browser-E2E | {{WIN_PYTEST}} |
| Linux (lokal) | ruff, ruff format, mypy --strict (77 Quelldateien, Linux + win32) | sauber |

Tests nach Bereich (Anzahl gesammelter Tests): agents 90 · api 95 · evaluation 262 ·
examples 15 · memory 92 · models 131 · packaging 5 · router 98 · tools 124 · ui 10.

### Abdeckung der geforderten Prüfungen (Phase 7)

| Anforderung | Tests |
|---|---|
| Authentifizierung, fehlender/ungültiger Schlüssel | `test_v1.py::test_missing_or_invalid_key`, `test_key_is_stored_hashed_revoke_and_rotate`, Ablauf |
| Berechtigungen/Scopes, Modellfreigabe | `test_v1.py` (Scope-, Modell-, Werkzeug-Tests) |
| Chat ohne Modell (503 `no_model`) | `test_v1.py`, `test_api.py`, HQ-Beispiel |
| erfolgreicher Chat, Streaming | `test_v1.py`, `test_streaming_core.py`, E2E |
| Abbruch | Client-Disconnect bricht Runtime-Stream ab; Agent-Cancel |
| Zeitüberschreitung | `_Timeout`/504, Agent `timed_out`, HQ-Client |
| nicht verfügbares/unbekanntes Modell | `model_not_found` 404, `model_not_allowed` 403 |
| Agent-Aufgaben mit eingeschränkten Rechten | Werkzeug nur mit `agent:tool:*` **und** Anforderung; fremde Aufgaben 404 |
| Version/Health | `test_health_is_public_and_minimal`, CLI `--version`, Installationstest |
| OpenAI-Kompatibilität | offizielles `openai`-SDK 3.26.1: Modelle, Chat, Streaming, Fehlerklassen |
| HQ-Beispiel | `tests/examples/test_hq_integration.py` (15 Tests gegen echte NOVA-App) |
| Schutz vor Netzwerkzugriff | Middleware-Test (fremde Client-Adresse → 403); Installationstest: Dienst lauscht nur auf 127.0.0.1 |
| Upgrade-Verhalten, Erhalt der Benutzerdaten | echter Hintergrunddienst Start/Stop/Neustart mit Datenerhalt (Linux); MSI-Upgrade/Reparatur/Deinstallation (Windows, §2) |
| Modell-Download | Prüfsumme, Nicht-GGUF, Unterbrechung + Fortsetzen (HTTP Range), Server ohne Range, RAM/Platz, Lizenzpflicht |

## 2. Windows-Installer

| Stufe | Status | Nachweis |
|---|---|---|
| Programmordner gebaut (PyInstaller) | {{WIN_BUILD}} | Workflow-Schritt „Build program folder“ |
| Rauchtest der gebauten EXE (`--version`, Server-Start, Health, UI ausgeliefert) | {{WIN_SMOKE}} | `packaging/build.py` |
| MSI gebaut (WiX 5.0.2) + ICE-Validierung | {{WIN_MSI}} | `build-msi.ps1` |
| **Installationstest** (unbeaufsichtigt, `msiexec /qn`) | {{WIN_INSTALL}} | `install-test.md` in den Artefakten |
| SHA-256-Prüfsummen | {{WIN_SHA}} | `SHA256SUMS.txt` |

Der Installationstest prüft auf `windows-latest`:

1. Installation;
2. Programmdateien;
3. Startmenü-Verknüpfungen;
4. keine Desktop-Verknüpfung ohne Auswahl;
5. Version in der EXE und in der Windows-Installer-Registrierung;
6. Dienststart im Setup-Modus;
7. der Dienst lauscht nur auf Loopback;
8. Benutzerdaten anlegen;
9. **Upgrade bei laufendem Dienst** (Exit 0, kein Neustart nötig);
10. Version danach;
11. Daten erhalten;
12. neue Version läuft;
13. **Downgrade abgelehnt**;
14. **Reparatur** stellt eine gelöschte Programmdatei wieder her;
15. **Deinstallation bei laufendem Dienst**;
16. Programmordner und Verknüpfungen entfernt;
17. Dienst gestoppt;
18. **Benutzerdaten erhalten**.

Ergebnis des letzten Laufs: {{WIN_RUN}}

Gefunden und behoben durch die Windows-Läufe:

* Typfehler in plattformabhängigem Code;
* CRLF-Zeilenenden wurden beim Bearbeiten verdoppelt;
* Pfade mit `\` in der Werkzeugausgabe;
* eine offene Datei wurde vor dem Löschen nicht geschlossen;
* der Launcher startete sich unter Linux selbst;
* die Produktregistrierung wurde im Test falsch abgefragt.

## 3. Nicht ausgeführt / nicht geprüft

| Was | Grund |
|---|---|
| Installation auf einem **Windows-10/11-Desktop** mit grafischem Assistenten (Dialoge, SmartScreen, Doppelklick aufs Startmenü, Browser öffnet sich) | Nur Windows Server (`windows-latest`) ohne interaktive Sitzung verfügbar. Der Assistent (WixUI_FeatureTree) wurde **nicht** durchgeklickt. |
| Installer-Features „Desktop shortcut“ und „Start with Windows“ | im CI nicht ausgewählt (nur Standardauswahl getestet) |
| Browser-E2E unter Windows | Chromium-Test nur unter Linux |
| Generierung mit einem **echten Sprachmodell** (llama.cpp/Ollama) | kein Modell in der Testumgebung, und laut Vorgabe keine großen Modelle herunterladen. Alle Chat-Tests nutzen eine simulierte OpenAI-kompatible Runtime. Marker `integration` (2 Tests) abgewählt. |
| echter Modell-Download aus dem Internet | kein Katalog mit echten Einträgen (NOVA erfindet keine URLs/Prüfsummen); Download-Logik gegen lokale Test-Server geprüft |
| Fernzugriff mit TLS (`--allow-remote`) über ein echtes Netzwerk | nur die Start-Bedingungen (TLS + Token erforderlich) und die Zugriffsprüfung sind getestet |
| Code-Signing, SmartScreen-Reputation | nicht umgesetzt (kein Zertifikat) |
| Last-/Dauertests, mehrere gleichzeitige Integrationen unter Last | nicht durchgeführt |
