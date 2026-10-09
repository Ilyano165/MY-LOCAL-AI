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
| Windows (GitHub Actions `windows-release`, `windows-latest`) | ruff, mypy, komplette Suite **ohne** Browser-E2E | **913 bestanden**, 3 übersprungen (Browser-/plattformabhängige Tests), 0 fehlgeschlagen |
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
| Programmordner gebaut (PyInstaller) | **gebaut** | Workflow-Schritt „Build program folder“ |
| Rauchtest der gebauten EXE (`--version`, Server-Start, Health, UI ausgeliefert) | **bestanden** („NOVA 0.1.0, health OK, UI served“) | `packaging/build.py` |
| MSI gebaut (WiX 5.0.2) + ICE-Validierung | **gebaut + ICE-Validierung bestanden** (0.1.0 und Upgrade-Testpaket 0.1.1) | `build-msi.ps1` |
| **Installationstest** (unbeaufsichtigt, `msiexec /qn`) | **bestanden: 18 von 18 Prüfungen** | `install-test.md` in den Artefakten |
| SHA-256-Prüfsummen | `b6f0c98806f5a4fbb00fe0888dd5491890f88ccd61061e2d2eba20cf4fa1996a  NOVA-0.1.0-x64.msi` | `SHA256SUMS.txt` |

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

Ergebnis des letzten Laufs: [windows-release #7](https://github.com/Ilyano165/MY-LOCAL-AI/actions/runs/37944321857) auf Commit `ea617bc`: alle Schritte grün. Das MSI (≈ 26 MB) liegt mit `SHA256SUMS.txt` im Artefakt `nova-windows-0.1.0`, Logs und `install-test.md` im Artefakt `nova-windows-logs`. Ein GitHub-Release wird erst bei einem Tag `v0.1.0` erzeugt; bisher wurde keins angelegt.

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

## 4. Release v0.1.0

Veröffentlicht über `windows-release` (manueller Start mit `release=true`,
[Lauf](https://github.com/Ilyano165/MY-LOCAL-AI/actions/runs/37945517772), Commit `1b18794`):
<https://github.com/Ilyano165/MY-LOCAL-AI/releases/tag/v0.1.0>. Das MSI wurde in diesem Lauf neu
gebaut, und der Installationstest bestand erneut 18 von 18 Prüfungen (`install-test.md` im Release).

`aab2db96d77915c5857c366bb1e3ca9ab5b479e9f32bd6d3bba6617aeedb9ce0  NOVA-0.1.0-x64.msi`

## 5. Desktop-App (Architekturkorrektur, Abschnitt 2)

Stand: 2026-10-09 · Lauf [windows-release](https://github.com/Ilyano165/MY-LOCAL-AI/actions/runs/37955664795)
auf Commit `d6a2dee`, alle Schritte grün; Linux-CI auf demselben Commit grün.

| Prüfung | Umgebung | Ergebnis |
|---|---|---|
| Suite lokal (Linux, inkl. Browser-E2E) | lokal | **933 bestanden**, 2 abgewählt |
| Suite Windows (ohne Browser-E2E) | `windows-latest` | **922 bestanden**, 3 übersprungen |
| Desktop-Unit-/Integrationstests (Einstellungen, WebView2-Erkennung, Core-Steuerung, Bridge, Fensterlebenszyklus mit **echtem Core-Prozess**) | Linux | 9 bestanden |
| JS-Units Desktop-Bridge | Node | 5 bestanden |
| Browser-E2E: Desktop-Modus (simulierte pywebview-Bridge) und Browser-Modus (kein Desktop-Panel) | Chromium | bestanden |
| PyInstaller-Build mit `nova-desktop.exe` + Rauchtest | Linux lokal + Windows CI | bestanden (Linux-Ordner 52 MB) |
| MSI + ICE-Validierung | Windows CI | bestanden |
| **Installationstest 20/20**, darunter: Startmenü „NOVA“ → `nova-desktop.exe`; **echtes Desktop-Fenster** startet im Selbsttest mit Renderer `edgechromium`, Titel „NOVA“, Desktop-Bridge aktiv, UI hat den Core-Status geladen („No local model available“), Core läuft danach weiter (Fenster hatte ihn nicht gestartet) | Windows CI | bestanden |

MSI dieses Laufs (CI-Artefakt, **kein** Release):
`d8a2739841fbc59f84a82d5b649e1710cd5a630f4aa4124df2a3440853042938  NOVA-0.1.0-x64.msi`.
Das Release v0.1.0 enthält die Desktop-App noch **nicht**.

Nicht geprüft: Desktop-Fenster auf einem echten Windows-10/11-Arbeitsplatz durch einen Menschen
(Darstellung, HiDPI, Fenstergröße merken, Schließen-Verhalten interaktiv), Rückfall-Dialog bei
fehlender WebView2 auf echter Maschine (nur per Test mit simulierter Registry), Knöpfe der
Desktop-Bridge im echten WebView2-Fenster (nur in Chromium mit simulierter Bridge).
