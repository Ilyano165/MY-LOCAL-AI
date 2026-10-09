# api/ – NOVA API und Web-Oberfläche

```
Browser (api/static)  →  NOVA API (FastAPI, nur 127.0.0.1)  →  NovaService
                                                               → InferenceEngine / Agent
                                                               → Model Router
                                                               → lokale Runtime (llama.cpp, Ollama, …)
```

Die Oberfläche spricht **ausschließlich** mit dieser API; Runtime-Adressen verlassen den Server
nie.

## Starten

```bash
uv pip install --python .venv/bin/python -e ".[ui]"
.venv/bin/python -m api --config ~/.nova/models.toml          # http://127.0.0.1:8765
.venv/bin/python -m api --dev                                  # ohne Modell: „No local model available.“
```

Optionen: `--port`, `--data-dir` (Standard `~/.nova`), `--router rules|learned`,
`--learned-ranker PFAD`. Optionales API-Token über die Umgebungsvariable `NOVA_API_TOKEN`.

## Endpunkte

| Methode | Pfad | Zweck |
|---|---|---|
| POST | `/chat` | Antwort ohne Streaming |
| POST | `/chat/stream` | Server-Sent Events: `run` → `routing` → `status` → `token`* → `done` \| `stopped` \| `error` |
| POST | `/agent/run` | Agent-Lauf (SSE mit `phase`-Ereignissen, Ergebnis mit Verifikationsurteil) |
| POST | `/agent/stop` | laufende Generierung oder Agent-Lauf stoppen (`run_id`) |
| GET | `/models` | Modelle mit Fähigkeiten und Datenstatus (MEASURED/UNMEASURED) |
| GET | `/models/status` | Verfügbarkeit je Modell, Runtime-Health (`?refresh=true`) |
| GET | `/router/status` | Router, Klassifikator, Learned Ranker, letzte Entscheidungen |
| GET | `/system/status` | Version, Modus, Hardware, freier Speicher, aktive Läufe |
| GET/PUT | `/settings` | Einstellungen (Modell, Temperatur, Tokens, System-Prompt, Agent-Workspace …) |
| GET/POST/PATCH/DELETE | `/conversations[/{id}]` | Verlauf |
| GET | `/uploads/{datei}` | hochgeladene Bilder (nur generierte Dateinamen) |

Interaktive Doku: `/api/docs`.

## Ehrlichkeit der Anzeige

* Kennzahlen erscheinen nur, wenn sie vorliegen: Tokens nur, wenn die Runtime `usage` meldet;
  Tokens/s von NOVA gemessen (Dekodierzeit ab erstem Token) oder als „runtime“ gekennzeichnet;
  Verifikation nur bei Agent-Läufen; „Streaming not supported by runtime“, wenn die Antwort
  in einem Stück kam.
* Ohne verfügbares Modell: HTTP 503 `no_model` mit Begründung, kein Verlaufseintrag, keine
  simulierte Antwort. Die UI zeigt „No local model available.“ und sperrt das Senden.
* Gestoppte oder abgebrochene Antworten werden als solche gespeichert.

## Sicherheit

Bindung an localhost (`--allow-remote` nötig für anderes), schreibende Anfragen nur mit Header
`X-NOVA-Client` und eigener `Origin` (Schutz vor fremden Webseiten/CSRF), Content-Security-
Policy ohne Inline-Skripte, Markdown-Renderer ohne Roh-HTML, Links nur http/https/mailto,
Anhänge mit Größenlimit (10 MB) und Typprüfung (Text, Bilder).

## Frontend

Ohne Build-Schritt und ohne externe Ressourcen (läuft offline): `static/index.html`,
`static/app.css`, ES-Module in `static/js/` – `api.js` (Client, SSE-Parser), `markdown.js`, `highlight.js`
(Syntaxhervorhebung für Python, JS/TS, Bash, Rust, Go, C-Familie, SQL, CSS, TOML/YAML),
`format.js` (Kennzahlen), `app.js` (Controller). Erweiterungen: neue Ansicht als Modul,
Endpunkt in `api.js` ergänzen.

## Tests

`tests/api/` (Endpunkte, Streaming, Stoppen, Fehler, Sicherheit, Agent), `tests/ui/*.test.mjs`
(Markdown inkl. XSS, Kennzahlen, SSE-Parser; über `node --test`, in pytest eingebunden),
`tests/ui/test_e2e.py` (Playwright/Chromium gegen echte Server mit Test-Runtime:
Development Mode, Streaming, Kennzahlen, Copy, Stop, Verlauf, Drag & Drop, Settings, Status,
Mobile).

## Grenzen (aktuell)

* Tool-Bestätigungen im Agent-Modus haben noch keinen UI-Dialog (bestätigungspflichtige Tools
  werden abgelehnt); Agent-Modus ohne Anhänge.
* PDF/Office-Anhänge noch nicht unterstützt (klare Fehlermeldung).
* Router-Begründungen und Verfügbarkeitsgründe stammen aus dem Core und sind auf Deutsch (in der UI als „Routing reason“ gekennzeichnet); UI und API-Fehlermeldungen sind Englisch.
