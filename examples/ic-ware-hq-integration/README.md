# IC WARE HQ ↔ NOVA – Beispielintegration

Dieses Beispiel zeigt, wie ein Backend (hier exemplarisch „IC WARE HQ“) die lokale
NOVA-Integrations-API **serverseitig** nutzt. Das HQ-eigene Technologie-Stack ist NOVA nicht
bekannt; das Beispiel ist deshalb in Python (FastAPI + httpx) geschrieben und lässt sich in
jede Sprache übertragen – es ist reines HTTP + JSON (siehe `docs/integration-api.md`).

| Datei | Zweck |
|---|---|
| `nova_client.py` | austauschbarer Client: Base URL/Schlüssel/Timeout nur aus Parametern oder Umgebung, Fehler → eigene Ausnahmen, Wiederholung nur bei 429/503 |
| `hq_backend.py` | Beispiel-HQ-Endpunkt `POST /hq/assistant`: prüft HQ-Login, ruft NOVA, gibt Antwort + **freigegebene** Metadaten zurück |
| `.env.example` | Vorlage für die Server-Umgebung (Schlüssel leer) |
| Tests | `tests/examples/test_hq_integration.py` (gegen eine echte NOVA-App mit Test-Runtime) |

## Ablauf

```
Browser (HQ-Frontend)
   │  HQ-Session (HQ-eigene Anmeldung) – KEIN NOVA-Schlüssel im Browser
   ▼
HQ-Backend  /hq/assistant
   │  1. HQ-Benutzer authentifiziert/autorisiert?        (HQ-Sache)
   │  2. Authorization: Bearer <NOVA_API_KEY>             (nur Server-Umgebung)
   ▼
NOVA  /api/v1/chat/completions
   │  3. Schlüssel gültig, nicht widerrufen/abgelaufen?   → sonst 401
   │  4. Scope chat:complete vorhanden? Modell erlaubt?    → sonst 403
   │  5. Rate Limit / Größenlimit                          → 429 / 413
   │  6. Router wählt lokales Modell („nova-auto“)        → 503 wenn keins verfügbar
   ▼
Antwort + Metadaten (model, routing.category, request_id)
   ▼
HQ-Backend filtert auf ALLOWED_METADATA und antwortet dem Browser
```

## Einrichtung

1. Auf dem Rechner, auf dem NOVA läuft, einen Integrationsschlüssel mit minimalen Rechten
   erzeugen – der Schlüssel wird **nur einmal** angezeigt und von NOVA nur gehasht gespeichert:

   ```powershell
   nova integrations create --name ic-ware-hq --scope chat:complete --scope models:read
   ```

   Weitere Optionen: `--rate-limit 30`, `--max-request-bytes 200000`, `--model <name>`
   (Modelle einschränken), `--expires-at 2027-01-01T00:00:00+00:00`.
   Widerrufen: `nova integrations revoke ic-ware-hq` · Rotieren: `nova integrations rotate ic-ware-hq`.

2. Auf dem HQ-Server (nicht im Frontend, nicht im Repository) setzen:

   ```
   NOVA_BASE_URL=http://127.0.0.1:8765
   NOVA_API_KEY=<Schlüssel aus Schritt 1>
   ```

3. Beispiel starten: `uvicorn hq_backend:app --port 9000` (im Verzeichnis dieses Beispiels).

## Scopes

| Scope | Erlaubt | Für HQ-Chat nötig? |
|---|---|---|
| `chat:complete` | Chat-Completions (`/api/v1`, `/v1`) | ja |
| `models:read` | Modellliste/-status | optional |
| `agent:run` | Agent-Aufgaben starten (ohne Werkzeuge) | nein |
| `agent:read` / `agent:cancel` | eigene Aufgaben lesen / abbrechen | nein |
| `agent:tool:<name>` | einzelnes Agent-Werkzeug freigeben (privilegiert, explizit) | nein |
| `system:read` | Systemstatus (ohne lokale Pfade) | nein |

Der Chat-Endpunkt führt **keine** Werkzeuge oder Shell-Befehle aus (Parameter `tools` wird mit
400 abgelehnt). Agent-Werkzeuge laufen nur in Agent-Aufgaben, nur wenn der Schlüssel den
passenden `agent:tool:<name>`-Scope hat **und** die Aufgabe das Werkzeug ausdrücklich anfordert;
Werkzeuge mit Bestätigungspflicht werden über die API grundsätzlich verweigert.

## Fehlerbehandlung im Beispiel

| NOVA | Client-Ausnahme | HQ antwortet dem Browser |
|---|---|---|
| 401 | `NovaAuthError` | 502 „misconfigured (ref …)“ – Details nur im Server-Log |
| 403 | `NovaPermissionError` | 502 wie oben |
| 429 | `NovaRateLimitError` (nach Wiederholung gem. `Retry-After`) | 429 |
| 503 | `NovaUnavailableError` (kein Modell) | 503 |
| 504 / Zeitüberschreitung | `NovaTimeoutError` | 504 |
| nicht erreichbar | `NovaConnectionError` | 503 |

## Wo läuft was? (Netzwerkszenarien)

NOVA rechnet **lokal auf dem Rechner, auf dem NOVA installiert ist** – mit dessen Modellen und
dessen Hardware. Daraus folgt:

| Szenario | Funktioniert? | Bedingungen |
|---|---|---|
| HQ-Backend läuft auf **demselben PC** wie NOVA | ja | `NOVA_BASE_URL=http://127.0.0.1:8765` (Standard, nur Loopback) |
| anderes Programm/Prozess auf demselben PC | ja | gleiche Loopback-Adresse, eigener Schlüssel pro Programm |
| HQ-Backend auf **einem anderen Rechner im LAN** | nur nach bewusster Konfiguration | NOVA mit `--host <LAN-IP> --allow-remote --tls-cert … --tls-key …` und `NOVA_API_TOKEN` starten, Firewall-Regel; `NOVA_BASE_URL=https://<host>:8765` |
| HQ auf einem **entfernten Server** (Rechenzentrum/Cloud), NOVA auf dem PC des Benutzers | **nicht direkt** | der entfernte Server kann den PC des Benutzers normalerweise nicht erreichen (NAT/Firewall) und es gibt dort kein lokales Modell. Möglich wären: NOVA zusätzlich auf dem Server installieren (Modelle laufen dann dort), oder ein vom Benutzer bewusst eingerichteter Tunnel/VPN. Ein Relay-Dienst ist **nicht** Teil von NOVA. |

Standardmäßig lauscht NOVA nur auf `127.0.0.1` und weist Anfragen von anderen Adressen mit 403
ab. CORS ist für die Integrations-API ausgeschaltet (kein Browser-Zugriff fremder Seiten), solange
nicht ausdrücklich `--cors-origin` gesetzt wird – und auch dann gehört der Schlüssel nicht in den
Browser.

## Andere Programme anbinden

- **OpenAI-kompatible Programme/SDKs**: Base URL `http://127.0.0.1:8765/v1`, API-Key = NOVA-Schlüssel,
  Modell `nova-auto` oder ein Name aus `GET /v1/models`. Nur Chat-Completions und Modellliste sind
  umgesetzt – Einschränkungen in `docs/integration-api.md`.
- **Eigene Programme**: `nova_client.py` kopieren oder die HTTP-Aufrufe nachbauen.
