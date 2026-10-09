# NOVA Integrations-API (v1)

Stand: 2026-10-09 · Implementierung: `api/v1.py`, `api/integrations.py`, `api/agent_tasks.py` ·
Tests: `tests/api/test_v1.py`, `tests/examples/test_hq_integration.py`

NOVA bietet anderen Programmen zwei Schnittstellen über **denselben** Generierungspfad wie die
Oberfläche (`NovaService.generate()` → Router → InferenceEngine). Es gibt keine zweite Engine.

| Basis | Format | Zweck |
|---|---|---|
| `http://127.0.0.1:8765/api/v1` | NOVA-Format (Fehler mit `code`, `request_id`; Routing-Metadaten) | eigene Integrationen (z. B. IC WARE HQ) |
| `http://127.0.0.1:8765/v1` | OpenAI-kompatibel (Teilmenge, s. §6) | vorhandene OpenAI-Clients/SDKs |

Port `8765` ist der Standard (`nova serve --port`, `nova service start --port`).

## 1. Status: implementiert / getestet / nicht vorhanden

| Funktion | Status |
|---|---|
| Schlüsselverwaltung (erstellen, auflisten, widerrufen, rotieren, Scopes ändern, Ablaufdatum) | implementiert + getestet (CLI und Store) |
| Authentifizierung `Authorization: Bearer` oder `X-API-Key` | implementiert + getestet |
| Scopes, Modell-Einschränkung, Rate Limit, Größenlimit | implementiert + getestet |
| Chat-Completions (normal + SSE-Streaming), Timeout, Abbruch durch Verbindungsende | implementiert + getestet |
| OpenAI-Kompatibilität (Modelle, Chat, Streaming, Fehlerformat) | implementiert + mit dem offiziellen `openai`-Python-SDK getestet |
| Agent-Aufgaben (anlegen, abfragen, abbrechen, Zeitlimit, Werkzeugfreigabe) | implementiert + getestet (mit Test-Modell) |
| Audit-Log ohne Inhalte | implementiert + getestet |
| Netzwerkschutz (nur Loopback), restriktives CORS | implementiert + getestet |
| Tests gegen ein **echtes** Sprachmodell | **nicht ausgeführt** (kein Modell in der Testumgebung; Runtime wird simuliert) |
| Embeddings, Bilder erzeugen, Audio, Dateien, Assistants, Function Calling | **nicht vorhanden** |
| Memory/RAG über die API | **nicht vorhanden** |

## 2. Schlüssel und Berechtigungen

```powershell
nova integrations create --name ic-ware-hq --scope chat:complete --scope models:read `
     [--model <name>]... [--rate-limit 60] [--max-request-bytes 1000000] `
     [--expires-at 2027-01-01T00:00:00+00:00] [--agent-workspace C:\Pfad]
nova integrations list
nova integrations scopes ic-ware-hq --scope chat:complete
nova integrations rotate ic-ware-hq     # neuer Schlüssel, alter sofort ungültig
nova integrations revoke ic-ware-hq
nova integrations list-scopes
```

* Schlüsselformat `nova_<12 hex>_<Geheimnis>`; angezeigt **nur einmal** bei `create`/`rotate`.
* Gespeichert wird nur der SHA-256-Hash (`<Daten>/integrations.db`); Vergleich in konstanter Zeit,
  auch für unbekannte Präfixe.
* Widerrufene oder abgelaufene Schlüssel → 401.

| Scope | erlaubt |
|---|---|
| `chat:complete` | `POST /api/v1/chat/completions`, `POST /v1/chat/completions` |
| `models:read` | `GET /api/v1/models`, `GET /api/v1/models/status`, `GET /v1/models` |
| `agent:run` | `POST /api/v1/agent/tasks` (ohne Werkzeuge) |
| `agent:read` | `GET /api/v1/agent/tasks/{id}` (nur eigene Aufgaben) |
| `agent:cancel` | `POST /api/v1/agent/tasks/{id}/cancel` (nur eigene Aufgaben) |
| `agent:tool:<name>` | ein bestimmtes Agent-Werkzeug (privilegiert, muss einzeln vergeben werden) |
| `system:read` | `GET /api/v1/system/status` (ohne lokale Pfade) |

Unbekannte Scopes oder unbekannte Werkzeugnamen werden beim Anlegen abgelehnt. Ohne Scope gibt es
nur `/health` (öffentlich) und `/capabilities` (beliebiger gültiger Schlüssel).

**Agent-Werkzeuge:** Ein Werkzeug läuft nur, wenn (1) der Schlüssel `agent:tool:<name>` hat **und**
(2) die Aufgabe es in `allowed_tools` anfordert. Werkzeuge, die laut NOVA eine Bestätigung
brauchen, werden über die API **immer verweigert** (es gibt keinen Menschen, der bestätigen
könnte). Der Chat-Endpunkt führt nie Werkzeuge oder Shell-Befehle aus.

## 3. Allgemeines Verhalten

| Thema | Verhalten |
|---|---|
| Content-Type | `application/json` (Anfragen und Antworten); Streaming `text/event-stream` |
| Request-ID | jede Antwort hat `X-Request-ID`; im NOVA-Fehlerformat auch im Body |
| Größenlimit | `max_request_bytes` je Integration (Standard 1 000 000) → 413 `request_too_large` |
| Rate Limit | `rate_limit_per_minute` je Integration (gleitendes 60-s-Fenster, im Speicher) → 429 `rate_limited` mit `Retry-After` |
| Zeitlimit Chat | Server-Einstellung `request_timeout_s` (Standard 300 s, 5–3600). Anfrage kann mit `nova_timeout_s` **kürzer**, nie länger → 504 `timeout` |
| Zeitlimit Agent | `timeout_s` je Aufgabe (Standard 600 s, max. 7200) → Status `timed_out` |
| Abbruch Chat | Verbindung schließen (Streaming und normal) bricht die Generierung in der Runtime ab |
| CORS | aus; nur für ausdrücklich gesetzte `--cors-origin` erlaubt. Fremde `Origin` → 403 |
| Netzwerk | nur Loopback-Clients; andere Adressen → 403 `forbidden` (Audit-Ereignis `network_denied`), außer `--allow-remote` (s. §8) |
| Versionierung | Pfad-Präfix `/api/v1`. Innerhalb von v1 nur abwärtskompatible Ergänzungen (neue optionale Felder/Endpunkte). Inkompatible Änderungen → `/api/v2`, v1 bleibt mindestens eine Minor-Version parallel. `GET /api/v1/health` liefert `version` (NOVA) und `api_version`. |

### Fehlerformat (NOVA, `/api/v1`)

```json
{"error": {"code": "insufficient_scope", "message": "…", "detail": null, "request_id": "req_…"}}
```

### Fehlerformat (OpenAI, `/v1`)

```json
{"error": {"message": "…", "type": "permission_error", "param": null, "code": "insufficient_scope"}}
```

### Statuscodes und Codes

| HTTP | `code` | Bedeutung |
|---|---|---|
| 400 | `invalid_request` | Schema verletzt (`param` nennt das Feld) |
| 400 | `unsupported_parameter` | Parameter wird nicht unterstützt (s. §6) |
| 401 | `missing_api_key` / `invalid_api_key` | Schlüssel fehlt / falsch / widerrufen / abgelaufen (`WWW-Authenticate: Bearer`) |
| 403 | `insufficient_scope` | Scope fehlt |
| 403 | `model_not_allowed` | Modell für diese Integration nicht freigegeben |
| 403 | `tool_not_allowed` | Agent-Werkzeug nicht freigegeben oder unbekannt |
| 403 | `forbidden` | Client-Adresse (nicht Loopback) oder Browser-Origin nicht erlaubt. Diese Prüfung läuft vor der Authentifizierung und antwortet immer im NOVA-Format `{"error":{"code":"forbidden",…}}`, auch auf `/v1`. |
| 404 | `model_not_found` / `task_not_found` | unbekanntes Modell / Aufgabe (auch: Aufgabe gehört anderer Integration) |
| 413 | `request_too_large` | Größenlimit überschritten |
| 429 | `rate_limited` | Rate Limit, `Retry-After` beachten |
| 502 | `model_error` | lokale Runtime meldet Fehler |
| 503 | `no_model` | kein lokales Modell verfügbar (Setup-Modus, Runtime aus) |
| 504 | `timeout` | Zeitlimit überschritten |

OpenAI-`type`: 400 `invalid_request_error`, 401 `authentication_error`, 403 `permission_error`,
404 `not_found_error`, 413 `invalid_request_error`, 429 `rate_limit_error`, 503
`service_unavailable_error`, 504 `timeout_error`, sonst `api_error`.

## 4. Endpunkte `/api/v1`

| Methode + Pfad | Scope | Antwort |
|---|---|---|
| `GET /health` | – (öffentlich) | `{"status":"ok","version":"0.1.0","api_version":"v1"}` – sonst nichts |
| `GET /capabilities` | gültiger Schlüssel | eigene Scopes/Limits, Features (`tool_calling_via_chat: false`, `embeddings: false`, …), Anzahl verfügbarer Modelle |
| `GET /models` | `models:read` | `{"auto_model":"nova-auto","models":[…]}` (nur freigegebene Modelle) |
| `GET /models/status[?refresh=true]` | `models:read` | Erreichbarkeit der Modelle/Runtimes |
| `POST /chat/completions` | `chat:complete` | s. §5 |
| `POST /agent/tasks` | `agent:run` | 202 + `Location`, Aufgabe (s. §7) |
| `GET /agent/tasks/{id}` | `agent:read` | Aufgabe inkl. Ereignissen und Ergebnis |
| `POST /agent/tasks/{id}/cancel` | `agent:cancel` | Aufgabe (Status wird `cancelled`) |
| `GET /system/status` | `system:read` | Modus (`normal`/`setup`/`development`), Modelle, Ressourcen – ohne lokale Pfade |

## 5. Chat-Completions

Anfrage (beide Basen gleich):

```json
{
  "model": "nova-auto",
  "messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}],
  "temperature": 0.2, "top_p": 0.9, "max_tokens": 512, "stop": ["\n\n"], "seed": 1,
  "stream": false, "stream_options": {"include_usage": true},
  "nova_timeout_s": 60
}
```

* `model`: `nova-auto` = NOVA-Router wählt das lokale Modell; oder ein Name aus `GET /models`.
* Rollen: `system`, `developer` (wie system), `user`, `assistant`. Inhalt: Text oder Teile
  `{"type":"text"}` / `{"type":"image_url","image_url":{"url":"data:image/…;base64,…"}}`.
* Grenzen: bis 500 Nachrichten, `max_tokens` ≤ 131 072, `temperature` 0–2, `top_p` (0,1].

Antwort (normal):

```json
{"id": "chatcmpl-…", "object": "chat.completion", "created": 1760000000, "model": "<lokales Modell>",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 12, "completion_tokens": 40, "total_tokens": 52},
 "nova": {"routing": {"category": "…", "model": "…", "reason": "…"}, "stats": {…}}}
```

* `usage` **nur**, wenn die lokale Runtime Token-Zahlen meldet (nichts wird geschätzt).
* `nova` nur auf `/api/v1`.

Streaming (`"stream": true`), Server-Sent Events, jede Zeile `data: <JSON>`:

1. nur `/api/v1`: `{"object":"nova.routing","routing":{…}}`
2. `chat.completion.chunk` mit `delta: {"role":"assistant","content":""}`
3. `chat.completion.chunk` mit `delta: {"content":"…"}` (beliebig viele)
4. Abschluss-Chunk mit `finish_reason`
5. optional (`include_usage` und von der Runtime gemeldet): Chunk mit `choices: []` und `usage`
6. nur `/api/v1`: `{"object":"nova.stats","stats":{…}}`
7. `data: [DONE]`

Fehler **vor** dem ersten Token (Auth, Scope, kein Modell, …) kommen als normale HTTP-Fehler.
Fehler **während** des Streams kommen als `data: {"error":{…}}` gefolgt von `data: [DONE]`.

## 6. OpenAI-Kompatibilität – was geht, was nicht

Base URL `http://127.0.0.1:8765/v1`, API-Key = NOVA-Schlüssel.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key=os.environ["NOVA_API_KEY"])
client.chat.completions.create(model="nova-auto", messages=[{"role": "user", "content": "Hi"}])
```

Getestet mit dem offiziellen Python-SDK (`openai` 3.26.1): `models.list()`, `chat.completions.create`
(normal und `stream=True`), Fehlerklassen (`AuthenticationError`, `BadRequestError`, …).

**Umgesetzt:** `GET /v1/models`, `POST /v1/chat/completions` (normal + Streaming), Parameter
`model, messages, temperature, top_p, max_tokens, max_completion_tokens, stop, seed, n=1, stream,
stream_options.include_usage, user` (`user` wird ignoriert), Fehlerformat.

**Bekannte Inkompatibilitäten** (werden mit 400 `unsupported_parameter` abgelehnt statt still
ignoriert):

* `tools`, `tool_choice`, `functions`, `function_call` – kein Function/Tool Calling über Chat
* `n > 1`
* `logprobs`
* `response_format` außer `text` (kein JSON-Modus, kein Structured Output)
* Bilder nur als `data:`-URL (keine http(s)-URLs – NOVA lädt nichts aus dem Internet)
* Rollen `tool`/`function`

**Nicht vorhanden** (404 `not_found`, NOVA-Format): Embeddings, Completions (legacy), Images, Audio, Files, Batches,
Assistants/Threads, Responses-API, Moderation, Fine-Tuning.

Weitere Unterschiede: `model` in der Antwort ist das tatsächlich genutzte lokale Modell (auch bei
`nova-auto`); `usage` kann fehlen; `system_fingerprint` fehlt; Zeitlimits/Rate Limits sind NOVA-
eigene Einstellungen; Abbruch nur durch Schließen der Verbindung.

## 7. Agent-Aufgaben (nur NOVA-Format)

```json
POST /api/v1/agent/tasks
{"objective": "Fasse die Datei README.md zusammen", "allowed_tools": ["read_file"], "timeout_s": 600}
```

* `allowed_tools` leer → Agent arbeitet nur mit dem Modell, ohne Werkzeuge.
* Antwort 202, `Location: /api/v1/agent/tasks/task_…`, Body = Aufgabe.
* Status: `queued` → `running` → `succeeded` | `failed` | `cancelled` | `timed_out`.
* Ergebnis: `result.answer`, `result.verification` (`status`, `verified`, `summary`, `quality`),
  `result.models_used`. Eine unbestätigte Antwort bleibt `verified: false` – NOVA behauptet keine
  Prüfung, die nicht stattfand.
* Ereignisse: die letzten 50 (`events`). Höchstens 2 Aufgaben gleichzeitig, weitere warten.
* Aufgaben werden in `<Daten>/agent-tasks/` gespeichert; nach einem Neustart sind unterbrochene
  Aufgaben `failed` mit `error: "interrupted"`.
* Eine Integration sieht nur ihre eigenen Aufgaben (fremde → 404).

## 8. Netzwerk und Betrieb

| Szenario | Konfiguration |
|---|---|
| Programm auf demselben PC | Standard. NOVA lauscht nur auf `127.0.0.1`. |
| anderer Prozess/Benutzer auf demselben PC | erreicht Loopback ebenfalls → **Schlüssel erforderlich**, jedes Programm eigener Schlüssel |
| anderer Rechner (LAN) | nur bewusst: `nova serve --host <IP> --allow-remote --tls-cert cert.pem --tls-key key.pem` **und** Umgebungsvariable `NOVA_API_TOKEN` (zusätzliches Token für die Oberfläche). Ohne alle drei startet NOVA nicht. Firewall-Regel selbst setzen. |
| entfernter Server (Cloud/Rechenzentrum) | kann eine NOVA-Installation auf einem Benutzer-PC normalerweise **nicht** erreichen (NAT/Firewall). NOVA bietet kein Relay. Optionen: NOVA auf dem Server selbst betreiben (Modelle laufen dann dort) oder ein vom Betreiber verantworteter VPN/Tunnel mit TLS. |

Audit-Log: `<Daten>/logs/audit.jsonl` (rotiert bei 10 MB) mit Ereignis, Integration, Pfad,
Status, Request-ID – **ohne** Schlüssel, Prompts, Nachrichten oder Antworten.

## 9. Andere Programme anbinden – Kurzanleitung

1. `nova integrations create --name <programm> --scope chat:complete` (nur nötige Scopes).
2. Schlüssel serverseitig/geschützt ablegen (Umgebungsvariable, Secret-Store) – nie im Browser,
   nie im Repository.
3. OpenAI-kompatible Programme: Base URL `http://127.0.0.1:8765/v1`, Modell `nova-auto`.
   Eigene Programme: `/api/v1` (Beispielclient: `examples/ic-ware-hq-integration/nova_client.py`).
4. 503 `no_model` behandeln (NOVA läuft, aber kein Modell eingerichtet) und 429 mit `Retry-After`.
5. Schlüssel regelmäßig rotieren; nicht mehr genutzte Integrationen widerrufen.
