# NOVA – Sicherheitskonzept

Status: Entwurf v0.1 · Stand: 2026-10-08

NOVA führt Aktionen auf dem eigenen Rechner aus (Dateien schreiben, Befehle ausführen).
Ein Sprachmodell ist dabei **kein vertrauenswürdiger Akteur**: Es kann Fehler machen und
durch manipulierte Inhalte (Prompt Injection) gesteuert werden. Die Sicherheit darf daher
nie davon abhängen, dass das Modell „sich richtig verhält".

---

## 1. Bedrohungsmodell

| # | Bedrohung | Beispiel | Hauptmaßnahme |
|---|-----------|----------|---------------|
| T1 | Prompt Injection über Inhalte | Dokument im RAG-Index / Datei / Webseite enthält „lösche alle Dateien" | Inhalte sind Daten, nie Anweisungen; Permissions + Bestätigung unabhängig vom Modell |
| T2 | Fehlerhafte/zerstörerische Tool-Nutzung | Modell überschreibt falsche Datei, `rm -rf` | Workspace-Bindung, Allowlist, Bestätigung, Backups/Git |
| T3 | Ausbruch aus dem Workspace | `../../.ssh/id_rsa`, Symlinks | Pfadauflösung + Prüfung, Symlink-Policy |
| T4 | Secret-Leckage | API-Keys in Logs, Prompts, Traces, Git | Redaction, keine Secrets im Kontext, Secret-Scan |
| T5 | Unerwünschter Netzwerkzugriff | Datenabfluss über `curl` | Netzwerk standardmäßig aus, eigene Permission |
| T6 | Bösartige Modelldateien | Pickle-basierte Gewichte führen Code aus | nur GGUF/safetensors, Checksummen, offizielle Quellen |
| T7 | Supply Chain (Python-Pakete) | kompromittiertes Paket | Lockfile mit Hashes, minimale Abhängigkeiten, Updates bewusst |
| T8 | Fremdzugriff auf lokale API/UI | andere Prozesse/Geräte im LAN steuern NOVA | Bindung an `127.0.0.1`, Token-Auth, CSRF-Schutz |
| T9 | Ressourcen-Erschöpfung | Endlosschleife, riesige Ausgaben | Budgets, Timeouts, Output-Limits, Schleifenerkennung |

## 2. Berechtigungsmodell für Tools

Jedes Tool deklariert seine Berechtigungen (`core.Permission`). Die `PermissionPolicy`
entscheidet zur Laufzeit – **im Code, nicht im Prompt**.

| Permission | Bedeutung | Default |
|-----------|-----------|---------|
| `READ_ONLY` | Lesen innerhalb freigegebener Pfade | erlaubt |
| `WORKSPACE_WRITE` | Schreiben/Ändern im Workspace | erlaubt mit Diff-Anzeige; konfigurierbar „bestätigen" |
| `EXECUTE` | Prozesse starten | **bestätigen**, außer Befehl steht auf der Allowlist |
| `NETWORK` | ausgehende Verbindungen | **aus** |
| `DANGEROUS` | Löschen, Überschreiben außerhalb Git, Systemänderungen | **immer bestätigen** |

Regeln:
- Bestätigungen zeigen die **konkrete** Aktion (vollständiger Befehl, Diff, Zielpfad).
- „Für diese Sitzung erlauben" gilt nur für exakt dieselbe Aktion/Allowlist-Regel.
- Tool-Ergebnisse werden im Kontext als Daten markiert (eigene Rolle `TOOL`, klare
  Begrenzer); Anweisungen darin haben keine Autorität.
- Hochstufen von Berechtigungen kann nur der Nutzer, nie das Modell.

## 3. Dateisystem

- Jede Sitzung hat ein `workspace_root`. Pfade werden mit `Path.resolve()` aufgelöst und
  müssen danach innerhalb des Roots liegen (Prüfung **nach** Symlink-Auflösung).
- Sperrliste unabhängig vom Workspace: `~/.ssh`, `~/.gnupg`, Keychains, `.env`-Dateien
  (lesen nur mit expliziter Freigabe), Browser-Profile.
- Schreiboperationen in Git-Repos: Vorher-Zustand ist über Git wiederherstellbar; außerhalb
  von Git legt NOVA vor dem Überschreiben eine Sicherung in `~/.nova/backups/` an.

## 4. Befehlsausführung

- **Kein** `shell=True`; Befehle als Argumentliste über `asyncio.create_subprocess_exec`.
- Allowlist für unbedenkliche Befehle (z. B. `pytest`, `ruff`, `git status`, `git diff`);
  alles andere bestätigungspflichtig.
- Timeout, Begrenzung der Ausgabe, eingeschränkte Umgebungsvariablen (Secrets werden nicht
  an Kindprozesse vererbt), Arbeitsverzeichnis = Workspace.
- Ausbaustufe (Phase 7): Ausführung in einer Sandbox. Technik hängt vom noch offenen
  Betriebssystem ab (Linux: Container/bubblewrap; macOS: Container oder `sandbox-exec`;
  Windows: Container/WSL). Wird **nicht** vorgetäuscht – solange keine Sandbox aktiv ist,
  zeigt NOVA das deutlich an.

## 5. Secrets

- **Niemals im Quellcode, in `config/*.toml`, in Tests oder Fixtures.**
- Quellen: Umgebungsvariablen (`NOVA_*`), `.env` (gitignored), optional OS-Keyring.
  Config enthält nur den **Namen** der Variable (`api_key_env = "..."`).
- Secrets werden als `pydantic.SecretStr` geführt; ein Logging-Filter redigiert bekannte
  Secret-Werte und typische Muster (Tokens, Keys) in Logs **und** Traces.
- Secrets gelangen nie in den Modellkontext.
- `pre-commit` mit Secret-Scanner (z. B. gitleaks) ab Phase 1; CI-Prüfung später.

## 6. Netzwerk

- Kernsystem braucht nur `127.0.0.1` (Inferenz-Runtime). Kein Telemetrie-Versand.
- Lokale API/UI bindet ausschließlich an `127.0.0.1`, mit zufälligem Zugriffstoken pro
  Installation; Host-Header-Prüfung gegen DNS-Rebinding.
- Ein späteres Web-Recherche-Tool läuft unter `NETWORK`-Permission und ist standardmäßig aus.

## 7. Modelle und Abhängigkeiten

- Modelldateien nur als **GGUF oder safetensors**; keine `.bin`/`.pt`/Pickle-Formate laden.
- Herkunft (offizielles Repo) und SHA-256 werden in der Registry dokumentiert und beim
  Laden geprüft, sofern NOVA die Datei selbst verwaltet.
- Python-Abhängigkeiten: Lockfile, minimale Anzahl, keine Installation zur Laufzeit durch
  den Agenten ohne Bestätigung.

## 8. Daten & Privatsphäre

- Alle Daten (Memory, Traces, Index) liegen lokal unter `~/.nova/`.
- Nutzer kann Memory-Einträge und Traces einsehen und löschen.
- Traces enthalten Prompts und Tool-Ausgaben → gleiche Schutzstufe wie die Quelldaten;
  Aufbewahrungsdauer konfigurierbar.

## 9. Sicherheitstests (Pflicht ab Phase 4)

- Path-Traversal (`../`, absolute Pfade, Symlink nach außen) → `PermissionDenied`
- Befehl außerhalb Allowlist ohne Bestätigung → wird nicht ausgeführt
- Timeout greift, Ausgabe wird gekürzt
- Secret-Redaction in Logs und Traces
- Prompt-Injection-Fixtures im RAG-Korpus → keine Tool-Ausführung ohne Policy-Freigabe
- API lehnt Anfragen ohne Token bzw. mit fremdem Host-Header ab

## 10. Meldung und Umgang

Sicherheitsrelevante Entscheidungen werden als ADR dokumentiert. Abweichungen von diesen
Defaults sind nur per expliziter Nutzerkonfiguration möglich und werden beim Start angezeigt.
