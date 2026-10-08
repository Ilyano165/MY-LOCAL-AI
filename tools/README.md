# tools/

Kontrollierter Zugriff des Agenten auf den lokalen Rechner. **Jeder** Aufruf läuft über
`ToolRegistry.execute(ToolInvocation(tool, arguments, reason), ctx)` und liefert ein
strukturiertes Ergebnis:

```json
{"success": true,  "output": "...", "error": null,  "metadata": {"tool": "...", "duration_ms": 3.1, "...": "..."}}
{"success": false, "output": null,  "error": "...", "metadata": {"...": "..."}}
```

| Datei | Inhalt |
|---|---|
| `base.py` | `ToolResult`, `ToolInvocation` (Tool + Argumente + **Begründung**), `Permission`, `ToolContext` (erlaubte Verzeichnisse, Sperrliste, Symlink-Schutz), Schema-Validierung, sichere Prozessumgebung |
| `registry.py` | Ausführungs-Pipeline, `PermissionPolicy`, Bestätigung (`ApprovalHandler`), Audit-Log (`JsonlAuditLog`, `MemoryAuditLog`) mit Redaction |
| `filesystem.py` | `list_directory`, `read_file`, `write_file`, `edit_file`, `search_files` |
| `terminal.py` | `execute_command` |
| `git.py` | `git_status`, `git_diff`, `git_log` |

## Ablauf eines Aufrufs

1. Tool nachschlagen → 2. `reason` vorhanden? → 3. Argumente gegen JSON-Schema →
4. benötigte Berechtigungen (je nach Argumenten) → 5. Policy: ALLOW / ASK / DENY
(ASK → `ApprovalHandler`; ohne Handler verweigert) → 6. Ausführung mit hartem Timeout →
7. Ausgabe begrenzen → 8. Audit-Eintrag (auch für verweigerte/fehlerhafte Aufrufe).

## Sicherheitsmechanismen

| Anforderung | Umsetzung |
|---|---|
| Erlaubte Verzeichnisse | `ToolContext.workspace` + `extra_roots`; Prüfung nach Symlink-Auflösung; Sperrliste (`.env*`, `.ssh/`, Schlüsseldateien …); kein Schreiben in `.git/` |
| Timeouts | Registry-Timeout je Tool; `execute_command` beendet bei Timeout die ganze Prozessgruppe; Git 30 s je Aufruf |
| Ausgabegrenzen | Registry kürzt Ausgaben (`max_output_chars`); Terminal liest höchstens `max_output_bytes` (Rest wird verworfen, Prozess blockiert nicht); Dateien > 1 MB werden nicht gelesen |
| Fehlerbehandlung | Jede Ausnahme → `ToolResult.fail(...)`; nie eine Exception zum Agenten |
| Berechtigungen | `READ`, `WRITE`, `EXECUTE_ALLOWLISTED` → erlaubt; `EXECUTE`, `DANGEROUS` → Bestätigung; `NETWORK` → verweigert (alles konfigurierbar) |
| Audit | JSON-Lines: Zeit, Task/Subtask, Tool, Argumente (redigiert, lange Inhalte nur als Hash), Begründung, Berechtigungen, Entscheidung, Erfolg, Fehler, Dauer |

### execute_command
Keine Shell: Argumentliste oder einfacher String; Shell-Operatoren (`|`, `&&`, `;`, `>` …)
außerhalb von Anführungszeichen werden abgelehnt. Klassifizierung: gesperrt (`sudo`, `mkfs`, `dd` …),
Netzwerk (`curl`, `git push`, `pip install` …), gefährlich (Shells, `rm`, `python -c` …),
Allowlist (`python -m pytest`, `ruff check`, `ls` …), sonst `EXECUTE`. Kein stdin, Umgebung ohne
Secrets.

### Git-Tools
Auch lesende Git-Befehle können über die Repo-Konfiguration Code ausführen (`core.fsmonitor`,
`diff.external`, Textconv, `filter.*`; vgl. CVE-2026-45033). Schutz: sichere `-c`-Overrides,
`--no-ext-diff --no-textconv`, `safe.bareRepository=explicit` **und** Verweigerung, wenn die
repo-lokale Konfiguration ausführbare Einträge enthält. Repository-Wurzel muss in einem erlaubten
Verzeichnis liegen.
