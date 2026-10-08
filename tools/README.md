# tools/

Tool-System. **Minimalstand** für den Agent Core; Ausbau (Shell mit Allowlist,
Bestätigungsdialoge, Permission-Policy) in Phase 4 gemäß `docs/security.md`.

| Datei | Inhalt |
|---|---|
| `base.py` | `Tool` (ABC), `ToolRegistry.execute`: JSON-Schema-Validierung, Timeout, Output-Kürzung, jede Ausnahme → fehlgeschlagenes `ToolOutput` |
| `filesystem.py` | `read_file`, `write_file`, `list_dir` – strikt im Workspace (`..`, absolute Pfade und Symlink-Ausbrüche werden abgelehnt) |
