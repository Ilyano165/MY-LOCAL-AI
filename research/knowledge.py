"""Knowledge Store: geprüfte Erkenntnisse mit Quellen (Wissensverbesserung ohne Training).

SQLite + FTS5. Jede Erkenntnis behält Klassifikation, Quellen (URL, Titel, Abrufzeit) und
Herkunftslauf; sie kann als veraltet markiert oder gelöscht werden. Die Anbindung an den
Chat (Retrieval mit Quellenangaben) folgt in einem eigenen Schritt.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,           -- <run_id>:<finding_id>
    run_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    subquestion TEXT NOT NULL,
    statement TEXT NOT NULL,
    classification TEXT NOT NULL,
    sources TEXT NOT NULL,         -- JSON: [{id,url,title,fetched_at,domain,score}]
    reasons TEXT NOT NULL,         -- JSON
    created_at TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(statement, subquestion, objective,
    content='findings', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
  INSERT INTO findings_fts(rowid, statement, subquestion, objective)
  VALUES (new.rowid, new.statement, new.subquestion, new.objective);
END;
CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
  INSERT INTO findings_fts(findings_fts, rowid, statement, subquestion, objective)
  VALUES ('delete', old.rowid, old.statement, old.subquestion, old.objective);
END;
"""


class KnowledgeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Zugriff aus Event-Loop und Worker-Threads → eine Verbindung, serialisiert per Lock
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def add_findings(
        self,
        run_id: str,
        objective: str,
        findings: list[dict[str, Any]],
        subquestions: dict[str, str],
        sources: dict[str, dict[str, Any]],
    ) -> int:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        added = 0
        with self.lock, self.db:
            for f in findings:
                refs = [
                    {
                        k: sources[s].get(k)
                        for k in ("id", "url", "title", "fetched_at", "domain", "score", "via")
                    }
                    for s in f["source_ids"]
                    if s in sources
                ]
                cur = self.db.execute(
                    "INSERT OR REPLACE INTO findings (id, run_id, objective, subquestion,"
                    " statement, classification, sources, reasons, created_at, stale)"
                    " VALUES (?,?,?,?,?,?,?,?,?,0)",
                    (
                        f"{run_id}:{f['id']}",
                        run_id,
                        objective,
                        subquestions.get(f["subquestion"], f["subquestion"]),
                        f["statement"],
                        f["classification"],
                        json.dumps(refs),
                        json.dumps(f["reasons"]),
                        now,
                    ),
                )
                added += cur.rowcount
        return added

    def search(
        self, query: str, *, limit: int = 10, include_stale: bool = False
    ) -> list[dict[str, Any]]:
        terms = " OR ".join(f'"{t}"' for t in query.replace('"', " ").split() if len(t) > 2)
        if not terms:
            return []
        with self.lock:
            rows = self.db.execute(
                "SELECT f.* FROM findings_fts JOIN findings f ON f.rowid = findings_fts.rowid"
                " WHERE findings_fts MATCH ? AND (? OR f.stale = 0) ORDER BY rank LIMIT ?",
                (terms, int(include_stale), limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def all_findings(self, *, include_stale: bool = True) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM findings WHERE (? OR stale = 0) ORDER BY created_at, id",
                (int(include_stale),),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM findings WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def mark_stale(self, finding_id: str, stale: bool = True) -> bool:
        with self.lock, self.db:
            cur = self.db.execute(
                "UPDATE findings SET stale = ? WHERE id = ?", (int(stale), finding_id)
            )
        return cur.rowcount == 1

    def delete(self, finding_id: str) -> bool:
        with self.lock, self.db:
            cur = self.db.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
        return cur.rowcount == 1

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["sources"] = json.loads(data["sources"])
        data["reasons"] = json.loads(data["reasons"])
        data["stale"] = bool(data["stale"])
        return data
