# memory/

Mehrschichtiges Gedächtnis. Der `MemoryManager` entscheidet automatisch, **ob** und **wo**
etwas gespeichert wird – nicht jede Nachricht wird dauerhaft gespeichert.

| Schicht | Datei | Scope | Speicher | Halbwertszeit | Ablauf | Kapazität |
|---|---|---|---|---|---|---|
| Working | `working_memory.py` | Task-ID | im Prozess | 1 h | 12 h / Task-Ende | 100 |
| Session | `session_memory.py` | Session-ID | SQLite | 24 h | 7 Tage | 1000 |
| Project | `project_memory.py` | Projekt-ID (aus Pfad) | SQLite | 30 Tage | – | 5000 |
| Long-Term | `long_term_memory.py` | `global` | SQLite | 180 Tage | – | 10 000 |

Weitere Module: `base.py` (Datentypen, Store-Schnittstelle, Scoring), `sqlite_store.py`
(SQLite + FTS5, atomar, WAL), `relevance.py` (Relevanzbewertung), `retrieval.py` (Abruf über
alle Schichten), `memory_manager.py` (zentrale API).

## Was wird wo gespeichert? (`HeuristicRelevanceAssessor`)

| Eingabe | Ergebnis |
|---|---|
| Gruß, Dank, „ok“ | nicht gespeichert |
| Frage | nur Session-Protokoll |
| enthält Secret (Passwort, Token, Key …) | **nie** gespeichert; im Session-Protokoll geschwärzt |
| Präferenz („ich bevorzuge …“) / Anweisung („antworte immer …“) | Long-Term |
| Entscheidung/Konvention mit Projektbezug („wir verwenden …“) | Project |
| „Merk dir: …“ | Long-Term oder Project (je nach Projektbezug), hohe Wichtigkeit |
| gewöhnliche Aussage | nur Session |
| Tool-Ausgabe | nur Working Memory der laufenden Aufgabe |
| Aufgabenergebnis | Project (Lehren nur aus **verifiziert** erfolgreichen Aufgaben) |

Mehrsätzige Nachrichten werden satzweise bewertet („Was ist 2+2? Und antworte ab jetzt auf
Deutsch.“ → nur die Anweisung wird gespeichert). Jede Entscheidung trägt Begründungen
(`StoreResult.reasons`). Identische Inhalte werden nicht dupliziert, sondern aufgefrischt.

## Abruf (`recall`)

Score = 0.6 · Relevanz + 0.25 · Wichtigkeit + 0.15 · Aktualität
(vgl. Park et al. 2023, „Generative Agents“):
* **Relevanz:** lexikalisch, nach Spezifität gewichtet, mit Sättigung (lange Aufgabentexte
  bestrafen kurze, treffende Erinnerungen nicht); Präfix-/Kompositumtreffer zählen teilweise.
* **Aktualität:** exponentieller Zerfall seit dem letzten *Zugriff*, Halbwertszeit je Schicht.
* Mindestrelevanz: Wichtiges, aber Themenfremdes wird nicht geliefert.
* Präferenzen und Anweisungen aus Long-Term kommen immer mit (`guidance`).

Der Agent ruft bei jeder neuen Aufgabe automatisch ab; das Ergebnis steht in
`Task.recalled_memories` und als Kontextblock in den Prompts von Planner und Executor.

## Bekannte Grenzen

* Keine semantische Suche (Synonyme wie „nutzen“/„verwenden“ werden nicht erkannt) –
  Embeddings kommen mit der RAG-Phase; Schnittstelle ist dafür vorbereitet.
* Widersprüche („ich bevorzuge Tabs“ → später „… Spaces“) werden nicht automatisch erkannt;
  beide Einträge existieren, der neuere gewinnt über die Aktualität. Korrektur über `update`/`delete`.
* Die Regeln sind auf Deutsch/Englisch zugeschnitten.
