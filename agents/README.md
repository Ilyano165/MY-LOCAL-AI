# agents/

Agent Core: zerlegt Aufgaben in Teilschritte, führt sie mit Tools aus und **verifiziert**
jedes Ergebnis, bevor es als erledigt gilt.

```
User Task → ANALYZE → PLAN → [ EXECUTE → OBSERVE → VERIFY → (CORRECT) ]* → FINALIZE
```

| Datei | Inhalt |
|---|---|
| `task.py` | Persistenter Task State: `objective`, `constraints`, `subtasks`, `current_step`, `observations`, `tool_results`, `errors`, `verification_results`, `final_result` (+ Status, Phase, Budget-Zähler) |
| `state.py` | `TaskStore`, `JsonFileTaskStore` (atomares Schreiben), `InMemoryTaskStore` |
| `planner.py` | ANALYZE: `HeuristicTaskAnalyzer` (deterministisch). PLAN: `LLMPlanner` mit strikter JSON-Validierung, Retry mit Fehlerrückmeldung, `replan` für alternative Wege |
| `executor.py` | EXECUTE/OBSERVE: Tool-Calling-Schleife pro Teilaufgabe. CORRECT: `FailureAnalyzer` (Fakten + Modell-Ursachenanalyse) |
| `verifier.py` | VERIFY: deterministische Checks + automatische Checks aus Tool-Ergebnissen |
| `agent.py` | Der Loop, Persistenz nach jedem Phasenwechsel, `run` / `resume`, ehrliche Endbewertung |

## Regeln, die der Code erzwingt

- **Status kommt vom Verifier, nie vom Modell.** Behauptet das Modell „erledigt“, ohne dass die
  Prüfungen bestehen, gilt der Schritt als fehlgeschlagen.
- **Ohne Prüfung kein Erfolg:** Schritte ohne anwendbare Prüfung sind `unverified`; das
  Endergebnis ist dann höchstens `unverified`, nie `success`.
- **Nach Dateiänderungen:** Datei erneut lesen und per SHA-256 mit dem Geschriebenen vergleichen;
  bei `.py` Syntaxprüfung; mit `Constraints.test_command` Tests ausführen.
- **Tool-Fehler:** landen in `task.errors`, werden dem Modell mit der Aufforderung zur
  Ursachenanalyse zurückgegeben; fehlgeschlagene Schreibvorgänge ohne erfolgreiche Wiederholung
  lassen die Verifikation scheitern.
- **Korrekturkaskade:** Fehlschlag → Ursachenanalyse → neuer Versuch mit Feedback (bis
  `max_attempts_per_subtask`) → Neuplanung (bis `max_replans`) → ehrliches Scheitern.
- **Prüfbefehle aus dem Plan** (vom Modell erzeugt) laufen nur, wenn sie auf der Allowlist stehen
  (Default: `python -m pytest`, `python -m py_compile`, `pytest`); ohne Shell, im Workspace,
  mit Timeout, ohne Secrets in der Umgebung.

Endstatus (`FinalResult.status`): `success` · `unverified` · `partial` · `failed` · `aborted`.

## Verwendung

```python
agent = Agent(engine, ToolRegistry(default_tools()), JsonFileTaskStore("~/.nova/tasks"), workspace)
task = await agent.run(
    "Implementiere add() in calc.py", Constraints(test_command=["python", "-m", "pytest", "-q"])
)
print(task.final_result.status, task.final_result.answer)
task = await agent.resume(task.id)  # nach Absturz/Abbruch
```
