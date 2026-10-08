"""Der Agent-Loop: ANALYZE → PLAN → (EXECUTE → OBSERVE → VERIFY → CORRECT)* → FINALIZE.

Grundsätze:
* Der Task State wird nach jedem Phasenwechsel persistiert → jederzeit nachvollziehbar
  und per :meth:`Agent.resume` fortsetzbar.
* Der Status einer Teilaufgabe und des Gesamtergebnisses wird vom Code aus den
  Verifikationsergebnissen abgeleitet – nie aus einer Behauptung des Modells.
* Fehlschläge führen zu: Fehler im State → Ursachenanalyse → Korrekturversuch →
  nach ausgeschöpften Versuchen Neuplanung → erst dann ehrliches Scheitern.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from agents.executor import ExecutionOutcome, Executor, FailureAnalyzer, reset_interrupted
from agents.planner import HeuristicTaskAnalyzer, LLMPlanner, Planner, PlanningError, TaskAnalyzer
from agents.state import TaskStore
from agents.task import (
    Complexity,
    Constraints,
    FinalResult,
    FinalStatus,
    Phase,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
    Verdict,
)
from agents.verifier import Verifier
from models.base import ChatRequest, GenerationParams, Message, ModelError
from models.capabilities import TaskRequirements, TaskType
from models.inference import InferenceEngine
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

EventHandler = Callable[[Task, Phase, str], None]

_FINAL_SYSTEM = (
    "You are NOVA. Write the final answer for the user based ONLY on the subtask results "
    "below. The status block is authoritative and was determined by automated verification: "
    "never claim more success than it states. Mention clearly what failed or is unverified."
)


class Agent:
    def __init__(
        self,
        engine: InferenceEngine,
        tools: ToolRegistry,
        store: TaskStore,
        workspace: str | Path,
        *,
        analyzer: TaskAnalyzer | None = None,
        planner: Planner | None = None,
        executor: Executor | None = None,
        verifier: Verifier | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.engine = engine
        self.tools = tools
        self.store = store
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace {self.workspace} existiert nicht")
        self.verifier = verifier or Verifier(self.workspace)
        self.analyzer = analyzer or HeuristicTaskAnalyzer()
        self.planner = planner or LLMPlanner(
            engine, tools=tools.specs(), known_checks=self.verifier.known_checks()
        )
        self.executor = executor or Executor(engine, tools, self.workspace)
        self.failure_analyzer = failure_analyzer or FailureAnalyzer(engine)
        self.on_event = on_event

    # ------------------------------------------------------------------ öffentliche API

    async def run(self, objective: str, constraints: Constraints | None = None) -> Task:
        task = Task(objective=objective, constraints=constraints or Constraints())
        await self._save(task)
        return await self._loop(task)

    async def resume(self, task_id: str) -> Task:
        task = await self.store.load(task_id)
        if task.status.terminal:
            return task
        reset_interrupted(task)
        self._event(task, task.phase, "Task wird fortgesetzt")
        return await self._loop(task)

    # ------------------------------------------------------------------ Helfer

    async def _save(self, task: Task) -> None:
        await self.store.save(task)

    def _event(self, task: Task, phase: Phase, message: str) -> None:
        logger.info("[%s] %s: %s", task.id, phase.value, message)
        if self.on_event is not None:
            self.on_event(task, phase, message)

    async def _enter(self, task: Task, phase: Phase, message: str) -> None:
        task.phase = phase
        self._event(task, phase, message)
        await self._save(task)

    # ------------------------------------------------------------------ Schleife

    async def _loop(self, task: Task) -> Task:
        task.status = TaskStatus.RUNNING
        try:
            if task.phase == Phase.ANALYZE:
                await self._analyze(task)
            if task.phase == Phase.PLAN:
                await self._plan(task)
            if task.phase != Phase.FINALIZE:
                aborted = await self._execute_all(task)
            else:
                aborted = False
            await self._finalize(task, aborted=aborted)
        except Exception as exc:
            task.record_error(task.phase, "internal", f"{type(exc).__name__}: {exc}")
            task.status = TaskStatus.FAILED
            task.final_result = FinalResult(
                FinalStatus.FAILED, "", False, f"Interner Fehler in Phase {task.phase.value}: {exc}"
            )
            await self._save(task)
            raise
        return task

    async def _analyze(self, task: Task) -> None:
        await self._enter(task, Phase.ANALYZE, "Aufgabe wird analysiert")
        task.analysis = await self.analyzer.analyze(task)
        a = task.analysis
        task.observe(
            "agent",
            f"Analyse: typ={a.task_type.value}, komplexität={a.complexity.value}, "
            f"tools={a.requires_tools}, signale={a.signals}",
        )
        task.phase = Phase.PLAN
        await self._save(task)

    async def _plan(self, task: Task) -> None:
        await self._enter(task, Phase.PLAN, "Plan wird erstellt")
        if task.analysis is None:
            raise RuntimeError("PLAN ohne vorherige Analyse")
        if task.analysis.complexity == Complexity.SIMPLE:
            task.subtasks = [Subtask(id="s1", description=task.objective)]
            task.observe("agent", "Einfache Aufgabe: Ein-Schritt-Plan ohne Planungsmodell")
        else:
            try:
                task.subtasks = await self.planner.plan(task)
            except PlanningError as exc:
                task.record_error(Phase.PLAN, "planning", str(exc))
                task.subtasks = [Subtask(id="s1", description=task.objective)]
                task.observe("agent", "Planung fehlgeschlagen – Rückfall auf Ein-Schritt-Plan")
        task.plan_version += 1
        task.current_step = 0
        task.observe(
            "agent", "Plan: " + " | ".join(f"{s.id}: {s.description}" for s in task.subtasks)
        )
        task.phase = Phase.EXECUTE
        await self._save(task)

    async def _execute_all(self, task: Task) -> bool:
        """Arbeitet alle Teilaufgaben ab. Rückgabe: True, wenn wegen Budget abgebrochen."""
        while task.current_step < len(task.subtasks):
            sub = task.subtasks[task.current_step]
            if sub.status in (SubtaskStatus.DONE, SubtaskStatus.FAILED, SubtaskStatus.SKIPPED):
                task.current_step += 1
                continue
            blocked = [d for d in sub.depends_on if task.subtask(d).status != SubtaskStatus.DONE]
            if blocked:
                sub.status = SubtaskStatus.SKIPPED
                task.observe(
                    "agent", f"{sub.id} übersprungen: Voraussetzung {blocked} nicht erfüllt", sub.id
                )
                task.current_step += 1
                await self._save(task)
                continue
            if task.steps_taken >= task.constraints.max_steps:
                task.record_error(
                    Phase.EXECUTE,
                    "budget",
                    f"Schrittbudget erschöpft ({task.constraints.max_steps})",
                    sub.id,
                )
                return True

            sub.status = SubtaskStatus.IN_PROGRESS
            sub.attempts += 1
            task.steps_taken += 1
            await self._enter(
                task, Phase.EXECUTE, f"{sub.id} Versuch {sub.attempts}: {sub.description}"
            )
            outcome = await self.executor.execute(task, sub)

            await self._enter(
                task,
                Phase.OBSERVE,
                f"{sub.id}: {outcome.tool_calls} Tool-Aufrufe, {len(outcome.errors)} Fehler",
            )

            await self._enter(task, Phase.VERIFY, f"{sub.id} wird verifiziert")
            verdict = await self.verifier.verify(task, sub, outcome)
            sub.verdict = verdict
            self._event(task, Phase.VERIFY, f"{sub.id}: {verdict.value}")

            if verdict != Verdict.FAILED:
                sub.status = SubtaskStatus.DONE
                sub.output = outcome.output
                task.current_step += 1
                await self._save(task)
                continue

            await self._correct(task, sub, outcome)
        return False

    async def _correct(self, task: Task, sub: Subtask, outcome: ExecutionOutcome) -> None:
        await self._enter(task, Phase.CORRECT, f"{sub.id} fehlgeschlagen – Ursachenanalyse")
        feedback = await self.failure_analyzer.analyze(task, sub, outcome)
        sub.feedback.append(feedback)
        sub.output = outcome.output

        if sub.attempts < task.constraints.max_attempts_per_subtask:
            sub.status = SubtaskStatus.PENDING
            task.observe(
                "agent", f"{sub.id}: neuer Versuch mit Korrekturhinweis", sub.id, sub.attempts
            )
            await self._save(task)
            return

        if task.replans < task.constraints.max_replans:
            try:
                new_subtasks = await self.planner.replan(task, sub)
            except PlanningError as exc:
                task.record_error(
                    Phase.CORRECT, "planning", f"Neuplanung fehlgeschlagen: {exc}", sub.id
                )
            else:
                sub.status = SubtaskStatus.FAILED
                sub.superseded = True
                task.replans += 1
                task.plan_version += 1
                done_before = task.subtasks[: task.current_step]
                task.subtasks = [*done_before, sub, *new_subtasks]
                task.current_step += 1  # gescheiterte Teilaufgabe bleibt als Historie stehen
                task.observe(
                    "agent",
                    f"Neuplanung {task.replans}: alternative Teilaufgaben "
                    + ", ".join(s.id for s in new_subtasks),
                    sub.id,
                )
                await self._save(task)
                return

        sub.status = SubtaskStatus.FAILED
        task.observe(
            "agent", f"{sub.id} endgültig fehlgeschlagen nach {sub.attempts} Versuchen", sub.id
        )
        task.current_step += 1
        await self._save(task)

    # ------------------------------------------------------------------ FINALIZE

    @staticmethod
    def _determine_status(task: Task, aborted: bool) -> tuple[FinalStatus, list[str], list[str]]:
        # Durch Neuplanung ersetzte Teilaufgaben bleiben als Historie im State, maßgeblich
        # für das Ergebnis ist aber der aktuelle Plan.
        relevant = [s for s in task.subtasks if not s.superseded]
        failed = [
            s.id for s in relevant if s.status in (SubtaskStatus.FAILED, SubtaskStatus.SKIPPED)
        ]
        unverified = [
            s.id for s in relevant if s.status == SubtaskStatus.DONE and s.verdict != Verdict.PASSED
        ]
        done = [s for s in relevant if s.status == SubtaskStatus.DONE]
        open_ = [
            s for s in relevant if s.status in (SubtaskStatus.PENDING, SubtaskStatus.IN_PROGRESS)
        ]
        if aborted or open_:
            return FinalStatus.ABORTED, unverified, failed + [s.id for s in open_]
        if not relevant or (not done):
            return FinalStatus.FAILED, unverified, failed
        if failed:
            return FinalStatus.PARTIAL, unverified, failed
        return (FinalStatus.UNVERIFIED if unverified else FinalStatus.SUCCESS), unverified, failed

    async def _finalize(self, task: Task, *, aborted: bool) -> None:
        await self._enter(task, Phase.FINALIZE, "Ergebnis wird zusammengefasst")
        status, unverified, failed = self._determine_status(task, aborted)
        done = [s for s in task.subtasks if s.status == SubtaskStatus.DONE]
        summary = (
            f"Status: {status.value}. {len(done)} Teilaufgabe(n) erledigt, davon "
            f"{len(done) - len([s for s in done if s.id in unverified])} verifiziert"
            + (f"; nicht verifiziert: {', '.join(unverified)}" if unverified else "")
            + (f"; fehlgeschlagen/offen: {', '.join(failed)}" if failed else "")
            + f". Fehler im Verlauf: {len(task.errors)}."
        )
        answer = await self._synthesize(task, status, summary)
        task.final_result = FinalResult(
            status=status,
            answer=answer,
            verified=status == FinalStatus.SUCCESS,
            summary=summary,
            unverified_subtasks=unverified,
            failed_subtasks=failed,
        )
        task.status = {
            FinalStatus.SUCCESS: TaskStatus.COMPLETED,
            FinalStatus.UNVERIFIED: TaskStatus.COMPLETED,
            FinalStatus.ABORTED: TaskStatus.ABORTED,
        }.get(status, TaskStatus.FAILED)
        self._event(task, Phase.FINALIZE, summary)
        await self._save(task)

    async def _synthesize(self, task: Task, status: FinalStatus, summary: str) -> str:
        results = "\n".join(
            f"- {s.id} [{s.status.value}/{(s.verdict or Verdict.UNVERIFIED).value}] "
            f"{s.description}: {s.output[:1500]}"
            for s in task.subtasks
        )
        fallback = f"{summary}\n\n{results}"
        if len(task.subtasks) == 1 and task.subtasks[0].status == SubtaskStatus.DONE:
            return task.subtasks[0].output  # keine zusätzliche Umformulierung nötig
        request = ChatRequest(
            messages=(
                Message.system(_FINAL_SYSTEM),
                Message.user(
                    f"Objective:\n{task.objective}\n\nStatus:\n{summary}\n\n"
                    f"Subtask results:\n{results}"
                ),
            ),
            params=GenerationParams(temperature=0.2),
        )
        try:
            result = await self.engine.chat(
                request, task=TaskRequirements(task_type=TaskType.GENERAL), fallback=True
            )
        except ModelError as exc:
            task.record_error(Phase.FINALIZE, "model", f"Zusammenfassung nicht möglich: {exc}")
            return fallback
        return result.response.message.content.strip() or fallback
