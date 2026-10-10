"""Verwaltung von Research-Läufen im Core-Dienst: starten, auflisten, abbrechen, fortsetzen.

Läufe laufen als Hintergrund-Tasks (höchstens ``max_parallel`` gleichzeitig, weitere warten),
damit der Chat nicht blockiert wird. Nach einem Neustart des Core gelten laufende Läufe als
``paused`` und können fortgesetzt werden – der Checkpoint enthält den vollständigen Zustand.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from research.engine import TERMINAL, Budget, ResearchRun, RunState, write_report
from research.fetch import Fetcher
from research.knowledge import KnowledgeStore
from research.llm import ResearchModel
from research.search import SearchError, SearchProvider, provider_from_config

log = logging.getLogger("nova.research")

DEFAULT_CONFIG: dict[str, Any] = {
    "search": {"provider": "none"},
    "fetch": {"min_interval_s": 2.0, "respect_robots": True, "max_bytes": 3_000_000},
}


class ResearchError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def summary(state: RunState) -> dict[str, Any]:
    unique = [s for s in state.sources if not s.duplicate_of]
    return {
        "id": state.id,
        "objective": state.objective,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "active_seconds": round(state.active_seconds, 1),
        "budget": {
            "max_duration_s": state.budget.max_duration_s,
            "max_sources": state.budget.max_sources,
        },
        "model": state.model,
        "search_provider": state.provider,
        "counts": {
            "subquestions": len(state.subquestions),
            "queued_queries": len(state.queue),
            "searches": state.searches,
            "fetches": state.fetches,
            "sources": len(unique),
            "claims": len(state.claims),
            "findings": len(state.findings),
            "errors": len(state.errors),
        },
    }


class ResearchManager:
    def __init__(
        self,
        root: Path,
        *,
        knowledge_path: Path,
        config_path: Path,
        model_factory: Callable[[], Awaitable[ResearchModel | None]],
        max_parallel: int = 1,
        fetcher_factory: Callable[[dict[str, Any]], Fetcher] | None = None,
        provider_factory: Callable[[dict[str, Any]], SearchProvider | None] | None = None,
    ) -> None:
        self.root = root
        self.config_path = config_path
        self.knowledge = KnowledgeStore(knowledge_path)
        self.model_factory = model_factory
        self.fetcher_factory = fetcher_factory or (
            lambda cfg: Fetcher(
                min_interval_s=float(cfg.get("min_interval_s", 2.0)),
                respect_robots=bool(cfg.get("respect_robots", True)),
                max_bytes=int(cfg.get("max_bytes", 3_000_000)),
            )
        )
        self.provider_factory = provider_factory or provider_from_config
        self._slots = asyncio.Semaphore(max_parallel)
        self._active: dict[str, ResearchRun] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._mark_interrupted()

    # ------------------------------------------------------------------ Konfiguration

    def config(self) -> dict[str, Any]:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        merged: dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
        for key in merged:
            if isinstance(data.get(key), dict):
                merged[key].update(data[key])
        return merged

    def set_config(self, search: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config()
        candidate = {k: v for k, v in search.items() if k in ("provider", "url", "api_key_env")}
        if "api_key" in search:
            raise ResearchError(
                "invalid_config",
                "Do not store API keys in the config – use an environment variable",
            )
        try:
            self.provider_factory(candidate)
        except SearchError as exc:
            raise ResearchError("invalid_config", str(exc)) from exc
        cfg["search"] = candidate
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        os.replace(tmp, self.config_path)
        return cfg

    def status(self) -> dict[str, Any]:
        cfg = self.config()
        try:
            provider = self.provider_factory(cfg["search"])
            problem = None
        except SearchError as exc:
            provider, problem = None, str(exc)
        return {
            "search_provider": provider.name if provider else None,
            "search_problem": problem,
            "respect_robots": cfg["fetch"].get("respect_robots", True),
            "active_runs": list(self._active),
        }

    # ------------------------------------------------------------------ Läufe

    def _mark_interrupted(self) -> None:
        for path in self.root.glob("*/state.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("status") in ("running", "queued"):
                data["status"] = "paused"
                data["stop_reason"] = "interrupted (NOVA was restarted) – resume to continue"
                path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def _dir(self, run_id: str) -> Path:
        if not run_id.replace("-", "").isalnum() or "/" in run_id or ".." in run_id:
            raise ResearchError("not_found", f"Research run {run_id} not found", 404)
        path = self.root / run_id
        if not (path / "state.json").is_file():
            raise ResearchError("not_found", f"Research run {run_id} not found", 404)
        return path

    def _state(self, run_id: str) -> RunState:
        if run_id in self._active:
            return self._active[run_id].state
        data = json.loads((self._dir(run_id) / "state.json").read_text(encoding="utf-8"))
        return RunState.from_dict(data)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*/state.json"), reverse=True):
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                runs.append(summary(self._state(path.parent.name)))
        return runs

    def get(self, run_id: str) -> dict[str, Any]:
        state = self._state(run_id)
        data = summary(state)
        data["subquestions"] = [{"id": q.id, "question": q.question} for q in state.subquestions]
        data["plan_note"] = state.plan_note
        data["errors"] = state.errors[-20:]
        return data

    def report(self, run_id: str) -> dict[str, Any]:
        path = self._dir(run_id)
        md = path / "report.md"
        if not md.is_file():
            write_report(path, self._state(run_id))
        return {
            "markdown": md.read_text(encoding="utf-8"),
            "report": json.loads((path / "report.json").read_text(encoding="utf-8")),
        }

    async def _components(self) -> tuple[ResearchModel | None, SearchProvider | None, Fetcher]:
        cfg = self.config()
        try:
            provider = self.provider_factory(cfg["search"])
        except SearchError as exc:
            raise ResearchError("invalid_config", str(exc)) from exc
        return await self.model_factory(), provider, self.fetcher_factory(cfg["fetch"])

    async def start(self, objective: str, budget: Budget, seed_urls: list[str]) -> dict[str, Any]:
        model, provider, fetcher = await self._components()
        if provider is None and not seed_urls:
            await fetcher.aclose()
            raise ResearchError(
                "no_search_provider",
                "No web search provider is configured. Configure SearXNG or Brave, "
                "or give seed URLs to research.",
            )
        try:
            run = ResearchRun.create(
                self.root, objective, budget, seed_urls=seed_urls, model=model,
                provider=provider, fetcher=fetcher, knowledge=self.knowledge,
            )  # fmt: skip
        except ValueError as exc:
            await fetcher.aclose()
            raise ResearchError("invalid_request", str(exc)) from exc
        self._launch(run)
        return summary(run.state)

    async def resume(self, run_id: str) -> dict[str, Any]:
        if run_id in self._active:
            raise ResearchError("already_running", "Research run is already running", 409)
        state = self._state(run_id)
        if state.status == "completed":
            raise ResearchError("already_completed", "Research run is already completed", 409)
        model, provider, fetcher = await self._components()
        run = ResearchRun.load(
            self._dir(run_id), model=model, provider=provider, fetcher=fetcher,
            knowledge=self.knowledge,
        )  # fmt: skip
        if run.state.status == "budget_exhausted":
            raise ResearchError(
                "budget_exhausted", "Budget used up – start a new run with a larger budget", 409
            )
        run.state.status = "paused"
        self._launch(run)
        return summary(run.state)

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self._active.get(run_id)
        if run is None:
            state = self._state(run_id)
            if state.status in TERMINAL:
                return summary(state)
            raise ResearchError("not_running", "Research run is not running", 409)
        run.cancel()
        return summary(run.state)

    def _launch(self, run: ResearchRun) -> None:
        run.state.status = "queued" if self._slots.locked() else run.state.status
        self._active[run.state.id] = run

        async def go() -> None:
            try:
                async with self._slots:
                    if run.cancel_event.is_set():
                        if run.pausing:
                            run.state.status = "paused"
                            run.state.stop_reason = "paused before start – resume to continue"
                        else:
                            run.state.status = "cancelled"
                            run.state.stop_reason = "cancelled before start"
                        run.checkpoint()
                        write_report(run.dir, run.state)
                        return
                    if run.state.status == "queued":
                        run.state.status = "paused"
                    await run.run()
            except Exception:
                log.exception("research run %s crashed", run.state.id)
            finally:
                await run.fetcher.aclose()
                self._active.pop(run.state.id, None)
                self._tasks.pop(run.state.id, None)

        self._tasks[run.state.id] = asyncio.create_task(go())

    async def wait(self, run_id: str, timeout_s: float | None = None) -> dict[str, Any]:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        return summary(self._state(run_id))

    async def shutdown(self) -> None:
        """Laufende Läufe pausieren (fortsetzbar), nicht abbrechen."""
        for run in list(self._active.values()):
            run.pause()
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.wait(tasks, timeout=15)
        self.knowledge.close()
