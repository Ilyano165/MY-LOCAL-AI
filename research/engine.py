"""Research-Lauf: Arbeitszyklus mit Budget, Checkpoints, Abbruch/Fortsetzen, Audit und Bericht.

Zyklus: nächste Suchanfrage → Treffer abrufen (robots, Rate Limit) → deduplizieren → Quelle
bewerten → Aussagen extrahieren (wörtliches Zitat geprüft) → speichern → Checkpoint.
Ist die Warteschlange leer: Abgleich (gleiche/widersprüchliche Aussagen), bei Bedarf neue
Suchanfragen für schwach belegte Teilfragen, sonst Abschluss mit Bericht.

Alles, was ein Lauf erzeugt, liegt in ``<Daten>/research/<run_id>/``:
``state.json`` (Checkpoint), ``audit.jsonl``, ``sources/<id>.txt`` (Text-Snapshots),
``report.md`` / ``report.json``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research import analysis
from research.analysis import Claim, Finding, SubQuestion, normalize_query
from research.fetch import Fetcher, FetchError
from research.knowledge import KnowledgeStore
from research.llm import ResearchModel
from research.scoring import assess
from research.search import SearchError, SearchProvider
from research.sources import (
    Source,
    canonical_url,
    content_hash,
    registrable_domain,
    shingles,
    similarity,
)

NEAR_DUPLICATE = 0.85
MIN_SOURCES_PER_SUBQUESTION = 2
TERMINAL = ("completed", "cancelled", "failed", "budget_exhausted")


@dataclass
class Budget:
    max_duration_s: float = 3600.0
    max_sources: int = 30
    max_searches: int = 20
    max_fetches: int = 60
    results_per_query: int = 5
    max_replans: int = 3

    def validate(self) -> None:
        if not 10 <= self.max_duration_s <= 24 * 3600:
            raise ValueError("max_duration_s must be between 10 s and 24 h")
        for name in ("max_sources", "max_searches", "max_fetches", "results_per_query"):
            if not 1 <= getattr(self, name) <= 1000:
                raise ValueError(f"{name} must be between 1 and 1000")


@dataclass
class RunState:
    id: str
    objective: str
    budget: Budget
    seed_urls: list[str] = field(default_factory=list)
    status: str = "created"
    created_at: str = ""
    updated_at: str = ""
    active_seconds: float = 0.0  # verbrauchte Laufzeit (über Fortsetzungen hinweg)
    plan_note: str = ""
    model: str | None = None
    provider: str | None = None
    subquestions: list[SubQuestion] = field(default_factory=list)
    queue: list[list[str]] = field(default_factory=list)  # [subquestion_id, query]
    used_queries: list[str] = field(default_factory=list)
    seen_urls: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    rejected_claims: int = 0
    searches: int = 0
    fetches: int = 0
    replans: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [s.to_dict() for s in self.sources]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        state = cls(data["id"], data["objective"], Budget(**data["budget"]))
        for key in (
            "seed_urls", "status", "created_at", "updated_at", "active_seconds", "plan_note",
            "model", "provider", "queue", "used_queries", "seen_urls", "rejected_claims",
            "searches", "fetches", "replans", "errors", "stop_reason",
        ):  # fmt: skip
            if key in data:
                setattr(state, key, data[key])
        state.subquestions = [SubQuestion(**q) for q in data.get("subquestions", [])]
        state.sources = [Source.from_dict(s) for s in data.get("sources", [])]
        state.claims = [Claim(**c) for c in data.get("claims", [])]
        state.findings = [Finding(**f) for f in data.get("findings", [])]
        return state


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now(UTC).strftime("r%Y%m%d-%H%M%S-") + secrets.token_hex(3)


class ResearchRun:
    def __init__(
        self,
        run_dir: Path,
        state: RunState,
        *,
        model: ResearchModel | None,
        provider: SearchProvider | None,
        fetcher: Fetcher,
        knowledge: KnowledgeStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dir = run_dir
        self.state = state
        self.model = model
        self.provider = provider
        self.fetcher = fetcher
        self.knowledge = knowledge
        self.clock = clock
        self.cancel_event = asyncio.Event()
        self._pausing = False
        self._started: float | None = None
        self._base_seconds = state.active_seconds
        (self.dir / "sources").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Erzeugen/Laden

    @classmethod
    def create(
        cls,
        root: Path,
        objective: str,
        budget: Budget,
        *,
        seed_urls: list[str] | None = None,
        **kwargs: Any,
    ) -> ResearchRun:
        objective = " ".join(objective.split())
        if not 5 <= len(objective) <= 2000:
            raise ValueError("objective must be 5–2000 characters")
        budget.validate()
        urls = [u for u in (seed_urls or []) if u.startswith(("http://", "https://"))]
        run_id = new_run_id()
        state = RunState(run_id, objective, budget, seed_urls=urls[:50])
        state.created_at = state.updated_at = _now()
        run = cls(root / run_id, state, **kwargs)
        run.checkpoint()
        run.audit("run_created", objective=objective, budget=asdict(budget), seed_urls=urls[:50])
        return run

    @classmethod
    def load(cls, run_dir: Path, **kwargs: Any) -> ResearchRun:
        data = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        return cls(run_dir, RunState.from_dict(data), **kwargs)

    # ------------------------------------------------------------------ Persistenz

    def checkpoint(self) -> None:
        self.state.updated_at = _now()
        if self._started is not None:
            self.state.active_seconds = self._base_seconds + (self.clock() - self._started)
        tmp = self.dir / "state.json.tmp"
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.state.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, self.dir / "state.json")

    def audit(self, event: str, **fields: Any) -> None:
        line = json.dumps({"ts": _now(), "event": event, **fields}, ensure_ascii=False)
        with (self.dir / "audit.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _error(self, stage: str, detail: str, **fields: Any) -> None:
        self.state.errors.append({"stage": stage, "detail": detail[:500], "ts": _now()})
        self.audit("error", stage=stage, detail=detail[:500], **fields)

    # ------------------------------------------------------------------ Budget

    def elapsed(self) -> float:
        running = self.clock() - self._started if self._started is not None else 0.0
        return self._base_seconds + running

    def _budget_left(self) -> str | None:
        b, s = self.state.budget, self.state
        if self.elapsed() >= b.max_duration_s:
            return "time budget used up"
        if len([x for x in s.sources if not x.duplicate_of]) >= b.max_sources:
            return "source budget reached"
        if s.fetches >= b.max_fetches:
            return "fetch budget reached"
        return None

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def pausing(self) -> bool:
        return self._pausing

    def pause(self) -> None:
        """Wie Abbrechen, aber als fortsetzbar markiert (z. B. beim Beenden des Core)."""
        self._pausing = True
        self.cancel_event.set()

    # ------------------------------------------------------------------ Ablauf

    async def run(self) -> RunState:
        s = self.state
        if s.status in TERMINAL:
            return s
        self._started = self.clock()
        resumed = s.status != "created"
        s.status = "running"
        self.audit("run_resumed" if resumed else "run_started", model=self._model_name())
        try:
            if not s.subquestions:
                await self._plan()
            await self._seeds()
            await self._loop()
        except asyncio.CancelledError:
            s.status, s.stop_reason = "cancelled", "task cancelled"
            self.checkpoint()
            raise
        except Exception as exc:  # unerwartet: Lauf kontrolliert beenden, Zustand behalten
            self._error("run", f"{type(exc).__name__}: {exc}")
            s.status, s.stop_reason = "failed", f"{type(exc).__name__}: {exc}"
        await self._finish()
        return s

    def _model_name(self) -> str | None:
        return getattr(self.model, "last_model", None) if self.model else None

    async def _plan(self) -> None:
        s = self.state
        try:
            subquestions, note = await analysis.plan(self.model, s.objective)
        except Exception as exc:
            subquestions, note = analysis.fallback_plan(s.objective), f"planner failed: {exc}"
            self._error("plan", str(exc))
        s.subquestions, s.plan_note = subquestions, note
        for q in subquestions:
            for query in q.queries:
                key = normalize_query(query)
                if key not in s.used_queries:
                    s.queue.append([q.id, query])
                    s.used_queries.append(key)
        self.audit("plan", note=note, subquestions=[asdict(q) for q in subquestions])
        self.checkpoint()

    async def _seeds(self) -> None:
        s = self.state
        target = s.subquestions[0] if s.subquestions else None
        for url in list(s.seed_urls):
            if self.cancel_event.is_set() or self._budget_left() or target is None:
                return
            if canonical_url(url) not in s.seen_urls:
                await self._take(url, target, via="seed")
        s.seed_urls = []
        self.checkpoint()

    async def _loop(self) -> None:
        s = self.state
        while True:
            if self.cancel_event.is_set():
                if self._pausing:
                    s.status, s.stop_reason = "paused", "paused – resume to continue"
                else:
                    s.status, s.stop_reason = "cancelled", "cancelled by user"
                return
            reason = self._budget_left()
            if reason:
                s.status, s.stop_reason = "budget_exhausted", reason
                return
            if s.queue:
                if self.provider is None:
                    s.queue.clear()
                    self._error("search", "no search provider configured – web search skipped")
                    continue
                if s.searches >= s.budget.max_searches:
                    s.status, s.stop_reason = "budget_exhausted", "search budget reached"
                    return
                sq_id, query = s.queue.pop(0)
                await self._search(self._subquestion(sq_id), query)
                self.checkpoint()
                continue
            await self._reconcile()
            weak = [
                q for q in s.subquestions if self._source_count(q.id) < MIN_SOURCES_PER_SUBQUESTION
            ]
            if weak and self.provider is not None and s.replans < s.budget.max_replans:
                s.replans += 1
                used = set(s.used_queries)
                try:
                    new = await analysis.replan(self.model, s.objective, weak, used)
                except Exception as exc:
                    new = []
                    self._error("replan", str(exc))
                s.used_queries = sorted(used)
                self.audit("replan", weak=[q.id for q in weak], queries=new)
                if new:
                    s.queue.extend([sq, q] for sq, q in new)
                    self.checkpoint()
                    continue
            s.status, s.stop_reason = "completed", "no further queries"
            return

    def _subquestion(self, sq_id: str) -> SubQuestion:
        return next(
            (q for q in self.state.subquestions if q.id == sq_id), self.state.subquestions[0]
        )

    def _source_count(self, sq_id: str) -> int:
        return sum(1 for x in self.state.sources if sq_id in x.subquestions and not x.duplicate_of)

    async def _search(self, sq: SubQuestion, query: str) -> None:
        s = self.state
        assert self.provider is not None
        s.searches += 1
        try:
            hits = await self.provider.search(query, s.budget.results_per_query)
        except SearchError as exc:
            self._error("search", str(exc), query=query)
            return
        s.provider = self.provider.name
        self.audit(
            "search",
            query=query,
            subquestion=sq.id,
            provider=self.provider.name,
            results=[h.url for h in hits],
        )
        for hit in hits:
            if self.cancel_event.is_set() or self._budget_left():
                return
            canonical = canonical_url(hit.url)
            if canonical in s.seen_urls:
                for src in s.sources:  # bekannte Quelle: nur der Teilfrage zuordnen
                    if src.canonical == canonical and sq.id not in src.subquestions:
                        src.subquestions.append(sq.id)
                continue
            await self._take(hit.url, sq, via=f"search:{self.provider.name}")

    async def _take(self, url: str, sq: SubQuestion, *, via: str) -> None:
        s = self.state
        canonical = canonical_url(url)
        s.seen_urls.append(canonical)
        s.fetches += 1
        try:
            page = await self.fetcher.fetch(url)
        except FetchError as exc:
            self._error("fetch", str(exc), url=url, reason=exc.reason)
            return
        text = page.page.text
        digest = content_hash(text)
        source = Source(
            id=f"S{len(s.sources) + 1}",
            url=page.final_url,
            canonical=canonical_url(page.final_url),
            domain=registrable_domain(page.final_url),
            title=page.page.title[:300] or page.final_url,
            fetched_at=_now(),
            content_hash=digest,
            chars=len(text),
            subquestions=[sq.id],
            published=page.page.published,
            author=page.page.author,
            via=via,
        )
        duplicate = self._duplicate_of(source, text)
        if duplicate:
            source.duplicate_of = duplicate
            s.sources.append(source)
            self.audit("duplicate", url=page.final_url, duplicate_of=duplicate)
            return
        assess(source)
        snapshot = self.dir / "sources" / f"{source.id}.txt"
        snapshot.write_text(text, encoding="utf-8")
        source.snapshot = f"sources/{source.id}.txt"
        s.sources.append(source)
        self.audit(
            "source",
            id=source.id,
            url=source.url,
            via=via,
            hash=digest,
            bytes=page.bytes,
            kind=source.kind,
            score=source.score,
            reasons=source.score_reasons,
        )
        self.checkpoint()
        if self.model is None or len(text) < 200:
            return
        try:
            claims, rejected = await analysis.extract_claims(
                self.model, source, text, sq, len(s.claims) + 1
            )
        except Exception as exc:
            self._error("extract", f"{type(exc).__name__}: {exc}", source=source.id)
            return
        s.claims.extend(claims)
        s.rejected_claims += rejected
        s.model = self._model_name() or s.model
        self.audit(
            "claims",
            source=source.id,
            accepted=[c.id for c in claims],
            rejected=rejected,
            model=s.model,
        )

    def _duplicate_of(self, source: Source, text: str) -> str | None:
        own = shingles(text)
        for other in self.state.sources:
            if other.duplicate_of:
                continue
            if other.content_hash == source.content_hash:
                return other.id
            path = self.dir / other.snapshot if other.snapshot else None
            comparable = (
                path is not None
                and path.is_file()
                and abs(other.chars - source.chars) < 0.3 * max(other.chars, 1)
            )
            if not comparable or path is None:
                continue
            if similarity(own, shingles(path.read_text(encoding="utf-8"))) >= NEAR_DUPLICATE:
                return other.id
        return None

    async def _reconcile(self) -> None:
        s = self.state
        sources = {x.id: x for x in s.sources}
        findings: list[Finding] = []
        for q in s.subquestions:
            claims = [c for c in s.claims if c.subquestion == q.id]
            if not claims:
                continue
            try:
                same, contradictions = await analysis.compare(self.model, claims)
            except Exception as exc:
                same, contradictions = [], []
                self._error("reconcile", str(exc), subquestion=q.id)
            findings.extend(
                analysis.build_findings(claims, sources, same, contradictions, len(findings) + 1)
            )
            self.audit("reconcile", subquestion=q.id, same=same, contradictions=contradictions)
        s.findings = findings
        self.checkpoint()

    async def _finish(self) -> None:
        s = self.state
        if (
            s.status in ("cancelled", "paused", "budget_exhausted", "completed")
            and s.claims
            and not s.findings
        ):
            await self._reconcile()
        self.checkpoint()
        write_report(self.dir, s)
        if self.knowledge is not None and s.findings:
            sources = {x.id: x.to_dict() for x in s.sources}
            questions = {q.id: q.question for q in s.subquestions}
            stored = self.knowledge.add_findings(
                s.id, s.objective, [f.to_dict() for f in s.findings], questions, sources
            )
            self.audit("knowledge_stored", findings=stored)
        self.audit(
            "run_finished",
            status=s.status,
            reason=s.stop_reason,
            sources=len(s.sources),
            claims=len(s.claims),
            findings=len(s.findings),
        )


# ---------------------------------------------------------------------- Bericht

LABELS = {
    "supported": "Supported facts (≥ 2 independent sources or a strong primary source)",
    "contested": "Contradictory / unresolved",
    "unverified": "Unverified (single non-primary source)",
    "hypothesis": "Hypotheses and expectations",
    "opinion": "Opinions",
}


def build_report(state: RunState) -> dict[str, Any]:
    sources = [x for x in state.sources if not x.duplicate_of]
    return {
        "run_id": state.id,
        "objective": state.objective,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "model": state.model,
        "search_provider": state.provider,
        "active_seconds": round(state.active_seconds, 1),
        "counts": {
            "searches": state.searches,
            "fetches": state.fetches,
            "sources": len(sources),
            "duplicates": len(state.sources) - len(sources),
            "claims": len(state.claims),
            "rejected_claims": state.rejected_claims,
            "findings": len(state.findings),
            "errors": len(state.errors),
        },
        "subquestions": [asdict(q) for q in state.subquestions],
        "findings": [f.to_dict() for f in state.findings],
        "sources": [x.to_dict() for x in sources],
        "errors": state.errors,
        "notes": _notes(state),
    }


def _notes(state: RunState) -> list[str]:
    notes = []
    if state.model is None:
        notes.append(
            "No language model was used: no statements were extracted or compared. "
            "Configure a local model for claim extraction."
        )
    if state.provider is None:
        notes.append("No web search provider was used (only seed URLs, if any).")
    if state.rejected_claims:
        notes.append(
            f"{state.rejected_claims} extracted statements were discarded because their quote "
            "was not found verbatim in the source or the answer was malformed."
        )
    return notes


def render_markdown(report: dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        f"# Research report – {report['objective']}",
        "",
        f"Run `{report['run_id']}` · status **{report['status']}** ({report['stop_reason']}) · "
        f"{report['active_seconds']} s · model: {report['model'] or 'none'} · "
        f"search: {report['search_provider'] or 'none'}",
        "",
        f"{c['searches']} searches · {c['fetches']} fetches · {c['sources']} sources "
        f"({c['duplicates']} duplicates removed) · {c['claims']} statements "
        f"({c['rejected_claims']} discarded) · {c['findings']} findings · {c['errors']} errors",
        "",
    ]
    for note in report["notes"]:
        lines.append(f"> {note}")
    if report["notes"]:
        lines.append("")
    questions = {q["id"]: q["question"] for q in report["subquestions"]}
    for key, label in LABELS.items():
        items = [f for f in report["findings"] if f["classification"] == key]
        if not items:
            continue
        lines += [f"## {label}", ""]
        for f in items:
            refs = ", ".join(f"[{s}]" for s in f["source_ids"])
            lines.append(
                f"- **{f['statement']}** {refs}  \n  _{questions.get(f['subquestion'], '')}_ – "
                f"{'; '.join(f['reasons'])}"
            )
        lines.append("")
    lines += ["## Sources", ""]
    for s in report["sources"]:
        date = f", published {s['published']}" if s.get("published") else ""
        lines.append(
            f"- [{s['id']}] {s['title']} – [{s['url']}]({s['url']}) "
            f"(retrieved {s['fetched_at']}{date}; "
            f"{s['kind']}, score {s['score']}: {'; '.join(s['score_reasons'])})"
        )
    if report["errors"]:
        lines += ["", "## Errors (run continued)", ""]
        lines += [f"- {e['stage']}: {e['detail']}" for e in report["errors"][:50]]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, state: RunState) -> dict[str, Any]:
    report = build_report(state)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
