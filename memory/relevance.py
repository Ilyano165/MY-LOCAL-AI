"""Relevanzbewertung: *ob* und *wo* eine Information gespeichert wird.

:class:`HeuristicRelevanceAssessor` ist deterministisch, schnell und erklärbar (jede
Entscheidung trägt Begründungen). Ein modellbasierter Bewerter kann dieselbe Schnittstelle
(:class:`RelevanceAssessor`) erfüllen und z. B. Fakten extrahieren – die Routing-Regeln im
MemoryManager bleiben gleich.

Grundsatz: Im Zweifel **nicht** dauerhaft speichern. Projekt- und Langzeit-Memory bekommen nur,
was einen klaren Hinweis auf dauerhafte Relevanz hat (explizite Bitte, Präferenz, Anweisung,
Entscheidung). Secrets werden nie gespeichert.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field

from memory.base import MemoryKind, MemoryLayer, MemorySource, contains_secret, tokenize


@dataclass(frozen=True)
class MemoryContext:
    """Wer/wo: bestimmt die Scopes der Schichten."""

    session_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None


@dataclass
class Assessment:
    store: bool
    layer: MemoryLayer | None
    kind: MemoryKind
    importance: float
    content: str
    reasons: list[str] = field(default_factory=list)


class RelevanceAssessor(abc.ABC):
    @abc.abstractmethod
    def assess(self, content: str, source: MemorySource, ctx: MemoryContext) -> Assessment: ...


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_EXPLICIT = _rx(
    r"^\s*(bitte\s+)?(merk|merke)\s*(dir)?\b[:,]?",
    r"^\s*(please\s+)?remember\b( that)?[:,]?",
    r"^\s*(notier|notiere|speicher|speichere)\s*(dir|dir das)?\b[:,]?",
    r"^\s*vergiss nicht\b[:,]?",
    r"^\s*don'?t forget\b( that)?[:,]?",
    r"^\s*note that\b[:,]?",
)
_PREFERENCE = _rx(
    r"\bich (bevorzuge|mag|möchte lieber|will lieber|hasse)\b",
    r"\bi (prefer|like|love|hate|dislike)\b",
    # Deutsche Verbzweitstellung („Grundsätzlich bevorzuge ich …“); 1. Person ist eindeutig.
    r"\b(bevorzuge|präferiere)\b",
    r"\b(mag|hasse) ich\b",
    r"\bmir ist wichtig\b",
    r"\bmy preference\b",
    r"\blieber\b.*\bals\b",
)
_INSTRUCTION = _rx(
    r"\b(immer|niemals|nie)\b",
    r"\b(always|never)\b",
    r"\bab jetzt\b",
    r"\bfrom now on\b",
    r"\bantworte\b.*\b(auf|in)\b",
    r"\b(respond|answer|reply) in\b",
    r"\bin zukunft\b",
)
_DECISION = _rx(
    r"\bwir (verwenden|nutzen|benutzen|setzen auf|haben uns entschieden)\b",
    r"\b(we use|we're using|we decided|decided to|we will use)\b",
    r"\b(entschieden|entscheidung|konvention|architektur|vereinbart)\b",
    r"\b(convention|architecture decision|adr)\b",
)
_PROJECT_HINT = _rx(
    r"\b(projekt|project|repo|repository|codebase|code-?basis)\b",
    r"[\w-]+/[\w./-]+",
    r"\b[\w-]+\.(py|ts|js|rs|go|java|toml|json|ya?ml|md)\b",
)
_GLOBAL_HINT = _rx(r"\b(generell|grundsätzlich|überall|in allen projekten|in general|everywhere)\b")
_QUESTION = _rx(
    r"\?\s*$",
    r"^\s*(wie|was|warum|wieso|weshalb|wann|wo|wer|welche[rsn]?|kannst|könntest|"
    r"how|what|why|when|where|who|which|can you|could you)\b",
)
_TRIVIAL = frozenset(
    {
        "hallo",
        "hi",
        "hey",
        "danke",
        "danke schön",
        "dankeschön",
        "vielen dank",
        "ok",
        "okay",
        "ja",
        "nein",
        "gut",
        "super",
        "cool",
        "passt",
        "alles klar",
        "thanks",
        "thank you",
        "yes",
        "no",
        "great",
        "nice",
        "sure",
        "bye",
        "tschüss",
        "moin",
        "servus",
        "perfekt",
        "top",
    }
)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜa-zäöü])")
_LEADING_CONJ = re.compile(r"^(und|aber|also|außerdem|and|but|also)\b[,]?\s*", re.IGNORECASE)
_LAYER_RANK = {
    None: -1,
    MemoryLayer.WORKING: 0,
    MemoryLayer.SESSION: 1,
    MemoryLayer.PROJECT: 2,
    MemoryLayer.LONG_TERM: 2,
}


class HeuristicRelevanceAssessor(RelevanceAssessor):
    def __init__(self, *, durable_threshold: float = 0.6) -> None:
        self.durable_threshold = durable_threshold

    def assess(self, content: str, source: MemorySource, ctx: MemoryContext) -> Assessment:
        """Bewertet eine Nachricht; mehrere Sätze werden einzeln bewertet und der dauerhafteste
        gewinnt („Was ist 2+2? Und antworte ab jetzt auf Deutsch.“ → die Anweisung)."""
        if contains_secret(content):
            return self._assess_sentence(content, source, ctx)
        sentences = [s for s in _SENTENCE_SPLIT.split(content.strip()) if s.strip()]
        if len(sentences) <= 1:
            return self._assess_sentence(content, source, ctx)
        candidates = [
            self._assess_sentence(_LEADING_CONJ.sub("", s).strip(), source, ctx) for s in sentences
        ]
        best = max(candidates, key=lambda a: (_LAYER_RANK.get(a.layer, -1), a.importance))
        if best.layer in (MemoryLayer.PROJECT, MemoryLayer.LONG_TERM):
            best.reasons.append(f"aus mehrsätziger Nachricht ({len(sentences)} Sätze) übernommen")
            return best
        whole = self._assess_sentence(content, source, ctx)
        return whole

    def _assess_sentence(
        self, content: str, source: MemorySource, ctx: MemoryContext
    ) -> Assessment:
        text = content.strip()
        reasons: list[str] = []

        if contains_secret(text):
            return Assessment(
                False,
                None,
                MemoryKind.NOTE,
                0.0,
                text,
                ["enthält vermutlich ein Secret – wird nie gespeichert"],
            )
        normalized = re.sub(r"[!.?\s]+$", "", text.lower())
        if not text or normalized in _TRIVIAL or len(tokenize(text)) == 0:
            return Assessment(
                False,
                None,
                MemoryKind.MESSAGE,
                0.0,
                text,
                ["kein inhaltlicher Wert (Gruß/Bestätigung/leer)"],
            )

        if source is MemorySource.TOOL:
            # Tool-Ausgaben gehören zur laufenden Aufgabe, nicht ins Gedächtnis.
            layer = MemoryLayer.WORKING if ctx.task_id else None
            return Assessment(
                layer is not None,
                layer,
                MemoryKind.OBSERVATION,
                0.2,
                text,
                ["Tool-Ausgabe: nur für die laufende Aufgabe relevant"],
            )

        explicit = _EXPLICIT.search(text)
        if explicit:
            text = text[explicit.end() :].strip(" :,-") or text
            reasons.append("ausdrückliche Bitte, sich etwas zu merken")
        preference = bool(_PREFERENCE.search(text))
        instruction = bool(_INSTRUCTION.search(text))
        decision = bool(_DECISION.search(text))
        question = bool(_QUESTION.search(text)) and not explicit

        if question:
            reasons.append("Frage – enthält keine zu merkende Information")
            return self._session(text, MemoryKind.MESSAGE, 0.15, ctx, reasons)

        if instruction:
            kind, importance = MemoryKind.INSTRUCTION, 0.8
            reasons.append("dauerhafte Anweisung")
        elif preference:
            kind, importance = MemoryKind.PREFERENCE, 0.75
            reasons.append("Präferenz des Nutzers")
        elif decision:
            kind, importance = MemoryKind.DECISION, 0.65
            reasons.append("Entscheidung/Konvention")
        else:
            kind, importance = MemoryKind.FACT, 0.35
        if explicit:
            importance = max(importance, 0.9)
        if source is not MemorySource.USER and not explicit:
            importance -= 0.1  # Aussagen des Agenten sind weniger verbindlich als die des Nutzers
            reasons.append(f"Quelle {source.value}: geringeres Gewicht")

        if importance < self.durable_threshold:
            reasons.append("nicht dauerhaft relevant genug")
            return self._session(text, kind, max(importance, 0.1), ctx, reasons)

        project_hint = bool(_PROJECT_HINT.search(text))
        global_hint = bool(_GLOBAL_HINT.search(text))
        if ctx.project_id and (decision or project_hint) and not global_hint:
            reasons.append("projektbezogen")
            return Assessment(True, MemoryLayer.PROJECT, kind, importance, text, reasons)
        if decision and not ctx.project_id and not (preference or instruction or explicit):
            reasons.append("Entscheidung ohne Projektkontext – nur Session")
            return self._session(text, kind, importance, ctx, reasons)
        reasons.append("langfristig relevant")
        return Assessment(True, MemoryLayer.LONG_TERM, kind, importance, text, reasons)

    @staticmethod
    def _session(
        text: str, kind: MemoryKind, importance: float, ctx: MemoryContext, reasons: list[str]
    ) -> Assessment:
        if ctx.session_id is None:
            reasons.append("keine Session – nicht gespeichert")
            return Assessment(False, None, kind, importance, text, reasons)
        return Assessment(True, MemoryLayer.SESSION, kind, importance, text, reasons)
