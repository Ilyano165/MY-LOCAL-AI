"""Versionierte SFT-Datensätze mit Herkunft, Qualitätsfiltern, Gruppen-Splits und Leck-Prüfung.

Quellen für Beispiele:
* geprüfte Research-Erkenntnisse (nur Klassifikation ``supported``; Quellen, deren Suchanbieter
  Training untersagt, werden ausgeschlossen),
* manuell kuratierte Beispiele (JSONL).

Eine Version wird nie überschrieben: ``<Daten>/datasets/<name>/<version>/`` mit
``train.jsonl``, ``val.jsonl``, ``test.jsonl`` und ``manifest.json`` (SHA-256 je Datei).
Splits werden nach **Gruppen** vergeben (gleiche Frage nie in zwei Splits); bereits vergebene
Gruppen behalten ihren Split in Folgeversionen (festes Testset).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.sources import normalize_text, shingles, similarity

SPLITS = ("train", "val", "test")
# Anbieter, deren Bedingungen Training mit Suchergebnissen untersagen
TRAINING_RESTRICTED_VIA = ("search:brave",)
SYSTEM_PROMPT = "You are NOVA, a careful local assistant. Answer precisely and name your sources."
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SECRETS = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwort|token)\s*[:=]\s*\S+"
    r"|sk-[A-Za-z0-9]{16,}|nova_[0-9a-f]{12}_[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s/-]?)?(?:\(?\d{2,5}\)?[\s/-]?){2,4}\d{3,}(?!\d)")


class DatasetError(Exception):
    pass


@dataclass
class Example:
    id: str
    messages: list[dict[str, str]]  # system, user, assistant
    group: str  # Split-Einheit (normalisierte Frage)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def user(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "user"), "")

    @property
    def assistant(self) -> str:
        return next((m["content"] for m in self.messages if m["role"] == "assistant"), "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group_key(question: str) -> str:
    return hashlib.sha256(normalize_text(question).lower().encode()).hexdigest()[:16]


def _example_id(user: str, assistant: str) -> str:
    raw = f"{normalize_text(user).lower()}\x00{normalize_text(assistant).lower()}"
    return "ex_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_example(question: str, answer: str, provenance: dict[str, Any]) -> Example:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_text(question)},
        {"role": "assistant", "content": answer.strip()},
    ]
    return Example(_example_id(question, answer), messages, group_key(question), provenance)


# ---------------------------------------------------------------------- Quellen


def from_findings(findings: list[dict[str, Any]]) -> tuple[list[Example], Counter[str]]:
    """Beispiele aus Knowledge-Store-Zeilen. Gibt (Beispiele, Ausschlussgründe) zurück."""
    examples: list[Example] = []
    skipped: Counter[str] = Counter()
    for f in findings:
        if f.get("stale"):
            skipped["stale"] += 1
            continue
        if f.get("classification") != "supported":
            skipped[f"classification:{f.get('classification')}"] += 1
            continue
        sources = [s for s in f.get("sources", []) if s.get("url")]
        if not sources:
            skipped["no_source"] += 1
            continue
        if any(str(s.get("via", "")) in TRAINING_RESTRICTED_VIA for s in sources):
            skipped["source_terms_forbid_training"] += 1
            continue
        refs = "; ".join(f"{s.get('title') or s['url']} ({s['url']})" for s in sources[:3])
        answer = f"{f['statement']}\n\nSources: {refs}"
        examples.append(
            make_example(
                f["subquestion"],
                answer,
                {
                    "type": "research_finding",
                    "finding_id": f["id"],
                    "run_id": f.get("run_id"),
                    "source_urls": [s["url"] for s in sources],
                    "retrieved": [s.get("fetched_at") for s in sources],
                },
            )
        )
    return examples, skipped


def from_jsonl(path: Path) -> list[Example]:
    """Manuelle Beispiele: je Zeile ``{"question": ..., "answer": ..., "note": ...}``."""
    examples = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            question, answer = str(item["question"]), str(item["answer"])
        except (ValueError, KeyError, TypeError) as exc:
            raise DatasetError(f"{path}:{n}: expected JSON with question and answer") from exc
        examples.append(
            make_example(
                question,
                answer,
                {
                    "type": "manual",
                    "file": path.name,
                    "line": n,
                    "note": str(item.get("note", ""))[:200],
                },
            )
        )
    return examples


# ---------------------------------------------------------------------- Qualität


def quality_problem(ex: Example) -> str | None:
    user, answer = ex.user, ex.assistant
    if len(user) < 8 or len(answer) < 8:
        return "too_short"
    if len(user) > 4000 or len(answer) > 8000:
        return "too_long"
    text = f"{user}\n{answer}"
    if _SECRETS.search(text):
        return "secret"
    if _EMAIL.search(text) or _PHONE.search(text):
        return "personal_data"
    return None


def filter_examples(examples: list[Example]) -> tuple[list[Example], Counter[str]]:
    """Qualität, exakte und Nahdubletten, widersprüchliche Paare (gleiche Frage, andere Antwort)."""
    dropped: Counter[str] = Counter()
    kept: list[Example] = []
    seen: set[str] = set()
    for ex in examples:
        problem = quality_problem(ex)
        if problem:
            dropped[problem] += 1
            continue
        if ex.id in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(ex.id)
        kept.append(ex)
    # gleiche Frage mit unterschiedlichen Antworten → beide verwerfen (Widerspruch unklar)
    answers: dict[str, set[str]] = {}
    for ex in kept:
        answers.setdefault(ex.group, set()).add(
            normalize_text(ex.assistant.split("\n\nSources:")[0]).lower()
        )
    conflicting = {g for g, a in answers.items() if len(a) > 1}
    result: list[Example] = []
    shingle_cache: list[tuple[Example, set[int]]] = []
    for ex in kept:
        if ex.group in conflicting:
            dropped["conflicting_answers"] += 1
            continue
        own = shingles(ex.user + " " + ex.assistant)
        if any(similarity(own, other) >= 0.9 for _, other in shingle_cache):
            dropped["near_duplicate"] += 1
            continue
        shingle_cache.append((ex, own))
        result.append(ex)
    return result, dropped


# ---------------------------------------------------------------------- Splits & Lecks


def assign_split(group: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    value = int(hashlib.sha256(f"split:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < ratios[0]:
        return "train"
    return "val" if value < ratios[0] + ratios[1] else "test"


def _ngrams(text: str, n: int = 8) -> set[str]:
    words = normalize_text(text).lower().split()
    return {" ".join(words[i : i + n]) for i in range(max(len(words) - n + 1, 0))}


def leak_report(splits: dict[str, list[Example]], n: int = 8) -> list[dict[str, Any]]:
    """Testbeispiele, die 8-Gramme mit Trainingsbeispielen teilen (Frage oder Antwort)."""
    train_grams: dict[str, str] = {}
    for ex in splits["train"]:
        for g in _ngrams(ex.user + " " + ex.assistant.split("\n\nSources:")[0], n):
            train_grams.setdefault(g, ex.id)
    leaks = []
    for name in ("val", "test"):
        for ex in splits[name]:
            hits = {
                train_grams[g]
                for g in _ngrams(ex.user + " " + ex.assistant.split("\n\nSources:")[0], n)
                if g in train_grams
            }
            if hits:
                leaks.append({"split": name, "example": ex.id, "overlaps_with": sorted(hits)[:5]})
    return leaks


# ---------------------------------------------------------------------- Versionen


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def versions(self, name: str) -> list[str]:
        base = self.root / name
        found = (
            [p.name for p in base.glob("*") if (p / "manifest.json").is_file()]
            if base.is_dir()
            else []
        )
        return sorted(found, key=lambda v: tuple(int(x) for x in v.split(".")))

    def manifest(self, name: str, version: str) -> dict[str, Any]:
        path = self.root / name / version / "manifest.json"
        if not path.is_file():
            raise DatasetError(f"dataset {name}@{version} not found")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def list_all(self) -> list[dict[str, Any]]:
        out = []
        for base in (
            sorted(p for p in self.root.glob("*") if p.is_dir()) if self.root.is_dir() else []
        ):
            for v in self.versions(base.name):
                m = self.manifest(base.name, v)
                out.append(
                    {
                        "name": base.name,
                        "version": v,
                        "counts": m["counts"],
                        "created_at": m["created_at"],
                    }
                )
        return out

    def verify(self, name: str, version: str) -> list[str]:
        """Prüft die Unveränderlichkeit (Hashes laut Manifest). Leere Liste = in Ordnung."""
        m = self.manifest(name, version)
        problems = []
        for split, digest in m["files"].items():
            path = self.root / name / version / f"{split}.jsonl"
            if not path.is_file():
                problems.append(f"{split}.jsonl missing")
            elif _sha256(path) != digest:
                problems.append(f"{split}.jsonl was modified")
        return problems

    def load(self, name: str, version: str, split: str) -> list[Example]:
        path = self.root / name / version / f"{split}.jsonl"
        return [
            Example(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def build(
        self,
        name: str,
        examples: list[Example],
        *,
        skipped: Counter[str] | None = None,
        description: str = "",
        version: str | None = None,
    ) -> dict[str, Any]:
        if not _NAME.match(name):
            raise DatasetError("dataset name: lowercase letters, digits, . _ - (2–64 chars)")
        previous = self.versions(name)
        parent = previous[-1] if previous else None
        if version is None:
            if parent:
                major, minor, _ = (int(x) for x in parent.split("."))
                version = f"{major}.{minor + 1}.0"
            else:
                version = "0.1.0"
        if not _VERSION.match(version):
            raise DatasetError("version must be X.Y.Z")
        target = self.root / name / version
        if target.exists():
            raise DatasetError(f"{name}@{version} already exists – versions are immutable")
        kept, dropped = filter_examples(examples)
        # feste Zuordnung: Gruppen aus Vorversionen behalten ihren Split
        fixed: dict[str, str] = {}
        if parent:
            for split in SPLITS:
                for ex in self.load(name, parent, split):
                    fixed[ex.group] = split
        splits: dict[str, list[Example]] = {s: [] for s in SPLITS}
        for ex in kept:
            splits[fixed.get(ex.group) or assign_split(ex.group)].append(ex)
        leaks = leak_report(splits)
        if leaks:
            leaked = {item["example"] for item in leaks}
            for s in (
                "val",
                "test",
            ):  # Leck → Beispiel aus val/test entfernen, nie still trainieren
                splits[s] = [ex for ex in splits[s] if ex.id not in leaked]
            dropped["leak_removed_from_eval"] += len(leaked)
        if not splits["train"]:
            raise DatasetError("no training examples left after filtering")
        target.mkdir(parents=True)
        files = {}
        for split, items in splits.items():
            path = target / f"{split}.jsonl"
            path.write_text(
                "".join(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n" for ex in items),
                encoding="utf-8",
            )
            files[split] = _sha256(path)
        manifest = {
            "name": name,
            "version": version,
            "parent": parent,
            "description": description,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "format": "chat-messages-v1",
            "counts": {s: len(v) for s, v in splits.items()},
            "input_examples": len(examples),
            "excluded_before_build": dict(skipped or {}),
            "dropped": dict(dropped),
            "leaks": leaks,
            "provenance": dict(Counter(ex.provenance.get("type", "unknown") for ex in kept)),
            "split_policy": "group = normalised question; groups keep their split across versions",
            "files": files,
        }
        tmp = target / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, target / "manifest.json")
        return manifest
