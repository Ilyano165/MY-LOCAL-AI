"""Kleines, lokales Ranking-Modell: lineare Bewertung s(x) = w·x̂ + b.

Bewusst kein neuronales Netz: wenige hundert Beispiele, ~80 Merkmale, nachvollziehbare
Gewichte, deterministisches Training in reinem Python (keine zusätzliche Abhängigkeit).

Trainingsziel (kombiniert, beide auf demselben Score):

* **paarweise** (RankNet, linear): Innerhalb einer Gruppe (= eine Anfrage mit mehreren
  bewerteten Kandidaten) soll der Kandidat mit höherem Nutzen den höheren Score bekommen:
  Verlust ``log(1 + exp(-(s_i - s_j)))`` gewichtet mit dem Nutzenabstand.
* **punktweise** (logistische Regression): ``σ(s)`` ≈ Erfolgswahrscheinlichkeit. Nötig für
  echte Routing-Logs, in denen meist nur das *gewählte* Modell ein Ergebnis hat.

Merkmale werden mit Mittelwert/Standardabweichung **der Trainingsdaten** standardisiert.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_FORMAT = 1


@dataclass(frozen=True)
class Example:
    group: str
    """Vergleichsgruppe (eine Anfrage); Paare entstehen nur innerhalb einer Gruppe."""
    task_id: str
    """Leakage-Einheit: dieselbe Aufgabe nie in Training und Test."""
    model: str
    features: Mapping[str, float]
    utility: float
    success: bool
    source: str = "routing_log"


@dataclass
class TrainingConfig:
    epochs: int = 60
    learning_rate: float = 0.05
    l2: float = 1e-3
    pointwise_weight: float = 0.3
    pair_margin: float = 0.01
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("epochs ≥ 1, learning_rate > 0, l2 ≥ 0 erforderlich")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class RankingModel:
    names: list[str]
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    config: TrainingConfig = field(default_factory=TrainingConfig)
    trained_on: dict[str, Any] = field(default_factory=dict)

    @property
    def trained(self) -> bool:
        return bool(self.weights)

    # ------------------------------------------------------------------ Vorhersage

    def _standardize(self, features: Mapping[str, float]) -> list[float]:
        return [
            (float(features.get(n, 0.0)) - m) / s
            for n, m, s in zip(self.names, self.mean, self.std, strict=True)
        ]

    def score(self, features: Mapping[str, float]) -> float:
        if not self.trained:
            raise RuntimeError("Ranking-Modell ist nicht trainiert")
        x = self._standardize(features)
        value = self.bias + sum(w * v for w, v in zip(self.weights, x, strict=True))
        if not math.isfinite(value):
            raise ValueError("Ranking-Score ist nicht endlich")
        return value

    def success_probability(self, features: Mapping[str, float]) -> float:
        return _sigmoid(self.score(features))

    # ------------------------------------------------------------------ Training

    def fit(self, examples: Sequence[Example]) -> dict[str, float]:
        if not examples:
            raise ValueError("Keine Trainingsbeispiele")
        n_features = len(self.names)
        rows = [[float(e.features.get(n, 0.0)) for n in self.names] for e in examples]
        self.mean = [sum(r[i] for r in rows) / len(rows) for i in range(n_features)]
        self.std = []
        for i in range(n_features):
            var = sum((r[i] - self.mean[i]) ** 2 for r in rows) / len(rows)
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        xs = [[(v - m) / s for v, m, s in zip(r, self.mean, self.std, strict=True)] for r in rows]

        groups: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(examples):
            groups[e.group].append(i)
        pairs: list[tuple[int, int, float]] = []
        for idx in groups.values():
            for hi in idx:
                for lo in idx:
                    diff = examples[hi].utility - examples[lo].utility
                    if diff > self.config.pair_margin:
                        pairs.append((hi, lo, min(diff * 4, 1.0)))

        cfg = self.config
        rng = random.Random(cfg.seed)
        w = [0.0] * n_features
        b = 0.0
        items: list[tuple[str, int, int, float]] = [("pair", a, c, wt) for a, c, wt in pairs]
        items += [("point", i, -1, cfg.pointwise_weight) for i in range(len(examples))]
        for epoch in range(cfg.epochs):
            rng.shuffle(items)
            lr = cfg.learning_rate / (1 + 0.05 * epoch)
            for kind, i, j, weight in items:
                if kind == "pair":
                    xi, xj = xs[i], xs[j]
                    margin = sum(wk * (xi[k] - xj[k]) for k, wk in enumerate(w))
                    grad = -weight * _sigmoid(-margin)  # d/dmargin log(1+e^-margin)
                    for k in range(n_features):
                        w[k] -= lr * (grad * (xi[k] - xj[k]) + cfg.l2 * w[k])
                else:
                    xi = xs[i]
                    z = b + sum(wk * xi[k] for k, wk in enumerate(w))
                    grad = weight * (_sigmoid(z) - float(examples[i].success))
                    for k in range(n_features):
                        w[k] -= lr * (grad * xi[k] + cfg.l2 * w[k])
                    b -= lr * grad
        self.weights, self.bias = w, b
        stats = self.training_stats(examples, pairs, xs)
        self.trained_on = {
            "examples": len(examples),
            "groups": len(groups),
            "pairs": len(pairs),
            "tasks": len({e.task_id for e in examples}),
            "sources": sorted({e.source for e in examples}),
            **{k: round(v, 4) for k, v in stats.items()},
        }
        return stats

    def training_stats(
        self,
        examples: Sequence[Example],
        pairs: Sequence[tuple[int, int, float]],
        xs: Sequence[Sequence[float]],
    ) -> dict[str, float]:
        scores = [self.bias + sum(wk * x[k] for k, wk in enumerate(self.weights)) for x in xs]
        pair_acc = (sum(scores[a] > scores[c] for a, c, _ in pairs) / len(pairs)) if pairs else 0.0
        return {"train_pair_accuracy": pair_acc}

    # ------------------------------------------------------------------ Persistenz

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "names": self.names,
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "config": vars(self.config),
            "trained_on": self.trained_on,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RankingModel:
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unbekanntes Modellformat: {data.get('format')!r}")
        names = list(data["names"])
        model = cls(
            names=names,
            weights=[float(x) for x in data["weights"]],
            bias=float(data["bias"]),
            mean=[float(x) for x in data["mean"]],
            std=[float(x) for x in data["std"]],
            config=TrainingConfig(**data.get("config", {})),
            trained_on=dict(data.get("trained_on", {})),
        )
        if not (len(model.weights) == len(model.mean) == len(model.std) == len(names)):
            raise ValueError("Modelldatei inkonsistent (Längen der Vektoren)")
        return model

    def top_weights(self, k: int = 10) -> list[tuple[str, float]]:
        pairs = sorted(zip(self.names, self.weights, strict=True), key=lambda p: -abs(p[1]))
        return [(n, round(w, 4)) for n, w in pairs[:k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RankingModel:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
