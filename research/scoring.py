"""Deterministische, nachvollziehbare Quellenbewertung (Punktzahl 0–1 mit Begründungen).

Bewertet wird die Quelle, nicht die Wahrheit einzelner Aussagen. Die Kategorien sind
Heuristiken über Domain/Pfad und Metadaten – bewusst einfach und überprüfbar.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlsplit

from research.sources import Source

KIND_BASE = {
    "primary": 0.75,  # Behörden, Standards, Hersteller-/Projektdokumentation
    "academic": 0.75,  # Paper, Hochschulen
    "reference": 0.6,  # Enzyklopädien, Nachschlagewerke
    "news": 0.5,
    "blog": 0.4,
    "forum": 0.3,
    "unknown": 0.4,
}
_ACADEMIC = (
    "arxiv.org",
    "doi.org",
    "acm.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "semanticscholar.org",
    "aclanthology.org",
    "openreview.net",
)
_REFERENCE = ("wikipedia.org", "britannica.com", "mozilla.org")
_FORUM = (
    "reddit.com",
    "stackexchange.com",
    "stackoverflow.com",
    "quora.com",
    "news.ycombinator.com",
    "discord.com",
    "forum.",
)
_NEWS = (
    "reuters.com",
    "apnews.com",
    "bbc.",
    "heise.de",
    "theverge.com",
    "arstechnica.com",
    "golem.de",
    "spiegel.de",
    "nytimes.com",
    "theguardian.com",
)
_BLOG = ("medium.com", "substack.com", "dev.to", "blogspot.", "wordpress.", "hashnode.")


def classify(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path.lower()
    if host.endswith((".gov", ".gov.uk", ".europa.eu", ".bund.de")) or host.startswith(
        ("docs.", "developer.", "learn.")
    ):
        return "primary"
    if any(h in host for h in _ACADEMIC) or host.endswith(".edu") or ".ac." in host:
        return "academic"
    if any(h in host for h in _REFERENCE):
        return "reference"
    if any(h in host for h in _FORUM) or "/forum" in path or "/comments/" in path:
        return "forum"
    if any(h in host for h in _NEWS):
        return "news"
    if any(h in host for h in _BLOG) or "/blog" in path:
        return "blog"
    # Projektseiten/Dateien der Projekte selbst (Paketindex, Repository-Rohdateien)
    if host in ("pypi.org", "raw.githubusercontent.com", "crates.io", "www.npmjs.com"):
        return "primary"
    if host.endswith(("github.com", "gitlab.com")) or "/docs/" in path or "/documentation" in path:
        return "primary"
    return "unknown"


def _age_days(published: str | None, now: datetime) -> float | None:
    if not published:
        return None
    try:
        when = datetime.fromisoformat(published.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (now - when).total_seconds() / 86400


def assess(source: Source, now: datetime | None = None) -> Source:
    now = now or datetime.now(UTC)
    kind = classify(source.url)
    score = KIND_BASE[kind]
    reasons = [f"type {kind} (base {KIND_BASE[kind]:.2f})"]
    if source.url.startswith("https://"):
        score += 0.05
        reasons.append("https +0.05")
    else:
        score -= 0.1
        reasons.append("no https -0.10")
    age = _age_days(source.published, now)
    if age is None:
        reasons.append("no publication date")
    elif age <= 730:
        score += 0.05
        reasons.append("published within 2 years +0.05")
    elif age > 365 * 6:
        score -= 0.1
        reasons.append("older than 6 years -0.10")
    if source.author:
        score += 0.05
        reasons.append("author named +0.05")
    if source.chars < 800:
        score -= 0.15
        reasons.append("very little text -0.15")
    source.kind = kind
    source.score = round(min(max(score, 0.0), 1.0), 2)
    source.score_reasons = reasons
    return source
