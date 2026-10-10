"""Quellenverwaltung: URL-Normalisierung, Inhalts-Hash, Nahdubletten, HTML→Text."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid|si)$", re.I)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[\w'-]+", re.UNICODE)


def canonical_url(url: str) -> str:
    """Kanonische Form für die Deduplizierung (Host klein, ohne Fragment/Tracking/Default-Port)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)))
    return urlunsplit((scheme, host, path, query, ""))


def registrable_domain(url: str) -> str:
    """Grobe „unabhängige Quelle“-Einheit: die letzten zwei Labels (co.uk etc. drei)."""
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in {"co", "com", "ac", "gov", "org", "net", "edu"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 5) -> set[int]:
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < size:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i : i + size])) for i in range(len(words) - size + 1)}


def similarity(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _TextExtractor(HTMLParser):
    SKIP = frozenset(
        {
            "script",
            "style",
            "noscript",
            "template",
            "svg",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "iframe",
        }
    )
    BLOCK = frozenset(
        {
            "p",
            "div",
            "li",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "tr",
            "pre",
            "blockquote",
            "section",
            "article",
            "td",
            "dd",
            "dt",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.meta: dict[str, str] = {}
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            key = (a.get("name") or a.get("property") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


@dataclass
class PageText:
    title: str
    text: str
    published: str | None = None
    author: str | None = None


def html_to_text(markup: str) -> PageText:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    lines = [normalize_text(line) for line in "".join(parser.parts).split("\n")]
    text = "\n".join(line for line in lines if line)
    meta = parser.meta
    published = (
        meta.get("article:published_time")
        or meta.get("datepublished")
        or meta.get("date")
        or meta.get("dc.date")
        or None
    )
    author = meta.get("author") or meta.get("article:author") or None
    return PageText(html.unescape(normalize_text(parser.title)), text, published, author)


@dataclass
class Source:
    id: str
    url: str
    canonical: str
    domain: str
    title: str
    fetched_at: str
    content_hash: str
    chars: int
    subquestions: list[str] = field(default_factory=list)
    published: str | None = None
    author: str | None = None
    kind: str = "unknown"
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    snapshot: str = ""  # Pfad relativ zum Laufverzeichnis
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
