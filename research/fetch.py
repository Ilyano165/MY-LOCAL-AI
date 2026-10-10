"""Höflicher, begrenzter Abruf von Webseiten.

* robots.txt wird beachtet (je Host zwischengespeichert); nicht abrufbares robots.txt (4xx)
  gilt als „erlaubt“, Serverfehler (5xx) als „nicht erlaubt“ (konservativ, wie RFC 9309).
* Mindestabstand je Host, Größenlimit, nur Text/HTML, keine Cookies, keine Logins,
  keine Umgehung von Paywalls (401/402/403 werden als Fehler protokolliert, nicht umgangen).
"""

from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from research.search import USER_AGENT
from research.sources import PageText, html_to_text, normalize_text

ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class FetchError(Exception):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class FetchedPage:
    url: str
    final_url: str
    status: int
    content_type: str
    page: PageText
    bytes: int


class Fetcher:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        min_interval_s: float = 2.0,
        max_bytes: int = 3_000_000,
        timeout_s: float = 20.0,
        respect_robots: bool = True,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
        )
        self._own_client = client is None
        self.min_interval_s = min_interval_s
        self.max_bytes = max_bytes
        self.respect_robots = respect_robots
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_problem: dict[str, str] = {}
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    @staticmethod
    def _origin(url: str) -> str:
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}"

    async def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        origin = self._origin(url)
        if origin not in self._robots:
            rules: urllib.robotparser.RobotFileParser | None = None
            try:
                r = await self._client.get(f"{origin}/robots.txt")
                if r.status_code < 500:  # 5xx: unklar → nicht abrufen
                    rules = urllib.robotparser.RobotFileParser()
                    # 4xx: kein robots.txt → alles erlaubt
                    rules.parse([] if r.status_code >= 400 else r.text.splitlines())
                else:
                    self._robots_problem[origin] = f"robots.txt returned HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                rules = None
                self._robots_problem[origin] = f"robots.txt not reachable ({type(exc).__name__})"
            self._robots[origin] = rules
        parser = self._robots[origin]
        return parser is not None and parser.can_fetch(USER_AGENT, url)

    async def _pace(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last.get(host, 0.0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[host] = time.monotonic()

    async def fetch(self, url: str) -> FetchedPage:
        if not url.startswith(("http://", "https://")):
            raise FetchError(f"unsupported URL scheme: {url}", reason="scheme")
        if not await self.allowed(url):
            problem = self._robots_problem.get(self._origin(url))
            if problem:  # Regeln unbekannt → konservativ nicht abrufen
                raise FetchError(f"{problem} – not fetched: {url}", reason="robots_unavailable")
            raise FetchError(f"robots.txt disallows {url}", reason="robots")
        await self._pace(urlsplit(url).netloc)
        try:
            async with self._client.stream("GET", url) as r:
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if r.status_code in (401, 402, 403, 451):
                    raise FetchError(f"HTTP {r.status_code} – access restricted", reason="access")
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code}", reason="http")
                if ctype not in ALLOWED_TYPES:
                    raise FetchError(f"unsupported content type {ctype!r}", reason="type")
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > self.max_bytes:
                        raise FetchError("page larger than limit", reason="size")
                final_url = str(r.url)
                encoding = r.encoding or "utf-8"
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}", reason="network") from exc
        text = bytes(body).decode(encoding, errors="replace")
        if ctype == "text/plain":
            page = PageText(title=final_url, text=normalize_text(text))
        else:
            page = html_to_text(text)
        return FetchedPage(url, final_url, 200, ctype, page, len(body))
