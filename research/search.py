"""Austauschbare Suchanbieter. Ohne konfigurierten Anbieter gibt es keine Websuche.

* ``searxng`` – selbst gehostete SearXNG-Instanz mit aktiviertem JSON-Format
* ``brave`` – Brave Search API (Schlüssel aus Umgebungsvariable, nie im Code/Config-Text).
  Hinweis: Deren Bedingungen schränken die Nutzung der Ergebnisse für Training ein – Treffer
  werden mit ``provider`` markiert, damit die Lernpipeline sie ausschließen kann.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

USER_AGENT = "NOVA-Research/0.1 (local personal research assistant)"


class SearchError(Exception):
    pass


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    provider: str


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[SearchHit]: ...


class SearxngProvider:
    name = "searxng"

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise SearchError("SearXNG URL must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self._client = client

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        client = self._client or httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT})
        try:
            r = await client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
        except httpx.HTTPError as exc:
            raise SearchError(f"SearXNG not reachable: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code == 403:
            raise SearchError("SearXNG refused JSON output – enable 'json' under search.formats")
        if r.status_code != 200:
            raise SearchError(f"SearXNG HTTP {r.status_code}")
        try:
            results = r.json().get("results", [])
        except ValueError as exc:
            raise SearchError("SearXNG returned no JSON") from exc
        return [
            SearchHit(str(x["url"]), str(x.get("title", "")), str(x.get("content", "")), self.name)
            for x in results[:limit]
            if isinstance(x, dict) and str(x.get("url", "")).startswith(("http://", "https://"))
        ]


class BraveProvider:
    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        if not api_key:
            raise SearchError("Brave API key missing (NOVA_BRAVE_API_KEY)")
        self._key = api_key
        self._client = client

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        client = self._client or httpx.AsyncClient(timeout=20)
        try:
            r = await client.get(
                self.ENDPOINT,
                params={"q": query, "count": min(limit, 20)},
                headers={"X-Subscription-Token": self._key, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise SearchError(f"Brave Search not reachable: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()
        if r.status_code != 200:
            raise SearchError(f"Brave Search HTTP {r.status_code}")
        results = (r.json().get("web") or {}).get("results", [])
        return [
            SearchHit(
                str(x["url"]), str(x.get("title", "")), str(x.get("description", "")), self.name
            )
            for x in results[:limit]
            if isinstance(x, dict) and str(x.get("url", "")).startswith(("http://", "https://"))
        ]


def provider_from_config(config: dict[str, Any]) -> SearchProvider | None:
    """``{"provider": "searxng", "url": ...}`` oder ``{"provider": "brave"}`` (Key aus Env)."""
    kind = str(config.get("provider") or "").lower()
    if not kind or kind == "none":
        return None
    if kind == "searxng":
        return SearxngProvider(str(config.get("url") or ""))
    if kind == "brave":
        env = str(config.get("api_key_env") or "NOVA_BRAVE_API_KEY")
        return BraveProvider(os.environ.get(env, ""))
    raise SearchError(f"Unknown search provider {kind!r} (supported: searxng, brave)")
