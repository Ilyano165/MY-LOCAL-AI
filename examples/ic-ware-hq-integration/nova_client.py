"""Austauschbarer Client für die NOVA-Integrations-API (serverseitig verwenden!).

Konfiguration ausschließlich über Parameter oder Umgebung – keine fest codierten Zugangsdaten:

    NOVA_BASE_URL   z. B. http://127.0.0.1:8765  (ohne /api/v1)
    NOVA_API_KEY    Integrationsschlüssel (nova integrations create …)
    NOVA_TIMEOUT_S  Gesamt-Zeitlimit je Anfrage (Standard 120)

Nur Python-Standardbibliothek + ``httpx``. Fehler werden auf eigene Ausnahmen abgebildet, damit
der aufrufende Dienst (z. B. IC WARE HQ) sie gezielt behandeln kann.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx


class NovaError(Exception):
    """Basis: ``status`` (HTTP), ``code`` (NOVA-Fehlercode), ``request_id`` (für Support)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.request_id = request_id


class NovaAuthError(NovaError):
    """401 – Schlüssel fehlt, ist falsch, widerrufen oder abgelaufen."""


class NovaPermissionError(NovaError):
    """403 – Scope oder Modell für diese Integration nicht freigegeben."""


class NovaRateLimitError(NovaError):
    """429 – zu viele Anfragen."""


class NovaUnavailableError(NovaError):
    """503 – kein lokales Modell verfügbar / NOVA nicht bereit."""


class NovaTimeoutError(NovaError):
    """504 oder Netzwerk-Zeitüberschreitung."""


class NovaConnectionError(NovaError):
    """NOVA nicht erreichbar (Dienst gestoppt, falsche Adresse, Netzwerk)."""


_BY_STATUS: dict[int, type[NovaError]] = {
    401: NovaAuthError,
    403: NovaPermissionError,
    429: NovaRateLimitError,
    503: NovaUnavailableError,
    504: NovaTimeoutError,
}


@dataclass
class ChatResult:
    content: str
    model: str
    request_id: str | None
    usage: dict[str, int] | None = None
    routing: dict[str, Any] = field(default_factory=dict)


class NovaClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout_s: float | None = None,
        retries: int = 2,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("NOVA_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("NOVA_API_KEY", "")
        if not self.base_url:
            raise ValueError("NOVA base URL missing (NOVA_BASE_URL)")
        if not self.api_key:
            raise ValueError("NOVA API key missing (NOVA_API_KEY)")
        self.timeout_s = timeout_s or float(os.environ.get("NOVA_TIMEOUT_S", "120"))
        self.retries = retries
        # Zeitlimits gelten über den httpx-Client (eigener Client → eigene Zeitlimits)
        self._http = http or httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=5.0))

    # ------------------------------------------------------------------ intern

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _request(self, method: str, path: str, json: Any = None) -> dict[str, Any]:
        url = f"{self.base_url}/api/v1{path}"
        for attempt in range(self.retries + 1):
            try:
                response = self._http.request(method, url, json=json, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise NovaTimeoutError(f"NOVA did not answer within {self.timeout_s:g} s") from exc
            except httpx.TransportError as exc:
                raise NovaConnectionError(f"NOVA not reachable at {self.base_url}") from exc
            # Wiederholen nur, wenn sicher: Rate Limit (mit Retry-After) und „nicht bereit“
            if response.status_code in (429, 503) and attempt < self.retries:
                delay = float(response.headers.get("retry-after", 1 + attempt))
                time.sleep(min(delay, 10.0))
                continue
            return self._parse(response)
        raise NovaError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        request_id = response.headers.get("x-request-id")
        try:
            data: dict[str, Any] = response.json()
        except ValueError:
            data = {}
        if response.status_code >= 400:
            error = data.get("error") or {}
            cls = _BY_STATUS.get(response.status_code, NovaError)
            raise cls(
                str(error.get("message") or f"HTTP {response.status_code}"),
                status=response.status_code,
                code=error.get("code"),
                request_id=request_id,
            )
        data.setdefault("_request_id", request_id)
        return data

    # ------------------------------------------------------------------ API

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/capabilities")

    def models(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/models")["models"])

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "nova-auto",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "nova_timeout_s": self.timeout_s,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        data = self._request("POST", "/chat/completions", body)
        return ChatResult(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            request_id=data.get("_request_id"),
            usage=data.get("usage"),
            routing=(data.get("nova") or {}).get("routing", {}),
        )

    def chat_stream(
        self, messages: list[dict[str, Any]], *, model: str = "nova-auto"
    ) -> Iterator[str]:
        """Liefert Textstücke; Abbrechen = Iterator verlassen (schließt die Verbindung)."""
        import json as _json

        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "nova_timeout_s": self.timeout_s,
        }
        with self._http.stream(
            "POST", f"{self.base_url}/api/v1/chat/completions", json=body, headers=self._headers()
        ) as response:
            if response.status_code >= 400:
                response.read()
                self._parse(response)
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = _json.loads(line[6:])
                if "error" in event:
                    raise NovaError(event["error"]["message"], code=event["error"].get("code"))
                for choice in event.get("choices") or []:
                    if text := choice.get("delta", {}).get("content"):
                        yield text

    def create_task(
        self, objective: str, *, allowed_tools: list[str] | None = None, timeout_s: float = 600
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/agent/tasks",
            {"objective": objective, "allowed_tools": allowed_tools or [], "timeout_s": timeout_s},
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/agent/tasks/{task_id}")

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        return self._request("POST", f"/agent/tasks/{task_id}/cancel")

    def wait_for_task(
        self, task_id: str, *, poll_s: float = 1.0, max_wait_s: float = 600
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            task = self.get_task(task_id)
            if task["status"] not in ("queued", "running"):
                return task
            time.sleep(poll_s)
        raise NovaTimeoutError(f"Task {task_id} did not finish within {max_wait_s:g} s")

    def close(self) -> None:
        self._http.close()
