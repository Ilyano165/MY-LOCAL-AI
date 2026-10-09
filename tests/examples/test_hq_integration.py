"""IC-WARE-HQ-Beispiel gegen eine echte NOVA-App (In-Process, Fake-Runtime)."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from tests.api.conftest import Factory, SSERuntime, engine_with

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "ic-ware-hq-integration"


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, ModuleType]]:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for name in ("nova_client", "hq_backend"):
        sys.modules.pop(name, None)
    import hq_backend
    import nova_client

    yield nova_client, hq_backend
    for name in ("nova_client", "hq_backend"):
        sys.modules.pop(name, None)


def nova(make_client: Factory, scopes: list[str], engine: Any = None, **kw: Any) -> tuple[Any, str]:
    client, service = make_client(engine)
    _, key = service.integrations.create("ic-ware-hq", scopes, **kw)
    return client, str(key)


def test_no_hardcoded_credentials(
    example: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    nova_client, _ = example
    monkeypatch.delenv("NOVA_BASE_URL", raising=False)
    monkeypatch.delenv("NOVA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="base URL"):
        nova_client.NovaClient()
    with pytest.raises(ValueError, match="API key"):
        nova_client.NovaClient("http://127.0.0.1:8765")
    for path in EXAMPLE.iterdir():
        if path.is_file():
            assert "nova_" + "0" not in path.read_text(encoding="utf-8")
    assert "NOVA_API_KEY=\n" in (EXAMPLE / ".env.example").read_text(encoding="utf-8")


def test_env_configuration(
    example: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    nova_client, _ = example
    monkeypatch.setenv("NOVA_BASE_URL", "http://10.0.0.5:8765/")
    monkeypatch.setenv("NOVA_API_KEY", "k")
    monkeypatch.setenv("NOVA_TIMEOUT_S", "7")
    c = nova_client.NovaClient()
    assert (c.base_url, c.api_key, c.timeout_s) == ("http://10.0.0.5:8765", "k", 7.0)


def test_chat_models_and_routing(
    example: tuple[ModuleType, ModuleType], make_client: Factory
) -> None:
    nova_client, _ = example
    http, key = nova(
        make_client,
        ["chat:complete", "models:read"],
        engine_with(("m", SSERuntime("Hello from NOVA"))),
    )
    c = nova_client.NovaClient("http://testserver", key, http=http)
    assert c.health()["status"] == "ok"
    assert "chat:complete" in c.capabilities()["integration"]["scopes"]
    assert [m["name"] for m in c.models()] == ["m"]
    result = c.chat([{"role": "user", "content": "Hi"}])
    assert result.content == "Hello from NOVA"
    assert result.model == "m" and result.request_id
    assert result.routing.get("category")


def test_streaming(example: tuple[ModuleType, ModuleType], make_client: Factory) -> None:
    nova_client, _ = example
    http, key = nova(
        make_client, ["chat:complete"], engine_with(("m", SSERuntime("one two three")))
    )
    c = nova_client.NovaClient("http://testserver", key, http=http)
    assert "".join(c.chat_stream([{"role": "user", "content": "Hi"}])) == "one two three"


@pytest.mark.parametrize(
    ("scopes", "engine", "error"),
    [
        (["models:read"], True, "NovaPermissionError"),  # Scope fehlt
        (["chat:complete"], False, "NovaUnavailableError"),  # kein Modell
    ],
)
def test_error_mapping(
    example: tuple[ModuleType, ModuleType],
    make_client: Factory,
    scopes: list[str],
    engine: bool,
    error: str,
) -> None:
    nova_client, _ = example
    http, key = nova(make_client, scopes, engine_with(("m", SSERuntime())) if engine else None)
    c = nova_client.NovaClient("http://testserver", key, http=http, retries=0)
    with pytest.raises(getattr(nova_client, error)) as info:
        c.chat([{"role": "user", "content": "Hi"}])
    assert info.value.status in (403, 503) and info.value.code and info.value.request_id


def test_invalid_key(example: tuple[ModuleType, ModuleType], make_client: Factory) -> None:
    nova_client, _ = example
    http, _ = nova(make_client, ["chat:complete"])
    c = nova_client.NovaClient("http://testserver", "nova_bad_key", http=http)
    with pytest.raises(nova_client.NovaAuthError) as info:
        c.models()
    assert info.value.status == 401


def test_rate_limit_retries_then_raises(
    example: tuple[ModuleType, ModuleType], make_client: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    nova_client, _ = example
    sleeps: list[float] = []
    monkeypatch.setattr(nova_client.time, "sleep", sleeps.append)
    http, key = nova(make_client, ["models:read"], rate_limit_per_minute=1)
    c = nova_client.NovaClient("http://testserver", key, http=http, retries=2)
    c.models()
    with pytest.raises(nova_client.NovaRateLimitError):
        c.models()
    assert len(sleeps) == 2 and all(0 < s <= 10 for s in sleeps)


def test_connection_error_and_timeout(example: tuple[ModuleType, ModuleType]) -> None:
    nova_client, _ = example

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    for handler, exc in (
        (refuse, nova_client.NovaConnectionError),
        (slow, nova_client.NovaTimeoutError),
    ):
        c = nova_client.NovaClient(
            "http://127.0.0.1:9", "k", http=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(exc):
            c.health()


def test_agent_task_via_client(
    example: tuple[ModuleType, ModuleType], make_client: Factory
) -> None:
    from tests.agents.fakes import ScriptedProvider, engine_for

    nova_client, _ = example
    http, key = nova(
        make_client, ["agent:run", "agent:read"], engine_for(ScriptedProvider(execute=["Done."]))
    )
    with http:
        c = nova_client.NovaClient("http://testserver", key, http=http)
        with pytest.raises(nova_client.NovaPermissionError) as info:
            c.create_task("read a file", allowed_tools=["read_file"])  # nicht freigegeben
        assert info.value.code == "tool_not_allowed"
        task = c.create_task("Explain recursion briefly")
        done = c.wait_for_task(task["id"], poll_s=0.05, max_wait_s=10)
        assert done["status"] == "succeeded" and done["result"]["answer"]


# ---------------------------------------------------------------------- HQ-Backend


def hq(example: tuple[ModuleType, ModuleType], http: Any, key: str) -> Any:
    from fastapi.testclient import TestClient

    nova_client, hq_backend = example
    return TestClient(
        hq_backend.create_app(
            nova_client.NovaClient("http://testserver", key, http=http, retries=0)
        )
    )


def test_hq_backend_returns_answer_and_allowed_metadata_only(
    example: tuple[ModuleType, ModuleType], make_client: Factory
) -> None:
    http, key = nova(make_client, ["chat:complete"], engine_with(("m", SSERuntime("Answer"))))
    app = hq(example, http, key)
    r = app.post(
        "/hq/assistant",
        json={"question": "What is up?"},
        headers={"Authorization": "Bearer hq-session"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "Answer"
    assert set(data["metadata"]) == {"model", "routing_category", "request_id"}
    assert key not in r.text


def test_hq_backend_requires_hq_login(
    example: tuple[ModuleType, ModuleType], make_client: Factory
) -> None:
    http, key = nova(make_client, ["chat:complete"], engine_with(("m", SSERuntime())))
    assert hq(example, http, key).post("/hq/assistant", json={"question": "x"}).status_code == 401


@pytest.mark.parametrize(
    ("scopes", "engine", "status"),
    [(["chat:complete"], False, 503), (["models:read"], True, 502)],
)
def test_hq_backend_error_handling(
    example: tuple[ModuleType, ModuleType],
    make_client: Factory,
    scopes: list[str],
    engine: bool,
    status: int,
) -> None:
    http, key = nova(make_client, scopes, engine_with(("m", SSERuntime())) if engine else None)
    r = hq(example, http, key).post(
        "/hq/assistant", json={"question": "x"}, headers={"Authorization": "Bearer s"}
    )
    assert r.status_code == status
    assert key not in r.text and "nova_" not in r.text


def test_hq_backend_timeout(example: tuple[ModuleType, ModuleType]) -> None:
    from fastapi.testclient import TestClient

    nova_client, hq_backend = example

    def slow(request: httpx.Request) -> httpx.Response:
        time.sleep(0)
        raise httpx.ReadTimeout("slow", request=request)

    client = nova_client.NovaClient(
        "http://127.0.0.1:9",
        "k",
        timeout_s=1,
        http=httpx.Client(transport=httpx.MockTransport(slow)),
    )
    r = TestClient(hq_backend.create_app(client)).post(
        "/hq/assistant", json={"question": "x"}, headers={"Authorization": "Bearer s"}
    )
    assert r.status_code == 504
