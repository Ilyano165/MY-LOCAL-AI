"""Integrations-API: /api/v1 (NOVA) und /v1 (OpenAI-kompatibel)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from api.integrations import IntegrationError, IntegrationStore, RateLimiter
from models.base import FinishReason, StreamChunk
from models.inference import InferenceEngine, ProviderRegistry
from models.model_registry import ModelRegistry
from tests.agents.fakes import ScriptedProvider
from tests.api.conftest import Factory, SSERuntime, engine_with
from tests.models.fakes import StubProvider, make_meta

ALL = ["chat:complete", "models:read", "agent:run", "agent:read", "agent:cancel", "system:read"]


def key_for(service: Any, scopes: list[str], **kw: Any) -> str:
    _, key = service.integrations.create(kw.pop("name", "hq"), scopes, **kw)
    return str(key)


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "nova-auto",
        "messages": [{"role": "user", "content": "Hello there"}],
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------- Health & Version


def test_health_is_public_and_minimal(make_client: Factory) -> None:
    client, _ = make_client()
    data = client.get("/api/v1/health").json()
    assert data["status"] == "ok" and data["api_version"] == "v1"
    from api.version import __version__

    assert data["version"] == __version__
    assert set(data) == {"status", "version", "api_version"}


# ---------------------------------------------------------------------- Authentifizierung


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Bearer nova_000000000000_" + "x" * 43},
    ],
)
def test_missing_or_invalid_key(make_client: Factory, headers: dict[str, str]) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    r = client.post("/api/v1/chat/completions", json=chat_body(), headers=headers)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] in ("missing_api_key", "invalid_api_key")
    assert service.audit.recent[-1]["event"] == "auth_failed"
    o = client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    assert o.status_code == 401 and o.json()["error"]["type"] == "authentication_error"


def test_key_is_stored_hashed_revoke_and_rotate(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "i.db")
    integration, key = store.create("hq", ["chat:complete"])
    assert key not in (tmp_path / "i.db").read_bytes().decode("latin-1")
    assert store.authenticate(key) is not None
    _, new_key = store.rotate(integration.id)
    assert store.authenticate(key) is None and store.authenticate(new_key) is not None
    store.revoke(integration.id)
    assert store.authenticate(new_key) is None
    with pytest.raises(IntegrationError):
        store.rotate(integration.id)


def test_expired_key(tmp_path: Path) -> None:
    store = IntegrationStore(tmp_path / "i.db")
    _, key = store.create("old", ["models:read"], expires_at="2000-01-01T00:00:00+00:00")
    assert store.authenticate(key) is None


@pytest.mark.parametrize("scopes", [[], ["admin:everything"], ["agent:tool:rm_rf"]])
def test_invalid_scopes_rejected(tmp_path: Path, scopes: list[str]) -> None:
    store = IntegrationStore(tmp_path / "i.db")
    with pytest.raises(IntegrationError):
        store.create("x", scopes, known_tools={"read_file"})


# ---------------------------------------------------------------------- Berechtigungen


def test_scope_checks(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime("hi"))))
    models_only = key_for(service, ["models:read"])
    r = client.post("/api/v1/chat/completions", json=chat_body(), headers=auth(models_only))
    assert r.status_code == 403 and r.json()["error"]["code"] == "insufficient_scope"
    assert client.get("/api/v1/models", headers=auth(models_only)).status_code == 200
    assert client.get("/api/v1/system/status", headers=auth(models_only)).status_code == 403
    o = client.post("/v1/chat/completions", json=chat_body(), headers=auth(models_only))
    assert o.status_code == 403 and o.json()["error"]["type"] == "permission_error"
    events = [e["event"] for e in service.audit.recent]
    assert "permission_denied" in events


def test_capabilities_reflect_integration(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    key = key_for(service, ["chat:complete", "agent:tool:read_file"])
    caps = client.get("/api/v1/capabilities", headers=auth(key)).json()
    assert caps["features"]["chat_completions"] is True
    assert caps["features"]["agent_tasks"] is False
    assert caps["features"]["agent_tools"] == ["read_file"]
    assert caps["features"]["tool_calling_via_chat"] is False
    assert caps["integration"]["scopes"] == ["agent:tool:read_file", "chat:complete"]


def test_allowed_models_per_integration(make_client: Factory) -> None:
    client, service = make_client(engine_with(("a", SSERuntime("A")), ("b", SSERuntime("B"))))
    key = key_for(service, ["chat:complete", "models:read"], allowed_models=["a"])
    ids = [m["id"] for m in client.get("/v1/models", headers=auth(key)).json()["data"]]
    assert ids == ["a"]
    r = client.post("/v1/chat/completions", json=chat_body(model="b"), headers=auth(key))
    assert r.status_code == 403 and r.json()["error"]["code"] == "model_not_allowed"
    ok = client.post("/v1/chat/completions", json=chat_body(model="a"), headers=auth(key))
    assert ok.json()["choices"][0]["message"]["content"] == "A"


def test_system_status_hides_local_paths(make_client: Factory) -> None:
    client, service = make_client()
    data = client.get("/api/v1/system/status", headers=auth(key_for(service, ["system:read"])))
    assert data.status_code == 200
    assert "data_dir" not in data.json() and "config_path" not in data.json()


# ---------------------------------------------------------------------- Limits


def test_rate_limit(make_client: Factory) -> None:
    client, service = make_client()
    key = key_for(service, ["models:read"], rate_limit_per_minute=2)
    assert [client.get("/api/v1/models", headers=auth(key)).status_code for _ in range(3)] == [
        200,
        200,
        429,
    ]
    r = client.get("/api/v1/models", headers=auth(key))
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "rate_limited"


def test_rate_limiter_window() -> None:
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert limiter.allow("x", 1)[0] and not limiter.allow("x", 1)[0]
    now[0] = 61.0
    assert limiter.allow("x", 1)[0]


def test_request_size_limit(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    key = key_for(service, ["chat:complete"], max_request_bytes=2048)
    big = chat_body(messages=[{"role": "user", "content": "x" * 5000}])
    r = client.post("/api/v1/chat/completions", json=big, headers=auth(key))
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


# ---------------------------------------------------------------------- Chat


def test_chat_without_model(make_client: Factory) -> None:
    client, service = make_client()
    key = key_for(service, ["chat:complete"])
    r = client.post("/api/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert r.status_code == 503 and r.json()["error"]["message"] == "No local model available."
    o = client.post("/v1/chat/completions", json=chat_body(), headers=auth(key))
    assert o.status_code == 503 and o.json()["error"]["type"] == "service_unavailable_error"


def test_successful_chat_nova_format(make_client: Factory) -> None:
    runtime = SSERuntime("Hello from NOVA")
    client, service = make_client(engine_with(("m", runtime)))
    key = key_for(service, ["chat:complete"])
    r = client.post(
        "/api/v1/chat/completions",
        headers=auth(key),
        json=chat_body(
            messages=[
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ],
            temperature=0.1,
            max_tokens=50,
            stop=["END"],
        ),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "chat.completion" and data["model"] == "m"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "Hello from NOVA"}
    assert data["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert (
        data["nova"]["routing"]["mode"] == "auto" and "decision_id" not in data["nova"]["routing"]
    )
    sent = runtime.payloads[-1]
    assert sent["temperature"] == 0.1 and sent["max_tokens"] == 50 and sent["stop"] == ["END"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert r.headers["x-request-id"].startswith("req_")
    assert client.get("/conversations").json()["conversations"] == []  # nicht im UI-Verlauf


def test_no_usage_when_runtime_does_not_report(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime("a b", usage=False))))
    r = client.post(
        "/v1/chat/completions", json=chat_body(), headers=auth(key_for(service, ["chat:complete"]))
    )
    assert "usage" not in r.json() and "nova" not in r.json()


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"n": 2}, "n"),
        ({"tools": [{"type": "function", "function": {"name": "x"}}]}, "tools"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"logprobs": True}, "logprobs"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages[0].role"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://example.org/a.png"}}
                        ],
                    }
                ]
            },
            "messages[0].image_url",
        ),
    ],
)
def test_unsupported_features_are_rejected_not_ignored(
    make_client: Factory, extra: dict[str, Any], param: str
) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(**extra),
        headers=auth(key_for(service, ["chat:complete"])),
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_unknown_model(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(model="gpt-4o"),
        headers=auth(key_for(service, ["chat:complete"])),
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_unavailable_model(make_client: Factory) -> None:
    client, service = make_client(engine_with(("a", SSERuntime()), ("b", SSERuntime())))
    service.engine.providers.get("p-b").served = None  # type: ignore[attr-defined]
    service.availability.mark_unavailable("b", "Runtime nicht erreichbar")
    r = client.post(
        "/v1/chat/completions",
        json=chat_body(model="b"),
        headers=auth(key_for(service, ["chat:complete"])),
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_streaming_nova_and_openai_format(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime("one two three"))))
    key = key_for(service, ["chat:complete"])
    raw = client.post(
        "/v1/chat/completions",
        headers=auth(key),
        json=chat_body(stream=True, stream_options={"include_usage": True}),
    ).text
    lines = [line[6:] for line in raw.split("\n\n") if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "one two three"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 3 and chunks[-1]["choices"] == []
    assert all(c.get("object") == "chat.completion.chunk" for c in chunks)
    nova = client.post(
        "/api/v1/chat/completions", headers=auth(key), json=chat_body(stream=True)
    ).text
    objects = [
        json.loads(x[6:]).get("object") for x in nova.split("\n\n") if x.startswith("data: {")
    ]
    assert objects[0] == "nova.routing" and objects[-1] == "nova.stats"


async def test_timeout(tmp_path: Path) -> None:
    from api.config import ApiConfig
    from api.service import NovaService, _Timeout

    class Slow(StubProvider):
        async def stream(self, model, request):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(5)
            yield StreamChunk(delta="late", finish_reason=FinishReason.STOP)

    engine = InferenceEngine(
        ModelRegistry([make_meta(provider="local")]),
        ProviderRegistry([Slow("local", served=["test-model"])]),
    )
    service = NovaService(ApiConfig(dev_mode=True, data_dir=tmp_path), engine=engine)
    from models.base import ChatRequest, Message

    started = time.perf_counter()
    with pytest.raises(_Timeout):
        async for _ in service.generate(ChatRequest((Message.user("x"),)), deadline_s=0.3):
            pass
    assert time.perf_counter() - started < 2


def test_openai_timeout_maps_to_504(make_client: Factory) -> None:
    class Slow(StubProvider):
        async def stream(self, model, request):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(5)
            yield StreamChunk(delta="late")

    engine = InferenceEngine(
        ModelRegistry([make_meta(provider="local")]),
        ProviderRegistry([Slow("local", served=["test-model"])]),
    )
    client, service = make_client(engine)
    r = client.post(
        "/v1/chat/completions",
        headers=auth(key_for(service, ["chat:complete"])),
        json=chat_body(nova_timeout_s=0.3),
    )
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


async def test_client_disconnect_cancels_generation(tmp_path: Path) -> None:
    """Abbruch: Schließt der Client den Stream, endet die Generierung bei der Runtime."""
    from api.config import ApiConfig
    from api.service import NovaService
    from models.base import ChatRequest, Message

    produced: list[int] = []
    closed: list[bool] = []

    class Endless(StubProvider):
        async def stream(self, model, request):  # type: ignore[no-untyped-def,override]
            try:
                for i in range(10_000):
                    produced.append(i)
                    await asyncio.sleep(0.005)
                    yield StreamChunk(delta=f"{i} ")
            finally:
                closed.append(True)

    engine = InferenceEngine(
        ModelRegistry([make_meta(provider="local")]),
        ProviderRegistry([Endless("local", served=["test-model"])]),
    )
    service = NovaService(ApiConfig(dev_mode=True, data_dir=tmp_path), engine=engine)
    events = service.generate(ChatRequest((Message.user("x"),)))
    async for event in events:
        if event["event"] == "token" and len(produced) > 5:
            break
    await events.aclose()
    count = len(produced)
    await asyncio.sleep(0.1)
    assert closed == [True] and len(produced) == count  # Runtime-Stream beendet


# ---------------------------------------------------------------------- Offizieller OpenAI-Client


def test_official_openai_sdk(make_client: Factory) -> None:
    openai = pytest.importorskip("openai")
    client, service = make_client(engine_with(("local-model", SSERuntime("Hi from NOVA"))))
    key = key_for(service, ["chat:complete", "models:read"])
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0
    )
    assert [m.id for m in sdk.models.list()] == ["nova-auto", "local-model"]
    result = sdk.chat.completions.create(
        model="nova-auto", messages=[{"role": "user", "content": "Hello"}]
    )
    assert result.choices[0].message.content == "Hi from NOVA"
    assert result.model == "local-model" and result.usage is not None
    stream = sdk.chat.completions.create(
        model="local-model", stream=True, messages=[{"role": "user", "content": "Hello"}]
    )
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hi from NOVA"
    with pytest.raises(openai.AuthenticationError):
        openai.OpenAI(
            base_url="http://testserver/v1", api_key="bad", http_client=client, max_retries=0
        ).models.list()
    with pytest.raises(openai.BadRequestError) as info:
        sdk.chat.completions.create(
            model="nova-auto", n=2, messages=[{"role": "user", "content": "x"}]
        )
    assert info.value.code == "unsupported_parameter"


# ---------------------------------------------------------------------- Agent-Tasks


def agent_engine() -> InferenceEngine:
    from tests.agents.fakes import engine_for

    return engine_for(ScriptedProvider(execute=["Recursion means a function calls itself."]))


def poll(client: Any, key: str, task_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data: dict[str, Any] = client.get(
            f"/api/v1/agent/tasks/{task_id}", headers=auth(key)
        ).json()
        if data["status"] not in ("queued", "running"):
            return data
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_agent_task_lifecycle(make_client: Factory) -> None:
    client, service = make_client(agent_engine())
    with client:
        key = key_for(service, ["agent:run", "agent:read"])
        r = client.post(
            "/api/v1/agent/tasks", headers=auth(key), json={"objective": "Erkläre kurz Rekursion"}
        )
        assert r.status_code == 202, r.text
        task = r.json()
        assert task["id"].startswith("task_") and task["status"] in ("queued", "running")
        assert r.headers["location"] == f"/api/v1/agent/tasks/{task['id']}"
        done = poll(client, key, task["id"])
        assert done["status"] == "succeeded"
        assert done["result"]["verification"]["status"] == "unverified"  # ehrlich
        assert done["result"]["answer"]
        assert any(e["phase"] == "finalize" for e in done["events"])
        other = key_for(service, ["agent:read"], name="other")
        assert (
            client.get(f"/api/v1/agent/tasks/{task['id']}", headers=auth(other)).status_code == 404
        )  # fremde Tasks unsichtbar


def test_agent_tools_require_explicit_scope(make_client: Factory) -> None:
    client, service = make_client(agent_engine())
    with client:
        key = key_for(service, ["agent:run"])
        r = client.post(
            "/api/v1/agent/tasks",
            headers=auth(key),
            json={"objective": "Read file", "allowed_tools": ["read_file"]},
        )
        assert r.status_code == 403 and r.json()["error"]["code"] == "tool_not_allowed"
        unknown = client.post(
            "/api/v1/agent/tasks",
            headers=auth(key),
            json={"objective": "x", "allowed_tools": ["format_disk"]},
        )
        assert unknown.status_code == 403
        granted = key_for(service, ["agent:run", "agent:tool:read_file"], name="reader")
        ok = client.post(
            "/api/v1/agent/tasks",
            headers=auth(granted),
            json={"objective": "Read file", "allowed_tools": ["read_file"]},
        )
        assert ok.status_code == 202 and ok.json()["allowed_tools"] == ["read_file"]


def test_agent_task_scopes_read_and_cancel(make_client: Factory) -> None:
    class Hanging(ScriptedProvider):
        async def chat(self, model, request):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(30)
            return await super().chat(model, request)

    from tests.agents.fakes import engine_for

    client, service = make_client(engine_for(Hanging(execute=["x"])))
    with client:
        runner = key_for(service, ["agent:run"])
        task = client.post(
            "/api/v1/agent/tasks",
            headers=auth(runner),
            json={"objective": "Erkläre kurz Rekursion"},
        ).json()
        assert (
            client.get(f"/api/v1/agent/tasks/{task['id']}", headers=auth(runner)).status_code == 403
        )  # agent:read fehlt
        assert (
            client.post(
                f"/api/v1/agent/tasks/{task['id']}/cancel", headers=auth(runner)
            ).status_code
            == 403
        )
        service.integrations.update_scopes(
            service.integrations.authenticate(runner).id,  # type: ignore[union-attr]
            ["agent:run", "agent:read", "agent:cancel"],
        )
        cancelled = client.post(f"/api/v1/agent/tasks/{task['id']}/cancel", headers=auth(runner))
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        events = [e["event"] for e in service.audit.recent]
        assert "agent_task_created" in events and "agent_task_finished" in events


def test_agent_task_timeout(make_client: Factory) -> None:
    class Hanging(ScriptedProvider):
        async def chat(self, model, request):  # type: ignore[no-untyped-def,override]
            await asyncio.sleep(30)
            return await super().chat(model, request)

    from tests.agents.fakes import engine_for

    client, service = make_client(engine_for(Hanging(execute=["x"])))
    with client:
        key = key_for(service, ["agent:run", "agent:read"])
        task = client.post(
            "/api/v1/agent/tasks",
            headers=auth(key),
            json={"objective": "Erkläre kurz Rekursion", "timeout_s": 0.3},
        ).json()
        done = poll(client, key, task["id"])
        assert done["status"] == "timed_out" and done["error"]["code"] == "timeout"


def test_agent_task_without_model(make_client: Factory) -> None:
    client, service = make_client()
    r = client.post(
        "/api/v1/agent/tasks",
        headers=auth(key_for(service, ["agent:run"])),
        json={"objective": "x"},
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_model"


def test_unfinished_tasks_marked_after_restart(tmp_path: Path) -> None:
    from api.agent_tasks import AgentTaskManager, AgentTaskRecord, TaskState

    state = tmp_path / "tasks"
    state.mkdir()
    record = AgentTaskRecord(id="task_x", owner="hq", objective="x", status=TaskState.RUNNING)
    (state / "task_x.json").write_text(json.dumps(record.to_dict()))
    manager = AgentTaskManager(agent_engine(), state)
    reloaded = manager.get("task_x", "hq")
    assert reloaded is not None and reloaded.status is TaskState.FAILED
    assert reloaded.error == {"code": "interrupted", "message": "NOVA was restarted"}


# ---------------------------------------------------------------------- Netzwerk & CORS


def test_remote_clients_are_rejected_by_default(make_client: Factory) -> None:
    client, service = make_client(trusted_clients=("127.0.0.1",))
    for path in ("/", "/system/status", "/api/v1/health"):
        r = client.get(path)
        assert r.status_code == 403, path
    assert service.audit.recent[-1]["event"] == "network_denied"


def test_remote_mode_must_be_explicit(make_client: Factory) -> None:
    client, _ = make_client(trusted_clients=("127.0.0.1",), allow_remote=True)
    assert client.get("/api/v1/health").status_code == 200


def test_cors_is_restrictive(make_client: Factory) -> None:
    client, service = make_client()
    key = key_for(service, ["models:read"])
    denied = client.get("/api/v1/models", headers={**auth(key), "Origin": "https://evil.example"})
    assert denied.status_code == 403
    assert "access-control-allow-origin" not in denied.headers
    allowed_client, svc2 = make_client(cors_origins=("https://hq.example",))
    key2 = key_for(svc2, ["models:read"])
    ok = allowed_client.get(
        "/api/v1/models", headers={**auth(key2), "Origin": "https://hq.example"}
    )
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://hq.example"
    preflight = allowed_client.options("/api/v1/models", headers={"Origin": "https://hq.example"})
    assert preflight.status_code == 204


def test_no_secrets_in_audit_log(make_client: Factory) -> None:
    client, service = make_client(engine_with(("m", SSERuntime())))
    key = key_for(service, ["models:read"])
    client.post("/api/v1/chat/completions", json=chat_body(), headers=auth(key))
    client.get("/api/v1/models", headers=auth("nova_123456789abc_" + "y" * 43))
    text = service.audit.path.read_text()  # type: ignore[union-attr]
    assert key not in text and "Hello there" not in text and "y" * 43 not in text
