"""NOVA API: Endpunkte, Streaming, Verlauf, Status, Fehler, Sicherheit, Development Mode."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from api.config import ApiConfig
from api.service import AttachmentIn, NovaService, ServiceError
from tests.agents.fakes import ScriptedProvider, engine_for
from tests.api.conftest import Factory, SSERuntime, engine_with, sse_events


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------- Development Mode


def test_dev_mode_without_model_is_honest(make_client: Factory) -> None:
    client, _ = make_client()
    system = client.get("/system/status").json()
    assert system["dev_mode"] is True and system["models_available"] == 0
    assert system["model_message"] == "No local model available."
    status = client.get("/models/status").json()
    assert status["any_available"] is False and status["message"] == "No local model available."
    r = client.post("/chat/stream", json={"message": "hi"})
    assert r.status_code == 503
    assert r.json()["error"] == {
        "code": "no_model",
        "message": "No local model available.",
        "detail": status["config_error"],
    }
    assert client.get("/conversations").json()["conversations"] == []  # kein Fake-Verlauf
    assert client.post("/chat", json={"message": "hi"}).status_code == 503
    agent = client.post("/agent/run", json={"task": "do it"})
    assert agent.status_code == 503 and agent.json()["error"]["code"] == "no_model"


def test_config_required_outside_dev_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--dev"):
        ApiConfig(models_config=None, dev_mode=False, data_dir=tmp_path)
    with pytest.raises(ValueError, match="not found"):
        ApiConfig(models_config=tmp_path / "missing.toml", dev_mode=False, data_dir=tmp_path)


def test_ui_is_served(make_client: Factory) -> None:
    client, _ = make_client()
    r = client.get("/")
    assert r.status_code == 200 and "<title>NOVA</title>" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/js/app.js").status_code == 200


# ---------------------------------------------------------------------- Chat


def test_chat_stream_full_flow(make_client: Factory) -> None:
    runtime = SSERuntime("Hello **world** from NOVA")
    client, service = make_client(engine_with(("test-model", runtime)))
    r = client.post("/chat/stream", json={"message": "Write a haiku about autumn"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    kinds = [e["event"] for e in events]
    assert kinds[:3] == ["run", "routing", "status"] and kinds[-1] == "done"
    assert "".join(e["delta"] for e in events if e["event"] == "token") == runtime.reply
    routing = events[1]["routing"]
    assert routing["mode"] == "auto" and routing["model"] == "test-model"
    assert routing["category"] and routing["complexity"] and routing["reason"]
    meta = events[-1]["message"]["meta"]
    assert meta["model"] == "test-model" and meta["streamed"] is True
    assert meta["completion_tokens"] == 4 and meta["prompt_tokens"] == 5
    assert meta["tokens_per_second"] > 0 and meta["generation_time_s"] > 0
    assert "verification" not in meta  # Chat ohne Verifikation → keine Angabe
    cid = events[0]["conversation_id"]
    conv = client.get(f"/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["title"] == "Write a haiku about autumn"
    router = client.get("/router/status").json()
    assert router["recent_decisions"][0]["selected_model"] == "test-model"
    assert not service.runs


def test_history_is_sent_on_followup(make_client: Factory) -> None:
    runtime = SSERuntime("first answer")
    client, _ = make_client(engine_with(("test-model", runtime)))
    first = sse_events(client.post("/chat/stream", json={"message": "Remember the code 4711"}).text)
    cid = first[0]["conversation_id"]
    client.post("/chat/stream", json={"message": "Which code?", "conversation_id": cid})
    messages = runtime.payloads[-1]["messages"]
    assert [m["content"] for m in messages] == [
        "Remember the code 4711",
        "first answer",
        "Which code?",
    ]


def test_missing_usage_means_no_token_stats(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("test-model", SSERuntime("a b c", usage=False))))
    meta = sse_events(client.post("/chat/stream", json={"message": "hi"}).text)[-1]["message"][
        "meta"
    ]
    assert "completion_tokens" not in meta and "tokens_per_second" not in meta
    assert meta["generation_time_s"] > 0


def test_non_streaming_chat_endpoint(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("test-model", SSERuntime("plain reply"))))
    data = client.post("/chat", json={"message": "hi"}).json()
    assert data["message"]["content"] == "plain reply"
    assert data["user_message"]["content"] == "hi"


def test_fallback_model_event(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("a", SSERuntime(fail=True)), ("b", SSERuntime("ok"))))
    client.put("/settings", json={"model": "auto"})
    events = sse_events(client.post("/chat/stream", json={"message": "hi there"}).text)
    done = events[-1]
    if events[1]["model"] == "a":
        assert any(e["event"] == "model" and e["fallback"] for e in events)
    assert done["event"] == "done" and done["message"]["meta"]["model"] == "b"


def test_runtime_failure_is_reported_not_faked(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("a", SSERuntime(fail=True))))
    events = sse_events(client.post("/chat/stream", json={"message": "hi"}).text)
    assert events[-1]["event"] == "error" and events[-1]["code"] == "model_error"
    cid = events[0]["conversation_id"]
    stored = client.get(f"/conversations/{cid}").json()["messages"][-1]
    assert stored["content"] == "" and stored["meta"]["error"]["code"] == "model_error"


def test_manual_model_selection(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("a", SSERuntime("from a")), ("b", SSERuntime("from b"))))
    assert client.put("/settings", json={"model": "b"}).json()["model"] == "b"
    events = sse_events(client.post("/chat/stream", json={"message": "hi"}).text)
    assert events[1]["routing"] == {"mode": "manual", "model": "b"}
    assert events[-1]["message"]["content"] == "from b"
    r = client.put("/settings", json={"model": "ghost"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_settings"


def test_empty_message_rejected(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("a", SSERuntime())))
    r = client.post("/chat/stream", json={"message": "   "})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------- Stoppen


class SlowRuntime(SSERuntime):
    pass


async def test_stop_keeps_partial_text_marked_as_stopped(tmp_path: Path) -> None:
    from models.base import FinishReason, StreamChunk
    from tests.models.fakes import StubProvider

    class Slow(StubProvider):
        async def stream(self, model, request):  # type: ignore[no-untyped-def,override]
            for i in range(100):
                await asyncio.sleep(0.01)
                yield StreamChunk(delta=f"t{i} ")
            yield StreamChunk(finish_reason=FinishReason.STOP)

    from models.inference import InferenceEngine, ProviderRegistry
    from models.model_registry import ModelRegistry
    from tests.models.fakes import make_meta

    engine = InferenceEngine(
        ModelRegistry([make_meta(provider="local")]),
        ProviderRegistry([Slow("local", served=["test-model"])]),
    )
    service = NovaService(ApiConfig(dev_mode=True, data_dir=tmp_path), engine=engine)
    events: list[dict[str, Any]] = []
    async for event in service.chat_stream(None, "hello", run_id="run-123"):
        events.append(event)
        if event["event"] == "token" and len(events) > 5:
            assert service.stop("run-123") is True
    assert events[-1]["event"] == "stopped"
    message = events[-1]["message"]
    assert message["meta"]["stopped"] is True and message["meta"]["stop_reason"] == "stopped"
    assert message["content"].startswith("t0 ")
    assert "run-123" not in service.runs
    assert service.stop("run-123") is False


async def test_client_disconnect_stores_partial(tmp_path: Path) -> None:
    runtime = SSERuntime(" ".join(f"w{i}" for i in range(50)))
    service = NovaService(
        ApiConfig(dev_mode=True, data_dir=tmp_path), engine=engine_with(("m", runtime))
    )
    stream = service.chat_stream(None, "hello")
    cid = None
    async for event in stream:
        cid = event.get("conversation_id", cid)
        if event["event"] == "token":
            break
    await stream.aclose()
    assert cid is not None
    last = service.store.messages(cid)[-1]
    assert last.meta["stopped"] is True and last.meta["stop_reason"] == "disconnected"


def test_stop_endpoint_unknown_run(make_client: Factory) -> None:
    client, _ = make_client()
    assert client.post("/agent/stop", json={"run_id": "nope"}).json() == {
        "run_id": "nope",
        "stopped": False,
    }


# ---------------------------------------------------------------------- Anhänge


def test_text_attachment_goes_into_prompt(make_client: Factory) -> None:
    runtime = SSERuntime("seen")
    client, _ = make_client(engine_with(("m", runtime)))
    att = {"name": "notes.md", "mime": "text/markdown", "data_b64": b64(b"# Secret plan\n42")}
    events = sse_events(
        client.post("/chat/stream", json={"message": "Summarize", "attachments": [att]}).text
    )
    prompt = runtime.payloads[-1]["messages"][-1]["content"]
    assert "[File: notes.md]" in prompt and "# Secret plan" in prompt
    user = events[0]["user_message"]
    assert user["attachments"] == [
        {"name": "notes.md", "mime": "text/markdown", "size": 16, "kind": "text"}
    ]  # Textinhalt wird nicht zurückgeschickt


def test_image_attachment_stored_and_served(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("m", SSERuntime("ok"))))
    png = b"\x89PNG\r\n\x1a\nfake"
    att = {"name": "shot.png", "mime": "image/png", "data_b64": b64(png)}
    r = client.post("/chat/stream", json={"message": "What is this?", "attachments": [att]})
    events = sse_events(r.text)
    stored = events[0]["user_message"]["attachments"][0]
    assert stored["kind"] == "image"
    assert client.get(f"/uploads/{stored['file']}").content == png
    assert client.get("/uploads/../conversations.db").status_code == 404
    # Flotte ohne Vision → ehrlicher Routing-Fehler statt Antwort ohne Bild
    assert events[-1]["event"] == "error" and events[-1]["code"] == "no_suitable_model"


@pytest.mark.parametrize(
    ("att", "code"),
    [
        (
            {"name": "a.pdf", "mime": "application/pdf", "data_b64": b64(b"%PDF")},
            "unsupported_attachment",
        ),
        ({"name": "a.txt", "mime": "text/plain", "data_b64": "!!!"}, "invalid_attachment"),
        (
            {"name": "a.txt", "mime": "text/plain", "data_b64": b64(b"\xff\xfe\xfa")},
            "invalid_attachment",
        ),
    ],
)
def test_bad_attachments_rejected(make_client: Factory, att: dict[str, str], code: str) -> None:
    client, _ = make_client(engine_with(("m", SSERuntime())))
    r = client.post("/chat/stream", json={"message": "x", "attachments": [att]})
    assert r.status_code == 400 and r.json()["error"]["code"] == code


def test_attachment_size_limit(tmp_path: Path) -> None:
    service = NovaService(
        ApiConfig(dev_mode=True, data_dir=tmp_path, max_attachment_bytes=10),
        engine=engine_with(("m", SSERuntime())),
    )
    with pytest.raises(ServiceError, match="larger than"):
        service._attachments([AttachmentIn("big.txt", "text/plain", b64(b"x" * 11))])


# ---------------------------------------------------------------------- Verlauf & Settings


def test_conversation_crud(make_client: Factory) -> None:
    client, _ = make_client()
    created = client.post("/conversations").json()
    cid = created["id"]
    assert client.patch(f"/conversations/{cid}", json={"title": "Plans"}).json()["title"] == "Plans"
    assert [c["id"] for c in client.get("/conversations").json()["conversations"]] == [cid]
    assert client.delete(f"/conversations/{cid}").json() == {"deleted": cid}
    assert client.get(f"/conversations/{cid}").status_code == 404
    assert client.delete(f"/conversations/{cid}").status_code == 404


def test_settings_validation_and_persistence(make_client: Factory, tmp_path: Path) -> None:
    client, service = make_client()
    assert client.get("/settings").json()["model"] == "auto"
    updated = client.put("/settings", json={"temperature": 0.2, "system_prompt": "Be brief"})
    assert updated.json()["temperature"] == 0.2
    assert service.settings_store.load().system_prompt == "Be brief"
    for bad in ({"temperature": 5}, {"unknown": 1}, {"agent_workspace": str(tmp_path / "x")}):
        r = client.put("/settings", json=bad)
        assert r.status_code == 400, bad


def test_models_endpoints(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("m", SSERuntime())))
    models = client.get("/models").json()["models"]
    assert models[0]["name"] == "m" and models[0]["data_status"] == "UNMEASURED"
    status = client.get("/models/status?refresh=true").json()
    assert status["any_available"] and status["models"][0]["available"] is True
    assert status["providers"][0]["ready"] is True


# ---------------------------------------------------------------------- Sicherheit


def test_mutations_require_client_header_and_same_origin(make_client: Factory) -> None:
    client, _ = make_client()
    r = client.post("/conversations", headers={"X-NOVA-Client": ""})
    assert r.status_code in (200, 403)
    from fastapi.testclient import TestClient

    raw = TestClient(client.app)
    assert raw.post("/conversations").status_code == 403
    assert (
        raw.post(
            "/conversations", headers={"X-NOVA-Client": "x", "Origin": "http://evil.example"}
        ).status_code
        == 403
    )
    assert (
        raw.post(
            "/conversations", headers={"X-NOVA-Client": "x", "Origin": "http://testserver"}
        ).status_code
        == 200
    )


def test_optional_api_token(make_client: Factory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVA_API_TOKEN", "s3cret")
    client, _ = make_client()
    assert client.get("/conversations").status_code == 401
    assert client.get("/conversations", headers={"X-NOVA-Token": "s3cret"}).status_code == 200
    assert client.get("/system/status").json()["auth_required"] is True
    assert client.get("/").status_code == 200


# ---------------------------------------------------------------------- Agent


def test_agent_disabled_without_workspace(make_client: Factory) -> None:
    client, _ = make_client(engine_with(("m", SSERuntime())))
    r = client.post("/agent/run", json={"task": "do it"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "agent_disabled"


def test_agent_run_reports_real_verification(make_client: Factory, tmp_path: Path) -> None:
    provider = ScriptedProvider(execute=["Recursion means a function calls itself."])
    client, _ = make_client(engine_for(provider))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client.put("/settings", json={"agent_workspace": str(workspace)})
    events = sse_events(client.post("/agent/run", json={"task": "Erkläre kurz Rekursion"}).text)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run" and "phase" in kinds and kinds[-1] == "done"
    meta = events[-1]["message"]["meta"]
    assert meta["agent"] is True
    assert meta["verification"]["status"] == "unverified"  # ehrlich: nichts prüfbar
    assert meta["verification"]["verified"] is False
    assert events[-1]["message"]["content"]
