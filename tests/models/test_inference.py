from __future__ import annotations

import pytest

from models.base import (
    ChatRequest,
    InvalidResponseError,
    Message,
    ModelNotFoundError,
    ModelProvider,
    ModelTimeoutError,
    NoSuitableModelError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
)
from models.capabilities import TaskType
from models.inference import (
    PROVIDER_TYPES,
    InferenceEngine,
    ProviderRegistry,
    create_provider,
    register_provider_type,
)
from models.local_provider import OpenAICompatibleProvider
from models.model_registry import ModelRegistry
from tests.models.fakes import StubProvider, make_meta

REQUEST = ChatRequest(messages=(Message.user("hi"),))


def engine_with(provider: StubProvider, **kwargs: float) -> InferenceEngine:
    models = ModelRegistry(
        [
            make_meta(name="coder", coding_capability="strong"),
            make_meta(name="backup", coding_capability="good"),
            make_meta(name="tiny", coding_capability="basic", speed="fast"),
        ]
    )
    return InferenceEngine(models, ProviderRegistry([provider]), **kwargs)


async def test_chat_by_task_picks_best_model() -> None:
    result = await engine_with(StubProvider()).chat(REQUEST, task=TaskType.CODING)
    assert result.model.name == "coder"
    assert result.response.message.content == "answer from coder"
    assert result.attempts == ("coder",)


async def test_chat_by_explicit_registry_name() -> None:
    result = await engine_with(StubProvider()).chat(REQUEST, model="tiny")
    assert result.model.name == "tiny"


async def test_default_task_is_general() -> None:
    engine = engine_with(StubProvider())
    assert engine.resolve().name == engine.models.find_best_for(TaskType.GENERAL).name


def test_resolve_rejects_model_and_task() -> None:
    with pytest.raises(ValueError):
        engine_with(StubProvider()).resolve(model="coder", task="coding")


async def test_unknown_model_and_provider() -> None:
    engine = engine_with(StubProvider())
    with pytest.raises(ModelNotFoundError):
        await engine.chat(REQUEST, model="ghost")
    engine.models.register(make_meta(name="orphan", provider="nowhere"))
    with pytest.raises(ProviderNotRegisteredError):
        await engine.chat(REQUEST, model="orphan")
    assert engine.validate() == ["orphan: Provider 'nowhere' nicht registriert"]


async def test_hard_timeout() -> None:
    engine = engine_with(StubProvider(behaviors={"coder": 0.5}), default_timeout_s=0.05)
    with pytest.raises(ModelTimeoutError, match=r"0\.[01]s"):
        await engine.chat(REQUEST, model="coder")
    request = ChatRequest(messages=REQUEST.messages, timeout_s=0.01)
    with pytest.raises(ModelTimeoutError):
        await engine_with(StubProvider(behaviors={"coder": 0.5})).chat(request, model="coder")


async def test_fallback_to_next_best_model() -> None:
    provider = StubProvider(behaviors={"coder": ProviderUnavailableError("down")})
    result = await engine_with(provider).chat(REQUEST, task=TaskType.CODING, fallback=True)
    assert result.model.name == "backup"
    assert result.attempts == ("coder", "backup")


async def test_fallback_exhausted() -> None:
    provider = StubProvider(
        behaviors={n: ProviderUnavailableError(f"{n} down") for n in ("coder", "backup", "tiny")}
    )
    with pytest.raises(ProviderUnavailableError, match="Alle geeigneten Modelle"):
        await engine_with(provider).chat(REQUEST, task=TaskType.CODING, fallback=True)
    assert provider.calls == ["coder", "backup", "tiny"]


async def test_non_retryable_error_does_not_fallback() -> None:
    provider = StubProvider(behaviors={"coder": InvalidResponseError("garbage")})
    with pytest.raises(InvalidResponseError):
        await engine_with(provider).chat(REQUEST, task=TaskType.CODING, fallback=True)
    assert provider.calls == ["coder"]


async def test_no_fallback_for_explicit_model() -> None:
    provider = StubProvider(behaviors={"coder": ProviderUnavailableError("down")})
    with pytest.raises(ProviderUnavailableError):
        await engine_with(provider).chat(REQUEST, model="coder", fallback=True)


async def test_aclose_closes_providers() -> None:
    provider = StubProvider()
    await engine_with(provider).aclose()
    assert provider.closed


def test_provider_registry_duplicates() -> None:
    reg = ProviderRegistry([StubProvider("a")])
    with pytest.raises(ValueError):
        reg.register(StubProvider("a"))
    reg.register(StubProvider("a"), replace=True)
    assert reg.names() == ["a"]


def test_engine_from_mapping() -> None:
    config = {
        "inference": {"default_timeout_s": 42},
        "providers": [
            {"name": "local", "type": "openai_compatible", "base_url": "http://127.0.0.1:8080"}
        ],
        "models": [make_meta(name="m").to_dict()],
    }
    engine = InferenceEngine.from_mapping(config)
    assert engine.default_timeout_s == 42
    assert isinstance(engine.providers.get("local"), OpenAICompatibleProvider)
    assert engine.models.get("m").provider == "local"


def test_engine_from_mapping_rejects_dangling_provider() -> None:
    config = {"providers": [], "models": [make_meta(name="m").to_dict()]}
    with pytest.raises(ValueError, match="nicht registriert"):
        InferenceEngine.from_mapping(config)


def test_create_provider_validation() -> None:
    with pytest.raises(ValueError, match="name"):
        create_provider({"type": "openai_compatible"})
    with pytest.raises(ValueError, match="Unbekannter Provider-Typ"):
        create_provider({"name": "x", "type": "quantum"})


def test_register_provider_type_extends_factory() -> None:
    def factory(cfg: object) -> ModelProvider:
        return StubProvider(str(cfg["name"]))  # type: ignore[index]

    try:
        register_provider_type("stub", factory)
        assert isinstance(create_provider({"name": "s", "type": "stub"}), StubProvider)
        with pytest.raises(ValueError):
            register_provider_type("stub", factory)
    finally:
        PROVIDER_TYPES.pop("stub", None)


def test_invalid_default_timeout() -> None:
    with pytest.raises(ValueError):
        InferenceEngine(ModelRegistry(), ProviderRegistry(), default_timeout_s=0)


def test_no_suitable_model_propagates() -> None:
    engine = InferenceEngine(ModelRegistry(), ProviderRegistry())
    with pytest.raises(NoSuitableModelError):
        engine.resolve(task=TaskType.VISION)
