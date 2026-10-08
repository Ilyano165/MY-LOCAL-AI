# models/

Model-Engine-Layer: einheitliche Schnittstelle zu lokalen LLMs. Enthält **keine**
Modellgewichte (die liegen außerhalb des Repos, Default `~/.nova/models`) und **keine**
hardcodierten Modellnamen (erzwungen durch `tests/test_no_hardcoded_models.py`).

| Datei | Inhalt |
|---|---|
| `capabilities.py` | `ModelMetadata` (name, provider, parameter_count, context_length, reasoning/coding/vision_capability, tool_calling, speed, memory_requirement, quantization, local_path), `CapabilityLevel`, `Speed`, `TaskType`, `TaskRequirements`, Ranking |
| `base.py` | `ModelProvider` (ABC), Nachrichten-/Request-/Response-Typen, Fehler-Hierarchie |
| `local_provider.py` | `OpenAICompatibleProvider` – lokale Runtime per HTTP (Referenz: llama.cpp `llama-server`; auch Ollama `/v1`, vLLM, LM Studio). Nur Loopback, außer `allow_remote=True` |
| `model_registry.py` | `ModelRegistry`: `register`, `get`, `list`, `find_best_for(task)`, `rank_for`, Laden aus TOML |
| `inference.py` | `InferenceEngine` (Auswahl per Aufgabe oder Registry-Name, harter Timeout, optionaler Fallback), `ProviderRegistry`, Backend-Factory `PROVIDER_TYPES` |
| `health.py` | `ModelHealthChecker`: present → loadable → inference → response_time → context (Needle) → tool_calling |

## Verwendung

```python
engine = InferenceEngine.from_toml("~/.nova/models.toml")
result = await engine.chat(
    ChatRequest(messages=(Message.user("…"),)), task=TaskType.CODING, fallback=True
)
```

Health Check: `python -m scripts.model_health --config ~/.nova/models.toml`
Konfigurationsvorlage: `config/models.example.toml`

## Neues Backend hinzufügen

1. Klasse von `ModelProvider` ableiten (`chat`, `list_models`, `health`, optional `ensure_loaded`).
2. `register_provider_type("mein_typ", factory)` oder Eintrag in `PROVIDER_TYPES`.
3. In der Config `type = "mein_typ"` verwenden. Restlicher Code bleibt unverändert.
