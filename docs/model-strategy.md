# NOVA – Modellstrategie

Status: Entwurf v0.1 · Stand: 2026-10-08

> In dieser Phase werden **keine** Modelle heruntergeladen und **kein** Fine-Tuning durchgeführt.
> Dieses Dokument legt fest, *wie* Modelle ausgewählt, eingebunden und gewechselt werden.

---

## 1. Grundsätze

1. **Modelle sind Konfiguration, nicht Code.** Ein Modellwechsel ist eine Änderung in
   `config/`, keine Architekturänderung.
2. **Rollen statt Modellnamen.** Der Code spricht von `CODING`, `REASONING` usw.; welches
   Modell eine Rolle erfüllt, steht in der Registry.
3. **Auswahl per eigener Messung.** Öffentliche Benchmarks sind nur eine Vorauswahl; die
   Entscheidung fällt über NOVAs eigene Eval-Suites (`evaluation.md`) auf der Zielhardware.
4. **Fähigkeiten ehrlich deklarieren.** Kontextfenster, Tool-Calling, Vision und
   JSON-Schema-Support werden pro Modell in der Registry gepflegt und durch einen
   Capability-Check (Integrationstest) verifiziert.
5. **Lizenz ist ein hartes Kriterium.** Bevorzugt Apache-2.0/MIT; Modelle mit
   Non-Commercial- oder restriktiven Lizenzen nur nach bewusster Entscheidung.

## 2. Rollen

| Rolle | Zweck | Anforderungen | Priorität |
|------|-------|---------------|-----------|
| `FAST` | Klassifikation, Zusammenfassung, kurze Antworten, Kontextkompression | niedrige Latenz, klein, Tool-Calling wünschenswert | hoch |
| `GENERAL` | Standard-Assistent, Schreiben, Analyse | gutes Instruction-Following, Tool-Calling, ≥ 32k Kontext | hoch |
| `CODING` | Code lesen/schreiben, Agentic Coding | stark in Code + Tool-Calling, ≥ 64k Kontext | hoch |
| `REASONING` | Planung, Mathe, mehrstufige Analyse | Reasoning-/Thinking-Modus, strukturierte Ausgabe | mittel |
| `VISION` | Bilder, Screenshots, Diagramme | multimodaler Input | niedrig (Phase 10) |
| `EMBEDDING` | RAG, Memory-Suche | mehrsprachig (DE/EN), gute Retrieval-Qualität | hoch (Phase 6) |
| `RERANK` | Neusortierung von Retrieval-Kandidaten | Cross-Encoder, klein | mittel (Phase 6) |

Eine Rolle darf auf dasselbe Modell zeigen wie eine andere. Auf begrenzter Hardware ist
**ein starkes Generalisten-/Coding-Modell + ein kleines FAST-Modell + Embedder** der
realistische Start.

## 3. Runtimes (Inferenz-Backends)

NOVA spricht Runtimes über HTTP an (ADR-002). Bewertung nach Recherche (Stand 10/2026,
Quellen unten; vor Einsatz gegen aktuelle Projektdokumentation prüfen):

| Runtime | Stärken | Schwächen | Einsatz in NOVA |
|--------|---------|-----------|-----------------|
| **llama.cpp** (`llama-server`) | maximale Kontrolle (Quantisierung, Offload), CPU+GPU+Apple Silicon, OpenAI-kompatible API, Grammatik/JSON-Schema-Constrained-Output | manuelle Modellverwaltung | Referenz-Backend für Einzelplatz |
| **Ollama** | einfachste Modellverwaltung, OpenAI-kompatible + native API, Embeddings | eigene Abstraktion über llama.cpp, Features teils verzögert, standardmäßig wenig Parallelität | bequemer Einstieg, Embeddings |
| **vLLM** | hoher Durchsatz, Continuous Batching, robustes Tool-Calling/Structured Output | braucht starke NVIDIA/AMD-GPU, VRAM-Vorbelegung, Overhead für Einzelnutzer | nur bei passender GPU / vielen parallelen Agenten |
| **LM Studio** | GUI, OpenAI-kompatible API | proprietäre App | optional, über OpenAI-Adapter |

Konsequenz: Der erste Adapter ist `OpenAICompatibleProvider` – damit sind alle vier
abgedeckt. Runtime-spezifische Features (Ollama-Modellverwaltung, llama.cpp-Grammatiken)
kommen als optionale Erweiterungen.

## 4. Modell-Registry (Konfigurationsschema, Entwurf)

```toml
[[models]]
id            = "coder-main"            # interne ID, frei wählbar
backend       = "openai_compatible"     # | "ollama"
base_url      = "http://127.0.0.1:8080/v1"
model_name    = "<name wie von der Runtime erwartet>"
roles         = ["coding", "general"]
context_window = 65536
supports_tools = true
supports_vision = false
supports_json_schema = true
license       = "apache-2.0"
priority      = 10                      # höher = bevorzugt innerhalb der Rolle
# api_key_env = "NOVA_CODER_API_KEY"    # nur Name der Env-Variable, nie der Wert

[routing.fallbacks]
coding    = ["coder-main", "general-main"]
reasoning = ["reasoner", "general-main"]
```

## 5. Kandidaten (Vorauswahl, NICHT final)

Stand der Recherche: Oktober 2026. Die Landschaft ändert sich monatlich; Benchmark-Zahlen
stammen aus unterschiedlichen Harnesses und sind nicht direkt vergleichbar. **Jeder Kandidat
wird vor Einsatz auf Model-Card, Lizenz, Tool-Calling-Qualität und Speicherbedarf geprüft
und mit NOVAs Eval-Suite gemessen.** Die finale Auswahl hängt von der noch unbekannten
Zielhardware ab.

| Hardware-Stufe | FAST | GENERAL / CODING | REASONING | EMBEDDING / RERANK |
|---|---|---|---|---|
| **S** – ~16 GB RAM, keine/kleine GPU | Qwen3 4B, Phi-4-mini, Gemma (klein) | Qwen3 8B (quantisiert) | gleiches Modell im Thinking-Modus | Qwen3-Embedding-0.6B / bge-m3; bge-reranker-v2-m3 |
| **M** – 24–32 GB VRAM | Qwen3 4B | Qwen3.x-27B-Klasse, Gemma 4 26B-A4B (MoE), Devstral Small 2 | gpt-oss-20b | wie S, ggf. größeres Embedding-Modell |
| **L** – ≥ 64–128 GB (Unified Memory / Multi-GPU) | wie M | Qwen3-Coder-Next (80B MoE, ~3B aktiv), gpt-oss-120b | gpt-oss-120b | Qwen3-Embedding-4B/8B |

Nicht realistisch lokal (Frontier-Größe, nur mit Server-Infrastruktur): DeepSeek-V4-Pro,
Kimi K3, GLM 5.x u. ä. – als Referenzpunkt interessant, nicht als Planungsgrundlage.

## 6. Speicherbedarf abschätzen

```
Gewichte  ≈ Parameter × Bits_pro_Gewicht / 8
            (Q4_K_M ≈ 4,8 Bit; Q5_K_M ≈ 5,7 Bit; Q8_0 ≈ 8,5 Bit; FP16 = 16 Bit)
KV-Cache  ≈ 2 × Layer × KV-Heads × Head-Dim × Kontextlänge × Bytes_pro_Wert
Gesamt    ≈ Gewichte + KV-Cache + Runtime-Overhead (~0,5–1,5 GB)
```

Beispiel: 27B-Modell in Q4_K_M ≈ 16 GB Gewichte; bei langem Kontext kann der KV-Cache
mehrere GB zusätzlich belegen. Bei MoE-Modellen müssen **alle** Experten in den Speicher,
nur die Rechenlast richtet sich nach den aktiven Parametern.

Quantisierungsregel: Q4_K_M als Default, Q5/Q6 wenn Speicher reicht und Evals einen Gewinn
zeigen; unter Q4 nur für FAST-Rolle.

## 7. Auswahlprozess für ein neues Modell

1. Model-Card & Lizenz prüfen; Herkunft der Gewichte (offizielles Repo), Format
   (GGUF/safetensors – **keine** Pickle-Formate, siehe `security.md`), Checksumme.
2. Speicherbedarf nach §6 gegen Zielhardware rechnen.
3. Capability-Check (automatisierter Integrationstest): Tool-Calling, JSON-Schema-Output,
   Kontextlänge, Tokenizer. Danach `python -m scripts.benchmark_models --model <name>`:
   ersetzt die Konfigurationsannahmen durch gemessene Werte (`docs/benchmarking.md`).
4. Eval-Suite der vorgesehenen Rolle laufen lassen; Ergebnis gegen aktuelles Modell vergleichen.
5. Nur bei messbarem Gewinn (oder gleicher Qualität bei klar besserer Latenz) in die
   Registry übernehmen. Entscheidung mit Eval-Report in `docs/` festhalten.

## 8. Fine-Tuning (später, nicht jetzt)

Voraussetzungen, bevor überhaupt darüber entschieden wird:
- stabile Eval-Suite mit Baseline,
- ausreichend kuratierte, eigene Trace-Daten (erfolgreiche Läufe, korrigierte Fehlschläge),
- klar identifizierte Schwäche, die nicht durch Routing, Prompting, Tools oder RAG lösbar ist.

## Quellen (Recherche 10/2026)

- [HF Blog – Open-weight LLMs to run locally 2026](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally)
- [HF Blog – Best open-source LLMs 2026](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [BentoML – Open-source LLMs 2026](https://www.bentoml.com/blog/navigating-the-world-of-open-source-large-language-models)
- [Modal – Best open-source models for SWE-bench / coding agents](https://modal.com/resources/best-open-source-models-swe-bench-coding-agents)
- [Tembo – Best LLM for agentic coding 2026](https://www.tembo.io/blog/best-llm-for-agentic-coding)
- [Thunder Compute – Best open-source LLMs (Oct 2026)](https://www.thundercompute.com/blog/best-open-source-llms)
- [insiderllm – llama.cpp vs Ollama vs vLLM](https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/)
- [morphllm – vLLM vs Ollama 2026](https://www.morphllm.com/comparisons/vllm-vs-ollama)
- [d-central – Local embedding models for RAG 2026](https://d-central.tech/local-embedding-models/)
- [promptquorum – Best embedding models for local RAG 2026](https://www.promptquorum.com/power-local-llm/best-embedding-models-local-rag-2026)

Hinweis: Viele dieser Quellen sind Blog-/Vergleichsartikel, teils widersprüchlich. Sie dienen
der Vorauswahl, nicht als Beleg.
