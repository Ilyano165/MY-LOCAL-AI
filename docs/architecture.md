# NOVA – Technische Architektur

Status: Entwurf v0.1 · Stand: 2026-10-08 · Gilt für: Phase 0/1 (Foundation)

---

## 1. Zweck

NOVA ist eine vollständig lokal laufende, persönliche KI-Plattform. Das Ziel ist
nicht, ein Frontier-Modell per Prompting zu imitieren, sondern **ein System zu bauen**,
dessen Gesamtleistung aus dem Zusammenspiel vieler Komponenten entsteht:

```
Leistung(NOVA) = Modell × Routing × Planung × Werkzeuge × Kontext × Gedächtnis × Verifikation
```

Ein mittelgroßes lokales Modell mit guten Werkzeugen, präzisem Kontext, Selbstprüfung
(z. B. Tests ausführen) und Messung schlägt in konkreten Aufgaben oft ein größeres Modell
ohne diese Infrastruktur. Genau dort setzt die Architektur an.

## 2. Leitprinzipien (verbindlich)

| # | Prinzip | Konsequenz im Code |
|---|---------|--------------------|
| P1 | Lokal zuerst | Kernsystem hat keine Cloud-Abhängigkeit. Netzwerkzugriff nur über explizit freigegebene Tools. |
| P2 | Interfaces vor Implementierungen | `core/` definiert Verträge; alle anderen Pakete implementieren sie. |
| P3 | Austauschbarkeit | Modelle, Backends, Vektorspeicher, Embedder sind per Konfiguration wechselbar. |
| P4 | Keine Fake-Implementierungen | Test-Doubles (z. B. `ScriptedModelProvider`) existieren nur in `tests/` bzw. sind klar als Test-Hilfe deklariert. |
| P5 | Testbarkeit | Jede Komponente ist ohne echtes Modell testbar (Dependency Injection). |
| P6 | Beobachtbarkeit | Jeder Agent-Lauf erzeugt einen vollständigen, maschinenlesbaren Trace. |
| P7 | Sicherheit by Default | Tools mit Seiteneffekten sind standardmäßig gesperrt oder bestätigungspflichtig (siehe `security.md`). |
| P8 | Messen statt glauben | Änderungen an Prompts, Routing oder Modellen werden über `evaluation/` bewertet. |
| P9 | Klein anfangen | Keine verteilte Infrastruktur, keine Queues, kein Kubernetes. Ein Prozess, SQLite, Dateien. |

## 3. Annahmen und offene Fragen

Folgende Informationen fehlen und werden **nicht** erfunden:

| Thema | Status | Auswirkung |
|-------|--------|------------|
| Zielhardware (GPU/VRAM, RAM, CPU, Apple Silicon?) | **offen** | Bestimmt Modellgrößen, Quantisierung, Backend-Wahl. Architektur ist davon unabhängig. |
| Betriebssystem (Linux/macOS/Windows) | **offen** | Beeinflusst Sandbox-Mechanismus für Code-Ausführung. |
| Hauptsprache der Nutzung (DE/EN/gemischt) | **offen** | Relevant für Embedding-Modell und Evaluationsdatensätze. |
| Mehrbenutzerbetrieb | Annahme: **Einzelnutzer** | Kein Benutzer-/Rollenmodell in v1. |
| Implementierungssprache | Vorschlag: **Python ≥ 3.12** | Siehe §12. |

## 4. Systemübersicht

```
                         ┌─────────────────────────────────────────────┐
  Nutzer ──▶  app/       │  CLI  ·  lokale HTTP-API  ·  Web-UI          │
                         └──────────────────────┬──────────────────────┘
                                                │ (Composition Root: verdrahtet alles)
                         ┌──────────────────────▼──────────────────────┐
  agents/                │  Agent (Orchestrator / Coding-Agent / …)     │
                         │   ├─ Planner          (zerlegt Aufgaben)     │
                         │   ├─ ContextManager   (baut Prompt-Budget)   │
                         │   ├─ Verifier         (prüft Ergebnisse)     │
                         │   └─ Agent-Loop       (denken→handeln→prüfen)│
                         └───┬───────────┬───────────┬───────────┬─────┘
                             │           │           │           │
               ┌─────────────▼──┐ ┌──────▼─────┐ ┌───▼──────┐ ┌──▼───────────┐
  router/      │  ModelRouter   │ │  tools/    │ │ memory/  │ │  rag/        │
               │  Aufgabe→Rolle │ │ Registry,  │ │ Memory-  │ │ Ingest,      │
               │  Rolle→Modell  │ │ Sandbox,   │ │ Store    │ │ Retriever,   │
               └───────┬────────┘ │ Permissions│ └──────────┘ │ Reranker     │
                       │          └────────────┘              └──────────────┘
               ┌───────▼────────────────────────────────┐
  models/      │  ModelProvider-Adapter                  │
               │  OpenAI-kompatibel (llama.cpp-server,   │
               │  vLLM, LM Studio) · Ollama-nativ · …    │
               └───────┬────────────────────────────────┘
                       │ HTTP (localhost)
               ┌───────▼────────────────────────────────┐
               │  Lokale Inferenz-Runtime (extern)       │
               │  llama.cpp / Ollama / vLLM              │
               └────────────────────────────────────────┘

  Querschnitt:  core/ (Verträge, Typen, Fehler, Config, Logging/Tracing)
                evaluation/ (Evaluator, Suites, Metriken)   config/ (TOML)
```

**Wichtige Entscheidung:** NOVA lädt Modelle nicht selbst in den Prozess, sondern spricht
mit einer lokalen Inferenz-Runtime über HTTP auf `127.0.0.1`. Gründe:

- Modellwechsel und Runtime-Wechsel ohne Code-Änderung (P3).
- Der NOVA-Prozess bleibt schlank, testbar und stürzt nicht mit dem Modell ab.
- Alle relevanten Runtimes (llama.cpp `llama-server`, Ollama, vLLM, LM Studio) bieten eine
  OpenAI-kompatible Chat-Completions-API; ein Adapter deckt damit die meisten Fälle ab.
- Ein In-Process-Provider (z. B. `llama-cpp-python`) bleibt als weiterer Adapter möglich.

## 5. Verzeichnisstruktur und Abhängigkeitsregeln

```
/app         Composition Root, CLI, lokale API, UI
/core        Interfaces (Protocols/ABCs), Datentypen, Fehler, Config, Logging/Tracing
/models      ModelProvider-Implementierungen + Modell-Registry (KEINE Gewichte!)
/router      ModelRouter-Implementierungen, Task-Klassifikation
/agents      Agent-Implementierungen, Planner, ContextManager, Verifier
/tools       Tool-Implementierungen, ToolRegistry, Sandbox, Permissions
/memory      MemoryStore-Implementierungen
/rag         Ingestion, Chunking, Embedding, Retriever, Reranker
/evaluation  Evaluator-Implementierungen, Eval-Suites, Metriken, Reports
/tests       Unit-, Integrations- und Vertrags-Tests
/config      Default-Konfiguration (TOML), Beispiel-Overrides
/docs        Architektur, Roadmap, Strategien, ADRs
/scripts     Entwickler-Skripte (Setup, Lint, Eval-Läufe)
```

Modellgewichte liegen **außerhalb** des Repos (konfigurierbar, Default `~/.nova/models`)
und werden niemals eingecheckt.

### Abhängigkeitsregeln (werden per Test erzwungen)

```
core        → (nur Standardbibliothek + pydantic)
models      → core
router      → core
tools       → core
memory      → core
rag         → core
agents      → core            (bekommt Router/Tools/Memory/Retriever injiziert)
evaluation  → core            (bekommt Agents injiziert)
app         → alles           (einziger Ort, an dem konkrete Klassen verdrahtet werden)
```

Kein Paket außer `app` importiert eine konkrete Implementierung aus einem anderen Paket.
Ein Architektur-Test (`tests/architecture/test_import_rules.py`) prüft das per AST-Analyse.

## 6. Kern-Verträge (`core/`)

Alle Verträge sind **asynchron** (`async`), weil Modellaufrufe, Tool-Ausführung und I/O
dominieren. Datentypen sind unveränderliche `pydantic`-Modelle (Validierung + Serialisierung
für Traces). Die folgenden Signaturen sind der Zielzustand für Phase 1; Details werden bei
der Implementierung über Tests festgezurrt.

### 6.1 Gemeinsame Datentypen (Auszug)

```python
class Role(StrEnum): SYSTEM, USER, ASSISTANT, TOOL

class ContentPart:            # Text oder Bild (für Vision-Modelle)
    type: Literal["text", "image"]; text: str | None; image_path: Path | None

class Message:
    role: Role
    content: list[ContentPart]
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None

class ToolCall:   id: str; name: str; arguments: dict[str, JsonValue]

class GenerationParams:
    temperature: float = 0.2; top_p: float = 1.0; max_tokens: int | None = None
    stop: list[str] = []; seed: int | None = None
    response_schema: dict | None = None      # strukturierte Ausgabe (JSON Schema)

class ChatRequest:
    messages: list[Message]; tools: list[ToolSpec] = []; params: GenerationParams

class ChatResponse:
    message: Message; finish_reason: Literal["stop","length","tool_calls","error"]
    usage: TokenUsage; model_id: str; latency_ms: float

class ModelCapabilities:
    context_window: int; supports_tools: bool; supports_vision: bool
    supports_json_schema: bool; supports_streaming: bool; supports_embeddings: bool
```

### 6.2 `ModelProvider`

Kapselt **ein** Backend mit **einem** konkreten Modell. Kennt keine Aufgaben, keine Rollen.

```python
class ModelProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def capabilities(self) -> ModelCapabilities: ...
    async def chat(self, request: ChatRequest) -> ChatResponse: ...
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]: ...
    async def count_tokens(self, messages: list[Message]) -> int: ...
    async def health(self) -> HealthStatus: ...

class EmbeddingProvider(Protocol):          # separat: Embedder ≠ Chat-Modell
    @property
    def dimensions(self) -> int: ...
    async def embed(self, texts: list[str], kind: Literal["query","document"]) -> list[list[float]]: ...
```

Geplante Adapter (`models/`):
- `OpenAICompatibleProvider` – llama.cpp `llama-server`, vLLM, LM Studio, Ollama `/v1`.
- `OllamaProvider` – native Ollama-API (Modellverwaltung, `keep_alive`, Embeddings).
- Weitere nach Bedarf (z. B. In-Process `llama-cpp-python`), ohne Änderung an `core`.

Fähigkeiten, die ein Modell **nicht** nativ kann (z. B. Tool-Calls), werden **nicht**
vorgetäuscht: der Provider meldet `supports_tools=False`, und der Router wählt ein anderes
Modell bzw. der Agent nutzt ein explizites, getestetes Fallback-Format (JSON per
`response_schema`/Grammatik).

### 6.3 `ModelRouter`

Entscheidet, **welches Modell** eine Anfrage bearbeitet.

```python
class ModelRole(StrEnum):
    FAST, CODING, REASONING, VISION, GENERAL      # erweiterbar: EMBEDDING, RERANK, JUDGE

class TaskProfile:                 # Ergebnis der Klassifikation
    role: ModelRole; confidence: float
    needs_tools: bool; needs_vision: bool; est_context_tokens: int
    signals: dict[str, str]        # Begründung, landet im Trace

class RoutingDecision:
    provider: ModelProvider; role: ModelRole; profile: TaskProfile
    fallbacks: list[ModelProvider]; reason: str

class ModelRouter(Protocol):
    async def route(self, request: ChatRequest, hint: ModelRole | None = None) -> RoutingDecision: ...
```

Zweistufiges Verfahren:
1. **Klassifikation** (Aufgabe → Rolle): zuerst deterministische Regeln
   (Bild im Input → VISION; Code-Blöcke/Dateiendungen/Stacktraces → CODING;
   kurze Anfrage ohne Tools → FAST; explizite Planungs-/Mathe-Signale → REASONING).
   Später optional ein kleines Modell als Klassifikator. Ein expliziter `hint`
   (z. B. vom Planner) hat Vorrang.
2. **Auflösung** (Rolle → Modell): über die Modell-Registry aus `config/`, gefiltert nach
   harten Anforderungen (Kontextfenster, Tools, Vision, Health). Fallback-Kette pro Rolle.

Jede Entscheidung wird mit Begründung geloggt → Router-Genauigkeit ist messbar (`evaluation.md`).

### 6.4 `Tool`

```python
class Permission(StrEnum):
    READ_ONLY, WORKSPACE_WRITE, EXECUTE, NETWORK, DANGEROUS

class ToolSpec:            # das, was das Modell sieht
    name: str; description: str; parameters: dict   # JSON Schema

class ToolResult:
    ok: bool; output: str; data: JsonValue | None; error: str | None
    truncated: bool; duration_ms: float

class Tool(Protocol):
    spec: ToolSpec
    permissions: frozenset[Permission]
    async def run(self, args: dict[str, JsonValue], ctx: ToolContext) -> ToolResult: ...

class ToolContext:         # wird von der Laufzeit injiziert, nie vom Modell
    workspace_root: Path; trace: TraceSpan; policy: PermissionPolicy; cancel: CancelToken
```

Die `ToolRegistry` validiert Argumente gegen das JSON-Schema **bevor** ein Tool läuft, prüft
Berechtigungen über die `PermissionPolicy`, erzwingt Timeouts und kürzt Ausgaben.

### 6.5 `MemoryStore`

```python
class MemoryKind(StrEnum): EPISODIC, SEMANTIC, PROCEDURAL, PREFERENCE

class MemoryItem:
    id: str; kind: MemoryKind; content: str; metadata: dict
    created_at: datetime; source: str; importance: float

class MemoryStore(Protocol):
    async def add(self, item: MemoryItem) -> str: ...
    async def get(self, item_id: str) -> MemoryItem | None: ...
    async def search(self, query: str, *, kinds: set[MemoryKind] | None = None, limit: int = 10) -> list[ScoredMemory]: ...
    async def delete(self, item_id: str) -> bool: ...
    async def list(self, *, kinds=None, since=None, limit=100) -> list[MemoryItem]: ...
```

- **Kurzzeitgedächtnis** = Konversationsverlauf, verwaltet vom ContextManager (kein Store).
- **Langzeitgedächtnis** = `MemoryStore` (Start: SQLite + FTS5; Vektor-Suche optional).
- Schreibvorgänge ins Langzeitgedächtnis sind explizit (Tool `remember` oder
  Konsolidierung am Sitzungsende) und für den Nutzer einsehbar/löschbar.

### 6.6 `Retriever` (RAG)

```python
class Chunk:  id: str; doc_id: str; text: str; metadata: dict; span: tuple[int, int]
class RetrievedChunk: chunk: Chunk; score: float; retriever: str

class Retriever(Protocol):
    async def retrieve(self, query: str, *, k: int = 8, filters: dict | None = None) -> list[RetrievedChunk]: ...
```

Pipeline in `rag/`: `Loader → Chunker → Embedder → Index` (Ingestion) und
`Query → [BM25 ‖ Vektor] → Reciprocal Rank Fusion → (Reranker) → Top-k` (Abfrage).
Jede Stufe ist ein eigener austauschbarer Baustein; `Retriever` ist nur die Fassade.
Quellen werden immer mit Zitat-Referenz (`doc_id`, `span`) an den Agenten gegeben.

### 6.7 `Planner`

```python
class PlanStep:
    id: str; goal: str; role_hint: ModelRole | None
    depends_on: list[str]; success_criteria: str; status: StepStatus

class Plan: task: str; steps: list[PlanStep]; version: int

class Planner(Protocol):
    async def plan(self, task: str, context: PlanningContext) -> Plan: ...
    async def replan(self, plan: Plan, observation: StepOutcome) -> Plan: ...
```

Pläne werden als **strukturiertes JSON** (per `response_schema`) erzeugt, validiert und
versioniert. Jeder Schritt hat ein prüfbares Erfolgskriterium – das ist die Brücke zur
Selbstverifikation.

### 6.8 `Agent`

```python
class AgentTask: id: str; instruction: str; attachments: list[Path]; constraints: Budget
class Budget: max_steps: int; max_tokens: int; max_wall_seconds: float; max_tool_calls: int

class AgentResult:
    task_id: str; status: Literal["success","failed","aborted","needs_user"]
    output: str; artifacts: list[Path]; trace_id: str; usage: TokenUsage

class Agent(Protocol):
    name: str
    async def run(self, task: AgentTask, events: EventSink | None = None) -> AgentResult: ...
```

Abhängigkeiten (Router, ToolRegistry, MemoryStore, Retriever, Planner, Verifier, ContextManager)
werden über den Konstruktor injiziert. Der `EventSink` streamt Zwischenschritte an die UI.

### 6.9 `Evaluator`

```python
class EvalCase:    id: str; input: AgentTask; expected: JsonValue | None; checks: list[CheckSpec]; tags: list[str]
class EvalResult:  case_id: str; passed: bool; score: float; metrics: dict[str, float]; trace_id: str; notes: str

class Evaluator(Protocol):
    async def evaluate(self, case: EvalCase, result: AgentResult) -> EvalResult: ...
```

Details in `evaluation.md`.

## 7. Agent-Laufzeit (Request-Lebenszyklus)

```
1. app nimmt Aufgabe an → erzeugt trace_id, Budget
2. Agent: Kontext sammeln
      ├─ Memory.search(task)            (Präferenzen, frühere Ergebnisse)
      └─ Retriever.retrieve(task)       (nur wenn Wissensbasis relevant)
3. Planner.plan()  — nur wenn Aufgabe nicht trivial (Router-Signal / Heuristik)
4. für jeden Schritt (Schleife, budgetbegrenzt):
      a. ContextManager.build(step)     → Messages innerhalb des Token-Budgets
      b. Router.route(request, hint)    → Modell
      c. Provider.chat()                → Antwort oder Tool-Calls
      d. ToolRegistry.execute(calls)    → Ergebnisse (validiert, berechtigt, gekürzt)
      e. Verifier.check(step)           → bestanden / Korrektur / Replan
5. Abschluss: Endprüfung, Ergebnis, optional Memory-Konsolidierung
6. Trace wird vollständig persistiert
```

Abbruchbedingungen: Budget erschöpft, wiederholte identische Aktion (Schleifenerkennung),
Nutzerabbruch, nicht behebbarer Fehler. Jeder Abbruch ist ein expliziter Status, nie ein
stilles „Fertig".

## 8. Context Management

Lokale Modelle haben begrenzte und oft schlechter genutzte Kontextfenster. Der
`ContextManager` ist deshalb eine eigene Komponente (in `agents/`, Vertrag in `core/`):

- **Budgetierung:** Kontextfenster des gewählten Modells minus Antwortreserve; feste Anteile
  für System, Plan, Tool-Spezifikationen, Retrieval, Verlauf.
- **Priorisierung:** System-Prompt und aktueller Schritt > aktuelle Tool-Ergebnisse >
  relevante Retrieval-Chunks > ältere Historie.
- **Kompression:** alte Turns werden zusammengefasst (vom FAST-Modell), große Tool-Ausgaben
  gekürzt mit Verweis auf die vollständige Datei/Trace-ID statt sie zu löschen.
- **Exakte Token-Zählung:** über `ModelProvider.count_tokens` (Tokenizer des Modells);
  Heuristik nur als Fallback und dann mit Sicherheitsmarge.

## 9. Self-Verification

Verifikation ist eine eigene Schicht (`Verifier`), keine Prompt-Floskel. Rangfolge nach
Verlässlichkeit:

1. **Ausführbare Prüfungen** – Tests, Linter, Typchecker, Compiler, Schema-Validierung.
2. **Deterministische Checks** – Datei existiert, Ausgabe matcht Format, Zahl im Bereich.
3. **Konsistenzprüfungen** – Quellenabgleich bei RAG (Aussage durch zitierten Chunk gedeckt?).
4. **Modellbasierte Kritik** – zweites Modell/zweiter Durchlauf mit Rubrik. Nur, wenn 1–3 nicht
   anwendbar sind; Ergebnis gilt als schwaches Signal.

Ein Schritt gilt nur als erledigt, wenn sein `success_criteria` geprüft wurde.

## 10. Konfiguration und Secrets

- Format: **TOML** in `config/` (`default.toml`), Überschreibung durch `~/.nova/config.toml`
  und Umgebungsvariablen mit Präfix `NOVA_`. Validierung mit `pydantic-settings`.
- Modell-Registry ist Teil der Konfiguration (siehe `model-strategy.md`).
- **Secrets niemals im Code oder in eingecheckten Dateien.** Nur über Umgebungsvariablen,
  `.env` (gitignored) oder OS-Keyring. Details in `security.md`.

## 11. Logging, Tracing, Fehler

- **Strukturierte Logs** (JSON-Lines) über die Standardbibliothek `logging`, mit
  `trace_id`/`span_id` in jedem Eintrag. Secrets werden per Filter redigiert.
- **Traces:** Jeder Agent-Lauf schreibt einen Baum aus Spans (Routing, Modellaufruf inkl.
  Prompt/Antwort/Tokens/Latenz, Tool-Aufruf inkl. Argumente/Ergebnis, Verifikation) nach
  `~/.nova/traces/<trace_id>.jsonl`. Traces sind Grundlage für Debugging, Evaluation und
  späteres Fine-Tuning-Datenmaterial.
- **Fehler-Hierarchie** in `core/errors.py`: `NovaError` → `ModelError`
  (`ModelUnavailable`, `ContextOverflow`, `InvalidModelOutput`), `ToolError`
  (`ToolValidationError`, `PermissionDenied`, `ToolTimeout`), `RoutingError`,
  `RetrievalError`, `BudgetExceeded`, `ConfigError`. Keine nackten `except Exception`.

## 12. Technologie-Entscheidungen

| Bereich | Wahl | Begründung |
|--------|------|------------|
| Sprache | Python ≥ 3.12 | Größtes lokales-LLM-Ökosystem; alle Runtimes und Embedding-Libs verfügbar. |
| Paket-/Env-Management | `uv` + `pyproject.toml` + Lockfile | Reproduzierbar, schnell. |
| Datenmodelle | `pydantic` v2 | Validierung, JSON-Schema-Export für Tools/Pläne, Serialisierung für Traces. |
| HTTP | `httpx` (async) | Testbar über `MockTransport` ohne laufendes Modell. |
| Tests | `pytest`, `pytest-asyncio` | Standard. Marker `integration` für Tests mit echter Runtime. |
| Qualität | `ruff` (Lint+Format), `mypy --strict` | Verträge sind nur wertvoll, wenn sie typgeprüft sind. |
| Persistenz | SQLite (+ FTS5) | Null Infrastruktur, transaktional, lokal. |
| Vektorsuche | zunächst `sqlite-vec` **oder** LanceDB – Entscheidung in Phase 6 per Messung | Beide embedded, kein Server. Hinter `VectorIndex`-Interface. |
| API/UI | FastAPI (lokal, `127.0.0.1`) + schlanke Web-UI; CLI zuerst | Phase 9. |

Bewusst **nicht** in v1: LangChain/LlamaIndex als Kernabhängigkeit (zu viel Abstraktion über
unseren eigenen Verträgen; einzelne Bausteine dürfen hinter unseren Interfaces genutzt
werden), Datenbank-Server, Message-Queues, Container-Orchestrierung.

## 13. Teststrategie (Kurzfassung)

- **Vertragstests:** Eine gemeinsame Test-Suite pro Interface (`tests/contracts/`), die jede
  Implementierung bestehen muss (z. B. jeder `MemoryStore` findet, was er gespeichert hat).
- **Unit-Tests:** ohne Netzwerk, ohne Modell; Modelle über einen deterministischen
  `ScriptedModelProvider` (Test-Double in `tests/`), HTTP über `httpx.MockTransport`.
- **Integrationstests:** `@pytest.mark.integration`, laufen nur, wenn eine lokale Runtime
  konfiguriert ist; mit einem sehr kleinen Modell.
- **Architekturtests:** Importregeln aus §5.
- **Evaluation** ist getrennt von Tests (Qualität statt Korrektheit) – siehe `evaluation.md`.

## 14. Architekturentscheidungen (ADR-Liste)

| ADR | Entscheidung | Status |
|-----|--------------|--------|
| 001 | Top-Level-Verzeichnisse wie vorgegeben (`/core`, `/models`, …) als Python-Pakete. Risiko: generische Namen (`app`, `core`, `models`) können mit Drittpaketen kollidieren. Alternative: Namespace `nova/` mit gleichen Unterpaketen. | **zu entscheiden** (Empfehlung: `nova/`-Namespace vor Phase 1) |
| 002 | Inferenz über externe Runtime per HTTP statt In-Process | angenommen |
| 003 | Asynchrone Verträge | angenommen |
| 004 | SQLite als einziger Persistenz-Speicher in v1 | angenommen |
| 005 | Kein Agent-Framework als Kernabhängigkeit | angenommen |
| 006 | Vektorindex-Wahl (sqlite-vec vs. LanceDB) | offen bis Phase 6 |
