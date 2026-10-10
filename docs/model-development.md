# NOVA – Modellentwicklung (eigenes, versioniertes NOVA-Modell)

Stand: 2026-10-09 · Status: **Strategie** – es wurde noch kein Modell heruntergeladen, kein
Training ausgeführt und kein NOVA-Checkpoint erzeugt.

## 0. Begriffe – was „NOVA-Modell“ heißen darf

| Bezeichnung | Bedeutung |
|---|---|
| **Basismodell** | unverändertes Open-Weight-Modell eines Dritten (z. B. „Qwen3.5-9B-Instruct“). Wird immer unter seinem Originalnamen geführt. |
| **NOVA-System** | Basismodell + NOVA-Router, Agenten, Tools, Memory, Retrieval, Verifikation. Das ist NOVAs eigene Leistung – aber **kein** eigenes Modell. |
| **NOVA-Checkpoint `nova-<basis>-<version>`** | Basismodell + von NOVA trainierte Gewichtsänderung (LoRA-Adapter oder gemergte Gewichte) aus **NOVAs eigenem, versioniertem Datensatz**, mit Eval-Report und Freigabe. Nur das darf „NOVA-Modell“ heißen. |

Ein unverändertes Basismodell wird nie als NOVA-Modell bezeichnet – auch nicht in UI, Model
Card oder Release-Notes. Jeder Checkpoint nennt Basismodell, Basis-Lizenz und Datensatzversion.

## 1. Die acht Stufen

| # | Stufe | Inhalt | Ergebnis / Artefakt | Status |
|---|---|---|---|---|
| 1 | Basismodell | Auswahl nach Lizenz, Hardware, eigener Eval (§2) | Eintrag in `models.toml` + Eval-Baseline | ⬜ (Hardware unbekannt) |
| 2 | NOVA-Systemarchitektur | Router, Agent, Tools, Memory, Verifikation | Code (`router/`, `agents/`, …) | ✅ vorhanden |
| 3 | Eigener Datensatz | aus geprüften Research-Erkenntnissen, manuell kuratierten Beispielen (Agent-Traces ⬜) | `datasets/<name>/<version>/` mit Manifest | ✅ Pipeline (`nova dataset`) – noch kein echter Datensatz |
| 4 | SFT | LoRA/QLoRA auf dem Basismodell | Adapter + Trainingslog + Config-Hash | ⬜ |
| 5 | Präferenzoptimierung (optional) | DPO/ORPO auf geprüften Paaren (gewählt/abgelehnt) | Adapter v+1 | ⬜ erst wenn SFT stabil |
| 6 | Unabhängige Evaluation | feste Testsets, deterministische Checks, Basis vs. Checkpoint, Regressionen | Eval-Report (JSON + MD) | 🟡 Evaluationsrahmen vorhanden (`evaluation/`), Modell-Vergleich fehlt |
| 7 | Versionierte Checkpoints | `checkpoints/nova-<basis>-<semver>/` mit Model Card, Hashes, Lineage | Registry-Eintrag | ⬜ |
| 8 | Sichere Auslieferung | nur freigegebene Checkpoints, GGUF-Export, SHA-256, Lizenzhinweis, Rollback | Download über Modellkatalog (bestehende Prüfungen) | 🟡 Download-/Prüfsummenmechanik vorhanden |

**Nicht** geplant: ein Sprachmodell von Grund auf trainieren (Kosten, Daten, Hardware stehen in
keinem Verhältnis zum Nutzen für einen persönlichen Assistenten).

## 2. Basismodell – Auswahlkriterien und Kandidaten

Kriterien (hart): Lizenz erlaubt Modifikation, Weitergabe und kommerzielle Nutzung ohne
Namens-/Nutzerzahl-Auflagen · Gewichte in safetensors (Training) und GGUF (Inferenz) verfügbar ·
Fine-Tuning von gängigen Werkzeugen unterstützt · passt auf die Zielhardware (§3).

Kandidaten (Recherche 2026-10-09, **vor Auswahl gegen Model Card und LICENSE-Datei prüfen**):

| Familie | Größen (dicht) | Lizenz (laut Recherche) | Bemerkung |
|---|---|---|---|
| Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B (+ MoE 35B-A3B, 122B-A10B) | Apache-2.0 | Thinking/Non-Thinking, langer Kontext, multimodal; gute Fine-Tuning-Unterstützung |
| Gemma 4 | E2B/E4B, 26B MoE, 31B (Angaben uneinheitlich) | Apache-2.0 (seit 03/2026, vorher eigene Terms) | Lizenzwechsel prüfen; Größen bestätigen |
| Phi-4-mini / Phi-4 | 3.8B / 14B | MIT | stark bei Reasoning, schwächer mehrsprachig |
| Llama 3.x | 1B–70B | Llama Community License | **nicht bevorzugt**: Namensauflage für Ableitungen, Nutzerzahlgrenze |

Vorläufige Empfehlung (abhängig von §3): **Qwen3.5-4B-Instruct** als erstes Trainingsziel
(klein genug für QLoRA auf Consumer-GPUs, schnell iterierbar), **Qwen3.5-9B** als
Produktivkandidat bei ≥ 12 GB VRAM. Endgültig entscheidet NOVAs eigene Evaluation auf der
Zielhardware (`docs/model-strategy.md` §7) – nicht diese Tabelle.

Technische Grenzen kleiner Basismodelle, die Fine-Tuning **nicht** behebt: begrenztes
Weltwissen (→ Retrieval statt Training), Halluzinationen bei seltenen Fakten (→ Quellenpflicht),
Kontextqualität bei sehr langen Eingaben, begrenzte Mehrschritt-Planung.

## 3. Hardwareabhängige Planung

Zielhardware des Nutzers: siehe §3.1. Die Entwicklungsumgebung (Cloud-Container:
4 CPU-Kerne, 15 GB RAM, keine GPU) ist **nicht** die Zielmaschine und für Training ungeeignet.

| Stufe | Hardware | Inferenz (GGUF Q4) | Fine-Tuning | Einordnung |
|---|---|---|---|---|
| A | keine GPU, 16 GB RAM | 4B flüssig, 9B langsam (CPU) | **nicht sinnvoll** lokal (nur Datensatz-/Eval-Pipeline) | Training extern (gemietete GPU) |
| B | 8 GB VRAM | 9B | QLoRA 4B (Kontext ≤ 2k), 8–9B nur knapp | erstes eigenes SFT möglich |
| C | 12–16 GB VRAM | 9B–14B | QLoRA 8–9B komfortabel, 14B knapp | empfohlenes Minimum für Produktiv-Fine-Tuning |
| D | 24 GB VRAM | 27B Q4 | QLoRA 14B; LoRA-16bit 8B | stärkere Checkpoints |
| E | ≥ 48 GB / mehrere GPUs | 27B+ / MoE | LoRA größerer Modelle, DPO | „später, leistungsfähigere Maschine“ |

Richtwerte QLoRA (Unsloth-Herstellerangaben, absolute Untergrenzen bei kurzer Sequenz,
Batch 1): 3B ≈ 3,5 GB, 8B ≈ 6 GB, 14B ≈ 8,5 GB VRAM; LoRA 16-bit 8B ≈ 22 GB. Praxis: +30–50 %
Puffer einplanen. Speicher: Basismodell safetensors (4B ≈ 8 GB, 9B ≈ 18 GB) + Checkpoints +
GGUF-Export → ≥ 50 GB frei.

Quellen: [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements),
[Qwen3.5-Übersicht (The Batch)](https://www.deeplearning.ai/the-batch/alibabas-latest-flagship-models-are-open-weights-moe-performers-in-sizes-from-less-than-1b-parameters),
[Gemma 4 unter Apache 2.0 (Google Open Source Blog)](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html).
Sekundärquellen dienen der Vorauswahl, nicht als Beleg.

### 3.1 Zielhardware des Nutzers (Angabe vom 2026-10-10)

**32 GB RAM · Intel i7 · NVIDIA GeForce GTX 1080 Ti (11 GB VRAM, Pascal, Compute Capability 6.1)
· > 500 GB SSD.** Die Angaben stammen vom Nutzer und sind noch nicht durch einen Hardware-Check
auf dem Gerät bestätigt (NOVA erkennt RAM/VRAM beim Start: Status → System).

| Bereich | Einschätzung | Begründung / Risiko |
|---|---|---|
| Inferenz | **gut**: 9B-Modell in Q4 (≈ 6,6 GB) vollständig in VRAM, 4B mit viel Kontext | llama.cpp unterstützt Pascal; Richtwert für 8B Q4 auf 1080 Ti: ≈ 60 Token/s (Schätzung einer Vergleichsseite, **nicht gemessen**) |
| Wichtig für Inferenz | llama.cpp-Build mit **CUDA 12.x** verwenden | CUDA 13 unterstützt nur noch GPUs ab Turing (sm_75) |
| Fine-Tuning | **möglich, aber unbewiesen und langsam** | Pascal hat keine Tensor Cores und kein bf16, fp16 ist auf GP102 stark gedrosselt → Berechnung in fp32 |
| Software für Training | PyTorch **≤ 2.14 mit CUDA 12.6** (letzte Version mit Pascal-Binaries); cu128-Builds haben keine Pascal-Kernel | ab PyTorch 2.15 nur noch Eigenbau |
| Unsloth | offiziell Minimum Compute Capability 7.0, „GTX 1070, 1080 works, but is slow“ | Unterstützung kann jederzeit entfallen |
| bitsandbytes 4-bit | laut Drittquellen ab Compute Capability 6.0 | auf 6.1 **nicht** Ende-zu-Ende bestätigt |

**Festlegung für das erste Fine-Tuning auf diesem PC:** Qwen3.5-4B (Apache-2.0) mit QLoRA,
Sequenzlänge ≤ 1024, Batch 1 + Gradient Accumulation, fp32-Compute. Vor jedem echten Training
läuft ein **Hardware-Probelauf** (0.8B-Modell, wenige Schritte), der Software-Stack, VRAM-Bedarf
und Geschwindigkeit misst – erst danach wird entschieden, ob 4B lokal trainiert wird.
Rückfalloption: dieselbe, reproduzierbare Pipeline auf einer gemieteten GPU (Entscheidung des
Nutzers; Daten verlassen dann den PC).

Quellen: [PyTorch: Pascal aus CUDA-12.8-Builds entfernt](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128),
[PyTorch: CUDA-12.6-Wheels enden mit 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432),
[Erfahrungsbericht GTX 1080 Ti mit cu128](https://github.com/jhj0517/Whisper-WebUI/issues/653),
[Unsloth Requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements),
[Schätzwert 8B auf 1080 Ti](https://willitrunai.com/can-run/llama-3.1-8b-on-gtx-1080-ti-11gb).

## 4. Werkzeugkette (geplant)

| Aufgabe | Werkzeug | Lizenz | Begründung |
|---|---|---|---|
| SFT / DPO | Hugging Face TRL + PEFT (LoRA/QLoRA) | Apache-2.0 | Referenzimplementierung, reproduzierbar |
| Speicher-/Tempo-Optimierung | Unsloth (optional) | Apache-2.0 (Kern) | weniger VRAM auf Consumer-GPUs |
| Export | llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` | MIT | passt zur bestehenden Inferenz-Runtime |
| Eval | NOVA `evaluation/` + feste Testsets | – | unabhängig vom trainierten Modell |

Training läuft als **separater Prozess** (`nova train …`), nie im Chat-Pfad, mit
GPU-/Zeitbudget, Checkpoint-Wiederaufnahme und vollständigem Log. Unter Windows wird
Training über WSL2/CUDA oder eine Linux-Maschine erwartet (TRL/bitsandbytes unter nativem
Windows eingeschränkt) – vor Umsetzung zu verifizieren.

## 5. Versionierung und Lineage

```
checkpoints/nova-qwen3.5-4b-0.1.0/
  adapter/                 LoRA-Gewichte (safetensors)
  model.gguf               optional, gemergt + quantisiert
  model-card.md            Basis, Lizenz, Datensatz, Hyperparameter, Grenzen
  lineage.json             base_model + Revision-Hash, dataset@version + Manifest-Hash,
                           config-Hash, Code-Commit, Zufallsseed, Hardware
  eval/report.json|md      Basis vs. Checkpoint, alle Suites, Regressionen
  approval.json            wer/wann/auf Basis welches Reports freigegeben
  SHA256SUMS
```

SemVer: Major = anderes Basismodell, Minor = neuer Datensatz/Trainingsziel, Patch = Fix
ohne neue Fähigkeiten.

## 6. Freigabekriterien (Gate)

Ein Checkpoint wird nur freigegeben, wenn **alle** gelten:
1. Auf jeder Pflicht-Suite mindestens so gut wie die aktuell ausgelieferte Version
   (Toleranz je Suite festgelegt, statistisch über mehrere Seeds/Wiederholungen).
2. Auf der Zielfähigkeit ein signifikanter Gewinn gegenüber dem Basismodell.
3. Keine Regression in Sicherheits-/Verweigerungs- und Tool-Calling-Tests.
4. Testdaten nachweislich nicht im Trainingsdatensatz (Leak-Check, `continuous-learning.md` §4).
5. Bewertung durch deterministische Checks und Referenzantworten; LLM-Bewertung höchstens
   ergänzend und **nie durch das trainierte Modell selbst**.
6. Menschliche Freigabe (`approval.json`).

## 7. Voraussetzungen für das erste eigene Fine-Tuning

1. Zielhardware bekannt ✅ (§3.1) → Basismodell Qwen3.5-4B (QLoRA); Hardware-Probelauf auf dem PC steht aus.
2. Basismodell lokal eingebunden und Baseline mit NOVAs Eval gemessen.
3. Datensatz v0.1: ≥ einige hundert geprüfte Beispiele für **eine** klar definierte
   Zielfähigkeit (z. B. NOVA-Tool-Calling-Format oder Quellenzitate), mit getrenntem Testset.
4. Trainingspipeline: Datensätze, Probe und Plan ✅; Runner ⬜ (`docs/training-setup.md`).
5. Freigabe-Gate automatisiert.
