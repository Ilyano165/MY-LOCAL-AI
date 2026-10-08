# router/

Intelligenter Model Router: wählt je Aufgabe ein **verfügbares** und passendes Modell –
mit nachvollziehbarer Begründung.

| Datei | Inhalt |
|---|---|
| `base.py` | `RoutingCategory` (FAST, GENERAL, CODING, REASONING, VISION, LONG_CONTEXT), `Complexity`, `Latency`, `RoutingRequest`, `TaskClassification`, `RoutingDecision`, Schnittstellen `TaskClassifier` / `ModelRouter` |
| `classifier.py` | `RuleBasedClassifier` (deterministisch), `LLMClassifier` (Modell klassifiziert, Fakten bleiben regelbasiert, Rückfall auf Regeln), `HybridClassifier` (LLM nur bei unsicheren Fällen) |
| `availability.py` | `ProviderAvailability` (Runtime-Health + Modellliste, TTL-Cache, Ausfallmeldung), `StaticAvailability` |
| `resources.py` | `ResourceBudget`, `detect_resources()` (RAM über /proc/meminfo bzw. sysconf, VRAM über nvidia-smi) |
| `rule_router.py` | `RuleBasedRouter` |
| `routing_log.py` | `format_decision`, `MemoryRoutingLog`, `JsonlRoutingLog` (Entscheidungen, Fehlschläge, Ergebnisse) |

## Eingaben → Entscheidung

Eingaben: Aufgabe, Gesprächskontext (Tokens), verfügbare Modelle (Registry + Verfügbarkeit),
benötigte Tools, Komplexitätsschätzung (optional, sonst geschätzt), Latenzanforderung,
VRAM/RAM, Bildbedarf.

1. **Klassifikation** – Kategorie aus Fakten (Bild → VISION, sehr großer Kontext → LONG_CONTEXT
   mit inhaltlicher Sekundärkategorie) und Inhalt (Code-, Reasoning-Signale, Kürze);
   Komplexität aus Umfang, Mehrstufigkeit, Projektebene, Debug/Refactor/Fix, formalem
   Schließen, Tool-Bedarf, Kontextgröße.
2. **Harte Filter** (jeder Ausschluss wird begründet): nicht verfügbar · passt nicht in
   VRAM/RAM · Kontextfenster zu klein · kein Tool-Calling · keine Vision-/Coding-/Reasoning-
   Fähigkeit für die jeweilige Kategorie. **Ein nicht verfügbares Modell ist nie wählbar** – auch
   nicht als Fallback.
3. **Rangfolge**: „fähig genug“ für die Komplexität (LOW → basic, MEDIUM → good, HIGH → strong),
   dann Geschwindigkeit (nur im RAM lauffähig = langsamer), dann volle Fähigkeit, dann
   Speicherbedarf, dann Name. `batch` → Qualität zuerst; `realtime` → „basic“ genügt.

Ohne passendes Modell: `NoModelAvailableError` mit allen Ausschlussgründen – kein stilles
Ausweichen auf ein ungeeignetes Modell.

## Routing Logs

```
TASK:
"Debug this Python project"
ROUTING:
category = CODING
complexity = HIGH
selected_model = coder
reason = "Erfordert Codeverständnis und mehrstufiges Vorgehen (z. B. Debugging über mehrere Dateien). Gewählt: stärkste Coding-Fähigkeit (strong) unter 3 geeigneten Modellen"
classifier = rules (confidence 0.85)
signals = code: debug, python; tiefe Arbeit; projektebene + tiefe Arbeit
fallbacks = thinker, small-fast
rejected = seer: keine Coding-Fähigkeit
```

`JsonlRoutingLog` speichert zusätzlich das **Ergebnis** jeder gerouteten Aufgabe (Urteil der
Verification Engine, Quality Score), das der Agent beim Abschluss meldet. `joined()` liefert
Entscheidung + Ergebnis – Grundlage, um Regeln auszuwerten und später einen Router zu lernen
(ein Router sieht nur das Ergebnis des Modells, das er gewählt hat).

## Integration

`InferenceEngine(..., router=RuleBasedRouter(...))`: Ohne explizites Modell entscheidet der
Router; Fallbacks stammen ebenfalls vom Router; Ausfälle werden gemeldet, damit das Modell bis
zur nächsten Prüfung gemieden wird. Ohne Router bleibt das bisherige Verhalten.

Trockenlauf: `python -m scripts.route --config ~/.nova/models.toml "Aufgabe"`

## Evaluation

`python -m scripts.evaluate_routing` – Datensatz `evaluation/datasets/routing_tasks.jsonl`,
Baseline und Verbesserungsvorschläge in `evaluation/reports/routing_baseline.md`.

## Grenzen

* Regeln sind auf Deutsch/Englisch zugeschnitten; Token-Zählung ist eine Schätzung.
* Speichererkennung ist eine Momentaufnahme; bereits geladene Modelle belegen Speicher.
* Ohne Benchmark-Profil stammen Fähigkeits- und Geschwindigkeitsangaben aus der Konfiguration
  (Annahmen). Mit Profil (`scripts/benchmark_models.py`, `evaluation.attach_profiles`) nutzt der
  Router gemessene Werte; jede Begründung kennzeichnet „gemessen“ bzw. „konfiguriert –
  UNMEASURED“, das Routing-Log enthält `data = MEASURED|PARTIAL|UNMEASURED`.
