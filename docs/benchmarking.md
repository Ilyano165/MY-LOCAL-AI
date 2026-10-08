# NOVA – Reales Model-Benchmarking

Status: umgesetzt (Suite 1.0) · Stand: 2026-10-08

## 1. Ziel und Grundsatz

Bis zu diesem Schritt stammten alle Modellangaben (Fähigkeit, Geschwindigkeit, Speicher) aus der
Konfiguration – also aus **Annahmen**. Das Benchmarking ersetzt sie durch **Messwerte aus echten
Läufen auf der eigenen Hardware**. Drei Regeln gelten ausnahmslos:

1. **Kein Wert gilt als gemessen, der nicht aus einem echten Lauf stammt.** Jeder Wert trägt
   einen Status und eine Methode. Ein Messwert ohne Methode ist im Code unzulässig
   (`Measurement.__post_init__`).
2. **Modellfähigkeit und Hardware-Leistung sind getrennt.** Ein langsames Modell ist nicht
   automatisch ein schlechtes Modell. Fähigkeiten gelten hardwareunabhängig, Leistungswerte nur
   für die Hardware, auf der gemessen wurde.
3. **Ohne Profil ist ein Modell ausdrücklich `UNMEASURED`.** Der Router nennt das in jeder
   Begründung („konfiguriert – UNMEASURED“) und im Routing-Log (`data = UNMEASURED`).

## 2. Messstatus

| Status | Bedeutung | Beispiel |
|---|---|---|
| `MEASURED` | von NOVA im echten Lauf gemessen | Wanduhr, Sampling, Task-Scoring |
| `REPORTED` | von der Runtime im echten Lauf gemeldet | llama.cpp `timings`, `/props n_ctx`, Ollama `/api/ps` |
| `DETECTED` | per Systemabfrage erkannt | CPU, GPU, GGUF-Quantisierung |
| `CONFIGURED` | aus der Konfiguration – **nie** ein Messwert | Vergleichswerte im Profil |
| `NOT_SUPPORTED` | nachweislich nicht unterstützt | Runtime lehnt Bilder/Tools ab |
| `NOT_MEASURABLE` | in dieser Konstellation nicht messbar, mit Begründung | Ladezeit bei vorgeladenem Modell |
| `FAILED` | Messung versucht, fehlgeschlagen | Timeout, Runtime-Fehler |
| `UNMEASURED` | nie gemessen | kein Profil |

Nur `MEASURED`, `REPORTED` und `DETECTED` fließen in den Router ein.

## 3. Modellfähigkeit: Standardaufgaben

Alle Aufgaben laufen mit `temperature=0` und `seed=42`, haben feste Prompts und werden **ohne
Modellurteil** bewertet. Score = Anteil erfüllter Prüfkriterien.

| ID | Bereich | Aufgabe | Unabhängige Prüfung |
|---|---|---|---|
| `chat` | general | Hauptstadt Frankreichs, ein Satz | Regex „Paris“, Satzanzahl, Länge |
| `short_analysis` | reasoning | Quartalsumsätze: bestes Quartal, Veränderung | Ergebniszeile: Q4, +100 % |
| `complex_analysis` | reasoning | Logikrätsel (4 Personen × 4 Tiere, 5 Hinweise) | Eindeutigkeit per Brute Force im Test; 4 Zuordnungen |
| `coding` | coding | `is_palindrome` schreiben | 9 versteckte Unit-Tests im Subprozess |
| `debugging` | coding | `median` mit vertauschten Zweigen reparieren + `ValueError` | 7 versteckte Unit-Tests |
| `agent_multistep` | agentic | Tool-Schleife: CSV finden, Umsatz summieren, `result.txt` schreiben | Dateizustand: Wert 133.49, CSV gelesen, fertig in ≤ 8 Schritten |
| `tool_calling` | tool_calling | `get_weather(city, unit)` aufrufen, Ergebnis verwenden | Name, Argumente, Nutzung des Tool-Ergebnisses |
| `long_context` | long_context | 3 Codes bei 10/50/90 % im Fülltext (60 % des Fensters, max. 16k) | exakte Codes; übersprungen unter 4096 Tokens |
| `vision` (Probe) | vision | generiertes PNG: linke/rechte Farbe | rot/blau; `NOT_SUPPORTED`, wenn die Runtime Bilder ablehnt |

**Suite-Version** = `1.0+<Hash über alle Prompts und Prüfregeln>`. Ändert sich eine Aufgabe,
ändert sich die Version; ältere Profile gelten dann als `STALE` und werden nicht angewendet.

**Abbildung auf Stufen** (für den Router): Score ≥ 0.85 → strong, ≥ 0.6 → good, ≥ 0.3 → basic,
sonst none. Tool-Calling gilt ab Score 0.5 als vorhanden.

**Sicherheit:** Coding-Aufgaben führen vom Modell erzeugten Code aus – in einem Subprozess ohne
Shell, mit Zeitlimit, `python -I`, temporärem Verzeichnis und einer Umgebung ohne Secrets
(`tools.process.run_process`). Die Ergebnis-Marke ist pro Lauf zufällig, damit Modellcode kein
Ergebnis vortäuschen kann. `--no-exec` überspringt diese Aufgaben.

## 4. Hardware-Leistung

| Wert | Methode |
|---|---|
| `load_time` | MEASURED, wenn NOVA den Ladevorgang beobachtet: `--server-cmd` (Prozessstart bis `/health` bereit) oder Ollama-Lade-API (kalt; vorgeladen nur mit `--measure-load`). Bei externem llama-server: NOT_MEASURABLE |
| `first_token_latency` | Wanduhr für eine 1-Token-Antwort auf einen neuen Prompt (ohne Streaming, inkl. HTTP) |
| `tokens_per_second` | **Differenzmethode** (N−1)/(t_N − t_1) bei gleichem Prompt, Median über `--runs` Läufe. Runtime-unabhängig, ohne Streaming. Zuerst N, dann 1 → Prompt-Cache begünstigt t_1, Schätzung eher konservativ |
| `tokens_per_second_runtime` | REPORTED: llama.cpp `timings.predicted_per_second` |
| `prompt_tokens_per_second` | REPORTED (llama.cpp) oder MEASURED: langer neuer Prompt, 1 Ausgabetoken |
| `peak_vram` / `peak_ram` | Sampling (Standard 0.2 s) während aller Läufe. Mit Server-PID pro Prozess (`nvidia-smi --query-compute-apps`, `/proc/<pid>/status` VmRSS), sonst systemweit (gekennzeichnet) |
| `memory_footprint` | nur belastbar: Ollama `size` (REPORTED), Prozess-Spitze VRAM + RSS, oder System-Delta ab Ausgangswert **vor** dem Serverstart (`--server-cmd`). Sonst NOT_MEASURABLE |

Die Server-PID wird bei llama.cpp automatisch über den Port ermittelt (`/proc/net/tcp` →
Socket-Inode → `/proc/<pid>/fd`). Bei Ollama läuft das Modell in einem Kindprozess, daher dort
systemweit plus von Ollama gemeldete Werte.

## 5. Erkannte Modell- und Hardwaredaten

* Hardware (`evaluation/hardware.py`): CPU-Modell, Kerne, SIMD-Flags (`/proc/cpuinfo`), RAM
  (`/proc/meminfo`, macOS `sysctl`), NVIDIA (`nvidia-smi`), AMD (`rocm-smi --json`), Apple
  Silicon (Metal, Unified Memory). Der **Fingerprint** enthält nur stabile Merkmale
  (CPU, Kerne, RAM gerundet, GPU-Modelle und VRAM), keine freien Speicherwerte.
* Modell: Kontextfenster (`/props n_ctx` bzw. `/api/ps context_length`), Parameter, Dateigröße,
  Trainingskontext (`/v1/models` meta), Quantisierung und Architektur aus dem **GGUF-Header**
  (`evaluation/gguf.py`, nur Metadaten, keine Gewichte). Abweichungen zur Konfiguration werden
  vermerkt.

## 6. Profile und Router

Profil: `~/.nova/benchmarks/<modell>.json` (überschreibbar mit `--profiles` oder
`NOVA_BENCHMARK_DIR`), zusätzlich jede Messung in `history/`. Pflichtfelder: `model`,
`runtime`, `hardware`, `load_time`, `tokens_per_second`, `peak_vram`, `peak_ram`,
`capabilities`, `benchmark_version`, jeweils mit Status und Methode.

Anbindung: `models/measured.py` definiert die neutrale Schnittstelle (`MeasurementSource`,
`MeasuredOverrides`, `DataStatus`); `models` importiert `evaluation` nicht. `ProfileStore`
implementiert sie. `ModelRegistry.effective()`/`list_effective()` ersetzen konfigurierte durch
gemessene Werte; die Konfiguration bleibt in `extra["configured"]` sichtbar.

| Profilzustand | Router-Daten | Status |
|---|---|---|
| kein Profil | Konfiguration | `UNMEASURED` |
| gleiche Hardware + Suite | Fähigkeiten **und** Leistung gemessen | `MEASURED` |
| andere Hardware | nur Fähigkeiten (hardwareunabhängig) | `PARTIAL` |
| andere Suite-Version | nichts angewendet | `STALE` (Router: UNMEASURED) |

Der `RuleBasedRouter` nutzt die effektiven Daten für harte Filter (z. B. real gemeldetes
Kontextfenster, gemessenes Tool-Calling) und Rangfolge, und kennzeichnet in der Begründung,
ob die verwendete Fähigkeit gemessen ist.

## 7. Bedienung

```bash
python -m scripts.benchmark_hardware                       # Hardware + Fingerprint
python -m scripts.benchmark_hardware --sample 10 --pid 1234 # Speicherspitzen eines Prozesses
python -m scripts.benchmark_models --config ~/.nova/models.toml            # alle Modelle
python -m scripts.benchmark_models --config … --model local-general --tasks chat,coding
python -m scripts.benchmark_models --config … --model local-general \
    --server-cmd "llama-server -m ~/.nova/models/x.gguf --port 8080 --jinja"   # misst Ladezeit
python -m scripts.benchmark_models --config … --list       # Messstatus aller Modelle
python -m scripts.route --config … "Aufgabe"               # Routing mit Profilen
```

Echte Integration: `NOVA_IT_CONFIG=… pytest -m integration` (enthält einen kurzen
Benchmark-Lauf gegen die laufende Runtime).

## 8. Grenzen (ehrlich)

* **Kleine Stichprobe:** 1–2 Aufgaben je Bereich. Die Stufen sind grob; für Modellvergleiche
  zählen die Einzelscores und Prüfkriterien im Profil. Ausbau: mehr Aufgaben je Bereich.
* **Reproduzierbarkeit** gilt für gleiches Modell, gleiche Runtime-Version und gleiche
  Server-Optionen. Andere Builds/Batchgrößen können bei Greedy-Decoding abweichen.
* **Kein Streaming:** Die Dekodierrate wird per Differenzmethode bestimmt; echte TTFT-Messung
  folgt mit Streaming-Unterstützung im Provider.
* **Sampling** verpasst Spitzen kürzer als das Intervall; RSS zählt bei `mmap` nur berührte
  Seiten; systemweite Werte enthalten Fremdprozesse (im Profil gekennzeichnet).
* **Quantisierung** beeinflusst die Fähigkeit. Ein Profil gilt für den konfigurierten
  Registry-Eintrag; eine andere Quantisierung braucht einen eigenen Eintrag und Lauf.
* Speicher-Sampling pro Prozess: Linux (`/proc`) und NVIDIA. AMD/Apple: systemweit.
* Nicht verifiziert gegen eine echte Runtime in dieser Umgebung (keine Modelle heruntergeladen):
  End-to-End wurde gegen einen echten HTTP-Prozess mit simulierter llama.cpp-API getestet
  (PID-Erkennung, Prozess-Sampling, `--server-cmd`).
