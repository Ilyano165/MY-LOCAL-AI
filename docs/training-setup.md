# NOVA – Training: Umgebung, Ablauf, technische Entscheidungen

Stand: 2026-10-12 · Ziel-PC: GTX 1080 Ti (11 GB), 32 GB RAM, i7 (Angabe des Nutzers)

> **Status.** Diese Teile sind implementiert und in der Entwicklungsumgebung **echt**
> durchlaufen (CPU, winziges Zufallsmodell):
>
> Datensatz → LoRA-Training → Evaluation mit Gate → GGUF-Export (llama.cpp-Konverter) →
> Freigabe → Aktivierung über `--lora` → Chat.
>
> **Nicht** getestet:
>
> * Training auf einer echten GPU, auch nicht auf der GTX 1080 Ti;
> * Qwen3.5 als Basismodell;
> * bitsandbytes unter Windows;
> * die automatische uv-Einrichtung unter Windows;
> * ob ein echter `llama-server` den exportierten Adapter lädt.
>
> Details: [Testbericht](test-report.md).

Alles wird in der App bedient (Seitenmenü → **Training**). Die Trainingsumgebung ist von der
installierten App **getrennt**: PyTorch und die Bibliotheken sind mehrere GB groß und gehören
nicht in den Installer.

## 1. Umgebung (automatisch)

„Set up training environment“ führt diese Schritte aus. Fortschritt und Fehler stehen in
`%LOCALAPPDATA%\NOVA\logs\training-setup.log`.

1. **uv** aus dem offiziellen GitHub-Release `astral-sh/uv` laden; die SHA-256 kommt aus der
   GitHub-API und wird geprüft.
2. `uv python install 3.12`, dann `uv venv` nach `%LOCALAPPDATA%\NOVA\train-env\`.
3. **PyTorch** aus `https://download.pytorch.org/whl/cu126`. Ohne NVIDIA-GPU wird der CPU-Index
   genutzt.
4. transformers, peft, accelerate, huggingface_hub, safetensors, sentencepiece, protobuf,
   numpy sowie pyyaml, tqdm und requests für den GGUF-Konverter; mit GPU zusätzlich
   bitsandbytes.
5. Die Trainer-Skripte (`runner.py`, `evaluate.py`, `export.py`, `probe.py`) nach
   `%LOCALAPPDATA%\NOVA\trainer\` kopieren.
6. Die **Hardware-Probe** läuft und schreibt `train-env\probe.json`.

Eine vorhandene Umgebung lässt sich mit der Umgebungsvariable `NOVA_TRAIN_PYTHON=<python.exe>`
nutzen. WSL2 ist dafür nicht nötig.

### Warum CUDA 12.6

Die GTX 1080 Ti ist Pascal (Compute Capability 6.1). Die PyTorch-Builds für cu128 und neuer
enthalten keine sm_61-Kernels mehr; cu126 enthält sie laut PyTorch bis 2.14. Pascal hat kein
bf16 und nur sehr langsames fp16. Die Probe wählt deshalb **fp32** als Rechengenauigkeit; die
Gewichte liegen bei QLoRA trotzdem 4-bit (nf4) vor.

### Warum nicht TRL

TRL 1.15 (`SFTTrainer`) patcht immer einen *fused LM head*, der Triton braucht. Triton gibt es
unter Windows nicht, und Pascal-GPUs werden nicht unterstützt. In der Entwicklungsumgebung
brach das Training reproduzierbar ab („0 active drivers“). Der Runner nutzt deshalb
`transformers.Trainer` mit PEFT und maskiert den Verlust selbst: Gelernt wird nur auf der
Assistenten-Antwort, der Prompt-Präfix kommt aus dem Chat-Template. Beispiele, deren Template
nicht präfix-stabil ist, werden verworfen, nicht falsch maskiert.

## 2. Datensatz

„Build dataset from research knowledge“, oder per Kommandozeile:
`nova dataset build nova-knowledge --from-knowledge [--manual eigene.jsonl]`.

* Nur **belegte** Research-Erkenntnisse. Strittige, unbestätigte und veraltete Erkenntnisse
  sowie solche aus Quellen mit Trainingsverbot werden ausgeschlossen; die Gründe stehen im
  Manifest.
* Split nach Gruppen (gleiche Teilfrage nie in Train *und* Test), mit Leck-Prüfung.
* Versionen sind unveränderlich, Hashes stehen im Manifest. `nova dataset verify` prüft sie.

## 3. Training

| Einstellung | Wert |
|---|---|
| Basismodell | `training_base` aus `config/recommended-models.json`, an einen festen HF-Commit gebunden. Für den 9B-GGUF ist das `Qwen/Qwen3.5-9B`, also dieselbe Architektur, sonst passt der Adapter nicht. |
| Methode | QLoRA (nf4, double quant) auf der GPU, LoRA auf der CPU |
| Ziel-Layer | `all-linear` |
| Standardwerte | r 16, alpha 32, lr 2e-4, Batch 1 × Gradient Accumulation 16, max. 1024 Token, Cosine-Schedule |
| Robustheit | Checkpoints alle 50 Schritte (es wird automatisch fortgesetzt); Abbruch über eine `STOP`-Datei |
| Ergebnis | `status.json` (live in der UI), `training.log`, `adapter/`, `lineage.json` (Modell-Commit, Datensatz-Hash, Konfig-Hash, Versionen, Gerät) |

Während des GPU-Trainings wird `llama-server` gestoppt, damit genug VRAM frei ist. Danach
startet er automatisch wieder.

## 4. Evaluation und Gate

`evaluate.py` vergleicht **dasselbe** geladene Modell mit und ohne Adapter (greedy decoding):

* Token-F1 gegen die Referenzaussage des Testsplits und ob die Antwort eine Referenzquelle nennt;
* 10 feste Allgemeinfragen als Regressionsprüfung.

Gate (Standard): mindestens 10 Testbeispiele, F1-Gewinn ≥ 0,02 und höchstens 1
Allgemeinfrage schlechter als das Basismodell. Kein Modell bewertet sich selbst.

## 5. Freigabe, Export, Aktivierung

„Approve and activate“ erscheint nur bei erfülltem Gate. Vor der Aktivierung kommt eine
Bestätigung, und das trainierte Basismodell muss dem laufenden Modell entsprechen. Dann:

1. Quellcode von llama.cpp **im selben Tag** wie die installierte Runtime laden. Für das Zip
   veröffentlicht GitHub keine Prüfsumme; NOVA protokolliert den Hash.
2. `convert_lora_to_gguf.py` mit dem `gguf-py` desselben Tags. Das PyPI-Paket `gguf` passte in
   der Entwicklung nicht zu den aktuellen Konverter-Skripten.
3. `approval.json` (wer, wann, Gate, SHA-256 des Adapters) schreiben, dann `llama-server --lora`
   starten. Das Modell heißt jetzt `nova-<modell>-<job>`.

„Back to base model“ entfernt den Adapter wieder.

## 6. Bekannte Risiken

* **Qwen3.5-Architektur:**
  * Laufen hybride Attention-Schichten ohne Triton-Kernels unter Windows schnell genug?
  * Erfasst `all-linear` die richtigen Module?
  * Benennt `convert_lora_to_gguf.py` die Tensoren für Qwen3.5 korrekt?

  Alles drei ist ungetestet. Scheitert der 9B-Lauf, zuerst mit dem 4B-Modell probieren. Für
  4B muss im Setup auch das 4B-GGUF aktiv sein.
* **bitsandbytes unter Windows auf Pascal** ist nicht geprüft. Schlägt QLoRA fehl, steht das in
  `training.log`.
* **Dauer:** Auf der 1080 Ti mit fp32 sind für einige hundert Beispiele und 2 Epochen
  Stunden realistisch. Das ist nicht gemessen. Die ETA in der UI kommt aus den echten
  Schritten.
