# NOVA – Trainingsumgebung einrichten und Hardware prüfen

Stand: 2026-10-10 · Ziel-PC: GTX 1080 Ti (11 GB), 32 GB RAM, i7 (Angabe des Nutzers)

> Status: Datensatz-Pipeline, Hardware-Probe und Trainingsplan sind **implementiert und
> getestet**. Ein Trainingslauf ist **noch nicht implementiert und nie ausgeführt** worden.
> Die Probe wurde nur mit simulierten PyTorch-Modulen getestet, nicht auf einer echten GPU.

Die Trainingsumgebung ist **getrennt** von der installierten NOVA-App: PyTorch, Transformers
usw. sind mehrere GB groß und gehören nicht in den Installer.

## 1. Umgebung anlegen (Windows, PowerShell)

```powershell
# Python 3.12 oder 3.13 installiert vorausgesetzt
py -3.13 -m venv $env:LOCALAPPDATA\NOVA\train-env
& $env:LOCALAPPDATA\NOVA\train-env\Scripts\Activate.ps1
# PyTorch mit CUDA 12.6 – die letzten offiziellen Builds mit Pascal-Kernels (laut PyTorch bis 2.14)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft trl accelerate bitsandbytes
```

Nicht `cu128` oder neuer verwenden: Diese Builds enthalten keine Kernels für die GTX 1080 Ti
und scheitern erst zur Laufzeit. Wenn native Windows-Pakete (v. a. bitsandbytes) Probleme
machen, ist WSL2 mit Ubuntu die Alternative (gleiche Befehle, Linux-Pfade).

## 2. Hardware-Probe

Im NOVA-Quellordner (oder mit `PYTHONPATH` darauf):

```powershell
python -m training.probe > probe.json
```

`probe.json` enthält GPU, Compute Capability, VRAM, PyTorch-/CUDA-Version, die gefundenen
Pakete, den empfohlenen Compute-Typ und **Probleme** (`"ready": false` → erst beheben).
Für die 1080 Ti ist zu erwarten: Compute Capability 6.1, `compute_dtype: float32`, Hinweis
„Pascal GPU … slow“.

## 3. Datensatz bauen (in NOVA)

```powershell
nova dataset build nova-knowledge --from-knowledge [--manual eigene-beispiele.jsonl]
nova dataset verify nova-knowledge@0.1.0
```

* Nur **belegte** Research-Erkenntnisse; strittige, unbestätigte, veraltete und solche aus
  Quellen mit Trainingsverbot (Brave Search) werden ausgeschlossen. Die Gründe stehen im Manifest.
* Manuelle Beispiele: je Zeile `{"question": "...", "answer": "..."}`.
* Versionen sind unveränderlich; `verify` erkennt nachträgliche Änderungen.

## 4. Trainingsplan (Dry-Run)

`train.toml`:

```toml
name = "nova-qwen3.5-4b"
base_model = "Qwen/Qwen3.5-4B-Instruct"   # vor Nutzung Model Card + Lizenz prüfen
base_revision = "<exakter Commit-Hash des Modells>"
base_license = "apache-2.0"
dataset = "nova-knowledge@0.1.0"
max_seq_len = 1024
batch_size = 1
grad_accum = 16
```

```powershell
nova train plan --config train.toml --hardware probe.json
```

Der Plan prüft: feste Modell-Revision, unveränderter Datensatz mit Testsplit, geschätzter
VRAM-Bedarf gegen die GPU, bitsandbytes für QLoRA. Er trainiert **nichts**.

## 5. Was als Nächstes kommt

1. Trainings-Runner (TRL/PEFT) mit Checkpoint-Fortsetzung, Ressourcenlimit und Lineage –
   zuerst als Probelauf mit einem 0.8B-Modell und wenigen Schritten auf deinem PC.
2. Evaluation Basis vs. Checkpoint auf dem festen Testsplit + Regressionssuites.
3. Freigabe-Gate (`docs/model-development.md` §6) und GGUF-Export.
