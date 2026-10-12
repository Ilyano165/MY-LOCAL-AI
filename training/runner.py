"""Trainings-Runner (LoRA/QLoRA-SFT) – läuft in der separaten Trainingsumgebung.

    python runner.py job.json

``job.json`` (von NOVA erzeugt):
    job_id, base_model, base_revision, method (qlora|lora), dataset_dir, output_dir,
    compute_dtype (float32|float16|bfloat16), hyper{lora_r, lora_alpha, lora_dropout,
    learning_rate, epochs, max_seq_len, batch_size, grad_accum, seed, save_steps, max_steps?},
    local_model_dir? (Tests/Offline), dataset_manifest_sha256, config_hash

Schreibt laufend ``status.json`` (atomar) und ``training.log``. Abbruch über die Datei ``STOP``
im Ausgabeordner; nach einem Absturz wird am letzten Checkpoint fortgesetzt.
Ergebnis: ``adapter/`` (PEFT) und ``lineage.json``.

Trainiert mit ``transformers.Trainer`` + PEFT, Verlust nur auf der Assistenten-Antwort –
bewusst ohne TRL: dessen fused LM-Head braucht Triton, das es unter Windows nicht gibt und das
Pascal-GPUs (z. B. GTX 1080 Ti) nicht unterstützen. Importiert nichts aus NOVA (eigenständig).
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_dir / f"{split}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def only_supported(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Nur Parameter übergeben, die die installierte Bibliotheksversion kennt."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in values.items() if k in params}


def _ids(result: Any) -> list[int]:
    """``apply_chat_template(tokenize=True)`` liefert je nach Version eine Liste oder ein Dict."""
    if isinstance(result, dict) or hasattr(result, "keys"):
        result = result["input_ids"]
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(x) for x in result]


def features(
    tokenizer: Any, messages: list[dict[str, str]], max_len: int
) -> dict[str, list[int]] | None:
    """Tokenisiert ein Chat-Beispiel; Verlust nur auf der Assistenten-Antwort (Prompt maskiert)."""
    prompt = _ids(
        tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    )
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if full[: len(prompt)] != prompt or len(full) <= len(prompt):
        return None  # Template nicht präfix-stabil → Beispiel nicht sicher maskierbar
    full = full[:max_len]
    labels = [-100] * min(len(prompt), len(full)) + full[len(prompt) :]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int) -> Any:
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

        return {
            "input_ids": pad("input_ids", self.pad_id),
            "labels": pad("labels", -100),
            "attention_mask": pad("attention_mask", 0),
        }


def model_source(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if job.get("local_model_dir"):
        return str(job["local_model_dir"]), {}
    return str(job["base_model"]), {"revision": job["base_revision"]}


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    started = time.time()
    status: dict[str, Any] = {
        "status": "loading",
        "job_id": job["job_id"],
        "step": 0,
        "max_steps": None,
        "loss": None,
        "started_at": started,
    }
    write_json(status_path, status)
    log = (out / "training.log").open("a", encoding="utf-8")

    def emit(**fields: Any) -> None:
        status.update(fields)
        status["updated_at"] = time.time()
        write_json(status_path, status)
        log.write(
            json.dumps({k: status.get(k) for k in ("status", "step", "loss", "epoch")}) + "\n"
        )
        log.flush()

    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        hyper = job["hyper"]
        torch.manual_seed(int(hyper.get("seed", 42)))
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            job.get("compute_dtype", "float32")
        ]
        source, kwargs = model_source(job)
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": dtype}
        if job.get("method") == "qlora":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            model_kwargs["device_map"] = {"": 0}
        try:
            model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        except TypeError:  # ältere transformers-Versionen: torch_dtype statt dtype
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        if job.get("method") == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=int(hyper.get("lora_r", 16)),
            lora_alpha=int(hyper.get("lora_alpha", 32)),
            lora_dropout=float(hyper.get("lora_dropout", 0.05)),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        dataset_dir = Path(job["dataset_dir"])
        max_len = int(hyper.get("max_seq_len", 1024))
        train = [
            f
            for e in load_split(dataset_dir, "train")
            if (f := features(tokenizer, e["messages"], max_len))
        ]
        val = [
            f
            for e in load_split(dataset_dir, "val")
            if (f := features(tokenizer, e["messages"], max_len))
        ]
        if not train:
            raise ValueError("no usable training examples after tokenization")

        class Progress(TrainerCallback):  # type: ignore[misc]
            def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
                if (out / "STOP").exists():
                    control.should_training_stop = True
                elapsed = time.time() - started
                eta = (elapsed / max(state.global_step, 1)) * (state.max_steps - state.global_step)
                emit(
                    status="training",
                    step=state.global_step,
                    max_steps=state.max_steps,
                    epoch=round(state.epoch or 0, 3),
                    eta_s=round(eta),
                )
                return control

            def on_log(
                self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any
            ) -> Any:
                if logs and "loss" in logs:
                    emit(loss=round(float(logs["loss"]), 4))
                if logs and "eval_loss" in logs:
                    emit(eval_loss=round(float(logs["eval_loss"]), 4))
                return control

        on_gpu = torch.cuda.is_available()
        config_values: dict[str, Any] = {
            "output_dir": str(out / "checkpoints"),
            "num_train_epochs": float(hyper.get("epochs", 2)),
            "per_device_train_batch_size": int(hyper.get("batch_size", 1)),
            "per_device_eval_batch_size": int(hyper.get("batch_size", 1)),
            "gradient_accumulation_steps": int(hyper.get("grad_accum", 16)),
            "learning_rate": float(hyper.get("learning_rate", 2e-4)),
            "logging_steps": 1,
            "save_steps": int(hyper.get("save_steps", 50)),
            "save_total_limit": 2,
            "seed": int(hyper.get("seed", 42)),
            "report_to": [],
            "gradient_checkpointing": on_gpu,
            "fp16": job.get("compute_dtype") == "float16",
            "bf16": job.get("compute_dtype") == "bfloat16",
            "use_cpu": not on_gpu,
            "eval_strategy": "epoch" if val else "no",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
        }
        if hyper.get("max_steps"):
            config_values["max_steps"] = int(hyper["max_steps"])
        args = TrainingArguments(**only_supported(TrainingArguments, config_values))
        trainer = Trainer(
            **only_supported(
                Trainer,
                {
                    "model": model,
                    "args": args,
                    "train_dataset": train,
                    "eval_dataset": val or None,
                    "data_collator": Collator(tokenizer.pad_token_id),
                    "callbacks": [Progress()],
                },
            )
        )
        checkpoints = sorted((out / "checkpoints").glob("checkpoint-*"))
        emit(status="training", resumed_from=checkpoints[-1].name if checkpoints else None)
        result = trainer.train(resume_from_checkpoint=bool(checkpoints))
        if (out / "STOP").exists():
            emit(status="cancelled")
            return 3
        emit(status="saving")
        model.save_pretrained(str(out / "adapter"))
        tokenizer.save_pretrained(str(out / "adapter"))
        lineage = {
            "job_id": job["job_id"],
            "base_model": job["base_model"],
            "base_revision": job["base_revision"],
            "method": job.get("method"),
            "dataset_dir": str(dataset_dir),
            "dataset_manifest_sha256": job.get("dataset_manifest_sha256"),
            "config_hash": job.get("config_hash"),
            "hyper": hyper,
            "compute_dtype": job.get("compute_dtype"),
            "train_loss": float(result.training_loss),
            "steps": int(result.global_step),
            "duration_s": round(time.time() - started, 1),
            "train_examples": len(train),
            "trainer": "transformers.Trainer + PEFT (assistant-only loss)",
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "device": torch.cuda.get_device_name(0) if on_gpu else "cpu",
        }
        write_json(out / "lineage.json", lineage)
        emit(status="completed", loss=round(float(result.training_loss), 4))
        return 0
    except Exception as exc:
        emit(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.write(traceback.format_exc())
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
