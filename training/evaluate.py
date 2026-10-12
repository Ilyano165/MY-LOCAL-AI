"""Unabhängige Evaluation: Basismodell vs. trainierter Adapter (läuft in der Trainingsumgebung).

    python evaluate.py job.json

Bewertet **deterministisch** (kein Modell bewertet sich selbst):
* Testsplit des Datensatzes: Token-F1 der Antwort gegen die Referenzaussage, Quellenangabe
  (nennt die Antwort eine Referenzquelle?).
* Regressions-Suite: feste Allgemeinfragen mit erwarteten Teilantworten – das Training darf
  vorhandene Fähigkeiten nicht zerstören.

Ergebnis: ``eval/report.json`` + ``eval/report.md`` mit Freigabe-Entscheidung (``gate``).
Die Freigabe selbst bleibt eine Entscheidung des Menschen in NOVA.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REGRESSION_SUITE: list[tuple[str, list[str]]] = [
    ("What is 12 multiplied by 7? Answer with the number only.", ["84"]),
    ("What is 100 minus 37? Answer with the number only.", ["63"]),
    ("What is the capital of France?", ["paris"]),
    ("Name the largest planet in our solar system.", ["jupiter"]),
    ("How many days are in a week?", ["7", "seven"]),
    ("How many sides does a triangle have?", ["3", "three"]),
    ("Translate 'thank you' into German.", ["danke"]),
    ("Write the word 'apple' backwards.", ["elppa"]),
    ("What gas do plants absorb for photosynthesis?", ["carbon dioxide", "co2"]),
    ("At what temperature in Celsius does water boil at sea level?", ["100"]),
]
DEFAULT_GATE = {"min_test_examples": 10, "min_f1_gain": 0.02, "max_regression_drop": 1}
_WORD = re.compile(r"[\w-]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    common: dict[str, int] = {}
    for t in ref:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def reference_parts(answer: str) -> tuple[str, list[str]]:
    statement, _, sources = answer.partition("\n\nSources:")
    urls = re.findall(r"https?://[^\s)]+", sources)
    return statement.strip(), urls


def cites_source(prediction: str, urls: list[str]) -> bool:
    low = prediction.lower()
    for url in urls:
        host = re.sub(r"^https?://(www\.)?", "", url.lower()).split("/")[0]
        if url.lower() in low or (host and host in low):
            return True
    return False


def passes(prediction: str, expected: list[str]) -> bool:
    low = prediction.lower()
    return any(e in low for e in expected)


def gate(report: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    cfg = {**DEFAULT_GATE, **settings}
    base, tuned = report["base"], report["tuned"]
    reasons, ok = [], True
    if report["test_examples"] < cfg["min_test_examples"]:
        ok = False
        reasons.append(
            f"only {report['test_examples']} test examples (need {cfg['min_test_examples']})"
        )
    gain = tuned["f1"] - base["f1"]
    if gain < cfg["min_f1_gain"]:
        ok = False
        reasons.append(f"F1 gain {gain:+.3f} below required {cfg['min_f1_gain']:+.3f}")
    else:
        reasons.append(f"F1 gain {gain:+.3f}")
    drop = base["regression_passed"] - tuned["regression_passed"]
    if drop > cfg["max_regression_drop"]:
        ok = False
        reasons.append(f"general regression: {drop} fewer checks passed than the base model")
    else:
        reasons.append(
            f"general checks {tuned['regression_passed']}/{report['regression_total']} "
            f"(base {base['regression_passed']})"
        )
    return {"eligible": ok, "reasons": reasons, "settings": cfg}


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new: int) -> str:
    import torch

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return str(tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)).strip()


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["output_dir"])
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    status = eval_dir / "status.json"

    def emit(**fields: Any) -> None:
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps({**fields, "updated_at": time.time()}), encoding="utf-8")
        os.replace(tmp, status)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from runner import load_split, model_source  # type: ignore[import-not-found]

        emit(status="loading")
        source, kwargs = model_source(job)
        tokenizer = AutoTokenizer.from_pretrained(str(out / "adapter"))
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            job.get("compute_dtype", "float32")
        ]
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": dtype}
        if job.get("method") == "qlora":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            model_kwargs["device_map"] = {"": 0}
        try:
            base = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            base = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        model = PeftModel.from_pretrained(base, str(out / "adapter"))
        model.eval()
        max_new = int(job.get("hyper", {}).get("eval_max_new_tokens", 160))
        limit = int(job.get("hyper", {}).get("eval_max_examples", 100))
        tests = load_split(Path(job["dataset_dir"]), "test")[:limit]
        total = len(tests) + len(REGRESSION_SUITE)
        rows, done = [], 0
        scores: dict[str, dict[str, float]] = {
            k: {"f1": 0.0, "cited": 0.0} for k in ("base", "tuned")
        }
        for example in tests:
            messages = example["messages"][:-1]
            statement, urls = reference_parts(example["messages"][-1]["content"])
            with model.disable_adapter():
                base_answer = generate(model, tokenizer, messages, max_new)
            tuned_answer = generate(model, tokenizer, messages, max_new)
            row = {"id": example["id"], "question": messages[-1]["content"], "reference": statement}
            for name, answer in (("base", base_answer), ("tuned", tuned_answer)):
                f1, cited = token_f1(answer, statement), cites_source(answer, urls)
                scores[name]["f1"] += f1
                scores[name]["cited"] += float(cited)
                row[name] = {"answer": answer[:600], "f1": round(f1, 3), "cited": cited}
            rows.append(row)
            done += 1
            emit(status="evaluating", done=done, total=total)
        regression: dict[str, int] = {"base": 0, "tuned": 0}
        reg_rows = []
        for prompt, expected in REGRESSION_SUITE:
            msgs = [{"role": "user", "content": prompt}]
            with model.disable_adapter():
                b = generate(model, tokenizer, msgs, 48)
            t = generate(model, tokenizer, msgs, 48)
            regression["base"] += passes(b, expected)
            regression["tuned"] += passes(t, expected)
            reg_rows.append(
                {
                    "prompt": prompt,
                    "base": b[:200],
                    "tuned": t[:200],
                    "base_ok": passes(b, expected),
                    "tuned_ok": passes(t, expected),
                }
            )
            done += 1
            emit(status="evaluating", done=done, total=total)
        n = max(len(tests), 1)
        report: dict[str, Any] = {
            "job_id": job["job_id"],
            "base_model": job["base_model"],
            "base_revision": job["base_revision"],
            "test_examples": len(tests),
            "regression_total": len(REGRESSION_SUITE),
            "base": {
                "f1": round(scores["base"]["f1"] / n, 4),
                "cited_rate": round(scores["base"]["cited"] / n, 4),
                "regression_passed": regression["base"],
            },
            "tuned": {
                "f1": round(scores["tuned"]["f1"] / n, 4),
                "cited_rate": round(scores["tuned"]["cited"] / n, 4),
                "regression_passed": regression["tuned"],
            },
            "decoding": {"greedy": True, "max_new_tokens": max_new},
            "examples": rows,
            "regression": reg_rows,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        report["gate"] = gate(report, job.get("gate", {}))
        (eval_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (eval_dir / "report.md").write_text(render(report), encoding="utf-8")
        emit(status="completed", eligible=report["gate"]["eligible"])
        return 0
    except Exception as exc:
        emit(status="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


def render(report: dict[str, Any]) -> str:
    b, t, g = report["base"], report["tuned"], report["gate"]
    lines = [
        f"# Evaluation – {report['job_id']}",
        "",
        f"Base `{report['base_model']}@{report['base_revision'][:12]}` · "
        f"{report['test_examples']} test "
        f"examples · {report['regression_total']} general checks · greedy decoding",
        "",
        "| | Base | Trained |",
        "|---|---|---|",
        f"| Token-F1 vs. reference | {b['f1']:.3f} | {t['f1']:.3f} |",
        f"| Cites a reference source | {b['cited_rate']:.0%} | {t['cited_rate']:.0%} |",
        f"| General checks passed | {b['regression_passed']}/{report['regression_total']} | "
        f"{t['regression_passed']}/{report['regression_total']} |",
        "",
        f"**Gate: {'eligible for approval' if g['eligible'] else 'NOT eligible'}** – "
        f"{'; '.join(g['reasons'])}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
