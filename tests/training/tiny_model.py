"""Erzeugt ein winziges, zufällig initialisiertes Causal-LM samt Tokenizer (ohne Download).

Nur für Tests der Trainings-/Evaluationspipeline auf der CPU. Läuft in der Trainingsumgebung:
    python tiny_model.py <zielordner> <dataset_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def main(target: str, dataset_dir: str) -> None:
    out = Path(target)
    texts = []
    for split in ("train", "val", "test"):
        path = Path(dataset_dir) / f"{split}.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                texts += [m["content"] for m in json.loads(line)["messages"]]
    specials = ["<pad>", "<unk>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<s>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        [*texts, "hello world 84 Paris"], trainers.WordLevelTrainer(special_tokens=specials)
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|end|>",
        bos_token="<s>",
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out)
    config = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.eos_token_id,
        bos_token_id=fast.bos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
