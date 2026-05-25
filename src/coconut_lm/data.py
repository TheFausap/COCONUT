from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from coconut_lm.tokenizer import CharTokenizer

if TYPE_CHECKING:
    import torch


TRAIN_ALPHABET = "0123456789Q:+\nA: ="


def build_addition_tokenizer() -> CharTokenizer:
    return CharTokenizer.build(TRAIN_ALPHABET)


def render_addition_text(left: int, right: int, latent_steps: int) -> str:
    return f"Q: {left}+{right}=\nA:" + ("<latent>" * latent_steps) + f" {left + right}"


def write_addition_dataset(
    path: Path,
    *,
    examples: int,
    max_value: int,
    latent_steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for _ in range(examples):
            left = rng.randint(0, max_value)
            right = rng.randint(0, max_value)
            record = {
                "text": render_addition_text(left, right, latent_steps),
                "left": left,
                "right": right,
                "answer": left + right,
            }
            file.write(json.dumps(record) + "\n")


def read_jsonl_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no} must be JSON with a string 'text' field") from exc
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_no} field 'text' must be a string")
            texts.append(text)
    if not texts:
        raise ValueError(f"{path} does not contain any training examples")
    return texts


def encode_texts(tokenizer: CharTokenizer, texts: list[str], block_size: int) -> list[list[int]]:
    encoded: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > block_size:
            raise ValueError(f"example has {len(ids)} tokens, exceeding block size {block_size}")
        encoded.append(ids)
    return encoded


def make_batch_from_examples(
    examples: list[list[int]],
    *,
    tokenizer: CharTokenizer,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    max_len = max(len(example) for example in batch)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_id, dtype=torch.long)
    for row, example in enumerate(batch):
        input_ids[row, : len(example)] = torch.tensor(example, dtype=torch.long)
    input_ids = input_ids.to(device)
    return input_ids, input_ids.clone()
