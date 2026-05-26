from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import TYPE_CHECKING

from coconut_lm.tokenizer import CharTokenizer

if TYPE_CHECKING:
    import torch


DEFAULT_VOCAB_TEXT = string.ascii_letters + string.digits + string.punctuation + " \n\t"

PLAIN_TEXT_TEMPLATES = [
    "The sun is bright and warm.",
    "A small model learns from simple text.",
    "Cats and dogs are common pets.",
    "Water freezes when it is very cold.",
    "A book can contain many useful ideas.",
    "People ask questions and expect helpful answers.",
    "The sky is blue on a clear day.",
    "A number can be added to another number.",
    "Language models predict the next token.",
    "Reasoning can take more than one step.",
]

FACT_QA = [
    ("What color is the sky on a clear day?", "blue"),
    ("What do bees make?", "honey"),
    ("What do people drink when they are thirsty?", "water"),
    ("What is the opposite of hot?", "cold"),
    ("How many days are in a week?", "seven"),
    ("What animal says meow?", "cat"),
    ("What shape has three sides?", "triangle"),
    ("What color is grass usually?", "green"),
]


def build_default_tokenizer() -> CharTokenizer:
    return CharTokenizer.build(DEFAULT_VOCAB_TEXT)


def render_addition_text(left: int, right: int, latent_steps: int) -> str:
    return f"Q: {left}+{right}=\nA:" + ("<latent>" * latent_steps) + f" {left + right}"


def render_qa_text(question: str, answer: str, latent_steps: int = 0) -> str:
    return f"Question: {question}\nAnswer:" + ("<latent>" * latent_steps) + f" {answer}"


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_addition_dataset(
    path: Path,
    *,
    examples: int,
    max_value: int,
    latent_steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for _ in range(examples):
        left = rng.randint(0, max_value)
        right = rng.randint(0, max_value)
        records.append(
            {
                "text": render_addition_text(left, right, latent_steps),
                "kind": "addition",
                "left": left,
                "right": right,
                "answer": left + right,
            }
        )
    write_jsonl(path, records)


def write_plain_text_dataset(path: Path, *, examples: int, seed: int) -> None:
    rng = random.Random(seed)
    records = [
        {"text": rng.choice(PLAIN_TEXT_TEMPLATES), "kind": "plain-text"}
        for _ in range(examples)
    ]
    write_jsonl(path, records)


def write_qa_dataset(
    path: Path,
    *,
    examples: int,
    max_value: int,
    latent_steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for idx in range(examples):
        if idx % 2 == 0:
            left = rng.randint(0, max_value)
            right = rng.randint(0, max_value)
            question = f"What is {left} plus {right}?"
            answer = str(left + right)
            meta = {"left": left, "right": right}
        else:
            question, answer = rng.choice(FACT_QA)
            meta = {}
        records.append(
            {
                "text": render_qa_text(question, answer, latent_steps),
                "kind": "latent-qa" if latent_steps else "qa",
                "question": question,
                "answer": answer,
                **meta,
            }
        )
    write_jsonl(path, records)


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
