from __future__ import annotations

import random

import torch

from coconut_lm.tokenizer import CharTokenizer


TRAIN_ALPHABET = "0123456789Q:+\nA: ="


def build_addition_tokenizer() -> CharTokenizer:
    return CharTokenizer.build(TRAIN_ALPHABET)


def make_addition_example(
    tokenizer: CharTokenizer,
    *,
    max_value: int,
    latent_steps: int,
    rng: random.Random,
) -> list[int]:
    left = rng.randint(0, max_value)
    right = rng.randint(0, max_value)
    prompt = f"Q: {left}+{right}=\nA:"
    answer = f" {left + right}"
    return (
        [tokenizer.bos_id]
        + tokenizer.encode(prompt)
        + [tokenizer.latent_id] * latent_steps
        + tokenizer.encode(answer, eos=True)
    )


def make_batch(
    tokenizer: CharTokenizer,
    *,
    batch_size: int,
    max_value: int,
    latent_steps: int,
    block_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [
        make_addition_example(
            tokenizer,
            max_value=max_value,
            latent_steps=latent_steps,
            rng=rng,
        )
        for _ in range(batch_size)
    ]
    max_len = min(max(len(example) for example in examples), block_size)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_id, dtype=torch.long)
    for row, example in enumerate(examples):
        trimmed = example[:max_len]
        input_ids[row, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)
    input_ids = input_ids.to(device)
    return input_ids, input_ids.clone()

