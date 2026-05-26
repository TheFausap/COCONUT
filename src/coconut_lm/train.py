from __future__ import annotations

import argparse
import dataclasses
import random
from pathlib import Path

import torch

from coconut_lm.config import CoconutConfig
from coconut_lm.data import (
    DEFAULT_VOCAB_TEXT,
    collate_examples,
    encode_texts,
    make_batch_from_examples,
    read_jsonl_texts,
)
from coconut_lm.model import TinyCoconutLM
from coconut_lm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or continue a tiny COCONUT LM.")
    parser.add_argument("--dataset", type=Path, default=Path("data/addition_train.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Continue training from a checkpoint.")
    parser.add_argument("--steps", type=int, default=None, help="Random-sampling updates. If omitted, epochs are used.")
    parser.add_argument("--epochs", type=int, default=1, help="Full shuffled passes over the dataset when --steps is omitted.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/addition"))
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def load_or_create_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    texts: list[str],
) -> tuple[CharTokenizer, CoconutConfig, TinyCoconutLM]:
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        tokenizer = CharTokenizer.from_stoi(checkpoint["tokenizer"])
        config = CoconutConfig(**checkpoint["config"])
        model = TinyCoconutLM(config).to(device)
        model.load_state_dict(checkpoint["model"])
        return tokenizer, config, model

    tokenizer = CharTokenizer.build(DEFAULT_VOCAB_TEXT + "".join(texts))
    config = CoconutConfig(
        vocab_size=len(tokenizer.itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_id,
        latent_token_id=tokenizer.latent_id,
    )
    model = TinyCoconutLM(config).to(device)
    return tokenizer, config, model


def iter_epoch_batches(
    examples: list[list[int]],
    *,
    batch_size: int,
    epochs: int,
    rng: random.Random,
) -> list[list[list[int]]]:
    batches: list[list[list[int]]] = []
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batches.append([examples[idx] for idx in order[start : start + batch_size]])
    return batches


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    texts = read_jsonl_texts(args.dataset)
    tokenizer, config, model = load_or_create_model(args, device=device, texts=texts)
    examples = encode_texts(tokenizer, texts, config.block_size)
    latent_steps = texts[0].count("<latent>")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    if args.steps is None:
        epoch_batches = iter_epoch_batches(
            examples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            rng=rng,
        )
        total_steps = len(epoch_batches)
    else:
        epoch_batches = []
        total_steps = args.steps

    for step in range(1, total_steps + 1):
        if args.steps is None:
            input_ids, targets = collate_examples(
                epoch_batches[step - 1],
                tokenizer=tokenizer,
                device=device,
            )
        else:
            input_ids, targets = make_batch_from_examples(
                examples,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                rng=rng,
                device=device,
            )
        output = model(input_ids, targets=targets)
        assert output.loss is not None
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(f"step {step:05d} | loss {output.loss.item():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": dataclasses.asdict(config),
        "model": model.state_dict(),
        "tokenizer": tokenizer.stoi,
        "dataset": str(args.dataset),
        "latent_steps": latent_steps,
        "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
    }
    path = args.out_dir / "model.pt"
    torch.save(checkpoint, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
