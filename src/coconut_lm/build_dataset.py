from __future__ import annotations

import argparse
from pathlib import Path

from coconut_lm.data import write_addition_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visible JSONL corpus for COCONUT training.")
    parser.add_argument("--out", type=Path, default=Path("data/addition_train.jsonl"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_addition_dataset(
        args.out,
        examples=args.examples,
        max_value=args.max_value,
        latent_steps=args.latent_steps,
        seed=args.seed,
    )
    print(f"wrote {args.examples} examples to {args.out}")


if __name__ == "__main__":
    main()

