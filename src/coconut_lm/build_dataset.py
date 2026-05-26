from __future__ import annotations

import argparse
from pathlib import Path

from coconut_lm.data import (
    write_addition_dataset,
    write_hf_plain_text_dataset,
    write_plain_text_dataset,
    write_qa_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visible JSONL corpus for COCONUT training.")
    parser.add_argument(
        "--kind",
        choices=["plain-text", "hf-plain-text", "qa", "latent-qa", "addition"],
        default="latent-qa",
        help="Dataset curriculum stage to generate.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/addition_train.jsonl"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hf-dataset", default="shreyasharma/sentences_truthv2")
    parser.add_argument("--hf-config", default="default")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--max-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "plain-text":
        write_plain_text_dataset(args.out, examples=args.examples, seed=args.seed)
    elif args.kind == "hf-plain-text":
        write_hf_plain_text_dataset(
            args.out,
            dataset=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            examples=args.examples,
            text_column=args.text_column,
            max_chars=args.max_chars,
        )
    elif args.kind == "qa":
        write_qa_dataset(
            args.out,
            examples=args.examples,
            max_value=args.max_value,
            latent_steps=0,
            seed=args.seed,
        )
    elif args.kind == "latent-qa":
        write_qa_dataset(
            args.out,
            examples=args.examples,
            max_value=args.max_value,
            latent_steps=args.latent_steps,
            seed=args.seed,
        )
    else:
        write_addition_dataset(
            args.out,
            examples=args.examples,
            max_value=args.max_value,
            latent_steps=args.latent_steps,
            seed=args.seed,
        )
    print(f"wrote {args.examples} {args.kind} examples to {args.out}")


if __name__ == "__main__":
    main()
