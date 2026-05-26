from __future__ import annotations

import argparse
from pathlib import Path

from coconut_lm.data import (
    write_addition_dataset,
    write_hf_plain_text_dataset,
    write_hf_proofs3_qa_dataset,
    write_plain_text_dataset,
    write_qa_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visible JSONL corpus for COCONUT training.")
    parser.add_argument(
        "--kind",
        choices=[
            "plain-text",
            "hf-plain-text",
            "proofs3-qa",
            "proofs3-latent-qa",
            "qa",
            "latent-qa",
            "addition",
        ],
        default="latent-qa",
        help="Dataset curriculum stage to generate.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/addition_train.jsonl"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default="default")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Prefer --hf-token-env for shell history safety.")
    parser.add_argument("--no-streaming", action="store_true", help="Download/cache the dataset split instead of streaming rows.")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--max-context-sentences", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_dataset = args.hf_dataset
    if hf_dataset is None and args.kind == "hf-plain-text":
        hf_dataset = "shreyasharma/sentences_truthv2"
    elif hf_dataset is None and args.kind.startswith("proofs3-"):
        hf_dataset = "shreyasharma/proofs3"

    if args.kind == "plain-text":
        write_plain_text_dataset(args.out, examples=args.examples, seed=args.seed)
    elif args.kind == "hf-plain-text":
        write_hf_plain_text_dataset(
            args.out,
            dataset=hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            examples=args.examples,
            text_column=args.text_column,
            max_chars=args.max_chars,
            streaming=not args.no_streaming,
            token=args.hf_token,
            token_env=args.hf_token_env,
        )
    elif args.kind == "proofs3-qa":
        write_hf_proofs3_qa_dataset(
            args.out,
            dataset=hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            examples=args.examples,
            latent_steps=0,
            max_context_sentences=args.max_context_sentences,
            streaming=not args.no_streaming,
            token=args.hf_token,
            token_env=args.hf_token_env,
        )
    elif args.kind == "proofs3-latent-qa":
        write_hf_proofs3_qa_dataset(
            args.out,
            dataset=hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            examples=args.examples,
            latent_steps=args.latent_steps,
            max_context_sentences=args.max_context_sentences,
            streaming=not args.no_streaming,
            token=args.hf_token,
            token_env=args.hf_token_env,
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
