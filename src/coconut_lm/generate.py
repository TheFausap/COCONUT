from __future__ import annotations

import argparse
from pathlib import Path

import torch

from coconut_lm.config import CoconutConfig
from coconut_lm.model import TinyCoconutLM
from coconut_lm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with a trained tiny COCONUT LM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--latent-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_stoi(checkpoint["tokenizer"])
    config = CoconutConfig(**checkpoint["config"])
    model = TinyCoconutLM(config).to(device)
    model.load_state_dict(checkpoint["model"])

    latent_steps = args.latent_steps
    if latent_steps is None:
        latent_steps = int(checkpoint.get("latent_steps", 4))

    prefix = (
        [tokenizer.bos_id]
        + tokenizer.encode(args.prompt)
        + [tokenizer.latent_id] * latent_steps
    )
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_id,
        temperature=args.temperature,
        top_k=args.top_k,
        banned_token_ids={tokenizer.pad_id, tokenizer.bos_id, tokenizer.latent_id},
    )
    visible = output[0, len(prefix) :].tolist()
    print(tokenizer.decode(visible, skip_special=True))


if __name__ == "__main__":
    main()

