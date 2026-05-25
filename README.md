# COCONUT LM

This repository implements a small decoder-only language model that uses **COCONUT: Chain of Continuous Thought** style latent reasoning.

Instead of generating natural-language chain-of-thought tokens, the model reserves `<latent>` positions in the prompt. During a forward pass, each latent position is filled with the previous hidden state of the transformer. The model then predicts the answer after those continuous latent steps.

The implementation is intentionally compact:

- `TinyCoconutLM`: GPT-like decoder with causal self-attention.
- Continuous thought injection: `<latent>` token embeddings are replaced by hidden states.
- Dataset builder: writes visible JSONL examples such as `Q: 7+5=\nA:<latent><latent> 12`.
- Toy arithmetic trainer: reads that JSONL corpus instead of synthesizing batches inside the loop.
- Generation CLI: runs latent steps internally and decodes only visible answer tokens.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Build The Toy Pretraining File

```bash
coconut-build-dataset --out data/addition_train.jsonl --examples 1000 --latent-steps 4
```

Each JSONL row has a `text` field that acts like the model's tiny pretraining document:

```json
{"text": "Q: 79+68=\nA:<latent><latent><latent><latent> 147", "left": 79, "right": 68, "answer": 147}
```

## Train A Toy Model

```bash
coconut-train --dataset data/addition_train.jsonl --steps 1000 --batch-size 64 --out-dir runs/addition
```

## Generate

```bash
coconut-generate --checkpoint runs/addition/model.pt --prompt "Q: 8+9=\nA:" --latent-steps 4
```

## Why Continuous Thoughts?

A standard chain-of-thought model commits to discrete text tokens for intermediate reasoning. COCONUT-style models keep the intermediate reasoning in the model's hidden space, using continuous vectors as soft thought states. This repository is a minimal educational implementation of that mechanism rather than a large-scale reproduction.

## How Training Differs From A Normal LM

Normal decoder-only pretraining is straightforward next-token prediction: embed every token, run one causal forward pass, and compute cross-entropy on the next token at each position.

COCONUT keeps the same next-token objective for visible text, but changes the forward pass at `<latent>` positions:

1. Read the prefix before the first `<latent>`.
2. Run the transformer on that prefix.
3. Copy the last hidden state into the `<latent>` token's input embedding slot.
4. Repeat for later latent positions, so each continuous thought can depend on the previous continuous thought.
5. Run the final causal pass and compute loss, masking out `<latent>` labels because they are internal reasoning states rather than text to predict.

So the corpus still looks like pretraining text, but the model is trained with extra latent compute inserted between prompt and answer.

## Reference

- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
