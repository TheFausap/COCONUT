# COCONUT LM

This repository implements a small decoder-only language model that uses **COCONUT: Chain of Continuous Thought** style latent reasoning.

Instead of generating natural-language chain-of-thought tokens, the model reserves `<latent>` positions in the prompt. During a forward pass, each latent position is filled with the previous hidden state of the transformer. The model then predicts the answer after those continuous latent steps.

The implementation is intentionally compact:

- `TinyCoconutLM`: GPT-like decoder with causal self-attention.
- Continuous thought injection: `<latent>` token embeddings are replaced by hidden states.
- Toy arithmetic trainer: learns examples such as `Q: 7+5=\nA:<latent><latent> 12`.
- Generation CLI: runs latent steps internally and decodes only visible answer tokens.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Train A Toy Model

```bash
coconut-train --steps 1000 --batch-size 64 --latent-steps 4 --out-dir runs/addition
```

## Generate

```bash
coconut-generate --checkpoint runs/addition/model.pt --prompt "Q: 8+9=\nA:" --latent-steps 4
```

## Why Continuous Thoughts?

A standard chain-of-thought model commits to discrete text tokens for intermediate reasoning. COCONUT-style models keep the intermediate reasoning in the model's hidden space, using continuous vectors as soft thought states. This repository is a minimal educational implementation of that mechanism rather than a large-scale reproduction.

## Reference

- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
