# COCONUT LM

This repository implements a small decoder-only language model that uses **COCONUT: Chain of Continuous Thought** style latent reasoning.

Instead of generating natural-language chain-of-thought tokens, the model reserves `<latent>` positions in the prompt. During a forward pass, each latent position is filled with the previous hidden state of the transformer. The model then predicts the answer after those continuous latent steps.

The implementation is intentionally compact:

- `TinyCoconutLM`: GPT-like decoder with causal self-attention.
- Continuous thought injection: `<latent>` token embeddings are replaced by hidden states.
- Dataset builder: writes visible JSONL examples for plain text, QA, latent QA, or old-style addition.
- Curriculum trainer: can start from scratch or continue from an earlier checkpoint.
- Generation CLI: runs latent steps internally and decodes only visible answer tokens.

## Install

```bash
python -m pip install -e ".[dev]"
```

For Hugging Face dataset downloads, install the optional HF dependencies:

```bash
python -m pip install -e ".[dev,hf]"
```

## Build Toy Training Files

```bash
coconut-build-dataset --kind plain-text --out data/plain_text.jsonl --examples 1000
coconut-build-dataset --kind hf-plain-text --out data/truthv2_plain_text.jsonl --examples 5000
coconut-build-dataset --kind qa --out data/qa.jsonl --examples 1000
coconut-build-dataset --kind latent-qa --out data/latent_qa.jsonl --examples 1000 --latent-steps 4
coconut-build-dataset --kind proofs3-qa --out data/proofs3_qa.jsonl --examples 5000
coconut-build-dataset --kind proofs3-latent-qa --out data/proofs3_latent_qa.jsonl --examples 5000 --latent-steps 4
```

Each JSONL row has a `text` field that acts like one tiny training document:

```json
{"text": "Question: What is 79 plus 68?\nAnswer:<latent><latent><latent><latent> 147", "kind": "latent-qa", "question": "What is 79 plus 68?", "answer": "147", "left": 79, "right": 68}
```

The `hf-plain-text` mode defaults to `shreyasharma/sentences_truthv2` and fetches rows with the Hugging Face Python library. It tries common text columns such as `text` and `sentence`; pass `--text-column COLUMN_NAME` if you want to force one. Prefer putting your token in `HF_TOKEN` instead of passing it on the command line:

```bash
export HF_TOKEN=hf_...

coconut-build-dataset \
  --kind hf-plain-text \
  --hf-dataset shreyasharma/sentences_truthv2 \
  --hf-config default \
  --hf-split train \
  --out data/truthv2_plain_text.jsonl \
  --examples 5000
```

By default the builder streams rows. Add `--no-streaming` if you want `datasets`/`huggingface_hub` to download and cache the split locally, which can take advantage of the Hugging Face cache and Xet support when `hf_xet` is installed.

The `proofs3-qa` and `proofs3-latent-qa` modes default to `shreyasharma/proofs3`. They format each row as context facts from `triples`, then the dataset `question`, then either `Answer:` or `Answer:<latent>...`.

## Train With A Curriculum

```bash
coconut-train --dataset data/truthv2_plain_text.jsonl --block-size 1024 --epochs 1 --out-dir runs/plain
coconut-train --dataset data/proofs3_qa.jsonl --checkpoint runs/plain/model.pt --epochs 1 --out-dir runs/proofs3_qa
coconut-train --dataset data/proofs3_latent_qa.jsonl --checkpoint runs/proofs3_qa/model.pt --epochs 1 --batch-size 1 --grad-accum-steps 16 --out-dir runs/proofs3_latent_qa
```

The first stage is normal next-token language modeling. The second stage teaches a simple question-answer format. The third stage keeps the same visible-answer objective, but inserts `<latent>` slots before the answer and fills those slots with continuous hidden states during the forward pass.

When `--steps` is omitted, training uses `--epochs` and makes full shuffled passes over the dataset without replacement. For example, 5000 rows with `--batch-size 64 --epochs 1` runs 79 optimizer updates. Use `--steps N` when you explicitly want random sampling with replacement.

Latent QA uses much more memory than plain QA because every `<latent>` position requires extra transformer passes over the prefix. On MPS, prefer small micro-batches and gradient accumulation:

```bash
coconut-train --dataset data/proofs3_latent_qa.jsonl \
  --checkpoint runs/proofs3_qa/model.pt \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --device mps \
  --out-dir runs/proofs3_latent_qa
```

## Generate

```bash
coconut-generate --checkpoint runs/proofs3_latent_qa/model.pt --prompt "Question: What is 8 plus 9?\nAnswer:" --latent-steps 4
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
