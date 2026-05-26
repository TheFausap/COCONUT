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

## Build Toy Training Files

```bash
coconut-build-dataset --kind plain-text --out data/plain_text.jsonl --examples 1000
coconut-build-dataset --kind hf-plain-text --out data/truthv2_plain_text.jsonl --examples 1000
coconut-build-dataset --kind qa --out data/qa.jsonl --examples 1000
coconut-build-dataset --kind latent-qa --out data/latent_qa.jsonl --examples 1000 --latent-steps 4
```

Each JSONL row has a `text` field that acts like one tiny training document:

```json
{"text": "Question: What is 79 plus 68?\nAnswer:<latent><latent><latent><latent> 147", "kind": "latent-qa", "question": "What is 79 plus 68?", "answer": "147", "left": 79, "right": 68}
```

The `hf-plain-text` mode defaults to `shreyasharma/sentences_truthv2` and fetches rows through Hugging Face's dataset-viewer API, so local Parquet handling is not needed. It tries common text columns such as `text` and `sentence`; pass `--text-column COLUMN_NAME` if you want to force one:

```bash
coconut-build-dataset \
  --kind hf-plain-text \
  --hf-dataset shreyasharma/sentences_truthv2 \
  --hf-config default \
  --hf-split train \
  --out data/truthv2_plain_text.jsonl \
  --examples 5000
```

## Train With A Curriculum

```bash
coconut-train --dataset data/plain_text.jsonl --steps 1000 --out-dir runs/plain
coconut-train --dataset data/qa.jsonl --checkpoint runs/plain/model.pt --steps 1000 --out-dir runs/qa
coconut-train --dataset data/latent_qa.jsonl --checkpoint runs/qa/model.pt --steps 1000 --out-dir runs/latent_qa
```

The first stage is normal next-token language modeling. The second stage teaches a simple question-answer format. The third stage keeps the same visible-answer objective, but inserts `<latent>` slots before the answer and fills those slots with continuous hidden states during the forward pass.

## Generate

```bash
coconut-generate --checkpoint runs/latent_qa/model.pt --prompt "Question: What is 8 plus 9?\nAnswer:" --latent-steps 4
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
