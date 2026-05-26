from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from coconut_lm.tokenizer import CharTokenizer

if TYPE_CHECKING:
    import torch


DEFAULT_VOCAB_TEXT = string.ascii_letters + string.digits + string.punctuation + " \n\t"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
TEXT_COLUMN_CANDIDATES = [
    "text",
    "sentence",
    "content",
    "statement",
    "prompt",
    "question",
]
PROOFS3_DATASET = "shreyasharma/proofs3"

PLAIN_TEXT_TEMPLATES = [
    "The sun is bright and warm.",
    "A small model learns from simple text.",
    "Cats and dogs are common pets.",
    "Water freezes when it is very cold.",
    "A book can contain many useful ideas.",
    "People ask questions and expect helpful answers.",
    "The sky is blue on a clear day.",
    "A number can be added to another number.",
    "Language models predict the next token.",
    "Reasoning can take more than one step.",
]

FACT_QA = [
    ("What color is the sky on a clear day?", "blue"),
    ("What do bees make?", "honey"),
    ("What do people drink when they are thirsty?", "water"),
    ("What is the opposite of hot?", "cold"),
    ("How many days are in a week?", "seven"),
    ("What animal says meow?", "cat"),
    ("What shape has three sides?", "triangle"),
    ("What color is grass usually?", "green"),
]


def build_default_tokenizer() -> CharTokenizer:
    return CharTokenizer.build(DEFAULT_VOCAB_TEXT)


def render_addition_text(left: int, right: int, latent_steps: int) -> str:
    return f"Q: {left}+{right}=\nA:" + ("<latent>" * latent_steps) + f" {left + right}"


def render_qa_text(question: str, answer: str, latent_steps: int = 0) -> str:
    return f"Question: {question}\nAnswer:" + ("<latent>" * latent_steps) + f" {answer}"


def render_context_qa_text(context: list[str], question: str, answer: str, latent_steps: int = 0) -> str:
    context_block = "\n".join(f"- {fact}" for fact in context)
    return (
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        + ("<latent>" * latent_steps)
        + f" {answer}"
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_addition_dataset(
    path: Path,
    *,
    examples: int,
    max_value: int,
    latent_steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for _ in range(examples):
        left = rng.randint(0, max_value)
        right = rng.randint(0, max_value)
        records.append(
            {
                "text": render_addition_text(left, right, latent_steps),
                "kind": "addition",
                "left": left,
                "right": right,
                "answer": left + right,
            }
        )
    write_jsonl(path, records)


def write_plain_text_dataset(path: Path, *, examples: int, seed: int) -> None:
    rng = random.Random(seed)
    records = [
        {"text": rng.choice(PLAIN_TEXT_TEMPLATES), "kind": "plain-text"}
        for _ in range(examples)
    ]
    write_jsonl(path, records)


def fetch_hf_rows_page(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: float = 30.0,
) -> dict[str, Any]:
    query = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_ROWS_API}?{query}"
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Hugging Face returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"could not reach Hugging Face dataset API: {exc.reason}") from exc


def extract_text_from_hf_row(row: dict[str, Any], text_column: str | None = None) -> str:
    if text_column is not None:
        value = row.get(text_column)
        if not isinstance(value, str):
            raise ValueError(f"configured text column {text_column!r} is missing or is not a string")
        return value

    for candidate in TEXT_COLUMN_CANDIDATES:
        value = row.get(candidate)
        if isinstance(value, str) and value.strip():
            return value

    string_values = [value for value in row.values() if isinstance(value, str) and value.strip()]
    if not string_values:
        raise ValueError(f"could not infer a text column from row keys: {sorted(row.keys())}")
    return max(string_values, key=len)


def clean_plain_text(text: str, *, max_chars: int | None = None) -> str:
    cleaned = " ".join(text.split())
    if max_chars is not None:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def sorted_numbered_values(mapping: dict[str, Any], *, prefix: str, limit: int | None = None) -> list[str]:
    def key_number(key: str) -> int:
        suffix = key.removeprefix(prefix)
        return int(suffix) if suffix.isdigit() else 0

    values: list[str] = []
    for key in sorted(mapping, key=key_number):
        value = mapping[key]
        if isinstance(value, str) and value.strip():
            values.append(clean_plain_text(value))
        if limit is not None and len(values) >= limit:
            break
    return values


def write_hf_plain_text_dataset(
    path: Path,
    *,
    dataset: str,
    config: str,
    split: str,
    examples: int,
    text_column: str | None = None,
    max_chars: int | None = 500,
    page_size: int = 100,
    fetch_page: Callable[..., dict[str, Any]] = fetch_hf_rows_page,
) -> None:
    records: list[dict[str, object]] = []
    offset = 0
    page_size = max(1, min(page_size, 100))

    while len(records) < examples:
        page = fetch_page(
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=min(page_size, examples - len(records)),
        )
        rows = page.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", item)
            if not isinstance(row, dict):
                continue
            text = clean_plain_text(extract_text_from_hf_row(row, text_column), max_chars=max_chars)
            if not text:
                continue
            records.append(
                {
                    "text": text,
                    "kind": "hf-plain-text",
                    "source": dataset,
                    "split": split,
                    "row_idx": item.get("row_idx"),
                    "text_column": text_column,
                }
            )
            if len(records) >= examples:
                break
        offset += len(rows)

    if not records:
        raise ValueError(f"no text rows were fetched from {dataset}/{config}/{split}")
    write_jsonl(path, records)


def write_hf_proofs3_qa_dataset(
    path: Path,
    *,
    dataset: str,
    config: str,
    split: str,
    examples: int,
    latent_steps: int,
    max_context_sentences: int | None = 12,
    page_size: int = 100,
    fetch_page: Callable[..., dict[str, Any]] = fetch_hf_rows_page,
) -> None:
    records: list[dict[str, object]] = []
    offset = 0
    page_size = max(1, min(page_size, 100))

    while len(records) < examples:
        page = fetch_page(
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=min(page_size, examples - len(records)),
        )
        rows = page.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", item)
            if not isinstance(row, dict):
                continue
            question = row.get("question")
            answer = row.get("answer")
            triples = row.get("triples")
            if not isinstance(question, str) or not isinstance(answer, str) or not isinstance(triples, dict):
                continue
            context = sorted_numbered_values(triples, prefix="sent", limit=max_context_sentences)
            if not context:
                continue
            question = clean_plain_text(question)
            answer = clean_plain_text(answer)
            records.append(
                {
                    "text": render_context_qa_text(context, question, answer, latent_steps),
                    "kind": "proofs3-latent-qa" if latent_steps else "proofs3-qa",
                    "source": dataset,
                    "split": split,
                    "row_idx": item.get("row_idx"),
                    "question": question,
                    "answer": answer,
                    "hypothesis": clean_plain_text(row["hypothesis"]) if isinstance(row.get("hypothesis"), str) else None,
                    "step_proof": clean_plain_text(row["step_proof"]) if isinstance(row.get("step_proof"), str) else None,
                    "label": row.get("label"),
                    "context": context,
                }
            )
            if len(records) >= examples:
                break
        offset += len(rows)

    if not records:
        raise ValueError(f"no proofs3 QA rows were fetched from {dataset}/{config}/{split}")
    write_jsonl(path, records)


def write_qa_dataset(
    path: Path,
    *,
    examples: int,
    max_value: int,
    latent_steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for idx in range(examples):
        if idx % 2 == 0:
            left = rng.randint(0, max_value)
            right = rng.randint(0, max_value)
            question = f"What is {left} plus {right}?"
            answer = str(left + right)
            meta = {"left": left, "right": right}
        else:
            question, answer = rng.choice(FACT_QA)
            meta = {}
        records.append(
            {
                "text": render_qa_text(question, answer, latent_steps),
                "kind": "latent-qa" if latent_steps else "qa",
                "question": question,
                "answer": answer,
                **meta,
            }
        )
    write_jsonl(path, records)


def read_jsonl_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no} must be JSON with a string 'text' field") from exc
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_no} field 'text' must be a string")
            texts.append(text)
    if not texts:
        raise ValueError(f"{path} does not contain any training examples")
    return texts


def encode_texts(tokenizer: CharTokenizer, texts: list[str], block_size: int) -> list[list[int]]:
    encoded: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > block_size:
            raise ValueError(f"example has {len(ids)} tokens, exceeding block size {block_size}")
        encoded.append(ids)
    return encoded


def make_batch_from_examples(
    examples: list[list[int]],
    *,
    tokenizer: CharTokenizer,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    max_len = max(len(example) for example in batch)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_id, dtype=torch.long)
    for row, example in enumerate(batch):
        input_ids[row, : len(example)] = torch.tensor(example, dtype=torch.long)
    input_ids = input_ids.to(device)
    return input_ids, input_ids.clone()


def collate_examples(
    batch: list[list[int]],
    *,
    tokenizer: CharTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    max_len = max(len(example) for example in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_id, dtype=torch.long)
    for row, example in enumerate(batch):
        input_ids[row, : len(example)] = torch.tensor(example, dtype=torch.long)
    input_ids = input_ids.to(device)
    return input_ids, input_ids.clone()
