import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_lm.data import (
    build_default_tokenizer,
    read_jsonl_texts,
    write_addition_dataset,
    write_hf_plain_text_dataset,
    write_hf_proofs3_qa_dataset,
    write_hf_ultrafineweb_qa_dataset,
    write_plain_text_dataset,
    write_qa_dataset,
)
from coconut_lm.tokenizer import CharTokenizer


class DatasetTest(unittest.TestCase):
    def test_dataset_file_contains_visible_latent_markers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "addition.jsonl"
            write_addition_dataset(path, examples=3, max_value=9, latent_steps=2, seed=1)

            lines = path.read_text(encoding="utf-8").splitlines()
            texts = read_jsonl_texts(path)

        self.assertEqual(len(lines), 3)
        first = json.loads(lines[0])
        self.assertIn("<latent><latent>", first["text"])
        self.assertEqual(texts, [json.loads(line)["text"] for line in lines])

    def test_curriculum_dataset_kinds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plain_path = tmp_path / "plain.jsonl"
            qa_path = tmp_path / "qa.jsonl"
            latent_path = tmp_path / "latent.jsonl"

            write_plain_text_dataset(plain_path, examples=2, seed=1)
            write_qa_dataset(qa_path, examples=2, max_value=9, latent_steps=0, seed=1)
            write_qa_dataset(latent_path, examples=2, max_value=9, latent_steps=3, seed=1)

            plain = json.loads(plain_path.read_text(encoding="utf-8").splitlines()[0])
            qa = json.loads(qa_path.read_text(encoding="utf-8").splitlines()[0])
            latent = json.loads(latent_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(plain["kind"], "plain-text")
        self.assertEqual(qa["kind"], "qa")
        self.assertNotIn("<latent>", qa["text"])
        self.assertEqual(latent["kind"], "latent-qa")
        self.assertIn("<latent><latent><latent>", latent["text"])

    def test_hf_plain_text_dataset_uses_rows_api_shape(self):
        def fake_fetch_page(**kwargs):
            self.assertEqual(kwargs["dataset"], "shreyasharma/sentences_truthv2")
            return {
                "rows": [
                    {
                        "row_idx": 0,
                        "row": {
                            "sentence": "  The sky is blue.  ",
                            "truth": True,
                        },
                    },
                    {
                        "row_idx": 1,
                        "row": {
                            "sentence": "Water is wet.",
                            "truth": True,
                        },
                    },
                ]
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "hf.jsonl"
            write_hf_plain_text_dataset(
                path,
                dataset="shreyasharma/sentences_truthv2",
                config="default",
                split="train",
                examples=2,
                rows=(
                    (item["row_idx"], item["row"])
                    for item in fake_fetch_page(dataset="shreyasharma/sentences_truthv2")["rows"]
                ),
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["text"], "The sky is blue.")
        self.assertEqual(records[0]["kind"], "hf-plain-text")
        self.assertEqual(records[0]["source"], "shreyasharma/sentences_truthv2")
        self.assertEqual(records[1]["row_idx"], 1)

    def test_proofs3_qa_dataset_formats_context_question_and_latents(self):
        def fake_fetch_page(**kwargs):
            self.assertEqual(kwargs["dataset"], "shreyasharma/proofs3")
            return {
                "rows": [
                    {
                        "row_idx": 7,
                        "row": {
                            "triples": {
                                "sent1": "leo is a kind of constellation",
                                "sent2": "constellations contain stars",
                                "sent3": None,
                            },
                            "question": "What contains stars?",
                            "answer": "constellations",
                            "hypothesis": "constellations contain stars",
                            "step_proof": "sent1 & sent2 -> hypothesis;",
                            "label": 1,
                        },
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "proofs3.jsonl"
            write_hf_proofs3_qa_dataset(
                path,
                dataset="shreyasharma/proofs3",
                config="default",
                split="train",
                examples=1,
                latent_steps=2,
                rows=(
                    (item["row_idx"], item["row"])
                    for item in fake_fetch_page(dataset="shreyasharma/proofs3")["rows"]
                ),
            )
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(record["kind"], "proofs3-latent-qa")
        self.assertIn("Context:\n- leo is a kind of constellation", record["text"])
        self.assertIn("Question: What contains stars?", record["text"])
        self.assertIn("Answer:<latent><latent> constellations", record["text"])
        self.assertEqual(record["row_idx"], 7)

    def test_ultrafineweb_qa_splits_one_source_into_qa_examples(self):
        rows = [
            (
                3,
                {
                    "uid": "abc",
                    "style": "qa",
                    "content": (
                        "Source paragraph about tides and the moon.\n\n"
                        "Question: What causes tides? Answer: Gravity from the moon and sun.\n\n"
                        "Question: Which body has the strongest effect?\n"
                        "A) Mars\nB) The moon Answer: B) The moon"
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ultra.jsonl"
            write_hf_ultrafineweb_qa_dataset(
                path,
                dataset="openbmb/Ultra-FineWeb-L3",
                config="Ultra-FineWeb-L3-en-QA-Synthetic",
                split="train",
                examples=2,
                latent_steps=2,
                rows=rows,
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["kind"], "ultrafineweb-latent-qa")
        self.assertIn("Context:\nSource paragraph", records[0]["text"])
        self.assertIn("Answer:<latent><latent> Gravity from the moon and sun.", records[0]["text"])
        self.assertIn("A) Mars B) The moon", records[1]["question"])
        self.assertEqual(records[1]["answer"], "B) The moon")


class TokenizerTest(unittest.TestCase):
    def test_encode_recognizes_special_tokens_inside_text(self):
        tokenizer = CharTokenizer.build("Q: 1+2=\nA: 3")

        ids = tokenizer.encode("Q: 1+2=\nA:<latent> 3", bos=True, eos=True)

        self.assertIn(tokenizer.latent_id, ids)
        self.assertEqual(tokenizer.decode(ids), "Q: 1+2=\nA: 3")

    def test_default_tokenizer_covers_toy_english_curriculum(self):
        tokenizer = build_default_tokenizer()

        ids = tokenizer.encode("Question: What is 8 plus 9?\nAnswer:<latent> 17")

        self.assertIn(tokenizer.latent_id, ids)

    def test_tokenizer_uses_unknown_token_for_unseen_characters(self):
        tokenizer = CharTokenizer.build("abc")

        ids = tokenizer.encode("abc🙂")

        self.assertEqual(ids[-1], tokenizer.unk_id)


if __name__ == "__main__":
    unittest.main()
