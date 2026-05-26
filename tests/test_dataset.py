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
                fetch_page=fake_fetch_page,
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["text"], "The sky is blue.")
        self.assertEqual(records[0]["kind"], "hf-plain-text")
        self.assertEqual(records[0]["source"], "shreyasharma/sentences_truthv2")
        self.assertEqual(records[1]["row_idx"], 1)


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


if __name__ == "__main__":
    unittest.main()
