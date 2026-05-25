import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_lm.data import read_jsonl_texts, write_addition_dataset
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


class TokenizerTest(unittest.TestCase):
    def test_encode_recognizes_special_tokens_inside_text(self):
        tokenizer = CharTokenizer.build("Q: 1+2=\nA: 3")

        ids = tokenizer.encode("Q: 1+2=\nA:<latent> 3", bos=True, eos=True)

        self.assertIn(tokenizer.latent_id, ids)
        self.assertEqual(tokenizer.decode(ids), "Q: 1+2=\nA: 3")


if __name__ == "__main__":
    unittest.main()
