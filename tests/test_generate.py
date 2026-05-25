import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coconut_lm.generate import normalize_prompt


class GenerateCliTest(unittest.TestCase):
    def test_normalize_prompt_accepts_shell_newline_escape(self):
        self.assertEqual(normalize_prompt(r"Q: 8+9=\nA:"), "Q: 8+9=\nA:")


if __name__ == "__main__":
    unittest.main()

