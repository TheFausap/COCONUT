import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is not installed")
class CoconutModelTest(unittest.TestCase):
    def test_forward_returns_logits_loss_and_latents(self):
        from coconut_lm.config import CoconutConfig
        from coconut_lm.model import TinyCoconutLM

        config = CoconutConfig(
            vocab_size=8,
            block_size=12,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            pad_token_id=0,
            latent_token_id=3,
        )
        model = TinyCoconutLM(config)
        input_ids = torch.tensor([[1, 4, 5, 3, 3, 6, 2]])

        output = model(input_ids, targets=input_ids, return_latents=True)

        self.assertEqual(output.logits.shape, (1, 7, 8))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(len(output.latent_embeddings), 2)
        self.assertEqual(output.latent_embeddings[0].shape, (1, 16))

    def test_generate_appends_tokens(self):
        from coconut_lm.config import CoconutConfig
        from coconut_lm.model import TinyCoconutLM

        config = CoconutConfig(
            vocab_size=8,
            block_size=12,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            pad_token_id=0,
            latent_token_id=3,
        )
        model = TinyCoconutLM(config)
        input_ids = torch.tensor([[1, 4, 3, 3]])

        output = model.generate(
            input_ids,
            max_new_tokens=3,
            temperature=0.0,
            banned_token_ids={0, 1, 3},
        )

        self.assertEqual(output.shape[1], 7)


if __name__ == "__main__":
    unittest.main()

