import unittest
from unittest.mock import patch

import torch

from ai_from_scratch.config import get_settings
from ai_from_scratch.model.model import GPTModel
from ai_from_scratch.model.model import generate_text_simple

cfg = get_settings()

class GenerateTextTest(unittest.TestCase):
    def test_single_iteration(self):
        start_context = "Hello, I am"

        try:
            tokenizer = tiktoken.get_encoding("gpt2")
        except Exception as exc:
            self.skipTest(f"gpt2 tiktoken encoding is unavailable: {exc}")

        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        model = GPTModel(cfg)

        model.eval()

        out = generate_text_simple(
                model=model,
                idx=encoded_tensor,
                max_new_tokens=6,
                context_size=cfg["context_length"]
            )

        decoded_text = tokenizer.decode(out.squeeze(0).tolist())

        self.assertGreater(decoded_text, start_context)


if __name__ == "__main__":
    unittest.main()
