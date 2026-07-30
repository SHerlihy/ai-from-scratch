import unittest
from unittest.mock import patch

import torch

from ai_from_scratch.config import get_settings
from ai_from_scratch.model.transformer import TransformerBlock

cfg = get_settings()

class TransformerTests(unittest.TestCase):
    def test_vector_shape(self):
        torch.manual_seed(123)
        x = torch.rand(2,4,768)

        block = TransformerBlock(cfg)
        output = block(x)

        self.assertEqual(x.shape, output.shape)

if __name__ == "__main__":
    unittest.main()
