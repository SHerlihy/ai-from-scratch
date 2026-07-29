import unittest
from unittest.mock import patch

import torch

from ai_from_scratch.model import CausalAttention
from ai_from_scratch.data import GPTDatasetV1, create_dataloader_v1

class CausalAttentionTests(unittest.TestCase):
    def test_vector_shape(self):
        text = "Your journey starts with one step"
        try:
            tokenizer = tiktoken.get_encoding("gpt2")
        except Exception as exc:
            self.skipTest(f"gpt2 tiktoken encoding is unavailable: {exc}")

        token_ids = torch.tensor(tokenizer.encode(text))
        embedding_layer = torch.nn.Embedding(
            num_embeddings=tokenizer.n_vocab,
            embedding_dim=3,
        )

        inputs = embedding_layer(token_ids)

        batch = torch.stack((inputs, inputs), dim=0)

        torch.manual_seed(123)
        context_length = batch.shape[1]
        ca = CausalAttention(d_in, d_out, context_length, 0.0)
        context_vecs = ca(batch)

        self.assertEqual(context_vecs.shape, torch.Size([2, 6, 2]))

if __name__ == "__main__":
    unittest.main()
