import unittest
from unittest.mock import patch

import tiktoken
import torch

from ai_from_scratch.data import GPTDatasetV1, create_dataloader_v1


class FakeTokenizer:
    def encode(self, text):
        return [int(token) for token in text.split()]


class GPTDatasetV1Tests(unittest.TestCase):
    def test_dataset_creates_input_and_target_chunks(self):
        dataset = GPTDatasetV1(
            "0 1 2 3 4 5",
            tokenizer=FakeTokenizer(),
            max_length=3,
            stride=2,
        )

        self.assertEqual(len(dataset), 2)

        first_inputs, first_targets = dataset[0]
        second_inputs, second_targets = dataset[1]

        torch.testing.assert_close(first_inputs, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(first_targets, torch.tensor([1, 2, 3]))
        torch.testing.assert_close(second_inputs, torch.tensor([2, 3, 4]))
        torch.testing.assert_close(second_targets, torch.tensor([3, 4, 5]))

    def test_create_dataloader_returns_expected_batch_shape(self):
        with patch(
            "ai_from_scratch.data.dataset.tiktoken.get_encoding",
            return_value=FakeTokenizer(),
        ):
            dataloader = create_dataloader_v1(
                "0 1 2 3 4 5",
                batch_size=2,
                max_length=2,
                stride=1,
                shuffle=False,
                drop_last=True,
            )

        inputs, targets = next(iter(dataloader))

        self.assertEqual(inputs.shape, torch.Size([2, 2]))
        self.assertEqual(targets.shape, torch.Size([2, 2]))

    def test_gets_embeddings_from_journey_sentence(self):
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

        embeddings = embedding_layer(token_ids)

        self.assertGreater(len(token_ids), 0)
        self.assertEqual(embeddings.shape, torch.Size([6, 3]))
        torch.testing.assert_close(
            embeddings[0],
            embedding_layer.weight[token_ids[0]],
        )


if __name__ == "__main__":
    unittest.main()
