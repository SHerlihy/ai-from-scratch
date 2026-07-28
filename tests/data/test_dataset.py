import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
