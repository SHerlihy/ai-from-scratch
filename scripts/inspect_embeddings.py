from pathlib import Path

import torch

from ai_from_scratch.data import create_dataloader_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = PROJECT_ROOT / "the-verdict.txt"


def main() -> None:
    raw_text = TEXT_PATH.read_text(encoding="utf-8")

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, _targets = next(data_iter)

    print("Token IDs\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


if __name__ == "__main__":
    main()

