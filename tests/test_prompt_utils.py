"""CPU regression tests for text-conditioning padding."""

from __future__ import annotations

import pytest
import torch

from model.prompt_utils import mask_prompt_padding


def test_mask_prompt_padding_hides_only_padded_tokens() -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    original = embeddings.clone()
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int64)

    masked_embeddings, returned_mask = mask_prompt_padding(embeddings, attention_mask)

    expected = original.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
    torch.testing.assert_close(masked_embeddings, expected)
    torch.testing.assert_close(returned_mask, attention_mask.bool())
    assert masked_embeddings.dtype == embeddings.dtype
    # The helper must not mutate a text encoder's output in-place.
    torch.testing.assert_close(embeddings, original)


@pytest.mark.parametrize(
    ("embeddings_shape", "mask_shape"),
    [((2, 4), (2, 4)), ((2, 4, 8), (4,)), ((2, 4, 8), (2, 3))],
)
def test_mask_prompt_padding_rejects_incompatible_shapes(
    embeddings_shape: tuple[int, ...], mask_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        mask_prompt_padding(torch.zeros(embeddings_shape), torch.ones(mask_shape))
