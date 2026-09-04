"""Utilities for preparing text-conditioning tensors."""

from __future__ import annotations

import torch


def mask_prompt_padding(
    prompt_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero padded prompt rows while retaining their attention mask.

    Text encoders return a fixed-width sequence for batched prompts.  Keeping
    the tokenizer mask is important: a zero embedding is still a valid key to
    attention unless its corresponding mask entry is false.
    """

    if prompt_embeddings.ndim != 3:
        raise ValueError(
            "prompt_embeddings must be [B, L, D], "
            f"got {tuple(prompt_embeddings.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must be [B, L], "
            f"got {tuple(attention_mask.shape)}"
        )
    if prompt_embeddings.shape[:2] != attention_mask.shape:
        raise ValueError(
            "prompt_embeddings and attention_mask must agree on [B, L], "
            f"got {tuple(prompt_embeddings.shape[:2])} and {tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(device=prompt_embeddings.device, dtype=torch.bool)
    prompt_embeddings = prompt_embeddings.masked_fill(~mask.unsqueeze(-1), 0)
    return prompt_embeddings, mask
