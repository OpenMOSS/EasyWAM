from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch
from PIL import Image


class ActionInferenceResult(TypedDict):
    action: torch.Tensor


class JointInferenceResult(ActionInferenceResult, total=False):
    video: list[Image.Image]


class ActionBatchResult(TypedDict):
    action: torch.Tensor


class JointBatchResult(ActionBatchResult, total=False):
    video: list[list[Image.Image]]


def normalize_seeds(
    seeds: int | Sequence[int | None] | None,
    batch_size: int,
) -> list[int | None]:
    if seeds is None or isinstance(seeds, int):
        return [seeds] * batch_size
    values = list(seeds)
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} seeds, got {len(values)}.")
    if any(value is not None and not isinstance(value, int) for value in values):
        raise TypeError("Seeds must be integers or None.")
    return values


def batch_single_tensor(
    value: torch.Tensor | None, sample_ndim: int, name: str
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim == sample_ndim:
        return value.unsqueeze(0)
    if value.ndim != sample_ndim + 1 or value.shape[0] != 1:
        raise ValueError(
            f"Single-sample {name} requires rank {sample_ndim} or a batch of size 1, "
            f"got {tuple(value.shape)}."
        )
    return value


def batch_single_state(
    state: torch.Tensor | None, *, history: bool
) -> torch.Tensor | None:
    if state is None:
        return None
    if not isinstance(state, torch.Tensor):
        raise TypeError("proprio must be a torch.Tensor.")
    if history and state.ndim == 1:
        state = state.unsqueeze(0)
    return batch_single_tensor(state, 2 if history else 1, "proprio")


def validate_batch_inputs(
    input_image: torch.Tensor,
    prompt: Sequence[str] | None,
    context: torch.Tensor | None,
    context_mask: torch.Tensor | None,
) -> int:
    if not isinstance(input_image, torch.Tensor):
        raise TypeError("input_image must be a torch.Tensor.")
    if input_image.ndim != 4 or input_image.shape[1] != 3:
        raise ValueError(f"input_image must be [B,3,H,W], got {tuple(input_image.shape)}.")
    if not input_image.is_floating_point() or input_image.shape[0] == 0:
        raise ValueError(
            "input_image must contain a nonempty batch of normalized floating-point RGB images."
        )
    if prompt is None:
        if context is None or context_mask is None:
            raise ValueError("Either prompt or both context/context_mask must be provided.")
    elif context is not None or context_mask is not None:
        raise ValueError("prompt and context/context_mask are mutually exclusive.")
    batch_size = input_image.shape[0]
    if prompt is not None and (isinstance(prompt, str) or len(prompt) != batch_size):
        raise ValueError(f"Batch prompt must contain {batch_size} strings.")
    if prompt is not None and any(not isinstance(value, str) for value in prompt):
        raise TypeError("Batch prompt values must be strings.")
    if context is not None and (
        not isinstance(context, torch.Tensor) or not context.is_floating_point()
    ):
        raise TypeError("context must be a floating-point torch.Tensor.")
    if context_mask is not None and (
        not isinstance(context_mask, torch.Tensor) or context_mask.dtype != torch.bool
    ):
        raise TypeError("context_mask must be a bool torch.Tensor.")
    if context is not None and (context.ndim != 3 or context.shape[0] != batch_size):
        raise ValueError(f"context must be [B,L,D] with B={batch_size}, got {tuple(context.shape)}.")
    if context_mask is not None and (context_mask.ndim != 2 or context_mask.shape[0] != batch_size):
        raise ValueError(f"context_mask must be [B,L] with B={batch_size}, got {tuple(context_mask.shape)}.")
    if context is not None and context_mask is not None and context.shape[:2] != context_mask.shape:
        raise ValueError("context_mask must match context batch and sequence dimensions.")
    return batch_size


def randn_per_sample(
    sample_shape: tuple[int, ...],
    *,
    seeds: int | Sequence[int | None] | None,
    batch_size: int,
    rand_device: str,
) -> torch.Tensor:
    seed_values = normalize_seeds(seeds, batch_size)
    samples = []
    for seed in seed_values:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device).manual_seed(seed)
        samples.append(
            torch.randn(
                (1, *sample_shape),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            )
        )
    return torch.cat(samples, dim=0)


def encode_image_batch(vae, input_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    videos = input_image.to(device=device).unsqueeze(2)
    return vae.encode(videos, device=device)


def decode_video_batch(vae, latents: torch.Tensor, device: torch.device) -> list[list[Image.Image]]:
    decoded = vae.decode(latents, device=device)
    if decoded.ndim != 5:
        raise ValueError(f"Decoded video must be [B,C,T,H,W], got {tuple(decoded.shape)}")
    decoded = ((decoded.detach().float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
    return [
        [Image.fromarray(video[:, t].permute(1, 2, 0).numpy()) for t in range(video.shape[1])]
        for video in decoded
    ]
