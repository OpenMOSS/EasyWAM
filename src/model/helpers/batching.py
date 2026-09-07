from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image


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


def validate_single_inference_image(args, kwargs) -> None:
    image = kwargs.get("input_image")
    if image is None and len(args) > 1:
        image = args[1]
    if isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[0] != 1:
        raise ValueError(f"Single-sample inference requires batch size 1, got {image.shape[0]}.")


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
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if input_image.ndim != 4 or input_image.shape[1] != 3:
        raise ValueError(f"`input_image` must be [B,3,H,W], got {tuple(input_image.shape)}")
    videos = [image.unsqueeze(1) for image in input_image.to(device=device)]
    latents = vae.encode(videos, device=device)
    if isinstance(latents, list):
        latents = torch.stack(latents, dim=0)
    return latents


def decode_video_batch(vae, latents: torch.Tensor, device: torch.device) -> list[list[Image.Image]]:
    decoded = vae.decode(latents, device=device)
    if isinstance(decoded, list):
        decoded = torch.stack(decoded, dim=0)
    if decoded.ndim == 4:
        decoded = decoded.unsqueeze(0)
    if decoded.ndim != 5:
        raise ValueError(f"Decoded video must be [B,C,T,H,W], got {tuple(decoded.shape)}")
    decoded = ((decoded.detach().float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
    return [
        [Image.fromarray(video[:, t].permute(1, 2, 0).numpy()) for t in range(video.shape[1])]
        for video in decoded
    ]
