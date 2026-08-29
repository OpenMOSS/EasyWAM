"""Cosmos-Predict2.5 video tokenizer interface."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .cosmos_vae_arch import CosmosTokenizerNetwork


COSMOS_LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
COSMOS_LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class CosmosVideoVAE(nn.Module):
    latent_channels = 16
    spatial_compression_ratio = 8
    temporal_compression_ratio = 4
    upsampling_factor = 8
    temporal_downsample_factor = 4
    z_dim = 16

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("latent_mean", torch.tensor(COSMOS_LATENT_MEAN), persistent=False)
        self.register_buffer("latent_std", torch.tensor(COSMOS_LATENT_STD), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "CosmosVideoVAE":
        path = Path(checkpoint_path)
        try:
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu", weights_only=True)
        with torch.device("meta"):
            network = CosmosTokenizerNetwork(z_dim=cls.latent_channels)
        result = network.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Cosmos tokenizer mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        network.eval().requires_grad_(False)
        return cls(network).to(
            device=device, dtype=torch_dtype
        )

    @property
    def _scale(self) -> list[torch.Tensor]:
        return [self.latent_mean, self.latent_std.reciprocal()]

    @torch.no_grad()
    def encode(
        self,
        video,
        device: str | torch.device | None = None,
    ):
        input_was_list = isinstance(video, (list, tuple))
        if input_was_list:
            video = torch.stack([value for value in video])
        if device is not None:
            video = video.to(device=device)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] % 4 != 1:
            raise ValueError(f"video frame count must satisfy T % 4 == 1, got {video.shape[2]}")
        if video.shape[3] % 8 or video.shape[4] % 8:
            raise ValueError("video height and width must be divisible by 8.")
        outputs = [self.model.encode(item[None], self._scale)[0] for item in video]
        result = torch.stack(outputs)
        if input_was_list:
            return [item for item in result]
        return result

    @torch.no_grad()
    def decode(
        self,
        latents,
        device: str | torch.device | None = None,
    ):
        input_was_list = isinstance(latents, (list, tuple))
        if input_was_list:
            latents = torch.stack([value for value in latents])
        if device is not None:
            latents = latents.to(device=device)
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"latents must be [B,16,T,H,W], got {tuple(latents.shape)}")
        outputs = [self.model.decode(item[None], self._scale)[0] for item in latents]
        result = torch.stack(outputs).clamp(-1, 1)
        if input_was_list:
            return [item for item in result]
        return result
