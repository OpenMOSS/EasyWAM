from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from .flux2_vae_arch import AutoEncoder, AutoEncoderParams


class Flux2VideoVAE(nn.Module):
    """FLUX image autoencoder used independently for video endpoint frames.

    The endpoint helper flattens frames to [B, C, H, W] before encoding;
    temporal selection and reshaping remain in the shared policy helper.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.micro_batch_size: int | None = None

    def set_micro_batch_size(self, value: int | None) -> None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("VAE micro batch size must be a positive integer or None.")
        self.micro_batch_size = value

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Flux2VideoVAE":

        with torch.device("meta"):
            network = AutoEncoder(AutoEncoderParams()).to(torch_dtype)
        state = load_safetensors(str(checkpoint_path), device=str(device))
        network.load_state_dict(state, strict=True, assign=True)
        return cls(network).eval().requires_grad_(False).to(
            device=device, dtype=torch_dtype
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode(images)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents)
