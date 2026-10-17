from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from .protocol import BLOCK_PROTOCOL_STANDARD


@dataclass
class BackboneComponents:
    name: str
    dit: nn.Module
    vae: nn.Module
    text_encoder: nn.Module | None
    tokenizer: Any | None
    text_dim: int
    block_protocol: str = BLOCK_PROTOCOL_STANDARD
    dit_path: str | None = None
    vae_path: str | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextEncoding:
    context: torch.Tensor
    mask: torch.Tensor
    token_mask: torch.Tensor


@dataclass
class TextEncoderComponents:
    model: nn.Module
    encode_batch: Callable[[Sequence[str]], TextEncoding]
    batch_size: int
    model_path: str
    tokenizer_path: str
