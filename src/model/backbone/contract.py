from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from .protocol import BLOCK_PROTOCOL_MAIN


@dataclass
class BackboneComponents:
    """The common loading contract consumed by EasyWAM model recipes."""

    name: str
    dit: nn.Module
    vae: nn.Module
    text_encoder: nn.Module | None
    tokenizer: Any | None
    text_dim: int
    block_protocol: str = BLOCK_PROTOCOL_MAIN
    dit_path: str | None = None
    vae_path: str | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
